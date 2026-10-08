"""Train a lightweight UTC segmentation decoder on frozen I-JEPA grids.

Consumes cached 16x16x1280 I-JEPA patch-token features. The I-JEPA encoder stays
fully frozen. Checkpoint selection and threshold selection use validation only;
test metrics are reported at both fixed 0.50 and validation-selected threshold.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

try:
    from scripts.train_utc_unet import dice_loss,evaluate
except ModuleNotFoundError:
    from train_utc_unet import dice_loss,evaluate


class IJEPAFeatureDataset(Dataset):
    def __init__(self,dataset_dir:Path,records:list[dict],feature_npz:Path,augment:bool=False):
        self.dataset_dir=dataset_dir
        self.records=records
        self.augment=augment
        with np.load(feature_npz) as z:
            self.features={k:z[k].astype(np.float32) for k in z.files}

    def __len__(self): return len(self.records)

    def __getitem__(self,idx):
        r=self.records[idx]
        key=f"patch_{int(r['id']):03d}"
        feat=self.features[key]  # H,W,C
        mask=(np.asarray(Image.open(self.dataset_dir/"masks"/r["mask_file"]).convert("L"))>0).astype(np.float32)

        if self.augment:
            if random.random()<0.5:
                feat=np.flip(feat,axis=1).copy();mask=np.flip(mask,axis=1).copy()
            if random.random()<0.5:
                feat=np.flip(feat,axis=0).copy();mask=np.flip(mask,axis=0).copy()
            k=random.randint(0,3)
            if k:
                feat=np.rot90(feat,k,axes=(0,1)).copy();mask=np.rot90(mask,k,axes=(0,1)).copy()

        x=torch.from_numpy(np.moveaxis(feat,-1,0))
        y=torch.from_numpy(mask[None,...])
        return x,y


class IJEPAFeatureDecoder(nn.Module):
    def __init__(self,in_channels:int=1280,hidden:int=256):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels,hidden,3,padding=1,bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden,hidden,3,padding=1,bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden,hidden//2,3,padding=1,bias=False),
            nn.BatchNorm2d(hidden//2),
            nn.GELU(),
            nn.Conv2d(hidden//2,1,1),
        )

    def forward(self,x):
        logits=self.net(x)
        return F.interpolate(logits,size=(256,256),mode="bilinear",align_corners=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-dir",type=Path,required=True)
    p.add_argument("--features",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--lr",type=float,default=5e-4)
    p.add_argument("--patience",type=int,default=8)
    p.add_argument("--hidden",type=int,default=256)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()

    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)

    dm=json.loads((args.dataset_dir/"dataset_manifest.json").read_text(encoding="utf-8"))
    splits={s:[r for r in dm["records"] if r["split"]==s] for s in ("train","val","test")}
    train_pos=sum(r["canopy_pixels"] for r in splits["train"])
    train_total=sum(r["total_pixels"] for r in splits["train"])
    pos_weight=(train_total-train_pos)/max(train_pos,1)

    loaders={
      "train":DataLoader(IJEPAFeatureDataset(args.dataset_dir,splits["train"],args.features,True),batch_size=args.batch_size,shuffle=True),
      "val":DataLoader(IJEPAFeatureDataset(args.dataset_dir,splits["val"],args.features,False),batch_size=args.batch_size,shuffle=False),
      "test":DataLoader(IJEPAFeatureDataset(args.dataset_dir,splits["test"],args.features,False),batch_size=args.batch_size,shuffle=False),
    }

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=IJEPAFeatureDecoder(hidden=args.hidden).to(device)
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight],device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)

    args.output_dir.mkdir(parents=True,exist_ok=True)
    best_path=args.output_dir/"ijepa_decoder_best.pt"
    history=[];best_iou=-1.0;stale=0

    print(f"Device: {device}; encoder=I-JEPA ViT-H/14; encoder_frozen=True")
    print(f"Split sizes: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"Train canopy fraction: {train_pos/train_total:.3f}; BCE pos_weight={pos_weight:.2f}")

    for epoch in range(1,args.epochs+1):
        model.train();losses=[];start=time.time()
        for x,y in loaders["train"]:
            x=x.to(device);y=y.to(device)
            opt.zero_grad(set_to_none=True)
            z=model(x)
            loss=bce(z,y)+dice_loss(z,y)
            loss.backward();opt.step();losses.append(float(loss.item()))
        vm=evaluate(model,loaders["val"],device,0.5)
        history.append({"epoch":epoch,"loss":float(np.mean(losses)),"val":vm})
        print(f"epoch={epoch:03d} loss={np.mean(losses):.4f} val_iou={vm['iou']:.3f} val_f1={vm['f1']:.3f} val_bal={vm['balanced_accuracy']:.3f} sec={time.time()-start:.1f}")
        if vm["iou"]>best_iou:
            best_iou=vm["iou"];stale=0
            torch.save({"model_state":model.state_dict(),"epoch":epoch,"hidden":args.hidden},best_path)
        else:
            stale+=1
            if stale>=args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    ck=torch.load(best_path,map_location=device,weights_only=False)
    model.load_state_dict(ck["model_state"])

    sweep=[]
    for t in np.arange(.2,.81,.05):
        m=evaluate(model,loaders["val"],device,float(t))
        sweep.append({"threshold":round(float(t),2),**m})
    best_thr=max(sweep,key=lambda r:(r["iou"],r["f1"],r["balanced_accuracy"]))
    threshold=float(best_thr["threshold"])

    fixed_test=evaluate(model,loaders["test"],device,0.5)
    calibrated_test=evaluate(model,loaders["test"],device,threshold)

    result={
      "model":"ijepa_vith14_frozen_decoder_rgb_v1",
      "encoder":"Meta FAIR I-JEPA ViT-H/14",
      "encoder_frozen":True,
      "features":str(args.features),
      "best_epoch":ck["epoch"],
      "decoder_hidden":args.hidden,
      "threshold_selection":"validation_iou",
      "fixed_threshold_test":{"threshold":0.5,**fixed_test},
      "best_validation":best_thr,
      "test_at_locked_threshold":{"threshold":threshold,**calibrated_test},
      "history":history,
      "validation_sweep":sweep,
      "license_note":"I-JEPA upstream is CC BY-NC 4.0; research/prototype use unless separately licensed.",
      "note":"Frozen I-JEPA patch-token benchmark. Checkpoint and threshold use validation only; both fixed-0.5 and validation-calibrated test results retained."
    }
    out=args.output_dir/"ijepa_decoder_metrics.json"
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"Selected threshold: {threshold:.2f}")
    print("FIXED-0.50 TEST:",json.dumps(result["fixed_threshold_test"],indent=2))
    print("BEST VALIDATION:",json.dumps(best_thr,indent=2))
    print("LOCKED-THRESHOLD TEST:",json.dumps(result["test_at_locked_threshold"],indent=2))
    print(f"Wrote {best_path}")
    print(f"Wrote {out}")


if __name__=="__main__":
    main()
