"""Benchmark a frozen DINOv2 encoder with a lightweight UTC segmentation decoder.

The pretrained encoder is frozen. A small convolutional head learns canopy
segmentation from patch-token features on the exact same spatial train/val/test
splits used by U-Net, DeepLabV3, and SegFormer.

Checkpoint selection and threshold selection use validation only. Both fixed
0.50 and validation-calibrated test metrics are retained because the benchmark
contains a strong canopy-prevalence shift across spatial splits.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

try:
    from scripts.train_utc_unet import PatchDataset, dice_loss, evaluate
except ModuleNotFoundError:
    from train_utc_unet import PatchDataset, dice_loss, evaluate


IMAGENET_MEAN=torch.tensor([0.485,0.456,0.406]).view(3,1,1)
IMAGENET_STD=torch.tensor([0.229,0.224,0.225]).view(3,1,1)


class NormalizedPatchDataset(PatchDataset):
    def __getitem__(self,idx):
        x,y=super().__getitem__(idx)
        x=(x-IMAGENET_MEAN)/IMAGENET_STD
        return x,y


def tokens_to_feature_map(tokens:torch.Tensor)->torch.Tensor:
    """Convert BxNxC square patch tokens into BxCxHxW."""
    b,n,c=tokens.shape
    side=int(round(n**0.5))
    if side*side!=n:
        raise ValueError(f"Expected square token grid, got {n} patch tokens")
    return tokens.transpose(1,2).reshape(b,c,side,side)


class DINOv2Binary(nn.Module):
    def __init__(self,model_name:str="facebook/dinov2-small",hidden:int=128):
        super().__init__()
        try:
            from transformers import AutoModel
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "transformers is required. Install requirements-utc-segmentation.txt"
            ) from e
        self.encoder=AutoModel.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad=False
        dim=int(self.encoder.config.hidden_size)
        self.decoder=nn.Sequential(
            nn.Conv2d(dim,hidden,3,padding=1,bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden,hidden//2,3,padding=1,bias=False),
            nn.BatchNorm2d(hidden//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden//2,1,1),
        )

    def train(self,mode:bool=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self,x):
        with torch.no_grad():
            out=self.encoder(pixel_values=x)
            patch_tokens=out.last_hidden_state[:,1:,:]
        fmap=tokens_to_feature_map(patch_tokens)
        logits=self.decoder(fmap)
        return F.interpolate(logits,size=x.shape[-2:],mode="bilinear",align_corners=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--lr",type=float,default=5e-4)
    p.add_argument("--patience",type=int,default=7)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--model-name",default="facebook/dinov2-small")
    p.add_argument("--decoder-hidden",type=int,default=128)
    args=p.parse_args()

    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    dm=json.loads((args.dataset_dir/"dataset_manifest.json").read_text(encoding="utf-8"))
    splits={s:[r for r in dm["records"] if r["split"]==s] for s in ("train","val","test")}
    train_pos=sum(r["canopy_pixels"] for r in splits["train"])
    train_total=sum(r["total_pixels"] for r in splits["train"])
    pos_weight=(train_total-train_pos)/max(train_pos,1)

    loaders={
      "train":DataLoader(NormalizedPatchDataset(args.dataset_dir,splits["train"],True),batch_size=args.batch_size,shuffle=True),
      "val":DataLoader(NormalizedPatchDataset(args.dataset_dir,splits["val"],False),batch_size=args.batch_size,shuffle=False),
      "test":DataLoader(NormalizedPatchDataset(args.dataset_dir,splits["test"],False),batch_size=args.batch_size,shuffle=False),
    }

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=DINOv2Binary(args.model_name,args.decoder_hidden).to(device)
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight],device=device))
    opt=torch.optim.AdamW(model.decoder.parameters(),lr=args.lr,weight_decay=1e-4)

    args.output_dir.mkdir(parents=True,exist_ok=True)
    best_path=args.output_dir/"dinov2_decoder_best.pt"
    history=[];best_iou=-1.0;stale=0

    print(f"Device: {device}; encoder={args.model_name}; encoder_frozen=True")
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
            torch.save({
                "decoder_state":model.decoder.state_dict(),
                "epoch":epoch,
                "model_name":args.model_name,
                "decoder_hidden":args.decoder_hidden,
            },best_path)
        else:
            stale+=1
            if stale>=args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    ck=torch.load(best_path,map_location=device,weights_only=False)
    model.decoder.load_state_dict(ck["decoder_state"])

    sweep=[]
    for t in np.arange(.2,.81,.05):
        m=evaluate(model,loaders["val"],device,float(t))
        sweep.append({"threshold":round(float(t),2),**m})
    best_thr=max(sweep,key=lambda r:(r["iou"],r["f1"],r["balanced_accuracy"]))
    threshold=float(best_thr["threshold"])

    fixed_test=evaluate(model,loaders["test"],device,0.5)
    calibrated_test=evaluate(model,loaders["test"],device,threshold)

    result={
      "model":"dinov2_frozen_decoder_rgb_v1",
      "encoder":args.model_name,
      "encoder_frozen":True,
      "best_epoch":ck["epoch"],
      "decoder_hidden":args.decoder_hidden,
      "threshold_selection":"validation_iou",
      "fixed_threshold_test":{"threshold":0.5,**fixed_test},
      "best_validation":best_thr,
      "test_at_locked_threshold":{"threshold":threshold,**calibrated_test},
      "history":history,
      "validation_sweep":sweep,
      "note":"Frozen DINOv2 representation benchmark. Checkpoint and threshold use validation only. Both fixed-0.5 and validation-calibrated test metrics are retained."
    }
    out=args.output_dir/"dinov2_decoder_metrics.json"
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"Selected threshold: {threshold:.2f}")
    print("FIXED-0.50 TEST:",json.dumps(result["fixed_threshold_test"],indent=2))
    print("BEST VALIDATION:",json.dumps(best_thr,indent=2))
    print("LOCKED-THRESHOLD TEST:",json.dumps(result["test_at_locked_threshold"],indent=2))
    print(f"Wrote {best_path}")
    print(f"Wrote {out}")


if __name__=="__main__":
    main()
