"""Train SegFormer-B0 for UTC canopy segmentation.

Uses the exact spatial train/val/test splits from dataset_manifest.json.
Checkpoint selection and threshold selection use validation only; test is
evaluated after the threshold is locked.

Pretrained mode uses the MIT-B0 encoder weights from Hugging Face.
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


class SegFormerBinary(nn.Module):
    def __init__(self,pretrained:bool=True,model_name:str="nvidia/mit-b0"):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "transformers is required. Install requirements-utc-segmentation.txt"
            ) from e

        if pretrained:
            self.model=SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=1,
                ignore_mismatched_sizes=True,
            )
        else:
            cfg=SegformerConfig(
                num_labels=1,
                depths=[2,2,2,2],
                hidden_sizes=[32,64,160,256],
                decoder_hidden_size=256,
                num_attention_heads=[1,2,5,8],
                sr_ratios=[8,4,2,1],
                patch_sizes=[7,3,3,3],
                strides=[4,2,2,2],
            )
            self.model=SegformerForSemanticSegmentation(cfg)

    def forward(self,x):
        logits=self.model(pixel_values=x).logits
        if logits.shape[-2:]!=x.shape[-2:]:
            logits=F.interpolate(logits,size=x.shape[-2:],mode="bilinear",align_corners=False)
        return logits


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=25)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--patience",type=int,default=6)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--model-name",default="nvidia/mit-b0")
    p.add_argument("--scratch",action="store_true",help="Train SegFormer-B0 from random initialization")
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
    pretrained=not args.scratch
    model=SegFormerBinary(pretrained=pretrained,model_name=args.model_name).to(device)
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight],device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)

    args.output_dir.mkdir(parents=True,exist_ok=True)
    best_path=args.output_dir/"segformer_b0_best.pt"
    history=[];best_iou=-1.0;stale=0
    print(f"Device: {device}; pretrained={pretrained}; model={args.model_name}")
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
            torch.save({"model_state":model.state_dict(),"epoch":epoch,"model_name":args.model_name,"pretrained":pretrained},best_path)
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
      "model":"segformer_b0_rgb_v1",
      "pretrained":pretrained,
      "model_name":args.model_name,
      "best_epoch":ck["epoch"],
      "threshold_selection":"validation_iou",
      "fixed_threshold_test":{"threshold":0.5,**fixed_test},
      "best_validation":best_thr,
      "test_at_locked_threshold":{"threshold":threshold,**calibrated_test},
      "history":history,
      "validation_sweep":sweep,
      "note":"Checkpoint and threshold use validation only. Both fixed-0.5 and validation-calibrated test results are retained because the spatial splits have strong canopy-prevalence shift."
    }
    (args.output_dir/"segformer_b0_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"Selected threshold: {threshold:.2f}")
    print("FIXED-0.50 TEST:",json.dumps(result["fixed_threshold_test"],indent=2))
    print("BEST VALIDATION:",json.dumps(best_thr,indent=2))
    print("LOCKED-THRESHOLD TEST:",json.dumps(result["test_at_locked_threshold"],indent=2))
    print(f"Wrote {best_path}")
    print(f"Wrote {args.output_dir/'segformer_b0_metrics.json'}")


if __name__=="__main__":
    main()
