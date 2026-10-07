"""Calibrate a trained UTC segmentation model threshold on validation only.

Currently supports the U-Net checkpoint produced by train_utc_unet.py.
The selected threshold is chosen by validation IoU, then locked and applied once
to the untouched test split.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.train_utc_unet import PatchDataset,UNetSmall,evaluate

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-dir",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--min-threshold",type=float,default=0.20)
    p.add_argument("--max-threshold",type=float,default=0.80)
    p.add_argument("--step",type=float,default=0.05)
    args=p.parse_args()

    dm=json.loads((args.dataset_dir/"dataset_manifest.json").read_text(encoding="utf-8"))
    splits={s:[r for r in dm["records"] if r["split"]==s] for s in ("val","test")}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=UNetSmall(ck["base_channels"]).to(device)
    model.load_state_dict(ck["model_state"])

    val_loader=DataLoader(PatchDataset(args.dataset_dir,splits["val"],False),batch_size=args.batch_size,shuffle=False)
    test_loader=DataLoader(PatchDataset(args.dataset_dir,splits["test"],False),batch_size=args.batch_size,shuffle=False)

    rows=[]
    thresholds=np.arange(args.min_threshold,args.max_threshold+1e-9,args.step)
    for t in thresholds:
        m=evaluate(model,val_loader,device,float(t))
        rows.append({"threshold":round(float(t),4),**m})
    best=max(rows,key=lambda r:(r["iou"],r["f1"],r["balanced_accuracy"]))
    locked=float(best["threshold"])
    test=evaluate(model,test_loader,device,locked)

    out={
      "status":"validation_calibrated_test_evaluation",
      "selection_metric":"validation_iou",
      "checkpoint":str(args.checkpoint),
      "best_validation":best,
      "test_at_locked_threshold":{"threshold":locked,**test},
      "validation_sweep":rows,
      "note":"Threshold chosen on validation only; test split was evaluated once after locking threshold."
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(f"Selected threshold: {locked:.2f}")
    print("BEST VALIDATION:",json.dumps(best,indent=2))
    print("LOCKED-THRESHOLD TEST:",json.dumps(out["test_at_locked_threshold"],indent=2))
    print(f"Wrote {args.output}")

if __name__=="__main__": main()
