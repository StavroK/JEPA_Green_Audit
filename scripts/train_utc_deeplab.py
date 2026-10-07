"""Train DeepLabV3-ResNet50 for UTC canopy segmentation with validation-only threshold calibration."""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50,DeepLabV3_ResNet50_Weights
from scripts.train_utc_unet import PatchDataset,dice_loss,evaluate

class DeepLabBinary(nn.Module):
    def __init__(self,pretrained:bool):
        super().__init__()
        weights=DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        m=deeplabv3_resnet50(weights=weights,weights_backbone=None if pretrained else None)
        m.classifier[-1]=nn.Conv2d(256,1,1)
        if m.aux_classifier is not None:
            m.aux_classifier=None
        self.model=m
    def forward(self,x):
        return self.model(x)["out"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=25)
    p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--patience",type=int,default=6)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--pretrained",action="store_true")
    args=p.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)

    dm=json.loads((args.dataset_dir/"dataset_manifest.json").read_text(encoding="utf-8"))
    splits={s:[r for r in dm["records"] if r["split"]==s] for s in ("train","val","test")}
    train_pos=sum(r["canopy_pixels"] for r in splits["train"]);train_total=sum(r["total_pixels"] for r in splits["train"])
    pos_weight=(train_total-train_pos)/max(train_pos,1)

    loaders={
      "train":DataLoader(PatchDataset(args.dataset_dir,splits["train"],True),batch_size=args.batch_size,shuffle=True),
      "val":DataLoader(PatchDataset(args.dataset_dir,splits["val"],False),batch_size=args.batch_size,shuffle=False),
      "test":DataLoader(PatchDataset(args.dataset_dir,splits["test"],False),batch_size=args.batch_size,shuffle=False)
    }
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=DeepLabBinary(args.pretrained).to(device)
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight],device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    best_path=args.output_dir/"deeplabv3_best.pt"
    history=[];best=-1;stale=0
    print(f"Device: {device}; pretrained={args.pretrained}")
    print(f"Split sizes: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    for epoch in range(1,args.epochs+1):
        model.train();losses=[];start=time.time()
        for x,y in loaders["train"]:
            x=x.to(device);y=y.to(device);opt.zero_grad(set_to_none=True)
            z=model(x);loss=bce(z,y)+dice_loss(z,y);loss.backward();opt.step();losses.append(float(loss.item()))
        vm=evaluate(model,loaders["val"],device,0.5)
        history.append({"epoch":epoch,"loss":float(np.mean(losses)),"val":vm})
        print(f"epoch={epoch:03d} loss={np.mean(losses):.4f} val_iou={vm['iou']:.3f} val_f1={vm['f1']:.3f} sec={time.time()-start:.1f}")
        if vm["iou"]>best:
            best=vm["iou"];stale=0;torch.save({"model_state":model.state_dict(),"epoch":epoch},best_path)
        else:
            stale+=1
            if stale>=args.patience:
                print(f"Early stopping at epoch {epoch}");break

    ck=torch.load(best_path,map_location=device,weights_only=False);model.load_state_dict(ck["model_state"])
    sweep=[]
    for t in np.arange(.2,.81,.05):
        m=evaluate(model,loaders["val"],device,float(t));sweep.append({"threshold":round(float(t),2),**m})
    best_thr=max(sweep,key=lambda r:(r["iou"],r["f1"],r["balanced_accuracy"]))
    threshold=float(best_thr["threshold"])
    test=evaluate(model,loaders["test"],device,threshold)
    result={
      "model":"deeplabv3_resnet50_rgb_v1","pretrained":args.pretrained,"best_epoch":ck["epoch"],
      "threshold_selection":"validation_iou","best_validation":best_thr,
      "test_at_locked_threshold":{"threshold":threshold,**test},
      "history":history,"validation_sweep":sweep,
      "note":"Threshold chosen on validation only; test evaluated after threshold lock."
    }
    (args.output_dir/"deeplabv3_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"Selected threshold: {threshold:.2f}")
    print("BEST VALIDATION:",json.dumps(best_thr,indent=2))
    print("INDEPENDENT TEST:",json.dumps(result["test_at_locked_threshold"],indent=2))

if __name__=="__main__":main()
