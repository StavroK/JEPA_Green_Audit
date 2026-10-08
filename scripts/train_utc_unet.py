"""Train a compact U-Net baseline for UTC canopy segmentation.

Uses spatial train/val/test splits from dataset_manifest.json. The test split is
never used for model selection. Training loss combines BCEWithLogits (with a
train-derived positive-class weight) and soft Dice loss.

Designed to run on CPU or GPU with PyTorch.
"""
from __future__ import annotations

import argparse, json, random, time
from pathlib import Path

from jepa_green_audit.utc_benchmark import load_dataset_manifest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class PatchDataset(Dataset):
    def __init__(self, root:Path, records:list[dict], augment:bool=False):
        self.root=root
        self.records=records
        self.augment=augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r=self.records[idx]
        x=np.asarray(Image.open(self.root/"images"/r["file"]).convert("RGB"),dtype=np.float32)/255.0
        y=(np.asarray(Image.open(self.root/"masks"/r["mask_file"]).convert("L"),dtype=np.uint8)>0).astype(np.float32)
        if self.augment:
            if random.random()<0.5:
                x=np.flip(x,axis=1).copy(); y=np.flip(y,axis=1).copy()
            if random.random()<0.5:
                x=np.flip(x,axis=0).copy(); y=np.flip(y,axis=0).copy()
            k=random.randint(0,3)
            if k:
                x=np.rot90(x,k,axes=(0,1)).copy(); y=np.rot90(y,k,axes=(0,1)).copy()
        x=torch.from_numpy(np.moveaxis(x,-1,0))
        y=torch.from_numpy(y[None,...])
        return x,y


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(cin,cout,3,padding=1,bias=False),
            nn.BatchNorm2d(cout),nn.ReLU(inplace=True),
            nn.Conv2d(cout,cout,3,padding=1,bias=False),
            nn.BatchNorm2d(cout),nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.e1=ConvBlock(3,base)
        self.e2=ConvBlock(base,base*2)
        self.e3=ConvBlock(base*2,base*4)
        self.b=ConvBlock(base*4,base*8)
        self.pool=nn.MaxPool2d(2)
        self.u3=nn.ConvTranspose2d(base*8,base*4,2,2); self.d3=ConvBlock(base*8,base*4)
        self.u2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=ConvBlock(base*4,base*2)
        self.u1=nn.ConvTranspose2d(base*2,base,2,2); self.d1=ConvBlock(base*2,base)
        self.out=nn.Conv2d(base,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); b=self.b(self.pool(e3))
        d3=self.d3(torch.cat([self.u3(b),e3],1))
        d2=self.d2(torch.cat([self.u2(d3),e2],1))
        d1=self.d1(torch.cat([self.u1(d2),e1],1))
        return self.out(d1)


def dice_loss(logits,target,eps=1e-6):
    p=torch.sigmoid(logits)
    inter=(p*target).sum(dim=(1,2,3))
    den=p.sum(dim=(1,2,3))+target.sum(dim=(1,2,3))
    return (1-((2*inter+eps)/(den+eps))).mean()


def metrics_from_counts(tp,fp,tn,fn):
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    specificity=tn/(tn+fp) if tn+fp else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    iou=tp/(tp+fp+fn) if tp+fp+fn else 0.0
    bal=(recall+specificity)/2
    return {"precision":precision,"recall":recall,"specificity":specificity,"f1":f1,"iou":iou,"balanced_accuracy":bal,
            "tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn)}


@torch.no_grad()
def evaluate(model,loader,device,threshold=0.5):
    model.eval(); tp=fp=tn=fn=0
    probs_all=[]; targets_all=[]
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        p=torch.sigmoid(model(x))
        pred=p>=threshold; truth=y>=0.5
        tp+=(pred & truth).sum().item(); fp+=(pred & ~truth).sum().item()
        tn+=(~pred & ~truth).sum().item(); fn+=(~pred & truth).sum().item()
        probs_all.append(p.cpu()); targets_all.append(y.cpu())
    m=metrics_from_counts(tp,fp,tn,fn)
    m["predicted_canopy_fraction"]=(tp+fp)/(tp+fp+tn+fn)
    m["reference_canopy_fraction"]=(tp+fn)/(tp+fp+tn+fn)
    m["utc_area_bias_pp"]=100*(m["predicted_canopy_fraction"]-m["reference_canopy_fraction"])
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-dir",type=Path,required=True)
    p.add_argument("--manifest",type=Path,help="Optional label-efficiency manifest; defaults to dataset_manifest.json")
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--base-channels",type=int,default=32)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--patience",type=int,default=8)
    p.add_argument("--num-workers",type=int,default=0)
    args=p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    dm=load_dataset_manifest(args.dataset_dir,args.manifest)
    splits={s:[r for r in dm["records"] if r["split"]==s] for s in ("train","val","test")}
    if not all(splits.values()): raise ValueError("train/val/test splits must all be non-empty")

    train_pos=sum(r["canopy_pixels"] for r in splits["train"]); train_total=sum(r["total_pixels"] for r in splits["train"])
    train_neg=train_total-train_pos
    pos_weight=train_neg/max(train_pos,1)

    train_ds=PatchDataset(args.dataset_dir,splits["train"],augment=True)
    val_ds=PatchDataset(args.dataset_dir,splits["val"],augment=False)
    test_ds=PatchDataset(args.dataset_dir,splits["test"],augment=False)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
    test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=UNetSmall(args.base_channels).to(device)
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight],device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)

    args.output_dir.mkdir(parents=True,exist_ok=True)
    best_path=args.output_dir/"unet_best.pt"
    history=[]; best_iou=-1.0; stale=0
    print(f"Device: {device}")
    print(f"Split sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"Train canopy fraction: {train_pos/train_total:.3f}; BCE pos_weight={pos_weight:.2f}")

    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]; start=time.time()
        for x,y in train_loader:
            x=x.to(device); y=y.to(device)
            opt.zero_grad(set_to_none=True)
            logits=model(x)
            loss=bce(logits,y)+dice_loss(logits,y)
            loss.backward(); opt.step(); losses.append(float(loss.item()))
        vm=evaluate(model,val_loader,device)
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),"val":vm}
        history.append(row)
        print(f"epoch={epoch:03d} loss={row['train_loss']:.4f} val_iou={vm['iou']:.3f} val_f1={vm['f1']:.3f} val_bal={vm['balanced_accuracy']:.3f} sec={time.time()-start:.1f}")
        if vm["iou"]>best_iou:
            best_iou=vm["iou"]; stale=0
            torch.save({"model_state":model.state_dict(),"base_channels":args.base_channels,"epoch":epoch,"val":vm},best_path)
        else:
            stale+=1
            if stale>=args.patience:
                print(f"Early stopping at epoch {epoch}"); break

    ck=torch.load(best_path,map_location=device,weights_only=False)
    model.load_state_dict(ck["model_state"])
    val_metrics=evaluate(model,val_loader,device)
    test_metrics=evaluate(model,test_loader,device)

    result={
        "model":"unet_small_rgb_v1",
        "device":str(device),
        "seed":args.seed,
        "best_epoch":ck["epoch"],
        "base_channels":args.base_channels,
        "loss":"weighted_bce_plus_soft_dice",
        "train_canopy_fraction":train_pos/train_total,
        "pos_weight":pos_weight,
        "split_sizes":{k:len(v) for k,v in splits.items()},
        "validation":val_metrics,
        "test":test_metrics,
        "history":history,
        "note":"Test split was not used for model selection or threshold tuning."
    }
    (args.output_dir/"unet_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("BEST VALIDATION:",json.dumps(val_metrics,indent=2))
    print("INDEPENDENT TEST:",json.dumps(test_metrics,indent=2))
    print(f"Wrote {best_path}")
    print(f"Wrote {args.output_dir/'unet_metrics.json'}")


if __name__=="__main__":
    main()
