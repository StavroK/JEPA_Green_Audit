"""Zero-shot cross-site UTC point evaluation for Fundidora-trained RGB models.

Runs a trained model over a larger RGB image using overlapping tiles, averages
probabilities in overlap regions, samples probabilities at pre-generated human
validation points, and reports point-based classification metrics.

Important: thresholds are supplied explicitly and must come from Fundidora
validation. Do not tune them on La Pastora labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, jaccard_score, precision_score, recall_score

try:
    from scripts.train_utc_unet import UNetSmall
    from scripts.train_utc_segformer import SegFormerBinary, IMAGENET_MEAN as SF_MEAN, IMAGENET_STD as SF_STD
    from scripts.train_utc_dinov2 import DINOv2Binary, IMAGENET_MEAN as DINO_MEAN, IMAGENET_STD as DINO_STD
    from scripts.train_utc_ijepa_decoder import IJEPAFeatureDecoder
except ModuleNotFoundError:
    from train_utc_unet import UNetSmall
    from train_utc_segformer import SegFormerBinary, IMAGENET_MEAN as SF_MEAN, IMAGENET_STD as SF_STD
    from train_utc_dinov2 import DINOv2Binary, IMAGENET_MEAN as DINO_MEAN, IMAGENET_STD as DINO_STD
    from train_utc_ijepa_decoder import IJEPAFeatureDecoder
from jepa_green_audit.ijepa_encoder import FrozenIJEPAEncoder


def percentile_stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    out=np.zeros_like(rgb,dtype=np.uint8)
    for b in range(3):
        band=rgb[...,b].astype(np.float32)
        vals=band[np.isfinite(band)]
        if vals.size==0:
            continue
        lo,hi=np.percentile(vals,[2,98])
        if hi<=lo:
            hi=lo+1.0
        out[...,b]=(np.clip((band-lo)/(hi-lo),0,1)*255).astype(np.uint8)
    return out


def tile_starts(length:int,tile:int,stride:int)->list[int]:
    if tile>length:
        return [0]
    starts=list(range(0,max(1,length-tile+1),stride))
    last=length-tile
    if starts[-1]!=last:
        starts.append(last)
    return starts


def load_model(model_type:str, checkpoint:Path, device:torch.device):
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    if model_type=="unet":
        model=UNetSmall(int(ck.get("base_channels",32))).to(device)
        model.load_state_dict(ck["model_state"])
        normalizer=None
    elif model_type=="segformer":
        model_name=ck.get("model_name","nvidia/mit-b0")
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ModuleNotFoundError as e:
            raise RuntimeError("transformers is required for SegFormer inference") from e
        cfg=SegformerConfig.from_pretrained(model_name)
        cfg.num_labels=1
        cfg.id2label={0:"canopy"}
        cfg.label2id={"canopy":0}
        model=SegFormerBinary.__new__(SegFormerBinary)
        torch.nn.Module.__init__(model)
        model.model=SegformerForSemanticSegmentation(cfg)
        model=model.to(device)
        model.load_state_dict(ck["model_state"])
        normalizer=(SF_MEAN,SF_STD)
    elif model_type=="dinov2":
        model_name=ck.get("model_name","facebook/dinov2-small")
        hidden=int(ck.get("decoder_hidden",128))
        model=DINOv2Binary(model_name=model_name,hidden=hidden).to(device)
        model.decoder.load_state_dict(ck["decoder_state"])
        normalizer=(DINO_MEAN,DINO_STD)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    model.eval()
    return model,normalizer,ck


@torch.no_grad()
def predict_full_image(model, rgb:np.ndarray, device:torch.device, tile:int, stride:int, normalizer=None, preprocessing:str="percentile"):
    h,w=rgb.shape[:2]
    prob_sum=np.zeros((h,w),dtype=np.float32)
    weight=np.zeros((h,w),dtype=np.float32)
    for y in tile_starts(h,tile,stride):
        for x in tile_starts(w,tile,stride):
            patch=rgb[y:min(y+tile,h),x:min(x+tile,w)]
            ph,pw=patch.shape[:2]
            if preprocessing=="percentile":
                patch=percentile_stretch_rgb(patch)
            if ph<tile or pw<tile:
                padded=np.zeros((tile,tile,3),dtype=np.uint8)
                padded[:ph,:pw]=patch
                patch=padded
            arr=patch.astype(np.float32)/255.0
            ten=torch.from_numpy(np.moveaxis(arr,-1,0))[None].to(device)
            if normalizer is not None:
                mean,std=normalizer
                ten=(ten-mean.to(device))/std.to(device)
            p=torch.sigmoid(model(ten))[0,0].cpu().numpy()[:ph,:pw]
            prob_sum[y:y+ph,x:x+pw]+=p
            weight[y:y+ph,x:x+pw]+=1.0
    if np.any(weight==0):
        raise RuntimeError("Sliding-window inference left uncovered pixels")
    return prob_sum/weight


def load_ijepa_decoder(checkpoint:Path, device:torch.device):
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    hidden=int(ck.get("hidden",256))
    decoder=IJEPAFeatureDecoder(hidden=hidden).to(device)
    decoder.load_state_dict(ck["model_state"])
    decoder.eval()
    return decoder,ck


@torch.no_grad()
def predict_full_image_ijepa(
    decoder,
    encoder:FrozenIJEPAEncoder,
    rgb:np.ndarray,
    device:torch.device,
    tile:int,
    stride:int,
    preprocessing:str="percentile",
):
    h,w=rgb.shape[:2]
    prob_sum=np.zeros((h,w),dtype=np.float32)
    weight=np.zeros((h,w),dtype=np.float32)
    for y in tile_starts(h,tile,stride):
        for x in tile_starts(w,tile,stride):
            patch=rgb[y:min(y+tile,h),x:min(x+tile,w)]
            ph,pw=patch.shape[:2]
            if preprocessing=="percentile":
                patch=percentile_stretch_rgb(patch)
            if ph<tile or pw<tile:
                padded=np.zeros((tile,tile,3),dtype=np.uint8)
                padded[:ph,:pw]=patch
                patch=padded
            grid=encoder.encode_rgb_patch_grid(patch)
            ten=torch.from_numpy(np.moveaxis(grid,-1,0))[None].to(device)
            p=torch.sigmoid(decoder(ten))[0,0].cpu().numpy()[:ph,:pw]
            prob_sum[y:y+ph,x:x+pw]+=p
            weight[y:y+ph,x:x+pw]+=1.0
    if np.any(weight==0):
        raise RuntimeError("Sliding-window I-JEPA inference left uncovered pixels")
    return prob_sum/weight


def point_metrics(labels:list[str], probs:list[float], threshold:float)->dict:
    keep=[i for i,h in enumerate(labels) if h in {"tree","non"}]
    y=np.asarray([1 if labels[i]=="tree" else 0 for i in keep],dtype=np.uint8)
    p=np.asarray([probs[i] for i in keep],dtype=np.float32)
    pred=(p>=threshold).astype(np.uint8)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {
        "n_definite":int(len(y)),
        "tree_reference":int((y==1).sum()),
        "non_tree_reference":int((y==0).sum()),
        "threshold":float(threshold),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "precision_tree":float(precision_score(y,pred,zero_division=0)),
        "recall_tree":float(recall_score(y,pred,zero_division=0)),
        "f1_tree":float(f1_score(y,pred,zero_division=0)),
        "iou_tree":float(jaccard_score(y,pred,zero_division=0)),
        "predicted_tree_fraction_points":float(pred.mean()),
        "reference_tree_fraction_points":float(y.mean()),
        "point_prevalence_bias_pp":float(100*(pred.mean()-y.mean())),
        "confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)},
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-type",choices=("unet","segformer","dinov2","ijepa"),required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--threshold",type=float,required=True,help="Locked Fundidora validation threshold")
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--labels",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--tile-size",type=int,default=256)
    p.add_argument("--stride",type=int,default=128)
    p.add_argument("--preprocessing",choices=("raw","percentile"),default="percentile",
                   help="Use percentile for legacy Fundidora v1 models trained on per-patch stretched RGB.")
    p.add_argument("--ijepa-upstream",type=Path,default=Path("vendor/ijepa"))
    p.add_argument("--ijepa-encoder-checkpoint",type=Path,default=Path("models/IN1K-vit.h.14-300e.pth.tar"))
    args=p.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb=np.asarray(Image.open(args.image).convert("RGB"),dtype=np.uint8)
    payload=json.loads(args.labels.read_text(encoding="utf-8"))
    if int(payload.get("width_px",-1))!=rgb.shape[1] or int(payload.get("height_px",-1))!=rgb.shape[0]:
        raise ValueError("Point-label dimensions do not match evaluation image")

    if args.model_type=="ijepa":
        decoder,ck=load_ijepa_decoder(args.checkpoint,device)
        encoder=FrozenIJEPAEncoder(
            upstream_dir=args.ijepa_upstream,
            checkpoint_path=args.ijepa_encoder_checkpoint,
            device=str(device),
        )
        prob=predict_full_image_ijepa(
            decoder,encoder,rgb,device,args.tile_size,args.stride,args.preprocessing
        )
    else:
        model,normalizer,ck=load_model(args.model_type,args.checkpoint,device)
        prob=predict_full_image(model,rgb,device,args.tile_size,args.stride,normalizer,args.preprocessing)

    labels=[];probs=[];rows=[]
    for pt in payload["points"]:
        x=int(pt["x"]);y=int(pt["y"])
        if not (0<=x<rgb.shape[1] and 0<=y<rgb.shape[0]):
            raise ValueError(f"Point {pt.get('id')} outside image: {(x,y)}")
        pr=float(prob[y,x])
        labels.append(pt.get("human_label",""))
        probs.append(pr)
        rows.append({
            "id":pt.get("id"),"x":x,"y":y,
            "human_label":pt.get("human_label",""),
            "detail_label":pt.get("detail_label",""),
            "model_probability":pr,
            "model_class":"tree" if pr>=args.threshold else "non",
        })

    metrics=point_metrics(labels,probs,args.threshold)
    result={
        "status":"zero_shot_cross_site_point_validation",
        "source_training_site":"fundidora_2007",
        "target_validation_site":payload.get("site","la_pastora"),
        "target_date_label":payload.get("date_label"),
        "model_type":args.model_type,
        "checkpoint":str(args.checkpoint),
        "threshold_source":"locked externally from Fundidora validation; not tuned on target labels",
        "threshold":args.threshold,
        "preprocessing":args.preprocessing,
        "tile_size":args.tile_size,
        "stride":args.stride,
        "device":str(device),
        "metrics":metrics,
        "points":rows,
        "note":"Point metrics validate cross-site classification at sampled locations. They are not dense UTC area estimates.",
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"Model: {args.model_type} | device={device} | threshold={args.threshold:.2f}")
    print(json.dumps(metrics,indent=2))
    print(f"Wrote {args.output}")


if __name__=="__main__":
    main()
