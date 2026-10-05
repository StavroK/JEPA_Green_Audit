"""Train a lightweight RGB canopy classifier from human point labels.

This is a calibration step for the UTC workflow. It uses human tree/non-tree
point labels plus local RGB/color/texture features to train a RandomForest,
then predicts a canopy-probability raster and candidate binary mask.

The result is still a model estimate and must be independently reviewed before
being reported as authoritative UTC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from scripts.propose_utc_canopy_mask import box_mean, robust_rgb


def feature_stack(rgb: np.ndarray) -> tuple[np.ndarray, list[str]]:
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
    denom=r+g+b+1e-6
    rn,gn,bn=r/denom,g/denom,b/denom
    exg=2*gn-rn-bn
    brightness=(r+g+b)/3.0
    green_dom=g-np.maximum(r,b)
    lum=0.299*r+0.587*g+0.114*b
    mean3=box_mean(lum,1)
    mean5=box_mean(lum,2)
    tex3=np.sqrt(np.maximum(box_mean(lum*lum,1)-mean3*mean3,0))
    tex5=np.sqrt(np.maximum(box_mean(lum*lum,2)-mean5*mean5,0))
    feats=np.stack([r,g,b,rn,gn,bn,exg,brightness,green_dom,mean3,mean5,tex3,tex5],axis=-1)
    names=["r","g","b","rn","gn","bn","exg","brightness","green_dominance","mean3","mean5","texture3","texture5"]
    return feats.astype(np.float32),names


def parse_labels(payload: dict, width: int, height: int):
    xs=[];ys=[];targets=[]
    for item in payload.get("labels",[]):
        label=item.get("label")
        if label not in {"tree","non"}:
            continue
        x=int(item["x"]); y=int(item["y"])
        if 0 <= x < width and 0 <= y < height:
            xs.append(x);ys.append(y);targets.append(1 if label=="tree" else 0)
    return np.asarray(xs),np.asarray(ys),np.asarray(targets,dtype=np.uint8)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--labels",type=Path,required=True)
    p.add_argument("--output-probability",type=Path,required=True)
    p.add_argument("--output-mask",type=Path,required=True)
    p.add_argument("--output-preview",type=Path,required=True)
    p.add_argument("--output-json",type=Path,required=True)
    p.add_argument("--threshold",type=float,default=0.5)
    p.add_argument("--trees",type=int,default=300)
    args=p.parse_args()

    with rasterio.open(args.image) as src:
        data=src.read()
        rgb=robust_rgb(data)
        feats,names=feature_stack(rgb)
        h,w=rgb.shape[:2]
        profile=src.profile.copy()

    payload=json.loads(args.labels.read_text(encoding="utf-8"))
    xs,ys,y=parse_labels(payload,w,h)
    if len(y)<20 or len(np.unique(y))<2:
        raise ValueError("Need at least 20 usable labels including both tree and non-tree classes")
    counts=np.bincount(y,minlength=2)
    if counts.min()<5:
        raise ValueError("Need at least 5 labels in each class")

    X=feats[ys,xs]
    clf=RandomForestClassifier(
        n_estimators=args.trees,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )

    folds=min(5,int(counts.min()))
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=42)
    cv_scores=cross_val_score(clf,X,y,cv=cv,scoring="balanced_accuracy",n_jobs=-1)
    clf.fit(X,y)

    flat=feats.reshape(-1,feats.shape[-1])
    prob=np.empty(flat.shape[0],dtype=np.float32)
    chunk=250000
    for start in range(0,len(flat),chunk):
        end=min(start+chunk,len(flat))
        prob[start:end]=clf.predict_proba(flat[start:end])[:,1]
    prob=prob.reshape(h,w)
    mask=(prob>=args.threshold).astype(np.uint8)

    profile_prob=profile.copy();profile_prob.update(count=1,dtype="float32",nodata=None,compress="deflate")
    profile_mask=profile.copy();profile_mask.update(count=1,dtype="uint8",nodata=255,compress="deflate")
    args.output_probability.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(args.output_probability,"w",**profile_prob) as dst: dst.write(prob,1)
    with rasterio.open(args.output_mask,"w",**profile_mask) as dst: dst.write(mask,1)

    base=np.clip(rgb*255,0,255).astype(np.uint8)
    overlay=base.copy()
    sel=mask.astype(bool)
    overlay[sel,0]=(0.45*overlay[sel,0]).astype(np.uint8)
    overlay[sel,1]=np.clip(0.55*overlay[sel,1]+115,0,255).astype(np.uint8)
    overlay[sel,2]=(0.45*overlay[sel,2]).astype(np.uint8)
    Image.fromarray(overlay,mode="RGB").save(args.output_preview)

    imp=sorted(zip(names,clf.feature_importances_.tolist()),key=lambda x:x[1],reverse=True)
    result={
      "method":"human_calibrated_rgb_random_forest_v1",
      "status":"model_estimate_requires_independent_validation",
      "source_image":str(args.image),
      "labels_file":str(args.labels),
      "n_labels":int(len(y)),
      "tree_labels":int((y==1).sum()),
      "non_tree_labels":int((y==0).sum()),
      "cv_balanced_accuracy_mean":float(cv_scores.mean()),
      "cv_balanced_accuracy_sd":float(cv_scores.std()),
      "threshold":args.threshold,
      "predicted_canopy_fraction":float(mask.mean()),
      "feature_importance":[{"feature":n,"importance":float(v)} for n,v in imp],
      "warning":"Do not report predicted canopy fraction as final UTC until independent spatial validation."
    }
    args.output_json.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"Labels: {len(y)} (tree={(y==1).sum()}, non-tree={(y==0).sum()})")
    print(f"CV balanced accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"Predicted canopy fraction: {100*mask.mean():.2f}%")
    print(f"Wrote {args.output_probability}")
    print(f"Wrote {args.output_mask}")
    print(f"Wrote {args.output_preview}")
    print(f"Wrote {args.output_json}")
    print("STATUS: model estimate — independent validation still required")


if __name__=="__main__":
    main()
