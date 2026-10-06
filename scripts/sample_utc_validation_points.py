"""Create a spatially distributed independent UTC validation sample.

The sampler reads a canopy probability raster, excludes pixels near training
clicks, and selects deterministic validation points across spatial blocks and
predicted classes. Human reviewers should label these points on the original
RGB imagery without seeing the model class/probability.

This is intended for independent validation of the Fundidora UTC pilot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


def load_training_points(path: Path):
    payload=json.loads(path.read_text(encoding="utf-8"))
    pts=[]
    for item in payload.get("labels",[]):
        if item.get("label") in {"tree","non"}:
            pts.append((int(item["x"]),int(item["y"])))
    return pts


def exclusion_mask(height:int,width:int,points:list[tuple[int,int]],radius:int)->np.ndarray:
    mask=np.zeros((height,width),dtype=bool)
    for x,y in points:
        x0=max(0,x-radius); x1=min(width,x+radius+1)
        y0=max(0,y-radius); y1=min(height,y+radius+1)
        yy,xx=np.ogrid[y0:y1,x0:x1]
        local=(xx-x)**2+(yy-y)**2 <= radius*radius
        mask[y0:y1,x0:x1] |= local
    return mask


def sample_points(prob:np.ndarray,excluded:np.ndarray,n:int,blocks:int,threshold:float,seed:int):
    rng=np.random.default_rng(seed)
    h,w=prob.shape
    candidates=[]
    per_group=max(1,int(np.ceil(n/(blocks*blocks*2))))
    for br in range(blocks):
        y0=round(br*h/blocks); y1=round((br+1)*h/blocks)
        for bc in range(blocks):
            x0=round(bc*w/blocks); x1=round((bc+1)*w/blocks)
            sub=prob[y0:y1,x0:x1]
            ex=excluded[y0:y1,x0:x1]
            for cls in (0,1):
                eligible=(sub>=threshold) if cls==1 else (sub<threshold)
                yy,xx=np.where(eligible & ~ex & np.isfinite(sub))
                if len(xx)==0:
                    continue
                k=min(per_group,len(xx))
                choose=rng.choice(len(xx),size=k,replace=False)
                for idx in choose:
                    x=int(x0+xx[idx]); y=int(y0+yy[idx])
                    candidates.append({
                        "x":x,"y":y,
                        "block_row":br,"block_col":bc,
                        "model_probability":float(prob[y,x]),
                        "model_class":"tree" if prob[y,x]>=threshold else "non",
                    })
    rng.shuffle(candidates)
    return candidates[:n]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--probability",type=Path,required=True)
    p.add_argument("--training-labels",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--n",type=int,default=200)
    p.add_argument("--blocks",type=int,default=4)
    p.add_argument("--threshold",type=float,default=0.5)
    p.add_argument("--exclude-radius",type=int,default=15)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()

    with rasterio.open(args.probability) as src:
        prob=src.read(1).astype(np.float32)
        h,w=prob.shape

    training=load_training_points(args.training_labels)
    excluded=exclusion_mask(h,w,training,args.exclude_radius)
    pts=sample_points(prob,excluded,args.n,args.blocks,args.threshold,args.seed)
    if len(pts)<min(args.n,50):
        raise RuntimeError(f"Only {len(pts)} validation points could be sampled")

    for i,item in enumerate(pts,1):
        item["id"]=i
        item["human_label"]=None

    payload={
        "schema_version":1,
        "task":"utc_independent_validation",
        "image_width_px":w,
        "image_height_px":h,
        "threshold":args.threshold,
        "exclude_radius_px":args.exclude_radius,
        "spatial_blocks":args.blocks,
        "seed":args.seed,
        "points":pts,
        "review_instruction":"Label on original RGB only. Do not inspect model_probability/model_class while reviewing."
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Validation points: {len(pts)}")
    print(f"Training points excluded within radius: {args.exclude_radius}px")


if __name__=="__main__":
    main()
