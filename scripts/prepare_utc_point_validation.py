"""Prepare deterministic spatially distributed UTC validation points.

This creates one jittered point per regular grid cell over a high-resolution RGB
reference image. The points are selected before human review and without model
predictions, so they can support independent cross-site validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def make_points(width:int,height:int,rows:int,cols:int,seed:int,margin_fraction:float=0.15):
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")
    rng=np.random.default_rng(seed)
    x_edges=np.linspace(0,width,cols+1,dtype=int)
    y_edges=np.linspace(0,height,rows+1,dtype=int)
    points=[]
    pid=1
    for r in range(rows):
        for c in range(cols):
            x0,x1=int(x_edges[c]),int(x_edges[c+1])
            y0,y1=int(y_edges[r]),int(y_edges[r+1])
            mx=max(1,int(round((x1-x0)*margin_fraction)))
            my=max(1,int(round((y1-y0)*margin_fraction)))
            lo_x=min(x1-1,x0+mx); hi_x=max(lo_x,min(x1-1,x1-mx-1))
            lo_y=min(y1-1,y0+my); hi_y=max(lo_y,min(y1-1,y1-my-1))
            x=int(rng.integers(lo_x,hi_x+1)) if hi_x>lo_x else lo_x
            y=int(rng.integers(lo_y,hi_y+1)) if hi_y>lo_y else lo_y
            points.append({
                "id":pid,"grid_row":r,"grid_col":c,"x":x,"y":y,
                "human_label":"","detail_label":"","reviewer":"","notes":""
            })
            pid+=1
    return points


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--rows",type=int,default=15)
    p.add_argument("--cols",type=int,default=16)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--site",default="la_pastora")
    p.add_argument("--date-label",default="2007")
    args=p.parse_args()

    with Image.open(args.image) as im:
        width,height=im.size

    points=make_points(width,height,args.rows,args.cols,args.seed)
    payload={
        "schema_version":1,
        "task":"utc_independent_point_validation",
        "site":args.site,
        "date_label":args.date_label,
        "image":str(args.image),
        "width_px":width,
        "height_px":height,
        "sampling":{
            "strategy":"one deterministic jittered point per regular grid cell",
            "rows":args.rows,"cols":args.cols,"seed":args.seed,
            "model_blind":True,
            "n_points":len(points)
        },
        "label_definition":{
            "tree":"point falls on visible tree-crown canopy",
            "non":"point falls on non-tree surface",
            "uncertain":"tree/non-tree cannot be determined confidently from RGB"
        },
        "points":points
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Image: {width}x{height}")
    print(f"Validation points: {len(points)} ({args.rows}x{args.cols})")


if __name__=="__main__":
    main()
