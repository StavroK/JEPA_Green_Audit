"""Rasterize dense UTC patch polygons into binary canopy masks.

Input annotations are exported by tools/utc_patch_mask_labeler.html.
Coordinates are clipped to the patch bounds so edge polygons remain valid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def clip_polygon(poly, width:int, height:int):
    out=[]
    for pt in poly:
        if len(pt)!=2:
            continue
        x=max(0,min(width-1,int(round(pt[0]))))
        y=max(0,min(height-1,int(round(pt[1]))))
        out.append((x,y))
    return out


def rasterize_patch(polygons, width:int, height:int)->np.ndarray:
    img=Image.new("L",(width,height),0)
    draw=ImageDraw.Draw(img)
    for poly in polygons:
        pts=clip_polygon(poly,width,height)
        if len(pts)>=3:
            draw.polygon(pts,fill=255)
    return np.asarray(img,dtype=np.uint8)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--annotations",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()

    ann=json.loads(args.annotations.read_text(encoding="utf-8"))
    man=json.loads(args.manifest.read_text(encoding="utf-8"))
    patch_size=int(man["patch_size"])

    masks=args.output_dir/"masks"
    masks.mkdir(parents=True,exist_ok=True)

    records=[]
    missing=[]
    incomplete=[]
    for patch in man["patches"]:
        key=str(patch["id"])
        rec=ann.get("patches",{}).get(key)
        if rec is None:
            missing.append(key)
            continue
        if not rec.get("complete",False):
            incomplete.append(key)
        mask=rasterize_patch(rec.get("polygons",[]),patch_size,patch_size)
        out_name=Path(patch["file"]).with_suffix(".png").name
        Image.fromarray(mask,mode="L").save(masks/out_name)
        canopy_px=int((mask>0).sum())
        total_px=int(mask.size)
        records.append({
            **patch,
            "mask_file":out_name,
            "canopy_pixels":canopy_px,
            "total_pixels":total_px,
            "canopy_fraction":canopy_px/total_px,
        })

    if missing:
        raise RuntimeError(f"Missing annotations for patch IDs: {missing}")
    if incomplete:
        raise RuntimeError(f"Incomplete annotations for patch IDs: {incomplete}")

    split_stats={}
    for split in ("train","val","test"):
        rows=[r for r in records if r["split"]==split]
        canopy=sum(r["canopy_pixels"] for r in rows)
        total=sum(r["total_pixels"] for r in rows)
        split_stats[split]={
            "patches":len(rows),
            "canopy_pixels":canopy,
            "total_pixels":total,
            "canopy_fraction":canopy/total if total else None,
        }

    dataset={
        "schema_version":1,
        "task":"utc_semantic_segmentation",
        "source_manifest":str(args.manifest),
        "source_annotations":str(args.annotations),
        "patch_size":patch_size,
        "records":records,
        "split_stats":split_stats,
        "mask_definition":"255=tree canopy, 0=non-canopy",
    }
    out=args.output_dir/"dataset_manifest.json"
    out.write_text(json.dumps(dataset,indent=2),encoding="utf-8")
    print(f"Wrote masks: {masks}")
    print(f"Wrote {out}")
    print(f"Patches: {len(records)}")
    for split,stats in split_stats.items():
        print(f"{split}: patches={stats['patches']} canopy_fraction={stats['canopy_fraction']:.3f}")


if __name__=="__main__":
    main()
