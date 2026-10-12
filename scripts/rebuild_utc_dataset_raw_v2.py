"""Rebuild the Fundidora UTC dense dataset with raw-consistent RGB inputs.

This reuses the exact v1 patch coordinates, split assignments, and human masks,
but re-extracts RGB directly from the source orthophoto without per-patch
percentile stretching. The goal is a v2 benchmark whose source and target sites
can both be evaluated with raw RGB inputs.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio


def raw_rgb(arr: np.ndarray) -> np.ndarray:
    rgb=arr[:3]
    if rgb.dtype==np.uint8:
        return np.moveaxis(rgb,0,-1)
    if np.issubdtype(rgb.dtype,np.integer):
        info=np.iinfo(rgb.dtype)
        x=(rgb.astype(np.float32)-info.min)/max(info.max-info.min,1)
        return np.moveaxis((np.clip(x,0,1)*255).astype(np.uint8),0,-1)
    vals=rgb[np.isfinite(rgb)]
    if vals.size==0:
        return np.zeros((rgb.shape[1],rgb.shape[2],3),dtype=np.uint8)
    lo=float(vals.min()); hi=float(vals.max())
    if hi<=lo: hi=lo+1.0
    x=(np.clip((rgb-lo)/(hi-lo),0,1)*255).astype(np.uint8)
    return np.moveaxis(x,0,-1)


def rebuild(v1_dir:Path, source_image:Path, output_dir:Path) -> dict:
    dm=json.loads((v1_dir/"dataset_manifest.json").read_text(encoding="utf-8"))
    images_out=output_dir/"images"
    masks_out=output_dir/"masks"
    images_out.mkdir(parents=True,exist_ok=True)
    masks_out.mkdir(parents=True,exist_ok=True)

    records=[]
    with rasterio.open(source_image) as src:
        if src.count<3:
            raise ValueError("Source image must contain at least 3 bands")
        for rec in dm["records"]:
            x=int(rec["x"]); y=int(rec["y"])
            w=int(rec.get("width",dm.get("patch_size",256)))
            h=int(rec.get("height",dm.get("patch_size",256)))
            if x<0 or y<0 or x+w>src.width or y+h>src.height:
                raise ValueError(f"Patch {rec['id']} window outside source raster")
            win=rasterio.windows.Window(x,y,w,h)
            arr=src.read([1,2,3],window=win)
            rgb=raw_rgb(arr)
            if rgb.shape[:2]!=(h,w):
                raise RuntimeError(f"Patch {rec['id']} shape mismatch: {rgb.shape}")
            Image.fromarray(rgb,mode="RGB").save(images_out/rec["file"])

            mask_name=rec["mask_file"]
            src_mask=v1_dir/"masks"/mask_name
            if not src_mask.exists():
                raise FileNotFoundError(src_mask)
            shutil.copy2(src_mask,masks_out/mask_name)
            records.append(dict(rec))

        source_meta={
            "width":src.width,
            "height":src.height,
            "count":src.count,
            "dtype":src.dtypes[0],
            "crs":str(src.crs),
        }

    out=dict(dm)
    out["schema_version"]=2
    out["source_dataset_v1"]=str(v1_dir)
    out["source_image"]=str(source_image)
    out["preprocessing"]="raw_rgb_no_per_patch_stretch"
    out["preprocessing_guardrail"]="RGB values are re-extracted from the source orthophoto using the original v1 coordinates. Human masks and spatial splits are unchanged."
    out["source_raster_metadata"]=source_meta
    out["records"]=records

    output_dir.mkdir(parents=True,exist_ok=True)
    manifest=output_dir/"dataset_manifest.json"
    manifest.write_text(json.dumps(out,indent=2),encoding="utf-8")
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v1-dataset-dir",type=Path,required=True)
    p.add_argument("--source-image",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    dm=rebuild(args.v1_dataset_dir,args.source_image,args.output_dir)
    print(f"Wrote {args.output_dir/'dataset_manifest.json'}")
    print(f"Patches: {len(dm['records'])}")
    print(f"Preprocessing: {dm['preprocessing']}")
    print("Masks/splits preserved from v1; RGB re-extracted raw from source image.")


if __name__=="__main__":
    main()
