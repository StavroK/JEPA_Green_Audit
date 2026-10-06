"""Prepare spatially disjoint RGB patches for UTC semantic-segmentation annotation."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio

def block_split(block_row:int, block_col:int, blocks:int)->str:
    idx=block_row*blocks+block_col
    mod=idx%5
    return "test" if mod==0 else ("val" if mod==1 else "train")

def patch_origins(width:int,height:int,patch:int,n:int,blocks:int,seed:int):
    if patch>width or patch>height:
        raise ValueError("Patch size exceeds image dimensions")
    rng=np.random.default_rng(seed)
    per_block=int(math.ceil(n/(blocks*blocks)))
    out=[]; seen=set()
    for br in range(blocks):
        by0=round(br*height/blocks); by1=round((br+1)*height/blocks)
        for bc in range(blocks):
            bx0=round(bc*width/blocks); bx1=round((bc+1)*width/blocks)
            xlo=max(0,bx0); xhi=min(width-patch,bx1-patch)
            ylo=max(0,by0); yhi=min(height-patch,by1-patch)
            if xhi<xlo:
                xlo=max(0,min(width-patch,(bx0+bx1-patch)//2)); xhi=xlo
            if yhi<ylo:
                ylo=max(0,min(height-patch,(by0+by1-patch)//2)); yhi=ylo
            attempts=0; added=0
            while added<per_block and attempts<per_block*50:
                attempts+=1
                x=int(rng.integers(xlo,xhi+1)) if xhi>xlo else int(xlo)
                y=int(rng.integers(ylo,yhi+1)) if yhi>ylo else int(ylo)
                if (x,y) in seen: continue
                seen.add((x,y))
                out.append({"x":x,"y":y,"block_row":br,"block_col":bc,"split":block_split(br,bc,blocks)})
                added+=1
    rng.shuffle(out)
    return out[:n]

def stretch_rgb(arr):
    rgb=arr[:3].astype(np.float32); out=np.zeros_like(rgb,dtype=np.uint8)
    for b in range(3):
        band=rgb[b]; vals=band[np.isfinite(band)]
        if vals.size==0: continue
        lo,hi=np.percentile(vals,[2,98])
        if hi<=lo: hi=lo+1.0
        out[b]=(np.clip((band-lo)/(hi-lo),0,1)*255).astype(np.uint8)
    return np.moveaxis(out,0,-1)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--patch-size",type=int,default=256)
    p.add_argument("--n",type=int,default=64)
    p.add_argument("--blocks",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    patch_dir=args.output_dir/"images"; patch_dir.mkdir(exist_ok=True)
    with rasterio.open(args.image) as src:
        if src.count<3: raise ValueError("Expected RGB image with at least 3 bands")
        origins=patch_origins(src.width,src.height,args.patch_size,args.n,args.blocks,args.seed)
        records=[]
        for i,rec in enumerate(origins,1):
            win=rasterio.windows.Window(rec["x"],rec["y"],args.patch_size,args.patch_size)
            rgb=stretch_rgb(src.read([1,2,3],window=win))
            name=f"patch_{i:03d}_{rec['split']}_b{rec['block_row']}{rec['block_col']}.png"
            Image.fromarray(rgb,mode="RGB").save(patch_dir/name)
            records.append({"id":i,"file":name,"x":rec["x"],"y":rec["y"],"width":args.patch_size,"height":args.patch_size,**{k:rec[k] for k in ("block_row","block_col","split")}})
    counts={k:sum(r["split"]==k for r in records) for k in ("train","val","test")}
    manifest={"schema_version":1,"task":"utc_dense_patch_annotation","source_image":str(args.image),"patch_size":args.patch_size,"spatial_blocks":args.blocks,"seed":args.seed,"n_patches":len(records),"split_counts":counts,"patches":records}
    out=args.output_dir/"manifest.json"
    out.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Wrote {out}"); print(f"Patch images: {patch_dir}"); print(f"Patches: {len(records)}"); print(f"Split counts: {counts}")
if __name__=="__main__": main()
