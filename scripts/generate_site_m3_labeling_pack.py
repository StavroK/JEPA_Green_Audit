"""Generate a blinded RGB-only M3b human-labeling pack for any site.

The pack uses a regular review grid over each RGB image. Labels are intentionally
independent of NDVI, Dynamic World and SCL.

For La Pastora we use a 16x16 grid to align with the I-JEPA 16x16 patch-token
grid after 224x224 encoding. This is a coarse vegetation-presence benchmark,
not Urban Tree Canopy crown delineation.

Allowed labels:
- vegetation
- non_vegetation
- uncertain
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def grid_edges(length:int,cells:int)->np.ndarray:
    return np.linspace(0,length,cells+1,dtype=int)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rgb-dir",type=Path,required=True)
    p.add_argument("--site",required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--grid",type=int,default=16)
    p.add_argument("--years",nargs="+",default=["2025","2026"])
    p.add_argument("--overview-scale",type=int,default=8)
    args=p.parse_args()

    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for year in args.years:
        src=args.rgb_dir/f"{args.site}_{year}_rgb.png"
        image=Image.open(src).convert("RGB")
        width,height=image.size
        xe=grid_edges(width,args.grid); ye=grid_edges(height,args.grid)

        year_dir=args.output_dir/year
        year_dir.mkdir(parents=True,exist_ok=True)

        overview=image.resize(
            (width*args.overview_scale,height*args.overview_scale),
            Image.Resampling.NEAREST
        )
        draw=ImageDraw.Draw(overview)
        for x in xe:
            draw.line((x*args.overview_scale,0,x*args.overview_scale,height*args.overview_scale),fill="white",width=1)
        for y in ye:
            draw.line((0,y*args.overview_scale,width*args.overview_scale,y*args.overview_scale),fill="white",width=1)
        overview_path=args.output_dir/f"{args.site}_{year}_review_grid_{args.grid}x{args.grid}.png"
        overview.save(overview_path)

        for row in range(args.grid):
            for col in range(args.grid):
                box=(int(xe[col]),int(ye[row]),int(xe[col+1]),int(ye[row+1]))
                patch=image.crop(box)
                patch_name=f"{year}_r{row:02d}_c{col:02d}.png"
                patch_path=year_dir/patch_name
                patch.resize((224,224),Image.Resampling.NEAREST).save(patch_path)
                rows.append({
                    "year":year,
                    "row":row,
                    "col":col,
                    "patch_path":str(patch_path),
                    "source_path":str(src),
                    "grid_rows":args.grid,
                    "grid_cols":args.grid,
                    "label":"",
                    "reviewer":"",
                    "notes":"",
                })

    csv_path=args.output_dir/f"{args.site}_patch_labels_{args.grid}x{args.grid}.csv"
    fields=["year","row","col","patch_path","source_path","grid_rows","grid_cols","label","reviewer","notes"]
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

    (args.output_dir/"README.md").write_text(
        f"# {args.site} M3b RGB-only labeling pack — {args.grid}x{args.grid}\n\n"
        "Label each cell using RGB only. Do not consult NDVI, Dynamic World, SCL, "
        "or any model prediction while labeling.\n\n"
        "Allowed labels: vegetation, non_vegetation, uncertain.\n\n"
        "Vegetation means vegetation is the dominant visible cover in the cell. "
        "Use uncertain for genuinely mixed or visually ambiguous cells. "
        "This benchmark measures coarse visible vegetation presence at Sentinel-2 "
        "scale; it is not tree-crown delineation or biological tree health.\n",
        encoding="utf-8"
    )
    print(f"Wrote {csv_path}")
    print(f"Generated {len(rows)} review cells")


if __name__=="__main__":
    main()
