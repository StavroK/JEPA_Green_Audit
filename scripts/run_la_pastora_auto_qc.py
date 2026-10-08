"""Run the complete La Pastora auto-AOI visual QC gate.

This orchestrates:
1. Sentinel-2 two-date baseline selection,
2. aligned RGB export,
3. NDVI mask export for both years,
4. Dynamic World classification/confidence export for both years,
5. one 6-panel RGB | NDVI | Dynamic World montage.

It does not adopt the AOI or create human labels. The montage must be reviewed
first. Earth Engine authentication is required only for the Dynamic World step.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd:list[str]):
    print("\n$ "+" ".join(cmd),flush=True)
    subprocess.run(cmd,check=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--project",default="mtygreenaudit")
    p.add_argument("--aoi",type=Path,default=Path("config/aoi_la_pastora_green_core_auto.geojson"))
    p.add_argument("--output-root",type=Path,default=Path("outputs/la_pastora_auto_qc"))
    p.add_argument("--site",default="la_pastora_auto")
    p.add_argument("--min-top-prob",type=float,default=0.5)
    args=p.parse_args()

    py=sys.executable
    root=args.output_root
    rgb_dir=root/"rgb"
    ndvi_dir=root/"ndvi"
    dw_dir=root/"dynamic_world"
    baseline=root/"sentinel2_baseline.json"
    montage=root/"rgb_ndvi_dynamic_world_qc.png"

    run([py,"scripts/fetch_sentinel2.py",
         "--aoi",str(args.aoi),
         "--output",str(baseline)])

    run([py,"scripts/export_rgb_tiles.py",
         "--aoi",str(args.aoi),
         "--baseline-json",str(baseline),
         "--site",args.site,
         "--out-dir",str(rgb_dir)])

    for year in ("2025","2026"):
        run([py,"scripts/export_ndvi_qc.py",
             "--aoi",str(args.aoi),
             "--baseline-json",str(baseline),
             "--site",args.site,
             "--year",year,
             "--output",str(ndvi_dir/f"{args.site}_{year}_ndvi.png")])

    periods={
        "2025":("2025-08-01","2025-09-30"),
        "2026":("2026-08-01","2026-09-30"),
    }
    for year,(start,end) in periods.items():
        run([py,"scripts/export_dynamic_world_qc.py",
             "--aoi",str(args.aoi),
             "--start",start,
             "--end",end,
             "--project",args.project,
             "--site",args.site,
             "--year",year,
             "--min-top-prob",str(args.min_top_prob),
             "--output-dir",str(dw_dir)])

    run([py,"scripts/make_rgb_ndvi_dynamic_world_qc.py",
         "--site",args.site,
         "--rgb-dir",str(rgb_dir),
         "--ndvi-dir",str(ndvi_dir),
         "--dw-dir",str(dw_dir),
         "--output",str(montage)])

    print("\nQC gate complete.")
    print(f"Review montage: {montage}")
    print("Do not adopt the AOI or start human labeling until the montage is visually reviewed.")


if __name__=="__main__":
    main()
