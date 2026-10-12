"""Find which locally downloaded INEGI orthophoto archives cover an AOI.

Scans ZIP archives, opens each BIL dataset with its georeferencing sidecars,
transforms the WGS84 AOI into the INEGI source CRS, and reports overlap/coverage.
No imagery is written.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from rasterio.features import bounds as geom_bounds

try:
    from scripts.ingest_inegi_orthophoto_bil import extract_bil_dataset, load_aoi, SOURCE_CRS
except ModuleNotFoundError:
    from ingest_inegi_orthophoto_bil import extract_bil_dataset, load_aoi, SOURCE_CRS


def intersects(a,b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def contains(outer,inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def inspect_archive(archive:Path,aoi_wgs84:dict)->dict:
    aoi_src=transform_geom("EPSG:4326",SOURCE_CRS,aoi_wgs84)
    aoi_bounds=geom_bounds(aoi_src)
    with tempfile.TemporaryDirectory() as td:
        bil=extract_bil_dataset(archive,Path(td))
        with rasterio.open(bil) as src:
            rb=(src.bounds.left,src.bounds.bottom,src.bounds.right,src.bounds.top)
            return {
                "archive":str(archive),
                "raster_bounds":list(map(float,rb)),
                "aoi_bounds_source_crs":list(map(float,aoi_bounds)),
                "intersects":intersects(rb,aoi_bounds),
                "fully_contains_aoi":contains(rb,aoi_bounds),
                "width":src.width,
                "height":src.height,
                "bands":src.count,
                "dtype":src.dtypes[0],
            }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--aoi",type=Path,required=True)
    p.add_argument("--archive-dir",type=Path,default=Path("data/raw/inegi"))
    p.add_argument("--output",type=Path)
    args=p.parse_args()

    geom=load_aoi(args.aoi)
    archives=sorted(args.archive_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"No ZIP archives found under {args.archive_dir}")

    rows=[]
    for archive in archives:
        try:
            row=inspect_archive(archive,geom)
        except Exception as e:
            row={"archive":str(archive),"error":f"{type(e).__name__}: {e}"}
        rows.append(row)

    covering=[r for r in rows if r.get("fully_contains_aoi")]
    overlapping=[r for r in rows if r.get("intersects") and not r.get("fully_contains_aoi")]

    print(f"AOI: {args.aoi}")
    print(f"Archives scanned: {len(rows)}")
    for r in rows:
        if r.get("fully_contains_aoi"):
            status="FULL COVERAGE"
        elif r.get("intersects"):
            status="PARTIAL OVERLAP"
        elif "error" in r:
            status="ERROR"
        else:
            status="NO OVERLAP"
        print(f"{status:15s} {r['archive']}")
        if "error" in r:
            print(f"  {r['error']}")

    result={"aoi":str(args.aoi),"archives":rows,
            "full_coverage":[r["archive"] for r in covering],
            "partial_overlap":[r["archive"] for r in overlapping]}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(f"Wrote {args.output}")

    if not covering:
        print("No downloaded archive fully covers this AOI.")
        print("Acquire the required INEGI orthophoto sheet before ingestion.")


if __name__=="__main__":
    main()
