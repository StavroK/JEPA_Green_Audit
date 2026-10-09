"""Ingest the INEGI G14C26A3 2007 1 m RGB BIL orthophoto.

The archive contains a 3-band uint8 BIL plus HDR/BLW/PRJ sidecars. The supplied
AUX advertises one band, so this loader deliberately extracts the BIL/HDR/BLW/
PRJ files while excluding AUX to preserve the correct 3-band interpretation.

The result is cropped to a WGS84 GeoJSON AOI and exported as a georeferenced
3-band GeoTIFF plus RGB PNG for visual-model experiments.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.warp import transform_geom

DEFAULT_SOURCE = Path("data/raw/inegi/889463305477_b.zip")
DEFAULT_AOI = Path("config/aoi_fundidora.geojson")
DEFAULT_TIF = Path("data/interim/inegi_orthophoto/fundidora_2007_1m_rgb.tif")
DEFAULT_PNG = Path("data/interim/inegi_orthophoto/fundidora_2007_1m_rgb.png")
DEFAULT_META = Path("data/interim/inegi_orthophoto/fundidora_2007_1m_rgb.json")

# INEGI sidecar metadata: UTM zone 14, ITRF92, GRS80, meters.
SOURCE_CRS = CRS.from_proj4("+proj=utm +zone=14 +north +ellps=GRS80 +units=m +no_defs")


def load_aoi(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        if not payload.get("features"):
            raise ValueError("AOI FeatureCollection is empty")
        return payload["features"][0]["geometry"]
    if payload.get("type") == "Feature":
        return payload["geometry"]
    return payload


def extract_bil_dataset(archive_path: Path, output_dir: Path) -> Path:
    """Extract BIL + required sidecars, intentionally excluding misleading AUX."""
    wanted_ext = {".bil", ".hdr", ".blw", ".prj"}
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name for name in archive.namelist()
            if Path(name).suffix.lower() in wanted_ext
        ]
        bils = [name for name in members if Path(name).suffix.lower() == ".bil"]
        if len(bils) != 1:
            raise ValueError(f"Expected exactly one BIL in {archive_path}; found {len(bils)}")
        stem = Path(bils[0]).stem.lower()
        for name in members:
            if Path(name).stem.lower() == stem:
                archive.extract(name, output_dir)
        return output_dir / bils[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--sheet", default="G14C26A3")
    parser.add_argument("--acquisition-year", type=int, default=2007)
    parser.add_argument("--resolution-m", type=float, default=1.0)
    parser.add_argument("--aoi", type=Path, default=DEFAULT_AOI)
    parser.add_argument("--output-tif", type=Path, default=DEFAULT_TIF)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    args = parser.parse_args()

    geometry_wgs84 = load_aoi(args.aoi)
    geometry_src = transform_geom("EPSG:4326", SOURCE_CRS, geometry_wgs84)

    with tempfile.TemporaryDirectory() as td:
        bil_path = extract_bil_dataset(args.source, Path(td))

        with rasterio.open(bil_path) as src:
            if src.count != 3:
                raise ValueError(
                    f"Expected 3 RGB bands after excluding AUX; raster reports {src.count}"
                )
            if src.dtypes[0] != "uint8":
                raise ValueError(f"Expected uint8 orthophoto; got {src.dtypes}")

            cropped, transform = mask(
                src,
                [geometry_src],
                crop=True,
                indexes=[1, 2, 3],
                filled=True,
            )

            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                count=3,
                dtype="uint8",
                height=cropped.shape[1],
                width=cropped.shape[2],
                transform=transform,
                crs=SOURCE_CRS,
                compress="deflate",
                photometric="RGB",
            )

    args.output_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output_tif, "w", **profile) as dst:
        dst.write(cropped)

    rgb = np.moveaxis(cropped, 0, -1)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(args.output_png)

    meta = {
        "source_archive": str(args.source),
        "source_sheet": args.sheet,
        "source_acquisition_year": args.acquisition_year,
        "source_resolution_m": args.resolution_m,
        "source_layout": "BIL",
        "source_bands": 3,
        "source_dtype": "uint8",
        "source_crs": SOURCE_CRS.to_string(),
        "aoi": str(args.aoi),
        "output_tif": str(args.output_tif),
        "output_png": str(args.output_png),
        "output_shape_hwc": list(rgb.shape),
        "temporal_guardrail": (
            "2007 RGB is for spatial-resolution/representation experiments. "
            "Do not evaluate it against 2025/2026 vegetation labels as if contemporaneous."
        ),
    }
    args.metadata.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {args.output_tif}")
    print(f"Wrote {args.output_png}")
    print(f"Wrote {args.metadata}")
    print(f"RGB shape: {rgb.shape}")
    print(
        f"Acquisition year: {args.acquisition_year} | "
        f"GSD: {args.resolution_m} m | bands: 3 | sheet: {args.sheet}"
    )


if __name__ == "__main__":
    main()
