"""Crop a georeferenced RGB INEGI orthophoto to a GeoJSON AOI.

The source may be any GDAL/rasterio-readable georeferenced raster with at least
three bands. This script deliberately does not scrape or redistribute INEGI
imagery; it prepares a user-obtained source file for the reproducible benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.mask import mask
from rasterio.warp import transform_geom


def load_aoi(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        if not payload.get("features"):
            raise ValueError("AOI FeatureCollection is empty")
        return payload["features"][0]["geometry"]
    if payload.get("type") == "Feature":
        return payload["geometry"]
    return payload


def to_uint8_rgb(data: np.ndarray) -> np.ndarray:
    """Robustly scale first three bands to uint8 for visual-model input."""
    if data.shape[0] < 3:
        raise ValueError("source raster must contain at least three bands")

    rgb = data[:3].astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.uint8)
    for b in range(3):
        band = rgb[b]
        valid = np.isfinite(band)
        if not valid.any():
            continue
        lo, hi = np.percentile(band[valid], (2, 98))
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
        out[b] = np.round(scaled * 255).astype(np.uint8)
    return np.moveaxis(out, 0, -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--aoi", type=Path, required=True)
    parser.add_argument(
        "--output-tif",
        type=Path,
        default=Path("data/interim/inegi/orthophoto_aoi.tif"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("data/interim/inegi/orthophoto_aoi_rgb.png"),
    )
    args = parser.parse_args()

    geometry_wgs84 = load_aoi(args.aoi)

    with rasterio.open(args.source) as src:
        if src.crs is None:
            raise ValueError("source raster has no CRS/georeferencing")
        geometry_src = transform_geom("EPSG:4326", src.crs, geometry_wgs84)
        cropped, transform = mask(src, [geometry_src], crop=True)
        profile = src.profile.copy()
        profile.update(
            height=cropped.shape[1],
            width=cropped.shape[2],
            transform=transform,
            count=cropped.shape[0],
        )

    args.output_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output_tif, "w", **profile) as dst:
        dst.write(cropped)

    rgb = to_uint8_rgb(cropped)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(args.output_png)

    print(f"Wrote {args.output_tif}")
    print(f"Wrote {args.output_png}")
    print(f"RGB shape: {rgb.shape}")


if __name__ == "__main__":
    main()
