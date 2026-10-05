"""Rasterize human UTC canopy polygons from pixel coordinates to GeoTIFF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize


def pixel_polygon_to_map(coords, transform):
    return [list(transform * (float(x), float(y))) for x, y in coords]


def annotations_to_shapes(payload: dict, transform):
    shapes = []
    for polygon in payload.get("polygons", []):
        coords = polygon.get("points") or []
        if len(coords) < 3:
            continue
        mapped = pixel_polygon_to_map(coords, transform)
        if mapped[0] != mapped[-1]:
            mapped.append(mapped[0])
        geom = {"type": "Polygon", "coordinates": [mapped]}
        shapes.append((geom, 1))
    return shapes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--crop-tif", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output-mask", type=Path, required=True)
    args = p.parse_args()

    payload = json.loads(args.annotations.read_text(encoding="utf-8"))

    with rasterio.open(args.crop_tif) as src:
        shapes = annotations_to_shapes(payload, src.transform)
        mask = rasterize(
            shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            default_value=1,
            dtype="uint8",
            all_touched=False,
        )
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=255, compress="deflate")

    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output_mask, "w", **profile) as dst:
        dst.write(mask, 1)

    canopy_px = int(mask.sum())
    total_px = int(mask.size)
    print(f"Canopy pixels: {canopy_px}/{total_px} ({100*canopy_px/total_px:.2f}%)")
    print(f"Wrote {args.output_mask}")


if __name__ == "__main__":
    main()
