"""Prepare a high-resolution RGB annotation pack for Urban Tree Canopy labeling.

This creates a georeferenced AOI crop plus a large PNG preview for human canopy
polygon annotation. It does not infer canopy automatically.

Example:
    python scripts/prepare_utc_annotation_pack.py \
      --image data/raw/inegi/FUNDIDORA_2007_ORTHO.tif \
      --aoi config/aoi_fundidora.geojson \
      --site fundidora \
      --date-label 2007 \
      --output-dir outputs/utc/fundidora_2007
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom


def load_geometry(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features") or []
        if len(features) != 1:
            raise ValueError("Expected a single AOI feature")
        return features[0]["geometry"]
    if payload.get("type") == "Feature":
        return payload["geometry"]
    raise ValueError("Expected GeoJSON Feature or single-feature FeatureCollection")


def stretch_rgb(arr: np.ndarray, low=2.0, high=98.0) -> np.ndarray:
    """Percentile-stretch CxHxW RGB to uint8 HxWx3 for human review."""
    if arr.shape[0] < 3:
        raise ValueError(f"Expected >=3 bands; got {arr.shape}")
    rgb = arr[:3].astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.uint8)
    for b in range(3):
        band = rgb[b]
        finite = np.isfinite(band)
        vals = band[finite]
        if vals.size == 0:
            continue
        lo, hi = np.percentile(vals, [low, high])
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((band - lo) / (hi - lo), 0, 1)
        out[b] = (scaled * 255).astype(np.uint8)
    return np.moveaxis(out, 0, -1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--aoi", type=Path, required=True)
    p.add_argument("--site", required=True)
    p.add_argument("--date-label", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    geom_wgs84 = load_geometry(args.aoi)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    crop_tif = args.output_dir / f"{args.site}_{args.date_label}_rgb_crop.tif"
    preview_png = args.output_dir / f"{args.site}_{args.date_label}_rgb_preview.png"
    config_json = args.output_dir / f"{args.site}_{args.date_label}_annotation_config.json"

    with rasterio.open(args.image) as src:
        if src.crs is None:
            raise ValueError("Source orthophoto has no CRS")
        geom_src = transform_geom("EPSG:4326", src.crs, geom_wgs84)
        cropped, transform = rio_mask(src, [geom_src], crop=True, filled=True)
        profile = src.profile.copy()
        profile.update(
            height=cropped.shape[1],
            width=cropped.shape[2],
            transform=transform,
            count=min(cropped.shape[0], 3),
        )
        with rasterio.open(crop_tif, "w", **profile) as dst:
            dst.write(cropped[: profile["count"]])

        preview = stretch_rgb(cropped)
        Image.fromarray(preview, mode="RGB").save(preview_png)

        cfg = {
            "schema_version": 1,
            "task": "urban_tree_canopy_polygon_annotation",
            "site": args.site,
            "date_label": args.date_label,
            "source_image": str(args.image),
            "aoi": str(args.aoi),
            "crop_tif": str(crop_tif),
            "preview_png": str(preview_png),
            "width_px": int(preview.shape[1]),
            "height_px": int(preview.shape[0]),
            "crs": src.crs.to_string(),
            "transform": list(transform)[:6],
            "annotation_definition": (
                "Draw polygons over visible tree-crown canopy only. Exclude grass, "
                "shrubs where distinguishable, buildings, roads, bare soil, water, "
                "and ambiguous shadow unless crown structure is visually supported."
            ),
            "limitations": [
                "This is a human interpretation of RGB imagery, not a field tree inventory.",
                "Occluded or shadowed crowns may be uncertain.",
                "UTC is crown cover area, not tree count.",
            ],
        }
        config_json.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"Wrote {crop_tif}")
    print(f"Wrote {preview_png}")
    print(f"Wrote {config_json}")
    print(f"Preview size: {preview.shape[1]}x{preview.shape[0]}")


if __name__ == "__main__":
    main()
