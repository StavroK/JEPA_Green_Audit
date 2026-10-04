"""Build a real two-date Sentinel-2 vegetation baseline for Parque Fundidora.

Usage:
    python scripts/fetch_sentinel2.py
    python scripts/fetch_sentinel2.py --cloud-cover 80 --candidates 5

The script queries public Earth Search metadata and window-reads only the AOI
from Cloud-Optimized GeoTIFF assets. It writes small JSON provenance/summary
files, not large raster files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from jepa_green_audit.sentinel2 import (
    EARTH_SEARCH,
    COLLECTION,
    load_aoi,
    search_scenes,
    summarize_item,
    valid_pixel_mask,
)
from jepa_green_audit.spectral import ndvi


DEFAULT_WINDOWS = {
    "2025": "2025-08-01/2025-09-29",
    "2026": "2026-08-01/2026-09-29",
}


def read_aoi(path: Path) -> tuple[dict, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features") or []
        if len(features) != 1:
            raise ValueError(
                f"Expected exactly one AOI feature in {path}; found {len(features)}"
            )
        feature = features[0]
    elif payload.get("type") == "Feature":
        feature = payload
    else:
        raise ValueError("AOI must be a GeoJSON Feature or single-feature FeatureCollection")

    coords = feature["geometry"]["coordinates"][0]
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return feature, [min(xs), min(ys), max(xs), max(ys)]


def choose_scene(bbox, window, cloud_cover, candidates):
    items = search_scenes(
        bbox,
        window,
        cloud_cover_lt=cloud_cover,
        limit=max(candidates, 1),
    )
    if not items:
        raise RuntimeError(f"No Sentinel-2 L2A scenes found for {window}")

    evaluated = []
    for item in items[:candidates]:
        arrays = load_aoi(item, bbox)
        valid = valid_pixel_mask(arrays["scl"])
        valid_fraction = float(valid.mean()) if valid.size else 0.0
        evaluated.append((valid_fraction, item, arrays))

    # For a small AOI, local usable-pixel fraction matters more than scene-level
    # cloud percentage. Break ties using lower scene-level cloud cover.
    evaluated.sort(
        key=lambda x: (
            -x[0],
            float(x[1].properties.get("eo:cloud_cover", 100.0)),
        )
    )
    return evaluated[0], evaluated


def analyze(item, arrays, vegetation_threshold):
    valid = valid_pixel_mask(arrays["scl"])
    index = ndvi(arrays["nir"], arrays["red"])
    finite = np.isfinite(index)
    usable = valid & finite
    vegetation = usable & (index >= vegetation_threshold)

    usable_count = int(usable.sum())
    vegetation_count = int(vegetation.sum())
    coverage = (
        100.0 * vegetation_count / usable_count if usable_count else float("nan")
    )

    summary = summarize_item(item)
    return {
        "scene": {
            "item_id": summary.item_id,
            "datetime": summary.datetime,
            "scene_cloud_cover_pct": summary.cloud_cover,
            "platform": summary.platform,
            "mgrs_tile": summary.mgrs_tile,
            "assets": {
                "red": summary.red_href,
                "nir": summary.nir_href,
                "scl": summary.scl_href,
            },
        },
        "grid": {
            "shape": list(arrays["red"].shape),
            "crs": arrays["crs"],
            "transform": list(arrays["transform"]),
        },
        "quality": {
            "usable_pixel_fraction": round(float(usable.mean()), 6),
            "usable_pixels": usable_count,
            "total_pixels": int(usable.size),
        },
        "vegetation": {
            "ndvi_threshold": vegetation_threshold,
            "vegetation_pixels": vegetation_count,
            "coverage_pct_of_usable_pixels": round(float(coverage), 3),
            "ndvi_mean_valid": round(float(np.nanmean(index[usable])), 4)
            if usable_count
            else None,
            "ndvi_median_valid": round(float(np.nanmedian(index[usable])), 4)
            if usable_count
            else None,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aoi",
        type=Path,
        default=Path("config/aoi_fundidora.geojson"),
    )
    parser.add_argument("--cloud-cover", type=float, default=60.0)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--vegetation-threshold", type=float, default=0.30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/fundidora_sentinel2_baseline.json"),
    )
    args = parser.parse_args()

    feature, bbox = read_aoi(args.aoi)
    result = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "catalog": EARTH_SEARCH,
            "collection": COLLECTION,
            "imagery": "Copernicus Sentinel-2 Level-2A surface reflectance",
            "access_pattern": "public STAC metadata + COG AOI window reads",
        },
        "aoi": {
            "id": feature.get("properties", {}).get(
                "id", feature.get("properties", {}).get("name", args.aoi.stem)
            ),
            "name": feature.get("properties", {}).get("name", args.aoi.stem),
            "bbox_wgs84": bbox,
            "geometry": feature["geometry"],
        },
        "selection_policy": {
            "comparison_windows": DEFAULT_WINDOWS,
            "scene_cloud_cover_lt_pct": args.cloud_cover,
            "candidate_scenes_evaluated_per_window": args.candidates,
            "primary_rank": "highest usable SCL pixel fraction over AOI",
            "tie_break": "lowest scene-level eo:cloud_cover",
        },
        "observations": {},
    }

    for label, window in DEFAULT_WINDOWS.items():
        (valid_fraction, item, arrays), evaluated = choose_scene(
            bbox, window, args.cloud_cover, args.candidates
        )
        obs = analyze(item, arrays, args.vegetation_threshold)
        obs["selection"] = {
            "window": window,
            "selected_local_usable_fraction": round(valid_fraction, 6),
            "evaluated_candidates": [
                {
                    "item_id": candidate.id,
                    "scene_cloud_cover_pct": candidate.properties.get("eo:cloud_cover"),
                    "local_usable_fraction": round(score, 6),
                }
                for score, candidate, _ in evaluated
            ],
        }
        result["observations"][label] = obs

    a = result["observations"]["2025"]["vegetation"]["coverage_pct_of_usable_pixels"]
    b = result["observations"]["2026"]["vegetation"]["coverage_pct_of_usable_pixels"]
    result["change"] = {
        "coverage_change_percentage_points": round(b - a, 3),
        "direction": "gain" if b > a else "loss" if b < a else "stable",
        "interpretation": (
            "Remote-sensing screening signal only; review acquisition quality, "
            "seasonality and source imagery before attributing a cause."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(
        f"2025 coverage: {a:.2f}% | 2026 coverage: {b:.2f}% | "
        f"change: {b-a:+.2f} pp"
    )


if __name__ == "__main__":
    main()
