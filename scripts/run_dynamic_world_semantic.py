"""Run Dynamic World semantic land-cover statistics for an AOI.

Requires Google Earth Engine access and an initialized project.

One-time setup:
    pip install earthengine-api
    earthengine authenticate

Example:
    python scripts/run_dynamic_world_semantic.py \
      --aoi config/aoi_la_pastora.geojson \
      --start 2025-08-01 --end 2025-09-30 \
      --project YOUR_GCP_PROJECT \
      --output outputs/semantic/la_pastora_dynamic_world_2025.json

Notes:
- Dynamic World is Sentinel-2 L1C-derived, 10 m, near-real-time land cover.
- This script computes AOI mean class probabilities and confident top-1 class
  fractions; it does not claim individual-tree crown delineation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASSES = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

PROJECT_GROUPS = {
    "tree_canopy": ["trees"],
    "other_vegetation": [
        "grass",
        "flooded_vegetation",
        "crops",
        "shrub_and_scrub",
    ],
    "developed": ["built"],
    "bare_ground": ["bare"],
    "water": ["water"],
    "uncertain": ["snow_and_ice"],
}


def load_geometry(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features") or []
        if len(features) != 1:
            raise ValueError("Expected a single-feature FeatureCollection")
        return features[0]["geometry"]
    if payload.get("type") == "Feature":
        return payload["geometry"]
    raise ValueError("Expected GeoJSON Feature or FeatureCollection")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--min-top-prob", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import ee

    ee.Initialize(project=args.project)
    geometry = ee.Geometry(load_geometry(args.aoi))

    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(geometry)
        .filterDate(args.start, args.end)
    )

    count = int(dw.size().getInfo())
    if count == 0:
        raise RuntimeError("No Dynamic World images found for the requested AOI/date range")

    # Median probability composite is more robust than selecting one arbitrary
    # scene while preserving the same class probability semantics.
    probs = dw.select(CLASSES).median()
    max_prob = probs.reduce(ee.Reducer.max())
    label = probs.toArray().arrayArgmax().arrayGet([0])

    scale = 10
    mean_probs = probs.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=scale,
        maxPixels=1_000_000,
        bestEffort=True,
    ).getInfo()

    valid_mask = max_prob.gte(args.min_top_prob)
    confident_label = label.updateMask(valid_mask)

    hist = confident_label.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=geometry,
        scale=scale,
        maxPixels=1_000_000,
        bestEffort=True,
    ).getInfo()

    raw_hist = hist.get("array", hist.get("max", hist.get("label", {}))) or {}
    native_fraction_pct = {name: 0.0 for name in CLASSES}
    total = sum(float(v) for v in raw_hist.values())
    if total:
        for k, v in raw_hist.items():
            idx = int(float(k))
            if 0 <= idx < len(CLASSES):
                native_fraction_pct[CLASSES[idx]] = round(100.0 * float(v) / total, 3)

    project_fraction_pct = {}
    for group, names in PROJECT_GROUPS.items():
        project_fraction_pct[group] = round(
            sum(native_fraction_pct.get(name, 0.0) for name in names), 3
        )

    result = {
        "dataset": "GOOGLE/DYNAMICWORLD/V1",
        "date_range": [args.start, args.end],
        "image_count": count,
        "scale_m": scale,
        "min_top_probability": args.min_top_prob,
        "mean_class_probability": {
            name: round(float(mean_probs.get(name, 0.0)), 4) for name in CLASSES
        },
        "confident_native_fraction_pct": native_fraction_pct,
        "confident_project_fraction_pct": project_fraction_pct,
        "limitations": [
            "10 m land cover is not individual-tree crown delineation.",
            "Fractions include only pixels whose top class probability meets the configured threshold.",
            "Dynamic World predictions are model outputs, not ground truth.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Images: {count}")
    print(f"Project fractions: {project_fraction_pct}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
