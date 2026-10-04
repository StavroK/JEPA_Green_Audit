"""Export Dynamic World semantic classification and probability maps for QC.

Requires Google Earth Engine access.

Example:
  python scripts/export_dynamic_world_qc.py \
    --aoi config/aoi_la_pastora_green_core_auto.geojson \
    --start 2025-08-01 --end 2025-09-30 \
    --project mtygreenaudit \
    --site la_pastora_auto --year 2025 \
    --output-dir outputs/dynamic_world
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve

import ee
from PIL import Image

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

COLORS = {
    "water": (65, 155, 223),
    "trees": (0, 100, 0),
    "grass": (124, 252, 0),
    "flooded_vegetation": (0, 207, 117),
    "crops": (255, 255, 0),
    "shrub_and_scrub": (177, 255, 0),
    "built": (196, 40, 27),
    "bare": (255, 187, 34),
    "snow_and_ice": (255, 255, 255),
    "uncertain": (100, 100, 100),
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


def hex_color(rgb):
    return "%02x%02x%02x" % rgb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aoi", type=Path, required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--site", required=True)
    p.add_argument("--year", required=True)
    p.add_argument("--min-top-prob", type=float, default=0.5)
    p.add_argument("--scale", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/dynamic_world"))
    args = p.parse_args()

    ee.Initialize(project=args.project)
    geometry = ee.Geometry(load_geometry(args.aoi))

    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(geometry)
        .filterDate(args.start, args.end)
    )
    count = int(dw.size().getInfo())
    if count == 0:
        raise RuntimeError("No Dynamic World images found")

    probs = dw.select(CLASSES).median()
    max_prob = probs.reduce(ee.Reducer.max())
    label = probs.toArray().arrayArgmax().arrayGet([0])

    confident = label.updateMask(max_prob.gte(args.min_top_prob))
    palette = [hex_color(COLORS[name]) for name in CLASSES]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_png = args.output_dir / f"{args.site}_{args.year}_dynamic_world.png"
    prob_png = args.output_dir / f"{args.site}_{args.year}_dynamic_world_confidence.png"

    class_vis = confident.visualize(min=0, max=8, palette=palette)
    prob_vis = max_prob.visualize(min=0, max=1, palette=["000000", "ffffff"])

    region = geometry.bounds().getInfo()["coordinates"]

    class_url = class_vis.getThumbURL({
        "region": region,
        "scale": args.scale,
        "format": "png",
    })
    prob_url = prob_vis.getThumbURL({
        "region": region,
        "scale": args.scale,
        "format": "png",
    })

    urlretrieve(class_url, class_png)
    urlretrieve(prob_url, prob_png)

    # Normalize to RGB so downstream montage has predictable modes.
    Image.open(class_png).convert("RGB").save(class_png)
    Image.open(prob_png).convert("RGB").save(prob_png)

    out = {
        "dataset": "GOOGLE/DYNAMICWORLD/V1",
        "image_count": count,
        "date_range": [args.start, args.end],
        "min_top_probability": args.min_top_prob,
        "scale_m": args.scale,
        "classification_png": str(class_png),
        "confidence_png": str(prob_png),
        "classes": CLASSES,
        "palette_rgb": {name: COLORS[name] for name in CLASSES},
    }
    meta = args.output_dir / f"{args.site}_{args.year}_dynamic_world_qc.json"
    meta.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Images: {count}")
    print(f"Wrote {class_png}")
    print(f"Wrote {prob_png}")
    print(f"Wrote {meta}")


if __name__ == "__main__":
    main()
