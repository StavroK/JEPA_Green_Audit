"""Export aligned RGB PNGs for the exact Fundidora Sentinel-2 scenes from Issue #1.

Outputs:
  data/interim/fundidora/fundidora_2025_rgb.png
  data/interim/fundidora/fundidora_2026_rgb.png
  data/interim/fundidora/rgb_tiles_provenance.json

The script reuses the same AOI and exact scene IDs validated in Issue #1.
It applies shared per-channel percentile scaling across both years so the
display transform is consistent before I-JEPA inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from pystac_client import Client
from rasterio.enums import Resampling

from jepa_green_audit.sentinel2 import (
    EARTH_SEARCH,
    COLLECTION,
    _read_band_to_grid,
)

DEFAULT_AOI_PATH = Path("config/aoi_fundidora.geojson")
DEFAULT_OUT_DIR = Path("data/interim/fundidora")
DEFAULT_BASELINE = Path("outputs/fundidora_sentinel2_baseline.json")
DEFAULT_SITE = "fundidora"

FUNDIDORA_SCENES = {
    "2025": "S2A_14RLP_20250828_0_L2A",
    "2026": "S2B_14RLP_20260915_0_L2A",
}

ASSETS = {
    "red": "red",
    "green": "green",
    "blue": "blue",
}


def read_aoi_bbox(path: Path) -> tuple[dict, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features") or []
        if len(features) != 1:
            raise ValueError(f"Expected one AOI feature in {path}; found {len(features)}")
        feature = features[0]
    elif payload.get("type") == "Feature":
        feature = payload
    else:
        raise ValueError("AOI must be a GeoJSON Feature or single-feature FeatureCollection")

    coords = feature["geometry"]["coordinates"][0]
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return feature, [min(xs), min(ys), max(xs), max(ys)]


def scenes_from_baseline(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    observations = payload.get("observations", {})
    scenes = {}
    for year in ("2025", "2026"):
        try:
            scenes[year] = observations[year]["scene"]["item_id"]
        except KeyError as exc:
            raise ValueError(
                f"{path} does not contain observations.{year}.scene.item_id"
            ) from exc
    return scenes


def fetch_item(item_id: str):
    catalog = Client.open(EARTH_SEARCH)
    search = catalog.search(
        collections=[COLLECTION],
        ids=[item_id],
        max_items=1,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(f"Sentinel-2 item not found: {item_id}")
    return items[0]


def load_rgb(item, bbox_wgs84: list[float]) -> tuple[np.ndarray, dict]:
    # Red band defines the exact 10 m grid used in Issue #1.
    red, crs, transform = _read_band_to_grid(
        item.assets[ASSETS["red"]].href,
        bbox_wgs84,
    )
    height, width = red.shape

    green, _, _ = _read_band_to_grid(
        item.assets[ASSETS["green"]].href,
        bbox_wgs84,
        dst_crs=crs,
        dst_transform=transform,
        dst_width=width,
        dst_height=height,
        resampling=Resampling.bilinear,
    )
    blue, _, _ = _read_band_to_grid(
        item.assets[ASSETS["blue"]].href,
        bbox_wgs84,
        dst_crs=crs,
        dst_transform=transform,
        dst_width=width,
        dst_height=height,
        resampling=Resampling.bilinear,
    )

    rgb = np.stack(
        [
            red.astype(np.float32),
            green.astype(np.float32),
            blue.astype(np.float32),
        ],
        axis=-1,
    )

    meta = {
        "item_id": item.id,
        "datetime": str(item.datetime or item.properties.get("datetime")),
        "scene_cloud_cover_pct": item.properties.get("eo:cloud_cover"),
        "shape": [height, width, 3],
        "crs": str(crs),
        "transform": list(transform),
        "assets": {
            "red": item.assets[ASSETS["red"]].href,
            "green": item.assets[ASSETS["green"]].href,
            "blue": item.assets[ASSETS["blue"]].href,
        },
    }
    return rgb, meta


def shared_channel_stretch(
    rgb_by_year: dict[str, np.ndarray],
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Apply the same percentile transform to both years, independently per channel."""
    stacked = np.concatenate(
        [rgb.reshape(-1, 3) for rgb in rgb_by_year.values()],
        axis=0,
    )

    lows = np.percentile(stacked, low_pct, axis=0)
    highs = np.percentile(stacked, high_pct, axis=0)
    denom = np.maximum(highs - lows, 1e-6)

    outputs = {}
    for year, rgb in rgb_by_year.items():
        scaled = (rgb - lows.reshape(1, 1, 3)) / denom.reshape(1, 1, 3)
        scaled = np.clip(scaled, 0.0, 1.0)
        outputs[year] = (scaled * 255.0).round().astype(np.uint8)

    scaling = {
        "method": "shared per-channel percentile stretch",
        "low_percentile": low_pct,
        "high_percentile": high_pct,
        "channel_order": ["red", "green", "blue"],
        "channel_low_values": lows.tolist(),
        "channel_high_values": highs.tolist(),
    }
    return outputs, scaling


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", type=Path, default=DEFAULT_AOI_PATH)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--site", default=DEFAULT_SITE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    feature, bbox = read_aoi_bbox(args.aoi)
    scenes = (
        scenes_from_baseline(args.baseline_json)
        if args.baseline_json.exists()
        else FUNDIDORA_SCENES
    )

    raw = {}
    provenance = {
        "schema_version": "1.1",
        "purpose": "Aligned RGB inputs for frozen representation comparison",
        "source": {
            "catalog": EARTH_SEARCH,
            "collection": COLLECTION,
            "baseline_json": str(args.baseline_json) if args.baseline_json.exists() else None,
        },
        "aoi": {
            "id": feature.get("properties", {}).get(
                "id", feature.get("properties", {}).get("name", args.aoi.stem)
            ),
            "name": feature.get("properties", {}).get("name", args.aoi.stem),
            "bbox_wgs84": bbox,
        },
        "site": args.site,
        "scenes": {},
    }

    for year, item_id in scenes.items():
        item = fetch_item(item_id)
        rgb, meta = load_rgb(item, bbox)
        raw[year] = rgb
        provenance["scenes"][year] = meta

    rgb8, scaling = shared_channel_stretch(raw)
    provenance["display_scaling"] = scaling

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for year, image in rgb8.items():
        path = args.out_dir / f"{args.site}_{year}_rgb.png"
        Image.fromarray(image, mode="RGB").save(path)
        provenance["scenes"][year]["png"] = str(path)

    prov_path = args.out_dir / "rgb_tiles_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"Wrote {args.out_dir / f'{args.site}_2025_rgb.png'}")
    print(f"Wrote {args.out_dir / f'{args.site}_2026_rgb.png'}")
    print(f"Wrote {prov_path}")
    print("Source grid shape:", provenance["scenes"]["2025"]["shape"])
    print("Shared scaling:", scaling)


if __name__ == "__main__":
    main()
