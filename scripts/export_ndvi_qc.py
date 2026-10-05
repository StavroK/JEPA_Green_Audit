"""Export NDVI vegetation mask PNG for a selected Sentinel baseline scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pystac_client import Client

from jepa_green_audit.spectral import ndvi
from jepa_green_audit.sentinel2 import EARTH_SEARCH, COLLECTION, load_aoi, valid_pixel_mask

try:
    from scripts.fetch_sentinel2 import read_aoi
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from fetch_sentinel2 import read_aoi


def fetch_item(item_id: str):
    catalog = Client.open(EARTH_SEARCH)
    items = list(
        catalog.search(
            collections=[COLLECTION],
            ids=[item_id],
            max_items=1,
        ).items()
    )
    if not items:
        raise RuntimeError(f"Sentinel-2 item not found: {item_id}")
    return items[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aoi", type=Path, required=True)
    p.add_argument("--baseline-json", type=Path, required=True)
    p.add_argument("--site", required=True)
    p.add_argument("--year", required=True, choices=["2025", "2026"])
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    _, bbox = read_aoi(args.aoi)
    payload = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    item_id = payload["observations"][args.year]["scene"]["item_id"]
    item = fetch_item(item_id)
    arrays = load_aoi(item, bbox)

    index = ndvi(arrays["nir"], arrays["red"])
    valid = valid_pixel_mask(arrays["scl"])
    vegetation = valid & np.isfinite(index) & (index >= args.threshold)

    rgb = np.zeros((*vegetation.shape, 3), dtype=np.uint8)
    rgb[~valid] = (100, 100, 100)
    rgb[valid & ~vegetation] = (190, 70, 50)
    rgb[vegetation] = (40, 160, 60)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(args.output)

    frac = float(vegetation.sum() / max(valid.sum(), 1))
    print(f"Vegetation fraction: {frac:.3f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
