"""Build the Fundidora M3 patch-level benchmark dataset.

This is an engineering smoke-test dataset. Vegetation targets are weak labels
from Sentinel-2 SCL class 4, not independent scientific ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pystac_client import Client

from jepa_green_audit.ijepa_encoder import FrozenIJEPAEncoder
from jepa_green_audit.sentinel2 import EARTH_SEARCH, COLLECTION, load_aoi, valid_pixel_mask
from jepa_green_audit.spectral import ndvi

AOI_PATH = Path("config/aoi_fundidora.geojson")
RGB_DIR = Path("data/interim/fundidora")
OUT_NPZ = Path("data/processed/fundidora_m3_patch_dataset.npz")
OUT_META = Path("data/processed/fundidora_m3_patch_dataset.json")

SCENES = {
    "2025": "S2A_14RLP_20250828_0_L2A",
    "2026": "S2B_14RLP_20260915_0_L2A",
}


def read_aoi_bbox(path: Path) -> tuple[dict, list[float]]:
    feature = json.loads(path.read_text(encoding="utf-8"))
    coords = feature["geometry"]["coordinates"][0]
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    return feature, [min(xs), min(ys), max(xs), max(ys)]


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


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def aggregate_to_grid(
    values: np.ndarray,
    valid: np.ndarray,
    out_shape: tuple[int, int] = (16, 16),
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate a raster to a regular grid using valid-pixel means."""
    arr = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    if arr.shape != mask.shape:
        raise ValueError("values and valid must have matching shapes")

    out_h, out_w = out_shape
    row_edges = np.linspace(0, arr.shape[0], out_h + 1, dtype=int)
    col_edges = np.linspace(0, arr.shape[1], out_w + 1, dtype=int)

    means = np.full((out_h, out_w), np.nan, dtype=np.float32)
    fractions = np.zeros((out_h, out_w), dtype=np.float32)

    for row in range(out_h):
        for col in range(out_w):
            rs = slice(row_edges[row], row_edges[row + 1])
            cs = slice(col_edges[col], col_edges[col + 1])
            local_valid = mask[rs, cs]
            total = local_valid.size
            count = int(local_valid.sum())
            fractions[row, col] = count / total if total else 0.0
            if count:
                means[row, col] = float(arr[rs, cs][local_valid].mean())

    return means, fractions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=OUT_NPZ)
    parser.add_argument("--metadata", type=Path, default=OUT_META)
    args = parser.parse_args()

    feature, bbox = read_aoi_bbox(AOI_PATH)
    encoder = FrozenIJEPAEncoder(
        upstream_dir=args.upstream,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    years: list[str] = []
    patch_features: list[np.ndarray] = []
    patch_ndvi: list[np.ndarray] = []
    weak_labels: list[np.ndarray] = []
    valid_fraction: list[np.ndarray] = []

    metadata = {
        "schema_version": "1.0",
        "purpose": "M3 patch-level engineering smoke-test benchmark",
        "label_source": {
            "type": "weak_label",
            "source": "Sentinel-2 Scene Classification Layer",
            "positive_class": 4,
            "decision_rule": "vegetation_fraction >= 0.5 among valid SCL pixels",
            "warning": "Not independent scientific ground truth.",
        },
        "aoi": {
            "id": feature["properties"]["id"],
            "name": feature["properties"]["name"],
            "bbox_wgs84": bbox,
        },
        "encoder": encoder.metadata(),
        "scenes": {},
    }

    for year, item_id in SCENES.items():
        item = fetch_item(item_id)
        bands = load_aoi(item, bbox)
        valid = valid_pixel_mask(bands["scl"])
        index = ndvi(bands["nir"], bands["red"])

        ndvi_grid, valid_grid = aggregate_to_grid(index, valid)

        scl_veg = (bands["scl"] == 4).astype(np.float32)
        veg_fraction, _ = aggregate_to_grid(scl_veg, valid)
        labels = veg_fraction >= 0.5

        rgb_path = RGB_DIR / f"fundidora_{year}_rgb.png"
        dense = encoder.encode_rgb_patch_grid(read_rgb(rgb_path))

        if dense.shape[:2] != ndvi_grid.shape:
            raise RuntimeError(
                f"Grid mismatch: I-JEPA={dense.shape[:2]} NDVI={ndvi_grid.shape}"
            )

        years.extend([year] * dense.shape[0] * dense.shape[1])
        patch_features.append(dense.reshape(-1, dense.shape[-1]))
        patch_ndvi.append(ndvi_grid.reshape(-1, 1))
        weak_labels.append(labels.reshape(-1))
        valid_fraction.append(valid_grid.reshape(-1))

        metadata["scenes"][year] = {
            "item_id": item.id,
            "datetime": str(item.datetime or item.properties.get("datetime")),
            "rgb_path": str(rgb_path),
            "source_grid_shape": list(index.shape),
            "patch_grid_shape": list(ndvi_grid.shape),
        }

    X_jepa = np.concatenate(patch_features, axis=0).astype(np.float32)
    X_ndvi = np.concatenate(patch_ndvi, axis=0).astype(np.float32)
    y = np.concatenate(weak_labels, axis=0).astype(np.uint8)
    valid_fraction_arr = np.concatenate(valid_fraction, axis=0).astype(np.float32)

    rows, cols = np.indices((16, 16))
    patch_row = np.tile(rows.reshape(-1), len(SCENES))
    patch_col = np.tile(cols.reshape(-1), len(SCENES))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X_jepa=X_jepa,
        X_ndvi=X_ndvi,
        y_weak=y,
        valid_fraction=valid_fraction_arr,
        year=np.asarray(years),
        patch_row=patch_row,
        patch_col=patch_col,
    )

    metadata["dataset"] = {
        "samples": int(y.size),
        "positive_samples": int(y.sum()),
        "negative_samples": int(y.size - y.sum()),
        "jepa_feature_dim": int(X_jepa.shape[1]),
        "ndvi_feature_dim": int(X_ndvi.shape[1]),
        "output_npz": str(args.output),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"Wrote {args.metadata}")
    print(
        f"Samples: {y.size} | weak vegetation positives: {int(y.sum())} | "
        f"negatives: {int(y.size - y.sum())}"
    )


if __name__ == "__main__":
    main()
