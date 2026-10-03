"""Build 8x8 structural nDSM features for the Fundidora M3 benchmark.

The input nDSM is MDS - MDT at ~1.5 m. It is treated as static 2024 structural
context and aggregated onto the same 8x8 review grid used by the human labels.

This script does not infer vegetation from height. It only creates structural
features that can be evaluated against independent RGB-only human labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

DEFAULT_NDSM = Path("data/interim/inegi_elevation/fundidora_ndsm.tif")
DEFAULT_OUTPUT = Path("data/processed/fundidora_m3_ndsm_features.npz")
DEFAULT_META = Path("data/processed/fundidora_m3_ndsm_features.json")
GRID = 8
FEATURE_NAMES = (
    "height_mean_m",
    "height_median_m",
    "height_p90_m",
    "fraction_gt_2m",
    "fraction_gt_5m",
)


def aggregate_ndsm_grid(
    ndsm: np.ndarray,
    grid: int = GRID,
    cap_percentile: float = 99.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Aggregate nDSM into regular review cells using robust height features."""
    arr = np.asarray(ndsm, dtype=np.float32).copy()
    valid = np.isfinite(arr)
    if not valid.any():
        raise ValueError("nDSM contains no finite pixels")

    # Small negative residuals are alignment/interpolation artifacts, not
    # meaningful below-ground object heights.
    arr[valid] = np.maximum(arr[valid], 0.0)
    cap = float(np.percentile(arr[valid], cap_percentile))
    arr[valid] = np.minimum(arr[valid], cap)

    y_edges = np.linspace(0, arr.shape[0], grid + 1, dtype=int)
    x_edges = np.linspace(0, arr.shape[1], grid + 1, dtype=int)

    features = np.full((grid, grid, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    valid_fraction = np.zeros((grid, grid), dtype=np.float32)

    for row in range(grid):
        for col in range(grid):
            cell = arr[
                y_edges[row] : y_edges[row + 1],
                x_edges[col] : x_edges[col + 1],
            ]
            cell_valid = np.isfinite(cell)
            valid_fraction[row, col] = float(cell_valid.mean()) if cell.size else 0.0
            values = cell[cell_valid]
            if values.size == 0:
                continue

            features[row, col] = (
                float(values.mean()),
                float(np.median(values)),
                float(np.percentile(values, 90)),
                float(np.mean(values > 2.0)),
                float(np.mean(values > 5.0)),
            )

    return features, valid_fraction, cap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndsm", type=Path, default=DEFAULT_NDSM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--grid", type=int, default=GRID)
    parser.add_argument("--cap-percentile", type=float, default=99.5)
    args = parser.parse_args()

    with rasterio.open(args.ndsm) as src:
        ndsm = src.read(1).astype(np.float32)
        if src.nodata is not None and np.isfinite(src.nodata):
            ndsm[ndsm == src.nodata] = np.nan
        raster_meta = {
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "transform": list(src.transform)[:6],
            "bounds": list(src.bounds),
        }

    features, valid_fraction, cap = aggregate_ndsm_grid(
        ndsm,
        grid=args.grid,
        cap_percentile=args.cap_percentile,
    )

    rows, cols = np.indices((args.grid, args.grid))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X_ndsm=features.reshape(-1, features.shape[-1]),
        valid_fraction=valid_fraction.reshape(-1),
        row=rows.reshape(-1),
        col=cols.reshape(-1),
        feature_names=np.asarray(FEATURE_NAMES),
    )

    metadata = {
        "source": str(args.ndsm),
        "output": str(args.output),
        "grid": [args.grid, args.grid],
        "feature_names": list(FEATURE_NAMES),
        "preprocessing": {
            "negative_heights_clipped_to_zero": True,
            "upper_cap_percentile": args.cap_percentile,
            "upper_cap_m": cap,
        },
        "temporal_note": (
            "INEGI MDS/MDT source coverage is 2024. These features are static "
            "structural context when evaluated against 2025/2026 labels; they "
            "must not be interpreted as contemporaneous vegetation condition."
        ),
        "raster": raster_meta,
        "valid_fraction": {
            "min": float(valid_fraction.min()),
            "mean": float(valid_fraction.mean()),
            "max": float(valid_fraction.max()),
        },
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"Wrote {args.metadata}")
    print(f"Features: {', '.join(FEATURE_NAMES)}")
    print(f"Robust height cap ({args.cap_percentile:g}th percentile): {cap:.2f} m")
    print(
        "Cell valid fraction: "
        f"min={valid_fraction.min():.3f} | "
        f"mean={valid_fraction.mean():.3f} | "
        f"max={valid_fraction.max():.3f}"
    )


if __name__ == "__main__":
    main()
