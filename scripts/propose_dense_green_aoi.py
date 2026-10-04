"""Propose a dense-green sub-AOI from a broader candidate using two-date Sentinel NDVI.

This avoids manually guessing coordinates from a display screenshot. The script:
1. selects usable Sentinel-2 scenes for 2025 and 2026 over the broad candidate;
2. computes valid NDVI vegetation masks on the 10 m grid;
3. searches fixed-size windows;
4. ranks windows by the minimum vegetation fraction across both years;
5. writes the best window as GeoJSON for visual QC.

The output is a research AOI candidate, not an official park boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.warp import transform_bounds

from jepa_green_audit.sentinel2 import load_aoi, valid_pixel_mask
from jepa_green_audit.spectral import ndvi
try:
    from scripts.fetch_sentinel2 import DEFAULT_WINDOWS, choose_scene, read_aoi
except ModuleNotFoundError as exc:
    # When executed as "python scripts/propose_dense_green_aoi.py", Python puts
    # the scripts/ directory (not the repository root) on sys.path.
    if exc.name != "scripts":
        raise
    from fetch_sentinel2 import DEFAULT_WINDOWS, choose_scene, read_aoi


def vegetation_mask(arrays: dict, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    valid = valid_pixel_mask(arrays["scl"])
    index = ndvi(arrays["nir"], arrays["red"])
    usable = valid & np.isfinite(index)
    vegetation = usable & (index >= threshold)
    return vegetation, usable


def score_window(
    masks: dict[str, np.ndarray],
    valids: dict[str, np.ndarray],
    row: int,
    col: int,
    size: int,
) -> dict[str, float] | None:
    stats = {}
    for year in ("2025", "2026"):
        veg = masks[year][row:row+size, col:col+size]
        valid = valids[year][row:row+size, col:col+size]
        total = valid.size
        valid_n = int(valid.sum())
        if total == 0 or valid_n == 0:
            return None
        stats[f"{year}_valid_fraction"] = valid_n / total
        stats[f"{year}_vegetation_fraction"] = int((veg & valid).sum()) / valid_n

    stats["score"] = min(
        stats["2025_vegetation_fraction"],
        stats["2026_vegetation_fraction"],
    )
    stats["mean_vegetation_fraction"] = (
        stats["2025_vegetation_fraction"] + stats["2026_vegetation_fraction"]
    ) / 2
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--candidate-aoi",
        type=Path,
        default=Path("config/aoi_la_pastora_candidate_v1.geojson"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("config/aoi_la_pastora_green_core_auto.geojson"),
    )
    p.add_argument("--window-pixels", type=int, default=60)
    p.add_argument("--stride-pixels", type=int, default=5)
    p.add_argument("--vegetation-threshold", type=float, default=0.30)
    p.add_argument("--cloud-cover", type=float, default=60.0)
    p.add_argument("--candidates", type=int, default=5)
    args = p.parse_args()

    feature, bbox = read_aoi(args.candidate_aoi)

    arrays_by_year = {}
    selected = {}
    for year, date_window in DEFAULT_WINDOWS.items():
        (usable_fraction, item, arrays), _ = choose_scene(
            bbox,
            date_window,
            args.cloud_cover,
            args.candidates,
        )
        arrays_by_year[year] = arrays
        selected[year] = {
            "item_id": item.id,
            "datetime": str(item.datetime or item.properties.get("datetime")),
            "usable_fraction": round(float(usable_fraction), 6),
        }

    shape0 = arrays_by_year["2025"]["red"].shape
    shape1 = arrays_by_year["2026"]["red"].shape
    if shape0 != shape1:
        raise RuntimeError(f"Scene grids differ: 2025={shape0}, 2026={shape1}")

    masks = {}
    valids = {}
    for year in ("2025", "2026"):
        masks[year], valids[year] = vegetation_mask(
            arrays_by_year[year], args.vegetation_threshold
        )

    h, w = shape0
    size = args.window_pixels
    if size > h or size > w:
        raise ValueError(f"window-pixels={size} exceeds candidate grid {h}x{w}")

    ranked = []
    for row in range(0, h - size + 1, args.stride_pixels):
        for col in range(0, w - size + 1, args.stride_pixels):
            stats = score_window(masks, valids, row, col, size)
            if stats is None:
                continue
            if min(
                stats["2025_valid_fraction"],
                stats["2026_valid_fraction"],
            ) < 0.90:
                continue
            ranked.append((stats["score"], stats["mean_vegetation_fraction"], row, col, stats))

    if not ranked:
        raise RuntimeError("No valid candidate windows found")

    ranked.sort(reverse=True)
    _, _, row, col, stats = ranked[0]

    transform = Affine(*arrays_by_year["2025"]["transform"][:6])
    crs = arrays_by_year["2025"]["crs"]
    win = Window(col, row, size, size)
    left, bottom, right, top = window_bounds(win, transform)
    west, south, east, north = transform_bounds(
        crs, "EPSG:4326", left, bottom, right, top, densify_pts=21
    )

    out = {
        "type": "FeatureCollection",
        "name": "la_pastora_dense_green_auto_candidate",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": "la_pastora_green_core_auto",
                    "name": "La Pastora auto-selected dense-green candidate",
                    "purpose": "M3b dense-green replication candidate selected from two-date Sentinel NDVI; requires visual QC",
                    "source_candidate": str(args.candidate_aoi),
                    "selection_method": "maximize minimum 2025/2026 vegetation fraction in fixed Sentinel window",
                    "window_pixels": size,
                    "vegetation_threshold": args.vegetation_threshold,
                    "score_min_vegetation_fraction": round(stats["score"], 6),
                    "mean_vegetation_fraction": round(stats["mean_vegetation_fraction"], 6),
                    "2025_vegetation_fraction": round(stats["2025_vegetation_fraction"], 6),
                    "2026_vegetation_fraction": round(stats["2026_vegetation_fraction"], 6),
                    "selected_scenes": selected,
                    "note": "Research AOI candidate only; not an official park/protected-area boundary",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [west, south],
                        [east, south],
                        [east, north],
                        [west, north],
                        [west, south],
                    ]],
                },
            }
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Candidate grid: {h}x{w}")
    print(f"Selected window: row={row}, col={col}, size={size}")
    print(
        "Vegetation fraction: "
        f"2025={stats['2025_vegetation_fraction']:.3f}, "
        f"2026={stats['2026_vegetation_fraction']:.3f}, "
        f"score={stats['score']:.3f}"
    )
    print(f"WGS84 bbox: {[west, south, east, north]}")
    print(f"Wrote {args.output}")
    print("Top 5 windows:")
    for _, _, r, c, st in ranked[:5]:
        print(
            f"  row={r:3d} col={c:3d} "
            f"2025={st['2025_vegetation_fraction']:.3f} "
            f"2026={st['2026_vegetation_fraction']:.3f} "
            f"score={st['score']:.3f}"
        )


if __name__ == "__main__":
    main()
