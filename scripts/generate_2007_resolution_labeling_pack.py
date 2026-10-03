"""Generate a sampled 2007 RGB-only labeling pack for resolution sensitivity.

The native review geography uses a 64x36 grid (~40 m-scale cells over the
Fundidora AOI). To keep manual review practical, exactly 384 cells are sampled
with a deterministic spatially distributed pattern.

Labels are created from the 1 m 2007 orthophoto itself, avoiding temporal
confounding with 2025/2026 Sentinel labels. The same geographic cells can later
be evaluated at 1 m, 2 m, 5 m, and 10 m.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_SOURCE = Path("data/interim/inegi_orthophoto/fundidora_2007_1m_rgb.png")
DEFAULT_OUT = Path("data/labels/fundidora_2007_resolution")
GRID_COLS = 64
GRID_ROWS = 36
TARGET_SAMPLE_COUNT = 384


def sampled_cells(
    grid_rows: int = GRID_ROWS,
    grid_cols: int = GRID_COLS,
    target_count: int = TARGET_SAMPLE_COUNT,
) -> list[tuple[int, int]]:
    """Return a deterministic, spatially distributed subset of review cells.

    For the default 64x36 grid there are 2304 cells. Taking every sixth cell in
    raster order yields exactly 384 cells. Because 64 is not divisible by 6,
    the selected columns shift across rows rather than forming vertical stripes.
    """
    total = grid_rows * grid_cols
    if target_count <= 0 or target_count > total:
        raise ValueError("target_count must be between 1 and the number of grid cells")

    if total % target_count == 0:
        stride = total // target_count
        indices = np.arange(0, total, stride, dtype=int)
        if len(indices) == target_count:
            return [(int(i // grid_cols), int(i % grid_cols)) for i in indices]

    # Generic deterministic fallback: evenly spaced positions across the raster.
    indices = np.linspace(0, total - 1, target_count, dtype=int)
    indices = np.unique(indices)
    if len(indices) != target_count:
        raise RuntimeError("could not create the requested number of unique sample cells")
    return [(int(i // grid_cols), int(i % grid_cols)) for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--grid-cols", type=int, default=GRID_COLS)
    parser.add_argument("--grid-rows", type=int, default=GRID_ROWS)
    parser.add_argument("--sample-count", type=int, default=TARGET_SAMPLE_COUNT)
    args = parser.parse_args()

    image = Image.open(args.source).convert("RGB")
    width, height = image.size
    x_edges = np.linspace(0, width, args.grid_cols + 1, dtype=int)
    y_edges = np.linspace(0, height, args.grid_rows + 1, dtype=int)
    selected = sampled_cells(args.grid_rows, args.grid_cols, args.sample_count)
    selected_set = set(selected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = args.output_dir / "2007"
    patch_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale patch PNGs from older grid configurations.
    for path in patch_dir.glob("*.png"):
        path.unlink()

    # Overview: full underlying grid in subtle white, sampled cells in red.
    overview = image.copy()
    draw = ImageDraw.Draw(overview)
    grid_width = max(1, round(max(width, height) / 1800))
    sample_width = max(2, round(max(width, height) / 700))

    for x in x_edges:
        draw.line((int(x), 0, int(x), height), fill=(255, 255, 255), width=grid_width)
    for y in y_edges:
        draw.line((0, int(y), width, int(y)), fill=(255, 255, 255), width=grid_width)

    for row, col in selected:
        x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
        y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
        draw.rectangle(
            (x0, y0, max(x0 + 1, x1 - 1), max(y0 + 1, y1 - 1)),
            outline=(255, 0, 0),
            width=sample_width,
        )

    overview.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
    overview.save(args.output_dir / "fundidora_2007_review_grid.png")

    rows_out = []
    for row, col in selected:
        box = (
            int(x_edges[col]),
            int(y_edges[row]),
            int(x_edges[col + 1]),
            int(y_edges[row + 1]),
        )
        patch = image.crop(box)
        patch_path = patch_dir / f"2007_r{row:02d}_c{col:02d}.png"

        # Keep enough detail for human inspection while avoiding unnecessary
        # derived-file size. Do not distort aspect ratio.
        patch.thumbnail((512, 512), Image.Resampling.LANCZOS)
        patch.save(patch_path)

        rows_out.append(
            {
                "year": "2007",
                "row": row,
                "col": col,
                "patch_path": str(patch_path),
                "source_path": str(args.source),
                "grid_rows": args.grid_rows,
                "grid_cols": args.grid_cols,
                "sampled": "1",
                "label": "",
                "reviewer": "",
                "notes": "",
            }
        )

    csv_path = args.output_dir / "fundidora_2007_labels.csv"
    fields = [
        "year",
        "row",
        "col",
        "patch_path",
        "source_path",
        "grid_rows",
        "grid_cols",
        "sampled",
        "label",
        "reviewer",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {csv_path}")
    print(
        f"Underlying grid: {args.grid_cols} x {args.grid_rows} "
        f"({args.grid_cols * args.grid_rows} geographic cells)"
    )
    print(f"Generated {len(rows_out)} sampled review cells")
    print(
        "Labels must be based on the 2007 1 m RGB only; the same cell IDs will "
        "be reused at 1/2/5/10 m."
    )


if __name__ == "__main__":
    main()
