"""Generate a 2007 RGB-only labeling pack for resolution sensitivity.

Labels are created from the 1 m 2007 orthophoto itself, avoiding temporal
confounding with 2025/2026 Sentinel labels. The default 16x9 grid gives 144
review cells across the Fundidora AOI while preserving enough high-resolution
visual context for supervised / SSL representation experiments.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_SOURCE = Path("data/interim/inegi_orthophoto/fundidora_2007_1m_rgb.png")
DEFAULT_OUT = Path("data/labels/fundidora_2007_resolution")
GRID_COLS = 16
GRID_ROWS = 9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--grid-cols", type=int, default=GRID_COLS)
    parser.add_argument("--grid-rows", type=int, default=GRID_ROWS)
    args = parser.parse_args()

    image = Image.open(args.source).convert("RGB")
    width, height = image.size
    x_edges = np.linspace(0, width, args.grid_cols + 1, dtype=int)
    y_edges = np.linspace(0, height, args.grid_rows + 1, dtype=int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = args.output_dir / "2007"
    patch_dir.mkdir(parents=True, exist_ok=True)

    overview = image.copy()
    draw = ImageDraw.Draw(overview)
    line_width = max(1, round(max(width, height) / 1000))
    for x in x_edges:
        draw.line((int(x), 0, int(x), height), fill="white", width=line_width)
    for y in y_edges:
        draw.line((0, int(y), width, int(y)), fill="white", width=line_width)
    overview.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
    overview.save(args.output_dir / "fundidora_2007_review_grid.png")

    rows_out = []
    for row in range(args.grid_rows):
        for col in range(args.grid_cols):
            box = (
                int(x_edges[col]),
                int(y_edges[row]),
                int(x_edges[col + 1]),
                int(y_edges[row + 1]),
            )
            patch = image.crop(box)
            patch_path = patch_dir / f"2007_r{row:02d}_c{col:02d}.png"
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
        "label",
        "reviewer",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {csv_path}")
    print(f"Generated {len(rows_out)} review cells ({args.grid_cols} x {args.grid_rows})")
    print("Labels must be based on the 2007 RGB only.")


if __name__ == "__main__":
    main()
