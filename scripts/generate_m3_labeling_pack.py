"""Generate a human-review labeling pack for the Fundidora M3 benchmark.

Creates 8x8-grid RGB review cells for each validated year plus a CSV template.
The labels are intended to be independent of NDVI/SCL so they can support the
scientific label-efficiency benchmark.

Label vocabulary:
- vegetation
- non_vegetation
- uncertain

Do not infer biological tree health from these visual labels.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RGB_DIR = Path("data/interim/fundidora")
OUT_DIR = Path("data/labels/fundidora_m3")
GRID = 8
YEARS = ("2025", "2026")


def grid_edges(length: int, cells: int = GRID) -> np.ndarray:
    return np.linspace(0, length, cells + 1, dtype=int)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_out = []

    for year in YEARS:
        image_path = RGB_DIR / f"fundidora_{year}_rgb.png"
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        x_edges = grid_edges(width)
        y_edges = grid_edges(height)

        year_dir = OUT_DIR / year
        year_dir.mkdir(parents=True, exist_ok=True)

        overview = image.resize((width * 4, height * 4), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(overview)
        for x in x_edges:
            draw.line((x * 4, 0, x * 4, height * 4), fill="white", width=1)
        for y in y_edges:
            draw.line((0, y * 4, width * 4, y * 4), fill="white", width=1)
        overview.save(OUT_DIR / f"fundidora_{year}_review_grid_8x8.png")

        for row in range(GRID):
            for col in range(GRID):
                box = (
                    int(x_edges[col]),
                    int(y_edges[row]),
                    int(x_edges[col + 1]),
                    int(y_edges[row + 1]),
                )
                patch = image.crop(box)
                patch_name = f"{year}_r{row:02d}_c{col:02d}.png"
                patch_path = year_dir / patch_name
                patch.resize((224, 224), Image.Resampling.NEAREST).save(patch_path)

                rows_out.append({
                    "year": year,
                    "row": row,
                    "col": col,
                    "patch_path": str(patch_path),
                    "label": "",
                    "reviewer": "",
                    "notes": "",
                })

    csv_path = OUT_DIR / "fundidora_patch_labels_8x8.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["year", "row", "col", "patch_path", "label", "reviewer", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    instructions = OUT_DIR / "README.md"
    instructions.write_text(
        "# Fundidora M3 human review pack — 8x8 scientific benchmark\n\n"
        "Review each 8x8 benchmark cell using RGB only. Do not consult NDVI or SCL while labeling. Each review cell corresponds to a 2x2 group of I-JEPA patch tokens.\n\n"
        "Allowed labels: vegetation, non_vegetation, uncertain.\n\n"
        "Use uncertain when the patch is mixed, visually ambiguous, obscured, or the "
        "dominant class cannot be determined confidently. These labels describe visible "
        "vegetation presence only; they are not tree-health diagnoses.\n",
        encoding="utf-8",
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {OUT_DIR / 'README.md'}")
    print(f"Generated {len(rows_out)} review patches")


if __name__ == "__main__":
    main()
