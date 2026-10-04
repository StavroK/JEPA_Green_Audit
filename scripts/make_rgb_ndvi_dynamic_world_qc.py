"""Create 2025/2026 RGB | NDVI | Dynamic World QC montage."""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def fit_nn(image: Image.Image, target_h: int) -> Image.Image:
    scale = target_h / image.height
    w = max(1, round(image.width * scale))
    return image.resize((w, target_h), Image.Resampling.NEAREST)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True)
    p.add_argument("--rgb-dir", type=Path, required=True)
    p.add_argument("--ndvi-dir", type=Path, required=True)
    p.add_argument("--dw-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--panel-height", type=int, default=500)
    args = p.parse_args()

    specs = []
    for year in ("2025", "2026"):
        specs.extend([
            (year, "RGB", args.rgb_dir / f"{args.site}_{year}_rgb.png"),
            (year, "NDVI mask", args.ndvi_dir / f"{args.site}_{year}_ndvi.png"),
            (year, "Dynamic World", args.dw_dir / f"{args.site}_{year}_dynamic_world.png"),
        ])

    panels = [(year, label, fit_nn(Image.open(path).convert("RGB"), args.panel_height))
              for year, label, path in specs]

    gap = 18
    header = 48
    row_w = sum(p.width for _, _, p in panels[:3]) + gap * 2
    canvas = Image.new("RGB", (row_w, 2 * (args.panel_height + header) + gap), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for row in range(2):
        y0 = row * (args.panel_height + header + gap)
        x = 0
        for year, label, panel in panels[row*3:(row+1)*3]:
            draw.text((x + 6, y0 + 14), f"{year} — {label}", fill="black", font=font)
            canvas.paste(panel, (x, y0 + header))
            x += panel.width + gap

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Wrote {args.output}")
    print(f"Output size: {canvas.width}x{canvas.height}")


if __name__ == "__main__":
    main()
