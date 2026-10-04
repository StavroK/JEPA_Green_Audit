"""Create a side-by-side visual QC image for a two-date Sentinel RGB site."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    images = []
    for year in ("2025", "2026"):
        path = args.rgb_dir / f"{args.site}_{year}_rgb.png"
        image = Image.open(path).convert("RGB")
        images.append((year, image))

    max_h = max(img.height for _, img in images)
    total_w = sum(img.width for _, img in images)
    header = 36
    canvas = Image.new("RGB", (total_w, max_h + header), "white")
    draw = ImageDraw.Draw(canvas)

    x = 0
    for year, image in images:
        canvas.paste(image, (x, header))
        draw.text((x + 8, 10), f"{args.site} {year}", fill="black")
        x += image.width

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
