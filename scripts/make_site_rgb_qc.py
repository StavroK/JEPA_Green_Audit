"""Create a large side-by-side visual QC image for a two-date Sentinel RGB site.

The source RGB tiles are intentionally enlarged with nearest-neighbor sampling so
each original Sentinel pixel remains visually discrete. This is for human QC,
not for model input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def enlarge_nearest(image: Image.Image, scale: int) -> Image.Image:
    if scale < 1:
        raise ValueError("scale must be >= 1")
    return image.resize(
        (image.width * scale, image.height * scale),
        resample=Image.Resampling.NEAREST,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scale",
        type=int,
        default=10,
        help="Nearest-neighbor enlargement factor for the source Sentinel RGB tiles.",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=24,
        help="Horizontal gap in output pixels between year panels.",
    )
    args = parser.parse_args()

    images = []
    for year in ("2025", "2026"):
        path = args.rgb_dir / f"{args.site}_{year}_rgb.png"
        image = Image.open(path).convert("RGB")
        images.append((year, image, enlarge_nearest(image, args.scale)))

    max_h = max(img.height for _, _, img in images)
    total_w = sum(img.width for _, _, img in images) + args.gap * (len(images) - 1)
    header = 64
    footer = 34
    canvas = Image.new("RGB", (total_w, max_h + header + footer), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    x = 0
    for year, source, image in images:
        canvas.paste(image, (x, header))
        title = f"{args.site} — {year}"
        draw.text((x + 8, 16), title, fill="black", font=font)
        draw.text(
            (x + 8, header + max_h + 10),
            f"source {source.width}x{source.height}px | {args.scale}x nearest-neighbor",
            fill="black",
            font=font,
        )
        x += image.width + args.gap

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)

    print(f"Wrote {args.output}")
    print(f"Output size: {canvas.width}x{canvas.height}")
    print(
        "QC note: nearest-neighbor enlargement preserves the original Sentinel "
        "pixel grid; it does not add spatial detail."
    )


if __name__ == "__main__":
    main()
