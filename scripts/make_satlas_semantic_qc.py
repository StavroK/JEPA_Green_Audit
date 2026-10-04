"""Build an auditable TCI/semantic montage for Satlas QC."""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageDraw


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True)
    p.add_argument("--semantic-dir", type=Path, default=Path("outputs/semantic"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    panels = []
    for year in ("2025", "2026"):
        for kind, label in (("satlas_tci", "TCI"), ("semantic", "Satlas semantic")):
            path = args.semantic_dir / f"{args.site}_{year}_{kind}.png"
            im = Image.open(path).convert("RGB")
            panels.append((f"{year} {label}", im))

    target_h = max(im.height for _, im in panels)
    resized = []
    for label, im in panels:
        scale = target_h / im.height
        resized.append((label, im.resize((round(im.width * scale), target_h))))

    header = 34
    width = sum(im.width for _, im in resized)
    canvas = Image.new("RGB", (width, target_h + header), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, im in resized:
        canvas.paste(im, (x, header))
        draw.text((x + 6, 9), label, fill="black")
        x += im.width

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
