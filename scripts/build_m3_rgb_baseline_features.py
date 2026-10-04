"""Build frozen ResNet18 and DINOv2 features for the Sentinel M3 label-efficiency benchmark.

The 2025/2026 RGB images are partitioned directly into the same 8x8 human-review
cells used by the existing M3 labels. Each non-uncertain cell receives:
- frozen ImageNet ResNet18 embedding
- frozen DINOv2 ViT-S/14 embedding

The output ordering is deterministic: year, row, col, matching the existing
aggregate_16_to_8() ordering used by run_m3_geographic_cv.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from scripts.build_2007_resolution_embeddings import (
        FrozenDINOv2Encoder,
        FrozenResNet18Encoder,
    )
    from scripts.run_m3_human_benchmark import (
        DEFAULT_LABELS,
        REVIEW_GRID,
        load_human_labels,
    )
except ModuleNotFoundError:
    from build_2007_resolution_embeddings import (
        FrozenDINOv2Encoder,
        FrozenResNet18Encoder,
    )
    from run_m3_human_benchmark import (
        DEFAULT_LABELS,
        REVIEW_GRID,
        load_human_labels,
    )

DEFAULT_RGB_DIR = Path("data/interim/fundidora")
DEFAULT_OUTPUT = Path("data/processed/fundidora_m3_rgb_baselines.npz")
YEARS = ("2025", "2026")


def cell_box(width: int, height: int, row: int, col: int) -> tuple[int, int, int, int]:
    x_edges = np.linspace(0, width, REVIEW_GRID + 1, dtype=int)
    y_edges = np.linspace(0, height, REVIEW_GRID + 1, dtype=int)
    return (
        int(x_edges[col]),
        int(y_edges[row]),
        int(x_edges[col + 1]),
        int(y_edges[row + 1]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--skip-dinov2",
        action="store_true",
        help="Build only ResNet18 features.",
    )
    args = parser.parse_args()

    labels = load_human_labels(args.labels)
    resnet = FrozenResNet18Encoder(device=args.device)
    dinov2 = None if args.skip_dinov2 else FrozenDINOv2Encoder(device=args.device)

    X_resnet18 = []
    X_dinov2 = []
    y = []
    years = []
    rows = []
    cols = []

    for year in YEARS:
        image_path = args.rgb_dir / f"fundidora_{year}_rgb.png"
        image = Image.open(image_path).convert("RGB")
        print(f"{year}: encoding review cells from {image_path}")

        for row in range(REVIEW_GRID):
            for col in range(REVIEW_GRID):
                label = labels.get((year, row, col))
                if label is None:
                    raise ValueError(f"Missing label for {year} r{row:02d} c{col:02d}")
                if label == "uncertain":
                    continue

                patch = image.crop(cell_box(image.width, image.height, row, col))
                rgb = np.asarray(patch)

                X_resnet18.append(resnet.encode_rgb(rgb))
                if dinov2 is not None:
                    X_dinov2.append(dinov2.encode_rgb(rgb))

                y.append(label == "vegetation")
                years.append(year)
                rows.append(row)
                cols.append(col)

    payload = {
        "X_resnet18": np.asarray(X_resnet18, dtype=np.float32),
        "y": np.asarray(y, dtype=bool),
        "year": np.asarray(years),
        "row": np.asarray(rows, dtype=int),
        "col": np.asarray(cols, dtype=int),
    }
    if dinov2 is not None:
        payload["X_dinov2"] = np.asarray(X_dinov2, dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)

    print(f"Wrote {args.output}")
    print(
        f"Samples={len(y)} | vegetation={int(np.sum(y))} | "
        f"non-vegetation={len(y)-int(np.sum(y))}"
    )
    print(f"ResNet18 shape: {payload['X_resnet18'].shape}")
    if "X_dinov2" in payload:
        print(f"DINOv2 shape: {payload['X_dinov2'].shape}")


if __name__ == "__main__":
    main()
