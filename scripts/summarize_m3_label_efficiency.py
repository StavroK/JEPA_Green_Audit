"""Print a compact label-efficiency summary from M3 geographic CV JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_INPUT = Path("outputs/fundidora_m3_geographic_cv.json")


def pct(fraction: float) -> str:
    return f"{int(round(fraction * 100)):>3d}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    aggregate = payload["aggregate_results"]

    preferred_names = [
        "ndvi",
        "resnet18",
        "dinov2",
        "ijepa",
        "ijepa_plus_ndvi",
        "ijepa_pca",
        "ijepa_plus_ndvi_pca",
    ]
    names = [name for name in preferred_names if name in aggregate]

    print("M3 geographic CV label-efficiency summary")
    print("metric: mean IoU / mean F1 across completed folds and seeds")
    print()

    header = "labels".ljust(8) + "".join(name.rjust(28) for name in names)
    print(header)
    print("-" * len(header))

    fractions = [row["fraction"] for row in aggregate[names[0]]]
    for i, fraction in enumerate(fractions):
        row = pct(fraction).ljust(8)
        for name in names:
            result = aggregate[name][i]
            iou = result.get("iou_mean")
            f1 = result.get("f1_dice_mean")
            runs = result.get("runs_completed", 0)
            if iou is None or f1 is None:
                cell = f"n/a ({runs} runs)"
            else:
                cell = f"{iou:.3f} / {f1:.3f} ({runs})"
            row += cell.rjust(28)
        print(row)

    print()
    print("Trivial baselines (100% fold training labels, reference only):")
    for name, result in payload["trivial_baselines"].items():
        print(
            f"  {name}: IoU={result.get('iou_mean', float('nan')):.3f} | "
            f"F1={result.get('f1_dice_mean', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
