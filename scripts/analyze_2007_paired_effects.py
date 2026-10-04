"""Paired fold-level effect analysis for the 2007 Fundidora benchmark.

Reads:
- outputs/fundidora_2007_resolution_cv.json
- outputs/fundidora_2007_resnet18_finetune_cv.json

For each resolution, compares frozen I-JEPA against:
- frozen ResNet18
- frozen DINOv2
- partially fine-tuned ResNet18

Because there are only four geographic folds, this script avoids asymptotic
significance claims. It reports paired fold differences and an exact exhaustive
bootstrap confidence interval over the four fold pairs (4^4 resamples).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

DEFAULT_FROZEN = Path("outputs/fundidora_2007_resolution_cv.json")
DEFAULT_FINETUNE = Path("outputs/fundidora_2007_resnet18_finetune_cv.json")
DEFAULT_JSON = Path("outputs/fundidora_2007_paired_effects.json")
DEFAULT_MD = Path("outputs/fundidora_2007_paired_effects.md")
RESOLUTIONS = (1, 2, 5, 10)


def exact_bootstrap_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    means = []
    for indices in itertools.product(range(n), repeat=n):
        means.append(float(np.mean(values[list(indices)])))
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def fold_metric(block: dict, metric: str = "f1_dice") -> np.ndarray:
    runs = sorted(block["runs"], key=lambda run: int(run["fold"]))
    return np.asarray([float(run[metric]) for run in runs], dtype=float)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = exact_bootstrap_ci(diff)
    return {
        "paired_fold_differences": [float(v) for v in diff],
        "mean_difference": float(diff.mean()),
        "bootstrap_95_ci": [lo, hi],
        "ijepa_wins": int(np.sum(diff > 0)),
        "ties": int(np.sum(np.isclose(diff, 0.0))),
        "comparator_wins": int(np.sum(diff < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--finetune", type=Path, default=DEFAULT_FINETUNE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    finetune = json.loads(args.finetune.read_text(encoding="utf-8"))

    payload = {
        "method": (
            "Paired geographic-fold F1 differences with exact exhaustive bootstrap "
            "over four folds; descriptive uncertainty only because n_folds=4."
        ),
        "comparisons": {},
    }

    rows = []
    for resolution in RESOLUTIONS:
        key = str(resolution)
        ijepa = fold_metric(frozen["results"]["ijepa"][key])

        comparators = {
            "frozen_resnet18": fold_metric(frozen["results"]["resnet18"][key]),
            "frozen_dinov2": fold_metric(frozen["results"]["dinov2"][key]),
            "finetuned_resnet18": fold_metric(finetune["results"][key]),
        }

        payload["comparisons"][key] = {}
        for name, values in comparators.items():
            result = compare(ijepa, values)
            payload["comparisons"][key][name] = result
            rows.append((resolution, name, result))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Fundidora 2007 paired fold effects",
        "",
        "Metric: F1/Dice. Positive mean difference favors frozen I-JEPA.",
        "",
        "| Resolution | Comparator | ΔF1 I-JEPA − comparator | 95% bootstrap CI | Fold wins |",
        "|---:|---|---:|---:|---:|",
    ]
    for resolution, name, result in rows:
        lo, hi = result["bootstrap_95_ci"]
        lines.append(
            f"| {resolution} m | {name} | {result['mean_difference']:+.3f} | "
            f"[{lo:+.3f}, {hi:+.3f}] | "
            f"{result['ijepa_wins']}-{result['ties']}-{result['comparator_wins']} |"
        )

    lines.extend([
        "",
        "## Interpretation guardrail",
        "",
        "There are only four geographic folds. These intervals are descriptive paired-fold "
        "uncertainty summaries, not strong population-level significance tests. A multi-site "
        "replication is required before claiming general superiority.",
        "",
    ])
    args.output_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    for resolution, name, result in rows:
        lo, hi = result["bootstrap_95_ci"]
        print(
            f"{resolution:>2} m | {name:<20} | "
            f"ΔF1={result['mean_difference']:+.3f} "
            f"95% CI [{lo:+.3f}, {hi:+.3f}] | "
            f"wins {result['ijepa_wins']}-{result['ties']}-{result['comparator_wins']}"
        )


if __name__ == "__main__":
    main()
