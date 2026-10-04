"""Evaluate the controlled 2007 resolution benchmark with geographic CV.

Compares frozen supervised ImageNet ResNet18, frozen I-JEPA, and frozen DINOv2
at 1/2/5/10 m using the exact same 2007 labels and geographic cells. Includes
trivial references and per-fold diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jepa_green_audit.benchmark import binary_segmentation_metrics
try:
    from scripts.run_m3_human_benchmark import fit_predict
except ModuleNotFoundError:
    from run_m3_human_benchmark import fit_predict

DEFAULT_INPUT = Path("data/processed/fundidora_2007_resolution_embeddings.npz")
DEFAULT_OUTPUT = Path("outputs/fundidora_2007_resolution_cv.json")
RESOLUTIONS = (1, 2, 5, 10)


def geographic_column_folds(
    cols: np.ndarray,
    grid_cols: int,
    folds: int = 4,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    boundaries = np.linspace(0, grid_cols, folds + 1, dtype=int)
    out = []
    for i in range(folds):
        start, stop = int(boundaries[i]), int(boundaries[i + 1])
        test = (cols >= start) & (cols < stop)
        train = ~test
        out.append((train, test, (start, stop - 1)))
    return out


def summarize(runs: list[dict]) -> dict:
    result = {}
    for metric in ("iou", "f1_dice", "precision", "recall"):
        values = np.asarray([run[metric] for run in runs], dtype=float)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std(ddof=0))
    return result


def trivial_predictions(name: str, y_train: np.ndarray, n_test: int) -> np.ndarray:
    if name == "always_vegetation":
        value = True
    elif name == "always_non_vegetation":
        value = False
    elif name == "train_majority":
        value = bool(np.mean(y_train) >= 0.5)
    else:
        raise ValueError(name)
    return np.full(n_test, value, dtype=bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = np.load(args.input)
    y = data["y"].astype(bool)
    rows = data["row"].astype(int)
    cols = data["col"].astype(int)
    grid_cols = int(data["grid_cols"][0])

    payload = {
        "benchmark_type": "controlled_same_date_resolution_sensitivity",
        "year": 2007,
        "resolutions_m": list(RESOLUTIONS),
        "samples": int(y.size),
        "vegetation": int(y.sum()),
        "non_vegetation": int((~y).sum()),
        "folds": 4,
        "results": {},
        "trivial_baselines": {},
    }

    folds = geographic_column_folds(cols, grid_cols)
    for baseline in ("always_vegetation", "always_non_vegetation", "train_majority"):
        runs = []
        for fold_index, (train, test, col_range) in enumerate(folds):
            pred = trivial_predictions(baseline, y[train], int(test.sum()))
            metrics = binary_segmentation_metrics(y[test], pred)
            metrics.update({
                "fold": fold_index,
                "test_columns": list(col_range),
                "test_samples": int(test.sum()),
                "test_vegetation": int(y[test].sum()),
            })
            runs.append(metrics)
        summary = {"runs": runs}
        summary.update(summarize(runs))
        payload["trivial_baselines"][baseline] = summary

    families = ["resnet18", "ijepa"]
    if f"X_dinov2_{RESOLUTIONS[0]}m" in data.files:
        families.append("dinov2")

    for family in families:
        payload["results"][family] = {}
        for resolution in RESOLUTIONS:
            X = data[f"X_{family}_{resolution}m"].astype(np.float32)
            runs = []
            for fold_index, (train, test, col_range) in enumerate(
                folds
            ):
                finite = np.all(np.isfinite(X), axis=1)
                train_mask = train & finite
                test_mask = test & finite

                pred, score = fit_predict(
                    X[train_mask],
                    y[train_mask],
                    X[test_mask],
                    use_pca=False,
                )
                metrics = binary_segmentation_metrics(y[test_mask], pred)
                metrics.update(
                    {
                        "fold": fold_index,
                        "test_columns": list(col_range),
                        "train_samples": int(train_mask.sum()),
                        "train_vegetation": int(y[train_mask].sum()),
                        "test_samples": int(test_mask.sum()),
                        "test_vegetation": int(y[test_mask].sum()),
                        "predicted_vegetation": int(pred.sum()),
                        "mean_positive_probability": float(score.mean()),
                    }
                )
                runs.append(metrics)

            summary = {"runs": runs}
            summary.update(summarize(runs))
            payload["results"][family][str(resolution)] = summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {args.output}")
    print("Trivial baselines:")
    for name, result in payload["trivial_baselines"].items():
        print(
            f"  {name}: IoU={result['iou_mean']:.3f} ±{result['iou_std']:.3f} | "
            f"F1={result['f1_dice_mean']:.3f} ±{result['f1_dice_std']:.3f}"
        )

    for family, by_resolution in payload["results"].items():
        print(f"{family}:")
        for resolution in RESOLUTIONS:
            result = by_resolution[str(resolution)]
            print(
                f"  {resolution:>2} m | "
                f"IoU={result['iou_mean']:.3f} ±{result['iou_std']:.3f} | "
                f"F1={result['f1_dice_mean']:.3f} ±{result['f1_dice_std']:.3f}"
            )
            fold_text = " | ".join(
                f"fold{run['fold']}:F1={run['f1_dice']:.3f},veg={run['test_vegetation']}/{run['test_samples']}"
                for run in result["runs"]
            )
            print(f"       {fold_text}")


if __name__ == "__main__":
    main()
