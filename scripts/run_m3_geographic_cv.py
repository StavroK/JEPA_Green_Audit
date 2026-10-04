"""Contiguous geographic cross-validation for the M3 human-labeled benchmark.

Four folds hold out adjacent two-column vertical bands of the 8x8 review grid.
The same geographic cells from both years stay in the same fold, preventing
spatial leakage across dates.

The benchmark reports:
- NDVI
- frozen I-JEPA
- PCA I-JEPA
- I-JEPA + NDVI
- PCA fusion
- trivial always-vegetation / always-non-vegetation / train-majority baselines

For learned models, label-efficiency fractions are sampled by whole 2x2 spatial
blocks within the training geography and repeated across deterministic seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jepa_green_audit.benchmark import (
    LABEL_FRACTIONS,
    binary_segmentation_metrics,
    select_labeled_training_blocks,
    spatial_block_ids,
)
try:
    from scripts.run_m3_human_benchmark import (
        DEFAULT_DATASET,
        DEFAULT_LABELS,
        REVIEW_GRID,
        SEEDS,
        aggregate_16_to_8,
        fit_predict,
        load_human_labels,
    )
except ModuleNotFoundError:
    # Support direct execution: python scripts/run_m3_geographic_cv.py
    from run_m3_human_benchmark import (
        DEFAULT_DATASET,
        DEFAULT_LABELS,
        REVIEW_GRID,
        SEEDS,
        aggregate_16_to_8,
        fit_predict,
        load_human_labels,
    )

DEFAULT_OUTPUT = Path("outputs/fundidora_m3_geographic_cv.json")
DEFAULT_RGB_BASELINES = Path("data/processed/fundidora_m3_rgb_baselines.npz")
FOLD_WIDTH = 2


def contiguous_column_folds(
    shape: tuple[int, int] = (REVIEW_GRID, REVIEW_GRID),
    fold_width: int = FOLD_WIDTH,
) -> list[dict[str, np.ndarray | int]]:
    """Return non-overlapping contiguous held-out column bands."""
    height, width = shape
    if width % fold_width != 0:
        raise ValueError("grid width must be divisible by fold_width")

    folds = []
    columns = np.arange(width)[None, :]
    for fold_index, start in enumerate(range(0, width, fold_width)):
        stop = start + fold_width
        test = np.broadcast_to(
            (columns >= start) & (columns < stop),
            (height, width),
        ).copy()
        train = ~test
        folds.append(
            {
                "fold": fold_index,
                "test_col_start": start,
                "test_col_stop_exclusive": stop,
                "train": train,
                "test": test,
            }
        )
    return folds


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


def summarize_runs(runs: list[dict]) -> dict[str, float]:
    summary = {}
    if not runs:
        return summary
    for metric in ("iou", "f1_dice", "precision", "recall"):
        values = np.asarray([run[metric] for run in runs], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rgb-baselines",
        type=Path,
        default=DEFAULT_RGB_BASELINES,
        help="Optional frozen ResNet18/DINOv2 feature NPZ built from the same 8x8 RGB cells.",
    )
    args = parser.parse_args()

    data = np.load(args.dataset)
    human_labels = load_human_labels(args.labels)
    X_jepa, X_ndvi, y, years, rows, cols = aggregate_16_to_8(
        data["X_jepa"],
        data["X_ndvi"],
        data["year"].astype(str),
        data["patch_row"].astype(int),
        data["patch_col"].astype(int),
        human_labels,
    )

    representations = {
        "ndvi": (X_ndvi, False),
        "ijepa": (X_jepa, False),
        "ijepa_pca": (X_jepa, True),
        "ijepa_plus_ndvi": (np.concatenate([X_jepa, X_ndvi], axis=1), False),
        "ijepa_plus_ndvi_pca": (
            np.concatenate([X_jepa, X_ndvi], axis=1),
            True,
        ),
    }

    if args.rgb_baselines.exists():
        rgb = np.load(args.rgb_baselines)
        expected = {
            "y": y,
            "year": years.astype(str),
            "row": rows,
            "col": cols,
        }
        actual = {
            "y": rgb["y"].astype(bool),
            "year": rgb["year"].astype(str),
            "row": rgb["row"].astype(int),
            "col": rgb["col"].astype(int),
        }
        for key in expected:
            if not np.array_equal(expected[key], actual[key]):
                raise ValueError(
                    f"RGB baseline alignment mismatch for {key}; rebuild features "
                    "from the same M3 labels before running CV."
                )

        representations["resnet18"] = (rgb["X_resnet18"].astype(np.float32), False)
        if "X_dinov2" in rgb.files:
            representations["dinov2"] = (rgb["X_dinov2"].astype(np.float32), False)
    else:
        print(
            f"Warning: {args.rgb_baselines} not found; ResNet18/DINOv2 will be omitted."
        )
    trivial_names = (
        "always_vegetation",
        "always_non_vegetation",
        "train_majority",
    )

    block_grid = spatial_block_ids((REVIEW_GRID, REVIEW_GRID), block_size=2)
    folds = contiguous_column_folds()

    payload = {
        "benchmark_type": "human_labeled_contiguous_geographic_cross_validation",
        "target": "RGB-only human visible-cover label",
        "review_grid": [REVIEW_GRID, REVIEW_GRID],
        "fold_definition": "4 contiguous vertical holdouts, 2 columns each",
        "same_location_same_fold_across_years": True,
        "rgb_baselines_file": str(args.rgb_baselines) if args.rgb_baselines.exists() else None,
        "label_fractions": list(LABEL_FRACTIONS),
        "seeds": list(SEEDS),
        "dataset_summary": {
            "samples": int(y.size),
            "vegetation": int(y.sum()),
            "non_vegetation": int(y.size - y.sum()),
        },
        "folds": [],
        "aggregate_results": {},
        "trivial_baselines": {},
    }

    all_model_runs: dict[str, dict[str, list[dict]]] = {
        name: {str(fraction): [] for fraction in LABEL_FRACTIONS}
        for name in representations
    }
    all_trivial_runs = {name: [] for name in trivial_names}

    for fold in folds:
        train_grid = fold["train"]
        test_grid = fold["test"]
        sample_train = train_grid[rows, cols]
        sample_test = test_grid[rows, cols]

        fold_payload = {
            "fold": int(fold["fold"]),
            "test_columns": [
                int(fold["test_col_start"]),
                int(fold["test_col_stop_exclusive"]) - 1,
            ],
            "train_samples": int(sample_train.sum()),
            "train_vegetation": int(y[sample_train].sum()),
            "test_samples": int(sample_test.sum()),
            "test_vegetation": int(y[sample_test].sum()),
            "results": {},
            "trivial_baselines": {},
        }

        # Trivial references use all available training labels for the fold.
        for baseline in trivial_names:
            pred = trivial_predictions(
                baseline,
                y[sample_train],
                int(sample_test.sum()),
            )
            metrics = binary_segmentation_metrics(y[sample_test], pred)
            metrics["fold"] = int(fold["fold"])
            fold_payload["trivial_baselines"][baseline] = metrics
            all_trivial_runs[baseline].append(metrics)

        for name, (X, use_pca) in representations.items():
            fraction_results = []
            finite = np.all(np.isfinite(X), axis=1)

            for fraction in LABEL_FRACTIONS:
                runs = []
                skipped = []

                for seed in SEEDS:
                    selected_grid = select_labeled_training_blocks(
                        train_grid,
                        block_grid,
                        fraction=fraction,
                        seed=seed,
                    )
                    selected = selected_grid[rows, cols] & sample_train & finite
                    test = sample_test & finite

                    try:
                        pred, score = fit_predict(
                            X[selected],
                            y[selected],
                            X[test],
                            use_pca=use_pca,
                        )
                    except ValueError as exc:
                        skipped.append({"seed": seed, "reason": str(exc)})
                        continue

                    metrics = binary_segmentation_metrics(y[test], pred)
                    metrics.update(
                        {
                            "fold": int(fold["fold"]),
                            "seed": seed,
                            "fraction": fraction,
                            "labeled_samples": int(selected.sum()),
                            "labeled_positive_samples": int(y[selected].sum()),
                            "labeled_blocks": int(
                                np.unique(
                                    block_grid[rows[selected], cols[selected]]
                                ).size
                            ),
                            "test_samples": int(test.sum()),
                            "test_positive_samples": int(y[test].sum()),
                            "predicted_positive_samples": int(pred.sum()),
                            "mean_positive_probability": float(score.mean()),
                        }
                    )
                    runs.append(metrics)
                    all_model_runs[name][str(fraction)].append(metrics)

                result = {
                    "fraction": fraction,
                    "runs": runs,
                    "skipped": skipped,
                }
                result.update(summarize_runs(runs))
                fraction_results.append(result)

            fold_payload["results"][name] = fraction_results

        payload["folds"].append(fold_payload)

    for name in representations:
        payload["aggregate_results"][name] = []
        for fraction in LABEL_FRACTIONS:
            runs = all_model_runs[name][str(fraction)]
            result = {
                "fraction": fraction,
                "runs_completed": len(runs),
            }
            result.update(summarize_runs(runs))
            payload["aggregate_results"][name].append(result)

    for baseline in trivial_names:
        runs = all_trivial_runs[baseline]
        result = {"folds": runs}
        result.update(summarize_runs(runs))
        payload["trivial_baselines"][baseline] = result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(
        f"Dataset: {y.size} samples | vegetation={int(y.sum())} | "
        f"non-vegetation={int(y.size - y.sum())}"
    )
    for fold in payload["folds"]:
        print(
            f"fold {fold['fold']} cols {fold['test_columns'][0]}-{fold['test_columns'][1]}: "
            f"test={fold['test_samples']} | vegetation={fold['test_vegetation']}"
        )

    print("100% label aggregate across folds/seeds:")
    for name, results in payload["aggregate_results"].items():
        full = results[0]
        print(
            f"  {name}: IoU={full.get('iou_mean', float('nan')):.3f} "
            f"±{full.get('iou_std', float('nan')):.3f} | "
            f"F1={full.get('f1_dice_mean', float('nan')):.3f} "
            f"±{full.get('f1_dice_std', float('nan')):.3f}"
        )

    print("Trivial baselines across folds:")
    for name, result in payload["trivial_baselines"].items():
        print(
            f"  {name}: IoU={result.get('iou_mean', float('nan')):.3f} "
            f"±{result.get('iou_std', float('nan')):.3f} | "
            f"F1={result.get('f1_dice_mean', float('nan')):.3f} "
            f"±{result.get('f1_dice_std', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
