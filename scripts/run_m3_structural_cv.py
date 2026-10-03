"""Run M3 geographic CV with spectral, visual, and structural features.

Structural nDSM features come from the 2024 INEGI 1.5 m MDS/MDT pair and are
static across the 2025/2026 human-label observations. The experiment therefore
measures whether persistent height structure is useful for visible-cover
classification; it is not a contemporaneous vegetation-health assessment.
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
    from scripts.run_m3_geographic_cv import (
        contiguous_column_folds,
        summarize_runs,
        trivial_predictions,
    )
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
    from run_m3_geographic_cv import (
        contiguous_column_folds,
        summarize_runs,
        trivial_predictions,
    )
    from run_m3_human_benchmark import (
        DEFAULT_DATASET,
        DEFAULT_LABELS,
        REVIEW_GRID,
        SEEDS,
        aggregate_16_to_8,
        fit_predict,
        load_human_labels,
    )

DEFAULT_NDSM = Path("data/processed/fundidora_m3_ndsm_features.npz")
DEFAULT_OUTPUT = Path("outputs/fundidora_m3_structural_cv.json")


def align_ndsm_to_samples(
    ndsm_path: Path,
    years: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    data = np.load(ndsm_path)
    grid_rows = data["row"].astype(int)
    grid_cols = data["col"].astype(int)
    X_grid = data["X_ndsm"].astype(np.float32)
    feature_names = [str(x) for x in data["feature_names"].tolist()]

    lookup = {
        (int(r), int(c)): X_grid[i]
        for i, (r, c) in enumerate(zip(grid_rows, grid_cols))
    }

    X = []
    for _year, row, col in zip(years, rows, cols):
        key = (int(row), int(col))
        if key not in lookup:
            raise ValueError(f"Missing nDSM features for review cell {key}")
        X.append(lookup[key])

    return np.asarray(X, dtype=np.float32), feature_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--ndsm", type=Path, default=DEFAULT_NDSM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    X_ndsm, ndsm_feature_names = align_ndsm_to_samples(
        args.ndsm,
        years,
        rows,
        cols,
    )

    representations = {
        "ndvi": (X_ndvi, False),
        "ndsm": (X_ndsm, False),
        "ndvi_plus_ndsm": (np.concatenate([X_ndvi, X_ndsm], axis=1), False),
        "ijepa": (X_jepa, False),
        "ijepa_plus_ndsm": (np.concatenate([X_jepa, X_ndsm], axis=1), False),
        "ijepa_plus_ndvi": (np.concatenate([X_jepa, X_ndvi], axis=1), False),
        "ijepa_plus_ndvi_plus_ndsm": (
            np.concatenate([X_jepa, X_ndvi, X_ndsm], axis=1),
            False,
        ),
    }

    trivial_names = (
        "always_vegetation",
        "always_non_vegetation",
        "train_majority",
    )
    block_grid = spatial_block_ids((REVIEW_GRID, REVIEW_GRID), block_size=2)
    folds = contiguous_column_folds()

    all_model_runs = {
        name: {str(fraction): [] for fraction in LABEL_FRACTIONS}
        for name in representations
    }
    all_trivial_runs = {name: [] for name in trivial_names}

    payload = {
        "benchmark_type": "human_labeled_structural_geographic_cross_validation",
        "target": "RGB-only human visible-cover label",
        "review_grid": [REVIEW_GRID, REVIEW_GRID],
        "ndsm_feature_names": ndsm_feature_names,
        "ndsm_temporal_note": (
            "Static structural context derived from 2024 INEGI MDS/MDT; reused "
            "for 2025 and 2026 observations at the same geographic cells."
        ),
        "label_fractions": list(LABEL_FRACTIONS),
        "seeds": list(SEEDS),
        "folds": [],
        "aggregate_results": {},
        "trivial_baselines": {},
    }

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
    print("100% label aggregate across folds/seeds:")
    for name, results in payload["aggregate_results"].items():
        full = results[0]
        print(
            f"  {name}: IoU={full.get('iou_mean', float('nan')):.3f} "
            f"±{full.get('iou_std', float('nan')):.3f} | "
            f"F1={full.get('f1_dice_mean', float('nan')):.3f} "
            f"±{full.get('f1_dice_std', float('nan')):.3f}"
        )

    print("Trivial baselines:")
    for name, result in payload["trivial_baselines"].items():
        print(
            f"  {name}: IoU={result.get('iou_mean', float('nan')):.3f} | "
            f"F1={result.get('f1_dice_mean', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
