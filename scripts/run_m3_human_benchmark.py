"""Run the M3 human-labeled 8x8 Fundidora benchmark.

Human labels are RGB-only and independent of NDVI/SCL. Each 8x8 review cell
corresponds to a 2x2 group of the frozen I-JEPA 16x16 patch-token grid.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jepa_green_audit.benchmark import (
    LABEL_FRACTIONS,
    binary_segmentation_metrics,
    geographic_holdout_masks,
    select_labeled_training_blocks,
    spatial_block_ids,
)

DEFAULT_DATASET = Path("data/processed/fundidora_m3_patch_dataset.npz")
DEFAULT_LABELS = Path("data/labels/fundidora_m3/fundidora_patch_labels_8x8.csv")
DEFAULT_OUTPUT = Path("outputs/fundidora_m3_human_benchmark.json")
SEEDS = (7, 19, 42, 73, 101)
SOURCE_GRID = 16
REVIEW_GRID = 8


def load_human_labels(path: Path) -> dict[tuple[str, int, int], str]:
    labels = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row["label"].strip()
            if label not in {"vegetation", "non_vegetation", "uncertain"}:
                raise ValueError(
                    f"Invalid or missing label for {row['year']} r{row['row']} c{row['col']}: {label!r}"
                )
            key = (row["year"], int(row["row"]), int(row["col"]))
            if key in labels:
                raise ValueError(f"Duplicate human label: {key}")
            labels[key] = label
    return labels


def aggregate_16_to_8(
    X_jepa: np.ndarray,
    X_ndvi: np.ndarray,
    years: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    labels: dict[tuple[str, int, int], str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate four 16x16 token cells into one human-reviewed 8x8 cell."""
    features_jepa = []
    features_ndvi = []
    targets = []
    out_years = []
    out_rows = []
    out_cols = []

    for year in sorted(set(years.tolist())):
        year_mask = years == year
        for review_row in range(REVIEW_GRID):
            for review_col in range(REVIEW_GRID):
                label = labels.get((str(year), review_row, review_col))
                if label is None:
                    raise ValueError(
                        f"Missing human label for {year} r{review_row:02d} c{review_col:02d}"
                    )
                if label == "uncertain":
                    continue

                source_rows = {2 * review_row, 2 * review_row + 1}
                source_cols = {2 * review_col, 2 * review_col + 1}
                mask = (
                    year_mask
                    & np.isin(rows, list(source_rows))
                    & np.isin(cols, list(source_cols))
                )
                if int(mask.sum()) != 4:
                    raise RuntimeError(
                        f"Expected 4 source tokens for {year} r{review_row} c{review_col}; got {int(mask.sum())}"
                    )

                features_jepa.append(X_jepa[mask].mean(axis=0))
                features_ndvi.append(X_ndvi[mask].mean(axis=0))
                targets.append(label == "vegetation")
                out_years.append(str(year))
                out_rows.append(review_row)
                out_cols.append(review_col)

    return (
        np.asarray(features_jepa, dtype=np.float32),
        np.asarray(features_ndvi, dtype=np.float32),
        np.asarray(targets, dtype=bool),
        np.asarray(out_years),
        np.asarray(out_rows, dtype=int),
        np.asarray(out_cols, dtype=int),
    )


def fit_predict(X_train, y_train, X_test, use_pca=False):
    if np.unique(y_train).size < 2:
        raise ValueError("selected labeled training set contains only one class")

    steps = [StandardScaler()]
    if use_pca:
        n_components = max(1, min(16, X_train.shape[0] - 1, X_train.shape[1]))
        steps.append(PCA(n_components=n_components, random_state=0))
    steps.append(
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=0,
        )
    )
    model = make_pipeline(*steps)
    model.fit(X_train, y_train)
    pred = model.predict(X_test).astype(bool)
    score = model.predict_proba(X_test)[:, 1]
    return pred, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
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

    grid_masks = geographic_holdout_masks((REVIEW_GRID, REVIEW_GRID))
    block_grid = spatial_block_ids((REVIEW_GRID, REVIEW_GRID), block_size=2)

    sample_train = grid_masks["train"][rows, cols]
    sample_val = grid_masks["val"][rows, cols]
    sample_test = grid_masks["test"][rows, cols]

    payload = {
        "benchmark_type": "human_labeled_scientific_benchmark",
        "target": "RGB-only human visible-cover label",
        "review_grid": [REVIEW_GRID, REVIEW_GRID],
        "source_ijepa_grid": [SOURCE_GRID, SOURCE_GRID],
        "aggregation": "mean of each 2x2 I-JEPA token group; mean NDVI over same source cells",
        "split": "fixed west 50% train / middle 25% validation / east 25% test",
        "label_block_size": 2,
        "seeds": list(SEEDS),
        "dataset_summary": {
            "samples": int(y.size),
            "vegetation": int(y.sum()),
            "non_vegetation": int(y.size - y.sum()),
            "uncertain_excluded": int(
                sum(label == "uncertain" for label in human_labels.values())
            ),
        },
        "split_diagnostics": {
            "train_samples": int(sample_train.sum()),
            "train_positive_samples": int(y[sample_train].sum()),
            "val_samples": int(sample_val.sum()),
            "val_positive_samples": int(y[sample_val].sum()),
            "test_samples": int(sample_test.sum()),
            "test_positive_samples": int(y[sample_test].sum()),
        },
        "results": {},
    }

    for name, (X, use_pca) in representations.items():
        results = []
        for fraction in LABEL_FRACTIONS:
            runs = []
            skipped = []
            for seed in SEEDS:
                selected_grid = select_labeled_training_blocks(
                    grid_masks["train"],
                    block_grid,
                    fraction=fraction,
                    seed=seed,
                )
                selected = (
                    selected_grid[rows, cols]
                    & sample_train
                    & np.all(np.isfinite(X), axis=1)
                )
                test = sample_test & np.all(np.isfinite(X), axis=1)

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
                        "seed": seed,
                        "labeled_samples": int(selected.sum()),
                        "labeled_positive_samples": int(y[selected].sum()),
                        "labeled_blocks": int(
                            np.unique(block_grid[rows[selected], cols[selected]]).size
                        ),
                        "test_samples": int(test.sum()),
                        "test_positive_samples": int(y[test].sum()),
                        "predicted_positive_samples": int(pred.sum()),
                        "mean_positive_probability": float(score.mean()),
                    }
                )
                runs.append(metrics)

            aggregate = {"fraction": fraction, "runs": runs, "skipped": skipped}
            if runs:
                for metric in ("iou", "f1_dice", "precision", "recall"):
                    values = np.asarray([run[metric] for run in runs], dtype=float)
                    aggregate[f"{metric}_mean"] = float(values.mean())
                    aggregate[f"{metric}_std"] = float(values.std(ddof=0))
            results.append(aggregate)
        payload["results"][name] = results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {args.output}")
    summary = payload["dataset_summary"]
    diag = payload["split_diagnostics"]
    print(
        f"Human labels: {summary['samples']} | vegetation={summary['vegetation']} | "
        f"non-vegetation={summary['non_vegetation']} | uncertain excluded={summary['uncertain_excluded']}"
    )
    print(
        "Split diagnostics: "
        f"train={diag['train_samples']} ({diag['train_positive_samples']} vegetation) | "
        f"val={diag['val_samples']} ({diag['val_positive_samples']} vegetation) | "
        f"test={diag['test_samples']} ({diag['test_positive_samples']} vegetation)"
    )
    for name, results in payload["results"].items():
        full = results[0]
        print(
            f"{name}: 100% labels | IoU={full.get('iou_mean', float('nan')):.3f} | "
            f"F1={full.get('f1_dice_mean', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
