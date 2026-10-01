"""Run the M3 weak-label Fundidora benchmark.

The result is an engineering smoke test only because the target is Sentinel-2
SCL class 4 rather than independent vegetation ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
DEFAULT_OUTPUT = Path("outputs/fundidora_m3_smoke_benchmark.json")
SEEDS = (7, 19, 42, 73, 101)


def fit_predict(X_train, y_train, X_test):
    if np.unique(y_train).size < 2:
        raise ValueError("selected labeled training set contains only one class")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=0,
        ),
    )
    model.fit(X_train, y_train)
    return model.predict(X_test).astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = np.load(args.dataset)
    X_jepa = data["X_jepa"]
    X_ndvi = data["X_ndvi"]
    y = data["y_weak"].astype(bool)
    rows = data["patch_row"].astype(int)
    cols = data["patch_col"].astype(int)
    valid_fraction = data["valid_fraction"]

    representations = {
        "ndvi": X_ndvi,
        "ijepa": X_jepa,
        "ijepa_plus_ndvi": np.concatenate([X_jepa, X_ndvi], axis=1),
    }

    grid_masks = geographic_holdout_masks((16, 16))
    block_grid = spatial_block_ids((16, 16), block_size=2)

    train_grid = grid_masks["train"]
    test_grid = grid_masks["test"]

    sample_train = train_grid[rows, cols] & np.isfinite(X_ndvi[:, 0]) & (valid_fraction > 0)
    sample_test = test_grid[rows, cols] & np.isfinite(X_ndvi[:, 0]) & (valid_fraction > 0)

    payload = {
        "benchmark_type": "engineering_smoke_test",
        "target": "Sentinel-2 SCL class 4 weak vegetation label",
        "warning": "Do not interpret as independent scientific ground truth.",
        "split": "fixed west 50% train / middle 25% validation / east 25% test",
        "label_block_size": 2,
        "seeds": list(SEEDS),
        "results": {},
    }

    for name, X in representations.items():
        rows_out = []
        for fraction in LABEL_FRACTIONS:
            seed_metrics = []
            skipped = []
            for seed in SEEDS:
                selected_grid = select_labeled_training_blocks(
                    train_grid,
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
                    pred = fit_predict(X[selected], y[selected], X[test])
                except ValueError as exc:
                    skipped.append({"seed": seed, "reason": str(exc)})
                    continue

                metrics = binary_segmentation_metrics(y[test], pred)
                metrics["seed"] = seed
                metrics["labeled_samples"] = int(selected.sum())
                metrics["labeled_blocks"] = int(
                    np.unique(block_grid[rows[selected], cols[selected]]).size
                )
                seed_metrics.append(metrics)

            aggregate = {"fraction": fraction, "runs": seed_metrics, "skipped": skipped}
            if seed_metrics:
                for metric in ("iou", "f1_dice", "precision", "recall"):
                    values = np.asarray([run[metric] for run in seed_metrics], dtype=float)
                    aggregate[f"{metric}_mean"] = float(values.mean())
                    aggregate[f"{metric}_std"] = float(values.std(ddof=0))
            rows_out.append(aggregate)
        payload["results"][name] = rows_out

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    for name, results in payload["results"].items():
        full = results[0]
        print(
            f"{name}: 100% labels | IoU={full.get('iou_mean', float('nan')):.3f} | "
            f"F1={full.get('f1_dice_mean', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
