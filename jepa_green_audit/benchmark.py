"""Deterministic utilities for label-efficiency segmentation benchmarks."""

from __future__ import annotations

import math
import numpy as np

LABEL_FRACTIONS = (1.0, 0.5, 0.25, 0.10, 0.05, 0.01)


def binary_segmentation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute binary vegetation segmentation metrics on valid samples."""
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    valid = np.ones(truth.shape, dtype=bool)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != truth.shape:
            raise ValueError("valid_mask must have the same shape as y_true")

    truth = truth[valid]
    pred = pred[valid]
    if truth.size == 0:
        raise ValueError("no valid samples to score")

    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    tn = int(np.sum(~truth & ~pred))

    def ratio(num: float, den: float) -> float:
        return float(num / den) if den else 1.0

    return {
        "iou": ratio(tp, tp + fp + fn),
        "f1_dice": ratio(2 * tp, 2 * tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": int(truth.size),
    }


def spatial_block_ids(shape: tuple[int, int], block_size: int = 4) -> np.ndarray:
    """Assign each cell to a deterministic square geographic block."""
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    height, width = shape
    rows = np.arange(height)[:, None] // block_size
    cols = np.arange(width)[None, :] // block_size
    n_col_blocks = int(math.ceil(width / block_size))
    return rows * n_col_blocks + cols


def geographic_holdout_masks(
    shape: tuple[int, int],
    train_fraction: float = 0.50,
    val_fraction: float = 0.25,
) -> dict[str, np.ndarray]:
    """Create contiguous west-to-east train/validation/test holdout bands."""
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be < 1")

    height, width = shape
    train_end = max(1, int(round(width * train_fraction)))
    val_end = max(train_end + 1, int(round(width * (train_fraction + val_fraction))))
    val_end = min(val_end, width - 1)

    cols = np.arange(width)[None, :]
    train = np.broadcast_to(cols < train_end, (height, width)).copy()
    val = np.broadcast_to((cols >= train_end) & (cols < val_end), (height, width)).copy()
    test = np.broadcast_to(cols >= val_end, (height, width)).copy()
    return {"train": train, "val": val, "test": test}


def select_labeled_training_blocks(
    train_mask: np.ndarray,
    block_ids: np.ndarray,
    fraction: float,
    seed: int = 42,
) -> np.ndarray:
    """Select whole geographic blocks to simulate limited labeled data."""
    train = np.asarray(train_mask, dtype=bool)
    groups = np.asarray(block_ids)
    if train.shape != groups.shape:
        raise ValueError("train_mask and block_ids must have the same shape")
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0, 1]")

    candidates = np.unique(groups[train])
    if candidates.size == 0:
        raise ValueError("train_mask contains no samples")

    if fraction == 1.0:
        chosen = candidates
    else:
        count = max(1, int(math.ceil(candidates.size * fraction)))
        rng = np.random.default_rng(seed)
        chosen = rng.choice(candidates, size=count, replace=False)

    return train & np.isin(groups, chosen)
