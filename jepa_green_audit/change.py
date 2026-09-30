"""Temporal and latent-representation change utilities."""

from __future__ import annotations

import numpy as np


def coverage_change(before_pct: float, after_pct: float) -> dict:
    """Return percentage-point vegetation change and direction."""
    delta = float(after_pct) - float(before_pct)
    if delta > 0:
        direction = "gain"
    elif delta < 0:
        direction = "loss"
    else:
        direction = "stable"
    return {
        "before_pct": float(before_pct),
        "after_pct": float(after_pct),
        "delta_percentage_points": delta,
        "direction": direction,
    }


def cosine_embedding_change(before: np.ndarray, after: np.ndarray) -> float:
    """Return cosine distance in [0, 2] for two embedding vectors.

    A higher value indicates larger representation change. It does not by
    itself explain the cause of that change.
    """
    a = np.asarray(before, dtype=np.float32).reshape(-1)
    b = np.asarray(after, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("embedding vectors must have the same shape")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        raise ValueError("embedding vectors must have non-zero norm")
    cosine_similarity = float(np.dot(a, b) / denom)
    return float(1.0 - np.clip(cosine_similarity, -1.0, 1.0))
