"""Spectral vegetation utilities."""

from __future__ import annotations

import numpy as np


def ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index.

    Parameters
    ----------
    nir, red:
        Arrays with matching shape. Reflectance-scaled values are preferred.
    eps:
        Numerical guard for a near-zero denominator.
    """
    nir = np.asarray(nir, dtype=np.float32)
    red = np.asarray(red, dtype=np.float32)
    if nir.shape != red.shape:
        raise ValueError("nir and red arrays must have the same shape")
    return (nir - red) / (nir + red + eps)


def vegetation_mask(index: np.ndarray, threshold: float = 0.30) -> np.ndarray:
    """Create a simple vegetation mask from an index such as NDVI."""
    return np.asarray(index) >= threshold


def vegetation_coverage(mask: np.ndarray) -> float:
    """Return vegetation coverage as a percentage from a boolean/binary mask."""
    arr = np.asarray(mask)
    if arr.size == 0:
        raise ValueError("mask cannot be empty")
    return float(np.mean(arr.astype(bool)) * 100.0)
