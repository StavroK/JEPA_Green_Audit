import numpy as np

from jepa_green_audit.change import cosine_embedding_change, coverage_change
from jepa_green_audit.spectral import ndvi, vegetation_coverage, vegetation_mask


def test_ndvi_green_pixel():
    value = ndvi(np.array([0.8]), np.array([0.2]))
    assert value[0] > 0.5


def test_coverage():
    mask = np.array([[True, False], [True, True]])
    assert vegetation_coverage(mask) == 75.0


def test_coverage_change_loss():
    result = coverage_change(30.0, 24.0)
    assert result["direction"] == "loss"
    assert result["delta_percentage_points"] == -6.0


def test_embedding_no_change():
    vector = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_embedding_change(vector, vector)) < 1e-6
