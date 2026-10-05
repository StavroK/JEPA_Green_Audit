import numpy as np
import pytest

from scripts.compute_utc import utc_metrics


def test_utc_metrics_basic():
    mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    out = utc_metrics(mask, pixel_area_m2=1.0)
    assert out["utc_pct"] == 50.0
    assert out["canopy_area_m2"] == 2.0
    assert out["analysis_area_m2"] == 4.0


def test_utc_metrics_excludes_nodata():
    mask = np.array([[1, 0], [255, 255]], dtype=np.uint8)
    out = utc_metrics(mask, pixel_area_m2=4.0, nodata=255)
    assert out["utc_pct"] == 50.0
    assert out["analysis_area_m2"] == 8.0


def test_utc_metrics_rejects_non_binary_values():
    mask = np.array([[0, 2]], dtype=np.uint8)
    with pytest.raises(ValueError):
        utc_metrics(mask, pixel_area_m2=1.0)
