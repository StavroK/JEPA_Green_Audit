import numpy as np

from scripts.aggregate_utc_by_zones import zone_metrics


def test_zone_metrics_basic():
    canopy = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    valid = np.ones_like(canopy, dtype=bool)
    zone = np.ones_like(canopy, dtype=bool)

    out = zone_metrics(canopy, valid, zone, pixel_area_m2=100.0)

    assert out["utc_pct"] == 50.0
    assert out["canopy_area_ha"] == 0.02
    assert out["analysis_area_ha"] == 0.04


def test_zone_metrics_target_deficit():
    canopy = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    valid = np.ones_like(canopy, dtype=bool)
    zone = np.ones_like(canopy, dtype=bool)

    out = zone_metrics(
        canopy,
        valid,
        zone,
        pixel_area_m2=100.0,
        target_utc=50.0,
    )

    assert out["utc_pct"] == 25.0
    assert out["utc_deficit_pp"] == 25.0
    assert out["canopy_deficit_ha"] == 0.01


def test_zone_metrics_empty_zone():
    canopy = np.zeros((2, 2), dtype=np.uint8)
    valid = np.ones_like(canopy, dtype=bool)
    zone = np.zeros_like(canopy, dtype=bool)

    out = zone_metrics(canopy, valid, zone, pixel_area_m2=100.0)
    assert out["utc_pct"] is None
