import numpy as np

from scripts.build_m3_ndsm_features import aggregate_ndsm_grid
from scripts.run_m3_structural_cv import align_ndsm_to_samples


def test_aggregate_ndsm_grid_clips_negative_and_builds_features():
    arr = np.arange(64, dtype=np.float32).reshape(8, 8) - 5.0
    features, valid_fraction, cap = aggregate_ndsm_grid(
        arr,
        grid=2,
        cap_percentile=100.0,
    )
    assert features.shape == (2, 2, 5)
    assert valid_fraction.shape == (2, 2)
    assert np.all(valid_fraction == 1.0)
    assert np.all(features[..., 0] >= 0.0)
    assert cap == 58.0


def test_aggregate_ndsm_grid_reports_missing_cells():
    arr = np.ones((8, 8), dtype=np.float32)
    arr[:4, :4] = np.nan
    features, valid_fraction, _ = aggregate_ndsm_grid(arr, grid=2)
    assert np.isnan(features[0, 0]).all()
    assert valid_fraction[0, 0] == 0.0
