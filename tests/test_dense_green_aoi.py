import numpy as np

from scripts.propose_dense_green_aoi import score_window


def test_dense_green_window_uses_worst_year_score():
    masks = {
        "2025": np.ones((10, 10), dtype=bool),
        "2026": np.ones((10, 10), dtype=bool),
    }
    masks["2026"][0:5, 0:5] = False
    valids = {
        "2025": np.ones((10, 10), dtype=bool),
        "2026": np.ones((10, 10), dtype=bool),
    }

    stats = score_window(masks, valids, 0, 0, 10)
    assert stats is not None
    assert stats["2025_vegetation_fraction"] == 1.0
    assert stats["2026_vegetation_fraction"] == 0.75
    assert stats["score"] == 0.75


def test_dense_green_window_respects_valid_mask():
    masks = {
        "2025": np.ones((4, 4), dtype=bool),
        "2026": np.ones((4, 4), dtype=bool),
    }
    valids = {
        "2025": np.ones((4, 4), dtype=bool),
        "2026": np.ones((4, 4), dtype=bool),
    }
    valids["2025"][0, 0] = False

    stats = score_window(masks, valids, 0, 0, 4)
    assert stats is not None
    assert stats["2025_valid_fraction"] == 15 / 16
    assert stats["2025_vegetation_fraction"] == 1.0
