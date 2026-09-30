import numpy as np

from scripts.export_rgb_tiles import shared_channel_stretch


def test_shared_channel_stretch_same_mapping():
    a = np.array(
        [[[100, 200, 300], [200, 300, 400]]],
        dtype=np.float32,
    )
    b = np.array(
        [[[300, 400, 500], [400, 500, 600]]],
        dtype=np.float32,
    )

    out, meta = shared_channel_stretch(
        {"2025": a, "2026": b},
        low_pct=0,
        high_pct=100,
    )

    assert out["2025"].dtype == np.uint8
    assert out["2026"].dtype == np.uint8
    assert meta["method"] == "shared per-channel percentile stretch"
    assert out["2025"][0, 0, 0] == 0
    assert out["2026"][0, 1, 0] == 255
