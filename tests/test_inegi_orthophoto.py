import numpy as np

from scripts.ingest_inegi_orthophoto import to_uint8_rgb


def test_to_uint8_rgb_shape_and_range():
    x = np.stack(
        [
            np.arange(100, dtype=np.float32).reshape(10, 10),
            np.arange(100, dtype=np.float32).reshape(10, 10) * 2,
            np.arange(100, dtype=np.float32).reshape(10, 10) * 3,
        ]
    )
    rgb = to_uint8_rgb(x)
    assert rgb.shape == (10, 10, 3)
    assert rgb.dtype == np.uint8
    assert int(rgb.min()) >= 0
    assert int(rgb.max()) <= 255
