import numpy as np

from jepa_green_audit.sentinel2 import INVALID_SCL, valid_pixel_mask


def test_invalid_scl_contains_cloud_and_shadow_classes():
    for value in (1, 3, 7, 8, 9, 10, 11):
        assert value in INVALID_SCL


def test_valid_pixel_mask():
    scl = np.array([[4, 5, 6], [3, 8, 9], [10, 11, 2]], dtype=np.uint8)
    mask = valid_pixel_mask(scl)
    expected = np.array(
        [[True, True, True], [False, False, False], [False, False, True]]
    )
    np.testing.assert_array_equal(mask, expected)
