import numpy as np

from scripts.propose_utc_canopy_mask import box_mean, canopy_candidate


def test_box_mean_preserves_shape():
    a = np.arange(25, dtype=np.float32).reshape(5, 5)
    assert box_mean(a, 1).shape == a.shape


def test_green_textured_patch_becomes_candidate():
    rgb = np.full((9, 9, 3), 0.25, dtype=np.float32)
    # Create a green patch with crown-like luminance variation.
    vals = np.array([
        [0.35, 0.55, 0.38],
        [0.60, 0.40, 0.58],
        [0.37, 0.62, 0.42],
    ], dtype=np.float32)
    for i in range(3):
        for j in range(3):
            rgb[3+i, 3+j] = [0.12, vals[i, j], 0.10]

    mask, _ = canopy_candidate(
        rgb,
        exg_threshold=0.05,
        green_margin=0.01,
        min_texture=0.01,
        texture_radius=1,
    )
    assert mask[4, 4]


def test_gray_area_is_not_candidate():
    rgb = np.full((7, 7, 3), 0.5, dtype=np.float32)
    mask, _ = canopy_candidate(rgb, min_texture=0.0)
    assert not mask.any()
