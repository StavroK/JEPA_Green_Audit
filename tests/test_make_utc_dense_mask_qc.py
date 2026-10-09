import numpy as np

from scripts.make_utc_dense_mask_qc import choose_records, overlay_rgb_mask


def test_overlay_rgb_mask_changes_only_positive_pixels():
    rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
    mask = np.array([[0, 255], [0, 0]], dtype=np.uint8)
    out = overlay_rgb_mask(rgb, mask, alpha=0.5)
    assert np.array_equal(out[0, 0], rgb[0, 0])
    assert out[0, 1, 1] > rgb[0, 1, 1]


def test_choose_records_samples_each_split_deterministically():
    rows = []
    for split in ("train", "val", "test"):
        for i in range(5):
            rows.append({"id": f"{split}-{i}", "split": split})
    picked = choose_records(rows, per_split=3)
    assert len(picked) == 9
    assert [r["split"] for r in picked].count("train") == 3
    assert [r["split"] for r in picked].count("val") == 3
    assert [r["split"] for r in picked].count("test") == 3
