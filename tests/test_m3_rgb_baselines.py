import numpy as np

from scripts.build_m3_rgb_baseline_features import cell_box
from scripts.run_m3_geographic_cv import contiguous_column_folds


def test_m3_rgb_cell_box_partitions_image_edges():
    width, height = 232, 146
    boxes = [cell_box(width, height, r, c) for r in range(8) for c in range(8)]
    assert boxes[0] == (0, 0, 29, 18)
    assert boxes[-1][2] == width
    assert boxes[-1][3] == height


def test_m3_geographic_folds_keep_columns_disjoint():
    folds = contiguous_column_folds((8, 8), fold_width=2)
    coverage = np.zeros((8, 8), dtype=int)
    for fold in folds:
        train = fold["train"]
        test = fold["test"]
        assert np.all(~(train & test))
        coverage += test.astype(int)
    assert np.all(coverage == 1)
