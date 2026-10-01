import numpy as np

from scripts.run_m3_geographic_cv import (
    contiguous_column_folds,
    trivial_predictions,
)


def test_contiguous_column_folds_cover_grid_once():
    folds = contiguous_column_folds((8, 8), fold_width=2)
    assert len(folds) == 4

    coverage = np.zeros((8, 8), dtype=int)
    for fold in folds:
        train = fold["train"]
        test = fold["test"]
        assert np.all(~(train & test))
        assert np.all(train | test)
        assert int(test.sum()) == 16
        coverage += test.astype(int)

    assert np.all(coverage == 1)


def test_trivial_predictions():
    y_train = np.array([True, True, False], dtype=bool)

    assert np.all(trivial_predictions("always_vegetation", y_train, 4))
    assert not np.any(trivial_predictions("always_non_vegetation", y_train, 4))
    assert np.all(trivial_predictions("train_majority", y_train, 4))
