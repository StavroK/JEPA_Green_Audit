import numpy as np

from scripts.build_2007_resolution_embeddings import (
    DINOV2_COMMIT,
    DINOV2_MODEL,
)
from scripts.run_2007_resolution_cv import (
    geographic_column_folds,
    trivial_predictions,
)


def test_dinov2_is_pinned():
    assert len(DINOV2_COMMIT) == 40
    assert DINOV2_MODEL == "dinov2_vits14"


def test_2007_trivial_predictions():
    y = np.asarray([True, False, False], dtype=bool)
    assert np.all(trivial_predictions("always_vegetation", y, 3))
    assert not np.any(trivial_predictions("always_non_vegetation", y, 3))
    assert not np.any(trivial_predictions("train_majority", y, 3))


def test_2007_geographic_folds_cover_samples_once():
    cols = np.asarray([0, 5, 16, 17, 33, 49, 63])
    folds = geographic_column_folds(cols, 64, folds=4)
    coverage = np.zeros(len(cols), dtype=int)
    for train, test, _ in folds:
        assert np.all(~(train & test))
        coverage += test.astype(int)
    assert np.all(coverage == 1)
