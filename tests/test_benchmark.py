import numpy as np

from jepa_green_audit.benchmark import (
    binary_segmentation_metrics,
    geographic_holdout_masks,
    select_labeled_training_blocks,
    spatial_block_ids,
)


def test_binary_segmentation_metrics():
    truth = np.array([[1, 1], [0, 0]], dtype=bool)
    pred = np.array([[1, 0], [1, 0]], dtype=bool)
    metrics = binary_segmentation_metrics(truth, pred)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["iou"] == 1 / 3
    assert metrics["f1_dice"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_geographic_holdouts_do_not_overlap():
    masks = geographic_holdout_masks((16, 16))
    union = masks["train"].astype(int) + masks["val"].astype(int) + masks["test"].astype(int)
    assert np.all(union == 1)
    assert masks["train"].sum() == 128
    assert masks["val"].sum() == 64
    assert masks["test"].sum() == 64


def test_label_fraction_selects_whole_spatial_blocks():
    train = geographic_holdout_masks((16, 16))["train"]
    groups = spatial_block_ids((16, 16), block_size=4)
    selected = select_labeled_training_blocks(train, groups, fraction=0.5, seed=7)
    assert np.all(~selected | train)
    for group in np.unique(groups[selected]):
        group_in_train = train & (groups == group)
        assert np.all(selected[group_in_train])
