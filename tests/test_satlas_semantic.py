import numpy as np

from scripts.run_satlas_landcover import (
    LAND_COVER_CLASSES,
    PROJECT_GROUPS,
    center_pad_512,
    project_map,
)


def test_satlas_project_taxonomy_covers_native_classes():
    covered = set().union(*PROJECT_GROUPS.values())
    assert covered == set(LAND_COVER_CLASSES)


def test_satlas_center_pad_round_trip_shape():
    rgb = np.zeros((63, 55, 3), dtype=np.uint8)
    padded, (top, bottom, left, right) = center_pad_512(rgb)
    assert padded.shape == (512, 512, 3)
    restored = padded[top:top+63, left:left+55]
    assert restored.shape == rgb.shape


def test_satlas_project_map_groups_tree_and_grass():
    native_names = np.asarray([["tree", "grass", "developed", "bare", "water"]], dtype=object)
    mapped = project_map(native_names)
    names = list(PROJECT_GROUPS)
    decoded = [names[i] for i in mapped[0]]
    assert decoded == [
        "tree_canopy",
        "other_vegetation",
        "developed",
        "bare_ground",
        "water",
    ]
