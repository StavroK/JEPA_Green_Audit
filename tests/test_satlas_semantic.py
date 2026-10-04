import numpy as np

from scripts.run_satlas_landcover import (
    LAND_COVER_CLASSES,
    PROJECT_GROUPS,
    context_window_for_aoi,
    project_map,
)


def test_satlas_project_taxonomy_covers_native_classes():
    covered = set().union(*PROJECT_GROUPS.values())
    assert covered == set(LAND_COVER_CLASSES)


def test_satlas_context_window_contains_aoi():
    from rasterio.windows import Window

    aoi = Window(col_off=1000.25, row_off=2000.75, width=55.4, height=63.2)
    context, (rows, cols) = context_window_for_aoi(aoi, context_size=512)

    assert context.width == 512
    assert context.height == 512
    assert 0 <= rows.start < rows.stop <= 512
    assert 0 <= cols.start < cols.stop <= 512
    assert rows.stop - rows.start in (63, 64, 65)
    assert cols.stop - cols.start in (55, 56, 57)


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
