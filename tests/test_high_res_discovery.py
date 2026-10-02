import json

from scripts.discover_high_res_imagery import bbox_from_geometry


def test_bbox_from_polygon():
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [-100.25, 25.66],
            [-100.24, 25.66],
            [-100.24, 25.67],
            [-100.25, 25.67],
            [-100.25, 25.66],
        ]],
    }
    assert bbox_from_geometry(geometry) == [-100.25, 25.66, -100.24, 25.67]
