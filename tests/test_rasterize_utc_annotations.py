from affine import Affine

from scripts.rasterize_utc_annotations import pixel_polygon_to_map, annotations_to_shapes


def test_pixel_polygon_to_map():
    t = Affine.translation(100, 200) * Affine.scale(2, -2)
    pts = pixel_polygon_to_map([[0, 0], [10, 5]], t)
    assert pts == [[100.0, 200.0], [120.0, 190.0]]


def test_annotations_close_polygon():
    payload = {"polygons": [{"points": [[0, 0], [10, 0], [10, 10]]}]}
    shapes = annotations_to_shapes(payload, Affine.identity())
    coords = shapes[0][0]["coordinates"][0]
    assert coords[0] == coords[-1]
    assert shapes[0][1] == 1
