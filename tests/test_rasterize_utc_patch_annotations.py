import numpy as np
from scripts.rasterize_utc_patch_annotations import clip_polygon, rasterize_patch

def test_clip_polygon_keeps_edge_polygons_in_bounds():
    pts=clip_polygon([[-1,-2],[256,0],[255,256],[0,255]],256,256)
    assert pts==[(0,0),(255,0),(255,255),(0,255)]

def test_rasterize_patch_binary_mask():
    mask=rasterize_patch([[[2,2],[8,2],[8,8],[2,8]]],16,16)
    assert mask.shape==(16,16)
    assert set(np.unique(mask)).issubset({0,255})
    assert mask[5,5]==255
    assert mask[0,0]==0
