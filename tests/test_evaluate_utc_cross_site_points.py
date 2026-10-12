import numpy as np
from scripts.evaluate_utc_cross_site_points import percentile_stretch_rgb, point_metrics, tile_starts

def test_tile_starts_covers_tail():
    assert tile_starts(612,256,128)==[0,128,256,356]

def test_percentile_stretch_shape_dtype():
    x=np.arange(10*10*3,dtype=np.uint8).reshape(10,10,3)
    y=percentile_stretch_rgb(x)
    assert y.shape==x.shape
    assert y.dtype==np.uint8

def test_point_metrics_perfect():
    m=point_metrics(["tree","non","tree","non"],[.9,.1,.8,.2],.5)
    assert m["balanced_accuracy"]==1.0
    assert m["f1_tree"]==1.0
    assert m["confusion_matrix"]=={"tn":2,"fp":0,"fn":0,"tp":2}
