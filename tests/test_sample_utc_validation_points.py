import numpy as np
from scripts.sample_utc_validation_points import exclusion_mask, sample_points

def test_exclusion_mask_marks_training_neighborhood():
    m=exclusion_mask(20,20,[(10,10)],2)
    assert m[10,10]
    assert m[10,12]
    assert not m[10,13]

def test_sample_points_is_spatial_and_deterministic():
    p=np.tile(np.linspace(0,1,20,dtype=np.float32),(20,1))
    ex=np.zeros_like(p,dtype=bool)
    a=sample_points(p,ex,20,2,.5,42)
    b=sample_points(p,ex,20,2,.5,42)
    assert a==b
    assert any(x["model_class"]=="tree" for x in a)
    assert any(x["model_class"]=="non" for x in a)
