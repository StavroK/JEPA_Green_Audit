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


def test_load_training_points_includes_hard_negative_subtypes(tmp_path):
    import json
    from scripts.sample_utc_validation_points import load_training_points

    p=tmp_path/"labels.json"
    p.write_text(json.dumps({"labels":[
        {"x":1,"y":1,"label":"tree"},
        {"x":2,"y":2,"label":"pavement"},
        {"x":3,"y":3,"label":"water"},
        {"x":4,"y":4,"label":"grass"},
        {"x":5,"y":5,"label":"roof"},
        {"x":6,"y":6,"label":"shadow"},
        {"x":7,"y":7,"label":"bare"},
        {"x":8,"y":8,"label":"uncertain"},
    ]}),encoding="utf-8")

    pts=load_training_points(p)
    assert pts==[(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)]
