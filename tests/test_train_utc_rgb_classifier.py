import numpy as np

from scripts.train_utc_rgb_classifier import feature_stack, parse_labels


def test_feature_stack_shape():
    rgb=np.zeros((5,7,3),dtype=np.float32)
    feats,names=feature_stack(rgb)
    assert feats.shape[:2]==(5,7)
    assert feats.shape[2]==len(names)
    assert "exg" in names
    assert "texture5" in names
    assert "texture9" in names
    assert "texture15" in names
    assert "color_var9" in names


def test_parse_labels_ignores_uncertain_and_out_of_bounds():
    payload={"labels":[
        {"x":1,"y":2,"label":"tree"},
        {"x":3,"y":4,"label":"non"},
        {"x":0,"y":0,"label":"uncertain"},
        {"x":999,"y":1,"label":"tree"},
    ]}
    xs,ys,y=parse_labels(payload,10,10)
    assert xs.tolist()==[1,3]
    assert ys.tolist()==[2,4]
    assert y.tolist()==[1,0]


def test_parse_labels_accepts_hard_negative_subtypes():
    payload={"labels":[
        {"x":1,"y":1,"label":"tree"},
        {"x":2,"y":2,"label":"water"},
        {"x":3,"y":3,"label":"pavement"},
        {"x":4,"y":4,"label":"grass"},
        {"x":5,"y":5,"label":"roof"},
        {"x":6,"y":6,"label":"shadow"},
        {"x":7,"y":7,"label":"bare"},
    ]}
    xs,ys,y=parse_labels(payload,10,10)
    assert y.tolist()==[1,0,0,0,0,0,0]


def test_v2_hard_negatives_need_tree_labels_from_another_file():
    payload={"labels":[
        {"x":1,"y":1,"label":"pavement"},
        {"x":2,"y":2,"label":"water"},
        {"x":3,"y":3,"label":"grass"},
    ]}
    xs,ys,y=parse_labels(payload,10,10)
    assert y.tolist()==[0,0,0]
