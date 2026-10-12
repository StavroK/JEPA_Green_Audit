import json
from scripts.make_utc_cross_site_figures import aggregate_rows, consensus_points

def test_aggregate_rows_extracts_metrics():
    s={"aggregate":{"dinov2":{"25":{"metrics":{
        "balanced_accuracy":{"mean":.8,"sd":.1},
        "f1_tree":{"mean":.85,"sd":.02},
        "iou_tree":{"mean":.74,"sd":.03},
        "point_prevalence_bias_pp":{"mean":-4,"sd":2}
    }}}}}
    rows=aggregate_rows(s)
    assert len(rows)==1
    assert rows[0]["model"]=="dinov2"
    assert rows[0]["budget"]==25
    assert rows[0]["f1_mean"]==.85

def test_consensus_points_marks_majority_error():
    runs=[
        {"points":[{"id":1,"x":3,"y":4,"human_label":"tree","model_class":"non"}]},
        {"points":[{"id":1,"x":3,"y":4,"human_label":"tree","model_class":"non"}]},
        {"points":[{"id":1,"x":3,"y":4,"human_label":"tree","model_class":"tree"}]},
    ]
    p=consensus_points(runs)[0]
    assert p["majority_class"]=="non"
    assert p["correct"] is False
    assert abs(p["agreement"]-2/3)<1e-9
