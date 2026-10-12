from scripts.validate_utc_point_labels import summarize

def test_summarize_point_labels():
    payload={"points":[
        {"id":1,"grid_row":0,"grid_col":0,"human_label":"tree","detail_label":"tree","reviewer":"B"},
        {"id":2,"grid_row":0,"grid_col":1,"human_label":"non","detail_label":"roof","reviewer":"B"},
        {"id":3,"grid_row":0,"grid_col":2,"human_label":"uncertain","detail_label":"uncertain","reviewer":"B"},
    ]}
    r=summarize(payload)
    assert r["n_points"]==3
    assert r["human_counts"]=={"tree":1,"non":1,"uncertain":1}
    assert r["invalid_labels"]==[]
    assert r["duplicate_grid_cells"]==[]
