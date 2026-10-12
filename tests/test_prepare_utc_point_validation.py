from scripts.prepare_utc_point_validation import make_points

def test_make_points_is_deterministic_and_spatially_distributed():
    a=make_points(612,612,15,16,42)
    b=make_points(612,612,15,16,42)
    assert a==b
    assert len(a)==240
    assert len({(p["grid_row"],p["grid_col"]) for p in a})==240
    for p in a:
        assert 0 <= p["x"] < 612
        assert 0 <= p["y"] < 612
        assert p["human_label"]==""
