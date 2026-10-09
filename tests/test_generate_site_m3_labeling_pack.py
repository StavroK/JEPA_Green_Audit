from scripts.generate_site_m3_labeling_pack import grid_edges

def test_grid_edges_cover_source():
    e=grid_edges(61,16)
    assert len(e)==17
    assert e[0]==0
    assert e[-1]==61
    assert all(b>a for a,b in zip(e[:-1],e[1:]))
