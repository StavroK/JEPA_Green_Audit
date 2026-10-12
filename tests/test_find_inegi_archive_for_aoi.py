from scripts.find_inegi_archive_for_aoi import contains, intersects

def test_bounds_relationships():
    outer=(0,0,10,10)
    assert contains(outer,(2,2,8,8))
    assert not contains(outer,(-1,2,8,8))
    assert intersects(outer,(9,9,12,12))
    assert not intersects(outer,(10,0,12,2))
