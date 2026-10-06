from scripts.prepare_utc_segmentation_patches import block_split, patch_origins

def test_block_split_keeps_whole_block_in_one_split():
    vals={(r,c):block_split(r,c,4) for r in range(4) for c in range(4)}
    assert set(vals.values())=={"train","val","test"}
    assert vals[(0,0)]=="test"
    assert vals[(0,1)]=="val"

def test_patch_origins_are_deterministic_and_in_bounds():
    a=patch_origins(2174,1463,256,32,4,42)
    b=patch_origins(2174,1463,256,32,4,42)
    assert a==b
    assert len(a)==32
    for p in a:
        assert 0<=p["x"]<=2174-256
        assert 0<=p["y"]<=1463-256
        assert p["split"] in {"train","val","test"}


def test_patch_origins_reject_patch_larger_than_spatial_block():
    import pytest
    with pytest.raises(ValueError, match="does not fit inside every"):
        patch_origins(1024,768,256,32,4,42)
