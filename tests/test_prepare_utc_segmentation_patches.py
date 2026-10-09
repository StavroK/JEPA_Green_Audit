import numpy as np
from scripts.prepare_utc_segmentation_patches import block_split, patch_origins, raw_rgb

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


def test_la_pastora_patch_layout_fits_spatial_blocks():
    origins=patch_origins(width=612,height=612,patch=192,n=36,blocks=3,seed=42)
    assert len(origins)==36
    assert {r["split"] for r in origins}=={"train","val","test"}
    for r in origins:
        assert 0 <= r["x"] <= 420
        assert 0 <= r["y"] <= 420

def test_raw_rgb_preserves_uint8_values():
    arr=np.arange(3*4*5,dtype=np.uint8).reshape(3,4,5)
    out=raw_rgb(arr)
    assert out.shape==(4,5,3)
    assert out.dtype==np.uint8
    assert np.array_equal(out[...,0],arr[0])
