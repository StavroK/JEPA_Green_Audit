import pytest
import torch
from scripts.train_utc_segformer import IMAGENET_MEAN,IMAGENET_STD

def test_segformer_normalization_constants():
    assert IMAGENET_MEAN.shape==(3,1,1)
    assert IMAGENET_STD.shape==(3,1,1)

def test_segformer_binary_output_shape_when_transformers_available():
    pytest.importorskip("transformers")
    from scripts.train_utc_segformer import SegFormerBinary
    m=SegFormerBinary(pretrained=False)
    m.eval()
    x=torch.randn(1,3,64,64)
    with torch.no_grad():
        y=m(x)
    assert y.shape==(1,1,64,64)
