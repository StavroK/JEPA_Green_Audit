import pytest
import torch
from scripts.train_utc_dinov2 import tokens_to_feature_map

def test_tokens_to_feature_map():
    x=torch.randn(2,16,32)
    y=tokens_to_feature_map(x)
    assert y.shape==(2,32,4,4)

def test_tokens_to_feature_map_rejects_non_square():
    x=torch.randn(1,15,8)
    with pytest.raises(ValueError,match="square token grid"):
        tokens_to_feature_map(x)
