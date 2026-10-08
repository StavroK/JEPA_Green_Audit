import numpy as np
import torch
from scripts.train_utc_ijepa_decoder import IJEPAFeatureDecoder

def test_ijepa_decoder_output_shape():
    m=IJEPAFeatureDecoder(in_channels=1280,hidden=32)
    x=torch.randn(2,1280,16,16)
    y=m(x)
    assert y.shape==(2,1,256,256)

def test_ijepa_decoder_is_trainable():
    m=IJEPAFeatureDecoder(in_channels=1280,hidden=16)
    assert any(p.requires_grad for p in m.parameters())
