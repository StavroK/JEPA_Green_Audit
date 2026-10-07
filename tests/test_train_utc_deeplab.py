import torch
from scripts.train_utc_deeplab import DeepLabBinary

def test_deeplab_binary_output_shape():
    m=DeepLabBinary(pretrained=False)
    m.eval()
    x=torch.randn(1,3,64,64)
    with torch.no_grad():
        y=m(x)
    assert y.shape==(1,1,64,64)
