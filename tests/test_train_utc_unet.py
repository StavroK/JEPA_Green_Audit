import torch
from scripts.train_utc_unet import UNetSmall, dice_loss, metrics_from_counts

def test_unet_output_shape():
    m=UNetSmall(base=8)
    x=torch.randn(2,3,64,64)
    y=m(x)
    assert y.shape==(2,1,64,64)

def test_dice_loss_is_finite():
    logits=torch.zeros(1,1,8,8)
    target=torch.zeros(1,1,8,8)
    loss=dice_loss(logits,target)
    assert torch.isfinite(loss)

def test_metrics_from_counts():
    m=metrics_from_counts(8,2,9,1)
    assert round(m["precision"],3)==0.8
    assert round(m["recall"],3)==0.889
    assert round(m["iou"],3)==0.727
