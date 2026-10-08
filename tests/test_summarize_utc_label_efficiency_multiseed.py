from scripts.summarize_utc_label_efficiency_multiseed import mean_sd

def test_mean_sd():
    mean,sd=mean_sd([1.0,2.0,3.0])
    assert mean==2.0
    assert round(sd,6)==1.0

def test_single_seed_sd_zero():
    mean,sd=mean_sd([0.5])
    assert mean==0.5
    assert sd==0.0
