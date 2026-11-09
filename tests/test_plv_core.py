import numpy as np
import torch
from dataset_maker.plv_dataset import plv_edges
from models.plv_model import Predictor
from plv_metrics import metrics


def test_phase_locking():
    phase=np.tile(np.linspace(0,40,1000),(26,1))+np.arange(26)[:,None]
    np.testing.assert_allclose(plv_edges(phase),1,atol=1e-6)


def test_metrics_identity_and_constant():
    y=np.random.default_rng(1).uniform(size=(30,325))
    m=metrics(y,y,y.mean(0))
    assert m['mse']==0 and m['f1_0.25']==1 and m['temporal_r']>.999
    c=metrics(y,np.broadcast_to(y.mean(0),y.shape),y.mean(0))
    assert c['temporal_valid']==0 and np.isnan(c['temporal_r'])


def test_model_shape_and_backward():
    for kind in ('mlp','tcn'):
        m=Predictor(kind=kind)
        y=m(torch.randn(3,10,64))
        assert y.shape==(3,6,325) and bool(((y>0)&(y<1)).all())
        y.mean().backward()

