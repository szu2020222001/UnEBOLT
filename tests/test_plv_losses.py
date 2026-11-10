import pytest
import torch
from plv_losses import objective


@pytest.mark.parametrize('kind',['mse','temporal','delta'])
def test_loss_identity_and_finite_backward(kind):
    torch.manual_seed(4)
    y=torch.rand(32,2,10)
    loss,_,_=objective(y,y,kind,1)
    assert loss.item()<1e-4
    pred=torch.full_like(y,.5,requires_grad=True)
    loss,_,_=objective(pred,y,kind,.01)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()


def test_delta_matches_changes_and_temporal_ignores_level():
    y=torch.rand(32,2,10)
    _,_,delta=objective(y+.2,y,'delta')
    _,_,corr=objective(y+.2,y,'temporal')
    assert delta<1e-12 and corr<1e-4
    _,_,delta=objective(torch.flip(y,[0]),y,'delta')
    assert delta>.01
    pred=torch.full((1,2,10),.5,requires_grad=True)
    for kind in ('delta','temporal'):
        loss,_,_=objective(pred,torch.ones_like(pred),kind)
        assert torch.isfinite(loss)
