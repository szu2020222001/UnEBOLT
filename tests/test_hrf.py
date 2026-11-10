import torch
from models.plv_model import Predictor


def test_hrf_mask_and_gradients():
    x=torch.randn(4,10,64)
    models=[]
    for prior in ('none','fixed','learnable'):
        torch.manual_seed(2026)
        m=Predictor(bands=1,kind='tcn_attention',hrf_prior=prior).eval()
        pred=m(x)
        assert pred.shape==(4,1,325)
        torch.testing.assert_close(m.last_attention.sum(-1),torch.ones(4,4,1))
        if prior!='none':
            mu,sigma=m.hrf_parameters()
            torch.testing.assert_close(mu,torch.tensor([4.,5.,6.,7.]))
            torch.testing.assert_close(sigma,torch.full((4,),3.5))
        if prior=='learnable':
            pred.sum().backward()
            assert torch.isfinite(m.hrf_raw_mu.grad).all() and m.hrf_raw_mu.grad.abs().sum()>0
            assert torch.isfinite(m.hrf_raw_sigma.grad).all()
        models.append(m)
    # 加先验不消耗额外初始化随机数：共享网络参数逐项相同。
    for key,value in models[0].state_dict().items():
        torch.testing.assert_close(value,models[2].state_dict()[key])
    torch.testing.assert_close(models[1](x),models[2](x))
