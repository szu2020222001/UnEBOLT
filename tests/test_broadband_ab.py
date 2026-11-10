import json
import numpy as np
import pytest
import torch
from models.plv_model import Predictor
from plv_losses import objective
from engine_plv import main


def test_attention_model_time_and_combined_loss():
    model=Predictor(bands=1,kind='tcn_attention')
    x=torch.randn(3,10,64)
    pred=model(x)
    assert pred.shape==(3,1,325)
    assert not torch.equal(model.time_position[0],model.time_position[1])
    np.testing.assert_allclose(model.last_attention.detach().sum(-1).numpy(),1,atol=1e-6)
    y=torch.rand_like(pred)
    total,mse,_=objective(pred,y,'combined',.01,1)
    _,_,corr=objective(pred,y,'temporal')
    _,_,delta=objective(pred,y,'delta')
    torch.testing.assert_close(total,mse+.01*corr+delta)
    total.backward()
    assert torch.isfinite(model.query.grad).all()


@pytest.mark.parametrize('kind',['tcn','tcn_attention'])
def test_single_band_selection_and_reload(tmp_path,kind):
    root=tmp_path/'data';root.mkdir()
    scans=[];rng=np.random.default_rng(3)
    for i,split in enumerate(('train','val','test')):
        file=f'{split}.npz'
        np.savez(root/file,x=rng.normal(size=(12,10,3)).astype('float32'),
            y=rng.uniform(size=(12,2,325)).astype('float32'),frame_index=np.arange(12))
        scans.append(dict(file=file,split=split,subject=f'sub{i}',scan=f'sub{i}-scan1',windows=12))
    (root/'manifest.json').write_text(json.dumps(dict(signature='test',config=dict(rois=3,frames=10,bands={'alpha':[8,13],'broadband':[.5,45]}),scans=scans)))
    run=tmp_path/'runs'/'example'
    args=['--data',str(root),'--output',str(run),'--model',kind,'--bands','broadband',
          '--selection','temporal_r','--sampling','scan_sequential','--loss','combined','--device','cpu','--epochs','2']
    main(args)
    config=json.loads((run/'training'/'run.json').read_text())
    assert config['bands']==['broadband'] and config['best_epoch'] in (1,2)
    main(['--data',str(root),'--output',str(run),'--model',kind,'--eval-only','--eval-split','test','--device','cpu'])
    for path in (run/'evaluations').glob('*/test_*_predictions.npz'):
        with np.load(path) as d:assert d['prediction'].shape==(12,1,325)
