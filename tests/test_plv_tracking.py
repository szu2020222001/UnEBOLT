"""Exercise real training/evaluation exports with tiny synthetic scans."""
import json
from pathlib import Path
import numpy as np
import pytest
from engine_plv import main
from experiment_ledger import generate


@pytest.mark.parametrize('kind', ['template', 'mlp'])
def test_run_and_repeated_evaluation_are_separate(tmp_path, kind):
    data=tmp_path/'data'; data.mkdir()
    rng=np.random.default_rng(4)
    scans=[]
    for i,split in enumerate(('train','val','test')):
        filename=f'{split}.npz'
        np.savez(data/filename,x=rng.normal(size=(4,10,3)).astype('float32'),
                 y=rng.uniform(size=(4,1,325)).astype('float32'),frame_index=np.arange(4))
        scans.append(dict(file=filename,split=split,scan=f'sub{i}-scan1',subject=f'sub{i}'))
    manifest=dict(signature='fixture',config=dict(bands={'alpha':[8,13]},frames=10,rois=3),scans=scans)
    (data/'manifest.json').write_text(json.dumps(manifest))
    run=tmp_path/'runs'/'template_seed4'
    args=['--data',str(data),'--output',str(run),'--model',kind,'--device','cpu','--epochs','1']
    main(args)
    training=run/'training'
    assert (training/'state.npz').exists()
    saved=json.loads((training/'run.json').read_text())
    assert saved['status']=='completed' and saved['data_identity']['manifest_sha256']
    assert (training/'source'/'plv_metrics.py').exists()
    before=(training/'run.json').read_bytes()
    main(args+['--eval-only'])
    evaluations=list((run/'evaluations').iterdir())
    assert len(evaluations)==2
    for directory in evaluations:
        meta=json.loads((directory/'evaluation.json').read_text())
        assert meta['status']=='completed' and meta['checkpoint_sha256']
        assert (directory/'val_summary.csv').exists()
    assert (training/'run.json').read_bytes()==before
    assert not list(run.glob('*_summary.csv'))
    html=(run.parent/'index.html').read_text()
    assert 'template_seed4' in html and 'temporal_valid' in html
    with pytest.raises(ValueError,match='not empty'):
        main(args)
    # 配置签名未变，但实际划分/文件内容改变时也必须拒绝复用检查点。
    manifest['scans'][0]['subject']='changed'
    (data/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='contents or split changed'):
        main(args+['--eval-only'])


def test_empty_ledger(tmp_path):
    path=generate(tmp_path)
    assert path.exists() and '暂无匹配' in path.read_text()
