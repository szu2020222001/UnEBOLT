"""Summarize latest frozen order-ablation metrics and paired subject differences."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def latest(run,split):
    paths=[p for p in sorted((run/'evaluations').glob('*/evaluation.json'))
           if (lambda m:m['split']==split and m['status']=='completed')(json.loads(p.read_text()))]
    return paths[-1].parent


if __name__=='__main__':
    out=ROOT/'runs'/'sampling_comparison_20261110';out.mkdir(exist_ok=True)
    rows=[]; selections=[]; paired=[]
    for model in ('mlp','tcn'):
        subjects={}
        for sampling in ('global_random','scan_random','scan_sequential'):
            suffix='' if sampling=='global_random' else f'_{sampling}'
            run=ROOT/'runs'/f'20261110_{model}{suffix}_s2026_v1'
            log=pd.read_csv(run/'training'/'training.csv'); best=log.loc[log.val_mse.idxmin()]
            config=json.loads((run/'training'/'run.json').read_text())
            selections.append(dict(model=model,sampling=sampling,best_epoch=int(best.epoch)+1,
                best_val_window_mse=best.val_mse,batches_per_epoch=config.get('batches_per_epoch',159)))
            for split in ('val','test'):
                path=latest(run,split)
                frame=pd.read_csv(path/f'{split}_summary.csv')
                frame['model']=model;frame['sampling']=sampling;frame['split']=split
                rows.append(frame)
                if split=='test': subjects[sampling]=pd.read_csv(path/'test_subjects.csv')
        # 配对差只针对更新次数匹配的两组；整名被试重采样，不重采样窗口。
        left=subjects['scan_sequential'].set_index(['subject','band'])
        right=subjects['scan_random'].set_index(['subject','band'])
        assert left.index.equals(right.index)
        for band in left.index.get_level_values('band').unique():
            for metric in ('mse','temporal_r','spatial_r','f1_0.25','skill'):
                delta=(left.xs(band,level='band')[metric]-right.xs(band,level='band')[metric]).to_numpy()
                delta=delta[np.isfinite(delta)]
                rng=np.random.default_rng(2026)
                boot=rng.choice(delta,(10000,len(delta)),replace=True).mean(1)
                low,high=np.quantile(boot,[.025,.975])
                paired.append(dict(model=model,band=band,metric=metric,
                    mean_subject_difference_seq_minus_random=delta.mean(),ci_low=low,ci_high=high,
                    subjects=len(delta)))
    frame=pd.concat(rows,ignore_index=True)
    frame.to_csv(out/'all_metrics.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'training_selection.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_subject_differences.csv',index=False)
    test=frame[frame.split=='test'].pivot(index=['model','band'],columns=['sampling','metric'],values='mean')
    test.to_csv(out/'test_comparison.csv')
    print(pd.DataFrame(selections).to_string(index=False))
    print(frame[(frame.split=='test')&frame.metric.isin(['mse','temporal_r'])].pivot(index=['model','band','metric'],columns='sampling',values='mean').to_string())
