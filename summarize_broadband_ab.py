"""Summarize latest broadband restart results without test-based selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from summarize_sampling import latest

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    out=ROOT/'runs'/'broadband_comparison';out.mkdir(exist_ok=True)
    rows=[];selection=[];subjects={}
    for model in ('tcn','tcn_attention'):
        for loss in ('mse','temporal','delta','combined'):
            run=ROOT/'runs'/f'bb_{model}_{loss}_s2026'
            config=json.loads((run/'training'/'run.json').read_text())
            selection.append({k:config.get(k) for k in ('model','loss','best_epoch','best_selection_score','parameter_count','selection')})
            for split in ('val','test'):
                path=latest(run,split)
                frame=pd.read_csv(path/f'{split}_summary.csv')
                frame['model']=model;frame['loss']=loss;frame['split']=split
                rows.append(frame)
                if split=='test':subjects[(model,loss)]=pd.read_csv(path/'test_subjects.csv').set_index('subject')
    frame=pd.concat(rows,ignore_index=True);frame.to_csv(out/'all_metrics.csv',index=False)
    pd.DataFrame(selection).to_csv(out/'selection.csv',index=False)
    comparison=frame[frame.split=='test'].pivot(index=['model','loss'],columns='metric',values='mean')
    comparison.to_csv(out/'test_comparison.csv')
    paired=[]
    contrasts=[((m,l),(m,'mse')) for m in ('tcn','tcn_attention') for l in ('temporal','delta','combined')]
    contrasts += [(('tcn_attention',l),('tcn',l)) for l in ('mse','temporal','delta','combined')]
    for a,b in contrasts:
        left,right=subjects[a].align(subjects[b],join='inner',axis=0)
        delta=(left.temporal_r-right.temporal_r).dropna().to_numpy()
        rng=np.random.default_rng(2026)
        lo,hi=np.quantile(rng.choice(delta,(10000,len(delta)),replace=True).mean(1),[.025,.975])
        paired.append(dict(experiment=str(a),reference=str(b),mean_subject_r_difference=delta.mean(),ci_low=lo,ci_high=hi,n=len(delta)))
    pd.DataFrame(paired).to_csv(out/'paired_temporal_differences.csv',index=False)
    print(pd.DataFrame(selection).to_string(index=False))
    print(comparison[['temporal_r','temporal_valid','mse','spatial_r','skill']].to_string())
