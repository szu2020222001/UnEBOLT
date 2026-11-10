"""Compare separately added losses with the matched sequential MSE control."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from summarize_sampling import latest

ROOT=Path(__file__).resolve().parent


if __name__=='__main__':
    out=ROOT/'runs'/'loss_comparison_20261110';out.mkdir(exist_ok=True)
    rows=[];selection=[];paired=[]
    for model in ('mlp','tcn'):
        subjects={}
        for loss in ('mse','temporal','delta'):
            suffix='' if loss=='mse' else '_'+loss
            run=ROOT/'runs'/f'20261110_{model}_scan_sequential{suffix}_s2026_v1'
            log=pd.read_csv(run/'training'/'training.csv');best=log.loc[log.val_mse.idxmin()]
            selection.append(dict(model=model,loss=loss,best_epoch=int(best.epoch)+1,val_mse=best.val_mse))
            for split in ('val','test'):
                path=latest(run,split)
                df=pd.read_csv(path/f'{split}_summary.csv');df['model']=model;df['loss']=loss;df['split']=split
                rows.append(df)
                if split=='test':subjects[loss]=pd.read_csv(path/'test_subjects.csv').set_index(['subject','band'])
        for loss in ('temporal','delta'):
            for band in subjects[loss].index.get_level_values('band').unique():
                for metric in ('mse','temporal_r','spatial_r','skill','f1_0.25'):
                    d=(subjects[loss].xs(band,level='band')[metric]-subjects['mse'].xs(band,level='band')[metric]).dropna().to_numpy()
                    rng=np.random.default_rng(2026)
                    lo,hi=np.quantile(rng.choice(d,(10000,len(d)),replace=True).mean(1),[.025,.975])
                    paired.append(dict(model=model,loss=loss,band=band,metric=metric,mean_difference_vs_mse=d.mean(),ci_low=lo,ci_high=hi,subjects=len(d)))
    df=pd.concat(rows,ignore_index=True);df.to_csv(out/'all_metrics.csv',index=False)
    pd.DataFrame(selection).to_csv(out/'training_selection.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_subject_differences.csv',index=False)
    table=df[(df.split=='test')&df.metric.isin(['mse','temporal_r'])].pivot(index=['model','band','metric'],columns='loss',values='mean')
    table.to_csv(out/'test_comparison.csv')
    print(pd.DataFrame(selection).to_string(index=False));print(table.to_string())
