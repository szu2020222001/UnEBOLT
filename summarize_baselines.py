"""Export the frozen baseline run summaries without selecting by test score."""
import json
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parent
rows=[]
training=[]
for model in ('template','ridge','mlp','tcn'):
    run=root/'runs'/f'20261110_{model}_s2026_v1'
    for meta in sorted((run/'evaluations').glob('*/evaluation.json')):
        record=json.loads(meta.read_text())
        if record['status']!='completed':
            continue
        split=record['split']
        df=pd.read_csv(meta.parent/f'{split}_summary.csv')
        df['model']=model; df['split']=split; df['evaluation']=meta.parent.name
        rows.append(df)
    log=run/'training'/'training.csv'
    if log.exists():
        df=pd.read_csv(log); best=df.loc[df.val_mse.idxmin()]
        training.append(dict(model=model,best_epoch_zero_based=int(best.epoch),
                             best_val_window_mse=best.val_mse,final_train_mse=df.iloc[-1].train_mse,
                             final_val_mse=df.iloc[-1].val_mse))
out=root/'runs'/'baseline_summary_20261110';out.mkdir(exist_ok=True)
allrows=pd.concat(rows,ignore_index=True).sort_values('evaluation').drop_duplicates(['model','split','band','metric'],keep='last')
allrows.to_csv(out/'all_metrics.csv',index=False)
pd.DataFrame(training).to_csv(out/'training_selection.csv',index=False)
test=allrows[allrows.split=='test']
table=test.pivot(index=['band','model'],columns='metric',values='mean')
table.to_csv(out/'test_comparison.csv')
print(pd.DataFrame(training).to_string(index=False))
print(table[['mse','skill','temporal_r','spatial_r','f1_0.25']].to_string())
