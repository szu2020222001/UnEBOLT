"""Three-seed HRF summary and fixed-case attention diagnostics (local results)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_sampling import latest

ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    out=ROOT/'runs'/'hrf_comparison';out.mkdir(exist_ok=True)
    rows=[];parameters=[];subjects=[]
    for seed in (2026,2027,2028):
        for prior in ('none','fixed','learnable'):
            run=ROOT/'runs'/f'hrf_{prior}_s{seed}'
            config=json.loads((run/'training'/'run.json').read_text())
            path=latest(run,'test')
            frame=pd.read_csv(path/'test_summary.csv');frame['seed']=seed;frame['prior']=prior
            rows.append(frame)
            sub=pd.read_csv(path/'test_subjects.csv');sub['seed']=seed;sub['prior']=prior;subjects.append(sub)
            with np.load(path/'attention_sub07-scan01.npz') as d:
                actual=d['actual'].mean(0)[:,0,:];content=d['content'].mean(0)[:,0,:]
                time=(np.arange(actual.shape[-1])-2)*2.1
                fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,layout='constrained')
                for h,ax in enumerate(axes.flat):
                    ax.plot(time,content[h],label='Content only',color='#555')
                    ax.plot(time,actual[h],label='Actual',color='#2463d5')
                    if prior!='none':
                        mu=float(d['mu'][h]);sigma=float(d['sigma'][h])
                        bias=-.1*(time-mu)**2/(2*sigma**2)
                        weights=np.exp(bias-bias.max());weights/=weights.sum()
                        ax.plot(time,weights,label='Gaussian prior',ls='--',color='#ca7e23')
                        parameters.append(dict(seed=seed,prior=prior,head=h,mu=mu,sigma=sigma,best_epoch=config['best_epoch']))
                        ax.set_title(f'Head {h+1}: mu={mu:.3f}s, sigma={sigma:.3f}s')
                    else:ax.set_title(f'Head {h+1}: no prior')
                    ax.set(xlabel='Seconds relative to EEG window center',ylabel='Attention',ylim=(0,1));ax.legend(fontsize=8)
                fig.suptitle(f'{prior}, seed {seed} | sub07-scan01 | mean over windows; TCN-mixed features')
                fig.savefig(out/f'attention_{prior}_{seed}.png',dpi=130);plt.close(fig)
            if prior!='none':
                history=pd.read_csv(run/'training'/'training.csv')
                fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
                for h in range(4):
                    axes[0].plot(history.epoch+1,history[f'hrf_mu_{h}'],label=f'Head {h+1}')
                    axes[1].plot(history.epoch+1,history[f'hrf_sigma_{h}'],label=f'Head {h+1}')
                for ax,title in zip(axes,('Time center (seconds)','Width (seconds)')):
                    ax.axvline(config['best_epoch'],color='#aaa',ls='--');ax.set(xlabel='Epoch',title=title);ax.legend()
                fig.savefig(out/f'parameters_{prior}_{seed}.png',dpi=130);plt.close(fig)
    df=pd.concat(rows,ignore_index=True);df.to_csv(out/'all_metrics.csv',index=False)
    table=df.pivot(index=['prior','seed'],columns='metric',values='mean')
    table.to_csv(out/'per_seed.csv')
    summary=table.groupby('prior').agg(['mean','std'])
    summary.to_csv(out/'seed_mean_sd.csv')
    pd.DataFrame(parameters).to_csv(out/'learned_parameters.csv',index=False)
    sub=pd.concat(subjects,ignore_index=True)
    # 每个测试被试先平均三 seed，再做被试配对 bootstrap；不把15个subject-seed当独立样本。
    per=sub.groupby(['prior','subject']).temporal_r.mean().unstack('prior')
    paired=[]
    for prior in ('fixed','learnable'):
        delta=(per[prior]-per['none']).dropna().to_numpy()
        rng=np.random.default_rng(2026)
        low,high=np.quantile(rng.choice(delta,(10000,len(delta)),replace=True).mean(1),[.025,.975])
        paired.append(dict(prior=prior,mean_subject_r_difference=delta.mean(),ci_low=low,ci_high=high,n=len(delta)))
    pd.DataFrame(paired).to_csv(out/'paired_subject_differences.csv',index=False)
    page='<meta charset="utf-8"><title>HRF diagnostics</title><style>body{font:16px system-ui;margin:30px}img{max-width:1100px;width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}</style><h1>HRF 三 seed 对照</h1><p><a href="../index.html">实验台账</a> · <a href="seed_mean_sd.csv">seed均值和标准差</a> · <a href="learned_parameters.csv">最佳检查点的时间参数</a></p><p>seed 标准差仅反映三次训练波动；台账中每次评测的 CI 是被试 bootstrap，二者不同。曲线是 TCN 混合后特征的读取权重，不是生理 HRF 测量。内容注意力是同一模型去除偏置后的诊断，不是重新训练的基线。</p>'
    page+=table[['temporal_r','temporal_valid','mse']].to_html()
    for seed in (2026,2027,2028):
        for prior in ('none','fixed','learnable'):
            page+=f'<h2>{prior} · seed {seed}</h2><img src="attention_{prior}_{seed}.png">'
            if prior!='none':page+=f'<img src="parameters_{prior}_{seed}.png">'
    (out/'index.html').write_text(page,encoding='utf-8')
    print(table[['temporal_r','temporal_valid','mse']].to_string())
    print(summary['temporal_r'].to_string())
    print(pd.DataFrame(paired).to_string(index=False))
