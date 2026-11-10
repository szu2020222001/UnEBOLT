"""Fixed-case, offline PLV visualizations; never select examples by model score."""
import argparse
import html
import json
from pathlib import Path
from uuid import uuid4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import mne
import numpy as np
from plv_metrics import corr_rows
from plv_tracking import now, sha256, write_json, training_dir

PAIRS=[('F3','F4'),('C3','C4'),('O1','O2'),('FZ','PZ')]


def matrix(edges, channels):
    """严格上三角向量恢复对称矩阵；对角线只用于显示，不进入统计。"""
    value=np.eye(channels)
    ij=np.triu_indices(channels,1)
    value[ij]=edges; value[(ij[1],ij[0])]=edges
    return value


def sensors(channels):
    # 使用 MNE 公共 montage API；标准位置不表示个体脑源位置。
    montage=mne.channels.make_standard_montage('standard_1020')
    lookup={name.upper():name for name in montage.ch_names}
    names=[lookup[c.upper()] for c in channels]
    info=mne.create_info(names,200,ch_types='eeg')
    info.set_montage(montage)
    positions=montage.get_positions()['ch_pos']
    xyz=np.array([positions[n] for n in names])
    # 极角投影到二维头皮平面，用于连接图；topomap 使用 MNE 自身投影。
    radius=np.linalg.norm(xyz,axis=1)
    angle=np.arccos(np.clip(xyz[:,2]/radius,-1,1))
    azimuth=np.arctan2(xyz[:,1],xyz[:,0])
    xy=np.c_[angle*np.cos(azimuth),angle*np.sin(azimuth)]
    xy*=.095/np.max(np.linalg.norm(xy,axis=1))
    return info,xy


def connections(ax, edges, xy, names, error=False):
    ij=np.array(np.triu_indices(len(names),1)).T
    count=int(np.ceil(.25*len(edges)))
    order=np.argsort(-np.abs(edges) if error else -edges,kind='stable')[:count]
    lines=LineCollection(xy[ij[order]],cmap='RdBu_r' if error else 'viridis',
                         norm=Normalize(-1,1) if error else Normalize(0,1),linewidths=1,alpha=.8)
    lines.set_array(edges[order]);ax.add_collection(lines)
    ax.add_patch(plt.Circle((0,0),.102,fill=False,color='#777'))
    ax.plot([-.012,0,.012],[.101,.114,.101],color='#777')
    ax.scatter(xy[:,0],xy[:,1],s=12,c='#222',zorder=3)
    for pos,name in zip(xy,names): ax.annotate(name,pos,xytext=(3,3),textcoords='offset points',fontsize=6)
    ax.set(xlim=(-.12,.12),ylim=(-.12,.12),aspect='equal');ax.axis('off')
    return lines


def snapshot(y,pred,n,band,channels,info,xy,start,end,frame,path):
    real,estimate=matrix(y[n],len(channels)),matrix(pred[n],len(channels))
    error=estimate-real
    fig,axes=plt.subplots(3,3,figsize=(16,14),layout='constrained')
    for col,(values,title) in enumerate(zip((real,estimate,error),('True','Predicted','Predicted - True'))):
        signed=col==2
        masked=values.copy();np.fill_diagonal(masked,np.nan)
        image=axes[0,col].imshow(masked,vmin=-1 if signed else 0,vmax=1,cmap='RdBu_r' if signed else 'viridis')
        axes[0,col].set_title(title+' | PLV matrix')
        axes[0,col].set_xticks(range(len(channels)),channels,rotation=90,fontsize=6)
        axes[0,col].set_yticks(range(len(channels)),channels,fontsize=6)
        fig.colorbar(image,ax=axes[0,col],shrink=.8)
        edges=values[np.triu_indices(len(channels),1)]
        image=connections(axes[1,col],edges,xy,channels,error=signed)
        axes[1,col].set_title('Largest absolute errors (top 25%)' if signed else title+' | strongest 25% edges')
        fig.colorbar(image,ax=axes[1,col],shrink=.8)
        # 非对角连接均值；误差列是有符号均值，可能相互抵消。
        strength=np.nansum(masked,axis=1)/(len(channels)-1)
        image,_=mne.viz.plot_topomap(strength,info,axes=axes[2,col],show=False,
            cmap='RdBu_r' if signed else 'viridis',vlim=(-1,1) if signed else (0,1),
            contours=0,sensors=True,names=info.ch_names,extrapolate='head')
        axes[2,col].set_title(title+' | mean incident PLV' if not signed else 'Mean signed edge error per electrode')
        fig.colorbar(image,ax=axes[2,col],shrink=.8)
    spatial=float(corr_rows(y[n:n+1],pred[n:n+1])[0])
    fig.suptitle(f'{band} | EEG [{start[n]:.2f}, {end[n]:.2f}) s | BOLD frames {frame[n]}..{frame[n]+9}\n'
                 f'MAE={np.abs(pred[n]-y[n]).mean():.4f}; spatial r={spatial:.4f} | standard 10-20 montage',fontsize=13)
    fig.savefig(path,dpi=110);plt.close(fig)


def timeplot(y,pred,template,channels,start,end,band,path):
    ij=list(zip(*np.triu_indices(len(channels),1)))
    t=(start+end)/2
    fig,axes=plt.subplots(6,1,figsize=(15,15),sharex=True,layout='constrained')
    for ax,(a,b) in zip(axes,PAIRS):
        i,j=sorted((channels.index(a),channels.index(b)));edge=ij.index((i,j))
        ax.plot(t,y[:,edge],label='True',color='#222',lw=1)
        ax.plot(t,pred[:,edge],label='Predicted',color='#2463d5',lw=1)
        ax.axhline(template[edge],label='Training template',color='#c27622',ls='--',lw=1)
        r=float(corr_rows(y[:,edge][None],pred[:,edge][None])[0])
        ax.set_title(f'{a}-{b} | temporal r={r:.3f}, MAE={np.abs(pred[:,edge]-y[:,edge]).mean():.4f}',loc='left',fontsize=10)
        ax.set_ylim(0,1);ax.set_ylabel('PLV');ax.grid(alpha=.2)
    axes[0].legend(ncols=3,loc='upper right')
    axes[4].plot(t,y.mean(1),label='True',color='#222')
    axes[4].plot(t,pred.mean(1),label='Predicted',color='#2463d5')
    axes[4].axhline(template.mean(),color='#c27622',ls='--')
    axes[4].set(ylabel='Mean PLV',ylim=(0,1),title='Mean over all 325 edges')
    axes[5].plot(t,np.abs(pred-y).mean(1),color='#b53c48')
    axes[5].set(ylabel='MAE',ylim=(0,1),title='Window-wise MAE over all 325 edges',xlabel='EEG label window center (seconds from recording start)')
    fig.suptitle(f'{band} | fixed electrode pairs | 8.4 s EEG windows, stride 2.1 s; overlapping observations',fontsize=13)
    fig.savefig(path,dpi=110);plt.close(fig)


def generate(run,data,scan='sub07-scan01'):
    run,data=Path(run),Path(data)
    evaluations=[]
    for path in sorted((run/'evaluations').glob('*/evaluation.json')):
        meta=json.loads(path.read_text())
        if meta['status']=='completed' and meta['split']=='test':evaluations.append((path.parent,meta))
    if not evaluations: raise ValueError(f'No completed test evaluation: {run}')
    evaluation,meta=evaluations[-1]
    manifest=json.loads((data/'manifest.json').read_text())
    if sha256(data/'manifest.json')!=meta['data_identity']['manifest_sha256']:
        raise ValueError('Visualization manifest differs from evaluated data')
    row=next(r for r in manifest['scans'] if r['scan']==scan)
    if row['split']!='test':raise ValueError('Fixed case must be in test split')
    if sha256(data/row['file'])!=meta['data_identity']['files'][row['file']]:raise ValueError('Scan content changed')
    with np.load(data/row['file']) as d:
        y=d['y'];frame=d['frame_index'];start=d['eeg_start'];end=d['eeg_end']
    with np.load(evaluation/f'test_{scan}_predictions.npz') as d:
        pred=d['prediction']
        trained=json.loads((training_dir(run)/'run.json').read_text())
        source_bands=list(manifest['config']['bands'])
        bands=trained.get('bands') or source_bands
        y=y[:,[source_bands.index(b) for b in bands]]
        np.testing.assert_array_equal(y,d['target'])
    if pred.shape!=y.shape or not np.isfinite(pred).all():raise ValueError('Invalid predictions')
    with np.load(training_dir(run)/'state.npz') as d:template=d['template']
    channels=manifest['config']['channels']
    info,xy=sensors(channels)
    out=evaluation/'visualizations'/f'{scan}_{uuid4().hex[:8]}';out.mkdir(parents=True)
    selected=[int(round(q*(len(y)-1))) for q in (.25,.5,.75)]
    for f,band in enumerate(bands):
        timeplot(y[:,f],pred[:,f],template[f],channels,start,end,band,out/f'{band}_time.png')
        for q,n in zip((25,50,75),selected):
            snapshot(y[:,f],pred[:,f],n,band,channels,info,xy,start,end,frame,out/f'{band}_{q}.png')
    settings=dict(scan=scan,selection='25/50/75 percent of valid windows; fixed before viewing scores',
        windows=[dict(index=n,frame=int(frame[n]),start=float(start[n]),end=float(end[n])) for n in selected],
        electrode_pairs=PAIRS,montage='standard_1020 (standard coordinates, not individual localization)',
        scales=dict(plv=[0,1],signed_error=[-1,1]),created_at=now(),
        evaluation=str(evaluation),prediction_sha256=sha256(evaluation/f'test_{scan}_predictions.npz'),
        script_sha256=sha256(__file__))
    write_json(out/'visualization.json',settings)
    title=html.escape(run.name+' / '+scan)
    page='''<!doctype html><meta charset="utf-8"><title>PLV fixed case</title>
<style>body{font:16px system-ui;max-width:1450px;margin:24px auto;padding:0 20px;color:#203047}img{width:100%;border:1px solid #ddd}select{padding:8px;margin:10px}p{line-height:1.7}a{color:#2463d5}</style>'''
    page+=f'<h1>{title}</h1><p><a href="../../../../../index.html">返回实验台账</a></p>'
    page+='''<p>固定测试案例；各实验共用同一扫描和窗口。每点是 8.4 秒 EEG 窗口的 PLV，步长 2.1 秒；横轴为窗口中心，非瞬时或独立观测。曲线不平滑、不按模型分数选边。此案例不代表测试集总体。</p>
<p>矩阵屏蔽对角线；头皮连接图各取最强 25% 边，误差图取绝对误差最大的 25%。所有实验 PLV 色标固定 [0,1]、误差固定 [-1,1]。拓扑图每电极数值为其 25 条连接的均值，误差列是有符号均值，可能抵消；不是 EEG 电位或脑源激活图。位置为标准 10-20 montage 示意。</p>
<label>频段 <select id="band">'''
    page+=''.join(f'<option {"selected" if b=="alpha" else ""}>{b}</option>' for b in bands if b=='broadband')
    page+='''</select></label><label>固定窗口位置 <select id="window"><option value="25">25%</option><option value="50" selected>50%</option><option value="75">75%</option></select></label>
<h2>时间变化与逐窗口误差</h2><a id="timeLink"><img id="time" alt="PLV time courses"></a><h2>单窗口：矩阵、头皮连接和电极均值拓扑</h2><a id="snapshotLink"><img id="snapshot" alt="True, predicted and error spatial comparison"></a>
<p>点击图片打开原图。各模型在实验台账中进入对应页面，可使用相同频段和窗口并排查看。</p>
<script>const band=document.getElementById('band'),win=document.getElementById('window');function render(){const t=band.value+'_time.png',s=band.value+'_'+win.value+'.png';document.getElementById('time').src=t;document.getElementById('timeLink').href=t;document.getElementById('snapshot').src=s;document.getElementById('snapshotLink').href=s}band.onchange=win.onchange=render;render();</script>'''
    (out/'index.html').write_text(page,encoding='utf-8')
    # 只添加可视化引用，不改模型或评测分数；旧版本目录保留。
    meta['visualization']=str((out/'index.html').relative_to(evaluation))
    write_json(evaluation/'evaluation.json',meta)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',default='runs')
    parser.add_argument('--data',default='artifacts/data_common_10bold_4tr_6bands')
    parser.add_argument('--scan',default='sub07-scan01')
    parser.add_argument('--only-run')
    args=parser.parse_args()
    runs=[Path(args.only_run)] if args.only_run else sorted(p for p in Path(args.runs).iterdir() if (training_dir(p)/'run.json').exists())
    for run in runs: print(generate(run,args.data,args.scan),flush=True)
    from experiment_ledger import generate as ledger
    print(ledger(args.runs))
