"""Train/evaluate cross-subject baselines with a frozen train-only scaler."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
import joblib
from plv_tracking import now, write_json, save_provenance, training_dir, new_evaluation, data_identity, sha256
from experiment_ledger import generate
from models.plv_model import Predictor
from plv_sampling import ScanBatchSampler
from plv_losses import objective
from plv_metrics import metrics, fisher_mean, corr_rows


def read_scan(root, row):
    with np.load(root/row['file']) as d:
        return d['x'],d['y']


def aggregate(rows):
    """扫描记录 → 被试记录 → 每频段总体统计。

    rows 每行是一个扫描/频段的 metrics() 结果。
    一人多扫描先等权汇总，再对被试等权；不把重叠窗口当独立统计样本。
    返回扫描表、被试表和逐指标汇总表。
    """
    frame = pd.DataFrame(rows)
    keys = [c for c in frame if c not in ('scan','subject','band')]
    subjects=[]
    # 相关用 Fisher-z 平均；误差、F1、有效率等用算术平均。
    for (u,f), group in frame.groupby(['subject','band']):
        row=dict(subject=u,band=f)
        for k in keys:
            row[k] = fisher_mean(group[k]) if k in ('spatial_r','temporal_r') else float(group[k].mean())
        subjects.append(row)
    summary=[]
    sub=pd.DataFrame(subjects)
    # 固定 bootstrap 随机数，按整名被试有放回重采样 1000 次。
    rng=np.random.default_rng(2026)
    for f,group in sub.groupby('band'):
        for k in keys:
            # 各指标排除 NaN 后的被试数可能不同，valid_subjects 会明确记录。
            v=group[k].to_numpy(); v=v[np.isfinite(v)]
            fn=fisher_mean if k in ('spatial_r','temporal_r') else np.mean
            point=float(fn(v)) if len(v) else np.nan
            boot=[fn(rng.choice(v,len(v),replace=True)) for _ in range(1000)] if len(v)>1 else []
            ci=np.quantile(boot,[.025,.975]) if boot else [np.nan,np.nan]
            # CI 为 bootstrap 百分位区间；subject_sd 是原始指标尺度上的样本标准差。
            # 只有一名有效被试时不计算跨被试 SD 或 CI。
            summary.append(dict(band=f,metric=k,mean=point,subject_sd=float(np.std(v,ddof=1)) if len(v)>1 else np.nan,
                                ci_low=ci[0],ci_high=ci[1],valid_subjects=len(v)))
    return frame,sub,pd.DataFrame(summary)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--model',choices=['template','ridge','mlp','tcn','tcn_attention'],default='tcn')
    p.add_argument('--hrf-prior',choices=['none','fixed','learnable'],default='none')
    p.add_argument('--hrf-strength',type=float,default=.1)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--sampling',choices=['global_random','scan_random','scan_sequential'],default='global_random')
    p.add_argument('--loss',choices=['mse','temporal','delta','combined'],default='mse')
    p.add_argument('--bands',nargs='+',default=None,help='Select label bands, e.g. broadband')
    p.add_argument('--selection',choices=['mse','temporal_r'],default='mse')
    p.add_argument('--delta-weight',type=float,default=1.)
    p.add_argument('--aux-weight',type=float,default=.01)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--device',default='cuda')
    p.add_argument('--eval-only',action='store_true')
    p.add_argument('--notes',default='',help='Purpose/change being tested; saved in the experiment ledger')
    p.add_argument('--eval-split',choices=['val','test'],default='val')
    a=p.parse_args(argv)
    if a.aux_weight<0 or a.delta_weight<0: p.error('--aux-weight must be nonnegative')
    if not a.eval_only and a.loss!='mse' and (a.sampling!='scan_sequential' or a.model not in ('mlp','tcn','tcn_attention')):
        p.error('Sequence losses require neural model and --sampling scan_sequential')
    if a.hrf_prior!='none' and a.model!='tcn_attention': p.error('HRF prior requires tcn_attention')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False
    root,run=Path(a.data),Path(a.output)
    run.mkdir(parents=True,exist_ok=True)
    out=training_dir(run) if a.eval_only else run/'training'
    if not a.eval_only:
        if any(run.iterdir()):
            raise ValueError('Run directory is not empty; use a new output directory')
        out.mkdir()
    manifest=json.loads((root/'manifest.json').read_text())
    config=manifest['config']; all_bands=list(config['bands'])
    if a.eval_only:
        saved_config=json.loads((out/'run.json').read_text())
        if a.bands is not None and a.bands!=saved_config.get('bands',all_bands):
            raise ValueError('Requested bands differ from trained bands')
        a.bands=saved_config.get('bands',all_bands)
        a.hrf_prior=saved_config.get('hrf_prior','none')
        a.hrf_strength=saved_config.get('hrf_strength',.1)
    bands=a.bands or all_bands
    if not bands or len(set(bands))!=len(bands) or any(b not in all_bands for b in bands):
        raise ValueError('Invalid bands')
    a.bands=bands
    band_indices=[all_bands.index(b) for b in bands]
    def load_scan(row):
        x,y=read_scan(root,row)
        return x,y[:,band_indices]
    identity=data_identity(root,manifest)
    by={s:[r for r in manifest['scans'] if r['split']==s] for s in ('train','val','test')}
    if not all(by.values()): raise ValueError('A complete train/val/test manifest is required')
    def concat(part):
        pairs=[load_scan(r) for r in by[part]]
        return tuple(np.concatenate([z[i] for z in pairs]) for i in (0,1))
    model=None; reg=None
    statepath=out/'state.npz'
    if not a.eval_only and statepath.exists():
        raise ValueError('Run already exists; use a new output directory or --eval-only')
    if a.eval_only:
        state=np.load(statepath); mean,std,template=state['mean'],state['std'],state['template']
        saved=json.loads((out/'run.json').read_text())
        if saved['data_signature']!=manifest['signature'] or saved['model']!=a.model:
            raise ValueError('Model/data configuration mismatch')
        if saved.get('data_identity') and saved['data_identity']!=identity:
            raise ValueError('Dataset contents or split changed since training')
    else:
        xt,yt=concat('train'); xv,yv=concat('val')
        # One contribution per physical BOLD frame, not duplicated overlapping windows.
        frames=[]; scan_templates=[]
        for r in by['train']:
            with np.load(root/r['file']) as d:
                idx=(d['frame_index'][:,None]+np.arange(config['frames'])).ravel()
                _,first=np.unique(idx,return_index=True)
                frames.append(d['x'].reshape(-1,config['rois'])[first])
                scan_templates.append((r['subject'],d['y'][:,band_indices].mean(0)))
        allframes=np.concatenate(frames)
        mean=allframes.mean(0); std=allframes.std(0).clip(1e-6)
        us=sorted(set(u for u,_ in scan_templates))
        template=np.mean([np.mean([v for u,v in scan_templates if u==s],axis=0) for s in us],axis=0)
        np.savez(statepath,mean=mean,std=std,template=template)
        record=dict(vars(a),data_signature=manifest['signature'],data_identity=identity,
                    torch=torch.__version__,cuda=torch.version.cuda,started_at=now(),status='running',
                    provenance=save_provenance(out))
        write_json(out/'run.json',record)
        write_json(out/'manifest.json',manifest)
        xt=((xt-mean)/std).astype('float32'); xv=((xv-mean)/std).astype('float32')
        if a.model=='ridge':
            best=float('inf'); scores=[]
            for alpha in (.1,1,10,100,1000):
                candidate=Ridge(alpha=alpha).fit(xt.reshape(len(xt),-1),yt.reshape(len(yt),-1))
                pred=np.clip(candidate.predict(xv.reshape(len(xv),-1)),0,1).reshape(yv.shape)
                score=float(np.mean((pred-yv)**2)); scores.append(dict(alpha=alpha,val_mse=score))
                if score<best: best,reg=score,candidate
            joblib.dump(reg,out/'ridge.joblib'); pd.DataFrame(scores).to_csv(out/'ridge_search.csv',index=False)
        elif a.model in ('mlp','tcn','tcn_attention'):
            model=Predictor(config['rois'],config['frames'],len(bands),kind=a.model,tr=config.get('tr',2.1),label_trs=config.get('label_trs',4),hrf_prior=a.hrf_prior,hrf_strength=a.hrf_strength).to(a.device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
            dataset=TensorDataset(torch.from_numpy(xt),torch.from_numpy(yt))
            scan_sampler=None
            if a.sampling=='global_random':
                loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=True)
            else:
                indices=[]
                for row in by['train']:
                    with np.load(root/row['file']) as scan:
                        indices.append(scan['frame_index'].copy())
                scan_sampler=ScanBatchSampler(indices,a.batch_size,
                    mode='random' if a.sampling=='scan_random' else 'sequential',seed=a.seed)
                loader=DataLoader(dataset,batch_sampler=scan_sampler)
            record['batches_per_epoch']=len(loader)
            record['samples_per_epoch']=len(dataset)
            write_json(out/'run.json',record)
            best=float('inf') if a.selection=='mse' else -float('inf'); log=[]
            for epoch in range(a.epochs):
                if scan_sampler is not None: scan_sampler.set_epoch(epoch)
                model.train(); total=0; mse_total=0; aux_total=0; delta_total=0
                for x,y in loader:
                    x,y=x.to(a.device),y.to(a.device); optimizer.zero_grad(set_to_none=True)
                    loss,batch_mse,aux=objective(model(x),y,a.loss,a.aux_weight,a.delta_weight); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                    total+=loss.item()*len(x)
                    mse_total+=batch_mse.item()*len(x); aux_total+=aux.item()*len(x)
                model.eval(); val_sum=0; val_predictions=[]
                with torch.no_grad():
                    for i in range(0,len(xv),a.batch_size):
                        prediction=model(torch.from_numpy(xv[i:i+a.batch_size]).to(a.device)).cpu().numpy()
                        val_predictions.append(prediction)
                        val_sum+=float(((prediction-yv[i:i+a.batch_size])**2).sum())
                val=val_sum/yv.size
                # 按完整扫描计算时间相关，再扫描→被试→总体 Fisher 平均；不拼接扫描。
                vp=np.concatenate(val_predictions); cursor=0; subject_r={}; subject_valid={}
                for row in by['val']:
                    n=row['windows'] if 'windows' in row else len(load_scan(row)[0])
                    correlations=corr_rows(yv[cursor:cursor+n].transpose(1,2,0),vp[cursor:cursor+n].transpose(1,2,0))
                    subject_r.setdefault(row['subject'],[]).append(fisher_mean(correlations))
                    subject_valid.setdefault(row['subject'],[]).append(float(np.isfinite(correlations).mean()))
                    cursor+=n
                val_r=fisher_mean([fisher_mean(values) for values in subject_r.values()])
                val_valid=float(np.mean([np.mean(values) for values in subject_valid.values()]))
                log.append(dict(epoch=epoch,train_mse=mse_total/len(xt),train_loss=total/len(xt),train_aux=aux_total/len(xt),val_mse=val,val_temporal_r=val_r,val_temporal_valid=val_valid))
                if model.hrf_prior!='none':
                    mu,sigma=model.hrf_parameters()
                    for h in range(4):
                        log[-1][f'hrf_mu_{h}']=float(mu[h].detach())
                        log[-1][f'hrf_sigma_{h}']=float(sigma[h].detach())
                print(log[-1],flush=True)
                pd.DataFrame(log).to_csv(out/'training.csv',index=False)
                score=val if a.selection=='mse' else val_r
                improved=score<best if a.selection=='mse' else np.isfinite(score) and score>best
                if improved:
                    best=score; torch.save(model.state_dict(),out/'best.pt')
                    record.update(best_epoch=epoch+1,best_selection_score=float(score),parameter_count=sum(p.numel() for p in model.parameters()))
                    write_json(out/'run.json',record)
    if not a.eval_only:
        record.update(status='completed',finished_at=now())
        write_json(out/'run.json',record)
    # 评测阶段：神经网络加载验证所选主指标最佳的检查点，不在测试集重新训练。
    if a.model=='ridge': reg=joblib.load(out/'ridge.joblib')
    if a.model in ('mlp','tcn','tcn_attention'):
        model=Predictor(config['rois'],config['frames'],len(bands),kind=a.model,tr=config.get('tr',2.1),label_trs=config.get('label_trs',4),hrf_prior=a.hrf_prior,hrf_strength=a.hrf_strength).to(a.device)
        model.load_state_dict(torch.load(out/'best.pt',map_location=a.device,weights_only=True)); model.eval()
    evaluation=new_evaluation(run,a.eval_split)
    checkpoint=out/({'ridge':'ridge.joblib','mlp':'best.pt','tcn':'best.pt','tcn_attention':'best.pt'}.get(a.model,'state.npz'))
    eval_record=dict(split=a.eval_split,model=a.model,status='running',started_at=now(),
                     data_identity=identity,checkpoint_sha256=sha256(checkpoint),
                     checkpoint=str(checkpoint.resolve()),provenance=save_provenance(evaluation),
                     arguments=vars(a))
    write_json(evaluation/'evaluation.json',eval_record)
    rows=[]
    # 按完整扫描预测，避免跨扫描拼接计算时间相关。
    for r in by[a.eval_split]:
        # 仅应用训练集保存的逐 ROI 标准化参数。
        x,y=load_scan(r); xn=((x-mean)/std).astype('float32')
        if a.model=='template': pred=np.broadcast_to(template,y.shape).copy()
        elif a.model=='ridge': pred=np.clip(reg.predict(xn.reshape(len(xn),-1)),0,1).reshape(y.shape)
        else:
            with torch.no_grad():
                pred=np.concatenate([model(torch.from_numpy(xn[i:i+a.batch_size]).to(a.device)).cpu().numpy() for i in range(0,len(xn),a.batch_size)])
        if a.model=='tcn_attention':
            # 固定扫描保存逐窗口、逐头内容注意力及实际权重，分批避免显存占用。
            if r['scan']=='sub07-scan01':
                actual=[]; content=[]
                prior_mode=model.hrf_prior
                with torch.no_grad():
                    for i in range(0,len(xn),a.batch_size):
                        bx=torch.from_numpy(xn[i:i+a.batch_size]).to(a.device)
                        model(bx); actual.append(model.last_attention.cpu().numpy())
                        model.hrf_prior='none'
                        model(bx); content.append(model.last_attention.cpu().numpy())
                        model.hrf_prior=prior_mode
                params=model.hrf_parameters()
                extra={} if params is None else dict(mu=params[0].detach().cpu().numpy(),sigma=params[1].detach().cpu().numpy())
                np.savez_compressed(evaluation/'attention_sub07-scan01.npz',actual=np.concatenate(actual),content=np.concatenate(content),**extra)
        np.savez_compressed(evaluation/f'{a.eval_split}_{r["scan"]}_predictions.npz',prediction=pred,target=y)
        # 六频段各自计算指标；保留扫描和被试身份供分层汇总。
        for f,band in enumerate(bands):
            rows.append(dict(scan=r['scan'],subject=r['subject'],band=band,**metrics(y[:,f],pred[:,f],template[f])))
    # 三张 CSV 分别用于逐扫描排查、被试层面比较和总体报告。
    scan,subject,summary=aggregate(rows)
    for name,df in [('scans',scan),('subjects',subject),('summary',summary)]:
        df.to_csv(evaluation/f'{a.eval_split}_{name}.csv',index=False)
    eval_record.update(status='completed',finished_at=now())
    write_json(evaluation/'evaluation.json',eval_record)
    print(summary.to_string(index=False),flush=True)
    print(f'Evaluation: {evaluation}',flush=True)
    print(f'Ledger: {generate(run.parent)}',flush=True)


if __name__=='__main__': main()
