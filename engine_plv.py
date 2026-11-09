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
from models.plv_model import Predictor
from plv_metrics import metrics, fisher_mean


def read_scan(root, row):
    with np.load(root/row['file']) as d:
        return d['x'],d['y']


def aggregate(rows):
    frame = pd.DataFrame(rows)
    keys = [c for c in frame if c not in ('scan','subject','band')]
    subjects=[]
    for (u,f), group in frame.groupby(['subject','band']):
        row=dict(subject=u,band=f)
        for k in keys:
            row[k] = fisher_mean(group[k]) if k in ('spatial_r','temporal_r') else float(group[k].mean())
        subjects.append(row)
    summary=[]
    sub=pd.DataFrame(subjects)
    rng=np.random.default_rng(2026)
    for f,group in sub.groupby('band'):
        for k in keys:
            v=group[k].to_numpy(); v=v[np.isfinite(v)]
            fn=fisher_mean if k in ('spatial_r','temporal_r') else np.mean
            point=float(fn(v)) if len(v) else np.nan
            boot=[fn(rng.choice(v,len(v),replace=True)) for _ in range(1000)] if len(v)>1 else []
            ci=np.quantile(boot,[.025,.975]) if boot else [np.nan,np.nan]
            summary.append(dict(band=f,metric=k,mean=point,subject_sd=float(np.std(v,ddof=1)) if len(v)>1 else np.nan,
                                ci_low=ci[0],ci_high=ci[1],valid_subjects=len(v)))
    return frame,sub,pd.DataFrame(summary)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--model',choices=['template','ridge','mlp','tcn'],default='tcn')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--device',default='cuda')
    p.add_argument('--eval-only',action='store_true')
    p.add_argument('--eval-split',choices=['val','test'],default='val')
    a=p.parse_args(argv)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False
    root,out=Path(a.data),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text())
    config=manifest['config']; bands=list(config['bands'])
    by={s:[r for r in manifest['scans'] if r['split']==s] for s in ('train','val','test')}
    if not all(by.values()): raise ValueError('A complete train/val/test manifest is required')
    def concat(part):
        pairs=[read_scan(root,r) for r in by[part]]
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
    else:
        xt,yt=concat('train'); xv,yv=concat('val')
        # One contribution per physical BOLD frame, not duplicated overlapping windows.
        frames=[]; scan_templates=[]
        for r in by['train']:
            with np.load(root/r['file']) as d:
                idx=(d['frame_index'][:,None]+np.arange(config['frames'])).ravel()
                _,first=np.unique(idx,return_index=True)
                frames.append(d['x'].reshape(-1,config['rois'])[first])
                scan_templates.append((r['subject'],d['y'].mean(0)))
        allframes=np.concatenate(frames)
        mean=allframes.mean(0); std=allframes.std(0).clip(1e-6)
        us=sorted(set(u for u,_ in scan_templates))
        template=np.mean([np.mean([v for u,v in scan_templates if u==s],axis=0) for s in us],axis=0)
        np.savez(statepath,mean=mean,std=std,template=template)
        (out/'run.json').write_text(json.dumps(dict(vars(a),data_signature=manifest['signature'],
                    torch=torch.__version__,cuda=torch.version.cuda),indent=2))
        xt=((xt-mean)/std).astype('float32'); xv=((xv-mean)/std).astype('float32')
        if a.model=='ridge':
            best=float('inf'); scores=[]
            for alpha in (.1,1,10,100,1000):
                candidate=Ridge(alpha=alpha).fit(xt.reshape(len(xt),-1),yt.reshape(len(yt),-1))
                pred=np.clip(candidate.predict(xv.reshape(len(xv),-1)),0,1).reshape(yv.shape)
                score=float(np.mean((pred-yv)**2)); scores.append(dict(alpha=alpha,val_mse=score))
                if score<best: best,reg=score,candidate
            joblib.dump(reg,out/'ridge.joblib'); pd.DataFrame(scores).to_csv(out/'ridge_search.csv',index=False)
        elif a.model in ('mlp','tcn'):
            model=Predictor(config['rois'],config['frames'],len(bands),kind=a.model).to(a.device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
            loader=DataLoader(TensorDataset(torch.from_numpy(xt),torch.from_numpy(yt)),batch_size=a.batch_size,shuffle=True)
            best=float('inf'); log=[]
            for epoch in range(a.epochs):
                model.train(); total=0
                for x,y in loader:
                    x,y=x.to(a.device),y.to(a.device); optimizer.zero_grad(set_to_none=True)
                    loss=(model(x)-y).square().mean(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                    total+=loss.item()*len(x)
                model.eval(); val_sum=0
                with torch.no_grad():
                    for i in range(0,len(xv),a.batch_size):
                        prediction=model(torch.from_numpy(xv[i:i+a.batch_size]).to(a.device)).cpu().numpy()
                        val_sum+=float(((prediction-yv[i:i+a.batch_size])**2).sum())
                val=val_sum/yv.size
                log.append(dict(epoch=epoch,train_mse=total/len(xt),val_mse=val))
                print(log[-1],flush=True)
                pd.DataFrame(log).to_csv(out/'training.csv',index=False)
                if val<best:
                    best=val; torch.save(model.state_dict(),out/'best.pt')
    if a.model=='ridge': reg=joblib.load(out/'ridge.joblib')
    if a.model in ('mlp','tcn'):
        model=Predictor(config['rois'],config['frames'],len(bands),kind=a.model).to(a.device)
        model.load_state_dict(torch.load(out/'best.pt',map_location=a.device,weights_only=True)); model.eval()
    rows=[]
    for r in by[a.eval_split]:
        x,y=read_scan(root,r); xn=((x-mean)/std).astype('float32')
        if a.model=='template': pred=np.broadcast_to(template,y.shape).copy()
        elif a.model=='ridge': pred=np.clip(reg.predict(xn.reshape(len(xn),-1)),0,1).reshape(y.shape)
        else:
            with torch.no_grad():
                pred=np.concatenate([model(torch.from_numpy(xn[i:i+a.batch_size]).to(a.device)).cpu().numpy() for i in range(0,len(xn),a.batch_size)])
        np.savez_compressed(out/f'{a.eval_split}_{r["scan"]}_predictions.npz',prediction=pred,target=y)
        for f,band in enumerate(bands):
            rows.append(dict(scan=r['scan'],subject=r['subject'],band=band,**metrics(y[:,f],pred[:,f],template[f])))
    scan,subject,summary=aggregate(rows)
    for name,df in [('scans',scan),('subjects',subject),('summary',summary)]:
        df.to_csv(out/f'{a.eval_split}_{name}.csv',index=False)
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__': main()
