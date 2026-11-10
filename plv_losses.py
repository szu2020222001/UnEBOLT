"""Sequence losses for [consecutive windows, frequency bands, edges]."""
import torch


def objective(pred, target, kind='mse', weight=.01):
    mse=(pred-target).square().mean()
    auxiliary=pred.sum()*0
    if kind=='temporal':
        if len(pred)>1:
            # 以 batch 为时间轴；逐频段/边中心化。稳定分母允许常量预测获得梯度。
            a=pred-pred.mean(0,keepdim=True)
            b=target-target.mean(0,keepdim=True)
            va=a.square().mean(0); vb=b.square().mean(0)
            valid=vb>1e-10
            corr=(a*b).mean(0)/torch.sqrt((va+1e-8)*(vb+1e-8))
            if valid.any(): auxiliary=1-corr[valid].mean()
    elif kind=='delta':
        if len(pred)>1:
            auxiliary=((pred[1:]-pred[:-1])-(target[1:]-target[:-1])).square().mean()
    elif kind!='mse':
        raise ValueError(f'Unknown loss: {kind}')
    return mse+weight*auxiliary,mse,auxiliary
