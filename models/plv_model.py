"""Small PyTorch baselines and explicitly defined scan-level PLV metrics."""
import math
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=dilation,
                                           dilation=dilation), nn.GELU(), nn.Dropout(.1))

    def forward(self, x):
        return x + self.net(x)


class Predictor(nn.Module):
    def __init__(self, rois=64, frames=10, bands=6, edges=325, kind='tcn', tr=2.1, label_trs=4, hrf_prior="none", hrf_strength=.1):
        super().__init__()
        self.kind = kind
        if hrf_prior not in ('none','fixed','learnable'): raise ValueError('Unknown HRF prior')
        self.hrf_prior=hrf_prior
        self.hrf_strength=hrf_strength
        if kind == 'tcn_attention':
            # EEG 标签中心为零点，BOLD 各帧位置用真实秒数编码。
            times=(torch.arange(frames).float()-label_trs/2)*tr
            scale=torch.exp(torch.arange(0,64,2).float()*(-math.log(10000.)/64))
            position=torch.zeros(frames,64)
            position[:,0::2]=torch.sin(times[:,None]*scale)
            position[:,1::2]=torch.cos(times[:,None]*scale)
            self.register_buffer('time_position',position)
            if hrf_prior!='none':
                self.register_buffer('relative_seconds',times)
                mu=torch.tensor([4.,5.,6.,7.])
                raw_mu=torch.logit(mu/12.)
                raw_sigma=torch.full((4,),math.log(math.expm1(2.5)))
                if hrf_prior=='learnable':
                    self.hrf_raw_mu=nn.Parameter(raw_mu)
                    self.hrf_raw_sigma=nn.Parameter(raw_sigma)
                else:
                    self.register_buffer('hrf_raw_mu',raw_mu)
                    self.register_buffer('hrf_raw_sigma',raw_sigma)
            self.query=nn.Parameter(torch.randn(1,1,64)*.02)
            self.attention=nn.MultiheadAttention(64,4,dropout=0.,batch_first=True)
            self.attention_norm=nn.LayerNorm(64)
        self.project = nn.Sequential(nn.Linear(rois, 64), nn.GELU())
        self.temporal = nn.Sequential(*[ResidualBlock(64, d) for d in (1, 2, 4)])
        self.hidden = nn.Sequential(nn.Linear((64 if kind == 'tcn_attention' else frames * (64 if kind == 'tcn' else rois)), 128),
                                    nn.GELU(), nn.Dropout(.1))
        self.heads = nn.ModuleList([nn.Linear(128, edges) for _ in range(bands)])

    def hrf_parameters(self):
        if self.hrf_prior=='none': return None
        return 12*torch.sigmoid(self.hrf_raw_mu),1+torch.nn.functional.softplus(self.hrf_raw_sigma)

    def forward(self, x):
        if self.kind == 'tcn_attention':
            x=self.project(x)+self.time_position
            x=self.temporal(x.transpose(1,2)).transpose(1,2)
            x=self.attention_norm(x)
            query=self.query.expand(len(x),-1,-1)
            mask=None
            if self.hrf_prior!='none':
                mu,sigma=self.hrf_parameters()
                bias=-self.hrf_strength*(self.relative_seconds[None,:]-mu[:,None]).square()/(2*sigma[:,None].square())
                mask=bias[None,:,None,:].expand(len(x),-1,-1,-1).reshape(len(x)*4,1,x.shape[1])
            x,weights=self.attention(query,x,x,attn_mask=mask,average_attn_weights=False)
            self.last_attention=weights.detach()
        elif self.kind == 'tcn':
            x = self.temporal(self.project(x).transpose(1, 2)).transpose(1, 2)
        x = self.hidden(x.flatten(1))
        return torch.stack([h(x).sigmoid() for h in self.heads], dim=1)
