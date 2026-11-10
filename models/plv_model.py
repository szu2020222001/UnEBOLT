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
    def __init__(self, rois=64, frames=10, bands=6, edges=325, kind='tcn', tr=2.1, label_trs=4):
        super().__init__()
        self.kind = kind
        if kind == 'tcn_attention':
            # EEG 标签中心为零点，BOLD 各帧位置用真实秒数编码。
            times=(torch.arange(frames).float()-label_trs/2)*tr
            scale=torch.exp(torch.arange(0,64,2).float()*(-math.log(10000.)/64))
            position=torch.zeros(frames,64)
            position[:,0::2]=torch.sin(times[:,None]*scale)
            position[:,1::2]=torch.cos(times[:,None]*scale)
            self.register_buffer('time_position',position)
            self.query=nn.Parameter(torch.randn(1,1,64)*.02)
            self.attention=nn.MultiheadAttention(64,4,dropout=0.,batch_first=True)
            self.attention_norm=nn.LayerNorm(64)
        self.project = nn.Sequential(nn.Linear(rois, 64), nn.GELU())
        self.temporal = nn.Sequential(*[ResidualBlock(64, d) for d in (1, 2, 4)])
        self.hidden = nn.Sequential(nn.Linear((64 if kind == 'tcn_attention' else frames * (64 if kind == 'tcn' else rois)), 128),
                                    nn.GELU(), nn.Dropout(.1))
        self.heads = nn.ModuleList([nn.Linear(128, edges) for _ in range(bands)])

    def forward(self, x):
        if self.kind == 'tcn_attention':
            x=self.project(x)+self.time_position
            x=self.temporal(x.transpose(1,2)).transpose(1,2)
            x=self.attention_norm(x)
            x,self.last_attention=self.attention(self.query.expand(len(x),-1,-1),x,x)
        elif self.kind == 'tcn':
            x = self.temporal(self.project(x).transpose(1, 2)).transpose(1, 2)
        x = self.hidden(x.flatten(1))
        return torch.stack([h(x).sigmoid() for h in self.heads], dim=1)
