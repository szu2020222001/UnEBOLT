"""Small PyTorch baselines and explicitly defined scan-level PLV metrics."""
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
    def __init__(self, rois=64, frames=10, bands=6, edges=325, kind='tcn'):
        super().__init__()
        self.kind = kind
        self.project = nn.Sequential(nn.Linear(rois, 64), nn.GELU())
        self.temporal = nn.Sequential(*[ResidualBlock(64, d) for d in (1, 2, 4)])
        self.hidden = nn.Sequential(nn.Linear(frames * (64 if kind == 'tcn' else rois), 128),
                                    nn.GELU(), nn.Dropout(.1))
        self.heads = nn.ModuleList([nn.Linear(128, edges) for _ in range(bands)])

    def forward(self, x):
        if self.kind == 'tcn':
            x = self.temporal(self.project(x).transpose(1, 2)).transpose(1, 2)
        x = self.hidden(x.flatten(1))
        return torch.stack([h(x).sigmoid() for h in self.heads], dim=1)

