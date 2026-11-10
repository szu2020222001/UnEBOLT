"""Scan-boundary-aware batches for controlled sample-order experiments."""
import numpy as np
from torch.utils.data import Sampler


class ScanBatchSampler(Sampler):
    """每轮每样本恰好一次；不跨扫描或时间索引缺口，不循环补齐。

    sequential 保留块内时间顺序，random 在同一连续区段内打散窗口。
    两种模式使用相同区段、batch 大小分布和更新次数。每轮随机打乱块顺序。
    使用独立 RNG，采样不消耗模型初始化或 dropout 的随机数流。
    """
    def __init__(self, frame_indices, batch_size=64, mode='sequential', seed=2026):
        if mode not in ('sequential', 'random') or batch_size < 1:
            raise ValueError('Invalid sampling mode or batch size')
        self.batch_size, self.mode, self.seed, self.epoch = batch_size, mode, seed, 0
        self.segments = []
        offset = 0
        for frames in frame_indices:
            frames = np.asarray(frames)
            if frames.ndim != 1 or not len(frames) or np.any(np.diff(frames) <= 0):
                raise ValueError('Each scan needs strictly increasing nonempty frame indices')
            for segment in np.split(np.arange(offset, offset+len(frames)), np.flatnonzero(np.diff(frames) != 1)+1):
                self.segments.append(segment)
            offset += len(frames)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return sum((len(s)+self.batch_size-1)//self.batch_size for s in self.segments)

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        batches = []
        for segment in self.segments:
            # 两组都生成 permutation，保证后续块顺序的 RNG 状态一致。
            shuffled = rng.permutation(segment)
            indices = shuffled if self.mode == 'random' else segment
            batches.extend(indices[i:i+self.batch_size].tolist() for i in range(0,len(indices),self.batch_size))
        return iter([batches[i] for i in rng.permutation(len(batches))])
