import numpy as np
import pytest
from plv_sampling import ScanBatchSampler


def test_coverage_boundaries_gaps_and_matched_batches():
    frames=[np.r_[np.arange(75),np.arange(80,99)],np.arange(103)]
    samplers=[ScanBatchSampler(frames,16,mode=m) for m in ('random','sequential')]
    modes=[]
    for sampler in samplers:
        batches=list(sampler)
        assert len(batches)==len(sampler)
        assert sorted(i for batch in batches for i in batch)==list(range(197))
        for batch in batches:
            assert all(i<75 for i in batch) or all(75<=i<94 for i in batch) or all(i>=94 for i in batch)
            if sampler.mode=='sequential': assert all(b-a==1 for a,b in zip(batch,batch[1:]))
        assert batches==list(sampler)
        modes.append(batches)
        sampler.set_epoch(1)
        assert batches!=list(sampler)
    assert sorted(map(len,modes[0]))==sorted(map(len,modes[1]))
    assert modes[0]!=modes[1]


def test_actual_layout_has_162_batches():
    sampler=ScanBatchSampler([np.arange(562)]*18,64)
    assert len(sampler)==162
    assert sum(map(len,sampler))==10116


def test_invalid_indices():
    with pytest.raises(ValueError): ScanBatchSampler([[2,1]])
