"""Three seeds x no/fixed/learnable Gaussian time prior; freeze before test."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')

if __name__=='__main__':
    for test in (False,True):
        for seed in (2026,2027,2028):
            for prior in ('none','fixed','learnable'):
                name=f'hrf_{prior}_s{seed}'
                args=[sys.executable,'-u','inference_plv.py' if test else 'main_plv.py',
                  '--data','artifacts/data_common_10bold_4tr_6bands','--output',f'runs/{name}',
                  '--model','tcn_attention','--hrf-prior',prior,'--hrf-strength','.1',
                  '--bands','broadband','--selection','temporal_r','--sampling','scan_sequential',
                  '--loss','combined','--aux-weight','.01','--delta-weight','1',
                  '--epochs','50','--seed',str(seed),'--batch-size','64','--lr','.0003','--device','cuda',
                  '--notes','HRF Gaussian prior: mu init 4/5/6/7 s, sigma 3.5 s, strength 0.1; matched 3-seed exploratory comparison']
                if test:args+=['--eval-split','test']
                log=ROOT/'runs'/'batch_logs'/f'{name}_{"test" if test else "train"}.log'
                print(f'Start {name} test={test}',flush=True)
                with log.open('w') as stream:subprocess.run(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                print(f'Completed {name} test={test}',flush=True)
