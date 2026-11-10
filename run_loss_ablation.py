"""Frozen separate temporal/difference loss experiment; all validation before test."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')


if __name__=='__main__':
    for evaluate in (False,True):
        for model in ('mlp','tcn'):
            for loss,weight in (('temporal','.01'),('delta','1.0')):
                run=f'runs/20261110_{model}_scan_sequential_{loss}_s2026_v1'
                args=[sys.executable,'-u','inference_plv.py' if evaluate else 'main_plv.py',
                      '--data','artifacts/data_common_10bold_4tr_6bands','--output',run,
                      '--model',model,'--sampling','scan_sequential','--loss',loss,'--aux-weight',weight,
                      '--epochs','50','--batch-size','64','--lr','0.0003','--seed','2026','--device','cuda',
                      '--notes','Separate loss ablation; fixed weights; validation MSE checkpoint; same sequential sampler as control']
                if evaluate:args+=['--eval-split','test']
                log=ROOT/'runs'/'batch_logs'/f'{model}_{loss}_{"test" if evaluate else "train"}.log'
                print(f'Start {model} {loss} test={evaluate}',flush=True)
                with log.open('w') as stream:
                    subprocess.run(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                print(f'Completed {model} {loss} test={evaluate}',flush=True)
