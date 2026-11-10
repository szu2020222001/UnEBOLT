"""Matched scan-random vs scan-sequential MSE training; freeze before test."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')


def execute(model, sampling, evaluate=False):
    run=f'runs/20261110_{model}_{sampling}_s2026_v1'
    log=ROOT/'runs'/'batch_logs'/f'{model}_{sampling}_{"test" if evaluate else "train"}.log'
    args=[sys.executable,'-u','inference_plv.py' if evaluate else 'main_plv.py',
          '--data','artifacts/data_common_10bold_4tr_6bands','--output',run,
          '--model',model,'--sampling',sampling,'--epochs','50','--batch-size','64',
          '--lr','0.0003','--seed','2026','--device','cuda',
          '--notes','Matched scan-boundary sampling ablation: each sample once; 162 updates/epoch; MSE only']
    if evaluate: args+=['--eval-split','test']
    print(f'Start {model} {sampling} eval={evaluate}',flush=True)
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as stream:
        subprocess.run(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print(f'Completed {model} {sampling} eval={evaluate}',flush=True)


if __name__=='__main__':
    for evaluate in (False,True):
        for model in ('mlp','tcn'):
            for sampling in ('scan_random','scan_sequential'):
                execute(model,sampling,evaluate)
