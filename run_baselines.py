"""Run the frozen first-round baselines, then evaluate all four on test."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
DATA='artifacts/data_common_10bold_4tr_6bands'
PREFIX='20261110'
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
logs=ROOT/'runs'/'batch_logs'
logs.mkdir(parents=True,exist_ok=True)

def execute(args, name):
    print(f'Starting {name}',flush=True)
    with (logs/f'{name}.log').open('w') as stream:
        subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=env,
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    print(f'Completed {name}',flush=True)

if __name__=='__main__':
    # Freeze all four validation-selected models before looking at test results.
    for model in ('template','ridge','mlp','tcn'):
        run=f'runs/{PREFIX}_{model}_s2026_v1'
        execute(['main_plv.py','--data',DATA,'--output',run,'--model',model,
                 '--device','cuda' if model in ('mlp','tcn') else 'cpu',
                 '--epochs','50','--batch-size','64','--lr','0.0003','--seed','2026',
                 '--notes','First frozen six-band cross-subject baseline; common reference; 10 BOLD frames to 4TR EEG PLV'],f'{model}_train')
    for model in ('template','ridge','mlp','tcn'):
        execute(['inference_plv.py','--data',DATA,'--output',f'runs/{PREFIX}_{model}_s2026_v1',
                 '--model',model,'--eval-split','test','--device',
                 'cuda' if model in ('mlp','tcn') else 'cpu'],f'{model}_test')
