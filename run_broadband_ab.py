"""Frozen broadband A/B x four losses; validate all before test."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')


if __name__=='__main__':
    logs=ROOT/'runs'/'batch_logs';logs.mkdir(parents=True,exist_ok=True)
    for test in (False,True):
        for model in ('tcn','tcn_attention'):
            for loss in ('mse','temporal','delta','combined'):
                name=f'bb_{model}_{loss}_s2026'
                args=[sys.executable,'-u','inference_plv.py' if test else 'main_plv.py',
                      '--data','artifacts/data_common_10bold_4tr_6bands','--output',f'runs/{name}',
                      '--model',model,'--bands','broadband','--selection','temporal_r',
                      '--sampling','scan_sequential','--loss',loss,'--aux-weight','1.0' if loss=='delta' else '.01',
                      '--delta-weight','1.0','--epochs','50','--seed','2026','--batch-size','64',
                      '--lr','.0003','--device','cuda','--notes','Broadband restart: A/B x 4 losses; val subject-balanced temporal-r checkpoint']
                if test:args+=['--eval-split','test']
                print(f'Start {name} test={test}',flush=True)
                with (logs/f'{name}_{"test" if test else "train"}.log').open('w') as stream:
                    subprocess.run(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                print(f'Completed {name} test={test}',flush=True)
