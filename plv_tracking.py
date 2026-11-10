"""Local experiment provenance and append-only evaluation directories."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def provenance():
    """记录实际执行的代码版本；未提交修改另存 patch 供追溯。"""
    root = Path(__file__).resolve().parent
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL).decode('utf-8', errors='replace').strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return dict(commit=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
                status=git('status', '--porcelain'), patch=git('diff', 'HEAD', '--binary'),
                python=sys.version, platform=platform.platform())


def save_provenance(directory):
    info = provenance()
    patch = info.pop('patch')
    if patch:
        (Path(directory) / 'source.patch').write_text(patch + '\n', encoding='utf-8')
    # 保存当前执行的 PLV 源文件，包含尚未被 Git 跟踪的新模块。
    root = Path(__file__).resolve().parent
    paths = ['engine_plv.py', 'run_broadband_ab.py', 'plv_losses.py', 'run_loss_ablation.py', 'plv_sampling.py', 'run_sampling_ablation.py', 'plv_metrics.py', 'plv_tracking.py', 'experiment_ledger.py',
             'main_plv.py', 'inference_plv.py', 'models/plv_model.py',
             'dataset_maker/plv_dataset.py', 'build_plv_data.py']
    hashes = {}
    for name in paths:
        src = root / name
        if src.exists():
            dest = Path(directory) / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
            hashes[name] = sha256(src)
    info['source_sha256'] = hashes
    return info


def training_dir(run):
    """兼容旧版直接在运行根目录保存 state.npz 的实验，只读使用旧训练产物。"""
    run = Path(run)
    return run / 'training' if (run / 'training' / 'state.npz').exists() else run


def new_evaluation(run, split):
    # 时间戳 + UUID，重复评测和并发评测都不会复用旧目录。
    token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:8]
    directory = Path(run) / 'evaluations' / f'{token}_{split}'
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def data_identity(root, manifest):
    """配置签名以外，再固定完整清单（含 split）与文件的实际内容。"""
    root = Path(root)
    return dict(manifest_sha256=sha256(root / 'manifest.json'),
                files={r['file']: sha256(root / r['file']) for r in manifest['scans']})
