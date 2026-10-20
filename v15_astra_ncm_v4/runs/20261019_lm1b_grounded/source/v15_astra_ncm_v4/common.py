"""Run identity, atomic persistence and single-owner resource protection."""
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent


def runtime_identity():
    return {'python':sys.version,'executable':sys.executable,
            'packages':{name:importlib.metadata.version(name) for name in
                        ('torch','numpy','transformers','omegaconf','networkx')}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_json(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, allow_nan=False) + '\n')


def source_hashes():
    files = list(ROOT.glob('*.py'))
    # Include every imported local/upstream source in the resume identity.
    for base in (WORKSPACE/'v15_astra_v3', WORKSPACE/'v15_astra_m2/references/ncm'):
        files += list(base.glob('*.py'))
        if base.name == 'ncm':
            for folder in ('models', 'tokenization', 'utils'):
                files += list((base/folder).glob('*.py'))
    return {str(p.relative_to(WORKSPACE)): sha(p) for p in sorted(set(files))}


@contextlib.contextmanager
def heavy_job():
    from v15_astra_v3.isolation import require_legacy_idle
    require_legacy_idle()
    lock = WORKSPACE/'v15_astra_v3/reports/job.lock'
    with lock.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another training/codec job owns the machine') from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def resource_issue():
    import psutil
    if psutil.virtual_memory().available < 1.5 * 1024**3:
        return 'low_memory'
    if psutil.disk_usage(ROOT).free < 3 * 1024**3:
        return 'low_disk'
    battery = psutil.sensors_battery()
    if battery and not battery.power_plugged and battery.percent < 20:
        return 'low_battery'
    return None


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state():
    import torch
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'mps': torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def restore_rng(state):
    import torch
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if state['mps'] is not None:
        torch.mps.set_rng_state(state['mps'])


def save_torch(path, obj):
    import torch
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
