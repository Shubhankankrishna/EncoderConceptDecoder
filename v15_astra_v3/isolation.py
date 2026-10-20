"""Read-only protection of the original experiment and guarded v3 writes."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / 'v15_astra_m2'
RUN = LEGACY / 'runs/20261018_48h'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def output_path(path):
    """All v3 CLI writes are confined to the new directory, including symlinks."""
    p = Path(path).resolve()
    if not p.is_relative_to(ROOT.resolve()):
        raise ValueError(f'Output must be inside {ROOT}; got {p}')
    return p


def atomic_json(path, obj):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def active_legacy_processes():
    import psutil
    found=[]
    for proc in psutil.process_iter(['pid','cmdline']):
        cmd = proc.info['cmdline'] or []
        # Match executable arguments, not a shell command containing this text.
        if any(x in ('v15_astra_m2.train','v15_astra_m2.supervise') or
               x.endswith(('/v15_astra_m2/train.py','/v15_astra_m2/supervise.py')) for x in cmd):
            found.append(proc.info['pid'])
    return found


def require_legacy_idle():
    pids=active_legacy_processes()
    if pids:
        raise RuntimeError(f'Active original experiment {pids}: v3 training/codec/inference is blocked. CPU unit tests remain available.')


def exclusive_job(fn):
    """One v3 heavy job at a time, after the registered experiment is idle."""
    import functools
    @functools.wraps(fn)
    def wrapped(*args,**kwargs):
        import fcntl
        require_legacy_idle()  # no lock-file writes while legacy is active
        lock=output_path(ROOT/'reports/job.lock');lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Another v3 heavy job is active') from None
            try:return fn(*args,**kwargs)
            finally:fcntl.flock(handle,fcntl.LOCK_UN)
    return wrapped


def protected_snapshot():
    """Exclude mutable logs/checkpoints; include source, plan and immutable data."""
    plan=json.loads((RUN/'plan.json').read_text())
    sources={p.name:sha256(p) for p in LEGACY.glob('*.py')}
    if sources != plan['source_files']:
        raise RuntimeError('Original source no longer matches its registered plan')
    paths=[RUN/'plan.json', LEGACY/'artifacts/data/manifest.json',
           LEGACY/'artifacts/data/source_manifest.json',
           LEGACY/'artifacts/data/samples.npz', LEGACY/'artifacts/data/codes.npy',
           LEGACY/'artifacts/data/texts.json', LEGACY/'artifacts/data/episodes.jsonl']
    return dict(source_files=sources, immutable_files={str(p.relative_to(LEGACY)):sha256(p) for p in paths})


def verify_protected(snapshot):
    if protected_snapshot() != snapshot:
        raise RuntimeError('Protected original experiment files changed')
    return True
