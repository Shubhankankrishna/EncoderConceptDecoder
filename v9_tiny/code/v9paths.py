"""Paths shared by V9 code. The frozen V8 testbed is imported from v9_tiny/frozen/code."""
import hashlib, json, os, sys
from pathlib import Path

V9 = Path(__file__).resolve().parents[1]
PROJECT = V9.parent
FROZEN_CODE = V9 / 'frozen' / 'code'
DATA = V9 / 'frozen' / 'data'
RUNS = Path(os.environ['V9_RUNS']) if os.environ.get('V9_RUNS') else V9 / 'runs'  # override only for smoke tests
SMOKE = bool(os.environ.get('V9_SMOKE'))
SPEC = V9 / 'spec'
TESTS = V9 / 'tests'
CODE = V9 / 'code'
V8 = PROJECT / 'v8_tiny_12h'
if str(FROZEN_CODE) not in sys.path:
    sys.path.insert(0, str(FROZEN_CODE))
if str(CODE) not in sys.path:
    sys.path.insert(1, str(CODE))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def files_hash(paths):
    """SHA-256 over (name, bytes) of an explicit file list; used by per-block locks."""
    h = hashlib.sha256()
    for p in sorted(Path(p) for p in paths):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
