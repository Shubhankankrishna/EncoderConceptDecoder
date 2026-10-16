"""Byte-hash data, completed work, and pending lock inputs without loading sealed data."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

V9 = Path(__file__).resolve().parents[1]
MON = V9 / 'monitor'


def read(p):
    return json.loads(p.read_text())


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def files_hash(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


fd = V9 / 'frozen/data'
m = read(fd / 'manifest.json')
targets = {fd / (n + suffix): e['sha256'] for group, suffix in (('text', '.npy'), ('symbolic', '.npz')) for n, e in m[group].items()}
targets[fd / 'tokenizer.json'] = m['tokenizer_sha256']
targets[V9.parent / 'v5_tiny/data/ts_train.npy'] = m['tiny_stories']['sha256']
for part, row in m['tiny_stories']['partitions'].items():
    targets[fd / f'ts_{part}_ids.npy'] = row['ids_sha256']
    targets[fd / f'ts_{part}_hashes.txt'] = row['content_hashes_sha256']
m2 = read(V9 / 'data_v9/manifest.json')
for group in ('colourcount', 'calibration'):
    for n, row in m2[group].items():
        targets[V9 / f'data_v9/{group}_{n}.npz'] = row['sha256']
targets[V9 / 'data_v9/colourcount_eval_hashes.txt'] = m2['eval_hashes_sha256']
checks = {str(p.relative_to(V9.parent)): digest(p) == h for p, h in targets.items()}
completed = {}
for mapping in (read(MON / 'q3_completion_notification.json')['artifacts_sha256'],
                read(MON / 'q4_completed_artifacts.json'), read(MON / 'q5_completed_artifacts.json')):
    completed.update({p: digest(V9.parent / p) == h for p, h in mapping.items()})
locks = {}
frozen_hash = files_hash(list((V9 / 'frozen/code').glob('*.py')))
scratch = Path(sys.argv[1]) if len(sys.argv) > 1 else None
for block in ('Q5', 'Q6', 'QF'):
    spec = scratch if scratch and block in ('Q5', 'Q6') else V9 / 'spec'
    lock = read(spec / f'LOCK_{block}.json')
    ident = {k: lock[k] for k in ('block', 'frozen_implementation_sha256', 'files_sha256', 'data_manifest_sha256', 'exclusion_list_sha256')}
    locks[block] = dict(code=files_hash([V9 / 'code' / n for n in lock['files']]) == lock['files_sha256'],
                        frozen=frozen_hash == lock['frozen_implementation_sha256'],
                        manifest=digest(fd / 'manifest.json') == lock['data_manifest_sha256'],
                        exclusions=digest(fd / 'evaluation_symbolic_hashes.txt') == lock['exclusion_list_sha256'],
                        identity=hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest() == lock['lock_sha256'])
result = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), actual_data_files=len(checks),
              data=checks, all_data_match=all(checks.values()), completed_files=len(completed),
              completed=completed, all_completed_match=all(completed.values()), pending_locks=locks,
              lock_source='isolated validation' if scratch else 'official',
              source_checkpoint_match=digest(V9 / 'runs/S2_K8_dense_s911/checkpoint.pt') == '3a42a0a6399d8382658f8dac23fa184d75cdd4534704d91db8b40064fbef22f5',
              validated_python_launcher_match=digest(MON / 'python_startup.py') == '7e030f4bdb98f916f8074a80d166a8afa21da9fe7d26a76ab23942a3c868a09c')
(MON / 'q5_memory_inputs.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k not in ('data', 'completed')}), flush=True)
assert result['all_data_match'] and result['all_completed_match'] and result['source_checkpoint_match'] and result['validated_python_launcher_match']
if scratch:
    assert all(all(c.values()) for c in locks.values()), locks
