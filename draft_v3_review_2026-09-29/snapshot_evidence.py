"""Read-only, low-memory evidence snapshot for the draft review."""
from pathlib import Path
import json, hashlib, datetime, zipfile, collections
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
inventory = json.loads((OUT / 'inventory.json').read_text())
payload = {'captured': datetime.datetime.now().astimezone().isoformat(), 'runs': {}, 'arrays': [], 'archives': []}
paths = ['run_m2_25M/data/MANIFEST.json']
paths += ['run_m2_25M/code/' + f for f in ['hrm_encoder.py', 'hrm_decoder.py', 'concept_core_v2.py', 'model.py', 'train.py', 'data.py', 'evaluate.py', 'build_data.py', 'tasks.py']]
paths += ['run_v2_19M/code/' + f for f in ['train.py', 'model.py', 'concept_core_d6.py', 'data.py', 'evaluate.py'] if (ROOT / 'run_v2_19M/code' / f).exists()]
for name, run in [('v1', 'run_m2_25M/runs/main'), ('v1_cont', 'run_m2_25M/runs/cont'), ('v2', 'run_v2_19M/runs/v2_48h')]:
    payload['runs'][name] = {}
    for file in ['log.jsonl', 'train_log.jsonl', 'config.json', 'eval_final.json']:
        p = ROOT / run / file
        if not p.exists(): continue
        paths.append(str(p.relative_to(ROOT)))
        data = p.read_text()
        parsed = [json.loads(line) for line in data.splitlines() if line.strip()] if file.endswith('jsonl') else json.loads(data)
        payload['runs'][name][file] = parsed
for rel in paths:
    data = (ROOT / rel).read_bytes()
    dest = OUT / 'source_snapshot' / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
for rec in inventory['files']:
    if '.venv/' in rec['path']: continue
    p = ROOT / rec['path']
    if p.suffix == '.npy':
        try:
            a = np.load(p, mmap_mode='r', allow_pickle=False)
            payload['arrays'].append({'path': rec['path'], 'shape': list(a.shape), 'dtype': str(a.dtype), 'nbytes': a.nbytes})
        except Exception as e:
            payload['arrays'].append({'path': rec['path'], 'error': str(e)})
    if p.suffix in ('.pt', '.pth'):
        try:
            with zipfile.ZipFile(p) as z:
                entries = z.infolist()
                payload['archives'].append({'path': rec['path'], 'entries': len(entries), 'uncompressed_bytes': sum(i.file_size for i in entries)})
        except Exception as e:
            payload['archives'].append({'path': rec['path'], 'error': str(e)})
manifest = json.loads((ROOT / paths[0]).read_text())
train = manifest['splits']['train']
rates = [1290, 1770]
slots = [rate * 45.5 * 3600 for rate in rates]
payload['data_arithmetic'] = {'rows': train['rows'], 'slots_per_epoch': train['row_tokens'], 'nonpad_per_epoch': train['nonpad_tokens'], 'pad_fraction_exact': 1 - train['nonpad_tokens']/train['row_tokens'], 'projected_slots': slots, 'projected_epochs': [n/train['row_tokens'] for n in slots], 'projected_nonpad_approx': [n*train['nonpad_tokens']/train['row_tokens'] for n in slots], 'projected_nonpad_per_compute_param_approx': [n*train['nonpad_tokens']/train['row_tokens']/18986871 for n in slots]}
payload['draft_snapshot_sha256'] = hashlib.sha256((OUT/'source_snapshot/draft_v3.md').read_bytes()).hexdigest()
payload['draft_current_sha256'] = hashlib.sha256((ROOT/'draft_v3.md').read_bytes()).hexdigest()
(OUT / 'evidence_snapshot.json').write_text(json.dumps(payload, indent=2))
for name, data in payload['runs'].items():
    print(name)
    for rec in data.get('log.jsonl', []):
        if rec['step'] in [1009, 1217, 1792, 2376, 2764, 2796, 4050] or rec is data['log.jsonl'][-1]:
            print({k: rec.get(k) for k in ['step', 'wall_h', 'time', 'val_lang_mean']}, 'TS', rec['val_tinystories']['nats'], 'WK', rec['val_wikitext']['nats'], 'probes', {k:v['exact'] for k,v in rec['probes'].items()})
    print('last_train', data.get('train_log.jsonl', [{}])[-1].get('step'))
print('data_arithmetic', payload['data_arithmetic'])
print('draft_unchanged', payload['draft_snapshot_sha256'] == payload['draft_current_sha256'])
print('array_count', len(payload['arrays']), 'checkpoint_archive_count', len(payload['archives']))
