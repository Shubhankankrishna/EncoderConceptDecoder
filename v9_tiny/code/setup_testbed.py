"""Q0: freeze the V8 12-hour testbed by SHA-256 and draw V9's fresh sealed manifests.

Run once. Refuses to overwrite an existing testbed. Writes only under v9_tiny/frozen and v9_tiny/spec.
"""
import hashlib, json, shutil, sys, time
from pathlib import Path
import numpy as np
import torch

V9 = Path(__file__).resolve().parents[1]
PROJECT = V9.parent
V8 = PROJECT / 'v8_tiny_12h'
FROZEN = V9 / 'frozen'
SPEC = V9 / 'spec'

# V8 suite table (data.py:162) in V8 order, with V8's item counts.
SUITES = [('standard', 10, 'graph', False, 192, 'shuffle', 'random'), ('hard10', 10, 'graph', True, 192, 'shuffle', 'random'),
          ('hard14', 14, 'graph', True, 384, 'shuffle', 'random'), ('long18', 18, 'graph', True, 384, 'shuffle', 'random'),
          ('lookup', 10, 'lookup', False, 192, 'shuffle', 'random'), ('compose', 10, 'compose', False, 192, 'shuffle', 'random'),
          ('pointer', 10, 'pointer', False, 192, 'shuffle', 'random'), ('bind', 10, 'bind', False, 192, 'shuffle', 'random'),
          ('shifted', 10, 'graph', False, 192, 'shifted', 'ordered')]
FRESH_SEED = 97300  # distinct from V8's 88300-89308 evaluation seeds, 89999 probe and 81000/82000+seed training seeds
SEED811 = ['A_alpha0', 'A_delayed_ncp', 'A_flat']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def copy_verified(src, dst):
    shutil.copyfile(src, dst)
    a, b = digest(src), digest(dst)
    if a != b:
        raise RuntimeError(f'copy mismatch {src}')
    return a


def main():
    if (SPEC / 'FROZEN.json').exists():
        raise SystemExit('Testbed already frozen; refusing to overwrite')
    code = {p.name: copy_verified(p, FROZEN / 'code' / p.name) for p in sorted((V8 / 'code').glob('*.py'))}
    data = {p.name: copy_verified(p, FROZEN / 'data' / p.name) for p in sorted((V8 / 'data').iterdir()) if p.is_file()}
    sys.path.insert(0, str(FROZEN / 'code'))
    import data as D
    import runtime as R
    if D.DATA.resolve() != (FROZEN / 'data').resolve():
        raise RuntimeError('frozen data.py does not resolve to the frozen data directory')
    if digest(D.ROOT.parent / 'v5_tiny/data/ts_train.npy') != json.loads((FROZEN / 'data/manifest.json').read_text())['tiny_stories']['sha256']:
        raise RuntimeError('TinyStories corpus path or content mismatch')
    # Exclusions: V8 dev/test evaluation items and every symbolic item V8's seed-811 runs trained on.
    v8_eval = set((FROZEN / 'data/evaluation_symbolic_hashes.txt').read_text().split())
    trained = set()
    seed811 = {}
    for name in SEED811:
        ck = V8 / 'runs' / name / 'checkpoint.pt'
        state = torch.load(ck, map_location='cpu', weights_only=False)
        trained |= set(state['mixer']['seen_symbols'])
        summary = json.loads((V8 / 'runs' / name / 'train_summary.json').read_text())
        seed811[name] = dict(checkpoint=str(ck), checkpoint_sha256=digest(ck), summary_sha256=digest(V8 / 'runs' / name / 'train_summary.json'),
                             status=summary['status'], completed_steps=summary['completed_steps'], seed=summary['seed'], alpha=summary['alpha'],
                             flat=summary['flat'], training_implementation_sha256=summary['implementation_sha256'], symbolic_exposure=summary['symbolic_exposure'])
        del state
    seen = set(v8_eval)
    fresh = {}
    rejected = dict(v8_eval=0, v8_trained=0, within_v9=0)
    for i, (name, n, task, hard, context, layout, labels) in enumerate(SUITES):
        count = 2048 if name in ('standard', 'lookup') else 512
        rng = np.random.RandomState(FRESH_SEED + i)
        rows = []
        while len(rows) < count:
            row = D.gen_symbolic(rng, n, task, hard, 4, layout, context, labels)
            key = row[3]['canonical']
            if key in v8_eval:
                rejected['v8_eval'] += 1; continue
            if key in trained:
                rejected['v8_trained'] += 1; continue
            if key in seen:
                rejected['within_v9'] += 1; continue
            seen.add(key); rows.append(row)
        path = FROZEN / 'data' / f'v9test_{name}.npz'
        np.savez(path, x=np.stack([r[0] for r in rows]), m=np.stack([r[1] for r in rows]), docs=np.stack([r[2] for r in rows]),
                 prompt_len=np.array([r[3]['prompt_len'] for r in rows]))
        hashes = sorted(r[3]['canonical'] for r in rows)
        fresh[name] = dict(n=count, nodes=n, task=task, hard=hard, context=context, layout=layout, label_style=labels, rng_seed=FRESH_SEED + i,
                           sha256=digest(path), canonical_hashes=hashes)
    new_hashes = sorted(h for v in fresh.values() for h in v['canonical_hashes'])
    excl = FROZEN / 'data/evaluation_symbolic_hashes.txt'
    excl.write_text('\n'.join(sorted(v8_eval | set(new_hashes))) + '\n')
    manifest = json.loads((FROZEN / 'data/manifest.json').read_text())
    for name, v in fresh.items():
        manifest['symbolic'][f'v9test_{name}'] = {k: v[k] for k in ('n', 'nodes', 'task', 'hard', 'context', 'layout', 'label_style', 'sha256')}
    manifest['v9'] = dict(protocol='V9 fresh sealed symbolic manifests from the frozen V8 generator; disjoint by canonical hash from V8 dev/test and from every symbolic item in V8 seed-811 training (mixer seen_symbols); appended to the training exclusion list before any V9 training',
                          fresh_seed_base=FRESH_SEED, rejected=rejected, exclusion_list_sha256=digest(excl), v8_exclusion_hashes=len(v8_eval),
                          v8_trained_hashes=len(trained), fresh_hashes=len(new_hashes),
                          sealed_text='No fresh text documents exist; V9 sealed text uses V8 test split (edu, ts), which V8 evaluated. V9 models never train on it.')
    (FROZEN / 'data/manifest.json').write_text(json.dumps(manifest, indent=2))
    frozen = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source=str(V8), code_sha256=code, data_sha256_at_copy=data,
                  frozen_implementation_sha256=R.implementation_hash(), v8_training_files_sha256={k: code[k] for k in ('model.py', 'train.py', 'data.py', 'runtime.py')},
                  data_manifest_sha256=digest(FROZEN / 'data/manifest.json'), exclusion_list_sha256=digest(excl),
                  layout='frozen/code holds the V8 code byte-identical; frozen/data holds V8 data plus V9 fresh sealed manifests; v9_tiny/v5_tiny symlinks ../v5_tiny for the TinyStories array')
    (SPEC / 'FROZEN.json').write_text(json.dumps(frozen, indent=2))
    (SPEC / 'SEED811.json').write_text(json.dumps(seed811, indent=2))
    (SPEC / 'FRESH_SEALED.json').write_text(json.dumps(dict(rejected=rejected, suites=fresh), indent=1))
    print(json.dumps(dict(implementation=frozen['frozen_implementation_sha256'], rejected=rejected, fresh=len(new_hashes), manifest=frozen['data_manifest_sha256'])))


if __name__ == '__main__':
    main()
