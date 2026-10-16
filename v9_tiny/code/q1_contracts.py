"""Q1 preflight contracts (draft §13.1 items 11 and 13, plus testbed integrity and V8's own contract suite).

Runs from a temporary working directory, writes smoke outputs only there, and fails if any file under any
project runs/ directory changes during the session (item 13). On success it writes spec/LOCK_Q1.json and
tests/q1_contracts.json; on failure it writes only tests/q1_contracts.json.
"""
import copy, hashlib, json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import numpy as np
import v9paths
from v9paths import V9, PROJECT, V8, RUNS, SPEC, TESTS, FROZEN_CODE, DATA, CODE, digest, files_hash, read_json, write_json
import torch

PY = sys.executable
tests = []


def check(name, cond, **details):
    if not cond:
        raise AssertionError(name + ': ' + json.dumps(details, default=str)[:2000])
    tests.append(dict(name=name, passed=True, **details))


def runs_snapshot():
    snap = {}
    for root in sorted(PROJECT.glob('*/runs')):
        for dirpath, _, files in os.walk(root):
            for f in files:
                p = os.path.join(dirpath, f)
                try:
                    st = os.stat(p); snap[p] = (st.st_size, st.st_mtime_ns)
                except FileNotFoundError:
                    pass
    return snap


def identity():
    import runtime
    from drive_q1 import Q1_FILES
    ident = dict(frozen_implementation_sha256=runtime.implementation_hash(), q1_files_sha256=files_hash(Q1_FILES),
                 contracts_file_sha256=digest(CODE / 'q1_contracts.py'), data_manifest_sha256=digest(DATA / 'manifest.json'),
                 exclusion_list_sha256=digest(DATA / 'evaluation_symbolic_hashes.txt'))
    ident['lock_sha256'] = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    return ident


def frozen_integrity():
    frozen = read_json(SPEC / 'FROZEN.json')
    for name, sha in frozen['code_sha256'].items():
        check('frozen code byte-identical ' + name, digest(FROZEN_CODE / name) == sha == digest(V8 / 'code' / name))
    check('no extra frozen code files', sorted(p.name for p in FROZEN_CODE.glob('*.py')) == sorted(frozen['code_sha256']))
    import runtime
    check('frozen implementation hash', runtime.implementation_hash() == frozen['frozen_implementation_sha256'] == 'fcc93d876201367618a195d2d56feb6767800cb633ec7565ec161a74c3a42b58')
    for name in ('model.py', 'train.py', 'data.py', 'runtime.py'):
        check('V8 training recipe file ' + name, frozen['code_sha256'][name] == read_json(V8 / 'spec/CONTINUATION.json')['unchanged_training_files_sha256']['code/' + name])
    check('manifest and exclusion recorded', digest(DATA / 'manifest.json') == frozen['data_manifest_sha256'] and digest(DATA / 'evaluation_symbolic_hashes.txt') == frozen['exclusion_list_sha256'])
    for name, sha in frozen['data_sha256_at_copy'].items():
        if name not in ('manifest.json', 'evaluation_symbolic_hashes.txt'):
            check('frozen data unchanged ' + name, digest(DATA / name) == sha)
    import data as D
    check('frozen data path', D.DATA.resolve() == DATA.resolve())


def fresh_sealed():
    """Contract 11 (S1 part): chunk alignment, reference answers verify, disjointness, training exclusion."""
    import data as D
    fresh = read_json(SPEC / 'FRESH_SEALED.json')['suites']
    v8_eval = set((V8 / 'data/evaluation_symbolic_hashes.txt').read_text().split())
    trained = set()
    for v in read_json(SPEC / 'SEED811.json').values():
        trained |= set(torch.load(v['checkpoint'], map_location='cpu', weights_only=False)['mixer']['seen_symbols'])
    excluded = set((DATA / 'evaluation_symbolic_hashes.txt').read_text().split())
    allh = []
    for name, meta in fresh.items():
        a = np.load(DATA / f'v9test_{name}.npz'); n = len(a['x']); hashes = []; bad = 0
        check(f'{name} count', n == meta['n'] == (2048 if name in ('standard', 'lookup') else 512))
        for i in range(n):
            p = int(a['prompt_len'][i]); r = int(a['m'][i].sum())
            if p % 4 or r % 4 or not (a['docs'][i, :p + r] == 0).all() or not (a['docs'][i, p + r:] == -1).all():
                bad += 1; continue
            if not D.verify(a['x'][i, :p], a['x'][i, p:p + r])['success']:
                bad += 1; continue
            hashes.append(D.canonical(a['x'][i, :p]))
        check(f'{name} chunk-aligned and reference answers verify', bad == 0, bad=bad)
        check(f'{name} hashes match manifest', sorted(hashes) == meta['canonical_hashes'] and len(set(hashes)) == n)
        check(f'{name} file hash', digest(DATA / f'v9test_{name}.npz') == meta['sha256'] == read_json(DATA / 'manifest.json')['symbolic'][f'v9test_{name}']['sha256'])
        corrupt = a['x'][0, p:p + r].copy(); corrupt[0] = D.PAD
        check(f'{name} verifier rejects a corrupted answer', not D.verify(a['x'][0, :p], corrupt)['success'])
        allh += hashes
    s = set(allh)
    check('fresh suites mutually disjoint', len(s) == len(allh))
    check('fresh disjoint from V8 dev/test', not (s & v8_eval), overlap=len(s & v8_eval))
    check('fresh disjoint from V8 seed-811 training symbols', not (s & trained), trained=len(trained))
    check('fresh in training exclusion list', s <= excluded and v8_eval <= excluded)
    for seed in (901, 902):
        mix = D.Mixer(81000 + seed)
        for _ in range(20):
            mix.batch()
        check(f'Mixer seed {seed}: no fresh or V8 evaluation item drawn', not (mix.seen_symbols & excluded), drawn=len(mix.seen_symbols))
        check(f'Mixer seed {seed}: V8 per-update quota', mix.seen_tasks == dict(graph=480, lookup=80, compose=40, pointer=40))
    check('S1 seeds differ in data order', D.Mixer(81901).batch()['x'].tolist() != D.Mixer(81902).batch()['x'].tolist())


def seed811():
    for name, v in read_json(SPEC / 'SEED811.json').items():
        check('seed-811 checkpoint unchanged ' + name, digest(v['checkpoint']) == v['checkpoint_sha256'])
        check('seed-811 summary completed ' + name, v['status'] == 'completed' and v['completed_steps'] == 24000 and v['seed'] == 811)


def compile_parity():
    """V8 stage_checks' eager-vs-compiled gradient parity for stage A (hierarchy and flat): decides --compiled as V8 did."""
    from model import System, Foundation, Flat
    from runtime import DEVICE, sync, optimizer, seed_all
    from data import Mixer
    from train import forward_function, objective
    seed_all(817); b = Mixer(81701).batch(DEVICE); out = {}
    for label, flat in (('A/dense', False), ('A_FLAT/dense', True)):
        s = System(Flat() if flat else Foundation()).to(DEVICE); grads = {}; losses = {}
        for mode, model in (('eager', copy.deepcopy(s)), ('compiled', copy.deepcopy(s))):
            fn = forward_function(model, 'A', mode == 'compiled'); loss, *_ = objective(model, 'A', b, fn, .1, 0.); loss.backward(); sync()
            losses[mode] = float(loss.detach()); grads[mode] = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
        errors = [dict(parameter=n, max_abs=float((g - grads['compiled'][n]).abs().max()), relative=float((g - grads['compiled'][n]).norm() / g.norm().clamp_min(1e-8))) for n, g in grads['eager'].items()]
        safe = abs(losses['eager'] - losses['compiled']) < 1e-4 and all(e['max_abs'] < 2e-4 and e['relative'] < .01 for e in errors)
        out[label] = safe
        tests.append(dict(name='compile parity ' + label, passed=True, compiled_safe=safe, loss=losses, worst=max(errors, key=lambda e: e['relative'])))
    return all(out.values())


def driver_args():
    from drive_q1 import train_args, ARMS, SEEDS
    for arm, v8 in (('A_delayed', 'A_delayed_ncp'), ('A_alpha0', 'A_alpha0'), ('A_flat', 'A_flat')):
        cfg = read_json(V8 / 'runs' / v8 / 'config.json'); args = train_args(arm, 901, True)
        val = lambda k: args[args.index('--' + k) + 1]
        same = dict(stage=val('stage') == cfg['stage'], steps=int(val('steps')) == cfg['steps'], batch=int(val('batch')) == cfg['batch'],
                    context=int(val('context')) == cfg['context'], alpha=float(val('alpha')) == cfg['alpha'], mode=val('mode') == cfg['mode'],
                    core=val('core') == cfg['core'], flat=('--flat' in args) == cfg['flat'], curves=('--development-curves' in args) == cfg['development_curves'],
                    compiled=('--compiled' in args) == cfg['compiled'], lr_default=cfg['lr'] == .002 and '--lr' not in args, beta=cfg['beta'] == .25 and '--beta' not in args)
        check('driver reproduces V8 recipe ' + arm, all(same.values()), **same)
    check('seeds avoid V8 generator seeds', all(81000 + s not in range(88300, 89309) and 82000 + s != 89999 and s != 811 for s in SEEDS))


def smoke(tmp):
    """Production-path smoke on the frozen train.py: compiled hierarchy and flat updates, then interrupted-and-resumed training."""
    env = dict(os.environ)
    def train(out, extra):
        r = subprocess.run([PY, str(FROZEN_CODE / 'train.py'), '--stage', 'A', '--out', str(out), '--batch', '64', '--context', '192', '--seed', '901', '--mode', 'foundation', '--core', 'dense', *extra],
                           cwd=tmp, env=env, capture_output=True, text=True, timeout=1800)
        if r.returncode:
            raise AssertionError('smoke train failed: ' + r.stdout[-1500:] + r.stderr[-1500:])
        return read_json(Path(out) / 'train_summary.json')
    a = train(Path(tmp) / 'hier', ['--steps', '20', '--seconds', '600', '--alpha', '0.1', '--compiled'])
    check('smoke compiled hierarchy 20 updates', a['status'] == 'completed' and a['completed_steps'] == 20 and np.isfinite(a['last']['loss']), loss=a['last']['loss'], wall=a['wall_s'])
    f = train(Path(tmp) / 'flat', ['--steps', '10', '--seconds', '600', '--alpha', '0', '--flat', '--compiled'])
    check('smoke compiled flat 10 updates', f['status'] == 'completed' and f['completed_steps'] == 10 and f['active_parameters'] == 2450048, loss=f['last']['loss'])
    extra = ['--steps', '400', '--seconds', '6', '--alpha', '0.1']
    r1 = train(Path(tmp) / 'resume', extra)
    r2 = train(Path(tmp) / 'resume', extra + ['--resume'])
    log = [json.loads(l) for l in (Path(tmp) / 'resume' / 'log.jsonl').read_text().splitlines()]
    check('interrupted run resumes from its checkpoint', r1['status'] == 'partial_time_budget' and r2['completed_steps'] > r1['completed_steps'],
          first=r1['completed_steps'], second=r2['completed_steps'], logged_steps=[v['step'] for v in log])


def eval_smoke():
    from runtime import load_system
    from model import freeze
    from evaluate import symbolic_eval, text_eval
    s, _ = load_system(read_json(SPEC / 'SEED811.json')['A_delayed_ncp']['checkpoint']); freeze(s); s.eval(); s.fast_generation = True
    r = symbolic_eval(s, 'standard', 'v9test', n=64); t = text_eval(s, 'test', source='edu', n=32)
    check('frozen evaluator runs on fresh sealed split', r['n'] == 64 and 0 <= r['rate'] <= 1 and np.isfinite(t['nll']), seed811_standard_64=r['rate'], edu_nll_32=t['nll'])


def main():
    out_path = TESTS / 'q1_contracts.json'
    before = runs_snapshot(); tmp = tempfile.mkdtemp(prefix='v9_q1_contracts_'); os.chdir(tmp)
    t0 = time.time(); status = 'failed'; compiled = None; ident = None
    try:
        torch.set_num_threads(4)
        ident = identity()
        frozen_integrity(); seed811(); fresh_sealed(); driver_args()
        import stats; check('statistics self-test', stats.self_test())
        import checks as v8checks  # V8's executable contract suite on the frozen copy, untrained model
        j = v8checks.run(source=None, fast=True); check('V8 contract suite (frozen checks.py)', j['status'] == 'passed', contracts=j['contracts'])
        import revision_checks  # data integrity, TinyStories disjointness, cache parity incl. hard14/long18 prompts
        j = revision_checks.run(None); check('V8 revision contract suite (frozen revision_checks.py)', j['status'] == 'passed', checks=len(j['checks']))
        compiled = compile_parity()
        check('compiled stage A matches V8 execution choice', compiled is True)
        smoke(tmp); eval_smoke()
        after = runs_snapshot()
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        check('item 13: no file under any runs/ changed during the test session', not changed, changed=changed[:20], files_watched=len(before))
        check('identity unchanged during session', identity() == ident)
        status = 'passed'
    except Exception:
        tests.append(dict(name='EXCEPTION', passed=False, traceback=traceback.format_exc()))
    finally:
        result = dict(status=status, block='Q1', items=['11 (fresh sealed, S1 part)', '13 (test hygiene)', 'testbed integrity', 'V8 contract suites', 'compile parity', 'smoke'],
                      lock_sha256=ident['lock_sha256'] if ident else None, identity=ident, compiled=compiled, tests=tests, seconds=time.time() - t0, temp_dir=tmp,
                      device=os.environ.get('V8_DEVICE', 'mps' if torch.backends.mps.is_available() else 'cpu'), torch=torch.__version__)
        write_json(out_path, result)
        if status == 'passed':
            write_json(SPEC / 'LOCK_Q1.json', dict(**ident, compiled=compiled, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                                   files=[str(p.relative_to(V9)) for p in __import__('drive_q1').Q1_FILES]))
        print(json.dumps(dict(status=status, tests=len(tests), seconds=round(time.time() - t0))))
        if status != 'passed':
            print(tests[-1].get('traceback', '')); sys.exit(1)


if __name__ == '__main__':
    main()
