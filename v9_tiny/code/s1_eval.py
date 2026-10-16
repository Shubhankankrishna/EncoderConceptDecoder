"""S1 sealed evaluation and the S1 hypothesis tests (draft §5.3; procedure fixed in spec/S1_SEALED_PROCEDURE.json).

Symbolic suites come from the fresh V9 sealed manifests (split 'v9test'). No fresh text documents exist, so
sealed text uses V8's test split, which V8 evaluated; V9 models never trained on it.
"""
import argparse, json, time
import numpy as np
import v9paths
from v9paths import RUNS, SPEC, read_json, write_json
import torch
from model import Core, freeze
from runtime import load_system, implementation_hash
from evaluate import symbolic_eval, text_eval
import stats

SUITES = ('standard', 'lookup', 'compose', 'pointer', 'hard10', 'hard14', 'long18')
TEXT = ('edu', 'ts')


def sealed(source, out, split='v9test', n=None, text_n=256):
    s, ck = load_system(source); freeze(s); s.eval(); s.fast_generation = not isinstance(s.core, Core)
    t0 = time.time()
    result = dict(source=str(source), checkpoint_sha256=v9paths.digest(source), split=split, text_split='test (V8 sealed, historical)',
                  symbolic={}, text={}, frozen_implementation_sha256=implementation_hash())
    for name in SUITES:
        r = symbolic_eval(s, name, split, n=n)
        r.pop('records', None)  # responses are not needed for the S1 tests; success vectors are kept
        result['symbolic'][name] = r
    for src in TEXT:
        result['text'][src] = text_eval(s, 'test' if split == 'v9test' else split, source=src, n=text_n)
    result['seconds'] = time.time() - t0
    write_json(out, result)
    return result


def load(name):
    return read_json(RUNS / f'{name}_sealed' / 'summary.json')


def analyze(seeds=(901, 902)):
    """H3, H4, H5 as pre-declared in spec/S1_SEALED_PROCEDURE.json; seed 811 reported per seed, never in the tests."""
    arms = ('A_delayed', 'A_alpha0', 'A_flat')
    ev = {(a, s): load(f'S1_{a}_s{s}') for a in arms for s in seeds}
    hist = {}
    for a, v8 in (('A_delayed', 'A_delayed_ncp'), ('A_alpha0', 'A_alpha0'), ('A_flat', 'A_flat')):
        p = RUNS / f'S1_{a}_s811_sealed' / 'summary.json'
        if p.exists():
            hist[a] = read_json(p)

    def vec(a, s, suite='standard'):
        return np.array(ev[(a, s)]['symbolic'][suite]['success_vector'], bool)

    def success_test(a, b):
        per_seed = {}
        for s in seeds:
            x, y = vec(a, s), vec(b, s)
            per_seed[s] = dict(rate_a=float(x.mean()), rate_b=float(y.mean()), mcnemar=stats.mcnemar_exact(x, y), bootstrap=stats.paired_bootstrap(x, y, seed=s))
        X = np.stack([vec(a, s) for s in seeds]); Y = np.stack([vec(b, s) for s in seeds])
        pooled = stats.paired_bootstrap(X, Y, seed=7)
        return dict(per_seed=per_seed, pooled=pooled, p=stats.paired_signflip(X, Y, seed=7), sign_agrees=all(per_seed[s]['bootstrap']['delta'] > 0 for s in seeds))

    def text_test(a, b):
        per_seed = {}; nums_a = []; nums_b = []
        for s in seeds:
            na = np.concatenate([ev[(a, s)]['text'][t]['numerators'] for t in TEXT]); nb = np.concatenate([ev[(b, s)]['text'][t]['numerators'] for t in TEXT])
            den = np.concatenate([ev[(a, s)]['text'][t]['denominators'] for t in TEXT])
            den_b = np.concatenate([ev[(b, s)]['text'][t]['denominators'] for t in TEXT])
            if not np.array_equal(den, den_b):
                raise RuntimeError('text documents or targets differ between arms')
            per_seed[s] = dict(pooled_families=stats.document_bootstrap(na, nb, den, seed=s),
                               by_family={t: stats.document_bootstrap(ev[(a, s)]['text'][t]['numerators'], ev[(b, s)]['text'][t]['numerators'], ev[(a, s)]['text'][t]['denominators'], seed=s) for t in TEXT})
            nums_a.append(na); nums_b.append(nb)
        pooled = stats.document_bootstrap(np.mean(nums_a, 0), np.mean(nums_b, 0), den, seed=7)
        return dict(per_seed=per_seed, pooled=pooled, p=pooled['p'])

    h3 = success_test('A_delayed', 'A_flat')
    h5 = success_test('A_delayed', 'A_alpha0')
    h4 = text_test('A_flat', 'A_delayed')  # negative delta = flat lower NLL
    adj = stats.holm(dict(H3=h3['p'], H4=h4['p'], H5=h5['p']))
    h3['holm_p'] = adj['H3']; h4['holm_p'] = adj['H4']; h5['holm_p'] = adj['H5']
    h3['supported'] = bool(h3['sign_agrees'] and h3['pooled']['interval'][0] > 0 and adj['H3'] < .05)
    h5['supported'] = bool(h5['sign_agrees'] and h5['pooled']['interval'][0] > 0 and adj['H5'] < .05)
    h4['supported'] = bool(all(h4['per_seed'][s]['pooled_families']['delta'] <= -.02 and h4['per_seed'][s]['pooled_families']['interval'][1] < 0 for s in seeds) and adj['H4'] < .05)
    table = {f'{a}_s{s}': dict(**{k: ev[(a, s)]['symbolic'][k]['rate'] for k in SUITES}, **{'nll_' + t: ev[(a, s)]['text'][t]['nll'] for t in TEXT}) for a in arms for s in seeds}
    for a, e in hist.items():
        table[f'{a}_s811_historical'] = dict(**{k: e['symbolic'][k]['rate'] for k in SUITES}, **{'nll_' + t: e['text'][t]['nll'] for t in TEXT})
    secondary = dict(alpha0_vs_flat=success_test('A_alpha0', 'A_flat'), hard10_delayed_vs_alpha0={s: stats.paired_bootstrap(vec('A_delayed', s, 'hard10'), vec('A_alpha0', s, 'hard10'), seed=s) for s in seeds})
    out = dict(procedure=read_json(SPEC / 'S1_SEALED_PROCEDURE.json'), H3=h3, H4=h4, H5=h5, holm=adj, table=table, secondary_not_in_family=secondary,
               note='Seed 811 rows are historical/development evidence (draft §5.2) and are not used in any test.')
    write_json(RUNS / 'S1_RESULTS.json', out)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--source'); ap.add_argument('--out'); ap.add_argument('--analyze', action='store_true'); a = ap.parse_args()
    torch.set_num_threads(4)
    if a.analyze:
        r = analyze(); print(json.dumps({k: r[k].get('supported') for k in ('H3', 'H4', 'H5')}))
    else:
        smoke = dict(split='dev', n=64, text_n=32) if v9paths.SMOKE else {}
        r = sealed(a.source, a.out, **smoke); print(json.dumps({k: v['rate'] for k, v in r['symbolic'].items()}))
