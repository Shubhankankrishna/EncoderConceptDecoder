"""Paired comparisons and the v11.md §7.2 Tier-1 gates, computed mechanically.

  python -B -m v11_tiny.compare        (writes v11_tiny/comparison.json and v11_tiny/COMPARISON.md)

Estimator: family NLL = sum(ce) / sum(scored tokens) over the fixed dev panel documents (the inherited
definition). Uncertainty: paired document bootstrap (documents matched by (source, index), resampled
with replacement, both arms recomputed on the same resample), 2,000 draws, per seed; 95% percentile
interval. A gate passes only if its condition holds in EVERY seed with the interval on the stated side
of the margin (v11.md §7.2). Seed variation is reported separately (per-seed rows), never pooled.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'
ARMS = ('C0', 'ORD', 'ORDE', 'X8', 'NOAUX', 'K1NB', 'K1', 'K1SC', 'NOAUXSC', 'K1NBSC')   # K1SC: diagnostic (separate clip groups), seed 1 only
SEEDS = (1, 2, 3)          # seed 3 exists only for K1 and C0 (pre-committed promotion seed)
FAMILIES = ('text', 'math', 'newmath', 'sudoku_easy', 'sudoku_hard')
MODULES = ('arithmetic__add_sub_multiple', 'arithmetic__mul', 'comparison__sort')
DRAWS = 2000


def load(arm, seed):
    p = RUNS / f'{arm}_s{seed}' / 'evaluation.json'
    return json.loads(p.read_text()) if p.exists() else None


def docs(ev, fam, module=None):
    rows = ev['nll'][fam]['documents']
    if module: rows = [r for r in rows if r['source'] == module]
    return {(r['source'], r['index']): r for r in rows}


def paired(a, b, key_a='ce', key_b='ce', seed=0):
    """Delta = NLL(a) - NLL(b) with paired bootstrap CI. a, b: dict key -> row; must share keys and n."""
    keys = sorted(set(a) & set(b))
    assert len(keys) == len(a) == len(b), 'panels differ between arms'
    n = np.array([a[k]['n'] for k in keys], float); assert all(a[k]['n'] == b[k]['n'] for k in keys)
    ca = np.array([a[k][key_a] for k in keys]); cb = np.array([b[k][key_b] for k in keys])
    delta = (ca.sum() - cb.sum()) / n.sum()
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(keys), (DRAWS, len(keys)))
    boot = (ca[idx].sum(1) - cb[idx].sum(1)) / n[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(delta=float(delta), lo=float(lo), hi=float(hi), documents=len(keys))


def within(ev, fam, key, module=None, seed=0):
    d = docs(ev, fam, module)
    return paired(d, d, key, 'ce', seed)


def fmt(r): return 'n/a' if r is None else f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"


def main():
    ev = {(a, s): load(a, s) for a in ARMS for s in SEEDS}
    have = {k for k, v in ev.items() if v is not None}
    out = dict(available=sorted(f'{a}_s{s}' for a, s in have), absolute={}, pairs={}, interventions={}, gates={}, extras={})
    # absolute NLL
    for (a, s) in sorted(have):
        e = ev[(a, s)]
        bm = e['nll']['math']['summary']['by_module']
        out['absolute'][f'{a}_s{s}'] = {f: e['nll'][f]['summary']['nll'] for f in FAMILIES} | \
            {m: bm[m]['nll'] if m in bm else None for m in MODULES}
    # pairwise vs references
    pairs = [('K1', 'X8'), ('K1', 'C0'), ('K1', 'ORD'), ('K1', 'K1NB'), ('K1', 'NOAUX'), ('K1NB', 'X8'), ('K1NB', 'C0'),
             ('NOAUX', 'C0'), ('X8', 'C0'), ('ORD', 'C0'), ('K1NB', 'NOAUX'), ('NOAUX', 'X8'),
             ('K1', 'ORDE'), ('C0', 'ORDE'), ('ORDE', 'ORD'), ('X8', 'ORDE'), ('K1NB', 'ORDE'),
             ('NOAUXSC', 'C0'), ('K1NBSC', 'C0'), ('K1NBSC', 'NOAUXSC'), ('K1SC', 'K1NBSC'), ('NOAUX', 'NOAUXSC'), ('K1NB', 'K1NBSC'),
             ('K1SC', 'K1NB'), ('K1', 'K1SC'), ('K1SC', 'X8'), ('K1SC', 'C0'), ('K1SC', 'NOAUX'),
             ('K1NBSC', 'X8'), ('NOAUXSC', 'X8'), ('NOAUXSC', 'NOAUX')]
    for a, b in pairs:
        for s in SEEDS:
            if (a, s) not in have or (b, s) not in have: continue
            row = {f: paired(docs(ev[(a, s)], f), docs(ev[(b, s)], f), seed=s) for f in FAMILIES}
            row |= {m: paired(docs(ev[(a, s)], 'math', m), docs(ev[(b, s)], 'math', m), seed=s) if docs(ev[(a, s)], 'math', m) else None
                    for m in MODULES}
            out['pairs'][f'{a}-{b}_s{s}'] = row
    # interventions (within-model paired deltas)
    for (a, s) in sorted(have):
        e = ev[(a, s)]; first = e['nll']['text']['documents'][0]
        keys = [k for k in first if k not in ('index', 'source', 'n', 'ce')]
        if not keys: continue
        out['interventions'][f'{a}_s{s}'] = {k: {f: within(e, f, k, seed=s) for f in FAMILIES} |
                                                {m: within(e, 'math', k, m, seed=s) if docs(e, 'math', m) else None for m in MODULES} for k in keys}
    # gates
    def per_seed(fn):
        res = {}
        for s in SEEDS:
            try: res[s] = fn(s)
            except (KeyError, TypeError, AssertionError) as exc: res[s] = dict(passed=None, missing=repr(exc))
        core = [res[s] for s in (1, 2)]
        if any(r.get('passed') == 'n/a' for r in core): verdict = 'n/a (informational)'
        elif any(r.get('passed') is None for r in core): verdict = None
        else: verdict = all(r['passed'] for r in res.values() if r.get('passed') is not None)
        return dict(per_seed=res, passed=verdict, seeds_used=[s for s, r in res.items() if r.get('passed') is not None])
    P = out['pairs']; I = out['interventions']
    def g1(s):
        r = P[f'K1-X8_s{s}']; mods = [r[m]['hi'] <= -.02 for m in MODULES]
        return dict(math=fmt(r['math']), text=fmt(r['text']), modules=[fmt(r[m]) for m in MODULES],
                    passed=bool(r['math']['hi'] <= -.02 and sum(mods) >= 2 and r['text']['hi'] <= .01))
    def g2(s):
        r = P[f'K1-C0_s{s}']
        return dict(text=fmt(r['text']), modules=[fmt(r[m]) for m in MODULES],
                    passed=bool(r['text']['hi'] <= .01 and all(r[m]['hi'] <= .02 for m in MODULES)))
    def g2b(s):
        return dict(noaux_minus_c0={f: fmt(P[f'NOAUX-C0_s{s}'][f]) for f in ('text', 'math')},
                    k1_minus_noaux={f: fmt(P[f'K1-NOAUX_s{s}'][f]) for f in ('text', 'math')},
                    criterion='informational: NOAUX-C0 and K1-NOAUX, no threshold', passed='n/a')
    def g3(s):
        off = I[f'K1_s{s}']['all_concept_off']['math']; lat = ev[('K1', s)]['latent']['math']['normalized']
        f_ok = lat['ridge_forecast'] <= lat['ridge_input'] - .02
        return dict(concept_off_math=fmt(off), ridge_forecast=lat['ridge_forecast'], ridge_input=lat['ridge_input'],
                    ridge_state=lat['ridge_state'], direct_forecast=lat['direct_f2'],
                    passed=bool(off['lo'] >= .02 and f_ok))
    def g4(s):
        r = P[f'K1-K1NB_s{s}']; gain = r['text']['hi'] <= -.005 or r['math']['hi'] <= -.01
        guard = all(r[f]['delta'] <= .005 for f in FAMILIES)
        return dict(text=fmt(r['text']), math=fmt(r['math']), worst_family=max(r[f]['delta'] for f in FAMILIES),
                    passed=bool(gain and guard))
    def g5(s):
        r = P[f'K1-ORD_s{s}']; return dict(text=fmt(r['text']), math=fmt(r['math']), passed=bool(r['text']['hi'] <= .01),
                                            label='vs ORD, which has no Engram: NOT a plain-model test (see K1-ORDE, seed 1)')
    def align(s):
        c0 = ev[('C0', s)]['alignment']
        res = {a: {f: ev[(a, s)]['alignment'][f]['sensitivity'] - c0[f]['sensitivity'] for f in ('math', 'newmath')}
               for a in ('X8', 'NOAUX', 'K1NB', 'K1') if (a, s) in have}
        if not res: raise KeyError('no concept arm at this seed')
        r17 = lambda a_: {f: (lambda c: max(c[1:]) - min(c[1:]))(ev[(a_, s)]['alignment'][f]['nll_by_offset']) for f in ('math', 'newmath')}
        c17 = r17('C0')
        res17 = {a: {f: r17(a)[f] - c17[f] for f in ('math', 'newmath')} for a in res}
        return dict(minus_c0=res, minus_c0_r1_7=res17, note='pre-registered: max-min over r=0..7 (no CI); r=1..7 excludes the format shock of the first inserted token',
                    passed=all(v <= .01 for r in res.values() for v in r.values()))
    for name, fn in (('G1_latent_beats_V10.3', g1), ('G2_noninferior_to_C0', g2), ('G2b_topology_vs_objectives', g2b),
                     ('G3_concept_state_used', g3), ('G4_bridge_earns_place', g4), ('G5_vs_ordinary', g5),
                     ('alignment_sensitivity', align)):
        out['gates'][name] = per_seed(fn)
    # post hoc: the same G1-G3 margins applied to other arms (labelled; the pre-registered verdicts are on K1 only)
    out['post_hoc_gates'] = {}
    for arm in ('K1NB', 'K1SC', 'K1NBSC', 'NOAUXSC'):
        for s in SEEDS:
            if (arm, s) not in have: continue
            row = {}
            try:
                r = P[f'{arm}-X8_s{s}']; mods = [r[m]['hi'] <= -.02 for m in MODULES]
                row['G1'] = dict(math=fmt(r['math']), text=fmt(r['text']), passed=bool(r['math']['hi'] <= -.02 and sum(mods) >= 2 and r['text']['hi'] <= .01))
                r = P[f'{arm}-C0_s{s}']
                row['G2'] = dict(text=fmt(r['text']), modules=[fmt(r[m]) for m in MODULES], passed=bool(r['text']['hi'] <= .01 and all(r[m]['hi'] <= .02 for m in MODULES)))
                off = I[f'{arm}_s{s}']['all_concept_off']['math']; lat = ev[(arm, s)]['latent']['math']['normalized']
                row['G3'] = dict(concept_off_math=fmt(off), ridge_forecast=lat['ridge_forecast'], ridge_input=lat['ridge_input'], ridge_state=lat['ridge_state'],
                                 passed=bool(off['lo'] >= .02 and lat['ridge_forecast'] <= lat['ridge_input'] - .02))
            except KeyError as exc: row['missing'] = repr(exc)
            out['post_hoc_gates'][f'{arm}_s{s}'] = row
    # extras: generation, grid shift, latent, scales
    for (a, s) in sorted(have):
        e = ev[(a, s)]
        out['extras'][f'{a}_s{s}'] = dict(
            generation={f: dict(correct=v['correct'], n=v['n'], truncated=v['truncated'],
                                by_module={m: (x['correct'], x['n'], x['truncated']) for m, x in v['by_module'].items()})
                        for f, v in e['generation'].items()},
            grid_shift={f: v['degradation'] for f, v in e['grid_shift'].items()},
            alignment={f: v['sensitivity'] for f, v in e['alignment'].items()},
            alignment_r1_7={f: max(v['nll_by_offset'][1:]) - min(v['nll_by_offset'][1:]) for f, v in e['alignment'].items()},
            alignment_curve={f: v['nll_by_offset'] for f, v in e['alignment'].items()},
            forecast_variance_ratio={f: v['forecast_variance_ratio'] for f, v in e.get('latent', {}).items()},
            pooled_participation={f: v['participation_ratio'] for f, v in e.get('pooled_participation', {}).items()},
            latent_math=e.get('latent', {}).get('math'), latent_text=e.get('latent', {}).get('text'),
            fvu_v103_math=e.get('fvu_v103', {}).get('math', {}).get('fvu'), scales=e['scales'])
    (ROOT / 'comparison.json').write_text(json.dumps(out, indent=1) + '\n')
    lines = ['# v11_tiny comparison (generated by compare.py)', '',
             'Δ = NLL(first) − NLL(second), nats/token, dev panels; [95% paired document bootstrap, 2,000 draws]. Per seed.', '']
    lines += ['## Gates', '', '| Gate | Seed 1 | Seed 2 | Seed 3 | Pass |', '|---|---|---|---|---|']
    for name, g in out['gates'].items():
        cells = [json.dumps({k: v for k, v in g['per_seed'][s].items() if k != 'passed'}) for s in SEEDS]
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | {g['passed']} (seeds {g['seeds_used']}) |")
    lines += ['', '## Post-hoc: G1-G3 margins applied to other arms (not pre-registered)', '', '| Run | G1 | G2 | G3 |', '|---|---|---|---|']
    for k, v in out['post_hoc_gates'].items():
        lines.append(f"| {k} | " + ' | '.join(json.dumps(v.get(g, v.get('missing'))) for g in ('G1', 'G2', 'G3')) + ' |')
    lines += ['', '## Absolute dev NLL', '', '| Run | ' + ' | '.join(FAMILIES + MODULES) + ' |', '|---' * (1 + len(FAMILIES) + len(MODULES)) + '|']
    for k, v in out['absolute'].items():
        lines.append(f'| {k} | ' + ' | '.join('n/a' if v[f] is None else f'{v[f]:.4f}' for f in FAMILIES + MODULES) + ' |')
    lines += ['', '## Paired differences', '', '| Pair | ' + ' | '.join(FAMILIES + MODULES) + ' |', '|---' * (1 + len(FAMILIES) + len(MODULES)) + '|']
    for k, v in out['pairs'].items():
        lines.append(f'| {k} | ' + ' | '.join(fmt(v[f]) for f in FAMILIES + MODULES) + ' |')
    lines += ['', '## Interventions (Δ = intervened − base, same model)', '']
    for k, v in out['interventions'].items():
        lines += [f'### {k}', '', '| Intervention | ' + ' | '.join(FAMILIES + MODULES) + ' |', '|---' * (1 + len(FAMILIES) + len(MODULES)) + '|']
        for name, row in v.items():
            lines.append(f'| {name} | ' + ' | '.join(fmt(row[f]) for f in FAMILIES + MODULES) + ' |')
        lines.append('')
    (ROOT / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')
    for name, g in out['gates'].items(): print(name, g['passed'])


if __name__ == '__main__':
    main()
