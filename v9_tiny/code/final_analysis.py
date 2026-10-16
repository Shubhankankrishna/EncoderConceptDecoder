"""Final analysis (draft §13.2, I18): the Holm-corrected families within each stage, per-seed and pooled paired tests,
margin claims by one-sided sign-flip tests against the margin, and the summary report. Missing runs are reported as
missing, never as passing or failing."""
import json
from pathlib import Path
import numpy as np
import v9paths
from v9paths import RUNS, SPEC, read_json, write_json
from stats import mcnemar_exact, paired_bootstrap, paired_signflip, holm
import register_s2 as R2

RESAMPLES = 10000


def onesided(d, direction, seed=0):
    """Sign-flip p-value for H1: mean(d) > 0 (direction +1) or < 0 (direction -1); d may be (seeds, items)."""
    d = np.asarray(d, float); d = d.mean(0) if d.ndim == 2 else d; rng = np.random.RandomState(seed)
    null = (rng.choice([-1., 1.], size=(RESAMPLES, len(d))) * d).mean(1); pt = d.mean()
    return float((1 + ((null >= pt - 1e-12) if direction > 0 else (null <= pt + 1e-12)).sum()) / (RESAMPLES + 1))


def compare(a_list, b_list, name, kind, margin=None, direction=+1):
    """a_list/b_list: per-seed per-item vectors (same items within a seed). kind: diff | noninf | bound | equiv | drop."""
    if not a_list or any(v is None for v in a_list + b_list):
        return dict(name=name, status='missing')
    n = min(min(len(v) for v in a_list), min(len(v) for v in b_list)); A = np.array([v[:n] for v in a_list], float); B = np.array([v[:n] for v in b_list], float)
    per_seed = [dict(mcnemar=mcnemar_exact(A[s], B[s]) if set(np.unique(np.r_[A[s], B[s]])) <= {0., 1.} else None, interval=paired_bootstrap(A[s], B[s])) for s in range(len(A))]
    pooled = paired_bootstrap(A, B); d = A - B
    if kind == 'diff':
        p = paired_signflip(A, B)
    elif kind == 'noninf':
        p = onesided(d + margin, +1)                       # H0: mean <= -margin
    elif kind == 'bound':
        p = onesided(d - margin, -1)                       # H0: mean >= +margin
    elif kind == 'equiv':
        p = max(onesided(d + margin, +1), onesided(d - margin, -1))   # TOST
    else:
        raise ValueError(kind)
    signs = [int(np.sign(x['interval']['delta'])) for x in per_seed]
    return dict(name=name, kind=kind, margin=margin, direction=direction, p=p, per_seed=per_seed, pooled=pooled, signs=signs, n_items=n, seeds=len(A))


def s5_vec(seed, cell):
    f = RUNS / 'S5' / f's{seed}_{cell}.json'
    return read_json(f)['success'] if f.exists() else None


def main():
    out = dict(families={}, notes=[])
    # S1 (Q1): H3, H4, H5 are already corrected in S1_RESULTS.json.
    out['families']['S1'] = read_json(RUNS / 'S1_RESULTS.json') if (RUNS / 'S1_RESULTS.json').exists() else 'missing'
    q5 = read_json(RUNS / 'Q5_RESULT.json') if (RUNS / 'Q5_RESULT.json').exists() else None
    seeds = [r['seed'] for r in q5['seeds']] if q5 else []
    # S3: H11 (S3b loses >= 20% of its swap sensitivity): d_i = s3b_i - .8 s2_i, H1: mean < 0.
    fam = {}
    s3 = read_json(RUNS / 'Q4_RESULT.json') if (RUNS / 'Q4_RESULT.json').exists() else None
    if s3 and all('joint' in r['arms'] for r in s3['seeds']):
        a, b = [], []
        promo = read_json(RUNS / 'S2_PROMOTION.json')['models']
        for idx, r in enumerate(s3['seeds']):
            assert promo[idx]['s3_seed'] == r['seed'], 'S3 seed / S2 promoted model pairing mismatch'
            m3 = read_json(r['arms']['joint']['measure']); m2 = read_json(promo[idx]['source']['measure'])
            a.append([x['sensitivity'] for _, x in R2.pooled_states(m3, 'first')]); b.append([.8 * x['sensitivity'] for _, x in R2.pooled_states(m2, 'first')])
        n = min(min(map(len, a)), min(map(len, b))); d = np.array([np.array(x[:n]) - np.array(y[:n]) for x, y in zip(a, b)])
        fam['H11'] = dict(name='H11', kind='drop', p=onesided(d, -1), pooled=paired_bootstrap(np.array([x[:n] for x in a]), np.array([y[:n] for y in b])), seeds=len(a))
    else:
        fam['H11'] = dict(name='H11', status='missing (S3b not run or K8 not promoted)')
    out['families']['S3'] = fam
    # S4: L3 (EI vs RFT, greedy sealed hard10); leaves the family if RH(1) was not allowed (§13.2).
    fam = {}
    rh = q5 and q5.get('rh_allowed')
    ei = [read_json(RUNS / 'S4F' / f's{s}_EI.json')['hard10']['success'] if (RUNS / 'S4F' / f's{s}_EI.json').exists() else None for s in seeds]
    rft = [read_json(RUNS / 'S4F' / f's{s}_rft.json')['hard10']['success'] if (RUNS / 'S4F' / f's{s}_rft.json').exists() else None for s in seeds]
    l3 = compare(ei, rft, 'L3', 'diff')
    if rh:
        fam['L3'] = l3
    else:
        out['notes'].append('L3 reported uncorrected and exploratory: RH(1) was not allowed (§13.2)'); out['L3_exploratory'] = l3
    out['families']['S4'] = fam
    # S5: L1 (RH1 vs token tree at 16x), H14 (answer-loop GradCuit - random < +2 points), H16 (decoder-free RH1 non-inferior to grounded RH1, -2 points).
    fam = {}
    fam['L1'] = compare([s5_vec(s, 'sealed_RH1_hard10_16x') for s in seeds], [s5_vec(s, 'sealed_token_tree_hard10_16x') for s in seeds], 'L1', 'diff')
    fam['H14'] = compare([s5_vec(s, 'sealed_answerloop_gradcuit_hard10_16x') for s in seeds], [s5_vec(s, 'sealed_answerloop_random_hard10_16x') for s in seeds], 'H14', 'bound', .02)
    fam['H16'] = compare([s5_vec(s, 'sealed_RH1_hard10_16x') for s in seeds], [s5_vec(s, 'sealed_RH1grounded_hard10_16x') for s in seeds], 'H16', 'noninf', .02)
    out['families']['S5'] = fam
    # S6: H12 (P3: LSL - always-open retained standard within +-5 points), TOST.
    q2s = [p for p in sorted(RUNS.glob('S6_s*_lsl_P3/measure.json'))]
    a = [read_json(p)['symbolic']['standard']['success'] for p in q2s]; b = [read_json(Path(str(p).replace('_lsl_P3', '_open_P3')))['symbolic']['standard']['success'] for p in q2s]
    out['families']['S6'] = dict(H12=compare(a, b, 'H12', 'equiv', .05) if q2s else dict(name='H12', status='missing'))
    # Holm within each stage family (difference tests need the predicted direction; margin claims are one-sided already).
    for stage, fam in out['families'].items():
        if not isinstance(fam, dict) or stage == 'S1':
            continue
        ps = {k: v['p'] for k, v in fam.items() if isinstance(v, dict) and 'p' in v}
        adj = holm(ps) if ps else {}
        for k, v in fam.items():
            if k in adj:
                v['p_holm'] = adj[k]
                if v['kind'] == 'diff':
                    v['supported'] = bool(adj[k] < .05 and v['pooled']['delta'] > 0)
                else:
                    v['supported'] = bool(adj[k] < .05)
    write_json(RUNS / 'FINAL_RESULTS.json', out)
    lines = ['# V9 final results (generated)', '', '| stage | hypothesis | kind | pooled delta [95% CI] | p (Holm) | supported |', '|---|---|---|---|---|---|']
    for stage, fam in out['families'].items():
        if not isinstance(fam, dict) or stage == 'S1':
            continue
        for k, v in fam.items():
            if 'pooled' in v:
                pl = v['pooled']; lines.append(f"| {stage} | {k} | {v['kind']} | {pl['delta']:+.3f} [{pl['interval'][0]:+.3f}, {pl['interval'][1]:+.3f}] | {v.get('p_holm', float('nan')):.4f} | {v.get('supported')} |")
            else:
                lines.append(f"| {stage} | {k} | — | {v.get('status', 'missing')} | — | — |")
    lines += ['', 'S1 (H3/H4/H5): see runs/S1_RESULTS.json.'] + [f'- {n}' for n in out['notes']]
    (RUNS / 'FINAL_REPORT.md').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
