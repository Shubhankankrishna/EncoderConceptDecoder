"""Paired comparisons between evaluated tiny arms, and the predeclared verdicts.

  python -B -m v10_3_tiny.compare --runs v10_3_tiny/runs/C0_s1 v10_3_tiny/runs/A_s1 ... --output v10_3_tiny/comparison.json

Every arm was scored on identical documents (evaluate.panels), so per-document
NLL differences are paired. Intervals are 95% document bootstraps of the
token-weighted difference (resampling documents, 4,000 draws). G1/G2/G3 follow
ARCHITECTURE.md exactly; nothing here is tuned after reading results.
"""
import argparse, json
from pathlib import Path
import numpy as np

BOOT = 4000


def paired(a, b, rng):
    """Token-weighted NLL difference b - a over shared documents, with a bootstrap interval."""
    ka = {(r['index'], r['source']): r for r in a}; kb = {(r['index'], r['source']): r for r in b}
    keys = sorted(set(ka) & set(kb))
    n = np.array([ka[k]['n'] for k in keys], float)
    d = np.array([kb[k]['ce'] - ka[k]['ce'] for k in keys])
    assert all(ka[k]['n'] == kb[k]['n'] for k in keys), 'panels differ'
    idx = rng.integers(0, len(keys), (BOOT, len(keys)))
    boot = d[idx].sum(1) / n[idx].sum(1)
    return dict(documents=len(keys), delta=float(d.sum() / n.sum()), ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))])


def intervention_ci(rows, name, rng):
    n = np.array([r['n'] for r in rows], float); d = np.array([r[name] - r['ce'] for r in rows])
    idx = rng.integers(0, len(rows), (BOOT, len(rows))); boot = d[idx].sum(1) / n[idx].sum(1)
    return dict(delta=float(d.sum() / n.sum()), ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))])


def paired_correct(a_rows, b_rows, rng):
    """Accuracy difference b - a on identical problems, with a 95% problem bootstrap."""
    ka = {(r['index'], r['source']): r['correct'] for r in a_rows}; kb = {(r['index'], r['source']): r['correct'] for r in b_rows}
    keys = sorted(set(ka) & set(kb)); assert len(keys) == len(ka) == len(kb), 'generation panels differ'
    d = np.array([float(kb[k]) - float(ka[k]) for k in keys])
    boot = d[rng.integers(0, len(d), (BOOT, len(d)))].mean(1)
    return dict(problems=len(d), a_correct=int(sum(ka.values())), b_correct=int(sum(kb.values())), delta=float(d.mean()),
                ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))])


def math_accuracy(ev, fam='math', exclude=()):
    rows = [r for r in ev['generation'][fam]['rows'] if r['source'] not in exclude]
    return sum(r['correct'] for r in rows), len(rows)


def main(a):
    rng = np.random.default_rng(0)
    evs = {}
    for run in a.runs:
        ev = json.loads((Path(run) / 'evaluation.json').read_text()); evs[Path(run).name] = ev
    out = dict(runs=list(evs), nll={}, pairs={}, interventions={}, verdicts={})
    for name, ev in evs.items():
        out['nll'][name] = {f: v['summary']['nll'] for f, v in ev['nll'].items()}
        if ev['arm'] != 'C0':
            out['interventions'][name] = {f: {k: intervention_ci(v['documents'], k, rng) for k in v['summary']['delta']}
                                          for f, v in ev['nll'].items()}
    names = list(evs)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            entry = {f: paired(evs[x]['nll'][f]['documents'], evs[y]['nll'][f]['documents'], rng)
                     for f in evs[x]['nll'] if f in evs[y]['nll']}
            for m in sorted({r['source'] for r in evs[x]['nll']['math']['documents']}):
                entry[m] = paired([r for r in evs[x]['nll']['math']['documents'] if r['source'] == m],
                                  [r for r in evs[y]['nll']['math']['documents'] if r['source'] == m], rng)
            out['pairs'][f'{y} - {x}'] = entry
    by_arm = {}
    for name, ev in evs.items(): by_arm.setdefault(ev['arm'], []).append(name)
    seed1 = {arm: [n for n in runs if n.endswith('_s1')][0] for arm, runs in by_arm.items() if any(n.endswith('_s1') for n in runs)}
    if {'A', 'X8', 'X4', 'C0'} <= set(seed1):
        xs = {arm: out['nll'][seed1[arm]]['text'] for arm in ('X8', 'X4')}
        best = min(xs, key=xs.get)
        g1 = paired(evs[seed1['A']]['nll']['text']['documents'], evs[seed1[best]]['nll']['text']['documents'], rng)
        acc = paired_correct(evs[seed1['A']]['generation']['math']['rows'], evs[seed1[best]]['generation']['math']['rows'], rng)
        g2 = paired(evs[seed1['C0']]['nll']['text']['documents'], evs[seed1[best]]['nll']['text']['documents'], rng)
        ev = evs[seed1[best]]
        f = ev['fvu']['text']; e = ev['exploration']
        n2 = f['decoder_forecast_input_variance_ratio']
        n3 = ev['scales']['codebook_rms']
        first = json.loads((Path(a.runs[names.index(seed1[best])]) / 'train.jsonl').open().readline())
        n3_ref = first['codebook_rms']
        n5 = {fam: dict(selector_ce=v['selector_winner_ce'], winner_entropy=v['winner_entropy'],
                        mixture=v['loglik_per_token']['mixture_learned_prior'], committed=v['loglik_per_token']['committed'])
              for fam, v in e.items() if fam in ('text', 'math', 'sudoku_easy', 'sudoku_hard')}
        n5_newmath_reported_not_scored = e.get('newmath')
        out['verdicts'] = dict(
            best_x=best,
            G1=dict(text_delta_x_minus_a=g1, math_exact_x_minus_a=acc,
                    passed=bool(g1['ci95'][1] < 0 and not acc['ci95'][1] < 0)),
            G2=dict(text_delta_x_minus_c0=g2, margin=0.02, passed=bool(g2['delta'] <= 0.02)),
            G3=dict(N2_decoder_forecast_variance_ratio=n2, N2_passed=bool(n2 >= .25),
                    N3_codebook_rms=n3, N3_step1_rms=n3_ref, N3_passed=bool(abs(n3 - n3_ref) <= .2 * n3_ref),
                    N5=n5, N5_newmath_reported_not_scored=n5_newmath_reported_not_scored, N5_passed=bool(all(v['selector_ce'] < v['winner_entropy'] and v['mixture'] > v['committed'] for v in n5.values()))),
        )
        gs = {}
        for arm in ('C0', 'best'):
            name = seed1['C0'] if arm == 'C0' else seed1[best]
            gs[arm] = {f: v['degradation'] for f, v in evs[name].get('grid_shift', {}).items()}
        out['verdicts']['grid_shift'] = dict(c0=gs['C0'], best_x=gs['best'],
                                             best_x_minus_c0={f: gs['best'][f] - gs['C0'][f] for f in gs['C0'] if f in gs['best']})
        out['verdicts']['G3']['passed'] = all(out['verdicts']['G3'][k] for k in ('N2_passed', 'N3_passed', 'N5_passed'))
        out['verdicts']['promote_to_10M'] = all(out['verdicts'][g]['passed'] for g in ('G1', 'G2', 'G3'))
    Path(a.output).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['nll'], indent=1)); print(json.dumps(out.get('verdicts', {}), indent=1)[:3000])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--runs', nargs='+', required=True); p.add_argument('--output', required=True)
    main(p.parse_args())
