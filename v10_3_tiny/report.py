"""Render every numeric table of the tiny V10.3 report from the saved JSON.

  python -B -m v10_3_tiny.report --runs C0_s1 A_s1 X8_s1 X4_s1 [C0_s2 X8_s2] --sleep sleep_X8_s1 --out v10_3_tiny/REPORT_TABLES.md

No number in the report is typed by hand; prose is written around this file.
"""
import argparse, json, statistics as st
from pathlib import Path

RUNS = Path(__file__).resolve().parent / 'runs'
FAMS = ('text', 'math', 'newmath', 'sudoku_easy', 'sudoku_hard')
PRE = ('text', 'math', 'sudoku_easy', 'sudoku_hard')


def f(x, d=3):
    return '—' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))


def ci(p, d=3):
    return f"{p['delta']:+.{d}f} [{p['ci95'][0]:+.{d}f}, {p['ci95'][1]:+.{d}f}]"


def table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join('---' if i == 0 else '---:' for i in range(len(head))) + '|']
    return '\n'.join(out + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def load(name):
    d = RUNS / name
    rows = [json.loads(l) for l in (d / 'train.jsonl').open()]
    return dict(manifest=json.loads((d / 'manifest.json').read_text()), train=rows,
                quick=json.loads((d / 'quick_eval.json').read_text()),
                ev=json.loads((d / 'evaluation.json').read_text()) if (d / 'evaluation.json').exists() else None)


def training(runs):
    s = ['## Training', table(['Run', 'Arm', 'Device', 'Dense', 'Steps', 'Input tokens', 'Active min', 'Median s/step', 'Final total loss (last 100)'],
        [[n, r['manifest']['arm'], r['manifest']['args']['device'], f"{r['manifest']['parameters']['dense']:,}", r['manifest'].get('steps'),
          f"{r['manifest'].get('tokens', 0):,}", f(r['manifest'].get('active_seconds', 0) / 60, 1),
          f(st.median(x['step_seconds'] for x in r['train']), 3), f(st.mean(x['total'] for x in r['train'][-100:]))] for n, r in runs.items()])]
    steps = sorted({e['step'] for r in runs.values() for e in r['quick']})
    s.append('\nQuick dev text NLL during training (8×4 text windows; the step grid differs per run because evaluation is wall-clock driven):')
    s.append(table(['Run', 'step: text NLL', 'final math', 'final sudoku easy', 'final sudoku hard'],
                   [[n, '; '.join(f"{e['step']}: {e['nll']['text']:.3f}" for e in r['quick']), f(r['quick'][-1]['nll']['math']),
                     f(r['quick'][-1]['nll']['sudoku_easy']), f(r['quick'][-1]['nll']['sudoku_hard'])] for n, r in runs.items()]))
    s.append('\nConcept-path scales (from train.jsonl):')
    rows = []
    for n, r in runs.items():
        sc = [x for x in r['train'] if 'codebook_rms' in x or 'fusion_rms' in x or 'fusion_map_rms' in x]
        if not sc: continue
        a, b = sc[0], sc[-1]
        rows.append([n, f(a.get('codebook_rms')), f(b.get('codebook_rms')), f(a.get('fusion_rms', a.get('fusion_map_rms')), 4),
                     f(b.get('fusion_rms', b.get('fusion_map_rms')), 4), f(a.get('code_projection_rms')), f(b.get('code_projection_rms'))])
    s.append(table(['Run', 'Codebook RMS step 1', 'final', 'Forecast gain/map RMS step 1', 'final', 'Code projection RMS step 1', 'final'], rows))
    cand = []
    for n, r in runs.items():
        c = [x for x in r['train'] if x.get('candidate') and 'mixture' in x]
        a = [x for x in r['train'] if x.get('candidate') and 'explore' in x]
        if c:
            last = c[-200:]
            cand.append([n, 'mixture', f(st.mean(x['mixture'] for x in last)), f(st.mean(x['responsibility_entropy'] for x in last)),
                         f(st.mean(x['prior_entropy'] for x in last)), f(st.mean(x['spread'] for x in last)), f(st.mean(x['gap'] for x in last), 4)])
        elif a:
            last = a[-200:]
            cand.append([n, 'hard argmin (V10.1)', f(st.mean(x['explore'] for x in last)), '—', '—', '—', '—'])
    bins = [(1, 1000), (1001, 2000), (2001, 3000), (3001, 4000), (4001, 5000), (5001, 6000)]
    rows = []
    for n, r in runs.items():
        keys = [k for k in ('future', 'latent', 'chunk', 'mixture', 'explore') if any(k in x for x in r['train'])]
        for k in keys:
            # mixture/explore are logged as 0 on non-candidate updates; average candidate updates only.
            only_candidates = k in ('mixture', 'explore')
            for fam in PRE:
                vals = []
                for lo, hi in bins:
                    v = [x[k] for x in r['train'] if lo <= x['step'] <= hi and x['family'] == fam and k in x
                         and (x.get('candidate') or not only_candidates)]
                    vals.append(f(st.mean(v)) if v else '—')
                rows.append([n, k, fam] + vals)
    if rows:
        s.append('\nConcept-prediction losses by family and step bin (every update; mixture/explore on candidate updates only). The training stream is effectively fresh data: under one epoch per family.')
        s.append(table(['Run', 'Loss', 'Family'] + [f'{lo}-{hi}' for lo, hi in bins], rows))
    if cand:
        s.append('\nExploration objective, mean over the last 200 candidate updates:')
        s.append(table(['Run', 'Objective', 'Loss', 'Responsibility entropy', 'Prior entropy', 'Spread', 'Gap'], cand))
    return '\n\n'.join(s)


def results(runs, comp):
    ev = {n: r['ev'] for n, r in runs.items() if r['ev']}
    s = ['## Dev results', table(['Run'] + list(FAMS), [[n] + [f(e['nll'][x]['summary']['nll']) for x in FAMS] for n, e in ev.items()])]
    mods = sorted(next(iter(ev.values()))['nll']['math']['summary']['by_module'])
    s.append('\nMath NLL by module:')
    s.append(table(['Run'] + mods, [[n] + [f(e['nll']['math']['summary']['by_module'][m]['nll']) for m in mods] for n, e in ev.items()]))
    s.append('\nGreedy exact match (correct / attempted):')
    s.append(table(['Run'] + list(next(iter(ev.values()))['generation']), [[n] + [f"{g['correct']}/{g['n']}" for g in e['generation'].values()] for n, e in ev.items()]))
    s.append('\nPaired document differences (later − earlier run, nats/token, 95% document bootstrap):')
    keys = list(comp['pairs']); fams = list(next(iter(comp['pairs'].values())))
    s.append(table(['Pair'] + fams, [[k] + [ci(comp['pairs'][k][x]) for x in fams] for k in keys]))
    s.append('\nSudoku grid-shift control (NLL increase when the board moves 4 tokens):')
    s.append(table(['Run', 'easy', 'hard'], [[n, f(e['grid_shift']['sudoku_easy']['degradation']), f(e['grid_shift']['sudoku_hard']['degradation'])]
                                             for n, e in ev.items() if 'grid_shift' in e]))
    return '\n\n'.join(s)


def mechanisms(runs, comp):
    ev = {n: r['ev'] for n, r in runs.items() if r['ev'] and r['ev']['arm'] != 'C0'}
    s = ['## Concept-path diagnostics']
    for n, e in ev.items():
        inter = comp['interventions'][n]
        names = list(inter['text'])
        s.append(f'\nInterventions, {n} (Δ NLL vs intact, 95% CI):')
        s.append(table(['Intervention'] + list(PRE), [[k] + [ci(inter[x][k]) for x in PRE] for k in names]))
    s.append('\nForecast quality, FVU of next-group pooled state (lower is better; 1 = calibration mean):')
    rows = []
    for n, e in ev.items():
        for x in PRE:
            v = e['fvu'][x]; q = v['fvu']
            rows.append([n, x, v['pairs'], f(q['calibration_mean']), f(q['forecast']), f(q['forecast_affine']), f(q['ridge_forecast']),
                         f(q['ridge_state']), f(q['ridge_input']), f(v['forecast_variance_ratio']), f(v['decoder_forecast_input_variance_ratio'], 4),
                         f(v.get('decoder_forecast_input_to_e3_variance_ratio'), 4)])
    s.append(table(['Run', 'Family', 'Pairs', 'Mean', 'F', 'affine(F)', 'ridge(F)', 'ridge(U)', 'ridge(P3)', 'var F / var Y', 'var decoder F / var Y', 'var decoder F / var E3'], rows))
    s.append('\nExploration on dev boundaries (utility = each arm\'s own training utility):')
    rows = []
    for n, e in ev.items():
        for x, v in e['exploration'].items():
            ll = v['loglik_per_token']
            rows.append([n, x, v['boundaries'], f(v['spread_median']), f(v['gap_median'], 4), f(v['regret_mean']), f(v['code_main_effect_share']),
                         f(v['selector_accuracy']), f(v['selector_winner_ce']), f(v['winner_entropy']), f(v['prior_entropy']),
                         f(ll['mixture_learned_prior']), f(ll['mixture_uniform']), f(ll['committed']), f(ll['best_in_hindsight'])])
    s.append(table(['Run', 'Family', 'Boundaries', 'Spread', 'Gap', 'Regret', 'Code main-effect share', 'Selector acc', 'Selector CE(winner)', 'H(winners)',
                    'Prior entropy', 'LL mixture', 'LL uniform mix', 'LL committed', 'LL best'], rows))
    s.append('\nWinner and committed code counts:')
    s.append(table(['Run', 'Family', 'Winners', 'Committed'], [[n, x, v['winner_counts'], v['committed_counts']] for n, e in ev.items() for x, v in e['exploration'].items()]))
    return '\n\n'.join(s)


def verdicts(comp):
    v = comp.get('verdicts') or {}
    if not v: return '## Verdicts\n\nNot computed (missing seed-1 arm).'
    g1, g2, g3 = v['G1'], v['G2'], v['G3']
    rows = [['G1 text (best X − A)', ci(g1['text_delta_x_minus_a']), g1['passed']],
            ['G1 math exact (best X − A)', f"{g1['math_exact_x_minus_a']['b_correct']} vs {g1['math_exact_x_minus_a']['a_correct']}; " + ci(g1['math_exact_x_minus_a']), ''],
            ['G2 text (best X − C0), margin 0.02', ci(g2['text_delta_x_minus_c0']), g2['passed']],
            ['G3 N2 var(decoder F)/var(Y) ≥ 0.25', f(g3['N2_decoder_forecast_variance_ratio'], 4), g3['N2_passed']],
            ['G3 N3 codebook RMS within 20% of step 1', f"{f(g3['N3_codebook_rms'])} vs {f(g3['N3_step1_rms'])}", g3['N3_passed']]]
    for fam, x in g3['N5'].items():
        rows.append([f'G3 N5 {fam}: selector CE < H(winners); LL mixture > committed',
                     f"{f(x['selector_ce'])} vs {f(x['winner_entropy'])}; {f(x['mixture'])} vs {f(x['committed'])}", ''])
    rows += [['G3 overall', '', g3['passed']], ['Promote to sub-10M', '', v['promote_to_10M']]]
    return '## Verdicts (best X = ' + v['best_x'] + ')\n\n' + table(['Gate', 'Value', 'Passed'], rows)


def sleep(name):
    d = json.loads((RUNS / name / 'sleep.json').read_text()); c = d['conditions']
    s = [f"## Sleep tests ({d['arm']}, checkpoint step {d['checkpoint_step']})"]
    fit = d.get('gate_fit') or {}
    if fit:
        s.append('Gate fit (train-split rows; open rate of the fitted gate on its own fitting rows):')
        s.append(table(['Site', 'Φpos rows', 'Φneg rows', 'Open on newmath', 'Open on pretraining', 'torch = sklearn'],
                       [[k, v['pos_rows'], v['neg_rows'], f(v['fit_open_pos']), f(v['fit_open_neg']), f(v['torch_sklearn_agreement'], 4)] for k, v in sorted(fit.items())]))
    mods = sorted(c['base']['nll']['math']['summary']['by_module'])
    s.append('\nDev NLL after adaptation:')
    s.append(table(['Condition', 'text'] + mods + ['newmath', 'sudoku_easy', 'sudoku_hard'],
                   [[n, f(r['nll']['text']['summary']['nll'])] + [f(r['nll']['math']['summary']['by_module'][m]) for m in mods] +
                    [f(r['nll'][x]['summary']['nll']) for x in ('newmath', 'sudoku_easy', 'sudoku_hard')] for n, r in c.items()]))
    s.append('\nGreedy exact match:')
    s.append(table(['Condition'] + mods + ['newmath'],
                   [[n] + [f"{r['generation']['math'][m]['correct']}/{r['generation']['math'][m]['n']}" for m in mods] +
                    [f"{r['generation']['newmath']['algebra__linear_1d']['correct']}/{r['generation']['newmath']['algebra__linear_1d']['n']}"] for n, r in c.items()]))
    cs = d.get('contrasts') or {}
    if cs:
        keys = ['text'] + mods + ['newmath', 'sudoku_easy', 'sudoku_hard']
        s.append('\nPaired contrasts (nats/token, 95% CI):')
        s.append(table(['Contrast'] + keys, [[k] + [ci(v[x]) if x in v else '—' for x in keys] for k, v in cs.items()]))
    for n in ('gated', 'consolidated'):
        if n in c and c[n].get('gate_open_rates'):
            r = c[n]['gate_open_rates']; labels = sorted({l for v in r.values() for l in v})
            s.append(f'\nGate open rate on dev ({n}; "@answer" rows are answer tokens only):')
            s.append(table(['Site'] + labels, [[k] + [f(v.get(l)) for l in labels] for k, v in sorted(r.items())]))
    cal = d.get('bank_calibration')
    if cal:
        s.append(f"\nRBF calibration on train-split documents (base newmath {f(cal['base_new'])}, base text+math {f(cal['base_old'])}; constraint met: {cal['constraint_met']}):")
        s.append(table(['σ', 'η', 'Δ newmath', 'Δ text+math', 'Value RMS', 'Median effective η'],
                       [[g['sigma'], g['eta'], f(g['new_delta'], 4), f(g['old_delta'], 4), f(g['value_rms'], 5), f(g['effective_eta_median'], 3)] for g in cal['grid']]))
        if 'rbf' in c: s.append('\nChosen bank: ' + json.dumps(c['rbf'].get('bank_summary')))
    if d.get('G4'):
        g = d['G4']
        s.append('\nG4 checks (gated vs plain): ' + json.dumps(g['checks']) + f"\n\nG4 passed: {g['passed']}")
    logs = {n: r['train_log'] for n, r in c.items() if r.get('train_log')}
    if logs:
        s.append('\nAdaptation loss (first → last logged step): ' + '; '.join(f"{n} {f(l[0]['loss'])} → {f(l[-1]['loss'])}" for n, l in logs.items()))
    return '\n\n'.join(s)


def audit():
    d = json.loads((Path(__file__).resolve().parent / 'data_audit.json').read_text())
    rows = []
    for fam in FAMS:
        for split in ('train', 'dev'):
            r = d[fam][split]
            rows.append([fam, split, r['records'], r.get('answer_start_mod8', f"{r.get('fraction_longer_than_window', 0):.0%} cropped at random offset"),
                         f(r.get('answer_start_mod8_chi2_df7'), 1), f(r['frequent_token_worst'])])
    return '## Data audit\n\n' + table(['Family', 'Split', 'Records', 'Answer start mod 8', 'χ² (df 7)', 'Worst token residue share'], rows)


def main(a):
    global RUNS
    RUNS = Path(a.root)
    runs = {n: load(n) for n in a.runs}
    comp = json.loads(Path(a.comparison).read_text())
    parts = [training(runs), results(runs, comp), mechanisms(runs, comp), verdicts(comp)]
    if a.sleep and (RUNS / a.sleep / 'sleep.json').exists(): parts.append(sleep(a.sleep))
    parts.append(audit())
    Path(a.out).write_text('\n\n'.join(parts) + '\n'); print('wrote', a.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--runs', nargs='+', required=True); p.add_argument('--comparison', required=True)
    p.add_argument('--sleep'); p.add_argument('--out', required=True); p.add_argument('--root', default=str(RUNS))
    main(p.parse_args())
