"""Collect every V14-tiny result file into REPORT_TABLES.md (numbers only; the narrative lives in REPORT.md).

    python3 make_report.py
"""
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = Path(os.environ.get('V14_RUNS', HERE / 'runs'))                # overridable for plumbing tests
EV = Path(os.environ.get('V14_EVAL', R / 'eval'))
CLD = Path(os.environ.get('V14_CL', R / 'cl'))
FAMS = ('grid', 'prog', 'state', 'plan', 'text')
V13_REF = dict(launch_tok_s=3234, average_tok_s=2471)               # V13-tiny REPORT §4


def jl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def j(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def fmt(x, n=3):
    if x is None: return '—'
    if isinstance(x, float) and np.isnan(x): return 'nan'
    return f'{x:.{n}f}' if isinstance(x, (float, int)) and not isinstance(x, bool) else str(x)


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def training(L, cfg, log):
    last = log[-1]
    tok_s = [r['tok_s'] for r in log if r['step'] > 60]
    s_step = [r['s_step'] for r in log if r['step'] > 60]
    wait = [r['wait_s'] for r in log if r['step'] > 60]
    avg = last['tokens'] / max(last['h'] * 3600, 1e-9)
    L.append(f"Base run: {last['step']:,} steps, {last['tokens']:,} processed tokens, {last['h']:.2f} h of training time; "
             f"{cfg.get('params', 0):,} trained parameters (twin {cfg.get('twin_params', 0):,}); activation checkpointing "
             f"{cfg.get('ckpt')}; skipped updates {last.get('skipped', 0)}.\n")
    L.append('## Throughput\n')
    L.append('| read | V14-tiny | V13-tiny |')
    L.append('|---|---|---|')
    L.append(f"| median tokens/s over 50-step intervals (after step 60) | {int(np.median(tok_s)) if tok_s else '—'} | "
             f"{V13_REF['launch_tok_s']} (launch benchmark) |")
    L.append(f"| average tokens/s (tokens / training time) | {int(avg):,} | {V13_REF['average_tok_s']} |")
    L.append(f"| median s/step | {fmt(np.median(s_step) if s_step else None, 3)} | 2.53 (launch benchmark) |")
    L.append(f"| median wait for rows, s/step | {fmt(np.median(wait) if wait else None, 3)} | — |")
    L.append(f"| peak MPS driver memory, GB | {fmt(max(r['mem_gb'] for r in log), 2)} | 5.4 (checkpointing on) |")
    pct = [10, 50, 90]
    if tok_s: L.append(f"\nTokens/s percentiles {pct}: {[int(np.percentile(tok_s, p)) for p in pct]}.")
    ch = j(R / 'base' / 'ckpt_choice.json')
    if ch: L.append(f"\nCheckpointing rule (DEVIATIONS item 7): {json.dumps(ch)}\n")
    L.append('\n## Training curve (the last logged values of each 10% window)\n')
    L.append('| window | ' + ' | '.join(['grid', 'prog', 'state', 'plan', 'text', 'NCP CE', 'h1 head a/b', 'h1 copy a/b',
                                         'same4', 'code ppl (EMA)', 'resets', 'lam', 'tok/s']) + ' |')
    L.append('|' + '---|' * 14)
    for k in range(10):
        win = [r for r in log if k / 10 <= r['frac'] < (k + 1) / 10]
        if not win: continue
        row = []
        for f in FAMS:
            v = [r['ntp'] for r in win if r['fam'] == f]
            row.append(fmt(np.mean(v[-5:]) if v else None))
        tail = win[-20:]
        ha = mean([r['hor'][0][1] for r in tail]); hb = mean([r['hor'][0][2] for r in tail])
        ca = mean([r['hor'][0][3] for r in tail]); cb_ = mean([r['hor'][0][4] for r in tail])
        row += [fmt(mean([r['ncp'] for r in tail]), 2), f'{fmt(ha, 2)}/{fmt(hb, 2)}', f'{fmt(ca, 2)}/{fmt(cb_, 2)}',
                fmt(mean([r['same4'] for r in tail]), 2), fmt(mean([np.mean(r['ppl_ema']) for r in tail]), 1),
                fmt(tail[-1]['resets'], 0), fmt(tail[-1]['lam'], 3), str(int(np.median([r['tok_s'] for r in win])))]
        L.append(f'| {k * 10}-{k * 10 + 10}% | ' + ' | '.join(row) + ' |')
    L.append('')
    caps = [r for r in log if 'cap' in r]
    if caps:
        L.append('Gradient-share cap readings (||grad_C lam L_NCP|| / ||grad_C L_NTP||; G-side norms ncp / ntp / sig):\n')
        for r in caps[::max(1, len(caps) // 12)] + ([caps[-1]] if len(caps) > 12 else []):
            c = r['cap']; ratio = c['ncp'][0] / max(c['ntp'][0], 1e-12)
            L.append(f"- step {r['step']}: ratio {ratio:.3f}; lam scale {r['lam_scale']}; G {c['ncp'][1]:.3g} / "
                     f"{c['ntp'][1]:.3g} / {c['sig'][1]:.3g}")
        L.append('')


def base_test(L, ev):
    L.append(f"## Base test (step {ev.get('step')}, {ev.get('tokens'):,} tokens)\n")
    L.append(f"**Reasoning = {ev['reasoning']:.4f}** (mean over 4 families of 0.75 x ID area + 0.25 x challenge area, "
             f"both above floor). Text NLL {ev['text_nll']:.4f} nats/token.\n")
    L.append('| family | ID area (above floor) | challenge area (above floor) | score |')
    L.append('|---|---|---|---|')
    for f, v in ev['family_scores'].items():
        L.append(f"| {f} | {v['id_area']:.4f} | {v['challenge_area']:.4f} | {v['score']:.4f} |")
    L.append('')
    L.append('| task | exact success per rung (0-7 trained, 8-9 test-only) | floors | ID area [95% CI] | ext area | L50 |')
    L.append('|---|---|---|---|---|---|')
    for t, s in ev['ladders'].items():
        L.append(f"| {t} | {' '.join(f'{x:.2f}' for x in s['acc'])} | {' '.join(f'{x:.2f}' for x in s['floor'])} | "
                 f"{s['id_area']:.3f} {s.get('ci_id')} | {fmt(s['ext_area'])} | {s['L50']} |")
    L.append('\n### Challenge panels\n')
    L.append('| panel | per rung | floors | trained rungs | above floor (all rungs) [CI] |')
    L.append('|---|---|---|---|---|')
    for k, v in ev['challenge'].items():
        L.append(f"| {k} | {' '.join(f'{x:.2f}' for x in v['acc'])} | {' '.join(f'{x:.2f}' for x in v['floor'])} | "
                 f"{v['trained_rungs']:.3f} | {v['above_floor']:.3f} {v['ci']} |")
    L.append(f"\nTag subscores: {ev.get('tag_subscores')}\n")
    if ev.get('interventions'):
        L.append('### Concept-path interventions (dev panels, teacher-forced exact; text NLL)\n')
        iv = ev['interventions']; tasks = [t for t in iv['normal'] if t != 'text_nll']
        L.append('| read | ' + ' | '.join(tasks) + ' | text NLL |')
        L.append('|' + '---|' * (len(tasks) + 2))
        for nm, v in iv.items():
            L.append(f'| {nm} | ' + ' | '.join(f"{v[t]['exact']:.3f}" for t in tasks) + f" | {v['text_nll']:.3f} |")
        L.append('')
    if ev.get('newparam_curve'):
        L.append('### New-parameter curve over the 10% snapshots (trained rungs, teacher-forced exact)\n')
        L.append('| % | step | grid1 new-param | grid1 ID (dev) | grid2 new-param | grid2 ID (dev) |')
        L.append('|---|---|---|---|---|---|')
        for r in ev['newparam_curve']:
            L.append(f"| {r['pct']} | {r['step']} | {r['grid1_newparam']:.3f} | {r['grid1_id_dev']:.3f} | "
                     f"{r['grid2_newparam']:.3f} | {r['grid2_id_dev']:.3f} |")
        L.append('')


def diagnostics(L, dg):
    L.append('## Pre-registered readings V1-V5 (diagnostics_v14.py)\n')
    Vd = dg.get('V', {})
    for name in ('V1', 'V2', 'V3', 'V4', 'V5'):
        if name in Vd: L.append(f"- **{name}**: {json.dumps(Vd[name])}")
        else: L.append(f'- **{name}**: not computed')
    L.append(f"- **§6.1 guard** (twin-versus-online agreement, final, pooled): {json.dumps(dg.get('guard'))}\n")
    for ck, R_ in dg.get('per_ckpt', {}).items():
        L.append(f"### {ck} (step {R_.get('step')}; T_glob gain {R_.get('t_glob_gain')}; codebook frozen {R_.get('frozen')}; "
                 f"agreement a/b {R_['agreement_pooled']['a']}/{R_['agreement_pooled']['b']})\n")
        L.append('| family | h1 head a / b (cc pairs) | copy a / b | marginal a / b | probe u a / b | probe p a / b | '
                 'probe L a / b | chi twin a / b | chi online a / b | R2 own / pooled | xcorr | eff rank twin a / b | ppl |')
        L.append('|' + '---|' * 13)
        for f, r in R_['fam'].items():
            h1 = r['head']['h1']; pr = r.get('probes', {})
            pp = lambda s: f"{fmt(pr[s]['a'])} / {fmt(pr[s]['b'])}" if s in pr else '—'
            L.append(f"| {f} | {fmt(h1['head_a_cc'])} / {fmt(h1['head_b_cc'])} | {fmt(h1['copy_a'])} / {fmt(h1['copy_b'])} | "
                     f"{fmt(h1['marg_a'])} / {fmt(h1['marg_b'])} | {pp('u')} | {pp('p')} | {pp('lrow')} | "
                     f"{fmt(r['chi']['twin_norm_a'])} / {fmt(r['chi']['twin_norm_b'])} | "
                     f"{fmt(r['chi']['online_a'])} / {fmt(r['chi']['online_b'])} | "
                     f"{fmt(r['residue_r2']['own'])} / {fmt(r['residue_r2']['pooled'])} | {fmt(r['xcorr_ab'])} | "
                     f"{fmt(r['eff_rank']['twin_a'], 1)} / {fmt(r['eff_rank']['twin_b'], 1)} | {r['usage_perplexity']} |")
        L.append('')
        heavy = {f: r for f, r in R_['fam'].items() if 'action_sensitivity' in r}
        if heavy:
            L.append('| family | action sensitivity a / b (control max) | b below gate / below a tenth of a | '
                     'cross-pair head / sampled / targets | local share of head signal h1-h4 | same4 |')
            L.append('|---|---|---|---|---|---|')
            for f, r in heavy.items():
                s_ = r['action_sensitivity']; cp = r.get('cross_pair', {})
                L.append(f"| {f} | {fmt(s_.get('ratio_a'), 4)} / {fmt(s_.get('ratio_b'), 4)} ({fmt(s_.get('control_max'), 6)}) | "
                         f"{s_.get('b_below_gate')} / {s_.get('b_below_tenth_of_a')} | {cp.get('head_argmax')} / "
                         f"{cp.get('head_sampled')} / {cp.get('targets')} | {r.get('signal_share_local')} | "
                         f"{r.get('same_argmax_all_horizons')} |")
            L.append('')
    v5 = dg.get('V5_reassignment')
    if v5: L.append(f"Code reassignment (V5): {json.dumps(v5)}\n")


def continual(L, cl):
    L.append('## Continual learning (S1-S6, equal incoming rows, A[i, j] teacher-forced; V13-tiny\'s rows and items)\n')
    L.append('| method | final average | backward transfer | forgetting | S0 before return (S5) | S0 after S6 | ' +
             ' | '.join(f'acq {p}' for p in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']) + ' |')
    L.append('|' + '---|' * 12)
    for meth, v in cl.items():
        if not isinstance(v, dict) or 'metrics' not in v: continue
        mm = v['metrics']
        L.append(f"| {meth} | {mm['final_average']:.3f} | {mm['backward_transfer']:+.3f} | {mm['forgetting']:.3f} | "
                 f"{mm['retention_before_return']:.3f} | {mm['s0_retention']['S6']:.3f} | " +
                 ' | '.join(f"{a['after']:.2f} ({a['gain']:+.2f})" for a in mm['acquisition'].values()) + ' |')
    L.append('')
    if cl.get('paired'):
        L.append('Paired differences (same items for every method; 95% bootstrap over items within columns):\n')
        L.append('| pair | final average | [CI] | S0 retention before return (S5) | [CI] |')
        L.append('|---|---|---|---|---|')
        for k, v in cl['paired'].items():
            L.append(f"| {k} | {v['final_average']:+.4f} | {v['final_average_ci']} | {v['s0_before_return']:+.4f} | "
                     f"{v['s0_before_return_ci']} |")
        L.append('')
    L.append('S0 retention by state, S3 counterparts and text NLL:\n')
    for meth, v in cl.items():
        if not isinstance(v, dict) or 'metrics' not in v: continue
        mm = v['metrics']
        L.append(f"- {meth}: S0 {mm['s0_retention']}; S3 counterparts {mm['s3_counterparts']}; text NLL {mm['text_nll']}")
    L.append('')


def main():
    L = []
    cfg = j(R / 'base' / 'config.json') or {}
    # the log is appended across restarts (DEVIATIONS item 17): keep the last record of each step
    log = sorted({d['step']: d for d in jl(R / 'base' / 'log.jsonl')}.values(), key=lambda d: d['step'])
    L.append('# V14-tiny: result tables (generated by make_report.py)\n')
    if log: training(L, cfg, log)
    ev = j(EV / 'base' / 'results.json')
    if ev: base_test(L, ev)
    dg = j(EV / 'diag' / 'diagnostics_v14.json')
    if dg: diagnostics(L, dg)
    era = j(R / 'base' / 'era.json')
    if era: L.append(f"## Era (§6.1 guard and refit)\n\n{json.dumps(era)}\n")
    ch = j(CLD / 'choice.json')
    if ch: L.append(f"## Continual default (§9.4, cl_choice.py)\n\n{json.dumps(ch)}\n")
    cl = j(EV / 'cl' / 'cl_results.json')
    if cl: continual(L, cl)
    for meth in ('seq', 'v14', 'replay'):
        rows = jl(CLD / meth / 'log.jsonl')
        if rows:
            L.append(f'### {meth}: phase log\n')
            for r in rows: L.append('- ' + json.dumps(r))
            L.append('')
    (Path(os.environ.get('V14_TABLES', HERE / 'REPORT_TABLES.md'))).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
