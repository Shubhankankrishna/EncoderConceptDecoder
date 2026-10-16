"""The continual matrix A[i, j] (Dataset_v13.md §10.2), evaluated retrospectively on the saved phase states.

    python3 cl_eval.py --cl runs/cl --out runs/eval/cl

Rows i: the base (before S1) and the state after each phase S1..S6, for each method (frozen, seq, replay, v13).
Columns j: each phase's incoming task panels (the new skills), the S3 counterparts (the retention side of the
confusable pairs), the S0 retention panels of all eight task types, and text NLL. Panels: 24 test items per trained
rung (0-7). Teacher-forced exact match everywhere, including planning (exact match to the canonical shortest plan, a
lower bound on simulator success; the final base test uses the simulator).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch

import cl as CL
import data as D
import evaluate as E
import model as M
import tok as TK

PHASES = CL.PHASES
STAGE = {'S0': 0, 'S1': 1, 'S2': 2, 'S3': 3, 'S3c': 0, 'S4': 4, 'S5': 5, 'S6': 6}


def columns(panel):
    cols = {}
    for (task, rung, mode, phase, want), items in panel.items():
        cols.setdefault(f'{phase}/{task}', []).extend(items)
    return dict(sorted(cols.items()))


@torch.no_grad()
def eval_state(model, cols, stories, tk, dev):
    fn = lambda ids, doc: model(ids, doc)[0]
    row, oks = {}, {}
    for c, items in cols.items():
        ok, _ = E.tf_exact(fn, items, tk, dev)
        row[c] = round(float(ok.mean()), 4); oks[c] = ok
    row['text_nll'] = round(E.text_nll(fn, stories, tk, dev), 4)
    return row, oks


def paired_ci(okA, okB, cols, n=1000, seed=0):
    """95% interval of mean_c mean(okA[c]) - mean_c mean(okB[c]); items resampled within columns, the same draw for
    both methods (every method is scored on the same items)."""
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        d = []
        for c in cols:
            a, b = okA[c], okB[c]; idx = rng.integers(0, len(a), len(a))
            d.append(a[idx].mean() - b[idx].mean())
        vals.append(np.mean(d))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def comparisons(OK, cols):
    learned = [c for c in cols if not c.startswith('S3c')]; s0 = [c for c in cols if c.startswith('S0/')]
    comp = {}
    for a_, b_ in (('v14', 'seq'), ('v14', 'replay'), ('replay', 'seq'), ('v13', 'seq'), ('v13', 'replay')):
        if all(m in OK and 'S6' in OK[m] for m in (a_, b_)):
            fa = lambda m, st, cc: float(np.mean([OK[m][st][c].mean() for c in cc]))
            comp[f'{a_}-{b_}'] = dict(
                final_average=round(fa(a_, 'S6', learned) - fa(b_, 'S6', learned), 4),
                final_average_ci=paired_ci(OK[a_]['S6'], OK[b_]['S6'], learned),
                s0_before_return=round(fa(a_, 'S5', s0) - fa(b_, 'S5', s0), 4),
                s0_before_return_ci=paired_ci(OK[a_]['S5'], OK[b_]['S5'], s0))
    return comp


def metrics(A, cols):
    """A: {state: {col: acc}} with states base, S1..S6. s(j) from the column's phase tag."""
    states = ['base'] + PHASES
    T_ = states[-1]
    learned = [c for c in cols if not c.startswith('S3c')]
    out = {}
    out['final_average'] = round(float(np.mean([A[T_][c] for c in learned])), 4)
    acq = {}
    for i, ph in enumerate(PHASES):
        new = [c for c in cols if c.startswith(ph + '/')]
        prev = states[i]
        acq[ph] = dict(before=round(float(np.mean([A[prev][c] for c in new])), 4),
                       after=round(float(np.mean([A[ph][c] for c in new])), 4))
        acq[ph]['gain'] = round(acq[ph]['after'] - acq[ph]['before'], 4)
    out['acquisition'] = acq
    bwt, fgt = [], []
    for c in learned:
        s = STAGE[c.split('/')[0]]
        if s >= 6: continue
        st = states[s]
        bwt.append(A[T_][c] - A[st][c])
        fgt.append(max(A[k][c] for k in states[s:]) - A[T_][c])
    out['backward_transfer'] = round(float(np.mean(bwt)), 4)
    out['forgetting'] = round(float(np.mean(fgt)), 4)
    s0 = [c for c in cols if c.startswith('S0/')]
    out['s0_retention'] = {st: round(float(np.mean([A[st][c] for c in s0])), 4) for st in states}
    out['retention_before_return'] = out['s0_retention']['S5']
    s3c = [c for c in cols if c.startswith('S3c/')]
    out['s3_counterparts'] = {st: round(float(np.mean([A[st][c] for c in s3c])), 4) for st in states}
    out['text_nll'] = {st: A[st]['text_nll'] for st in states}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cl', default='runs/cl'); ap.add_argument('--base', default='runs/base/final.pt')
    ap.add_argument('--out', default='runs/eval/cl'); ap.add_argument('--device', default='mps')
    ap.add_argument('--methods', default='seq,replay,v13'); ap.add_argument('--quick', type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(a.device); out = HERE / a.out; out.mkdir(parents=True, exist_ok=True)
    tk = TK.Tok()
    cols = columns(E.build_panel('cl_test', E.cl_cells(24, 'test')))
    if a.quick: cols = {k: v[:a.quick] for k, v in cols.items()}
    stories = D.val_stories(tk, 128 if not a.quick else 8)
    z = torch.load(HERE / a.base, map_location='cpu', weights_only=False)
    def fresh():
        m = M.V13(M.Cfg(V=tk.vocab)).to(dev); m.load_state_dict(z['model']); m.eval(); return m
    res = {}; t0 = time.time(); OK = {}
    base_row, base_ok = eval_state(fresh(), cols, stories, tk, dev)
    print('base', json.dumps(base_row), flush=True)
    res['frozen'] = {'A': {st: base_row for st in ['base'] + PHASES}}
    for meth in a.methods.split(','):
        A = {'base': base_row}; OK[meth] = {'base': base_ok}
        for ph in PHASES:
            p = HERE / a.cl / meth / f'after_{ph}.pt'
            if not p.exists(): print('missing', p); break
            st = torch.load(p, map_location='cpu', weights_only=False)
            m = fresh(); CL.load_adapter_state(m, st['adapters'])
            A[ph], OK[meth][ph] = eval_state(m, cols, stories, tk, dev); del m
            print(meth, ph, json.dumps(A[ph]), f'{(time.time() - t0) / 60:.1f} min', flush=True)
        res[meth] = {'A': A}
        if len(A) == 7: res[meth]['metrics'] = metrics(A, cols)
        (out / 'cl_results.json').write_text(json.dumps(res, indent=1))
    res['frozen']['metrics'] = metrics(res['frozen']['A'], cols)
    res['paired'] = comparisons(OK, cols)
    (out / 'cl_results.json').write_text(json.dumps(res, indent=1))
    torch.save({m: {st: {c: v.tolist() for c, v in d.items()} for st, d in s.items()} for m, s in OK.items()},
               out / 'cl_items.pt')
    print('paired differences', json.dumps(res['paired']), flush=True)
    for meth in res:
        if 'metrics' in res[meth]:
            mm = res[meth]['metrics']
            print(f'{meth:7s} final avg {mm["final_average"]:.3f}  BWT {mm["backward_transfer"]:+.3f}  forgetting '
                  f'{mm["forgetting"]:.3f}  S0 before return {mm["retention_before_return"]:.3f}  acquisition '
                  + ' '.join(f'{k}:{v["after"]:.2f}({v["gain"]:+.2f})' for k, v in mm['acquisition'].items()), flush=True)


if __name__ == '__main__':
    main()
