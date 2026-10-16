"""Final evaluation of the base checkpoint (Dataset_v13.md §7.4, §10.1, §10.4).

    python3 final_eval.py --ckpt runs/base/final.pt --out runs/eval/base

Reports, per task type, exact success per ladder rung with floors, ID and extension ladder areas (raw and above
floor), frontier L50; the composition and new-parameter panels; the frozen primary score
  Reasoning = mean over the 4 families [0.75 x ID ladder area + 0.25 x challenge area] (both above floor);
the §5.7 tag subscores; text NLL on held-out stories; the new-parameter checkpoint curve (snapshots at every 10%);
and the concept-path interventions of §10.4 (diagnostics, never part of the score).
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

import data as D
import evaluate as E
import model as M
import ncp as N
import tasks as T
import tok as TK

FAMILY_OF = {t: v[0] for t, v in T.TASKS.items()}


def load(ckpt, dev, tk):
    m = M.V13(M.Cfg(V=tk.vocab)).to(dev)
    z = torch.load(ckpt, map_location=dev, weights_only=False)
    m.load_state_dict(z['model']); m.eval()
    cb = N.PQCodebook(m.c.d, m.c.pq_S, m.c.pq_N).to(dev); cb.load_state_dict(z['codebook'])
    return m, cb, z


def make_fn(m, hook=None, compile_=False):
    f = lambda ids, doc: m(ids, doc, u_hook=hook)[0]
    return torch.compile(f, dynamic=False) if compile_ else f


def ladders(fn, panel, tk, dev, gen=True):
    by_task = {}
    for (task, rung, mode, phase, want), items in panel.items():
        by_task.setdefault(task, {})[rung] = items
    res = {}; raw = {}
    for task in T.TASKS:
        if task not in by_task: continue
        rr = {}
        for rung, items in sorted(by_task[task].items()):
            if gen: ok, nll, exc = E.score_items(fn, items, tk, dev)
            else: (ok, nll), exc = E.tf_exact(fn, items, tk, dev), {}
            rr[rung] = (ok, items)
            raw[(task, rung)] = dict(ok=ok.tolist(), nll=np.nan_to_num(nll).tolist(),
                                     excess=[int(v) for v in exc.values()])
        s = E.ladder_summary(task, rr, T.TASKS[task][3])
        s['ci_id'] = E.bootstrap_ci([rr[r][0] for r in sorted(rr) if r < T.TASKS[task][3]])
        if task == 'plan':
            ex = [v for r in rr for v in raw[(task, r)]['excess']]
            s['excess_len_mean'] = float(np.mean(ex)) if ex else None
        if task == 'graph':                                        # by query kind
            for qk in ('reach', 'dist', 'path'):
                s[f'acc_{qk}'] = [round(float(np.mean([o for o, e in zip(rr[r][0], rr[r][1]) if e['meta']['qkind'] == qk]
                                                      or [np.nan])), 3) for r in sorted(rr)]
        if task == 'state':
            for qk in ('loc', 'owner', 'count'):
                s[f'acc_{qk}'] = [round(float(np.mean([o for o, e in zip(rr[r][0], rr[r][1]) if e['meta']['qkind'] == qk]
                                                      or [np.nan])), 3) for r in sorted(rr)]
        res[task] = s
        print(f'{task:6s} ID {s["id_area"]:.3f} (above floor {s["id_area_above_floor"]:.3f}) ext {s["ext_area"]} '
              f'L50 {s["L50"]} acc {s["acc"]}', flush=True)
    return res, raw


def panel_acc(fn, panel, tk, dev, select):
    out = {}
    for (task, rung, mode, phase, want), items in panel.items():
        if not select(task, mode): continue
        ok, _ = E.tf_exact(fn, items, tk, dev)
        out.setdefault((task, mode), {})[rung] = (ok, items)
    summ = {}
    for (task, mode), rr in out.items():
        acc = [float(np.mean(rr[r][0])) for r in sorted(rr)]
        flo = [E.floor_of(task, rr[r][1]) for r in sorted(rr)]
        ntr = T.TASKS[task][3]
        summ[f'{task}/{mode}'] = dict(acc=[round(x, 4) for x in acc], floor=[round(x, 4) for x in flo],
                                      trained_rungs=round(float(np.mean(acc[:ntr])), 4),
                                      above_floor=round(float(np.mean([E.above_floor(p, c) for p, c in zip(acc, flo)])), 4),
                                      ci=E.bootstrap_ci([rr[r][0] for r in sorted(rr)]))
    return summ


def reasoning_score(lad, ch):
    fam = {}
    for f, tasks in T.FAMILY_TASKS.items():
        idv = [lad[t]['id_area_above_floor'] for t in tasks if t in lad]
        chv = [lad[t]['ext_area_above_floor'] for t in tasks if t in lad and lad[t]['ext_area_above_floor'] is not None]
        if f == 'grid':
            chv += [ch[k]['above_floor'] for k in ('grid2/comp_test', 'grid1/newparam', 'grid2/newparam') if k in ch]
        if f == 'prog' and 'list/comp_test' in ch: chv.append(ch['list/comp_test']['above_floor'])
        fid, fch = float(np.mean(idv)), float(np.mean(chv)) if chv else 0.0
        fam[f] = dict(id_area=round(fid, 4), challenge_area=round(fch, 4), score=round(0.75 * fid + 0.25 * fch, 4))
    return round(float(np.mean([v['score'] for v in fam.values()])), 4), fam


@torch.no_grad()
def interventions(m, cb, panel_items, stories, tk, dev):
    """§10.4: zero / batch-mean / shuffled concept read, teacher-forced exact and text NLL."""
    def zero(u, mm, ctx): return torch.zeros_like(u)
    def mean(u, mm, ctx):
        keep = (ctx['gdoc'] > 0)[..., None].to(u.dtype)
        mu = (u * keep).sum((0, 1)) / keep.sum().clamp(min=1)
        return mu.expand_as(u)
    def shuffle(u, mm, ctx):
        return u.roll(1, 0)                                         # another row's concepts (other documents)
    out = {}
    for name, h in (('normal', None), ('zero', zero), ('mean', mean), ('shuffle', shuffle)):
        fn = make_fn(m, h)
        acc = {}
        for task in T.TASKS:
            items = [e for e in panel_items if e['meta']['task'] == task]
            if not items: continue
            ok, nll = E.tf_exact(fn, items, tk, dev)
            acc[task] = dict(exact=round(float(ok.mean()), 4), answer_nll=round(float(np.nanmean(nll)), 4))
        acc['text_nll'] = round(E.text_nll(fn, stories, tk, dev), 4)
        out[name] = acc
        print('intervention', name, json.dumps(acc), flush=True)
    return out


def newparam_curve(run, panel_np, panel_dev, tk, dev):
    curve = []
    for k in range(10, 101, 10):
        p = run / f'snap_{k:03d}.pt'
        if not p.exists(): continue
        m, cb, z = load(p, dev, tk); fn = make_fn(m)
        rec = dict(pct=k, step=z['step'], tokens=z['tokens'])
        for task in ('grid1', 'grid2'):
            items = [e for (t, r, mode, ph, w), its in panel_np.items() if t == task and r < 8 for e in its]
            ok, _ = E.tf_exact(fn, items, tk, dev); rec[f'{task}_newparam'] = round(float(ok.mean()), 4)
            items = [e for (t, r, mode, ph, w), its in panel_dev.items() if t == task and r < 8 for e in its]
            ok, _ = E.tf_exact(fn, items, tk, dev); rec[f'{task}_id_dev'] = round(float(ok.mean()), 4)
        curve.append(rec); print('newparam curve', rec, flush=True)
        del m
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/base/final.pt'); ap.add_argument('--out', default='runs/eval/base')
    ap.add_argument('--device', default='mps'); ap.add_argument('--no-curve', action='store_true')
    ap.add_argument('--quick', type=int, default=0, help='keep this many items per panel cell (plumbing test)')
    a = ap.parse_args()
    dev = torch.device(a.device); out = HERE / a.out; out.mkdir(parents=True, exist_ok=True)
    tk = TK.Tok(); t0 = time.time()
    m, cb, z = load(HERE / a.ckpt, dev, tk)
    test, devc, ch = E.base_cells()
    P_test = E.build_panel('base_test', test); P_dev = E.build_panel('base_dev', devc); P_ch = E.build_panel('challenge', ch)
    if a.quick:
        P_test, P_dev, P_ch = ({k: v[:a.quick] for k, v in P.items()} for P in (P_test, P_dev, P_ch))
    fn = make_fn(m)
    res = dict(ckpt=a.ckpt, step=z.get('step'), tokens=z.get('tokens'), elapsed_h=round(z.get('elapsed', 0) / 3600, 3))
    lad, raw = ladders(fn, P_test, tk, dev, gen=True)
    res['ladders'] = lad
    chs = panel_acc(fn, P_ch, tk, dev, lambda t, mode: True)
    res['challenge'] = chs
    print('challenge', json.dumps(chs), flush=True)
    res['reasoning'], res['family_scores'] = reasoning_score(lad, chs)
    tags = {}
    for t, (fav, cc) in T.TAGS.items():
        if t in lad: tags.setdefault(fav, []).append(lad[t]['id_area'])
    res['tag_subscores'] = {k: round(float(np.mean(v)), 4) for k, v in tags.items()}
    stories = D.val_stories(tk, 512 if not a.quick else 16)
    res['text_nll'] = round(E.text_nll(fn, stories, tk, dev), 4)
    print('REASONING', res['reasoning'], json.dumps(res['family_scores']), 'text NLL', res['text_nll'], flush=True)
    dev_items = [e for its in P_dev.values() for e in its if e['meta']['rung'] < 8]
    res['interventions'] = interventions(m, cb, dev_items, stories[:256], tk, dev)
    (out / 'results.json').write_text(json.dumps(res, indent=1, default=str))
    torch.save(raw, out / 'raw_items.pt')
    if not a.no_curve:
        P_np = {k: v for k, v in P_ch.items() if k[2] == 'newparam'}
        res['newparam_curve'] = newparam_curve(HERE / Path(a.ckpt).parent, P_np, P_dev, tk, dev)
        (out / 'results.json').write_text(json.dumps(res, indent=1, default=str))
    print(f'final eval done in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
