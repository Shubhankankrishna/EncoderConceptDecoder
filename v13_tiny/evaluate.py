"""Evaluation for the V13 tiny test (Dataset_v13.md §7.4, §10). Architecture-neutral: everything takes a
`logits_fn(ids, doc) -> logits` on fixed (B, 1,024) rows, so tomorrow's baseline uses the same code.

Exact match by teacher forcing: an answer is exactly right under greedy decoding iff, with the correct prefix fed,
the argmax at every answer position (EOS included) is the target token; so for tasks with one correct output the
teacher-forced exact match *is* greedy single-output accuracy (§10.1). Plans (any valid plan reaching the goals within
the cap counts) and graph paths (any shortest valid path counts) have several correct outputs: they are decoded
greedily, by full re-forward, and checked by the simulator / verifier.

Panels are built once from the frozen generators (test split by hash, or dev) and cached under data/panels/.
"""
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

import data as D
import tasks as T

HERE = Path(__file__).resolve().parent
PANELS = HERE / 'data' / 'panels'


# ------------------------------------------------------------------------------------------------ panels
def _gen_cell(args):
    task, rung, n, mode, phase, want, seed = args
    rng = np.random.default_rng(D.seed_of('panel', seed, task, rung, mode, phase, want))
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 30:
        tries += 1
        e = T.generate(task, rung, rng, mode=mode, phase=phase, want=want)
        if e is None or e['key'] in seen: continue
        seen.add(e['key']); out.append(e)
    return (task, rung, mode, phase, want), out


def build_panel(name, cells, workers=3):
    """cells: list of (task, rung, n, mode, phase, want). Cached as data/panels/<name>.pkl."""
    PANELS.mkdir(parents=True, exist_ok=True)
    p = PANELS / f'{name}.pkl'
    if p.exists(): return pickle.loads(p.read_bytes())
    import multiprocessing as mp
    args = [c + (13,) for c in cells]
    t0 = time.time()
    with mp.get_context('spawn').Pool(workers) as pool:
        res = dict(pool.map(_gen_cell, args, chunksize=1))
    p.write_bytes(pickle.dumps(res))
    print(f'panel {name}: {sum(len(v) for v in res.values())} items in {time.time() - t0:.0f}s', flush=True)
    return res


def base_cells(n_test=128, n_dev=32):
    test, dev = [], []
    for task, (fam, fn, nr, ntr, gen) in T.TASKS.items():
        for r in range(nr):
            test.append((task, r, n_test, 'id', 'S0', 'test')); dev.append((task, r, n_dev, 'id', 'S0', 'dev'))
    ch = []
    for r in range(len(T.GRID_RUNGS)):                       # never-trained pairs and held-out values: no hash bucket
        ch.append(('grid2', r, 64, 'comp_test', 'S0', 'any'))
        ch.append(('grid1', r, 64, 'newparam', 'S0', 'any')); ch.append(('grid2', r, 64, 'newparam', 'S0', 'any'))
    for r in range(len(T.LIST_RUNGS)):
        ch.append(('list', r, 64, 'comp_test', 'S0', 'any'))
    return test, dev, ch


# phase -> tasks whose generator changes in that phase (the incoming data of §8.1)
PHASE_TASKS = {
    'S1': ['grid2', 'list', 'state', 'plan'],
    'S2': ['grid1', 'arith', 'plan'],
    'S3': ['grid1', 'grid2', 'arith', 'list', 'state', 'plan'],
    'S4': ['grid1', 'grid2', 'reg', 'state', 'plan'],
    'S5': ['grid2', 'arith', 'plan'],
    'S6': ['grid2', 'list', 'state', 'plan'],
}
CL_RUNGS = list(range(8))                                        # trained rungs only for the continual matrix


def cl_cells(n=24, want='test'):
    cells = []
    for ph, tl in PHASE_TASKS.items():
        for task in tl:
            for r in CL_RUNGS:
                cells.append((task, r, n, 'id', ph, want))
    for task in PHASE_TASKS['S3']:                               # S3 counterparts (retention side of the pairs)
        for r in CL_RUNGS: cells.append((task, r, n, 'id', 'S3c', want))
    for task in T.TASKS:                                         # S0 retention panel at the same size
        for r in CL_RUNGS: cells.append((task, r, n, 'id', 'S0', want))
    return cells


# ------------------------------------------------------------------------------------------------ packing and scoring
def pack(items, tok, B=8, T_=D.T_ROW):
    """Teacher-forced rows: returns batches [(ids, doc, [(item_idx, row, pos, t0, L)])]."""
    rows, metas = [], []
    rb = D.RowBuilder(tok, T_); cur = []
    for i, e in enumerate(items):
        ids, t0, _ = rb.encode(e)
        if len(ids) > T_: continue
        if not rb.fits(len(ids)):
            rows.append(rb.out()); metas.append(cur); rb.reset(); cur = []
        p = rb.pos; rb.add(ids, list(range(t0, len(ids))))
        cur.append((i, p, t0, len(ids)))
    if cur: rows.append(rb.out()); metas.append(cur)
    batches = []
    for s in range(0, len(rows), B):
        rr = rows[s:s + B]; mm = metas[s:s + B]
        while len(rr) < B:                                       # pad the batch to the fixed shape
            rb.reset(); rr.append(rb.out()); mm.append([])
        batches.append((np.stack([r['ids'] for r in rr]), np.stack([r['doc'] for r in rr]),
                        [(i, j, p, t0, L) for j, m in enumerate(mm) for (i, p, t0, L) in m]))
    return batches


@torch.no_grad()
def tf_exact(logits_fn, items, tok, dev, B=8):
    """Per item: exact (all answer tokens argmax-correct), and the answer NLL (nats per answer token)."""
    ok = np.zeros(len(items), bool); nll = np.full(len(items), np.nan)
    for ids, doc, meta in pack(items, tok, B):
        it = torch.from_numpy(ids).long().to(dev); dc = torch.from_numpy(doc).long().to(dev)
        lg = logits_fn(it, dc).float()
        pred = lg.argmax(-1).cpu().numpy()
        lp = torch.log_softmax(lg, -1)
        tgt = torch.from_numpy(np.concatenate([ids[:, 1:], ids[:, :1]], 1)).long().to(dev)
        tl = lp.gather(-1, tgt[..., None])[..., 0].cpu().numpy()
        for (i, r, p, t0, L) in meta:
            seg = slice(p + t0 - 1, p + L - 1)
            ok[i] = bool((pred[r, seg] == ids[r, p + t0:p + L]).all())
            nll[i] = -float(tl[r, seg].mean())
    return ok, nll


def gen_cap(e):
    m = e['meta']
    if m.get('qkind') == 'path': return 18
    if m['task'] == 'plan': return T.PLAN_RUNGS[m['rung']][1] + 6
    return len(e['target']) + 2


@torch.no_grad()
def greedy(logits_fn, items, tok, dev, B=8, T_=D.T_ROW):
    """Greedy decoding by full re-forward for items that need it; returns the generated token-id lists (EOS cut)."""
    slots = []
    for i, e in enumerate(items):
        pids, _ = tok.pieces(e['prompt'])
        L = len(pids) + gen_cap(e); slots.append((i, pids, ((L + 3) // 4) * 4))
    slots.sort(key=lambda s: gen_cap(items[s[0]]))                 # similar caps share a batch
    out = {}
    rows = []; cur = []; used = 0
    for s in slots:
        if used + s[2] > T_: rows.append(cur); cur = []; used = 0
        cur.append((s, used)); used += s[2]
    if cur: rows.append(cur)
    for b0 in range(0, len(rows), B):
        rr = rows[b0:b0 + B]
        ids = np.full((B, T_), tok.pad, np.int64); doc = np.zeros((B, T_), np.int64)
        state = []
        for r, row in enumerate(rr):
            for d, ((i, pids, Lp), p) in enumerate(row):
                ids[r, p:p + len(pids)] = pids; doc[r, p:p + Lp] = d + 1
                state.append([i, r, p + len(pids), p + Lp, []])
        cap = max(gen_cap(items[st[0]]) for st in state)
        it = torch.from_numpy(ids).to(dev); dc = torch.from_numpy(doc).to(dev)
        for _ in range(cap):
            act = [st for st in state if st[2] < st[3] and (not st[4] or st[4][-1] != tok.eos)]
            if not act: break
            lg = logits_fn(it, dc)
            rr_ = torch.tensor([st[1] for st in act], device=dev); pp = torch.tensor([st[2] - 1 for st in act], device=dev)
            nxt = lg[rr_, pp].argmax(-1).tolist()
            for st, tkn in zip(act, nxt):
                it[st[1], st[2]] = tkn; st[4].append(tkn); st[2] += 1
        for st in state:
            g = st[4]
            out[st[0]] = g[:g.index(tok.eos)] if tok.eos in g else g + [-1]   # -1: no EOS within the cap
    return [out.get(i) for i in range(len(items))]


def verify(e, gen_ids, tok):
    """Simulator / verifier for generated outputs. Returns (success, excess length or None)."""
    if gen_ids is None or (gen_ids and gen_ids[-1] == -1): return False, None
    toks = [tok.tk.id_to_token(i) for i in gen_ids]
    m = e['meta']
    if m['task'] == 'plan':
        mp = np.array(m['map'])
        ok = T.plan_simulate(mp, tuple(m['start']), [tuple(g) for g in m['goals']], toks, m['keydoor'], m['legend'])
        return ok, (len(toks) - m['optimal']) if ok else None
    if m.get('qkind') == 'path':
        edges = {frozenset(x) for x in m['edges']}
        path = [t_ for t_ in toks]
        ok = (len(path) == m['dist'] + 1 and path[0] == m['src'] and path[-1] == m['dst'] and
              all(frozenset((a, b)) in edges for a, b in zip(path, path[1:])))
        return ok, 0 if ok else None
    raise ValueError(m['task'])


def needs_gen(e):
    return e['meta']['task'] == 'plan' or e['meta'].get('qkind') == 'path'


def score_items(logits_fn, items, tok, dev, B=8):
    """Exact success per item (generation + verifier where needed), plus answer NLL from the teacher-forced pass."""
    ok, nll = tf_exact(logits_fn, items, tok, dev, B)
    gi = [i for i, e in enumerate(items) if needs_gen(e)]
    excess = {}
    if gi:
        gens = greedy(logits_fn, [items[i] for i in gi], tok, dev, B)
        for i, g in zip(gi, gens):
            s, x = verify(items[i], g, tok); ok[i] = s
            if x is not None: excess[i] = x
    return ok, nll, excess


# ------------------------------------------------------------------------------------------------ floors and ladders
def floor_of(task, items):
    """Floor c per rung (§7.4): the larger of chance and the registered shortcuts that can be computed exactly here:
    grid copy-input and all-background; graph reach chance 1/2 (balanced); state: the most frequent answer;
    the others have open answer spaces (chance ~ 0)."""
    if not items: return 0.0
    if task in ('grid1', 'grid2'):                                 # copy the query input; all-background output
        def qgrid(e):
            pr = e['prompt']; qi = len(pr) - 1 - pr[::-1].index('<in>'); return pr[qi + 1:-1]
        cp = np.mean([qgrid(e) == e['target'][:-1] for e in items])
        bg = np.mean([all(x == '<c0>' for x in e['target'][:-1] if x.startswith('<c') or x.startswith('<k'))
                      for e in items])
        return float(max(cp, bg))
    if task == 'graph':
        n = len(items); nr = sum(e['meta']['qkind'] == 'reach' for e in items)
        return 0.5 * nr / max(n, 1)
    if task == 'state':
        ans = [e['meta']['answer'] for e in items]
        vals, cnt = np.unique(ans, return_counts=True)
        return float(cnt.max() / len(ans))
    return 0.0


def above_floor(p, c):
    return (p - c) / (1 - c) if c < 1 else 0.0


def fit_l50(levels, acc, floors, n_boot=0, item_ok=None):
    """Fit p = c + (1 - c) sigmoid(beta (L50 - level)) by least squares on a grid; 'beyond'/'below' if no crossing."""
    lv = np.asarray(levels, float); a = np.asarray(acc, float); c = np.asarray(floors, float)
    mid = c + (1 - c) / 2
    if np.all(a >= mid): return 'beyond ladder'
    if np.all(a < mid): return 'below ladder'
    best = None
    for L50 in np.linspace(lv.min() - 1, lv.max() + 1, 241):
        for beta in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8):
            pr = c + (1 - c) / (1 + np.exp(-beta * (L50 - lv)))
            err = float(((pr - a) ** 2).sum())
            if best is None or err < best[0]: best = (err, L50, beta)
    return round(best[1], 2)


def ladder_summary(task, res_by_rung, ntr):
    """res_by_rung: {rung: (ok array, items)}. Returns accuracy and floor per rung, ID/extension areas, L50."""
    rungs = sorted(res_by_rung)
    acc = [float(np.mean(res_by_rung[r][0])) if len(res_by_rung[r][0]) else float('nan') for r in rungs]
    flo = [floor_of(task, res_by_rung[r][1]) for r in rungs]
    af = [above_floor(p, c) for p, c in zip(acc, flo)]
    idr = [i for i, r in enumerate(rungs) if r < ntr]; exr = [i for i, r in enumerate(rungs) if r >= ntr]
    return dict(acc=[round(x, 4) for x in acc], floor=[round(x, 4) for x in flo], n=[len(res_by_rung[r][0]) for r in rungs],
                id_area=round(float(np.mean([acc[i] for i in idr])), 4),
                id_area_above_floor=round(float(np.mean([af[i] for i in idr])), 4),
                ext_area=round(float(np.mean([acc[i] for i in exr])), 4) if exr else None,
                ext_area_above_floor=round(float(np.mean([af[i] for i in exr])), 4) if exr else None,
                L50=fit_l50(rungs, acc, flo))


def bootstrap_ci(ok_lists, n=1000, seed=0):
    """95% interval of a mean of per-rung means, resampling items within rungs."""
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        vals.append(np.mean([np.mean(o[rng.integers(0, len(o), len(o))]) for o in ok_lists if len(o)]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


@torch.no_grad()
def text_nll(logits_fn, stories, tok, dev, B=8):
    tot, n = 0.0, 0
    rb = D.RowBuilder(tok); rows = []
    for s in stories:
        if not rb.add(s, None, text=True):
            rows.append(rb.out()); rb.reset(); rb.add(s, None, text=True)
    rows.append(rb.out())
    for s0 in range(0, len(rows), B):
        rr = rows[s0:s0 + B]
        while len(rr) < B: rb.reset(); rr.append(rb.out())
        ids = torch.from_numpy(np.stack([r['ids'] for r in rr])).long().to(dev)
        doc = torch.from_numpy(np.stack([r['doc'] for r in rr])).long().to(dev)
        tgt = torch.from_numpy(np.stack([r['tgt'] for r in rr])).long().to(dev)
        w = torch.from_numpy(np.stack([r['w'] for r in rr])).to(dev)
        lp = torch.log_softmax(logits_fn(ids, doc).float(), -1).gather(-1, tgt[..., None])[..., 0]
        tot += float(-(lp * w).sum()); n += float(w.sum())
    return tot / max(n, 1)
