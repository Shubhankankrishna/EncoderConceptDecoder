"""V13 mechanism diagnostics on the final base checkpoint (kept out of every score, Dataset_v13.md §10.4).

    python3 diagnostics.py --ckpt runs/base/final.pt --out runs/eval/diag

1. §6.2's acceptance probe: an MLP probe on u_g against an MLP probe on its own input p_g, both predicting the next
   group's PQ codes (sg m_{g+1}); pre-registered threshold: u better by >= 5 points (V12 measured +7.7). Per family.
2. NCP head accuracy per family, and the codebook's usage perplexity per segment.
3. E2's loop at inference: R = 2 (trained) against R = 1 (exit after the first loop) on the dev panels. The loop was
   trained at R = 2 only, so this reads whether loop 2 is used, not what an R = 1 model would learn.
4. GDN-2 output RMS per M layer and loop (§12.7 asks for state norms; the chunkwise kernel exposes its output).
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data as D
import evaluate as E
import final_eval as FE
import ncp as N
import tasks as T
import tok as TK


@torch.no_grad()
def collect(m, cb, fam, n_rows, dev, seed=99):
    """(u_g, p_g, next codes) for valid groups of fresh rows of one family (dev-time rows, seed disjoint from training)."""
    U, P, Y = [], [], []
    for k in range(0, n_rows, 8):
        rows = [D.make_row(fam, 10 ** 7 + k + j, 1.0, seed) for j in range(8)] if fam != 'text' else None
        if rows is None: break
        b = D.collate(rows)
        ids = torch.from_numpy(b['ids']).long().to(dev); doc = torch.from_numpy(b['doc']).long().to(dev)
        out = m(ids, doc, return_all=True)
        mm, u = out['m'], out['u']
        p = m.wp(mm)
        gdoc = doc[:, ::m.c.k]
        valid, idx = N.ncp_targets(mm, gdoc, cb, update=False)
        U.append(u[:, :-1][valid].cpu()); P.append(p[:, :-1][valid].cpu()); Y.append(idx[valid].cpu())
    return torch.cat(U), torch.cat(P), torch.cat(Y)


def probe(X, Y, S, Nw, epochs=8, seed=0):
    torch.manual_seed(seed)
    n = len(X); perm = torch.randperm(n); tr, te = perm[:int(0.8 * n)], perm[int(0.8 * n):]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-5; Xn = (X - mu) / sd
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, S * Nw))
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=0.01)
    for ep in range(epochs):
        for i in range(0, len(tr), 512):
            bi = tr[i:i + 512]
            loss = F.cross_entropy(net(Xn[bi]).view(-1, Nw), Y[bi].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (net(Xn[te]).view(-1, S, Nw).argmax(-1) == Y[te]).float().mean()
    return float(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/base/final.pt'); ap.add_argument('--out', default='runs/eval/diag')
    ap.add_argument('--device', default='mps'); ap.add_argument('--rows', type=int, default=48)
    ap.add_argument('--quick', type=int, default=0, help='dev items per task for the loop test (plumbing test)')
    a = ap.parse_args()
    dev = torch.device(a.device); out = HERE / a.out; out.mkdir(parents=True, exist_ok=True)
    tk = TK.Tok(); m, cb, z = FE.load(HERE / a.ckpt, dev, tk)
    res = {}
    # 1-2: probes and NCP accuracy per family
    pr = {}
    for fam in ('grid', 'prog', 'state', 'plan'):
        U, P, Y = collect(m, cb, fam, a.rows, dev)
        with torch.no_grad():
            ncp_acc = float((m.ncp(U.to(dev)).view(len(U), m.c.pq_S, m.c.pq_N).argmax(-1).cpu() == Y).float().mean())
        au, ap_ = probe(U, Y, m.c.pq_S, m.c.pq_N), probe(P, Y, m.c.pq_S, m.c.pq_N)
        pr[fam] = dict(groups=len(U), probe_u=round(au, 4), probe_p=round(ap_, 4), u_minus_p=round(au - ap_, 4),
                       ncp_head=round(ncp_acc, 4), passes_5pt=bool(au - ap_ >= 0.05))
        print('probe', fam, pr[fam], flush=True)
    res['probes'] = pr
    p = cb.count / cb.count.sum(-1, keepdim=True)
    res['codebook_perplexity_per_segment'] = [round(float(x), 2) for x in torch.exp(-(p * p.clamp(min=1e-12).log()).sum(-1))]
    # 3: E2 loop at inference
    test, devc, ch = E.base_cells(); P_dev = E.build_panel('base_dev', devc)
    items = [e for its in P_dev.values() for e in its if e['meta']['rung'] < 8]
    if a.quick: items = [e for task in T.TASKS for e in [x for x in items if x['meta']['task'] == task][:a.quick]]
    loop = {}
    for R in (2, 1):
        m.c.R = R
        fn = FE.make_fn(m)
        acc = {}
        for task in T.TASKS:
            it = [e for e in items if e['meta']['task'] == task]
            ok, nll = E.tf_exact(fn, it, tk, dev)
            acc[task] = dict(exact=round(float(ok.mean()), 4), answer_nll=round(float(np.nanmean(nll)), 4))
        loop[f'R={R}'] = acc; print('loop', R, json.dumps(acc), flush=True)
    m.c.R = 2
    res['e2_loop_inference'] = loop
    # 4: GDN-2 output RMS per layer and loop
    rows = [D.make_row('prog', 10 ** 7 + j, 1.0, 99) for j in range(8)]
    b = D.collate(rows)
    ids = torch.from_numpy(b['ids']).long().to(dev); doc = torch.from_numpy(b['doc']).long().to(dev)
    with torch.no_grad():
        ctx = m.context(doc); ctx['log'] = []
        m.encode(ids, m.engram.keys(ids, doc), ctx)
    res['gdn2_output_rms'] = [round(float(x), 4) for x in ctx['log']]          # order: loop 1 M0..M2, loop 2 M0..M2
    (out / 'diagnostics.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
