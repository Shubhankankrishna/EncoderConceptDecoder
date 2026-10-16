"""Linear probes of ground-truth latents from frozen residual streams (paper Fig. 3 / A1).

For every depth, a multinomial logistic regression is fitted on episodes [0, n/2) of the dev split and scored on
[n/2, n). Rows: trajectory tokens (after p_0) whose ground truth is defined. Targets: the abstract action (pair
index, 4 classes) and the current target colour (8 classes). Chance = held-out majority-class rate.
"""
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def collect(model, split, device, n=1200, B=64, extra=None):
    feats, aa, tc = None, [], []
    for s in range(0, n, B):
        b = split.batch(np.arange(s, min(s + B, n)), device=device, pad_to=176)
        out = model(b['ids'])
        keep = (b['abs_act'] >= 0).cpu()                         # boolean indexing on MPS is very slow
        layers = list(out['resid']) + (extra(out) if extra else [])
        if feats is None: feats = [[] for _ in layers]
        for i, e in enumerate(layers): feats[i].append(e.float().cpu()[keep])
        aa.append(b['abs_act'].cpu()[keep]); tc.append(b['tgt_col'].cpu()[keep])
    return [torch.cat(f) for f in feats], torch.cat(aa), torch.cat(tc)


def fit_probe(X, y, Xt, yt, classes, steps=300, lr=0.05, wd=1e-4):
    mu, sd = X.mean(0), X.std(0) + 1e-5
    X, Xt = (X - mu) / sd, (Xt - mu) / sd
    W = torch.zeros(X.shape[1], classes, requires_grad=True); b = torch.zeros(classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr)
    for _ in range(steps):
        l = F.cross_entropy(X @ W + b, y) + wd * W.pow(2).sum()
        opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        return float(((Xt @ W + b).argmax(-1) == yt).float().mean())


def probe_all(model, split, device, n=1200, extra=None):
    was = model.training; model.eval()
    feats, aa, tc = collect(model, split, device, n, extra=extra)
    model.train(was)
    # split rows by episode half: rows are in episode order, so a row split at the halfway episode is clean
    half = collect_half_index(split, n)
    out = dict(abs_act=[], tgt_col=[])
    for X in feats:
        out['abs_act'].append(fit_probe(X[:half], aa[:half], X[half:], aa[half:], 4))
        out['tgt_col'].append(fit_probe(X[:half], tc[:half], X[half:], tc[half:], 8))
    out['chance_abs_act'] = float(np.bincount(aa[half:].numpy(), minlength=4).max() / (len(aa) - half))
    out['chance_tgt_col'] = float(np.bincount(tc[half:].numpy(), minlength=8).max() / (len(tc) - half))
    return out


def collect_half_index(split, n):
    cnt = 0
    for i in range(n // 2):
        _, _, ab, _ = split.episode(i); cnt += int((ab >= 0).sum())
    return cnt
