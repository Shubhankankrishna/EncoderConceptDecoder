"""Declared concept-usage metric (Decision 2.1): dev NLL with and without the concept route, per family.
C counts as used on a family if zeroing it costs >= 0.02 nats/token (pinpad: per action) — the declared metric.
Also reported (advisors #13/#14): shuffle-ablation (u from another sequence in the batch: in-distribution
removal) and mean-ablation, u replaced by its per-family dev mean. Zeroing is a distribution
shift for D (u is RMS-normalised, never zero in training); mean-ablation removes the information while keeping
D's input in-distribution, so it bounds usage from below. Forward-only; model.py untouched (the final concept
norm `cn` is swapped for a module returning the mean).

  python -B usage.py RUN CKPT      -> runs/RUN/usage_CKPT.json
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from data import Split, BPEFamily
from evaluate import load_run, dev_nll, bpe_dev_nll
from losses import group_valid

ROOT = Path(__file__).resolve().parent


class MeanU(nn.Module):
    def __init__(self, mu): super().__init__(); self.mu = mu
    def forward(self, x): return self.mu.expand(*x.shape[:-1], -1)


class ShuffleU(nn.Module):
    """In-distribution removal (advisor #14): u of another sequence in the batch (a fixed derangement)."""
    def __init__(self, cn): super().__init__(); self.cn = cn
    def forward(self, x):
        y = self.cn(x)
        return torch.roll(y, 1, 0) if y.shape[0] > 1 else y


@torch.no_grad()
def mean_u(model, fam, device):
    tot, cnt = 0, 0
    if fam == 'pinpad':
        sp = Split('dev'); batches = (sp.batch(np.arange(s, s + 128), device=device, pad_to=176) for s in range(0, 1024, 128))
    else:
        f = BPEFamily(fam, 'dev'); bs = 32 if fam == 'text' else 128; n = 400 if fam == 'text' else min(f.n, 1024)
        batches = (f.batch(np.arange(s, min(s + bs, n)), None, device, pad_to=256 if fam == 'text' else 96) for s in range(0, n, bs))
    for b in batches:
        out = model(b['ids']); G = out['G']
        v = group_valid(b, G, model.k)
        tot = tot + out['u'][v].sum(0); cnt += int(v.sum())
    return tot / max(cnt, 1)


@torch.no_grad()
def main(run, ckpt, device='mps'):
    model, arm = load_run(run, device, ckpt); model.eval()
    dev = Split('dev'); res = dict(run=run, ckpt=ckpt)
    cn = model.cn
    def with_mean(fam, fn):
        model.cn = MeanU(mean_u(model, fam, device))
        try: return fn()
        finally: model.cn = cn
    def with_shuffle(fn):
        model.cn = ShuffleU(cn)
        try: return fn()
        finally: model.cn = cn
    on, off = dev_nll(model, dev, device), dev_nll(model, dev, device, concept_off=True)
    mean = with_mean('pinpad', lambda: dev_nll(model, dev, device))
    shuf = with_shuffle(lambda: dev_nll(model, dev, device))
    res['pinpad'] = dict(on=on, off=off, mean=mean, shuffle=shuf, delta_act=off['act'] - on['act'],
                         delta_mean_act=mean['act'] - on['act'], delta_shuffle_act=shuf['act'] - on['act'])
    for f in ('text', 'math', 'newmath'):
        a, b = bpe_dev_nll(model, f, device), bpe_dev_nll(model, f, device, concept_off=True)
        m = with_mean(f, lambda: bpe_dev_nll(model, f, device))
        s = with_shuffle(lambda: bpe_dev_nll(model, f, device))
        res[f] = dict(on=a, off=b, mean=m, shuffle=s, delta=b - a, delta_mean=m - a, delta_shuffle=s - a)
    res['used'] = {f: (res[f]['delta_act'] if f == 'pinpad' else res[f]['delta']) >= 0.02 for f in ('pinpad', 'text', 'math', 'newmath')}
    res['used_mean_ablation'] = {f: (res[f]['delta_mean_act'] if f == 'pinpad' else res[f]['delta_mean']) >= 0.02 for f in ('pinpad', 'text', 'math', 'newmath')}
    (ROOT / 'runs' / run / f'usage_{ckpt}.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
