"""Pre-registered gradient-share check (Decision 3.5): per family, ||d(lam*L_aux)/d theta|| / ||dL_NTP/d theta||
on the encoder (emb excluded: shared with the head) and on the concept module. Acceptance: aux share on E <= 1.

  python -B grad_share.py RUN CKPT     -> runs/RUN/grad_share_CKPT.json
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from data import Mixed
from evaluate import load_run
from losses import ntp, ncp_ce, nextlat_loss
import train

ROOT = Path(__file__).resolve().parent


def norm(gs):
    gs = [g for g in gs if g is not None]
    return float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs))) if gs else 0.0    # 0: loss does not reach the group


def main(run, ckpt, device='mps', per_family=3):
    model, arm = load_run(run, device, ckpt); model.train()
    a = train.ARMS[arm]
    groups = dict(E=[p for blk in getattr(model, 'enc', []) for p in blk.parameters()],
                  C=[p for blk in getattr(model, 'cblocks', []) for p in blk.parameters()],
                  D=[p for blk in getattr(model, 'dec', getattr(model, 'blocks', [])) for p in blk.parameters()])
    stream = Mixed(123); res = {}
    seen = {}
    while min(seen.get(f, 0) for f in ('pinpad', 'text', 'math')) < per_family:
        b, _ = stream.next(device); f = b['family']
        if seen.get(f, 0) >= per_family: continue
        seen[f] = seen.get(f, 0) + 1
        out = model(b['ids'])
        terms = dict(ntp=ntp(out, b)[0])
        if a.get('ncp'): terms['ncp'] = train.LAM['ncp'] * ncp_ce(model, out, b, model.codebook, model.aux['ncp_head'], a['ncp'])[0]
        if a.get('nextlat'):
            lh, lk = nextlat_loss(model, out, b, model.aux['nl']); terms['nextlat'] = lh + lk
        for g, ps in groups.items():
            if not ps: continue
            n = {k: norm(torch.autograd.grad(v, ps, retain_graph=True, allow_unused=True)) for k, v in terms.items()}
            for k in n:
                if k == 'ntp': continue
                res.setdefault(f, {}).setdefault(f'{k}/ntp_{g}', []).append(n[k] / max(n['ntp'], 1e-12))
    res = {f: {k: float(np.mean(v)) for k, v in d.items()} for f, d in res.items()}
    (ROOT / 'runs' / run / f'grad_share_{ckpt}.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
