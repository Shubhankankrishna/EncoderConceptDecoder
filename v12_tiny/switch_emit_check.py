"""Scaling check (§7 #5): with binarised β (inference / internal RL), z is piecewise constant, so the hypernetwork
needs to run only at switch events and U can be cached per segment. Verifies exact equivalence to the per-row
path on a real batch and reports the materialised-controller memory of both.

  python -B switch_emit_check.py RUN
"""
import sys
import numpy as np
import torch
from data import Split
from evaluate import load_run
from meta import MetaController, decision_rows


@torch.no_grad()
def main(run, device='cpu'):
    torch.manual_seed(0)
    model, _ = load_run(run, device)
    mc = MetaController(model.d)
    for p in mc.hyp[-1].parameters(): p.normal_(0, 0.02)                         # non-trivial controllers
    b = Split('dev_meta').batch(np.arange(64), lam_obs=0.0, device=device)
    rows, mask = decision_rows(b)
    grab = {}
    def cap(x):
        grab['e'] = torch.gather(x, 1, rows[..., None].expand(-1, -1, x.shape[-1])); return x
    model(b['ids'], ctrl=cap)
    e = grab['e']
    z, beta, _, _ = mc.sequence(e, mask, sample=False, beta_threshold=0.5)
    # per-row path (as in meta.Controlled)
    A, Bm = mc.controllers(z)
    per_row = torch.einsum('bjdr,bjr->bjd', A, torch.einsum('bjrd,bjd->bjr', Bm, e))
    # switch-emitted path: one hypernetwork call per segment
    sw = (beta > 0.5); sw[:, 0] = True
    seg = torch.cumsum(sw.long(), 1) - 1                                          # (B, J) segment id
    zs = z[sw]                                                                    # (n_segments, nz): z at each switch
    As, Bs = mc.controllers(zs)
    off = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), sw.sum(1)[:-1]]), 0)
    gid = off[:, None] + seg                                                      # global segment index per row
    emitted = torch.einsum('bjdr,bjr->bjd', As[gid], torch.einsum('bjrd,bjd->bjr', Bs[gid], e))
    err = float((per_row - emitted).abs().max())
    d, r = model.d, mc.r
    print(dict(max_abs_diff=err, rows=int(mask.sum()), segments=int(sw[mask].sum()),
               per_row_floats=int(mask.sum()) * 2 * d * r, emitted_floats=int(sw[mask].sum()) * 2 * d * r,
               ratio=round(int(mask.sum()) / max(int(sw[mask].sum()), 1), 1)))
    print('note: the gather As[gid] above re-materialises per row for the check only; a kernel indexes the cache.')


if __name__ == '__main__':
    main(sys.argv[1])
