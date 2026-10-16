"""Ablation (DESIGN_STEPS Step 6.2): XM codes on the VAE's switches.

A trained paper-faithful metacontroller's switch unit is frozen and its posterior-mean pass fixes the segments
(β > 0.5, first step always a switch). K = 8 learned controller codes c_1..c_K (in z-space) replace the Gaussian
posterior. Training is forward explorative modelling in the memory-saving mode (XM App. C):
  1. K no-grad forward passes, pass k assigning code k to every segment; per-segment action NLL ℓ_{s,k}
     (an approximation: other segments' codes in pass k are also k);
  2. per segment, the soft assignment r_{s,k} = softmax_k(−ℓ_{s,k} / τ) (τ → hard best-of-K as τ → 0);
  3. one gradient pass with z_s = Σ_k r_{s,k} c_k... (soft) — here hard: z_s = c_{argmin_k ℓ_{s,k}}, so only the
     best code per segment receives gradient (XM Eq. 1).
The hypernetwork is initialised from the VAE metacontroller and trained jointly with the codes.
Reported on dev_meta: action NLL with hindsight-best codes (the XM analogue of the posterior NLL), code usage, and
the same switch metrics (identical to the VAE's, since the switch unit is shared).

Rate: the code costs at most log K nats per segment (XM's bound), reported per action:
rate_logK = log K · segments / actions, so the point sits on the α sweep's rate–distortion plot.
Gate (written before any result): run only for a VAE α whose thresholded switches/episode is within [1, 4]
(true pair switches: 2) and pair F1 ≥ 0.3; otherwise thresholding gives ~1 segment and this degenerates to
best-of-K global controllers. Token-rate (TF) bases only.

  python -B meta_xm.py RUN TAG [STEPS]      (TAG = a trained meta/{TAG}.pt)
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import Split, Sampler
from evaluate import load_run
from meta import MetaController, decision_rows

ROOT = Path(__file__).resolve().parent


class XMCodes(nn.Module):
    def __init__(self, mc, K=8):
        super().__init__()
        self.mc = mc; self.K = K
        self.codes = nn.Parameter(torch.randn(K, mc.nz))

    @torch.no_grad()
    def segments(self, e, mask):
        """Fixed segmentation from the frozen VAE (posterior-mean pass). Returns segment id per unit (B, J)."""
        z, beta, _, _ = self.mc.sequence(e, mask, sample=False, beta_threshold=0.5)
        sw = (beta > 0.5) & mask; sw[:, 0] = True
        return torch.cumsum(sw.long(), 1) - 1


def controlled_logits(model, mc, b, z_units, rows, mask):
    """Apply U(z_units) at token-rate decision rows (and the following row), return logits."""
    def ctrl(e):
        B, N, D = e.shape
        A, Bm = mc.controllers(z_units.to(next(mc.parameters()).device))
        A, Bm = A.to(e.device), Bm.to(e.device)
        e2 = e.clone()
        for off in (0, 1):
            ok = mask & ((rows + off) <= N - 1); r = (rows + off).clamp(max=N - 1)
            er = torch.gather(e, 1, r[..., None].expand(-1, -1, D))
            delta = torch.einsum('bjdr,bjr->bjd', A, torch.einsum('bjrd,bjd->bjr', Bm, er)) * ok[..., None]
            e2 = e2.scatter_add(1, r[..., None].expand(-1, -1, D), delta)
        return e2
    return model(b['ids'], ctrl=ctrl)['logits']


def unit_nll(logits, b, rows, mask):
    lp = F.cross_entropy(logits.flatten(0, 1).float(), b['targets'].flatten(), reduction='none').view_as(b['weights'])
    return torch.gather(lp, 1, rows) * mask                                        # action NLL at each decision row


def step(model, xm, b, train=True):
    rows, mask = decision_rows(b)
    md = next(xm.mc.parameters()).device
    captured = {}
    def grab(e):
        captured['e'] = torch.gather(e, 1, rows[..., None].expand(-1, -1, e.shape[-1])); return e
    with torch.no_grad(): model(b['ids'], ctrl=grab)
    seg = xm.segments(captured['e'].to(md), mask.to(md)).to(rows.device)        # (B, J)
    S = int(seg.max()) + 1
    segnll = []
    with torch.no_grad():
        for k in range(xm.K):
            z = xm.codes[k].expand(*rows.shape, -1)
            u = unit_nll(controlled_logits(model, xm.mc, b, z, rows, mask), b, rows, mask)
            segnll.append(torch.zeros(rows.shape[0], S, device=u.device).scatter_add(1, seg, u))
    L = torch.stack(segnll, -1)                                                   # (B, S, K)
    best = L.argmin(-1)                                                           # hard best-of-K per segment
    z = xm.codes[torch.gather(best, 1, seg)]                                      # (B, J, nz)
    u = unit_nll(controlled_logits(model, xm.mc, b, z, rows, mask), b, rows, mask)
    nll = u.sum() / mask.sum()
    return nll, torch.gather(best, 1, seg)[mask], dict(segments=int((seg.max(1).values + 1).sum()), actions=int(mask.sum()))


def main(run, tag, steps=600, device='mps', B=64, lr=1e-3):
    model, arm = load_run(run, device)
    for p in model.parameters(): p.requires_grad_(False)
    if model.kind == 'ECD': raise SystemExit('token-rate control only in this ablation')
    cp = tag.split('_a')[0]
    if cp.startswith('L'): model.ctrl_layer = int(cp[1:])
    mc = MetaController(model.d); mc.load_state_dict(torch.load(ROOT / 'runs' / run / 'meta' / f'{tag}.pt'))
    for n, p in mc.named_parameters(): p.requires_grad_(n.startswith('hyp'))      # switch/encoder frozen
    torch.set_num_threads(2)
    xm = XMCodes(mc)
    opt = torch.optim.AdamW([xm.codes] + list(mc.hyp.parameters()), lr, weight_decay=0.0)
    meta, dm = Split('meta'), Split('dev_meta'); s = Sampler(meta, B, 1); t0 = time.time(); hist = []
    for it in range(steps):
        b = meta.batch(s.next_idx(), lam_obs=0.0, device=device)
        nll, best, _ = step(model, xm, b)
        opt.zero_grad(); nll.backward(); opt.step()
        if it % 50 == 0: hist.append(dict(it=it, nll=float(nll), usage=np.bincount(best.flatten().cpu().numpy(), minlength=xm.K).tolist()))
    with torch.no_grad():
        tot = cnt = 0; usage = np.zeros(xm.K); nseg = nact = 0
        for st in range(0, 1024, 128):
            b = dm.batch(np.arange(st, st + 128), lam_obs=0.0, device=device)
            nll, best, sc = step(model, xm, b, train=False); nseg += sc['segments']; nact += sc['actions']
            na = int(b['is_act'].sum()); tot += float(nll) * na; cnt += na; usage += np.bincount(best.flatten().cpu().numpy(), minlength=xm.K)
    res = dict(run=run, tag=tag, K=xm.K, steps=steps, dev_nll_hindsight_best=tot / cnt,
               rate_logK=float(np.log(xm.K)) * nseg / max(nact, 1), segments_per_ep=nseg / 1024, dev_code_usage=usage.tolist(), hist=hist, elapsed=time.time() - t0)
    (ROOT / 'runs' / run / 'meta' / f'xm_{tag}.json').write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != 'hist'}))


if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 600)
