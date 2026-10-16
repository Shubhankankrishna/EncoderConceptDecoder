"""Pilot (advisor #1): MPS throughput of the plain base at several widths, then a short training run with
linear probes of the ground-truth subgoal by depth. Decides the world size and the phase budget.

  python -B pilot.py bench            -> pilot_bench.json
  python -B pilot.py train D MIN       -> runs/pilot_TF_dD/ (train.jsonl, final.pt, probes.json)
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pinpad as P
from data import Split, Sampler
from model import Transformer, n_params
from probes import probe_all

ROOT = Path(__file__).resolve().parent
DEV = 'mps'


def loss_fn(out, b):
    lp = F.cross_entropy(out['logits'].flatten(0, 1), b['targets'].flatten(), reduction='none').view_as(b['weights'])
    w = b['weights']
    act = (lp * b['is_act']).sum() / b['is_act'].sum()
    return (lp * w).sum() / w.sum(), act


def bench():
    tr = Split('train'); res = []
    for d, L in ((128, 6), (192, 6), (256, 6)):
        for B in (64, 128):
            torch.manual_seed(0); m = Transformer(P.World().vocab, d, L).to(DEV)
            opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=0.03)
            s = Sampler(tr, B, 0); bs = [tr.batch(s.next_idx(), device=DEV) for _ in range(30)]
            for i in range(30):
                if i == 10: torch.mps.synchronize(); t0 = time.time()
                l, _ = loss_fn(m(bs[i]['ids']), bs[i]); opt.zero_grad(); l.backward(); opt.step()
            float(l); torch.mps.synchronize(); dt = (time.time() - t0) / 20
            ntok = np.mean([float(b['lengths'].sum()) for b in bs])
            r = dict(d=d, L=L, B=B, params=n_params(m), step_s=dt, tok_s=ntok / dt); res.append(r); print(r, flush=True)
    (ROOT / 'pilot_bench.json').write_text(json.dumps(res, indent=1))


def train(d, minutes, L=6, B=128, lr=1e-3, seed=1):
    run = ROOT / 'runs' / f'pilot_TF_d{d}'; run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tr, dev = Split('train'), Split('dev')
    m = Transformer(P.World().vocab, d, L).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr, betas=(0.9, 0.95), weight_decay=0.03)
    s = Sampler(tr, B, seed); log = open(run / 'train.jsonl', 'w')
    t0 = time.time(); step = 0; probes_at = []
    while time.time() - t0 < minutes * 60:
        b = tr.batch(s.next_idx(), device=DEV)
        l, act = loss_fn(m(b['ids']), b)
        opt.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        step += 1
        if step % 100 == 0:
            log.write(json.dumps(dict(step=step, t=time.time() - t0, loss=float(l), act=float(act))) + '\n'); log.flush()
        if step in (300, 1000, 2000, 3000, 5000, 8000, 12000):
            pr = probe_all(m, dev, DEV); pr['step'] = step; probes_at.append(pr)
            print(step, f'{time.time()-t0:.0f}s', 'act', round(float(act), 3), 'probe', [round(x, 3) for x in pr['abs_act']], flush=True)
    pr = probe_all(m, dev, DEV); pr['step'] = step; probes_at.append(pr)
    torch.save(m.state_dict(), run / 'final.pt')
    (run / 'probes.json').write_text(json.dumps(probes_at, indent=1))
    print('final', step, [round(x, 3) for x in pr['abs_act']], flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'bench': bench()
    else: train(int(sys.argv[2]), float(sys.argv[3]))
