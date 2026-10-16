"""Ground-truth subgoal controllers on a frozen base (paper §"internal controllers", C.3, Fig. 4).

One rank-16 linear residual controller per abstract action (4 pairs), e <- e + A_g B_g e, inserted at the base's
control point (TF: residual before block L//2; ECD: the concept stream or the decoder input). Trained with the
base frozen on the clean `meta` split, labels = the ground-truth pair of the *next* token (row t gets
abs_act[t+1]; a group-rate controller reads the label of its release row k g + k - 1), action NLL only.
Evaluated closed-loop on the post-training task (0-1-2-3-4-5-6-7-0-1-2-3: new order, longer) by activating the
controllers in the right order from the environment's state.

  python -B gtctrl.py RUN [ctrl_point]     -> runs/RUN/gtctrl[_POINT].json   (TF: 'mid' = e_{L//2}, or 'L5' = e_5)
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pinpad as P
from data import Split, Sampler
from rollout import rollout
from evaluate import load_run

ROOT = Path(__file__).resolve().parent
SHORT_TASK = '0-1-2-3-4-5'


class LowRankBank(nn.Module):
    def __init__(self, d, G=4, r=16):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(G, d, r)); self.B = nn.Parameter(torch.randn(G, r, d) / d ** 0.5)
        self.G = G

    def forward(self, e, lab):
        oh = F.one_hot(lab.clamp(min=0), self.G).to(e.dtype) * (lab >= 0)[..., None].to(e.dtype)   # (B, N, G)
        y = torch.einsum('bnd,grd->bngr', e, self.B) * oh[..., None]
        return e + torch.einsum('bngr,gdr->bnd', y, self.A)


def group_labels(sub, k, G):
    rows = torch.arange(G, device=sub.device) * k + k - 1
    return sub[:, rows.clamp(max=sub.shape[1] - 1)]


def make_ctrl(model, bank, sub):
    """sub: (B, T) per-row labels. Returns the callable passed as model(ids, ctrl=...)."""
    def ctrl(e):
        if e.shape[1] == sub.shape[1]: return bank(e, sub)                         # token rate
        return bank(e, group_labels(sub, model.k, e.shape[1]))                      # group rate
    return ctrl


def train_bank(model, ctrl_point, steps=1500, B=128, lr=1e-3, device='mps', seed=0):
    torch.manual_seed(seed)
    if hasattr(model, 'ctrl_point'): model.ctrl_point = ctrl_point
    bank = LowRankBank(model.d).to(device)
    opt = torch.optim.AdamW(bank.parameters(), lr, weight_decay=0.03)
    meta = Split('meta'); s = Sampler(meta, B, seed)
    hist = []
    for step in range(steps):
        b = meta.batch(s.next_idx(), lam_obs=0.0, device=device, pad_to=176)      # fixed shape: no allocator bloat
        lab = torch.full_like(b['abs_act'], -1); lab[:, :-1] = b['abs_act'][:, 1:]
        out = model(b['ids'], ctrl=make_ctrl(model, bank, lab))
        lp = F.cross_entropy(out['logits'].flatten(0, 1).float(), b['targets'].flatten(), reduction='none').view_as(b['weights'])
        loss = (lp * b['is_act']).sum() / b['is_act'].sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 100 == 0:
            hist.append((step, float(loss)))
            if device == 'mps': torch.mps.empty_cache()
    return bank, hist


@torch.no_grad()
def base_action_nll(model, split, device, n=1024, ctrl_fn=None):
    tot, cnt = 0.0, 0
    for s in range(0, n, 128):
        b = split.batch(np.arange(s, min(s + 128, n)), lam_obs=0.0, device=device, pad_to=176)
        ctrl = ctrl_fn(b) if ctrl_fn else None
        out = model(b['ids'], ctrl=ctrl)
        lp = F.cross_entropy(out['logits'].flatten(0, 1).float(), b['targets'].flatten(), reduction='none').view_as(b['weights'])
        tot += float((lp * b['is_act']).sum()); cnt += int(b['is_act'].sum())
    return tot / cnt


def main(run, ctrl_point=None, device='mps', n_roll=256, steps=1500):
    model, arm = load_run(run, device)
    for p in model.parameters(): p.requires_grad_(False)
    ctrl_point = ctrl_point or ('concept' if model.kind == 'ECD' else 'mid')
    if model.kind == 'TF' and ctrl_point.startswith('L'): model.ctrl_layer = int(ctrl_point[1:])   # e.g. L5 = before block 5
    t0 = time.time()
    bank, hist = train_bank(model, ctrl_point, steps=steps, device=device)
    world = P.World(); task = P.parse(P.POST_TASK)
    fac = lambda x, sub: make_ctrl(model, bank, sub)
    res = dict(run=run, ctrl_point=ctrl_point, train_hist=hist, train_s=time.time() - t0)
    dm = Split('dev_meta')
    def lab_ctrl(b):
        lab = torch.full_like(b['abs_act'], -1); lab[:, :-1] = b['abs_act'][:, 1:]
        return make_ctrl(model, bank, lab)
    res['dev_meta_act_nll_base'] = base_action_nll(model, dm, device)
    res['dev_meta_act_nll_gt'] = base_action_nll(model, dm, device, ctrl_fn=lab_ctrl)
    r_post = rollout(model, world, n_roll, 7, task=task, ctrl_factory=fac, device=device)
    r_post0 = rollout(model, world, n_roll, 7, task=task, ctrl_factory=None, device=device)
    res['post_success_gt'] = float(r_post['success'].mean()); res['post_success_base'] = float(r_post0['success'].mean())
    # 3-pair composition with two transitions absent from Table A1 (0-1 -> 2-3, 2-3 -> 4-5): the internal-RL task
    short = P.parse(SHORT_TASK)
    r_s = rollout(model, world, n_roll, 8, task=short, ctrl_factory=fac, device=device)
    r_s0 = rollout(model, world, n_roll, 8, task=short, ctrl_factory=None, device=device)
    res['short_success_gt'] = float(r_s['success'].mean()); res['short_success_base'] = float(r_s0['success'].mean())
    # pretraining tasks with GT controllers (in-distribution controllability)
    succ = []
    for j, tk in enumerate(P.PRETRAIN_TASKS[:8]):
        r = rollout(model, world, 64, 100 + j, task=P.parse(tk), ctrl_factory=fac, device=device)
        succ.append(float(r['success'].mean()))
    res['pretrain_success_gt'] = float(np.mean(succ))
    res['elapsed'] = time.time() - t0
    out = ROOT / 'runs' / run / f'gtctrl_{ctrl_point}.json'
    out.write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != 'train_hist'}), flush=True)
    return res


if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2] if len(a) > 2 and a[2] != '-' else None, steps=int(a[3]) if len(a) > 3 else 1500)
