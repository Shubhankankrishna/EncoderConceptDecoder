"""Internal RL (ETA paper §"Internal reinforcement learning", C.5, Algorithms 1–3) and the raw-action RL baseline.

Internal RL: the frozen base, the frozen metacontroller switch unit and controller decoder are the environment.
A causal policy pi(z | e_{1:t}) (1-layer GRU, Gaussian with learned log-std) replaces the acausal encoder. At
each decision unit the frozen switch emits beta (thresholded at 0.5; the first unit always switches); only at
switches is a new z sampled, otherwise z is held, so the policy acts on the abstract timescale. Sparse reward:
1 if the whole task is completed. Advantage = (R - mean) / (std + 1e-3) over the batch (critic-free, C.5.2).
Objective: PPO-clip on log pi(z) of the switch decisions (2 epochs per batch).

Raw-action RL baseline: the same estimator on the base's own action log-probs, fine-tuning the whole base.

  python -B internal_rl.py internal RUN TAG TASK ITERS SEED
  python -B internal_rl.py raw RUN TASK ITERS SEED
"""
import json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pinpad as P
from evaluate import load_run
from meta import MetaController

ROOT = Path(__file__).resolve().parent


class Policy(nn.Module):
    def __init__(self, d, nz=8, nh=64):
        super().__init__()
        self.inp = nn.Linear(d, nh); self.gru = nn.GRUCell(nh, nh); self.mu = nn.Linear(nh, nz)
        self.log_std = nn.Parameter(torch.zeros(nz))
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)       # start at the prior N(0, I)


class Env:
    """Batched pinpad episodes driven by a (controlled) frozen model; full recompute each step."""
    def __init__(self, world, n, seed, task):
        self.world, self.n, self.task = world, n, task
        rng = random.Random(seed)
        self.st, seqs = [], []
        for _ in range(n):
            s, q = P.env_reset(world, rng); self.st.append(s); seqs.append(q)
        self.H = len(seqs[0]); L = self.H + 2 * world.T + 1
        self.ids = torch.zeros(n, L, dtype=torch.long)
        for i, q in enumerate(seqs): self.ids[i, :self.H] = torch.tensor(q)
        self.cur = self.H; self.done = np.zeros(n, bool); self.success = np.zeros(n, bool)

    def step(self, acts):
        self.ids[:, self.cur] = torch.tensor(P.ACT0 + acts)
        newpos = []
        for i, s in enumerate(self.st):
            if self.done[i]: newpos.append(P.PAD); continue
            newpos.append(P.env_step(self.world, s, int(acts[i]), self.task))
            if s['done']: self.done[i] = True; self.success[i] = s['success']
        self.ids[:, self.cur + 1] = torch.tensor(newpos); self.cur += 2


class ZController:
    """ctrl callable for internal RL: decision units = token decision rows (header + 2j) or concept groups.
    Holds z per unit (B, J, nz); at the newest unit it queries the switch + policy (if `act`)."""
    def __init__(self, model, mc, pol, H, group_rate):
        self.model, self.mc, self.pol, self.H, self.gr = model, mc, pol, H, group_rate
        self.z = None; self.h_mc = None; self.h_pi = None; self.n_units = 0
        self.records = []                               # (e_j, switch, z_sample) per unit for the PPO update

    def units(self, N, T):
        hdr = self.H - 1                                    # index of p_0 (the first decision row)
        if not self.gr:
            return [hdr + 2 * j for j in range((T - self.H) // 2 + 1)]
        k = self.model.k
        return [g for g in range(N) if g * k + k - 1 >= hdr]

    def new_unit(self, e_j, sample=True):
        B = e_j.shape[0]
        x = self.mc.inp(e_j)
        if self.z is None:
            self.z = x.new_zeros(B, self.mc.nz); self.h_mc = x.new_zeros(B, x.shape[-1])
            self.h_pi = x.new_zeros(B, self.pol.gru.hidden_size)
            sw = torch.ones(B, dtype=torch.bool, device=x.device)
        else:
            beta = self.mc.switch(x, self.h_mc, self.z)[:, 0]
            sw = beta > 0.5
        self.h_pi = self.pol.gru(self.pol.inp(e_j), self.h_pi)
        mu = self.pol.mu(self.h_pi); std = self.pol.log_std.exp()
        zs = mu + std * torch.randn_like(mu) if sample else mu
        self.z = torch.where(sw[:, None], zs, self.z)
        self.h_mc = self.mc.gru(x[:, None], self.h_mc[None])[1][0]
        alive = torch.tensor(~self.env.done, device=sw.device) if hasattr(self, 'env') else torch.ones_like(sw)
        self.records.append((e_j.detach(), (sw & alive).detach(), zs.detach()))
        return self.z

    def __call__(self, e):
        B, N, D = e.shape
        T = self.T
        if not self.gr:
            rows = self.units(N, T)
            while self.n_units < len(rows):
                self.zs.append(self.new_unit(e[:, rows[self.n_units]])); self.n_units += 1
            Z = torch.stack(self.zs, 1)
            A, Bm = self.mc.controllers(Z)
            out = e.clone()
            for off in (0, 1):
                r = torch.tensor([min(x + off, N - 1) for x in rows], device=e.device)
                er = e[:, r]
                out[:, r] = out[:, r] + torch.einsum('bjdr,bjr->bjd', A, torch.einsum('bjrd,bjd->bjr', Bm, er))
            return out
        gs = self.units(N, T)
        while self.n_units < len(gs):
            self.zs.append(self.new_unit(e[:, gs[self.n_units]])); self.n_units += 1
        if not gs: return e
        Z = torch.stack(self.zs, 1)
        A, Bm = self.mc.controllers(Z)
        gi = torch.tensor(gs, device=e.device)
        out = e.clone()
        out[:, gi] = e[:, gi] + torch.einsum('bjdr,bjr->bjd', A, torch.einsum('bjrd,bjd->bjr', Bm, e[:, gi]))
        return out


@torch.no_grad()
def run_internal_batch(model, mc, pol, world, n, seed, task, device, sample=True):
    env = Env(world, n, seed, task)
    zc = ZController(model, mc, pol, env.H, model.kind == 'ECD' and getattr(model, 'ctrl_point', '') == 'concept')
    zc.zs = []; zc.env = env
    gen = torch.Generator().manual_seed(seed)
    for t in range(world.T):
        zc.T = env.cur
        x = env.ids[:, :env.cur].to(device)
        logits = model(x, ctrl=zc)['logits'][:, -1, P.ACT0:P.ACT0 + 4].float().cpu()
        acts = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen)[:, 0].numpy()
        env.step(acts)
        if env.done.all(): break
    return env, zc


def ppo_update(pol, opt, recs, adv, clip=0.2, epochs=2):
    """recs: list over units of (e_j (B,d), switch (B,), z (B,nz)). Recompute log pi under the current policy."""
    E = torch.stack([r[0] for r in recs], 1); S = torch.stack([r[1] for r in recs], 1); Z = torch.stack([r[2] for r in recs], 1)
    def logp():
        B, J, _ = E.shape
        h = E.new_zeros(B, pol.gru.hidden_size); lps = []
        for j in range(J):
            h = pol.gru(pol.inp(E[:, j]), h)
            mu = pol.mu(h); std = pol.log_std.exp()
            lps.append((-0.5 * ((Z[:, j] - mu) / std) ** 2 - pol.log_std).sum(-1))
        return (torch.stack(lps, 1) * S).sum(1)                       # sum over switch decisions
    with torch.no_grad(): old = logp()
    a = torch.tensor(adv, device=E.device, dtype=E.dtype)
    for _ in range(epochs):
        lp = logp(); ratio = (lp - old).exp()
        loss = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()


def internal(run, tag, task_str, iters, seed, n=256, lr=1e-3, device='mps'):
    model, arm = load_run(run, device)
    for p in model.parameters(): p.requires_grad_(False)
    cp = tag.split('_a')[0]
    if model.kind == 'ECD': model.ctrl_point = cp
    elif cp.startswith('L'): model.ctrl_layer = int(cp[1:])
    mc = MetaController(model.d).to(device); mc.load_state_dict(torch.load(ROOT / 'runs' / run / 'meta' / f'{tag}.pt'))
    for p in mc.parameters(): p.requires_grad_(False)
    torch.manual_seed(seed)
    pol = Policy(model.d).to(device); opt = torch.optim.Adam(pol.parameters(), lr)
    world = P.World(); task = P.parse(task_str); hist = []; t0 = time.time()
    for it in range(iters):
        env, zc = run_internal_batch(model, mc, pol, world, n, 1000 * seed + it, task, device)
        R = env.success.astype(np.float32)
        hist.append(dict(it=it, success=float(R.mean()), switches=float(torch.stack([r[1] for r in zc.records], 1).float().sum(1).mean()),
                         t=round(time.time() - t0)))
        if R.std() > 0:
            with torch.enable_grad():
                ppo_update(pol, opt, zc.records, (R - R.mean()) / (R.std() + 1e-3))
        if it % 5 == 0: print(json.dumps(hist[-1]), flush=True)
    out = ROOT / 'runs' / run / 'rl'; out.mkdir(exist_ok=True)
    (out / f'internal_{tag}_{task_str}_s{seed}.json').write_text(json.dumps(dict(run=run, tag=tag, task=task_str, hist=hist), indent=1))
    return hist


def raw(run, task_str, iters, seed, n=256, lr=3e-5, device='mps'):
    model, arm = load_run(run, device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr)
    world = P.World(); task = P.parse(task_str); hist = []; t0 = time.time()
    for it in range(iters):
        env = Env(world, n, 1000 * seed + it, task)
        gen = torch.Generator().manual_seed(seed * 7 + it)
        with torch.no_grad():
            for t in range(world.T):
                logits = model(env.ids[:, :env.cur].to(device))['logits'][:, -1, P.ACT0:P.ACT0 + 4].float().cpu()
                env.step(torch.multinomial(torch.softmax(logits, -1), 1, generator=gen)[:, 0].numpy())
                if env.done.all(): break
        R = env.success.astype(np.float32)
        hist.append(dict(it=it, success=float(R.mean()), t=round(time.time() - t0)))
        if R.std() > 0:
            adv = torch.tensor((R - R.mean()) / (R.std() + 1e-3), device=device)
            ids = env.ids[:, :env.cur].to(device)
            logp = F.log_softmax(model(ids)['logits'][:, :-1].float(), -1)
            tgt = ids[:, 1:]
            is_act = (tgt >= P.ACT0) & (tgt < P.ACT0 + 4)
            is_act[:, :env.H - 1] = False
            nreal = torch.tensor([st['steps'] for st in env.st], device=device)
            col = torch.arange(tgt.shape[1], device=device)[None]
            is_act &= col < (env.H - 1 + 2 * nreal[:, None])            # only actions taken before the episode ended
            lp = (logp.gather(-1, tgt[..., None])[..., 0] * is_act).sum(1)
            loss = -(lp * adv).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if it % 5 == 0: print(json.dumps(hist[-1]), flush=True)
    out = ROOT / 'runs' / run / 'rl'; out.mkdir(exist_ok=True)
    (out / f'raw_{task_str}_s{seed}.json').write_text(json.dumps(dict(run=run, task=task_str, hist=hist), indent=1))
    return hist


if __name__ == '__main__':
    a = sys.argv
    if a[1] == 'internal': internal(a[2], a[3], a[4], int(a[5]), int(a[6]))
    else: raw(a[2], a[3], int(a[4]), int(a[5]))
