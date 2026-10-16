"""Phase D: discover persistent latent actions over the frozen transplanted system, then internal RL over Z.

Stage 1 (discovery, ETA-style metacontroller [R5] adapted to our Z pathway):
  switch    beta_m = sigmoid(MLP([e_m ; Z_{m-1}]))                  causal
  proposal  Zt_m ~ q_D(. | demonstration)  (bidirectional GRU over [e_m ; c_{m+1}])   training only
  Z_m = (1 - beta_m) Z_{m-1} + beta_m Zt_m ;  forced beta=1 at the first walk boundary (chunk 30)
  loss = walk NLL through the frozen core (prior guidance) + decoder
         + lam_D * sum_m KL(q_D(Zt_m) || p_ref(Zt_m | e_m))   (p_ref = Phase B prior; ungated, per review §7.3)
Stage 2 (RL): Gaussian policy pi(Z | e_m, Z_prev) = prior mean + learned residual (zero-init), switch = frozen
  hardened beta (kappa=0.5) from stage 1; REINFORCE with a learned value baseline, gamma = 1, reward = verifier success.
usage: python phase_d.py --core_run C_v6_s0 --out D_v6_s0
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data_v5 as dv
from model_v5 import token_nll, concept_mask, gauss_kl, MLP2, K, D
from common_v5 import DEV, RUNS, Logger, save_json, eval_sets, cosine_lr, maybe_compile
from system_v5 import window_tensors
from eval_b import load_system, noise, hop_colours, walk_stats, starts_hold, starts_every

M = dv.T // K
W0 = dv.HEADER_CHUNKS - 1          # 30: first walk boundary
NB = dv.MAX_HOPS + 1               # boundaries 30..46 decide hops 1..16 and <done>
LAM_D = 0.1
KAPPA = 0.5


class Meta(nn.Module):
    def __init__(self, dz=8):
        super().__init__()
        self.switch = MLP2(D + dz, 1)
        self.gru = nn.GRU(2 * D, D, batch_first=True, bidirectional=True)
        self.post = MLP2(2 * D, 2 * dz)
        nn.init.constant_(self.switch.b.bias, 0.0)

    def beta(self, e_m, z_prev):
        return torch.sigmoid(self.switch(torch.cat([e_m, z_prev], -1))).squeeze(-1)

    def proposals(self, e, y):
        h, _ = self.gru(torch.cat([e, y], -1))
        mu, ls = self.post(h).chunk(2, -1)
        return mu, ls.clamp(-5, 2)


class Policy(nn.Module):
    def __init__(self, dz=8):
        super().__init__()
        self.res = MLP2(D + dz, 2 * dz)
        nn.init.zeros_(self.res.b.weight); nn.init.zeros_(self.res.b.bias)
        self.value = MLP2(D + dz, 1)

    def dist(self, e_m, z_prev, mu_p, ls_p):
        r_mu, r_ls = self.res(torch.cat([e_m, z_prev], -1)).chunk(2, -1)
        return mu_p + r_mu, (ls_p + r_ls).clamp(-5, 2)


def z_sequence(meta, e, mu_t, ls_t, hard=False, sample=True, z_header=None):
    """Builds Z_pos [B,M,dz] for the walk boundaries by the soft/hard integration recursion.
    Positions < W0 use z_header (convention shared with eval_b/phase_e: one prior draw at position 0)."""
    B = e.shape[0]
    zt = mu_t + ls_t.exp() * torch.randn_like(mu_t) if sample else mu_t
    Z = [None] * M
    betas = torch.zeros(B, M, device=e.device)
    z_prev = zt[:, W0]
    for m in range(M):
        if m < W0:
            Z[m] = z_header if z_header is not None else zt[:, W0]; continue
        if m == W0:
            b = torch.ones(B, device=e.device)
        else:
            b = meta.beta(e[:, m], z_prev)
            if hard:
                b = (b > KAPPA).float()
        betas[:, m] = b
        z = (1 - b.unsqueeze(-1)) * z_prev + b.unsqueeze(-1) * zt[:, m]
        Z[m] = z; z_prev = z
    return torch.stack(Z, 1), betas


def walk_mask(m):
    cm = F.pad(concept_mask(m), (0, 1))
    return cm


def discovery(sysm, a, log):
    meta = Meta().to(DEV)
    opt = torch.optim.AdamW(meta.parameters(), lr=a.lr_d, weight_decay=0.0)
    rng = np.random.RandomState(3000 + a.seed)
    core, t = sysm.core, sysm.t
    eps_mode = "prior" if core.use_eps else "off"
    for step in range(a.steps_d):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, a.steps_d, a.lr_d, warm=100)
        xg, mg, _ = dv.graph_batch(rng, a.bs)
        x = torch.from_numpy(xg).to(DEV); m = torch.from_numpy(mg).to(DEV)
        with torch.no_grad():
            H, C = sysm.context(x)
            e = core.in_proj(C)
            mu_p, ls_p = core.z_prior(e, None)
        y = F.pad(C[:, 1:], (0, 0, 0, 1))
        mu_t, ls_t = meta.proposals(e, y)
        zh = (mu_p[:, 0] + ls_p[:, 0].exp() * torch.randn_like(mu_p[:, 0])).detach()
        Z, betas = z_sequence(meta, e, mu_t, ls_t, z_header=zh)
        u, A, _ = core(C, Z, eps_mode=eps_mode, e=e)
        _, _, chat = t.pq(u)
        nll = token_nll(t.decode(H, chat, A), x, m)
        ntok = m[:, :-1].sum()
        l_nll = nll.sum() / ntok
        cm = walk_mask(m)
        klm = gauss_kl(mu_t, ls_t, mu_p, ls_p).sum(-1) * cm
        if a.gate_kl:
            klm = klm * betas                     # beta not detached: each switch pays for its information
        kl = klm.sum() / ntok
        loss = l_nll + LAM_D * kl
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(meta.parameters(), 1.0); opt.step()
        if step % 50 == 0:
            bw = (betas * cm)[:, W0 + 1:].sum() / cm[:, W0 + 1:].sum()
            log(stage="discovery", step=step, nll=l_nll.item(), kl=kl.item(), mean_beta=bw.item())
            if step % 200 == 0:
                print(f"disc {step} nll {l_nll.item():.4f} kl {kl.item():.4f} beta {bw.item():.3f}", flush=True)
    return meta


@torch.no_grad()
def discovery_eval(sysm, meta, xg, mg, metas):
    """Soft vs hardened NLL (same proposals, proposal means) and switch alignment with true colour switches."""
    core, t = sysm.core, sysm.t
    eps_mode = "prior" if core.use_eps else "off"
    res = {"soft": [0.0, 0.0], "hard": [0.0, 0.0]}
    tp = fp = fn = 0
    betas_true, betas_false = [], []
    for i in range(0, len(xg), 128):
        x = torch.from_numpy(xg[i:i + 128]).to(DEV); m = torch.from_numpy(mg[i:i + 128]).to(DEV)
        H, C = sysm.context(x)
        e = core.in_proj(C)
        y = F.pad(C[:, 1:], (0, 0, 0, 1))
        mu_t, ls_t = meta.proposals(e, y)
        ez, nz = noise(x.shape[0], 77 + i)
        mu_p, ls_p = core.z_prior(e, None)
        zh = mu_p[:, 0] + ls_p[:, 0].exp() * ez[:, 0]
        for mode in ("soft", "hard"):
            Z, betas = z_sequence(meta, e, mu_t, ls_t, hard=(mode == "hard"), sample=False, z_header=zh)
            u, A, _ = core(C, Z, eps_mode=eps_mode, noise=nz, e=e)
            _, _, chat = t.pq(u)
            nll = token_nll(t.decode(H, chat, A), x, m)
            res[mode][0] += nll.sum().item(); res[mode][1] += m[:, :-1].sum().item()
            if mode == "soft":
                bs = betas.cpu().numpy()
        # true switch at boundary W0+h (h>=1): colour of hop h+1 differs from hop h
        for r in range(x.shape[0]):
            hc = metas[i + r]["hop_cols"]
            for h in range(1, len(hc)):
                true_sw = hc[h] != hc[h - 1]
                b = bs[r, W0 + h]
                (betas_true if true_sw else betas_false).append(b)
                pred = b > KAPPA
                tp += pred and true_sw; fp += pred and not true_sw; fn += (not pred) and true_sw
    out = {k: v[0] / v[1] for k, v in res.items()}
    out["soft_minus_hard_nll"] = out["soft"] - out["hard"]
    out["switch_precision"] = tp / max(tp + fp, 1); out["switch_recall"] = tp / max(tp + fn, 1)
    out["mean_beta_at_true_switch"] = float(np.mean(betas_true)); out["mean_beta_at_non_switch"] = float(np.mean(betas_false))
    out["fraction_beta_intermediate"] = float(np.mean([(0.1 < b < 0.9) for b in betas_true + betas_false]))
    return out


# ----------------------------------------------------------------------------- controlled generation
_STEP = {}


def step_fns(sysm):
    """Compiled pieces for rollouts: context (encode+pool+in_proj+prior) and core+decoder logits."""
    key = id(sysm)
    if key not in _STEP:
        core, t = sysm.core, sysm.t
        eps_mode = "prior" if core.use_eps else "off"

        def ctx(x):
            H = t.encode(x); C = t.pool(H); e = core.in_proj(C)
            mu, ls = core.z_prior(e, None)
            return H, C, e, mu, ls

        def dec(H, C, e, Z, ee):
            u, A, _ = core(C, Z, eps_mode=eps_mode, noise=ee, e=e)
            _, _, chat = t.pq(u)
            return t.decode(H, chat, A)
        _STEP[key] = (maybe_compile(ctx), maybe_compile(dec))
    return _STEP[key]


@torch.no_grad()
def rollout(sysm, meta, policy, xg, controller, seed, temperature=0.0, record=False, eps_seed=None):
    """Generate walks; at each walk boundary m the controller decides Z_m.
    controller in {'policy','prior','prior_hold','always','fixed3'}; switch for 'policy'/'prior' = hardened beta.
    Returns tokens, and (if record) per-decision (e_m, z_prev, z, is_decision) for the policy gradient."""
    core, t = sysm.core, sysm.t
    eps_mode = "prior" if core.use_eps else "off"
    x = torch.from_numpy(xg.copy()); x[:, dv.WALK_START:] = dv.PAD; x = x.to(DEV)
    B = x.shape[0]
    g = torch.Generator().manual_seed(seed)
    ee = noise(B, eps_seed if eps_seed is not None else seed)[1]
    Z = torch.zeros(B, M, core.dz, device=DEV)
    z_prev = torch.zeros(B, core.dz, device=DEV)
    recs = []
    gen_tok = torch.Generator().manual_seed(seed + 1)
    ctx_c, dec_c = step_fns(sysm)
    for pos in range(dv.WALK_START - 1, dv.T - 1):
        H, C, e, mu_p, ls_p = ctx_c(x)
        if (pos - 3) % 4 == 0:                    # new boundary m: chunk m complete, decide Z_m
            mb = (pos - 3) // 4
            em, mp, lp = e[:, mb], mu_p[:, mb], ls_p[:, mb]
            if mb == W0:                          # header convention (shared with eval_b/phase_e)
                zh = mu_p[:, 0] + ls_p[:, 0].exp() * torch.randn(mu_p[:, 0].shape, generator=g).to(DEV)
                if controller == "prior_mean":
                    zh = mu_p[:, 0]
                Z[:, :W0] = zh.unsqueeze(1)
            if mb == W0 or controller in ("always", "prior_mean"):
                sw = torch.ones(B, device=DEV)
            elif controller in ("policy", "prior"):
                sw = (meta.beta(em, z_prev) > KAPPA).float()
            elif controller == "fixed3":
                sw = torch.full((B,), float((mb - W0) % 3 == 0), device=DEV)
            elif controller == "prior_hold4":
                sw = torch.full((B,), float((mb - W0) % 4 == 0), device=DEV)
            else:                                 # prior_hold
                sw = torch.zeros(B, device=DEV)
            if controller == "policy":
                mu, ls = policy.dist(em, z_prev, mp, lp)
            else:
                mu, ls = mp, lp
            z_new = mu + ls.exp() * torch.randn(mu.shape, generator=g).to(DEV)
            if controller == "prior_mean":       # deterministic prior at every boundary (zero-cost policy)
                z_new = mu
            z = torch.where(sw.unsqueeze(-1) > 0, z_new, z_prev)
            if record:
                recs.append((em.clone(), z_prev.clone(), z_new.clone(), sw.clone(), mp.clone(), lp.clone()))
            Z[:, mb:] = z.unsqueeze(1)            # positions beyond mb are refreshed at later boundaries
            z_prev = z
        lg = dec_c(H, C, e, Z, ee)[:, pos].float()
        if temperature <= 0:
            nxt = lg.argmax(-1)
        else:
            nxt = torch.multinomial((lg / temperature).softmax(-1).cpu(), 1, generator=gen_tok).squeeze(1).to(DEV)
        x[:, pos + 1] = nxt
        if pos % 4 == 3 and bool((x[:, dv.WALK_START:pos + 2] == dv.DONE).any(1).all()):
            break
    return x.cpu().numpy(), recs


def rewards(X):
    return np.array([float(dv.verify(r[:dv.WALK_START], r[dv.WALK_START:])[0]) for r in X])


def n_decisions_used(X):
    """Number of boundaries actually used per episode (hops + done)."""
    return np.array([len(hop_colours(r)) + 1 for r in X])


def rl(sysm, meta, a, log):
    policy = Policy().to(DEV)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr_rl, weight_decay=0.0)
    rng = np.random.RandomState(4000 + a.seed)
    hist = []
    for it in range(a.iters_rl):
        # pre-registered (11:12): RL trains on the hard-goal generator (seeds disjoint from the eval set, seed 31)
        xg = np.stack([dv.gen_hardgoal(rng)[0] for _ in range(a.bs_rl)])
        X, recs = rollout(sysm, meta, policy, xg, "policy", seed=10_000 + it, record=True)
        R = torch.from_numpy(rewards(X)).float().to(DEV)
        used = torch.from_numpy(n_decisions_used(X)).to(DEV)
        lp_sum = 0.0; v_loss = 0.0; n_dec = 0
        for j, (em, zp, zn, sw, mp, lp) in enumerate(recs):
            active = (sw > 0) & (j < used)
            if not active.any():
                continue
            mu, ls = policy.dist(em, zp, mp, lp)
            logp = (-0.5 * ((zn - mu) / ls.exp()).pow(2) - ls).sum(-1)
            v = policy.value(torch.cat([em, zp], -1)).squeeze(-1)
            adv = (R - v).detach()
            lp_sum = lp_sum + -(adv * logp * active.float()).sum()
            v_loss = v_loss + ((R - v).pow(2) * active.float()).sum()
            n_dec += int(active.sum())
        loss = (lp_sum + 0.5 * v_loss) / max(n_dec, 1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        hist.append(R.mean().item())
        if it % 10 == 0:
            log(stage="rl", it=it, reward=R.mean().item(), reward_ma=float(np.mean(hist[-10:])), n_dec=n_dec)
            print(f"rl {it} reward {R.mean().item():.3f} ma10 {np.mean(hist[-10:]):.3f}", flush=True)
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core_run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps_d", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr_d", type=float, default=1e-3)
    ap.add_argument("--iters_rl", type=int, default=200)
    ap.add_argument("--bs_rl", type=int, default=128)
    ap.add_argument("--lr_rl", type=float, default=3e-3)
    ap.add_argument("--n_eval", type=int, default=512)
    ap.add_argument("--gate_kl", action="store_true",
                    help="post-hoc arm (16:02): KL_m weighted by the (non-detached) switch beta_m (review §7.3)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = os.path.join(RUNS, a.out); os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.jsonl"))
    sysm, ck = load_system(a.core_run)
    assert sysm.core.use_z, "Phase D needs a Z-pathway core"
    t0 = time.time()
    meta = discovery(sysm, a, log)
    _, xg, mg, metas = eval_sets("val")
    res = {"core_run": a.core_run, "discovery": discovery_eval(sysm, meta, xg[:a.n_eval], mg[:a.n_eval], metas[:a.n_eval])}
    print("discovery eval", json.dumps(res["discovery"]), flush=True)
    torch.save({"meta": meta.state_dict()}, os.path.join(out, "meta.pt"))
    # baselines before RL (equal attempts: one episode per instance, same seeds); both evaluation sets
    xh_hard, _ = dv.hardgoal_set(a.n_eval, seed=31)
    sets = {"standard": xg[:a.n_eval], "hardgoal": xh_hard}

    def run_ctrl(xs, ctrl, policy=None, temperature=0.0):
        X, _ = rollout(sysm, meta, policy, xs, ctrl, seed=123, temperature=temperature)
        r = rewards(X)
        out = dict(success=float(r.mean()), se=float(r.std(ddof=1) / np.sqrt(len(r))),
                   **{k: v for k, v in walk_stats(X)[0].items() if k not in ("success", "success_se")})
        out["flags"] = r.tolist()
        return out
    ev = {name: {} for name in sets}
    for name, xs in sets.items():
        for ctrl in ("prior", "prior_hold4", "prior_mean", "prior_hold", "always", "fixed3"):
            ev[name][ctrl] = run_ctrl(xs, ctrl)
        ev[name]["token_sampling_T1_prior_hold4"] = run_ctrl(xs, "prior_hold4", temperature=1.0)
    policy = rl(sysm, meta, a, log)
    for name, xs in sets.items():
        ev[name]["policy"] = run_ctrl(xs, "policy", policy)
        for base in ("prior", "prior_hold4", "prior_mean"):
            d = np.array(ev[name]["policy"]["flags"]) - np.array(ev[name][base]["flags"])
            ev[name][f"policy_minus_{base}"] = dict(delta=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))
    res["rollouts"] = ev
    res["wall_s"] = time.time() - t0
    torch.save({"policy": policy.state_dict()}, os.path.join(out, "policy.pt"))
    save_json(os.path.join(out, "phase_d.json"), res)
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "flags"}
        return o
    print(json.dumps(strip(res), indent=1)[:6000])


if __name__ == "__main__":
    main()
