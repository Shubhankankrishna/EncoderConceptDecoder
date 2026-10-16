"""Phase E: GradCuit-style test-time optimisation of the control offset dZ (review_v5 §8.3 corrected loop).

One control window per episode (Z held over the walk from the first walk boundary), Z = Z0 + dZ, dZ = 0 at start.
Weights frozen; tokens and reward detached; reward 0 (accepted) / -1 (rejected) from the header-only verifier.
Loss on rejection: -R * sum_t log p(y_t | ...) = sum log p  -> lowers the rejected walk's probability.
Generate + verify after EVERY update (U updates -> at most U+1 generations).
Baselines at the same number of generations: Z resampled from the prior; token sampling T=1; norm-matched random dZ.
usage: python phase_e.py --core_run C_v6_s0 [--d_run D_v6_s0] --out E_v6_s0
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

import data_v5 as dv
from model_v5 import token_nll, K, D
from common_v5 import maybe_compile, DEV, RUNS, eval_sets, save_json
from eval_b import load_system, noise, starts_hold4

M = dv.T // K
W0 = dv.HEADER_CHUNKS - 1


def make_fz(sysm):
    """fz(x, eta_z, delta, ee): Z = per-window prior (hold4 schedule, fixed draws eta_z) + one shared offset delta."""
    core, t = sysm.core, sysm.t
    eps_mode = "prior" if core.use_eps else "off"

    def fz(x, eta_z, delta, ee):
        H = t.encode(x); C = t.pool(H)
        e = core.in_proj(C)
        Z0 = sysm.z_prior_schedule(e, starts_hold4(x.shape[0]), eta_z)[0]
        walk = (torch.arange(M, device=x.device) >= W0).float().view(1, M, 1)   # dZ acts on the walk window only
        u, A, _ = core(C, Z0 + delta.unsqueeze(1) * walk, eps_mode=eps_mode, noise=ee, e=e)
        _, _, chat = t.pq(u)
        return t.decode(H, chat, A)
    return fz


def zpos_hold(z):
    """z [B,dz] -> single window; z [B,M,dz] (per-position, hold4 schedule) passes through."""
    return z if z.dim() == 3 else z.unsqueeze(1).expand(-1, M, -1).contiguous()


@torch.no_grad()
def generate(fz_c, xh, eta_z, delta, ee, temperature=0.0, gen=None):
    x = torch.from_numpy(xh.copy()); x[:, dv.WALK_START:] = dv.PAD; x = x.to(DEV)
    for pos in range(dv.WALK_START - 1, dv.T - 1):
        lg = fz_c(x, eta_z, delta, ee)[:, pos].float()
        if temperature <= 0:
            nxt = lg.argmax(-1)
        else:
            nxt = torch.multinomial((lg / temperature).softmax(-1).cpu(), 1, generator=gen).squeeze(1).to(DEV)
        x[:, pos + 1] = nxt
        if pos % 4 == 3 and bool((x[:, dv.WALK_START:pos + 2] == dv.DONE).any(1).all()):
            break
    return x


def verify_batch(x):
    xc = x.cpu().numpy()
    return np.array([dv.verify(r[:dv.WALK_START], r[dv.WALK_START:])[0] for r in xc])


def walk_loss_mask(x):
    """Score the generated walk: inputs WALK_START-1 .. through the end of the <done> chunk (or 16 hops)."""
    B, T = x.shape
    m = torch.zeros(B, T, device=x.device)
    xc = x.cpu().numpy()
    for r in range(B):
        end = dv.T - 1
        for ch in range(dv.MAX_HOPS + 1):
            p = dv.WALK_START + 4 * ch
            if p < T and xc[r, p] == dv.DONE:
                end = p + 3; break
        m[r, dv.WALK_START - 1:min(end, T - 1)] = 1.0
    return m


def base_controls(sysm, xh, seed):
    """Z0 = per-window prior samples on the hold4 schedule (windows of 4 chunks from the first walk boundary;
    in distribution for B's 1-4 chunk windows).  Uses the header-only context at each window start: windows after the
    first read the context of the *generated* prefix, so Z0 is recomputed after each generation (see gen_with_offset).
    Here we return the eta draws; the prior is applied inside generation."""
    return noise(len(xh), seed)[0]


def gradcuit(sysm, fz, fz_c, xh, Z0, ee, lr, U, random_ctrl=False, fresh_noise=False, seed=0):
    """Z0 here is the fixed eta_z draw [B,M,dz] defining the per-window prior controls.
    Returns accepted flags, attempts used, final |dZ| (per instance)."""
    B = len(xh)
    dz = Z0.shape[-1]
    delta = torch.zeros(B, dz, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    g = torch.Generator().manual_seed(seed)
    cur_ee = ee
    x = generate(fz_c, xh, Z0, delta.detach(), cur_ee)
    acc = verify_batch(x)
    attempts = np.ones(B)
    frozen = delta.detach().clone()
    for u in range(U):
        active = torch.from_numpy(~acc).to(DEV)
        if not active.any():
            break
        if random_ctrl:
            with torch.no_grad():   # norm-matched random step: same per-row norm as a typical Adam step (lr * sqrt(dz))
                r = torch.randn((B, dz), generator=g).to(DEV)
                r = r / r.norm(dim=-1, keepdim=True) * lr * (dz ** 0.5)
                delta += r * active.unsqueeze(-1).float()
        else:
            opt.zero_grad()
            m = walk_loss_mask(x)
            logp = -token_nll(fz(x, Z0, delta, cur_ee), x, m).sum(1)                 # [B] sum log p of the walk
            reward = torch.where(active, -1.0, 0.0)
            loss = (-reward * logp).sum()                                         # = sum over rejected of log p
            loss.backward()
            opt.step()
            with torch.no_grad():   # accepted rows keep their accepted control
                delta.data = torch.where(active.unsqueeze(-1), delta.data, frozen)
        if fresh_noise:
            cur_ee = torch.randn(ee.shape, generator=g).to(DEV)
        x_new = generate(fz_c, xh, Z0, delta.detach(), cur_ee)
        acc_new = verify_batch(x_new)
        act = ~acc
        attempts[act] += 1
        x = torch.where(torch.from_numpy(act).to(DEV).unsqueeze(-1), x_new, x)
        acc = acc | (act & acc_new)
        frozen = delta.detach().clone()
    return acc, attempts, delta.detach().norm(dim=-1).cpu().numpy()


def resample_baseline(sysm, fz_c, xh, ee, U, seed):
    """U+1 attempts, each with a fresh prior Z draw, same eps noise."""
    B = len(xh)
    acc = np.zeros(B, dtype=bool); attempts = np.zeros(B)
    zero = torch.zeros(B, 8, device=DEV)
    for k in range(U + 1):
        Z = base_controls(sysm, xh, seed + 1000 * k)
        x = generate(fz_c, xh, Z, zero, ee)
        a = verify_batch(x)
        attempts[~acc] += 1
        acc = acc | a
    return acc, attempts


def token_sampling_baseline(fz_c, xh, Z0, ee, U, seed):
    B = len(xh)
    acc = np.zeros(B, dtype=bool)
    g = torch.Generator().manual_seed(seed)
    zero = torch.zeros(B, Z0.shape[-1], device=DEV)
    for k in range(U + 1):
        x = generate(fz_c, xh, Z0, zero, ee, temperature=1.0, gen=g)
        acc = acc | verify_batch(x)
    return acc


def summ(flags):
    f = np.asarray(flags, dtype=float)
    return dict(success=float(f.mean()), se=float(f.std(ddof=1) / np.sqrt(len(f))))


def pair(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return dict(delta=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core_run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--U", type=int, default=10)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--lrs", default="0.01,0.1,0.3")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--set", default="hardgoal", help="hardgoal (primary, pre-registered 11:12) | standard")
    a = ap.parse_args()
    out = os.path.join(RUNS, a.out); os.makedirs(out, exist_ok=True)
    sysm, ck = load_system(a.core_run)
    for p in sysm.core.parameters():
        p.requires_grad_(False)
    fz = make_fz(sysm)
    fz_c = maybe_compile(fz)
    if a.set == "hardgoal":
        xs = dv.hardgoal_set(a.n, seed=31)[0]
    else:
        _, xg, mg, metas = eval_sets("val")
        xs = xg[:a.n]
    t0 = time.time()
    res = {"core_run": a.core_run, "U": a.U, "n": a.n, "set": a.set}
    flags = {k: [] for k in ["base"] + [f"gc_lr{lr}" for lr in a.lrs.split(",")] +
             ["random_dz", "gc_fresh_noise", "resample_prior", "token_T1",
              "base_from_mean", "gc_from_mean", "token_T1_from_mean"]}
    extra = {"attempts": {}, "dz_norm": {}}
    for i in range(0, len(xs), a.bs):
        xh = xs[i:i + a.bs]
        B = len(xh)
        ee = noise(B, 500 + i)[1]
        Z0 = base_controls(sysm, xh, 900 + i)
        x0 = generate(fz_c, xh, Z0, torch.zeros(B, Z0.shape[-1], device=DEV), ee)
        flags["base"] += verify_batch(x0).tolist()
        for lr in a.lrs.split(","):
            acc, att, dn = gradcuit(sysm, fz, fz_c, xh, Z0, ee, float(lr), a.U, seed=i)
            flags[f"gc_lr{lr}"] += acc.tolist()
            extra["attempts"].setdefault(f"gc_lr{lr}", []).extend(att.tolist())
            extra["dz_norm"].setdefault(f"gc_lr{lr}", []).extend(dn.tolist())
        best_lr = float(a.lrs.split(",")[-2]) if len(a.lrs.split(",")) > 1 else float(a.lrs)
        acc, _, _ = gradcuit(sysm, fz, fz_c, xh, Z0, ee, best_lr, a.U, random_ctrl=True, seed=i)
        flags["random_dz"] += acc.tolist()
        acc, _, _ = gradcuit(sysm, fz, fz_c, xh, Z0, ee, best_lr, a.U, fresh_noise=True, seed=i)
        flags["gc_fresh_noise"] += acc.tolist()
        acc, _ = resample_baseline(sysm, fz_c, xh, ee, a.U, seed=900 + i)
        flags["resample_prior"] += acc.tolist()
        flags["token_T1"] += token_sampling_baseline(fz_c, xh, Z0, ee, a.U, seed=i).tolist()
        # arms added 15:45 (pre-registered): start from the prior mean (eta_z = 0 -> Z0 = mu)
        Zm = torch.zeros_like(Z0)
        flags["base_from_mean"] += verify_batch(generate(fz_c, xh, Zm, torch.zeros(B, Z0.shape[-1], device=DEV), ee)).tolist()
        acc, _, _ = gradcuit(sysm, fz, fz_c, xh, Zm, ee, best_lr, a.U, seed=i)
        flags["gc_from_mean"] += acc.tolist()
        flags["token_T1_from_mean"] += token_sampling_baseline(fz_c, xh, Zm, ee, a.U, seed=i).tolist()
        print(f"batch {i}: " + json.dumps({k: float(np.mean(v)) for k, v in flags.items()}), flush=True)
    res["success"] = {k: summ(v) for k, v in flags.items()}
    res["paired"] = {k: {"vs_resample_prior": pair(v, flags["resample_prior"]), "vs_token_T1": pair(v, flags["token_T1"]),
                         "vs_random_dz": pair(v, flags["random_dz"]), "vs_base_from_mean": pair(v, flags["base_from_mean"]),
                         "vs_token_T1_from_mean": pair(v, flags["token_T1_from_mean"])}
                     for k, v in flags.items() if k.startswith("gc_")}
    res["mean_attempts"] = {k: float(np.mean(v)) for k, v in extra["attempts"].items()}
    res["mean_dz_norm"] = {k: float(np.mean(v)) for k, v in extra["dz_norm"].items()}
    res["flags"] = flags
    res["wall_s"] = time.time() - t0
    save_json(os.path.join(out, "phase_e.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k != "flags"}, indent=1))


if __name__ == "__main__":
    main()
