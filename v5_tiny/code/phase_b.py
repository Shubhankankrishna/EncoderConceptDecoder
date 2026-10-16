"""Phase B: train a replacement concept core against the frozen NCP encoder / PQ heads / codebook / decoder.

variants: det (V2), adapter (V2a), z (V3), gram (V4 = --xm 0, V6 = --xm 1), xmonly (V5: Z+eps prior only, XM loss).
usage: python phase_b.py --variant gram --xm 1 --teacher A_ncp_s0 --steps 6000 --out B_v6_s0
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

import data_v5 as dv
from model_v5 import GramCore, token_nll, concept_mask, random_windows, gauss_kl, count_params, K, D
from common_v5 import DEV, RUNS, Mixer, domain_mean, domain_mean_rows, cosine_lr, make_opt, Logger, save_json, B_TEXT
from system_v5 import System, load_ncp, window_tensors, chunk_sums

LAM_C = 1.0          # NCP-MSE weight (same as alpha in A)
BETA_Z = 0.1         # weighted ELBO (declared)
BETA_E = 0.1
LAM_X = 1.0


def kl_per_token(kl_rows, ntok_rows):
    """kl_rows [B] summed KL per row, ntok_rows [B] scored tokens -> domain-balanced per-token KL."""
    v = kl_rows / ntok_rows.clamp(min=1)
    return 0.5 * v[:B_TEXT].mean() + 0.5 * v[B_TEXT:].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="gram")
    ap.add_argument("--xm", type=int, default=1)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--teacher", default="A_ncp_s0")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--init", default="", help="Phase C: start from this run's core.pt")
    ap.add_argument("--distill", type=int, default=1, help="0 disables the distillation term (Phase C)")
    ap.add_argument("--lam_var", type=float, default=1.0, help="weight of the variational/reconstruction path")
    ap.add_argument("--beta_z", type=float, default=BETA_Z, help="KL_Z weight (slack arm: 1.0 = exact-ELBO weighting)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = os.path.join(RUNS, a.out); os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.jsonl"))
    teacher = load_ncp(os.path.join(RUNS, a.teacher, "ckpt.pt"))
    core_var = "gram" if a.variant == "xmonly" else a.variant
    core = GramCore(core_var).to(DEV)
    if a.init:
        ck0 = torch.load(os.path.join(RUNS, a.init, "core.pt"), map_location="cpu")
        assert ck0["variant"] == core_var
        core.load_state_dict(ck0["core"])
    sysm = System(teacher, core)
    print(f"{a.variant} core params {count_params(core):,}", flush=True)
    opt = make_opt(core.parameters(), a.lr)
    mix = Mixer(seed=2000 + a.seed)
    g = torch.Generator().manual_seed(a.seed)
    use_post = a.variant in ("z", "gram")          # variational path
    use_xm = (a.variant in ("gram", "z") and a.xm == 1) or a.variant == "xmonly"   # z+xm used by Phase C

    def dec_nll(H, chat, A, x, m):
        return token_nll(teacher.decode(H, chat, A), x, m)
    dnll = dec_nll if a.no_compile else torch.compile(dec_nll)

    def ctx(x):
        H = teacher.encode(x); C = teacher.pool(H)
        u_old = teacher.concept(C)
        return H, C, u_old, teacher.port(u_old)
    ctxf = ctx if a.no_compile else torch.compile(ctx)
    core_f = core if a.no_compile else torch.compile(core)

    t0 = time.time(); ntok = 0
    for step in range(a.steps):
        frac = step / a.steps
        delta = 1.0 if frac < 0.1 else max(0.0, 1.0 - (frac - 0.1) / 0.2)     # distillation weight schedule
        if not a.distill:
            delta = 0.0
        for gr in opt.param_groups:
            gr["lr"] = cosine_lr(step, a.steps, a.lr, warm=200)
        x, m, _ = mix.batch()
        B = x.shape[0]
        with torch.no_grad():
            H, C, u_old, A_old = ctxf(x)
        M = C.shape[1]
        cm = F.pad(concept_mask(m), (0, 1))                       # [B,M], last position has no target
        y = F.pad(C[:, 1:], (0, 0, 0, 1))                          # target c_{j+1}, zero at j=M-1
        ntok_rows = m[:, :-1].sum(1)
        e = core.in_proj(C)
        logs = {}
        loss = torch.zeros((), device=DEV)
        Z_pos = None
        if core.use_z:
            win, starts = random_windows(B, M, g, device=DEV)
            oh, first = window_tensors(win)
            wmask = (torch.einsum("bmw,bm->bw", oh, cm) > 0).float()
        # ---------------- reconstruction path (posterior for variational variants; deterministic otherwise)
        if a.variant != "xmonly":
            eps_mode = "post" if core.use_eps else "off"
            if core.use_z:
                mu_q, ls_q, _, _ = core.z_post(e, y, win)
                z_win = mu_q + ls_q.exp() * torch.randn_like(mu_q)
                Z_pos = torch.einsum("bmw,bwd->bmd", oh, z_win)
                mu_p, ls_p = core.z_prior(e, starts)
                mu_pw = torch.einsum("bmw,bmd->bwd", first, mu_p); ls_pw = torch.einsum("bmw,bmd->bwd", first, ls_p)
                klz = (gauss_kl(mu_q, ls_q, mu_pw, ls_pw).sum(-1) * wmask).sum(1)       # [B]
                l_klz = kl_per_token(klz, ntok_rows)
                loss = loss + a.lam_var * a.beta_z * l_klz; logs["kl_z"] = l_klz.item()
            u_new, A_new, kle = core_f(C, Z_pos, eps_mode=eps_mode, y=y, e=e)
            _, _, chat = teacher.pq(u_new)
            nll = dnll(H, chat, A_new, x, m)
            l_rec, lt, lg = domain_mean(nll, m)
            loss = loss + a.lam_var * l_rec
            logs.update(rec=l_rec.item(), text=lt.item(), graph=lg.item())
            if core.use_eps:
                l_kle = kl_per_token((kle * cm).sum(1), ntok_rows)
                loss = loss + a.lam_var * BETA_E * l_kle; logs["kl_eps"] = l_kle.item()
            err = (chat[:, :-1] - C[:, 1:]).pow(2).mean(-1)
            l_mse = domain_mean_rows(err, cm[:, :-1])
            dist = (u_new - u_old).pow(2).mean(-1) + (A_new - A_old).pow(2).mean(-1)
            l_dist = domain_mean_rows(dist, torch.ones_like(dist))
            loss = loss + LAM_C * l_mse + delta * l_dist
            logs.update(mse=l_mse.item(), dist=l_dist.item(), h_norm=u_new.norm(dim=-1).mean().item())
        # ---------------- Forward XM over prior-generated candidates (hard min per window, summed NLL)
        if use_xm:
            scores, nlls = [], []
            for i in range(a.K):
                eta_z = torch.randn(B, M, core.dz, device=DEV)
                Zc, _ = sysm.z_prior_schedule(e, starts, eta_z)
                u_c, A_c, _ = core_f(C, Zc, eps_mode="prior", e=e)
                _, _, chat_c = teacher.pq(u_c)
                nll_c = dnll(H, chat_c, A_c, x, m)
                sc = torch.einsum("bmw,bm->bw", oh, chunk_sums(nll_c, M))       # [B,nW] summed NLL per window
                scores.append(sc); nlls.append(nll_c)
            S = torch.stack(scores)                                           # [K,B,nW]
            win_idx = S.detach().argmin(0)                                    # [B,nW]
            sel = F.one_hot(win_idx, a.K).permute(2, 0, 1).float()            # [K,B,nW]
            l_xm_rows = (S * sel).sum((0, 2))                                 # [B] summed NLL of winners
            l_xm = kl_per_token(l_xm_rows, ntok_rows)
            loss = loss + LAM_X * l_xm
            logs.update(xm=l_xm.item(), xm_win_frac0=float((win_idx == 0).float().mean()))
            if a.variant == "xmonly":
                logs.update(rec=l_xm.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        opt.step()
        ntok += x.numel()
        if a.bench and step == 3:
            if DEV == "mps": torch.mps.synchronize()
            tb = time.time(); nb = ntok
        if a.bench and step == 3 + a.bench:
            if DEV == "mps": torch.mps.synchronize()
            print(f"BENCH {a.variant} xm={int(use_xm)}: {(ntok-nb)/(time.time()-tb):,.0f} tok/s, "
                  f"{(time.time()-tb)/a.bench*1000:.0f} ms/step", flush=True)
            return
        if step % 100 == 0:
            log(step=step, loss=loss.item(), gn=float(gn), delta=delta, tok_s=ntok / (time.time() - t0), **logs)
            if step % 500 == 0:
                print(f"step {step} loss {loss.item():.3f} {logs}", flush=True)
    torch.save({"core": core.state_dict(), "variant": core_var, "args": vars(a), "params": count_params(core)},
               os.path.join(out, "core.pt"))
    save_json(os.path.join(out, "train_summary.json"), dict(args=vars(a), params=count_params(core), tokens=ntok,
              wall_s=time.time() - t0))


if __name__ == "__main__":
    main()
