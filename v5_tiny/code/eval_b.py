"""Phase B/C evaluation of a replacement core.  usage: python eval_b.py --run B_v6_s0 [--lam 1.0] [--tag x]

NLL modes (z/gram variants): posterior (diagnostic), prior-sample x4 (deployable gate), prior-mean,
oracle best-of-8 per window (diagnostic).  Control tests on GraphNav (advisor consult 5 design):
hold vs resample, Z-swap, eps-only, with pre-drawn noise and greedy tokens.
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

import data_v5 as dv
from model_v5 import GramCore, token_nll, concept_mask, random_windows, K, D
from common_v5 import maybe_compile, DEV, RUNS, eval_sets, text_mask, save_json
from system_v5 import System, load_ncp, window_tensors, chunk_sums
from eval_a import per_item, paired, switch_masks
from common_v5 import Mixer, B_TEXT

M = dv.T // K
FIRST_WALK_BOUNDARY = dv.HEADER_CHUNKS - 1      # chunk 30 emits chat for the first hop


def load_system(run, teacher=None):
    ck = torch.load(os.path.join(RUNS, run, "core.pt"), map_location="cpu")
    teacher = teacher or ck["args"]["teacher"]
    t = load_ncp(os.path.join(RUNS, teacher, "ckpt.pt"))
    core = GramCore(ck["variant"])
    core.load_state_dict(ck["core"]); core.to(DEV).eval()
    for p in core.parameters():
        p.requires_grad_(False)
    return System(t, core), ck


def starts_hold(B, first=FIRST_WALK_BOUNDARY):
    s = torch.zeros(B, M, dtype=torch.bool, device=DEV); s[:, 0] = True; s[:, first] = True
    return s


def starts_hold4(B):
    """In-distribution schedule: a new window every 4 chunks from the first walk boundary (B trains lengths 1-4)."""
    s = torch.zeros(B, M, dtype=torch.bool, device=DEV); s[:, 0] = True
    s[:, FIRST_WALK_BOUNDARY::4] = True
    return s


def starts_every(B):
    return torch.ones(B, M, dtype=torch.bool, device=DEV)


def make_logits_fn(sysm, lam=1.0, compiled=True):
    """f(x, starts, eta_z, eta_eps, z_mode, eps_mode) -> logits [B,T,V].  z_mode: 'sample' | 'mean'."""
    core, t = sysm.core, sysm.t

    def f(x, starts, eta_z, eta_eps, z_mode="sample", eps_mode="prior"):
        H = t.encode(x); C = t.pool(H)
        e = core.in_proj(C)
        Z = sysm.z_prior_schedule(e, starts, eta_z, mode=z_mode)[0] if core.use_z else None
        if lam == 1.0:
            u, A, _ = core(C, Z, eps_mode=eps_mode, noise=eta_eps, e=e)
            _, _, chat = t.pq(u)
        else:
            chat, A, _ = sysm.plans(H, C, Z, eps_mode=eps_mode, noise=eta_eps, lam=lam)
        return t.decode(H, chat, A)
    return maybe_compile(f) if compiled else f


def noise(B, seed, dz=8, R=2):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, M, dz, generator=g).to(DEV), torch.randn(B, R, M, D, generator=g).to(DEV))


@torch.no_grad()
def nll_modes(sysm, xs, ms, f, lam, seed=0, n_prior=4, n_oracle=8):
    """Per-item results for each mode. Windows: fixed seeded segmentation (lengths 1-4)."""
    core = sysm.core
    out = {}
    if not core.use_z:
        out["det"] = per_item(lambda x, i: f(x, starts_every(x.shape[0]), *noise(x.shape[0], seed + i), "mean", "off"),
                              xs, ms)
        return out
    gw = torch.Generator().manual_seed(seed + 99)
    wins = {}

    def starts_for(i, B):
        if i not in wins:
            wins[i] = random_windows(B, M, gw, device=DEV)
        return wins[i][1]
    eps_prior = "prior" if core.use_eps else "off"
    acc = []
    for d in range(n_prior):
        acc.append(per_item(lambda x, i: f(x, starts_for(i, x.shape[0]), *noise(x.shape[0], seed + 1000 * d + i),
                                           "sample", eps_prior), xs, ms))
    out["prior_sample"] = {k: np.mean([a[k] for a in acc], 0) for k in acc[0]}
    out["prior_mean"] = per_item(lambda x, i: f(x, starts_for(i, x.shape[0]), *noise(x.shape[0], seed + i),
                                                "mean", "mean" if core.use_eps else "off"), xs, ms)
    # posterior (diagnostic; uses targets) and oracle best-of-n per window (diagnostic)
    post, orac = {"nll": [], "n": [], "col": [], "ncol": []}, {"nll": [], "n": []}
    mix = {"nll": [], "n": []}
    for i in range(0, len(xs), 64):
        x = torch.from_numpy(xs[i:i + 64]).to(DEV); m = torch.from_numpy(ms[i:i + 64]).to(DEV)
        B = x.shape[0]
        H, C = sysm.context(x)
        e = core.in_proj(C)
        st = starts_for(i, B); win = st.long().cumsum(1) - 1; oh, first = window_tensors(win)
        y = F.pad(C[:, 1:], (0, 0, 0, 1))
        mu_q, ls_q, _, _ = core.z_post(e, y, win)
        Zq = torch.einsum("bmw,bwd->bmd", oh, mu_q)
        u, A, _ = core(C, Zq, eps_mode="post" if core.use_eps else "off", y=y, noise=noise(B, seed + i)[1] * 0, e=e)
        _, _, chat = sysm.t.pq(u)
        nl = token_nll(sysm.t.decode(H, chat, A), x, m)
        tgt = x[:, 1:]
        iscol = ((tgt >= dv.COL0) & (tgt < dv.COL0 + dv.N_COL)).float() * m[:, :-1]
        post["nll"].append(nl.sum(1).cpu()); post["n"].append(m[:, :-1].sum(1).cpu())
        post["col"].append((nl * iscol).sum(1).cpu()); post["ncol"].append(iscol.sum(1).cpu())
        best = None
        wss = []
        for d in range(n_oracle):
            ez, ee = noise(B, seed + 7000 + 100 * d + i)
            nlc = token_nll(f(x, st, ez, ee, "sample", eps_prior), x, m)
            ws = torch.einsum("bmw,bm->bw", oh, chunk_sums(nlc, M))
            ws[:, 0] = ws[:, 0] + nlc[:, :3].sum(1)   # positions 0..2 read no plan; chunk_sums omits them (fixed 15:33)
            wss.append(ws)
            best = ws if best is None else torch.minimum(best, ws)
        orac["nll"].append(best.sum(1).cpu()); orac["n"].append(m[:, :-1].sum(1).cpu())
        # post-hoc diagnostic (12:12): per-window mixture NLL over the same 8 prior samples (XM smooth form)
        W = torch.stack(wss)                                              # [8,B,nW]
        mixw = -(torch.logsumexp(-W, 0) - np.log(len(wss)))
        mix["nll"].append(mixw.sum(1).cpu()); mix["n"].append(m[:, :-1].sum(1).cpu())
    out["posterior"] = {k: torch.cat(v).numpy() for k, v in post.items()}
    out["oracle_best_of_8"] = {k: torch.cat(v).numpy() for k, v in orac.items()}
    out["mixture_of_8_posthoc"] = {k: torch.cat(v).numpy() for k, v in mix.items()}
    return out


# ----------------------------------------------------------------------------- generation with a core
@torch.no_grad()
def generate(f, xg, starts_fn, seed, z_mode="sample", eps_mode="prior", temperature=0.0, bs=128, noise_fn=None):
    """Free generation from the header with a fixed noise draw per instance.  Returns tokens [N,T] (numpy)."""
    outs = []
    gen = torch.Generator().manual_seed(seed + 5)
    for i in range(0, len(xg), bs):
        x = torch.from_numpy(xg[i:i + bs].copy()); x[:, dv.WALK_START:] = dv.PAD; x = x.to(DEV)
        B = x.shape[0]
        ez, ee = noise_fn(B, i) if noise_fn else noise(B, seed + i)
        st = starts_fn(B)
        for pos in range(dv.WALK_START - 1, dv.T - 1):
            lg = f(x, st, ez, ee, z_mode, eps_mode)[:, pos].float()
            if temperature <= 0:
                nxt = lg.argmax(-1)
            else:
                nxt = torch.multinomial((lg / temperature).softmax(-1).cpu(), 1, generator=gen).squeeze(1).to(DEV)
            x[:, pos + 1] = nxt
            if pos % 4 == 3 and bool((x[:, dv.WALK_START:pos + 2] == dv.DONE).any(1).all()):
                break
        outs.append(x.cpu().numpy())
    return np.concatenate(outs)


def hop_colours(xrow):
    cols = []
    for ch in range(dv.MAX_HOPS):
        c4 = xrow[dv.WALK_START + 4 * ch: dv.WALK_START + 4 * ch + 4]
        if len(c4) < 4 or not (dv.COL0 <= c4[0] < dv.COL0 + dv.N_COL) or c4[1] != dv.TO:
            break
        cols.append(int(c4[0]))
    return cols


def walk_stats(X, return_persist=False):
    succ, cons, n_pairs, n_hops, per = [], 0, 0, [], []
    for r in X:
        ok = dv.verify(r[:dv.WALK_START], r[dv.WALK_START:])
        succ.append(float(ok[0]))
        c = hop_colours(r); n_hops.append(len(c))
        same = sum(c[j] == c[j + 1] for j in range(len(c) - 1)); pairs = max(0, len(c) - 1)
        cons += same; n_pairs += pairs
        per.append(same / pairs if pairs > 0 else np.nan)
    s = np.array(succ)
    out = (dict(success=float(s.mean()), success_se=float(s.std(ddof=1) / np.sqrt(len(s))),
                colour_persistence=cons / max(n_pairs, 1), mean_hops=float(np.mean(n_hops))), s)
    return out + (np.array(per),) if return_persist else out


def first_colour(X):
    return np.array([X[r, dv.WALK_START] for r in range(len(X))])


def control_tests(sysm, f, xg, seed=0):
    """P-control (pre-registered) + Z-swap / eps-only effects.  Greedy tokens, pre-drawn noise."""
    core = sysm.core
    res = {}
    eps_mode = "prior" if core.use_eps else "off"
    Xh = generate(f, xg, starts_hold, seed, "sample", eps_mode)
    X4 = generate(f, xg, starts_hold4, seed, "sample", eps_mode)
    Xr = generate(f, xg, starts_every, seed, "sample", eps_mode)
    sh, flag_h = walk_stats(Xh); s4, flag_4, per4 = walk_stats(X4, True); sr, flag_r, perr = walk_stats(Xr, True)
    res["hold_stress"] = sh; res["hold4"] = s4; res["resample"] = sr
    d = flag_4 - flag_r
    res["hold4_minus_resample_success"] = dict(delta=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))
    # P-control: within-window colour persistence, hold4 vs resample (pairs of consecutive hops)
    res["P_control_persistence_hold4_minus_resample"] = s4["colour_persistence"] - sr["colour_persistence"]
    ok = ~np.isnan(per4) & ~np.isnan(perr)
    dp = per4[ok] - perr[ok]
    res["P_control_persistence_paired"] = dict(delta=float(dp.mean()), se=float(dp.std(ddof=1) / np.sqrt(max(len(dp), 2))),
                                               n=int(ok.sum()))
    if core.use_z:
        # Z-swap: same eps, different Z draw
        nz_a = lambda B, i: noise(B, seed + i)
        nz_b = lambda B, i: (noise(B, seed + 50000 + i)[0], noise(B, seed + i)[1])
        Xa = generate(f, xg, starts_hold4, seed, "sample", eps_mode, noise_fn=nz_a)
        Xb = generate(f, xg, starts_hold4, seed, "sample", eps_mode, noise_fn=nz_b)
        diff_z = (first_colour(Xa) != first_colour(Xb)).astype(float)
        res["zswap_first_colour_differs"] = dict(rate=float(diff_z.mean()), se=float(diff_z.std(ddof=1) / np.sqrt(len(diff_z))))
        seq_diff = [float(hop_colours(a) != hop_colours(b)) for a, b in zip(Xa, Xb)]
        res["zswap_colour_sequence_differs"] = float(np.mean(seq_diff))
        if core.use_eps:
            nz_c = lambda B, i: (noise(B, seed + i)[0], noise(B, seed + 60000 + i)[1])
            Xc = generate(f, xg, starts_hold4, seed, "sample", eps_mode, noise_fn=nz_c)
            diff_e = (first_colour(Xa) != first_colour(Xc)).astype(float)
            res["epsonly_first_colour_differs"] = dict(rate=float(diff_e.mean()), se=float(diff_e.std(ddof=1) / np.sqrt(len(diff_e))))
            res["epsonly_colour_sequence_differs"] = float(np.mean([float(hop_colours(a) != hop_colours(c)) for a, c in zip(Xa, Xc)]))
            dd = diff_z - diff_e
            res["zswap_minus_epsonly_paired"] = dict(delta=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(len(dd))))
        Xm = generate(f, xg, starts_hold4, seed, "mean", "mean" if core.use_eps else "off")
        res["prior_mean_greedy"] = walk_stats(Xm)[0]
    return res


@torch.no_grad()
def route_split(sysm, xt, mt, xg, mg, metas, seed=0):
    """Replacement-route authority (deterministic path: prior-mean Z, eps mean): replace chat (pq_mean) or A (crc_mean)
    or both (mean) by the replacement's own training-data means per domain (20 train batches, Mixer seed 777)."""
    core, t = sysm.core, sysm.t
    eps_m = "mean" if core.use_eps else "off"

    def plans(x):
        B = x.shape[0]
        H, C = sysm.context(x)
        e = core.in_proj(C)
        Z = sysm.z_prior_schedule(e, starts_every(B), None, mode="mean")[0] if core.use_z else None
        u, A, _ = core(C, Z, eps_mode=eps_m, e=e)
        _, _, chat = t.pq(u)
        return H, chat, A
    mix = Mixer(seed=777)
    acc = {d: [0, 0, 0] for d in ("text", "graph")}
    for _ in range(20):
        x, m, _ = mix.batch()
        H, chat, A = plans(x)
        cm = F.pad(concept_mask(m), (0, 1))
        for d, sl in (("text", slice(0, B_TEXT)), ("graph", slice(B_TEXT, None))):
            w = cm[sl].unsqueeze(-1)
            acc[d][0] = acc[d][0] + (chat[sl] * w).sum((0, 1)); acc[d][1] = acc[d][1] + (A[sl] * w).sum((0, 1))
            acc[d][2] += w.sum().item()
    stats = {d: (v[0] / v[2], v[1] / v[2]) for d, v in acc.items()}
    swm, nsm = switch_masks(metas)
    out = {}
    for name, xs, ms, dom, kw in (("text", xt, mt, "text", {}), ("graph", xg, mg, "graph", dict(sw=swm, ns=nsm))):
        mc, mA = stats[dom]
        res = {}
        for mode in ("normal", "pq_mean", "crc_mean", "mean"):
            def fwd(x, i, mode=mode):
                H, chat, A = plans(x)
                if mode in ("pq_mean", "mean"):
                    chat = mc.expand_as(chat)
                if mode in ("crc_mean", "mean"):
                    A = mA.expand_as(A)
                return t.decode(H, chat, A)
            res[mode] = per_item(fwd, xs, ms, **kw)
        for mode in ("pq_mean", "crc_mean", "mean"):
            r = {"nll": paired(res["normal"], res[mode])}
            if dom == "graph":
                r["colour_at_switch"] = paired(res["normal"], res[mode], "sw", "nsw")
            out[f"{name}_{mode}"] = r
    return out


@torch.no_grad()
def hardgoal_success(sysm, f, n=512, seed=0):
    """Hard-goal success with hold4 prior samples and with the prior mean (per-instance flags saved for pairing)."""
    xh, metas = dv.hardgoal_set(n, seed=31)
    core = sysm.core
    out = {}
    for mode in ("sample", "mean"):
        eps_mode = ("prior" if mode == "sample" else "mean") if core.use_eps else "off"
        X = generate(f, xh, starts_hold4, seed, mode, eps_mode)
        s = np.array([float(dv.verify(r[:dv.WALK_START], r[dv.WALK_START:])[0]) for r in X])
        out[mode] = dict(success=float(s.mean()), se=float(s.std(ddof=1) / np.sqrt(len(s))), flags=s.tolist())
    out["success"], out["se"] = out["sample"]["success"], out["sample"]["se"]
    return out


@torch.no_grad()
def raw_norms(sysm, xs, seed=0):
    core = sysm.core
    if core.variant == "adapter":
        return None
    core.track_norm = True
    x = torch.from_numpy(xs[:64]).to(DEV)
    B = x.shape[0]
    H, C = sysm.context(x)
    e = core.in_proj(C)
    Z = sysm.z_prior_schedule(e, starts_every(B), noise(B, seed)[0])[0] if core.use_z else None
    core(C, Z, eps_mode="prior" if core.use_eps else "off", noise=noise(B, seed)[1], e=e)
    core.track_norm = False
    return list(core._hnorm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n_gen", type=int, default=512)
    ap.add_argument("--no_control", action="store_true")
    a = ap.parse_args()
    sysm, ck = load_system(a.run)
    xt, xg, mg, metas = eval_sets(a.split)
    mt = np.stack([text_mask(r[None])[0] for r in xt])
    f = make_logits_fn(sysm, lam=a.lam)
    res = {"run": a.run, "variant": ck["variant"], "lam": a.lam, "core_params": ck["params"], "oracle_window_fix": True}
    # teacher reference on the same items
    fa = maybe_compile(lambda x: sysm.t(x)["logits"])
    ref_t, ref_g = per_item(lambda x, i: fa(x), xt, mt), per_item(lambda x, i: fa(x), xg, mg)
    res["teacher"] = dict(text=float(ref_t["nll"].sum() / ref_t["n"].sum()), graph=float(ref_g["nll"].sum() / ref_g["n"].sum()))
    mt_ = nll_modes(sysm, xt, mt, f, a.lam)
    mg_ = nll_modes(sysm, xg, mg, f, a.lam, seed=1)
    res["nll"] = {}
    for mode in mt_:
        r = dict(text=paired(ref_t, mt_[mode]), graph=paired(ref_g, mg_[mode]))
        if "col" in mg_[mode]:
            r["colour"] = paired(ref_g, mg_[mode], "col", "ncol")
        res["nll"][mode] = r
    gate_mode = "prior_sample" if "prior_sample" in res["nll"] else "det"
    g = res["nll"][gate_mode]
    res["gate"] = dict(mode=gate_mode, text_ok=bool(g["text"]["delta"] <= 0.05), graph_ok=bool(g["graph"]["delta"] <= 0.05))
    res["raw_h_norm_per_transition"] = raw_norms(sysm, xt)
    if a.lam == 1.0:
        res["route_split"] = route_split(sysm, xt, mt, xg, mg, metas)
    if not a.no_control:
        res["control"] = control_tests(sysm, f, xg[:a.n_gen])
        res["hardgoal_hold4"] = hardgoal_success(sysm, f)
    tag = a.tag or f"lam{a.lam}"
    save_json(os.path.join(RUNS, a.run, f"eval_b_{a.split}_{tag}.json"), res)
    print(json.dumps(res, indent=1)[:8000])


if __name__ == "__main__":
    main()
