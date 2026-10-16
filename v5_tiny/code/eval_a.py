"""Phase A evaluation (pre-registered measurements).  usage: python eval_a.py --run A_ncp_s0 [--split val]"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

import data_v5 as dv
from model_v5 import NCPModel, TransformerLM, token_nll, concept_mask, rms, K, D, S_PQ, V_PQ
from common_v5 import maybe_compile, DEV, RUNS, Mixer, B_TEXT, eval_sets, text_mask, generate_walks, save_json


def load(run):
    ck = torch.load(os.path.join(RUNS, run, "ckpt.pt"), map_location="cpu")
    m = NCPModel(dv.VOCAB) if ck["arch"] == "ncp" else TransformerLM(dv.VOCAB)
    m.load_state_dict(ck["model"]); m.to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck["arch"]


@torch.no_grad()
def plan_stats(model, n_batches=50, seed=777):
    """Training-data means of chat and A per domain (pre-registered: 50 train batches, Mixer seed 777).
    Graph means use only positions whose target chunk is scored (walk/done)."""
    mix = Mixer(seed=seed)
    acc = {d: [torch.zeros(D, device=DEV), torch.zeros(D, device=DEV), 0.0] for d in ("text", "graph")}
    for _ in range(n_batches):
        x, m, _ = mix.batch()
        o = model(x)
        cm = F.pad(concept_mask(m), (0, 1))
        for d, sl in (("text", slice(0, B_TEXT)), ("graph", slice(B_TEXT, None))):
            w = cm[sl].unsqueeze(-1)
            acc[d][0] += (o["chat"][sl] * w).sum((0, 1)); acc[d][1] += (o["A"][sl] * w).sum((0, 1))
            acc[d][2] += w.sum().item()
    return {d: {"chat": v[0] / v[2], "A": v[1] / v[2]} for d, v in acc.items()}


def derangement(n, gen):
    while True:
        p = torch.randperm(n, generator=gen)
        if (p != torch.arange(n)).all():
            return p


@torch.no_grad()
def switch_masks(metas, T=dv.T):
    """[N,T-1] masks over input positions predicting colour tokens at switch hops / non-switch hops (h >= 1)."""
    sw = np.zeros((len(metas), T - 1), np.float32); ns = np.zeros_like(sw)
    for r, mt in enumerate(metas):
        hc = mt["hop_cols"]
        for h in range(1, len(hc)):
            pos = dv.WALK_START + 4 * h - 1
            (sw if hc[h] != hc[h - 1] else ns)[r, pos] = 1.0
    return sw, ns


def per_item(fwd, xs, ms, bs=64, sw=None, ns=None):
    """Per-row (NLL sum, token count) arrays, plus colour/node sums for graph rows."""
    out = {"nll": [], "n": [], "col": [], "ncol": [], "sw": [], "nsw": [], "ns": [], "nns": []}
    for i in range(0, len(xs), bs):
        x = torch.from_numpy(xs[i:i + bs]).to(DEV); m = torch.from_numpy(ms[i:i + bs]).to(DEV)
        nl = token_nll(fwd(x, i), x, m)
        tgt = x[:, 1:]
        iscol = ((tgt >= dv.COL0) & (tgt < dv.COL0 + dv.N_COL)).float() * m[:, :-1]
        out["nll"].append(nl.sum(1).cpu()); out["n"].append(m[:, :-1].sum(1).cpu())
        out["col"].append((nl * iscol).sum(1).cpu()); out["ncol"].append(iscol.sum(1).cpu())
        if sw is not None:
            w1 = torch.from_numpy(sw[i:i + bs]).to(DEV); w2 = torch.from_numpy(ns[i:i + bs]).to(DEV)
            out["sw"].append((nl * w1).sum(1).cpu()); out["nsw"].append(w1.sum(1).cpu())
            out["ns"].append((nl * w2).sum(1).cpu()); out["nns"].append(w2.sum(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items() if v}


def paired(a, b, key="nll", den="n"):
    """Mean per-token delta (b - a) with SE over items (per-item token-weighted means)."""
    da = a[key] / np.maximum(a[den], 1); db = b[key] / np.maximum(b[den], 1)
    ok = a[den] > 0
    d = (db - da)[ok]
    return dict(delta=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), base=float(da[ok].mean()))


@torch.no_grad()
def predicted_history_logits(model, x):
    """NCP inference protocol: the concept module consumes previously predicted concepts (position 0 = actual c_0)."""
    H = model.encode(x); C = model.pool(H)
    Cin = C.clone()
    M = C.shape[1]
    for j in range(1, M):
        u = model.concept(Cin[:, :j])
        _, _, chat = model.pq(u)
        Cin[:, j] = chat[:, j - 1]
    u = model.concept(Cin)
    _, _, chat = model.pq(u)
    return model.decode(H, chat, model.port(u))


@torch.no_grad()
def forecast_baselines(model, xt, xg, mg, n_fit=30):
    """MSE of chat_j vs c_{j+1} on val (scored targets) against copy-last, train-mean and a ridge linear predictor."""
    mix = Mixer(seed=778)
    Xs, Ys = [], []
    for _ in range(n_fit):
        x, m, _ = mix.batch()
        C = model.pool(model.encode(x)); cm = concept_mask(m).bool()
        Xs.append(C[:, :-1][cm]); Ys.append(C[:, 1:][cm])
    X = torch.cat(Xs).cpu().double(); Y = torch.cat(Ys).cpu().double()
    Xb = torch.cat([X, torch.ones(len(X), 1, device=X.device, dtype=X.dtype)], 1)
    W = torch.linalg.solve(Xb.t() @ Xb + 1e-2 * torch.eye(Xb.shape[1], device=X.device, dtype=X.dtype), Xb.t() @ Y)
    mean_c = Y.mean(0)
    res = {}
    for name, xs, ms in (("text", xt, np.stack([text_mask(r[None])[0] for r in xt])), ("graph", xg, mg)):
        acc = {"ncp": 0.0, "copy": 0.0, "mean": 0.0, "linear": 0.0, "n": 0.0}
        for i in range(0, len(xs), 64):
            x = torch.from_numpy(xs[i:i + 64]).to(DEV); m = torch.from_numpy(ms[i:i + 64]).to(DEV)
            o = model(x); C = o["C"]; cm = concept_mask(m).bool()
            tgt = C[:, 1:][cm].cpu().double()
            prev = C[:, :-1][cm].cpu().double()
            preds = {"ncp": o["chat"][:, :-1][cm].cpu().double(), "copy": prev,
                     "mean": mean_c.expand_as(tgt),
                     "linear": torch.cat([prev, torch.ones(len(tgt), 1, dtype=torch.float64)], 1) @ W}
            for k, p in preds.items():
                acc[k] += (p - tgt).pow(2).mean(-1).sum().item()
            acc["n"] += len(tgt)
        res[name] = {k: acc[k] / acc["n"] for k in ("ncp", "copy", "mean", "linear")}
    return res


@torch.no_grad()
def codebook_usage(model, xt):
    cnt = torch.zeros(S_PQ, V_PQ, device=DEV)
    for i in range(0, len(xt), 64):
        x = torch.from_numpy(xt[i:i + 64]).to(DEV)
        q, _ = model.vq(model.pool(model.encode(x)))
        model.count_usage(q) if False else None
        flat = (q.reshape(-1, S_PQ) + torch.arange(S_PQ, device=DEV) * V_PQ).flatten()
        cnt += torch.bincount(flat, minlength=S_PQ * V_PQ).view(S_PQ, V_PQ).float()
    p = cnt / cnt.sum(1, keepdim=True)
    ppl = torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum(1))
    return dict(used=int((cnt > 0).sum()), of=S_PQ * V_PQ, perplexity_per_segment=ppl.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_gen", type=int, default=512)
    a = ap.parse_args()
    model, arch = load(a.run)
    xt, xg, mg, metas = eval_sets(a.split)
    mt = np.stack([text_mask(r[None])[0] for r in xt])
    res = {"run": a.run, "arch": arch, "split": a.split}
    if arch == "ncp":
        fwd_logits = maybe_compile(lambda x: model(x)["logits"])
    else:
        fwd_logits = maybe_compile(lambda x: model(x))
    base_t = per_item(lambda x, i: fwd_logits(x), xt, mt)
    swm, nsm = switch_masks(metas)
    base_g = per_item(lambda x, i: fwd_logits(x), xg, mg, sw=swm, ns=nsm)
    res["nll"] = dict(text=float(base_t["nll"].sum() / base_t["n"].sum()),
                      graph=float(base_g["nll"].sum() / base_g["n"].sum()),
                      colour=float(base_g["col"].sum() / base_g["ncol"].sum()),
                      colour_at_switch=float(base_g["sw"].sum() / base_g["nsw"].sum()),
                      colour_non_switch=float(base_g["ns"].sum() / base_g["nns"].sum()))
    from common_v5 import eval_nll
    res["nll_split"] = eval_nll(fwd_logits, a.split)
    # free generation
    step = lambda x, pos: fwd_logits(x)[:, pos]
    res["gen_greedy"] = generate_walks(step, xg[:a.n_gen], temperature=0.0)
    res["gen_t1"] = generate_walks(step, xg[:a.n_gen], temperature=1.0, seed=1)
    for k in ("gen_greedy", "gen_t1"):
        res[k + "_success_flags"] = res[k].pop("success_flags")
    if arch == "ncp":
        ps = plan_stats(model)
        gen = torch.Generator().manual_seed(5)
        perms_t = [derangement(min(64, len(xt) - i), gen) for i in range(0, len(xt), 64)]
        perms_g = [derangement(min(64, len(xg) - i), gen) for i in range(0, len(xg), 64)]
        route = {}
        for mode in ("mean", "swap", "pq_mean", "crc_mean", "zero"):
            ft = lambda x, i, mode=mode: model(x, plan_mode=mode, plan_stats=ps["text"], perm=perms_t[i // 64])["logits"]
            fg = lambda x, i, mode=mode: model(x, plan_mode=mode, plan_stats=ps["graph"], perm=perms_g[i // 64])["logits"]
            it, ig = per_item(ft, xt, mt), per_item(fg, xg, mg, sw=swm, ns=nsm)
            route[mode] = dict(text=paired(base_t, it), graph=paired(base_g, ig),
                               colour=paired(base_g, ig, "col", "ncol"),
                               colour_at_switch=paired(base_g, ig, "sw", "nsw"),
                               colour_non_switch=paired(base_g, ig, "ns", "nns"))
        res["route"] = route
        g = res["route"]["mean"]
        res["route_gate"] = dict(
            text_used=bool(g["text"]["delta"] >= 0.02 and g["text"]["delta"] > 2 * g["text"]["se"]),
            graph_used=bool(g["graph"]["delta"] >= 0.05 and g["graph"]["delta"] > 2 * g["graph"]["se"]))
        if os.environ.get("SKIP_PH"):
            ph_t = ph_g = None
        ph_t = ph_t if os.environ.get("SKIP_PH") else per_item(lambda x, i: predicted_history_logits(model, x), xt, mt)
        if not os.environ.get("SKIP_PH"):
            ph_g = per_item(lambda x, i: predicted_history_logits(model, x), xg, mg)
            res["predicted_history"] = dict(text=paired(base_t, ph_t), graph=paired(base_g, ph_g))
        res["forecast"] = forecast_baselines(model, xt, xg, mg)
        res["codebook"] = codebook_usage(model, xt)
    save_json(os.path.join(RUNS, a.run, f"eval_a_{a.split}.json"), res)
    brief = {k: v for k, v in res.items() if not k.endswith("flags")}
    import json
    print(json.dumps(brief, indent=1)[:6000])


if __name__ == "__main__":
    main()
