"""Analysis of the v4-vs-LLM arms (CPU, reads checkpoints; never touches v3). Implements VS_LLM_REPORT.md §3 as
pre-registered: paired per-row / per-item / per-episode / per-layout readings, closed-loop rollouts (P2), linear
probes (P3) and v4's coarse-code NMI.

python analyze_vs.py arm <run name>          -> runs/<name>/analysis.json (+ per_unit.npz)
python analyze_vs.py compare                 -> runs/vs_compare.json, runs/vs_compare.md
"""
import json
import math
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

import model_v4 as M4
import data_v4 as D
import pinpad as PP
import run_vs as R

RUNS = R.RUNS
T = 480


def load(name):
    ck = torch.load(os.path.join(RUNS, name, "ckpt.pt"), map_location="cpu", weights_only=False)
    a = ck["args"]
    V = 8192 if a["data"] == "text" else PP.V
    over = {}
    for kv in filter(None, a.get("over", "").split(",")):
        kk, vv = kv.split("="); over[kk] = eval(vv)
    m = R.build(a["arch"], V, over)
    m.load_state_dict(ck["model"])
    return m.eval(), a


# ----------------------------------------------------------------------------- text: per row / per item
@torch.no_grad()
def text_units(m, a):
    S = R.text_setup(R.get_args(["--name", "x", "--arch", a["arch"], "--data", "text", "--seed", str(a["seed"])]))
    t, d = S["vrows"]
    rows = []
    for i in range(0, len(t), 8):
        ids, doc, ei = D.to_batch(t[i:i + 8], d[i:i + 8], S["canon"], m.ecfg)
        lg = R.fwd(m, ids, doc, ei)[0]
        tm = M4.target_mask(ids, doc)
        nll = F.cross_entropy(lg[:, :-1].transpose(1, 2), ids[:, 1:].masked_fill(~tm, 0), reduction="none")
        rows += ((nll * tm).sum(1) / tm.sum(1).clamp_min(1)).tolist()
    items = S["items_f"]
    hits = item_hits(m, items, S["canon"])
    fam = np.array([it["family"] for it in items])
    return dict(row_ntp=np.array(rows), item_hit=hits, item_family=fam)


@torch.no_grad()
def item_hits(m, items, canon, **kw):
    t, d, where = D.item_rows(items, T, R.K)
    by_row = {}
    for idx, (r, off, L) in enumerate(where):
        by_row.setdefault(r, []).append((idx, off))
    hit = np.zeros(len(items))
    for a in range(0, len(t), 8):
        rs = list(range(a, min(a + 8, len(t))))
        ids, doc, ei = D.to_batch(t[rs], d[rs], canon, m.ecfg)
        pred = R.fwd(m, ids, doc, ei, **kw)[0].argmax(-1)
        for bi, r in enumerate(rs):
            for idx, off in by_row.get(r, []):
                ok = True
                for (s, e, _) in items[idx]["spans"]:
                    ok &= bool((pred[bi, off + s - 1: off + e - 1] == ids[bi, off + s: off + e]).all())
                hit[idx] = ok
    return hit


# ----------------------------------------------------------------------------- pinpad: per episode
@torch.no_grad()
def pin_units(m, eps, **kw):
    """Per-episode: optimal-action rate (all actions; flag-2 actions), action NTP, position accuracy."""
    t, d, where = D.pack_sequences([e["tokens"] for e in eps], T, R.K)
    canon = PP.Ident()
    n = len(eps)
    keys = ["act_opt", "act_opt_b1", "act_opt_b2", "act_opt_b0", "ntp_action", "pos_acc"]
    keys += [f"{m_}_{z}" for z in ("near", "partial", "far") for m_ in ("act_opt", "act_opt_b2", "ntp_action")]
    keys += [f"n_{z}" for z in ("near", "partial", "far")]
    out = {k: np.full(n, np.nan) for k in keys}
    by_row = {}
    for ei_, (r, off, L) in enumerate(where):
        by_row.setdefault(r, []).append((ei_, off, L))
    for a in range(0, len(t), 8):
        rs = list(range(a, min(a + 8, len(t))))
        if kw.get("plan_swap") and len(rs) < 2:
            rs = [rs[0] - 1, rs[0]]
        ids, doc, eix = D.to_batch(t[rs], d[rs], canon, m.ecfg)
        lg = R.fwd(m, ids, doc, eix, **kw)[0]
        lp = F.log_softmax(lg.double(), -1)
        for bi, r in enumerate(rs):
            if r < a:
                continue
            for e_i, off, L in by_row.get(r, []):
                ep = eps[e_i]; bf = PP.boundary_flags(ep)
                ok, okb, nl, pa = [], {0: [], 1: [], 2: []}, [], []
                zone = {z: dict(o=[], b2=[], nl=[]) for z in ("near", "partial", "far")}
                for i in range(27, L):
                    p = off + i
                    if ep["kind"][i] == 1:
                        o = int(lg[bi, p - 1, PP.ACT0:PP.ACT0 + 4].argmax()) in ep["opt"][i]
                        ok.append(o); okb[bf[i]].append(o)
                        nl.append(-float(lp[bi, p - 1, ep["tokens"][i]]))
                        z = zone["near" if i <= 70 else ("far" if i >= 97 else "partial")]
                        z["o"].append(o); z["nl"].append(nl[-1])
                        if bf[i] == 2:
                            z["b2"].append(o)
                    elif ep["kind"][i] == 2:
                        pa.append(int(lg[bi, p - 1].argmax()) == ep["tokens"][i])
                out["act_opt"][e_i] = np.mean(ok); out["ntp_action"][e_i] = np.mean(nl); out["pos_acc"][e_i] = np.mean(pa)
                for b in (0, 1, 2):
                    if okb[b]:
                        out[f"act_opt_b{b}"][e_i] = np.mean(okb[b])
                for zn, z in zone.items():                                # reach diagnostic (§6)
                    out[f"n_{zn}"][e_i] = len(z["o"])
                    if z["o"]:
                        out[f"act_opt_{zn}"][e_i] = np.mean(z["o"]); out[f"ntp_action_{zn}"][e_i] = np.mean(z["nl"])
                    if z["b2"]:
                        out[f"act_opt_b2_{zn}"][e_i] = np.mean(z["b2"])
    return out


# ----------------------------------------------------------------------------- P2 rollouts
ROLL_N, ROLL_SEED = 256, 77


def roll_layouts():
    rng = random.Random(ROLL_SEED)
    out = []
    while len(out) < ROLL_N:
        colors, walls, start = PP.sample_layout(rng)
        out.append(dict(colors=list(colors), walls=sorted(walls), start=start))
    return out


@torch.no_grad()
def rollouts(m, greedy, seed=0):
    """All 256 layouts in one batch; finished rows are dropped from the forward (they are never read again)."""
    lays = roll_layouts()
    g = torch.Generator().manual_seed(1234 + seed)
    envs = [PP.Env(l) for l in lays]
    seqs = [PP.header_tokens(l["colors"], set(l["walls"]), l["start"]) for l in lays]
    active = list(range(len(lays)))
    for stepi in range(PP.TMAX):
        if not active:
            break
        n = len(seqs[active[0]]); Tp = -(-n // R.K) * R.K
        ids = torch.zeros(len(active), Tp, dtype=torch.long)
        ids[:, :n] = torch.tensor([seqs[j] for j in active])
        doc = torch.zeros_like(ids)
        ei = D.to_batch(ids.numpy(), doc.numpy(), PP.Ident(), m.ecfg)[2]
        lg = R.fwd(m, ids, doc, ei)[0][:, n - 1, PP.ACT0:PP.ACT0 + 4].double()
        acts = lg.argmax(-1) if greedy else torch.multinomial(F.softmax(lg, -1), 1, generator=g).squeeze(-1)
        still = []
        for row, j in enumerate(active):
            cell = envs[j].act(int(acts[row]))
            seqs[j] += [PP.ACT0 + int(acts[row]), PP.pos_tok(cell)]
            if len(envs[j].visits) < 6:
                still.append(j)
        active = still
    res = []
    for env in envs:
        v = tuple(env.visits[:6])
        ev = env.visits
        pairs = [(ev[i], ev[i + 1]) for i in range(len(ev) - 1) if ev[i] % 2 == 0]
        res.append(dict(task=float(len(v) == 6 and v in PP.TRAIN_TASKS),
                        comp=float(len(v) == 6 and v in PP.ALL3),
                        complete=float(len(v) == 6),
                        pair_ok=float(np.mean([b == a + 1 for a, b in pairs])) if pairs else np.nan,
                        steps=env.steps))
    return {k: np.array([r[k] for r in res], float) for k in res[0]}


# ----------------------------------------------------------------------------- P3 probes
@torch.no_grad()
def features(m, eps):
    """Causal feature sites at positions whose next token is an action. Returns {site: (N, d)}, sub (N,), tgt (N,)."""
    t, d, where = D.pack_sequences([e["tokens"] for e in eps], T, R.K)
    feats, subs, tgts = {}, [], []
    for a in range(0, len(t), 8):
        ids, doc, ei = D.to_batch(t[a:a + 8], d[a:a + 8], PP.Ident(), m.ecfg)
        if R.is_v4(m):
            lg, aux = R.fwd(m, ids, doc, ei)
            sites = dict(emb=M4.rms(m.enc.embed(ids)), enc_L=aux["tok"], enc_G=aux["enc"]["G"], plan=aux["P"], dec_h=aux["h"])
        else:
            lg, aux = R.fwd(m, ids, doc, ei, collect=True)
            sites = {("emb" if i == 0 else f"L{i:02d}"): s for i, s in enumerate(aux["states"])}
        sel = []
        for e_i, (r, off, L) in enumerate(where):
            if not (a <= r < a + 8):
                continue
            ep = eps[e_i]
            for i in range(27, L):
                if ep["kind"][i] == 1:
                    sel.append((r - a, off + i - 1)); subs.append(ep["sub"][i]); tgts.append(ep["tgt"][i])
        bi = torch.tensor([s[0] for s in sel]); pi = torch.tensor([s[1] for s in sel])
        for k, v in sites.items():
            feats.setdefault(k, []).append(v[bi, pi].float())
    return {k: torch.cat(v) for k, v in feats.items()}, torch.tensor(subs), torch.tensor(tgts)


def probe(Xtr, ytr, tests, ncls, steps=500):
    torch.manual_seed(0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    W = torch.zeros(Xtr.shape[1], ncls, requires_grad=True); b = torch.zeros(ncls, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=1e-2, weight_decay=1e-3)
    X = (Xtr - mu) / sd
    for _ in range(steps):
        loss = F.cross_entropy(X @ W + b, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return [float((((Xt - mu) / sd @ W + b).argmax(-1) == yt).float().mean()) for Xt, yt in tests]


def probes(m):
    tr = PP.eval_episodes("probe", 400, 0)
    te_in = PP.eval_episodes("heldin", 200, 9)
    te_out = PP.eval_episodes("heldout", 200, 9)
    Ftr, s_tr, g_tr = features(m, tr)
    Fin, s_in, g_in = features(m, te_in)
    Fout, s_out, g_out = features(m, te_out)
    out = {}
    for k in Ftr:
        a_sub = probe(Ftr[k], s_tr, [(Fin[k], s_in), (Fout[k], s_out)], 4)
        a_tgt = probe(Ftr[k], g_tr, [(Fin[k], g_in), (Fout[k], g_out)], 8)
        out[k] = dict(sub_heldin=a_sub[0], sub_heldout=a_sub[1], tgt_heldin=a_tgt[0], tgt_heldout=a_tgt[1])
    maj = lambda y: float(np.bincount(y.numpy()).max() / len(y))
    out["_majority"] = dict(sub_heldin=maj(s_in), sub_heldout=maj(s_out), tgt_heldin=maj(g_in), tgt_heldout=maj(g_out))
    return out


@torch.no_grad()
def code_nmi(m):
    """v4 only, non-causal: NMI between the coarse code of each complete chunk and the chunk's majority subgoal."""
    eps = PP.eval_episodes("heldin", 200, 9)
    t, d, where = D.pack_sequences([e["tokens"] for e in eps], T, R.K)
    lab = np.full(t.shape, -1)
    for e, (r, off, L) in zip(eps, where):
        lab[r, off:off + L] = e["sub"]
    codes, labs = [], []
    for a in range(0, len(t), 8):
        ids, doc, ei = D.to_batch(t[a:a + 8], d[a:a + 8], PP.Ident(), m.ecfg)
        _, aux = R.fwd(m, ids, doc, ei)
        n0, _, _ = m.core.assign(aux["c_hat"])
        ne, comp, _ = M4.chunk_masks(ids, doc, R.K)
        L = lab[a:a + 8].reshape(len(ids), -1, R.K)
        for bi in range(len(ids)):
            for j in range(L.shape[1]):
                v = L[bi, j][L[bi, j] >= 0]
                if comp[bi, j] and len(v) == R.K:
                    codes.append(int(n0[bi, j])); labs.append(int(np.bincount(v).argmax()))
    return dict(nmi=nmi(np.array(codes), np.array(labs)), n_chunks=len(codes), codes_used=len(set(codes)))


def nmi(x, y):
    def H(p):
        p = p[p > 0]; return float(-(p * np.log(p)).sum())
    xs, ys = np.unique(x, return_inverse=True)[1], np.unique(y, return_inverse=True)[1]
    J = np.zeros((xs.max() + 1, ys.max() + 1)); np.add.at(J, (xs, ys), 1); J /= J.sum()
    Hx, Hy, Hxy = H(J.sum(1)), H(J.sum(0)), H(J.ravel())
    return (Hx + Hy - Hxy) / max(math.sqrt(Hx * Hy), 1e-12)


# ----------------------------------------------------------------------------- per arm
def arm(name):
    m, a = load(name)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    import speed_v3
    speed_v3.enable_block_attention(60)
    res, units = dict(name=name, arch=a["arch"], data=a["data"], seed=a["seed"], budget=float(a["real_tokens"])), {}
    t0 = [time.time()]
    def phase(nm):
        print(f"{time.strftime('%T')} {name} {nm} done in {time.time() - t0[0]:.0f} s", flush=True); t0[0] = time.time()
    if a["data"] == "text":
        u = text_units(m, a); units.update(u)
        res["val_ntp"] = float(u["row_ntp"].mean())
        res["probes"] = {f: float(u["item_hit"][u["item_family"] == f].mean()) for f in np.unique(u["item_family"])}
    else:
        sets = R.pin_eval_sets()
        for s_name, eps in sets.items():
            u = pin_units(m, eps)
            for k, v in u.items():
                units[f"{s_name}/{k}"] = v
            res[s_name] = {k: float(np.nanmean(v)) for k, v in u.items()}
        phase("units")
        inter = dict(R.interventions(m))
        if R.is_v4(m):
            inter["engram_off"] = dict(engram_off=True)                   # all four Engram reads (encoder; decoder off anyway)
        for nm, kw in inter.items():
            u = pin_units(m, sets["heldout"], **kw)
            res.setdefault("interventions", {})[nm] = {k: float(np.nanmean(v)) for k, v in u.items()}
        phase("interventions")
        for pol in ("temp1", "greedy"):
            ro = rollouts(m, greedy=(pol == "greedy"), seed=a["seed"])
            for k, v in ro.items():
                units[f"roll_{pol}/{k}"] = v
            res[f"roll_{pol}"] = {k: float(np.nanmean(v)) for k, v in ro.items()}
            phase("rollouts " + pol)
        res["probes"] = probes(m)
        phase("probes")
        if R.is_v4(m):
            res["code_nmi"] = code_nmi(m)
            phase("code nmi")
    out = os.path.join(RUNS, name)
    json.dump(res, open(os.path.join(out, "analysis.json"), "w"), indent=1)
    np.savez(os.path.join(out, "per_unit.npz"), **{k: np.asarray(v) for k, v in units.items()})
    print(json.dumps(res, indent=1)[:3000])


# ----------------------------------------------------------------------------- compare (paired)
def paired(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    dd = a[ok] - b[ok]
    return float(dd.mean()), float(dd.std(ddof=1) / math.sqrt(len(dd))) if len(dd) > 1 else float("nan"), int(ok.sum())


def compare(prefix="vs_", allow_pilot=False):
    """Pairs v4 and LLM runs with the same (data, budget, seed); one table per (data, budget)."""
    arms = {}
    for n in sorted(os.listdir(RUNS)):
        f = os.path.join(RUNS, n, "per_unit.npz")
        if n.startswith(prefix) and (allow_pilot or "pilot" not in n) and os.path.exists(f):
            j = json.load(open(os.path.join(RUNS, n, "analysis.json")))
            key = (j["data"], j.get("budget", 0.0), j["arch"], j["seed"])
            assert key not in arms, f"two runs share {key}"
            arms[key] = np.load(f)
    out = {}
    for data, budget in sorted({(k[0], k[1]) for k in arms}):
        seeds = sorted({k[3] for k in arms if k[:2] == (data, budget)
                        and (data, budget, "v4", k[3]) in arms and (data, budget, "llm", k[3]) in arms})
        if not seeds:
            continue
        A = lambda arch, s: arms[(data, budget, arch, s)]
        keys = [k for k in A("v4", seeds[0]).files if k in A("llm", seeds[0]).files and A("v4", seeds[0])[k].dtype.kind == "f"]
        res = {}
        for k in keys:
            per = []
            for s in seeds:
                a, b = A("v4", s)[k], A("llm", s)[k]
                if k == "item_hit":
                    fam = A("v4", s)["item_family"]
                    for f in np.unique(fam):
                        res.setdefault(f"probe/{f}", {"seeds": []})["seeds"].append(paired(a[fam == f], b[fam == f]))
                    continue
                per.append(paired(a, b))
            if per:
                res[k] = {"seeds": per}
        for k, v in res.items():
            ms = [p[0] for p in v["seeds"]]; ses = [p[1] for p in v["seeds"]]
            v["mean"] = float(np.mean(ms)); v["se"] = float(math.sqrt(np.nansum(np.square(ses))) / len(ses))
            v["z"] = v["mean"] / v["se"] if v["se"] > 0 else float("nan")
        out[f"{data}@{budget / 1e6:g}M"] = dict(seeds=seeds, diff_v4_minus_llm=res)
    json.dump(out, open(os.path.join(RUNS, "vs_compare.json"), "w"), indent=1)
    lines = ["# v4 − LLM, paired (VS_LLM_REPORT §3)", ""]
    for key, r in out.items():
        lines += [f"## {key} (seeds {r['seeds']})", "", "| reading | v4 − LLM | SE | z | per seed |", "|---|---|---|---|---|"]
        for k, v in sorted(r["diff_v4_minus_llm"].items()):
            lines.append(f"| {k} | {v['mean']:+.4f} | {v['se']:.4f} | {v['z']:+.1f} | "
                         + ", ".join(f"{p[0]:+.4f}" for p in v["seeds"]) + " |")
        lines.append("")
    open(os.path.join(RUNS, "vs_compare.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "arm":
        arm(sys.argv[2])
    else:
        compare()
