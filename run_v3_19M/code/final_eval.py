"""Detailed testing of v3 (draft_v3.md r2 §7.4): at the 24 h milestone on VAL (the test split stays sealed) and at 48 h on
TEST + OOD (the number of record). v1's suite ported from v2; M sweep {1, 2a, 2b, 3}; the full intervention set (core,
u, c_pred, plan swap, shared FFN, table, Engram, Engram read 2, F layers); v3 loop dynamics; the per-term gradient audit.

  python final_eval.py --ckpt ../runs/v3_48h/ckpt_last.pt --out ../runs/v3_48h/eval_24h.json --split val
  python final_eval.py --ckpt ../runs/v3_48h/ckpt_last.pt --out ../runs/v3_48h/eval_final.json --split test

Parts (all by default):
  lang      test language (TinyStories-test, WikiText-test; 256 rows each): nats, bits/byte; interventions
            core_off, plan_swap, u_off, c_off; loop depth R in {1,2,3,4,6,8}
  trunc     context truncation r in {72,128,256,512,1024} on the last 16 positions + plan-swap cost (G-D2)
  items     synthetic families (200/family) and probes (512/family), test ID and OOD; interventions on every suite
  probes    linear probes: concept -> its 4 tokens (G-E1); u at the query chunk -> answer tokens per probe family;
            u after pass 1, 2, 3 -> bridge entity on lr_hop
  health    codebook health (G-C1, G-C2, G-E2) on test rows, telemetry, loop dynamics (concept spec 15.5)
  gen       10 greedy + 10 sampled (T = 0.8) continuations of 128 tokens from TinyStories and WikiText prompts
  audit     per-term gradient audit (§6.4): weighted g_main vs g_short per parameter group, norm of the sum, cosine,
            on fixed diagnostic batches from each data family and each shortcut schedule (diagnostic, not a gate)
Intervention deltas are fixed-checkpoint effects (not upper bounds, not training-time value).
"""
import argparse, json, math, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F

from model import build_model, forward, chunk_masks, rms, param_groups
from evaluate import Evaluator
from data import make_batch, pack_items, target_mask, Canon, PAD, EOS
from probes import PROBES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "run_m2_25M", "data"))   # shared with v1 (same tokenizer, splits, probes)
INTERV = {"intact": {}, "core_off": dict(core_off=True), "plan_swap": dict(plan_swap=True),
          "u_off": dict(use_u=False), "c_off": dict(use_c=False), "shared_ffn_off": dict(shared_ffn_off=True),
          "table_off": dict(table_off=True), "engram_off": dict(engram_off=True),
          "engram_read2_off": dict(engram_read2_off=True), "F_off": dict(F_off=True)}
R_SWEEP = ("1", "2a", "2b", "3")
SPLIT = ["test"]                                                  # set from --split in main()


def load(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    over = dict(table=not args.get("no_table", False))
    if args.get("shared_ff", -1) >= 0:
        over["shared_ff"] = args["shared_ff"]
    if args.get("core_ff", -1) > 0:
        over["core_ff"] = args["core_ff"]
    m, ecfg, dcfg = build_model(args.get("size", "full"), **over)
    import speed_v3
    speed_v3.enable_block_attention(args.get("block_attn", 0))             # same attention path as training
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ecfg, dcfg, ck


def part_lang(ev, m, R0, out):
    res = {}
    for name in ("tinystories", "wikitext"):
        tok, doc, _ = ev.lang_rows(SPLIT[0], name)
        r = {}
        for k, kw in INTERV.items():
            x = ev.language(m, tok, doc, R=R0, **kw); x.pop("per_row"); r[k] = x
        for R in R_SWEEP:
            if R != R0:
                x = ev.language(m, tok, doc, R=R); x.pop("per_row"); r[f"R{R}"] = x
        r[f"R{R0}"] = r["intact"]
        for k in list(r):
            if k != "intact":
                r[k]["delta_vs_intact"] = r[k]["nats"] - r["intact"]["nats"]
        res[name] = r
        print(f"lang {name}: " + " ".join(f"{k}={v['nats']:.4f}" for k, v in r.items()), flush=True)
    out["lang"] = res


def part_trunc(ev, m, R0, out):
    res = {}
    for name in ("tinystories", "wikitext"):
        tok, doc, ctx = ev.lang_rows(SPLIT[0], name)
        res[name] = ev.truncation(m, tok, doc, ctx, R=R0)
        print(f"trunc {name}: {res[name]['summary']}", flush=True)
    out["trunc"] = res


def part_items(ev, m, R0, out, split_list=None, probe_only_interv=False):
    split_list = split_list or (("val",) if SPLIT[0] == "val" else ("test", "ood"))
    res = {}
    for split in split_list:
        items = ev.items(split)
        synth = [x for x in items if x["family"] not in PROBES]
        probes = [x for x in items if x["family"] in PROBES]
        r = {}
        for k, kw in INTERV.items():
            r[k] = ev.item_accuracy(m, probes, R=R0, **kw)
            if not probe_only_interv or k in ("intact", "core_off", "plan_swap", "engram_off", "table_off", "shared_ffn_off"):
                r[k].update(ev.item_accuracy(m, synth, R=R0, **kw))
        for R in R_SWEEP:
            if R != R0:
                r[f"R{R}"] = ev.item_accuracy(m, probes, R=R)
                if not probe_only_interv:
                    r[f"R{R}"].update(ev.item_accuracy(m, synth, R=R))
        fams_s = sorted({x["family"] for x in synth})
        for k in list(r):
            got = [r[k][f]["exact"] for f in fams_s if f in r[k]]
            if got:
                r[k]["_macro_synthetic"] = float(np.mean(got))
        res[split] = r
        print(f"items {split}: intact probes " + " ".join(f"{f}={r['intact'][f]['exact']:.3f}" for f in PROBES) +
              f" | synthetic macro {r['intact'].get('_macro_synthetic', float('nan')):.3f}", flush=True)
        print(f"items {split}: plan_swap probes " + " ".join(f"{f}={r['plan_swap'][f]['exact']:.3f}" for f in PROBES), flush=True)
    out["items"] = res


# ---------------------------------------------------------------- linear probes
@torch.enable_grad()
def fit_probe(X, y, epochs=300, lr=1e-2, device="cpu"):
    """Multinomial logistic regression, 80/20 split, standardised features; returns held-out top-1 accuracy."""
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y)
    cls, yy = torch.unique(y, return_inverse=True)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(X), generator=g)
    X, yy = X[perm], yy[perm]
    n = int(0.8 * len(X))
    mu, sd = X[:n].mean(0), X[:n].std(0) + 1e-6
    X = ((X - mu) / sd).to(device); yy = yy.to(device)
    W = torch.zeros(X.shape[1], len(cls), device=device, requires_grad=True)
    b = torch.zeros(len(cls), device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(epochs):
        idx = torch.randint(0, n, (min(n, 8192),), device=device)
        loss = F.cross_entropy(X[idx] @ W + b, yy[idx]) + 1e-4 * W.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = ((X[n:] @ W + b).argmax(-1) == yy[n:]).float().mean().item()
        yc = yy.cpu()                                               # aten::mode has no MPS kernel
        maj = (yc[n:] == torch.mode(yc[:n]).values).float().mean().item()
    return dict(acc=acc, n_train=n, n_test=len(X) - n, classes=len(cls), majority_baseline=maj)


@torch.no_grad()
def part_probes(ev, m, R0, out, dev, tok_obj):
    res = {}
    # (1) G-E1: concept -> each of its 4 tokens, on test language rows + test probe rows
    feats, labs = [], [[] for _ in range(4)]
    tok, doc, _ = ev.lang_rows(SPLIT[0], "wikitext", 64)
    tok2, doc2, _ = ev.lang_rows(SPLIT[0], "tinystories", 64)
    for T_, D_ in ((tok, doc), (tok2, doc2)):
        for (ids, dc, ei), n_real in ev._batches(T_, D_):
            _, c_hat, _ = m.enc(ids.to(dev), dc.to(dev), ei.to(dev))
            c = c_hat[:n_real].float().cpu()
            ch = ids[:n_real].view(n_real, -1, 4)
            real = (ch != PAD).all(-1)
            feats.append(c[real])
            for i in range(4):
                labs[i].append(ch[..., i][real])
    X = torch.cat(feats)
    res["G_E1_concept_to_token"] = {f"pos{i}": fit_probe(X, torch.cat(labs[i]), device=dev) for i in range(4)}
    print("probe G-E1:", {k: round(v["acc"], 4) for k, v in res["G_E1_concept_to_token"].items()}, flush=True)
    # (2) u at the query chunk -> answer tokens, per probe family (u_{j*} read by the first answer token)
    items = ev.items(SPLIT[0])
    for fam in PROBES:
        its = [x for x in items if x["family"] == fam]
        tokp, docp, where = pack_items(its)
        U = {}
        r0 = 0
        for (ids, dc, ei), n_real in ev._batches(tokp, docp):
            _, x = forward(m, ids.to(dev), dc.to(dev), ei.to(dev), R0)
            U.update({r0 + i: x["u"][i].float().cpu() for i in range(n_real)})
            r0 += n_real
        fx, fy = [], {}
        for it, (row, off) in zip(its, where):
            for s, e, _ in it["spans"]:
                t = off + s - 1                                   # position predicting the first answer token
                j = (t + 1) // 4 - 1
                fx.append(U[row][j])
                for i in range(e - s):
                    fy.setdefault(i, []).append(it["ids"][s + i])
        fx = torch.stack(fx)
        res[f"u_query_to_answer_{fam}"] = {f"tok{i}": fit_probe(fx[:len(v)], torch.tensor(v), device=dev)
                                           for i, v in fy.items() if len(v) == len(fx)}
        print(f"probe u->answer {fam}:", {k: round(v["acc"], 3) for k, v in res[f'u_query_to_answer_{fam}'].items()}, flush=True)
    # (3) lr_hop: u after pass 1, 2, 4 -> bridge entity (first intermediate node's first token)
    its = [x for x in items if x["family"] == "lr_hop"]
    tokp, docp, where = pack_items(its)
    bridge, qpos = [], []
    for it in its:
        txt = tok_obj.decode(it["ids"])
        facts = dict(re.findall(r" (\w+) points to (\w+)\.", txt))
        st = re.search(r"Start at (\w+) and follow", txt).group(1)
        b = facts.get(st)
        bridge.append(tok_obj.encode(" " + b).ids[0] if b else -1)
        qpos.append(it["spans"][0][0] - 1)
    per_pass = {1: [], 2: [], 3: []}
    r0 = 0
    rows_u = {}
    for (ids, dc, ei), n_real in ev._batches(tokp, docp):
        ids_d, dc_d, ei_d = ids.to(dev), dc.to(dev), ei.to(dev)
        _, c_hat, _ = m.enc(ids_d, dc_d, ei_d)
        _, states, _ = m.core.run(c_hat, dc_d[:, ::4], "3")
        for p in per_pass:
            u = m.core.out_norm(states[p - 1])
            for i in range(n_real):
                rows_u[(r0 + i, p)] = u[i].float().cpu()
        r0 += n_real
    for it, (row, off), bt, q in zip(its, where, bridge, qpos):
        if bt < 0:
            continue
        j = (off + q + 1) // 4 - 1
        for p in per_pass:
            per_pass[p].append((rows_u[(row, p)][j], bt))
    res["lr_hop_bridge_by_pass"] = {f"pass{p}": fit_probe(torch.stack([a for a, _ in v]), torch.tensor([b for _, b in v]), device=dev)
                                    for p, v in per_pass.items()}
    print("probe lr_hop bridge by pass:", {k: round(v["acc"], 3) for k, v in res["lr_hop_bridge_by_pass"].items()}, flush=True)
    out["probes"] = res


def part_health(ev, m, R0, out):
    tok = np.load(os.path.join(DATA, SPLIT[0], "tok.npy"), mmap_mode="r")
    doc = np.load(os.path.join(DATA, SPLIT[0], "doc.npy"), mmap_mode="r")
    idx = np.sort(np.random.RandomState(7).choice(len(tok), 64, replace=False))
    t, d = np.asarray(tok[idx]), np.asarray(doc[idx])
    out["health"] = dict(codebook=ev.codebook_health(m, t, d, R=R0), telemetry=ev.telemetry(m, t, d))
    hop = [x for x in ev.items(SPLIT[0]) if x["family"] == "lr_hop"]
    out["health"]["loop_dynamics_lr_hop"] = {R: ev.loop_dynamics(m, hop, R=R) for R in ("3", "2b", "2a")}
    print("health:", out["health"]["codebook"], flush=True)
    print("loop dynamics M3:", {k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in out["health"]["loop_dynamics_lr_hop"]["3"].items()}, flush=True)


@torch.no_grad()
def part_gen(ev, m, R0, out, dev, tok_obj, ecfg, n=10, new=128, plen=64):
    canon = Canon(DATA)
    res = {}
    g = torch.Generator().manual_seed(0)
    for name in ("tinystories", "wikitext"):
        tok, doc, _ = ev.lang_rows(SPLIT[0], name)
        prompts = []
        for r in range(len(tok)):
            row, dr = tok[r], doc[r]
            # a prompt = the first plen tokens of a document that has at least plen tokens in this row
            starts = [0] + [i for i in range(1, len(dr)) if dr[i] != dr[i - 1]]
            for s in starts:
                if s + plen <= len(dr) and (dr[s:s + plen] == dr[s]).all() and PAD not in row[s:s + plen]:
                    prompts.append(row[s:s + plen].astype(np.int64)); break
            if len(prompts) == n:
                break
        outs = {}
        for mode in ("greedy", "sample_T0.8"):
            seqs = [list(p) for p in prompts]
            for _ in range(new):
                L = len(seqs[0]); Lp = L + (-L) % 4
                ids = torch.full((len(seqs), Lp), PAD, dtype=torch.long)
                for i, s in enumerate(seqs):
                    ids[i, :L] = torch.tensor(s)
                docz = torch.zeros_like(ids)
                ei = __import__("hrm_encoder").engram_indices(canon(ids), docz, ecfg)
                logits, _ = forward(m, ids.to(dev), docz.to(dev), ei.to(dev), R0)
                lg = logits[:, L - 1].float().cpu()
                if mode == "greedy":
                    nxt = lg.argmax(-1)
                else:
                    nxt = torch.multinomial(F.softmax(lg / 0.8, -1), 1, generator=g)[:, 0]
                for i in range(len(seqs)):
                    seqs[i].append(int(nxt[i]))
            outs[mode] = [dict(prompt=tok_obj.decode(list(p)), continuation=tok_obj.decode(s[plen:], skip_special_tokens=False))
                          for p, s in zip(prompts, seqs)]
        res[name] = outs
        print(f"gen {name}: greedy[0] = {outs['greedy'][0]['continuation'][:300]!r}", flush=True)
    out["generation"] = res

# ---------------------------------------------------------------- per-term gradient audit (draft r2 §6.4)
AUDIT_GROUPS = (("encoder", lambda n: n.startswith("enc.")), ("core_attn", lambda n: ".attn." in n and n.startswith("core.layers")),
                ("core_own_ffn", lambda n: ".ffn." in n and n.startswith("core.layers")),
                ("core_reads_writes", lambda n: n.startswith("core.layers") and (".read_" in n or ".write_" in n)),
                ("core_shared", lambda n: n.startswith("core.shared")), ("core_table", lambda n: n == "core.table"),
                ("ncp_head", lambda n: n in ("core.Wq.weight", "core.log_tau")),
                ("dec_plan", lambda n: n.startswith("dec.plan.")))


def audit_batches(ev, n_rows=2):
    """Fixed diagnostic batches: TinyStories, WikiText (language rows) and the probe / synthetic item families."""
    out = []
    for name in ("tinystories", "wikitext"):
        tok, doc, _ = ev.lang_rows(SPLIT[0], name, n_rows)
        out.append((name, tok, doc))
    items = ev.items(SPLIT[0])
    fams = {}
    for x in items:
        fams.setdefault("probes" if x["family"] in PROBES else "synthetic", []).append(x)
    for k in ("probes", "synthetic"):
        tok, doc, _ = pack_items(fams[k][:40])
        out.append((k, tok[:n_rows], doc[:n_rows]))
    return out


def part_audit(ev, m, out, dev):
    """g_main = grad(NTP + 0.1 NCP(u3)), g_short = grad(0.05 NCP(u_s) + 0.1 L_scons) per group; report |g_main|, |g_short|,
    |g_main + g_short| (norm of the SUM), cos(g_main, g_short) and the projection of g_short on g_main."""
    names = [n for n, _ in m.named_parameters()]
    params = dict(m.named_parameters())
    res = {}
    for fam, tok, doc in audit_batches(ev):
        (ids, dc, ei), n_real = next(iter(ev._batches(tok, doc)))
        ids, dc, ei = ids.to(dev), dc.to(dev), ei.to(dev)
        for sc in ("1", "2a", "2b"):
            with torch.enable_grad():
                tokv, c_hat, _ = m.enc(ids, dc, ei)
                cdoc = dc[:, ::4]
                nonempty, complete, pair_ok = chunk_masks(ids, dc)
                _, n = m.core.vq_sum(c_hat, complete, count_hits=False)
                S3, _, Ss = m.core.run(c_hat, cdoc, "3", short=sc)
                u = m.core.out_norm(S3); lc, cp = m.core.head(u)
                logits, _ = m.dec(tokv, u, cp, dc, m.enc.embed.weight)
                tm = target_mask(ids, dc)
                tgt = ids[:, 1:].masked_fill(~tm, -100)
                ntp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
                npair = pair_ok.sum().clamp_min(1)
                l_main = ntp + 0.1 * m.core.ncp_sum(lc, n, pair_ok) / npair
                lcs, _ = m.core.head(m.core.out_norm(Ss), pred=False)
                l_short = 0.05 * m.core.ncp_sum(lcs, n, pair_ok) / npair + \
                    0.1 * ((rms(Ss) - rms(S3).detach()).pow(2).mean(-1) * nonempty).sum() / nonempty.sum().clamp_min(1)
                gm = torch.autograd.grad(l_main, list(params.values()), retain_graph=True, allow_unused=True)
                gs = torch.autograd.grad(l_short, list(params.values()), allow_unused=True)
            row = {}
            for gname, sel in AUDIT_GROUPS:
                idx = [i for i, nm in enumerate(names) if sel(nm)]
                if not idx:                                           # arm without this component (no table / no shared)
                    continue
                vm = torch.cat([(gm[i] if gm[i] is not None else torch.zeros_like(params[names[i]])).detach().flatten().cpu().double() for i in idx])
                vs = torch.cat([(gs[i] if gs[i] is not None else torch.zeros_like(params[names[i]])).detach().flatten().cpu().double() for i in idx])
                nm_, ns_ = vm.norm().item(), vs.norm().item()
                row[gname] = dict(g_main=nm_, g_short=ns_, norm_of_sum=(vm + vs).norm().item(),
                                  cos=(vm @ vs).item() / max(nm_ * ns_, 1e-30), proj_short_on_main=(vm @ vs).item() / max(nm_, 1e-30))
            res[f"{fam}/{sc}"] = row
    m.zero_grad(set_to_none=True)
    summ = {}
    for gname, _ in AUDIT_GROUPS:
        if not all(gname in v for v in res.values()):
            continue
        cs = [v[gname]["cos"] for v in res.values()]
        rs = [v[gname]["g_short"] / max(v[gname]["g_main"], 1e-30) for v in res.values()]
        summ[gname] = dict(cos_median=float(np.median(cs)), cos_frac_negative=float(np.mean([c < 0 for c in cs])),
                           short_over_main_median=float(np.median(rs)))
    out["audit"] = dict(batches=res, summary=summ)
    print("audit summary:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in summ.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--parts", default="lang,trunc,items,probes,health,audit,gen")
    ap.add_argument("--split", default="test", choices=("val", "test"))   # 24 h milestone: val (test sealed until 48 h)
    ap.add_argument("--probe_only_interv", action="store_true")   # synthetic families: intact/core_off/swap only
    ap.add_argument("--R", default="")                            # evaluation schedule (default: the run's eval_R, "3")
    ap.add_argument("--limit", type=int, default=0)               # code-path testing only: rows / items per family
    a = ap.parse_args()
    dev = torch.device(a.device)
    SPLIT[0] = a.split
    m, ecfg, dcfg, ck = load(a.ckpt, dev)
    R0 = a.R or ck.get("args", {}).get("eval_R") or "3"
    from tokenizers import Tokenizer
    tok_obj = Tokenizer.from_file(os.path.join(DATA, "tokenizer.json"))
    ev = Evaluator(DATA, ecfg, dev, batch_rows=4)
    if a.limit:
        L = a.limit
        _lr, _it = ev.lang_rows, ev.items
        ev.lang_rows = lambda split, name, n=None: _lr(split, name, min(n or L, L))
        def _items(split, _it=_it):
            by, sel = {}, []
            for x in _it(split):
                if by.get(x["family"], 0) < L:
                    sel.append(x); by[x["family"]] = by.get(x["family"], 0) + 1
            return sel
        ev.items = _items
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    out.update(ckpt=a.ckpt, step=ck["step"], eval_R=R0, split=a.split, time=time.strftime("%Y-%m-%d %H:%M"))
    t0 = time.time()
    for p in a.parts.split(","):
        t1 = time.time()
        if p == "lang": part_lang(ev, m, R0, out)
        elif p == "trunc": part_trunc(ev, m, R0, out)
        elif p == "items": part_items(ev, m, R0, out, probe_only_interv=a.probe_only_interv)
        elif p == "probes": part_probes(ev, m, R0, out, dev, tok_obj)
        elif p == "health": part_health(ev, m, R0, out)
        elif p == "gen": part_gen(ev, m, R0, out, dev, tok_obj, ecfg)
        elif p == "audit": part_audit(ev, m, out, dev)
        out.setdefault("seconds", {})[p] = time.time() - t1
        json.dump(out, open(a.out, "w"), indent=1)                 # durable after every part
        print(f"part {p} done in {time.time() - t1:.0f}s", flush=True)
    print(f"ALL DONE in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
