"""Detailed testing after training (PROMPT section 7) on one checkpoint.

  python final_eval.py --ckpt ../runs/main/ckpt_last.pt --out ../runs/main/eval_final.json [--device mps] [--parts ...]

Parts (all by default):
  lang      test language (TinyStories-test, WikiText-test; 256 rows each): nats, bits/byte; interventions
            core_off, plan_swap, u_off, c_off; loop depth R in {1,2,3,4,6,8}
  trunc     context truncation r in {72,128,256,512,1024} on the last 16 positions + plan-swap cost (G-D2)
  items     synthetic families (200/family) and probes (512/family), test ID and OOD; interventions on every suite
  probes    linear probes: concept -> its 4 tokens (G-E1); u at the query chunk -> answer tokens per probe family;
            u after pass 1, 2, 4 -> bridge entity on lr_hop
  health    codebook health (G-C1, G-C2, G-E2) on test rows, telemetry, loop dynamics (concept spec 15.5)
  gen       10 greedy + 10 sampled (T = 0.8) continuations of 128 tokens from TinyStories and WikiText prompts
"""
import argparse, json, math, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F

from model import build_model, forward
from evaluate import Evaluator
from data import make_batch, pack_items, target_mask, Canon, PAD, EOS
from probes import PROBES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
INTERV = {"intact": {}, "core_off": dict(core_off=True), "plan_swap": dict(plan_swap=True),
          "u_off": dict(use_u=False), "c_off": dict(use_c=False)}
R_SWEEP = (1, 2, 3, 4, 6, 8)


def load(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = ck.get("args", {})
    if args.get("small"):
        d = args.get("d", 64)
        m, ecfg, dcfg = build_model(d=d, layout=args.get("layout", "confined"), S=8, N=32, n_heads=4, n_kv=2,
                                    head_dim=d // 4, d_ff=int(2.75 * d), engram_rows=1021, engram_dim=8, gr_rank=8)
    else:
        m, ecfg, dcfg = build_model(layout=args.get("layout", "confined"))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ecfg, dcfg, ck


def part_lang(ev, m, R0, out):
    res = {}
    for name in ("tinystories", "wikitext"):
        tok, doc, _ = ev.lang_rows("test", name)
        r = {}
        for k, kw in INTERV.items():
            x = ev.language(m, tok, doc, R=R0, **kw); x.pop("per_row"); r[k] = x
        for R in R_SWEEP:
            if R != R0:
                x = ev.language(m, tok, doc, R=R); x.pop("per_row"); r[f"R{R}"] = x
        r[f"R{R0}"] = r["intact"]
        res[name] = r
        print(f"lang {name}: " + " ".join(f"{k}={v['nats']:.4f}" for k, v in r.items()), flush=True)
    out["lang"] = res


def part_trunc(ev, m, R0, out):
    res = {}
    for name in ("tinystories", "wikitext"):
        tok, doc, ctx = ev.lang_rows("test", name)
        res[name] = ev.truncation(m, tok, doc, ctx, R=R0)
        print(f"trunc {name}: {res[name]['summary']}", flush=True)
    out["trunc"] = res


def part_items(ev, m, R0, out, split_list=("test", "ood"), probe_only_interv=False):
    res = {}
    for split in split_list:
        items = ev.items(split)
        synth = [x for x in items if x["family"] not in PROBES]
        probes = [x for x in items if x["family"] in PROBES]
        r = {}
        for k, kw in INTERV.items():
            r[k] = ev.item_accuracy(m, probes, R=R0, **kw)
            if not probe_only_interv or k in ("intact", "core_off", "plan_swap"):
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
        maj = (yy[n:] == torch.mode(yy[:n]).values).float().mean().item()
    return dict(acc=acc, n_train=n, n_test=len(X) - n, classes=len(cls), majority_baseline=maj)


@torch.no_grad()
def part_probes(ev, m, R0, out, dev, tok_obj):
    res = {}
    # (1) G-E1: concept -> each of its 4 tokens, on test language rows + test probe rows
    feats, labs = [], [[] for _ in range(4)]
    tok, doc, _ = ev.lang_rows("test", "wikitext", 64)
    tok2, doc2, _ = ev.lang_rows("test", "tinystories", 64)
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
    items = ev.items("test")
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
    from hrm_encoder import build_masks
    per_pass = {1: [], 2: [], 4: []}
    r0 = 0
    rows_u = {}
    for (ids, dc, ei), n_real in ev._batches(tokp, docp):
        ids_d, dc_d, ei_d = ids.to(dev), dc.to(dev), ei.to(dev)
        _, c_hat, _ = m.enc(ids_d, dc_d, ei_d)
        cdoc = dc_d[:, ::4]
        _, mask = build_masks(cdoc, 1)
        S = torch.zeros_like(c_hat)
        for p in range(1, 5):
            S = m.core.layer(c_hat, S, cdoc, mask)
            if p in per_pass:
                u = m.core.out_norm(S)
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
    tok = np.load(os.path.join(DATA, "test", "tok.npy"), mmap_mode="r")
    doc = np.load(os.path.join(DATA, "test", "doc.npy"), mmap_mode="r")
    idx = np.sort(np.random.RandomState(7).choice(len(tok), 64, replace=False))
    t, d = np.asarray(tok[idx]), np.asarray(doc[idx])
    out["health"] = dict(codebook=ev.codebook_health(m, t, d, R=R0), telemetry=ev.telemetry(m, t, d))
    hop = [x for x in ev.items("test") if x["family"] == "lr_hop"]
    out["health"]["loop_dynamics_lr_hop"] = ev.loop_dynamics(m, hop, R=8)
    print("health:", out["health"]["codebook"], flush=True)
    print("loop dynamics:", {k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in out["health"]["loop_dynamics_lr_hop"].items()}, flush=True)


@torch.no_grad()
def part_gen(ev, m, R0, out, dev, tok_obj, ecfg, n=10, new=128, plen=64):
    canon = Canon(DATA)
    res = {}
    g = torch.Generator().manual_seed(0)
    for name in ("tinystories", "wikitext"):
        tok, doc, _ = ev.lang_rows("test", name)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--parts", default="lang,trunc,items,probes,health,gen")
    ap.add_argument("--probe_only_interv", action="store_true")   # synthetic families: intact/core_off/swap only
    ap.add_argument("--R", type=int, default=0)                   # evaluation depth (default: 4, or fixed_R of S3)
    ap.add_argument("--limit", type=int, default=0)               # code-path testing only: rows / items per family
    a = ap.parse_args()
    dev = torch.device(a.device)
    m, ecfg, dcfg, ck = load(a.ckpt, dev)
    R0 = a.R or ck.get("args", {}).get("fixed_R") or 4
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
    out.update(ckpt=a.ckpt, step=ck["step"], eval_R=R0, time=time.strftime("%Y-%m-%d %H:%M"))
    t0 = time.time()
    for p in a.parts.split(","):
        t1 = time.time()
        if p == "lang": part_lang(ev, m, R0, out)
        elif p == "trunc": part_trunc(ev, m, R0, out)
        elif p == "items": part_items(ev, m, R0, out, probe_only_interv=a.probe_only_interv)
        elif p == "probes": part_probes(ev, m, R0, out, dev, tok_obj)
        elif p == "health": part_health(ev, m, R0, out)
        elif p == "gen": part_gen(ev, m, R0, out, dev, tok_obj, ecfg)
        out.setdefault("seconds", {})[p] = time.time() - t1
        json.dump(out, open(a.out, "w"), indent=1)                 # durable after every part
        print(f"part {p} done in {time.time() - t1:.0f}s", flush=True)
    print(f"ALL DONE in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
