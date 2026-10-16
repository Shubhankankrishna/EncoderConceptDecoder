"""v4 vs LLM-stack tiny experiment (CPU): train and evaluate one arm.

python run_vs.py --name vs_text_v4_s0 --arch v4  --data text   --seed 0 --real_tokens 5e6
python run_vs.py --name vs_pin_llm_s1 --arch llm --data pinpad --seed 1 --real_tokens 3e6
Writes v4_tiny/runs/<name>/{log.jsonl, final.json, ckpt.pt}.

Held equal across arch: training rows (same stream, same seed, rows padded per v4's k = 6 for both), tokenizer,
tied embedding, NTP target mask, optimizer (AdamW 0.9/0.95, wd 0.1 by v4's decay rule), lr schedule (warm-up 400,
cosine to 0.1 of peak over the token budget), B, T, clip, token budget, threads. v4 adds its own concept terms
(0.05 CE_coarse + 0.1 CE_fine + VQ); comparisons use NTP only.
v4 = recommended tiny config: k 6, pooler attention only (no FFN), decoder Engram reads off, F loop 2, core 4 x M 2.
Text: v3's corpus (v4_tiny/data) mixed with recall1 / carry3 / multi4 probe rows; the filler flaw of the first tiny
experiment is fixed: training fillers come from 60,000 training documents, validation fillers from the val store.
"""
import argparse
import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F

import model_v4 as M4
import model_llm as ML
import data_v4 as D
import pinpad as PP

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", "runs"))
K = 6


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", choices=["v4", "llm"], required=True)
    ap.add_argument("--data", choices=["text", "pinpad"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=480)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--p_syn", type=float, default=0.5)
    ap.add_argument("--real_tokens", type=float, default=5e6)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--eval_every_tokens", type=float, default=1e6)
    ap.add_argument("--n_probe", type=int, default=128)
    ap.add_argument("--n_probe_final", type=int, default=192)
    ap.add_argument("--restart_every", type=int, default=200)
    ap.add_argument("--block", type=int, default=60)
    ap.add_argument("--over", default="", help="model overrides, e.g. ff=296")
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------- model dispatch
def build(arch, V, over=None, size="tiny"):
    over = dict(over or {})
    if arch == "v4":
        cfg = dict(k=K, pool="attn", dec_reads=(), V=V); cfg.update(over)
        return M4.build_model(size, **cfg)
    cfg = dict(V=V); cfg.update(over)
    return ML.build_model(size, **cfg)


def is_v4(m):
    return isinstance(m, M4.HRMThinkerV4)


def fwd(m, ids, doc, ei, **kw):
    if is_v4(m):
        return M4.forward(m, ids, doc, ei, **kw)
    kw = {k: v for k, v in kw.items() if k in ("M", "enc_loop", "collect")}
    return ML.forward(m, ids, doc, ei, **kw)


def loss_fn(m, ids, doc, ei):
    return (M4 if is_v4(m) else ML).loss_terms(m, ids, doc, ei)


def ptable(m):
    t = (M4 if is_v4(m) else ML).param_table(m)
    return {k: v for k, v in t.items() if k != "parts"}


def interventions(m):
    if is_v4(m):
        return dict(core_off=dict(core_off=True), plan_swap=dict(plan_swap=True), M1=dict(M=1), loop_off=dict(enc_loop=1))
    return dict(M1=dict(M=1), loop_off=dict(enc_loop=1))


def flops_per_token(m, T=480):
    """Forward multiply-adds per token (x2 = FLOPs), matmuls + attention scores/values; v4's core at T / k positions."""
    d = m.c["d"]
    def attn_flops(heads, kv, dh, keys):                                  # projections + QK + AV
        return 2 * d * (2 * heads * dh + 2 * kv * dh) + 2 * heads * dh * 2 * keys
    if not is_v4(m):
        c = m.c; tot = 0
        for li in m.schedule:
            kind = m.layers[li].kind
            keys = min(c["window"], T) if kind == "S" else T / 2
            tot += attn_flops(c["heads"], c["kv"], c["head_dim"], keys) + 2 * 3 * d * c["ff"]
        return tot + 2 * d * c["V"]
    c = m.c; tot = 0.0
    enc_layers = [l.kind for l in m.enc.layers]
    for kind in enc_layers:
        reps = c["enc_loop"] if kind == "F" else 1
        heads = c["heads_full"] if kind == "F" else c["heads"]
        keys = 8 if kind == "S" else T / 2
        tot += reps * (attn_flops(heads, c["kv"], c["head_dim"], keys) + 2 * 3 * d * c["enc_ff"]
                       + 2 * 3 * d * c["enc_shared_ff"])
    nc = T / c["k"]
    core = c["core_layers"] * c["M"] * (attn_flops(c["heads_full"], c["kv"], c["head_dim"], nc / 2)
                                         + 2 * 3 * d * (c["core_ff"] + c["shared_ff"]))
    tot += core / c["k"]
    tot += 4 * (attn_flops(c["heads"], c["kv"], c["head_dim"], 8) + 2 * 3 * d * c["dec_ff"])
    return tot + 2 * d * c["V"]


# ----------------------------------------------------------------------------- text evaluation
@torch.no_grad()
def val_ntp(m, rows, canon, chunk=8, **kw):
    t, d = rows
    tot = n = 0.0
    for a in range(0, len(t), chunk):
        ids, doc, ei = D.to_batch(t[a:a + chunk], d[a:a + chunk], canon, m.ecfg)
        lg, _ = fwd(m, ids, doc, ei, **kw)
        tm = M4.target_mask(ids, doc)
        tgt = ids[:, 1:].masked_fill(~tm, -100)
        tot += F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=-100,
                               reduction="sum").item()
        n += tm.sum().item()
    return tot / max(n, 1)


@torch.no_grad()
def item_acc(m, items, canon, T, chunk=8, **kw):
    """Teacher-forced greedy exact match of every answer span (prediction for token i is logits[i-1])."""
    t, d, where = D.item_rows(items, T, K)
    by_row = {}
    for idx, (r, off, L) in enumerate(where):
        by_row.setdefault(r, []).append((idx, off))
    correct = np.zeros(len(items), bool)
    for a in range(0, len(t), chunk):
        rs = list(range(a, min(a + chunk, len(t))))
        if len(rs) < 2:
            rs = [rs[0] - 1, rs[0]] if rs[0] > 0 else rs
        ids, doc, ei = D.to_batch(t[rs], d[rs], canon, m.ecfg)
        pred = fwd(m, ids, doc, ei, **kw)[0].argmax(-1)
        for bi, r in enumerate(rs):
            if r < a:
                continue
            for idx, off in by_row.get(r, []):
                ok = True
                for (s, e, _) in items[idx]["spans"]:
                    ok &= bool((pred[bi, off + s - 1: off + e - 1] == ids[bi, off + s: off + e]).all())
                correct[idx] = ok
    fams = {}
    for it, c in zip(items, correct):
        fams.setdefault(it["family"], []).append(c)
    return {f: float(np.mean(v)) for f, v in fams.items()}


@torch.no_grad()
def item_ntp(m, items, canon, T, chunk=8):
    t, d, _ = D.item_rows(items, T, K)
    return val_ntp(m, (t, d), canon, chunk)


# ----------------------------------------------------------------------------- pinpad evaluation
def pin_rows(eps, T):
    """Pack episodes; per-row label arrays aligned with token positions."""
    t, d, where = D.pack_sequences([e["tokens"] for e in eps], T, K)
    R = len(t)
    lab = {k: np.full((R, T), -1, np.int64) for k in ("kind", "sub", "tgt", "bflag")}
    opt = {}
    for e, (r, off, L) in zip(eps, where):
        assert L == len(e["tokens"]), "episode longer than T"
        lab["kind"][r, off:off + L] = np.where(np.arange(L) < 27, 0, e["kind"])
        lab["sub"][r, off:off + L] = e["sub"]; lab["tgt"][r, off:off + L] = e["tgt"]
        lab["bflag"][r, off:off + L] = PP.boundary_flags(e)
        for i, A in enumerate(e["opt"]):
            if A is not None:
                opt[(r, off + i)] = A
    return t, d, lab, opt


@torch.no_grad()
def pin_metrics(m, eps, T, chunk=8, **kw):
    """NTP split by token kind; action optimality (argmax over the 4 actions in the expert's optimal set), split by
    boundary flag; position exact accuracy."""
    t, d, lab, opt = pin_rows(eps, T)
    canon = PP.Ident()
    agg = {}
    def add(key, v, n=1):
        s = agg.setdefault(key, [0.0, 0])
        s[0] += v; s[1] += n
    starts = list(range(0, len(t), chunk))
    ends = starts[1:] + [len(t)]
    if len(t) - starts[-1] == 1 and len(starts) > 1:                    # no 1-row chunk (plan_swap would be a no-op)
        starts, ends = starts[:-1], ends[:-2] + [len(t)]
    for a, z in zip(starts, ends):
        ids, doc, ei = D.to_batch(t[a:z], d[a:z], canon, m.ecfg)
        lg = fwd(m, ids, doc, ei, **kw)[0]
        tm = M4.target_mask(ids, doc)
        tgt = ids[:, 1:].masked_fill(~tm, 0)
        nll = F.cross_entropy(lg[:, :-1].transpose(1, 2), tgt, reduction="none")       # (b, T-1)
        kind = torch.from_numpy(lab["kind"][a:z, 1:])
        for kk, nm in ((0, "header"), (1, "action"), (2, "position"), (3, "end")):
            sel = tm & (kind == kk)
            add("ntp_" + nm, float(nll[sel].sum()), int(sel.sum()))
        add("ntp_all", float(nll[tm].sum()), int(tm.sum()))
        act_pred = lg[:, :-1, PP.ACT0:PP.ACT0 + 4].argmax(-1)                          # (b, T-1)
        full_pred = lg[:, :-1].argmax(-1)
        bf = lab["bflag"][a:z, 1:]
        for bi in range(ids.shape[0]):
            r = a + bi
            for j in np.flatnonzero(lab["kind"][r, 1:] == 1):                         # target index j+1 is an ACT
                A = opt[(r, j + 1)]
                ok = int(act_pred[bi, j]) in A
                add("act_opt", ok); add("act_chance", len(A) / 4)
                add(f"act_opt_b{bf[bi, j]}", ok); add(f"act_chance_b{bf[bi, j]}", len(A) / 4)
            for j in np.flatnonzero(lab["kind"][r, 1:] == 2):
                add("pos_acc", int(full_pred[bi, j] == ids[bi, j + 1]))
    return {k: v[0] / max(v[1], 1) for k, v in agg.items()}


def pin_eval_sets(n_heldin=192, n_heldout=200, n_long=64):
    return dict(heldin=PP.eval_episodes("heldin", n_heldin, 0), heldout=PP.eval_episodes("heldout", n_heldout, 0),
                long=PP.eval_episodes("long", n_long, 0))


# ----------------------------------------------------------------------------- data setup
def text_setup(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(D.SRC, "tokenizer.json"))
    tr, va = D.TextStore("train"), D.TextStore("val")
    fill_tr = D.filler_sentences(tr, tok, n_docs=60000, seed=0)
    fill_va = D.filler_sentences(va, tok, n_docs=va.n, seed=0)
    probes_tr = D.TinyProbes(tok, fill_tr, "train")
    probes_va = D.TinyProbes(tok, fill_va, "val")
    stream = D.MixedStream(tr, probes_tr, a.T, K, a.B, a.p_syn, seed=a.seed)
    return dict(tok=tok, canon=D.Canon(), stream=stream, vrows=D.val_text_rows(va, a.T, K, n_rows=64),
                items_q=D.probe_items(probes_va, a.n_probe, seed=1000),
                items_f=D.probe_items(probes_va, a.n_probe_final, seed=2000),
                items_trpool=D.probe_items(probes_tr, 96, seed=3000), n_fill=(len(fill_tr), len(fill_va)))


def pin_setup(a):
    return dict(canon=PP.Ident(), stream=PP.PinpadStream(a.T, K, a.B, seed=a.seed), sets=pin_eval_sets())


def periodic_eval(m, a, S):
    m.eval()
    if a.data == "text":
        ev = dict(val_ntp=val_ntp(m, S["vrows"], S["canon"]), probes=item_acc(m, S["items_q"], S["canon"], a.T))
        if is_v4(m):
            ev["swap"] = item_acc(m, [x for x in S["items_q"] if x["family"] == "lr_recall1"], S["canon"], a.T,
                                  plan_swap=True).get("lr_recall1")
    else:
        ev = {nm: pin_metrics(m, eps[:96], a.T) for nm, eps in S["sets"].items() if nm != "long"}
    m.train()
    return ev


def final_eval(m, a, S):
    m.eval()
    res = {}
    if a.data == "text":
        res["val_ntp"] = val_ntp(m, S["vrows"], S["canon"])
        res["probes"] = item_acc(m, S["items_f"], S["canon"], a.T)
        res["probe_row_ntp"] = dict(val_pool=item_ntp(m, S["items_f"][:96], S["canon"], a.T),
                                    train_pool=item_ntp(m, S["items_trpool"], S["canon"], a.T))
        res["interventions"] = {nm: dict(val_ntp=val_ntp(m, S["vrows"], S["canon"], **kw),
                                         probes=item_acc(m, S["items_f"], S["canon"], a.T, **kw))
                                for nm, kw in interventions(m).items()}
    else:
        res.update({nm: pin_metrics(m, eps, a.T) for nm, eps in S["sets"].items()})
        res["interventions"] = {nm: {s: pin_metrics(m, S["sets"][s], a.T, **kw) for s in ("heldin", "heldout")}
                                for nm, kw in interventions(m).items()}
    m.train()
    return res


# ----------------------------------------------------------------------------- training
def main(argv=None):
    a = get_args(argv)
    torch.set_num_threads(a.threads)
    if a.block:
        import speed_v3
        speed_v3.enable_block_attention(a.block)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    out = os.path.join(RUNS, a.name); os.makedirs(out, exist_ok=True)
    over = {}
    for kv in filter(None, a.over.split(",")):
        kk, vv = kv.split("="); over[kk] = eval(vv)
    V = 8192 if a.data == "text" else PP.V
    m = build(a.arch, V, over)
    S = text_setup(a) if a.data == "text" else pin_setup(a)
    canon, stream = S["canon"], S["stream"]
    pt = ptable(m)
    opt = torch.optim.AdamW((M4 if is_v4(m) else ML).param_groups(m, a.wd), lr=a.lr, betas=(0.9, 0.95))
    log = open(os.path.join(out, "log.jsonl"), "a")
    meta = dict(kind="meta", args=vars(a), params=pt, flops_per_token=flops_per_token(m, a.T),
                n_fill=S.get("n_fill"), time=time.strftime("%F %T"))
    log.write(json.dumps(meta) + "\n"); log.flush()
    print(f"{a.name}: {a.arch} on {a.data}, compute {pt['compute']:,} params, {meta['flops_per_token']/1e6:.2f} MFLOP/token",
          flush=True)
    real_seen = 0; step = 0; t0 = time.time(); next_eval = a.eval_every_tokens
    acc = {}
    m.train()
    stop_file = os.path.join(out, "STOP")
    while real_seen < a.real_tokens and (not a.max_steps or step < a.max_steps):
        if step % 25 == 0 and os.path.exists(stop_file):
            log.write(json.dumps(dict(kind="stopfile", step=step, real_tokens=real_seen)) + "\n"); log.flush()
            break
        tt, dd = stream.next()
        ids, doc, ei = D.to_batch(tt, dd, canon, m.ecfg)
        if step == 0 and is_v4(m):                                     # k-means++ on complete chunks of 3 batches
            with torch.no_grad():
                cands = []
                for bi in range(3):
                    b_ = (ids, doc, ei) if bi == 0 else D.to_batch(*stream.next(), canon, m.ecfg)
                    _, c_hat, _, _ = m.enc(*b_)
                    cands.append(c_hat[M4.chunk_masks(b_[0], b_[1], K)[1]])
                cands = torch.cat(cands)
                assert len(cands) >= m.core.N and len(cands) >= m.core.K, len(cands)
                m.core.init_codebooks(cands, torch.Generator().manual_seed(a.seed))
        elif step == 0:                                                 # consume the same 2 batches (identical rows)
            stream.next(); stream.next()
        prog = min(1.0, real_seen / a.real_tokens)
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lr
        loss, terms, aux = loss_fn(m, ids, doc, ei)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip)
        if not math.isfinite(loss.item()) or not torch.isfinite(gn):
            log.write(json.dumps(dict(kind="nonfinite", step=step, terms=terms)) + "\n"); log.flush()
            raise SystemExit(f"non-finite at step {step}")
        opt.step()
        step += 1
        real_seen += int((ids != 0).sum())
        if is_v4(m) and step % a.restart_every == 0:
            r = m.core.restart_dead(aux["c_hat"].detach()[aux["complete"]], opt)
            log.write(json.dumps(dict(kind="restart", step=step, **r)) + "\n")
        for kk, v in terms.items():
            acc[kk] = acc.get(kk, 0.0) + v
        acc["n"] = acc.get("n", 0) + 1
        if step % 50 == 0:
            el = time.time() - t0
            rec = dict(kind="train", step=step, real_tokens=real_seen, lr=lr, gn=float(gn), wall=el,
                       tok_s=real_seen / el, **{kk: v / acc["n"] for kk, v in acc.items() if kk != "n"})
            acc = {}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 200 == 0:
                print(f"{time.strftime('%T')} {a.name} step {step} real {real_seen/1e6:.2f}M ntp {rec['ntp']:.3f} "
                      f"gn {rec['gn']:.2f} {rec['tok_s']:.0f} real tok/s", flush=True)
        if real_seen >= next_eval:
            next_eval += a.eval_every_tokens
            ev = dict(kind="eval", step=step, real_tokens=real_seen, wall=time.time() - t0, **periodic_eval(m, a, S))
            log.write(json.dumps(ev) + "\n"); log.flush()
            print(f"{time.strftime('%T')} {a.name} EVAL step {step} real {real_seen/1e6:.2f}M "
                  f"{json.dumps({k: v for k, v in ev.items() if k not in ('kind', 'step', 'real_tokens', 'wall')})[:400]}",
                  flush=True)
    torch.save(dict(model=m.state_dict(), arch=a.arch, cfg=m.c, args=vars(a), step=step, real_tokens=real_seen),
               os.path.join(out, "ckpt.pt"))
    res = final_eval(m, a, S)
    res.update(step=step, real_tokens=real_seen, wall=time.time() - t0, params=pt, args=vars(a),
               flops_per_token=flops_per_token(m, a.T))
    json.dump(res, open(os.path.join(out, "final.json"), "w"), indent=1)
    log.write(json.dumps(dict(kind="final", step=step, real_tokens=real_seen)) + "\n")
    log.close()
    print(f"{time.strftime('%T')} {a.name} FINAL {json.dumps({k: v for k, v in res.items() if k in ('val_ntp', 'probes', 'heldin', 'heldout')})[:600]}",
          flush=True)


if __name__ == "__main__":
    main()
