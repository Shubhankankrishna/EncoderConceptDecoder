"""Train and evaluate one tiny v4 arm on CPU (v4_tiny experiment). Never touches the v3 run or the GPU.

python run_v4tiny.py --name k6_attnffn_s0 --k 6 --pool attnffn --seed 0 --real_tokens 12e6
Writes v4_tiny/runs/<name>/{log.jsonl, final.json, ckpt.pt}.
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
from tokenizers import Tokenizer

import model_v4 as M4
import data_v4 as D

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", "runs"))


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--pool", default="attnffn", choices=["attnffn", "attn", "linear", "gconv"])
    ap.add_argument("--enc_loop", type=int, default=2)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=480)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--p_syn", type=float, default=0.5)
    ap.add_argument("--real_tokens", type=float, default=12e6, help="stop after this many real (non-pad) input tokens")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--eval_every_tokens", type=float, default=1.5e6)
    ap.add_argument("--n_probe", type=int, default=128, help="items per family in periodic evals")
    ap.add_argument("--n_probe_final", type=int, default=192)
    ap.add_argument("--restart_every", type=int, default=200)
    ap.add_argument("--block", type=int, default=60, help="block-local window attention (0 = full-mask SDPA)")
    ap.add_argument("--over", default="", help="extra model config overrides, e.g. d=96,enc_ff=384")
    return ap.parse_args()


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def val_ntp(m, rows, canon, chunk=8, **kw):
    t, d = rows
    tot = n = 0.0
    for a in range(0, len(t), chunk):
        ids, doc, ei = D.to_batch(t[a:a + chunk], d[a:a + chunk], canon, m.ecfg)
        lg, _ = M4.forward(m, ids, doc, ei, **kw)
        tm = M4.target_mask(ids, doc)
        tgt = ids[:, 1:].masked_fill(~tm, -100)
        tot += F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=-100,
                               reduction="sum").item()
        n += tm.sum().item()
    return tot / max(n, 1)


@torch.no_grad()
def item_acc(m, items, canon, T, k, chunk=8, **kw):
    """Teacher-forced greedy exact match: every answer token of every span is the argmax (prediction for token i is
    logits[i-1]). plan_swap rolls plans across rows within a forward chunk, so chunks hold >= 2 rows."""
    t, d, where = D.item_rows(items, T, k)
    by_row = {}
    for idx, (r, off, L) in enumerate(where):
        by_row.setdefault(r, []).append((idx, off))
    correct = np.zeros(len(items), bool)
    for a in range(0, len(t), chunk):
        rs = list(range(a, min(a + chunk, len(t))))
        if len(rs) < 2:
            rs = [rs[0] - 1, rs[0]] if rs[0] > 0 else rs
        ids, doc, ei = D.to_batch(t[rs], d[rs], canon, m.ecfg)
        lg, _ = M4.forward(m, ids, doc, ei, **kw)
        pred = lg.argmax(-1)
        for bi, r in enumerate(rs):
            if r < a:
                continue
            for idx, off in by_row.get(r, []):
                okk = True
                for (s, e, _) in items[idx]["spans"]:
                    okk &= bool((pred[bi, off + s - 1: off + e - 1] == ids[bi, off + s: off + e]).all())
                correct[idx] = okk
    fams = {}
    for it, c in zip(items, correct):
        fams.setdefault(it["family"], []).append(c)
    return {f: float(np.mean(v)) for f, v in fams.items()}


@torch.no_grad()
def codebook_health(m, rows, canon):
    t, d = rows
    ids, doc, ei = D.to_batch(t[:16], d[:16], canon, m.ecfg)
    lg, aux = M4.forward(m, ids, doc, ei)
    ne, comp, pair = M4.chunk_masks(ids, doc, m.k)
    n0, n, r = m.core.assign(aux["c_hat"])
    c = aux["c_hat"][comp]
    q0 = m.core.E0[n0][comp]
    qf = m.core.e[torch.arange(m.core.S), n][comp].flatten(-2)
    var = (c - c.mean(0)).pow(2).sum(-1).mean()
    ve0 = 1 - (c - q0).pow(2).sum(-1).mean() / var
    ve = 1 - (c - q0 - qf).pow(2).sum(-1).mean() / var
    p0 = torch.bincount(n0[comp], minlength=m.core.K).float(); p0 = p0 / p0.sum()
    ppl0 = math.exp(-(p0[p0 > 0] * p0[p0 > 0].log()).sum().item())
    cec, cef = m.core.ncp_terms(aux["q"], aux["logits_c"], n0, n, pair)
    npair = max(int(pair.sum()), 1)
    return dict(ve_coarse=float(ve0), ve=float(ve), coarse_ppl=ppl0, coarse_ppl_frac=ppl0 / m.core.K,
                ce_coarse=float(cec) / npair, ce_fine=float(cef) / npair, ln_K=math.log(m.core.K))


@torch.no_grad()
def pool_telemetry(m, text_rows, items, canon, T, k, freq_top, tok):
    """Pooler weights vs the uniform baseline (1 / real tokens in the chunk):
    text: mean (alpha / uniform) on the 100 most frequent tokens vs the rest, per head; per-slot profile.
    probes: mean (alpha / uniform) on fact content (names, values) vs fact template words (' The code of', ' is', '.')."""
    if m.c["pool"] == "linear":
        return None
    out = {}
    t, d = text_rows
    ids, doc, ei = D.to_batch(t[:32], d[:32], canon, m.ecfg)
    _, aux = M4.forward(m, ids, doc, ei)
    a = aux["enc"]["pool_attn"]                                          # (B, M, k, H)
    B_, Mn = ids.shape[0], ids.shape[1] // k
    real = (ids != 0).view(B_, Mn, k)
    nreal = real.sum(-1, keepdim=True).clamp_min(1)
    ratio = a / (1.0 / nreal)[..., None]                                 # alpha / uniform
    top = torch.zeros(8192, dtype=torch.bool); top[torch.as_tensor(freq_top)] = True
    is_top = top[ids].view(B_, Mn, k) & real
    is_rest = (~top[ids]).view(B_, Mn, k) & real
    multi = (nreal.squeeze(-1) >= 2)[..., None].expand_as(real)
    H = a.shape[-1]
    out["text_top100_over_uniform"] = [float(ratio[..., h][is_top & multi].mean()) for h in range(H)]
    out["text_rest_over_uniform"] = [float(ratio[..., h][is_rest & multi].mean()) for h in range(H)]
    full = real.all(-1)
    out["slot_profile"] = [[float(a[..., s, h][full].mean()) for s in range(k)] for h in range(H)]
    ent = -(a.clamp_min(1e-12).log() * a).sum(2)[full]                  # (n, H)
    out["entropy_over_log_k"] = [float(ent[:, h].mean() / math.log(k)) for h in range(H)]
    # probe facts
    tmpl = set(tok.encode(" The code of").ids) | set(tok.encode(" is").ids) | {tok.encode(".").ids[0]}
    TC = tok.encode(" The code of").ids; Q = tok.encode(" What is the code of").ids
    pt, pd, where = D.item_rows(items[:96], T, k)
    ids, doc, ei = D.to_batch(pt, pd, canon, m.ecfg)
    _, aux = M4.forward(m, ids, doc, ei)
    a = aux["enc"]["pool_attn"]
    Bp = ids.shape[0]
    realp = (ids != 0).view(Bp, -1, k)
    ratio = a / (1.0 / realp.sum(-1, keepdim=True).clamp_min(1))[..., None]
    content, template = [], []
    for idx, (r, off, L) in enumerate(where):
        seq = items[idx]["ids"]
        qpos = next(i for i in range(len(seq)) if seq[i:i + len(Q)] == Q)
        i = 0
        while i < qpos:
            if seq[i:i + len(TC)] == TC:
                j = i
                while j < qpos and seq[j] != tok.encode(".").ids[0]:
                    j += 1
                for p_ in range(i, j + 1):
                    pos = off + p_
                    v = ratio[r, pos // k, pos % k]                      # (H,)
                    (template if seq[p_] in tmpl else content).append(v)
                i = j + 1
            else:
                i += 1
    if content and template:
        out["fact_content_over_uniform"] = torch.stack(content).mean(0).tolist()
        out["fact_template_over_uniform"] = torch.stack(template).mean(0).tolist()
    return out


def full_eval(m, a, canon, vrows, items, T, k, freq_top, tok):
    m.eval()
    res = dict(val_ntp=val_ntp(m, vrows, canon))
    res["probes"] = item_acc(m, items, canon, T, k)
    inter = dict(core_off=dict(core_off=True), plan_swap=dict(plan_swap=True), c_off=dict(use_c=False), M1=dict(M=1),
                 loop_off=dict(enc_loop=1), dec_reads_off=dict(dec_reads_off=(0, 1)))
    if m.c["pool"] in ("attnffn", "gconv"):
        inter["pool_ffn_off"] = dict(pool_ffn_off=True)
    res["interventions"] = {}
    for nme, kw in inter.items():
        res["interventions"][nme] = dict(val_ntp=val_ntp(m, vrows, canon, **kw), probes=item_acc(m, items, canon, T, k, **kw))
    res["codebook"] = codebook_health(m, vrows, canon)
    res["pool"] = pool_telemetry(m, vrows, items, canon, T, k, freq_top, tok)
    m.train()
    return res


# ----------------------------------------------------------------------------- training
def main():
    a = get_args()
    torch.set_num_threads(a.threads)
    if a.block:
        import speed_v3
        speed_v3.enable_block_attention(a.block)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    out = os.path.join(RUNS, a.name); os.makedirs(out, exist_ok=True)
    over = {}
    for kv in filter(None, a.over.split(",")):
        kk, vv = kv.split("="); over[kk] = type(M4.TINY[kk])(vv) if not isinstance(M4.TINY[kk], tuple) else eval(vv)
    m = M4.build_model("tiny", k=a.k, pool=a.pool, enc_loop=a.enc_loop, M=a.M, **over)
    pt = M4.param_table(m)
    tok = Tokenizer.from_file(os.path.join(D.SRC, "tokenizer.json"))
    tr, va = D.TextStore("train"), D.TextStore("val")
    canon = D.Canon()
    fill = D.filler_sentences(tr, tok)
    probes_tr = D.TinyProbes(tok, fill, "train")
    probes_va = D.TinyProbes(tok, D.filler_sentences(tr, tok, seed=99), "val")
    freq_top = np.argsort(-D.token_freq(tr))[:100]
    stream = D.MixedStream(tr, probes_tr, a.T, a.k, a.B, a.p_syn, seed=a.seed)
    vrows = D.val_text_rows(va, a.T, a.k, n_rows=64)
    items_q = D.probe_items(probes_va, a.n_probe, seed=1000)
    items_f = D.probe_items(probes_va, a.n_probe_final, seed=2000)
    opt = torch.optim.AdamW(M4.param_groups(m, a.wd), lr=a.lr, betas=(0.9, 0.95))
    log = open(os.path.join(out, "log.jsonl"), "a")
    meta = dict(kind="meta", args=vars(a), params=pt, time=time.strftime("%F %T"))
    log.write(json.dumps(meta) + "\n"); log.flush()
    print(f"{a.name}: compute {pt['compute']:,} params (pooler {pt['pooler']:,}), k={a.k} pool={a.pool} loop={a.enc_loop}",
          flush=True)
    real_seen = 0; step = 0; t0 = time.time(); next_eval = a.eval_every_tokens
    last_comp = None; acc = {}
    m.train()
    stop_file = os.path.join(out, "STOP")
    while real_seen < a.real_tokens and (not a.max_steps or step < a.max_steps):
        if step % 25 == 0 and os.path.exists(stop_file):               # owner window: stop early, still run the final eval
            log.write(json.dumps(dict(kind="stopfile", step=step, real_tokens=real_seen)) + "\n"); log.flush()
            break
        tt, dd = stream.next()
        ids, doc, ei = D.to_batch(tt, dd, canon, m.ecfg)
        if step == 0:                                                  # k-means++ on complete chunks of 3 batches
            with torch.no_grad():
                cands = []
                for bi in range(3):
                    b_ = (ids, doc, ei) if bi == 0 else D.to_batch(*stream.next(), canon, m.ecfg)
                    _, c_hat, _, _ = m.enc(*b_)
                    cands.append(c_hat[M4.chunk_masks(b_[0], b_[1], a.k)[1]])
                cands = torch.cat(cands)
                assert len(cands) >= m.core.N and len(cands) >= m.core.K, len(cands)
                m.core.init_codebooks(cands, torch.Generator().manual_seed(a.seed))
        prog = min(1.0, real_seen / a.real_tokens)
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = lr
        loss, terms, aux = M4.loss_terms(m, ids, doc, ei)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip)
        if not math.isfinite(loss.item()) or not torch.isfinite(gn):
            log.write(json.dumps(dict(kind="nonfinite", step=step, terms=terms)) + "\n"); log.flush()
            raise SystemExit(f"non-finite at step {step}")
        opt.step()
        step += 1
        real_seen += int((ids != 0).sum())
        last_comp = aux["c_hat"].detach()[aux["complete"]]
        if step % a.restart_every == 0:
            r = m.core.restart_dead(last_comp, opt)
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
                      f"cec {rec['ce_coarse']:.2f} cef {rec['ce_fine']:.2f} vqc {rec['vq_coarse']:.2f} vqf {rec['vq_fine']:.2f} "
                      f"gn {rec['gn']:.2f} {rec['tok_s']:.0f} real tok/s", flush=True)
        if real_seen >= next_eval:
            next_eval += a.eval_every_tokens
            m.eval()
            ev = dict(kind="eval", step=step, real_tokens=real_seen, wall=time.time() - t0,
                      val_ntp=val_ntp(m, vrows, canon), probes=item_acc(m, items_q, canon, a.T, a.k),
                      swap=item_acc(m, [x for x in items_q if x["family"] == "lr_recall1"], canon, a.T, a.k,
                                    plan_swap=True).get("lr_recall1"),
                      codebook=codebook_health(m, vrows, canon))
            m.train()
            log.write(json.dumps(ev) + "\n"); log.flush()
            print(f"{time.strftime('%T')} {a.name} EVAL step {step} real {real_seen/1e6:.2f}M val {ev['val_ntp']:.3f} "
                  f"probes {ev['probes']} swap {ev['swap']} ppl0 {ev['codebook']['coarse_ppl']:.1f} VE {ev['codebook']['ve']:.3f}",
                  flush=True)
    torch.save(dict(model=m.state_dict(), cfg=m.c, args=vars(a), step=step, real_tokens=real_seen),
               os.path.join(out, "ckpt.pt"))
    res = full_eval(m, a, canon, vrows, items_f, a.T, a.k, freq_top, tok)
    res.update(step=step, real_tokens=real_seen, wall=time.time() - t0, params=pt, args=vars(a))
    json.dump(res, open(os.path.join(out, "final.json"), "w"), indent=1)
    log.write(json.dumps(dict(kind="final", **{kk: v for kk, v in res.items() if kk in ("val_ntp", "probes")})) + "\n")
    log.close()
    print(f"{time.strftime('%T')} {a.name} FINAL val {res['val_ntp']:.3f} probes {res['probes']} "
          f"core_off {res['interventions']['core_off']['probes']} swap {res['interventions']['plan_swap']['probes']}",
          flush=True)


if __name__ == "__main__":
    main()
