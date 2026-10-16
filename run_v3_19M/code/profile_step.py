"""Where does one v3 training micro-step spend its time on MPS? (full model, real 80M table, real data, B2)
Sections are separated by torch.mps.synchronize(); medians over N repeats after warm-up.

  python profile_step.py [--block_attn 64] [--micro_bs 2]
"""
import argparse, os, time, statistics
import numpy as np
import torch, torch.nn.functional as F
import speed_v3 as SV
from model import build_model, chunk_masks, param_groups, rms
from data import ShardStream, target_mask

ap = argparse.ArgumentParser()
ap.add_argument("--block_attn", type=int, default=0)
ap.add_argument("--micro_bs", type=int, default=2)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--eval", action="store_true")
a = ap.parse_args()
DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run_m2_25M", "data"))
dev = torch.device("mps")
torch.manual_seed(0)
SV.enable_block_attention(a.block_attn)
m, ecfg, _ = build_model("full")
m = m.to(dev).train()
opt = torch.optim.AdamW(param_groups(m)[0], lr=1e-4, betas=(0.9, 0.95), eps=1e-8)
stream = ShardStream(DATA, "train", ecfg, a.micro_bs, seed=0, prefetch=4)
sync = torch.mps.synchronize
T = {k: [] for k in ("fetch", "enc_fwd", "core_fwd", "head_dec_fwd", "loss", "backward", "opt_step", "micro_total")}


def tick():
    sync(); return time.time()


for it in range(a.n + 2):
    t0 = time.time(); ids, doc, ei = stream.next(); t1 = time.time()
    ids, doc, ei = ids.to(dev), doc.to(dev), ei.to(dev)
    tA = tick()
    with SV.step_memo():
        tok, c_hat, _ = m.enc(ids, doc, ei); tB = tick()
        nonempty, complete, pair_ok = chunk_masks(ids, doc)
        vq, n = m.core.vq_sum(c_hat, complete, count_hits=True)
        S3, states, Ss = m.core.run(c_hat, doc[:, ::4], "3", short="2b"); tC = tick()
        u = m.core.out_norm(S3); lc, cp = m.core.head(u)
        logits, _ = m.dec(tok, u, cp, doc, m.enc.embed.weight); tD = tick()
        tgt = ids[:, 1:].masked_fill(~target_mask(ids, doc), -100)
        ntp = F.cross_entropy(logits[:, :-1].reshape(-1, 8192), tgt.reshape(-1), ignore_index=-100)
        us = m.core.out_norm(Ss); lcs, _ = m.core.head(us, pred=False)
        loss = ntp + 0.1 * m.core.ncp_sum(lc, n, pair_ok) / pair_ok.sum() + 0.05 * m.core.ncp_sum(lcs, n, pair_ok) / pair_ok.sum() \
            + 0.1 * ((rms(Ss) - rms(S3).detach()).pow(2).mean(-1) * nonempty).sum() / nonempty.sum() + vq / complete.sum()
        tE = tick()
        loss.backward(); tF = tick()
    opt.step(); opt.zero_grad(set_to_none=True); tG = tick()
    if it >= 2:
        for k, v in zip(T, (t1 - t0, tB - tA, tC - tB, tD - tC, tE - tD, tF - tE, tG - tF, tF - tA)):
            T[k].append(v)
med = {k: statistics.median(v) for k, v in T.items()}
micro = med["micro_total"]
print(f"block_attn {a.block_attn}, micro_bs {a.micro_bs}: median seconds per section (n={a.n})")
for k, v in med.items():
    print(f"  {k:14s} {v*1000:8.1f} ms  {100*v/micro:5.1f} % of micro-step" if k not in ("fetch", "opt_step", "micro_total")
          else f"  {k:14s} {v*1000:8.1f} ms")
tok_per_micro = a.micro_bs * 1024
step = 16 // (a.micro_bs // 2) * micro + med["opt_step"]
print(f"  => est. {32768 / step:.0f} tok/s per 32,768-token step (micro x {32768 // tok_per_micro} + opt step; fetch overlapped)")
print(f"  mps alloc {torch.mps.current_allocated_memory()/2**30:.2f} GiB driver {torch.mps.driver_allocated_memory()/2**30:.2f} GiB")
if a.eval:
    from evaluate import Evaluator
    ev = Evaluator(DATA, ecfg, dev, batch_rows=4)
    m.eval(); t0 = time.time(); r = ev.quick(m, R="3"); print(f"  quick eval: {time.time()-t0:.0f} s (val {r['val_lang_mean']:.3f})")
stream.close()
