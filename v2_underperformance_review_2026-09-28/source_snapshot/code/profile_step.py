"""Time the pieces of one v2 training micro-batch on MPS (synchronised wall clock). python profile_step.py [--B 2] [--K 10]"""
import argparse, contextlib, time, os
import numpy as np
import torch
import torch.nn.functional as F
from model import build_model, nextlat_losses
from data import ShardStream, target_mask
import train as T

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=2)
ap.add_argument("--K", type=int, default=10)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--amp", default="none")
ap.add_argument("--cf", type=float, default=1.5)
ap.add_argument("--cache", type=int, default=1)
ap.add_argument("--memo", type=int, default=1)
a = ap.parse_args()
dev = torch.device("mps")
torch.manual_seed(0)
m, ecfg, _ = build_model(); m = m.to(dev).train()
for l in m.core.layers: l.cf = a.cf
import speed, hrm_encoder
if not a.memo: hrm_encoder.same_doc_shift_mask = speed._orig_mask
data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run_m2_25M", "data")
st = ShardStream(data, "train", ecfg, a.B)
ids, doc, ei = [x.to(dev) for x in st.next()]
amp = (lambda: torch.autocast("mps", dtype=torch.bfloat16)) if a.amp == "bf16" else \
      (lambda: torch.autocast("mps", dtype=torch.float16)) if a.amp == "fp16" else contextlib.nullcontext
sync = torch.mps.synchronize
tm = {}


def tic(k, t0):
    sync(); tm[k] = tm.get(k, 0) + time.time() - t0; return time.time()


for rep in range(a.reps + 1):
    if rep == 1:
        tm.clear()
    sync(); t = time.time(); t_all = t
    with amp():
        tok, c_hat, _ = m.enc(ids, doc, ei)
    t = tic("enc_fwd", t)
    core = m.core; cdoc = doc[:, ::4]; mask = core.mask(cdoc)
    vq, tg = core.vocab.vq_loss(c_hat.float())
    cl = c_hat.detach().requires_grad_(); tl = tok.detach().requires_grad_()
    t = tic("vq", t)
    with amp():
        X, S = core.init_state(cl)
    losses = 0; cache = {} if a.cache else None
    for r in range(a.K):
        with amp():
            S, aux = core.one_pass(X, S, mask, first=(r == 0), cache=cache)
        t = tic("core_pass_fwd", t)
        with amp():
            u = core.readout(S); lc, cp = core.head(u.float())
        ce = core.ncp_ce(lc.float(), tg, cdoc)
        t = tic("ncp_head_fwd", t)
        losses = losses + ce + sum(x["lb"] for x in aux)
    with amp():
        logits, daux = m.dec(tl, u, cp, doc, m.enc.embed.weight)
    t = tic("dec_fwd", t)
    tgt = ids[:, 1:].masked_fill(~target_mask(ids, doc), -100)
    ntp = F.cross_entropy(logits[:, :-1].reshape(-1, 8192).float(), tgt.reshape(-1), ignore_index=-100)
    t = tic("ntp_ce", t)
    with amp():
        lh, lkl = nextlat_losses(m, daux["h"], logits, ids, doc)
    t = tic("nextlat_fwd", t)
    (losses + ntp + lh + lkl + vq).backward()
    t = tic("backward_core_dec", t)
    torch.autograd.backward([tok, c_hat], [tl.grad, cl.grad])
    t = tic("backward_enc", t)
    m.zero_grad(set_to_none=True)
    tm["total"] = tm.get("total", 0) + time.time() - t_all if rep else 0
tot = tm["total"]
print(f"cf {a.cf} cache {a.cache} memo {a.memo} B {a.B} K {a.K} amp {a.amp}: {tot / a.reps * 1000:.0f} ms per micro-batch -> {a.B * 1024 * a.reps / tot:.0f} tok/s (no optimizer step)")
for k, v in sorted(tm.items(), key=lambda x: -x[1]):
    if k != "total":
        print(f"  {k:20s} {v / a.reps * 1000:7.0f} ms  {100 * v / tot:5.1f} %")
