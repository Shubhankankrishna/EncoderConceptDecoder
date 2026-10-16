"""Interleaved throughput A/B on MPS (thermal drift on the fanless M2 Air makes sequential runs unusable: the same base
config measured 2,059 then 1,478 tok/s three minutes apart). One process, one model, real optimizer steps (32,768 tokens
each); configurations alternate every optimizer step (A B A B ...), so drift affects both equally. Reports the median
seconds per step and tok/s per configuration and the paired ratio (median of B_i / A_i over adjacent pairs).

Configs (comma list): base | block64 | block128 | memo0 | b4 | b4block64 | compcore (torch.compile core layers, train only)
  python ab_interleave.py --configs base,block64 --steps 16
"""
import argparse, os, statistics, time
import torch, torch.nn.functional as F
import speed_v3 as SV
from model import build_model, chunk_masks, param_groups
import train as TR
from data import ShardStream

ap = argparse.ArgumentParser()
ap.add_argument("--configs", default="base,block64")
ap.add_argument("--steps", type=int, default=16)                  # measured steps per config (after 2 warm-up rounds)
ap.add_argument("--warm", type=int, default=3)
ap.add_argument("--no_memo", action="store_true")                   # memo off in every arm (required with compile)
a = ap.parse_args()
DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "run_m2_25M", "data"))
dev = torch.device("mps")
torch.manual_seed(0)
m, ecfg, _ = build_model("full")
m = m.to(dev).train()
opt = torch.optim.AdamW(param_groups(m)[0], lr=1e-4, betas=(0.9, 0.95), eps=1e-8)
stream = ShardStream(DATA, "train", ecfg, 2, seed=0, prefetch=20)
gen = torch.Generator().manual_seed(1)
cfgs = a.configs.split(",")
comp_state = {"on": False}
comp_state["all"] = False
if any("comp" in c for c in cfgs):
    for l in m.core.layers:
        SV.compile_layer(l, lambda l_: comp_state["on"])
if any("compall" in c for c in cfgs):
    for l in list(m.enc.layers) + list(m.dec.layers):
        SV.compile_layer(l, lambda l_: comp_state["all"])


class Args:
    lam_ncp, lam_s, lam_c, main_sched = 0.1, 0.05, 0.1, "3"


def one_step(cfg):
    SV.enable_block_attention(128 if "block128" in cfg else 64 if "block64" in cfg else 0)
    comp_state["on"] = "comp" in cfg
    comp_state["all"] = "compall" in cfg
    memo = not a.no_memo and cfg not in ("memo0", "compcore")
    batches = [stream.next() for _ in range(16)]
    if cfg.startswith("b4"):
        batches = [tuple(torch.cat([x, y]) for x, y in zip(batches[i], batches[i + 1])) for i in range(0, 16, 2)]
    N = TR.batch_counts(batches)
    torch.mps.synchronize(); t0 = time.time()
    opt.zero_grad(set_to_none=True)
    for ids, doc, ei in batches:
        ids, doc, ei = ids.to(dev), doc.to(dev), ei.to(dev)
        short = TR.sample_short(gen)
        if memo:
            with SV.step_memo():
                TR.micro_step(m, ids, doc, ei, short, Args, N)
        else:
            TR.micro_step(m, ids, doc, ei, short, Args, N)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    torch.mps.synchronize()
    drv[cfg] = max(drv.get(cfg, 0.0), torch.mps.driver_allocated_memory() / 2**30)
    return time.time() - t0


import subprocess
swap = lambda: subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split("used =")[1].split("M")[0].strip()
drv = {}
swap0 = swap()
times = {c: [] for c in cfgs}
t_start = time.time()
for r in range(a.warm + a.steps):
    order = cfgs if r % 2 == 0 else cfgs[::-1]                      # ABBA ordering cancels within-round drift too
    for c in order:
        dt = one_step(c)
        if r >= a.warm:
            times[c].append(dt)
    if r == a.warm - 1:
        print(f"warm-up done ({time.time() - t_start:.0f}s incl. any compile)", flush=True)
base = cfgs[0]
print(f"interleaved A/B, {a.steps} measured steps per config, alloc {torch.mps.current_allocated_memory()/2**30:.2f} GiB, "
      f"driver {torch.mps.driver_allocated_memory()/2**30:.2f} GiB")
for c in cfgs:
    med = statistics.median(times[c])
    ratio = statistics.median([b / x for b, x in zip(times[c], times[base])])
    q = sorted(times[c]); lo, hi = q[len(q) // 4], q[(3 * len(q)) // 4]
    print(f"  {c:10s} driver max {drv[c]:.2f} GiB  median {med:.2f} s/step ({32768/med:5.0f} tok/s)  IQR {lo:.2f}-{hi:.2f}  paired time ratio vs {base}: {ratio:.3f}"
          f"  -> {100 * (1 / ratio - 1):+.1f} % tok/s")
print(f"swap before {swap0} MB, after {swap()} MB")
stream.close()
