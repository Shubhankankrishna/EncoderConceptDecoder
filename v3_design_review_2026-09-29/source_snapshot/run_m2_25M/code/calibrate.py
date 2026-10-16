"""Calibration on MPS (PROMPT 6.1): micro-batch B in {4, 2, 1} at T = 1024; 10 warm-up + 30 timed micro-steps each,
with real optimizer steps every `accum` micro-steps (accum = 32 / B so that a step sees 32,768 tokens). Keeps the
largest B whose peak allocated memory is <= 75% of torch.mps.recommended_max_memory() and causes no swap growth.
Writes runs/calibration.json. Each B runs in a fresh subprocess (one MPS job at a time, clean allocator)."""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "runs", "calibration.json")


def swap_mb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(s.split("used =")[1].split("M")[0])


def one(B, warm=10, timed=30):
    import random
    import numpy as np
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, HERE)
    from model import build_model, forward, param_groups
    from data import ShardStream, target_mask
    dev = torch.device("mps")
    torch.manual_seed(0); random.seed(0)
    m, ecfg, dcfg = build_model()
    m = m.to(dev).train()
    opt = torch.optim.AdamW(param_groups(m, 0.1), lr=1e-4, betas=(0.9, 0.95))
    stream = ShardStream(os.path.join(HERE, "..", "data"), "train", ecfg, B, seed=0)
    accum = 32 // B
    rec = torch.mps.recommended_max_memory()
    peak_fwd = peak_after = 0
    sw0 = swap_mb()
    times = []
    R = 4
    for i in range(warm + timed):
        if i % accum == 0:
            R = random.randint(1, 4)
        torch.mps.synchronize(); t0 = time.time()
        ids, doc, ei = [x.to(dev) for x in stream.next()]
        logits, x = forward(m, ids, doc, ei, R)
        msk = target_mask(ids, doc)
        ntp = F.cross_entropy(logits[:, :-1][msk].float(), ids[:, 1:][msk])
        ncp, vq = m.core.losses(x["c_hat"], x["logits_c"], x["cdoc"])
        peak_fwd = max(peak_fwd, torch.mps.current_allocated_memory())
        ((ntp + 0.1 * ncp + vq) / accum).backward()
        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        peak_after = max(peak_after, torch.mps.current_allocated_memory())
        if i >= warm:
            times.append(time.time() - t0)
    stream.close()
    tok_s = B * 1024 * len(times) / sum(times)
    res = dict(B=B, accum=accum, tok_s=tok_s, sec_per_micro=sum(times) / len(times),
               peak_alloc_after_forward_gib=peak_fwd / 2**30, peak_alloc_gib=max(peak_fwd, peak_after) / 2**30,
               driver_gib=torch.mps.driver_allocated_memory() / 2**30, recommended_max_gib=rec / 2**30,
               frac_of_recommended=max(peak_fwd, peak_after) / rec, swap_delta_mb=swap_mb() - sw0)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        one(int(sys.argv[2])); sys.exit(0)
    results = []
    for B in (4, 2, 1):
        p = subprocess.run([sys.executable, __file__, "--one", str(B)], capture_output=True, text=True,
                           env=dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1"))
        line = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if not line:
            print(p.stdout[-2000:], p.stderr[-3000:]); results.append(dict(B=B, error=p.stderr[-500:])); continue
        r = json.loads(line[-1]); results.append(r); print(r, flush=True)
    ok = [r for r in results if "error" not in r and r["frac_of_recommended"] <= 0.75 and r["swap_delta_mb"] <= 50]
    choice = max(ok, key=lambda r: r["B"]) if ok else None
    out = dict(time=time.strftime("%Y-%m-%d %H:%M"), results=results, choice=choice)
    if choice:
        tokens_per_step = 32768
        sec_per_step = tokens_per_step / choice["tok_s"]
        budget = 24 * 3600 * 0.95                          # 5% of the 24 h for evaluations
        out.update(tokens_per_step=tokens_per_step, sec_per_step=sec_per_step,
                   total_steps=int(budget / sec_per_step), tokens_24h=int(budget / sec_per_step) * tokens_per_step)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))
