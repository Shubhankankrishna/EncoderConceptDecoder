"""Phase A: train the reduced NCP foundation (or the V0 transformer) on the TinyStories + GraphNav mix.

usage: python phase_a.py --arch ncp|v0 --seed 0 --steps 18000 --out <name>
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

import data_v5 as dv
from model_v5 import NCPModel, TransformerLM, token_nll, concept_mask, ncp_mse, count_params
from common_v5 import (DEV, RUNS, Mixer, domain_mean, domain_mean_rows, cosine_lr, make_opt, eval_nll, Logger,
                       save_json)

ALPHA, BETA = 1.0, 1.0      # NCP-MSE (per-feature mean) and VQ weights; declared, not paper values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="ncp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=18000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--alpha", type=float, default=ALPHA, help="NCP-MSE weight (slack ablation: 0)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = os.path.join(RUNS, a.out); os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.jsonl"))
    model = (NCPModel(dv.VOCAB) if a.arch == "ncp" else TransformerLM(dv.VOCAB)).to(DEV)
    n_par = count_params(model)
    print(f"{a.arch} params {n_par:,}", flush=True)
    if a.arch == "ncp":     # codebook: own group, 10x lr, no weight decay (declared; VQ loss is codebook-only)
        opt = make_opt([p for n, p in model.named_parameters() if n != "Q"], a.lr)
        opt.add_param_group({"params": [model.Q], "weight_decay": 0.0, "lr_mult": 10.0})
    else:
        opt = make_opt(model.parameters(), a.lr)
    mix = Mixer(seed=1000 + a.seed)
    is_ncp = a.arch == "ncp"

    def per_tok(x, m):
        """Compiled region: forward + per-token/per-concept tensors only.  All reductions happen eagerly
        (inductor-MPS miscomputes reductions over sliced tensors; found by test_v5 eager-vs-compiled check)."""
        if is_ncp:
            o = model(x)
            nll = token_nll(o["logits"], x, m)
            err = (o["chat"][:, :-1] - o["C"][:, 1:].detach()).pow(2).mean(-1)
            return nll, err, o["C"]
        return token_nll(model(x), x, m), None, None

    def loss_fn(x, m):
        nll, err, C = fwd(x, m)
        l_ntp, lt, lg = domain_mean(nll, m)
        if is_ncp:
            cm = concept_mask(m)
            l_ncp = domain_mean_rows(err, cm)
            q, l_vq = model.vq(C)
            return l_ntp + a.alpha * l_ncp + BETA * l_vq, (lt, lg, l_ncp, l_vq, q, C)
        z = torch.zeros((), device=x.device)
        return l_ntp, (lt, lg, z, z, None, None)

    fwd = per_tok if a.no_compile else torch.compile(per_tok)
    f = loss_fn
    if is_ncp:                                   # data-dependent codebook init (outside compile)
        x, m, _ = mix.batch()
        with torch.no_grad():
            model.vq_init_or_restart(model.pool(model.encode(x)))
    t0 = time.time(); ntok = 0
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, a.steps, a.lr) * g.get("lr_mult", 1.0)
        x, m, _ = mix.batch()
        loss, (lt, lg, lncp, lvq, q, C) = f(x, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ntok += x.numel()
        if is_ncp:
            model.count_usage(q)
            if step > 0 and step % 250 == 0 and step < 0.8 * a.steps:
                nd = model.vq_init_or_restart(C)
                log(step=step, dead_codes_restarted=nd)
        if step % 100 == 0:
            dt = time.time() - t0
            log(step=step, loss=loss.item(), text=lt.item(), codes_used=int((model.usage > 0).sum()) if is_ncp else 0, graph=lg.item(), ncp=float(lncp), vq=float(lvq),
                gn=float(gn), lr=opt.param_groups[0]["lr"], tok_s=ntok / dt)
            if step % 500 == 0:
                print(f"step {step} loss {loss.item():.3f} text {lt.item():.3f} graph {lg.item():.3f} "
                      f"ncp {float(lncp):.3f} {ntok/dt:,.0f} tok/s", flush=True)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            model.eval()
            fwd_eval = (lambda x: model(x)["logits"]) if is_ncp else model     # distinct name: `fwd` is the train closure
            ev = eval_nll(fwd_eval)
            model.train()
            log(step=step + 1, eval=ev)
            print(f"eval {step+1}: {ev}", flush=True)
            torch.save({"model": model.state_dict(), "arch": a.arch, "step": step + 1, "params": n_par,
                        "args": vars(a), "opt": opt.state_dict()}, os.path.join(out, "ckpt.pt"))
    save_json(os.path.join(out, "train_summary.json"), dict(arch=a.arch, params=n_par, steps=a.steps,
              tokens=ntok, wall_s=time.time() - t0, final_eval=ev))


if __name__ == "__main__":
    main()
