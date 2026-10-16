"""Shared harness pieces: mixed batches, Phase A evaluation, GraphNav free generation."""
import json
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

import data_v5 as dv
from model_v5 import (NCPModel, TransformerLM, token_nll, concept_mask, ncp_mse, apply_plan_mode, shift_to_tokens,
                      K, D)

DEV = os.environ.get("V5_DEV") or ("mps" if torch.backends.mps.is_available() else "cpu")
RUNS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
B_TEXT, B_GRAPH = 32, 32          # pre-registered mix: 50% of rows each


def maybe_compile(f):
    """torch.compile unless V5_NOCOMPILE is set (compiled CPU training fails here with an OpenMP error)."""
    return f if os.environ.get("V5_NOCOMPILE") else torch.compile(f)


class Mixer:
    def __init__(self, seed, split="train"):
        self.rng = np.random.RandomState(seed)
        self.text = dv.TextStream(split)

    def batch(self, dev=DEV):
        xt, mt = self.text.batch(self.rng, B_TEXT)
        mt[:, -1] = 0.0                                  # last position has no target
        xg, mg, metas = dv.graph_batch(self.rng, B_GRAPH)
        x = torch.from_numpy(np.concatenate([xt, xg])).to(dev)
        m = torch.from_numpy(np.concatenate([mt, mg])).to(dev)
        return x, m, metas


def domain_mean(per_tok, m):
    """Pre-registered reduction: per-token mean within each domain (rows 0..B_TEXT-1 text, rest graph), 0.5/0.5."""
    rows_tok = per_tok.sum(1)
    rows_m = m[:, :-1].sum(1)
    lt = rows_tok[:B_TEXT].sum() / rows_m[:B_TEXT].sum().clamp(min=1)
    lg = rows_tok[B_TEXT:].sum() / rows_m[B_TEXT:].sum().clamp(min=1)
    return 0.5 * lt + 0.5 * lg, lt, lg


def domain_mean_rows(per_row_vals, w):
    """Same 0.5/0.5 domain reduction for per-concept quantities. per_row_vals,w [B,M']"""
    a = (per_row_vals[:B_TEXT] * w[:B_TEXT]).sum() / w[:B_TEXT].sum().clamp(min=1)
    b = (per_row_vals[B_TEXT:] * w[B_TEXT:]).sum() / w[B_TEXT:].sum().clamp(min=1)
    return 0.5 * a + 0.5 * b


def cosine_lr(step, total, base, warm=300, floor=0.1):
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def make_opt(params, lr, wd=0.1):
    params = [p for p in params if p.requires_grad]
    dec = [p for p in params if p.dim() >= 2]
    nod = [p for p in params if p.dim() < 2]
    return torch.optim.AdamW([{"params": dec, "weight_decay": wd}, {"params": nod, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95))


# ----------------------------------------------------------------------------- fixed eval sets
_EVAL = {}


def eval_sets(split="val"):
    if split not in _EVAL:
        ts = dv.TextStream(split)
        xt = ts.fixed(384, seed=11 if split == "val" else 12)
        xg, mg, metas = dv.fixed_graph_set(512, seed=21 if split == "val" else 22)
        if os.environ.get("V5_QUICK"):          # smoke tests only
            xt, xg, mg, metas = xt[:32], xg[:32], mg[:32], metas[:32]
        _EVAL[split] = (xt, xg, mg, metas)
    return _EVAL[split]


def text_mask(xt):
    m = np.ones_like(xt, dtype=np.float32); m[:, -1] = 0
    return m


@torch.no_grad()
def eval_nll(fwd, split="val", bs=64, dev=DEV):
    """fwd(x) -> logits.  Returns text NLL/token, graph walk NLL/token, colour-token NLL, node-token NLL."""
    xt, xg, mg, metas = eval_sets(split)
    tot = {"text": [0.0, 0.0], "graph": [0.0, 0.0], "colour": [0.0, 0.0], "node": [0.0, 0.0], "done": [0.0, 0.0]}
    for i in range(0, len(xt), bs):
        x = torch.from_numpy(xt[i:i + bs]).to(dev); m = torch.from_numpy(text_mask(xt[i:i + bs])).to(dev)
        nl = token_nll(fwd(x), x, m)
        tot["text"][0] += nl.sum().item(); tot["text"][1] += m[:, :-1].sum().item()
    for i in range(0, len(xg), bs):
        x = torch.from_numpy(xg[i:i + bs]).to(dev); m = torch.from_numpy(mg[i:i + bs]).to(dev)
        nl = token_nll(fwd(x), x, m)
        tgt = x[:, 1:]
        tot["graph"][0] += nl.sum().item(); tot["graph"][1] += m[:, :-1].sum().item()
        iscol = ((tgt >= dv.COL0) & (tgt < dv.COL0 + dv.N_COL)).float() * m[:, :-1]
        isnode = ((tgt >= dv.NODE0) & (tgt < dv.NODE0 + dv.N_NODE_TOK)).float() * m[:, :-1]
        isdone = (tgt == dv.DONE).float() * m[:, :-1]
        off = (torch.arange(1, x.shape[1], device=dev) % 4).expand_as(tgt)
        offs = [(f"off{o}", (off == o).float() * m[:, :-1]) for o in range(4)]
        for k, w in [("colour", iscol), ("node", isnode), ("done", isdone)] + offs:
            tot.setdefault(k, [0.0, 0.0])
            tot[k][0] += (nl * w).sum().item(); tot[k][1] += w.sum().item()
    return {k: v[0] / max(v[1], 1) for k, v in tot.items()}


# ----------------------------------------------------------------------------- generation
@torch.no_grad()
def generate_walks(step_logits, xg, temperature=0.0, seed=0, dev=DEV, bs=128):
    """Free generation from the GraphNav header.  step_logits(x [B,T], pos) -> logits at position pos [B,V].
    Fixed T=192 inputs (pads after the cursor are invisible under the causal mask).  Returns success flags etc."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    res = []
    for i in range(0, len(xg), bs):
        x = torch.from_numpy(xg[i:i + bs].copy())
        x[:, dv.WALK_START:] = dv.PAD
        x = x.to(dev)
        n = x.shape[0]
        for pos in range(dv.WALK_START - 1, dv.T - 1):
            lg = step_logits(x, pos).float()
            if temperature <= 0:
                nxt = lg.argmax(-1)
            else:
                p = (lg / temperature).softmax(-1).cpu()
                nxt = torch.multinomial(p, 1, generator=gen).squeeze(1).to(dev)
            x[:, pos + 1] = nxt
        xc = x.cpu().numpy()
        for r in range(n):
            res.append(dv.verify(xc[r, :dv.WALK_START], xc[r, dv.WALK_START:]))
    succ = np.array([r[0] for r in res], dtype=float)
    fmt = np.array([r[3] for r in res], dtype=float)
    reach = np.array([r[2] for r in res], dtype=float)
    return dict(success=float(succ.mean()), format_ok=float(fmt.mean()), reached=float(reach.mean()),
                success_flags=succ.tolist())


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


class Logger:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, **kw):
        kw["time"] = time.strftime("%H:%M:%S")
        self.f.write(json.dumps(kw) + "\n"); self.f.flush()
