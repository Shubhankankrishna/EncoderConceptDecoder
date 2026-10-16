"""torch.compile parity on MPS (V3-006): one training micro-step eager vs with the core layers (and optionally the
encoder / decoder layers) compiled, same weights, same batch (B 2, T 1024, block-local attention 64, memo off).
Pass: max |Δ logits| < 1e-3, per-parameter gradient relative error < 1e-2, and the core.table / core.shared.w2
gradients nonzero in both and within tolerance. Exit 1 on failure.   python compile_check.py [--all]
"""
import copy, sys, time
import torch, torch.nn.functional as F
import speed_v3 as SV
from model import build_model, chunk_masks, rms
from hrm_encoder import engram_indices
from data import target_mask

ALL = "--all" in sys.argv
dev = torch.device("mps")
SV.enable_block_attention(64)
torch.manual_seed(0)
m, ecfg, _ = build_model("full", engram_rows=1021)
with torch.no_grad():
    for n, p in m.named_parameters():
        if "engram.table" not in n:
            p.add_(torch.randn_like(p) * 0.02)
m = m.to(dev).train()
B, T = 2, 1024
ids = torch.randint(2, 8192, (B, T)); doc = torch.zeros(B, T, dtype=torch.long); doc[0, 400:] = 1; doc[1, 700:] = 1
ei = engram_indices(ids, doc, ecfg)
ids, doc, ei = ids.to(dev), doc.to(dev), ei.to(dev)


def step(mm, short):
    mm.zero_grad(set_to_none=True)
    tok, c_hat, _ = mm.enc(ids, doc, ei)
    ne, co, po = chunk_masks(ids, doc)
    vq, n = mm.core.vq_sum(c_hat, co, count_hits=False)
    S3, _, Ss = mm.core.run(c_hat, doc[:, ::4], "3", short=short)
    u = mm.core.out_norm(S3); lc, cp = mm.core.head(u)
    logits, _ = mm.dec(tok, u, cp, doc, mm.enc.embed.weight)
    tgt = ids[:, 1:].masked_fill(~target_mask(ids, doc), -100)
    lcs, _ = mm.core.head(mm.core.out_norm(Ss), pred=False)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 8192), tgt.reshape(-1), ignore_index=-100) + 0.1 * mm.core.ncp_sum(lc, n, po) / po.sum() \
        + 0.05 * mm.core.ncp_sum(lcs, n, po) / po.sum() + 0.1 * ((rms(Ss) - rms(S3).detach()).pow(2).mean(-1) * ne).sum() / ne.sum() + vq / co.sum()
    loss.backward()
    torch.mps.synchronize()
    return logits.detach().float().cpu(), {k: p.grad.detach().float().cpu() for k, p in mm.named_parameters() if p.grad is not None}


mc = copy.deepcopy(m)
layers = list(mc.core.layers) + (list(mc.enc.layers) + list(mc.dec.layers) if ALL else [])
for l in layers:
    SV.compile_layer(l, lambda l_: True)
ok = True
for short in ("1", "2a", "2b"):
    le, ge = step(m, short)
    t0 = time.time(); lc_, gc = step(mc, short); tcomp = time.time() - t0
    dl = (le - lc_).abs().max().item()
    rel = {k: ((ge[k] - gc[k]).norm() / ge[k].norm().clamp_min(1e-12)).item() for k in ge if ge[k].norm() > 0}
    worst = max(rel, key=rel.get)
    crit = all(ge[k].abs().sum() > 0 and gc[k].abs().sum() > 0 and rel[k] < 1e-2 for k in ("core.table", "core.shared.w2.weight"))
    good = dl < 1e-3 and rel[worst] < 1e-2 and crit and ge.keys() == gc.keys()
    ok &= good
    print(f"shortcut {short}: max|Δ logits| {dl:.2e}, worst grad rel {rel[worst]:.2e} ({worst}), table rel {rel['core.table']:.2e}, "
          f"shared.w2 rel {rel['core.shared.w2.weight']:.2e}, compiled step {tcomp:.1f}s -> {'PASS' if good else 'FAIL'}", flush=True)
print(f"COMPILE PARITY ({'core+enc+dec' if ALL else 'core'})", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
