"""Parity tests for the v3 speed paths (CPU float64; the MPS fp32 check is in mps_check.py). Exit 1 on any failure.

  nice -n 19 python test_speed.py
"""
import copy, sys
import torch, torch.nn.functional as F
import model as M
import speed_v3 as SV
from model import build_model, forward, rms
from hrm_encoder import engram_indices

RES = {}


def check(name, ok, info=""):
    RES[name] = bool(ok)
    print(f"{name:58s} {'PASS' if ok else 'FAIL'} {info}", flush=True)


def loss_and_grads(m, ids, doc, ei, short="2b"):
    m.zero_grad(set_to_none=True)
    tok, C, _ = m.enc(ids, doc, ei)
    S3, _, Ss = m.core.run(C, doc[:, ::4], "3", short=short)
    u = m.core.out_norm(S3); lc, cp = m.core.head(u)
    logits, _ = m.dec(tok, u, cp, doc, m.enc.embed.weight)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)) + \
        ((rms(Ss) - rms(S3).detach()) ** 2).mean()
    loss.backward()
    return logits.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}


def main():
    torch.manual_seed(0); torch.set_default_dtype(torch.float64)
    m, ecfg, _ = build_model("test")
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "engram.table" not in n:
                p.add_(torch.randn_like(p) * 0.1)
    B, T = 2, 128
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(2, 64, (B, T), generator=g); doc = torch.zeros(B, T, dtype=torch.long)
    doc[0, 36:] = 1; doc[0, 92:] = 2; doc[1, 64:] = 1; ids[1, 60:64] = 0          # doc boundaries inside blocks
    ei = engram_indices(ids, doc, ecfg)
    SV.enable_block_attention(0)
    l0, g0 = loss_and_grads(m, ids, doc, ei)
    for blk in (32, 16):
        SV.enable_block_attention(blk)
        l1, g1 = loss_and_grads(m, ids, doc, ei)
        dl = (l1 - l0).abs().max().item(); dg = max((g1[k] - g0[k]).abs().max().item() for k in g0)
        check(f"S1_block_attention_{blk}_forward_and_grad_parity", dl < 1e-12 and dg < 1e-12 and g1.keys() == g0.keys(),
              f"dlogit={dl:.1e} dgrad={dg:.1e}")
    SV.enable_block_attention(0)
    with SV.step_memo():
        l2, g2 = loss_and_grads(m, ids, doc, ei)
    ok = torch.equal(l2, l0) and all(torch.equal(g2[k], g0[k]) for k in g0)
    check("S2_step_memo_bit_identical", ok)
    SV.enable_block_attention(32)
    with SV.step_memo():
        l3, g3 = loss_and_grads(m, ids, doc, ei)
    dl = (l3 - l0).abs().max().item(); dg = max((g3[k] - g0[k]).abs().max().item() for k in g0)
    check("S3_block_attention_plus_step_memo_parity", dl < 1e-12 and dg < 1e-12, f"{dl:.1e} {dg:.1e}")
    # eval forward with interventions under block attention
    with torch.no_grad():
        a, _ = forward(m, ids, doc, ei, R="3", core_off=True)
        SV.enable_block_attention(0)
        b, _ = forward(m, ids, doc, ei, R="3", core_off=True)
    check("S4_block_attention_eval_core_off_parity", (a - b).abs().max().item() < 1e-12)
    # memo must not leak across steps: a different doc at a reused address must not see stale masks
    SV.enable_block_attention(32)
    doc2 = doc.clone(); doc2[0, 20:] = 3
    with SV.step_memo():
        x1, _ = forward(m, ids, doc, ei)
    with SV.step_memo():
        x2, _ = forward(m, ids, doc2, engram_indices(ids, doc2, ecfg))
    SV.enable_block_attention(0)
    with torch.no_grad():
        y2, _ = forward(m, ids, doc2, engram_indices(ids, doc2, ecfg))
    check("S5_no_stale_masks_across_steps", (x2 - y2).abs().max().item() < 1e-12)
    torch.set_default_dtype(torch.float32)
    bad = [k for k, v in RES.items() if not v]
    print(f"\n{len(RES) - len(bad)}/{len(RES)} PASS" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
