"""Exact CPU tests for HRM Thinker v2 (float64, small config).  python test_v2.py
T1 capacity MoE dispatch == masked dense MoE when nothing is dropped
T2 per-window backward (c_hat / tok leaves) == one-graph backward with the carry detached between windows
T3 identity at init: pass 1 leaves S_0 unchanged (W_o, W_2 zero), readout finite
T4 causality + document isolation through encoder -> D6 core -> decoder (eval mode)
T5 parameter table of the full model == budget (18,833,529 = 18,833,513 + 16 per-head temperatures)
T6 NextLat never updates the head E; gradients reach psi, decoder, core (final window) and encoder
T7 windows(K) layout for K = 8..12
"""
import copy, math, sys
import torch
import torch.nn.functional as F
from model import build_model, forward, nextlat_losses, param_table
from concept_core_d6 import moe_dispatch, ExpertPool
from data import target_mask
import train as T


def cfg_args(**kw):
    class A: pass
    a = A()
    a.lam_lb, a.lam_z, a.lam_cons, a.lam_h, a.lam_kl = 0.01, 0.001, 0.1, 1.0, 1.0
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def randomize(m, scale=0.1):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "engram.table" not in n and "vocab.e" not in n:
                p.add_(torch.randn_like(p) * scale)


def batch(m, B=2, T_=128, seed=0, two_docs=False):
    from hrm_encoder import engram_indices
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, 8192, (B, T_), generator=g)
    doc = torch.zeros(B, T_, dtype=torch.long)
    if two_docs:
        doc[:, T_ // 2:] = 1
    return ids, doc, engram_indices(ids, doc, m.ecfg)


def t1():
    torch.manual_seed(0)
    pool = ExpertPool(16, 24, 6).double()
    with torch.no_grad():
        pool.w2.normal_()
    x = torch.randn(50, 16, dtype=torch.float64); lg = torch.randn(50, 6, dtype=torch.float64)
    y1, a1 = moe_dispatch(x, lg, pool, 2, None)
    y2, a2 = moe_dispatch(x, lg, pool, 2, None, dense=True)
    assert torch.allclose(y1, y2, atol=1e-12), (y1 - y2).abs().max()
    y3, a3 = moe_dispatch(x, lg, pool, 2, 0.5)
    assert a3["dropped"] > 0 and torch.isfinite(y3).all()
    print(f"PASS T1 capacity dispatch == dense MoE (max diff {(y1 - y2).abs().max():.1e}); cf 0.5 drops {a3['dropped']:.2f} and stays finite")


def reference_step(m, ids, doc, ei, K, a):
    """One graph: same arithmetic as train.micro_step, carry detached between windows, single backward."""
    core = m.core
    tok, c_hat, _ = m.enc(ids, doc, ei)
    cdoc = doc[:, ::4]; mask = core.mask(cdoc)
    vq, targets = core.vocab.vq_loss(c_hat)
    X, S = core.init_state(c_hat)
    wins = T.windows(K); total = 0.0
    for wi, (p0, p1) in enumerate(wins):
        last = wi == len(wins) - 1
        if wi:
            S = S.detach()
        lw, us = 0.0, []
        for r in range(p0, p1 + 1):
            S, auxs = core.one_pass(X, S, mask, first=(r == 1))
            u = core.readout(S); lc, cp = core.head(u)
            ce = core.ncp_ce(lc, targets, cdoc)
            lb = torch.stack([x["lb"] for x in auxs]).mean(); z = torch.stack([x["z"] for x in auxs]).mean()
            lw = lw + (0.1 * ce + a.lam_lb * lb + a.lam_z * z) / K
            if last:
                us.append(u)
        total = total + lw
        if last:
            uK = us[-1]
            cons = torch.stack([(uK.detach() - ur).pow(2).mean() for ur in us[:-1]]).mean()
            logits, daux = m.dec(tok, uK, cp, doc, m.enc.embed.weight)
            tgt = ids[:, 1:].masked_fill(~target_mask(ids, doc), -100)
            ntp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            lh, lkl = nextlat_losses(m, daux["h"], logits, ids, doc)
            total = total + a.lam_cons * cons + ntp + vq + a.lam_h * lh + a.lam_kl * lkl
    total.backward()


def t2():
    import contextlib
    torch.set_default_dtype(torch.float64); torch.manual_seed(0)
    m, _, _ = build_model(small=True); randomize(m); m.train()
    for l in m.core.layers:                                            # no drops: identical routing in both paths
        l.cf = None
    ids, doc, ei = batch(m, two_docs=True)
    m2 = copy.deepcopy(m)
    a = cfg_args()
    for K in (8, 11):
        m.zero_grad(); m2.zero_grad()
        T.micro_step(m, ids, doc, ei, K, a, 1, contextlib.nullcontext)
        reference_step(m2, ids, doc, ei, K, a)
        worst, name = 0.0, ""
        for (n1, p1), (n2, p2) in zip(m.named_parameters(), m2.named_parameters()):
            g1 = p1.grad if p1.grad is not None else torch.zeros_like(p1)
            g2 = p2.grad if p2.grad is not None else torch.zeros_like(p2)
            d = (g1 - g2).abs().max().item() / (g2.abs().max().item() + 1e-30)
            if d > worst:
                worst, name = d, n1
        assert worst < 1e-9, (K, worst, name)
        print(f"PASS T2 K={K} windows {T.windows(K)}: per-window backward == one-graph backward (max rel diff {worst:.1e})")
    torch.set_default_dtype(torch.float32)


def t3():
    torch.manual_seed(0)
    m, _, _ = build_model(small=True); m.eval()
    ids, doc, ei = batch(m)
    with torch.no_grad():
        _, c_hat, _ = m.enc(ids, doc, ei)
        X, S0 = m.core.init_state(c_hat)
        S1, _ = m.core.one_pass(X, S0, m.core.mask(doc[:, ::4]), first=True)
    assert torch.equal(S0, S1), (S0 - S1).abs().max()
    print("PASS T3 identity at init: pass 1 leaves S_0 = W_s c_hat unchanged (W_o and expert W_2 zero-init)")


def t4():
    torch.set_default_dtype(torch.float64); torch.manual_seed(0)
    m, _, _ = build_model(small=True); randomize(m, 0.2); m.eval()
    from hrm_encoder import engram_indices
    ids, doc, ei = batch(m, B=2, T_=256)
    p = 101
    ids2 = ids.clone(); ids2[0, p] = (ids[0, p] + 7) % 8192
    with torch.no_grad():
        a, _ = forward(m, ids, doc, ei, 6)
        b, _ = forward(m, ids2, doc, engram_indices(ids2, doc, m.ecfg), 6)
    ch = torch.nonzero((a[0] - b[0]).abs().amax(-1) != 0).flatten()
    assert ch.min().item() == p and torch.equal(a[1], b[1]), "causality / row isolation"
    doc2 = torch.zeros_like(doc); doc2[:, 128:] = 1
    ids3 = ids.clone(); ids3[:, 120] = (ids[:, 120] + 5) % 8192
    with torch.no_grad():
        c1, _ = forward(m, ids, doc2, engram_indices(ids, doc2, m.ecfg), 6)
        c2, _ = forward(m, ids3, doc2, engram_indices(ids3, doc2, m.ecfg), 6)
    assert torch.equal(c1[:, 128:], c2[:, 128:]), "document isolation"
    print(f"PASS T4 causality (first change at the edited position {p}; other row untouched) and document isolation through the experts (eval: no drops)")
    torch.set_default_dtype(torch.float32)


def t5():
    m, _, _ = build_model(engram_rows=11)
    t = param_table(m)
    assert t["compute_params"] == 18833529, t["compute_params"]
    print(f"PASS T5 compute params {t['compute_params']:,} (budget 18,833,513 + 16 per-head temperatures); NextLat psi {t['nextlat_psi_training_only']:,} (training only)")


def t6():
    torch.manual_seed(0)
    m, _, _ = build_model(small=True); randomize(m); m.train()
    ids, doc, ei = batch(m)
    logits, x = forward(m, ids, doc, ei, 6)
    lh, lkl = nextlat_losses(m, x["h"], logits, ids, doc)
    E = m.enc.embed.weight
    gE, gpsi, gdec, gcore, genc = torch.autograd.grad(lh + lkl, [E, m.nl.f[0].weight, m.dec.layers[0].q.weight,
                                                                  m.core.layers[0].attn.q.weight, m.enc.layers[3].attn.q.weight], allow_unused=True)
    assert gpsi.abs().sum() > 0 and gdec.abs().sum() > 0 and gcore.abs().sum() > 0 and genc.abs().sum() > 0
    h = x["h"].detach().requires_grad_()                              # head path alone: E must get no gradient
    lh2, lkl2 = nextlat_losses(m, h, logits.detach(), ids, doc)
    gE2 = torch.autograd.grad(lh2 + lkl2, [E], allow_unused=True)[0]
    assert gE2 is None or gE2.abs().sum() == 0
    print("PASS T6 NextLat gradients reach psi, decoder, core and encoder; the head E enters detached (E.detach() in nextlat_losses)")


def t7():
    exp = {8: [(1, 4), (5, 8)], 9: [(1, 4), (5, 9)], 10: [(1, 4), (5, 10)], 11: [(1, 4), (5, 11)], 12: [(1, 4), (5, 8), (9, 12)]}
    for K, w in exp.items():
        assert T.windows(K) == w, (K, T.windows(K))
    print("PASS T7 windows: K 8..12 -> [4,4] [4,5] [4,6] [4,7] [4,4,4]; eval R 6 lies in window 2 for every K")


if __name__ == "__main__":
    for f in (t7, t1, t3, t5, t6, t4, t2):
        f()
    print("ALL PASS")
