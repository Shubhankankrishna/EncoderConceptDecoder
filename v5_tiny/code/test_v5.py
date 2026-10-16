"""Pre-launch tests (advisor consult 3) + review_v5 §12 numeric checks.  Run: python test_v5.py"""
import math
import numpy as np
import torch

import data_v5 as dv
from model_v5 import (NCPModel, GramCore, token_nll, concept_mask, ncp_mse, random_windows, shift_to_tokens,
                      apply_plan_mode, K)

torch.manual_seed(0)
FAIL = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""))
    if not cond:
        FAIL.append(name)


def t_review_numeric():
    p = [0.81, 0.09, 0.09, 0.01]
    check("review12 length-avg mass 1.6", abs(sum(math.sqrt(x) for x in p) - 1.6) < 1e-12)
    c, alpha = 0.49, 0.1
    opt = (1 + c - math.sqrt((1 - c) ** 2 + 2 * alpha)) / 2
    check("review12 MSE+CE optimum", abs(opt - 0.4058466423577676) < 1e-12)
    z = 0.0
    for _ in range(10):
        z = 0.51 * z + 0.49 * 2.0
    check("review12 soft gate 1.9976 vs hard 0", abs(z - 1.9976191515234476) < 1e-12)
    check("review12 PQ bounds", 4 * min(32, 15) == 60 and 8 * min(32, 31) == 248 and 32 * min(128, 127) == 4064)
    check("v5 PQ bound S=8,V=32,d=128 is 128", 8 * min(128 // 8, 32 - 1) == 128)
    # E4 loop: original evaluates offsets [0,1]; corrected [0,1,2] for 2 updates
    def orig(U):
        ev, d = [], 0
        for _ in range(U):
            ev.append(d); d += 1
        return ev
    def corrected(U):
        d = 0; ev = [d]
        for _ in range(U):
            d += 1; ev.append(d)
        return ev
    check("review12 E4 corrected loop evaluates final update", orig(2) == [0, 1] and corrected(2) == [0, 1, 2])


def t_graph_mask_and_verifier():
    rng = np.random.RandomState(5)
    x, m, meta = dv.gen_instance(rng)
    nh = meta["n_hops"]
    check("graph mask count", m.sum() == (nh + 1) * 4, f"{m.sum()} vs {(nh+1)*4}")
    cm = concept_mask(torch.from_numpy(m)[None])[0]
    check("concept_mask header/walk/done", cm[29] == 0 and cm[30] == 1 and cm[29 + nh + 1] == 1 and
          (31 + nh >= len(cm) or cm[30 + nh + 1] == 0), f"nh={nh} cm[28:]={cm[28:].tolist()}")
    hdr, walk = x[:dv.WALK_START], x[dv.WALK_START:].copy()
    check("verifier accepts demo", dv.verify(hdr, walk)[0])
    w2 = walk.copy(); w2[2] = dv.NODE0 + ((w2[2] - dv.NODE0 + 1) % dv.N_NODE_TOK)
    check("verifier rejects wrong node", not dv.verify(hdr, w2)[0])
    w3 = walk.copy(); w3[4 * nh] = dv.PAD
    check("verifier rejects missing done", not dv.verify(hdr, w3)[0])
    w4 = walk.copy(); w4[4 * (nh - 1):4 * nh] = [dv.DONE, dv.SEP, dv.SEP, dv.SEP]
    check("verifier rejects early done", not dv.verify(hdr, w4)[0])


def t_shift_leak():
    model = NCPModel(dv.VOCAB).eval()
    rng = np.random.RandomState(7)
    x, m, meta = dv.gen_instance(rng)
    x = torch.from_numpy(x)[None]
    j = 33                                   # poison chunk j+1 onward (inside the walk)
    x2 = x.clone(); x2[:, 4 * (j + 1):] = torch.randint(40, 2000, (1, x.shape[1] - 4 * (j + 1)))
    with torch.no_grad():
        o1, o2 = model(x), model(x2)
    check("shift: chat[:, :j+1] unchanged", torch.allclose(o1["chat"][:, :j + 1], o2["chat"][:, :j + 1], atol=1e-6))
    check("shift: A[:, :j+1] unchanged", torch.allclose(o1["A"][:, :j + 1], o2["A"][:, :j + 1], atol=1e-6))
    check("shift: logits[:, :4j+4] unchanged",
          torch.allclose(o1["logits"][:, :4 * j + 4], o2["logits"][:, :4 * j + 4], atol=1e-5))
    # load-bearing alignment: position 4j+3 (predicts first token of chunk j+1) reads chat_j
    T = x.shape[1]
    v = torch.zeros(1, T // K, 4); v[0, j] = 1.0
    b = shift_to_tokens(v, T)
    check("shift: chat_j lands exactly on positions 4j+3..4j+6",
          b[0, :, 0].nonzero().flatten().tolist() == [4 * j + 3, 4 * j + 4, 4 * j + 5, 4 * j + 6])
    # changing chat_j must change logits at 4j+3
    with torch.no_grad():
        H = model.encode(x); C = model.pool(H); u = model.concept(C); _, _, chat = model.pq(u); A = model.port(u)
        chat2 = chat.clone(); chat2[:, j] += 3.0
        l1 = model.decode(H, chat, A); l2 = model.decode(H, chat2, A)
    check("shift: perturbing chat_j changes logits at 4j+3", (l1[:, 4 * j + 3] - l2[:, 4 * j + 3]).abs().max() > 1e-4)
    check("shift: perturbing chat_j leaves logits <4j+3", torch.allclose(l1[:, :4 * j + 3], l2[:, :4 * j + 3], atol=1e-6))


def t_gram_causal():
    for var in ("det", "adapter", "z", "gram"):
        core = GramCore(var).eval()
        B, M = 2, 48
        C = torch.randn(B, M, 128)
        Z = torch.randn(B, M, 8)
        noise = torch.randn(B, 2, M, 128)
        y = torch.randn(B, M, 128)
        j = 20
        C2, Z2, y2 = C.clone(), Z.clone(), y.clone()
        C2[:, j + 1:] += 5; Z2[:, j + 1:] += 5
        with torch.no_grad():
            u1, A1, _ = core(C, Z if core.use_z else None, eps_mode="prior", noise=noise)
            u2, A2, _ = core(C2, Z2 if core.use_z else None, eps_mode="prior", noise=noise)
        check(f"gram[{var}] causal over concepts", torch.allclose(u1[:, :j + 1], u2[:, :j + 1], atol=1e-5) and
              torch.allclose(A1[:, :j + 1], A2[:, :j + 1], atol=1e-5))
        if core.use_z:
            with torch.no_grad():
                u3, _, _ = core(C, Z * 0 + torch.randn(B, M, 8), eps_mode="prior", noise=noise)
            check(f"gram[{var}] Z changes output at init", (u3 - u1).abs().max() > 1e-3)


def t_windows_and_swap():
    g = torch.Generator().manual_seed(0)
    win, st = random_windows(3, 48, g)
    lens = [np.diff(np.flatnonzero(np.r_[st[b].numpy(), True])) for b in range(3)]
    check("windows lengths in 1..4", all(((l >= 1) & (l <= 4)).all() for l in lens))
    perm = torch.tensor([1, 0, 3, 2])
    c = torch.arange(4.).view(4, 1, 1).expand(4, 2, 3)
    cs, _ = apply_plan_mode(c, c, "swap", None, perm)
    check("swap permutes rows", (cs[:, 0, 0] != c[:, 0, 0]).all())


def t_compile_equiv():
    from common_v5 import DEV, Mixer, domain_mean
    model = NCPModel(dv.VOCAB).to(DEV)
    mix = Mixer(seed=3)
    x, m, _ = mix.batch()
    def per_tok(x, m):
        o = model(x)
        return token_nll(o["logits"], x, m), (o["chat"][:, :-1] - o["C"][:, 1:].detach()).pow(2).mean(-1)
    def loss(fn):
        nll, err = fn(x, m)
        cm = concept_mask(m)
        return domain_mean(nll, m)[0] + (err * cm).sum() / cm.sum()
    cf = torch.compile(per_tok)
    a = loss(per_tok); model.zero_grad(); a.backward()
    ga = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
    b = loss(cf); model.zero_grad(); b.backward()
    gb = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
    check("eager vs compiled loss (eager reductions)", abs(a.item() - b.item()) < 1e-4, f"{a.item():.6f} vs {b.item():.6f}")
    rel = ((ga - gb).norm() / ga.norm()).item()
    check("eager vs compiled gradients", rel < 1e-3, f"rel diff {rel:.2e}")


def t_core_compile_equiv():
    from common_v5 import DEV
    core = GramCore("gram").to(DEV)
    for mlp in (core.post_e,):
        torch.nn.init.normal_(mlp.b.weight, std=0.02)
    B, M = 8, 48
    C = torch.randn(B, M, 128, device=DEV); Z = torch.randn(B, M, 8, device=DEV)
    y = torch.randn(B, M, 128, device=DEV); nz = torch.randn(B, 2, M, 128, device=DEV)
    a = core(C, Z, eps_mode="post", y=y, noise=nz)
    b = torch.compile(core)(C, Z, eps_mode="post", y=y, noise=nz)
    ok = all((u - v).abs().max().item() < 1e-4 * max(1.0, u.abs().max().item()) for u, v in zip(a, b))
    check("GramCore eager vs compiled (u, A, kl)", ok, str([(u - v).abs().max().item() for u, v in zip(a, b)]))


def t_phase_d_zseq():
    import phase_d as pdm
    meta = pdm.Meta()
    with torch.no_grad():
        torch.nn.init.normal_(meta.switch.b.weight, std=1.0)
    B, M = 4, 48
    e = torch.randn(B, M, 128); mu = torch.randn(B, M, 8); ls = torch.full((B, M, 8), -1.0)
    with torch.no_grad():
        Zh, bh = pdm.z_sequence(meta, e, mu, ls, hard=True, sample=False)
        Zs, bs = pdm.z_sequence(meta, e, mu, ls, hard=False, sample=False)
    W0 = pdm.W0
    check("phaseD forced switch at first walk boundary", bool((bh[:, W0] == 1).all()) and torch.allclose(Zh[:, W0], mu[:, W0]))
    hard_ok = bool(((bh[:, W0:] == 0) | (bh[:, W0:] == 1)).all())
    persist = all(torch.allclose(Zh[b, m], Zh[b, m - 1]) for b in range(B) for m in range(W0 + 1, M) if bh[b, m] == 0)
    switch = all(torch.allclose(Zh[b, m], mu[b, m]) for b in range(B) for m in range(W0 + 1, M) if bh[b, m] == 1)
    check("phaseD hardened Z persists between switches and resets at switches", hard_ok and persist and switch)
    check("phaseD soft betas strictly inside (0,1)", bool(((bs[:, W0 + 1:] > 0) & (bs[:, W0 + 1:] < 1)).all()))


if __name__ == "__main__":
    t_phase_d_zseq()
    t_core_compile_equiv()
    t_review_numeric()
    t_graph_mask_and_verifier()
    t_shift_leak()
    t_gram_causal()
    t_windows_and_swap()
    t_compile_equiv()
    print("\nALL PASS" if not FAIL else f"\nFAILED: {FAIL}")
