"""Acceptance tests for model_v4 (draft_v4 §3.8, §4.9, §5.5, §6.2), float64 on CPU, at k = 4, 6 and 8.

Every structural test runs on a RANDOMIZED model: at init the pooler is uniform, the FFN branches and decoder Engram
reads output exactly zero, so an un-randomized test would be vacuous for every new branch.
Run:  python test_v4.py        (prints one PASS line per test; any failure raises)
"""
import math
import sys
import torch
import torch.nn.functional as F

import model_v4 as M4
from hrm_encoder import engram_indices
from hrm_decoder import plan_index

torch.set_num_threads(2)
T, B = 240, 2
RESULTS = []


def ok(name, detail=""):
    RESULTS.append((name, detail))
    print(f"PASS {name}" + (f" | {detail}" if detail else ""), flush=True)


def randomize(m, scale=0.2, skip=()):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if not any(s in n for s in skip):
                p.add_(torch.randn_like(p) * scale)
    return m


def mk(k, rand=True, **over):
    torch.manual_seed(1234 + k)
    m = M4.build_model("test", k=k, **over).double()
    if rand:
        randomize(m)
    return m.eval()


def batch(V=64, doc_split=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, V, (B, T), generator=g)
    doc = torch.zeros(B, T, dtype=torch.long)
    if doc_split is not None:
        doc[:, doc_split:] = 1
    return ids, doc


def logits_of(m, ids, doc, **kw):
    with torch.no_grad():
        lg, aux = M4.forward(m, ids, doc, engram_indices(ids, doc, m.ecfg), **kw)
    return lg, aux


def changed(a, b):
    """positions (row 0) whose last-dim values differ at all"""
    return torch.nonzero((a - b).abs().amax(-1)[0] != 0).flatten()


# ----------------------------------------------------------------------------- 1. end-to-end causality
def test_causality():
    for k in (4, 6, 8):
        m = mk(k)
        ids, doc = batch()
        for p in (100, 101, 103, 150):
            ids2 = ids.clone(); ids2[:, p] = (ids[:, p] + 7) % 62 + 2
            a, _ = logits_of(m, ids, doc); b, _ = logits_of(m, ids2, doc)
            ch = changed(a, b)
            assert torch.equal(a[:, :p], b[:, :p]), (k, p, "future token changed a past logit")
            assert ch.min().item() == p, (k, p, ch.min())
            assert ch.max().item() > p + 70, (k, p, "the concept route should carry the change beyond reach 70")
    ok("1 end-to-end causality, k = 4/6/8", "perturbing token p never changes a logit at t < p; it reaches beyond 70 through the core")


# ----------------------------------------------------------------------------- 2. document isolation
def test_doc_isolation():
    for k in (4, 6, 8):
        m = mk(k)
        ids, doc = batch(doc_split=120)
        for p in (20, 100, 119):
            ids2 = ids.clone(); ids2[:, p] = (ids[:, p] + 5) % 62 + 2
            a, _ = logits_of(m, ids, doc); b, _ = logits_of(m, ids2, doc)
            assert torch.equal(a[:, 120:], b[:, 120:]), (k, p)
    ok("2 document isolation, k = 4/6/8", "editing document 0 leaves every logit of document 1 bit-identical")


# ----------------------------------------------------------------------------- 3. reach(L) = 63 with the F loop
def test_reach_L():
    for loops in (1, 2):
        m = mk(6, enc_loop=loops)
        ids, doc = batch()
        p = 60
        ids2 = ids.clone(); ids2[:, p] = (ids[:, p] + 3) % 62 + 2
        with torch.no_grad():
            _, _, _, a = m.enc(ids, doc, engram_indices(ids, doc, m.ecfg))
            _, _, _, b = m.enc(ids2, doc, engram_indices(ids2, doc, m.ecfg))
        ch = changed(a["L"], b["L"])
        assert ch.min().item() == p and ch.max().item() - p == 63, (loops, ch.min(), ch.max() - p)
        chG = changed(a["G"], b["G"])
        assert chG.max().item() == T - 1
    ok("3 reach(L) = 63 with enc_loop = 1 and 2", "looping the F layers leaves the decoder-visible stream's reach unchanged; G reaches the row end")


# ----------------------------------------------------------------------------- 4. pooler init = masked mean
def test_pool_init():
    torch.manual_seed(0)
    for k in (4, 6, 8):
        P = M4.PoolAttnFFN(64, k, 4, 96).double()
        L = M4.PoolLinear(64, k).double()
        m = torch.randn(B, T, 64, dtype=torch.float64)
        real = torch.ones(B, T, dtype=torch.bool)
        real[:, T - k + 2:] = False                       # partial last chunk (2 real tokens)
        real[0, 5 * k + 1: 6 * k] = False                 # another partial chunk (1 real token)
        c, a = P(m, real)
        r = real.view(B, T // k, k, 1).double()
        mean = (m.view(B, T // k, k, 64) * r).sum(2) / r.sum(2)
        assert torch.allclose(c, mean, atol=1e-12, rtol=0), (k, (c - mean).abs().max())
        cl, _ = L(m, real)
        assert not torch.allclose(cl[:, -1], mean[:, -1]), "v3's linear pool averages pads in; it must differ here"
        assert torch.allclose(cl[:, 1], mean[:, 1], atol=1e-12)
        G = M4.PoolGConv(64, k, 4, 96).double()
        cg, _ = G(m, real)
        unit = lambda x: x / x.norm(dim=-1, keepdim=True)
        assert torch.allclose(unit(cg), unit(mean), atol=1e-12, rtol=0), (k, "gconv direction != masked mean")
    ok("4 pooler init = masked mean of the real tokens, k = 4/6/8 (gated conv: same direction, so the same c_hat)",
       "exact to 1e-12; v3's linear pool equals it on full chunks and differs on partial ones (it averages pads in)")


# ----------------------------------------------------------------------------- 5. alive at step 1 (and step 2)
def test_alive():
    torch.manual_seed(5)
    m = M4.build_model("test", k=6).double().train()
    ids, doc = batch()
    ei = engram_indices(ids, doc, m.ecfg)
    with torch.no_grad():
        _, c_hat, _, _ = m.enc(ids, doc, ei)
        _, comp, _ = M4.chunk_masks(ids, doc, 6)
        m.core.init_codebooks(c_hat[comp])
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    loss, _, _ = M4.loss_terms(m, ids, doc, ei)
    loss.backward()
    g = {n: (p.grad.abs().sum().item() if p.grad is not None else 0.0) for n, p in m.named_parameters()}
    alive1 = ["enc.shared.w2.weight", "enc.pooler.q", "enc.pooler.b", "enc.pooler.p", "enc.pooler.ffn.w2.weight",
              "core.shared.w2.weight", "dec.eng.0.Wv.weight", "dec.eng.1.Wv.weight", "enc.engram2.Wv.weight"]
    zero1 = ["enc.pooler.Wk.weight", "enc.pooler.g", "enc.shared_gain"]
    for n in alive1:
        assert g[n] > 0, ("dead at step 1", n)
    for n in zero1:
        assert g[n] == 0, ("expected exactly zero at step 1 by construction", n, g[n])
    opt.step(); opt.zero_grad()
    loss, _, _ = M4.loss_terms(m, ids, doc, ei)
    loss.backward()
    for n in zero1:
        p = dict(m.named_parameters())[n]
        assert p.grad is not None and p.grad.abs().sum() > 0, ("still dead at step 2", n)
    torch.manual_seed(6)
    m = M4.build_model("test", k=6, pool="gconv").double().train()
    with torch.no_grad():
        _, c_hat, _, _ = m.enc(ids, doc, ei); m.core.init_codebooks(c_hat[comp])
    loss, _, _ = M4.loss_terms(m, ids, doc, ei); loss.backward()
    g = {n: (p.grad.abs().sum().item() if p.grad is not None else 0.0) for n, p in m.named_parameters()}
    for n in ("enc.pooler.Wg.weight", "enc.pooler.bg", "enc.pooler.w", "enc.pooler.Wo.weight", "enc.pooler.ffn.w2.weight"):
        assert g[n] > 0, ("gconv dead at step 1", n)
    assert g["enc.pooler.g"] == 0
    ok("5 every new branch is alive (gated conv too: W_g, b_g, w, W_O, FFN W2 > 0 at step 1)", "step 1: shared W2 (enc, core), pooler q/b/p/FFN W2, decoder read W_V, read-2 W_V > 0; "
       "pooler W_K, pooler gain g, encoder shared gains = 0 by construction at step 1 and > 0 at step 2")


# ----------------------------------------------------------------------------- 6. codebook gradient isolation
def test_codebook_isolation():
    m = mk(6).train()
    ids, doc = batch()
    ei = engram_indices(ids, doc, m.ecfg)
    logits, aux = M4.forward(m, ids, doc, ei)
    ne, comp, pair = M4.chunk_masks(ids, doc, 6)
    vqc, vqf, n0, n = m.core.vq_terms(aux["c_hat"], comp)
    cec, cef = m.core.ncp_terms(aux["q"], aux["logits_c"], n0, n, pair)
    params = dict(m.named_parameters())
    gr = torch.autograd.grad(cec + cef, list(params.values()), allow_unused=True, retain_graph=True)
    gd = {nme: (x.abs().sum().item() if x is not None else 0.0) for nme, x in zip(params, gr)}
    assert gd["core.E0"] == 0 and gd["core.e"] == 0, "NCP must not train the codebooks"
    assert gd["core.Wq.weight"] > 0 and gd["enc.pooler.Wo.weight"] > 0, "NCP must train head, core and encoder"
    gr = torch.autograd.grad(vqc + vqf, list(params.values()), allow_unused=True)
    gv = {nme: (x.abs().sum().item() if x is not None else 0.0) for nme, x in zip(params, gr)}
    others = [nme for nme, v in gv.items() if v != 0 and nme not in ("core.E0", "core.e")]
    assert gv["core.E0"] > 0 and gv["core.e"] > 0 and not others, others
    gr = torch.autograd.grad(vqc, [params["core.e"]], allow_unused=True)
    assert gr[0] is None or gr[0].abs().sum() == 0, "VQ_coarse must not train the fine codebook"
    ok("6 codebook gradient isolation", "CE_coarse + CE_fine: zero gradient on E0 and PQ, nonzero on head/core/encoder; "
       "VQ terms: gradient ONLY on E0 and PQ; VQ_coarse never touches PQ")


# ----------------------------------------------------------------------------- 7. §6.2 frames
def test_frames():
    m = mk(6)
    ids, doc = batch()
    ei = engram_indices(ids, doc, m.ecfg)
    with torch.no_grad():
        _, aux = M4.forward(m, ids, doc, ei)
        ne, comp, pair = M4.chunk_masks(ids, doc, 6)
        n0, n, _ = m.core.assign(aux["c_hat"])
        base = m.core.ncp_terms(aux["q"], aux["logits_c"], n0, n, pair)
        used = set(n0[:, 1:][pair].tolist())
        unused = [i for i in range(m.core.K) if i not in set(n0.flatten().tolist())]
        assert unused, "need an unused coarse row for this test"
        E0 = m.core.E0.clone()
        m.core.E0[unused[0]] += 1e-3
        n0b, nb, _ = m.core.assign(aux["c_hat"])
        assert torch.equal(n0b, n0)
        cf = m.core.ncp_terms(aux["q"], aux["logits_c"], n0b, nb, pair)[1]
        assert cf == base[1], "fine CE must not depend on a coarse row no target uses"
        _, _, cp_a = m.core.head(aux["u"])
        m.core.E0.copy_(E0)
        _, _, cp_b = m.core.head(aux["u"])
        assert not torch.equal(cp_a, cp_b), "the plan's c_pred reads every coarse row through cbar"
        j = next(iter(used))
        m.core.E0[j] += 1e-3
        n0c, nc_, _ = m.core.assign(aux["c_hat"])
        cf2 = m.core.ncp_terms(aux["q"], aux["logits_c"], n0c, nc_, pair)[1]
        m.core.E0.copy_(E0)
        assert cf2 != base[1], "fine CE must depend on the true next coarse centroid"
    ok("7 §6.2 frames", "fine CE depends on E0 only through the TRUE next coarse code; the plan's c_pred depends on all of E0 via cbar")


# ----------------------------------------------------------------------------- 8. decoder D1 / D2 / D3 per k
def test_decoder():
    for k in (4, 6, 8):
        m = mk(k)
        dec = m.dec
        d = m.c["d"]
        Mn = T // k
        g = torch.Generator().manual_seed(k)
        tok = torch.randn(B, T, d, generator=g, dtype=torch.float64)
        u = torch.randn(B, Mn, d, generator=g, dtype=torch.float64)
        cp = torch.randn(B, Mn, d, generator=g, dtype=torch.float64)
        e = torch.randn(B, T, m.ecfg.n_tables * m.ecfg.engram_dim, generator=g, dtype=torch.float64)
        E = m.enc.embed.weight
        doc = torch.zeros(B, T, dtype=torch.long)
        with torch.no_grad():
            a, _ = dec(tok, u, cp, doc, E, e)
            p = 100
            t2 = tok.clone(); t2[:, p] += 1.0
            ch = changed(a, dec(t2, u, cp, doc, E, e)[0])
            assert ch.min() == p and ch.max() - p == 7, (k, "D1 tok", ch.min(), ch.max())
            e2 = e.clone(); e2[:, p] += 1.0
            ch = changed(a, dec(tok, u, cp, doc, E, e2)[0])
            assert ch.tolist() == [p], (k, "D1 e is position-wise", ch.tolist())
            j = 10
            u2 = u.clone(); u2[:, j] += 1.0
            ch = changed(a, dec(tok, u2, cp, doc, E, e)[0]).tolist()
            assert ch == list(range(k * j + k - 1, k * j + 2 * k - 1)), (k, "D2", ch)
            c2 = cp.clone(); c2[:, j] += 1.0
            assert changed(a, dec(tok, u, c2, doc, E, e)[0]).tolist() == ch, (k, "D2 c_pred")
        doc2 = torch.zeros(B, T, dtype=torch.long); doc2[:, 120:] = 1
        idx, okm, r = plan_index(doc2, k)
        want = [not ((t if t < 120 else t - 120) < k - 1) for t in range(T)]
        assert okm[0].tolist() == want, (k, "D3 null plan")
        with torch.no_grad():
            u3 = u.clone(); u3[:, : 120 // k] += 3.0
            a3, _ = dec(tok, u, cp, doc2, E, e); b3, _ = dec(tok, u3, cp, doc2, E, e)
        assert torch.equal(a3[:, 120:], b3[:, 120:]), (k, "D3 leak")
    # D1 id path through the real Engram indices: token id at p changes e at p..p+2, logits (tok fixed) at p..p+2
    m = mk(6)
    ids, doc = batch()
    p = 100
    ids2 = ids.clone(); ids2[:, p] = (ids[:, p] + 9) % 62 + 2
    with torch.no_grad():
        e1 = m.enc.engram.table(engram_indices(ids, doc, m.ecfg)).reshape(B, T, -1)
        e2 = m.enc.engram.table(engram_indices(ids2, doc, m.ecfg)).reshape(B, T, -1)
        assert changed(e1, e2).tolist() == [p, p + 1, p + 2]
        tok = torch.randn(B, T, 64, dtype=torch.float64); u = torch.randn(B, T // 6, 64, dtype=torch.float64)
        a, _ = m.dec(tok, u, u, doc, m.enc.embed.weight, e1); b, _ = m.dec(tok, u, u, doc, m.enc.embed.weight, e2)
    assert changed(a, b).tolist() == [p, p + 1, p + 2]
    ok("8 decoder D1/D2/D3 at k = 4/6/8 with both Engram reads",
       "tok reach exactly 7; e is position-wise; a token id reaches logits through the reads at exactly p..p+2; "
       "u_j / c_pred_j change exactly positions kj+k-1 .. kj+2k-2 (shift one chunk, repeat k); null at the first k-1 tokens; no cross-document leak")


# ----------------------------------------------------------------------------- 9. empty chunk and pad masking
def test_empty_and_pads():
    for k in (4, 6, 8):
        m = mk(k).train()
        ids, doc = batch()
        # real packing: document 0 = [0, 100) real + trailing pads (same doc id) up to the next k boundary; document 1
        # starts on that boundary; the row ends with an all-PAD chunk carrying a fresh doc id (only place packing makes one)
        end = 100; nb = ((end + k - 1) // k) * k
        ids[:, end:nb] = 0
        doc[:, :nb] = 0; doc[:, nb:] = 1
        ids[:, T - k:] = 0; doc[:, T - k:] = 2
        ei = engram_indices(ids, doc, m.ecfg)
        with torch.no_grad():
            m.core.init_codebooks(torch.randn(500, 64, dtype=torch.float64))
        loss, terms, aux = M4.loss_terms(m, ids, doc, ei)
        loss.backward()
        assert math.isfinite(loss.item()) and all(math.isfinite(v) for v in terms.values())
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        je = T // k - 1
        assert not aux["nonempty"][0, je] and not aux["complete"][0, je] and not aux["pair_ok"][0, je - 1]
        if nb != end:
            jp = end // k                                                # the partial chunk of document 0
            assert aux["nonempty"][0, jp] and not aux["complete"][0, jp] and not aux["pair_ok"][0, jp - 1]
        assert torch.isfinite(aux["c_hat"]).all()
        # pad masking: change the PAD embedding -> no nonempty concept changes
        m.eval()
        with torch.no_grad():
            a = m.enc(ids, doc, ei)[1]
            m.enc.embed.weight[0] += 1.0
            b = m.enc(ids, doc, ei)[1]
        ne = aux["nonempty"]
        assert torch.equal(a[ne], b[ne]), (k, "pad embedding leaked into a nonempty concept")
        assert not torch.equal(a[~ne], b[~ne]), (k, "the empty chunk uses the uniform fallback over pads")
    ok("9 empty chunk and pad masking, k = 4/6/8 (rows packed as the data are)",
       "row-end all-PAD chunk: finite loss and gradients, nonempty = 0, excluded from VQ/NCP; a partial chunk is nonempty but "
       "never a target; changing the PAD embedding changes no nonempty concept")


# ----------------------------------------------------------------------------- 13. the pooler can select by content
def test_pool_selects():
    """§3.8 test 9: train only the pooler (toy objective: copy the marked token's vector) -> one head puts > 0.9 of its
    weight on the marked token at EVERY slot. v3's linear pool (per-slot weights) cannot."""
    torch.manual_seed(7)
    k, d = 6, 64
    marker = torch.randn(d, dtype=torch.float64)
    def data(n=256):
        m = torch.randn(n, k, d, dtype=torch.float64)
        pos = torch.randint(k, (n,))
        m[torch.arange(n), pos] += 3 * marker                                 # the marked token, at a random slot
        return m.reshape(n, 1, k, d).reshape(n, k, d), pos
    res = {}
    sel = {}
    for name, P in (("attn", M4.PoolAttnFFN(d, k, 4, 96, use_ffn=False)), ("gconv", M4.PoolGConv(d, k, 4, 96, use_ffn=False)),
                    ("linear", M4.PoolLinear(d, k))):
        P = P.double(); opt = torch.optim.Adam(P.parameters(), lr=3e-3)
        for _ in range(400):
            m, pos = data()
            c, _ = P(m, torch.ones(len(m), k, dtype=torch.bool))
            tgt = m[torch.arange(len(m)), pos]
            loss = (c[:, 0] - tgt).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m, pos = data(1024)
        with torch.no_grad():
            c, a = P(m, torch.ones(len(m), k, dtype=torch.bool))
            res[name] = (c[:, 0] - m[torch.arange(len(m)), pos]).pow(2).mean().item() / m.pow(2).mean().item()
            if a is not None:
                w = a[:, 0][torch.arange(len(m)), pos]                       # (n, H): weight on the marked slot
                best = w.mean(0).argmax()
                sel[name] = [w[pos == s, best].mean().item() for s in range(k)]
    per_slot = sel["attn"]
    assert min(sel["attn"]) > 0.9 and min(sel["gconv"]) > 2.0 / k, sel        # equal budget: attention selects, gconv partly
    assert res["attn"] < 0.01 and res["linear"] > 10 * res["attn"] and res["gconv"] < res["linear"], res
    # capacity: given more steps the gated conv also selects (sigmoid gates, no competition: slower)
    P = M4.PoolGConv(d, k, 4, 96, use_ffn=False).double(); opt = torch.optim.Adam(P.parameters(), lr=1e-2)
    for _ in range(1000):
        m, pos = data(); c, _ = P(m, torch.ones(len(m), k, dtype=torch.bool))
        loss = (c[:, 0] - m[torch.arange(len(m)), pos]).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    m, pos = data(1024)
    with torch.no_grad():
        c, a = P(m, torch.ones(len(m), k, dtype=torch.bool))
        cap_err = ((c[:, 0] - m[torch.arange(len(m)), pos]).pow(2).mean() / m.pow(2).mean()).item()
        cap_share = a[:, 0][torch.arange(len(m)), pos].mean(0).max().item()
    assert cap_err < 0.01 and cap_share > 0.8, (cap_err, cap_share)
    ok("13 the pooler selects by content (toy)", f"best head's weight on the marked token per slot {[round(x, 3) for x in per_slot]}; "
       f"gated conv {[round(x, 3) for x in sel['gconv']]}; relative error attention {res['attn']:.3f}, gated conv {res['gconv']:.3f}, "
       f"v3 linear pool / plain conv {res['linear']:.3f} (400 steps each); gated conv given 1,000 steps at lr 1e-2: "
       f"error {cap_err:.4f}, gate share {cap_share:.3f}")


# ----------------------------------------------------------------------------- 10. restart order
def test_restart():
    torch.manual_seed(3)
    m = M4.build_model("test", k=6).double().train()
    x = torch.randn(400, 64, dtype=torch.float64) * 3
    m.core.init_codebooks(x)
    m.core.hits0.fill_(1.0); m.core.hits0[:5] = 0
    m.core.hits.fill_(1.0); m.core.hits[:, :7] = 0
    old = m.core.E0.clone()
    m.core.restart_dead(x)
    E0 = m.core.E0
    assert not torch.equal(E0[:5], old[:5]) and torch.equal(E0[5:], old[5:])
    n0 = torch.cdist(x, E0).argmin(-1)
    r_new = (x - E0[n0]).view(-1, 8, 8)
    r_old = (x - old[torch.cdist(x, old).argmin(-1)]).view(-1, 8, 8)
    hit_new = hit_old_only = 0
    for j in range(8):
        for nn_ in range(7):
            v = m.core.e[j, nn_]
            dn = (r_new[:, j] - v).abs().sum(-1).min().item()
            do = (r_old[:, j] - v).abs().sum(-1).min().item()
            hit_new += dn == 0
            hit_old_only += (do == 0) and (dn != 0)
    assert hit_new == 56 and hit_old_only == 0, (hit_new, hit_old_only)
    ok("10 restart order", "dead coarse rows reseeded first; all 56 dead fine codes are residuals against the UPDATED E0")


# ----------------------------------------------------------------------------- 11. bit identity vs the v3 encoder
def test_v3_identity():
    import model_v3ref as V3
    torch.manual_seed(11)
    m3, _, _ = V3.build_model("test")
    m3 = m3.double()
    randomize(m3.enc)
    m4 = M4.build_model("test", k=4, enc_ff=128, engram_dim=8, gr_rank=32, pool="linear", enc_loop=1).double()
    sd = {kk.replace("pool.weight", "pooler.W.weight"): v for kk, v in m3.enc.state_dict().items()}
    missing, unexpected = m4.enc.load_state_dict(sd, strict=False)
    assert not unexpected and all(s.startswith("shared") for s in missing), (missing, unexpected)
    ids, doc = batch(); doc[:, 120:] = 1
    ei = engram_indices(ids, doc, m4.ecfg)
    with torch.no_grad():
        t3, c3, a3 = m3.enc(ids, doc, ei)
        t4, c4, _, a4 = m4.enc(ids, doc, ei)
        assert torch.equal(t3, t4) and torch.equal(c3, c4) and torch.equal(a3["G"], a4["G"])
        t5, c5, _, _ = m4.enc(ids, doc, ei, enc_loop=2)
        assert not torch.equal(c3, c5) and torch.equal(t3, t5)
    ok("11 bit-identity with v3's encoder", "v4 encoder with enc_loop = 1, linear pool, shared W2 = 0 == v3 encoder bit for bit "
       "(tok, c_hat, G); enc_loop = 2 changes c_hat but leaves tok bit-identical (F layers write only G)")


# ----------------------------------------------------------------------------- 12. switch-offs run and are finite
def test_switch_offs():
    m = mk(6)
    ids, doc = batch()
    a, _ = logits_of(m, ids, doc)
    outs = dict(core_off=dict(core_off=True), plan_swap=dict(plan_swap=True), u_off=dict(use_u=False),
                c_off=dict(use_c=False), M1=dict(M=1), loop_off=dict(enc_loop=1), engram_off=dict(engram_off=True),
                enc_shared_off=dict(enc_shared_off=True), core_shared_off=dict(core_shared_off=True),
                pool_ffn_off=dict(pool_ffn_off=True), dec_read0_off=dict(dec_reads_off=(0,)),
                dec_read2_off=dict(dec_reads_off=(1,)), dec_reads_off=dict(dec_reads_off=(0, 1)))
    for nme, kw in outs.items():
        b, _ = logits_of(m, ids, doc, **kw)
        assert torch.isfinite(b).all() and not torch.equal(a, b), nme
    ok("12 all 13 switch-offs run, are finite and change the output", ", ".join(outs))


# ----------------------------------------------------------------------------- 14. block-local attention parity
def test_block_parity():
    import speed_v3
    for k in (4, 6, 8):
        m = mk(k)
        ids, doc = batch(doc_split=120)
        speed_v3.enable_block_attention(0)
        a, _ = logits_of(m, ids, doc)
        speed_v3.enable_block_attention(60)
        b, _ = logits_of(m, ids, doc)
        speed_v3.enable_block_attention(BLOCK)
        assert torch.allclose(a, b, atol=1e-10, rtol=0), (k, (a - b).abs().max())
    ok("14 block-local window attention (block 60) == full-mask SDPA, k = 4/6/8", "float64, max abs diff < 1e-10")


# ----------------------------------------------------------------------------- 15. decoder bypass == v3's decoder
def test_decoder_bypass():
    from hrm_decoder import Decoder
    for k in (4, 6, 8):
        torch.manual_seed(20 + k)
        m4 = M4.build_model("test", k=k).double()
        d3 = Decoder(m4.dcfg).double()
        randomize(d3)
        missing, unexpected = m4.dec.load_state_dict(d3.state_dict(), strict=False)
        assert not unexpected and all(x.startswith("eng.") for x in missing), (missing, unexpected)
        randomize(m4.dec.eng, skip=("Wv",))                              # reads random except W_V = 0 (the bypass)
        g = torch.Generator().manual_seed(k)
        tok = torch.randn(B, T, 64, generator=g, dtype=torch.float64)
        u = torch.randn(B, T // k, 64, generator=g, dtype=torch.float64); cp = torch.randn(B, T // k, 64, generator=g, dtype=torch.float64)
        e = torch.randn(B, T, m4.ecfg.n_tables * m4.ecfg.engram_dim, generator=g, dtype=torch.float64)
        doc = torch.zeros(B, T, dtype=torch.long); doc[:, 120:] = 1
        E = torch.randn(64, 64, generator=g, dtype=torch.float64)
        with torch.no_grad():
            a, _ = d3(tok, u, cp, doc, E)
            b, _ = m4.dec(tok, u, cp, doc, E, e)
            assert torch.equal(a, b), (k, (a - b).abs().max())
            m4.dec.eng[0].Wv.weight.add_(0.1)
            c, _ = m4.dec(tok, u, cp, doc, E, e)
            assert not torch.equal(a, c)
    ok("15 decoder bypass == v3's decoder, k = 4/6/8", "with both read W_V = 0, the v4 decoder equals v3's decoder (k-chunk plan) bit for bit; any W_V != 0 changes it")


BLOCK = 0
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    if "--block" in sys.argv:
        i = sys.argv.index("--block"); BLOCK = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
        import speed_v3
        speed_v3.enable_block_attention(BLOCK)
        print(f"block-local attention ON (block {BLOCK}) for every test")
    tests = [test_causality, test_doc_isolation, test_reach_L, test_pool_init, test_alive, test_codebook_isolation,
             test_frames, test_decoder, test_empty_and_pads, test_restart, test_v3_identity, test_switch_offs,
             test_pool_selects, test_block_parity, test_decoder_bypass]
    only = sys.argv[1:]
    for t in tests:
        if not only or any(o in t.__name__ for o in only):
            t()
    print(f"ALL {len(RESULTS)} TESTS PASS")
