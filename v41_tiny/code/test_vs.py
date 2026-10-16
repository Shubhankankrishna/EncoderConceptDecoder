"""Acceptance tests for the v4-vs-LLM experiment (CPU, float64 where exactness matters).

python test_vs.py [--block 60]
"""
import argparse
import random
import sys
import numpy as np
import torch

import model_v4 as M4
import model_llm as ML
import data_v4 as D
import pinpad as PP
import run_vs as R

torch.set_num_threads(2)


def rand_llm(V=64, T=None, seed=0, **over):
    torch.manual_seed(seed)
    m = ML.build_model("test", V=V, **over).double()
    with torch.no_grad():
        for n, p in m.named_parameters():                              # randomise gains / convs so nothing is vacuous
            if p.dim() == 1 or n.endswith(".w"):
                p.add_(0.3 * torch.randn_like(p))
    return m.eval()


def rows(B, T, V, seed, docs=None):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, V, (B, T), generator=g)
    doc = torch.zeros(B, T, dtype=torch.long) if docs is None else docs
    return ids, doc


# ----------------------------------------------------------------------------- 1. accounting
def test_param_match():
    v4 = M4.param_table(M4.build_model("tiny", pool="attn", dec_reads=()))["compute"]
    assert v4 == 1_212_032, v4
    t = ML.param_table(ML.build_model("tiny"))
    assert abs(t["compute"] / v4 - 1) < 0.01, (t["compute"], v4)
    m = ML.build_model("tiny")
    kinds = "".join(l.kind for l in m.layers)
    assert kinds.count("S") == 3 * kinds.count("F") and kinds == "SSSF" * 4, kinds
    assert t["unique_layers"] == 16 and t["depth"] == 22, t
    assert m.schedule == [0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15], m.schedule
    v = M4.build_model("tiny", pool="attn", dec_reads=())
    depth_v4 = sum(v.c["enc_loop"] if l.kind == "F" else 1 for l in v.enc.layers) + v.c["core_layers"] * v.c["M"] + 4
    assert depth_v4 == 22, depth_v4
    assert len(v.dec.eng) == 0 and v.enc.pooler.ffn is None
    return f"LLM {t['compute']:,} vs v4-rec {v4:,} ({100 * (t['compute'] / v4 - 1):+.2f}%), 16 unique / 22 deep both, SSSF x4"


# ----------------------------------------------------------------------------- 2. causality, documents, window
def test_causal_and_docs():
    V, T = 64, 120
    m = rand_llm(V)
    docs = torch.tensor([[0] * 50 + [1] * 70, [0] * 120])
    ids, doc = rows(2, T, V, 1, docs)
    base = R.fwd(m, ids, doc, None)[0]
    for p in (0, 30, 59, 60, 100, 119):
        ids2 = ids.clone(); ids2[:, p] = (ids2[:, p] + 7) % (V - 5) + 5
        out = R.fwd(m, ids2, doc, None)[0]
        assert torch.allclose(out[:, :p], base[:, :p], atol=1e-10), f"future leak at p={p}"
        assert (out[:, p:] - base[:, p:]).abs().max() > 1e-6, f"no effect of p={p}"
    ids2 = ids.clone(); ids2[0, :50] = 5                                # change document 0 of row 0
    out = R.fwd(m, ids2, doc, None)[0]
    assert torch.allclose(out[0, 50:], base[0, 50:], atol=1e-10), "document leak"
    return "no future leak at 6 positions; no cross-document leak"


def test_window():
    """An S attention module sees exactly the last 60 positions; an F module sees everything causal."""
    V, T = 64, 180
    m = rand_llm(V)
    c = m.ecfg
    S = m.layers[0].attn; Fm = m.layers[3].attn
    assert S.kind == "S" and Fm.kind == "F" and c.window == 60
    x = torch.randn(1, T, c.d_model, dtype=torch.double)
    doc = torch.zeros(1, T, dtype=torch.long)
    import speed_v3
    speed_v3._CTX["doc"] = doc
    swa, full = ML.build_masks(doc, c.window)                           # the masks the LLM itself builds
    from hrm_encoder import rope_cache
    rope = rope_cache(T, c.rope_dims, c.rope_base, x.device, x.dtype)
    t = 150
    def dep(mod, mask, j):
        x2 = x.clone(); x2[0, j] += 1.0
        with torch.no_grad():
            return (mod(x2, doc, mask, rope)[0, t] - mod(x, doc, mask, rope)[0, t]).abs().max().item()
    # q/k/v short convs (kernel 4) extend a window-60 attention's reach by 3 positions: keys t-59..t, each reading x
    # back to 3 more; so x_j influences y_t iff t - j <= 59 + 3.
    assert dep(S, swa, t - 62) > 1e-8 and dep(S, swa, t - 63) == 0.0, "S reach != 62 (window 60 + conv 3)"
    assert dep(Fm, full, 0) > 1e-8, "F does not see position 0"
    return "S reach exactly 62 (window 60 + short conv 3); F reaches position 0"


def test_loops():
    V, T = 64, 120
    m = rand_llm(V)
    ids, doc = rows(2, T, V, 3)
    o = R.fwd(m, ids, doc, None)[0]
    o1 = R.fwd(m, ids, doc, None, M=1)[0]
    o2 = R.fwd(m, ids, doc, None, enc_loop=1)[0]
    assert (o - o1).abs().max() > 1e-4 and (o - o2).abs().max() > 1e-4
    assert len(m.make_schedule(M=1)) == 18 and len(m.make_schedule(enc_loop=1)) == 20
    _, aux = R.fwd(m, ids, doc, None, collect=True)
    assert len(aux["states"]) == 23
    return "M = 1 and enc_loop = 1 change the output; schedules 22 / 18 / 20"


def test_block_parity():
    import speed_v3
    V, T = 64, 120
    m = rand_llm(V)
    ids, doc = rows(2, T, V, 4, torch.tensor([[0] * 37 + [1] * 83, [5] * 120]))
    speed_v3.enable_block_attention(0)
    a = R.fwd(m, ids, doc, None)[0]
    speed_v3.enable_block_attention(60)
    b = R.fwd(m, ids, doc, None)[0]
    speed_v3.enable_block_attention(BLOCK)
    err = (a - b).abs().max().item()
    assert err < 1e-9, err
    return f"block-60 vs full-mask SDPA max |diff| {err:.1e}"


# ----------------------------------------------------------------------------- 3. identical rows for both arms
def test_same_rows():
    a = R.get_args(["--name", "x", "--arch", "v4", "--data", "pinpad", "--seed", "3"])
    s1, s2 = PP.PinpadStream(a.T, 6, a.B, seed=3), PP.PinpadStream(a.T, 6, a.B, seed=3)
    seq1 = [s1.next() for _ in range(3)]                                 # v4: step-0 batch + 2 for codebook init
    seq2 = [s2.next() for _ in range(3)]                                 # llm: step-0 batch + 2 consumed
    for _ in range(5):
        seq1.append(s1.next()); seq2.append(s2.next())
    for (t1, d1), (t2, d2) in zip(seq1, seq2):
        assert (t1 == t2).all() and (d1 == d2).all()
    t, d = seq1[0]
    for r in range(len(t)):                                              # every doc starts on a k boundary
        starts = np.flatnonzero(np.diff(d[r]) != 0) + 1
        assert all(s % 6 == 0 for s in starts)
    src = open(R.__file__).read()
    assert "stream.next(); stream.next()" in src
    return "pinpad streams identical across arch (8 batches), documents start on 6-token boundaries"


def test_runner_rows():
    """The real training loop (R.main, 4 steps) feeds byte-identical batches to v4 and the LLM."""
    import shutil, os
    rec = {}
    orig_tb, orig_fe = D.to_batch, R.final_eval
    def tb(t, d, canon, ecfg):
        rec.setdefault(cur[0], []).append((np.asarray(t).copy(), np.asarray(d).copy()))
        return orig_tb(t, d, canon, ecfg)
    cur = [None]
    try:
        R.final_eval = lambda m, a, S: {}
        D.to_batch = tb
        for arch in ("v4", "llm"):
            cur[0] = arch
            R.main(["--name", f"_test_rows_{arch}", "--arch", arch, "--data", "pinpad", "--seed", "5", "--max_steps", "4",
                    "--eval_every_tokens", "1e12", "--threads", "2"])
            shutil.rmtree(os.path.join(R.RUNS, f"_test_rows_{arch}"))
    finally:
        D.to_batch, R.final_eval = orig_tb, orig_fe
    v4_train = [rec["v4"][0]] + rec["v4"][3:]                           # v4: batches 1-2 go to the codebook init
    llm_train = rec["llm"]
    assert len(v4_train) == len(llm_train) == 4, (len(rec["v4"]), len(rec["llm"]))
    for (t1, d1), (t2, d2) in zip(v4_train, llm_train):
        assert (t1 == t2).all() and (d1 == d2).all(), "training rows differ between arch"
    return "R.main: the 4 training batches are identical for v4 and the LLM"


# ----------------------------------------------------------------------------- 4. pinpad environment
def test_pinpad_expert():
    rng = random.Random(7)
    n = 0
    for kind in ("heldin", "heldout", "long"):
        for ep in PP.eval_episodes(kind, 60, 5):
            lay = ep["layout"]
            env = PP.Env(lay)
            toks = ep["tokens"]
            assert toks[:27] == PP.header_tokens(lay["colors"], set(lay["walls"]), lay["start"])
            assert toks[-1] == PP.END and len(toks) % 2 == 0
            i = 27
            while toks[i] != PP.END:
                a = toks[i] - PP.ACT0; assert 0 <= a < 4
                assert a in ep["opt"][i], "expert action not optimal"
                cell = env.act(a)
                assert toks[i + 1] == PP.pos_tok(cell), "position token != dynamics"
                assert ep["sub"][i] == ep["tgt"][i] // 2
                i += 2
            assert tuple(env.visits) == ep["task"], (env.visits, ep["task"])
            assert env.steps <= PP.TMAX
            n += 1
    # optimal actions really reduce the obstacle-aware distance, and BFS agrees with a brute-force Bellman sweep
    colors, walls, start = PP.sample_layout(rng)
    target = colors[0]
    blocked = set(walls) | {c for c in colors if c != target}
    dist = PP.bfs_dist(target, blocked)
    bf = np.full(49, 10 ** 6); bf[target] = 0
    for _ in range(60):
        for u in range(49):
            if u in blocked or u == target:
                continue
            for a in range(4):
                v = PP.step(u, a, walls)
                if v != u and v not in blocked:
                    bf[u] = min(bf[u], bf[v] + 1)
    for u in range(49):
        if u not in blocked:
            assert (dist[u] == -1 and bf[u] >= 10 ** 6) or dist[u] == bf[u], (u, dist[u], bf[u])
    assert set(PP.HELDOUT_TASKS).isdisjoint(PP.TRAIN_TASKS) and len(PP.HELDOUT_TASKS) == 20
    long_max = max(len(e["tokens"]) for e in PP.eval_episodes("long", 64, 0))
    assert long_max <= 480, long_max
    return f"{n} expert episodes replayed: optimal actions, dynamics, visit order == task; BFS == Bellman; long max {long_max} tokens"


def test_boundary_flags():
    ep = PP.eval_episodes("heldin", 1, 11)[0]
    bf = PP.boundary_flags(ep)
    acts = [i for i, k in enumerate(ep["kind"]) if k == 1]
    firsts = [i for i in acts if bf[i] in (1, 2)]
    assert len(firsts) == len(ep["task"]), (len(firsts), ep["task"])
    for j, i in enumerate(firsts):
        assert ep["tgt"][i] == ep["task"][j] and bf[i] == (1 if ep["task"][j] % 2 == 0 else 2)
    assert all(bf[i] == 0 for i in acts if i not in firsts)
    return "one flagged first action per visit; flag 1 for pair starts, 2 for pair seconds"


def test_metrics_oracle():
    """pin_metrics on an oracle (logits = one-hot of the true next token, with a non-optimal action shift for a
    mutant check) gives action optimality 1 and position accuracy 1; a shifted oracle does not."""
    eps = PP.eval_episodes("heldin", 12, 1)
    class Oracle(torch.nn.Module):
        def __init__(self, shift=0):
            super().__init__(); self.shift = shift
            self.c = dict(d=8); self.ecfg = ML.build_model("test", V=PP.V).ecfg
        def __call__(self, ids, doc, **kw):
            nxt = torch.roll(ids, -1 - self.shift, dims=1)
            return torch.nn.functional.one_hot(nxt, PP.V).double() * 30.0, {}
    orig = R.fwd
    try:
        R.fwd = lambda m, ids, doc, ei, **kw: m(ids, doc)
        good = R.pin_metrics(Oracle(), eps, 480)
        bad = R.pin_metrics(Oracle(shift=1), eps, 480)
    finally:
        R.fwd = orig
    assert good["act_opt"] == 1.0 and good["pos_acc"] == 1.0 and good["ntp_action"] < 1e-6, good
    assert bad["pos_acc"] < 0.2, bad
    return f"oracle act_opt {good['act_opt']:.2f} pos_acc {good['pos_acc']:.2f}; shifted oracle pos_acc {bad['pos_acc']:.2f}"


# ----------------------------------------------------------------------------- 5. text filler fix, end-to-end steps
def test_filler_fix():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(D.SRC + "/tokenizer.json")
    tr, va = D.TextStore("train"), D.TextStore("val")
    f_tr = D.filler_sentences(tr, tok, n_docs=60000, seed=0)
    f_va = D.filler_sentences(va, tok, n_docs=va.n, seed=0)
    s_tr = set(map(tuple, f_tr)); s_va = set(map(tuple, f_va))
    ov = len(s_va & s_tr) / len(s_va)
    assert len(f_tr) > 100_000 and len(f_va) > 3_000 and ov < 0.05, (len(f_tr), len(f_va), ov)
    return f"fillers: train pool {len(f_tr):,} sentences (60,000 docs; was 3,000 docs), val pool {len(f_va):,} from the val store; overlap {100 * ov:.2f}%"


def test_train_steps():
    out = []
    for arch in ("v4", "llm"):
        for data in ("pinpad", "text"):
            torch.manual_seed(0)
            V = PP.V if data == "pinpad" else 8192
            m = R.build(arch, V)
            if data == "pinpad":
                s = PP.PinpadStream(480, 6, 2, seed=0); canon = PP.Ident()
            else:
                tr = D.TextStore("val"); canon = D.Canon()
                s = D.MixedStream(tr, None, 480, 6, 2, p_syn=0.0, seed=0)
            ids, doc, ei = D.to_batch(*s.next(), canon, m.ecfg)
            if arch == "v4":
                with torch.no_grad():
                    cands = []
                    for _ in range(4):
                        b_ = D.to_batch(*s.next(), canon, m.ecfg)
                        cands.append(m.enc(*b_)[1][M4.chunk_masks(b_[0], b_[1], 6)[1]])
                    m.core.init_codebooks(torch.cat(cands), torch.Generator().manual_seed(0))
            opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
            l0 = None
            for i in range(3):
                loss, terms, _ = R.loss_fn(m, ids, doc, ei)
                opt.zero_grad(); loss.backward(); opt.step()
                l0 = l0 or terms["ntp"]
            assert np.isfinite(terms["ntp"]) and terms["ntp"] < l0, (arch, data, l0, terms)
            out.append(f"{arch}/{data} ntp {l0:.2f}->{terms['ntp']:.2f}")
    return "; ".join(out)


TESTS = [test_param_match, test_causal_and_docs, test_window, test_loops, test_block_parity, test_same_rows, test_runner_rows,
         test_pinpad_expert, test_boundary_flags, test_metrics_oracle, test_filler_fix, test_train_steps]
BLOCK = 60

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--block", type=int, default=60); ap.add_argument("--only", default="")
    args = ap.parse_args()
    BLOCK = args.block
    import speed_v3
    speed_v3.enable_block_attention(BLOCK)
    fails = 0
    for t in TESTS:
        if args.only and args.only not in t.__name__:
            continue
        try:
            print(f"PASS {t.__name__}: {t()}", flush=True)
        except Exception as e:
            fails += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}", flush=True)
    print(f"{len(TESTS) - fails if not args.only else ''} passed, {fails} failed (block {BLOCK})")
    sys.exit(1 if fails else 0)
