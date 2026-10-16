"""HRM Thinker v3 — reference implementation of the draft_v3.md architecture (CPU, for exact counts and tests).

Encoder: v1 hrm_encoder at d 256, SSSFSSSF, F layers with 6 heads (n_heads_full), Engram read 1 (after layer 0, with conv)
and read 2 (after layer 4, shared table, no conv, W_v zero-init). Core: 4 distinct F-anatomy layers over {C, S} with
per-stream (t, dt)-table read gains and write gains, own SwiGLU + shared SwiGLU (parallel, w2 zero-init), S0 = 0,
trajectories on the grid of 1/3 (main M=3, shortcuts 2a, 2b, 1). NCP head / PQ 32x128x8 = v1's ConceptCoreV2 pieces.
Decoder: v1 confined decoder at d 256, ff 1024. Not a trainer; no GPU. Run: nice -n 19 python3 v3_ref.py
"""
import copy, math, os, sys
from dataclasses import replace
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "run_m2_25M", "code"))
sys.path.insert(0, HERE)
import torch, torch.nn as nn, torch.nn.functional as F
from hrm_encoder import (EncoderConfig, Encoder, GatedAttention, SwiGLU, GRRead, GRWrite, ZCRMSNorm, rms, build_masks,
                         engram_indices)
from hrm_decoder import DecoderConfig, Decoder
from concept_core_v2 import ConceptVocab
from check_engram2 import Encoder2

PAIRS = {(0, 1): 0, (1, 1): 1, (2, 1): 2, (0, 2): 3, (1, 2): 4, (0, 3): 5}      # (t, dt) in thirds -> table row
SCHED = {"3": [(0, 1), (1, 1), (2, 1)], "2a": [(0, 1), (1, 2)], "2b": [(0, 2), (2, 1)], "1": [(0, 3)]}
N_SITES = 7   # gC_attn, gS_attn, a_attn, gC_ffn, gS_ffn, a_own, a_shared


def encoder_v3(cfg, heads_full=6):
    enc = Encoder2(cfg, read2_after=4, conv2=False)
    cfg6 = replace(cfg, n_heads=heads_full)
    for li, kind in enumerate(cfg.layer_pattern):
        if kind == "F":
            enc.layers[li].attn = GatedAttention(cfg6, "F")
    return enc


def gr_read_mod(read: GRRead, streams, mods):
    """v1 GRRead with an extra per-stream multiplicative gain (1 + mod_i); mod_i = 0 -> exactly v1."""
    out = 0
    for i, (R, g) in enumerate(zip(streams, mods)):
        n = rms(R)
        out = out + torch.sigmoid(read.U[i](read.V[i](n))) * n * (1 + read.gamma[i]) * (1 + g)
    return out / len(streams)


class CoreLayerV3(nn.Module):
    def __init__(self, acfg, d_ff):
        super().__init__()
        d = acfg.d_model
        self.read_attn, self.read_ffn = GRRead(d, 2, acfg.gr_rank, acfg.gr_read_bias), GRRead(d, 2, acfg.gr_rank, acfg.gr_read_bias)
        self.write_attn, self.write_ffn = GRWrite(d, 1), GRWrite(d, 1)
        self.attn, self.ffn = GatedAttention(acfg, "F"), SwiGLU(d, d_ff)

    def forward(self, C, S, cdoc, mask, m, shared):
        x = gr_read_mod(self.read_attn, [C, S], [m[0], m[1]])
        S = S + self.write_attn(x) * (1 + m[2]) * self.attn(x, cdoc, mask, None)
        x = gr_read_mod(self.read_ffn, [C, S], [m[3], m[4]])
        y = (1 + m[5]) * self.ffn(x)
        if shared is not None:
            y = y + (1 + m[6]) * shared(x)
        return S + self.write_ffn(x) * y


class ConceptCoreV3(nn.Module):
    def __init__(self, acfg, n_layers=4, d_ff=768, shared_ff=1024, pq=(32, 128, 8), use_table=True):
        super().__init__()
        d = acfg.d_model
        self.layers = nn.ModuleList([CoreLayerV3(acfg, d_ff) for _ in range(n_layers)])
        self.shared = SwiGLU(d, shared_ff) if shared_ff else None
        if self.shared is not None:
            nn.init.zeros_(self.shared.w2.weight)                     # branch writes exactly 0 at init
        self.table = nn.Parameter(torch.zeros(len(PAIRS), n_layers, N_SITES, d)) if use_table else None
        self.out_norm = ZCRMSNorm(d)
        S_, N_, ds = pq
        assert S_ * ds == d
        self.vocab = ConceptVocab(d, S_, N_)
        self.Wq = nn.Linear(d, d, bias=False)
        self.log_tau = nn.Parameter(torch.full((S_,), math.log(2 * d // S_)))
        self.d = d

    def mods(self, pair, li, ref):
        if self.table is None:
            return [torch.zeros((), dtype=ref.dtype)] * N_SITES
        return list(self.table[PAIRS[pair], li])

    def one_pass(self, C, S, cdoc, mask, pair):
        for li, layer in enumerate(self.layers):
            S = layer(C, S, cdoc, mask, self.mods(pair, li, C), self.shared)
        return S

    def trajectory(self, C, cdoc, sched="3", S0=None, return_states=False):
        _, mask = build_masks(cdoc, 1)
        S = torch.zeros_like(C) if S0 is None else S0
        states = []
        for pair in SCHED[sched] if isinstance(sched, str) else sched:
            S = self.one_pass(C, S, cdoc, mask, pair)
            states.append(S)
        return (S, states) if return_states else S

    def head(self, u):
        e = self.vocab.e.detach(); V = self.vocab
        q = self.Wq(u).unflatten(-1, (V.S, V.ds))
        d2 = q.pow(2).sum(-1, keepdim=True) - 2 * torch.einsum("...sd,snd->...sn", q, e) + e.pow(2).sum(-1)
        logits = -d2 / self.log_tau.exp()[:, None]
        return logits, torch.einsum("...sn,snd->...sd", logits.softmax(-1), e).flatten(-2)


class HRMThinkerV3(nn.Module):
    def __init__(self, d=256, V=8192, enc_ff=1024, dec_ff=1024, core_ff=736, shared_ff=1024, heads=4, heads_full=6,
                 engram_rows=208333, small=False):
        super().__init__()
        kw = dict(vocab_size=V, d_model=d, n_heads=heads, n_kv_heads=2, head_dim=64, d_ff=enc_ff, d_concept=d,
                  engram_rows=engram_rows, engram_dim=48)
        if small:
            kw.update(head_dim=16, rope_dims=8, engram_dim=8)
        self.ecfg = EncoderConfig(**kw)
        self.enc = encoder_v3(self.ecfg, heads_full)
        acfg = replace(self.ecfg, n_heads=heads_full)
        self.core = ConceptCoreV3(acfg, d_ff=core_ff, shared_ff=shared_ff,
                                  pq=(32, 128, 8) if not small else (8, 16, d // 8))
        dk = dict(d_model=d, n_heads=heads, n_kv_heads=2, head_dim=kw["head_dim"], d_ff=dec_ff, n_layers=4, window=8,
                  layout="confined", plan_in="add", d_concept=d, rope_dims=kw.get("rope_dims", 16))
        self.dcfg = DecoderConfig(**dk)
        self.dec = Decoder(self.dcfg)

    def forward(self, ids, doc, eidx, sched="3", shortcut=None):
        tok, c_hat, _ = self.enc(ids, doc, eidx)
        cdoc = doc[:, ::4]
        S = self.core.trajectory(c_hat, cdoc, sched)
        u = self.core.out_norm(S)
        lc, c_pred = self.core.head(u)
        logits, _ = self.dec(tok, u, c_pred, doc, self.enc.embed.weight)
        out = dict(c_hat=c_hat, cdoc=cdoc, u=u, logits_c=lc)
        if shortcut is not None:
            us = self.core.out_norm(self.core.trajectory(c_hat, cdoc, shortcut))
            out.update(u_s=us, logits_cs=self.core.head(us)[0])
        return logits, out


def param_table(m):
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    table = m.enc.engram.table.weight.numel(); emb = m.enc.embed.weight.numel(); code = m.core.vocab.e.numel()
    enc = n(m.enc) - table - emb
    core = n(m.core) - code
    dec = n(m.dec)
    c = m.core
    parts = dict(core_layers=sum(n(l) for l in c.layers), core_layer=n(c.layers[0]), shared=n(c.shared),
                 table=c.table.numel(), head=n(c.Wq) + c.log_tau.numel(), out_norm=n(c.out_norm),
                 enc_S=n(m.enc.layers[0]), enc_F=n(m.enc.layers[3]), eng1=n(m.enc.engram) - table, eng2=n(m.enc.engram2),
                 dec_layer=n(m.dec.layers[0]), plan=n(m.dec.plan))
    return dict(encoder=enc, core=core, decoder=dec, compute=enc + core + dec, engram_table=table, embedding=emb,
                codebook=code, stored=enc + core + dec + table + emb + code, parts=parts)


# ----------------------------------------------------------------------------- exact tests (float64, small config)
def tests():
    torch.manual_seed(0); torch.set_default_dtype(torch.float64)
    m = HRMThinkerV3(d=64, V=64, enc_ff=128, dec_ff=128, core_ff=96, shared_ff=128, engram_rows=1021, small=True)
    with torch.no_grad():                                           # randomise everything except the init-identity parts
        for nme, p in m.named_parameters():
            if "engram.table" in nme or "core.table" in nme or "core.shared.w2" in nme:
                continue
            p.add_(torch.randn_like(p) * 0.1)
    m.eval(); core = m.core
    B, T = 2, 128
    ids = torch.randint(0, 64, (B, T)); doc = torch.zeros(B, T, dtype=torch.long); doc[1, 64:] = 1
    eidx = engram_indices(ids, doc, m.ecfg)
    res = {}
    with torch.no_grad():
        tok, C, _ = m.enc(ids, doc, eidx); cdoc = doc[:, ::4]
        # T1 init identities: table = 0 and shared w2 = 0  ==  plain 4-layer v1-grammar core without table / shared
        ref = copy.deepcopy(core); ref.table = None; ref.shared = None
        res["T1_init_identity_table0_sharedw2_0"] = torch.equal(core.trajectory(C, cdoc, "3"), ref.trajectory(C, cdoc, "3"))
        # after randomising the table and w2, outputs must differ, and M=1 != M=3
        core.table.add_(torch.randn_like(core.table) * 0.1); core.shared.w2.weight.add_(torch.randn_like(core.shared.w2.weight) * 0.1)
        S3 = core.trajectory(C, cdoc, "3"); S1 = core.trajectory(C, cdoc, "1")
        res["T1b_trained_table_changes_output"] = not torch.equal(S3, ref.trajectory(C, cdoc, "3"))
        res["T1c_M1_differs_from_M3"] = not torch.allclose(core.out_norm(S1), core.out_norm(S3))
        # T2 causality over concepts: edit concept p -> states at concepts < p unchanged
        p = 10; C2 = C.clone(); C2[:, p] += 1.0                     # row 0: one doc; row 1: concept 10 is in doc 0 (0-15)
        S3b = core.trajectory(C2, cdoc, "3")
        res["T2_causal_over_concepts"] = (torch.equal(S3[:, :p], S3b[:, :p]) and not torch.equal(S3[0, p:], S3b[0, p:])
                                          and not torch.equal(S3[1, p:16], S3b[1, p:16]))
        # T3 document isolation: edit a concept in doc 1 of row 1 -> doc 0 of row 1 unchanged; row 0 unchanged
        C3 = C.clone(); C3[1, 5] += 1.0                              # edit doc 0 of row 1 (concept 5 < boundary 16)
        S3c = core.trajectory(C3, cdoc, "3")                        # LATER concepts of doc 1 must not see it
        res["T3_doc_isolation"] = (torch.equal(S3[1, 16:], S3c[1, 16:]) and torch.equal(S3[0], S3c[0])
                                   and not torch.equal(S3[1, 5:16], S3c[1, 5:16]))
        # T5 reuse of the (1/3, 2/3) first step: continue from main's S1 == 2a from scratch
        _, st = core.trajectory(C, cdoc, "3", return_states=True)
        cont = core.trajectory(C, cdoc, [(1, 2)], S0=st[0])
        res["T5_2a_reuses_main_step1_exactly"] = torch.equal(cont, core.trajectory(C, cdoc, "2a"))
        # T6 batch-row permutation invariance (dense layers: no capacity exception)
        perm = torch.tensor([1, 0])
        res["T6_batch_permutation_invariant"] = torch.allclose(core.trajectory(C[perm], cdoc[perm], "3"), S3[perm], atol=0, rtol=0)
    # T4 full-gradient reach: final NTP loss has nonzero gradient at S1, S2, S3 and C; shortcut loss reaches C
    m.train()
    tok, C, _ = m.enc(ids, doc, eidx); cdoc = doc[:, ::4]
    C.retain_grad()
    S, st = core.trajectory(C, cdoc, "3", return_states=True)
    for s_ in st:
        s_.retain_grad()
    u = core.out_norm(S); lc, cp = core.head(u)
    logits, _ = m.dec(tok, u, cp, doc, m.enc.embed.weight)
    F.cross_entropy(logits[:, :-1].reshape(-1, 64), ids[:, 1:].reshape(-1)).backward(retain_graph=True)
    res["T4_ntp_grad_reaches_S1_S2_S3_C"] = all(s_.grad is not None and s_.grad.abs().sum() > 0 for s_ in st) and C.grad.abs().sum() > 0
    C.grad = None
    u.retain_grad(); u.grad = None
    core.out_norm.gamma.grad = None
    Ss = core.trajectory(C, cdoc, "1")
    ((rms(Ss) - rms(S).detach()) ** 2).mean().backward()          # L_scons is gain-free: rms(S_s) vs sg(rms(S_3))
    res["T4b_shortcut_scons_reaches_C_not_u3"] = C.grad.abs().sum() > 0 and (u.grad is None or u.grad.abs().sum() == 0)
    res["T4c_scons_does_not_touch_out_norm_gain"] = core.out_norm.gamma.grad is None or core.out_norm.gamma.grad.abs().sum() == 0
    torch.set_default_dtype(torch.float32)
    return res


if __name__ == "__main__":
    r = tests()
    for k, v in r.items():
        print(f"{k:40s} {'PASS' if v else 'FAIL'}")
    torch.manual_seed(0)
    m = HRMThinkerV3(engram_rows=11)                                 # tiny table; the real one is added analytically
    t = param_table(m)
    t["engram_table"] = 8 * 208333 * 48
    t["stored"] = t["compute"] + t["engram_table"] + t["embedding"] + t["codebook"]
    for k in ("encoder", "core", "decoder", "compute", "engram_table", "embedding", "codebook", "stored"):
        print(f"{k:14s} {t[k]:>12,d}")
    print({k: f"{v:,d}" for k, v in t["parts"].items()})
    print(f"shares enc/core/dec: {t['encoder']/t['compute']*100:.1f} / {t['core']/t['compute']*100:.1f} / {t['decoder']/t['compute']*100:.1f} %")
