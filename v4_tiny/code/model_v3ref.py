"""HRM THINKER v3 (19M compute, d 256) — the draft_v3.md (r2) architecture, trainer-grade.

Encoder: v1 hrm_encoder at d 256, SSSFSSSF, F layers with 6 heads; ONE Engram gather feeding read 1 (after layer 0, with
conv, v1's module) and read 2 (after layer 4, no conv, W_v zero-init, gate reads L only).
Core: 4 distinct F-anatomy layers over {C, S}; per-stream (t, dt)-table read gains and write gains; own SwiGLU 736 +
shared SwiGLU 1024 in parallel (shared w2 zero-init); S0 = 0; main trajectory M = 3 on the grid of 1/3 plus a
training-only shortcut (1 | 2a | 2b), batched with the main passes at concept rate (rows [main; shortcut]).
Decoder: v1 confined decoder (d 256, ff 1024, 4 heads), tied head. No MoE, no NextLat.

Module names follow v3_design/v3_ref.py so a state_dict transfers (bit-parity test in test_v3.py). v3_ref aliases the
table as enc.engram2.table; here read 2 receives the gathered rows instead (one gather, review C4/C5).

forward(m, ids, doc, eidx, R="3", ...) keeps v1's evaluation signature; R is a schedule name "1" | "2a" | "2b" | "3"
(ints map 1 -> "1", 2 -> "2b", 3 -> "3").
"""
import math
from dataclasses import replace
import torch
import torch.nn as nn
import torch.nn.functional as F

import compat  # noqa: F401  (EncoderLayer._apply rename; .to(device) works)
import speed as _speed
from hrm_encoder import (EncoderConfig, Encoder, GatedAttention, SwiGLU, GRRead, GRWrite, ZCRMSNorm, rms, build_masks,
                         rope_cache)
from hrm_decoder import DecoderConfig, Decoder
from concept_core_v2 import ConceptVocab

PAD = 0
PAIRS = {(0, 1): 0, (1, 1): 1, (2, 1): 2, (0, 2): 3, (1, 2): 4, (0, 3): 5}      # (t, dt) in thirds -> table row
SCHED = {"3": [(0, 1), (1, 1), (2, 1)], "2a": [(0, 1), (1, 2)], "2b": [(0, 2), (2, 1)], "1": [(0, 3)]}
SHORTCUTS = ("1", "2a", "2b")
N_SITES = 7   # gC_attn, gS_attn, a_attn, gC_ffn, gS_ffn, a_own, a_shared
R_MAP = {1: "1", 2: "2b", 3: "3"}

FULL = dict(d=256, V=8192, heads=4, heads_full=6, kv=2, head_dim=64, rope=16, enc_ff=1024, dec_ff=1024, core_ff=736,
            shared_ff=1024, engram_rows=208333, engram_dim=48, gr_rank=32, pq=(32, 128, 8), table=True)
SMALL = dict(d=64, V=8192, heads=4, heads_full=6, kv=2, head_dim=16, rope=8, enc_ff=128, dec_ff=128, core_ff=96,
             shared_ff=128, engram_rows=1021, engram_dim=8, gr_rank=8, pq=(8, 16, 8), table=True)   # CPU trainer smoke
TEST = dict(SMALL, V=64, gr_rank=32)                                                         # = v3_ref small config


# ============================================================================ encoder
class EngramRead2(nn.Module):
    """Second read of the shared Engram rows: own W_k, W_v (zero-init), gated read of L only, write to {L, G}, no conv."""
    def __init__(self, cfg):
        super().__init__()
        d, de = cfg.d_model, cfg.n_tables * cfg.engram_dim
        self.Wk = nn.Linear(de, d, bias=False)
        self.Wv = nn.Linear(de, d, bias=False); nn.init.zeros_(self.Wv.weight)
        self.read = GRRead(d, 1, cfg.gr_rank, cfg.gr_read_bias)
        self.write = GRWrite(d, 2)
        self.d = d

    def forward(self, L, G, e, doc):
        h = self.read([L]); k, v = self.Wk(e), self.Wv(e)
        a = torch.sigmoid((rms(h) * rms(k)).sum(-1, keepdim=True) / math.sqrt(self.d))
        y = F.silu(a * v)
        s = self.write(h)
        return L + s[..., 0:1] * y, G + s[..., 1:2] * y


def engram_read1(eng, L, G, e, doc):
    """v1 Engram.forward with the gathered rows passed in (identical arithmetic)."""
    h = eng.read([L])
    k, v = eng.Wk(e), eng.Wv(e)
    alpha = torch.sigmoid((rms(h) * rms(k)).sum(-1, keepdim=True) / math.sqrt(eng.d))
    y = F.silu(eng.conv(alpha * v, doc))
    s = eng.write(h)
    return L + s[..., 0:1] * y, G + s[..., 1:2] * y, alpha


class EncoderV3(Encoder):
    def __init__(self, cfg, heads_full=6, read2_after=4):
        super().__init__(cfg)
        self.read2_after = read2_after
        self.engram2 = EngramRead2(cfg)
        cfg6 = replace(cfg, n_heads=heads_full)
        for li, kind in enumerate(cfg.layer_pattern):
            if kind == "F":
                self.layers[li].attn = GatedAttention(cfg6, "F")
        assert cfg.layer_pattern[read2_after] == "S"

    def forward(self, ids, doc, eng_idx, engram_off=False, engram_read2_off=False, F_off=False):
        own = getattr(_speed._tls, "memo", None) is None                # doc-shift mask memo (speed.py); an outer
        if own:                                                          # step-wide memo (speed_v3.step_memo) is reused
            _speed._tls.memo = {}
        try:
            return self._fwd(ids, doc, eng_idx, engram_off, engram_read2_off, F_off)
        finally:
            if own:
                _speed._tls.memo = None

    def _fwd(self, ids, doc, eng_idx, engram_off, engram_read2_off, F_off):
        c = self.cfg
        B, T = ids.shape
        assert T % c.chunk == 0
        swa_mask, full_mask = build_masks(doc, c.window)
        cos, sin = rope_cache(T, c.rope_dims, c.rope_base, ids.device, self.embed.weight.dtype)
        e0 = rms(self.embed(ids))
        L, G = e0, e0
        e = self.engram.table(eng_idx).reshape(B, T, -1)                # ONE gather, both reads
        alpha = None
        for li, layer in enumerate(self.layers):
            if not (F_off and layer.kind == "F"):                        # F layers write only G: skipping == writes 0
                L, G = layer(L, G, doc, swa_mask if layer.kind == "S" else full_mask, (cos, sin))
            if li == c.engram_after and not engram_off:
                L, G, alpha = engram_read1(self.engram, L, G, e, doc)
            if li == self.read2_after and not (engram_off or engram_read2_off):
                L, G = self.engram2(L, G, e, doc)
        mix = torch.softmax(self.crc_router(rms(G)), dim=-1)
        m = mix[..., 0:1] * e0 + mix[..., 1:2] * rms(L) + mix[..., 2:3] * rms(G)
        M = T // c.chunk
        cvec = self.pool(m.reshape(B, M, c.chunk * c.d_model))
        c_hat = math.sqrt(c.dc) * cvec / (cvec.norm(dim=-1, keepdim=True) + c.eps)
        if alpha is None:
            alpha = torch.zeros(B, T, 1, device=ids.device, dtype=e0.dtype)
        return self.tok_norm(L), c_hat, dict(L=L, G=G, mix=mix, engram_alpha=alpha)


# ============================================================================ concept module
class CoreLayerV3(nn.Module):
    def __init__(self, acfg, d_ff):
        super().__init__()
        d = acfg.d_model
        self.read_attn, self.read_ffn = GRRead(d, 2, acfg.gr_rank, acfg.gr_read_bias), GRRead(d, 2, acfg.gr_rank, acfg.gr_read_bias)
        self.write_attn, self.write_ffn = GRWrite(d, 1), GRWrite(d, 1)
        self.attn, self.ffn = GatedAttention(acfg, "F"), SwiGLU(d, d_ff)

    @staticmethod
    def c_half(read, C):
        """Pass-independent C half of a gated read: sigma(U V n) * n * (1 + gamma_C) (same association as v3_ref)."""
        n = rms(C)
        return torch.sigmoid(read.U[0](read.V[0](n))) * n * (1 + read.gamma[0])

    @staticmethod
    def read_mod(read, xc, S, gC, gS):
        """v3_ref gr_read_mod over [C, S] with the C half precomputed; g* are (1 + mod) factors or None (= exactly 1)."""
        tC = xc if gC is None else xc * gC
        n = rms(S)
        tS = torch.sigmoid(read.U[1](read.V[1](n))) * n * (1 + read.gamma[1])
        if gS is not None:
            tS = tS * gS
        return (0 + tC + tS) / 2

    def forward(self, xca, xcf, S, cdoc, mask, g, shared):
        """g: list of 7 (1 + mod) factors (None = no table), xca / xcf: cached C halves of the two reads."""
        x = self.read_mod(self.read_attn, xca, S, g[0], g[1])
        a = self.attn(x, cdoc, mask, None)
        S = S + (self.write_attn(x) * a if g[2] is None else self.write_attn(x) * g[2] * a)
        x = self.read_mod(self.read_ffn, xcf, S, g[3], g[4])
        y = self.ffn(x) if g[5] is None else g[5] * self.ffn(x)
        if shared is not None:
            y = y + (shared(x) if g[6] is None else g[6] * shared(x))
        return S + self.write_ffn(x) * y


class ConceptCoreV3(nn.Module):
    def __init__(self, acfg, n_layers=4, d_ff=736, shared_ff=1024, pq=(32, 128, 8), use_table=True, table_rows=None):
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

    # ---------------------------------------------------------------- passes
    def _gains(self, onep, rows, li, n_rows):
        """(1 + mod) factors for layer li: rows = one table row (vector gains) or a list of per-row-block rows."""
        if onep is None:
            return [None] * N_SITES
        if isinstance(rows, int):
            return list(onep[rows, li])                                  # 7 x (d,)
        blocks = onep[rows, li]                                          # (nb, 7, d)
        per = blocks.repeat_interleave(n_rows, 0)                        # (nb * n_rows, 7, d)
        return [per[:, s, None, :] for s in range(N_SITES)]              # 7 x (rows, 1, d)

    def one_pass(self, cache, S, cdoc, mask, onep, rows, n_rows, shared):
        for li, layer in enumerate(self.layers):
            xca, xcf = cache[li]
            S = layer(xca, xcf, S, cdoc, mask, self._gains(onep, rows, li, n_rows), shared)
        return S

    def c_cache(self, C):
        return [(CoreLayerV3.c_half(l.read_attn, C), CoreLayerV3.c_half(l.read_ffn, C)) for l in self.layers]

    def run(self, C, cdoc, sched="3", short=None, table_off=False, shared_off=False, return_states=False):
        """Main trajectory `sched` from S0 = 0; if `short` is given (training), the shortcut trajectory runs batched with
        the main passes (rows [main; shortcut]); a common schedule prefix (2a's first step) is reused, not recomputed.
        Returns S_main, list of main states, S_short (or None)."""
        B = C.shape[0]
        onep = None if (self.table is None or table_off) else 1 + self.table
        shared = None if shared_off else self.shared
        _, mask = build_masks(cdoc, 1)
        cache = self.c_cache(C)
        main = SCHED[sched] if isinstance(sched, str) else list(sched)
        sh = SCHED[short] if short else []
        pre = 0
        while short and pre < min(len(main), len(sh)) and main[pre] == sh[pre]:
            pre += 1
        S = torch.zeros_like(C)
        states, S_s = [], None
        cache2 = mask2 = cdoc2 = None
        for k, pair in enumerate(main):
            if short and pre <= k < len(sh):
                if cache2 is None:
                    cache2 = [(torch.cat([a, a]), torch.cat([f, f])) for a, f in cache]
                    mask2, cdoc2 = torch.cat([mask, mask]), torch.cat([cdoc, cdoc])
                    S_s = S                                              # shortcut starts from the shared prefix state
                Sb = self.one_pass(cache2, torch.cat([S, S_s]), cdoc2, mask2, onep,
                                   [PAIRS[pair], PAIRS[sh[k]]], B, shared)
                S, S_s = Sb[:B], Sb[B:]
            else:
                S = self.one_pass(cache, S, cdoc, mask, onep, PAIRS[pair], B, shared)
            states.append(S)
        return S, states, S_s

    def trajectory(self, C, cdoc, sched="3", **kw):
        return self.run(C, cdoc, sched, **kw)[0]

    # ---------------------------------------------------------------- readout, NCP, VQ
    def head(self, u, pred=True):
        e = self.vocab.e.detach(); V = self.vocab
        q = self.Wq(u).unflatten(-1, (V.S, V.ds))
        d2 = q.pow(2).sum(-1, keepdim=True) - 2 * torch.einsum("...sd,snd->...sn", q, e) + e.pow(2).sum(-1)
        logits = -d2 / self.log_tau.exp()[:, None]
        if not pred:
            return logits, None
        return logits, torch.einsum("...sn,snd->...sd", logits.softmax(-1), e).flatten(-2)

    def forward(self, c_hat, cdoc, R="3", **kw):
        """Evaluation interface (v1's m.core(c_hat, cdoc, R)): u, NCP logits, c_pred."""
        S = self.trajectory(c_hat, cdoc, R_MAP.get(R, R), **kw)
        u = self.out_norm(S)
        logits, c_pred = self.head(u)
        return u, logits, c_pred

    def assign(self, c):
        return self.vocab.assign(c)

    def vq_sum(self, c_hat, complete, count_hits):
        """Sum over COMPLETE real chunks of the per-chunk mean (over segments) squared error; codebook-only gradient.
        Returns (sum, code indices for every chunk)."""
        V = self.vocab
        c = c_hat.detach()
        n = V.assign(c)
        if count_hits:
            with torch.no_grad():
                V.hits += F.one_hot(n[complete].reshape(-1, V.S), V.N).sum(0).to(V.hits.dtype)
                V.step += 1
        cs = c.unflatten(-1, (V.S, V.ds))
        q = V.e[torch.arange(V.S, device=c.device), n]
        per = (cs - q).pow(2).sum(-1).mean(-1)                           # (B, M)
        return (per * complete).sum(), n

    @staticmethod
    def ncp_sum(logits, n, pair_ok):
        """Sum over valid pairs (j -> j+1) of the per-segment-mean CE of chunk j+1's code, predicted at chunk j."""
        ce = F.cross_entropy(logits[:, :-1].flatten(0, 2), n[:, 1:].flatten(), reduction="none").view(n[:, 1:].shape)
        return (ce.mean(-1) * pair_ok).sum()


def chunk_masks(ids, doc, k=4):
    """nonempty (B, M): chunk has >= 1 real token; complete (B, M): 4 real tokens; pair_ok (B, M-1): current chunk
    nonempty, target chunk complete, same document (documents start on chunk boundaries, verified in the data)."""
    B, T = ids.shape
    real = (ids != PAD).view(B, T // k, k)
    nonempty, complete = real.any(-1), real.all(-1)
    cdoc = doc[:, ::k]
    pair_ok = nonempty[:, :-1] & complete[:, 1:] & (cdoc[:, 1:] == cdoc[:, :-1])
    return nonempty, complete, pair_ok


# ============================================================================ full model
class HRMThinkerV3(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = dict(c)
        kw = dict(vocab_size=c["V"], d_model=c["d"], n_heads=c["heads"], n_kv_heads=c["kv"], head_dim=c["head_dim"],
                  d_ff=c["enc_ff"], d_concept=c["d"], engram_rows=c["engram_rows"], engram_dim=c["engram_dim"],
                  gr_rank=c["gr_rank"], rope_dims=c["rope"])
        self.ecfg = EncoderConfig(**kw)
        self.enc = EncoderV3(self.ecfg, c["heads_full"])
        acfg = replace(self.ecfg, n_heads=c["heads_full"])
        self.core = ConceptCoreV3(acfg, d_ff=c["core_ff"], shared_ff=c["shared_ff"], pq=tuple(c["pq"]),
                                  use_table=c.get("table", True))
        self.dcfg = DecoderConfig(d_model=c["d"], n_heads=c["heads"], n_kv_heads=c["kv"], head_dim=c["head_dim"],
                                  d_ff=c["dec_ff"], n_layers=4, window=8, layout="confined", plan_in="add",
                                  d_concept=c["d"], gr_rank=c["gr_rank"], rope_dims=c["rope"])
        self.dec = Decoder(self.dcfg)


def build_model(size="full", **over):
    c = dict({"full": FULL, "small": SMALL, "test": TEST}[size]); c.update(over)
    m = HRMThinkerV3(c)
    return m, m.ecfg, m.dcfg


def forward(m, ids, doc, eidx, R="3", core_off=False, plan_swap=False, use_u=True, use_c=True, shared_ffn_off=False,
            table_off=False, engram_off=False, engram_read2_off=False, F_off=False, return_passes=False):
    """Evaluation / generation path (one trajectory, no shortcut). Returns logits (B, T, V) and aux."""
    sched = R_MAP.get(R, R)
    tok, c_hat, eaux = m.enc(ids, doc, eidx, engram_off=engram_off, engram_read2_off=engram_read2_off, F_off=F_off)
    cdoc = doc[:, ::4]
    S, states, _ = m.core.run(c_hat, cdoc, sched, table_off=table_off, shared_off=shared_ffn_off)
    u = m.core.out_norm(S)
    lc, c_pred = m.core.head(u)
    if plan_swap:
        u_dec, c_dec = u.roll(1, dims=0), c_pred.roll(1, dims=0)
    else:
        u_dec, c_dec = u, c_pred
    logits, daux = m.dec(tok, u_dec, c_dec, doc, m.enc.embed.weight, core_off=core_off, use_u=use_u, use_c=use_c)
    aux = dict(c_hat=c_hat, cdoc=cdoc, logits_c=lc, u=u, c_pred=c_pred, h=daux["h"], P=daux["P"], enc=eaux, tok=tok)
    if return_passes:
        aux["passes"] = states
    return logits, aux


# ============================================================================ accounting and optimizer groups
def param_table(m):
    """Repository convention: compute = encoder non-embedding + core without codebook + decoder. Parts reconcile."""
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    table = m.enc.engram.table.weight.numel(); emb = m.enc.embed.weight.numel(); code = m.core.vocab.e.numel()
    enc = n(m.enc) - table - emb
    core = n(m.core) - code
    dec = n(m.dec)
    c = m.core
    parts = dict(enc_layers=sum(n(l) for l in m.enc.layers), eng1=n(m.enc.engram) - table, eng2=n(m.enc.engram2),
                 crc_pool_norm=n(m.enc.crc_router) + n(m.enc.pool) + n(m.enc.tok_norm),
                 core_layers=sum(n(l) for l in c.layers), shared=n(c.shared) if c.shared is not None else 0,
                 table=c.table.numel() if c.table is not None else 0, head=n(c.Wq) + c.log_tau.numel(),
                 out_norm=n(c.out_norm), dec_layers=sum(n(l) for l in m.dec.layers), plan=n(m.dec.plan),
                 dec_out_norm=n(m.dec.out_norm))
    ledger = sum(parts.values())
    unique = sum(p.numel() for p in m.parameters())
    assert ledger + table + emb + code == unique, (ledger, table, emb, code, unique)   # review C5: ledger reconciles
    assert enc + core + dec == ledger
    return dict(encoder=enc, core=core, decoder=dec, compute=enc + core + dec, engram_table=table, embedding=emb,
                codebook=code, stored=unique, parts=parts)


NO_DECAY = (".bias", "embed.", "engram.table", "log_tau", "vocab.e", "plan.null", "plan.offset", "core.table")


def param_groups(model, weight_decay=0.1):
    """No decay: biases, embeddings, Engram table, temperatures, conv kernels (.w), codebook, null plan, chunk offsets,
    and the (t, dt) table (draft r2 §4.3). Everything else (matrices, zc gains) is decayed, as v1."""
    decay, no_decay, names = [], [], []
    for n, p in model.named_parameters():
        if n.endswith(".bias") or "embed." in n or "engram.table" in n or "log_tau" in n or n.endswith(".w") \
                or n.endswith("vocab.e") or n.endswith("plan.null") or n.endswith("plan.offset") or n == "core.table":
            no_decay.append(p); names.append(n)
        else:
            decay.append(p)
    return [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)], names


if __name__ == "__main__":
    torch.manual_seed(0)
    m, e, d = build_model(engram_rows=11)
    t = param_table(m)
    t["engram_table"] = 8 * 208333 * 48
    t["stored"] = t["compute"] + t["engram_table"] + t["embedding"] + t["codebook"]
    print({k: v for k, v in t.items() if k != "parts"}); print(t["parts"])
    assert t["compute"] == 18986871 and t["stored"] == 101116663, t
    print("compute params 18,986,871 OK")
