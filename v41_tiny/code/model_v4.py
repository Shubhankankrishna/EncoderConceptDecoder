"""HRM Thinker v4 (draft_v4.md, revision 2), size-parametric: FULL = 19,898,011 compute parameters at k = 6; TINY for
the CPU experiment in v4_tiny/.

Encoder (§3): v3's 8 layers SSSFSSSF (F layers 6 heads), L/G gated residual, ONE Engram gather feeding read 1 (after
  layer 0, with conv) and read 2 (after layer 4); NEW: F layers 3 and 7 applied twice in place (enc_loop = 2); NEW:
  shared SwiGLU beside every layer's own FFN, y = FFN_l(x) + g_l * FFN_shared(x) (g_l = 1, shared W2 = 0);
  NEW: pooler over k tokens (§3.5.4): 4-head attention (W_V = W_O = I, q = b = p = 0) then a SwiGLU on the pooled vector
  (W2 = 0, g = 1); pads masked; an empty chunk falls back to uniform weights. "linear" = v3's pool (ablation).
Core (§4): 4 distinct F-anatomy layers over {C, S}, own SwiGLU + shared SwiGLU (shared W2 = 0), M = 2 plain passes
  from S0 = 0, no (t, dt) table, terminal-only. Head: q = W_q u; coarse logits -|q - E0|^2 / tau0; fine PQ logits on a
  residual frame. Plan's c_pred = cbar + PQexp(q - cbar) (§6.2); the fine loss uses the frame q - E0[k_true].
Decoder (§5): v3's confined decoder; plan shifted one chunk and repeated over k tokens (PlanBuilder with chunk = k);
  NEW: two Engram reads of the encoder's gathered rows e, after decoder layers 0 and 2 (W_V = 0).
Objective (§6.1): NTP + 0.05 CE_coarse + 0.1 CE_fine + VQ_coarse + VQ_fine (VQ terms: codebook-only gradients).
"""
import math
import sys
from dataclasses import replace
import torch
import torch.nn as nn
import torch.nn.functional as F

import compat  # noqa: F401  (EncoderLayer._apply rename so .to() and .double() work)
import speed as _speed  # doc-shift mask memo (identical arithmetic; masks carry no gradient)
from hrm_encoder import (EncoderConfig, EncoderLayer, Engram, GatedAttention, SwiGLU, GRRead, GRWrite, ZCRMSNorm, rms,
                         build_masks, rope_cache)
from hrm_decoder import DecoderConfig, PlanBuilder, ConfinedLayer

PAD = 0

FULL = dict(d=256, V=8192, heads=4, heads_full=6, kv=2, head_dim=64, rope=16, enc_ff=1024, dec_ff=1024, core_ff=736,
            shared_ff=1024, enc_shared_ff=384, engram_rows=208333, engram_dim=48, gr_rank=32, coarse=1024,
            pq=(32, 256, 8), k=6, pool="attnffn", pool_heads=4, pool_ff=384, enc_loop=2, core_layers=4, M=2)
TINY = dict(d=64, V=8192, heads=4, heads_full=6, kv=2, head_dim=16, rope=8, enc_ff=256, dec_ff=256, core_ff=184,
            shared_ff=256, enc_shared_ff=96, engram_rows=4093, engram_dim=12, gr_rank=8, coarse=64,
            pq=(8, 256, 8), k=6, pool="attnffn", pool_heads=4, pool_ff=96, enc_loop=2, core_layers=4, M=2)
TEST = dict(TINY, V=64, engram_rows=1021, coarse=16, pq=(8, 16, 8))


# ============================================================================ encoder
class EngramRead2(nn.Module):
    """v3's second Engram read (also the anatomy of the decoder reads): own W_k, W_v (zero-init), gated read of one
    stream, write to n_write streams, no conv."""
    def __init__(self, d, de, rank, bias, n_write=2):
        super().__init__()
        self.Wk = nn.Linear(de, d, bias=False)
        self.Wv = nn.Linear(de, d, bias=False); nn.init.zeros_(self.Wv.weight)
        self.read = GRRead(d, 1, rank, bias)
        self.write = GRWrite(d, n_write)
        self.d = d

    def gate(self, X, e):
        h = self.read([X])
        a = torch.sigmoid((rms(h) * rms(self.Wk(e))).sum(-1, keepdim=True) / math.sqrt(self.d))
        return h, a

    def forward(self, L, G, e):
        h, a = self.gate(L, e)
        y = F.silu(a * self.Wv(e)); s = self.write(h)
        return L + s[..., 0:1] * y, G + s[..., 1:2] * y


def engram_read1(eng, L, G, e, doc):
    """v1/v3 Engram.forward with the gathered rows passed in (identical arithmetic)."""
    h = eng.read([L]); k, v = eng.Wk(e), eng.Wv(e)
    alpha = torch.sigmoid((rms(h) * rms(k)).sum(-1, keepdim=True) / math.sqrt(eng.d))
    y = F.silu(eng.conv(alpha * v, doc)); s = eng.write(h)
    return L + s[..., 0:1] * y, G + s[..., 1:2] * y, alpha


def layer_forward(layer, L, G, doc, mask, rope, shared=None, gain=None):
    """compat's EncoderLayer forward, plus the shared FFN as a parallel branch on the same read and write gate."""
    x = layer.read_attn([L] if layer.kind == "S" else [L, G])
    L, G = layer._write_streams(L, G, layer.attn(x, doc, mask, rope), layer.write_attn(x))
    x = layer.read_ffn([L] if layer.kind == "S" else [L, G])
    y = layer.ffn(x)
    if shared is not None:
        y = y + gain * shared(x)
    return layer._write_streams(L, G, y, layer.write_ffn(x))


class PoolLinear(nn.Module):
    """v3's pool: one linear map over the concatenated chunk, initialised to the exact mean (pads included)."""
    def __init__(self, d, k):
        super().__init__()
        self.k = k
        self.W = nn.Linear(k * d, d, bias=False)
        with torch.no_grad():
            self.W.weight.copy_(torch.cat([torch.eye(d)] * k, dim=1) / k)

    def forward(self, m, real, ffn_off=False):
        B, T, d = m.shape
        return self.W(m.reshape(B, T // self.k, self.k * d)), None


class PoolAttnFFN(nn.Module):
    """§3.5.4: H-head attention pooling over the chunk's k tokens (shared W_K, W_V across slots, position vectors p_r,
    position biases b_{h,r}), then an optional SwiGLU on the pooled vector. Init = masked mean of the real tokens."""
    def __init__(self, d, k, H=4, ff=96, use_ffn=True):
        super().__init__()
        assert d % H == 0
        self.k, self.H, self.dh = k, H, d // H
        self.Wk = nn.Linear(d, d, bias=False); nn.init.normal_(self.Wk.weight, std=0.02)
        self.Wv = nn.Linear(d, d, bias=False); self.Wo = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.Wv.weight.copy_(torch.eye(d)); self.Wo.weight.copy_(torch.eye(d))
        self.q = nn.Parameter(torch.zeros(H, self.dh))
        self.p = nn.Parameter(torch.zeros(k, d))
        self.b = nn.Parameter(torch.zeros(H, k))
        self.ffn = SwiGLU(d, ff) if use_ffn else None
        if self.ffn is not None:
            nn.init.zeros_(self.ffn.w2.weight)
            self.g = nn.Parameter(torch.ones(()))

    def forward(self, m, real, ffn_off=False):
        B, T, d = m.shape
        k, H, dh = self.k, self.H, self.dh
        M = T // k
        mc = m.view(B, M, k, d) + self.p                                       # values: m_i + p_r
        xk = rms(m).view(B, M, k, d) + self.p                                  # keys:   rms(m_i) + p_r
        K = self.Wk(xk).view(B, M, k, H, dh)
        V = self.Wv(mc).view(B, M, k, H, dh)
        s = (K * self.q).sum(-1) / math.sqrt(dh) + self.b.t()                  # (B, M, k, H)
        r = real.view(B, M, k)
        allow = r | ~r.any(-1, keepdim=True)                                   # empty chunk -> uniform fallback
        a = s.masked_fill(~allow[..., None], float("-inf")).softmax(2)         # over the chunk's slots
        c = self.Wo((a[..., None] * V).sum(2).reshape(B, M, d))
        if self.ffn is not None and not ffn_off:
            n = rms(c)
            c = c + self.g * self.ffn(n)
        return c, a


class PoolGConv(nn.Module):
    """Owner's convolution hypothesis (2026-10-01), content-gated so it is not v3's pool in disguise.
    A plain Conv1d(d, d, kernel k, stride k) over the chunk IS v3's linear pool (its weight reshapes to W_p), so the
    'linear' arm already covers it. Here: a depthwise multi-channel kernel (slot weights w_r per channel) times a per-token
    content gate per channel group ("multi-head"): g_i = 2 sigmoid(W_g rms(m_i) + b_{h, r(i)}) in R^H, expanded to d.
      c0 = W_O sum_i real_i * g_i * w_{r(i)} * m_i,  then the same SwiGLU residual as the attention pooler.
    Gates are sigmoids: absolute selection, no competition across the chunk (the difference from attention).
    Init: W_g = b = 0 (g = 1), w = 1/k, W_O = I -> c0 = (n_real / k) * masked mean: same direction, so c_hat equals the
    attention pooler's at step 0."""
    def __init__(self, d, k, H=4, ff=96, use_ffn=True):
        super().__init__()
        assert d % H == 0
        self.k, self.H = k, H
        self.Wg = nn.Linear(d, H, bias=False); nn.init.zeros_(self.Wg.weight)
        self.bg = nn.Parameter(torch.zeros(H, k))
        self.w = nn.Parameter(torch.full((k, d), 1.0 / k))
        self.Wo = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.Wo.weight.copy_(torch.eye(d))
        self.ffn = SwiGLU(d, ff) if use_ffn else None
        if self.ffn is not None:
            nn.init.zeros_(self.ffn.w2.weight)
            self.g = nn.Parameter(torch.ones(()))

    def forward(self, m, real, ffn_off=False):
        B, T, d = m.shape
        k, H = self.k, self.H
        M = T // k
        mc = m.view(B, M, k, d)
        r = real.view(B, M, k, 1).to(m.dtype)
        gate = 2 * torch.sigmoid(self.Wg(rms(m)).view(B, M, k, H) + self.bg.t())          # (B, M, k, H)
        ge = gate.repeat_interleave(d // H, dim=-1)                                       # (B, M, k, d)
        c = self.Wo((r * ge * self.w * mc).sum(2))
        if self.ffn is not None and not ffn_off:
            c = c + self.g * self.ffn(rms(c))
        a = gate * r
        a = a / a.sum(2, keepdim=True).clamp_min(1e-12)                                  # telemetry: gate share
        return c, a


class EncoderV4(nn.Module):
    def __init__(self, cfg: EncoderConfig, c):
        super().__init__()
        self.cfg, self.k, self.enc_loop = cfg, c["k"], c["enc_loop"]
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, d); nn.init.normal_(self.embed.weight, std=0.02)
        self.layers = nn.ModuleList([EncoderLayer(cfg, kind) for kind in cfg.layer_pattern])
        cfg6 = replace(cfg, n_heads=c["heads_full"])
        for li, kind in enumerate(cfg.layer_pattern):
            if kind == "F":
                self.layers[li].attn = GatedAttention(cfg6, "F")
        self.engram = Engram(cfg)
        self.engram2 = EngramRead2(d, cfg.n_tables * cfg.engram_dim, cfg.gr_rank, cfg.gr_read_bias)
        self.read2_after = 4
        self.shared = SwiGLU(d, c["enc_shared_ff"]) if c["enc_shared_ff"] else None
        if self.shared is not None:
            nn.init.zeros_(self.shared.w2.weight)
            self.shared_gain = nn.Parameter(torch.ones(len(cfg.layer_pattern)))
        self.crc_router = nn.Linear(d, 3)
        nn.init.zeros_(self.crc_router.weight)
        with torch.no_grad():
            self.crc_router.bias.copy_(torch.tensor(cfg.crc_bias))
        if c["pool"] == "linear":
            self.pooler = PoolLinear(d, self.k)
        elif c["pool"] == "gconv":
            self.pooler = PoolGConv(d, self.k, c["pool_heads"], c["pool_ff"])
        else:
            self.pooler = PoolAttnFFN(d, self.k, c["pool_heads"], c["pool_ff"], use_ffn=(c["pool"] == "attnffn"))
        self.tok_norm = ZCRMSNorm(d)

    def forward(self, ids, doc, eidx, **kw):
        own = getattr(_speed._tls, "memo", None) is None                # memoise same_doc_shift_mask for this call
        if own:
            _speed._tls.memo = {}
        try:
            return self._fwd(ids, doc, eidx, **kw)
        finally:
            if own:
                _speed._tls.memo = None

    def _fwd(self, ids, doc, eidx, enc_loop=None, engram_off=False, shared_off=False, pool_ffn_off=False):
        c, k = self.cfg, self.k
        loops = self.enc_loop if enc_loop is None else enc_loop
        B, T = ids.shape
        assert T % k == 0
        swa, full = build_masks(doc, c.window)
        rope = rope_cache(T, c.rope_dims, c.rope_base, ids.device, self.embed.weight.dtype)
        e0 = rms(self.embed(ids))
        L, G = e0, e0
        e = self.engram.table(eidx).reshape(B, T, -1)                          # ONE gather: 2 encoder + 2 decoder reads
        shared = None if (self.shared is None or shared_off) else self.shared
        alpha = None
        for li, layer in enumerate(self.layers):
            reps = loops if layer.kind == "F" else 1
            for _ in range(reps):
                L, G = layer_forward(layer, L, G, doc, swa if layer.kind == "S" else full, rope, shared,
                                     None if shared is None else self.shared_gain[li])
            if li == c.engram_after and not engram_off:
                L, G, alpha = engram_read1(self.engram, L, G, e, doc)
            if li == self.read2_after and not engram_off:
                L, G = self.engram2(L, G, e)
        mix = torch.softmax(self.crc_router(rms(G)), dim=-1)
        m = mix[..., 0:1] * e0 + mix[..., 1:2] * rms(L) + mix[..., 2:3] * rms(G)
        cvec, att = self.pooler(m, ids != PAD, ffn_off=pool_ffn_off)
        c_hat = math.sqrt(c.dc) * cvec / (cvec.norm(dim=-1, keepdim=True) + c.eps)
        return self.tok_norm(L), c_hat, e, dict(L=L, G=G, mix=mix, m=m, pool_attn=att, engram_alpha=alpha)


# ============================================================================ concept module
class CoreLayerV4(nn.Module):
    def __init__(self, acfg, d_ff):
        super().__init__()
        d = acfg.d_model
        self.read_attn = GRRead(d, 2, acfg.gr_rank, acfg.gr_read_bias)
        self.read_ffn = GRRead(d, 2, acfg.gr_rank, acfg.gr_read_bias)
        self.write_attn, self.write_ffn = GRWrite(d, 1), GRWrite(d, 1)
        self.attn, self.ffn = GatedAttention(acfg, "F"), SwiGLU(d, d_ff)

    def forward(self, C, S, cdoc, mask, shared):
        x = self.read_attn([C, S])
        S = S + self.write_attn(x) * self.attn(x, cdoc, mask, None)
        x = self.read_ffn([C, S])
        y = self.ffn(x)
        if shared is not None:
            y = y + shared(x)
        return S + self.write_ffn(x) * y


class ConceptCoreV4(nn.Module):
    def __init__(self, acfg, c):
        super().__init__()
        d = acfg.d_model
        self.layers = nn.ModuleList([CoreLayerV4(acfg, c["core_ff"]) for _ in range(c["core_layers"])])
        self.shared = SwiGLU(d, c["shared_ff"]) if c["shared_ff"] else None
        if self.shared is not None:
            nn.init.zeros_(self.shared.w2.weight)
        self.out_norm = ZCRMSNorm(d)
        S_, N_, ds = c["pq"]
        assert S_ * ds == d
        self.S, self.N, self.ds, self.K = S_, N_, ds, c["coarse"]
        self.E0 = nn.Parameter(torch.randn(self.K, d))                       # coarse codebook (excluded from compute)
        self.e = nn.Parameter(torch.randn(S_, N_, ds))                       # fine PQ codebook (excluded from compute)
        self.Wq = nn.Linear(d, d, bias=False)
        self.log_tau = nn.Parameter(torch.full((S_,), math.log(2 * ds)))     # fine temperatures (v3 rule)
        self.log_tau0 = nn.Parameter(torch.full((), math.log(2 * d)))        # coarse temperature (§6.2)
        self.M = c["M"]
        self.register_buffer("hits0", torch.zeros(self.K))
        self.register_buffer("hits", torch.zeros(S_, N_))
        self.d = d

    def run(self, C, cdoc, M=None, shared_off=False):
        _, mask = build_masks(cdoc, 1)
        shared = None if shared_off else self.shared
        S = torch.zeros_like(C)
        states = []
        for _ in range(self.M if M is None else M):
            for layer in self.layers:
                S = layer(C, S, cdoc, mask, shared)
            states.append(S)
        return S, states

    # ------------------------------------------------------------ head
    def fine_logits(self, frame):
        e = self.e.detach()
        q = frame.unflatten(-1, (self.S, self.ds))
        d2 = q.pow(2).sum(-1, keepdim=True) - 2 * torch.einsum("...sd,snd->...sn", q, e) + e.pow(2).sum(-1)
        return -d2 / self.log_tau.exp()[:, None]

    def head(self, u):
        """Returns q, coarse logits, and the plan's c_pred = cbar + PQexp(q - cbar) (no future information)."""
        E0 = self.E0.detach()
        q = self.Wq(u)
        d2 = q.pow(2).sum(-1, keepdim=True) - 2 * q @ E0.t() + E0.pow(2).sum(-1)
        lc = -d2 / self.log_tau0.exp()
        cbar = lc.softmax(-1) @ E0
        lf = self.fine_logits(q - cbar)
        pq = torch.einsum("...sn,snd->...sd", lf.softmax(-1), self.e.detach()).flatten(-2)
        return q, lc, cbar + pq

    # ------------------------------------------------------------ codes and VQ
    def assign(self, c):
        """Coarse code, then PQ codes of the residual against that coarse centroid."""
        c = c.detach()
        n0 = torch.cdist(c.reshape(-1, self.d), self.E0.detach()).argmin(-1).view(c.shape[:-1])
        r = c - self.E0.detach()[n0]
        rs = r.unflatten(-1, (self.S, self.ds))
        n = (rs.unsqueeze(-2) - self.e.detach()).pow(2).sum(-1).argmin(-1)
        return n0, n, r

    def vq_terms(self, c_hat, complete, count_hits=True):
        """Sums over complete chunks: VQ_coarse = |sg c - E0_k|^2 / S; VQ_fine = segment-mean |sg r - e_n|^2
        (r from a detached E0, so each term trains only its own codebook)."""
        n0, n, r = self.assign(c_hat)
        c = c_hat.detach()
        vqc = (c - self.E0[n0]).pow(2).sum(-1) / self.S
        rs = r.unflatten(-1, (self.S, self.ds))
        qf = self.e[torch.arange(self.S, device=c.device), n]
        vqf = (rs - qf).pow(2).sum(-1).mean(-1)
        if count_hits and self.training:
            with torch.no_grad():
                self.hits0 += torch.bincount(n0[complete], minlength=self.K).to(self.hits0.dtype)
                self.hits += F.one_hot(n[complete].reshape(-1, self.S), self.N).sum(0).to(self.hits.dtype)
        return (vqc * complete).sum(), (vqf * complete).sum(), n0, n

    def ncp_terms(self, q, lc, n0, n, pair_ok):
        """Sums over pair_ok (j -> j+1): CE of the next coarse code; segment-mean CE of the next fine codes with logits
        from the teacher-forced frame q_j - E0[k_{j+1}] (§6.2: inside the loss only, never in the plan)."""
        ce_c = F.cross_entropy(lc[:, :-1].flatten(0, 1), n0[:, 1:].flatten(), reduction="none").view(n0[:, 1:].shape)
        frame = q[:, :-1] - self.E0.detach()[n0[:, 1:]]
        lf = self.fine_logits(frame)
        ce_f = F.cross_entropy(lf.flatten(0, 2), n[:, 1:].flatten(), reduction="none").view(n[:, 1:].shape).mean(-1)
        return (ce_c * pair_ok).sum(), (ce_f * pair_ok).sum()

    @torch.no_grad()
    def init_codebooks(self, c, gen=None):
        """k-means++: coarse first, then the fine codes on the residuals."""
        x = c.detach().reshape(-1, self.d)
        self.E0.copy_(kpp(x, self.K, gen))
        n0 = torch.cdist(x, self.E0).argmin(-1)
        r = (x - self.E0[n0]).view(-1, self.S, self.ds)
        for j in range(self.S):
            self.e[j] = kpp(r[:, j], self.N, gen)

    @torch.no_grad()
    def restart_dead(self, c, opt=None, gen=None):
        """Coarse rows first, then residuals recomputed with the updated E0, then the fine codes (§6.2). Adam moments of
        restarted rows are zeroed."""
        x = c.detach().reshape(-1, self.d)
        out = dict(dead0=0, dead=0)
        if len(x) == 0:
            return out
        dead0 = self.hits0 == 0
        if dead0.any():
            pick = torch.randint(len(x), (int(dead0.sum()),), generator=gen)
            self.E0[dead0] = x[pick]
            _zero_moments(opt, self.E0, dead0)
            out["dead0"] = int(dead0.sum())
        n0 = torch.cdist(x, self.E0).argmin(-1)
        r = (x - self.E0[n0]).view(-1, self.S, self.ds)
        for j in range(self.S):
            dead = self.hits[j] == 0
            if dead.any():
                pick = torch.randint(len(r), (int(dead.sum()),), generator=gen)
                self.e[j, dead] = r[pick, j]
                _zero_moments(opt, self.e, (j, dead))
                out["dead"] += int(dead.sum())
        out["used0"] = float((self.hits0 > 0).float().mean()); out["used"] = float((self.hits > 0).float().mean())
        self.hits0.zero_(); self.hits.zero_()
        return out


def _zero_moments(opt, p, idx):
    if opt is None:
        return
    st = opt.state.get(p, {})
    for key in ("exp_avg", "exp_avg_sq"):
        if key in st:
            st[key][idx] = 0


def kpp(x, n, gen=None):
    """k-means++ seeding of n centres from rows of x (with replacement when x has fewer distinct rows)."""
    pick = [x[torch.randint(len(x), (1,), generator=gen)]]
    dmin = (x - pick[0]).pow(2).sum(-1)
    for _ in range(n - 1):
        w = dmin if dmin.sum() > 0 else torch.ones_like(dmin)
        i = torch.multinomial(w / w.sum(), 1, generator=gen)
        pick.append(x[i])
        dmin = torch.minimum(dmin, (x - x[i]).pow(2).sum(-1))
    return torch.cat(pick)


# ============================================================================ decoder
class DecoderV4(nn.Module):
    def __init__(self, dcfg: DecoderConfig, de, reads=(0, 2)):
        super().__init__()
        self.cfg = dcfg
        self.plan = PlanBuilder(dcfg)
        self.layers = nn.ModuleList([ConfinedLayer(dcfg) for _ in range(dcfg.n_layers)])
        self.reads_after = tuple(reads)
        self.eng = nn.ModuleList([EngramRead2(dcfg.d_model, de, dcfg.gr_rank, dcfg.gr_read_bias, n_write=1)
                                  for _ in self.reads_after])
        self.out_norm = ZCRMSNorm(dcfg.d_model)

    def forward(self, tok, u, c_pred, doc, E, e, core_off=False, use_u=True, use_c=True, reads_off=()):
        c = self.cfg
        B, T, _ = tok.shape
        P = self.plan(u, c_pred, doc, core_off, use_u, use_c)
        swa, _ = build_masks(doc, c.window)
        _sv3 = sys.modules.get("speed_v3")                               # block-local path needs doc (speed_v3._CTX)
        if _sv3 is not None:
            _sv3._CTX["doc"] = doc
        rope = rope_cache(T, c.rope_dims, c.rope_base, tok.device, tok.dtype)
        D = tok + P
        gates = []
        for li, layer in enumerate(self.layers):
            D = layer(D, P, tok, swa, rope)
            if li in self.reads_after:
                ri = self.reads_after.index(li)
                if ri not in reads_off:
                    rd = self.eng[ri]
                    h, a = rd.gate(D, e)
                    D = D + rd.write(h) * F.silu(a * rd.Wv(e))
                    gates.append(a.detach())
        h = self.out_norm(D)
        return h @ E.t(), dict(h=h, P=P, dec_gates=gates)


# ============================================================================ model
class HRMThinkerV4(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = dict(c)
        self.k = c["k"]
        self.ecfg = EncoderConfig(vocab_size=c["V"], d_model=c["d"], n_heads=c["heads"], n_kv_heads=c["kv"],
                                  head_dim=c["head_dim"], d_ff=c["enc_ff"], chunk=c["k"], d_concept=c["d"],
                                  engram_rows=c["engram_rows"], engram_dim=c["engram_dim"], gr_rank=c["gr_rank"],
                                  rope_dims=c["rope"])
        self.enc = EncoderV4(self.ecfg, c)
        acfg = replace(self.ecfg, n_heads=c["heads_full"])
        self.core = ConceptCoreV4(acfg, c)
        self.dcfg = DecoderConfig(d_model=c["d"], n_heads=c["heads"], n_kv_heads=c["kv"], head_dim=c["head_dim"],
                                  d_ff=c["dec_ff"], n_layers=4, window=8, chunk=c["k"], layout="confined",
                                  plan_in="add", d_concept=c["d"], gr_rank=c["gr_rank"], rope_dims=c["rope"])
        self.dec = DecoderV4(self.dcfg, self.ecfg.n_tables * self.ecfg.engram_dim, reads=c.get("dec_reads", (0, 2)))


def build_model(size="tiny", **over):
    c = dict({"full": FULL, "tiny": TINY, "test": TEST}[size]); c.update(over)
    return HRMThinkerV4(c)


def chunk_masks(ids, doc, k):
    """nonempty: >= 1 real token; complete: k real tokens; pair_ok (j -> j+1): j nonempty, j+1 complete, same doc."""
    B, T = ids.shape
    real = (ids != PAD).view(B, T // k, k)
    nonempty, complete = real.any(-1), real.all(-1)
    cdoc = doc[:, ::k]
    pair_ok = nonempty[:, :-1] & complete[:, 1:] & (cdoc[:, 1:] == cdoc[:, :-1])
    return nonempty, complete, pair_ok


def target_mask(ids, doc):
    return (doc[:, 1:] == doc[:, :-1]) & (ids[:, 1:] != PAD)


def forward(m, ids, doc, eidx, core_off=False, plan_swap=False, use_u=True, use_c=True, M=None, enc_loop=None,
            engram_off=False, enc_shared_off=False, core_shared_off=False, pool_ffn_off=False, dec_reads_off=(),
            enc_engram_off=False):
    """Evaluation path. Returns logits (B, T, V) and aux."""
    k = m.k
    tok, c_hat, e, eaux = m.enc(ids, doc, eidx, enc_loop=enc_loop, engram_off=engram_off or enc_engram_off,
                                shared_off=enc_shared_off,
                                pool_ffn_off=pool_ffn_off)
    cdoc = doc[:, ::k]
    S, states = m.core.run(c_hat, cdoc, M=M, shared_off=core_shared_off)
    u = m.core.out_norm(S)
    q, lc, c_pred = m.core.head(u)
    if plan_swap:
        u, c_pred = u.roll(1, dims=0), c_pred.roll(1, dims=0)
    e_dec = e if not engram_off else torch.zeros_like(e)
    logits, daux = m.dec(tok, u, c_pred, doc, m.enc.embed.weight, e_dec, core_off=core_off, use_u=use_u, use_c=use_c,
                         reads_off=dec_reads_off)
    return logits, dict(c_hat=c_hat, cdoc=cdoc, q=q, logits_c=lc, u=u, c_pred=c_pred, states=states, enc=eaux,
                        tok=tok, e=e, **daux)


LAM_C, LAM_F = 0.05, 0.1


def loss_terms(m, ids, doc, eidx, count_hits=True):
    """One batch, means over the batch's own counts. Returns (loss, dict of detached terms, aux)."""
    k = m.k
    logits, aux = forward(m, ids, doc, eidx)
    nonempty, complete, pair_ok = chunk_masks(ids, doc, k)
    vqc, vqf, n0, n = m.core.vq_terms(aux["c_hat"], complete, count_hits)
    cec, cef = m.core.ncp_terms(aux["q"], aux["logits_c"], n0, n, pair_ok)
    tm = target_mask(ids, doc)
    tgt = ids[:, 1:].masked_fill(~tm, -100)
    ntp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100,
                          reduction="sum")
    nt, npr, nc = tm.sum().clamp_min(1), pair_ok.sum().clamp_min(1), complete.sum().clamp_min(1)
    terms = dict(ntp=ntp / nt, ce_coarse=cec / npr, ce_fine=cef / npr, vq_coarse=vqc / nc, vq_fine=vqf / nc)
    loss = terms["ntp"] + LAM_C * terms["ce_coarse"] + LAM_F * terms["ce_fine"] + terms["vq_coarse"] + terms["vq_fine"]
    aux.update(complete=complete, nonempty=nonempty, pair_ok=pair_ok, n0=n0, n=n)
    return loss, {kk: float(v.detach()) for kk, v in terms.items()}, aux


# ============================================================================ accounting
def param_table(m):
    """Repository convention: compute = encoder (no embedding, no Engram table) + core (no codebooks) + decoder."""
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    table, emb = m.enc.engram.table.weight.numel(), m.enc.embed.weight.numel()
    code = m.core.E0.numel() + m.core.e.numel()
    enc, core, dec = n(m.enc) - table - emb, n(m.core) - code, n(m.dec)
    e, c = m.enc, m.core
    parts = dict(enc_layers=sum(n(l) for l in e.layers),
                 enc_shared=(n(e.shared) + e.shared_gain.numel()) if e.shared is not None else 0,
                 engram_reads=n(e.engram) - table + n(e.engram2), router_pooler_norm=n(e.crc_router) + n(e.pooler) + n(e.tok_norm),
                 core_layers=sum(n(l) for l in c.layers), core_shared=n(c.shared) if c.shared is not None else 0,
                 head=n(c.Wq) + c.log_tau.numel() + 1 + n(c.out_norm),
                 dec_layers_norm=sum(n(l) for l in m.dec.layers) + n(m.dec.out_norm), plan=n(m.dec.plan),
                 dec_engram=n(m.dec.eng))
    assert sum(parts.values()) == enc + core + dec, (parts, enc, core, dec)
    assert enc + core + dec + table + emb + code == sum(p.numel() for p in m.parameters())
    return dict(encoder=enc, core=core, decoder=dec, compute=enc + core + dec, pooler=n(e.pooler), engram_table=table,
                embedding=emb, codebooks=code, parts=parts)


NO_DECAY_KEYS = ("embed.", "engram.table", "log_tau", "core.E0", "core.e", "plan.null", "plan.offset", "pooler.p",
                 "pooler.b", "pooler.g", "pooler.bg", "pooler.w", "shared_gain")


def param_groups(model, weight_decay=0.1):
    decay, no_decay = [], []
    for nme, p in model.named_parameters():
        if nme.endswith(".bias") or nme.endswith(".w") or any(s in nme for s in NO_DECAY_KEYS):
            no_decay.append(p)
        else:
            decay.append(p)
    return [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]


if __name__ == "__main__":
    for kk, want in ((6, 19898011), (8, 19899043), (4, None)):
        mm = build_model("full", engram_rows=11, k=kk)
        t = param_table(mm)
        print(f"k={kk} compute {t['compute']:,} (enc {t['encoder']:,} core {t['core']:,} dec {t['decoder']:,}) "
              f"pooler {t['pooler']:,}", t["parts"])
        if want:
            assert t["compute"] == want, (t["compute"], want)
    print("full-size ledger OK")
