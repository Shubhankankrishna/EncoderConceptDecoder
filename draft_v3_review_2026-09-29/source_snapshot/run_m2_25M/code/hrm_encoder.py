"""HRM THINKER - Encoder v1.0 (frozen).  Reference implementation.

Two role streams:
  L  (local)  : bounded reach (63 tokens at window 8). Only stream the decoder may read.
  G  (global) : includes FULL-attention writes. Read only by the CRC -> concept core.
Layer pattern "SSSFSSSF", Engram after layer index 0 (S1).

Inputs  (all int64, shape (B, T), T % chunk == 0):
  ids        token ids
  doc        document id per position (packing); positions of different docs never interact
  eng_idx    (B, T, n_tables) Engram row indices, precomputed with `engram_indices` (CPU is fine)
Outputs:
  tok        (B, T, d)        normalised L stream  -> decoder token path
  c_hat      (B, M, d_c)      concepts on the sphere radius sqrt(d_c), M = T / chunk -> concept core
  aux        dict(L, G, mix)  for tests / telemetry
"""
from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    vocab_size: int = 8192
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    d_ff: int = 1408
    chunk: int = 4
    d_concept: int = 0            # 0 -> d_model
    window: int = 8               # SWA keys t-window+1 .. t
    rope_dims: int = 16           # partial RoPE on the first rope_dims dims of each head (SWA only)
    rope_base: float = 10000.0
    layer_pattern: str = "SSSFSSSF"
    engram_after: int = 0         # Engram inserted after this layer index
    engram_orders: tuple = (2, 3)
    engram_heads: int = 4
    engram_rows: int = 131071     # prime, coprime with vocab (2^13)
    engram_dim: int = 48
    conv_kernel: int = 4
    gr_rank: int = 32
    gr_read_bias: float = 2.0     # sigmoid(2) = 0.881
    crc_bias: tuple = (-1.0, -1.0, 1.0)   # e0 / L / G -> 0.107 / 0.107 / 0.787
    eps: float = 1e-6

    @property
    def dc(self):
        return self.d_concept or self.d_model

    @property
    def n_tables(self):
        return len(self.engram_orders) * self.engram_heads

    def reach(self):
        """Exact reach of the L stream (tokens back), derived in the spec section 'Reach'."""
        per_swa = (self.window - 1) + (self.conv_kernel - 1)
        n_swa = self.layer_pattern.count("S")
        s_before = self.layer_pattern[: self.engram_after + 1].count("S")
        before = s_before * per_swa                               # reach entering Engram's gate
        engram = max(before, max(self.engram_orders) - 1) + (self.conv_kernel - 1)
        return engram + (n_swa - s_before) * per_swa


# ----------------------------------------------------------------------------- helpers
def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class ZCRMSNorm(nn.Module):
    """y = (1 + gamma) * x / rms(x); gamma initialised to 0 and weight-decayed."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        return rms(x, self.eps) * (1 + self.gamma)


def same_doc_shift_mask(doc, i):
    """(B, T, 1) bool: position t and t-i are in the same document (False for t < i)."""
    T = doc.shape[1]
    prev = F.pad(doc, (i, 0), value=-1)[:, :T]
    return (prev == doc).unsqueeze(-1)


class ShortConv(nn.Module):
    """Causal depthwise conv, kernel K, dilation 1, residual form:
         y_t = x_t + sum_{i=0}^{K-1} w_i * x_{t-i}       (w initialised to 0 -> identity)
       Document-aware: x_{t-i} from another document contributes 0."""
    def __init__(self, channels, kernel=4, zero_init=True):
        super().__init__()
        self.kernel = kernel
        w = torch.zeros(channels, kernel) if zero_init else torch.randn(channels, kernel) * 0.3
        self.w = nn.Parameter(w)

    def forward(self, x, doc):
        T = x.shape[1]
        y = x * (1 + self.w[:, 0])
        for i in range(1, self.kernel):
            xi = F.pad(x, (0, 0, i, 0))[:, :T]
            y = y + xi * same_doc_shift_mask(doc, i) * self.w[:, i]
        return y


class GRRead(nn.Module):
    """Gated-residual read over the allowed streams:
         n_i = rms(R_i),  g_i = sigmoid(U_i V_i n_i + b_i),  x = mean_i g_i * (1+gamma_i) * n_i"""
    def __init__(self, d, n_streams, rank, bias):
        super().__init__()
        self.V = nn.ModuleList([nn.Linear(d, rank, bias=False) for _ in range(n_streams)])
        self.U = nn.ModuleList([nn.Linear(rank, d) for _ in range(n_streams)])
        self.gamma = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in range(n_streams)])
        for u in self.U:
            nn.init.zeros_(u.weight)
            nn.init.constant_(u.bias, bias)

    def forward(self, streams):
        out = 0
        for i, R in enumerate(streams):
            n = rms(R)
            out = out + torch.sigmoid(self.U[i](self.V[i](n))) * n * (1 + self.gamma[i])
        return out / len(streams)


class GRWrite(nn.Module):
    """Per-stream write scale s_i = 2 * sigmoid(u_i . x + a_i); u, a initialised to 0 -> s_i = 1."""
    def __init__(self, d, n_streams):
        super().__init__()
        self.u = nn.Linear(d, n_streams)
        nn.init.zeros_(self.u.weight)
        nn.init.zeros_(self.u.bias)

    def forward(self, x):
        return 2 * torch.sigmoid(self.u(x))     # (B, T, n_streams)


def rope_cache(T, rope_dims, base, device, dtype):
    half = rope_dims // 2
    inv = 1.0 / (base ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    ang = torch.arange(T, device=device, dtype=torch.float32)[:, None] * inv[None]
    return ang.cos().to(dtype), ang.sin().to(dtype)


def apply_partial_rope(x, cos, sin, rope_dims):
    half = rope_dims // 2
    a, b, rest = x[..., :half], x[..., half:rope_dims], x[..., rope_dims:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos, rest], dim=-1)


class GatedAttention(nn.Module):
    """kind 'S': sliding window, partial RoPE.  kind 'F': full causal, NoPE, per-head temperature."""
    def __init__(self, cfg: EncoderConfig, kind: str):
        super().__init__()
        c = cfg
        self.cfg, self.kind = c, kind
        H, Hk, dh, d = c.n_heads, c.n_kv_heads, c.head_dim, c.d_model
        self.q = nn.Linear(d, H * dh, bias=False)
        self.k = nn.Linear(d, Hk * dh, bias=False)
        self.v = nn.Linear(d, Hk * dh, bias=False)
        self.o = nn.Linear(H * dh, d, bias=False)
        self.gate = nn.Linear(d, H * dh, bias=False)
        self.conv_q = ShortConv(H * dh, c.conv_kernel)
        self.conv_k = ShortConv(Hk * dh, c.conv_kernel)
        self.conv_v = ShortConv(Hk * dh, c.conv_kernel)
        self.q_norm = ZCRMSNorm(dh)
        self.k_norm = ZCRMSNorm(dh)
        self.log_tau = nn.Parameter(torch.zeros(H)) if kind == "F" else None

    def forward(self, x, doc, mask, rope):
        B, T, _ = x.shape
        c = self.cfg
        H, Hk, dh = c.n_heads, c.n_kv_heads, c.head_dim
        q = self.conv_q(self.q(x), doc).view(B, T, H, dh).transpose(1, 2)
        k = self.conv_k(self.k(x), doc).view(B, T, Hk, dh).transpose(1, 2)
        v = self.conv_v(self.v(x), doc).view(B, T, Hk, dh).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.kind == "S":
            cos, sin = rope
            q = apply_partial_rope(q, cos, sin, c.rope_dims)
            k = apply_partial_rope(k, cos, sin, c.rope_dims)
        else:
            q = q * self.log_tau.exp().view(1, H, 1, 1)          # logits * tau_h
        rep = H // Hk
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)   # mask: True = attend
        y = y.transpose(1, 2).reshape(B, T, H * dh)
        return self.o(y * torch.sigmoid(self.gate(x)))


class SwiGLU(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.w1 = nn.Linear(d, f, bias=False)
        self.w3 = nn.Linear(d, f, bias=False)
        self.w2 = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class EncoderLayer(nn.Module):
    """'S': reads {L}, writes {L, G}.  'F': reads {L, G}, writes {G} (attention AND FFN)."""
    def __init__(self, cfg: EncoderConfig, kind: str):
        super().__init__()
        self.kind = kind
        n_read, n_write = (1, 2) if kind == "S" else (2, 1)
        d = cfg.d_model
        self.read_attn = GRRead(d, n_read, cfg.gr_rank, cfg.gr_read_bias)
        self.write_attn = GRWrite(d, n_write)
        self.read_ffn = GRRead(d, n_read, cfg.gr_rank, cfg.gr_read_bias)
        self.write_ffn = GRWrite(d, n_write)
        self.attn = GatedAttention(cfg, kind)
        self.ffn = SwiGLU(d, cfg.d_ff)

    def _apply(self, L, G, y, s):
        if self.kind == "S":
            return L + s[..., 0:1] * y, G + s[..., 1:2] * y
        return L, G + s[..., 0:1] * y

    def forward(self, L, G, doc, mask, rope):
        x = self.read_attn([L] if self.kind == "S" else [L, G])
        L, G = self._apply(L, G, self.attn(x, doc, mask, rope), self.write_attn(x))
        x = self.read_ffn([L] if self.kind == "S" else [L, G])
        L, G = self._apply(L, G, self.ffn(x), self.write_ffn(x))
        return L, G


# ----------------------------------------------------------------------------- Engram
_MASK61 = (1 << 61) - 1


def engram_indices(canon_ids, doc, cfg: EncoderConfig):
    """(B, T) canonical ids + doc ids -> (B, T, n_tables) int64 row indices (already offset per table).
    Order-n key at t = (x_{t-n+1}, ..., x_t); a position in another document is replaced by boundary id 0,
    real ids are shifted by +1. Hash: multiplicative-XOR, masked to 61 bits, then mod rows (prime)."""
    B, T = canon_ids.shape
    grams = []
    for i in range(max(cfg.engram_orders)):
        prev = F.pad(canon_ids, (i, 0))[:, :T] + 1
        ok = same_doc_shift_mask(doc, i).squeeze(-1)
        grams.append(torch.where(ok, prev, torch.zeros_like(prev)))
    out, table = [], 0
    for n in cfg.engram_orders:
        for h in range(cfg.engram_heads):
            x = torch.full_like(canon_ids, 1469598103934665603 & _MASK61)
            for i in range(n):
                mult = (0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019 * (table * 8 + i + 1)) & _MASK61 | 1
                x = ((x ^ grams[i]) * mult) & _MASK61
            out.append(torch.remainder(x, cfg.engram_rows) + table * cfg.engram_rows)
            table += 1
    return torch.stack(out, dim=-1)


class Engram(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        c, d = cfg, cfg.d_model
        self.table = nn.Embedding(c.n_tables * c.engram_rows, c.engram_dim)
        nn.init.normal_(self.table.weight, std=0.02)           # small random, NOT zero
        de = c.n_tables * c.engram_dim
        self.Wk = nn.Linear(de, d, bias=False)
        self.Wv = nn.Linear(de, d, bias=False)
        nn.init.zeros_(self.Wv.weight)                          # output exactly 0 at init
        self.read = GRRead(d, 1, c.gr_rank, c.gr_read_bias)
        self.conv = ShortConv(d, c.conv_kernel, zero_init=False)
        self.write = GRWrite(d, 2)
        self.d = d

    def forward(self, L, G, eng_idx, doc):
        B, T, _ = eng_idx.shape
        e = self.table(eng_idx).reshape(B, T, -1)              # (B, T, n_tables*engram_dim)
        h = self.read([L])
        k, v = self.Wk(e), self.Wv(e)
        alpha = torch.sigmoid((rms(h) * rms(k)).sum(-1, keepdim=True) / math.sqrt(self.d))
        y = F.silu(self.conv(alpha * v, doc))
        s = self.write(h)
        return L + s[..., 0:1] * y, G + s[..., 1:2] * y, alpha


# ----------------------------------------------------------------------------- masks
def build_masks(doc, window):
    """Boolean (B, 1, T, T) masks, True = may attend.  SWA: causal & same doc & t-j < window.  FULL: causal & same doc."""
    T = doc.shape[1]
    i = torch.arange(T, device=doc.device)
    causal = i[None, :] <= i[:, None]
    same = doc[:, :, None] == doc[:, None, :]
    full = (causal[None] & same).unsqueeze(1)
    swa = full & ((i[:, None] - i[None, :]) < window)[None, None]
    return swa, full


# ----------------------------------------------------------------------------- encoder
class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig = EncoderConfig()):
        super().__init__()
        c = cfg
        assert c.n_heads % c.n_kv_heads == 0 and c.rope_dims % 2 == 0 and c.rope_dims <= c.head_dim
        assert set(c.layer_pattern) <= {"S", "F"} and c.layer_pattern[c.engram_after] == "S"
        self.cfg = c
        d, dc, k = c.d_model, c.dc, c.chunk
        self.embed = nn.Embedding(c.vocab_size, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.layers = nn.ModuleList([EncoderLayer(c, kind) for kind in c.layer_pattern])
        self.engram = Engram(c)
        self.crc_router = nn.Linear(d, 3)
        nn.init.zeros_(self.crc_router.weight)
        with torch.no_grad():
            self.crc_router.bias.copy_(torch.tensor(c.crc_bias))
        self.pool = nn.Linear(k * d, dc, bias=False)
        with torch.no_grad():
            if dc == d:
                P = torch.eye(d)
            elif dc > d:
                P = torch.linalg.qr(torch.randn(dc, d))[0]        # (dc, d), orthonormal columns
            else:
                P = torch.linalg.qr(torch.randn(d, dc))[0].T      # (dc, d), orthonormal rows
            self.pool.weight.copy_(torch.cat([P] * k, dim=1) / k)  # mean pooling (projected)
        self.tok_norm = ZCRMSNorm(d)

    def forward(self, ids, doc, eng_idx):
        c = self.cfg
        B, T = ids.shape
        assert T % c.chunk == 0, "pad rows so that T and every document length are multiples of chunk"
        swa_mask, full_mask = build_masks(doc, c.window)
        cos, sin = rope_cache(T, c.rope_dims, c.rope_base, ids.device, self.embed.weight.dtype)
        e0 = rms(self.embed(ids))
        L, G = e0, e0
        alpha = None
        for li, layer in enumerate(self.layers):
            mask = swa_mask if layer.kind == "S" else full_mask
            L, G = layer(L, G, doc, mask, (cos, sin))
            if li == c.engram_after:
                L, G, alpha = self.engram(L, G, eng_idx, doc)
        mix = torch.softmax(self.crc_router(rms(G)), dim=-1)                 # (B, T, 3)
        m = mix[..., 0:1] * e0 + mix[..., 1:2] * rms(L) + mix[..., 2:3] * rms(G)
        M = T // c.chunk
        cvec = self.pool(m.reshape(B, M, c.chunk * c.d_model))               # (B, M, dc)
        c_hat = math.sqrt(c.dc) * cvec / (cvec.norm(dim=-1, keepdim=True) + c.eps)
        return self.tok_norm(L), c_hat, dict(L=L, G=G, mix=mix, engram_alpha=alpha)


def param_groups(model, weight_decay=0.1):
    """No weight decay: Engram table, FULL temperatures, token embedding, conv kernels, biases.
    Weight decay: all matrices and the zero-centred norm gammas (pulls effective scale to 1)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if ("engram.table" in n or "log_tau" in n or n.startswith("embed.") or n.endswith("conv_q.w")
                or n.endswith("conv_k.w") or n.endswith("conv_v.w") or n.endswith("conv.w") or n.endswith(".bias")):
            no_decay.append(p)
        else:
            decay.append(p)
    return [dict(params=decay, weight_decay=weight_decay), dict(params=no_decay, weight_decay=0.0)]


# ----------------------------------------------------------------------------- unit tests
def _randomize(model, scale=0.2):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "engram.table" not in n:
                p.add_(torch.randn_like(p) * scale)


def run_unit_tests(verbose=True):
    torch.manual_seed(0)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    cfg = EncoderConfig(vocab_size=64, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=128,
                        rope_dims=8, engram_rows=1021, engram_dim=8)
    enc = Encoder(cfg)
    _randomize(enc)
    enc.eval()
    T, p = 256, 100
    ids = torch.randint(0, 64, (1, T))
    doc = torch.zeros(1, T, dtype=torch.long)
    ids2 = ids.clone()
    ids2[0, p] = (ids[0, p] + 7) % 64
    with torch.no_grad():
        _, c1, a = enc(ids, doc, engram_indices(ids, doc, cfg))
        _, c2, b = enc(ids2, doc, engram_indices(ids2, doc, cfg))
    dL = (a["L"] - b["L"]).abs().amax(-1)[0]
    dG = (a["G"] - b["G"]).abs().amax(-1)[0]
    R = cfg.reach()
    got = int(torch.nonzero(dL != 0).max()) - p
    assert R == 63 and got == R, (R, got)
    assert torch.equal(a["L"][0, p + R + 1:], b["L"][0, p + R + 1:])
    assert dG[-1] != 0
    assert not (dL[:p] != 0).any() and not (dG[:p] != 0).any()
    first_c = int(torch.nonzero((c1 - c2).abs().amax(-1)[0] != 0).min())
    assert first_c == p // cfg.chunk
    # document isolation
    doc2 = torch.zeros(1, T, dtype=torch.long)
    doc2[0, 128:] = 1
    ids3 = ids.clone()
    ids3[0, 126] = (ids[0, 126] + 5) % 64
    with torch.no_grad():
        _, c3, y1 = enc(ids, doc2, engram_indices(ids, doc2, cfg))
        _, c4, y2 = enc(ids3, doc2, engram_indices(ids3, doc2, cfg))
    assert torch.equal(y1["L"][0, 128:], y2["L"][0, 128:]) and torch.equal(y1["G"][0, 128:], y2["G"][0, 128:])
    assert torch.equal(c3[0, 32:], c4[0, 32:])
    torch.set_default_dtype(old)
    # init values at the real size
    cfg = EncoderConfig()
    enc = Encoder(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 128))
    doc = torch.zeros(2, 128, dtype=torch.long)
    tok, c_hat, aux = enc(ids, doc, engram_indices(ids, doc, cfg))
    mix = aux["mix"].mean((0, 1)).tolist()
    assert abs(mix[2] - 0.787) < 0.01 and abs(mix[0] - 0.107) < 0.01
    assert torch.allclose(c_hat.norm(dim=-1), torch.full_like(c_hat[..., 0], math.sqrt(cfg.dc)), rtol=1e-4)
    assert tok.shape == (2, 128, 512) and c_hat.shape == (2, 32, 512)
    (tok.pow(2).mean() + c_hat.sum() * 1e-3).backward()
    assert enc.engram.Wv.weight.grad.abs().sum() > 0
    n_non = sum(p.numel() for n, p in enc.named_parameters() if not n.startswith("embed.") and "engram.table" not in n)
    n_tab = enc.engram.table.weight.numel()
    if verbose:
        print(f"unit tests PASS | reach(L)={R} | mix at init={[round(v, 3) for v in mix]}")
        print(f"params: non-embedding {n_non/1e6:.2f}M, Engram table {n_tab/1e6:.2f}M, embedding {enc.embed.weight.numel()/1e6:.2f}M")
    return True


def mps_smoke_test(B=2, T=1024):
    """Run on the MacBook: CPU vs MPS forward agreement, one backward pass, memory report."""
    if not torch.backends.mps.is_available():
        print("MPS not available on this machine; skipping")
        return
    torch.manual_seed(0)
    cfg = EncoderConfig()
    enc = Encoder(cfg)
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    doc = torch.zeros(B, T, dtype=torch.long)
    doc[:, T // 2:] = 1                                   # two documents per row
    eidx = engram_indices(ids, doc, cfg)                  # computed on CPU
    with torch.no_grad():
        tok_c, ch_c, _ = enc(ids, doc, eidx)
    enc = enc.to("mps")
    dev = lambda x: x.to("mps")
    tok_m, ch_m, _ = enc(dev(ids), dev(doc), dev(eidx))
    d_tok = (tok_m.detach().cpu() - tok_c).abs().max().item()
    d_ch = (ch_m.detach().cpu() - ch_c).abs().max().item()
    (tok_m.pow(2).mean() + ch_m.sum() * 1e-3).backward()
    torch.mps.synchronize()
    print(f"MPS vs CPU max abs diff: tok {d_tok:.2e}, c_hat {d_ch:.2e} (expect < 1e-3)")
    print(f"MPS allocated after backward: {torch.mps.current_allocated_memory() / 2**30:.2f} GiB (B={B}, T={T})")


if __name__ == "__main__":
    import sys
    run_unit_tests()
    if "--mps" in sys.argv:
        mps_smoke_test()
