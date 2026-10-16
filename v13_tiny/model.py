"""V13-tiny-v2: the V13 architecture of v13_NOTES.md at d = 128, sized for throughput on the M2 (MPS, fp32).

    E1  2 x [S, S, S, F], Engram after the first S. S: sliding window w_E = 8, reads L, writes L and G.
        F: full attention (NoPE, temperature), reads L and G, writes G. Within-role 2x2 mix after every layer.
    E2  [M, M, M, F] looped R = 2 over a workspace W (s0, per-loop embedding), M = Gated DeltaNet-2 (chunkwise, exact
        document resets); every sublayer reads {L, G, W} and writes W only; exit G_b += s_b * W.
    E3  [S', S', S', F] reads {L, G}, writes G only; mix within G.
    exit L_enc = GRread(L1, L2), G_enc = GRread(G1, G2).
    pool per-channel softmax over the 4 tokens of each group, three sources (x, sg L_enc, G_enc), beta-mixed -> m_g,
        p_g = W_P m_g.  NCP target: sg[m_{g+1}] quantised by the EMA PQ codebook (losses in train.py).
    C   n_C = 4 layers [self-attention over concepts <= g] -> [cross-attention to sg L_enc, 16-token window ending at
        4g+3] -> [FFN], Delta-AttnRes over every sublayer with p_g as source 0. Output u_g (no final norm).
    D   N_D = 6 layers [local attention: query from the decoder state, keys/values from L_enc, w_D = 16] ->
        [concept read: concepts j <= floor((t+1)/4) - 1 of the same document, null entry, sink logit] -> [FFN],
        Delta-AttnRes with L_enc as source 0; tied output head.
Reach: R_L = 6 (w_E - 1) + 2 + 9 = 53, R_D = R_L + w_D - 1 = 68 (tested in tests_g0.py).
Packed rows: every mixer is masked or reset at document boundaries (attention masks, convolution cuts, Engram boundary
ids, GDN-2 state reset, concept and window masks); documents start at multiples of 4 (data.py).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt


class Cfg:
    V = 2048; d = 128; H = 4; ffn = 256; k = 4
    wE = 8; wE3 = 16; wD = 16; wC = 16
    n_e1_blocks = 2; R = 2; n_c = 4; n_d = 6
    chunk = 16; rope_frac_S = 0.25
    eng_heads = 8; eng_dim = 8; eng_orders = (2, 3)
    pq_S = 4; pq_N = 32

    def __init__(self, **kw):
        for a, b in kw.items():
            assert hasattr(self, a), a; setattr(self, a, b)


PRIMES = [8191, 8179, 8171, 8167, 8161, 8147, 8123, 8117, 8111, 8101, 8093, 8089, 8087, 8081, 8069, 8059]


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class Norm(nn.Module):
    """Zero-centred RMSNorm: weight 1 + gamma, gamma initialised to 0 (decays toward the plain norm)."""
    def __init__(self, *shape):
        super().__init__(); self.g = nn.Parameter(torch.zeros(*shape))

    def forward(self, x):
        return rms(x) * (1 + self.g)


def rope_tables(pos, dim, base=10000.0):
    """cos, sin for integer positions `pos` (any shape) over `dim` rotated dimensions."""
    dt = torch.float32 if pos.device.type == 'mps' else torch.float64
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=pos.device, dtype=dt) / dim))
    ang = pos.to(dt)[..., None] * inv
    return ang.cos(), ang.sin()


def rope(x, cos, sin, rd):
    """Rotate the first rd dims of the last axis (pairs (i, i + rd/2)); cos/sin broadcast to x[..., :rd/2]."""
    if rd == 0: return x
    h = rd // 2
    x1, x2, rest = x[..., :h], x[..., h:rd], x[..., rd:]
    c, s = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c, rest], -1)


def lin(i, o, bias=False, std=0.02):
    m = nn.Linear(i, o, bias=bias); nn.init.normal_(m.weight, std=std)
    if bias: nn.init.zeros_(m.bias)
    return m


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.w13 = lin(d, 2 * h); self.w2 = lin(h, d)

    def forward(self, x):
        a, b = self.w13(x).chunk(2, -1); return self.w2(F.silu(a) * b)


class QKNorm(nn.Module):
    """RMSNorm on every query and key head, and a learned per-head temperature on the query (§3.3).
    key_gain=False drops the key's elementwise gain (the query gain and temperature carry the same freedom); used on
    the attentions over concepts, where the compiled backward of that gain produced non-finite values on MPS."""
    def __init__(self, H, dh, key_gain=True):
        super().__init__(); self.qn = Norm(dh); self.kn = Norm(dh) if key_gain else None
        self.log_tau = nn.Parameter(torch.zeros(H, 1))

    def q(self, q):                                               # (..., H, dh) with H at dim -2
        return self.qn(q) * self.log_tau.exp()

    def k(self, k):
        return self.kn(k) if self.kn is not None else rms(k)


# ------------------------------------------------------------------------------------------------ attention mixers
def banded(q, k, v, doc_q, doc_k, W, scale):
    """Causal window attention, window W (keys t-W+1..t), via blocks of W: each query block sees its own and the
    previous key block. q, k, v: (B, T, H, dh); doc_q, doc_k: (B, T)."""
    B, T, H, dh = q.shape; nb = T // W
    qb = q.view(B, nb, W, H, dh).permute(0, 3, 1, 2, 4)
    kb = k.view(B, nb, W, H, dh).permute(0, 3, 1, 2, 4); vb = v.view(B, nb, W, H, dh).permute(0, 3, 1, 2, 4)
    kp = torch.cat([F.pad(kb, (0, 0, 0, 0, 1, 0))[:, :, :-1], kb], 3)                    # (B, H, nb, 2W, dh)
    vp = torch.cat([F.pad(vb, (0, 0, 0, 0, 1, 0))[:, :, :-1], vb], 3)
    s = (qb @ kp.transpose(-1, -2)) * scale
    i = torch.arange(W, device=q.device)[:, None] + W; j = torch.arange(2 * W, device=q.device)[None]
    band = (j <= i) & (i - j < W)
    dq = doc_q.view(B, nb, W); dk = doc_k.view(B, nb, W)
    dkp = torch.cat([F.pad(dk, (0, 0, 1, 0), value=-1)[:, :-1], dk], 2)
    m = band[None, None] & (dq[:, :, :, None] == dkp[:, :, None, :])                    # (B, nb, W, 2W)
    s = s.masked_fill(~m[:, None], float('-inf'))
    return (s.softmax(-1) @ vp).permute(0, 2, 3, 1, 4).reshape(B, T, H * dh)


class LocalAttn(nn.Module):
    """Sliding-window attention; queries from xq, keys/values from xkv (the decoder passes L_enc as xkv)."""
    def __init__(s, d, H, W, rd):
        super().__init__(); s.H, s.dh, s.W, s.rd = H, d // H, W, rd
        s.wq = lin(d, d); s.wkv = lin(d, 2 * d); s.wg = lin(d, d); s.wo = lin(d, d); s.qk = QKNorm(H, d // H)

    def forward(s, xq, xkv, ctx):
        B, T, _ = xq.shape
        q = s.qk.q(s.wq(xq).view(B, T, s.H, s.dh))
        k, v = s.wkv(xkv).view(B, T, 2, s.H, s.dh).unbind(2); k = s.qk.k(k)
        cos, sin = ctx['rope_tok'] if s.rd == s.dh else ctx['rope_tok_part']
        q = rope(q, cos[:, None], sin[:, None], s.rd); k = rope(k, cos[:, None], sin[:, None], s.rd)
        y = banded(q, k, v, ctx['doc'], ctx['doc'], s.W, 1 / math.sqrt(s.dh))
        return s.wo(y * torch.sigmoid(s.wg(xq)))


class FullAttn(nn.Module):
    """Token-level full attention, NoPE, QK norm and temperature, document-causal mask, SDPA."""
    def __init__(s, d, H):
        super().__init__(); s.H, s.dh = H, d // H
        s.wqkv = lin(d, 3 * d); s.wg = lin(d, d); s.wo = lin(d, d); s.qk = QKNorm(H, d // H)

    def forward(s, x, ctx):
        B, T, _ = x.shape
        q, k, v = s.wqkv(x).view(B, T, 3, s.H, s.dh).unbind(2)
        q, k = s.qk.q(q), s.qk.k(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           attn_mask=ctx['full_mask'])
        return s.wo(y.transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(s.wg(x)))


def shortconv(x, w, doc, dil=1):
    """Causal depthwise convolution, taps t, t-dil, ..., cut at document boundaries. x (B, T, C), w (K, C)."""
    out = x * w[0]
    for j in range(1, w.shape[0]):
        sh = j * dil
        xs = F.pad(x, (0, 0, sh, 0))[:, :-sh]
        same = (F.pad(doc, (sh, 0), value=-1)[:, :-sh] == doc)[..., None]
        out = out + xs * same * w[j]
    return out


def unit_lower_inverse(A):
    """(I + A)^-1 for strictly lower-triangular A of size 16 (A^16 = 0): (I - A)(I + A^2)(I + A^4)(I + A^8)."""
    C = A.shape[-1]; I = torch.eye(C, device=A.device, dtype=A.dtype)
    out = I - A; P = A
    n = 2
    while n < C:
        P = P @ P; out = out @ (I + P); n *= 2
    return out


def gdn2_chunk(q, k, u, a, g, doc, C):
    """Exact chunkwise Gated DeltaNet-2 with per-channel decay and document resets.
    S_t = (I - k_t a_t^T) Diag(exp g_t) S_{t-1} + k_t u_t^T,  o_t = S_t^T q_t, with a = b * k, u = w * v.
    q, k, a, g: (B, T, H, dk); u: (B, T, H, dv); doc: (B, T); g >= -5 (so every factor below stays finite).
    Intra-chunk pair terms use the factorised decay e^{G_t} e^{-G_s} (G summed within the document only); the
    chunk-to-chunk recurrence S_{n+1} = M_n S_n + N_n runs as a log-depth scan. Returns o (B, T, H, dv)."""
    B, T, H, dk = q.shape; dv = u.shape[-1]; N = T // C
    tr = lambda x: x.view(B, N, C, H, x.shape[-1]).permute(0, 3, 1, 2, 4)
    q, k, u, a, g = map(tr, (q, k, u, a, g))
    dc = doc.view(B, N, C)
    prev = torch.cat([dc.new_full((B, 1), -1), dc[:, :-1, -1]], 1)
    cont = dc == prev[..., None]                                                         # (B, N, C)
    sd = dc[..., :, None] == dc[..., None, :]                                            # (B, N, C, C)
    tri = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    m_le = (sd & tri)[:, None]; m_lt = (sd & strict)[:, None]                           # (B, 1, N, C, C)
    Gc = torch.einsum('bnts,bhnsc->bhntc', (tri & sd).to(g.dtype), g)                   # decay sums within the document
    eG = Gc.exp(); kd = k * (-Gc).exp()
    kdT = kd.transpose(-1, -2)
    zero = q.new_zeros(())
    Bm = torch.where(m_le, (q * eG) @ kdT, zero)                                         # s <= t
    Am = torch.where(m_lt, (a * eG) @ kdT, zero)                                         # s < t
    Tinv = unit_lower_inverse(Am)
    cf = cont[:, None, ..., None].to(q.dtype)
    W1 = Tinv @ u; W2 = Tinv @ (a * eG * cf)
    lastm = sd[..., -1, :][:, None, ..., None].to(q.dtype)
    Khat = torch.exp(Gc[..., -1:, :] - Gc) * k * lastm                                   # (B, H, N, C, dk)
    endc = cont[..., -1][:, None, :, None, None].to(q.dtype)
    KhT = Khat.transpose(-1, -2)
    Mc = endc * torch.diag_embed(eG[..., -1, :]) - KhT @ W2                              # (B, H, N, dk, dk)
    Nc = KhT @ W1                                                                        # (B, H, N, dk, dv)
    I = torch.eye(dk, device=q.device, dtype=q.dtype).expand(B, H, 1, dk, dk)
    step = 1
    while step < N:                                                                      # inclusive scan of (M, N)
        Mp = torch.cat([I.expand(B, H, step, dk, dk), Mc[:, :, :-step]], 2)
        Np = torch.cat([Nc.new_zeros(B, H, step, dk, dv), Nc[:, :, :-step]], 2)
        Nc = Mc @ Np + Nc; Mc = Mc @ Mp
        step *= 2
    S0 = torch.cat([Nc.new_zeros(B, H, 1, dk, dv), Nc[:, :, :-1]], 2)                   # state entering each chunk
    Delta = W1 - W2 @ S0
    O = (q * eG * cf) @ S0 + Bm @ Delta
    return O.permute(0, 2, 3, 1, 4).reshape(B, T, H, dv)


def gdn2_reference(q, k, u, a, g, doc):
    """Token-by-token recurrence, for tests only."""
    B, T, H, dk = q.shape; dv = u.shape[-1]
    S = q.new_zeros(B, H, dk, dv); out = []
    for t in range(T):
        if t > 0:
            keep = (doc[:, t] == doc[:, t - 1]).to(q.dtype)[:, None, None, None]
            S = S * keep
        al = g[:, t].exp()                                                               # (B, H, dk)
        DS = al[..., None] * S
        S = DS - k[:, t][..., None] * (a[:, t][..., None] * DS).sum(-2, keepdim=True) + k[:, t][..., None] * u[:, t][..., None, :]
        out.append((S * q[:, t][..., None]).sum(-2))
    return torch.stack(out, 1)


class GDN2(nn.Module):
    def __init__(s, d, H, chunk):
        super().__init__(); s.H, s.dh, s.C = H, d // H, chunk
        s.wqkv = lin(d, 3 * d)
        cw = torch.zeros(4, 3 * d); cw[0] = 1.0; cw[1:] = 0.02 * torch.randn(3, 3 * d)
        s.conv = nn.Parameter(cw)
        s.wb = lin(d, d, bias=True); s.ww = lin(d, d, bias=True)
        s.wa1 = lin(d, 16); s.wa2 = lin(16, d, bias=True)
        y = torch.linspace(0.001, 0.1, s.dh).repeat(H)                                  # -log(alpha) spread per channel
        with torch.no_grad(): s.wa2.bias.copy_(torch.log(torch.expm1(y)))
        s.wg = lin(d, d); s.on = Norm(s.dh); s.wo = lin(d, d)

    def forward(s, x, ctx):
        B, T, _ = x.shape; H, dh = s.H, s.dh
        qkv = F.silu(shortconv(s.wqkv(x), s.conv, ctx['doc']))
        q, k, v = qkv.view(B, T, 3, H, dh).unbind(2)
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        b = torch.sigmoid(s.wb(x)).view(B, T, H, dh); w = torch.sigmoid(s.ww(x)).view(B, T, H, dh)
        g = (-F.softplus(s.wa2(s.wa1(x)))).clamp(min=-5.0).view(B, T, H, dh)          # alpha >= e^-5 per step
        o = gdn2_chunk(q, k, w * v, b * k, g, ctx['doc'], s.C)
        if ctx.get('log') is not None: ctx['log'].append(o.detach().pow(2).mean().sqrt())
        o = s.on(o).reshape(B, T, -1)
        return s.wo(torch.sigmoid(s.wg(x)) * o)


# ------------------------------------------------------------------------------------------------ residual operator
class GR(nn.Module):
    """Gated read over branches [r0, r1) and scalar writes into the branches in `writes` (§3.4)."""
    def __init__(s, d, nb, r0, r1, writes, wbias=None):
        super().__init__(); A = r1 - r0; s.r0, s.r1, s.A = r0, r1, A
        s.gam = nn.Parameter(torch.zeros(A, d))
        s.wd = lin(A * d, d // 8); s.wu = nn.Linear(d // 8, A * d)
        nn.init.zeros_(s.wu.weight); nn.init.constant_(s.wu.bias, 2.0)
        s.ww = None
        if writes:
            s.ww = nn.Linear(A * d, len(writes)); nn.init.zeros_(s.ww.weight)
            with torch.no_grad(): s.ww.bias.copy_(torch.tensor(wbias if wbias is not None else [0.0] * len(writes)))
        P = torch.zeros(len(writes), nb)
        for i, b in enumerate(writes): P[i, b] = 1.0
        s.register_buffer('P', P, persistent=False)

    def read(s, R):
        Rn = rms(R[:, :, s.r0:s.r1]) * (1 + s.gam)
        z = Rn.flatten(-2) / s.A
        gt = torch.sigmoid(s.wu(F.silu(s.wd(z)))).view(Rn.shape)
        return (gt * Rn).sum(-2) / s.A, (2 * torch.sigmoid(s.ww(z)) if s.ww is not None else None)

    def write(s, R, y, sc):
        return R + (sc @ s.P)[..., None] * y[:, :, None, :]


class WriteScale(nn.Module):
    """E2's exit: s = 2 sigmoid(W vec(RMSNorm branches) / A + b), one scalar per written branch (no read gate)."""
    def __init__(s, d, A, wbias):
        super().__init__(); s.A = A; s.gam = nn.Parameter(torch.zeros(A, d))
        s.ww = nn.Linear(A * d, len(wbias)); nn.init.zeros_(s.ww.weight)
        with torch.no_grad(): s.ww.bias.copy_(torch.tensor(wbias))

    def forward(s, R):
        return 2 * torch.sigmoid(s.ww((rms(R) * (1 + s.gam)).flatten(-2) / s.A))


class Mix(nn.Module):
    """Within-role 2x2 doubly stochastic mix [[a, 1-a], [1-a, a]], a = sigmoid(.) from the role's own branches."""
    def __init__(s, d, roles):
        super().__init__(); s.roles = roles
        s.w = nn.ModuleList()
        for _ in roles:
            m = nn.Linear(2 * d, 1); nn.init.zeros_(m.weight); nn.init.constant_(m.bias, 2.0); s.w.append(m)

    def forward(s, R):
        parts = []; last = 0
        for i0, m in zip(s.roles, s.w):
            parts.append(R[:, :, last:i0])
            r = R[:, :, i0:i0 + 2]
            a = torch.sigmoid(m(rms(r).flatten(-2) / 2))[..., None]                     # (B, T, 1, 1)
            parts.append(torch.cat([a * r[:, :, :1] + (1 - a) * r[:, :, 1:], (1 - a) * r[:, :, :1] + a * r[:, :, 1:]], 2))
            last = i0 + 2
        parts.append(R[:, :, last:])
        return torch.cat(parts, 2)


class EncLayer(nn.Module):
    """kind: S (local, reads L, writes L+G), F (full, reads L+G, writes G), S3 (local, reads L+G, writes G),
    M2 / F2 (E2: GDN-2 / full, reads L+G+W, writes W)."""
    def __init__(s, kind, c):
        super().__init__(); d = c.d; s.kind = kind
        if kind == 'S':   nb, r, wr, wb, roles = 4, (0, 2), [0, 1, 2, 3], [-.5, .5, -.5, .5], [0, 2]
        elif kind == 'F': nb, r, wr, wb, roles = 4, (0, 4), [2, 3], [-.5, .5], [0, 2]
        elif kind == 'S3': nb, r, wr, wb, roles = 4, (0, 4), [2, 3], [-.5, .5], [2]
        elif kind == 'F3': nb, r, wr, wb, roles = 4, (0, 4), [2, 3], [-.5, .5], [2]
        else:             nb, r, wr, wb, roles = 5, (0, 5), [4], [0.0], []
        if kind == 'S': s.att = LocalAttn(d, c.H, c.wE, int(c.rope_frac_S * d // c.H))
        elif kind == 'S3': s.att = LocalAttn(d, c.H, c.wE3, int(c.rope_frac_S * d // c.H))
        elif kind == 'M2': s.att = GDN2(d, c.H, c.chunk)
        else: s.att = FullAttn(d, c.H)
        s.gr1 = GR(d, nb, *r, wr, wb); s.ffn = SwiGLU(d, c.ffn); s.gr2 = GR(d, nb, *r, wr, wb)
        s.mix = Mix(d, roles) if roles else None

    def forward(s, R, ctx):
        x, sc = s.gr1.read(R)
        y = s.att(x, x, ctx) if isinstance(s.att, LocalAttn) else s.att(x, ctx)
        R = s.gr1.write(R, y, sc)
        x, sc = s.gr2.read(R)
        R = s.gr2.write(R, s.ffn(x), sc)
        return s.mix(R) if s.mix is not None else R


class Engram(nn.Module):
    """One table of 16 hashed n-gram heads (orders 2, 3 x 8), shared W_V, per-branch keys and gates, per-branch
    zero-initialised causal convolution (kernel 4, dilation 3), residual write into every branch (§3.5)."""
    def __init__(s, c, nb=4):
        super().__init__(); d = c.d; s.nb = nb; s.V = c.V
        nh = c.eng_heads * len(c.eng_orders); primes = PRIMES[:nh]
        s.orders = [o for o in c.eng_orders for _ in range(c.eng_heads)]
        s.table = nn.Embedding(sum(primes), c.eng_dim); nn.init.normal_(s.table.weight, std=0.02)
        off = torch.tensor([0] + list(torch.tensor(primes).cumsum(0)[:-1].tolist()))
        s.register_buffer('off', off, persistent=False)
        s.register_buffer('primes', torch.tensor(primes), persistent=False)
        gen = torch.Generator().manual_seed(1234)
        mult = torch.randint(1, 2 ** 20, (nh, 3), generator=gen) * 2 + 1
        s.register_buffer('mult', mult, persistent=False)
        dm = nh * c.eng_dim
        s.wv = lin(dm, d); s.wk = lin(dm, nb * d)
        s.hn = nn.Parameter(torch.zeros(nb, d)); s.kn = nn.Parameter(torch.zeros(nb, d))
        s.conv = nn.Parameter(torch.zeros(4, nb * d))

    @torch.no_grad()
    def keys(self, ids, doc):
        """Hashed row indices (B, T, 16); positions before the document start use the boundary id V."""
        bnd = self.V
        x0 = ids.long()
        x1 = torch.where(F.pad(doc, (1, 0), value=-1)[:, :-1] == doc, F.pad(ids, (1, 0))[:, :-1].long(), torch.full_like(x0, bnd))
        x2 = torch.where(F.pad(doc, (2, 0), value=-1)[:, :-2] == doc, F.pad(ids, (2, 0))[:, :-2].long(), torch.full_like(x0, bnd))
        m = self.mult
        h2 = (x0[..., None] * m[:, 0]) ^ (x1[..., None] * m[:, 1])
        h3 = h2 ^ (x2[..., None] * m[:, 2])
        o3 = torch.tensor([o == 3 for o in self.orders], device=ids.device)
        h = torch.where(o3, h3, h2)
        return h % self.primes + self.off

    def forward(s, R, eidx, ctx):
        B, T, nb, d = R.shape
        e = s.table(eidx).flatten(-2)
        v = s.wv(e); kk = s.wk(e).view(B, T, nb, d)
        alpha = torch.sigmoid((rms(R) * (1 + s.hn) * rms(kk) * (1 + s.kn)).sum(-1) / math.sqrt(d))   # (B, T, nb)
        u = (alpha[..., None] * v[:, :, None, :]).reshape(B, T, nb * d)
        y = u + shortconv(u, s.conv, ctx['doc'], dil=3)                                     # taps t, t-3, t-6, t-9
        return R + y.view(B, T, nb, d)


class Pool(nn.Module):
    """Per-channel softmax over the 4 positions of a group (W_Z = 0, B = 0, W_C = I: the exact mean at init)."""
    def __init__(s, d, k):
        super().__init__(); s.k = k
        s.wz = nn.Linear(d, d, bias=False); nn.init.zeros_(s.wz.weight)
        s.B = nn.Parameter(torch.zeros(k, d))
        s.wc = nn.Linear(d, d, bias=False)
        with torch.no_grad(): s.wc.weight.copy_(torch.eye(d))

    def forward(s, H):
        B, T, d = H.shape; G = T // s.k
        z = s.wz(H).view(B, G, s.k, d) + s.B
        return (z.softmax(2) * s.wc(H).view(B, G, s.k, d)).sum(2)


# ------------------------------------------------------------------------------------------------ concept module, decoder
class AttnRes(nn.Module):
    """Delta-AttnRes read: softmax over RMS-normalised sources against a learned (zero-initialised) query."""
    def __init__(s, d):
        super().__init__(); s.q = nn.Parameter(torch.zeros(d))

    def forward(s, S, Sn):
        """S, Sn: (..., n, d) running stacks of the sources and of their RMS-normalised copies."""
        p = (Sn @ s.q).softmax(-1)                                                       # (..., n)
        return (p.unsqueeze(-2) @ S).squeeze(-2)


class ConceptSelfAttn(nn.Module):
    def __init__(s, d, H):
        super().__init__(); s.H, s.dh = H, d // H
        s.wqkv = lin(d, 3 * d); s.wg = lin(d, d); s.wo = lin(d, d); s.qk = QKNorm(H, d // H, key_gain=False)

    def forward(s, x, ctx):
        B, G, _ = x.shape
        q, k, v = s.wqkv(x).view(B, G, 3, s.H, s.dh).unbind(2)
        q, k = s.qk.q(q), s.qk.k(k)
        cos, sin = ctx['rope_grp']
        q = rope(q, cos[:, None], sin[:, None], s.dh); k = rope(k, cos[:, None], sin[:, None], s.dh)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=ctx['grp_mask'])
        return s.wo(y.transpose(1, 2).reshape(B, G, -1) * torch.sigmoid(s.wg(x)))


class WindowRead(nn.Module):
    """Concept g reads sg[L_enc] at token positions 4g+3-15 .. 4g+3 (query at RoPE position 4g+3)."""
    def __init__(s, d, H, W):
        super().__init__(); s.H, s.dh, s.W = H, d // H, W
        s.wq = lin(d, d); s.wkv = lin(d, 2 * d); s.wg = lin(d, d); s.wo = lin(d, d); s.qk = QKNorm(H, d // H)

    def forward(s, x, ctx):
        B, G, _ = x.shape; H, dh, W = s.H, s.dh, s.W
        q = s.qk.q(s.wq(x).view(B, G, H, dh))
        cq, sq = ctx['rope_rel']; q = rope(q, cq[:, None], sq[:, None], dh)
        k, v = s.wkv(ctx['Lsg']).view(B, -1, 2, H, dh).unbind(2)
        k = s.qk.k(k); ct, st = ctx['rope_tok']; k = rope(k, ct[:, None], st[:, None], dh)
        idx = ctx['win_idx']                                                             # (G, W) into padded T
        kp = F.pad(k, (0, 0, 0, 0, W - 4, 0))[:, idx]; vp = F.pad(v, (0, 0, 0, 0, W - 4, 0))[:, idx]  # (B, G, W, H, dh)
        sc = torch.einsum('bghd,bgjhd->bghj', q, kp) / math.sqrt(dh)
        sc = sc.masked_fill(~ctx['win_mask'][:, :, None, :], float('-inf'))
        y = torch.einsum('bghj,bgjhd->bghd', sc.softmax(-1), vp).reshape(B, G, -1)
        return s.wo(y * torch.sigmoid(s.wg(x)))


class ConceptRead(nn.Module):
    """Decoder row t reads u_j for j <= floor((t+1)/4) - 1 in its document, plus a null entry and a sink logit."""
    def __init__(s, d, H):
        super().__init__(); s.H, s.dh = H, d // H
        s.wq = lin(d, d); s.wkv = lin(d, 2 * d); s.wg = lin(d, d); s.wo = lin(d, d); s.qk = QKNorm(H, d // H, key_gain=False)
        s.k_null = nn.Parameter(0.02 * torch.randn(H, d // H)); s.v_null = nn.Parameter(0.02 * torch.randn(H, d // H))
        s.sink = nn.Parameter(torch.zeros(H))

    def forward(s, x, ctx):
        B, T, _ = x.shape; H, dh = s.H, s.dh
        q0 = s.qk.q(s.wq(x).view(B, T, H, dh))
        ct, st = ctx['rope_tok']; q = rope(q0, ct[:, None], st[:, None], dh)
        un = ctx['u_norm']; G = un.shape[1]
        k, v = s.wkv(un).view(B, G, 2, H, dh).unbind(2); k = s.qk.k(k)
        cr, sr = ctx['rope_rel']; k = rope(k, cr[:, None], sr[:, None], dh)
        sc = torch.einsum('bthd,bghd->bhtg', q, k) / math.sqrt(dh)
        sc = sc.masked_fill(~ctx['read_mask'][:, None], float('-inf'))
        sn = torch.einsum('bthd,hd->bht', q0, s.qk.k(s.k_null)) / math.sqrt(dh)        # null entry: no position
        ss = s.sink[None, :, None].expand(B, H, T)
        allsc = torch.cat([sc, sn[..., None], ss[..., None]], -1)
        p = allsc.softmax(-1)
        y = torch.einsum('bhtg,bghd->bthd', p[..., :G], v) + p[..., G].transpose(1, 2)[..., None] * s.v_null
        return s.wo(y.reshape(B, T, -1) * torch.sigmoid(s.wg(x)))


class Stack(nn.Module):
    """A single stream with Delta-AttnRes over every sublayer; source 0 is the stream's input."""
    def __init__(s, d, subs):
        super().__init__()
        s.subs = nn.ModuleList(subs); s.res = nn.ModuleList([AttnRes(d) for _ in subs[1:]]); s.out = AttnRes(d)
        s.pre = nn.ModuleList([Norm(d) for _ in subs]); s.ckpt = False

    def forward(s, h0, ctx, kinds):
        S = h0.unsqueeze(-2); Sn = rms(h0).unsqueeze(-2)                                # running stacks (..., n, d)
        for i, (sub, pre, kind) in enumerate(zip(s.subs, s.pre, kinds)):
            x = pre(s.res[i - 1](S, Sn) if i else h0)                                     # one source: no softmax
            if s.training and s.ckpt:
                y = ckpt(sub, x, ctx, use_reentrant=False) if kind != 'ffn' else ckpt(sub, x, use_reentrant=False)
            else:
                y = sub(x, ctx) if kind != 'ffn' else sub(x)
            S = torch.cat([S, y.unsqueeze(-2)], -2); Sn = torch.cat([Sn, rms(y).unsqueeze(-2)], -2)
        return s.out(S, Sn)


class DecLocal(nn.Module):
    def __init__(s, d, H, W):
        super().__init__(); s.a = LocalAttn(d, H, W, d // H)

    def forward(s, x, ctx):
        return s.a(x, ctx['L_enc'], ctx)


# ------------------------------------------------------------------------------------------------ the model
class V13(nn.Module):
    def __init__(s, c=None):
        super().__init__(); c = c or Cfg(); s.c = c; d = c.d
        s.emb = nn.Embedding(c.V, d); nn.init.normal_(s.emb.weight, std=1.0)
        s.e1 = nn.ModuleList([EncLayer(k, c) for _ in range(c.n_e1_blocks) for k in ('S', 'S', 'S', 'F')])
        s.engram = Engram(c)
        s.e2 = nn.ModuleList([EncLayer(k, c) for k in ('M2', 'M2', 'M2', 'F2')])
        s.s0 = nn.Parameter(torch.randn(d)); s.e_r = nn.Parameter(0.02 * torch.randn(c.R, d))
        s.e2_exit = WriteScale(d, 5, [-.5, .5])
        s.e3 = nn.ModuleList([EncLayer(k, c) for k in ('S3', 'S3', 'S3', 'F3')])
        s.exit_L = GR(d, 4, 0, 2, []); s.exit_G = GR(d, 4, 2, 4, [])
        s.pool_x, s.pool_L, s.pool_G = Pool(d, c.k), Pool(d, c.k), Pool(d, c.k)
        s.src_norm = nn.Parameter(torch.zeros(3, d)); s.beta = nn.Parameter(torch.tensor([-1.0, -1.0, 1.0]))
        s.wp = lin(d, d)
        subs, s.c_kinds = [], []
        for _ in range(c.n_c):
            subs += [ConceptSelfAttn(d, c.H), WindowRead(d, c.H, c.wC), SwiGLU(d, c.ffn)]; s.c_kinds += ['sa', 'win', 'ffn']
        s.cm = Stack(d, subs)
        subs, s.d_kinds = [], []
        for _ in range(c.n_d):
            subs += [DecLocal(d, c.H, c.wD), ConceptRead(d, c.H), SwiGLU(d, c.ffn)]; s.d_kinds += ['loc', 'read', 'ffn']
        s.dec = Stack(d, subs)
        s.final = Norm(d)
        s.ncp = nn.Sequential(Norm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, c.pq_S * c.pq_N))
        s.ckpt = False

    def set_ckpt(s, on):
        s.ckpt = on; s.cm.ckpt = on; s.dec.ckpt = on

    # ---------------------------------------------------------------- per-batch context (masks, rope tables)
    def context(s, doc):
        c = s.c; B, T = doc.shape; G = T // c.k; dev = doc.device
        t = torch.arange(T, device=dev); g = torch.arange(G, device=dev)
        causal = t[None, :] <= t[:, None]
        ctx = dict(doc=doc)
        ctx['full_mask'] = (causal[None] & (doc[:, :, None] == doc[:, None, :]))[:, None]
        gdoc = doc[:, ::c.k]                                                              # doc of each group
        ctx['gdoc'] = gdoc
        ctx['grp_mask'] = ((g[None, :] <= g[:, None])[None] & (gdoc[:, :, None] == gdoc[:, None, :]))[:, None]
        rel = c.k * g + c.k - 1                                                          # release row 4g+3
        win = rel[:, None] - c.wC + 1 + torch.arange(c.wC, device=dev)[None]              # (G, W) token positions
        ctx['win_idx'] = win + (c.wC - c.k)                                              # into the left-padded axis
        dpad = F.pad(doc, (c.wC - c.k, 0), value=-1)
        ctx['win_mask'] = (dpad[:, ctx['win_idx']] == gdoc[:, :, None]) & (win >= 0)[None]
        last = (t + 1) // c.k - 1                                                        # newest legal concept per row
        ctx['read_mask'] = (g[None, :] <= last[:, None])[None] & (doc[:, :, None] == gdoc[:, None, :])   # (B, T, G)
        dt = s.emb.weight.dtype
        ctx['rope_tok'] = tuple(z.to(dt) for z in rope_tables(t, s.c.d // s.c.H))
        ctx['rope_tok_part'] = tuple(z.to(dt) for z in rope_tables(t, int(s.c.rope_frac_S * s.c.d // s.c.H)))
        ctx['rope_grp'] = tuple(z.to(dt) for z in rope_tables(g, s.c.d // s.c.H))
        ctx['rope_rel'] = tuple(z.to(dt) for z in rope_tables(rel, s.c.d // s.c.H))
        return ctx

    def encode(s, ids, eidx, ctx):
        c = s.c; B, T = ids.shape
        x = s.emb(ids)
        R = x[:, :, None, :].expand(B, T, 4, c.d).contiguous()
        run = (lambda f, *z: ckpt(f, *z, use_reentrant=False)) if (s.training and s.ckpt) else (lambda f, *z: f(*z))
        for i, layer in enumerate(s.e1):
            R = run(layer, R, ctx)
            if i == 0: R = run(s.engram, R, eidx, ctx)
        W = s.s0.expand(B, T, c.d)
        for r in range(c.R):
            R5 = torch.cat([R, (W + s.e_r[r])[:, :, None]], 2)
            for layer in s.e2: R5 = run(layer, R5, ctx)
            W = R5[:, :, 4]
        sc = s.e2_exit(torch.cat([R, W[:, :, None]], 2))
        R = R + torch.cat([torch.zeros_like(R[:, :, :2]), sc[..., None] * W[:, :, None, :]], 2)
        for layer in s.e3: R = run(layer, R, ctx)
        L_enc, _ = s.exit_L.read(R); G_enc, _ = s.exit_G.read(R)
        return x, L_enc, G_enc

    def concepts(s, x, L_enc, G_enc, ctx):
        beta = s.beta.softmax(0)
        srcs = [s.pool_x(x), s.pool_L(L_enc.detach()), s.pool_G(G_enc)]
        m = sum(beta[i] * rms(srcs[i]) * (1 + s.src_norm[i]) for i in range(3))
        p = s.wp(m)
        ctx['Lsg'] = L_enc.detach()
        u = s.cm(p, ctx, s.c_kinds)
        return m, u

    def decode(s, L_enc, u, ctx):
        ctx['L_enc'] = L_enc; ctx['u_norm'] = rms(u)
        h = s.dec(L_enc, ctx, s.d_kinds)
        return (s.final(h) @ s.emb.weight.t()) / math.sqrt(s.c.d)

    def forward(s, ids, doc, eidx=None, u_hook=None, return_all=False):
        if eidx is None: eidx = s.engram.keys(ids, doc)
        ctx = s.context(doc)
        x, L_enc, G_enc = s.encode(ids, eidx, ctx)
        m, u = s.concepts(x, L_enc, G_enc, ctx)
        if u_hook is not None: u = u_hook(u, m, ctx)
        logits = s.decode(L_enc, u, ctx)
        ncp_logits = s.ncp(u).view(*u.shape[:2], s.c.pq_S, s.c.pq_N)
        if return_all: return dict(logits=logits, m=m, u=u, ncp=ncp_logits, L_enc=L_enc, G_enc=G_enc)
        return logits, m, ncp_logits


def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    groups = {}
    for n, p in model.named_parameters():
        groups[n.split('.')[0]] = groups.get(n.split('.')[0], 0) + p.numel()
    return tot, groups
