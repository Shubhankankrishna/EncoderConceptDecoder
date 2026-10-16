"""HRM THINKER encoder v1.0 (frozen spec) - reference implementation + unit tests.

Two role streams: L (local, bounded reach 63) feeds the decoder; G (global) feeds the concept core via CRC.
Layer pattern S S S F S S S F; Engram after S1. Run `python encoder_v1.py` to execute the tests.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class ZCRMSNorm(nn.Module):
    """Zero-centred RMSNorm: (1 + gamma) * x / rms(x); gamma starts at 0 and IS weight-decayed (Qwen3.8)."""
    def __init__(s, d, learn=True):
        super().__init__(); s.gamma = nn.Parameter(torch.zeros(d)) if learn else None
    def forward(s, x):
        y = rms(x)
        return y * (1 + s.gamma) if s.gamma is not None else y


class ShortConv(nn.Module):
    """Causal depthwise conv, kernel 4, dilation 1, residual form x + sum_i w_i * x_{t-i}.
    w starts at 0 (exact identity at init). Document-aware: never mixes across segment boundaries."""
    def __init__(s, d, k=4, zero_init=True):
        super().__init__(); s.k = k
        s.w = nn.Parameter(torch.zeros(d, k) if zero_init else torch.randn(d, k) * 0.3)
    def forward(s, x, doc):
        T = x.shape[1]; out = x * (1 + s.w[:, 0])
        for i in range(1, s.k):
            sh = F.pad(x, (0, 0, i, 0))[:, :T]
            same = (F.pad(doc, (i, 0), value=-1)[:, :T] == doc).unsqueeze(-1)
            out = out + sh * same * s.w[:, i]
        return out


class GRRead(nn.Module):
    """x = mean_i g_i * zcRMS(R_i), g_i = sigmoid(U_i V_i rms(R_i) + b_i); rank 32, b_i = +2 (gate ~0.88)."""
    def __init__(s, d, n, r=32):
        super().__init__()
        s.norm = nn.ModuleList([ZCRMSNorm(d) for _ in range(n)])
        s.V = nn.ModuleList([nn.Linear(d, r, bias=False) for _ in range(n)])
        s.U = nn.ModuleList([nn.Linear(r, d) for _ in range(n)])
        for u in s.U: nn.init.zeros_(u.weight); nn.init.constant_(u.bias, 2.0)
    def forward(s, streams):
        out = 0
        for i, R in enumerate(streams):
            n = rms(R)                                   # one normalisation, reused by gate and value
            out = out + torch.sigmoid(s.U[i](s.V[i](n))) * n * (1 + s.norm[i].gamma)
        return out / len(streams)


class GRWrite(nn.Module):
    """s_i = 2 sigmoid(u_i^T x + a_i); u and a start at 0, so s_i = 1 exactly at init."""
    def __init__(s, d, n):
        super().__init__(); s.u = nn.Linear(d, n); nn.init.zeros_(s.u.weight); nn.init.zeros_(s.u.bias)
    def forward(s, x):
        return 2 * torch.sigmoid(s.u(x))


class Attn(nn.Module):
    """Gated attention. SWA: window 8, partial RoPE (16/64). FULL: NoPE + learnable per-head temperature (no WD)."""
    def __init__(s, d, H=8, kvh=2, dh=64, window=None, rope_dims=16):
        super().__init__(); s.H, s.kvh, s.dh, s.win, s.rd = H, kvh, dh, window, rope_dims
        s.q = nn.Linear(d, H * dh, bias=False); s.kv = nn.Linear(d, 2 * kvh * dh, bias=False)
        s.o = nn.Linear(H * dh, d, bias=False); s.gate = nn.Linear(d, H * dh, bias=False)
        s.cq, s.ck, s.cv = ShortConv(H * dh), ShortConv(kvh * dh), ShortConv(kvh * dh)
        s.qn, s.kn = ZCRMSNorm(dh), ZCRMSNorm(dh)
        s.log_tau = nn.Parameter(torch.zeros(H)) if window is None else None   # FULL only
    def rope(s, x):
        if s.win is None: return x                                            # NoPE
        T, half = x.shape[2], s.rd // 2
        inv = 1.0 / (10000 ** (torch.arange(half, dtype=x.dtype) / half))
        ang = torch.arange(T, dtype=x.dtype)[:, None] * inv[None]
        c, sn = ang.cos(), ang.sin()
        a, b, rest = x[..., :half], x[..., half:s.rd], x[..., s.rd:]
        return torch.cat([a * c - b * sn, a * sn + b * c, rest], -1)
    def forward(s, x, doc):
        B, T, _ = x.shape
        q = s.cq(s.q(x), doc); k, v = s.kv(x).chunk(2, -1); k, v = s.ck(k, doc), s.cv(v, doc)
        q = q.view(B, T, s.H, s.dh).transpose(1, 2)
        k = k.view(B, T, s.kvh, s.dh).transpose(1, 2); v = v.view(B, T, s.kvh, s.dh).transpose(1, 2)
        q, k = s.rope(s.qn(q)), s.rope(s.kn(k))
        k, v = k.repeat_interleave(s.H // s.kvh, 1), v.repeat_interleave(s.H // s.kvh, 1)
        i = torch.arange(T); mask = i[None, :] <= i[:, None]
        if s.win is not None: mask = mask & (i[:, None] - i[None, :] < s.win)
        mask = mask[None] & (doc[:, :, None] == doc[:, None, :])               # document mask
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(s.dh)
        if s.log_tau is not None: logits = logits * s.log_tau.exp()[None, :, None, None]
        att = logits.masked_fill(~mask[:, None], float('-inf')).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, -1)
        return s.o(y * torch.sigmoid(s.gate(x)))


class SwiGLU(nn.Module):
    def __init__(s, d, f):
        super().__init__(); s.w1 = nn.Linear(d, f, bias=False); s.w3 = nn.Linear(d, f, bias=False); s.w2 = nn.Linear(f, d, bias=False)
    def forward(s, x): return s.w2(F.silu(s.w1(x)) * s.w3(x))


class Layer(nn.Module):
    """'S': both sublayers read {L}, write {L,G}.  'F': both sublayers (attention AND FFN) read {L,G}, write {G} only."""
    def __init__(s, kind, d, f):
        super().__init__(); s.kind = kind
        nr, nw = (1, 2) if kind == 'S' else (2, 1)
        s.r1, s.w1, s.r2, s.w2 = GRRead(d, nr), GRWrite(d, nw), GRRead(d, nr), GRWrite(d, nw)
        s.attn = Attn(d, window=8 if kind == 'S' else None); s.ffn = SwiGLU(d, f)
    def _sub(s, L, G, read, write, fn, doc):
        x = read([L] if s.kind == 'S' else [L, G])
        y = s.attn(x, doc) if fn == 'attn' else s.ffn(x)
        sc = write(x)
        if s.kind == 'S': return L + sc[..., 0:1] * y, G + sc[..., 1:2] * y
        return L, G + sc[..., 0:1] * y
    def forward(s, L, G, doc):
        L, G = s._sub(L, G, s.r1, s.w1, 'attn', doc)
        return s._sub(L, G, s.r2, s.w2, 'ffn', doc)


class Engram(nn.Module):
    """Orders {2,3}, 4 hash heads each, prime-row tables x 48 dims; context gate; conv k=4 dilation 1.
    Keys use canonical token ids (case + leading-space merged, built offline) and reset at document starts."""
    def __init__(s, d, rows=131071, heads=4, dm=48, orders=(2, 3)):
        super().__init__(); s.orders, s.heads, s.rows = orders, heads, rows
        s.tab = nn.ModuleList([nn.Embedding(rows, dm) for _ in range(len(orders) * heads)])
        for t in s.tab: nn.init.normal_(t.weight, std=0.02)                    # small random, NOT zero
        de = len(orders) * heads * dm
        s.Wk = nn.Linear(de, d, bias=False); s.Wv = nn.Linear(de, d, bias=False); nn.init.zeros_(s.Wv.weight)
        s.read = GRRead(d, 1); s.conv = ShortConv(d, zero_init=False); s.write = GRWrite(d, 2)
        s.mult = [(2654435761 + 97 * j) | 1 for j in range(64)]
    def keys(s, ids, doc):
        T, out, grams = ids.shape[1], [], []
        for i in range(max(s.orders)):
            sh = F.pad(ids, (i, 0))[:, :T]
            ok = F.pad(doc, (i, 0), value=-1)[:, :T] == doc
            grams.append(torch.where(ok, sh + 1, torch.zeros_like(sh)))       # 0 = document boundary
        for n in s.orders:
            for h in range(s.heads):
                x = torch.zeros_like(ids)
                for i in range(n): x = (x * s.mult[h * 8 + i]) ^ grams[i]     # multiplicative-XOR hash
                out.append(x.abs() % s.rows)
        return out
    def forward(s, L, G, canon_ids, doc):
        e = torch.cat([t(k) for t, k in zip(s.tab, s.keys(canon_ids, doc))], -1)
        h = s.read([L]); k, v = s.Wk(e), s.Wv(e)
        a = torch.sigmoid((rms(h) * rms(k)).sum(-1, keepdim=True) / math.sqrt(L.shape[-1]))
        y = F.silu(s.conv(a * v, doc)); sc = s.write(h)
        return L + sc[..., 0:1] * y, G + sc[..., 1:2] * y


class Encoder(nn.Module):
    def __init__(s, V=8192, d=512, f=1408, k=4, dc=None, rows=131071):
        super().__init__(); dc = dc or d; s.k, s.dc = k, dc
        s.E = nn.Embedding(V, d)
        s.layers = nn.ModuleList([Layer(c, d, f) for c in "SSSFSSSF"])
        s.engram = Engram(d, rows=rows)
        s.Wa = nn.Linear(d, 3); nn.init.zeros_(s.Wa.weight)
        with torch.no_grad(): s.Wa.bias.copy_(torch.tensor([-1.0, -1.0, 1.0]))  # e0 / L / G = 0.11 / 0.11 / 0.79
        s.pool = nn.Linear(k * d, dc, bias=False)
        with torch.no_grad():
            P = torch.eye(d) if dc == d else torch.linalg.qr(torch.randn(max(dc, d), min(dc, d)))[0][:dc, :d] if dc > d \
                else torch.linalg.qr(torch.randn(d, dc))[0].T
            s.pool.weight.copy_(torch.cat([P] * k, 1) / k)                     # mean pooling (projected if dc != d)
        s.tok_norm = ZCRMSNorm(d)
    def forward(s, ids, doc, canon_ids=None):
        assert ids.shape[1] % s.k == 0, "rows must be padded so chunks never straddle documents"
        canon_ids = ids if canon_ids is None else canon_ids
        e0 = rms(s.E(ids)); L = G = e0
        for i, layer in enumerate(s.layers):
            L, G = layer(L, G, doc)
            if i == 0: L, G = s.engram(L, G, canon_ids, doc)
        a = torch.softmax(s.Wa(rms(G)), -1)
        m = a[..., 0:1] * e0 + a[..., 1:2] * rms(L) + a[..., 2:3] * rms(G)       # CRC depth mix
        B, T, d = m.shape
        c = s.pool(m.reshape(B, T // s.k, s.k * d))
        c_hat = math.sqrt(s.dc) * c / (c.norm(dim=-1, keepdim=True) + 1e-6)      # sphere
        return s.tok_norm(L), c_hat, dict(L=L, G=G, a=a)


def param_groups(model, wd=0.01):
    """No weight decay: Engram tables, FULL-layer temperatures, token embedding, conv kernels. Decay elsewhere (incl. zc gammas)."""
    no_wd, yes_wd = [], []
    for n, p in model.named_parameters():
        (no_wd if ('tab.' in n or 'log_tau' in n or n.startswith('E.') or n.endswith('.w')) else yes_wd).append(p)
    return [dict(params=yes_wd, weight_decay=wd), dict(params=no_wd, weight_decay=0.0)]


# ------------------------------------------------------------------ tests
def _randomize(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if 'tab.' not in n: p.add_(torch.randn_like(p) * 0.2)

def run_tests():
    torch.manual_seed(0); torch.set_default_dtype(torch.float64)
    enc = Encoder(V=64, d=64, f=128, rows=1021); _randomize(enc); enc.eval()
    T, p = 256, 100
    ids = torch.randint(0, 64, (1, T)); doc = torch.zeros(1, T, dtype=torch.long)
    ids2 = ids.clone(); ids2[0, p] = (ids[0, p] + 7) % 64
    with torch.no_grad(): _, c1, a = enc(ids, doc); _, c2, b = enc(ids2, doc)
    dL = (a['L'] - b['L']).abs().amax(-1)[0]; dG = (a['G'] - b['G']).abs().amax(-1)[0]
    reach = torch.nonzero(dL != 0).max().item() - p
    assert reach == 63, reach
    assert torch.equal(a['L'][0, p + 64:], b['L'][0, p + 64:]), "L must be bit-identical 64+ tokens later"
    assert dG[-1] != 0, "G must be global"
    assert not (dL[:p] != 0).any() and not (dG[:p] != 0).any(), "causality"
    assert torch.nonzero((c1 - c2).abs().amax(-1)[0] != 0).min().item() == p // 4, "concept causality"
    print(f"PASS reach(L)=63, L bit-identical at >=64, G global, causal, concept m depends on chunk<=m")

    doc2 = torch.zeros(1, T, dtype=torch.long); doc2[0, 128:] = 1
    ids3 = ids.clone(); ids3[0, 126] = (ids[0, 126] + 5) % 64
    with torch.no_grad(): _, _, y1 = enc(ids, doc2); _, _, y2 = enc(ids3, doc2)
    assert torch.equal(y1['G'][0, 128:], y2['G'][0, 128:]) and torch.equal(y1['L'][0, 128:], y2['L'][0, 128:])
    print("PASS no leakage across document boundaries (convs, attention, n-grams)")

    torch.set_default_dtype(torch.float32)
    enc = Encoder()
    ne = sum(q.numel() for n, q in enc.named_parameters() if 'tab.' not in n and not n.startswith('E.'))
    tab = sum(q.numel() for n, q in enc.named_parameters() if 'tab.' in n)
    ids = torch.randint(0, 8192, (2, 128)); doc = torch.zeros(2, 128, dtype=torch.long)
    tok, c_hat, ex = enc(ids, doc)
    w = ex['a'].mean((0, 1)).tolist()
    assert abs(w[2] - 0.787) < 0.01 and abs((c_hat.norm(dim=-1) / math.sqrt(512)).mean().item() - 1) < 1e-4
    (tok.pow(2).mean() + c_hat.sum() * 1e-3).backward()
    assert enc.engram.Wv.weight.grad.abs().sum() > 0
    print(f"PASS init: depth-mix {[round(x, 3) for x in w]}, |c_hat|=sqrt(dc), Engram W_V receives gradient")
    print(f"     params: non-embedding {ne/1e6:.2f}M | Engram tables {tab/1e6:.2f}M | embedding {enc.E.weight.numel()/1e6:.2f}M")

    enc2 = Encoder(dc=1024); _, c2h, _ = enc2(ids, doc); assert c2h.shape[-1] == 1024
    print("PASS pooling init works for d_c != d (d_c=1024)")

    # FULL layers can now exceed the QK-norm logit ceiling of sqrt(dh)=8
    need = math.log(131072) + math.log(9)
    print(f"INFO ceiling sqrt(dh)=8; sharp retrieval over 128k keys needs gap {need:.1f}; reachable with tau = {need/8:.2f}")


if __name__ == "__main__":
    run_tests()
