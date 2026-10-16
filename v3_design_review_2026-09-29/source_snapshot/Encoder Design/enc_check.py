"""Prototype of the frozen-candidate encoder, written exactly from the spec, to test its claims.
Tests: reach of L and G, causality, Engram init gradients, gate values at init, param count,
document-boundary leakage, QK-norm logit ceiling."""
import math, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

class ZCRMSNorm(nn.Module):          # zero-centred RMSNorm: scale = 1 + gamma, gamma init 0
    def __init__(s, d, learn=True):
        super().__init__(); s.g = nn.Parameter(torch.zeros(d)) if learn else None
    def forward(s, x):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return y * (1 + s.g) if s.g is not None else y

def rms(x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

class ShortConv(nn.Module):          # causal depthwise, kernel 4, residual form, doc-aware optional
    def __init__(s, d, k=4, zero=True):
        super().__init__(); s.k = k; s.w = nn.Parameter(torch.zeros(d, k) if zero else torch.randn(d, k) * 0.3)
    def forward(s, x, doc=None):      # x: B,T,d ; doc: B,T segment ids
        out = x.clone()
        for i in range(1, s.k):
            sh = F.pad(x, (0, 0, i, 0))[:, :x.shape[1]]
            if doc is not None:
                same = (F.pad(doc, (i, 0), value=-1)[:, :x.shape[1]] == doc).unsqueeze(-1)
                sh = sh * same
            out = out + sh * s.w[:, i]
        return out + x * s.w[:, 0]    # residual: x + sum_i w_i * x_{t-i}

class GRRead(nn.Module):             # x = mean_i g_i * zcRMS(R_i), g_i = sigmoid(U V rms(R_i) + b)
    def __init__(s, d, n, r=32):
        super().__init__()
        s.norm = nn.ModuleList([ZCRMSNorm(d) for _ in range(n)])
        s.U = nn.ModuleList([nn.Linear(r, d, bias=True) for _ in range(n)])
        s.V = nn.ModuleList([nn.Linear(d, r, bias=False) for _ in range(n)])
        for u in s.U: nn.init.constant_(u.bias, 2.0)
    def forward(s, streams):
        xs = [torch.sigmoid(s.U[i](s.V[i](rms(R)))) * s.norm[i](R) for i, R in enumerate(streams)]
        return sum(xs) / len(xs), [torch.sigmoid(s.U[i](s.V[i](rms(R)))) for i, R in enumerate(streams)]

class GRWrite(nn.Module):            # s_i = 2 sigmoid(u_i^T x + a_i), a_i = 0
    def __init__(s, d, n):
        super().__init__(); s.u = nn.Linear(d, n); nn.init.zeros_(s.u.bias)
    def forward(s, x): return 2 * torch.sigmoid(s.u(x))   # B,T,n

class Attn(nn.Module):
    def __init__(s, d, H=8, kvh=2, dh=64, window=None, rope_dims=16, conv_zero=True):
        super().__init__(); s.H, s.kvh, s.dh, s.win, s.rd = H, kvh, dh, window, rope_dims
        s.q = nn.Linear(d, H*dh, bias=False); s.kv = nn.Linear(d, 2*kvh*dh, bias=False)
        s.o = nn.Linear(H*dh, d, bias=False); s.gate = nn.Linear(d, H*dh, bias=False)
        s.cq, s.ck, s.cv = ShortConv(H*dh, zero=conv_zero), ShortConv(kvh*dh, zero=conv_zero), ShortConv(kvh*dh, zero=conv_zero)
        s.qn, s.kn = ZCRMSNorm(dh), ZCRMSNorm(dh)
    def rope(s, x):
        if s.win is None or s.rd == 0: return x      # NoPE for FULL layers
        T = x.shape[2]; half = s.rd // 2
        inv = 1.0 / (10000 ** (torch.arange(half, dtype=x.dtype) / half))
        ang = torch.arange(T, dtype=x.dtype)[:, None] * inv[None]
        c, sn = ang.cos(), ang.sin()
        a, b, rest = x[..., :half], x[..., half:s.rd], x[..., s.rd:]
        return torch.cat([a*c - b*sn, a*sn + b*c, rest], -1)
    def forward(s, x, doc):
        B, T, _ = x.shape
        q = s.cq(s.q(x), doc); k, v = s.kv(x).chunk(2, -1); k, v = s.ck(k, doc), s.cv(v, doc)
        q = q.view(B, T, s.H, s.dh).transpose(1, 2); k = k.view(B, T, s.kvh, s.dh).transpose(1, 2); v = v.view(B, T, s.kvh, s.dh).transpose(1, 2)
        q, k = s.rope(s.qn(q)), s.rope(s.kn(k))
        k = k.repeat_interleave(s.H // s.kvh, 1); v = v.repeat_interleave(s.H // s.kvh, 1)
        i = torch.arange(T); mask = (i[None, :] <= i[:, None])
        if s.win is not None: mask = mask & (i[:, None] - i[None, :] < s.win)
        mask = mask[None] & (doc[:, :, None] == doc[:, None, :])
        att = (q @ k.transpose(-1, -2)) / math.sqrt(s.dh)
        att = att.masked_fill(~mask[:, None], float('-inf')).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, -1)
        return s.o(y * torch.sigmoid(s.gate(x)))

class SwiGLU(nn.Module):
    def __init__(s, d, f): super().__init__(); s.w1 = nn.Linear(d, f, bias=False); s.w3 = nn.Linear(d, f, bias=False); s.w2 = nn.Linear(f, d, bias=False)
    def forward(s, x): return s.w2(F.silu(s.w1(x)) * s.w3(x))

class Layer(nn.Module):              # kind 'S' reads {L} writes {L,G}; 'F' reads {L,G} writes {G}
    def __init__(s, kind, d, f, conv_zero=True):
        super().__init__(); s.kind = kind
        nr = 1 if kind == 'S' else 2; nw = 2 if kind == 'S' else 1
        s.r1, s.w1, s.r2, s.w2 = GRRead(d, nr), GRWrite(d, nw), GRRead(d, nr), GRWrite(d, nw)
        s.attn = Attn(d, window=8 if kind == 'S' else None, conv_zero=conv_zero); s.ffn = SwiGLU(d, f)
    def _step(s, L, G, read, write, fn, doc):
        x, _ = read([L] if s.kind == 'S' else [L, G]); y = fn(x, doc) if fn is s.attn else fn(x); sc = write(x)
        if s.kind == 'S': return L + sc[..., :1] * y, G + sc[..., 1:2] * y
        return L, G + sc[..., :1] * y                        # FULL never writes L
    def forward(s, L, G, doc):
        L, G = s._step(L, G, s.r1, s.w1, s.attn, doc)
        return s._step(L, G, s.r2, s.w2, s.ffn, doc)

class Engram(nn.Module):
    P = 131071
    def __init__(s, d, heads=4, dm=48, orders=(2, 3), rows=None, conv_zero_wv=True):
        super().__init__(); s.orders, s.heads = orders, heads; rows = rows or s.P
        s.rows = rows
        s.tab = nn.ModuleList([nn.Embedding(rows, dm) for _ in range(len(orders) * heads)])
        for t in s.tab: nn.init.normal_(t.weight, std=0.02)
        de = len(orders) * heads * dm
        s.Wk = nn.Linear(de, d, bias=False); s.Wv = nn.Linear(de, d, bias=False)
        if conv_zero_wv: nn.init.zeros_(s.Wv.weight)
        s.read = GRRead(d, 1); s.conv = ShortConv(d, zero=False); s.write = GRWrite(d, 2)
        s.mult = [(2654435761 + 97 * j) | 1 for j in range(64)]
    def keys(s, ids, doc):
        out = []
        for n in s.orders:
            grams = []
            for i in range(n):
                sh = F.pad(ids, (i, 0), value=0)[:, :ids.shape[1]]
                valid = F.pad(doc, (i, 0), value=-1)[:, :ids.shape[1]] == doc
                grams.append(torch.where(valid, sh + 1, torch.zeros_like(sh)))   # 0 = boundary
            for h in range(s.heads):
                x = torch.zeros_like(ids)
                for i, g in enumerate(grams): x = (x * s.mult[h * 8 + i]) ^ g
                out.append(x.abs() % s.rows)
        return out
    def forward(s, L, G, ids, doc):
        e = torch.cat([t(k) for t, k in zip(s.tab, s.keys(ids, doc))], -1)
        hx, _ = s.read([L]); k, v = s.Wk(e), s.Wv(e)
        d = L.shape[-1]
        a = torch.sigmoid((rms(hx) * rms(k)).sum(-1, keepdim=True) / math.sqrt(d))
        y = F.silu(s.conv(a * v, doc)); sc = s.write(hx)
        return L + sc[..., :1] * y, G + sc[..., 1:2] * y, a

class Encoder(nn.Module):
    def __init__(s, V=8192, d=512, f=1408, k=4, dc=None, conv_zero=True, rows=None, beta=(-2., -2., 2.)):
        super().__init__(); dc = dc or d; s.k = k
        s.E = nn.Embedding(V, d); s.en = ZCRMSNorm(d, learn=False)
        s.layers = nn.ModuleList([Layer(c, d, f, conv_zero) for c in "SSSFSSSF"])
        s.engram = Engram(d, rows=rows)
        s.Wa = nn.Linear(d, 3); nn.init.zeros_(s.Wa.weight); s.Wa.bias.data = torch.tensor(beta)
        s.pool = nn.Linear(k * d, dc, bias=False)
        with torch.no_grad(): s.pool.weight.copy_(torch.cat([torch.eye(d)] * k, 1) / k)   # mean-pool init (dc=d)
        s.tokn = ZCRMSNorm(d)
    def forward(s, ids, doc):
        e0 = s.en(s.E(ids)); L = G = e0; alpha_eng = None
        for i, lyr in enumerate(s.layers):
            L, G = lyr(L, G, doc)
            if i == 0: L, G, alpha_eng = s.engram(L, G, ids, doc)
        a = torch.softmax(s.Wa(rms(G)), -1)                                   # depth-mix weights
        m = a[..., 0:1] * rms(e0) + a[..., 1:2] * rms(L) + a[..., 2:3] * rms(G)
        B, T, d = m.shape
        c = s.pool(m.view(B, T // s.k, s.k * d))
        chat = math.sqrt(c.shape[-1]) * c / (c.norm(dim=-1, keepdim=True) + 1e-6)
        return s.tokn(L), chat, dict(L=L, G=G, a=a, alpha_eng=alpha_eng)

def randomize(model):                 # give every gate/conv non-trivial values so reach is not hidden by zero init
    with torch.no_grad():
        for n, p in model.named_parameters():
            if 'tab' in n: continue
            p.add_(torch.randn_like(p) * 0.2)

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    # ---------- 1. reach and causality ----------
    enc = Encoder(V=64, d=64, f=128, rows=1021); randomize(enc); enc.eval()
    T = 256; ids = torch.randint(0, 64, (1, T)); doc = torch.zeros(1, T, dtype=torch.long)
    p = 100
    ids2 = ids.clone(); ids2[0, p] = (ids[0, p] + 7) % 64
    with torch.no_grad():
        o1, c1, x1 = enc(ids, doc); o2, c2, x2 = enc(ids2, doc)
    dL = (x1['L'] - x2['L']).abs().amax(-1)[0]; dG = (x1['G'] - x2['G']).abs().amax(-1)[0]
    chL = torch.nonzero(dL != 0).flatten(); chG = torch.nonzero(dG != 0).flatten()
    print(f"[reach] L changed at positions {chL.min().item()}..{chL.max().item()} -> reach = {chL.max().item()-p}")
    print(f"[reach] G changed up to position {chG.max().item()} (T-1={T-1}) -> unbounded: {chG.max().item()==T-1}")
    print(f"[causal] any change before p in L/G: {bool((dL[:p]!=0).any() or (dG[:p]!=0).any())}")
    dc_ = (c1 - c2).abs().amax(-1)[0]; chc = torch.nonzero(dc_ != 0).flatten()
    print(f"[causal] first changed concept index {chc.min().item()} (token {p} is in chunk {p//4})")
    # ---------- 2. document-boundary leakage ----------
    doc2 = torch.zeros(1, T, dtype=torch.long); doc2[0, 128:] = 1
    ids3 = ids.clone(); ids3[0, 126] = (ids[0, 126] + 5) % 64
    with torch.no_grad():
        _, _, y1 = enc(ids, doc2); _, _, y2 = enc(ids3, doc2)
    leak = (y1['G'][0, 128:] - y2['G'][0, 128:]).abs().max().item()
    print(f"[docs] change in doc 1 caused by editing doc 0 (doc-aware conv/attn/ngram): {leak:.2e}")
    # same test with conv made doc-UNaware
    for mod in enc.modules():
        if isinstance(mod, ShortConv): mod._fwd = mod.forward; mod.forward = (lambda self: (lambda x, doc=None: self._fwd(x, None)))(mod)
    with torch.no_grad():
        _, _, y1 = enc(ids, doc2); _, _, y2 = enc(ids3, doc2)
    print(f"[docs] same edit with doc-UNAWARE convs: {(y1['G'][0,128:]-y2['G'][0,128:]).abs().max().item():.2e}")

    # ---------- 3. init behaviour at the real M2 size ----------
    torch.set_default_dtype(torch.float32)
    enc = Encoder()   # spec init
    ne = sum(p.numel() for n, p in enc.named_parameters() if 'tab' not in n and not n.startswith('E.'))
    tab = sum(p.numel() for n, p in enc.named_parameters() if 'tab' in n)
    print(f"[params] non-embedding {ne/1e6:.2f}M | engram tables {tab/1e6:.2f}M | embedding {enc.E.weight.numel()/1e6:.2f}M")
    ids = torch.randint(0, 8192, (2, 128)); doc = torch.zeros(2, 128, dtype=torch.long)
    tok, chat, ex = enc(ids, doc)
    g = torch.sigmoid(enc.layers[0].r1.U[0](enc.layers[0].r1.V[0](rms(ex['L'])))).mean().item()
    s = enc.layers[0].w1(torch.randn(2, 128, 512)).mean().item()
    print(f"[init] read gate ~{g:.3f} | write scale ~{s:.3f} | depth-mix weights e0/L/G = {[round(v,3) for v in ex['a'].mean((0,1)).tolist()]}")
    print(f"[init] ||chat||/sqrt(dc) = {(chat.norm(dim=-1)/math.sqrt(512)).mean().item():.4f}")
    loss = tok.pow(2).mean() + chat.sum() * 1e-3
    loss.backward()
    eg = enc.engram
    print(f"[engram step0] |grad Wv|={eg.Wv.weight.grad.abs().sum():.3e} |grad Wk|={eg.Wk.weight.grad.abs().sum():.3e} |grad tables|={sum(t.weight.grad.abs().sum() for t in eg.tab):.3e}")
    print(f"[engram] gate alpha at init: mean {ex['alpha_eng'].mean().item():.3f}, std {ex['alpha_eng'].std().item():.3f}")
    ge = enc.Wa.weight.grad.abs().sum().item(); print(f"[crc] grad on depth-mix router at init: {ge:.3e}")
    # ---------- 4. QK-norm logit ceiling ----------
    dh = 64; qn = torch.randn(dh); qn = qn / qn.pow(2).mean().sqrt()
    print(f"[qk] max logit with unit-RMS q,k (no gain): {(qn @ qn / math.sqrt(dh)).item():.2f} = sqrt(dh)")
    for N in (1024, 131072):
        need = math.log(N) + math.log(0.9 / 0.1)
        print(f"[qk] to put 0.9 mass on 1 of {N} keys (others at logit 0) needs gap {need:.1f}")
