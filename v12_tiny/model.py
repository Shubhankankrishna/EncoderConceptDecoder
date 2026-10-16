"""V12 tiny models. Every model exposes the same interface:

  out = model(ids, ctrl=None)      -> dict(logits, resid=[e_0 .. e_L] (token-rate residual stream per depth), ...)

`ctrl` is an optional callable applied to the token-rate residual at the model's control depth
(`model.ctrl_layer`, between blocks): e <- ctrl(e). The metacontroller (meta.py) uses it on a frozen model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.g = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


def rope(x, pos):
    # x: (B, H, T, Dh); pos: (T,) or (B, T) float positions
    Dh = x.shape[-1]; half = Dh // 2
    freq = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    ang = pos[..., None].float() * freq                       # (T, half) or (B, T, half)
    if ang.dim() == 3: ang = ang[:, None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads; self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, pos, mask=None, kv=None, kv_pos=None):
        """Self-attention (kv None, causal) or cross-attention over `kv` with boolean mask (B, Tq, Tk)."""
        B, T, D = x.shape; H = self.h; Dh = D // H
        if kv is None:
            q, k, v = self.qkv(x).view(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
            q, k = rope(q, pos), rope(k, pos)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None] if mask is not None else None,
                                               is_causal=mask is None)
        else:
            Wq, Wk, Wv = self.qkv.weight.split(D, 0)
            q = F.linear(x, Wq).view(B, T, H, Dh).transpose(1, 2)
            Tk = kv.shape[1]
            k = F.linear(kv, Wk).view(B, Tk, H, Dh).transpose(1, 2); v = F.linear(kv, Wv).view(B, Tk, H, Dh).transpose(1, 2)
            q, k = rope(q, pos), rope(k, kv_pos)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
            y = torch.nan_to_num(y)                              # rows with no visible key -> 0
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, heads, mlp_mult=2):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, heads)
        self.fc1 = nn.Linear(d, mlp_mult * d, bias=False); self.fc2 = nn.Linear(mlp_mult * d, d, bias=False)

    def forward(self, x, pos, mask=None):
        x = x + self.attn(self.n1(x), pos, mask)
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


def init_weights(m, std=0.02):
    for n, p in m.named_parameters():
        if p.dim() >= 2: nn.init.normal_(p, 0, std)
        if n.endswith('fc2.weight') or n.endswith('attn.o.weight'): nn.init.normal_(p, 0, std / math.sqrt(2 * 6))


class Transformer(nn.Module):
    """Plain causal transformer (the paper's base model family): the size-matched control."""
    kind = 'TF'

    def __init__(self, vocab, d=128, layers=6, heads=4, mlp_mult=2, ctrl_layer=None):
        super().__init__()
        self.d, self.L = d, layers
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, heads, mlp_mult) for _ in range(layers))
        self.nf = RMSNorm(d)
        self.ctrl_layer = layers // 2 if ctrl_layer is None else ctrl_layer
        init_weights(self)

    def forward(self, ids, ctrl=None, **kw):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.emb(ids); resid = [x]
        for i, blk in enumerate(self.blocks):
            if ctrl is not None and i == self.ctrl_layer:
                x = ctrl(x)
            x = blk(x, pos); resid.append(x)
        h = self.nf(x)
        return dict(logits=h @ self.emb.weight.t(), resid=resid, final=x)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class CrossBlock(nn.Module):
    """Decoder block: causal self-attention, cross-attention to released concept states, MLP."""
    def __init__(self, d, heads, mlp_mult=2):
        super().__init__()
        self.n1, self.nc, self.n2 = RMSNorm(d), RMSNorm(d), RMSNorm(d)
        self.attn, self.xattn = Attention(d, heads), Attention(d, heads)
        self.fc1 = nn.Linear(d, mlp_mult * d, bias=False); self.fc2 = nn.Linear(mlp_mult * d, d, bias=False)

    def forward(self, x, pos, u, upos, rmask):
        x = x + self.attn(self.n1(x), pos)
        if u is not None:
            x = x + self.xattn(self.nc(x), pos, rmask, kv=u, kv_pos=upos)
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class ECD(nn.Module):
    """Encoder -> fixed-group pooling (k tokens) -> group-rate causal concept module (optionally a loop of 2,
    shared weights) -> decoder reading the released concept history by cross-attention.

    Group g covers tokens [k g, k g + k - 1] and is released to decoder rows t >= k g + k - 1 (the row that
    predicts token k g + k may use the whole group). The concept state of group g is u_g.
    Control points: 'concept' (group rate: e <- ctrl(e) on the concept stream, between visit 1 and visit 2 when
    looped, else between concept blocks), 'dec_in' (token rate, the decoder's input residual).
    """
    kind = 'ECD'

    def __init__(self, vocab, d=128, enc=2, concept=2, dec=2, heads=4, mlp_mult=2, k=4, loop=1,
                 ctrl_point='concept'):
        super().__init__()
        self.d, self.k, self.loop, self.ctrl_point = d, k, loop, ctrl_point
        self.emb = nn.Embedding(vocab, d)
        self.enc = nn.ModuleList(Block(d, heads, mlp_mult) for _ in range(enc))
        self.pool_n = RMSNorm(d); self.pool = nn.Linear(d, d, bias=False)
        self.cblocks = nn.ModuleList(Block(d, heads, mlp_mult) for _ in range(concept))
        self.visit_emb = nn.Parameter(torch.zeros(max(loop, 1), d))
        self.cn = RMSNorm(d)
        self.dec = nn.ModuleList(CrossBlock(d, heads, mlp_mult) for _ in range(dec))
        self.nf = RMSNorm(d)
        self.ctrl_layer = None
        init_weights(self)

    def groups(self, T):
        return T // self.k                                      # only complete groups are pooled

    def pooled(self, h):
        B, T, D = h.shape; G = self.groups(T)
        return self.pool(self.pool_n(h[:, :G * self.k].reshape(B, G, self.k, D).mean(2)))

    def concept(self, p, ctrl=None, cut=None):
        """Group-rate concept module. Returns final u and the per-visit states."""
        G = p.shape[1]; gpos = torch.arange(G, device=p.device)
        x, states = p, []
        nb = len(self.cblocks)
        for v in range(self.loop):
            x = x + self.visit_emb[v]
            for j, blk in enumerate(self.cblocks):
                at_ctrl = (self.loop > 1 and v == 1 and j == 0) or (self.loop == 1 and j == nb // 2)
                if ctrl is not None and self.ctrl_point == 'concept' and at_ctrl:
                    states.append(x); x = ctrl(x)
                elif at_ctrl:
                    states.append(x)
                x = blk(x, gpos)
            states.append(x)
        return self.cn(x), states

    def release_mask(self, T, G, device):
        t = torch.arange(T, device=device)[:, None]; g = torch.arange(G, device=device)[None]
        return (t >= self.k * g + self.k - 1)                    # (T, G)

    def forward(self, ids, ctrl=None, concept_off=False, **kw):
        B, T = ids.shape; dev = ids.device
        pos = torch.arange(T, device=dev)
        x = self.emb(ids); resid = [x]
        for blk in self.enc:
            x = blk(x, pos); resid.append(x)
        h = x
        p = self.pooled(h); G = p.shape[1]
        u, cstates = self.concept(p, ctrl)
        if concept_off: u = torch.zeros_like(u)
        rmask = self.release_mask(T, G, dev)[None].expand(B, T, G)
        gpos = (torch.arange(G, device=dev) * self.k + self.k - 1).float()
        if ctrl is not None and self.ctrl_point == 'dec_in':
            x = ctrl(x)
        dres = []
        for blk in self.dec:
            x = blk(x, pos, u if G > 0 else None, gpos, rmask); dres.append(x)
        resid += dres
        hf = self.nf(x)
        pooled_raw = h[:, :G * self.k].reshape(B, G, self.k, -1).mean(2)
        return dict(logits=hf @ self.emb.weight.t(), resid=resid, final=x, enc=h, pooled=p, pooled_raw=pooled_raw,
                    u=u, cstates=cstates, G=G)


def token_view(gstate, T, k):
    """Map a group-rate state (B, G, D) to token rows: row t sees the latest released group g with
    k g + k - 1 <= t (zeros before the first release)."""
    B, G, D = gstate.shape
    t = torch.arange(T, device=gstate.device)
    g = torch.div(t + 1, k, rounding_mode='floor') - 1
    g = g.clamp(max=G - 1)
    z = torch.cat([torch.zeros(B, 1, D, device=gstate.device, dtype=gstate.dtype), gstate], 1)
    return z[:, (g + 1).clamp(min=0)]
