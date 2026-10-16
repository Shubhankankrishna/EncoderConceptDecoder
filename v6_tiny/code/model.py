"""V6 tiny: document-isolated NCP, deterministic recurrence, zero-anchored Z.

No imports from mutable V5 code. FP32 is the reference execution mode.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

D, HEADS, FF, K, VOCAB, S, CODES, DZ = 128, 4, 384, 4, 2048, 8, 32, 8


def rms(x):
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)


def safe_rms(x):
    return x.float().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()


def ball(z, bound=2.0):
    return z * (bound / safe_rms(z)).clamp_max(1.0)


def init(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)


def positions(docs):
    n = docs.shape[1]
    idx = torch.arange(n, device=docs.device).expand_as(docs)
    first = torch.ones_like(docs, dtype=torch.bool)
    first[:, 1:] = docs[:, 1:] != docs[:, :-1]
    starts = torch.where(first, idx, torch.zeros_like(idx)).cummax(1).values
    return idx - starts


def rotate(q, pos):
    hd = q.shape[-1]
    inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, device=q.device).float() / hd))
    freq = pos.float().unsqueeze(-1) * inv
    freq = torch.cat([freq, freq], -1).unsqueeze(1)
    rot = torch.cat([-q[..., hd // 2:], q[..., :hd // 2]], -1)
    return q * freq.cos() + rot * freq.sin()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(D), nn.RMSNorm(D)
        self.qkv, self.o = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)
        self.w12, self.w3 = nn.Linear(D, 2 * FF, bias=False), nn.Linear(FF, D, bias=False)

    def forward(self, x, docs, pos=None, cache=None, return_cache=False):
        b, t, _ = x.shape
        if pos is None:
            pos = positions(docs)
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        q, k = rotate(q, pos), rotate(k, pos)
        offset = 0
        kd = docs
        if cache is not None:
            old_k, old_v, old_docs = cache
            offset = old_k.shape[2]
            k, v = torch.cat([old_k, k], 2), torch.cat([old_v, v], 2)
            kd = torch.cat([old_docs, docs], 1)
        causal = torch.arange(k.shape[2], device=x.device)[None, :] <= (offset + torch.arange(t, device=x.device))[:, None]
        mask = (docs[:, :, None] == kd[:, None, :]) & causal
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        x = x + self.o(a.transpose(1, 2).reshape(b, t, D))
        g, u = self.w12(self.n2(x)).chunk(2, -1)
        x = x + self.w3(F.silu(g) * u)
        return (x, (k, v, kd)) if return_cache else x

    def static_step(self,x,pos,cache):
        """Single-document inference only; fixed-size caches avoid length recompilation."""
        b=x.shape[0];q,k,v=self.qkv(self.n1(x)).view(b,1,3,HEADS,D//HEADS).permute(2,0,3,1,4)
        p=pos.reshape(1,1).expand(b,1);q,k=rotate(q,p),rotate(k,p)
        kc,vc=cache;index=pos.reshape(1,1,1,1).expand(b,HEADS,1,D//HEADS)
        kc.scatter_(2,index,k);vc.scatter_(2,index,v)
        allowed=torch.arange(kc.shape[2],device=x.device)<=pos
        a=F.scaled_dot_product_attention(q,kc,vc,attn_mask=allowed[None,None,None,:])
        x=x+self.o(a.transpose(1,2).reshape(b,1,D))
        g,u=self.w12(self.n2(x)).chunk(2,-1)
        return x+self.w3(F.silu(g)*u)


def chunk_info(docs):
    c = docs.view(docs.shape[0], -1, K)
    valid = (c >= 0).all(-1) & (c == c[..., :1]).all(-1)
    return torch.where(valid, c[..., 0], torch.full_like(c[..., 0], -1)), valid


def shift(v, docs):
    b, t = docs.shape
    cd, valid = chunk_info(docs)
    rep = F.pad(v.repeat_interleave(K, 1)[:, :t - 3], (0, 0, 3, 0))
    source = F.pad(cd.repeat_interleave(K, 1)[:, :t - 3], (3, 0), value=-2)
    allowed = (source == docs) & (docs >= 0)
    return rep * allowed.unsqueeze(-1)


class Foundation(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.enc = nn.ModuleList([Block() for _ in range(3)])
        self.enc_norm = nn.RMSNorm(D)
        self.con = nn.ModuleList([Block() for _ in range(2)])
        self.con_norm = nn.RMSNorm(D)
        self.heads = nn.Linear(D, S * CODES)
        self.Q = nn.Parameter(torch.randn(S, CODES, D // S) * 0.25)
        self.W_A = nn.Linear(D, D, bias=False)
        self.gamma = nn.Parameter(torch.full((D,), 0.1))
        self.dec = nn.ModuleList([Block() for _ in range(3)])
        self.nf = nn.RMSNorm(D)
        self.register_buffer('usage', torch.zeros(S, CODES))
        self.register_buffer('q_init', torch.tensor(False))
        self.apply(init)

    def encode(self, x, docs):
        h = self.emb(x)
        for block in self.enc:
            h = block(h, docs)
        return self.enc_norm(h)

    @staticmethod
    def pool(h):
        return rms(h.view(h.shape[0], -1, K, D).mean(2))

    def concept(self, c, cd):
        for block in self.con:
            c = block(c, cd)
        return self.con_norm(c)

    def pq(self, u):
        p = self.heads(u).view(*u.shape[:2], S, CODES).float().softmax(-1)
        return torch.einsum('bmsv,svd->bmsd', p, self.Q).reshape(*u.shape[:2], D)

    def port(self, u):
        return self.W_A(rms(u))

    def decode(self, h, chat, a, docs):
        h = h + shift(chat, docs) + self.gamma * shift(a, docs)
        for block in self.dec:
            h = block(h, docs)
        return self.nf(h) @ self.emb.weight.t()

    def forward(self, x, docs):
        h = self.encode(x, docs)
        c = self.pool(h)
        cd, valid = chunk_info(docs)
        u = self.concept(c, cd)
        chat, a = self.pq(u), self.port(u)
        return self.decode(h, chat, a, docs), c, chat, u, a

    def vq(self, c, valid):
        cs = c.detach().view(*c.shape[:2], S, D // S)
        dist = (cs.unsqueeze(-2) - self.Q[None, None]).float().square().sum(-1)
        q = dist.argmin(-1)
        selected = self.Q[torch.arange(S, device=c.device), q]
        error = (cs - selected).float().square().sum(-1).mean(-1)
        loss = (error * valid).sum() / valid.sum().clamp_min(1)
        return loss, q

    @torch.no_grad()
    def count_usage(self, q, valid):
        labels = q[valid]
        flat = (labels + torch.arange(S, device=q.device) * CODES).flatten()
        self.usage.add_(torch.bincount(flat, minlength=S * CODES).view(S, CODES))

    @torch.no_grad()
    def restart(self, c, valid, opt=None, generator=None):
        cs = c.detach()[valid].view(-1, S, D // S)
        if len(cs) < CODES:
            raise ValueError('Codebook initialization/restart needs >=32 valid chunks')
        dead = torch.ones_like(self.usage, dtype=torch.bool) if not bool(self.q_init) else self.usage == 0
        for s in range(S):
            ids = dead[s].nonzero().flatten()
            if len(ids):
                choice = torch.randperm(len(cs), generator=generator, device='cpu')[:len(ids)].to(c.device)
                self.Q[s, ids] = cs[choice, s]
        if opt is not None and self.Q in opt.state:
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in opt.state[self.Q]:
                    opt.state[self.Q][key][dead] = 0
        self.q_init.fill_(True)
        self.usage.zero_()
        return int(dead.sum())


class Adapters(nn.Module):
    def __init__(self, kind='gated', dz=DZ, rank=16, cap=0.1):
        super().__init__()
        self.kind, self.dz, self.cap = kind, dz, cap
        if kind == 'gated':
            self.G = nn.ModuleList([nn.Linear(D, rank) for _ in range(2)])
            self.V = nn.ModuleList([nn.Linear(dz, rank, bias=False) for _ in range(2)])
            self.U = nn.ModuleList([nn.Linear(rank, D, bias=False) for _ in range(2)])
        elif kind == 'direct':
            self.U = nn.ModuleList([nn.Linear(dz, D, bias=False) for _ in range(2)])
        else:
            raise ValueError(kind)
        self.apply(init)
        if kind == 'gated':
            for layer in self.G:
                nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def raw(self, e, z, q):
        if self.kind == 'direct':
            return self.U[q](z)
        return self.U[q]((1 + 0.5 * self.G[q](rms(e)).tanh()) * self.V[q](z))

    def forward(self, e, z):
        writes = []
        for q in range(2):
            raw = self.raw(e, z, q)
            writes.append(raw * (self.cap * safe_rms(e) / safe_rms(raw)).clamp_max(1))
        return writes

    @torch.no_grad()
    def calibrate(self, e, schedules):
        factors = []
        for q in range(2):
            ratio = max(float((safe_rms(self.raw(e, z, q)) / safe_rms(e)).max()) for z in schedules)
            factor = min(1.0, 0.01 / max(ratio, 1e-12))
            self.U[q].weight.mul_(factor); factors.append(factor)
        return factors


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(D, D, bias=False)
        self.fL, self.fH = Block(), Block()
        self.post_L, self.post_H = nn.RMSNorm(D), nn.RMSNorm(D)
        self.h0, self.l0 = nn.Parameter(torch.randn(D) * 0.02), nn.Parameter(torch.randn(D) * 0.02)
        self.out_n = nn.RMSNorm(D)
        self.B_u, self.B_A = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
        self.apply(init)

    def forward(self, c, docs, z=None, adapters=None, caches=None, return_cache=False, pos=None):
        e = self.in_proj(c)
        ih, il = (adapters(e, z) if adapters is not None and z is not None else (0.0, 0.0))
        h, l = self.h0.expand_as(e) + ih, self.l0.expand_as(e)
        new_caches, visit = [], 0
        for _ in range(2):
            for _ in range(2):
                args = dict(pos=pos, cache=caches[visit] if caches else None, return_cache=return_cache)
                out = self.fL(l + h + e + il, docs, **args)
                if return_cache:
                    out, cache = out; new_caches.append(cache)
                l = self.post_L(out); visit += 1
            out = self.fH(h + l, docs, pos=pos, cache=caches[visit] if caches else None, return_cache=return_cache)
            if return_cache:
                out, cache = out; new_caches.append(cache)
            h = self.post_H(out); visit += 1
        h = self.out_n(h)
        result = (self.B_u(h), self.B_A(h))
        return (*result, new_caches) if return_cache else result


class System(nn.Module):
    def __init__(self, foundation, core=None, adapters=None):
        super().__init__()
        self.foundation, self.core, self.adapters = foundation, core, adapters

    def context(self, x, docs):
        h = self.foundation.encode(x, docs)
        return h, self.foundation.pool(h), chunk_info(docs)[0]

    def plans(self, c, cd, z=None, lam=1.0):
        f = self.foundation
        if self.core is None or lam == 0:
            u = f.concept(c, cd)
            return f.pq(u), f.port(u), u
        u, a = self.core(c, cd, z, self.adapters)
        chat = f.pq(u)
        if lam < 1:
            old = f.concept(c, cd)
            chat, a = (1-lam)*f.pq(old)+lam*chat, (1-lam)*f.port(old)+lam*a
        return chat, a, u

    def forward(self, x, docs, z=None, lam=1.0):
        h, c, cd = self.context(x, docs)
        chat, a, u = self.plans(c, cd, z, lam)
        return self.foundation.decode(h, chat, a, docs)


def count(module):
    return sum(p.numel() for p in module.parameters())


def active_count(system):
    f = count(system.foundation)
    if system.core is None:
        return f
    return f - count(system.foundation.con) - count(system.foundation.con_norm) - count(system.foundation.W_A) + count(system.core) + (count(system.adapters) if system.adapters else 0)


def freeze(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module
