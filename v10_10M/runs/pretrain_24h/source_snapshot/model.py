"""Implementation of proposed_v10.md steps 1--32 (open choices in ARCHITECTURE.md).

Each batch row is one document/window. Right padding only; document packing is
deliberately unsupported. BOS is an observed token and counts toward groups.
All candidate branches preserve earlier selected code history.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, mask, rope):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.dh).unbind(2)
        q, k, v = [z.transpose(1, 2) for z in (q, k, v)]
        co, si = rope
        def rotate(z):
            a, b = z.chunk(2, -1)
            return torch.cat((a * co - b * si, a * si + b * co), -1)
        y = F.scaled_dot_product_attention(rotate(q), rotate(k), v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class FFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.up, self.down = nn.Linear(d, 4*d), nn.Linear(4*d, d)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.an, self.fn = RMSNorm(c.width), RMSNorm(c.width)
        self.attn, self.ffn = Attention(c.width, c.heads), FFN(c.width)

    def forward(self, x, mask, rope):
        x = x + self.attn(self.an(x), mask, rope)
        return x + self.ffn(self.fn(x))


class Reader(nn.Module):
    def __init__(self, c, writer=True):
        super().__init__()
        d = c.width
        self.n0, self.n1 = RMSNorm(d), RMSNorm(d)
        self.down = nn.Linear(2*d, c.gate_rank, bias=False)
        self.up = nn.Linear(c.gate_rank, 2*d)
        self.write = nn.Linear(2*d, 2) if writer else None
        nn.init.zeros_(self.up.weight)
        nn.init.constant_(self.up.bias, 2.)
        if writer:
            nn.init.zeros_(self.write.weight)
            # Asymmetry avoids identical stream dynamics at initialization.
            with torch.no_grad():
                self.write.bias.copy_(torch.tensor([-0.5, 0.5]))

    def forward(self, r0, r1):
        n0, n1 = self.n0(r0), self.n1(r1)
        j = torch.cat((n0, n1), -1)
        a0, a1 = self.up(F.silu(self.down(j))).sigmoid().chunk(2, -1)
        x = (a0*n0 + a1*n1) * .5
        return x, self.write(j).sigmoid() if self.write is not None else None


class TwoStreamBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ar, self.fr = Reader(c), Reader(c)
        self.attn, self.ffn = Attention(c.width, c.heads), FFN(c.width)

    def forward(self, r0, r1, mask, rope):
        x, beta = self.ar(r0, r1)
        y = self.attn(x, mask, rope)
        r0, r1 = r0 + beta[..., :1]*y, r1 + beta[..., 1:]*y
        x, beta = self.fr(r0, r1)
        y = self.ffn(x)
        return r0 + beta[..., :1]*y, r1 + beta[..., 1:]*y


class Engram(nn.Module):
    """Orders 2/3, two hashes each, context gate and dilation-3 causal conv.

    CPU sparse lookup avoids dense 20M-entry gradient/optimizer traffic on MPS.
    Only gathered rows cross the device boundary, with autograd intact.
    """
    def __init__(self, c):
        super().__init__()
        self.rows, self.dim = tuple(c.engram_rows), c.engram_dim
        self.offsets = np.cumsum([0] + list(self.rows[:-1]))
        self.table = nn.Embedding(sum(self.rows), c.engram_dim, sparse=c.engram_cpu)
        nn.init.normal_(self.table.weight, std=.02)
        self.key = nn.Linear(4*c.engram_dim, c.width, bias=False)
        self.value = nn.Linear(4*c.engram_dim, c.width, bias=False)
        self.qn, self.kn, self.vn = [RMSNorm(c.width) for _ in range(3)]
        self.conv = nn.Parameter(torch.zeros(4, c.width))
        self.register_buffer('canonical', torch.arange(c.vocab_size), persistent=True)

    def indices(self, ids):
        # Dataset workers can precompute this once per CPU batch.
        x = ids.detach().cpu().numpy().astype(np.int64)
        canon = self.canonical.detach().cpu().numpy()
        x = canon[x]
        result = []
        for h, (order, rows) in enumerate(zip((2, 2, 3, 3), self.rows)):
            z = np.zeros_like(x)
            for lag in range(order):
                prev = np.zeros_like(x)
                if lag == 0:
                    prev[:] = x + 1
                else:
                    prev[:, lag:] = x[:, :-lag] + 1
                z ^= prev * (1000003 + 104729*h + 7919*lag)
            result.append(z % rows + self.offsets[h])
        return torch.from_numpy(np.stack(result, -1))

    def forward(self, h, ids, indices=None):
        indices = self.indices(ids) if indices is None else indices
        e = self.table(indices.to(self.table.weight.device)).flatten(-2).to(h.device)
        k, v = self.key(e), self.value(e)
        gate = ((self.qn(h) * self.kn(k)).sum(-1, keepdim=True) / math.sqrt(h.shape[-1])).sigmoid()
        v = gate * v
        vn = self.vn(v)
        conv = vn * self.conv[0]
        for i in range(1, 4):
            lag = 3*i
            if lag < h.shape[1]:
                conv = conv + F.pad(vn[:, :-lag] * self.conv[i], (0, 0, lag, 0))
        return h + v + F.silu(conv)


class Pooler(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c.width
        self.c = c
        self.slot = nn.Parameter(torch.randn(2, d)*.02)
        self.pos = nn.Parameter(torch.randn(8, d)*.02)
        self.qn, self.mn, self.fn, self.outn = [RMSNorm(d) for _ in range(4)]
        self.q, self.k, self.v, self.o = [nn.Linear(d, d, bias=False) for _ in range(4)]
        self.ffn, self.alpha = FFN(d), nn.Parameter(torch.tensor(.1))

    def forward(self, h):
        b, t, d = h.shape
        g = t // 8
        if not g:
            return h.new_empty(b, 0, 2, d)
        x = h[:, :g*8].reshape(b*g, 8, d)
        anchor = x.reshape(b*g, 2, 4, d).mean(2)
        mem = self.mn(x) + self.pos
        def split(z):
            return z.reshape(b*g, -1, self.c.heads, d//self.c.heads).transpose(1, 2)
        q, k, v = split(self.q(self.qn(anchor)+self.slot)), split(self.k(mem)), split(self.v(mem))
        delta = self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b*g, 2, d))
        z = anchor + self.alpha*delta
        return self.outn(z + self.ffn(self.fn(z))).reshape(b, g, 2, d)


def geometry(t, lengths, device, heads_dim, window=None, group_slots=1):
    pos = torch.arange(t, device=device)
    units = pos // group_slots
    allowed = units[None, :] <= units[:, None]
    if window:
        allowed = allowed & ((pos[:, None]-pos[None, :]) < window)
    valid = pos[None, :] < lengths[:, None]
    mask = allowed[None, None] & valid[:, None, None, :]
    # Empty padded rows get a self key, then are ignored by all losses/valid reads.
    mask = mask | ((~valid)[:, None, :, None] & torch.eye(t, device=device, dtype=torch.bool)[None, None])
    angle = pos.float()[:, None] / (10000 ** (torch.arange(heads_dim//2, device=device).float()/(heads_dim//2)))
    return mask, (angle.cos(), angle.sin())


class ConceptCore(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.codes = nn.Embedding(c.codes, c.code_width)
        self.code_projection = nn.Linear(c.code_width, c.width, bias=False)
        nn.init.normal_(self.codes.weight, std=.1)
        self.anchor_norm, self.exit_norm = RMSNorm(c.width), RMSNorm(c.width)
        self.anchor_scale = nn.Parameter(torch.tensor(.25))
        self.layers = nn.ModuleList([Block(c) for _ in range(c.concept_layers)])
        # Prefix-only choice from both ordered, code-free observed slots.
        self.selector_norm = RMSNorm(2*c.width)
        self.selector = nn.Linear(2*c.width, c.codes)

    def select(self, pooled):
        return self.selector(self.selector_norm(pooled.flatten(-2)))

    def forward(self, pooled, lengths, codes, xi=None):
        b, g, p, d = pooled.shape
        if not g:
            return pooled
        e = self.codes(codes)
        if xi is not None:
            if self.training and bool(torch.count_nonzero(xi)):
                raise ValueError('Pretraining requires temporary GradCuit xi = 0')
            e = e + xi
        x = (pooled + self.code_projection(e).unsqueeze(-2)).reshape(b, g*p, d)
        a = self.anchor_norm(x)
        state = a
        mask, rope = geometry(g*p, lengths//8*p, x.device, d//self.c.heads, group_slots=p)
        for _ in range(self.c.concept_visits):
            state = state + self.anchor_scale*a
            for layer in self.layers:
                state = layer(state, mask, rope)
            state = self.exit_norm(state)
        return state.reshape(b, g, p, d)


class PQHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.logits = nn.Linear(c.width, c.pq_segments*c.pq_words)
        self.codebook = nn.Parameter(torch.randn(c.pq_segments, c.pq_words, c.width//c.pq_segments))

    def forward(self, u):
        c = self.c
        probs = self.logits(u).reshape(*u.shape[:-1], c.pq_segments, c.pq_words).softmax(-1)
        # Broadcasted matmul supports MPS and preserves all head/code gradients.
        out = torch.einsum('...sn,snd->...sd', probs, self.codebook)
        return out.flatten(-2), probs

    def fitting_loss(self, target, valid):
        c = self.c
        x = target.detach().reshape(*target.shape[:-1], c.pq_segments, c.width//c.pq_segments)
        dist = (x.unsqueeze(-2)-self.codebook).square().sum(-1)
        # Mean segment squared Euclidean distance, NOT mean coordinate distance.
        best = dist.min(-1).values.mean((-1, -2))
        return (best*valid).sum()/valid.sum().clamp_min(1)


class Decoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.fusion = nn.Parameter(torch.full((c.width,), .1))
        self.src_norm = nn.ModuleList([RMSNorm(c.width) for _ in range(2)])
        self.src_proj = nn.ModuleList([nn.Linear(c.width, c.width, bias=False) for _ in range(2)])
        self.route_norm = nn.ModuleList([RMSNorm(c.width) for _ in range(c.decoder_layers)])
        self.routes = nn.ModuleList([nn.Linear(c.width, 2) for _ in range(c.decoder_layers)])
        self.scales = nn.Parameter(torch.full((c.decoder_layers, c.width), .1))
        self.layers = nn.ModuleList([Block(c) for _ in range(c.decoder_layers)])
        self.outn = RMSNorm(c.width)

    @staticmethod
    def align(forecast, t):
        b, g, p, d = forecast.shape
        repeated = forecast.flatten(1, 2).repeat_interleave(4, dim=1)
        # Zero-based row 7 is the first legal row; it predicts token index 8.
        return F.pad(repeated, (0, 0, 7, max(0, t-7-8*g)))[:, :t]

    def forward(self, early, final, forecast, mask, rope, ablate=False):
        guidance = self.align(forecast, final.shape[1])
        if ablate:
            guidance = torch.zeros_like(guidance)
        x = final + self.fusion*guidance
        sources = torch.stack([p(n(h)) for p, n, h in zip(self.src_proj, self.src_norm, (early, final))], -2)
        for i, layer in enumerate(self.layers):
            weights = self.routes[i](self.route_norm[i](x)).softmax(-1)
            x = x + self.scales[i]*(sources*weights.unsqueeze(-1)).sum(-2)
            x = layer(x, mask, rope)
        return self.outn(x)


class V10(nn.Module):
    def __init__(self, c=None):
        super().__init__()
        self.c = c or ModelConfig()
        self.c.validate()
        c = self.c
        self.embedding = nn.Embedding(c.vocab_size, c.width)
        nn.init.normal_(self.embedding.weight, std=.02)
        self.engram = Engram(c)
        self.e1 = nn.ModuleList([Block(c) for _ in range(c.e1_layers)])
        self.e2 = nn.ModuleList([TwoStreamBlock(c) for _ in range(c.e2_layers)])
        self.exit_reader = Reader(c, writer=False)
        self.e3 = nn.ModuleList([Block(c) for _ in range(c.e3_layers)])
        self.pooler, self.core, self.head, self.decoder = Pooler(c), ConceptCore(c), PQHead(c), Decoder(c)

    def place(self, device):
        self.to(device)
        if self.c.engram_cpu:
            self.engram.table.cpu()
        return self

    def counts(self):
        total = sum(p.numel() for p in self.parameters())
        memory = self.engram.table.weight.numel()
        return dict(dense=total-memory, engram=memory, total=total,
                    encoder_distinct=len(self.e1)+len(self.e2)+len(self.e3),
                    encoder_executed=len(self.e1)+2*len(self.e2)+len(self.e3),
                    concept_distinct=len(self.core.layers), concept_executed=2*len(self.core.layers))

    def encode(self, ids, lengths, indices=None):
        d = self.c.width
        full, rope = geometry(ids.shape[1], lengths, ids.device, d//self.c.heads)
        local, _ = geometry(ids.shape[1], lengths, ids.device, d//self.c.heads, self.c.window)
        x = self.embedding(ids)
        for i, layer in enumerate(self.e1):
            x = layer(x, full if i%4 == 3 else local, rope)
            if i == 0:
                x = self.engram(x, ids, indices)
        early = x
        r0, r1 = x, x
        for _ in range(2):
            for i, layer in enumerate(self.e2):
                r0, r1 = layer(r0, r1, full if (len(self.e1)+i)%4 == 3 else local, rope)
        x, _ = self.exit_reader(r0, r1)
        for i, layer in enumerate(self.e3):
            x = layer(x, full if (len(self.e1)+len(self.e2)+i)%4 == 3 else local, rope)
        return early, x, full, rope

    def forward(self, ids, lengths=None, indices=None, codes=None, xi=None,
                bank=None, write_pass=False, ablate=False):
        if lengths is None:
            lengths = torch.full((ids.shape[0],), ids.shape[1], device=ids.device, dtype=torch.long)
        early, h, mask, rope = self.encode(ids, lengths, indices)
        pooled = self.pooler(h)
        selector = self.core.select(pooled)
        selected = selector.argmax(-1)
        if codes is None:
            codes = selected
        elif codes.shape[1] < selected.shape[1]:
            codes = torch.cat((codes, selected[:, codes.shape[1]:]), 1)
        u = self.core(pooled, lengths, codes, xi)
        correction, queries = None, None
        if bank is not None:
            queries = u.flatten(-2)/math.sqrt(2*self.c.width)
            correction = bank.read(queries, codes)
            if write_pass:
                correction = correction.detach().requires_grad_(True)
            u = u + correction.reshape_as(u)
        forecast, probs = self.head(u)
        hidden = self.decoder(early, h, forecast, mask, rope, ablate)
        # Token embedding and vocabulary head are tied.
        logits = F.linear(hidden, self.embedding.weight)
        valid = torch.arange(pooled.shape[1], device=ids.device)[None, :] < (lengths//8)[:, None]
        return dict(logits=logits, pooled=pooled, forecast=forecast, selector=selector,
                    codes=codes, valid_groups=valid, probs=probs, u=u,
                    correction=correction, queries=queries)

    def candidates(self, pooled, lengths, codes, boundary):
        """All K choices at one sampled boundary per document, with common past.

        This avoids K complete token encoder/decoder passes. A vector of
        boundaries permits different valid lengths within the same batch.
        """
        b, _, p, d = pooled.shape
        if isinstance(boundary,int):
            boundary=torch.full((b,),boundary,device=pooled.device,dtype=torch.long)
        k, n = self.c.codes, int(boundary.max().item())+1
        x = pooled[:, :n].unsqueeze(1).expand(b, k, n, p, d).reshape(b*k, n, p, d)
        history = codes[:, :n].unsqueeze(1).expand(b, k, n).clone()
        rows=torch.arange(b,device=pooled.device)
        history[rows[:,None],torch.arange(k,device=pooled.device)[None,:],boundary[:,None]]=torch.arange(k,device=pooled.device)[None,:]
        used_lengths=torch.minimum(lengths,(boundary+1)*8).repeat_interleave(k)
        u = self.core(x, used_lengths, history.reshape(b*k, n))
        current=u[torch.arange(b*k,device=pooled.device),boundary.repeat_interleave(k)]
        return self.head(current)[0].reshape(b, k, p, d)


def losses(model, output, targets, target_mask, boundary=None, concept_weight=.1,
           vq_weight=.01, selector_weight=.05):
    logits = output['logits']
    safe_targets = targets.masked_fill(~target_mask, -100)
    n = target_mask.sum()
    token = F.cross_entropy(logits.flatten(0, 1), safe_targets.flatten(), ignore_index=-100, reduction='sum')/n.clamp_min(1)
    pooled = output['pooled'].detach()  # Online, code-free stopped target; no hidden EMA claim.
    valid = output['valid_groups']
    zero = logits.sum()*0
    concept, explore, selector, vq = zero, zero, zero, zero
    if pooled.shape[1] > 1:
        pairs = valid[:, 1:]
        err = (output['forecast'][:, :-1]-pooled[:, 1:]).square().mean((-1, -2))
        concept = (err*pairs).sum()/pairs.sum().clamp_min(1)
        vq = model.head.fitting_loss(pooled[:, 1:], pairs)
        if boundary is not None:
            lengths = valid.sum(-1)*8
            pred = model.candidates(output['pooled'], lengths, output['codes'], boundary)
            if isinstance(boundary,int):
                boundary=torch.full((len(pooled),),boundary,device=pooled.device,dtype=torch.long)
            rows=torch.arange(len(pooled),device=pooled.device)
            eligible=valid.sum(-1)>1
            errors = (pred-pooled[rows,boundary+1,None]).square().mean((-1, -2))
            explore = (errors.min(-1).values*eligible).sum()/eligible.sum().clamp_min(1)
            labels = errors.detach().argmin(-1)
            classification=F.cross_entropy(output['selector'][rows,boundary],labels,reduction='none')
            selector=(classification*eligible).sum()/eligible.sum().clamp_min(1)
    total = token + concept_weight*(concept+explore) + vq_weight*vq + selector_weight*selector
    return dict(total=total, token=token, concept=concept, explore=explore, vq=vq, selector=selector)
