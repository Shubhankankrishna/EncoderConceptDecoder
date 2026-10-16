"""V6 tiny: document-isolated NCP, deterministic recurrence, zero-anchored Z.

No imports from mutable V5 code. FP32 is the reference execution mode.
"""
import math
import copy
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
    def __init__(self, ff=FF):
        super().__init__()
        self.banks = nn.ModuleList()
        self.capture = None
        self.n1, self.n2 = nn.RMSNorm(D), nn.RMSNorm(D)
        self.qkv, self.o = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)
        self.w12, self.w3 = nn.Linear(D, 2 * ff, bias=False), nn.Linear(ff, D, bias=False)

    def forward(self, x, docs, pos=None, cache=None, return_cache=False):
        b, t, _ = x.shape
        if pos is None:
            pos = positions(docs)
        a = self.n1(x)
        gate_state = cache[3] if cache is not None and len(cache)>3 else None
        qkv, next_gate = self.project(a, docs, gate_state)
        q, k, v = qkv.view(b, t, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        q, k = rotate(q, pos), rotate(k, pos)
        offset = 0
        kd = docs
        if cache is not None:
            old_k, old_v, old_docs = cache[:3]
            offset = old_k.shape[2]
            k, v = torch.cat([old_k, k], 2), torch.cat([old_v, v], 2)
            kd = torch.cat([old_docs, docs], 1)
        causal = torch.arange(k.shape[2], device=x.device)[None, :] <= (offset + torch.arange(t, device=x.device))[:, None]
        mask = (docs[:, :, None] == kd[:, None, :]) & causal
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        x = x + self.o(a.transpose(1, 2).reshape(b, t, D))
        g, u = self.w12(self.n2(x)).chunk(2, -1)
        x = x + self.w3(F.silu(g) * u)
        return (x, (k, v, kd, next_gate)) if return_cache else x

    def static_step(self,x,pos,cache):
        """Single-document inference only; fixed-size caches avoid length recompilation."""
        b=x.shape[0];qkv,ng=self.project(self.n1(x),torch.zeros(b,1,dtype=torch.long,device=x.device),cache[2] if len(cache)>2 else None);q,k,v=qkv.view(b,1,3,HEADS,D//HEADS).permute(2,0,3,1,4)
        p=pos.reshape(1,1).expand(b,1);q,k=rotate(q,p),rotate(k,p)
        kc,vc=cache[:2];
        if len(cache)>2: cache[2]=ng
        index=pos.reshape(1,1,1,1).expand(b,HEADS,1,D//HEADS)
        kc.scatter_(2,index,k);vc.scatter_(2,index,v)
        allowed=torch.arange(kc.shape[2],device=x.device)<=pos
        a=F.scaled_dot_product_attention(q,kc,vc,attn_mask=allowed[None,None,None,:])
        x=x+self.o(a.transpose(1,2).reshape(b,1,D))
        g,u=self.w12(self.n2(x)).chunk(2,-1)
        return x+self.w3(F.silu(g)*u)


def chunk_info(docs):
    c = docs[:, :docs.shape[1]//K*K].reshape(docs.shape[0], -1, K)
    valid = (c >= 0).all(-1) & (c == c[..., :1]).all(-1)
    return torch.where(valid, c[..., 0], torch.full_like(c[..., 0], -1)), valid


def shift(v, docs):
    b,t=docs.shape
    if v.shape[1]==0:return v.new_zeros(b,t,D)
    j=(torch.arange(t,device=docs.device)+1)//K-1
    idx=j.clamp(0,v.shape[1]-1)
    cd,_=chunk_info(docs)
    allowed=(j[None]>=0)&(cd[:,idx]==docs)&(docs>=0)
    return v[:,idx]*allowed.unsqueeze(-1)


class Foundation(nn.Module):
    def __init__(self, depth=4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.enc = nn.ModuleList([Block() for _ in range(3)])
        self.enc_norm = nn.RMSNorm(D)
        self.con = nn.ModuleList([Block() for _ in range(depth)])
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
        return rms(h[:, :h.shape[1]//K*K].reshape(h.shape[0], -1, K, D).mean(2))

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
            self.G = nn.ModuleList([nn.Linear(D, rank, bias=False) for _ in range(2)])
            self.V = nn.ModuleList([nn.Linear(dz, rank, bias=False) for _ in range(2)])
            self.U = nn.ModuleList([nn.Linear(rank, D, bias=False) for _ in range(2)])
        elif kind == 'direct':
            self.U = nn.ModuleList([nn.Linear(dz, D, bias=False) for _ in range(2)])
        else:
            raise ValueError(kind)
        self.apply(init)
        if kind == 'gated':
            for layer in self.G:
                nn.init.zeros_(layer.weight)
            for layer in list(self.U)+list(self.V):nn.init.normal_(layer.weight,std=.08)

    def raw(self, e, z, q):
        if self.kind == 'direct':
            return self.U[q](z)
        return self.U[q]((1 + 0.5 * self.G[q](rms(e)).tanh()) * self.V[q](z))

    def forward(self, e, z):
        writes = []
        for q in range(2):
            raw = self.raw(e, z, q)
            writes.append(self.cap * safe_rms(e) * raw / (1+raw.float().square().mean(-1,keepdim=True)).sqrt())
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


class DenseCore(nn.Module):
    def __init__(self, f):
        super().__init__();self.blocks=copy.deepcopy(f.con);self.norm=copy.deepcopy(f.con_norm);self.W_A=copy.deepcopy(f.W_A)
    def forward(self,c,docs,z=None,adapters=None,caches=None,return_cache=False,pos=None):
        nc=[]
        for i,block in enumerate(self.blocks):
            if adapters is not None and z is not None and i in (0,len(self.blocks)//2):c=c+adapters(c,z)[0 if i==0 else 1]
            out=block(c,docs,pos,caches[i] if caches else None,return_cache)
            if return_cache:c,cache=out;nc.append(cache)
            else:c=out
        u=self.norm(c);result=(u,self.W_A(rms(u)))
        return (*result,nc) if return_cache else result

class Flat(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(VOCAB,D);self.blocks=nn.ModuleList([Block(400 if i<9 else FF) for i in range(10)]);self.nf=nn.RMSNorm(D);self.apply(init)
    def forward(self,x,docs):
        h=self.emb(x)
        for b in self.blocks:h=b(h,docs)
        return self.nf(h)@self.emb.weight.t()

class System(nn.Module):
    def __init__(self, foundation=None, core=None, adapters=None):
        super().__init__();self.foundation=foundation if foundation is not None else Foundation();self.core=core;self.adapters=adapters
    @property
    def is_flat(self):return isinstance(self.foundation,Flat)
    def context(self,x,docs):
        h=self.foundation.encode(x,docs);return h,self.foundation.pool(h),chunk_info(docs)[0]
    def plans(self,c,cd,z=None,lam=1.0):
        f=self.foundation
        if c.shape[1]==0:return c,c,c
        if self.core is None:
            if self.adapters is None or z is None:u=f.concept(c,cd);a=f.port(u)
            else:u,a=DenseView(f)(c,cd,z,self.adapters)
        else:u,a=self.core(c,cd,z,self.adapters)
        if lam<1 and self.core is not None:
            old=f.concept(c,cd);u=(1-lam)*old+lam*u;a=(1-lam)*f.port(old)+lam*a
        return f.pq(u),a,u
    def forward(self,x,docs,z=None,lam=1.0,details=False):
        if self.is_flat:return self.foundation(x,docs)
        h,c,cd=self.context(x,docs);chat,a,u=self.plans(c,cd,z,lam)
        logits=self.foundation.decode(h,chat,a,docs)
        return (logits,c,chat,u,a) if details else logits
    def active_blocks(self):
        f=self.foundation
        if self.is_flat:return list(f.blocks)
        core=list(self.core.blocks) if isinstance(self.core,DenseCore) else ([self.core.fL,self.core.fH] if self.core else list(f.con))
        return list(f.enc)+core+list(f.dec)
    def add_phase(self,rho=1.,force=True):
        if len(self.active_blocks()[0].banks)>=8:raise ValueError('Continual bank cap is eight; consolidation requires a separately tested protocol')
        for i,b in enumerate(self.active_blocks()):
            if not hasattr(b,'projection'):
                generator=torch.Generator().manual_seed(88000+i)
                b.register_buffer('projection',torch.randn(32,D,generator=generator).to(b.qkv.weight.device)/32**.5)
                b.register_buffer('reference_var',torch.ones(32,device=b.qkv.weight.device));b.register_buffer('neg_mu',torch.zeros(8,32,device=b.qkv.weight.device));b.register_buffer('neg_var',torch.ones(8,32,device=b.qkv.weight.device));b.register_buffer('neg_logw',torch.full((8,),-math.log(8),device=b.qkv.weight.device))
            b.banks.append(LoRABank(rho).to(b.qkv.weight.device));b.banks[-1].force=force
    def metadata(self):
        blocks=self.active_blocks();return dict(gate_buffers=hasattr(blocks[0],'projection'),flat=self.is_flat,depth=0 if self.is_flat else len(self.foundation.con),core='recurrent' if isinstance(self.core,Core) else 'dense' if self.core else None,adapter=self.adapters.kind if self.adapters else None,phases=len(blocks[0].banks),rho=[v.rho for v in blocks[0].banks],forces=[v.force for v in blocks[0].banks])

class DenseView:
    def __init__(self,f):self.f=f
    def __call__(self,c,docs,z=None,adapters=None,caches=None,return_cache=False,pos=None):
        f=self.f;nc=[]
        for i,b in enumerate(f.con):
            if adapters is not None and z is not None and i in (0,len(f.con)//2):c=c+adapters(c,z)[0 if i==0 else 1]
            out=b(c,docs,pos,caches[i] if caches else None,return_cache)
            if return_cache:c,cache=out;nc.append(cache)
            else:c=out
        u=f.con_norm(c);result=(u,f.port(u))
        return (*result,nc) if return_cache else result

def count(module):
    return sum(p.numel() for p in module.parameters())


def active_count(system):
    f = count(system.foundation)
    if system.core is None:
        return f + (count(system.adapters) if system.adapters else 0)
    return f - count(system.foundation.con) - count(system.foundation.con_norm) - count(system.foundation.W_A) + count(system.core) + (count(system.adapters) if system.adapters else 0)


def freeze(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def log_density(h,mu,var,logw):
    return torch.logsumexp(logw-.5*((h[...,None,:]-mu).square()/var+var.log()+math.log(2*math.pi)).sum(-1),-1)

class LoRABank(nn.Module):
    def __init__(self,rho=1.):
        super().__init__();self.Aq=nn.Parameter(torch.randn(4,D)*.02);self.Bq=nn.Parameter(torch.zeros(D,4));self.Av=nn.Parameter(torch.randn(4,D)*.02);self.Bv=nn.Parameter(torch.zeros(D,4));self.rho=rho;self.force=True
        self.register_buffer('pos_mu',torch.zeros(4,32));self.register_buffer('pos_var',torch.ones(4,32));self.register_buffer('pos_logw',torch.full((4,),-math.log(4)))
    def delta(self,a):return 2*F.linear(F.linear(a,self.Aq),self.Bq),2*F.linear(F.linear(a,self.Av),self.Bv)

def project(self,a,docs,state=None):
    qkv=self.qkv(a)
    if self.capture is not None:self.capture(a.detach(),docs)
    if not self.banks:return qkv,None
    h=F.linear(a,self.projection);q,k,v=qkv.chunk(3,-1);ng=[]
    for i,bank in enumerate(self.banks):
        if bank.force is True:g=(docs>=0).float();last=g[:,-1]
        elif bank.force is False:g=torch.zeros_like(docs,dtype=a.dtype);last=g[:,-1]
        else:
            hit=(log_density(h,bank.pos_mu,bank.pos_var,bank.pos_logw)-log_density(h,self.neg_mu,self.neg_var,self.neg_logw)>0).float()*(docs>=0)
            if bank.rho==1.:g=hit;last=g[:,-1]
            else:
                previous=torch.zeros(len(a),device=a.device) if state is None else state[:,i];values=[]
                for t in range(a.shape[1]):
                    reset=(docs[:,t]<0) | ((docs[:,t]!=docs[:,t-1]) if t else torch.zeros(len(a),device=a.device,dtype=torch.bool))
                    previous=torch.where(reset,torch.zeros_like(previous),previous)
                    previous=bank.rho*hit[:,t]+(1-bank.rho)*previous;values.append(previous)
                last=previous;g=(torch.stack(values,1)>.5).float()*(docs>=0)
        dq,dv=bank.delta(a);q=q+g[...,None]*dq;v=v+g[...,None]*dv;ng.append(last)
    return torch.cat([q,k,v],-1),torch.stack(ng,1)

Block.project=project
