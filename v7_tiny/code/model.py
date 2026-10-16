"""V7 reference: causal distinct stacks, bounded IRC/CRC, soft PQ, exact-zero Z."""
from dataclasses import dataclass,asdict
import math,copy
import torch
from torch import nn
from torch.nn import functional as F

@torch.compiler.disable
def training_attention(q,k,v,mask):
    # Keep MPS SDPA backward outside Inductor: fused backward corrupted K-norm gradients.
    return F.scaled_dot_product_attention(q,k,v,attn_mask=mask)

@dataclass
class Config:
    d:int=192
    heads:int=6
    ff:int=576
    le:int=3
    lc:int=4
    ld:int=3
    vocab:int=2048
    k:int=4
    segments:int=12
    codes:int=64
    dz:int=48
    rank:int=48
    routing:bool=True
    forecast:bool=True
    continuous:bool=True
    kind:str='hierarchy'
    flat_layers:int=12
    flat_ff:int=464
    def __post_init__(self):
        if self.d%self.heads or (self.d//self.heads)%2 or self.d%self.segments:raise ValueError('Width must divide heads/segments with even rotary head width')
        if min(self.le,self.lc,self.ld)<2 or min(self.ff,self.flat_ff-16,self.dz,self.rank,self.k,self.codes)<1:raise ValueError('Invalid stack or control dimensions')

def norm(x):return x*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)
def scale(x):return (x.float().square().mean(-1,keepdim=True)+1e-6).sqrt()
def ball(z,bound=2.):return z*(bound/z.float().square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()).clamp_max(1)
def freeze(m):
    for p in m.parameters():p.requires_grad_(False)
    return m
def count(m):return sum(p.numel() for p in m.parameters())
def positions(docs):
    t=docs.shape[1];idx=torch.arange(t,device=docs.device).expand_as(docs)
    first=torch.ones_like(docs,dtype=torch.bool);first[:,1:]=docs[:,1:]!=docs[:,:-1]
    starts=torch.where(first,idx,torch.zeros_like(idx)).cummax(1).values
    return idx-starts
def chunks(docs,k=4):
    v=docs[:,:docs.shape[1]//k*k].view(len(docs),-1,k);valid=(v>=0).all(-1)&(v==v[...,:1]).all(-1)
    return torch.where(valid,v[...,0],torch.full_like(v[...,0],-1)),valid
def shift(v,docs,k=4):
    t=docs.shape[1];cd,_=chunks(docs,k)
    if v.shape[1]==0:return v.new_zeros(len(docs),t,v.shape[-1])
    j=(torch.arange(t,device=docs.device)+1)//k-1;index=j.clamp(0,v.shape[1]-1)
    rep=v[:,index];source=cd[:,index]
    return rep*((source==docs)&(docs>=0)&(j>=0)[None]).unsqueeze(-1)

class Block(nn.Module):
    def __init__(self,cfg,depth,ff=None):
        super().__init__();d=cfg.d;self.d=d;self.heads=cfg.heads;self.hd=d//cfg.heads
        self.n1=nn.RMSNorm(d);self.n2=nn.RMSNorm(d)
        self.qkv=nn.Linear(d,3*d,bias=False);self.o=nn.Linear(d,d,bias=False)
        self.w12=nn.Linear(d,2*(ff or cfg.ff),bias=False);self.w3=nn.Linear(ff or cfg.ff,d,bias=False)
        self.qn=nn.RMSNorm(self.hd);self.kn=nn.RMSNorm(self.hd)
        sigma=.02*math.sqrt(128/d)
        for layer in (self.qkv,self.w12):nn.init.normal_(layer.weight,std=sigma)
        for layer in (self.o,self.w3):nn.init.normal_(layer.weight,std=sigma/math.sqrt(2*depth))
    def rotate(self,q,pos):
        inv=1/(10000**(torch.arange(0,self.hd,2,device=q.device).float()/self.hd))
        a=pos.float().unsqueeze(-1)*inv;a=torch.cat([a,a],-1).unsqueeze(1)
        rot=torch.cat([-q[...,self.hd//2:],q[...,:self.hd//2]],-1)
        return q*a.cos()+rot*a.sin()
    def forward(self,x,docs,pos=None,cache=None,return_cache=False):
        b,t,_=x.shape;pos=positions(docs) if pos is None else pos
        q,k,v=self.qkv(self.n1(x)).view(b,t,3,self.heads,self.hd).permute(2,0,3,1,4)
        q,k=self.rotate(self.qn(q),pos),self.rotate(self.kn(k),pos)
        offset=0;kd=docs
        if cache is not None:
            oldk,oldv,oldd=cache;offset=oldk.shape[2]
            k,v=torch.cat([oldk,k],2),torch.cat([oldv,v],2);kd=torch.cat([oldd,docs],1)
        causal=torch.arange(k.shape[2],device=x.device)[None,:] <= (offset+torch.arange(t,device=x.device))[:,None]
        mask=(docs[:,:,None]==kd[:,None,:])&causal
        # Padding queries remain mathematically finite and are never scored/exported.
        a=training_attention(q,k,v,mask[:,None]) if torch.is_grad_enabled() else F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None])
        x=x+self.o(a.transpose(1,2).reshape(b,t,self.d));g,u=self.w12(self.n2(x)).chunk(2,-1)
        out=x+self.w3(F.silu(g)*u)
        return (out,(k,v,kd)) if return_cache else out
    def static_step(self,x,pos,cache):
        b=x.shape[0];q,k,v=self.qkv(self.n1(x)).view(b,1,3,self.heads,self.hd).permute(2,0,3,1,4)
        p=pos.reshape(1,1).expand(b,1);q,k=self.rotate(self.qn(q),p),self.rotate(self.kn(k),p)
        kc,vc=cache;index=pos.reshape(1,1,1,1).expand(b,self.heads,1,self.hd)
        kc.scatter_(2,index,k);vc.scatter_(2,index,v)
        allowed=torch.arange(kc.shape[2],device=x.device)<=pos
        a=F.scaled_dot_product_attention(q,kc,vc,attn_mask=allowed[None,None,None,:])
        x=x+self.o(a.transpose(1,2).reshape(b,1,self.d));g,u=self.w12(self.n2(x)).chunk(2,-1)
        return x+self.w3(F.silu(g)*u)

class IRC(nn.Module):
    def __init__(self,d,depth):
        super().__init__();self.router=nn.Linear(d,3);self.a=nn.Parameter(torch.zeros(d));self.limit=.25/math.sqrt(depth)
        nn.init.zeros_(self.router.weight);nn.init.zeros_(self.router.bias);self.router.bias.data[2]=2
    def forward(self,y,initial,group=None):
        scores=self.router(norm(y))
        if group is None:scores=scores+scores.new_tensor([0.,-1e9,0.]);group=y
        # Exactly coincident group/current at group completion is never passed as a bank entry.
        w=scores.float().softmax(-1);mix=w[...,0:1]*norm(initial)+w[...,1:2]*norm(group)+w[...,2:3]*norm(y)
        return y+self.limit*self.a.tanh()*scale(y)*(mix-norm(y))

class CRC(nn.Module):
    def __init__(self,d,depth):
        super().__init__();self.router=nn.Linear(d,2);self.a=nn.Parameter(torch.full((d,),math.atanh(.04)))
        self.limit=.5/math.sqrt(depth);nn.init.zeros_(self.router.weight);nn.init.zeros_(self.router.bias)
    def forward(self,t,sources):
        w=self.router(norm(t)).float().softmax(-1)
        return t+self.limit*self.a.tanh()*(w[...,0:1]*norm(sources[0])+w[...,1:2]*norm(sources[1]))

class Stack(nn.Module):
    def __init__(self,cfg,depth,incoming=False):
        super().__init__();self.depth=depth;self.blocks=nn.ModuleList([Block(cfg,depth) for _ in range(depth)])
        self.ircs=nn.ModuleList([IRC(cfg.d,depth) for _ in range(depth)]) if cfg.routing else None
        self.crcs=nn.ModuleList([CRC(cfg.d,depth) for _ in range(depth)]) if incoming and cfg.routing else None
        self.exports=nn.ModuleList([nn.RMSNorm(cfg.d),nn.RMSNorm(cfg.d)])
        self.final_norm=nn.RMSNorm(cfg.d)
    def forward(self,x,docs,sources=None,z=None,adapters=None,pos=None,caches=None,return_cache=False,static=False):
        initial=x;group=None;outs=[];new=[];mid=math.ceil(self.depth/2)
        for i,block in enumerate(self.blocks):
            if self.crcs is not None:x=self.crcs[i](x,sources)
            if adapters is not None and i in (0,self.depth//2):x=adapters(x,z,0 if i==0 else 1)
            if static:y=block.static_step(x,pos,caches[i])
            elif return_cache:y,c=block(x,docs,pos,None if caches is None else caches[i],True);new.append(c)
            else:y=block(x,docs,pos)
            x=self.ircs[i](y,initial,group) if self.ircs is not None else y
            if (i+1)%4==0:group=x
            if i+1 in (mid,self.depth):outs.append(x)
        exports=[self.exports[i](v) for i,v in enumerate(outs)]
        return (self.final_norm(x),exports,new) if return_cache else (self.final_norm(x),exports)

class Adapters(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;self.cap=.1
        self.G=nn.ModuleList([nn.Linear(cfg.d,cfg.rank,bias=False) for _ in range(2)])
        self.V=nn.ModuleList([nn.Linear(cfg.dz,cfg.rank,bias=False) for _ in range(2)])
        self.U=nn.ModuleList([nn.Linear(cfg.rank,cfg.d,bias=False) for _ in range(2)])
        for g in self.G:nn.init.zeros_(g.weight)
        for l in list(self.U)+list(self.V):nn.init.normal_(l.weight,std=.08)
    def write(self,v,z,site):
        a=self.U[site]((1+.5*self.G[site](norm(v)).tanh())*self.V[site](z))
        return self.cap*scale(v)*a/torch.sqrt(1+a.float().square().mean(-1,keepdim=True))
    def forward(self,v,z,site):return v+self.write(v,z,site)

class Backbone(nn.Module):
    def __init__(self,cfg=None):
        super().__init__();self.cfg=cfg or Config();c=self.cfg
        self.emb=nn.Embedding(c.vocab,c.d);nn.init.normal_(self.emb.weight,std=.02*math.sqrt(128/c.d))
        self.enc=Stack(c,c.le);self.con=Stack(c,c.lc,True)
        self.head=nn.Linear(c.d,c.segments*c.codes);nn.init.normal_(self.head.weight,std=.02*math.sqrt(128/c.d));nn.init.zeros_(self.head.bias)
        self.Q=nn.Parameter(torch.randn(c.segments,c.codes,c.d//c.segments)*.25)
        self.dec=Stack(c,c.ld)
        # Decoder exports are not consumed; do not count unused learned norm weights.
        self.dec.exports=nn.ModuleList([])
        self.ed=nn.ModuleList([CRC(c.d,c.ld) for _ in range(c.ld)]) if c.routing else None
        self.cd=nn.ModuleList([CRC(c.d,c.ld) for _ in range(c.ld)]) if c.routing else None
        self.forecast_a=nn.Parameter(torch.full((c.ld,c.d),math.atanh(.2)))
        self.register_buffer('usage',torch.zeros(c.segments,c.codes));self.register_buffer('q_initialized',torch.tensor(False))
    def pool(self,h):return norm(h.view(len(h),-1,self.cfg.k,self.cfg.d).mean(2))
    def context(self,x,docs):
        h,exports=self.enc(self.emb(x),docs)
        c=self.pool(h);eb=[self.pool(v) for v in exports];cd,valid=chunks(docs,self.cfg.k)
        return h,c,eb,exports,cd,valid
    def pq(self,u):
        c=self.cfg;p=self.head(u).view(*u.shape[:2],c.segments,c.codes).float().softmax(-1)
        return torch.einsum('bmsq,sqf->bmsf',p,self.Q).reshape(*u.shape[:2],c.d)
    def plans(self,c,eb,cd,core=None,z=None,adapters=None):
        if adapters is not None and z is None:z=c.new_zeros(*c.shape[:2],self.cfg.dz)
        u,ports=(self.con if core is None else core)(c,cd,eb,z,adapters)
        return self.pq(u),ports,u
    def decode(self,h,ee,qbar,ports,docs,pos=None,caches=None,return_cache=False,static=False):
        c=self.cfg;x=h;initial=h;group=None;new=[]
        qs= qbar if static else shift(qbar,docs,c.k)
        ps= ports if static else [shift(v,docs,c.k) for v in ports]
        for i,block in enumerate(self.dec.blocks):
            if self.ed is not None:x=self.ed[i](x,ee)
            if self.cd is not None and c.continuous:x=self.cd[i](x,ps)
            elif c.continuous:x=x+(.02/math.sqrt(c.ld))*norm(ps[-1])
            if c.forecast:x=x+(.5/math.sqrt(c.ld))*self.forecast_a[i].tanh()*qs
            if static:y=block.static_step(x,pos,caches[i])
            elif return_cache:y,cache=block(x,docs,pos,None if caches is None else caches[i],True);new.append(cache)
            else:y=block(x,docs,pos)
            x=self.dec.ircs[i](y,initial,group) if self.dec.ircs is not None else y
            if (i+1)%4==0:group=x
        logits=self.dec.final_norm(x)@self.emb.weight.t()
        return (logits,new) if return_cache else logits
    def vq(self,c,valid,domain):
        cfg=self.cfg;cs=c.detach().view(*c.shape[:2],cfg.segments,cfg.d//cfg.segments)
        dist=(cs.unsqueeze(-2)-self.Q[None,None]).float().square().sum(-1);ids=dist.argmin(-1)
        selected=self.Q[torch.arange(cfg.segments,device=c.device),ids]
        errors=(cs-selected).float().square().mean((-1,-2))
        vals=torch.stack([(errors*valid*(domain==i)[:,None]).sum()/(valid*(domain==i)[:,None]).sum().clamp_min(1) for i in (0,1)])
        return vals.mean(),ids,vals
    @torch.no_grad()
    def record_usage(self,ids,valid):
        flat=(ids[valid]+torch.arange(self.cfg.segments,device=ids.device)*self.cfg.codes).flatten()
        self.usage.add_(torch.bincount(flat,minlength=self.cfg.segments*self.cfg.codes).view_as(self.usage))
    @torch.no_grad()
    def restart(self,c,valid,opt=None,generator=None):
        cfg=self.cfg;cs=c.detach()[valid].view(-1,cfg.segments,cfg.d//cfg.segments)
        if len(cs)<cfg.codes:raise ValueError('Not enough valid training chunks to initialize codebooks')
        dead=self.usage==0 if bool(self.q_initialized) else torch.ones_like(self.usage,dtype=torch.bool)
        for s in range(cfg.segments):
            ids=dead[s].nonzero().flatten()
            choice=torch.randperm(len(cs),generator=generator,device='cpu')[:len(ids)].to(cs.device)
            self.Q[s,ids]=cs[choice,s]
        if opt is not None:
            for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                if key in opt.state.get(self.Q,{}):opt.state[self.Q][key][dead]=0
        self.q_initialized.fill_(True);self.usage.zero_();return int(dead.sum())

class Flat(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;c=cfg;self.emb=nn.Embedding(c.vocab,c.d)
        nn.init.normal_(self.emb.weight,std=.02*math.sqrt(128/c.d))
        self.blocks=nn.ModuleList([Block(c,c.flat_layers,c.flat_ff-(16 if i==c.flat_layers-1 else 0)) for i in range(c.flat_layers)]);self.final=nn.RMSNorm(c.d)
    def forward(self,x,docs):
        h=self.emb(x)
        for block in self.blocks:h=block(h,docs)
        return self.final(h)@self.emb.weight.t()

class System(nn.Module):
    def __init__(self,cfg=None,foundation=None,core=None,adapters=None):
        super().__init__();self.cfg=cfg or Config();self.foundation=foundation or (Flat(self.cfg) if self.cfg.kind=='flat' else Backbone(self.cfg))
        self.core=core;self.adapters=adapters
    def context(self,x,docs):return self.foundation.context(x,docs)
    def from_context(self,context,docs,z=None,blend=1.):
        h,c,eb,ee,cd,valid=context;f=self.foundation
        q,ports,u=f.plans(c,eb,cd,self.core,z,self.adapters)
        if blend!=1:
            oldq,oldports,_=f.plans(c,eb,cd)
            q=(1-blend)*oldq+blend*q;ports=[(1-blend)*o+blend*r for o,r in zip(oldports,ports)]
        return f.decode(h,ee,q,ports,docs),c,q,u,ports,valid
    def forward(self,x,docs,z=None,blend=1.,details=False):
        if self.cfg.kind=='flat':return self.foundation(x,docs)
        result=self.from_context(self.context(x,docs),docs,z,blend)
        return result if details else result[0]
    def make_refit(self):
        freeze(self.foundation);self.core=copy.deepcopy(self.foundation.con)
        for p in self.core.parameters():p.requires_grad_(True)
    def make_adapters(self):
        freeze(self);self.adapters=Adapters(self.cfg).to(next(self.foundation.parameters()).device)
    def active_count(self):
        if self.cfg.kind=='flat':return count(self.foundation)
        return count(self.foundation)-(count(self.foundation.con) if self.core is not None else 0)+(count(self.core) if self.core is not None else 0)+(count(self.adapters) if self.adapters is not None else 0)
