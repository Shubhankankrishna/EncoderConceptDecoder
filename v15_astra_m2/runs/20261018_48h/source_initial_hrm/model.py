"""Deterministic Astra HRM with a normalized, causal 12-level code distribution.

The 8K LM1B codebooks need low-rank trainable code tables to fit ~10M total.
All 641 codes are predicted, including the first/root code. Teacher forcing is
parallelized across levels with explicit block masks; generation is sequential
across levels and parallel within each level.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Config:
    width: int = 192
    code_dim: int = 64
    vocab: int = 8192
    levels: tuple = (1,8,16,24,32,40,52,64,76,92,108,128)
    heads: int = 6
    ff: int = 544
    ctx_layers: int = 2
    head_layers: int = 2
    slots: int = 16
    state_length: int = 88
    fast_steps: int = 2
    slow_steps: int = 2
    state_conditioned: bool = True


class Attention(nn.Module):
    def __init__(self,d,h):
        super().__init__();self.h=h;self.d=d
        self.q=nn.Linear(d,d,bias=False);self.kv=nn.Linear(d,2*d,bias=False)
        self.out=nn.Linear(d,d,bias=False)

    def forward(self,x,mem=None,mask=None):
        mem=x if mem is None else mem
        b,n,d=x.shape; m=mem.shape[1]
        q=self.q(x).view(b,n,self.h,d//self.h).transpose(1,2)
        k,v=self.kv(mem).view(b,m,2,self.h,d//self.h).permute(2,0,3,1,4).unbind(0)
        z=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        return self.out(z.transpose(1,2).reshape(b,n,d))


class Block(nn.Module):
    def __init__(self,c,cross=False):
        super().__init__();d=c.width
        self.norm1=nn.LayerNorm(d);self.attn=Attention(d,c.heads)
        self.norm2=nn.LayerNorm(d)
        self.gate=nn.Linear(d,c.ff,bias=False);self.value=nn.Linear(d,c.ff,bias=False)
        self.ffout=nn.Linear(c.ff,d,bias=False)
        self.cross=Attention(d,c.heads) if cross else None
        if cross:self.norm_cross=nn.LayerNorm(d);self.norm_mem=nn.LayerNorm(d)

    def forward(self,x,mem=None,self_mask=None,cross_mask=None):
        x=x+self.attn(self.norm1(x),mask=self_mask)
        if self.cross is not None:
            x=x+self.cross(self.norm_cross(x),self.norm_mem(mem),cross_mask)
        z=self.norm2(x)
        return x+self.ffout(F.silu(self.gate(z))*self.value(z))


class AstraHRM(nn.Module):
    def __init__(self,c=None):
        super().__init__();self.config=c or Config();c=self.config;d=c.width
        self.code=nn.ModuleList([nn.Embedding(c.vocab,c.code_dim) for _ in c.levels])
        self.code_in=nn.Linear(c.code_dim,d,bias=False)
        self.code_out=nn.Linear(d,c.code_dim,bias=False)
        self.level=nn.Parameter(torch.randn(len(c.levels),d)*.02)
        self.position=nn.Parameter(torch.randn(max(c.levels),d)*.02)
        self.pool_query=nn.Parameter(torch.randn(len(c.levels),d)*.02)
        self.pool_k=nn.Linear(d,d,bias=False);self.pool_v=nn.Linear(d,d,bias=False)
        self.state_embed=nn.Embedding(128,d)
        self.state_pos=nn.Parameter(torch.randn(c.state_length,d)*.02)
        self.context=nn.ModuleList([Block(c) for _ in range(c.ctx_layers)])
        self.h_init=nn.Parameter(torch.randn(c.slots,d)*.02)
        self.l_init=nn.Parameter(torch.randn(c.slots,d)*.02)
        self.init_proj=nn.Linear(d,2*d,bias=False)
        self.fast=Block(c,cross=True);self.slow=Block(c,cross=True)
        self.decoder=nn.ModuleList([Block(c,cross=True) for _ in range(c.head_layers)])
        self.final_norm=nn.LayerNorm(d)
        self.logit_scale=nn.Parameter(torch.zeros(len(c.levels)))
        self.logit_bias=nn.Parameter(torch.zeros(len(c.levels),c.vocab))
        self.offsets=[0]
        for n in c.levels:self.offsets.append(self.offsets[-1]+n)
        ids=torch.repeat_interleave(torch.arange(len(c.levels)),torch.tensor(c.levels))
        self.register_buffer('query_levels',ids,persistent=False)
        self.register_buffer('self_mask',ids[:,None]==ids[None,:],persistent=False)
        self.register_buffer('previous_mask',torch.arange(len(c.levels))[None,:]<ids[:,None],persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod,nn.Linear):nn.init.normal_(mod.weight,std=.02)
            if isinstance(mod,nn.Embedding):nn.init.normal_(mod.weight,std=.02)

    def summaries(self,codes):
        out=[]
        for i,n in enumerate(self.config.levels):
            z=self.code_in(self.code[i](codes[:,self.offsets[i]:self.offsets[i+1]]))
            z=z+self.position[:n]+self.level[i]
            w=(self.pool_k(z)*self.pool_query[i]).sum(-1)/math.sqrt(self.config.width)
            out.append((w.softmax(-1).unsqueeze(-1)*self.pool_v(z)).sum(1))
        return torch.stack(out,dim=1)

    def encode_context(self,codes,state):
        mem=self.summaries(codes)
        if self.config.state_conditioned:
            assert state.shape[1]==self.config.state_length
            mem=torch.cat([mem,self.state_embed(state)+self.state_pos],dim=1)
        for block in self.context:mem=block(mem)
        return mem

    def reason(self,mem,retain_states=False):
        h0,l0=self.init_proj(mem.mean(1)).chunk(2,-1)
        h=self.h_init[None]+h0[:,None];l=self.l_init[None]+l0[:,None]
        trace=[]
        for _ in range(self.config.slow_steps):
            for _ in range(self.config.fast_steps):
                l=self.fast(l,torch.cat([mem,h],dim=1))
            h=self.slow(h,torch.cat([mem,l],dim=1))
            if retain_states and h.requires_grad:h.retain_grad()
            trace.append(h)
        return h,trace

    def queries(self):
        return torch.cat([self.position[:n]+self.level[i] for i,n in enumerate(self.config.levels)],0)

    def decode_hidden(self,mem,h,target_codes,level=None):
        prev=self.summaries(target_codes)
        memory=torch.cat([mem,h,prev],1)
        q=self.queries();mask=self.previous_mask;sm=self.self_mask
        if level is not None:
            lo,hi=self.offsets[level:level+2];q=q[lo:hi];mask=mask[lo:hi];sm=sm[lo:hi,lo:hi]
        always=torch.ones(mask.shape[0],mem.shape[1]+h.shape[1],dtype=torch.bool,device=mem.device)
        cm=torch.cat([always,mask],1)
        q=q[None].expand(len(mem),-1,-1)
        for block in self.decoder:q=block(q,memory,self_mask=sm,cross_mask=cm)
        return self.code_out(self.final_norm(q))

    def logits(self,hidden,level):
        scale=self.logit_scale[level].clamp(-4,4).exp()
        return F.linear(hidden,self.code[level].weight,self.logit_bias[level])*scale

    def forward(self,x,state,y,objective='per_code',corrupt=0.):
        mem=self.encode_context(x,state);h,_=self.reason(mem)
        conditioning=y
        if self.training and corrupt:
            mask=torch.rand(y.shape,device=y.device)<corrupt
            conditioning=torch.where(mask,torch.randint(self.config.vocab,y.shape,device=y.device),y)
        hidden=self.decode_hidden(mem,h,conditioning)
        nll=[];acc=[]
        for i,n in enumerate(self.config.levels):
            a,b=self.offsets[i:i+2];logits=self.logits(hidden[:,a:b],i)
            ce=F.cross_entropy(logits.reshape(-1,self.config.vocab),y[:,a:b].reshape(-1),reduction='none').view(len(y),n)
            nll.append(ce.sum(-1));acc.append((logits.argmax(-1)==y[:,a:b]).float().mean())
        sums=torch.stack(nll,1)
        proper=sums.sum(1).mean()/self.offsets[-1]
        loss=proper if objective=='per_code' else (sums/torch.tensor(self.config.levels,device=y.device)).mean()
        return loss,dict(nll=proper.detach(),per_level=(sums.mean(0)/torch.tensor(self.config.levels,device=y.device)).detach(),accuracy=torch.stack(acc).detach())

    @torch.inference_mode()
    def generate(self,x,state,temperature=0.):
        mem=self.encode_context(x,state);h,_=self.reason(mem)
        y=torch.zeros(len(x),self.offsets[-1],dtype=torch.long,device=x.device)
        for i,n in enumerate(self.config.levels):
            hidden=self.decode_hidden(mem,h,y,level=i);logits=self.logits(hidden,i)
            if temperature:
                chosen=torch.multinomial((logits/temperature).softmax(-1).reshape(-1,self.config.vocab),1).view(len(x),n)
            else:chosen=logits.argmax(-1)
            a,b=self.offsets[i:i+2];y[:,a:b]=chosen
        return y

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
