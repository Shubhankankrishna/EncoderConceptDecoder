"""Actual-token concept refresh, arbitrary prefix lengths, version-owned caches."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import *
from data import T,PAD,DONE,EOS,parse_prompt,verify
from runtime import sync,nll

def expand_bank(bank,p,length,m):
    j0=max(0,p//K-1);j=torch.arange(m,device=bank.device);idx=((j-j0).clamp_min(0)//length).clamp_max(bank.shape[1]-1)
    return bank[:,idx]*(j>=j0)[None,:,None]
class Incremental:
    def __init__(self,s,z):self.s=s;self.z=z;self.enc=[];self.dec=[];self.con=[]
    def prefill(self,x,encoding=None):
        s=self.s;f=s.foundation;docs=torch.zeros_like(x);self.t=x.shape[1]
        if s.is_flat:
            h=f.emb(x)
            for b in f.blocks:h,c=b(h,docs,return_cache=True);self.dec.append(c)
            return f.nf(h[:,-1])@f.emb.weight.t()
        h=f.emb(x)
        for b in f.enc:h,c=b(h,docs,return_cache=True);self.enc.append(c)
        h=f.enc_norm(h);self.m=self.t//K;self.tail=h[:,self.m*K:];c=f.pool(h);cd=docs[:,:self.m]
        if self.m:
            u,a,self.con=(s.core or DenseView(f))(c,cd,self.z[:,:self.m],s.adapters,return_cache=True);q=f.pq(u);self.q=q[:,-1:];self.a=a[:,-1:]
        else:
            q=h[:,:0];a=q;self.q=h.new_zeros(len(x),1,D);self.a=self.q
        h=h+shift(q,docs)+f.gamma*shift(a,docs)
        for b in f.dec:h,c=b(h,docs,return_cache=True);self.dec.append(c)
        return f.nf(h[:,-1])@f.emb.weight.t()
    def step(self,token):
        s=self.s;f=s.foundation;docs=torch.zeros(len(token),1,dtype=torch.long,device=token.device);pos=torch.full_like(docs,self.t);h=f.emb(token[:,None])
        if s.is_flat:
            for i,b in enumerate(f.blocks):h,self.dec[i]=b(h,docs,pos,self.dec[i],True)
        else:
            for i,b in enumerate(f.enc):h,self.enc[i]=b(h,docs,pos,self.enc[i],True)
            h=f.enc_norm(h);self.tail=torch.cat([self.tail,h],1)
            if self.tail.shape[1]==K:
                c=f.pool(self.tail);cp=torch.full_like(docs,self.m)
                u,self.a,self.con=(s.core or DenseView(f))(c,docs,self.z[:,self.m:self.m+1],s.adapters,self.con or None,True,cp);self.q=f.pq(u);self.m+=1;self.tail=self.tail[:,:0]
            h=h+self.q+f.gamma*self.a
            for i,b in enumerate(f.dec):h,self.dec[i]=b(h,docs,pos,self.dec[i],True)
        self.t+=1;return f.nf(h[:,0])@f.emb.weight.t()

class FastIncremental(Incremental):
    def __init__(self,s,z):
        super().__init__(s,z)
        if isinstance(s.core,Core):raise ValueError('Recurrent cache uses independently audited dynamic visit caches')
        if not hasattr(s,'_fast_steps'):
            f=s.foundation
            if s.is_flat:
                def flat(token,pos,caches):
                    h=f.emb(token[:,None])
                    for b,c in zip(f.blocks,caches):h=b.static_step(h,pos,c)
                    return f.nf(h[:,0])@f.emb.weight.t()
                s._fast_steps=(torch.compile(flat),)
            else:
                def enc(token,pos,caches):
                    h=f.emb(token[:,None])
                    for b,c in zip(f.enc,caches):h=b.static_step(h,pos,c)
                    return f.enc_norm(h)
                def con(c,z,pos,caches):
                    blocks=s.core.blocks if s.core else f.con
                    for i,(b,cache) in enumerate(zip(blocks,caches)):
                        if s.adapters is not None and i in (0,len(blocks)//2):c=c+s.adapters(c,z)[0 if i==0 else 1]
                        c=b.static_step(c,pos,cache)
                    u=s.core.norm(c) if s.core else f.con_norm(c);a=s.core.W_A(rms(u)) if s.core else f.port(u)
                    return f.pq(u),a
                def dec(h,q,a,pos,caches):
                    h=h+q+f.gamma*a
                    for b,c in zip(f.dec,caches):h=b.static_step(h,pos,c)
                    return f.nf(h[:,0])@f.emb.weight.t()
                s._fast_steps=tuple(torch.compile(v) for v in (enc,con,dec))
    def prefill(self,x,encoding=None):
        out=super().prefill(x,encoding)
        def pad(caches,limit):return [[F.pad(k,(0,0,0,limit-k.shape[2])),F.pad(v,(0,0,0,limit-v.shape[2])),g] for k,v,docs,g in caches]
        self.dec=pad(self.dec,T)
        if not self.s.is_flat:
            self.enc=pad(self.enc,T)
            if self.con:self.con=pad(self.con,T//K)
            else:
                blocks=self.s.core.blocks if self.s.core else self.s.foundation.con
                self.con=[[x.new_zeros(len(x),HEADS,T//K,D//HEADS,dtype=self.s.foundation.emb.weight.dtype),x.new_zeros(len(x),HEADS,T//K,D//HEADS,dtype=self.s.foundation.emb.weight.dtype),torch.zeros(len(x),len(block.banks),device=x.device) if block.banks else None] for block in blocks]
        return out
    def step(self,token):
        s=self.s;pos=torch.tensor(self.t,device=token.device)
        if s.is_flat:out=s._fast_steps[0](token,pos,self.dec)
        else:
            enc,con,dec=s._fast_steps;h=enc(token,pos,self.enc);self.tail=torch.cat([self.tail,h],1)
            if self.tail.shape[1]==K:
                self.q,self.a=con(s.foundation.pool(self.tail),self.z[:,self.m:self.m+1],torch.tensor(self.m,device=token.device),self.con);self.m+=1;self.tail=self.tail[:,:0]
            out=dec(h,self.q,self.a,pos,self.dec)
        self.t+=1;return out

@torch.no_grad()
def generate(s,prompts,bank=None,length=1,temperature=0.,top_k=128,seed=0,max_new=100,cached=True,encoding=None,text=False,force_complete=False):
    device=next(s.parameters()).device;prompts=np.asarray(prompts);b,p=prompts.shape;limit=min(T,p+max_new)
    if p>=T:raise ValueError('No response capacity')
    z=torch.zeros(b,T//K,DZ,device=device) if bank is None else expand_bank(bank,p,length,T//K)
    x=torch.full((b,T),PAD,device=device,dtype=torch.long);x[:,:p]=torch.as_tensor(prompts,device=device);lengths=torch.full((b,),limit,device=device,dtype=torch.long);done=torch.zeros(b,dtype=torch.bool,device=device);numeric=done.clone();rng=torch.Generator().manual_seed(seed)
    cache=(FastIncremental if getattr(s,'fast_generation',False) and not isinstance(s.core,Core) else Incremental)(s,z) if cached else None
    sync(device);begin=time.perf_counter();calls=0
    if cache:logits=cache.prefill(x[:,:p]);calls+=1
    lookup=torch.tensor([parse_prompt(row)[0] in ('lookup','compose','bind') for row in prompts],device=device) if not text else None
    for pos in range(p,limit):
        if cache is None:logits=s(x[:,:pos],torch.zeros_like(x[:,:pos]),z[:,:pos//K])[:,-1];calls+=1
        good=torch.isfinite(logits).all(-1);numeric|=~good
        if temperature:
            scores=logits.float()/temperature
            if top_k:scores=scores.masked_fill(scores<scores.topk(min(top_k,scores.shape[-1]),-1).values[...,-1:],-1e9)
            probs=scores.softmax(-1);fallback=torch.zeros_like(probs);fallback[:,PAD]=1;token=torch.multinomial(torch.where(good[:,None],probs,fallback).cpu(),1,generator=rng).squeeze(1).to(device)
        else:token=logits.argmax(-1)
        x[:,pos]=torch.where(done,torch.full_like(token,PAD),token)
        if text:new=(token==EOS)&~done
        elif (pos-p)%K==K-1:new=(lookup|((x[:,pos-K+1]==DONE)))&~done
        else:new=torch.zeros_like(done)
        lengths=torch.where(new,torch.full_like(lengths,pos+1),lengths);done|=new
        if (bool(done.all()) and not force_complete) or pos==limit-1:break
        if cache:logits=cache.step(x[:,pos]);calls+=1
    sync(device);seconds=time.perf_counter()-begin;xx=x.cpu().numpy();ln=lengths.cpu().numpy();start=time.perf_counter();vv=[]
    for i in range(b):
        v=dict(success=None,reason='unverified_text') if text else verify(xx[i,:p],xx[i,p:int(ln[i])])
        if bool(numeric[i]):v.update(success=False,reason='numerical_failure')
        vv.append(v)
    vs=time.perf_counter()-start
    return dict(x=xx,lengths=ln,prompt_len=p,verification=vv,seconds=seconds+vs,generation_s=seconds,verifier_s=vs,generated_tokens=int((ln-p).sum()),forward_calls=calls,encoding=None)

def active_windows(lengths,p,length,device,m=T//K):
    last=(torch.as_tensor(lengths,device=device)-1)//K-1;j=torch.arange(m,device=device);j0=max(0,p//K-1);enabled=(j[None]>=j0)&(j[None]<=last[:,None]);ids=((j-j0).clamp_min(0)//length).clamp_max(m-1);counts=torch.zeros(len(lengths),m,device=device);counts.scatter_add_(1,ids[None].expand(len(lengths),-1),enabled.float());return counts>0
class MaskedAdam:
    def __init__(self,z,lr=.03):self.z=z;self.lr=lr;self.m=torch.zeros_like(z);self.v=torch.zeros_like(z);self.steps=torch.zeros_like(z,dtype=torch.long)
    @torch.no_grad()
    def update(self,grad,active,eligible):
        valid=torch.isfinite(grad).all((1,2))&torch.isfinite(self.z).all((1,2));use=(active&eligible[:,None]&valid[:,None])[...,None].expand_as(grad).clone();g=torch.where(use,grad,torch.zeros_like(grad));g*= (1/g.flatten(1).norm(dim=1).clamp_min(1e-12)).clamp_max(1)[:,None,None];nm=.9*self.m+.1*g;nv=.99*self.v+.01*g.square();ns=(self.steps+use).clamp_min(1)
        direction=(nm/(1-.9**ns))/((nv/(1-.99**ns)).sqrt()+1e-8);proposal=ball(self.z-self.lr*direction);valid&=torch.isfinite(proposal).all((1,2));use&=valid[:,None,None];delta=torch.where(use,proposal-self.z,torch.zeros_like(self.z));self.z.add_(delta);self.m.copy_(torch.where(use,nm,self.m));self.v.copy_(torch.where(use,nv,self.v));self.steps.add_(use)
        return dict(finite=valid,step_norm=delta.flatten(1).norm(dim=1))
def repair_gradient(s,result,bank,length=1,kl=.05,full_response=False):
    device=bank.device;x=torch.as_tensor(result['x'],device=device);ln=torch.as_tensor(result['lengths'],device=device);p=result['prompt_len'];pos=torch.arange(T,device=device)[None];docs=torch.where(pos<ln[:,None],torch.zeros_like(x),torch.full_like(x,-1));mask=((pos>=p-1)&(pos<ln[:,None]-1)).float();decision=mask if full_response else mask*(((pos-(p-1))%K==0)|((pos-(p-1))%K==2)).float()
    z=expand_bank(bank,p,length,T//K);logits=s(x,docs,z);loss=-nll(logits,x,decision).sum(1)/decision[:,:-1].sum(1).clamp_min(1)
    if kl:
        with torch.no_grad():reference=s(x,docs,torch.zeros_like(z)).float().softmax(-1)
        div=F.kl_div(logits.float().log_softmax(-1),reference,reduction='none').sum(-1)[:,:-1];loss+=kl*(div*mask[:,:-1]).sum(1)/mask[:,:-1].sum(1).clamp_min(1)
    grad=torch.autograd.grad(loss.sum(),bank)[0];return grad,dict(loss=float(loss.mean().detach()),gradient_norm=float(grad.norm()))
