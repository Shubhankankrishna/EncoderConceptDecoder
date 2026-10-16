"""Real-token refresh, independent branch caches, and full differentiable replay."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import chunks,shift,ball,norm
from data import T,K,PAD,DONE,EOS,verify
from runtime import sync,nll

def expand_bank(bank,prompt_len,length,m):
    j0=max(0,prompt_len//K-1);j=torch.arange(m,device=bank.device);idx=((j-j0).clamp_min(0)//length).clamp_max(bank.shape[1]-1)
    return bank[:,idx]*(j>=j0)[None,:,None]

class Incremental:
    def __init__(self,s,z):self.s=s;self.z=z;self.enc=[];self.dec=[];self.con=[]
    def prefill(self,x,encoding=None):
        s=self.s;f=s.foundation;docs=torch.zeros_like(x);cfg=s.cfg
        if cfg.kind=='flat':
            h=f.emb(x)
            for b in f.blocks:h,c=b(h,docs,return_cache=True);self.dec.append(c)
            self.t=x.shape[1];return f.final(h[:,-1])@f.emb.weight.t()
        if encoding is None:
            h,ee,self.enc=f.enc(f.emb(x),docs,return_cache=True)
            self.encoding=dict(owner=id(f),tokens=x.cpu().numpy().copy(),h=h,exports=ee,caches=list(self.enc))
        else:
            if encoding['owner']!=id(f) or not np.array_equal(x.cpu().numpy(),encoding['tokens']):raise ValueError('Encoder cache owner/prefix mismatch')
            h=encoding['h'];ee=encoding['exports'];self.enc=list(encoding['caches']);self.encoding=encoding
        self.t=x.shape[1];self.m=self.t//K;self.tail=h[:,self.m*K:];self.et=[e[:,self.m*K:] for e in ee]
        core=s.core or f.con
        if self.m:
            c=f.pool(h[:,:self.m*K]);eb=[f.pool(e[:,:self.m*K]) for e in ee];cd=torch.zeros(c.shape[:2],device=x.device,dtype=torch.long)
            u,ports,self.con=core(c,cd,eb,self.z[:,:self.m],s.adapters,return_cache=True);q=f.pq(u)
            self.q=q[:,-1:];self.ports=[p[:,-1:] for p in ports]
        else:
            q=h.new_zeros(len(x),0,cfg.d);ports=[q,q];self.q=h.new_zeros(len(x),1,cfg.d);self.ports=[self.q,self.q]
            self.con=[(h.new_zeros(len(x),cfg.heads,0,cfg.d//cfg.heads),h.new_zeros(len(x),cfg.heads,0,cfg.d//cfg.heads),docs[:,:0]) for _ in range(cfg.lc)]
        result,self.dec=f.decode(h,ee,q,ports,docs,return_cache=True)
        return result[:,-1]
    def step(self,token):
        s=self.s;f=s.foundation;docs=torch.zeros(len(token),1,device=token.device,dtype=torch.long);pos=torch.full_like(docs,self.t)
        if s.cfg.kind=='flat':
            h=f.emb(token[:,None])
            for i,b in enumerate(f.blocks):h,self.dec[i]=b(h,docs,pos,self.dec[i],True)
            self.t+=1;return f.final(h[:,0])@f.emb.weight.t()
        h,ee,self.enc=f.enc(f.emb(token[:,None]),docs,pos=pos,caches=self.enc,return_cache=True)
        self.tail=torch.cat([self.tail,h],1);self.et=[torch.cat([a,b],1) for a,b in zip(self.et,ee)]
        if self.tail.shape[1]==K:
            c=f.pool(self.tail);eb=[f.pool(v) for v in self.et];cp=torch.full_like(docs,self.m)
            u,self.ports,self.con=(s.core or f.con)(c,docs,eb,self.z[:,self.m:self.m+1],s.adapters,cp,self.con,True)
            self.q=f.pq(u);self.m+=1;self.tail=self.tail[:,:0];self.et=[v[:,:0] for v in self.et]
        # decode's static flag controls alignment AND cache execution; dynamic step aligns explicitly.
        x=h;initial=h;group=None;new=[]
        for i,b in enumerate(f.dec.blocks):
            if f.ed is not None:x=f.ed[i](x,ee)
            if s.cfg.continuous:x=f.cd[i](x,self.ports) if f.cd is not None else x+.02/(s.cfg.ld**.5)*norm(self.ports[-1])
            if s.cfg.forecast:x=x+.5/(s.cfg.ld**.5)*f.forecast_a[i].tanh()*self.q
            y,cache=b(x,docs,pos,self.dec[i],True);new.append(cache)
            x=f.dec.ircs[i](y,initial,group) if f.dec.ircs is not None else y
            if (i+1)%4==0:group=x
        self.dec=new;self.t+=1;return f.dec.final_norm(x[:,0])@f.emb.weight.t()

class FastIncremental(Incremental):
    def __init__(self,s,z):
        super().__init__(s,z)
        if not hasattr(s,'_fast_steps'):
            f=s.foundation
            if s.cfg.kind=='flat':
                def flat(token,pos,caches):
                    h=f.emb(token[:,None])
                    for b,c in zip(f.blocks,caches):h=b.static_step(h,pos,c)
                    return f.final(h[:,0])@f.emb.weight.t()
                s._fast_steps=(torch.compile(flat),)
            else:
                def enc(token,pos,caches):return f.enc(f.emb(token[:,None]),None,pos=pos,caches=caches,static=True)
                def con(c,eb,z,pos,caches):
                    u,p=(s.core or f.con)(c,None,eb,z,s.adapters,pos,caches,static=True);return f.pq(u),p
                def dec(h,ee,q,p,pos,caches):return f.decode(h,ee,q,p,None,pos,caches,static=True)[:,0]
                s._fast_steps=tuple(torch.compile(fn) for fn in (enc,con,dec))
    def prefill(self,x,encoding=None):
        result=super().prefill(x,encoding)
        def padded(caches,limit):return [(F.pad(k,(0,0,0,limit-k.shape[2])),F.pad(v,(0,0,0,limit-v.shape[2]))) for k,v,_ in caches]
        self.dec=padded(self.dec,T)
        if self.s.cfg.kind!='flat':self.enc=padded(self.enc,T);self.con=padded(self.con,T//K)
        return result
    def step(self,token):
        s=self.s;pos=torch.tensor(self.t,device=token.device)
        if s.cfg.kind=='flat':out=s._fast_steps[0](token,pos,self.dec);self.t+=1;return out
        enc,con,dec=s._fast_steps;h,ee=enc(token,pos,self.enc);self.tail=torch.cat([self.tail,h],1);self.et=[torch.cat([a,b],1) for a,b in zip(self.et,ee)]
        if self.tail.shape[1]==K:
            c=s.foundation.pool(self.tail);eb=[s.foundation.pool(v) for v in self.et];cp=torch.tensor(self.m,device=token.device)
            self.q,self.ports=con(c,eb,self.z[:,self.m:self.m+1],cp,self.con);self.m+=1;self.tail=self.tail[:,:0];self.et=[v[:,:0] for v in self.et]
        out=dec(h,ee,self.q,self.ports,pos,self.dec);self.t+=1;return out

@torch.no_grad()
def generate(s,prompts,bank=None,length=1,temperature=0.,top_k=0,seed=0,max_new=100,cached=True,encoding=None,text=False):
    device=next(s.parameters()).device;prompts=np.asarray(prompts);b,p=prompts.shape;cfg=s.cfg;limit=min(T,p+max_new)
    if p>=T:raise ValueError('Prompt leaves no generation positions')
    z=torch.zeros(b,T//K,cfg.dz,device=device) if bank is None else expand_bank(bank,p,length,T//K)
    x=torch.full((b,T),PAD,device=device,dtype=torch.long);x[:,:p]=torch.as_tensor(prompts,device=device)
    lengths=torch.full((b,),limit,device=device,dtype=torch.long);done=torch.zeros(b,device=device,dtype=torch.bool);numeric=torch.zeros_like(done)
    rng=torch.Generator().manual_seed(seed);cache=(FastIncremental if getattr(s,'fast_generation',False) else Incremental)(s,z) if cached else None
    sync(device);begin=time.perf_counter();calls=0
    if cache:logits=cache.prefill(x[:,:p],encoding);calls+=1
    for pos in range(p,limit):
        if cache is None:
            docs=torch.full_like(x,-1);docs[:,:pos]=0;logits=s(x,docs,z)[:,pos-1];calls+=1
        good=torch.isfinite(logits).all(-1);numeric|=~good
        if temperature:
            scores=logits.float()/temperature
            if top_k:scores=scores.masked_fill(scores<scores.topk(min(top_k,scores.shape[-1]),-1).values[...,-1:],-1e9)
            probs=scores.softmax(-1);fallback=torch.zeros_like(probs);fallback[:,PAD]=1
            token=torch.multinomial(torch.where(good[:,None],probs,fallback).cpu(),1,generator=rng).squeeze(1).to(device)
        else:token=logits.argmax(-1)
        x[:,pos]=torch.where(done,torch.full_like(token,PAD),token)
        if text:new=(x[:,pos]==EOS)&~done
        elif (pos-p)%K==K-1:new=((x[:,pos-3:pos+1]==DONE).any(-1))&~done
        else:new=torch.zeros_like(done)
        lengths=torch.where(new,torch.full_like(lengths,pos+1),lengths);done|=new
        if bool(done.all()) or pos==limit-1:break
        if cache:logits=cache.step(x[:,pos]);calls+=1
    sync(device);gen_s=time.perf_counter()-begin;xx=x.cpu().numpy();ln=lengths.cpu().numpy();verification=[];start=time.perf_counter()
    for i in range(b):
        v=dict(success=None,reason='unverified_text') if text else verify(xx[i,:p],xx[i,p:int(ln[i])])
        if bool(numeric[i]):v.update(success=False,reason='numerical_failure')
        verification.append(v)
    vs=time.perf_counter()-start
    return dict(x=xx,lengths=ln,prompt_len=p,verification=verification,seconds=gen_s+vs,generation_s=gen_s,verifier_s=vs,
        generated_tokens=int((ln-p).sum()),forward_calls=calls,encoding=getattr(cache,'encoding',None))

def replay_batch(result,device):
    x=torch.as_tensor(result['x'],device=device);ln=torch.as_tensor(result['lengths'],device=device);p=result['prompt_len'];pos=torch.arange(T,device=device)[None]
    docs=torch.where(pos<ln[:,None],torch.zeros_like(x),torch.full_like(x,-1));mask=((pos>=p-1)&(pos<ln[:,None]-1)).float()
    return x,docs,mask

def active_windows(lengths,p,length,device,m=T//K):
    last=(torch.as_tensor(lengths,device=device)-1)//K-1;j=torch.arange(m,device=device);j0=max(0,p//K-1)
    enabled=(j[None]>=j0)&(j[None]<=last[:,None]);ids=((j-j0).clamp_min(0)//length).clamp_max(m-1)
    counts=torch.zeros(len(lengths),m,device=device);counts.scatter_add_(1,ids[None].expand(len(lengths),-1),enabled.float());return counts>0

class MaskedAdam:
    def __init__(self,z,lr=.03):self.z=z;self.lr=lr;self.m=torch.zeros_like(z);self.v=torch.zeros_like(z);self.steps=torch.zeros(len(z),device=z.device,dtype=torch.long)
    @torch.no_grad()
    def update(self,grad,active,eligible):
        valid=torch.isfinite(grad).all((1,2))&torch.isfinite(self.z).all((1,2));use=active&eligible[:,None]&valid[:,None]
        g=torch.where(use[:,:,None],grad,torch.zeros_like(grad));g=g*(1/g.flatten(1).norm(dim=1).clamp_min(1e-12)).clamp_max(1)[:,None,None]
        nm=.9*self.m+.1*g;nv=.999*self.v+.001*g.square();ns=(self.steps+use.any(1)).clamp_min(1)
        direction=(nm/(1-.9**ns[:,None,None]))/((nv/(1-.999**ns[:,None,None])).sqrt()+1e-8)
        proposal=ball(self.z-self.lr*direction);valid&=torch.isfinite(proposal).all((1,2));use&=valid[:,None]
        delta=torch.where(use[:,:,None],proposal-self.z,torch.zeros_like(self.z));self.z.add_(delta)
        self.m.copy_(torch.where(use[:,:,None],nm,self.m));self.v.copy_(torch.where(use[:,:,None],nv,self.v));self.steps+=use.any(1)
        return dict(finite=valid,step_norm=delta.flatten(1).norm(dim=1))

def repair_gradient(s,result,bank,length=1,kl=.05):
    device=bank.device;x,docs,mask=replay_batch(result,device);p=result['prompt_len'];z=expand_bank(bank,p,length,T//K)
    logits=getattr(s,'repair_forward',s)(x,docs,z);counts=mask[:,:-1].sum(1).clamp_min(1)
    reject=-nll(logits,x,mask).sum(1)/counts;loss=reject
    if kl:
        with torch.no_grad():reference=s(x,docs,torch.zeros_like(z)).float().softmax(-1)
        logp=logits.float().log_softmax(-1);div=F.kl_div(logp,reference,reduction='none').sum(-1)[:,:-1]
        loss=loss+kl*(div*mask[:,:-1]).sum(1)/counts
    grad=torch.autograd.grad(loss.sum(),bank)[0];return grad,dict(loss=float(loss.mean().detach()),gradient_norm=float(grad.norm()))
