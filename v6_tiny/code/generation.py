"""Correct incremental caches; every recursive application owns a separate slot."""
import time
import numpy as np
import torch
from model import D,K,DZ,chunk_info,shift,ball
from data import T,WALK_START,PAD,DONE,verify
from runtime import sync


def expand_bank(bank,length,m):
    j=torch.arange(m,device=bank.device)- (WALK_START//K-1)
    idx=(j.clamp_min(0)//length).clamp_max(bank.shape[1]-1)
    return bank[:,idx]* (j>=0)[None,:,None]


class Incremental:
    def __init__(self,system,z):self.s=system;self.z=z;self.enc=[];self.dec=[];self.core=[];self.con=[]

    def prefill(self,x,encoding=None):
        f=self.s.foundation;docs=torch.zeros_like(x)
        if encoding is not None:
            if encoding['owner']!=id(f) or not np.array_equal(encoding['tokens'],x.cpu().numpy()):raise ValueError('Prompt encoding cache owner/prefix mismatch')
            h=encoding['features'];self.enc=list(encoding['caches']);self.prompt_encoding=encoding
        else:
            h=f.emb(x)
            for block in f.enc:
                h,cache=block(h,docs,return_cache=True);self.enc.append(cache)
            h=f.enc_norm(h)
            self.prompt_encoding=dict(owner=id(f),tokens=x.cpu().numpy().copy(),features=h,caches=list(self.enc))
        self.tail=h[:,(x.shape[1]//4)*4:];self.t=x.shape[1];self.m=x.shape[1]//4
        c=f.pool(h);cd=torch.zeros(c.shape[:2],device=x.device,dtype=torch.long)
        if self.s.core is not None:
            u,a,self.core=self.s.core(c,cd,self.z[:,:self.m],self.s.adapters,return_cache=True)
        else:
            u=c
            for block in f.con:
                u,cache=block(u,cd,return_cache=True);self.con.append(cache)
            u=f.con_norm(u);a=f.port(u)
        chat=f.pq(u);self.chat=chat[:,-1:];self.a=a[:,-1:]
        d=h+shift(chat,docs)+f.gamma*shift(a,docs)
        for block in f.dec:
            d,cache=block(d,docs,return_cache=True);self.dec.append(cache)
        return f.nf(d[:,-1])@f.emb.weight.t()

    def step(self,token):
        f=self.s.foundation;device=token.device
        docs=torch.zeros(len(token),1,device=device,dtype=torch.long)
        pos=torch.full_like(docs,self.t);h=f.emb(token[:,None])
        for i,block in enumerate(f.enc):h,self.enc[i]=block(h,docs,pos=pos,cache=self.enc[i],return_cache=True)
        h=f.enc_norm(h);self.tail=torch.cat([self.tail,h],1)
        if len(self.tail[0])==K:
            c=f.pool(self.tail);cd=docs;cp=torch.full_like(docs,self.m)
            if self.s.core is not None:
                u,a,self.core=self.s.core(c,cd,self.z[:,self.m:self.m+1],self.s.adapters,self.core,True,cp)
            else:
                u=c
                for i,block in enumerate(f.con):u,self.con[i]=block(u,cd,pos=cp,cache=self.con[i],return_cache=True)
                u=f.con_norm(u);a=f.port(u)
            self.chat,self.a=f.pq(u),a;self.tail=self.tail[:,:0];self.m+=1
        d=h+self.chat+f.gamma*self.a
        for i,block in enumerate(f.dec):d,self.dec[i]=block(d,docs,pos=pos,cache=self.dec[i],return_cache=True)
        self.t+=1
        return f.nf(d[:,0])@f.emb.weight.t()


class FastIncremental(Incremental):
    def __init__(self,system,z):
        super().__init__(system,z)
        if not hasattr(system,'_fast_steps'):
            f=system.foundation
            def enc(token,pos,caches):
                h=f.emb(token[:,None])
                for block,cache in zip(f.enc,caches):h=block.static_step(h,pos,cache)
                return f.enc_norm(h)
            def core(c,z,pos,caches):
                if system.core is None:
                    u=c
                    for block,cache in zip(f.con,caches):u=block.static_step(u,pos,cache)
                    u=f.con_norm(u);return f.pq(u),f.port(u)
                core=system.core;e=core.in_proj(c)
                ih,il=system.adapters(e,z) if system.adapters else (0.,0.)
                h,l=core.h0.expand_as(e)+ih,core.l0.expand_as(e);visit=0
                for _ in range(2):
                    for _ in range(2):l=core.post_L(core.fL.static_step(l+h+e+il,pos,caches[visit]));visit+=1
                    h=core.post_H(core.fH.static_step(h+l,pos,caches[visit]));visit+=1
                u=core.out_n(h);return f.pq(core.B_u(u)),core.B_A(u)
            def dec(h,chat,a,pos,caches):
                d=h+chat+f.gamma*a
                for block,cache in zip(f.dec,caches):d=block.static_step(d,pos,cache)
                return f.nf(d[:,0])@f.emb.weight.t()
            system._fast_steps=tuple(torch.compile(fn) for fn in (enc,core,dec))

    def prefill(self,x,encoding=None):
        result=super().prefill(x,encoding)
        def padded(caches,limit):
            return [(torch.nn.functional.pad(k,(0,0,0,limit-k.shape[2])),torch.nn.functional.pad(v,(0,0,0,limit-v.shape[2]))) for k,v,_ in caches]
        self.enc=padded(self.enc,T);self.dec=padded(self.dec,T)
        self.core=padded(self.core,T//4) if self.s.core else padded(self.con,T//4)
        return result

    def step(self,token):
        enc,core,dec=self.s._fast_steps;pos=torch.tensor(self.t,device=token.device)
        h=enc(token,pos,self.enc);self.tail=torch.cat([self.tail,h],1)
        if self.tail.shape[1]==K:
            c=self.s.foundation.pool(self.tail);cp=torch.tensor(self.m,device=token.device)
            self.chat,self.a=core(c,self.z[:,self.m:self.m+1],cp,self.core)
            self.tail=self.tail[:,:0];self.m+=1
        result=dec(h,self.chat,self.a,pos,self.dec);self.t+=1
        return result


@torch.no_grad()
def generate(system,headers,bank=None,length=1,temperature=0,seed=0,cached=True,prompt_encoding=None):
    device=next(system.parameters()).device;b=len(headers);m=T//K
    z=torch.zeros(b,m,DZ,device=device) if bank is None else expand_bank(bank,length,m)
    x=torch.full((b,T),PAD,dtype=torch.long,device=device)
    x[:,:WALK_START]=torch.as_tensor(headers[:,:WALK_START],device=device)
    lengths=torch.full((b,),T,device=device,dtype=torch.long);done=torch.zeros(b,device=device,dtype=torch.bool);numerical=torch.zeros_like(done)
    rng=torch.Generator().manual_seed(seed)
    cache=(FastIncremental(system,z) if getattr(system,'fast_generation',False) else Incremental(system,z)) if cached else None
    sync(device);start=time.perf_counter();forward_calls=0
    if cache:logits=cache.prefill(x[:,:WALK_START],prompt_encoding);forward_calls+=1
    for pos in range(WALK_START,T):
        if not cache:
            docs=torch.full_like(x,-1);docs[:,:pos]=0
            logits=getattr(system,'optimized_forward',system)(x,docs,z)[:,pos-1];forward_calls+=1
        finite=torch.isfinite(logits).all(-1);numerical|=~finite
        if temperature:
            p=(logits.float()/temperature).softmax(-1)
            fallback=torch.zeros_like(p);fallback[:,PAD]=1
            probabilities=torch.where(finite[:,None],p,fallback).cpu()
            nxt=torch.multinomial(probabilities,1,generator=rng).squeeze(1).to(device)
        else:nxt=logits.argmax(-1)
        x[:,pos]=torch.where(done,torch.full_like(nxt,PAD),nxt)
        # Stop only after all four actually generated terminator tokens exist.
        if (pos-WALK_START)%4==3:
            newly=(x[:,pos-3]==DONE)&~done;lengths=torch.where(newly,torch.full_like(lengths,pos+1),lengths);done|=newly
        if bool(done.all()) or pos==T-1:break
        if cache:logits=cache.step(x[:,pos]);forward_calls+=1
    sync(device);elapsed=time.perf_counter()-start
    xc=x.cpu().numpy();ln=lengths.cpu().numpy();verifications=[]
    verstart=time.perf_counter()
    numerical=numerical.cpu().tolist()
    for i in range(b):
        v=verify(xc[i,:WALK_START],xc[i,WALK_START:int(ln[i])])
        if numerical[i]:v.update(success=False,format_ok=False,reason='numerical_failure')
        verifications.append(v)
    verifier_s=time.perf_counter()-verstart
    return dict(x=xc,lengths=ln,verification=verifications,seconds=elapsed+verifier_s,generation_s=elapsed,
                verifier_s=verifier_s,forward_calls=forward_calls,generated_tokens=int((ln-WALK_START).sum()),numerical_failures=numerical,
                prompt_encoding=cache.prompt_encoding if cache else None,encoder_prefill_reused=prompt_encoding is not None)


def replay_batch(result,device):
    x=torch.as_tensor(result['x'],device=device);lengths=torch.as_tensor(result['lengths'],device=device)
    pos=torch.arange(T,device=device)[None]
    docs=torch.where(pos<lengths[:,None],torch.zeros_like(x),torch.full_like(x,-1))
    mask=((pos>=WALK_START-1)&(pos<lengths[:,None]-1)).float()
    return x,docs,mask


class MaskedAdam:
    """Per-problem clock, frozen inactive moments, and bounded window writes."""
    def __init__(self,z,lr=0.1):
        self.z=z;self.lr=lr;self.m=torch.zeros_like(z);self.v=torch.zeros_like(z)
        self.steps=torch.zeros(len(z),device=z.device,dtype=torch.long)

    @torch.no_grad()
    def update(self,grad,active,eligible):
        use=active&eligible[:,None]
        finite=torch.isfinite(grad).all((1,2))&torch.isfinite(self.z).all((1,2))
        use&=finite[:,None];used=use.any(1)
        g=torch.where(use[:,:,None],grad,torch.zeros_like(grad));norm=g.flatten(1).norm(dim=1).clamp_min(1e-12)
        g=g*(1/norm).clamp_max(1)[:,None,None]
        nm=0.9*self.m+0.1*g;nv=0.999*self.v+0.001*g.square()
        ns=self.steps+used.long();ns_safe=ns.clamp_min(1)
        direction=(nm/(1-0.9**ns_safe[:,None,None]))/((nv/(1-0.999**ns_safe[:,None,None])).sqrt()+1e-8)
        proposal=ball(self.z-self.lr*direction)
        finite&=torch.isfinite(proposal).all((1,2));use&=finite[:,None];used=use.any(1)
        delta=torch.where(use[:,:,None],proposal-self.z,torch.zeros_like(self.z))
        self.z.add_(delta);self.m.copy_(torch.where(use[:,:,None],nm,self.m));self.v.copy_(torch.where(use[:,:,None],nv,self.v))
        self.steps.add_(used.long())
        return dict(finite=finite,norm=norm,step_norm=delta.flatten(1).norm(dim=1))


def active_windows(lengths,length,device,nwindows=T//4):
    # Window contributes if its plan is read by at least one scored response token.
    last=((torch.as_tensor(lengths,device=device)-2+1)//4)-1
    chunk=torch.arange(T//4,device=device)
    enabled=(chunk[None]>=WALK_START//4-1)&(chunk[None]<=last[:,None])
    ids=((chunk-(WALK_START//4-1)).clamp_min(0)//length).clamp_max(nwindows-1)
    counts=torch.zeros(len(lengths),nwindows,device=device)
    counts.scatter_add_(1,ids[None].expand(len(lengths),-1),enabled.float())
    return counts>0
