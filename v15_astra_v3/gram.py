"""Astra GRAM: stochastic slow-state residuals, train-only q, full trajectory ELBO.

This is the blueprint's short-unroll NCM adaptation, not a paper replication.
The default diagonal variance and full BPTT differ from paper Eq. 14. A separate
last-transition surrogate is supplied and tested, but is not used by this run.
"""
from contextlib import nullcontext
from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
import torch.nn.functional as F

from .model import AstraHRM, Config
from .objectives import gaussian_kl


@dataclass
class GramConfig(Config):
    ff: int = 432
    gaussian_hidden: int = 128
    variance: str = 'diagonal'
    sigma_initial: float = .1
    sigma_min: float = 1e-4


class GaussianHead(nn.Module):
    def __init__(self, c, inputs):
        super().__init__(); self.c=c
        self.norm=nn.LayerNorm(inputs*c.width)
        self.gate=nn.Linear(inputs*c.width,c.gaussian_hidden,bias=False)
        self.value=nn.Linear(inputs*c.width,c.gaussian_hidden,bias=False)
        self.out=nn.Linear(c.gaussian_hidden,c.width+(c.width if c.variance=='diagonal' else 1))
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        with torch.no_grad():self.out.bias[c.width:]=math.log(math.expm1(c.sigma_initial-c.sigma_min))

    def forward(self,*xs):
        z=self.norm(torch.cat(xs,-1));v=self.out(F.silu(self.gate(z))*self.value(z))
        mu=v[...,:self.c.width]
        sd=F.softplus(v[...,self.c.width:].clamp(-12,4))+self.c.sigma_min
        return mu,sd.expand_as(mu)


class TargetEncoder(nn.Module):
    """Training-only target encoder; code tables are shared, not duplicated."""
    def __init__(self,c):
        super().__init__();self.proj=nn.Linear(c.code_dim,c.width,bias=False)
        self.pos=nn.Parameter(torch.randn(max(c.levels),c.width)*.02)
        self.level=nn.Parameter(torch.randn(len(c.levels),c.width)*.02)
        self.score=nn.Linear(c.width,1)

    def forward(self,codes,tables,offsets):
        levels=[]
        for i,table in enumerate(tables):
            a,b=offsets[i:i+2]
            z=self.proj(table(codes[:,a:b]))+self.pos[:b-a]+self.level[i]
            levels.append((self.score(z).softmax(1)*z).sum(1))
        return torch.stack(levels,1).mean(1)


class AstraGRAM(AstraHRM):
    def __init__(self,c=None,include_posterior=True):
        c=c or GramConfig();super().__init__(c)
        self.prior=GaussianHead(c,2)
        if include_posterior:
            self.posterior=GaussianHead(c,3)
            self.target_encoder=TargetEncoder(c)

    def _trajectory(self,mem,target=None,noise=None,mean=False,truncated=False,retain_states=False,phases=1):
        c=self.config;b=len(mem);pooled=mem.mean(1)
        if phases<1 or int(phases)!=phases:raise ValueError('phases must be positive integer')
        if truncated and phases!=1:raise ValueError('Last-transition surrogate uses one declared phase per optimizer update')
        total_steps=c.slow_steps*phases
        h0,l0=self.init_proj(pooled).chunk(2,-1)
        h=self.h_init[None]+h0[:,None];l=self.l_init[None]+l0[:,None]
        ctx=pooled[:,None].expand(-1,c.slots,-1)
        target_mem=None if target is None else self.target_encoder(target,self.code,self.offsets)[:,None].expand(-1,c.slots,-1)
        if noise is None:
            noise=torch.zeros(total_steps,b,c.slots,c.width,device=mem.device) if mean else torch.randn(total_steps,b,c.slots,c.width,device=mem.device)
        assert noise.shape==(total_steps,b,c.slots,c.width)
        kls=[];sigmas=[];trace=[]
        for r in range(total_steps):
            tracked=not truncated or r==total_steps-1
            with nullcontext() if tracked else torch.no_grad():
                for _ in range(c.fast_steps):l=self.fast(l,torch.cat([mem,h],1))
                u=self.slow(h,torch.cat([mem,l],1))
                mp,sp=self.prior(u,ctx)
                if target is not None:
                    mq,sq=self.posterior(u,ctx,target_mem)
                    # Both conditionals are evaluated on this SAME posterior path.
                    # Softplus guarantees positive scales. The runner checks
                    # finite gradients; avoid four GPU synchronizations here.
                    if tracked:kls.append(gaussian_kl(mq,sq,mp,sp,validate=False).sum((1,2)))
                    mu,sd=mq,sq
                else:mu,sd=mp,sp
                h=u+mu+sd*noise[r]
                if retain_states and h.requires_grad:h.retain_grad()
                trace.append(h);sigmas.append(sp.detach().flatten())
        kl=torch.stack(kls).sum(0) if kls else torch.zeros(b,device=mem.device)
        return h,dict(kl=kl,sigma=torch.cat(sigmas),trace=trace)

    def reason(self,mem,retain_states=False):
        h,d=self._trajectory(mem,retain_states=retain_states)
        return h,d['trace']

    def full_bptt(self,mem,target,noise=None,retain_states=False,phases=1):
        return self._trajectory(mem,target,noise,retain_states=retain_states,phases=phases)

    def gram_truncated_last_transition(self,mem,target,noise=None,retain_states=False):
        return self._trajectory(mem,target,noise,truncated=True,retain_states=retain_states)

    def conditional_nll(self,mem,h,y,corrupt=0.):
        conditioning=y
        if self.training and corrupt:
            conditioning=torch.where(torch.rand(y.shape,device=y.device)<corrupt,
                                     torch.randint(self.config.vocab,y.shape,device=y.device),y)
        z=self.decode_hidden(mem,h,conditioning);terms=[]
        for i in range(len(self.config.levels)):
            a,b=self.offsets[i:i+2];logits=self.logits(z[:,a:b],i)
            terms.append(F.cross_entropy(logits.reshape(-1,self.config.vocab),y[:,a:b].reshape(-1),reduction='none').view(len(y),b-a).sum(1))
        return torch.stack(terms,1)

    def forward(self,x,state,y,objective='per_code',corrupt=0.,noise=None,gradient_mode='full_bptt',phases=1,auxiliary_weight=0.):
        assert objective=='per_code','GRAM ELBO requires consistent code-sum/KL units'
        if corrupt:raise ValueError('Clean terminal ELBO cannot silently use corrupted conditioning')
        if auxiliary_weight<0:raise ValueError('Negative auxiliary weight')
        mem=self.encode_context(x,state)
        if gradient_mode=='full_bptt':h,d=self.full_bptt(mem,y,noise,phases=phases)
        elif gradient_mode=='gram_truncated_last_transition':
            if phases!=1 or auxiliary_weight:raise ValueError('Truncated and multi-phase auxiliary objectives are separate')
            h,d=self.gram_truncated_last_transition(mem,y,noise)
        else:raise ValueError('Unknown gradient mode')
        terms=self.conditional_nll(mem,h,y)
        n=self.offsets[-1];nll=terms.sum(1).mean()/n;kl=d['kl'].mean()
        loss=nll+kl/n
        auxiliary=loss.new_zeros(())
        if auxiliary_weight:
            if phases<2:raise ValueError('Intermediate supervision requires multiple phases')
            early=d['trace'][self.config.slow_steps-1:-1:self.config.slow_steps]
            auxiliary=torch.stack([self.conditional_nll(mem,state,y).sum(1).mean()/n for state in early]).mean()
            loss=loss+auxiliary_weight*auxiliary
        coords=self.config.slots*self.config.width*(self.config.slow_steps*phases if gradient_mode=='full_bptt' else 1)
        return loss,dict(nll=nll.detach(),per_level=(terms.mean(0)/torch.tensor(self.config.levels,device=y.device)).detach(),
                         kl_sum=kl.detach(),kl_per_coordinate=(kl/coords).detach(),
                         auxiliary_nll=auxiliary.detach(),
                         objective_kind='deep_supervision_augmented' if auxiliary_weight else ('terminal_full_ELBO' if gradient_mode=='full_bptt' else 'last_transition_surrogate'),
                         sigma_median=d['sigma'].median(),sigma_min=d['sigma'].min(),sigma_max=d['sigma'].max())

    @torch.inference_mode()
    def score_prior(self,x,state,y,samples=4):
        """Finite-MC logmeanexp estimate; not exact marginal likelihood."""
        mem=self.encode_context(x,state);logps=[];levels=[]
        for _ in range(samples):
            h,_=self._trajectory(mem);terms=self.conditional_nll(mem,h,y)
            logps.append(-terms.sum(1));levels.append(terms.mean(0))
        logp=torch.logsumexp(torch.stack(logps),0)-math.log(samples)
        return dict(nll=(-logp.mean()/self.offsets[-1]),
                    per_level=torch.stack(levels).mean(0)/torch.tensor(self.config.levels,device=y.device))

    @torch.inference_mode()
    def generate(self,x,state,temperature=0.,mean=False,noise=None):
        # There is deliberately no target/posterior argument in this public API.
        mem=self.encode_context(x,state);h,_=self._trajectory(mem,noise=noise,mean=mean)
        return self.generate_from_state(mem,h,temperature)

    @torch.inference_mode()
    def posterior_diagnostic(self,x,state,y):
        mem=self.encode_context(x,state);h,d=self.full_bptt(mem,y)
        return self.generate_from_state(mem,h),d

    def inference_state_dict(self):
        return {k:v.detach().cpu() for k,v in self.state_dict().items()
                if not k.startswith(('posterior.','target_encoder.'))}


