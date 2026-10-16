"""Executable V10.1 specification. Reuses unchanged V10 encoder primitives.

Default configuration is D+J+H+S: both CRC paths, normalized visit supervision,
sampled chunk readout and utility-based code exploration. No EMA or memory writes
occur in forward(). Source dependency hashes are recorded by validation tools.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from v10_10M.config import ModelConfig
from v10_10M.model import V10, RMSNorm, Block, Decoder, geometry


@dataclass
class Architecture:
    encoder_concept_crc: bool = True
    concept_decoder_crc: bool = True
    deep_supervision: bool = True
    chunk_objective: bool = True
    utility_exploration: bool = True

    @classmethod
    def arm(cls, name):
        if name not in ('A','B','C','D','J','H','S'):
            raise ValueError('Choose A/B/C/D/J/H/S')
        return cls(name in ('B','D','J','H','S'), name in ('C','D','J','H','S'),
                   name in ('J','H','S'), name in ('H','S'), name == 'S')


class EncoderConceptCRC(nn.Module):
    def __init__(self, c):
        super().__init__(); d=c.width
        self.norms=nn.ModuleList([RMSNorm(d) for _ in range(2)])
        self.projections=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(2)])
        self.route_norm=RMSNorm(d); self.route=nn.Linear(d,2)
        self.scale=nn.Parameter(torch.full((d,),.1))
        nn.init.zeros_(self.route.weight); nn.init.zeros_(self.route.bias)
        for p in self.projections: nn.init.xavier_uniform_(p.weight)

    def sources(self, early, final):
        return torch.stack([p(n(x)) for p,n,x in zip(self.projections,self.norms,(early,final))],-2)

    def forward(self, state, sources):
        weights=self.route(self.route_norm(state)).softmax(-1)
        return state+self.scale*(weights.unsqueeze(-1)*sources).sum(-2)


class Decoder101(nn.Module):
    """Reuse exact baseline parameters; expand only the CRC routers."""
    align=staticmethod(Decoder.align)

    def __init__(self, original, c):
        super().__init__(); self.c=c
        for name in ('fusion','src_norm','src_proj','route_norm','scales','layers','outn'):
            setattr(self,name,getattr(original,name))
        self.routes=nn.ModuleList([nn.Linear(c.width,3) for _ in original.routes])
        with torch.no_grad():
            for old,new in zip(original.routes,self.routes):
                new.weight[:2].copy_(old.weight);new.bias[:2].copy_(old.bias)
                new.weight[2].zero_();new.bias[2]=-2
        self.concept_projection=nn.Linear(c.width,c.width,bias=False)
        nn.init.xavier_uniform_(self.concept_projection.weight)

    def forward(self, early, final, forecast, mask, rope, u, lengths,
                ablate_pq=False, ablate_continuous=False, zero_continuous=False,
                detach_pq=False, detach_continuous=False):
        t=final.shape[1]
        guidance=self.align(forecast.detach() if detach_pq else forecast,t)
        if ablate_pq: guidance=torch.zeros_like(guidance)
        x=final+self.fusion*guidance
        enc=[p(n(h)) for p,n,h in zip(self.src_proj,self.src_norm,(early,final))]
        continuous=self.align(self.concept_projection(u.detach() if detach_continuous else u),t)
        if zero_continuous:continuous=torch.zeros_like(continuous)
        sources=torch.stack(enc+[continuous],-2)
        pos=torch.arange(t,device=final.device)
        legal=(pos[None,:]>=7) & (pos[None,:]<lengths[:,None])
        if ablate_continuous:legal=torch.zeros_like(legal)
        routes=[]
        for i,layer in enumerate(self.layers):
            logits=self.routes[i](self.route_norm[i](x))
            logits=torch.cat((logits[...,:2],logits[...,2:].masked_fill(~legal[...,None],-torch.inf)),-1)
            weights=logits.softmax(-1);routes.append(weights.detach())
            x=x+self.scales[i]*(sources*weights.unsqueeze(-1)).sum(-2)
            x=layer(x,mask,rope)
        return self.outn(x), routes


class ChunkReadout(nn.Module):
    def __init__(self,c):
        super().__init__();d=c.width
        self.projection=nn.Linear(d,d,bias=False)
        self.position=nn.Parameter(torch.randn(8,d)*.02)
        self.start=nn.Parameter(torch.randn(d)*.02)
        self.block=Block(c);self.outn=RMSNorm(d);self.c=c

    def forward(self, forecast, previous_ids, embedding):
        # previous_ids has seven earlier target-chunk tokens, never the current one.
        b=forecast.shape[0]
        previous=torch.cat((self.start[None,None].expand(b,1,-1),embedding(previous_ids)),1)
        x=previous+self.position+self.projection(forecast.repeat_interleave(4,1))
        mask,rope=geometry(8,torch.full((b,),8,device=x.device),x.device,self.c.width//self.c.heads)
        return F.linear(self.outn(self.block(x,mask,rope)),embedding.weight)


class V101(V10):
    def __init__(self,c=None,architecture=None):
        super().__init__(c)
        self.architecture=architecture or Architecture()
        if self.architecture.utility_exploration and not self.architecture.chunk_objective:
            raise ValueError('Utility exploration requires the chunk readout')
        self.encoder_concept_crc=EncoderConceptCRC(self.c) if self.architecture.encoder_concept_crc else None
        if self.architecture.concept_decoder_crc:self.decoder=Decoder101(self.decoder,self.c)
        self.chunk_readout=ChunkReadout(self.c) if self.architecture.chunk_objective else None
        self.memory_era=None

    def candidates(self,*args,**kwargs):
        raise TypeError('V10.1 candidates require cached CRC sources; use candidate_forecasts and v10_1.losses')

    def bind_memory_era(self,era,sigma):
        if not era or not math.isfinite(sigma) or sigma<=0 or self.training or any(p.requires_grad for p in self.parameters()):
            raise ValueError('Freeze and fingerprint the model before binding a memory era')
        self.memory_era=era;self.memory_sigma=sigma

    def concept(self,pooled,lengths,codes,sources=None,xi=None,visits=2):
        if visits not in (1,2):raise ValueError('Only the one-visit diagnostic or two-visit model is defined')
        b,g,p,d=pooled.shape
        if not g:return [pooled for _ in range(visits)]
        e=self.core.codes(codes)
        if xi is not None:
            if self.training and bool(torch.count_nonzero(xi)):
                raise ValueError('Pretraining requires xi=0')
            e=e+xi
        x=(pooled+self.core.code_projection(e).unsqueeze(-2)).reshape(b,g*p,d)
        anchor=self.core.anchor_norm(x);state=anchor
        mask,rope=geometry(g*p,lengths//8*p,x.device,d//self.c.heads,group_slots=p)
        z=sources.reshape(b,g*p,2,d) if sources is not None else None
        history=[]
        for _ in range(visits):
            state=state+self.core.anchor_scale*anchor
            if self.encoder_concept_crc is not None:state=self.encoder_concept_crc(state,z)
            for layer in self.core.layers:state=layer(state,mask,rope)
            state=self.core.exit_norm(state);history.append(state.reshape(b,g,p,d))
        return history

    def forward(self,ids,lengths=None,indices=None,codes=None,xi=None,bank=None,
                write_pass=False,ablate=False,ablate_pq=False,ablate_continuous=False,
                zero_continuous=False,visits=2,correction_override=None,
                detach_pq=False,detach_continuous=False):
        if lengths is None:lengths=torch.full((len(ids),),ids.shape[1],device=ids.device,dtype=torch.long)
        # Legacy "no concepts" must disable BOTH routes, never mislabel PQ-only ablation.
        ablate_pq=ablate_pq or ablate
        ablate_continuous=ablate_continuous or ablate
        if ids.ndim!=2 or bool((lengths<1).any()) or bool((lengths>ids.shape[1]).any()):
            raise ValueError('Expected nonempty sequences with valid right-padded lengths')
        if bank is not None and visits!=2:raise ValueError('A recurrence ablation cannot reuse a populated memory era')
        if bank is not None and (self.memory_era!=bank.era or self.memory_sigma!=bank.sigma or self.training or any(p.requires_grad for p in self.parameters())):
            raise ValueError('Memory inference requires a frozen model bound to this representation era')
        early,h,mask,rope=self.encode(ids,lengths,indices)
        pooled=self.pooler(h)
        sources=self.encoder_concept_crc.sources(self.pooler(early),pooled) if self.encoder_concept_crc is not None else None
        selector=self.core.select(pooled);selected=selector.argmax(-1)
        if codes is None:codes=selected
        elif codes.ndim!=2 or codes.shape[0]!=len(ids) or codes.shape[1]>selected.shape[1]:
            raise ValueError('Committed codes must be a valid prefix')
        elif codes.shape[1]<selected.shape[1]:codes=torch.cat((codes,selected[:,codes.shape[1]:]),1)
        states=self.concept(pooled,lengths,codes,sources,xi,visits)
        pre=states[-1];queries=pre.flatten(-2)/math.sqrt(2*self.c.width)
        correction=None
        if correction_override is not None:
            if bank is not None or correction_override.shape!=queries.shape:raise ValueError('Invalid correction override')
            correction=correction_override
        elif bank is not None:
            correction=bank.read(queries,codes)
            if write_pass:correction=correction.detach().requires_grad_(True)
        u=pre if correction is None else pre+correction.reshape_as(pre)
        forecast,probs=self.head(u)
        forecasts=[self.head(s)[0] for s in states[:-1]]+[forecast]
        if self.architecture.concept_decoder_crc:
            hidden,routes=self.decoder(early,h,forecast,mask,rope,u,lengths,ablate_pq,
                ablate_continuous,zero_continuous,detach_pq,detach_continuous)
        else:
            hidden=self.decoder(early,h,forecast.detach() if detach_pq else forecast,mask,rope,ablate_pq)
            routes=[]
        valid=torch.arange(pooled.shape[1],device=ids.device)[None,:]<(lengths//8)[:,None]
        return dict(logits=F.linear(hidden,self.embedding.weight),pooled=pooled,sources=sources,
                    forecast=forecast,forecasts=forecasts,states=states,selector=selector,codes=codes,
                    valid_groups=valid,probs=probs,u=u,pre_memory=pre,correction=correction,
                    queries=queries,decoder_routes=routes)

    def candidate_forecasts(self,output,lengths,boundaries):
        if bool((boundaries<0).any()) or bool((boundaries>=output['valid_groups'].sum(-1)).any()):
            raise ValueError('Each candidate boundary must be a completed current group')
        k=self.c.codes;b=len(lengths);n=int(boundaries.max())+1
        def expand(x):return x[:,:n].repeat_interleave(k,0)
        pooled=expand(output['pooled']);sources=expand(output['sources']) if output['sources'] is not None else None
        history=output['codes'][:,:n].repeat_interleave(k,0).clone()
        rows=torch.arange(b*k,device=pooled.device);at=boundaries.repeat_interleave(k)
        history[rows,at]=torch.arange(k,device=pooled.device).repeat(b)
        lens=torch.minimum(lengths,(boundaries+1)*8).repeat_interleave(k)
        states=self.concept(pooled,lens,history,sources)
        return self.head(states[-1][rows,at])[0].reshape(b,k,2,self.c.width)

    def chosen_forecasts(self,output,lengths,boundaries,choices):
        """Recompute only winning branches with gradients after detached K-way scoring."""
        n=int(boundaries.max())+1;b=len(lengths)
        history=output['codes'][:,:n].clone();rows=torch.arange(b,device=history.device)
        history[rows,boundaries]=choices
        source=output['sources'][:,:n] if output['sources'] is not None else None
        states=self.concept(output['pooled'][:,:n],torch.minimum(lengths,(boundaries+1)*8),history,source)
        return self.head(states[-1][rows,boundaries])[0]


def token_loss(output,targets,target_mask):
    valid=targets.masked_fill(~target_mask,-100)
    return F.cross_entropy(output['logits'].flatten(0,1),valid.flatten(),ignore_index=-100,reduction='sum')/target_mask.sum().clamp_min(1)


def latent_errors(output):
    valid=output['valid_groups'][:,1:]
    target=output['pooled'][:,1:].detach()
    errors=[(f[:,:-1]-target).square().mean((-1,-2)) for f in output['forecasts']]
    return [(e*valid).sum()/valid.sum().clamp_min(1) for e in errors]


def chunk_batch(ids,lengths,targets,target_mask,valid_groups,generator=None,boundaries=None):
    """One sampled boundary per eligible row; returns only eligible rows.

    targets[:,i] is x[i+1]. Current groups must be complete; future chunks may
    be partial and may end in EOS. Prompt targets remain context but not loss.
    """
    b,t=ids.shape;g=valid_groups.shape[1];device=ids.device
    if not g:return None
    positions=8*(torch.arange(g,device=device)+1)[:,None]+torch.arange(8,device=device)[None,:]-1
    index=positions.clamp_max(t-1)
    eligible_targets=target_mask[:,index] & (positions[None]<lengths[:,None,None])
    eligible=valid_groups & eligible_targets.any(-1)
    rows=eligible.any(-1).nonzero().flatten()
    if not len(rows):return None
    if boundaries is None:
        chosen=[]
        for row in rows:
            options=eligible[row].nonzero().flatten()
            choice=int(torch.randint(len(options),(1,),generator=generator,device='cpu'))
            chosen.append(options[choice])
        boundaries=torch.stack(chosen)
    elif boundaries.shape!=(len(rows),) or not bool(eligible[rows,boundaries].all()):
        raise ValueError('Invalid sampled chunk boundary')
    pos=positions[boundaries]; safe=pos.clamp_max(t-1)
    labels=targets[rows[:,None],safe]
    mask=eligible_targets[rows,boundaries]
    # At chunk position j>0, previous input is x[b+j-1] = targets[row,b+j-2].
    previous=targets[rows[:,None],safe[:,:7]]
    return dict(rows=rows,boundaries=boundaries,labels=labels,mask=mask,previous=previous,
                eligible_boundaries=int(eligible.sum()),supervised_targets=int(mask.sum()))


def per_document_chunk_ce(model,forecast,chunk):
    logits=model.chunk_readout(forecast,chunk['previous'],model.embedding)
    ce=F.cross_entropy(logits.flatten(0,1),chunk['labels'].flatten(),reduction='none').reshape_as(chunk['mask'])
    return (ce*chunk['mask']).sum(-1)/chunk['mask'].sum(-1).clamp_min(1)


def losses(model,output,ids,lengths,targets,target_mask,candidate_update=False,generator=None,boundaries=None,candidate_strategy='recompute'):
    if candidate_strategy not in ('recompute','all_graphs'):raise ValueError('Unknown candidate differentiation strategy')
    token=token_loss(output,targets,target_mask);zero=token*0
    visit_latent=latent_errors(output)
    latent=(visit_latent[0]/3+2*visit_latent[-1]/3) if model.architecture.deep_supervision else visit_latent[-1]
    valid=output['valid_groups'][:,1:]
    vq=model.head.fitting_loss(output['pooled'][:,1:],valid) if valid.numel() else zero
    chunk_loss=explore=selector=zero;chunk=None;selector_regret=None;winners=None
    if model.chunk_readout is not None:
        chunk=chunk_batch(ids,lengths,targets,target_mask,output['valid_groups'],generator,boundaries)
        if chunk is not None:
            rows,at=chunk['rows'],chunk['boundaries']
            ces=[per_document_chunk_ce(model,f[rows,at],chunk).mean() for f in output['forecasts']]
            chunk_loss=ces[0]/3+2*ces[-1]/3 if model.architecture.deep_supervision else ces[-1]
    if candidate_update:
        if model.architecture.utility_exploration:
            rows=chunk['rows'] if chunk is not None else torch.empty(0,device=ids.device,dtype=torch.long)
            at=chunk['boundaries'] if chunk is not None else rows
        else:
            rows=(output['valid_groups'].sum(-1)>1).nonzero().flatten()
            at=torch.tensor([int(torch.randint(int(output['valid_groups'][r].sum())-1,(1,),generator=generator)) for r in rows],device=ids.device,dtype=torch.long)
        if len(rows):
            subset={k:(output[k][rows] if output[k] is not None else None) for k in ['pooled','sources','codes','valid_groups']}
            has_next=(at+1)<output['valid_groups'][rows].sum(-1)
            target=output['pooled'][rows,(at+1).clamp_max(output['pooled'].shape[1]-1)].detach()
            # min_k differentiates only the winning branch. Scoring with no graph
            # and recomputing that branch preserves the objective and avoids K
            # retained recurrent graphs. All modules here are deterministic.
            with torch.set_grad_enabled(torch.is_grad_enabled() and candidate_strategy=='all_graphs'):
                predictions=model.candidate_forecasts(subset,lengths[rows],at)
                mse=(predictions-target[:,None]).square().mean((-1,-2))*has_next[:,None]
                if model.architecture.utility_exploration:
                    expanded={k:chunk[k].repeat_interleave(model.c.codes,0) for k in ['previous','labels','mask']}
                    utility=per_document_chunk_ce(model,predictions.flatten(0,1),expanded).reshape(len(rows),model.c.codes)+.1*mse
                else:utility=mse
                minimum,winners=utility.min(-1)
            if candidate_strategy=='recompute':
                chosen=model.chosen_forecasts(subset,lengths[rows],at,winners.detach())
                mse=(chosen-target).square().mean((-1,-2))*has_next
                chosen_utility=per_document_chunk_ce(model,chosen,chunk)+.1*mse if model.architecture.utility_exploration else mse
                explore=chosen_utility.mean()
            else:explore=minimum.mean()
            selector=F.cross_entropy(output['selector'][rows,at],winners.detach())
            selected=output['codes'][rows,at]
            selector_regret=float((utility.detach().gather(1,selected[:,None]).flatten()-minimum.detach()).mean())
    total=token+.1*latent+.01*vq+.1*chunk_loss+.1*explore+.05*selector
    return dict(total=total,token=token,latent=latent,vq=vq,chunk=chunk_loss,explore=explore,
                selector=selector,latent_visits=visit_latent,candidate_executed=bool(candidate_update and winners is not None),
                selector_regret=selector_regret,chunk_valid_documents=0 if chunk is None else len(chunk['rows']),
                chunk_supervised_targets=0 if chunk is None else chunk['supervised_targets'])
