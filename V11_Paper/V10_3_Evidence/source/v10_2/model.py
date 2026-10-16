"""V10.2 RF specification: completed concept history plus forecast-only utility."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from v10_10M.model import V10, Decoder, RMSNorm, FFN
from v10_1.model import V101, EncoderConceptCRC, token_loss, chunk_batch

@dataclass
class Architecture:
    variant: str = 'RF'
    encoder_concept_crc: bool = True
    concept_decoder_crc: bool = True
    forecast_only: bool = True
    history_reader: bool = True
    deep_supervision: bool = True

class ForecastReadout(nn.Module):
    def __init__(self,c):
        super().__init__();d=c.width
        self.projection=nn.Linear(d,d,bias=False)
        nn.init.xavier_uniform_(self.projection.weight)
        self.position=nn.Parameter(torch.randn(8,d)*.02)
        self.norm=RMSNorm(d);self.ffn=FFN(d);self.outn=RMSNorm(d)
    def forward(self,forecast,embedding):
        x=self.projection(forecast).repeat_interleave(4,dim=1)+self.position
        return F.linear(self.outn(x+self.ffn(self.norm(x))),embedding.weight)

def history_geometry(t,g,lengths,dh,device):
    pos=torch.arange(t,device=device)
    ends=(8*torch.arange(g,device=device)+7).repeat_interleave(2)
    allowed=(ends[None,None,:]<=pos[None,:,None]) & (ends[None,None,:]<lengths[:,None,None]) & (pos[None,:,None]<lengths[:,None,None])
    # Sentinel is enabled only when no real key is available, including T<8.
    mask=torch.cat((allowed,~allowed.any(-1,keepdim=True)),-1)[:,None]
    freq=10000**(torch.arange(dh//2,device=device).float()/(dh//2))
    def rope(p):
        a=p.float()[:,None]/freq
        return a.cos(),a.sin()
    return mask,rope(pos),rope(ends)

class ConceptReader(nn.Module):
    def __init__(self,c):
        super().__init__();self.heads=c.heads;self.dh=c.width//c.heads
        self.norm=RMSNorm(c.width)
        self.q,self.k,self.v,self.o=[nn.Linear(c.width,c.width,bias=False) for _ in range(4)]
        for p in (self.q,self.k,self.v):nn.init.xavier_uniform_(p.weight)
        nn.init.xavier_uniform_(self.o.weight,gain=.1)
    def forward(self,x,memory,geometry):
        b,t,d=x.shape;mask,qr,kr=geometry
        def split(z):return z.reshape(b,z.shape[1],self.heads,self.dh).transpose(1,2)
        def rotate(z,rope):
            a,b=z.chunk(2,-1);co,si=rope
            return torch.cat((a*co-b*si,a*si+b*co),-1)
        q=rotate(split(self.q(self.norm(x))),qr)
        k=rotate(split(self.k(memory)),kr);v=split(self.v(memory))
        sentinel=k.new_zeros(b,self.heads,1,self.dh)
        k=torch.cat((k,sentinel),-2);v=torch.cat((v,sentinel),-2)
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        return self.o(y.transpose(1,2).reshape(b,t,d))

class Decoder102(Decoder):
    def __init__(self,original,c):
        nn.Module.__init__(self);self.c=c
        for name in ('fusion','src_norm','src_proj','route_norm','routes','scales','layers','outn'):
            setattr(self,name,getattr(original,name))
        self.concept_projection=nn.Linear(c.width,c.width,bias=False)
        self.forecast_projection=nn.Linear(c.width,c.width,bias=False)
        for p in (self.concept_projection,self.forecast_projection):nn.init.xavier_uniform_(p.weight)
        self.readers=nn.ModuleList([ConceptReader(c) for _ in self.layers])
    def forward(self,early,final,forecast,mask,rope,u,lengths,ablate_pq=False,
                ablate_continuous=False,zero_continuous=False,detach_pq=False,detach_continuous=False,
                ablate_aligned=False,ablate_cross_pq=False,ablate_reader=False):
        t=final.shape[1];f=forecast.detach() if detach_pq else forecast
        state=u.detach() if detach_continuous else u
        guidance=self.align(f,t)
        if ablate_pq or ablate_aligned:guidance=torch.zeros_like(guidance)
        x=final+self.fusion*guidance
        cu=self.concept_projection(state);cf=self.forecast_projection(f)
        if ablate_continuous or zero_continuous:cu=torch.zeros_like(cu)
        if ablate_pq or ablate_cross_pq:cf=torch.zeros_like(cf)
        memory=(cu+cf).flatten(1,2)
        geom=history_geometry(t,u.shape[1],lengths,self.c.width//self.c.heads,x.device)
        enc=torch.stack([p(n(h)) for p,n,h in zip(self.src_proj,self.src_norm,(early,final))],-2)
        routes=[]
        for i,layer in enumerate(self.layers):
            w=self.routes[i](self.route_norm[i](x)).softmax(-1);routes.append(w.detach())
            x=x+self.scales[i]*(enc*w.unsqueeze(-1)).sum(-2)
            x=x+layer.attn(layer.an(x),mask,rope)
            if not ablate_reader:x=x+self.readers[i](x,memory,geom)
            x=x+layer.ffn(layer.fn(x))
        return self.outn(x),routes

class V102(V101):
    def __init__(self,c=None,architecture=None):
        V10.__init__(self,c)
        self.architecture=architecture or Architecture()
        if self.architecture!=Architecture():raise ValueError('Production implementation is the RF architecture')
        self.encoder_concept_crc=EncoderConceptCRC(self.c)
        self.decoder=Decoder102(self.decoder,self.c)
        self.forecast_readout=ForecastReadout(self.c)
        self.memory_era=None
    def place(self,device):
        super().place(device)
        # CPU hashing must never read a device buffer once per forward.
        self.engram.canonical=self.engram.canonical.cpu()
        return self
    def forward(self,ids,lengths=None,indices=None,codes=None,xi=None,bank=None,
                write_pass=False,ablate=False,ablate_pq=False,ablate_continuous=False,
                zero_continuous=False,visits=2,correction_override=None,
                detach_pq=False,detach_continuous=False,ablate_aligned=False,ablate_cross_pq=False,ablate_reader=False):
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
                ablate_continuous,zero_continuous,detach_pq,detach_continuous,
                ablate_aligned,ablate_cross_pq,ablate_reader)
        else:
            hidden=self.decoder(early,h,forecast.detach() if detach_pq else forecast,mask,rope,ablate_pq)
            routes=[]
        valid=torch.arange(pooled.shape[1],device=ids.device)[None,:]<(lengths//8)[:,None]
        return dict(logits=F.linear(hidden,self.embedding.weight),pooled=pooled,sources=sources,
                    forecast=forecast,forecasts=forecasts,states=states,selector=selector,codes=codes,
                    valid_groups=valid,probs=probs,u=u,pre_memory=pre,correction=correction,
                    queries=queries,decoder_routes=routes)


def future_ce(model,forecast,chunk,details=False):
    logits=model.forecast_readout(forecast,model.embedding)
    ce=F.cross_entropy(logits.flatten(0,1),chunk['labels'].flatten(),reduction='none').reshape_as(chunk['mask'])
    weight=torch.arange(1,9,device=ce.device,dtype=ce.dtype).rsqrt()*chunk['mask']
    per=(ce*weight).sum(-1)/weight.sum(-1).clamp_min(1e-12)
    return (per,ce) if details else per

def losses(model,output,ids,lengths,targets,target_mask,candidate_update=False,generator=None,
           boundaries=None,candidate_strategy='recompute',prepared_chunk=None,chunk_prepared=False):
    if candidate_strategy not in ('recompute','all_graphs'):raise ValueError('Unknown candidate strategy')
    token=token_loss(output,targets,target_mask);zero=token*0
    chunk=prepared_chunk if chunk_prepared else chunk_batch(ids,lengths,targets,target_mask,output['valid_groups'],generator,boundaries)
    visits=[zero,zero];future=explore=selector=zero;winner=None;utility=None;regret=zero;gap=zero;spread=zero;correct=zero
    offsets=zero.expand(8);offset_counts=zero.expand(8)
    if chunk is not None:
        rows,at=chunk['rows'],chunk['boundaries']
        details=[future_ce(model,f[rows,at],chunk,True) for f in output['forecasts']]
        visits=[v[0].mean() for v in details]
        future=visits[0]/3+2*visits[-1]/3
        ce=details[-1][1]
        offsets=(ce.detach()*chunk['mask']).sum(0);offset_counts=chunk['mask'].sum(0)
        if candidate_update:
            subset={k:output[k][rows] for k in ('pooled','sources','codes','valid_groups')}
            with torch.set_grad_enabled(torch.is_grad_enabled() and candidate_strategy=='all_graphs'):
                predictions=model.candidate_forecasts(subset,lengths[rows],at)
                expanded={k:chunk[k].repeat_interleave(model.c.codes,0) for k in ('labels','mask')}
                utility=future_ce(model,predictions.flatten(0,1),expanded).reshape(len(rows),model.c.codes)
                minimum,winner=utility.min(-1)
            if candidate_strategy=='recompute':
                chosen=model.chosen_forecasts(subset,lengths[rows],at,winner.detach())
                explore=future_ce(model,chosen,chunk).mean()
            else:explore=minimum.mean()
            selector=F.cross_entropy(output['selector'][rows,at],winner.detach())
            selected=output['codes'][rows,at];u=utility.detach();ranked=u.sort(-1).values
            regret=(u.gather(1,selected[:,None]).flatten()-minimum.detach()).mean()
            gap=(ranked[:,1]-ranked[:,0]).mean();spread=(ranked[:,-1]-ranked[:,0]).mean()
            correct=(selected==winner).float().mean()
    total=token+.1*future+.1*explore+.05*selector
    return dict(total=total,token=token,future=future,explore=explore,selector=selector,
                future_visits=visits,candidate_executed=winner is not None,selector_regret=regret,
                utility_gap=gap,utility_spread=spread,selector_agreement=correct,winners=winner,
                future_offset_numerators=offsets,future_offset_counts=offset_counts,
                future_documents=0 if chunk is None else len(chunk['rows']),
                future_supervised_targets=0 if chunk is None else int(chunk['mask'].sum()))
