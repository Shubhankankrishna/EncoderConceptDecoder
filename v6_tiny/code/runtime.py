"""Atomic recovery, exact loss units and coherent candidate replay."""
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import (Foundation, Core, Adapters, System, freeze, chunk_info, ball, K, DZ)
from data import WALK_START, T

DEVICE = os.environ.get('V6_DEVICE') or ('mps' if torch.backends.mps.is_available() else 'cpu')
ROOT = Path(__file__).resolve().parents[1]


def sync(device=DEVICE):
    if str(device)=='mps':torch.mps.synchronize()


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with open(temp,'w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def atomic_save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with open(temp,'wb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


class Log:
    def __init__(self,path):self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
    def __call__(self,**obj):
        obj['utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        with open(self.path,'a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush()


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def cpu_tree(obj):
    if isinstance(obj,torch.Tensor):return obj.detach().cpu()
    if isinstance(obj,dict):return {k:cpu_tree(v) for k,v in obj.items()}
    if isinstance(obj,list):return [cpu_tree(v) for v in obj]
    if isinstance(obj,tuple):return tuple(cpu_tree(v) for v in obj)
    return obj


def save_checkpoint(path,system,opt,step,mix,generator,config,extra=None):
    sync(next(system.parameters()).device)
    obj=dict(foundation=system.foundation.state_dict(),core=system.core.state_dict() if system.core else None,
             adapters=system.adapters.state_dict() if system.adapters else None,
             adapter_kind=system.adapters.kind if system.adapters else None,optimizer=opt.state_dict(),step=step,
             mixer=mix.state(),generator=generator.get_state(),torch_rng=torch.get_rng_state(),
             numpy_rng=np.random.get_state(),python_rng=random.getstate(),config=config,extra=extra or {})
    saved=cpu_tree(obj)
    def finite(value):
        if isinstance(value,torch.Tensor):return not value.is_floating_point() or bool(torch.isfinite(value).all())
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
        return True
    if not all(finite(saved[key]) for key in ('foundation','core','adapters','optimizer')):
        raise FloatingPointError('Nonfinite checkpoint rejected; previous atomic checkpoint preserved')
    atomic_save(path,saved)


def load_system(path,device=DEVICE):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    f=Foundation();f.load_state_dict(ck['foundation'])
    core=Core() if ck.get('core') is not None else None
    if core:core.load_state_dict(ck['core'])
    adapters=Adapters(ck['adapter_kind']) if ck.get('adapters') is not None else None
    if adapters:adapters.load_state_dict(ck['adapters'])
    return System(f,core,adapters).to(device),ck


def optimizer(params,lr,wd=0.1):
    ps=[p for p in params if p.requires_grad]
    return torch.optim.AdamW([dict(params=[p for p in ps if p.dim()>=2],weight_decay=wd),
                             dict(params=[p for p in ps if p.dim()<2],weight_decay=0)],lr=lr,betas=(0.9,0.95))


def lr_at(step,total,peak,warm):
    warm=min(warm,max(1,total//10))
    if step<warm:return peak*(step+1)/warm
    p=(step-warm)/max(1,total-warm)
    return peak*(0.1+0.9*0.5*(1+math.cos(math.pi*p)))


def nll(logits,x,m):
    loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),x[:,1:].reshape(-1),reduction='none').view(x.shape[0],-1)
    return loss*m[:,:-1]


def balanced_sums(values,weights,domain,denominators=None):
    nums=torch.stack([(values*(domain==i).reshape(-1,*([1]*(values.ndim-1)))).sum() for i in (0,1)])
    dens=torch.stack([(weights*(domain==i).reshape(-1,*([1]*(weights.ndim-1)))).sum() for i in (0,1)]) if denominators is None else denominators
    means=nums/dens.clamp_min(1)
    return means.mean(),means,dens


def ncp_weights(batch):
    cd,valid=chunk_info(batch['docs'])
    cm=(valid[:,:-1]&valid[:,1:]&(cd[:,:-1]==cd[:,1:])).float()
    # Forecast only pairs whose target has at least one scored next-token target.
    m=batch['m']; scored=F.pad(m,(0,1))[:,3:3+4*(cd.shape[1]-1)].view(len(m),-1,4).amax(-1)>0
    return cm*scored


def controls(batch,generator,lengths=None):
    b,t=batch['x'].shape;m=t//K;device=batch['x'].device
    if lengths is None:lengths=torch.tensor([1,2,4])[torch.randint(3,(b,),generator=generator)]
    boundaries=torch.randint(1,max(2,m//2),(b,),generator=generator)
    boundaries=torch.where(batch['domain'].cpu()==1,torch.full_like(boundaries,WALK_START//K-1),boundaries-1)
    chunks=torch.arange(m)[None].expand(b,m)
    cd=chunk_info(batch['docs'])[0].cpu()
    starts=torch.zeros_like(cd)
    change=torch.ones_like(cd,dtype=torch.bool);change[:,1:]=cd[:,1:]!=cd[:,:-1]
    starts=torch.where(change,chunks,starts).cummax(1).values
    begin=torch.maximum(starts,boundaries[:,None])
    text_index=begin+(chunks-begin).clamp_min(0)//lengths[:,None]
    graph_index=(chunks-boundaries[:,None]).clamp_min(0)//lengths[:,None]
    index=torch.where((batch['domain'].cpu()==1)[:,None],graph_index,text_index).clamp_max(m-1).to(device)
    enabled=((chunks>=boundaries[:,None])&(cd>=0)).to(device)
    bank=ball(torch.randn(b,m,DZ,generator=generator)).to(device)
    z=torch.gather(bank,1,index[:,:,None].expand(b,m,DZ))*enabled[:,:,None]
    suffix_start=(4*boundaries+3).to(device)
    suffix_mask=batch['m']*(torch.arange(t,device=device)[None]>=suffix_start[:,None])
    return z,suffix_mask,index,enabled,lengths


def candidate_rows(system,batch,schedules,mask,forward=None,context=None):
    forward=forward or system
    if context is None:
        with torch.no_grad():context=getattr(system,'optimized_context',system.context)(batch['x'],batch['docs'])
    h,c,cd=context
    if len(schedules)>1:
        # Candidate axis becomes a batch axis; the effective training batch/denominators are unchanged.
        k=len(schedules);b=len(h)
        def repeat(v):return v.unsqueeze(0).expand(k,*v.shape).reshape(k*b,*v.shape[1:])
        hh,cc,ccd,docs=repeat(h),repeat(c),repeat(cd),repeat(batch['docs'])
        zs=torch.stack(schedules).reshape(k*b,*schedules[0].shape[1:])
        if forward is system:
            chat,a,_=system.plans(cc,ccd,zs)
            logits=system.foundation.decode(hh,chat,a,docs)
        else:logits=forward(hh,cc,ccd,zs,docs)
        return nll(logits,repeat(batch['x']),repeat(mask)).sum(1).view(k,b)
    rows=[]
    for z in schedules:
        if forward is system:
            chat,a,_=system.plans(c,cd,z)
            logits=system.foundation.decode(h,chat,a,batch['docs'])
        else:logits=forward(h,c,cd,z,batch['docs'])
        rows.append(nll(logits,batch['x'],mask).sum(1))
    return torch.stack(rows)


def selection_loss(scores,mask,domain,beta,denominators=None):
    counts=mask[:,:-1].sum(1);eligible=(counts>0).float()
    rows=((1-beta)*scores.min(0).values+beta*scores.mean(0))/counts.clamp_min(1)
    return balanced_sums(rows*eligible,eligible,domain,denominators)[0]


def candidate_backward(system,batch,schedules,mask,beta,forward=None,replay=True):
    with torch.no_grad():context=getattr(system,'optimized_context',system.context)(batch['x'],batch['docs'])
    if not replay:
        scores=candidate_rows(system,batch,schedules,mask,forward,context)
        loss=selection_loss(scores,mask,batch['domain'],beta);loss.backward();return float(loss.detach())
    with torch.no_grad():scores=candidate_rows(system,batch,schedules,mask,forward,context);winner=scores.argmin(0)
    k=len(schedules);counts=mask[:,:-1].sum(1);eligible=counts>0
    denom=torch.stack([((batch['domain']==d)&eligible).sum() for d in (0,1)]).clamp_min(1)
    weights=0.5*eligible/denom[batch['domain']]/counts.clamp_min(1)
    for i,z in enumerate(schedules):
        row=candidate_rows(system,batch,[z],mask,forward,context)[0]
        w=((1-beta)*(winner==i).float()+beta/k)*weights
        (row*w).sum().backward()
    return float(selection_loss(scores,mask,batch['domain'],beta))


def implementation_hash():
    h=hashlib.sha256()
    for p in sorted((ROOT/'code').glob('*.py')):h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()
