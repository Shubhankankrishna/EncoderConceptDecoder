"""Privileged valid-target optimization: development diagnosis, never search credit."""
import time
import numpy as np
import torch
from data import DATA,T,K
from runtime import DEVICE,nll,atomic_json,sync
from generation import MaskedAdam,expand_bank,active_windows,generate

def privileged_reach(s,out,n=16,steps=32):
    ds=np.load(DATA/'dev_hard.npz');rows=[];begin=time.monotonic()
    for i in range(n):
        x=torch.as_tensor(ds['x'][i:i+1],device=DEVICE);docs=torch.as_tensor(ds['docs'][i:i+1],device=DEVICE);mask=torch.as_tensor(ds['m'][i:i+1],device=DEVICE);p=int(ds['prompt_len'][i]);ln=int((docs>=0).sum())
        bank=torch.zeros(1,T//K,s.cfg.dz,device=DEVICE,requires_grad=True);opt=MaskedAdam(bank,.03);active=active_windows([ln],p,1,DEVICE);losses=[];generations={}
        for step in range(steps+1):
            if step in (0,8,steps):
                r=generate(s,ds['x'][i:i+1,:p],bank.detach());generations[str(step)]=dict(success=bool(r['verification'][0]['success']),reason=r['verification'][0]['reason'],seconds=r['seconds'])
            loss=nll(s(x,docs,expand_bank(bank,p,1,T//K)),x,mask).sum()/mask.sum();losses.append(float(loss.detach()))
            if step==steps:break
            grad=torch.autograd.grad(loss,bank)[0];opt.update(grad,active,torch.ones(1,device=DEVICE,dtype=torch.bool))
        rows.append(dict(item=i,target_nll=losses,generation=generations))
    sync();result=dict(scope='Uses known valid held-out development targets; diagnostic only, no ordinary search credit',steps=steps,n=n,seconds=time.monotonic()-begin,rows=rows,
        success_rates={str(k):float(np.mean([r['generation'][str(k)]['success'] for r in rows])) for k in (0,8,steps)})
    atomic_json(out,result);return result
