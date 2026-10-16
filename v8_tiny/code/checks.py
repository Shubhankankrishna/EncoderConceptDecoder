"""Executable leakage, trained decoder, control, continual and gradient contracts."""
import argparse,copy,json,traceback
from pathlib import Path
import numpy as np
import torch
from model import *
from data import *
from runtime import *
from generation import Incremental,FastIncremental,generate,expand_bank,MaskedAdam,repair_gradient

def run(device=DEVICE,source=None,fast=False):
    torch.set_num_threads(4);seed_all(800);tests=[]
    def check(name,condition,**details):
        if not condition:raise AssertionError(name+': '+str(details))
        tests.append(dict(name=name,passed=True,**details))
    check('main and flat counts',count(System())==2448640 and count(System(Flat()))==2450048)
    rng=np.random.RandomState(82008)
    for task,layout in [('graph','shuffle'),('lookup','shuffle'),('compose','shuffle'),('bind','shuffle'),('graph','shifted'),('pointer','shuffle')]:
        for i in range(32):
            x,m,d,meta=gen_symbolic(rng,10,task,layout=layout);p=meta['prompt_len'];r=x[p:p+meta['response_len']];check(f'{task}/{layout} reference {i}',verify(x[:p],r)['success'])
        bad=r.copy();bad[0]=PAD;check(task+' rejects corrupt answer',not verify(x[:p],bad)['success'])
    s=load_system(source,device)[0] if source else System().to(device);s.eval();row=gen_symbolic(rng);x=torch.tensor(row[0][None,:],device=device);docs=torch.tensor(row[2][None,:],device=device)
    # Complete trained graph prefixes at all modulo-four offsets, not only text.
    with torch.no_grad():
        z=torch.zeros(1,T//K,DZ,device=device)
        for p in [1,2,3,4,121,122,123,124,125,126,127,128]:
            cache=Incremental(s,z);out=cache.prefill(x[:,:p]);full=s(x[:,:p],torch.zeros_like(x[:,:p]),z[:,:p//K])[:,-1];error=float((out-full).abs().max());check('dynamic prefix '+str(p),error<3e-4,error=error)
            for pos in range(p,min(p+5,150)):
                out=cache.step(x[:,pos]);full=s(x[:,:pos+1],torch.zeros_like(x[:,:pos+1]),z[:,: (pos+1)//K])[:,-1];error=float((out-full).abs().max());check('dynamic step '+str(pos),error<4e-4,error=error)
        if fast:
            cache=FastIncremental(s,z);out=cache.prefill(x[:,:124])
            for pos in range(124,132):
                out=cache.step(x[:,pos]);full=s(x[:,:pos+1],torch.zeros_like(x[:,:pos+1]),z[:,: (pos+1)//K])[:,-1];error=float((out-full).abs().max());check('compiled static step '+str(pos),error<4e-4,error=error)
        a=s(x,docs);xx=x.clone();xx[:,140:]=torch.randint(33,2048,xx[:,140:].shape,device=device);b=s(xx,docs);check('future tokens cannot alter earlier logits',float((a[:,:140]-b[:,:140]).abs().max())<2e-5)
        d=torch.zeros(1,192,dtype=torch.long,device=device);d[:,96:]=1;xx=x.clone();yy=xx.clone();yy[:,:96]=torch.randint(33,2048,yy[:,:96].shape,device=device);a=s(xx,d);b=s(yy,d);check('document attention and concept isolation',float((a[:,96:]-b[:,96:]).abs().max())<3e-5)
    if not s.is_flat:
        f=System(copy.deepcopy(s.foundation)).to(device);f.core=DenseCore(f.foundation).to(device)
        with torch.no_grad():check('dense exact-copy endpoint',float((f(x,docs)-s(x,docs,lam=0)).abs().max())<2e-5)
        f.adapters=Adapters().to(device);freeze(f.foundation);freeze(f.core)
        zero=torch.zeros(1,48,DZ,device=device);noise=ball(torch.randn_like(zero));a=f(x,docs,zero);b=f(x,docs,noise);check('Z zero anchor within FP32 tolerance',float((a-f(x,docs)).abs().max())<2e-5,error=float((a-f(x,docs)).abs().max()))
        loss=(b-a.detach()).square().mean();loss.backward();check('bilinear U/V and frozen-decoder gradient flows',all(v.weight.grad is not None and float(v.weight.grad.norm())>0 for v in list(f.adapters.U)+list(f.adapters.V)))
        bank=torch.zeros(1,96,DZ,device=device,requires_grad=True);r=generate(f,row[0][None,:124],max_new=16);grad,meta=repair_gradient(f,r,bank);check('repair gradient finite and active',bool(torch.isfinite(grad).all()) and float(grad.norm())>0)
        old=f(x,docs,zero).detach();freeze(f);f.add_phase();new=f(x,docs,zero);check('zero-initialized LoRA anchor',torch.allclose(old,new,atol=4e-4,rtol=1e-5),error=float((old-new).abs().max()),tolerance='4e-4 + 1e-5 relative, MPS FP32; B matrices separately exact zero')
        check('new LoRA B matrices exactly zero',all(bool((b.banks[-1].Bq==0).all()) and bool((b.banks[-1].Bv==0).all()) for b in f.active_blocks()));new.square().mean().backward();check('new B receives gradient through frozen network',all(float(b.banks[-1].Bq.grad.norm())>0 for b in f.active_blocks()))
    z=torch.zeros(2,5,8,device=device);adam=MaskedAdam(z);active=torch.zeros(2,5,dtype=torch.bool,device=device);active[0,1]=True;adam.update(torch.ones_like(z),active,torch.tensor([True,False],device=device));check('inactive Adam moments and per-coordinate clocks fixed',bool((adam.steps[1]==0).all()) and bool((adam.m[:,0]==0).all()) and bool((adam.steps[0,1]==1).all()))
    # Verify data partitions and train exclusion, including the newly added tasks.
    manifest=json.loads((DATA/'manifest.json').read_text());hashsets={name:{r['sha256'] for r in json.loads((DATA/f'{name}_documents.json').read_text())} for name in manifest['text']}
    for a,ha in hashsets.items():
        for b,hb in hashsets.items():
            if a<b:check('document split disjoint '+a+'/'+b,not (ha&hb))
    mix=Mixer(81233);batch=mix.batch();check('exact per-update graph quota',mix.seen_tasks==dict(graph=24,lookup=4,compose=2,pointer=2));check('continual distributions absent foundation',not any(v in mix.seen_tasks for v in ('bind','shifted')))
    check('training prompt exclusion',not(mix.seen_symbols&mix.excluded));check('padding/doc chunk boundaries',bool(((batch['docs'][:,::4]!=batch['docs'][:,3::4]) & (batch['docs'][:,::4]>=0)&(batch['docs'][:,3::4]>=0)).sum()==0))
    return dict(status='passed',device=device,source=source,fast=fast,contracts=len(tests),tests=tests,implementation_sha256=implementation_hash())
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source');ap.add_argument('--out',required=True);ap.add_argument('--fast',action='store_true');a=ap.parse_args()
    try:result=run(source=a.source,fast=a.fast);atomic_json(a.out,result);print(json.dumps({k:v for k,v in result.items() if k!='tests'}))
    except Exception as exc:atomic_json(a.out,dict(status='failed',error=repr(exc),traceback=traceback.format_exc()));raise
