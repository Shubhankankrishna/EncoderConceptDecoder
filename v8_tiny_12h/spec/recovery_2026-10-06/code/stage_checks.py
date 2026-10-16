"""Production-shaped optimizer updates and eager/compiled gradients for every train path."""
import copy,json,time
import numpy as np
import torch
from model import *
from data import Mixer,gen_symbolic,parse_prompt,verify
from runtime import *
from train import configure,forward_function,objective,candidate_backward
from generation import Incremental,MaskedAdam,generate
from search import Node,attach,select,public_grade,search_one

def main():
    torch.set_num_threads(4);seed_all(817);b=Mixer(81701).batch(DEVICE);records=[]
    for stage,core in [('A','dense'),('A_FLAT','dense'),('REFIT','dense'),('REFIT','recurrent'),('Z','dense'),('Z','recurrent'),('CL','dense'),('CL_GATED','dense'),('CL_FLAT','dense')]:
        name=stage+'/'+core;s=System(Flat() if stage in ('A_FLAT','CL_FLAT') else Foundation()).to(DEVICE)
        actual='A' if stage=='A_FLAT' else 'CL' if stage.startswith('CL') else stage
        if actual=='REFIT':freeze(s);s.core=(Core() if core=='recurrent' else DenseCore(s.foundation)).to(DEVICE);[p.requires_grad_(True) for p in s.core.parameters()]
        if actual=='Z':s.core=(Core() if core=='recurrent' else DenseCore(s.foundation)).to(DEVICE);freeze(s);s.adapters=Adapters().to(DEVICE)
        if actual=='CL':
            if stage in ('CL_GATED','CL_FLAT'):
                s.add_phase(force=None)
                for block in s.active_blocks():
                    block.banks[-1].Bq.data.normal_(std=.01);block.banks[-1].Bv.data.normal_(std=.01);block.banks[-1].pos_var.fill_(10.);block.neg_var.fill_(100.)
            freeze(s);s.add_phase();[p.requires_grad_(True) for block in s.active_blocks() for p in block.banks[-1].parameters()]
        eager=copy.deepcopy(s);compiled=copy.deepcopy(s);times={};losses={};gradients={}
        for mode,model in [('eager',eager),('compiled',compiled)]:
            start=time.monotonic();fn=forward_function(model,actual,mode=='compiled');g=torch.Generator().manual_seed(81702)
            if actual=='Z':loss=candidate_backward(model,b,fn,g,.25)
            else:loss,*_=objective(model,actual,b,fn,.1 if actual!='CL' else 0.,1. if actual=='REFIT' else 0.);loss.backward()
            sync();times[mode]=time.monotonic()-start;losses[mode]=float(loss.detach());gradients[mode]={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
            opt=optimizer(model,stage=actual);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
        errors=[]
        for n,grad in gradients['eager'].items():
            other=gradients['compiled'][n];errors.append(dict(parameter=n,max_abs=float((grad-other).abs().max()),relative=float((grad-other).norm()/grad.norm().clamp_min(1e-8))))
        passed=abs(losses['eager']-losses['compiled'])<1e-4 and all(e['max_abs']<2e-4 and e['relative']<.01 for e in errors)
        if not all(bool(torch.isfinite(v).all()) for v in gradients['eager'].values()):raise AssertionError(name+' eager nonfinite gradient')
        records.append(dict(stage=name,passed=True,compiled_safe=passed,execution='compiled' if passed else 'eager',loss=losses,seconds=times,gradient_errors=errors))
    # Trained-fixture nonzero banks: branch caches and smoothing clocks must agree with full causal execution.
    for rho in (1.,.3):
        s=System().to(DEVICE);s.add_phase(rho,False);s.add_phase(rho,None)
        for block in s.active_blocks():
            bank=block.banks[1];bank.Bq.data.normal_(std=.01);bank.Bv.data.normal_(std=.01);bank.pos_var.fill_(2.);block.neg_var.fill_(5.)
        x=b['x'][:2,:];z=torch.zeros(2,96,DZ,device=DEVICE);cache=Incremental(s,z)
        with torch.no_grad():
            cache.prefill(x[:,:124]);errs=[]
            for p in range(124,132):
                inc=cache.step(x[:,p]);full=s(x[:,:p+1],torch.zeros_like(x[:,:p+1]))[:,-1];errs.append(float((inc-full).abs().max()))
        if max(errs)>4e-4:raise AssertionError('gated cache mismatch rho='+str(rho)+str(errs))
        records.append(dict(stage='gated_cache',rho=rho,max_abs=max(errs),passed=True))
    # Recurrent execution has SIX independent visit caches.
    s=System();s.core=Core();s=s.to(DEVICE);x=b['x'][:2];cache=Incremental(s,torch.zeros(2,96,DZ,device=DEVICE));errs=[]
    with torch.no_grad():
        cache.prefill(x[:,:124])
        if len(cache.con)!=6:raise AssertionError('visit cache count')
        for p in range(124,132):errs.append(float((cache.step(x[:,p])-s(x[:,:p+1],torch.zeros_like(x[:,:p+1]))[:,-1]).abs().max()))
    if max(errs)>4e-4:raise AssertionError('recurrent cache mismatch')
    records.append(dict(stage='recurrent_cache',max_abs=max(errs),passed=True))
    root=Node();a=Node(depth=1);attach(root,a,.2);bnode=Node(depth=2);attach(a,bnode,.7)
    assert root.gen.rewards==[.2] and root.cont.rewards==[.7] and a.gen.rewards==[.7] and a.cont.rewards==[]
    assert select(root,np.random.RandomState(1),16) is not None
    row=gen_symbolic(np.random.RandomState(3),hard=True);p=row[3]['prompt_len'];resp=row[0][p:p+row[3]['response_len']];assert verify(row[0][:p],resp)['success'];assert 0<=public_grade(row[0][:p],resp)<=.9
    atomic_json(ROOT/'tests/stage_contracts.json',dict(status='passed',checks=records,implementation_sha256=implementation_hash()));print('All stage gradient/cache/search contracts passed')
if __name__=='__main__':main()
