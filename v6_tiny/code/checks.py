"""Executable V6 draft contracts. Skipped execution modes are labeled explicitly."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import time
import traceback
import numpy as np
import torch
from model import *
from data import Mixer,gen_instance,verify,oracle,WALK_START,T,PAD,DONE,SEP,fixed_text,DATA
from runtime import (candidate_backward,candidate_rows,selection_loss,controls,optimizer,save_checkpoint,load_system,
                     nll,balanced_sums,ncp_weights,atomic_json,DEVICE,sync)
from generation import Incremental,FastIncremental,generate,MaskedAdam,active_windows,expand_bank
from evaluate import repair_gradient,search_batch
from train import train_loss,tensors_forward,do_update


def close(a,b,atol=1e-5,rtol=1e-4):
    torch.testing.assert_close(a,b,atol=atol,rtol=rtol)


def make_system():
    torch.manual_seed(606);return System(Foundation(),Core(),Adapters()).eval()


def fixture(device='cpu',batch=4):return Mixer(606,batch=batch).batch(device)


def check_counts():
    s=make_system();assert count(s.foundation)==2022144;assert count(s.core)==476288
    assert count(s.adapters)==8480;assert count(Adapters('direct'))==2048;assert active_count(s)==2063904


def check_data():
    batch=fixture();x,m,d=batch['x'],batch['m'],batch['docs']
    allowed=(d[:,:-1]==d[:,1:])&(d[:,:-1]>=0)
    assert not bool((m[:,:-1]>allowed.float()).any())
    cd,valid=chunk_info(d);cm=ncp_weights(batch)
    assert not bool((cm> ((cd[:,:-1]==cd[:,1:])&valid[:,:-1]&valid[:,1:]).float()).any())
    dev=set(np.load(DATA/'dev_docs.npy').tolist());test=set(np.load(DATA/'test_docs.npy').tolist());assert not dev&test
    assert bool((m[:2,:3]>0).any()),'Initial text targets must be counted'


def check_verifier():
    x,m,meta=gen_instance(np.random.RandomState(1));response=x[WALK_START:]
    assert verify(x[:WALK_START],response)['success']
    bad=response.copy();j=4*meta['n_hops'];bad[j+1]=PAD
    assert verify(x[:WALK_START],bad)['reason']=='malformed_terminator'
    assert not verify(x[:WALK_START],response[:j+1])['success']
    bad=response.copy();bad[2]=NODE0 if 'NODE0' in globals() else 6
    if bad[2]==response[2]:bad[2]+=1
    assert not verify(x[:WALK_START],bad)['success']
    assert verify(x[:WALK_START],oracle(x[:WALK_START]))['success']


def check_causality():
    s=make_system();batch=fixture(batch=2);x,d=batch['x'],batch['docs'];x2=x.clone();x2[:,104:]=torch.randint(33,VOCAB,x2[:,104:].shape)
    z=torch.randn(2,T//4,DZ)*.1;z2=z.clone();z2[:,26:]+=1
    with torch.no_grad():
        a=s(x,d,z);b=s(x2,d,z);c=s(x,d,z2)
    close(a[:,:104],b[:,:104]);close(a[:,:107],c[:,:107])


def check_isolation():
    s=make_system();x=torch.randint(33,VOCAB,(1,32));d=torch.tensor([[0]*16+[1]*16]);changed=x.clone();changed[:,:16]=torch.randint(33,VOCAB,(1,16))
    with torch.no_grad():a=s(x,d);b=s(changed,d)
    close(a[:,16:],b[:,16:])
    v=torch.ones(1,8,D);shifted=shift(v,d);assert not bool(shifted[:,16:19].any())


def check_anchor():
    s=make_system();batch=fixture(batch=2);x,d=batch['x'],batch['docs'];z=torch.zeros(2,T//4,DZ)
    with torch.no_grad():
        reference=s(x,d);zero=s(x,d,z);close(reference,zero,atol=0,rtol=0)
        for p in s.adapters.parameters():p.add_(torch.randn_like(p)*.1)
        close(reference,s(x,d,z),atol=0,rtol=0)
        foundation=s.foundation(x,d)[0];close(foundation,s(x,d,z,0),atol=0,rtol=0)


def check_control_gradients():
    s=make_system();freeze(s.foundation);freeze(s.core);batch=fixture(batch=2)
    z=torch.zeros(2,T//4,DZ,requires_grad=True);loss=s(batch['x'],batch['docs'],z).square().mean()
    loss.backward();assert bool(torch.isfinite(z.grad).all());assert float(z.grad.norm())>0
    assert all(p.grad is None for p in s.foundation.parameters());assert all(p.grad is None for p in s.core.parameters())
    e=torch.randn(2,4,D);a=s.adapters(e,torch.randn(2,4,DZ)*50)
    for write in a:assert float((safe_rms(write)/safe_rms(e)).max())<=.10001
    zero=torch.zeros(2,4,DZ,requires_grad=True);sum(w.square().sum()+w.sum() for w in s.adapters(torch.zeros_like(e),zero)).backward()
    assert bool(torch.isfinite(zero.grad).all())


def check_finite_difference():
    s=make_system();freeze(s);batch=fixture(batch=2);z=torch.zeros(2,T//4,DZ,requires_grad=True)
    # A scalar response logit avoids subtracting nearly equal ~7.6-nat FP32 batch means.
    loss=s(batch['x'],batch['docs'],z)[1,WALK_START+3,40]
    grad=torch.autograd.grad(loss,z)[0];idx=int(grad.abs().argmax());epsilon=.1
    plus=z.detach().clone();minus=plus.clone();plus.view(-1)[idx]+=epsilon;minus.view(-1)[idx]-=epsilon
    with torch.no_grad():a=s(batch['x'],batch['docs'],plus)[1,WALK_START+3,40];b=s(batch['x'],batch['docs'],minus)[1,WALK_START+3,40]
    actual=float(grad.flatten()[idx]);finite=float((a-b)/(2*epsilon))
    assert abs(actual-finite)<max(2e-6,.05*abs(actual)),(actual,finite)


def check_selection():
    scores=torch.tensor([[0.,8.],[8.,0.]])
    assert scores.sum(1).min()==8 and scores.min(0).values.sum()==0
    s=make_system();freeze(s.foundation);freeze(s.core);b=fixture()
    g=torch.Generator().manual_seed(2);z,mask,_,_,_=controls(b,g);schedules=[z,-z]
    candidate_backward(s,b,schedules,mask,.25,replay=False)
    retained=[p.grad.clone() for p in s.adapters.parameters()];s.zero_grad(set_to_none=True)
    candidate_backward(s,b,schedules,mask,.25,replay=True)
    for old,p in zip(retained,s.adapters.parameters()):close(old,p.grad,atol=3e-7,rtol=2e-4)
    before={k:v.clone() for k,v in s.core.state_dict().items()}
    optimizer(s.adapters.parameters(),.001,0).step()
    assert all(torch.equal(v,s.core.state_dict()[k]) for k,v in before.items())
    z,mask,index,enabled,lengths=controls(b,g)
    for row in range(len(z)):
        for j in range(1,z.shape[1]):
            if enabled[row,j] and enabled[row,j-1] and index[row,j]==index[row,j-1]:close(z[row,j],z[row,j-1],atol=0,rtol=0)


def check_moments():
    f=Foundation();opt=optimizer(f.parameters(),.001);f.Q.square().sum().backward();opt.step()
    f.q_init.fill_(True);f.usage.fill_(1);f.usage[:,0]=0;old=opt.state[f.Q]['exp_avg'].clone()
    f.restart(torch.randn(2,48,D),torch.ones(2,48,dtype=torch.bool),opt,torch.Generator().manual_seed(1))
    assert not bool(opt.state[f.Q]['exp_avg'][:,0].any());close(old[:,1:],opt.state[f.Q]['exp_avg'][:,1:],atol=0,rtol=0)


def check_masked_adam():
    z=torch.zeros(3,4,DZ);opt=MaskedAdam(z);active=torch.tensor([[1,0,0,0],[1,1,0,0],[0,0,0,0]],dtype=torch.bool)
    use=torch.tensor([1,1,0],dtype=torch.bool);g=torch.randn_like(z)
    opt.update(g,active,use);old=z.clone();mom=opt.m.clone();steps=opt.steps.clone()
    opt.update(g,active,torch.tensor([0,1,0],dtype=torch.bool))
    close(old[0],z[0],atol=0,rtol=0);close(mom[0],opt.m[0],atol=0,rtol=0);assert opt.steps[0]==steps[0]
    assert not bool(z[:,2:].any());assert not bool(opt.m[:,2:].any());assert float(safe_rms(z).max())<=2.00001
    permutation=torch.tensor([1,0,2]);z1=torch.zeros_like(z);z2=torch.zeros_like(z);a=MaskedAdam(z1);b=MaskedAdam(z2)
    a.update(g,active,use);b.update(g[permutation],active[permutation],use[permutation]);close(z1[permutation],z2,atol=0,rtol=0)
    bad=g.clone();bad[0,0,0]=float('nan');before=z1.clone();r=a.update(bad,active,use);assert not r['finite'][0];close(before[0],z1[0],atol=0,rtol=0)


def check_accumulation(device='cpu'):
    # Trainable foundation fixture exercises exact denominators for both domains.
    s=System(Foundation()).to(device);batch=fixture(device);den=torch.stack([batch['m'][batch['domain']==i,:-1].sum() for i in (0,1)])
    balanced_sums(nll(s(batch['x'],batch['docs']),batch['x'],batch['m']),batch['m'][:,:-1],batch['domain'],den)[0].backward()
    full=[p.grad.clone() if p.grad is not None else None for p in s.parameters()];s.zero_grad(set_to_none=True)
    for ids in ([0,2],[1,3]):
        micro={k:v[ids] for k,v in batch.items() if isinstance(v,torch.Tensor)}
        balanced_sums(nll(s(micro['x'],micro['docs']),micro['x'],micro['m']),micro['m'][:,:-1],micro['domain'],den)[0].backward()
    for expected,p in zip(full,s.parameters()):
        if expected is not None:close(expected,p.grad,atol=3e-6,rtol=3e-4)


def check_resume(device='cpu'):
    torch.manual_seed(3);s=System(Foundation()).to(device);opt=optimizer(s.parameters(),.001);mix=Mixer(3,batch=4);g=torch.Generator().manual_seed(4)
    b=mix.batch(device);f=tensors_forward(s,'A');do_update(s,opt,'A',b,f,g,0,100)
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tests') as tmp:
        save_checkpoint(Path(tmp)/'fixture.pt',s,opt,1,mix,g,dict(implementation_sha256='fixture'))
        b1=mix.batch(device);r1=torch.randn(3,generator=g);do_update(s,opt,'A',b1,f,g,1,100);expected=copy.deepcopy(s.state_dict())
        restored,ck=load_system(Path(tmp)/'fixture.pt',device);o2=optimizer(restored.parameters(),.001);o2.load_state_dict(ck['optimizer'])
        mix.restore(ck['mixer']);g.set_state(ck['generator']);b2=mix.batch(device);r2=torch.randn(3,generator=g)
        assert torch.equal(b1['x'],b2['x']);assert torch.equal(r1,r2)
        do_update(restored,o2,'A',b2,tensors_forward(restored,'A'),g,1,100)
        for key,value in expected.items():close(value,restored.state_dict()[key],atol=0 if device=='cpu' else 2e-6,rtol=0 if device=='cpu' else 2e-5)


def check_cache(device='cpu',fast=False):
    s=make_system().to(device);freeze(s);batch=fixture(device,batch=2);x=batch['x'];x=x[1:2];docs=torch.zeros_like(x)
    z=torch.randn(1,T//4,DZ,device=device)*.2;cache=(FastIncremental if fast else Incremental)(s,z)
    with torch.no_grad():
        initial=cache.prefill(x[:,:WALK_START]);close(initial,s(x,docs,z)[:,WALK_START-1],atol=4e-5,rtol=4e-4)
        assert len(cache.core)==6;assert len({id(c[0]) for c in cache.core})==6
        for pos in range(WALK_START,WALK_START+12):
            actual=cache.step(x[:,pos]);close(actual,s(x,docs,z)[:,pos],atol=5e-5,rtol=5e-4)


def check_generation():
    s=make_system();freeze(s);x,_,_=gen_instance(np.random.RandomState(4));bank=torch.randn(1,T//4,DZ)*.2
    cached=generate(s,x[None],bank,seed=1);replay=generate(s,x[None],bank,seed=1,cached=False)
    assert np.array_equal(cached['x'],replay['x']);assert np.array_equal(cached['lengths'],replay['lengths'])
    changed=generate(s,x[None],-bank,seed=1,prompt_encoding=cached['prompt_encoding']);fresh=generate(s,x[None],-bank,seed=1)
    assert np.array_equal(changed['x'],fresh['x']);assert changed['encoder_prefill_reused']
    # Every updated answer, including the final slot, is generated and verified.
    result=search_batch(s,x[None],'gradcuit',updates=1)
    assert result['generations']==2;assert [r['attempt'] for r in result['records'][0]]==[0,1]
    zero=search_batch(s,x[None],'greedy',updates=0);assert zero['generations']==1
    bank=torch.zeros(1,T//4,DZ,requires_grad=True);grad,_=repair_gradient(s,cached,bank,1,0)
    before_x,docs,mask=__import__('generation').replay_batch(cached,'cpu')
    before=float(-nll(s(before_x,docs,expand_bank(bank,1,T//4)),before_x,mask).sum()/mask.sum())
    with torch.no_grad():changed=bank-.001*grad
    after=float(-nll(s(before_x,docs,expand_bank(changed,1,T//4)),before_x,mask).sum()/mask.sum())
    assert after<=before+1e-7,(before,after)


def check_compile():
    s=make_system().to(DEVICE);freeze(s.foundation);freeze(s.core);batch=fixture(DEVICE,batch=8);z=torch.randn(8,T//4,DZ,device=DEVICE)
    eager=tensors_forward(s,'B1');compiled=tensors_forward(s,'B1',True)
    with torch.no_grad():h,c,cd=s.context(batch['x'],batch['docs'])
    def evaluate(fn):
        s.zero_grad(set_to_none=True);logits=fn(h,c,cd,z,batch['docs'])[0];loss=nll(logits,batch['x'],batch['m']).sum()/batch['m'].sum();loss.backward()
        return loss.detach(),torch.cat([p.grad.flatten() for p in s.adapters.parameters() if p.grad is not None])
    la,ga=evaluate(eager);lb,gb=evaluate(compiled);close(la,lb,atol=1e-4,rtol=1e-4)
    relative=float((ga-gb).norm()/ga.norm().clamp_min(1e-12));assert relative<.002,relative
    a=System(Foundation()).to(DEVICE);f1=tensors_forward(a,'A');f2=tensors_forward(a,'A',True)
    l1,*_=train_loss(a,'A',batch,f1);a.zero_grad();l1.backward();g1=torch.cat([p.grad.flatten() for p in a.parameters() if p.grad is not None])
    l2,*_=train_loss(a,'A',batch,f2);a.zero_grad();l2.backward();g2=torch.cat([p.grad.flatten() for p in a.parameters() if p.grad is not None])
    close(l1,l2,atol=1e-4,rtol=1e-4);assert float((g1-g2).norm()/g1.norm())<.002


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--optimized',action='store_true');a=ap.parse_args();torch.set_num_threads(4)
    checks=[('parameter_ledger',check_counts),('document_masks_and_splits',check_data),('strict_verifier_and_bfs',check_verifier),('token_and_control_causality',check_causality),
            ('document_attention_and_plan_isolation',check_isolation),('zero_and_blend_anchors',check_anchor),('gradient_reach_and_safe_caps',check_control_gradients),
            ('zero_control_finite_difference',check_finite_difference),('coherent_selection_and_replay_gradients',check_selection),('codebook_moment_reset',check_moments),
            ('per_problem_optimizer_mask_and_permutation',check_masked_adam),('microbatch_global_denominators',check_accumulation),('atomic_exact_resume',check_resume),
            ('distinct_recursive_caches_cpu',check_cache),('generation_replay_and_final_update',check_generation)]
    if a.optimized:checks.extend([('mps_eager_compiled_losses_and_gradients',check_compile),('compiled_static_cache_equivalence',lambda:check_cache(DEVICE,True)),
                                 ('mps_microbatch_global_denominators',lambda:check_accumulation(DEVICE)),('mps_checkpoint_resume',lambda:check_resume(DEVICE))])
    results=[]
    for name,fn in checks:
        start=time.monotonic()
        try:fn();result=dict(name=name,status='passed',seconds=time.monotonic()-start)
        except Exception as exc:result=dict(name=name,status='failed',seconds=time.monotonic()-start,error=repr(exc),traceback=traceback.format_exc())
        results.append(result);print(json.dumps(result),flush=True)
        atomic_json(Path(__file__).resolve().parents[1]/'tests'/('optimized_checks.json' if a.optimized else 'checks.json'),dict(results=results,passed=all(r['status']=='passed' for r in results),
                     not_exercised=['Distributed workers: single-device experiment','BF16/FP16: FP32 deliberately selected','Long-context/width scaling: tiny experiment only']))
    raise SystemExit(0 if all(r['status']=='passed' for r in results) else 1)


if __name__=='__main__':main()
