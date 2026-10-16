"""First-principles contracts before and after training; smoke quality is not claimed."""
import argparse,copy,json,tempfile,time,traceback
from pathlib import Path
import numpy as np
import torch
from model import *
from data import *
from runtime import *
from train import train_loss,forward_function,configure,update
from generation import *
from search import Node,attach,select,search_one

def close(a,b,atol=1e-5,rtol=1e-4):torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
def tiny():return Config(d=32,heads=4,ff=96,dz=8,rank=8,segments=4,codes=8,flat_ff=64)
def system(adapters=True):
    torch.manual_seed(709);s=System(tiny());s.make_refit()
    if adapters:s.make_adapters()
    return s.eval()
def fixture(n=4):
    b=Mixer(710,batch=n).batch('cpu',single_text=True);return b

def parameters():
    a=System();b=System(Config(kind='flat'));ratio=abs(count(a)-count(b))/count(a)
    assert 5e6<count(a)<7e6 and ratio<.005;assert count(Adapters(Config()))==41472
    a.make_refit();assert a.active_count()==count(System());a.make_adapters();assert a.active_count()==count(System())+41472

def data_contract():
    sets=[set(json.loads((DATA/f'edu_{split}_documents.json').read_text())['hashes']) for split in ('train','dev','test')]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    b=fixture();allowed=(b['docs'][:,:-1]>=0)&(b['docs'][:,:-1]==b['docs'][:,1:]);assert not bool((b['m'][:,:-1]>allowed).any())
    cd,valid=chunks(b['docs']);w=ncp_weights(b);assert not bool((w>(valid[:,:-1]&valid[:,1:]&(cd[:,:-1]==cd[:,1:]))).any())
    assert all(len(set(row[row>=0].tolist()))==1 for row in b['docs'])
    evalhash=set((DATA/'evaluation_symbolic_hashes.txt').read_text().splitlines());assert not Mixer(1,batch=4).seen_symbols&evalhash

def verifiers():
    for task in ('graph','pointer','lookup','compose'):
        for n in (10,14,18):
            x,m,d,meta=gen_symbolic(np.random.RandomState(n),n,task);p=meta['prompt_len'];end=p+meta['response_len']
            assert verify(x[:p],x[p:end])['success'];assert not verify(x[:p],x[p:end-1])['success']
            bad=x[p:end].copy();bad[-1]=PAD;assert not verify(x[:p],bad)['success']
            bad=x[p:end].copy();bad[0]=EOS;assert not verify(x[:p],bad)['success']
            assert m.sum()==meta['response_len']

def causality():
    s=system();x=torch.randint(33,2048,(1,64));d=torch.zeros_like(x);z=torch.randn(1,16,8)*.1
    with torch.no_grad():a=s(x,d,z)
    for t in (3,4,5,6,15,31):
        changed=x.clone();changed[:,t+1:]=torch.randint(33,2048,changed[:,t+1:].shape)
        with torch.no_grad():close(a[:,:t+1],s(changed,d,z)[:,:t+1])
    zz=z.clone();zz[:,8:]+=1
    with torch.no_grad():close(a[:,:35],s(x,d,zz)[:,:35])

def boundaries_and_padding():
    s=system();x=torch.randint(33,2048,(1,32));d=torch.tensor([[0]*16+[1]*16]);changed=x.clone();changed[:,:16]=torch.randint(33,2048,(1,16))
    with torch.no_grad():a=s(x,d);b=s(changed,d);single=s(x[:,16:],torch.zeros(1,16,dtype=torch.long))
    close(a[:,16:],b[:,16:]);close(a[:,16:],single)
    h=torch.ones(1,8,32);assert not bool(shift(h,d)[:,16:19].any())
    padded=F.pad(x,(0,16));pd=F.pad(d,(0,16),value=-1)
    with torch.no_grad():close(a,s(padded,pd)[:,:32])

def anchors():
    s=system();x=torch.randint(33,2048,(1,32));d=torch.zeros_like(x);zero=torch.zeros(1,8,8)
    with torch.no_grad():
        a=s(x,d);close(a,s(x,d,zero),0,0);close(a,s(x,d,zero,blend=0),0,0)
        for p in s.adapters.parameters():p.add_(torch.randn_like(p)*.1)
        close(a,s(x,d,zero),0,0)
        for p in s.core.parameters():p.add_(torch.randn_like(p)*.01)
        foundation=s.foundation.decode(*[]) if False else System(s.cfg,foundation=s.foundation)(x,d)
        close(foundation,s(x,d,zero,blend=0),0,0)
        ref=copy.deepcopy(s);ref.adapters=None;close(ref(x,d),s(x,d,zero,blend=1),0,0)

def gradients():
    s=system();freeze(s);x=torch.randint(33,2048,(1,32));d=torch.zeros_like(x);z=torch.zeros(1,8,8,requires_grad=True)
    y=s(x,d,z)[0,15,40];g=torch.autograd.grad(y,z)[0];assert bool(torch.isfinite(g).all()) and float(g.norm())>0
    idx=int(g.abs().argmax());e=.05;plus=z.detach().clone();minus=plus.clone();plus.view(-1)[idx]+=e;minus.view(-1)[idx]-=e
    with torch.no_grad():fd=(s(x,d,plus)[0,15,40]-s(x,d,minus)[0,15,40])/(2*e)
    assert abs(float(fd)-float(g.flatten()[idx]))<max(3e-6,.04*abs(float(g.flatten()[idx])))
    v=torch.randn(2,5,32);write=s.adapters.write(v,torch.randn(2,5,8)*100,0);assert float((scale(write)/scale(v)).max())<=.1001

def objective_ownership():
    s=System(tiny());b=fixture();f=forward_function(s,'A');loss,logs,c,ids,valid=train_loss(s,'A',b,f)
    gs=torch.autograd.grad(logs['ncp'],[s.foundation.enc.blocks[0].qkv.weight,s.foundation.con.blocks[0].qkv.weight,s.foundation.head.weight,s.foundation.Q,s.foundation.dec.blocks[0].qkv.weight],allow_unused=True,retain_graph=True)
    assert all(g is not None and float(g.norm())>0 for g in gs[:4]);assert gs[4] is None
    target=c.detach().requires_grad_(True);vq,_,_=s.foundation.vq(target,valid,b['domain']);g=torch.autograd.grad(vq,[target,s.foundation.Q],allow_unused=True)
    assert g[0] is None and g[1] is not None
    s.make_refit();loss,*_=train_loss(s,'B0',b,forward_function(s,'B0'),delta=1.);loss.backward()
    assert all(p.grad is None for p in s.foundation.parameters());assert sum(float(p.grad.norm()) for p in s.core.parameters() if p.grad is not None)>0

def codebook_moments():
    s=System(tiny());o=optimizer(s,.001);s.foundation.Q.square().sum().backward();o.step();f=s.foundation
    f.q_initialized.fill_(True);f.usage.fill_(1);f.usage[:,0]=0;old=o.state[f.Q]['exp_avg'].clone()
    f.restart(torch.randn(2,16,32),torch.ones(2,16,dtype=torch.bool),o,torch.Generator().manual_seed(1))
    assert not bool(o.state[f.Q]['exp_avg'][:,0].any());close(old[:,1:],o.state[f.Q]['exp_avg'][:,1:],0,0)

def selection_and_ownership():
    s=system();b=fixture();b['dz']=8;g=torch.Generator().manual_seed(1);z,mask,index,enabled,lengths=controls(b,g);zs=[z,-z]
    fn=forward_function(s,'B1');candidate_backward(s,b,zs,mask,.25,fn);ret=[p.grad.clone() for p in s.adapters.parameters()];s.zero_grad(set_to_none=True)
    candidate_backward(s,b,zs,mask,.25,fn,True)
    for v,p in zip(ret,s.adapters.parameters()):close(v,p.grad,2e-6,3e-4)
    fh=tensor_hash(s.foundation);ch=tensor_hash(s.core);optimizer(s,.001).step();assert fh==tensor_hash(s.foundation) and ch==tensor_hash(s.core)
    scores=torch.tensor([[1.,9.],[9.,1.]]);assert float(scores.sum(1).min())==10
    for row in range(len(z)):
        for j in range(1,z.shape[1]):
            if enabled[row,j] and enabled[row,j-1] and index[row,j]==index[row,j-1]:close(z[row,j],z[row,j-1],0,0)

def masked_optimizer():
    z=torch.zeros(3,5,8);o=MaskedAdam(z);g=torch.randn_like(z);active=torch.tensor([[1,0,0,0,0],[1,1,0,0,0],[0,0,0,0,0]],dtype=torch.bool)
    o.update(g,active,torch.tensor([1,1,0],dtype=torch.bool));old=z.clone();mom=o.m.clone();step=o.steps.clone()
    o.update(g,active,torch.tensor([0,1,0],dtype=torch.bool));close(old[0],z[0],0,0);close(mom[0],o.m[0],0,0);assert o.steps[0]==step[0]
    assert not bool(z[:,2:].any());bad=g.clone();bad[0,0,0]=float('nan');old=z.clone();r=o.update(bad,active,torch.ones(3,dtype=torch.bool));assert not r['finite'][0];close(old[0],z[0],0,0)

def accumulation():
    s=System(tiny());b=fixture();den=torch.stack([b['m'][b['domain']==i,:-1].sum() for i in (0,1)])
    balanced(nll(s(b['x'],b['docs']),b['x'],b['m']),b['m'][:,:-1],b['domain'],den)[0].backward()
    old=[p.grad.clone() if p.grad is not None else None for p in s.parameters()];s.zero_grad(set_to_none=True)
    for ids in ([0,2],[1,3]):
        bb={k:v[ids] for k,v in b.items() if isinstance(v,torch.Tensor)};balanced(nll(s(bb['x'],bb['docs']),bb['x'],bb['m']),bb['m'][:,:-1],bb['domain'],den)[0].backward()
    for a,p in zip(old,s.parameters()):
        if a is not None:close(a,p.grad,3e-6,5e-4)

def recovery():
    s=System(tiny());o=optimizer(s,.001);mix=Mixer(1,batch=4);g=torch.Generator().manual_seed(2);fn=forward_function(s,'A');b=mix.batch('cpu');update(s,o,'A',b,fn,g,.3)
    with tempfile.TemporaryDirectory(dir=ROOT/'tests') as tmp:
        path=Path(tmp)/'checkpoint.pt';save_checkpoint(path,s,o,1,mix,g,dict(test=True),dict(targets=100))
        bb=mix.batch('cpu');noise=torch.randn(3,generator=g);update(s,o,'A',bb,fn,g,.4);expected=copy.deepcopy(s.state_dict())
        r,ck=load_system(path,'cpu');ro=optimizer(r,.001);ro.load_state_dict(ck['optimizer']);mix.restore(ck['mixer']);g.set_state(ck['generator']);torch.set_rng_state(ck['torch_rng'])
        bb2=mix.batch('cpu');close(bb['x'],bb2['x'],0,0);close(noise,torch.randn(3,generator=g),0,0);update(r,ro,'A',bb2,forward_function(r,'A'),g,.4)
        for k,v in expected.items():close(v,r.state_dict()[k],0,0)

def caches(device='cpu',fast=False):
    s=system().to(device);freeze(s);cfg=s.cfg
    for p in (1,2,3,4,12,13,14,15):
        x=torch.randint(33,2048,(1,32),device=device);z=torch.randn(1,8,8,device=device)*.1;d=torch.zeros_like(x)
        cache=(FastIncremental if fast else Incremental)(s,z)
        with torch.no_grad():
            close(cache.prefill(x[:,:p]),s(x,d,z)[:,p-1],5e-5,5e-4)
            for t in range(p,p+5):close(cache.step(x[:,t]),s(x,d,z)[:,t],5e-5,5e-4)

def tree_backups():
    root=Node();a=Node();b=Node();c=Node();d=Node();attach(root,a,.8);attach(root,b,0);attach(a,c,1);attach(c,d,.5)
    assert root.gen.rewards==[.8,0] and root.cont.rewards==[1,.5];assert a.gen.rewards==[1] and a.cont.rewards==[.5]
    assert a.answer.rewards==[.8,1,.5] and c.gen.rewards==[.5] and c.cont.rewards==[]
    for _ in range(40):assert select(root,np.random.RandomState(_)) is not None

def generation_and_repair():
    s=system();freeze(s);x,m,d,meta=gen_symbolic(np.random.RandomState(1),10);p=meta['prompt_len'];bank=torch.randn(1,T//K,8)*.1
    a=generate(s,x[None,:p],bank,max_new=12);b=generate(s,x[None,:p],bank,max_new=12,cached=False);assert np.array_equal(a['x'],b['x'])
    c=generate(s,x[None,:p],-bank,max_new=12,encoding=a['encoding']);d=generate(s,x[None,:p],-bank,max_new=12);assert np.array_equal(c['x'],d['x'])
    z=torch.zeros_like(bank,requires_grad=True);grad,stats=repair_gradient(s,a,z,kl=0);assert float(grad.norm())>0
    xx,docs,mask=replay_batch(a,'cpu')
    with torch.no_grad():before=float(-nll(s(xx,docs,expand_bank(z,p,1,T//K)),xx,mask).sum()/mask.sum());after=float(-nll(s(xx,docs,expand_bank(z-.001*grad,p,1,T//K)),xx,mask).sum()/mask.sum())
    assert after<=before+1e-6
    result=search_one(s,x[:p],'gradcuit',budget_s=30,candidate_limit=2);assert len(result['records'])==2;assert result['records'][-1]['operator']=='gradient'
    assert all(len(r['response'])%4==0 for r in result['records'])

def optimized():
    s=system().to(DEVICE);b=fixture();b={k:v.to(DEVICE) if isinstance(v,torch.Tensor) else v for k,v in b.items()};b['dz']=8
    g=torch.Generator().manual_seed(3);z,mask,*_=controls(b,g);fn=forward_function(s,'B1');compiled=forward_function(s,'B1',True)
    candidate_backward(s,b,[z,-z],mask,.25,fn);a=torch.cat([p.grad.flatten() for p in s.adapters.parameters()]);s.zero_grad(set_to_none=True)
    candidate_backward(s,b,[z,-z],mask,.25,compiled);bb=torch.cat([p.grad.flatten() for p in s.adapters.parameters()]);assert float((a-bb).norm()/a.norm().clamp_min(1e-12))<.003
    caches(DEVICE,True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--optimized',action='store_true');args=ap.parse_args();torch.set_num_threads(4)
    tests=[parameters,data_contract,verifiers,causality,boundaries_and_padding,anchors,gradients,objective_ownership,codebook_moments,
        selection_and_ownership,masked_optimizer,accumulation,recovery,caches,tree_backups,generation_and_repair]
    if args.optimized:tests+=[optimized]
    results=[]
    for test in tests:
        start=time.monotonic()
        try:test();r=dict(name=test.__name__,status='passed',seconds=time.monotonic()-start)
        except Exception as e:r=dict(name=test.__name__,status='failed',error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start)
        results.append(r);print(json.dumps(r),flush=True);atomic_json(ROOT/'tests'/('optimized_checks.json' if args.optimized else 'checks.json'),dict(passed=all(x['status']=='passed' for x in results),results=results,
            not_exercised=['distributed workers','BF16','billion-parameter scaling','untrained smoke quality']))
    raise SystemExit(0 if all(r['status']=='passed' for r in results) else 1)

if __name__=='__main__':main()
