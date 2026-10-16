"""Behavioral end-to-end contracts; small vocabulary/table, full width/depth."""
import argparse
import copy
import hashlib
import io
import json
import time
from dataclasses import replace
from pathlib import Path
import torch
from v10_10M.config import ModelConfig
from v10_10M.model import V10, losses as original_losses
from v10_10M.train import build_optimizers,clip
from .model import V101,Architecture,losses,token_loss,chunk_batch
from .memory import RBFMemory,learn_episode,write_objective,model_era
from .provenance import provenance
from .accumulate import backward_effective_batch


def run(device='cpu'):
    torch.set_num_threads(2);torch.manual_seed(127)
    c=replace(ModelConfig(),vocab_size=96,engram_rows=(101,103,107,109))
    tol=5e-5 if device=='mps' else 5e-6
    checks={}
    # Exact baseline reproduction, with the same initialization and no new routes.
    torch.manual_seed(311);old=V10(c).place(device).eval()
    torch.manual_seed(311);a=V101(c,Architecture.arm('A')).place(device).eval()
    x=torch.randint(4,96,(2,33),device=device);lengths=torch.tensor([33,25],device=device)
    y=torch.randint(4,96,x.shape,device=device);mask=torch.arange(33,device=device)[None]<lengths[:,None]
    with torch.no_grad():
        expected=old(x,lengths);actual=a(x,lengths)
        error=float((expected['logits']-actual['logits']).abs().max());assert error<tol,error
        ol=original_losses(old,expected,y,mask);nl=losses(a,actual,x,lengths,y,mask)
        assert torch.allclose(ol['total'],nl['total'],atol=tol,rtol=1e-5)
    checks['A_reproduces_V10_logits_and_objective']=error
    torch.manual_seed(311);m=V101(c).place(device).eval()
    with torch.no_grad():
        base=m(x,lengths)
        assert torch.allclose(base['logits'][:,:7],actual['logits'][:,:7],atol=tol,rtol=1e-5)
    for name,param in old.state_dict().items():
        candidate=m.state_dict()[name]
        if name.startswith('decoder.routes.'):candidate=candidate[:2]
        assert torch.equal(param,candidate),name
    checks['common_initialization_and_startup_routing']=True
    del old,a
    with torch.no_grad():
        errors={}
        committed=None
        for t in [1,7,8,9,15,16,17,23,24,25,33]:
            short_lengths=lengths.clamp_max(t)
            prefix=m(x[:,:t],short_lengths,codes=committed)
            for b,n in enumerate(short_lengths):
                error=float((prefix['logits'][b,:n]-base['logits'][b,:n]).abs().max())
                assert error<tol,(t,error);errors[t]=max(errors.get(t,0),error)
            if committed is not None:assert torch.equal(prefix['codes'][:,:committed.shape[1]],committed)
            committed=prefix['codes']
            if t<33:
                changed=x.clone();changed[:,t:]=torch.randint(4,96,changed[:,t:].shape,device=device)
                other=m(changed,lengths)
                error=float((other['logits'][:,:t]-base['logits'][:,:t]).abs().max())
                assert error<tol,(t,error)
        checks['future_perturbation_and_prefix_generation_errors']=errors
        changed=x.clone();changed[0]=90;changed[1,25:]=91
        other=m(changed,lengths);single=m(x[1:2,:25])
        assert torch.allclose(other['logits'][1,:25],base['logits'][1,:25],atol=tol,rtol=1e-5)
        assert torch.allclose(single['logits'][0],base['logits'][1,:25],atol=tol,rtol=1e-5)
        checks['padding_batch_and_unpadded_isolation']=True
        at=torch.tensor([3,1],device=device)
        candidates=m.candidate_forecasts(base,lengths,at)
        for code in range(8):
            history=base['codes'].clone();history[torch.arange(2,device=device),at]=code
            states=m.concept(base['pooled'],lengths,history,base['sources'])
            predicted=m.head(states[-1][torch.arange(2,device=device),at])[0]
            assert torch.allclose(candidates[:,code],predicted,atol=tol,rtol=1e-5),code
        later=base['codes'].clone();later[:,-1]=(later[:,-1]+1)%8
        other=m(x,lengths,codes=later)
        assert torch.allclose(other['pre_memory'][:,:-1],base['pre_memory'][:,:-1],atol=tol,rtol=1e-5)
        checks['candidate_separation_history_and_group_causality']=True
        both=m(x,lengths,codes=base['codes'],ablate=True)
        explicit=m(x,lengths,codes=base['codes'],ablate_pq=True,ablate_continuous=True)
        pq_only=m(x,lengths,codes=base['codes'],ablate_pq=True)
        assert torch.equal(both['logits'],explicit['logits'])
        assert not torch.allclose(both['logits'][:,7:],pq_only['logits'][:,7:],atol=tol,rtol=1e-5)
        checks['ablation_labels_disable_the_correct_routes']=True
    counts=dict(e1=0,e2=0,e3=0,core=0,decoder=0,pool=0,ec=0);handles=[]
    for name,modules in [('e1',m.e1),('e2',m.e2),('e3',m.e3),('core',m.core.layers),
                         ('decoder',m.decoder.layers),('pool',[m.pooler]),('ec',[m.encoder_concept_crc])]:
        for module in modules:
            def hook(mod,args,out,name=name):counts[name]+=1
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():m(x,lengths)
    for h in handles:h.remove()
    assert counts==dict(e1=4,e2=8,e3=4,core=8,decoder=4,pool=2,ec=2),counts
    assert not torch.equal(base['states'][0],base['states'][1])
    checks['actual_layer_applications_and_retained_state']=counts
    # Terminal answer: current group complete, future group absent, EOS supervised.
    xx=x[:1,:16];yy=y[:1,:16].clone();yy[0,-1]=2
    mm=torch.zeros_like(xx,dtype=torch.bool);mm[:,-1]=True;ll=lengths.new_tensor([16])
    out=m(xx,ll);chunk=chunk_batch(xx,ll,yy,mm,out['valid_groups'])
    assert chunk['boundaries'].tolist()==[1] and chunk['supervised_targets']==1 and chunk['labels'][0,0]==2
    assert chunk['mask'][0].tolist()==[True]+[False]*7
    terminal=losses(m,out,xx,ll,yy,mm,candidate_update=True)
    assert terminal['candidate_executed'] and terminal['chunk_valid_documents']==1
    checks['partial_chunk_EOS_and_utility_without_future_latent']=True
    # Auxiliary hidden inputs have a one-token delay, including current-target perturbations.
    with torch.no_grad():
        f=torch.randn(1,2,128,device=device);tokens=torch.randint(4,96,(1,8),device=device)
        original=m.chunk_readout(f,tokens[:,:7],m.embedding)
        for j in range(8):
            changed=tokens.clone();changed[:,j]=(changed[:,j]+1)%96
            altered=m.chunk_readout(f,changed[:,:7],m.embedding)
            assert torch.allclose(original[:,:j+1],altered[:,:j+1],atol=tol,rtol=1e-5)
    checks['auxiliary_current_future_token_isolation']=True
    # Actual combined loss differentiates through all requested new paths.
    m.train();m.zero_grad(set_to_none=True)
    out=m(x,lengths);parts=losses(m,out,x,lengths,y,mask,candidate_update=True,generator=torch.Generator().manual_seed(19))
    parts['total'].backward();gradients={}
    for name,p in [('encoder',m.e1[0].attn.qkv.weight),('pooler',m.pooler.q.weight),
        ('ec_projection',m.encoder_concept_crc.projections[0].weight),('ec_router',m.encoder_concept_crc.route.weight),
        ('core',m.core.layers[0].attn.qkv.weight),('code',m.core.codes.weight),('selector',m.core.selector.weight),
        ('pq',m.head.codebook),('continuous_projection',m.decoder.concept_projection.weight),
        ('decoder_router',m.decoder.routes[0].weight),('chunk',m.chunk_readout.projection.weight)]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,name
        gradients[name]=float(p.grad.norm())
    assert m.engram.table.weight.grad.is_sparse and torch.isfinite(m.engram.table.weight.grad.coalesce().values()).all()
    checks['end_to_end_gradient_norms']=gradients
    for t in (1,7,8):
        m.zero_grad(set_to_none=True);z=m(x[:1,:t]);empty=torch.zeros_like(x[:1,:t],dtype=torch.bool)
        p=losses(m,z,x[:1,:t],lengths.new_tensor([t]),y[:1,:t],empty,candidate_update=True)
        assert torch.isfinite(p['total']);p['total'].backward()
    checks['empty_masks_short_groups_finite_backward']=True
    try:m(x,lengths,xi=torch.ones_like(m.core.codes(base['codes'])))
    except ValueError:pass
    else:raise AssertionError('Nonzero training xi accepted')
    checks['zero_gradcuit_contract']=True
    # Compare ALL parameter gradients, including sparse Engram, for exact winner recomputation.
    comparison={};objectives=[];full_grads=None
    for strategy in ['all_graphs','recompute']:
        m.zero_grad(set_to_none=True);o=m(x,lengths)
        p=losses(m,o,x,lengths,y,mask,candidate_update=True,generator=torch.Generator().manual_seed(97),candidate_strategy=strategy)
        p['total'].backward();objectives.append(float(p['total'].detach()))
        grads={name:((v.grad.to_dense() if v.grad.is_sparse else v.grad).detach().cpu().clone()) for name,v in m.named_parameters() if v.grad is not None}
        if full_grads is None:full_grads=grads
        else:
            assert full_grads.keys()==grads.keys()
            comparison={name:float((value-full_grads[name]).abs().max()) for name,value in grads.items()}
    assert abs(objectives[0]-objectives[1])<tol and max(comparison.values())<tol
    checks['winner_recomputation_objective_and_all_parameter_gradients']=dict(loss_error=abs(objectives[0]-objectives[1]),max_gradient_error=max(comparison.values()))
    m.zero_grad(set_to_none=True)
    accumulated=backward_effective_batch(m,dict(ids=x,lengths=lengths,targets=y,target_mask=mask),microbatch_size=1,candidate_update=True,seed=97)
    accumulation_error=max(float(((p.grad.to_dense() if p.grad.is_sparse else p.grad).detach().cpu()-grads[name]).abs().max()) for name,p in m.named_parameters() if p.grad is not None)
    assert abs(accumulated['total']-objectives[-1])<tol and accumulation_error<tol
    checks['microbatch_accumulation_matches_full_batch']=dict(loss_error=abs(accumulated['total']-objectives[-1]),max_gradient_error=accumulation_error)
    # Bank identity, combined-path derivative, snapshots and finite gradient steps.
    m.eval().requires_grad_(False);m.zero_grad(set_to_none=True)
    xx=x[:1,:17];ll=lengths.new_tensor([17]);yy=y[:1,:17];mm=mask[:1,:17]
    era=model_era(m,'test-tokenizer','test-preprocessing',.5)
    m.bind_memory_era(era,.5)
    bank=RBFMemory(era,capacity=16,sigma=.5)
    with torch.no_grad():
        normal=m(xx,ll);empty=m(xx,ll,bank=bank.snapshot())
        assert torch.equal(normal['logits'],empty['logits'])
    shape=normal['queries'].shape
    def derivative(**flags):
        r=torch.zeros(shape,device=device,requires_grad=True)
        o=m(xx,ll,correction_override=r,**flags)
        value=token_loss(o,yy,mm)
        return torch.autograd.grad(value,r)[0]
    full=derivative();pq=derivative(detach_continuous=True);continuous=derivative(detach_pq=True)
    assert pq.norm()>0 and continuous.norm()>0
    assert torch.allclose(full,pq+continuous,atol=tol,rtol=1e-4)
    checks['memory_credit_both_routes']=dict(full_norm=float(full.norm()),pq_norm=float(pq.norm()),continuous_norm=float(continuous.norm()),sum_error=float((full-pq-continuous).abs().max()))
    # Directional finite differences are better conditioned than tiny coordinate derivatives in FP32.
    direction=full/full.norm();eps=.01
    with torch.no_grad():
        plus=token_loss(m(xx,ll,correction_override=eps*direction),yy,mm)
        minus=token_loss(m(xx,ll,correction_override=-eps*direction),yy,mm)
    numeric=float((plus-minus)/(2*eps));analytic=float((full*direction).sum())
    assert abs(numeric-analytic)<max(1e-4,.03*abs(analytic)),(numeric,analytic)
    checks['real_model_memory_directional_derivative']=dict(numeric=numeric,analytic=analytic)
    snap=bank.snapshot();before_keys=normal['queries'].detach().clone()
    replay=m(xx,ll,bank=snap,write_pass=True)
    write_grad=torch.autograd.grad(write_objective(replay,yy,mm),replay['correction'])[0].detach().flatten(0,1)
    q=before_keys.flatten(0,1);code=normal['codes'].flatten()
    gram=torch.exp(-(q[:,None]-q[None]).square().sum(-1)/(2*bank.sigma**2))*(code[:,None]==code[None,:])
    expected_read=-.1*(gram@write_grad)
    predicted_change=float((write_grad*expected_read).sum())
    assert predicted_change<0
    report=learn_episode(m,bank,xx,ll,yy,mm,'one',era,eta=.1)
    with torch.no_grad():
        changed=m(xx,ll,bank=bank.snapshot());after=float(write_objective(changed,yy,mm))
        assert torch.equal(changed['queries'],before_keys)
        assert after<=report['prewrite_loss']+2e-6,(after,report)
        assert torch.allclose(changed['correction'].flatten(0,1),expected_read,atol=1e-8,rtol=1e-4)
        assert changed['correction'].norm()>0
    assert len(snap.keys)==0 and bank.version==1 and report['writes']>0
    checks['real_model_small_memory_update']=dict(**report,after_loss=after,queries_unchanged=True,
        first_order_predicted_loss_change=predicted_change,loss_reduction_resolved_at_fp32=after<report['prewrite_loss'])
    # Explicit exact Gaussian, code partition, differentiable query, capacity and payload guards.
    small=RBFMemory('toy',width=2,capacity=1,sigma=.7)
    small.commit(small.snapshot(),torch.zeros(1,2),torch.tensor([1]),torch.tensor([[-1.,2.]]),'first',eta=.1,max_value_rms=10)
    q=torch.tensor([[.1,.2],[.1,.2]],requires_grad=True)
    read=small.snapshot().read(q,torch.tensor([1,0]));expected=torch.exp(torch.tensor(-.05/(2*.7**2)))*torch.tensor([.1,-.2])
    assert torch.allclose(read[0],expected) and torch.equal(read[1],torch.zeros(2));read.sum().backward();assert q.grad[0].norm()>0
    rejected={}
    calls={
        'capacity':lambda:small.commit(small.snapshot(),torch.ones(1,2),torch.tensor([1]),torch.ones(1,2),'overflow'),
        'duplicate':lambda:learn_episode(m,bank,xx,ll,yy,mm,'one',era),
        'stale':lambda:bank.commit(snap,torch.zeros(1,256),torch.tensor([0]),torch.ones(1,256),'stale'),
        'era':lambda:RBFMemory.from_state_dict(bank.state_dict(),'wrong'),
        'inference_era':lambda:m(xx,ll,bank=RBFMemory('wrong').snapshot()),
        'inference_bandwidth':lambda:m(xx,ll,bank=RBFMemory(era,sigma=1.).snapshot()),
        'nonfinite':lambda:bank.commit(bank.snapshot(),torch.zeros(1,256),torch.tensor([0]),torch.full((1,256),float('nan')),'nan')}
    for name,call in calls.items():
        try:call()
        except (ValueError,OverflowError,FloatingPointError):rejected[name]=True
        else:raise AssertionError(name+' accepted')
    bank.commit(bank.snapshot(),torch.zeros(1,256),torch.tensor([0]),torch.zeros(1,256),'zero')
    try:bank.commit(bank.snapshot(),torch.zeros(1,256),torch.tensor([0]),torch.ones(1,256),'zero')
    except ValueError:rejected['zero_write_event_replay']=True
    else:raise AssertionError('Zero event replay accepted')
    clone=RBFMemory.from_state_dict(bank.state_dict(),era)
    assert torch.equal(clone.snapshot().read(before_keys,normal['codes']),bank.snapshot().read(before_keys,normal['codes']))
    checks['exact_RBF_query_gradient_and_memory_guards']=rejected
    del m
    # Actual interrupted checkpoint replay: parameters, both optimizers, RNG, next update.
    torch.manual_seed(787);m=V101(c).place(device).train();denseopt,sparseopt,dense=build_optimizers(m)
    def update():
        denseopt.zero_grad(set_to_none=True);sparseopt.zero_grad(set_to_none=True)
        o=m(x,lengths);p=losses(m,o,x,lengths,y,mask,candidate_update=True)
        p['total'].backward();clip(m,dense);denseopt.step();sparseopt.step();return float(p['total'].detach())
    update()
    payload=io.BytesIO();torch.save(dict(model=m.state_dict(),dense=denseopt.state_dict(),sparse=sparseopt.state_dict(),rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if device=='mps' else None),payload)
    expected_loss=update();expected={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    payload.seek(0);checkpoint=torch.load(payload,map_location='cpu',weights_only=False)
    m.load_state_dict(checkpoint['model']);denseopt.load_state_dict(checkpoint['dense']);sparseopt.load_state_dict(checkpoint['sparse']);torch.set_rng_state(checkpoint['rng'])
    if device=='mps':torch.mps.set_rng_state(checkpoint['mps_rng'])
    resumed_loss=update();error=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in m.state_dict().items())
    assert abs(expected_loss-resumed_loss)<tol and error<tol,(expected_loss,resumed_loss,error)
    checks['optimizer_checkpoint_replay']=dict(loss_error=abs(expected_loss-resumed_loss),max_parameter_error=error)
    return checks


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','mps'],default='cpu');p.add_argument('--output',required=True);args=p.parse_args()
    started=time.time();checks=run(args.device)
    result=dict(**provenance(),status='passed',device=args.device,seconds=time.time()-started,
                scope='Full width/depth; vocabulary 96 and reduced hash table for behavioral contracts',checks=checks)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
