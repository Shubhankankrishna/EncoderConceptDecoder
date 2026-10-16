"""Trained endpoint, forecast, zero-anchor, routing and deployment audits."""
import argparse,copy,time
import numpy as np
import torch
from model import System,freeze,shift,count
from runtime import *
from data import fixed_text,T,DATA
from generation import generate,Incremental
from evaluate import text_metrics

def audit(path,out):
    s,ck=load_system(path);s.eval();freeze(s);start=time.monotonic();before=tensor_hash(s);b=fixed_text('dev',8);x=b['x'].to(DEVICE);docs=b['docs'].to(DEVICE)
    if s.cfg.kind=='flat':
        r=dict(status='completed',kind='flat',active_parameters=s.active_count(),seconds=0.);atomic_json(out,r);return r
    z=torch.zeros(len(x),T//s.cfg.k,s.cfg.dz,device=DEVICE)
    with torch.no_grad():
        original=System(s.cfg,foundation=s.foundation)(x,docs);zero=s(x,docs,z);automatic=s(x,docs)
        endpoint0=s(x,docs,z,blend=0.);anchor=System(s.cfg,foundation=s.foundation,core=s.core)(x,docs)
        errors=dict(default_zero=float((automatic-zero).abs().max()),lambda0_original=float((original-endpoint0).abs().max()),lambda1_refit=float((anchor-zero).abs().max()))
    if any(v!=0 for v in errors.values()):raise RuntimeError('Anchor/transplant endpoint mismatch '+str(errors))
    g=None
    if s.adapters is not None:
        bank=z.clone().requires_grad_(True);scalar=s(x[:1],docs[:1],bank[:1])[0,127,41];g=torch.autograd.grad(scalar,bank)[0]
        if not bool(torch.isfinite(g).all()) or float(g.norm())==0:raise RuntimeError('Dead control derivative')
    with torch.no_grad():
        ctx=s.context(x,docs);logits,c,q,u,ports,valid=s.from_context(ctx,docs);cm=valid[:,:-1]&valid[:,1:]&(ctx[4][:,:-1]==ctx[4][:,1:]);target=c[:,1:];forecast=q[:,:-1]
        mse=float(((forecast-target).square().mean(-1)*cm).sum()/cm.sum());persist=float(((c[:,:-1]-target).square().mean(-1)*cm).sum()/cm.sum())
        centered=target[cm].cpu().numpy();variance=float(np.var(centered,axis=0).mean());sing=np.linalg.svd(centered-centered.mean(0),compute_uv=False);energy=sing**2;energy/=max(energy.sum(),1e-20)
        erank=float(np.exp(-np.sum(energy*np.log(energy+1e-20))))
        probs=s.foundation.head(u).view(*u.shape[:2],s.cfg.segments,s.cfg.codes).softmax(-1);prob=probs[valid].mean(0);entropy=-(prob*prob.clamp_min(1e-12).log()).sum(-1)
        hard=probs.argmax(-1);usage=[]
        for segment in range(s.cfg.segments):usage.append(int(torch.unique(hard[...,segment][valid]).numel()))
        baseline_nll=float(nll(zero,x,b['m'].to(DEVICE)).sum()/b['m'][:,:-1].sum())
        interventions={}
        saved=(s.cfg.forecast,s.cfg.continuous)
        for name,f,p in [('forecast_only',True,False),('continuous_only',False,True),('neither',False,False)]:
            s.cfg.forecast=f;s.cfg.continuous=p;v=float(nll(s(x,docs),x,b['m'].to(DEVICE)).sum()/b['m'][:,:-1].sum());interventions[name]=dict(nll=v,delta=v-baseline_nll)
        s.cfg.forecast,s.cfg.continuous=saved
    # Actual trained cached/full agreement at a non-aligned text prefix.
    xx=x[:1,:64];dd=torch.zeros_like(xx);zz=torch.zeros(1,16,s.cfg.dz,device=DEVICE);cache=Incremental(s,zz)
    with torch.no_grad():
        err=float((cache.prefill(xx[:,:15])-s(xx,dd,zz)[:,14]).abs().max())
        for t in range(15,20):err=max(err,float((cache.step(xx[:,t])-s(xx,dd,zz)[:,t]).abs().max()))
    if err>2e-4:raise RuntimeError('Trained cache divergence '+str(err))
    exported=copy.deepcopy(s)
    if exported.core is not None:exported.foundation.con=exported.core;exported.core=None
    with torch.no_grad():deploy_error=float((exported(x,docs)-zero).abs().max())
    if deploy_error!=0:raise RuntimeError('Deployment core merge mismatch')
    deployment=Path(out).with_name('deployment.pt');save_checkpoint(deployment,exported,None,ck['step'],None,None,dict(exported_from=str(path)),ck['extra'])
    restored,_=load_system(deployment)
    with torch.no_grad():reload_error=float((restored(x,docs)-zero).abs().max())
    if reload_error!=0:raise RuntimeError('Deployment reload mismatch')
    if tensor_hash(s)!=before:raise RuntimeError('Audit mutated the trained model')
    result=dict(status='completed',checkpoint=str(path),anchor_errors=errors,control_gradient_norm=float(g.norm()) if g is not None else None,cache_max_abs_error=err,
        forecast=dict(mse=mse,persistence_mse=persist,skill_vs_persistence=1-mse/max(persist,1e-20),target_variance=variance,normalized_mse=mse/max(variance,1e-20),effective_rank=erank),
        vocabulary=dict(mean_marginal_entropy=float(entropy.mean()),hard_codes_used_per_segment=usage),port_interventions=interventions,
        intervention_scope='inference distribution-shift diagnostics; not retrained port ablations',active_parameters=s.active_count(),resident_parameters=count(s),
        deployment_parameters=count(exported),deployment=str(deployment),deployment_max_abs_error=deploy_error,reload_max_abs_error=reload_error,seconds=time.monotonic()-start)
    atomic_json(out,result);return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(4);audit(a.checkpoint,a.out)
