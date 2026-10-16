"""Post-training checks of frozen weights, latent authority and failure credit."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model import freeze,chunk_info,DZ,safe_rms,ball
from data import fixed_graph,fixed_text,PAD,T,WALK_START,COL0,N_COL,NODE0,N_NODE_TOK,DONE,SEP,TO
from runtime import DEVICE,load_system,atomic_json
from generation import generate,expand_bank,replay_batch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--anchor',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(4);s,ck=load_system(a.checkpoint);anchor,base=load_system(a.anchor);freeze(s);freeze(anchor)
    xs,_,_,_=fixed_graph(8,64001);x=torch.as_tensor(xs,device=DEVICE);docs=torch.where(x!=PAD,torch.zeros_like(x),torch.full_like(x,-1));z=torch.zeros(8,T//4,DZ,device=DEVICE,requires_grad=True)
    results={}
    results['frozen_foundation_exact']=all(torch.equal(v,base['foundation'][k]) for k,v in ck['foundation'].items())
    results['frozen_core_exact']=all(torch.equal(v,base['core'][k]) for k,v in ck['core'].items())
    with torch.no_grad():zero=s(x,docs,z);reference=anchor(x,docs)
    results['zero_anchor_max_logit_difference']=float((zero-reference).abs().max())
    from runtime import ROOT
    fast=json.loads((ROOT/'runs'/'calibration'/'benchmark.json').read_text())['selected_generation']=='compiled_static_cache'
    s.fast_generation=fast;anchor.fast_generation=fast
    baseline_generation=generate(anchor,xs[:4]);controlled_generation=generate(s,xs[:4])
    results['zero_anchor_cached_generation_exact']=bool(np.array_equal(baseline_generation['x'],controlled_generation['x']))
    h,c,cd=s.context(x,docs);e=s.core.in_proj(c)
    prior=ball(torch.randn(8,T//4,DZ,generator=torch.Generator().manual_seed(64002))).to(DEVICE)
    enabled=(torch.arange(T//4,device=DEVICE)>=WALK_START//4-1)[None,:,None]
    prior=prior*enabled
    results['interpolation']=[]
    for scale in (0.,.25,.5,1.):
        control=(prior*scale).detach().requires_grad_(True)
        logits=s(x,docs,control*enabled)
        # Symbolic demonstration NLL is a diagnostic, not deployed privileged search.
        m=torch.zeros_like(x,dtype=torch.float32);m[:,WALK_START-1:]=((x[:,WALK_START-1:]!=PAD)&(torch.arange(T-WALK_START+1,device=DEVICE)[None]<((x!=PAD).sum(1)-WALK_START)[:,None])).float()
        from runtime import nll
        loss=nll(logits,x,m).sum()/m.sum().clamp_min(1)
        gradient=torch.autograd.grad(loss,control)[0]
        with torch.no_grad():
            ratios=[safe_rms(s.adapters.raw(e,control,q))/(.1*safe_rms(e)) for q in (0,1)]
            u,a1=s.core(c,cd,control,s.adapters);u0,a0=s.core(c,cd)
            lp0=zero.float().log_softmax(-1);lp=logits.float().log_softmax(-1)
            divergence=(lp0.exp()*(lp0-lp)).sum(-1)
        results['interpolation'].append(dict(scale=scale,nll=float(loss.detach()),gradient_rms=float(safe_rms(gradient).mean()),
                 cap_hit_fraction=[float((r>=1).float().mean()) for r in ratios],u_shift_rms=float(safe_rms(u-u0).mean()),a_shift_rms=float(safe_rms(a1-a0).mean()),
                 replay_kl=float((divergence*m).sum()/m.sum().clamp_min(1))))
    generated=generate(s,xs[:4]);tokens,d,mask=replay_batch(generated,DEVICE);bank=torch.zeros(4,T//4,DZ,device=DEVICE,requires_grad=True)
    logits=s(tokens,d,expand_bank(bank,1,T//4));logp=logits[:,:-1].float().log_softmax(-1).gather(-1,tokens[:,1:,None]).squeeze(-1)
    target=tokens[:,1:];m=mask[:,:-1];den=m.sum(1).clamp_min(1)
    kinds={'colour':(target>=COL0)&(target<COL0+N_COL),'destination':(target>=NODE0)&(target<NODE0+N_NODE_TOK),
           'format':(target==SEP)|(target==TO),'stopping':target==DONE}
    known=torch.zeros_like(target,dtype=torch.bool)
    for kind in kinds.values():known|=kind
    kinds['other']=~known;contributions=[]
    totalgrad=torch.autograd.grad(((logp*m).sum(1)/den).sum(),bank,retain_graph=True)[0]
    for name,part in kinds.items():
        gradient=torch.autograd.grad(((logp*m*part).sum(1)/den).sum(),bank,retain_graph=True)[0]
        cosine=(gradient.flatten(1)*totalgrad.flatten(1)).sum(1)/(gradient.flatten(1).norm(dim=1)*totalgrad.flatten(1).norm(dim=1)).clamp_min(1e-12)
        contributions.append(dict(token_type=name,scored_tokens=int((part*m).sum()),gradient_norm=gradient.flatten(1).norm(dim=1).cpu().tolist(),cosine_to_total=cosine.cpu().tolist()))
    results['rejection_gradient_token_attribution']=contributions
    results['correctness_passed']=results['frozen_foundation_exact'] and results['frozen_core_exact'] and results['zero_anchor_max_logit_difference']==0 and results['zero_anchor_cached_generation_exact']
    atomic_json(a.out,results)
    if not results['correctness_passed']:raise AssertionError('Post-training frozen/zero-anchor contract failed')


if __name__=='__main__':main()
