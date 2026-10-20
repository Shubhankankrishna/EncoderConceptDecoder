"""Free decoded actions and exact environment outcomes, separate from code CE."""
import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from .codec import FrozenNCMCodec, ROOT
from .data import Dataset,atomic_json
from .model import AstraHRM,Config
from .gram import AstraGRAM,make_model
from .tasks import ACTIONS,parse_action,step,observed,distances,optimal_actions


def bootstrap_groups(values,groups,seed=1518):
    buckets={}
    for v,g in zip(values,groups):buckets.setdefault(g,[]).append(v)
    vals=np.array([np.mean(v) for v in buckets.values()]);rng=np.random.default_rng(seed)
    if not len(vals):return dict(mean=None,ci95=None,groups=0)
    samples=[rng.choice(vals,len(vals),replace=True).mean() for _ in range(1000)]
    return dict(mean=float(vals.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),groups=len(vals))


def evaluate(*args,seed=1518,**kwargs):
    cpu=torch.get_rng_state();mps=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        torch.manual_seed(seed)
        if mps is not None:torch.mps.manual_seed(seed)
        return _evaluate(*args,seed=seed,**kwargs)
    finally:
        torch.set_rng_state(cpu)
        if mps is not None:torch.mps.set_rng_state(mps)


@torch.inference_mode()
def _evaluate(model,data,codec,split='valid',n=128,batch=4,closed_loop=False,seed=1518):
    codec.assert_matches(data.manifest['codec'])
    model.eval();device=next(model.parameters()).device;rng=np.random.default_rng(seed)
    pool=data.indices[split];ix=rng.choice(pool,min(n,len(pool)),replace=False)
    losses=[];perlevel=[];preds=[];fullcodes=[];codeacc=[];samples=[];t=time.perf_counter();diagnostics=[]
    is_gram=isinstance(model,AstraGRAM)
    for start in range(0,len(ix),batch):
        jj=ix[start:start+batch];x,s,y=data.batch(jj,device)
        if is_gram:stats=model.score_prior(x,s,y,samples=4)
        else:_,stats=model(x,s,y)
        losses.append((float(stats['nll']),len(jj)))
        perlevel.append(stats['per_level'].cpu().numpy())
        codes=model.generate(x,s);txt=codec.decode_text(codes.cpu())
        preds.extend(parse_action(a) for a in txt)
        if is_gram and start<32:
            mean_text=codec.decode_text(model.generate(x,s,mean=True).cpu())
            second_text=codec.decode_text(model.generate(x,s).cpu())
            qtext=None
            if hasattr(model,'posterior'):
                qc,_=model.posterior_diagnostic(x,s,y);qtext=codec.decode_text(qc.cpu())
            for j,pt,mt,st,qt in zip(jj,txt,mean_text,second_text,qtext or [None]*len(jj)):
                e=data.episodes[int(data.rows['episode'][j])];row=e['trajectory'][int(data.rows['transition'][j])]
                pa,ma,sa,qa=map(lambda t:parse_action(t) if t is not None else None,(pt,mt,st,qt))
                valid=row['valid_actions']
                diagnostics.append(dict(prior=pa in valid,mean=ma in valid,posterior=None if qt is None else qa in valid,
                                        two_draw_valid_diversity=len({a for a in (pa,sa) if a in valid})))
        fullcodes.extend((codes==y).all(-1).cpu().tolist());codeacc.extend((codes==y).float().mean(-1).cpu().tolist())
        for j,z in zip(jj,txt):
            if len(samples)<12:samples.append(dict(row=int(j),target=int(data.rows['action'][j]),decoded=z[:400]))
    correct=[];legal=[];groups=[];domains=[];majority=[];random=[]
    train=data.indices['train'];domain_majority={}
    for d in (1,2):
        aa=data.rows['action'][train[data.rows['state'][train,0]==d]]
        domain_majority[d]=int(np.bincount(aa,minlength=7).argmax())
    for row,a in zip(ix,preds):
        episode=data.episodes[int(data.rows['episode'][row])]
        r=episode['trajectory'][int(data.rows['transition'][row])]
        env=episode['env'];correct.append(a in r['valid_actions']);groups.append(episode['group']);domains.append(env['domain'])
        legal.append(a is not None and step(env,r['current'],a) is not None)
        majority.append(domain_majority[int(data.rows['state'][row,0])] in r['valid_actions'])
        legal_actions=[a for a in range(7) if step(env,r['current'],a) is not None]
        random.append(len(r['valid_actions'])/len(legal_actions))
    result=dict(split=split,examples=len(ix),nll_per_code=sum(v*k for v,k in losses)/sum(k for _,k in losses),
                per_level_nll=np.mean(perlevel,axis=0).tolist(),exact_code_rate=float(np.mean(fullcodes)),
                code_accuracy=float(np.mean(codeacc)),parse_rate=float(np.mean([x is not None for x in preds])),
                legal_action_rate=float(np.mean(legal)),optimal_action_rate=float(np.mean(correct)),
                optimal_action_group_ci=bootstrap_groups(correct,groups),
                majority_baseline=float(np.mean(majority)),uniform_legal_baseline=float(np.mean(random)),
                by_domain={d:float(np.mean([v for v,dd in zip(correct,domains) if dd==d])) for d in set(domains)},
                samples=samples,seconds=time.perf_counter()-t,
                prediction_mode='all 12 levels freely generated; greedy frozen NCM decode; first-sentence parser')
    result['nll_kind']='prior_MC_logmeanexp_4_samples_estimate' if is_gram else 'exact_conditional_per_code'
    result['per_level_nll_kind']='mean_prior_conditional' if is_gram else 'conditional'
    if diagnostics:
        ds=dict(examples=len(diagnostics),prior_success=float(np.mean([r['prior'] for r in diagnostics])),
                prior_mean_success=float(np.mean([r['mean'] for r in diagnostics])),
                mean_distinct_valid_actions_two_draws=float(np.mean([r['two_draw_valid_diversity'] for r in diagnostics])))
        if diagnostics[0]['posterior'] is not None:
            ds['posterior_success']=float(np.mean([r['posterior'] for r in diagnostics]))
            ds['posterior_minus_prior_success']=ds['posterior_success']-ds['prior_success']
        result['gram_diagnostics']=ds
    if closed_loop:result['closed_loop']=rollouts(model,data,codec,split,24,seed)
    return result


@torch.inference_mode()
def rollouts(model,data,codec,split,n=24,seed=1518):
    device=next(model.parameters()).device;rng=np.random.default_rng(seed)
    eps=[e for e in data.episodes if e['split']==split]
    chosen=rng.choice(len(eps),min(n,len(eps)),replace=False)
    eps=[eps[i] for i in chosen];states=[e['initial'] for e in eps]
    done=[False]*len(eps);success_at=[None]*len(eps);invalid=0;t=time.perf_counter()
    for k in range(1,21):
        active=[i for i in range(len(eps)) if not done[i]]
        if not active:break
        for st in range(0,len(active),4):
            ii=active[st:st+4]
            records=[observed(eps[i]['env'],states[i],eps[i]['goal']) for i in ii]
            ids,_=codec.tokenize([r[0] for r in records]);codes=codec.encode_ids(ids).to(device)
            state=torch.tensor([r[1] for r in records],dtype=torch.long,device=device)
            text=codec.decode_text(model.generate(codes,state).cpu())
            for i,txt in zip(ii,text):
                action=parse_action(txt);nxt=None if action is None else step(eps[i]['env'],states[i],action)
                if nxt is None:done[i]=True;invalid+=1;continue
                states[i]=nxt
                if nxt==eps[i]['goal']:done[i]=True;success_at[i]=k
    return dict(episodes=len(eps),invalid=invalid,seconds=time.perf_counter()-t,
                solved_by_horizon={str(h):sum(k is not None and k<=h for k in success_at)/len(eps) for h in (5,10,20)},
                groups=[e['group'] for e in eps],steps_to_success=success_at,
                minimum_required_steps=[len(e['trajectory']) for e in eps],
                scope='closed loop at horizons 5/10/20; not all selected puzzles require 10/20 actions')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint');ap.add_argument('--split',default='test');ap.add_argument('--n',type=int,default=512);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(4)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    arm=ckpt.get('arm','hrm');inference=ckpt.get('posterior_included') is False
    m=make_model(arm,ckpt['config'],inference=inference).to('mps');m.load_state_dict(ckpt['model']);d=Dataset()
    assert ckpt['data_sha']==d.manifest['code_sha']
    codec=FrozenNCMCodec('cpu',family='lm1b');atomic_json(args.out,evaluate(m,d,codec,args.split,args.n,closed_loop=True))
