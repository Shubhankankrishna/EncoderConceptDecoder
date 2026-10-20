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
from .tasks import ACTIONS,parse_action,step,observed,distances,optimal_actions


def bootstrap_groups(values,groups,seed=1518):
    buckets={}
    for v,g in zip(values,groups):buckets.setdefault(g,[]).append(v)
    vals=np.array([np.mean(v) for v in buckets.values()]);rng=np.random.default_rng(seed)
    if not len(vals):return dict(mean=None,ci95=None,groups=0)
    samples=[rng.choice(vals,len(vals),replace=True).mean() for _ in range(1000)]
    return dict(mean=float(vals.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),groups=len(vals))


@torch.inference_mode()
def evaluate(model,data,codec,split='valid',n=128,batch=4,closed_loop=False,seed=1518):
    codec.assert_matches(data.manifest['codec'])
    model.eval();device=next(model.parameters()).device;rng=np.random.default_rng(seed)
    pool=data.indices[split];ix=rng.choice(pool,min(n,len(pool)),replace=False)
    losses=[];perlevel=[];preds=[];fullcodes=[];codeacc=[];samples=[];t=time.perf_counter()
    for start in range(0,len(ix),batch):
        jj=ix[start:start+batch];x,s,y=data.batch(jj,device)
        _,stats=model(x,s,y);losses.append((float(stats['nll']),len(jj)))
        perlevel.append(stats['per_level'].cpu().numpy())
        codes=model.generate(x,s);txt=codec.decode_text(codes.cpu())
        preds.extend(parse_action(a) for a in txt)
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
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True);m=AstraHRM(Config(**ckpt['config'])).to('mps');m.load_state_dict(ckpt['model']);d=Dataset()
    assert ckpt['data_sha']==d.manifest['code_sha']
    codec=FrozenNCMCodec('cpu',family='lm1b');atomic_json(args.out,evaluate(m,d,codec,args.split,args.n,closed_loop=True))
