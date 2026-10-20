"""Comparable saved decisions, paired baselines, and required-length rollouts."""
import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from .controls import StateActionPolicy,make_model
from .gram import AstraGRAM
from .dataset import Dataset
from .tasks import parse_action,step,observed
from .baselines import legal_actions,distance_heuristic,heuristic_rollout,bfs_plan
from .representation import serialize_observation,fixed_observed_blocks
from .metrics import cluster_interval,paired_difference,episode_report,code_diagnostics,length_bucket
from .isolation import atomic_json,require_legacy_idle,exclusive_job,output_path


@contextmanager
def evaluation_rng(seed,device):
    cpu=torch.get_rng_state();mps=torch.mps.get_rng_state() if str(device).startswith('mps') else None
    try:
        # CPU RNG only unless evaluating an actual MPS model.
        torch.random.default_generator.manual_seed(seed)
        if mps is not None:torch.mps.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(cpu)
        if mps is not None:torch.mps.set_rng_state(mps)


def stratified_episodes(data,split,per_bucket=32,seed=151803):
    rng=np.random.default_rng(seed);chosen=[];available={}
    for bucket in ('1-4','5-9','10-19','20+'):
        panel=[e for e in data.episodes if e['split']==split and length_bucket(len(e['trajectory']))==bucket]
        available[bucket]=len(panel)
        if panel:chosen.extend(panel[int(i)] for i in rng.choice(len(panel),min(len(panel),per_bucket),replace=False))
    return chosen,available


def observe_codes(codec,env,state,goal,blocks,device):
    ids=fixed_observed_blocks(codec.bpe.encode(serialize_observation(env,state,goal)),
                              blocks=blocks,pad_id=codec.pad)
    return codec.encode_ids(torch.tensor(ids,dtype=torch.long)).reshape(1,blocks,641).to(device)


@torch.inference_mode()
def rollout(model,data,codec,episode,horizon=40,legal_filter=False):
    device=next(model.parameters()).device;e=episode;s=e['initial'];start=time.perf_counter()
    done_at=None;invalid=False;actions=[]
    for k in range(1,horizon+1):
        state=torch.tensor([observed(e['env'],s,e['goal'])[1]],dtype=torch.long,device=device)
        if isinstance(model,StateActionPolicy):
            mask=None
            if legal_filter:
                mask=torch.zeros(1,7,dtype=torch.bool,device=device);mask[0,legal_actions(e['env'],s)]=True
            a=int(model.generate_actions(state,mask)[0])
        else:
            if legal_filter:raise ValueError('Code-policy legality needs a separately budgeted candidate/ranking protocol')
            x=observe_codes(codec,e['env'],s,e['goal'],data.encoding['context_blocks'],device)
            a=parse_action(codec.decode_text(model.generate(x,state).cpu())[0])
        actions.append(a);nxt=None if a is None else step(e['env'],s,a)
        if nxt is None:invalid=True;break
        s=nxt
        if s==e['goal']:done_at=k;break
    return dict(episode=e['episode'],group=e['group'],minimum_required_steps=len(e['trajectory']),
                steps_to_success=done_at,invalid=invalid,actions=actions,seconds=time.perf_counter()-start,
                legal_filter=legal_filter,recipe=e.get('recipe','unspecified'))


@torch.inference_mode()
def evaluate(model,data,codec=None,*,split='valid',n=128,per_bucket=32,seed=151803,closed_loop=True,batch=4):
    device=next(model.parameters()).device;was_training=model.training
    if min(n,per_bucket,batch)<1:raise ValueError('Positive evaluation budgets required')
    with evaluation_rng(seed,device):
        try:
            model.eval();rng=np.random.default_rng(seed)
            ix=rng.choice(data.indices[split],min(n,len(data.indices[split])),replace=False)
            if not len(ix):raise ValueError('Empty evaluation split')
            is_action=isinstance(model,StateActionPolicy);records=[];pred_codes=[];target_codes=[];nll=[]
            if not is_action:
                if codec is None or data.encoding is None:raise ValueError('Code evaluation requires the pinned codec and complete cache')
                codec.assert_matches(data.encoding['codec'])
            start=time.perf_counter()
            for offset in range(0,len(ix),batch):
                jj=ix[offset:offset+batch];b=data.batch(jj,device,require_codes=not is_action)
                diagnostics=None
                if is_action:
                    logits=model.logits(b['state']);preds=logits.argmax(-1).cpu().tolist()
                    filtered=logits.masked_fill(~b['legal_mask'],-torch.inf).argmax(-1).cpu().tolist()
                else:
                    stats=model.score_prior(b['x'],b['state'],b['y'],samples=4) if isinstance(model,AstraGRAM) else model(b['x'],b['state'],b['y'])[1]
                    nll.append((float(stats['nll']),len(jj)))
                    codes=model.generate(b['x'],b['state']);preds=[parse_action(t) for t in codec.decode_text(codes.cpu())]
                    pred_codes.extend(codes.cpu().tolist());target_codes.extend(b['y'].cpu().tolist());filtered=[None]*len(jj)
                    if isinstance(model,AstraGRAM):
                        mean=[parse_action(t) for t in codec.decode_text(model.generate(b['x'],b['state'],mean=True).cpu())]
                        other=[parse_action(t) for t in codec.decode_text(model.generate(b['x'],b['state']).cpu())]
                        diagnostics=list(zip(mean,other))
                for k,(idx,a) in enumerate(zip(jj,preds)):
                    e,r=data.record(idx);heur=distance_heuristic(e['env'],r['current'],e['goal'])
                    legal=legal_actions(e['env'],r['current']);opts=r['valid_actions']
                    rec=dict(row=int(idx),group=e['group'],domain=e['env']['domain'],
                             remaining=r['remaining'],prediction=a,reference=r['action'],optimal_actions=opts,
                             optimal=a in opts,legal=a in legal,heuristic=heur,heuristic_optimal=heur in opts,
                             uniform_legal_expected=len(opts)/len(legal),
                             filtered_prediction=filtered[k],filtered_optimal=filtered[k] in opts if is_action else None)
                    if diagnostics is not None:
                        mean,other=diagnostics[k];rec.update(prior_mean=mean,second_prior=other,
                            prior_mean_optimal=mean in opts,prior_disagreement=a!=other,
                            extra_distinct_optimal_actions=len({v for v in (a,other) if v in opts})-int(a in opts))
                    records.append(rec)
            result=dict(split=split,seed=seed,examples=len(records),seconds=time.perf_counter()-start,
                        decisions=records,model_optimal=cluster_interval([r['optimal'] for r in records],[r['group'] for r in records]),
                        model_legal=float(np.mean([r['legal'] for r in records])),
                        heuristic_optimal=cluster_interval([r['heuristic_optimal'] for r in records],[r['group'] for r in records]),
                        paired_difference_vs_heuristic=paired_difference([r['optimal'] for r in records],
                            [r['heuristic_optimal'] for r in records],[r['group'] for r in records]),
                        bootstrap_estimand='example_weighted_cluster_resampling',
                        heuristic_scope='known-rule legality and local distance; no shortest-path labels used in proposals')
            if pred_codes:
                result['code_diagnostics']=code_diagnostics(target_codes,pred_codes,data.target_codes)
                result['nll_per_code']=sum(v*n for v,n in nll)/sum(n for _,n in nll)
                result['nll_per_block']=result['nll_per_code']*data.target_codes.shape[1]
                result['nll_kind']='prior_MC_logmeanexp_4_estimate' if isinstance(model,AstraGRAM) else 'exact_conditional_code_NLL'
            result['by_domain']={domain:cluster_interval([r['optimal'] for r in records if r['domain']==domain],
                                    [r['group'] for r in records if r['domain']==domain]) for domain in sorted({r['domain'] for r in records})}
            result['by_remaining_length']={bucket:cluster_interval([r['optimal'] for r in panel],[r['group'] for r in panel])
                for bucket in ('1-4','5-9','10-19','20+') if (panel:=[r for r in records if length_bucket(r['remaining'])==bucket])}
            if closed_loop:
                eps,available=stratified_episodes(data,split,per_bucket,seed)
                roll=[rollout(model,data,codec,e) for e in eps];result['rollouts']=roll
                result['closed_loop']=episode_report(roll,horizons=(5,10,20,40));result['available_length_buckets']=available
                heuristic=[];solver=[]
                for e in eps:
                    h=heuristic_rollout(e['env'],e['initial'],e['goal'],40)
                    heuristic.append(dict(**h,group=e['group'],minimum_required_steps=len(e['trajectory'])))
                    t=time.perf_counter();p=bfs_plan(e['env'],e['initial'],e['goal']);p['seconds']=time.perf_counter()-t;solver.append(p)
                result['heuristic_closed_loop']=episode_report(heuristic,horizons=(5,10,20,40))
                result['paired_episode_difference_vs_heuristic']=paired_difference(
                    [r['steps_to_success'] is not None for r in roll],
                    [r['steps_to_success'] is not None for r in heuristic],[r['group'] for r in roll])
                result['exact_solver_control']=dict(label='BFS with known transitions, not a learned model',records=solver)
            return result
        finally:model.train(was_training)


@exclusive_job
def run(args):
    require_legacy_idle();torch.set_num_threads(1)
    output_path(args.out)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=make_model(ck['arm'],ck['config'],inference=ck.get('posterior_included') is False)
    model.load_state_dict(ck['model'],strict=True);model.to(args.device);data=Dataset(args.data)
    if ck['dataset_sha']!=data.manifest['episodes_sha']:raise ValueError('Checkpoint/dataset mismatch')
    codec=None
    if not isinstance(model,StateActionPolicy):
        from .codec import FrozenNCMCodec
        codec=FrozenNCMCodec('cpu',family='lm1b')
    atomic_json(args.out,evaluate(model,data,codec,split=args.split))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint',type=Path);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--split',choices=['valid','test'],default='valid')
    ap.add_argument('--device',choices=['cpu','mps'],default='cpu');run(ap.parse_args())
