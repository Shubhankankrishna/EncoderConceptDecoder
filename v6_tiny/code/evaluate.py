"""Per-item generation/search evidence, strict verifier and measured search costs."""
import argparse
from collections import Counter
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import DZ,ball,safe_rms,freeze,chunk_info,active_count
from data import T,WALK_START,fixed_text,fixed_graph,oracle,canonical,PAD,verify
from runtime import DEVICE,load_system,atomic_json,Log,nll,sync,controls
from generation import generate,expand_bank,replay_batch,MaskedAdam,active_windows


def repair_gradient(system,result,bank,length,kl=.05,target=None):
    device=bank.device;x,docs,mask=replay_batch(result,device)
    z=expand_bank(bank,length,T//4)
    with torch.no_grad():h,c,cd=getattr(system,'optimized_context',system.context)(x,docs)
    def replay(z):
        if hasattr(system,'optimized_replay'):return system.optimized_replay(h,c,cd,z,docs)[0]
        chat,a,_=system.plans(c,cd,z)
        return system.foundation.decode(h,chat,a,docs)
    logits=replay(z).float();lp=logits[:,:-1].log_softmax(-1)
    chosen=lp.gather(-1,x[:,1:,None]).squeeze(-1)
    counts=mask[:,:-1].sum(1).clamp_min(1)
    rejection=(chosen*mask[:,:-1]).sum(1)/counts
    if target is not None:rejection=-rejection
    replay_kl=torch.zeros_like(rejection)
    if kl:
        with torch.no_grad():base=replay(torch.zeros_like(z))[:,:-1].float().log_softmax(-1)
        divergence=(base.exp()*(base-lp)).sum(-1)
        replay_kl=(divergence*mask[:,:-1]).sum(1)/counts
    loss=rejection+kl*replay_kl
    gradient=torch.autograd.grad(loss.sum(),bank)[0]
    def numbers(value):return [float(v) if math.isfinite(float(v)) else None for v in value.detach().cpu().tolist()]
    return gradient.detach(),dict(loss=numbers(loss),replay_kl=numbers(replay_kl))


def search_batch(system,headers,method,updates=10,seed=0,length=1,kl=.05,budget=None,lr=.1):
    sync(next(system.parameters()).device);policy_start=time.perf_counter()
    device=next(system.parameters()).device;b=len(headers);bank=torch.zeros(b,T//4,DZ,device=device)
    opt=MaskedAdam(bank,lr);rng=torch.Generator().manual_seed(seed)
    solved=np.zeros(b,bool);attempts=[];records=[[] for _ in range(b)];total=0.;generations=0;backward=0;replay_calls=0
    numerical=np.zeros(b,bool)
    trace=[];prompt_encoding=None
    for attempt in range(updates+1):
        if attempt>0:
            eligible=torch.tensor(~solved&~numerical,device=device)
            active=active_windows(previous['lengths'],length,device,bank.shape[1])
            if method in ('gradcuit','gradcuit_kl','shared','random_matched'):
                variable=bank.detach().clone().requires_grad_(True);sync(device);start=time.perf_counter()
                grad,diag=repair_gradient(system,previous,variable,length,kl if method=='gradcuit_kl' else 0.)
                sync(device);elapsed=time.perf_counter()-start;total+=elapsed;backward+=1;replay_calls+=2 if method=='gradcuit_kl' else 1
                old=bank.clone();update=opt.update(grad,active,eligible)
                if method=='random_matched':
                    direction=torch.randn(*bank.shape,generator=rng).to(device)*active[:,:,None]*eligible[:,None,None]
                    direction=direction/direction.flatten(1).norm(dim=1).clamp_min(1e-12)[:,None,None]
                    target=update['step_norm'];low=torch.zeros(b,device=device);high=torch.ones(b,device=device)
                    def proposal(scale):return torch.where((active&eligible[:,None])[:,:,None],ball(old+scale[:,None,None]*direction),old)
                    for _ in range(8):
                        dist=(proposal(high)-old).flatten(1).norm(dim=1)
                        high=torch.where(dist<target,high*2,high)
                    for _ in range(16):
                        mid=(low+high)/2;dist=(proposal(mid)-old).flatten(1).norm(dim=1)
                        low=torch.where(dist<target,mid,low);high=torch.where(dist>=target,mid,high)
                    bank.copy_(proposal((low+high)/2))
                    diag['matched_actual_step_norm']=(bank-old).flatten(1).norm(dim=1).cpu().tolist()
                    diag['norm_source']='Online gradient/Adam step at this random trajectory; all gradient computation charged'
                numerical |= ~update['finite'].cpu().numpy()
                trace.append(dict(attempt=attempt,backward_s=elapsed,gradient_norm=update['norm'].cpu().tolist(),step_norm=update['step_norm'].cpu().tolist(),**diag))
            elif method=='random':
                perturb=torch.randn(*bank.shape,generator=rng).to(device)
                perturb=perturb*active[:,:,None]
                perturb=perturb/perturb.flatten(1).norm(dim=1).clamp_min(1e-12)[:,None,None]*(lr*math.sqrt(DZ))
                bank.copy_(torch.where((active&eligible[:,None])[:,:,None],ball(bank+perturb),bank))
            elif method=='prior':
                proposal=ball(torch.randn(*bank.shape,generator=rng)).to(device)
                bank.copy_(torch.where(eligible[:,None,None],proposal,bank))
        sync(device);total=time.perf_counter()-policy_start
        if budget is not None and total>=budget:break
        if method=='shared':length=T//4
        result=generate(system,headers,bank,length,temperature=1 if method=='retry' and attempt else 0,seed=seed+attempt,prompt_encoding=prompt_encoding)
        prompt_encoding=result['prompt_encoding']
        numerical|=np.asarray(result['numerical_failures'],bool)
        total=time.perf_counter()-policy_start;generations+=1
        within=budget is None or total<=budget
        for i,v in enumerate(result['verification']):
            if solved[i] or numerical[i]:continue
            rec=dict(attempt=attempt,success=bool(v['success'] and within),verified_success=bool(v['success']),within_budget=within,
                     reason=v['reason'],format_ok=v['format_ok'],trajectory=v['trajectory'],length=int(result['lengths'][i]),
                     cumulative_batch_seconds=total,generated_response=result['x'][i,WALK_START:int(result['lengths'][i])].tolist())
            records[i].append(rec)
            if within and v['success']:solved[i]=True
        attempts.append(dict(attempt=attempt,success=solved.tolist(),seconds=total,generated_tokens=result['generated_tokens'],forward_calls=result['forward_calls'],verifier_s=result['verifier_s'],encoder_prefill_reused=result['encoder_prefill_reused']))
        previous=result
        if bool((solved|numerical).all()) or not within:break
    return dict(records=records,success=solved.tolist(),seconds=total,attempts=attempts,gradient_trace=trace,
                generations=generations,backward_passes=backward,probability_replays=replay_calls,numerical_failures=numerical.tolist())


@torch.no_grad()
def likelihood(system,split='dev',n=128,bs=16,lam=1.,prior=False,length=1,seed=60110):
    xt,mt,dt=fixed_text(n,seed,split)
    from data import DATA
    fixed=np.load(DATA/f'{split}_standard_s0.npz');xg,mg=fixed['x'][:n],fixed['m'][:n]
    arrays=[(xt,mt,dt),(xg,mg,np.where(xg!=PAD,0,-1))]
    totals={};generator=torch.Generator().manual_seed(seed+2)
    for domain,(xs,ms,docs) in enumerate(arrays):
        num=den=snum=sden=anchor_suffix=0.;per=[]
        for start in range(0,n,bs):
            x=torch.as_tensor(xs[start:start+bs],device=DEVICE);m=torch.as_tensor(ms[start:start+bs],device=DEVICE)
            d=torch.as_tensor(docs[start:start+bs],device=DEVICE);z=None;mask=m
            if prior:
                batch=dict(x=x,m=m,docs=d,domain=torch.full((len(x),),domain,device=DEVICE,dtype=torch.long))
                z,mask,_,_,_=controls(batch,generator,torch.full((len(x),),length,dtype=torch.long))
            logits=system(x,d,z,lam)
            losses=nll(logits,x,m);suffix=nll(logits,x,mask)
            if prior:anchor_suffix+=float(nll(system(x,d,None,lam),x,mask).sum())
            nums=losses.sum(1).cpu().numpy();dens=m[:,:-1].sum(1).cpu().numpy()
            num+=nums.sum();den+=dens.sum();snum+=float(suffix.sum());sden+=float(mask[:,:-1].sum())
            per.extend([dict(nll_sum=float(a),targets=float(b)) for a,b in zip(nums,dens)])
        totals['text' if domain==0 else 'graph']=dict(nll=float(num/max(den,1)),suffix_nll=float(snum/max(sden,1)),targets=float(den),per_row=per,
                                                  matched_anchor_suffix_nll=float(anchor_suffix/max(sden,1)) if prior else None)
    return totals


def bootstrap(a,b,seed=65100):
    a,b=np.asarray(a,float),np.asarray(b,float);rng=np.random.RandomState(seed);delta=a-b
    samples=delta[rng.randint(len(delta),size=(2000,len(delta)))].mean(1)
    return dict(gain_points=float(100*delta.mean()),ci95_points=[float(x) for x in 100*np.quantile(samples,[.025,.975])],n=len(delta))


def reachability(system,headers,out,starts=2,updates=25,limit=16):
    freeze(system);results=[]
    for i,header in enumerate(headers[:limit]):
        gold=oracle(header);x=np.full((1,T),PAD,np.int64);x[0,:WALK_START]=header[:WALK_START];x[0,WALK_START:WALK_START+len(gold)]=gold
        fixture=dict(x=x,lengths=np.array([WALK_START+len(gold)]))
        successes=[]
        for start in range(starts):
            bank=torch.zeros(1,T//4,DZ,device=DEVICE) if start==0 else ball(torch.randn(1,T//4,DZ,generator=torch.Generator().manual_seed(68000+i))).to(DEVICE)
            opt=MaskedAdam(bank,.1)
            for _ in range(updates):
                variable=bank.detach().clone().requires_grad_(True)
                gradient,_=repair_gradient(system,fixture,variable,1,0.,target=True)
                opt.update(gradient,active_windows(fixture['lengths'],1,DEVICE),torch.tensor([True],device=DEVICE))
            generated=generate(system,header[None],bank)
            successes.append(generated['verification'][0]['success'])
        results.append(dict(item=i,oracle_fit_verified=any(successes),starts=successes))
        atomic_json(out,dict(privileged=True,ordinary_repair=False,starts=starts,updates=updates,items=results))
    return results


def evaluate_checkpoint(checkpoint,out,seed=0,split='dev',n_standard=128,n_hard=256,bs=16,full=True,budget_per_batch=None,deadline=None,methods_override=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);system,_=load_system(checkpoint);freeze(system)
    from runtime import ROOT
    calibrated=json.loads((ROOT/'runs'/'calibration'/'benchmark.json').read_text())
    system.fast_generation=calibrated['selected_generation']=='compiled_static_cache'
    if system.core:
        from train import tensors_forward
        system.optimized_replay=tensors_forward(system,'B1',True)
    log=Log(out/'items.jsonl');summary=dict(checkpoint=str(checkpoint),active_parameters=active_count(system),seed=seed,split=split,
        protocol='One identical zero-control greedy first answer; strict full terminator; unconstrained decoding; FP32; complete cache rebuild per attempt',
        caveat='Within-seed development/held-out results; held-out text comes from previously inspected V5 validation corpus',likelihood=likelihood(system,split,n=min(192,n_standard)))
    if system.core:
        summary['blends']={str(l):likelihood(system,split,n=32,lam=l) for l in (0,.25,.5,.75,1)}
    if system.adapters:
        summary['prior_likelihood']={str(l):likelihood(system,split,n=96,prior=True,length=l) for l in (1,4)}
    from data import DATA
    excluded=set()
    for file in ('training_graph_hashes.txt','development_graph_hashes.txt'):
        if file=='development_graph_hashes.txt' and split!='test':continue
        if (DATA/file).exists():excluded.update((DATA/file).read_text().splitlines())
    sets={};all_flags={}
    methods=['greedy','retry']+(['prior','gradcuit','gradcuit_kl','shared','random_matched','random'] if system.adapters and full else [])
    if methods_override:methods=list(methods_override)
    summary['methods']=methods;summary['locked_batch_time_budget_s']=budget_per_batch
    for name,n,hard in [('standard',n_standard,False),('hard',n_hard,True)]:
        if seed==0:
            xs=np.load(DATA/f'{split}_{name}_s0.npz')['x'][:n]
            if any(canonical(x) in excluded for x in xs):raise AssertionError('Evaluation overlap with training/development')
            excluded.update(canonical(x) for x in xs)
        else:
            xs,_,_,seen=fixed_graph(n,61000+seed*100+(1000 if split=='test' else 0)+(10 if hard else 0),hard,excluded);excluded|=seen
        np.save(out/f'{name}_instances.npy',xs);sets[name]=dict(n=n,canonical_hashes=[canonical(x) for x in xs]);all_flags[name]={}
        warm_start=time.perf_counter();warm=generate(system,xs[:bs])
        if system.adapters and full:
            variable=torch.zeros(min(bs,n),T//4,DZ,device=DEVICE,requires_grad=True)
            repair_gradient(system,warm,variable,1,.05)
        sync();summary.setdefault('compilation_warmup_s',{})[name]=time.perf_counter()-warm_start
        aggregate={m:dict(flags=[],cost=0.,attempt_curves={str(u):[] for u in (1,2,4,8,11)},failures=Counter(),batch_costs=[]) for m in methods}
        for start in range(0,n,bs):
            # Interleave policy order by batch; no single long run gets all the cool-machine timings.
            order=methods[start//bs%len(methods):]+methods[:start//bs%len(methods)]
            for method in order:
                if deadline is not None and time.time()>deadline:
                    summary['partial_reason']='evaluation wall-clock deadline';atomic_json(out/'summary.json',summary);return summary
                result=search_batch(system,xs[start:start+bs],method,0 if method=='greedy' else 10,seed=seed*100000+start+62100,budget=budget_per_batch)
                agg=aggregate[method];agg['flags'].extend(result['success']);agg['cost']+=result['seconds'];agg['batch_costs'].append(result['seconds'])
                for cap in (1,2,4,8,11):
                    flags=[any(r['success'] for r in rs if r['attempt']<cap) for rs in result['records']]
                    agg['attempt_curves'][str(cap)].extend(flags)
                for i,recs in enumerate(result['records']):
                    if recs:agg['failures'][recs[-1]['reason']]+=1
                    log(set=name,method=method,item=start+i,success=result['success'][i],attempts=recs,
                        batch_seconds=result['seconds'],batch_size=len(result['records']),batch_gradient_trace=result['gradient_trace'] if i==0 else None,
                        generations=result['generations'],backward_passes=result['backward_passes'],probability_replays=result['probability_replays'],numerical_failure=result['numerical_failures'][i])
                print(json.dumps(dict(set=name,method=method,completed=start+len(result['records']),success=float(np.mean(agg['flags'])),seconds=agg['cost'])),flush=True)
            atomic_json(out/'progress.json',dict(set=name,completed=min(start+bs,n),total=n,methods={m:float(np.mean(a['flags'])) if a['flags'] else None for m,a in aggregate.items()}))
        summary[name]={m:dict(success=float(np.mean(a['flags'])),n=len(a['flags']),seconds=a['cost'],seconds_per_problem=a['cost']/len(a['flags']),
                             attempt_curves={u:float(np.mean(f)) for u,f in a['attempt_curves'].items()},failures=dict(a['failures']),flags=a['flags'],
                             batch_median_s=float(np.median(a['batch_costs'])),batch_p95_s=float(np.quantile(a['batch_costs'],.95))) for m,a in aggregate.items()}
        if 'gradcuit_kl' in aggregate:
            summary[name]['paired']={m:bootstrap(aggregate['gradcuit_kl']['flags'],aggregate[m]['flags']) for m in ('retry','random_matched','prior','gradcuit') if m in aggregate}
        atomic_json(out/'summary.json',summary)
    atomic_json(out/'split_manifest.json',sets)
    if system.adapters and full and (deadline is None or time.time()+120<deadline):
        hard=np.load(out/'hard_instances.npy')
        # Fixed prior draws on 64 development/held-out problems measure verified alternatives.
        coverage=[]
        for start in range(0,min(64,len(hard)),bs):
            valid=[set() for _ in range(min(bs,len(hard)-start))];solved=np.zeros(len(valid),bool)
            for draw in range(4):
                bank=ball(torch.randn(len(valid),T//4,DZ,generator=torch.Generator().manual_seed(69000+start+draw))).to(DEVICE)
                gen=generate(system,hard[start:start+bs],bank)
                for i,v in enumerate(gen['verification']):
                    if v['success']:valid[i].add(tuple(tuple(p) for p in v['trajectory']));solved[i]=True
            coverage.extend(dict(item=start+i,solved=bool(solved[i]),distinct_valid=len(v)) for i,v in enumerate(valid))
        summary['prior_coverage']=dict(draws=4,items=coverage,any_solved=float(np.mean([c['solved'] for c in coverage])),additional_valid_fraction=float(np.mean([c['distinct_valid']>1 for c in coverage])))
        if split=='dev':
            failures=[i for i,f in enumerate(summary['hard']['greedy']['flags']) if not f]
            summary['oracle_reachability']=reachability(system,hard[failures],out/'oracle.json')
    if deadline is None or time.time()+60<deadline:
        summary['text_samples']=text_samples(system,split)
    atomic_json(out/'summary.json',summary)
    return summary


@torch.no_grad()
def text_samples(system,split='dev',n=8):
    from data import Text,SOURCE,EOS
    from tokenizers import Tokenizer
    from generation import Incremental,FastIncremental
    stream=Text(split);tok=Tokenizer.from_file(str(SOURCE/'tokenizer.json'))
    ids=[int(i) for i in stream.ids if stream.offsets[i+1]-stream.offsets[i]>=96][:n]
    results=[]
    for i in ids:
        start=int(stream.offsets[i]);prompt=np.asarray(stream.flat[start:start+64],np.int64)
        x=torch.as_tensor(prompt[None],device=DEVICE);z=torch.zeros(1,T//4,DZ,device=DEVICE)
        cache=(FastIncremental if getattr(system,'fast_generation',False) else Incremental)(system,z)
        logits=cache.prefill(x);response=[]
        for pos in range(64,T):
            token=logits.argmax(-1);response.append(int(token))
            if int(token)==EOS or pos==T-1:break
            logits=cache.step(token)
        results.append(dict(document=i,prompt=tok.decode(prompt.tolist(),skip_special_tokens=False),
                            continuation=tok.decode(response,skip_special_tokens=False),generated_tokens=len(response),
                            symbolic_special_fraction=sum(2<=t<33 for t in response)/max(1,len(response))))
    return results


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--split',default='dev');ap.add_argument('--standard',type=int,default=128);ap.add_argument('--hard',type=int,default=256)
    ap.add_argument('--bs',type=int,default=16);ap.add_argument('--quick',action='store_true');ap.add_argument('--budget',type=float);ap.add_argument('--deadline',type=float);ap.add_argument('--methods')
    a=ap.parse_args();torch.set_num_threads(4)
    evaluate_checkpoint(a.checkpoint,a.out,a.seed,a.split,a.standard,a.hard,a.bs,not a.quick,a.budget,a.deadline,a.methods.split(',') if a.methods else None)


if __name__=='__main__':main()
