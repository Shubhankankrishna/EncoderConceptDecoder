"""Bounded development tuning and locked equal-time test comparisons."""
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from model import *
from data import *
from runtime import *
from generation import generate
from search import search_one
from evaluate import paired,wilson

def run(source,flat_source,out,seconds=3600):
    torch.set_num_threads(4);s,_=load_system(source);freeze(s);s.fast_generation=not isinstance(s.core,Core);flat,_=load_system(flat_source);freeze(flat);flat.fast_generation=True;out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();dev=np.load(DATA/'dev_hard10.npz');test=np.load(DATA/'test_hard10.npz');log=Log(out/'attempts.jsonl')
    # Synchronized per-request calibration, identical absolute deadline for both architectures.
    times=[]
    for i in range(8):
        p=int(dev['prompt_len'][i]);prompt=dev['x'][i:i+1,:p];generate(flat,prompt,max_new=68)
        before=time.monotonic()
        for draw in range(16):generate(flat,prompt,temperature=0. if draw==0 else .8,seed=8100+i*20+draw,max_new=68,force_complete=False)
        times.append(time.monotonic()-before)
    budget=float(np.median(times));tuning=[]
    for temp,topk in itertools.product((.6,.8,1.),(32,128)):
        hits=[]
        for i in range(8):
            p=int(dev['prompt_len'][i]);r=search_one(s,dev['x'][i,:p],'retry',budget,64,temperature=temp,top_k=topk,seed=8100+i);hits.append(r['success'])
        tuning.append(dict(kind='retry',temperature=temp,top_k=topk,successes=sum(hits)))
    chosen=max(tuning,key=lambda v:v['successes']);gradtune=[]
    for lr,kl in itertools.product((.01,.03,.1),(0.,.05)):
        hits=[]
        for i in range(8):
            p=int(dev['prompt_len'][i]);r=search_one(s,dev['x'][i,:p],'gradcuit_kl',budget,64,lr=lr,kl=kl,seed=8100+i);hits.append(r['success'])
        gradtune.append(dict(lr=lr,kl=kl,successes=sum(hits)))
    grad=max(gradtune,key=lambda v:v['successes']);lock=dict(budget_s=budget,calibration_zero_plus_15_retry_s=times,retry=chosen,gradient=grad,tuning=tuning,gradient_tuning=gradtune,candidate_limit=64,depth=16,feedback=['binary','graded'],training_seeds=1,search_rng_repeats_are_not_training_seeds=True)
    atomic_json(out/'LOCK.json',lock)
    methods=['retry','latent','combined','gradcuit_kl','hybrid_random','ab_prior','ab_hybrid','ab_hybrid_randomdir'];results={};n_done=0
    for i in range(128):
        if time.monotonic()-start>seconds:break
        p=int(test['prompt_len'][i]);prompt=test['x'][i,:p];order=np.random.RandomState(8150+i).permutation(len(methods))
        # Stop before a new paired block if there is insufficient reserved time for all methods.
        if seconds-(time.monotonic()-start)<len(methods)*budget*2:break
        for j in order:
            method=methods[j];r=search_one(s,prompt,method,budget,64,lr=grad['lr'],kl=grad['kl'],temperature=chosen['temperature'],top_k=chosen['top_k'],seed=8100+i,feedback='graded');log(problem=i,method=method,feedback='graded',**r);results.setdefault(method,[]).append(bool(r['success']))
        n_done+=1
    # Index all methods consistently: each list is append-once per completed problem.
    comparison={method:dict(n=len(v),successes=sum(v),rate=float(np.mean(v)) if v else None,interval=wilson(sum(v),len(v)),paired_vs_retry=paired(v,results['retry'],alpha=.05/max(1,len(methods)-1)) if method!='retry' and v else None) for method,v in results.items()}
    summary=dict(lock=lock,status='completed' if n_done==128 else 'partial_time_budget',paired_test_problems=n_done,comparisons=comparison,wall_s=time.monotonic()-start,limitations=['one training seed pilot','binary-only and 1x/2x/4x curves deferred unless separately budgeted','same-kernel nonadaptive allocation and equal-norm direction control included; no quality-adaptivity claim from binary-only success'])
    atomic_json(out/'summary.json',summary);return summary
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--flat-source',required=True);ap.add_argument('--out',required=True);ap.add_argument('--seconds',type=float,default=3600);a=ap.parse_args();run(a.source,a.flat_source,a.out,a.seconds)
