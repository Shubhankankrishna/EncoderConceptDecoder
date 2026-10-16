"""Measure throughput and reject optimizations that change loss/gradients."""
import argparse
import copy
import gc
import time
from pathlib import Path
import numpy as np
import torch
from model import Foundation,Core,Adapters,System,freeze,chunk_info,active_count
from data import Mixer,gen_instance
from train import tensors_forward,do_update
from runtime import DEVICE,ROOT,atomic_json,atomic_save,optimizer,sync,seed_all
from generation import generate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=20);a=ap.parse_args();torch.set_num_threads(4);seed_all(606)
    out=ROOT/'runs'/'calibration';out.mkdir(parents=True,exist_ok=True);results=dict(device=DEVICE,precision='float32',batch=64,training={},generation={})
    foundation=Foundation().to(DEVICE);mix=Mixer(6000);batch=mix.batch(DEVICE)
    with torch.no_grad():c=foundation.pool(foundation.encode(batch['x'],batch['docs']))
    foundation.restart(c,chunk_info(batch['docs'])[1],generator=torch.Generator().manual_seed(6))
    for stage in ('A','B0','B1'):
        if stage=='A':s=System(foundation)
        elif stage=='B0':freeze(foundation);s=System(foundation,Core().to(DEVICE))
        else:freeze(s.core);s.adapters=Adapters().to(DEVICE)
        opt=optimizer(s.parameters(),0,wd=0)
        eager=tensors_forward(s,stage);compiled=tensors_forward(s,stage,True)
        grads=[];losses=[];gns=[]
        for fn in (eager,compiled):
            value,gn,logs,_=do_update(s,opt,stage,batch,fn,torch.Generator().manual_seed(60100),0,10000)
            losses.append(value);gns.append(gn);grads.append(torch.cat([p.grad.detach().cpu().flatten().clone() for p in s.parameters() if p.grad is not None]))
        relative=float((grads[0]-grads[1]).norm()/grads[0].norm().clamp_min(1e-12))
        if abs(losses[0]-losses[1])>1e-4 or relative>.002:raise AssertionError((stage,losses,relative))
        for group in opt.param_groups:group['lr']=.001 if stage=='B1' else .002
        durations=[];generator=torch.Generator().manual_seed(60200)
        for i in range(a.steps+3):
            sync();start=time.perf_counter();b=mix.batch(DEVICE)
            do_update(s,opt,stage,b,compiled,generator,i,10000);sync();elapsed=time.perf_counter()-start
            if i>=3:durations.append(elapsed)
        median=float(np.median(durations));memory=dict(allocated=torch.mps.current_allocated_memory(),driver=torch.mps.driver_allocated_memory()) if DEVICE=='mps' else {}
        results['training'][stage]=dict(median_step_s=median,p90_step_s=float(np.quantile(durations,.9)),input_slots_per_s=12288/median,
                                       gradient_relative_error=relative,eager_compiled_losses=losses,gradient_norms=gns,memory_bytes=memory,compiled=True)
        if stage=='B1':
            # Retained graphs avoid the additional no-grad scoring forwards when memory permits.
            replay=[]
            for i in range(8):
                sync();start=time.perf_counter();b=mix.batch(DEVICE)
                do_update(s,opt,stage,b,compiled,generator,i,10000,replay=True);sync()
                if i>=2:replay.append(time.perf_counter()-start)
            results['training'][stage]['score_then_replay_step_s']=float(np.median(replay))
            results['training'][stage]['candidate_execution']='vectorized_two_candidate_batch; one retained graph; encoder features computed once'
        atomic_json(out/'benchmark.json',results);print(stage,results['training'][stage],flush=True)
    freeze(s);headers=np.stack([gen_instance(np.random.RandomState(700+i))[0] for i in range(8)])
    reference=None
    for mode in ('eager_cache','compiled_static_cache'):
        s.fast_generation=mode=='compiled_static_cache'
        generate(s,headers,seed=1)
        runs=[generate(s,headers,seed=1) for _ in range(2)]
        if reference is None:reference=runs[0]['x']
        disagreements=int((reference!=runs[0]['x']).sum())
        seconds=float(np.median([r['seconds'] for r in runs]));tokens=int(runs[0]['generated_tokens'])
        results['generation'][mode]=dict(batch=8,median_s=seconds,generated_tokens_per_s=tokens/seconds,
                                        greedy_token_disagreements=disagreements,cache_kv_elements=15*8*192*128)
        print(mode,results['generation'][mode],flush=True)
    valid=[(v['median_s'],k) for k,v in results['generation'].items() if v['greedy_token_disagreements']==0]
    results['selected_generation']=min(valid)[1]
    results['selected_training']='compiled_fp32_retained_candidates'
    results['not_claimed']='Short warmed timings; long-run thermal throughput is logged separately'
    atomic_json(out/'benchmark.json',results)


if __name__=='__main__':main()
