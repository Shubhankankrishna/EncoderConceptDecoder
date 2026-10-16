"""Correctness-checked warm throughput calibration on the actual device."""
import argparse,gc,json,time
import numpy as np
import torch
from model import Config,System,freeze
from data import Mixer,ROOT,T,gen_symbolic
from runtime import DEVICE,optimizer,sync,atomic_json,seed_all
from runtime import controls,candidate_backward
from train import configure,forward_function,train_loss,update
from generation import generate

def bench(stage,batch,compiled=True,flat=False):
    seed_all(702);cfg=Config(kind='flat' if flat else 'hierarchy');s=System(cfg).to(DEVICE);b=Mixer(701,batch=batch).batch(DEVICE,single_text=stage=='B1')
    if not flat:
        with torch.no_grad():ctx=s.context(b['x'],b['docs']);s.foundation.restart(ctx[1],ctx[-1],generator=torch.Generator().manual_seed(2))
    if stage=='B0':s.make_refit()
    if stage=='B1':s.make_refit();s.make_adapters()
    opt=optimizer(s,.001);g=torch.Generator().manual_seed(703);fn=forward_function(s,stage,compiled);eager=forward_function(s,stage)
    def vector():return torch.cat([p.grad.flatten() for p in s.parameters() if p.grad is not None])
    if stage!='B1':
        s.zero_grad(set_to_none=True);la,*_=train_loss(s,stage,b,eager);la.backward();ga=vector().detach().clone()
        s.zero_grad(set_to_none=True);lb,*_=train_loss(s,stage,b,fn);lb.backward();gb=vector().detach();relative=float((ga-gb).norm()/ga.norm().clamp_min(1e-12))
        if relative>.003:raise RuntimeError(f'Compiled gradient mismatch {relative}')
        loss_error=abs(float(la)-float(lb));del ga,gb,la,lb
    else:
        b['dz']=s.cfg.dz;z,mask,*_=controls(b,torch.Generator().manual_seed(977))
        s.zero_grad(set_to_none=True);candidate_backward(s,b,[z,-z],mask,.25,eager);ga=vector().detach().clone()
        s.zero_grad(set_to_none=True);candidate_backward(s,b,[z,-z],mask,.25,fn);gb=vector().detach();relative=float((ga-gb).norm()/ga.norm().clamp_min(1e-12));loss_error=None
        if relative>.003:raise RuntimeError(f'Compiled adapter gradient mismatch {relative}')
        del ga,gb
    timings=[]
    for i in range(14):
        sync();start=time.perf_counter();update(s,opt,stage,b,fn,g,.4);sync();elapsed=time.perf_counter()-start
        if i>=4:timings.append(elapsed)
    result=dict(stage=stage,batch=batch,compiled=compiled,median_step_s=float(np.median(timings)),p90_step_s=float(np.percentile(timings,90)),
        input_slots_per_s=batch*T/float(np.median(timings)),scored_targets_per_s=float(b['m'].sum())/float(np.median(timings)),targets_per_step=int(b['m'].sum()),
        eager_compiled_gradient_relative_error=relative,eager_compiled_loss_error=loss_error,active_parameters=s.active_count(),
        memory=dict(allocated=torch.mps.current_allocated_memory(),driver=torch.mps.driver_allocated_memory()) if DEVICE=='mps' else {})
    del s,opt,fn,eager,b;gc.collect()
    if DEVICE=='mps':torch.mps.empty_cache()
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'runs/calibration/benchmark.json'));args=ap.parse_args();torch.set_num_threads(4);rows=[]
    for batch in (8,16,32):
        r=bench('A',batch);rows.append(r);print(json.dumps(r),flush=True);atomic_json(args.out,dict(device=DEVICE,precision='FP32',training=rows,status='calibrating'))
    best=max(rows,key=lambda r:r['scored_targets_per_s']);batch=best['batch']
    for stage in ('B0','B1'):
        r=bench(stage,batch if stage=='B0' else max(8,batch//2));rows.append(r);print(json.dumps(r),flush=True)
    r=bench('A',batch,flat=True);r['stage']='flat';rows.append(r);print(json.dumps(r),flush=True)
    # Compare eager against compiled on the winning batch rather than assuming compilation helps.
    eager=bench('A',batch,compiled=False);rows.append(eager);print(json.dumps(eager),flush=True)
    s=System().to(DEVICE);s.make_refit();s.make_adapters();freeze(s);s.eval();x,_,_,meta=gen_symbolic(np.random.RandomState(1),10);p=meta['prompt_len'];gen=[]
    reference=generate(s,x[None,:p],max_new=20)
    for fast in (False,True):
        s.fast_generation=fast;times=[]
        for i in range(4):
            r=generate(s,x[None,:p],max_new=20)
            if not np.array_equal(reference['x'],r['x']):raise RuntimeError('Static generation disagrees with reference')
            if i:times.append(r['seconds'])
        gen.append(dict(compiled_static=fast,median_s=float(np.median(times)),tokens_per_s=20/float(np.median(times))))
        print(json.dumps(gen[-1]),flush=True)
    selected_compiled=best['median_step_s']<eager['median_step_s'];selected_gen=max(gen,key=lambda r:r['tokens_per_s'])['compiled_static']
    atomic_json(args.out,dict(device=DEVICE,precision='FP32',training=rows,generation=gen,batch=batch,compiled=selected_compiled,compiled_static_generation=selected_gen,status='completed',
        scope='Warmed calibration; sustained thermal throughput and complete elapsed costs are recorded during training.'))

if __name__=='__main__':main()
