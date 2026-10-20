"""Resumable time-sliced training for the paired native32 GRAM experiment."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import psutil
import torch

from ..controls import make_model, parameter_ledger
from ..isolation import ROOT, atomic_json, exclusive_job, sha256, require_legacy_idle
from ..train import save
from ..evaluate import evaluation_rng
from .codec import load_codec
from .data import DATA, ShortDataset
from .evaluate import evaluate


def sources():
    paths=list(ROOT.glob('*.py'))+list((ROOT/'shortblock').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}


def initialize(config, seed, device):
    torch.manual_seed(seed)
    return make_model('code_gram',config).to(device)


@exclusive_job
def train(args):
    require_legacy_idle()
    torch.set_num_threads(1)
    if args.device=='mps':torch.mps.set_per_process_memory_fraction(.5)
    data=ShortDataset(args.data)
    config=json.loads(args.config.read_text())
    if list(config['levels'])!=data.manifest['codec']['levels'] or config['context_source']!='codes':
        raise ValueError('Model and native codec contract differ')
    model=initialize(config,args.seed,args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.95),weight_decay=.01,foreach=False)
    rng=np.random.default_rng(args.seed)
    identity=dict(version='native32-paired-v1',task='raw',arm='code_gram',config=asdict(model.config),
        dataset_sha=data.manifest['episodes_sha'],codec_id=data.manifest['codec']['codec_id'],sources=sources(),
        seed=args.seed,batch=args.batch,accumulation=args.accum,lr=args.lr,device=args.device,
        schedule_steps=args.schedule_steps,maximum_steps=args.steps,wall_budget_seconds=args.total_seconds,
        deadline=args.deadline,eval_every=args.eval_every,eval_n=args.eval_n,cpu_threads=1,precision='float32')
    run=args.run
    run.mkdir(parents=True,exist_ok=True)
    checkpoint=run/'latest.pt'
    step=0; prior=0.
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location=args.device,weights_only=True)
        if ck['manifest_identity']!=identity:
            raise ValueError('Resume would change the registered paired experiment')
        model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer'])
        step,prior=ck['step'],ck['elapsed']
        rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'].cpu())
        if ck['mps_rng'] is not None:torch.mps.set_rng_state(ck['mps_rng'].cpu())
        del ck
    atomic_json(run/'manifest.json',dict(**identity,ledger=parameter_ledger(model),objective='terminal_full_ELBO',
        comparison='same corpus, codec, context, target, batch and recurrence; equal wall time with saved equal-step inference milestones'))
    stopped=[False]
    def stop(*_):stopped[0]=True
    previous={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    start=time.monotonic();elapsed=prior;last_save=start;last_eval=start;codec=None;reason='slice_complete'
    best_path=run/'best_validation.json'
    best=json.loads(best_path.read_text())['nll_per_code'] if best_path.exists() else float('inf')
    last_state=run/'last_evaluation.json'
    last_eval_elapsed=json.loads(last_state.read_text()).get('elapsed_seconds',-1e12) if last_state.exists() else -1e12
    def validation():
        nonlocal codec,best,last_eval_elapsed,elapsed
        if codec is None:
            with evaluation_rng(args.seed,args.device):
                codec=load_codec(data.manifest['codec'])
        result=evaluate(model,data,codec,n=args.eval_n)
        elapsed=prior+time.monotonic()-start
        result.update(step=step,elapsed_seconds=elapsed,timestamp=time.time())
        atomic_json(run/f'eval_{step:08d}.json',result)
        atomic_json(last_state,dict(step=step,elapsed_seconds=elapsed))
        last_eval_elapsed=elapsed
        if result['nll_per_code']<best:
            best=result['nll_per_code']
            save(run/'best_nll.pt',model,optimizer,step,rng,data,'code_gram',elapsed,identity)
            atomic_json(best_path,dict(step=step,nll_per_code=best,evaluation=f'eval_{step:08d}.json'))
    try:
        if step==0:validation()
        with (run/'metrics.jsonl').open('a',buffering=1) as log:
            while step<args.steps and elapsed<args.total_seconds and time.time()<args.deadline:
                if stopped[0] or (run/'STOP').exists():reason='stop_requested';break
                if time.monotonic()-start>=args.slice_seconds:break
                memory=psutil.virtual_memory().available
                battery=psutil.sensors_battery()
                if memory<1.5*1024**3:reason='low_memory';break
                if shutil.disk_usage(run).free<3*1024**3:reason='low_disk';break
                if battery and not battery.power_plugged and battery.percent<20:reason='low_battery';break
                if step%50==0:require_legacy_idle()
                tic=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
                value=0.;summaries={}
                fraction=min(1.,step/max(1,args.schedule_steps))
                lr=args.lr*min(1.,(step+1)/100)*(.15+.85*.5*(1+math.cos(math.pi*fraction)))
                for group in optimizer.param_groups:group['lr']=lr
                for _ in range(args.accum):
                    index=rng.choice(data.indices['train'],args.batch,replace=False)
                    b=data.batch(index,args.device)
                    loss,stats=model(b['x'],None,b['y'])
                    (loss/args.accum).backward();value+=float(loss.detach())/args.accum
                    for key in ('nll','kl_sum','kl_per_coordinate','sigma_median','sigma_min','sigma_max'):
                        summaries[key]=summaries.get(key,0.)+float(stats[key])/args.accum
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                if args.device=='mps':torch.mps.synchronize()
                step+=1;elapsed=prior+time.monotonic()-start
                row=dict(step=step,loss=value,lr=lr,grad_norm=float(norm),elapsed_seconds=elapsed,
                    update_seconds=time.perf_counter()-tic,examples_seen=step*args.batch*args.accum,
                    text_target_positions_seen=step*args.batch*args.accum*32,
                    timestamp=time.time(),available_ram=psutil.virtual_memory().available,
                    mps_driver_bytes=torch.mps.driver_allocated_memory() if args.device=='mps' else 0,**summaries)
                row['blocks_per_second']=args.batch*args.accum/row['update_seconds']
                if step<=3 or step%50==0:
                    log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                    atomic_json(run/'status.json',dict(status='training',**row))
                if step==1 or time.monotonic()-last_save>=300:
                    save(checkpoint,model,optimizer,step,rng,data,'code_gram',elapsed,identity)
                    last_save=time.monotonic()
                if step in (1000,5000,10000,25000,50000,100000):
                    payload=dict(version=identity['version'],arm='code_gram',config=asdict(model.config),
                        step=step,dataset_sha=identity['dataset_sha'],manifest_identity=identity,
                        model=model.inference_state_dict(),posterior_included=False)
                    temporary=run/f'equal_step_{step:08d}.tmp'
                    torch.save(payload,temporary);temporary.replace(run/f'equal_step_{step:08d}_inference.pt')
                if elapsed-last_eval_elapsed>=args.eval_every and time.time()<args.deadline-180:
                    validation()
        elapsed=prior+time.monotonic()-start
        save(checkpoint,model,optimizer,step,rng,data,'code_gram',elapsed,identity)
        atomic_json(run/'status.json',dict(status=reason,step=step,elapsed_seconds=elapsed,timestamp=time.time()))
    except BaseException as error:
        atomic_json(run/'status.json',dict(status='error',error=repr(error),step=step,elapsed_seconds=elapsed,timestamp=time.time()))
        raise
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def parser():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--data',type=Path,default=DATA);ap.add_argument('--device',default='mps',choices=['cpu','mps'])
    ap.add_argument('--seed',type=int,default=151803);ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--accum',type=int,default=2);ap.add_argument('--lr',type=float,default=.0003)
    ap.add_argument('--steps',type=int,default=100000000);ap.add_argument('--schedule-steps',type=int,required=True)
    ap.add_argument('--total-seconds',type=float,required=True);ap.add_argument('--slice-seconds',type=float,default=1800)
    ap.add_argument('--deadline',type=float,required=True);ap.add_argument('--eval-every',type=float,default=1800)
    ap.add_argument('--eval-n',type=int,default=16)
    return ap


if __name__=='__main__':train(parser().parse_args())
