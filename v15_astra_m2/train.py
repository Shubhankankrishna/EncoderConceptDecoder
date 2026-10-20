"""Resumable, bounded MPS training. All data preparation finishes before entry."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import psutil
import torch

from .codec import ROOT,FrozenNCMCodec,sha256
from .data import Dataset,atomic_json
from .evaluate import evaluate
from .model import AstraHRM,Config
from .gram import AstraGRAM,make_model

STOP=False


def request_stop(*_):
    global STOP
    STOP=True


def save_checkpoint(path,model,optimizer,step,rng,data,training_seconds,extra=None):
    payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        arm='gram' if isinstance(model,AstraGRAM) else 'hrm',
        optimizer=optimizer.state_dict(),step=step,config=asdict(model.config),
        numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
        mps_rng=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        data_sha=data.manifest['code_sha'],data_manifest_sha=sha256(data.root/'manifest.json'),
        training_seconds=training_seconds,extra=extra or {})
    assert all(torch.isfinite(t).all() for t in payload['model'].values())
    tmp=path.with_suffix('.tmp')
    with open(tmp,'wb') as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    # Readability checked before replacing the previous intact checkpoint.
    check=torch.load(tmp,map_location='cpu',weights_only=True)
    assert check['step']==step and check['data_sha']==payload['data_sha']
    del check
    os.replace(tmp,path)
    if isinstance(model,AstraGRAM):
        exported=dict(model=model.inference_state_dict(),config=asdict(model.config),arm='gram',
                      step=step,data_sha=data.manifest['code_sha'],posterior_included=False,
                      source_checkpoint=path.name)
        dst=path.with_name(path.stem+'_inference.pt');temp=dst.with_suffix('.tmp')
        with open(temp,'wb') as f:torch.save(exported,f);f.flush();os.fsync(f.fileno())
        verify=torch.load(temp,map_location='cpu',weights_only=True)
        assert not any(k.startswith(('posterior.','target_encoder.')) for k in verify['model'])
        os.replace(temp,dst)


def train(args):
    global STOP
    STOP=False
    torch.set_num_threads(2);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    if args.device=='mps':torch.mps.set_per_process_memory_fraction(.6)
    run=Path(args.run);run.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    data=Dataset();data.verify_hashes();rng=np.random.default_rng(args.seed)
    m=make_model(args.arm).to(args.device);c=m.config
    if args.arm=='gram':args.corrupt=0. # beta=1, no corrupt labels: full trajectory NELBO.
    opt=torch.optim.AdamW(m.parameters(),lr=args.lr,betas=(.9,.95),weight_decay=.01,foreach=False)
    checkpoint=run/'latest.pt';step=0;prior_seconds=0.;best=float('inf');best_success=-1.;policy_nll=float('inf')
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location=args.device,weights_only=True)
        assert ck['config']==asdict(c) and ck['data_sha']==data.manifest['code_sha']
        assert ck['data_manifest_sha']==sha256(data.root/'manifest.json')
        m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step=ck['step']
        rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'].cpu())
        if args.device=='mps' and ck['mps_rng'] is not None:torch.mps.set_rng_state(ck['mps_rng'].cpu())
        prior_seconds=ck['training_seconds'];best=ck['extra'].get('best',best);best_success=ck['extra'].get('best_success',best_success)
        policy_nll=ck['extra'].get('policy_nll',policy_nll)
        del ck
    manifest=dict(config=asdict(c),parameters=m.parameter_count(),arm=args.arm,seed=args.seed,
                  trainable_parameters={n:dict(shape=list(p.shape),numel=p.numel(),train_only=n.startswith(('posterior.','target_encoder.'))) for n,p in m.named_parameters()},
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cpu_threads=torch.get_num_threads(),
                  data_manifest_sha=sha256(data.root/'manifest.json'),codec=data.manifest['codec'],
                  optimizer=dict(name='AdamW',lr=args.lr,betas=[.9,.95],weight_decay=.01,clip=1.),
                  batch=args.batch,accumulation=args.accum,precision='float32',device=args.device,
                  objective=args.objective,conditioning_corruption=args.corrupt,deadline=args.deadline,
                  pytorch=str(torch.__version__),pid=os.getpid(),source_files={p.name:sha256(p) for p in ROOT.glob('*.py')},
                  only_stage=2 if args.arm=='gram' else 1,
                  gradient_mode='full_bptt',kl_beta=1 if args.arm=='gram' else None,
                  posterior_train_only=args.arm=='gram',
                  claim='Astra state-conditioned NCM action-policy pilot; not a full eight-stage validation')
    atomic_json(run/'manifest.json',manifest)
    codec=None;start=time.monotonic();last_save=start;last_eval=start;last_log=start;last_health=start
    elapsed_train=prior_seconds;termination='deadline';cpu_start=time.process_time();processed=step*args.batch*args.accum
    if args.steps:target_step=step+args.steps
    else:target_step=None
    log=open(run/'metrics.jsonl','a',buffering=1)
    try:
        while time.time()<args.deadline:
            if STOP or (run/'STOP').exists():termination='stopped';break
            if target_step and step>=target_step:termination='step_limit';break
            tic=time.monotonic();m.train();opt.zero_grad(set_to_none=True)
            # Schedule depends on completed optimizer steps, and is resume-stable.
            warm=min(1.,(step+1)/500)
            frac=min(1.,max(0,step-500)/max(1,args.schedule_steps-500))
            lr=args.lr*warm*(.15+.85*.5*(1+math.cos(math.pi*frac)))
            for g in opt.param_groups:g['lr']=lr
            loss_sum=0.;nll_sum=0.;gram_stats={}
            for _ in range(args.accum):
                idx=rng.choice(data.indices['train'],args.batch,replace=False)
                x,s,y=data.batch(idx,args.device)
                loss,stats=m(x,s,y,args.objective,args.corrupt)
                (loss/args.accum).backward();loss_sum+=float(loss.detach())/args.accum;nll_sum+=float(stats['nll'])/args.accum
                for k in ('kl_sum','kl_per_coordinate','sigma_median','sigma_min','sigma_max'):
                    if k in stats:gram_stats[k]=gram_stats.get(k,0.)+float(stats[k])/args.accum
            norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            if args.device=='mps':torch.mps.synchronize()
            duration=time.monotonic()-tic;elapsed_train+=duration;step+=1;processed+=args.batch*args.accum
            now=time.monotonic()
            if step<=3 or now-last_log>=20 or (target_step and step==target_step):
                row=dict(step=step,loss=loss_sum,nll_per_code=nll_sum,grad_norm=float(norm),lr=lr,
                         optimizer_seconds=duration,blocks_per_second=args.batch*args.accum/duration,
                         blocks_seen=processed,training_seconds=elapsed_train,wall_seconds=now-start,
                         cpu_seconds=time.process_time()-cpu_start,rss_bytes=psutil.Process().memory_info().rss,
                         available_ram=psutil.virtual_memory().available,timestamp=time.time(),
                         mps_driver_bytes=torch.mps.driver_allocated_memory() if args.device=='mps' else 0)
                row.update(gram_stats)
                row['nll_kind']='posterior_conditional' if args.arm=='gram' else 'conditional'
                log.write(json.dumps(row)+'\n');atomic_json(run/'status.json',dict(status='training',**row));print(json.dumps(row),flush=True);last_log=now
            if now-last_health>=60:
                if psutil.virtual_memory().available<900*1024**2:
                    termination='low_memory';break
                if __import__('shutil').disk_usage(run).free<2*1024**3:
                    termination='low_disk';break
                last_health=now
            if now-last_save>=args.save_every:
                save_checkpoint(checkpoint,m,opt,step,rng,data,elapsed_train,dict(best=best,best_success=best_success,policy_nll=policy_nll));last_save=time.monotonic()
            if args.eval_every and now-last_eval>=args.eval_every and time.time()<args.deadline-120:
                codec=codec or FrozenNCMCodec('cpu',family='lm1b')
                ev=evaluate(m,data,codec,'valid',n=args.eval_n,batch=4,closed_loop=False)
                ev.update(step=step,training_seconds=elapsed_train)
                atomic_json(run/f'eval_{step:07d}.json',ev);log.write(json.dumps(dict(event='evaluation',**ev))+'\n')
                if ev['nll_per_code']<best:
                    best=ev['nll_per_code'];save_checkpoint(run/'best_nll.pt',m,opt,step,rng,data,elapsed_train,dict(best=best,best_success=best_success))
                if (ev['optimal_action_rate'],-ev['nll_per_code'])>(best_success,-policy_nll):
                    best_success=ev['optimal_action_rate'];save_checkpoint(run/'best_policy.pt',m,opt,step,rng,data,elapsed_train,dict(best=best,best_success=best_success))
                    policy_nll=ev['nll_per_code']
                last_eval=time.monotonic()
    except Exception as e:
        termination='error';atomic_json(run/'error.json',dict(error=repr(e),step=step,timestamp=time.time()))
        raise
    finally:
        # A failed numerical step may have nonfinite gradients; parameters before
        # optimizer.step remain valid. Latest successful checkpoint is retained
        # on an error instead of overwriting it with uncertain state.
        if termination!='error':
            save_checkpoint(checkpoint,m,opt,step,rng,data,elapsed_train,dict(best=best,best_success=best_success,policy_nll=policy_nll))
        atomic_json(run/'status.json',dict(status=termination,step=step,training_seconds=elapsed_train,
                                         timestamp=time.time(),pid=os.getpid(),checkpoint=str(checkpoint)))
        log.close()
    return 75 if termination in ('low_memory','low_disk') else 0


def parser():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--arm',choices=['gram','hrm','onepass'],default='gram')
    ap.add_argument('--device',default='mps');ap.add_argument('--seed',type=int,default=1518);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--accum',type=int,default=2)
    ap.add_argument('--lr',type=float,default=.0003);ap.add_argument('--deadline',type=float,default=time.time()+48*3600)
    ap.add_argument('--steps',type=int,default=0);ap.add_argument('--schedule-steps',type=int,default=130000)
    ap.add_argument('--objective',choices=['per_code','level_balanced'],default='per_code');ap.add_argument('--corrupt',type=float,default=0.)
    ap.add_argument('--save-every',type=float,default=600);ap.add_argument('--eval-every',type=float,default=1800);ap.add_argument('--eval-n',type=int,default=128)
    return ap


if __name__=='__main__':raise SystemExit(train(parser().parse_args()))
