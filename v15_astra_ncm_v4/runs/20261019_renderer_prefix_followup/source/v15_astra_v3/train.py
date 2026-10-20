"""Guarded next-run training entry point. Never joins or resumes the live v2 run."""
import argparse
from dataclasses import asdict
import json
import math
import os
import signal
import shutil
from pathlib import Path
import time
import numpy as np
import torch
from . import VERSION
from .controls import ARMS,StateActionPolicy,make_model,parameter_ledger,copy_common_initialization
from .gram import AstraGRAM
from .model import Config
from .dataset import Dataset
from .representation import require_representation_contract
from .isolation import ROOT,output_path,atomic_json,sha256,require_legacy_idle,exclusive_job


def model_loss(model,batch):
    if isinstance(model,StateActionPolicy):return model(batch['state'],batch['action']),{}
    return model(batch['x'],batch.get('state'),batch['y'])


def source_identity():return {p.name:sha256(p) for p in ROOT.glob('*.py')}


def save(path,model,opt,step,rng,data,arm,elapsed,manifest_identity):
    path=output_path(path)
    state=dict(version=VERSION,arm=arm,config=asdict(model.config),model=model.state_dict(),
               optimizer=opt.state_dict(),step=step,numpy_rng=rng.bit_generator.state,
               torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if next(model.parameters()).device.type=='mps' else None,
               dataset_sha=data.manifest['episodes_sha'],elapsed=elapsed,manifest_identity=manifest_identity,
               posterior_included=isinstance(model,AstraGRAM))
    if not all(torch.isfinite(t).all() for t in state['model'].values()):raise ValueError('Nonfinite checkpoint rejected')
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    check=torch.load(tmp,map_location='cpu',weights_only=True)
    if check['step']!=step:raise ValueError('Checkpoint verification failed')
    del check;os.replace(tmp,path)
    if isinstance(model,AstraGRAM):
        payload={k:state[k] for k in ('version','arm','config','step','dataset_sha','manifest_identity')}
        payload.update(model=model.inference_state_dict(),posterior_included=False)
        target=path.with_name(path.stem+'_inference.pt');tmp=target.with_suffix('.tmp')
        with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,target)


@exclusive_job
def train(args):
    # This check happens before data loading, model allocation, or output writes.
    require_legacy_idle()
    import psutil
    if args.steps<1 or args.seconds<=0:raise ValueError('Positive step and wall-time limits are mandatory')
    if min(args.batch,args.accum,args.cpu_threads)<1 or args.lr<=0 or args.save_every<=0:raise ValueError('Positive optimizer/resource settings required')
    if args.eval_every<0 or args.eval_n<1:raise ValueError('Invalid evaluation settings')
    if args.deadline is not None and (not math.isfinite(args.deadline) or time.time()>=args.deadline):raise ValueError('Run deadline must be finite and in the future')
    if args.task!='raw' and args.eval_every:raise ValueError('Periodic integrated evaluation currently supports raw P1; use the separate grounded evaluator')
    run=output_path(args.run)
    if args.task=='raw':
        from .raw import RawCodeDataset
        data=RawCodeDataset(args.data)
        if args.arm=='state_action':raise ValueError('Raw P1 has no action or state supervision')
    else:data=Dataset(args.data)
    if not len(data.indices['train']):raise ValueError('Empty training split')
    config=json.loads(args.config.read_text()) if args.config else {}
    if args.task=='raw':config['context_source']='codes'
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(args.seed)
    model=make_model(args.arm,config)
    if args.task!='raw' and not isinstance(model,StateActionPolicy):
        if data.encoding is None:raise ValueError('Encode the complete observation dataset first')
        require_representation_contract(context_source=model.config.context_source,
            complete_observation=data.encoding['complete_observation'],codec_roundtrip_passed=data.encoding['code_only_accepted'])
    if not isinstance(model,StateActionPolicy):
        codec=data.manifest['codec'] if args.task=='raw' else data.encoding['codec']
        if tuple(model.config.levels)!=tuple(codec['levels']) or codec['vocab_sizes']!=[model.config.vocab]*len(model.config.levels):
            raise ValueError('Model hierarchy must match the cached codec')
    # Common core weights can be explicitly paired across deterministic/stochastic controls.
    paired_names=[]
    if args.arm=='code_gram':
        # Reset seed so reference is exactly the deterministic arm's initialization.
        torch.manual_seed(args.seed)
        reference=make_model('code_hrm',{k:v for k,v in asdict(model.config).items() if k in Config.__dataclass_fields__})
        paired_names=copy_common_initialization(reference,model);del reference
    if args.device=='mps':torch.mps.set_per_process_memory_fraction(.5)
    model.to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.95),weight_decay=.01,foreach=False)
    rng=np.random.default_rng(args.seed)
    schedule_steps=args.schedule_steps or args.steps
    identity=dict(version=VERSION,task=args.task,arm=args.arm,config=asdict(model.config),dataset_sha=data.manifest['episodes_sha'],
                  codec_id=(data.manifest['codec']['codec_id'] if args.task=='raw' else data.encoding['codec']['codec_id'] if data.encoding else None),
                  encoding_manifest_sha=sha256(data.root/'encoding_manifest.json') if data.encoding else None,
                  sources=source_identity(),seed=args.seed,batch=args.batch,accumulation=args.accum,lr=args.lr,
                  device=args.device,schedule_steps=schedule_steps,maximum_steps=args.steps,wall_budget_seconds=args.seconds,
                  deadline=args.deadline,eval_every=args.eval_every,eval_n=args.eval_n,
                  cpu_threads=args.cpu_threads,precision='float32')
    run.mkdir(parents=True,exist_ok=True);checkpoint=run/'latest.pt';step=0;elapsed=0.
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location=args.device,weights_only=True)
        if ck['manifest_identity']!=identity:raise ValueError('Resume would change the registered experiment')
        model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);step=ck['step'];elapsed=ck['elapsed']
        rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'].cpu())
        if ck['mps_rng'] is not None:torch.mps.set_rng_state(ck['mps_rng'].cpu())
    atomic_json(run/'manifest.json',dict(**identity,ledger=parameter_ledger(model),paired_parameter_names=paired_names,
                objective='action_CE' if isinstance(model,StateActionPolicy) else 'terminal_full_ELBO' if isinstance(model,AstraGRAM) else 'conditional_code_NLL',
                selection='raw validation NLL and fixed-budget latest checkpoint' if args.eval_every else 'fixed budget checkpoint; test split is never used for training or checkpoint selection'))
    start=time.monotonic();prior_elapsed=elapsed;last_save=start;last_eval=start;reason='budget_complete';codec=None
    best_path=run/'best_validation.json';best_nll=json.loads(best_path.read_text())['nll_per_code'] if best_path.exists() else float('inf')
    stopping=[False]
    def stop_requested(signum,frame):stopping[0]=True
    old_signals={sig:signal.signal(sig,stop_requested) for sig in (signal.SIGINT,signal.SIGTERM)}
    with (run/'metrics.jsonl').open('a',buffering=1) as log:
        try:
            while step<args.steps and elapsed<args.seconds and (args.deadline is None or time.time()<args.deadline):
                if stopping[0] or (run/'STOP').exists():reason='stop_requested';break
                if step%50==0:require_legacy_idle()
                if psutil.virtual_memory().available<1200*1024**2:reason='low_memory';break
                if shutil.disk_usage(run).free<2*1024**3:reason='low_disk';break
                battery=psutil.sensors_battery()
                if battery and not battery.power_plugged and battery.percent<20:reason='low_battery';break
                tic=time.perf_counter();model.train();opt.zero_grad(set_to_none=True);loss_value=0.;summaries={}
                warm=min(1.,(step+1)/100);fraction=min(1.,step/max(1,schedule_steps))
                lr=args.lr*warm*(.15+.85*.5*(1+math.cos(math.pi*fraction)))
                for g in opt.param_groups:g['lr']=lr
                for _ in range(args.accum):
                    pool=data.indices['train'];ix=rng.choice(pool,min(args.batch,len(pool)),replace=False)
                    b=data.batch(ix,args.device,require_codes=not isinstance(model,StateActionPolicy))
                    loss,stats=model_loss(model,b);(loss/args.accum).backward();loss_value+=float(loss.detach())/args.accum
                    for key in ('nll','kl_sum','kl_per_coordinate','sigma_median','sigma_min','sigma_max'):
                        if key in stats:summaries[key]=summaries.get(key,0.)+float(stats[key])/args.accum
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                if args.device=='mps':torch.mps.synchronize()
                step+=1;elapsed=prior_elapsed+time.monotonic()-start
                row=dict(step=step,loss=loss_value,lr=lr,grad_norm=float(norm),elapsed_seconds=elapsed,
                         update_seconds=time.perf_counter()-tic,examples_seen=step*min(args.batch,len(pool))*args.accum,
                         timestamp=time.time(),available_ram=psutil.virtual_memory().available,
                         mps_driver_bytes=torch.mps.driver_allocated_memory() if args.device=='mps' else 0,**summaries)
                row['blocks_per_second']=min(args.batch,len(pool))*args.accum/row['update_seconds']
                if step<=3 or step%50==0:
                    log.write(json.dumps(row)+'\n');atomic_json(run/'status.json',dict(status='training',**row));print(json.dumps(row),flush=True)
                if step==1 or time.monotonic()-last_save>=args.save_every:
                    save(checkpoint,model,opt,step,rng,data,args.arm,elapsed,identity);last_save=time.monotonic()
                if args.eval_every and (step==1 or time.monotonic()-last_eval>=args.eval_every) and (args.deadline is None or time.time()<args.deadline-180):
                    from .codec import FrozenNCMCodec
                    from .evaluate_raw import evaluate_raw
                    codec=codec or FrozenNCMCodec('cpu',family='lm1b')
                    evaluation=evaluate_raw(model,data,codec,n=args.eval_n,seed=args.seed,batch=2)
                    evaluation.update(step=step,elapsed_seconds=prior_elapsed+time.monotonic()-start,timestamp=time.time())
                    atomic_json(run/f'eval_{step:08d}.json',evaluation)
                    log.write(json.dumps(dict(event='evaluation',step=step,nll_per_code=evaluation['nll_per_code'],seconds=evaluation['seconds']))+'\n')
                    if evaluation['nll_per_code']<best_nll:
                        best_nll=evaluation['nll_per_code'];save(run/'best_nll.pt',model,opt,step,rng,data,args.arm,evaluation['elapsed_seconds'],identity)
                        atomic_json(best_path,dict(step=step,nll_per_code=best_nll,evaluation=f'eval_{step:08d}.json'))
                    last_eval=time.monotonic();elapsed=prior_elapsed+time.monotonic()-start
        except BaseException:
            atomic_json(run/'status.json',dict(status='error',step=step,elapsed_seconds=elapsed))
            raise
        else:
            elapsed=prior_elapsed+time.monotonic()-start
            save(checkpoint,model,opt,step,rng,data,args.arm,elapsed,identity)
            atomic_json(run/'status.json',dict(status=reason,step=step,elapsed_seconds=elapsed))
        finally:
            for sig,handler in old_signals.items():signal.signal(sig,handler)


def parser():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--config',type=Path)
    ap.add_argument('--task',choices=['grounded','raw'],default='grounded')
    ap.add_argument('--steps',type=int,required=True);ap.add_argument('--seconds',type=float,required=True)
    ap.add_argument('--schedule-steps',type=int);ap.add_argument('--deadline',type=float)
    ap.add_argument('--device',choices=['cpu','mps'],default='cpu');ap.add_argument('--seed',type=int,default=151803)
    ap.add_argument('--batch',type=int,default=4);ap.add_argument('--accum',type=int,default=2)
    ap.add_argument('--lr',type=float,default=.0003);ap.add_argument('--cpu-threads',type=int,default=1)
    ap.add_argument('--save-every',type=float,default=600)
    ap.add_argument('--eval-every',type=float,default=0);ap.add_argument('--eval-n',type=int,default=64)
    return ap


if __name__=='__main__':train(parser().parse_args())
