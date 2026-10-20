"""Resumable bounded structured training; no codec or generator in this process."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import signal
import time

import torch

from .common import (ROOT,atomic_json,append_json,heavy_job,resource_issue,rng_state,
                     restore_rng,save_torch,seed_all,sha,source_hashes)
from .evaluate import evaluate
from .graphs import batch_states,load_dataset
from .model import Agent,AgentConfig,objective

STOP=False


def stop(*_):
    global STOP
    STOP=True


def sample_batch(graphs,size,strategies,device):
    items=[]
    for _ in range(size):
        g=random.choice(graphs)
        items.append((g,random.randrange(g['n']),random.randrange(2) if strategies else 0))
    return batch_states(items,device)


def checkpoint(model,opt,step,identity,best,elapsed):
    return {'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,
            'identity':identity,'best':best,'elapsed':elapsed,'rng':rng_state()}


def main(args):
    global STOP
    STOP=False
    torch.set_num_threads(2)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    args.run.mkdir(parents=True,exist_ok=True)
    cfg=AgentConfig(steps=args.recurrence,mode=args.mode,strategies=args.strategies)
    data=load_dataset()
    training=[g for g in data['graphs'] if g['split']=='train']
    validation=[g for g in data['graphs'] if g['split']=='valid'][:64]
    identity={'config':asdict(cfg),'seed':args.seed,'batch':args.batch,'steps':args.steps,
              'dataset_sha':sha(ROOT/'data/graphs.json'),'sources':source_hashes(),
              'lr':3e-4,'device':args.device,'deadline':args.deadline}
    atomic_json(args.run/'process.json',{'pid':os.getpid(),'started':time.time()})
    seed_all(args.seed)
    model=Agent(cfg).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    step=0;best=-1.;elapsed=0.
    if (args.run/'latest.pt').exists():
        saved=torch.load(args.run/'latest.pt',map_location=args.device,weights_only=True)
        if saved['identity']!=identity:raise ValueError('Resume identity changed')
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
        step,best,elapsed=saved['step'],saved['best'],saved['elapsed']
        # RNG byte tensors must be on CPU even if model tensors load to MPS.
        saved['rng']['torch']=saved['rng']['torch'].cpu()
        if saved['rng']['mps'] is not None:saved['rng']['mps']=saved['rng']['mps'].cpu()
        restore_rng(saved['rng'])
    atomic_json(args.run/'plan.json',{**identity,'trainable_parameters':sum(p.numel() for p in model.parameters()),
        'unique_recurrent_parameters':sum(p.numel() for p in model.cells.parameters())})
    began=time.monotonic();last_eval=step;last_save=time.monotonic();last_metrics={}

    def status(reason):
        atomic_json(args.run/'status.json',{'status':reason,'step':step,'elapsed_seconds':elapsed+time.monotonic()-began,
            'timestamp':time.time(),'last_train':last_metrics,'target_steps':args.steps,'pid':os.getpid()})

    def save():
        save_torch(args.run/'latest.pt',checkpoint(model,opt,step,identity,best,elapsed+time.monotonic()-began))

    def validate(final=False):
        nonlocal best
        reports={str(z):evaluate(model,validation,args.device,z,search=final) for z in ([0,1] if cfg.strategies else [0])}
        score=sum(r['rollouts']['greedy']['success_rate'] for r in reports.values())/len(reports)
        result={'step':step,'timestamp':time.time(),'strategies':reports,'selection_greedy_success':score}
        atomic_json(args.run/f'eval_{step:07d}.json',result)
        atomic_json(args.run/'latest_eval.json',result)
        if score>best:
            best=score
            save_torch(args.run/'best.pt',{'model':model.state_dict(),'config':asdict(cfg),'step':step,'score':best,'identity':identity})
        return result

    status('initial_evaluation' if step==0 else 'resuming')
    if step==0:validate()
    reason='complete'
    while step<args.steps:
        if STOP or (args.run.parent/'STOP').exists():reason='stopped';break
        if time.time()>=args.deadline:reason='deadline';break
        if step%25==0:
            issue=resource_issue()
            if issue:reason=issue;break
        model.train()
        batch=sample_batch(training,args.batch,cfg.strategies,args.device)
        opt.zero_grad(set_to_none=True)
        out=model(batch);loss,metrics=objective(out,batch)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite structured loss')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();step+=1
        if step%25==0:
            last_metrics={**metrics,'grad_norm':float(grad),'residual_norms':out['residual_norms']}
            append_json(args.run/'metrics.jsonl',{'step':step,'timestamp':time.time(),**last_metrics})
            status('training')
        if step-last_eval>=args.eval_every:
            validate();last_eval=step;save();last_save=time.monotonic()
        elif time.monotonic()-last_save>=120:
            save();last_save=time.monotonic()
    save()
    if reason=='complete':validate(final=True);save()
    status(reason)
    print(json.dumps({'status':reason,'step':step,'best_validation_success':best}),flush=True)
    if reason in ('low_memory','low_disk','low_battery'):raise RuntimeError(reason)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--recurrence',type=int,default=2)
    ap.add_argument('--mode',choices=['shared','untied','untied_compute'],default='shared')
    ap.add_argument('--strategies',action='store_true')
    ap.add_argument('--steps',type=int,default=4000)
    ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--eval-every',type=int,default=500)
    ap.add_argument('--deadline',type=float,required=True)
    ap.add_argument('--device',choices=['cpu','mps'],default='mps')
    args=ap.parse_args()
    try:
        with heavy_job():main(args)
    except Exception as error:
        atomic_json(args.run/'status.json',{'status':'failed','error':repr(error),'timestamp':time.time()})
        raise
