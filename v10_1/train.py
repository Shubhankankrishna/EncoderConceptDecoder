"""Deadline-bounded V10.1 pretraining with reproducible checkpoints and telemetry."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
import torch
from tokenizers import Tokenizer
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_10M.prepare_data import digest
from v10_10M.train import atomic_json,canonical_ids,build_optimizers,memory_stats
from .model import V101,Architecture
from .accumulate import backward_effective_batch
from .evaluate import evaluate
from .prepare_data import DATA
from .provenance import provenance

ROOT=Path(__file__).resolve().parent


def identity():
    p=provenance()
    # The trainer imports the safe cleaner and generation primitive from V10;
    # hash all these dependencies as well as the local reference implementation.
    p['cleaner_sha256']=digest(ROOT.parent/'v10_10M/data_tools/clean_math.py')
    return hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),p


def verify_data(data):
    manifest=json.loads((data/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if digest(data/name)!=expected:raise ValueError('Data changed: '+name)
    if manifest['vocab_size']!=8192:raise ValueError('Unexpected vocabulary size')
    return digest(data/'manifest.json')


def gradient_metrics(model,dense,detailed=False):
    components={}
    if detailed:
        for name,p in model.named_parameters():
            if p.grad is None or name=='engram.table.weight':continue
            component=name.split('.')[0]
            components[component]=components.get(component,0.)+float(p.grad.detach().square().sum())
        components={k:math.sqrt(v) for k,v in components.items()}
    norm=float(torch.nn.utils.clip_grad_norm_(dense,1.,error_if_nonfinite=True))
    g=model.engram.table.weight.grad
    if g is None:raise ValueError('Missing Engram gradient')
    g=g.coalesce();model.engram.table.weight.grad=g;v=g.values();sn=float(v.norm())
    if not math.isfinite(sn):raise FloatingPointError('Nonfinite Engram gradient')
    v.mul_(min(1.,1./max(sn,1e-12)))
    return dict(dense_grad_norm=norm,dense_grad_norm_post=min(norm,1.),dense_clip_factor=min(1.,1./max(norm,1e-12)),
        engram_grad_norm=sn,engram_grad_norm_post=min(sn,1.),engram_clip_factor=min(1.,1./max(sn,1e-12)),
        engram_rows_touched=g._nnz(),component_grad_norms=components)


def main(args):
    if args.hours<=0 or args.hours>24:raise ValueError('Choose a positive total budget of at most 24 hours')
    if min(args.batch_size,args.microbatch_size,args.context,args.candidate_every,args.log_every)<1:raise ValueError('Invalid training dimensions')
    if args.context!=384:raise ValueError('Production context is fixed at 384')
    run=Path(args.run_dir).resolve();run.mkdir(parents=True,exist_ok=True);data=Path(args.data_dir).resolve()
    if (run/'latest.pt').exists() and not args.resume:raise ValueError('Existing run requires --resume')
    torch.set_num_threads(args.threads);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.device=='mps':
        if not torch.backends.mps.is_available():raise RuntimeError('Metal unavailable')
        torch.mps.set_per_process_memory_fraction(args.memory_fraction)
    data_sha=verify_data(data);code_sha,dependencies=identity()
    corpus=Corpus('train',seed=args.seed,data_dir=data);tokenizer=Tokenizer.from_file(str(data/'tokenizer.json'))
    c=ModelConfig(vocab_size=tokenizer.get_vocab_size());architecture=Architecture()
    model=V101(c,architecture);counts=model.counts()
    if counts['dense']>=10_000_000:raise ValueError('Dense budget exceeded')
    model.engram.canonical.copy_(canonical_ids(tokenizer,c.vocab_size));model.place(args.device).train()
    optimizer,engram_optimizer,dense=build_optimizers(model,args.lr)
    step=0;tokens=0;family_tokens=Counter();family_inputs=Counter();elapsed_prior=0.;schedule_hours=args.hours
    original_start=time.time();previous_manifest=None
    if args.resume:
        ck=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
        if ck['source_sha256']!=code_sha or ck['data_sha256']!=data_sha:raise ValueError('Source or corpus changed; cannot resume silently')
        for name in ('batch_size','microbatch_size','context','lr','candidate_every','seed','device','threads'):
            if getattr(args,name)!=ck['training_config'][name]:raise ValueError('Resume setting changed: '+name)
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);engram_optimizer.load_state_dict(ck['engram_optimizer'])
        step,tokens,elapsed_prior,schedule_hours,original_start=[ck[k] for k in ('step','tokens','elapsed_seconds','schedule_hours','start')]
        family_tokens.update(ck['family_tokens']);family_inputs.update(ck['family_inputs'])
        corpus.rng.bit_generator.state=ck['data_rng'];torch.set_rng_state(ck['torch_rng'])
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        if args.device=='mps':torch.mps.set_rng_state(ck['mps_rng'])
        previous_manifest=json.loads((run/'manifest.json').read_text());del ck
    if elapsed_prior>=args.hours*3600:raise ValueError('Budget already consumed; increase --hours to extend, up to 24')
    session_start=time.time();session_clock=time.monotonic();deadline=session_start+args.hours*3600-elapsed_prior
    def elapsed():return elapsed_prior+time.monotonic()-session_clock
    stop={'requested':False}
    def handle(*_):stop['requested']=True
    signal.signal(signal.SIGTERM,handle);signal.signal(signal.SIGINT,handle)
    sessions=(previous_manifest or {}).get('sessions',[])+[dict(start_unix=session_start,resume=args.resume,step=step,hours=args.hours)]
    manifest=dict(version='V10.1 RCDL',model=c.to_dict(),architecture=asdict(architecture),parameters=counts,
        training=vars(args),start_unix=original_start,deadline_unix=deadline,sessions=sessions,
        source_sha256=code_sha,dependency_hashes=dependencies,data_sha256=data_sha,tokenizer_sha256=digest(data/'tokenizer.json'),
        device=args.device,torch=torch.__version__,precision='float32',loss_weights=dict(token=1,latent=.1,vq=.01,chunk=.1,explore=.1,selector=.05),
        gradcuit_xi=0,rbf_pretraining_writes=0,initialization='Fresh random weights and new clean-data tokenizer',
        auxiliary_candidate_every=args.candidate_every,candidate_strategy='Detached K-way scoring; differentiate recomputed winner',
        effective_batch=args.batch_size,actual_microbatch=args.microbatch_size,schedule_hours=schedule_hours,
        extension_policy='--resume --hours 24 extends total active wall budget; original cosine schedule continues at its 10% floor after 16 hours',
        elapsed_semantics='active process wall time, including evaluation and checkpoint I/O; excludes stopped downtime',
        table_optimizer='CPU SparseAdam, 5x dense LR',baseline_evaluation='Balanced development only; sealed test reserved')
    atomic_json(run/'manifest.json',manifest)
    metrics={};status='running';error=None
    def save():
        state=dict(model_config=c.to_dict(),architecture=asdict(architecture),model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(),engram_optimizer=engram_optimizer.state_dict(),start=original_start,
            deadline=deadline,elapsed_seconds=elapsed(),schedule_hours=schedule_hours,step=step,tokens=tokens,
            family_tokens=dict(family_tokens),family_inputs=dict(family_inputs),source_sha256=code_sha,data_sha256=data_sha,
            training_config=vars(args),data_rng=corpus.rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state() if args.device=='mps' else None,python_rng=random.getstate(),numpy_rng=np.random.get_state())
        tmp=run/'latest.tmp.pt';torch.save(state,tmp);tmp.replace(run/'latest.pt')
    def write_status():
        state=dict(metrics);state.update(status=status,pid=os.getpid(),step=step,tokens=tokens,
            deadline_unix=deadline,elapsed_hours=elapsed()/3600,remaining_hours=max(0.,(deadline-time.time())/3600),
            error=error,checkpoint=str(run/'latest.pt'),last_metrics=metrics if status!='running' else None)
        atomic_json(run/'status.json',state)
    def assess(initial=False):
        rng=torch.get_rng_state();mrng=torch.mps.get_rng_state() if args.device=='mps' else None
        try:
            result=evaluate(model,args.device,data_dir=data,batches=2,
                math_per_module=0 if initial else 4,sudoku_per_family=0 if initial else 2,
                deadline=min(deadline-10,time.time()+240))
            result.update(schema_version=1,run_id=run.name,step=step,source_sha256=code_sha,data_sha256=data_sha,wall_time=time.time())
            name='evaluation_initial.json' if initial else f'evaluation_{step:07d}.json'
            atomic_json(run/name,result)
        finally:
            torch.set_rng_state(rng)
            if mrng is not None:torch.mps.set_rng_state(mrng)
    write_status();last_save=last_eval=time.monotonic();last_log_tokens=tokens;last_log=time.monotonic()
    try:
        save()
        if step==0 and not args.no_eval:assess(initial=True)
        with (run/'train.jsonl').open('a',buffering=1) as log:
            while elapsed()<args.hours*3600 and not stop['requested'] and (not args.max_steps or step<args.max_steps):
                tick=time.monotonic();progress=min(1.,elapsed()/(schedule_hours*3600))
                scale=min(1.,(step+1)/200)*(.1+.9*.5*(1+math.cos(math.pi*progress)))
                for group in optimizer.param_groups:group['lr']=args.lr*scale
                for group in engram_optimizer.param_groups:group['lr']=args.lr*5*scale
                optimizer.zero_grad(set_to_none=True);engram_optimizer.zero_grad(set_to_none=True)
                raw=corpus.batch(args.batch_size,args.context,progress=progress,device='cpu');family=raw['family']
                batch={k:raw[k].to(args.device) for k in ('ids','lengths','targets','target_mask')}
                candidate=step%args.candidate_every==0;diagnostics={}
                parts=backward_effective_batch(model,batch,args.microbatch_size,candidate,seed=args.seed+step,diagnostics=diagnostics)
                grads=gradient_metrics(model,dense,detailed=(step+1)%args.log_every==0 or step==0)
                optimizer.step();engram_optimizer.step();step+=1
                inp=int(raw['lengths'].sum());sup=int(raw['target_mask'].sum());tokens+=inp;family_tokens[family]+=sup;family_inputs[family]+=inp
                seconds=time.monotonic()-tick
                metrics=dict(schema_version=1,run_id=run.name,source_sha256=code_sha,data_sha256=data_sha,split='train',
                    step=step,tokens=tokens,wall_time=time.time(),loss=parts,**grads,
                    tokens_per_second=inp/max(seconds,1e-6),step_seconds=seconds,elapsed_hours=elapsed()/3600,
                    remaining_hours=max(0.,(deadline-time.time())/3600),learning_rate=args.lr*scale,
                    family=family,batch_indices=raw['chosen'],input_tokens=inp,supervised_tokens=sup,
                    family_supervised_tokens=dict(family_tokens),family_input_tokens=dict(family_inputs),candidate_requested=candidate,
                    candidate_executed=diagnostics['candidate_documents']>0,nonfinite_count=0,
                    selector_regret=(diagnostics['selector_regret_sum']/diagnostics['candidate_documents'] if diagnostics['candidate_documents'] else None),
                    **{k:v for k,v in diagnostics.items() if k!='selector_regret_sum'},**memory_stats(args.device))
                log.write(json.dumps(metrics,allow_nan=False)+'\n')
                if step%args.log_every==0 or step==1:
                    write_status();print(json.dumps(metrics,allow_nan=False),flush=True)
                    last_log_tokens=tokens;last_log=time.monotonic()
                if time.monotonic()-last_save>=args.checkpoint_seconds:save();last_save=time.monotonic()
                if not args.no_eval and time.monotonic()-last_eval>=args.eval_seconds and time.time()<deadline-260:
                    assess();last_eval=time.monotonic()
                if args.device=='mps' and step%200==0:torch.mps.empty_cache()
            status='stopped' if stop['requested'] else ('smoke_complete' if args.max_steps and step>=args.max_steps else 'complete')
    except Exception:
        status='failed';error=traceback.format_exc();print(error,flush=True)
    finally:
        save();write_status()
    if error:raise SystemExit(1)


def parser():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default=str(ROOT/'runs/pretrain_16h'))
    p.add_argument('--data-dir',default=str(DATA));p.add_argument('--hours',type=float,default=16)
    p.add_argument('--device',choices=['cpu','mps'],default='mps');p.add_argument('--seed',type=int,default=1012)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--microbatch-size',type=int,default=1)
    p.add_argument('--context',type=int,default=384);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--candidate-every',type=int,default=4);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--memory-fraction',type=float,default=.55);p.add_argument('--log-every',type=int,default=20)
    p.add_argument('--checkpoint-seconds',type=float,default=600);p.add_argument('--eval-seconds',type=float,default=1800)
    p.add_argument('--max-steps',type=int,default=0);p.add_argument('--resume',action='store_true');p.add_argument('--no-eval',action='store_true')
    return p


if __name__=='__main__':main(parser().parse_args())
