"""A deadline-bounded, resumable 24-hour M2/MPS pretraining run."""
import argparse
from collections import Counter
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
import psutil
import torch
from tokenizers import Tokenizer
import unicodedata
from .config import ModelConfig
from .model import V10, losses
from .data import Corpus
from .evaluate import evaluate
from .prepare_data import DATA, ROOT, digest


def atomic_json(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def source_hash():
    h=hashlib.sha256()
    for p in sorted(ROOT.glob('*.py')):
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def canonical_ids(tokenizer, vocab):
    # Keep bytes/specials that do not decode uniquely distinct. Lossless tokenizer
    # IDs stay unchanged outside this lookup-only compression map.
    seen={}; ids=[]
    for i in range(vocab):
        raw=tokenizer.id_to_token(i)
        decoded=tokenizer.decode([i],skip_special_tokens=False)
        norm=unicodedata.normalize('NFKC',decoded).lower().strip()
        key=('unique',i) if '\ufffd' in decoded or not norm or i<4 else ('text',norm)
        if key not in seen: seen[key]=i
        ids.append(seen[key])
    return torch.tensor(ids)


def memory_stats(device):
    stats=dict(rss_gb=psutil.Process().memory_info().rss/2**30,
               system_available_gb=psutil.virtual_memory().available/2**30)
    if device=='mps':
        stats.update(mps_allocated_gb=torch.mps.current_allocated_memory()/2**30,
                     mps_driver_gb=torch.mps.driver_allocated_memory()/2**30)
    return stats


def build_optimizers(model, lr=3e-4):
    dense=[p for n,p in model.named_parameters() if n!='engram.table.weight']
    decay=[p for p in dense if p.ndim>=2]
    no_decay=[p for p in dense if p.ndim<2]
    adam=torch.optim.AdamW([dict(params=decay,weight_decay=.1),dict(params=no_decay,weight_decay=0.)],
                           lr=lr,betas=(.9,.95),foreach=False)
    if model.c.engram_cpu:
        table=torch.optim.SparseAdam([model.engram.table.weight],lr=lr*5,betas=(.9,.99))
    else:
        table=torch.optim.AdamW([model.engram.table.weight],lr=lr*5,weight_decay=0,foreach=False)
    return adam,table,dense


def clip(model,dense):
    norm=torch.nn.utils.clip_grad_norm_(dense,1.,error_if_nonfinite=True)
    g=model.engram.table.weight.grad
    if g is not None:
        if g.is_sparse:
            g=g.coalesce();model.engram.table.weight.grad=g;v=g.values()
        else: v=g
        n=v.norm()
        if not torch.isfinite(n):raise FloatingPointError('Engram gradient nonfinite')
        v.mul_((1./n.clamp_min(1.)).item())
    return float(norm)


def main(args):
    run=Path(args.run_dir);run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.device=='mps':
        if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
        torch.mps.set_per_process_memory_fraction(.55)
    corpus=Corpus('train',seed=args.seed)
    tokenizer=Tokenizer.from_file(str(DATA/'tokenizer.json'))
    c=ModelConfig(vocab_size=tokenizer.get_vocab_size())
    model=V10(c)
    counts=model.counts()
    if counts['dense']>=10_000_000:raise ValueError(counts)
    model.engram.canonical.copy_(canonical_ids(tokenizer,c.vocab_size))
    model.place(args.device).train()
    optimizer,engram_optimizer,dense=build_optimizers(model,args.lr)
    now=time.time();start=now;deadline=start+args.hours*3600;step=0;tokens=0;family_tokens=Counter()
    code_sha=source_hash();data_sha=digest(DATA/'manifest.json')
    if args.resume:
        checkpoint=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
        if checkpoint['source_sha256']!=code_sha or checkpoint['data_sha256']!=data_sha:
            raise ValueError('Source/data changed; explicit new run required')
        if checkpoint.get('training_config') is not None:
            for name in ('batch_size','accumulate','context','lr','candidate_every','seed','device'):
                if getattr(args,name)!=checkpoint['training_config'][name]:
                    raise ValueError('Resume training setting changed: '+name)
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        engram_optimizer.load_state_dict(checkpoint['engram_optimizer'])
        start,deadline,step,tokens=[checkpoint[k] for k in ('start','deadline','step','tokens')]
        corpus.rng.bit_generator.state=checkpoint['data_rng'];torch.set_rng_state(checkpoint['torch_rng'])
        if args.device=='mps' and checkpoint.get('mps_rng') is not None:torch.mps.set_rng_state(checkpoint['mps_rng'])
        family_tokens.update(checkpoint['family_tokens'])
    stop={'requested':False}
    def handler(*_):stop['requested']=True
    signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
    manifest=dict(model=c.to_dict(),parameters=counts,training=vars(args),start_unix=start,deadline_unix=deadline,
                  device=args.device,torch=torch.__version__,source_sha256=code_sha,data_sha256=data_sha,
                  loss_weights=dict(token=1,concept=.1,vq=.01,selector=.05,explore=.1),
                  auxiliary_candidate_every=args.candidate_every,gradcuit_xi=0,rbf_pretraining_writes=0,
                  target='online code-free pooler with stop-gradient',precision='float32',
                  table_optimizer='CPU SparseAdam, 5x dense LR',baseline_evaluation='development only; sealed test excluded')
    atomic_json(run/'manifest.json',manifest)
    def save():
        # state_dict preserves CPU sparse-table placement; load always map_location=cpu.
        state=dict(model_config=c.to_dict(),model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                   optimizer=optimizer.state_dict(),engram_optimizer=engram_optimizer.state_dict(),
                   start=start,deadline=deadline,step=step,tokens=tokens,family_tokens=dict(family_tokens),
                   source_sha256=code_sha,data_sha256=data_sha,training_config=vars(args),data_rng=corpus.rng.bit_generator.state,
                   torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if args.device=='mps' else None)
        tmp=run/'latest.tmp.pt';torch.save(state,tmp);tmp.replace(run/'latest.pt')
    last_save,last_eval=time.time(),time.time()
    last_log_time,last_log_tokens=time.monotonic(),tokens
    metric_sum=Counter();metric_steps=0
    status='running';error=None;metrics={}
    atomic_json(run/'status.json',dict(status=status,pid=os.getpid(),step=step,tokens=tokens,deadline_unix=deadline))
    try:
        if step==0 and not args.skip_initial_eval:
            initial=evaluate(model,args.device,batches=1,greedy_math=0,greedy_sudoku=0,deadline=min(deadline,time.time()+120))
            atomic_json(run/'evaluation_initial.json',initial)
        save()
        while time.time()<deadline and not stop['requested'] and (not args.max_steps or step<args.max_steps):
            progress=min(1.,(time.time()-start)/(deadline-start))
            # A time schedule means throughput changes do not extend the 24h budget.
            warmup=min(1.,(step+1)/200)
            scale=warmup*(.1+.9*.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups:group['lr']=args.lr*scale
            for group in engram_optimizer.param_groups:group['lr']=args.lr*5*scale
            optimizer.zero_grad(set_to_none=True);engram_optimizer.zero_grad(set_to_none=True)
            aggregate=Counter();target_count=0
            for micro in range(args.accumulate):
                batch=corpus.batch(args.batch_size,args.context,progress=progress,device='cpu')
                indices=model.engram.indices(batch['ids'])
                for key in ('ids','lengths','targets','target_mask'):
                    batch[key]=batch[key].to(args.device)
                out=model(batch['ids'],batch['lengths'],indices)
                boundary=None
                if step%args.candidate_every==0:
                    counts_valid=out['valid_groups'].sum(-1).detach().cpu().numpy()-1
                    if counts_valid.max()>0:
                        boundary=torch.tensor([int(corpus.rng.integers(max(1,n))) for n in counts_valid],device=args.device)
                parts=losses(model,out,batch['targets'],batch['target_mask'],boundary)
                if not torch.isfinite(parts['total']):raise FloatingPointError('Training loss nonfinite')
                (parts['total']/args.accumulate).backward()
                supervised=int(batch['target_mask'].sum())
                tokens+=int(batch['lengths'].sum());target_count+=supervised
                family_tokens[batch['family']]+=supervised
                for k,v in parts.items():aggregate[k]+=float(v.detach())/args.accumulate
            gradnorm=clip(model,dense)
            optimizer.step();engram_optimizer.step();step+=1
            metric_sum.update(aggregate);metric_steps+=1
            if step%args.log_every==0 or step==1:
                t=time.monotonic();metrics=dict(step=step,tokens=tokens,loss={k:v/metric_steps for k,v in metric_sum.items()},dense_grad_norm=gradnorm,
                    tokens_per_second=(tokens-last_log_tokens)/max(t-last_log_time,1e-6),elapsed_hours=(time.time()-start)/3600,
                    remaining_hours=max(0.,(deadline-time.time())/3600),learning_rate=args.lr*scale,
                    family_supervised_tokens=dict(family_tokens),**memory_stats(args.device))
                with open(run/'train.jsonl','a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
                atomic_json(run/'status.json',dict(status='running',pid=os.getpid(),deadline_unix=deadline,**metrics))
                print(json.dumps(metrics),flush=True)
                last_log_time,last_log_tokens=t,tokens
                metric_sum.clear();metric_steps=0
            if time.time()-last_save>=args.checkpoint_seconds:
                save();last_save=time.time()
            if not args.no_eval and time.time()-last_eval>=args.eval_seconds and time.time()<deadline-120:
                result=evaluate(model,args.device,deadline=min(deadline-60,time.time()+180))
                atomic_json(run/f'evaluation_{step:07d}.json',result);last_eval=time.time()
            if args.device=='mps' and step%200==0:torch.mps.empty_cache()
        status='stopped' if stop['requested'] else ('smoke_complete' if args.max_steps and step>=args.max_steps else 'complete')
    except Exception:
        status='failed';error=traceback.format_exc();print(error,flush=True)
    finally:
        save()
        atomic_json(run/'status.json',dict(status=status,pid=os.getpid(),step=step,tokens=tokens,deadline_unix=deadline,
                    elapsed_hours=(time.time()-start)/3600,error=error,last_metrics=metrics,checkpoint=str(run/'latest.pt')))
    if error:raise SystemExit(1)


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',default=str(ROOT/'runs/pretrain_24h'))
    p.add_argument('--device',choices=['cpu','mps'],default='mps')
    p.add_argument('--hours',type=float,default=24)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--accumulate',type=int,default=1)
    p.add_argument('--context',type=int,default=384)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=1011)
    p.add_argument('--candidate-every',type=int,default=4)
    p.add_argument('--log-every',type=int,default=20)
    p.add_argument('--checkpoint-seconds',type=float,default=600)
    p.add_argument('--eval-seconds',type=float,default=1800)
    p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--no-eval',action='store_true')
    p.add_argument('--skip-initial-eval',action='store_true')
    return p


if __name__=='__main__':main(parser().parse_args())
