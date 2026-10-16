"""Bounded real-data integration runs; explicitly not a quality comparison.

Uses cleaned train/dev examples with the existing tokenizer for compatibility
tests only. No active arrays or sealed-test contents are read/write-modified.
Records per-step telemetry and matched CPU timing. Production clean-tokenizer
pretraining and sustained exclusive-device benchmarks remain separate work.
"""
import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import psutil
import torch
from tokenizers import Tokenizer
from v10_10M.config import ModelConfig
from v10_10M.train import canonical_ids,build_optimizers
from v10_10M.data_tools.clean_math import clean_math_field
from .model import V101,Architecture,losses,token_loss
from .provenance import provenance

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'v10_10M/data';DEST=ROOT/'v10_1/validation'
FAMILIES=['text','math','sudoku_easy','sudoku_hard']


def load_examples(split,context=192,count=8):
    tokenizer=Tokenizer.from_file(str(DATA/'tokenizer.json'));rng=np.random.default_rng(413 if split=='train' else 719)
    samples={};identities={}
    for family in FAMILIES:
        offsets=np.load(DATA/f'{family}_{split}_offsets.npy',mmap_mode='r')
        selected=sorted(rng.choice(len(offsets)-1,count,replace=False).tolist());wanted=set(selected)
        rows={}
        with (DATA/f'{family}_{split}_meta.jsonl').open() as f:
            for i,line in enumerate(f):
                if i in wanted:rows[i]=json.loads(line)
                if len(rows)==len(wanted):break
        arr=np.load(DATA/f'{family}_{split}.npy',mmap_mode='r')
        batch=[];identities[family]=[]
        for index in selected:
            row=rows[index]
            if family=='text':
                seq=np.array(arr[offsets[index]:offsets[index+1]],dtype=np.int64)
                begin=int(rng.integers(max(1,len(seq)-context)))
                seq=seq[begin:begin+context+1].tolist();answer=1
            else:
                if family=='math':
                    q=clean_math_field(row['prompt'][len('Math: '):-len('\nAnswer:')]);a=clean_math_field(row['answer'])
                    prompt='Math: '+q+'\nAnswer:';body=' '+a
                else:prompt='Sudoku:\n'+row['puzzle']+'\nSolution:\n';body=row['solution']
                prefix=tokenizer.encode(prompt).ids;seq=[1]+prefix+tokenizer.encode(body).ids+[2];answer=1+len(prefix)
                if len(seq)>context+1:raise ValueError('Never truncate symbolic answers')
            batch.append((seq,answer));identities[family].append(dict(index=index,identity=row['identity']))
        samples[family]=batch
    return samples,identities,tokenizer


def pack(rows):
    lengths=torch.tensor([len(seq)-1 for seq,_ in rows]);t=int(((lengths.max()+7)//8)*8)
    ids=torch.zeros(len(rows),t,dtype=torch.long);targets=torch.zeros_like(ids);mask=torch.zeros_like(ids,dtype=torch.bool)
    for i,(seq,start) in enumerate(rows):
        n=len(seq)-1;ids[i,:n]=torch.tensor(seq[:-1]);targets[i,:n]=torch.tensor(seq[1:]);mask[i,start-1:n]=True
    return dict(ids=ids,lengths=lengths,targets=targets,target_mask=mask)


def objective(model,batch,candidate=False,seed=0):
    out=model(batch['ids'],batch['lengths'])
    parts=losses(model,out,**batch,candidate_update=candidate,generator=torch.Generator().manual_seed(seed))
    return out,parts


@torch.no_grad()
def evaluate(model,dev):
    model.eval();result={}
    for family in FAMILIES:
        batch=pack(dev[family][:2]);out=model(batch['ids'],batch['lengths'])
        result[family]=dict(nll=float(token_loss(out,batch['targets'],batch['target_mask'])),supervised_tokens=int(batch['target_mask'].sum()))
    model.train();return result


def clip_and_log(model,dense):
    norm=float(torch.nn.utils.clip_grad_norm_(dense,1.,error_if_nonfinite=True))
    g=model.engram.table.weight.grad.coalesce();model.engram.table.weight.grad=g
    values=g.values();sn=float(values.norm())
    if not torch.isfinite(values).all():raise FloatingPointError('Nonfinite Engram gradient')
    values.mul_(min(1.,1./max(sn,1e-12)))
    return dict(dense_grad_norm_pre=norm,dense_grad_norm_post=min(norm,1.),dense_clip_factor=min(1.,1./max(norm,1e-12)),
                engram_grad_norm_pre=sn,engram_grad_norm_post=min(sn,1.),engram_rows_touched=g._nnz())


def main(steps,arms):
    torch.set_num_threads(2);DEST.mkdir(exist_ok=True)
    train,train_ids,tokenizer=load_examples('train');dev,dev_ids,_=load_examples('dev',count=4)
    manifest=dict(**provenance(),scope='Integration smoke, not trained quality comparison; cleaned math with legacy tokenizer; no sealed test',
        device='cpu',threads=2,context=192,batch_size=2,steps_per_arm=steps,seeds=[311],
        candidate_schedule='Every fifth smoke step rotates across four test families; production specification is every fourth optimizer update',
        candidate_strategy='detached scoring plus winning-branch recomputation',
        active_tokenizer_sha256=hashlib.sha256((DATA/'tokenizer.json').read_bytes()).hexdigest(),
        training_identities=train_ids,development_identities=dev_ids)
    (DEST/'integration_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results={}
    for arm in arms:
        torch.manual_seed(311);model=V101(ModelConfig(),Architecture.arm(arm))
        model.engram.canonical.copy_(canonical_ids(tokenizer,model.c.vocab_size))
        denseopt,sparseopt,dense=build_optimizers(model);initial=evaluate(model,dev)
        events=[];started=time.time();max_rss=0
        logfile=DEST/f'integration_{arm}.jsonl'
        with logfile.open('w') as f:
            for step in range(steps):
                family=FAMILIES[step%4];start=(step//4*2)%len(train[family]);batch=pack(train[family][start:start+2])
                tick=time.perf_counter();denseopt.zero_grad(set_to_none=True);sparseopt.zero_grad(set_to_none=True)
                # Candidate cadence rotates across families over these short test runs.
                candidate=step%5==0
                out,parts=objective(model,batch,candidate,seed=1100+step);parts['total'].backward()
                grads=clip_and_log(model,dense);denseopt.step();sparseopt.step()
                dt=time.perf_counter()-tick;rss=psutil.Process().memory_info().rss/2**30;max_rss=max(max_rss,rss)
                event=dict(schema_version=1,kind='integration_training',arm=arm,step=step+1,family=family,
                    source_sha256=manifest['v10_1_source_sha256'],candidate_strategy='recompute',
                    input_tokens=int(batch['lengths'].sum()),supervised_tokens=int(batch['target_mask'].sum()),seconds=dt,
                    loss={k:float(parts[k].detach()) for k in ['total','token','latent','vq','chunk','explore','selector']},
                    **grads,candidate_requested=candidate,candidate_executed=parts['candidate_executed'],
                    selector_regret=parts['selector_regret'],chunk_documents=parts['chunk_valid_documents'],rss_gib=rss)
                f.write(json.dumps(event,allow_nan=False)+'\n');f.flush();events.append(event)
        final=evaluate(model,dev)
        results[arm]=dict(parameters=model.counts(),architecture=asdict(model.architecture),initial_dev=initial,final_dev=final,
            seconds=time.time()-started,peak_sampled_rss_gib=max_rss,
            ordinary_step_seconds_median=float(np.median([e['seconds'] for e in events if not e['candidate_requested']])),
            candidate_step_seconds_median=float(np.median([e['seconds'] for e in events if e['candidate_requested']])),
            final_loss=events[-1]['loss'],all_finite=True)
        print(json.dumps(dict(arm=arm,**results[arm])),flush=True)
        del model,denseopt,sparseopt,dense,out,parts;gc.collect()
        (DEST/'integration_summary.json').write_text(json.dumps(dict(manifest=manifest,arms=results),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=16);p.add_argument('--arms',nargs='+',default=['A','B','C','D','S']);a=p.parse_args()
    main(a.steps,a.arms)
