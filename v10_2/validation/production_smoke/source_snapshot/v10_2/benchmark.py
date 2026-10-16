"""Real-corpus, full-parameter Metal benchmark. Does not touch training runs."""
import argparse,json,time,gc,statistics
from pathlib import Path
import torch
from tokenizers import Tokenizer
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_10M.train import build_optimizers,canonical_ids,memory_stats
from .model import V102
from .accumulate import backward_effective_batch,gradient_metrics
from .prepare_data import DATA
from .provenance import provenance

def run(microbatch,foreach,steps,seed=1020):
    torch.manual_seed(seed);torch.set_num_threads(4);torch.mps.set_per_process_memory_fraction(.55)
    corpus=Corpus('train',seed=seed,data_dir=DATA);tok=Tokenizer.from_file(str(DATA/'tokenizer.json'))
    model=V102();model.engram.canonical.copy_(canonical_ids(tok,model.c.vocab_size));model.place('mps').train()
    opt,sparse,dense=build_optimizers(model)
    for group in opt.param_groups:group['foreach']=foreach
    records=[];start=time.monotonic()
    for step in range(steps+8):
        tick=time.monotonic();opt.zero_grad(set_to_none=True);sparse.zero_grad(set_to_none=True)
        # Exercise every family in warmup, then the actual production mixture.
        family=['text','math','sudoku_easy','sudoku_hard'][step%4] if step<8 else None
        raw=corpus.batch(8,384,family=family,progress=.25)
        diag={};parts=backward_effective_batch(model,raw,microbatch,step%4==0,seed+step,diag)
        grad=gradient_metrics(model,dense,detailed=(step+1)%20==0);opt.step();sparse.step();torch.mps.synchronize()
        rec=dict(step=step,seconds=time.monotonic()-tick,family=raw['family'],candidate=step%4==0,input_tokens=int(raw['lengths'].sum()),loss=parts,**grad,**memory_stats('mps'))
        if step>=8:records.append(rec)
        if step%20==0:print(json.dumps(dict(microbatch=microbatch,foreach=foreach,step=step,seconds=rec['seconds'],driver_gib=rec['mps_driver_gb'])),flush=True)
    groups={}
    for family in corpus.arrays:
        for cand in (False,True):
            sub=[r for r in records if r['family']==family and r['candidate']==cand]
            if sub:groups[f'{family}/candidate={cand}']=dict(updates=len(sub),tokens_per_second=sum(r['input_tokens'] for r in sub)/sum(r['seconds'] for r in sub),median_seconds=statistics.median(r['seconds'] for r in sub))
    result=dict(microbatch=microbatch,foreach=foreach,updates=steps,warmup=8,elapsed_seconds=time.monotonic()-start,
        sustained_tokens_per_second=sum(r['input_tokens'] for r in records)/sum(r['seconds'] for r in records),
        max_mps_driver_gib=max(r['mps_driver_gb'] for r in records),max_rss_gib=max(r['rss_gb'] for r in records),
        min_system_available_gib=min(r['system_available_gb'] for r in records),groups=groups,records=records)
    del model,opt,sparse,dense;gc.collect();torch.mps.empty_cache()
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=24);p.add_argument('--microbatch',type=int,default=8);p.add_argument('--foreach',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    result=run(a.microbatch,a.foreach,a.steps);result['provenance']=provenance()
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('records','provenance')},indent=2))
