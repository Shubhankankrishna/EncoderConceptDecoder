"""Read-only progress review plus bounded CPU development checks.
Does not update telemetry's canonical audit files, train, or use the sealed test.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.request import urlopen
from zoneinfo import ZoneInfo
import numpy as np
import torch
from tokenizers import Tokenizer
from ..config import ModelConfig
from ..data import Corpus,verify_sudoku
from ..evaluate import generate
from ..model import V10,losses
from ..prepare_data import ROOT,DATA
from ..telemetry.store import atomic_json


def api(route):
    with urlopen('http://127.0.0.1:8766/api/v1/'+route,timeout=15) as f:return json.load(f)


def events(kind):
    result=[];cursor=0
    while True:
        d=api(f'events?kind={kind}&after_id={cursor}&limit=5000');result+=d['events'];cursor=d['next_after_id']
        if not d['has_more']:return result


def sudoku_detail(puzzle,response):
    text=response.strip()
    if len(text)!=81 or any(c not in '123456789' for c in text):return dict(format_valid=False)
    rows=[text[i:i+9] for i in range(0,81,9)];target=set('123456789')
    clues=[i for i,c in enumerate(puzzle) if c!='0'];correct=sum(text[i]==puzzle[i] for i in clues)
    return dict(format_valid=True,clue_correct=correct,clues=len(clues),clue_accuracy=correct/len(clues) if clues else None,
                valid_rows=sum(set(r)==target for r in rows),valid_columns=sum(set(text[c::9])==target for c in range(9)),
                valid_boxes=sum(set(''.join(rows[r+i][c:c+3] for i in range(3)))==target for r in (0,3,6) for c in (0,3,6)))


def telemetry_summary():
    overview=api('overview');training=events('training');evaluations=sorted(events('evaluation'),key=lambda e:(e['step'],e['id']));host=events('host');last=training[-1]['payload'];end=last['elapsed_hours']
    windows=[]
    for lo,hi in [(0,.5),(.5,1),(1,2),(2,3),(max(0,end-.5),end)]:
        rows=[e['payload'] for e in training if lo<=e['payload']['elapsed_hours']<=hi]
        if not rows:continue
        grad=np.array([p['dense_grad_norm'] for p in rows]);loss={k:float(np.mean([p['loss'][k] for p in rows])) for k in rows[0]['loss']}
        seconds=(rows[-1]['elapsed_hours']-rows[0]['elapsed_hours'])*3600
        windows.append(dict(start_hour=lo,end_hour=hi,records=len(rows),loss_means=loss,grad_median=float(np.median(grad)),grad_p95=float(np.percentile(grad,95)),grad_max=float(grad.max()),dense_clip_fraction=float(np.mean(grad>1)),throughput_median=float(np.median([p['tokens_per_second'] for p in rows])),throughput_token_delta_per_second=(rows[-1]['tokens']-rows[0]['tokens'])/seconds if seconds else None))
    meta={f:{r['identity']:r for r in [json.loads(x) for x in (DATA/f'{f}_dev_meta.jsonl').read_text().splitlines()]} for f in ['math','sudoku_easy','sudoku_hard']}
    eval_rows=[]
    for e in evaluations:
        q=e['payload'];row=dict(step=e['step'],wall_time=e['wall_time'],families=q['families'],generation={})
        for family,g in q.get('generation',{}).items():
            entry={k:v for k,v in g.items() if k!='examples'}
            if family=='math':entry['response_counts']=dict(Counter(x['response'] for x in g['examples']))
            else:entry['partial_diagnostics']=[sudoku_detail(meta[family][x['identity']]['puzzle'],x['response']) for x in g['examples']]
            row['generation'][family]=entry
        eval_rows.append(row)
    flat=[v for e in training for v in e['payload']['loss'].values()]+[e['payload']['dense_grad_norm'] for e in training]
    return dict(captured_unix=time.time(),overview=overview,collector=api('health'),training_records=len(training),all_logged_losses_gradients_finite=all(math.isfinite(v) for v in flat),windows=windows,evaluations=eval_rows,host_first=host[0] if host else None,host_latest=host[-1] if host else None)


@torch.no_grad()
def text_check(model):
    result={}
    for split in ['train','dev']:
        corpus=Corpus(split,seed=2861);values=[];deltas=[];weights=[];chosen=[];engram=[]
        for j in range(16):
            b=corpus.batch(2,context=384,family='text',device='cpu');out=model(b['ids'],b['lengths']);base=float(losses(model,out,b['targets'],b['target_mask'])['token']);n=int(b['target_mask'].sum())
            values.append(base);weights.append(n);chosen+=b['chosen']
            if split=='dev':
                off=model(b['ids'],b['lengths'],codes=out['codes'],ablate=True);deltas.append(float(losses(model,off,b['targets'],b['target_mask'])['token'])-base)
                if j<4:
                    hook=model.engram.register_forward_hook(lambda module,args,output:args[0])
                    try:noengram=model(b['ids'],b['lengths'])
                    finally:hook.remove()
                    engram.append(dict(tokens=n,delta_nll=float(losses(model,noengram,b['targets'],b['target_mask'])['token'])-base))
        nll=float(np.average(values,weights=weights));entry=dict(sampled_sequences=len(chosen),unique_sequences=len(set(chosen)),tokens=sum(weights),nll=nll,perplexity=math.exp(nll),batch_nlls=values,batch_weights=weights,selection='Seed 2861, 16 batches of 2 with replacement; text only; context 384')
        if deltas:
            rng=np.random.default_rng(2241);idx=rng.integers(0,len(deltas),(2000,len(deltas)));d=np.array(deltas);w=np.array(weights);boot=(d[idx]*w[idx]).sum(-1)/w[idx].sum(-1)
            entry['concept_ablation']=dict(delta_nll=float(np.average(d,weights=w)),batch_deltas=deltas,paired_batch_bootstrap_ci95=np.percentile(boot,[2.5,97.5]).tolist(),ci_limitation='Descriptive paired-batch resampling; small sampled documents, not a population guarantee.')
            entry['engram_ablation']=dict(sampled_sequences=8,delta_nll=float(np.average([x['delta_nll'] for x in engram],weights=[x['tokens'] for x in engram])),batch_deltas=engram)
        result[split]=entry;print(json.dumps(dict(text_split=split,nll=nll,sequences=len(chosen))),flush=True)
    result['train_dev_gap']=result['dev']['nll']-result['train']['nll']
    return result


@torch.no_grad()
def balanced_check(model):
    corpus=Corpus('dev');tokenizer=Tokenizer.from_file(str(DATA/'tokenizer.json'));rng=np.random.default_rng(1927);result={}
    for family in ['math','sudoku_easy','sudoku_hard']:
        arr,offsets,starts=corpus.arrays[family];meta=[json.loads(x) for x in (DATA/f'{family}_dev_meta.jsonl').read_text().splitlines()]
        if family=='math':
            groups={source:[i for i,r in enumerate(meta) if r['source']==source] for source in sorted({r['source'] for r in meta})}
            chosen=[int(i) for indices in groups.values() for i in rng.choice(indices,min(16,len(indices)),replace=False)]
        else:chosen=[int(i) for i in rng.choice(len(meta),min(16,len(meta)),replace=False)]
        examples=[]
        for i in chosen:
            tok=arr[offsets[i]:offsets[i+1]].tolist();p=int(starts[i]);predicted=generate(model,tok[:p],max_new_tokens=48 if family=='math' else 84,device='cpu');eos=bool(predicted and predicted[-1]==2);text=tokenizer.decode(predicted[:-1] if eos else predicted,skip_special_tokens=False)
            if family=='math':correct=eos and text.strip()==meta[i]['answer'].strip();reason='exact' if correct else 'incorrect_or_unterminated';detail={}
            else:
                check=verify_sudoku(meta[i]['puzzle'],text);correct=eos and check['valid'];reason=check['reason'] if eos else 'unterminated';detail=sudoku_detail(meta[i]['puzzle'],text)
            examples.append(dict(index=i,identity=meta[i]['identity'],source=meta[i]['source'],response=text,correct=bool(correct),eos=eos,reason=reason,partial_diagnostics=detail))
        summary=dict(n=len(examples),correct=sum(x['correct'] for x in examples),examples=examples,reason_counts=dict(Counter(x['reason'] for x in examples)))
        if family=='math':summary['by_module']={s:dict(n=sum(x['source']==s for x in examples),correct=sum(x['correct'] for x in examples if x['source']==s)) for s in groups}
        else:
            formatted=[x['partial_diagnostics'] for x in examples if x['partial_diagnostics']['format_valid']]
            summary['partial_diagnostics']=dict(format_valid=len(formatted),clue_consistent=sum(x['clue_correct']==x['clues'] for x in formatted),clue_accuracy=sum(x['clue_correct'] for x in formatted)/max(1,sum(x['clues'] for x in formatted)),mean_valid_rows=float(np.mean([x['valid_rows'] for x in formatted])) if formatted else None,mean_valid_columns=float(np.mean([x['valid_columns'] for x in formatted])) if formatted else None,mean_valid_boxes=float(np.mean([x['valid_boxes'] for x in formatted])) if formatted else None)
        result[family]=summary;print(json.dumps(dict(family=family,n=summary['n'],correct=summary['correct'],partial=summary.get('partial_diagnostics'))),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args();dest=Path(a.output_dir);dest.mkdir(parents=True,exist_ok=True);start=time.time();torch.set_num_threads(2);torch.manual_seed(1191)
    telemetry=telemetry_summary();atomic_json(dest/'telemetry_analysis.json',telemetry)
    with (ROOT/'runs/pretrain_24h/latest.pt').open('rb') as f:
        stat=os.fstat(f.fileno());ck=torch.load(f,map_location='cpu',weights_only=False)
    model=V10(ModelConfig(**ck['model_config']));model.load_state_dict(ck['model']);model.eval();result=dict(run_id='pretrain_24h',step=ck['step'],tokens=ck['tokens'],checkpoint_mtime=stat.st_mtime,source_sha256=ck['source_sha256'],device='cpu',threads=2,created_unix=start,selection='Same balanced development identities as checkpoint 17607 audit; seed 1927',limitations='Active math corpus contains bytes-literal wrappers. Strict scoring unchanged. Partial Sudoku diagnostics never count as solved. No test data, optimizer steps or memory writes.',memory_writes=0)
    del ck
    result['text_check']=text_check(model);result['generation']=balanced_check(model);result['seconds']=time.time()-start
    atomic_json(dest/'fresh_evaluation.json',result);print(json.dumps(dict(output=str(dest),step=result['step'],seconds=result['seconds'])),flush=True)

if __name__=='__main__':main()
