"""Stratified development generation, without teacher answers or output repair."""
import json
from pathlib import Path
import time
import torch
import numpy as np
from tokenizers import Tokenizer
from ..prepare_data import ROOT,DATA
from ..config import ModelConfig
from ..model import V10
from ..data import Corpus,verify_sudoku
from ..evaluate import generate
from ..telemetry.store import atomic_json


def main():
    torch.set_num_threads(2);start=time.time()
    ck=torch.load(ROOT/'runs/pretrain_24h/latest.pt',map_location='cpu',weights_only=False)
    m=V10(ModelConfig(**ck['model_config']));m.load_state_dict(ck['model']);m.eval()
    result=dict(run_id='pretrain_24h',kind='balanced_offline_dev_evaluation',step=ck['step'],source_sha256=ck['source_sha256'],
                split='dev',device='cpu',selection='Seed 1927, 16 per math module and 16 per Sudoku family, without replacement',
                limitation='Active corpus has byte-literal math wrappers. Metrics score its existing strict targets; not a clean-corpus benchmark.',
                generation={},families={},memory_writes=0)
    del ck
    corpus=Corpus('dev');tokenizer=Tokenizer.from_file(str(DATA/'tokenizer.json'));rng=np.random.default_rng(1927)
    for family in ['math','sudoku_easy','sudoku_hard']:
        arr,offsets,starts=corpus.arrays[family]
        meta=[json.loads(x) for x in (DATA/f'{family}_dev_meta.jsonl').read_text().splitlines()]
        if family=='math':
            groups={source:[i for i,r in enumerate(meta) if r['source']==source] for source in sorted({r['source'] for r in meta})}
            chosen=[int(i) for indices in groups.values() for i in rng.choice(indices,min(16,len(indices)),replace=False)]
        else:chosen=[int(i) for i in rng.choice(len(meta),min(16,len(meta)),replace=False)]
        examples=[]
        for i in chosen:
            tok=arr[offsets[i]:offsets[i+1]].tolist();p=int(starts[i])
            predicted=generate(m,tok[:p],max_new_tokens=48 if family=='math' else 84,device='cpu')
            eos=bool(predicted and predicted[-1]==2);text=tokenizer.decode(predicted[:-1] if eos else predicted,skip_special_tokens=False)
            if family=='math':correct=eos and text.strip()==meta[i]['answer'].strip();reason='exact' if correct else 'incorrect_or_unterminated'
            else:
                check=verify_sudoku(meta[i]['puzzle'],text);correct=eos and check['valid'];reason=check['reason'] if eos else 'unterminated'
            examples.append(dict(index=i,identity=meta[i]['identity'],source=meta[i]['source'],response=text,correct=bool(correct),eos=eos,reason=reason))
        summary=dict(n=len(examples),correct=sum(x['correct'] for x in examples),examples=examples)
        if family=='math':summary['by_module']={s:dict(n=sum(x['source']==s for x in examples),correct=sum(x['correct'] for x in examples if x['source']==s)) for s in groups}
        result['generation'][family]=summary
        print(json.dumps(dict(family=family,n=summary['n'],correct=summary['correct'],step=result['step'])),flush=True)
    result['seconds']=time.time()-start;atomic_json(ROOT/'audit/balanced_evaluation.json',result)
    print(json.dumps(dict(seconds=result['seconds'],step=result['step'])),flush=True)


if __name__=='__main__':main()
