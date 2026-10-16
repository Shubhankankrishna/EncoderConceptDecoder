"""Fixed, balanced development panels; explicit route ablations and strict solves."""
import json
import hashlib
import math
import re
from fractions import Fraction
from pathlib import Path
import time
import numpy as np
import torch
from tokenizers import Tokenizer
from v10_10M.data import Corpus,verify_sudoku
from v10_10M.evaluate import generate
from .model import token_loss,latent_errors
from .prepare_data import DATA


def parse_answer(text):
    """Exact numeric grammar, including ordered lists; no answer repair."""
    atoms=text.strip().split(',')
    if not atoms:return None
    values=[]
    for atom in atoms:
        atom=atom.strip()
        if not re.fullmatch(r'[+-]?(?:[0-9]+/[1-9][0-9]*|[0-9]+(?:\.[0-9]*)?|\.[0-9]+)',atom):return None
        try:values.append(Fraction(atom))
        except (ValueError,ZeroDivisionError):return None
    return tuple(values)


def score_math(text,answer,eos):
    raw=eos and text.strip()==answer.strip();expected=parse_answer(answer);predicted=parse_answer(text)
    return dict(correct=bool(eos and (predicted==expected if expected is not None and predicted is not None else raw)),
                raw_exact=bool(raw),target_grammar_supported=expected is not None)


def generation_indices(metadata,family,rng,math_per_module,sudoku_per_family):
    if family=='math':
        groups={s:[i for i,r in enumerate(metadata) if r['source']==s] for s in sorted({r['source'] for r in metadata})}
        return [int(i) for group in groups.values() for i in rng.choice(group,min(math_per_module,len(group)),replace=False)]
    return [int(i) for i in rng.choice(len(metadata),min(sudoku_per_family,len(metadata)),replace=False)]


@torch.no_grad()
def evaluate(model,device,data_dir=DATA,batches=2,math_per_module=4,sudoku_per_family=2,deadline=None):
    was_training=model.training;model.eval();started=time.time()
    corpus=Corpus('dev',seed=1811,data_dir=data_dir);tok=Tokenizer.from_file(str(Path(data_dir)/'tokenizer.json'))
    result=dict(split='dev',families={},generation={},memory_writes=0,selector_policy='prefix_argmax',
        selection='Fixed seed 1927, balanced math modules; random without replacement within each module/family',
        interpretation='Teacher-forced NLL and strict complete free-generation success are distinct; no output repair')
    try:
        for family in corpus.arrays:
            totals={};count=0;codes=torch.zeros(model.c.codes,dtype=torch.long);stds=[];latent=[0.,0.];groups=0
            for _ in range(batches):
                if deadline and time.time()>=deadline:break
                batch=corpus.batch(1,family=family,device=device)
                x,l,y,mask=[batch[k] for k in ('ids','lengths','targets','target_mask')];n=int(mask.sum())
                out=model(x,l);totals['nll']=totals.get('nll',0.)+float(token_loss(out,y,mask))*n;count+=n
                for label,flags in [('no_concepts_nll',dict(ablate=True)),('no_pq_nll',dict(ablate_pq=True)),
                                    ('no_continuous_nll',dict(ablate_continuous=True)),('one_visit_nll',dict(visits=1))]:
                    off=model(x,l,codes=out['codes'],**flags)
                    totals[label]=totals.get(label,0.)+float(token_loss(off,y,mask))*n
                valid=out['valid_groups'];ng=int(valid[:,1:].sum());groups+=ng
                for i,e in enumerate(latent_errors(out)):latent[i]+=float(e)*ng
                states=out['pooled'][valid].flatten(0,1)
                stds.append(float(states.std(0,unbiased=False).mean()) if len(states)>1 else 0.)
                codes+=torch.bincount(out['codes'][valid].cpu(),minlength=model.c.codes)
            if count:
                values={k:v/count for k,v in totals.items()};nll=values['nll']
                result['families'][family]=dict(**values,perplexity=math.exp(min(nll,30)),supervised_tokens=count,
                    concept_ablation_delta_nll=values['no_concepts_nll']-nll,pq_ablation_delta_nll=values['no_pq_nll']-nll,
                    continuous_ablation_delta_nll=values['no_continuous_nll']-nll,one_visit_delta_nll=values['one_visit_nll']-nll,
                    latent_visit_mse=[v/max(1,groups) for v in latent],latent_group_count=groups,
                    pooled_coordinate_std=float(np.mean(stds)),code_counts=codes.tolist())
        rng=np.random.default_rng(1927)
        for family in ('math','sudoku_easy','sudoku_hard'):
            meta=[json.loads(x) for x in (Path(data_dir)/f'{family}_dev_meta.jsonl').read_text().splitlines()]
            chosen=generation_indices(meta,family,rng,math_per_module,sudoku_per_family)
            arr,offsets,starts=corpus.arrays[family];examples=[]
            for i in chosen:
                if deadline and time.time()>=deadline:break
                row=meta[i];prefix=arr[offsets[i]:offsets[i]+starts[i]].tolist()
                pred=generate(model,prefix,max_new_tokens=48 if family=='math' else 84,device=device)
                eos=bool(pred and pred[-1]==2);text=tok.decode(pred[:-1] if eos else pred,skip_special_tokens=False)
                if family=='math':
                    scoring=score_math(text,row['answer'],eos);correct=scoring['correct'];reason='exact_numeric' if correct else 'incorrect_or_unterminated'
                else:
                    scoring={}
                    check=verify_sudoku(row['puzzle'],text);correct=eos and check['valid'];reason=check['reason'] if eos else 'unterminated'
                examples.append(dict(identity=row['identity'],source=row['source'],response=text,eos=eos,
                                     **{k:v for k,v in scoring.items() if k!='correct'},correct=bool(correct),reason=reason,generated_tokens=len(pred)))
            entry=dict(n=len(examples),correct=sum(x['correct'] for x in examples),planned_n=len(chosen),examples=examples,
                       identity_set_sha256=hashlib.sha256(json.dumps([meta[i]['identity'] for i in chosen]).encode()).hexdigest(),
                       metric='Complete exact numeric or ordered-list match / constraint-valid board, with EOS')
            if family=='math':entry['raw_exact_correct']=sum(x['raw_exact'] for x in examples)
            if family=='math':entry['by_module']={s:dict(n=sum(x['source']==s for x in examples),
                correct=sum(x['correct'] for x in examples if x['source']==s)) for s in sorted({r['source'] for r in meta})}
            result['generation'][family]=entry
        result.update(seconds=time.time()-started,budget_exhausted=bool(deadline and time.time()>=deadline))
        return result
    finally:model.train(was_training)
