"""Development NTP and unconstrained greedy symbolic evaluation. No memory writes."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from .model import losses
from .data import Corpus, verify_sudoku
from .prepare_data import DATA


@torch.no_grad()
def generate(model, prefix, max_new_tokens=96, bank=None, device='mps'):
    model.eval()
    ids = torch.tensor([prefix], device=device, dtype=torch.long)
    committed = None
    for _ in range(max_new_tokens):
        out = model(ids, codes=committed, bank=bank)
        committed = out['codes'].detach()
        nxt = out['logits'][:, -1].argmax(-1)
        ids = torch.cat((ids, nxt[:, None]), -1)
        if int(nxt.item()) == 2:
            break
    return ids[0, len(prefix):].cpu().tolist()


@torch.no_grad()
def evaluate(model, device, data_dir=DATA, batches=3, greedy_math=8, greedy_sudoku=2,
             deadline=None, split='dev'):
    was_training = model.training
    model.eval()
    corpus = Corpus(split, seed=1811, data_dir=data_dir)
    tokenizer = Tokenizer.from_file(str(Path(data_dir)/'tokenizer.json'))
    result = dict(split=split, families={}, generation={}, selector_policy='prefix_argmax',
                  memory_writes=0, interpretation='Teacher-forced loss and free-generation success are distinct metrics')
    for family in corpus.arrays:
        values, ablation, stds, code_counts = [], [], [], torch.zeros(model.c.codes, dtype=torch.long)
        weights=[]
        for _ in range(batches):
            if deadline and time.time() >= deadline: break
            batch = corpus.batch(2, family=family, device=device)
            out = model(batch['ids'], batch['lengths'])
            parts = losses(model, out, batch['targets'], batch['target_mask'])
            off = model(batch['ids'], batch['lengths'], codes=out['codes'], ablate=True)
            ablated = losses(model, off, batch['targets'], batch['target_mask'])
            values.append(float(parts['token'])); ablation.append(float(ablated['token']))
            weights.append(int(batch['target_mask'].sum()))
            valid = out['valid_groups']
            states=out['pooled'][valid].flatten(0,1)
            stds.append(float(states.std(0, unbiased=False).mean()) if len(states)>1 else 0.)
            code_counts += torch.bincount(out['codes'][valid].cpu(), minlength=model.c.codes)
        if values:
            nll=float(np.average(values, weights=weights)); offnll=float(np.average(ablation, weights=weights))
            result['families'][family]=dict(nll=nll, perplexity=math.exp(min(nll,30)), supervised_tokens=sum(weights),
                                            no_concepts_nll=offnll, concept_ablation_delta_nll=offnll-nll,
                                            pooled_coordinate_std=float(np.mean(stds)), code_counts=code_counts.tolist())
    for family, n in [('math',greedy_math), ('sudoku_easy',greedy_sudoku), ('sudoku_hard',greedy_sudoku)]:
        arr, offsets, starts=corpus.arrays[family]
        metadata=[json.loads(x) for x in (Path(data_dir)/f'{family}_{split}_meta.jsonl').read_text().splitlines()]
        examples=[]
        for i in range(min(n,len(starts))):
            if deadline and time.time() >= deadline: break
            tok=arr[offsets[i]:offsets[i+1]].tolist(); p=int(starts[i])
            predicted=generate(model,tok[:p],max_new_tokens=48 if family=='math' else 84,device=device)
            eos=bool(predicted and predicted[-1]==2)
            # Only remove the terminal EOS. Interior PAD/BOS/UNK remain visible
            # and invalidate an otherwise correct-looking decoded answer.
            content=predicted[:-1] if eos else predicted
            text=tokenizer.decode(content,skip_special_tokens=False)
            if family=='math':
                valid=text.strip()==metadata[i]['answer'].strip() and eos
                reason='exact_match' if valid else 'incorrect_or_unterminated'
            else:
                check=verify_sudoku(metadata[i]['puzzle'],text)
                valid=check['valid'] and eos; reason=check['reason'] if eos else 'unterminated'
            examples.append(dict(identity=metadata[i]['identity'],correct=valid,reason=reason,
                                 response=text,generated_tokens=len(predicted),eos=eos))
        result['generation'][family]=dict(correct=sum(x['correct'] for x in examples),n=len(examples),
                                           examples=examples,metric='complete exact/constraint-valid answer with EOS')
    model.train(was_training)
    return result


if __name__=='__main__':
    import argparse
    from .config import ModelConfig
    from .model import V10
    p=argparse.ArgumentParser(); p.add_argument('checkpoint');p.add_argument('--device',default='mps')
    p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--output',required=True)
    a=p.parse_args(); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    m=V10(ModelConfig(**ck['model_config'])); m.load_state_dict(ck['model']);m.place(a.device)
    res=evaluate(m,a.device,split=a.split,batches=16,greedy_math=64,greedy_sudoku=32)
    Path(a.output).write_text(json.dumps(res,indent=2));print(a.output)
