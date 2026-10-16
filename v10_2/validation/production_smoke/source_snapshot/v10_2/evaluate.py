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
from .model import token_loss,losses,chunk_batch,future_ce
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
            totals={};count=0;codes=torch.zeros(model.c.codes,dtype=torch.long);stds=[];future=[0.,0.];documents=0;offset_num=torch.zeros(8);offset_den=torch.zeros(8);indices=[];pq_sum=torch.zeros(model.c.pq_segments,model.c.pq_words);pq_n=0;forecast_spread=[];zero_ce=0.;shuffled_ce=0.
            for _ in range(batches):
                if deadline and time.time()>=deadline:break
                batch=corpus.batch(1,family=family,device=device)
                x,l,y,mask=[batch[k] for k in ('ids','lengths','targets','target_mask')];n=int(mask.sum())
                indices.extend(batch['chosen']);out=model(x,l);totals['nll']=totals.get('nll',0.)+float(token_loss(out,y,mask))*n;count+=n
                for label,flags in [('no_concepts_nll',dict(ablate=True)),('no_pq_nll',dict(ablate_pq=True)),
                                    ('no_continuous_nll',dict(ablate_continuous=True)),('one_visit_nll',dict(visits=1)),('no_aligned_nll',dict(ablate_aligned=True)),('no_cross_pq_nll',dict(ablate_cross_pq=True)),('no_reader_nll',dict(ablate_reader=True))]:
                    off=model(x,l,codes=out['codes'],**flags)
                    totals[label]=totals.get(label,0.)+float(token_loss(off,y,mask))*n
                valid=out['valid_groups']
                objective=losses(model,out,x,l,y,mask,generator=torch.Generator().manual_seed(9100+_))
                n_docs=objective['future_documents'];documents+=n_docs
                for i,e in enumerate(objective['future_visits']):future[i]+=float(e)*n_docs
                offset_num+=objective['future_offset_numerators'].cpu();offset_den+=objective['future_offset_counts'].cpu()
                probs=out['probs'][valid].reshape(-1,model.c.pq_segments,model.c.pq_words).cpu()
                if len(probs):pq_sum+=probs.sum(0);pq_n+=len(probs)
                f=out['forecast'][valid].flatten(0,1)
                if len(f)>1:forecast_spread.append(float(f.std(0,unbiased=False).mean()))
                chunk=chunk_batch(x,l,y,mask,valid,generator=torch.Generator().manual_seed(9100+_))
                if chunk is not None:
                    f=out['forecast'][chunk['rows'],chunk['boundaries']]
                    zero_ce+=float(future_ce(model,torch.zeros_like(f),chunk).sum())
                    # Deterministic wrong-group forecast; single-group documents have no shuffle control.
                    wrong=(chunk['boundaries']+1)%out['valid_groups'].sum(-1)[chunk['rows']].clamp_min(1)
                    other=out['forecast'][chunk['rows'],wrong]
                    shuffled_ce+=float(future_ce(model,other,chunk).sum())

                states=out['pooled'][valid].flatten(0,1)
                stds.append(float(states.std(0,unbiased=False).mean()) if len(states)>1 else 0.)
                codes+=torch.bincount(out['codes'][valid].cpu(),minlength=model.c.codes)
            if count:
                values={k:v/count for k,v in totals.items()};nll=values['nll']
                pq=pq_sum/max(1,pq_n);entropy=-(pq*pq.clamp_min(1e-12).log()).sum(-1)
                result['families'][family]=dict(**values,perplexity=math.exp(min(nll,30)),supervised_tokens=count,
                    concept_ablation_delta_nll=values['no_concepts_nll']-nll,pq_ablation_delta_nll=values['no_pq_nll']-nll,
                    continuous_ablation_delta_nll=values['no_continuous_nll']-nll,one_visit_delta_nll=values['one_visit_nll']-nll,
                    future_visit_ce=[v/max(1,documents) for v in future],future_document_count=documents,
                    future_offset_ce=(offset_num/offset_den.clamp_min(1)).tolist(),future_offset_counts=offset_den.tolist(),panel_indices=indices,
                    aligned_ablation_delta_nll=values['no_aligned_nll']-nll,cross_pq_ablation_delta_nll=values['no_cross_pq_nll']-nll,
                    reader_ablation_delta_nll=values['no_reader_nll']-nll,
                    pooled_coordinate_std=float(np.mean(stds)),code_counts=codes.tolist(),
                    pq_segment_entropy=entropy.tolist(),pq_segment_effective_words=entropy.exp().tolist(),pq_slots=pq_n,
                    forecast_coordinate_std=float(np.mean(forecast_spread)) if forecast_spread else None,
                    zero_forecast_ce=zero_ce/documents if documents else None,wrong_group_forecast_ce=shuffled_ce/documents if documents else None,
                    wrong_group_control='Next complete group modulo document groups; may include later context; descriptive readout sensitivity, not causal inference')
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
                                     prompt=tok.decode(prefix,skip_special_tokens=False),expected=row.get('answer',row.get('solution')), 
                                     **{k:v for k,v in scoring.items() if k!='correct'},correct=bool(correct),reason=reason,generated_tokens=len(pred)))
            entry=dict(n=len(examples),correct=sum(x['correct'] for x in examples),planned_n=len(chosen),examples=examples,
                       identity_set_sha256=hashlib.sha256(json.dumps([meta[i]['identity'] for i in chosen]).encode()).hexdigest(),
                       metric='Complete exact numeric or ordered-list match / constraint-valid board, with EOS')
            if family=='math':entry['raw_exact_correct']=sum(x['raw_exact'] for x in examples)
            if family=='math':entry['by_module']={s:dict(n=sum(x['source']==s for x in examples),
                correct=sum(x['correct'] for x in examples if x['source']==s)) for s in sorted({r['source'] for r in meta})}
            result['generation'][family]=entry
        if math_per_module and (deadline is None or time.time()<deadline-30):
            result['objective_gradient_probe']=gradient_probe(model,device,data_dir)
        result.update(seconds=time.time()-started,budget_exhausted=bool(deadline and time.time()>=deadline))
        return result
    finally:model.train(was_training)


def gradient_probe(model,device,data_dir=DATA):
    """Fixed text probe; no optimizer step or .grad mutation. Dense gradients only."""
    corpus=Corpus('train',seed=39117,data_dir=data_dir)
    raw=corpus.batch(1,384,family='text',device='cpu')
    batch={k:raw[k].to(device) for k in ('ids','lengths','targets','target_mask')}
    readers={};sources={};handles=[]
    for i,reader in enumerate(model.decoder.readers):
        def hook(module,args,output,i=i):readers[i]=(output,args[0].detach())
        handles.append(reader.register_forward_hook(hook))
    for name,projection in [('continuous',model.decoder.concept_projection),('pq',model.decoder.forecast_projection)]:
        def source_hook(module,args,output,name=name):sources[name]=output.detach()
        handles.append(projection.register_forward_hook(source_hook))
    with torch.enable_grad():
        try:out=model(batch['ids'],batch['lengths'],indices=model.engram.indices(raw['ids']))
        finally:
            for h in handles:h.remove()
        parts=losses(model,out,**batch,candidate_update=True,generator=torch.Generator().manual_seed(39117))
        parameters=[p for n,p in model.named_parameters() if n!='engram.table.weight']
        objectives={'token':parts['token'],'weighted_future':.1*parts['future'],
                    'weighted_candidate':.1*parts['explore']+.05*parts['selector']}
        vectors=[];reader_stats={}
        for name,value in objectives.items():
            gradients=torch.autograd.grad(value,parameters+[readers[i][0] for i in sorted(readers)],allow_unused=True,retain_graph=True)
            if name=='token':
                for i,g in enumerate(gradients[len(parameters):]):
                    y,x=readers[i]
                    rms=y.detach().square().mean().sqrt();base=x.square().mean().sqrt()
                    reader_stats[str(i)]=dict(output_rms=float(rms),residual_rms=float(base),relative_rms=float(rms/base.clamp_min(1e-12)),
                        output_gradient_norm=float(g.norm()) if g is not None else 0.)
            vectors.append(torch.cat([(torch.zeros_like(p) if g is None else g.detach()).flatten() for p,g in zip(parameters,gradients)]))
        vectors=torch.stack(vectors);norms=vectors.norm(dim=1)
        cos=(vectors@vectors.T)/(norms[:,None]*norms[None,:]).clamp_min(1e-12)
        values=torch.cat((norms,cos.flatten())).detach().cpu().tolist()
    names=list(objectives)
    return dict(split='train',family='text',indices=raw['chosen'],seed=39117,
        meaning='Fixed diagnostic probe; dense only; before clipping; no optimizer update. Auxiliary coefficients included.',
        reader_layers=reader_stats,source_projection_rms={k:float(v.square().mean().sqrt()) for k,v in sources.items()},
        norms=dict(zip(names,values[:3])),cosines={a:{b:values[3+i*3+j] for j,b in enumerate(names)} for i,a in enumerate(names)})
