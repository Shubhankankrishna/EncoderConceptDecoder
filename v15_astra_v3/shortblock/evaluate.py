"""Prior likelihood, shuffled-context control, and rolling decoded continuations."""
from collections import Counter
import re
import time

import numpy as np
import torch

from ..evaluate import evaluation_rng


def text_metrics(text, chunks):
    words = re.findall(r"\w+(?:['’]\w+)?", text.casefold())
    counts = Counter(words)
    triples = list(zip(words,words[1:],words[2:]))
    seams = []
    for left,right in zip(chunks,chunks[1:]):
        a,b=re.findall(r'\w+',left.casefold()),re.findall(r'\w+',right.casefold())
        seams.append(dict(left_tail=left[-100:],right_head=right[:100:],
            same_word_at_seam=bool(a and b and a[-1]==b[0])))
    return dict(words=len(words),dominant_word_fraction=max(counts.values(),default=0)/max(1,len(words)),
        distinct_word_fraction=len(counts)/max(1,len(words)),
        repeated_trigram_fraction=1-len(set(triples))/len(triples) if triples else 0.,
        duplicate_chunk_fraction=1-len(set(chunks))/len(chunks),
        same_word_seam_count=sum(s['same_word_at_seam'] for s in seams),seams=seams)


@torch.inference_mode()
def evaluate(model, data, codec, *, n=16, split='valid', seed=151803, chunks=4):
    codec.assert_matches(data.manifest['codec'])
    device=next(model.parameters()).device
    rng=np.random.default_rng(seed)
    pool=data.indices[split]
    selected=rng.choice(pool,min(n,len(pool)),replace=False)
    previous=model.training
    start=time.monotonic()
    records=[]
    with evaluation_rng(seed,device):
        try:
            model.eval()
            for position,index in enumerate(selected):
                e=data.examples[int(index)]
                alternate=next((int(j) for j in pool if data.examples[int(j)]['doc_id']!=e['doc_id']),None)
                if alternate is None:
                    raise ValueError('Context probe requires another document')
                batch=data.batch([index],device)
                wrong=data.batch([alternate],device)['x']
                # Identical random seeds within each true/shuffled pair reduce
                # noise in this context-sensitivity diagnostic.
                with evaluation_rng(seed+position,device):
                    likelihood=model.score_prior(batch['x'],None,batch['y'],samples=4)
                with evaluation_rng(seed+position,device):
                    shuffled=model.score_prior(wrong,None,batch['y'],samples=4)
                greedy=model.generate(batch['x'],None,temperature=0.)
                greedy_text=codec.decode_text(greedy.cpu())[0]
                context=batch['x'].clone()
                generated_text=[]; generated_ids=[]
                if device.type=='mps':torch.mps.synchronize()
                tic=time.perf_counter()
                for chunk in range(chunks):
                    codes=model.generate(context,None,temperature=.8)
                    ids=codec.decode_logits(codes.cpu()).argmax(-1)
                    generated_ids.extend(ids[0].tolist())
                    generated_text.append(codec.text_from_ids(ids)[0])
                    # Match the training context distribution: feed canonical
                    # codes of decoded text, rather than arbitrary sampled codes.
                    if chunk+1<chunks:
                        feedback=codec.encode_ids(ids).to(device)
                        context=torch.cat([context[:,1:],feedback[:,None]],1)
                if device.type=='mps':torch.mps.synchronize()
                elapsed=time.perf_counter()-tic
                special=set(codec.bpe.hf_tokenizer.all_special_ids)
                visible=sum(i not in special for i in generated_ids)
                text=codec.bpe.decode(generated_ids,skip_special_tokens=True)
                metrics=text_metrics(text,generated_text)
                records.append(dict(index=int(index),doc_id=e['doc_id'],group=e['group'],
                    nll_per_code=float(likelihood['nll']),shuffled_context_nll_per_code=float(shuffled['nll']),
                    context_nll_delta=float(shuffled['nll'])-float(likelihood['nll']),
                    observed_history=codec.bpe.decode(data.token_ids[e['context']].reshape(-1).tolist(),skip_special_tokens=True),
                    reference_pretokenized_text=codec.bpe.decode(data.token_ids[e['target']].tolist(),skip_special_tokens=True),
                    reference_codec_reconstruction=codec.decode_text(batch['y'].cpu())[0],
                    free_generated_text=text,greedy_first_chunk=greedy_text,generated_chunks=generated_text,
                    generation_seconds=elapsed,generated_content_tokens=visible,
                    decoded_positions_per_second=(chunks*32)/max(elapsed,1e-9),
                    end_to_end_content_tokens_per_second=visible/max(elapsed,1e-9),**metrics))
            total_seconds=sum(r['generation_seconds'] for r in records)
            return dict(split=split,seed=seed,indices=selected.tolist(),dataset_sha=data.manifest['episodes_sha'],
                codec_id=codec.identity['codec_id'],records=records,
                nll_per_code=float(np.mean([r['nll_per_code'] for r in records])),
                context_relevance=dict(mean_shuffled_minus_true_nll=float(np.mean([r['context_nll_delta'] for r in records])),
                    definition='Positive shuffled-context NLL delta indicates predictive context sensitivity; not a semantic relevance score'),
                repetition=dict(mean_dominant_word_fraction=float(np.mean([r['dominant_word_fraction'] for r in records])),
                    mean_repeated_trigram_fraction=float(np.mean([r['repeated_trigram_fraction'] for r in records]))),
                boundary_continuity=dict(same_word_seams=sum(r['same_word_seam_count'] for r in records),
                    seam_count=len(records)*(chunks-1),
                    definition='Descriptive seam repetition plus saved adjacent text for human review; no automatic coherence claim'),
                end_to_end_content_tokens_per_second=sum(r['generated_content_tokens'] for r in records)/max(total_seconds,1e-9),
                decoded_positions_per_second=len(records)*chunks*32/max(total_seconds,1e-9),
                generation_settings=dict(temperature=.8,chunks=chunks,positions_per_chunk=32,context_positions=256,
                    feedback='decode text then re-encode into canonical context codes'),
                nll_kind='prior_MC_logmeanexp_4_estimate',seconds=time.monotonic()-start,
                note='Code NLL is not text perplexity; throughput includes generator, CPU decode and intermediate feedback encoding')
        finally:
            model.train(previous)
