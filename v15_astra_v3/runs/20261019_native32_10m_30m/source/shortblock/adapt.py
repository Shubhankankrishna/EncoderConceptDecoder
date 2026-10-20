"""Bounded decoder-only adaptation; no codebook or quantizer-buffer updates."""
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..isolation import exclusive_job, atomic_json, sha256
from .codec import ShortNCMCodec, RECIPE
from .data import REPORT, windows, split_blocks


def accuracy(logits, ids, pad=0):
    predicted = logits.argmax(-1)
    mask = ids != pad
    last = mask.long().sum(-1).sub(1).clamp_min(0)
    edge = torch.stack([predicted[:,0] == ids[:,0],
        predicted.gather(1,last[:,None])[:,0] == ids.gather(1,last[:,None])[:,0]],1)
    return dict(correct=int(((predicted==ids)&mask).sum()), total=int(mask.sum()),
        exact=int(((predicted==ids)|~mask).all(-1).sum()), blocks=len(ids),
        edge_correct=int(edge.sum()), edge_total=edge.numel())


def merge(rows):
    sums = {k:sum(r[k] for r in rows) for k in rows[0]}
    return dict(nonpad_token_accuracy=sums['correct']/sums['total'],
        exact_nonpad_block_rate=sums['exact']/sums['blocks'],
        boundary_token_accuracy=sums['edge_correct']/sums['edge_total'],
        sampled_blocks=sums['blocks'])


@exclusive_job
def adapt(steps=1600, seconds=1200):
    torch.set_num_threads(2)
    torch.manual_seed(151803)
    torch.mps.set_per_process_memory_fraction(.5)
    REPORT.mkdir(parents=True, exist_ok=True)
    checkpoint = REPORT/'decoder32.pt'
    if (REPORT/'codec_acceptance.json').exists():
        result = json.loads((REPORT/'codec_acceptance.json').read_text())
        if result['passed'] and result['adaptation_sha'] == sha256(checkpoint):
            return result
    codec = ShortNCMCodec()
    raw = windows(codec)
    rng = np.random.default_rng(151803)
    train_pool, valid_pool = split_blocks(raw, 'train'), split_blocks(raw, 'valid')
    if set(train_pool)&set(valid_pool):
        raise ValueError('Adaptation train/validation blocks overlap')
    selected_train = rng.choice(train_pool, min(4096,len(train_pool)), replace=False)
    selected_valid = rng.choice(valid_pool, min(512,len(valid_pool)), replace=False)
    train_ids = torch.tensor(raw['block_ids'][selected_train])
    valid_ids = torch.tensor(raw['block_ids'][selected_valid])
    def latent(ids):
        rows=[]
        for start in range(0,len(ids),8):
            codes=codec.encode_ids(ids[start:start+8])
            rows.append(codec.model.vq.idx_to_z_q(codes).clone())
            if start%512==0:
                print(json.dumps(dict(event='fit_panel_encoding',completed=min(start+8,len(ids)),total=len(ids))),flush=True)
        return torch.cat(rows)
    atomic_json(REPORT/'adaptation_status.json', dict(status='encoding_fit_panel', timestamp=time.time(),
        train_blocks=len(train_ids), valid_blocks=len(valid_ids), test_blocks_used=0))
    train_z, valid_z = latent(train_ids), latent(valid_ids)
    decoder = codec.model.decoder
    def evaluate(device):
        decoder.eval()
        with torch.inference_mode():
            return merge([accuracy(decoder(z.to(device)),ids.to(device))
                for z,ids in zip(valid_z.split(16),valid_ids.split(16))])
    initial = evaluate('cpu')
    decoder.requires_grad_(True).to('mps').eval()
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=.01, foreach=False)
    best, start, best_metrics = -1., time.monotonic(), None
    with (REPORT/'adaptation_metrics.jsonl').open('a',buffering=1) as log:
        for step in range(1,steps+1):
            if time.monotonic()-start > seconds:
                break
            index=rng.choice(len(train_ids),16,replace=False)
            # Quantized latents and labels are training split only; adaptation
            # trains reconstruction, not the future GRAM generator.
            z, ids = train_z[index].to('mps'), train_ids[index].to('mps')
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(decoder(z).reshape(-1,30528),ids.reshape(-1),ignore_index=codec.pad)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            if step%100==0 or step==1:
                metrics=evaluate('mps')
                row=dict(step=step,loss=float(loss),grad_norm=float(norm),elapsed=time.monotonic()-start,
                    validation=metrics,timestamp=time.time())
                log.write(json.dumps(row)+'\n')
                atomic_json(REPORT/'adaptation_status.json',dict(status='decoder_adaptation',**row))
                print(json.dumps(row),flush=True)
                if metrics['nonpad_token_accuracy'] > best:
                    best,best_metrics=metrics['nonpad_token_accuracy'],metrics
                    payload=dict(recipe=RECIPE,decoder={k:v.detach().cpu() for k,v in decoder.state_dict().items()},
                        step=step,train_block_indices=selected_train.tolist(),valid_block_indices=selected_valid.tolist())
                    temp=checkpoint.with_suffix('.tmp')
                    with temp.open('wb') as f:
                        torch.save(payload,f);f.flush();os.fsync(f.fileno())
                    os.replace(temp,checkpoint)
                if metrics['nonpad_token_accuracy']>=.985 and metrics['boundary_token_accuracy']>=.97:
                    break
    del opt, decoder, codec, train_z, valid_z
    gc.collect();torch.mps.empty_cache()
    # Audit the saved/reloaded CPU codec, not just the in-memory MPS decoder.
    codec=ShortNCMCodec(adaptation=checkpoint)
    reloaded=merge([accuracy(codec.decode_logits(codec.encode_ids(ids)),ids)
        for ids in valid_ids.split(16)])
    passed=reloaded['nonpad_token_accuracy']>=.97 and reloaded['boundary_token_accuracy']>=.94
    result=dict(passed=passed,initial_native=initial,adapted=reloaded,
        minimum_nonpad_token_accuracy=.97,minimum_boundary_token_accuracy=.94,
        adaptation_sha=sha256(checkpoint),codec_id=codec.identity['codec_id'],
        train_blocks=len(selected_train),validation_blocks=len(selected_valid),test_blocks_used=0,
        training_scope='decoder only; pretrained encoder/codebooks/quantizer frozen',
        limitation='Validation used for decoder selection; not a sealed test or exact symbolic fidelity claim')
    atomic_json(REPORT/'codec_acceptance.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if not passed:
        raise RuntimeError('Native32 reconstruction acceptance failed; generator training is blocked')
    return result


if __name__ == '__main__':
    adapt()
