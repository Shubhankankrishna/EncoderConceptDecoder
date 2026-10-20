"""P1 evaluation and retention: same raw panel, prior likelihood and free text."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .gram import AstraGRAM
from .controls import make_model
from .raw import RawCodeDataset
from .evaluate import evaluation_rng
from .isolation import require_legacy_idle,exclusive_job,atomic_json,sha256


@torch.inference_mode()
def evaluate_raw(model,data,codec,*,indices=None,split='valid',n=32,seed=151803,batch=2):
    if model.config.context_source!='codes':raise ValueError('Raw evaluation has no typed state')
    codec.assert_matches(data.manifest['codec']);device=next(model.parameters()).device
    pool=data.indices[split];rng=np.random.default_rng(seed)
    indices=np.asarray(indices if indices is not None else rng.choice(pool,min(n,len(pool)),replace=False),dtype=np.int64)
    if not len(indices) or not set(indices).issubset(set(pool)):raise ValueError('Panel is empty or crosses split')
    was_training=model.training;records=[];start=time.perf_counter()
    with evaluation_rng(seed,device):
        try:
            model.eval()
            for offset in range(0,len(indices),batch):
                ix=indices[offset:offset+batch];b=data.batch(ix,device)
                # Per-example likelihood records permit regrouping by document.
                for row,index in enumerate(ix):
                    x,y=b['x'][row:row+1],b['y'][row:row+1]
                    stats=model.score_prior(x,None,y,samples=4) if isinstance(model,AstraGRAM) else model(x,None,y)[1]
                    generated=model.generate(x,None)
                    records.append(dict(index=int(index),doc_id=data.examples[int(index)]['doc_id'],
                        group=data.examples[int(index)]['group'],nll_per_code=float(stats['nll']),
                        free_generated_text=codec.decode_text(generated.cpu())[0],
                        reference_codec_reconstruction=codec.decode_text(y.cpu())[0],
                        exact_code_match=bool(torch.equal(generated,y))))
                    if data.token_ids is not None:
                        example=data.examples[int(index)]
                        records[-1]['observed_history']=codec.bpe.decode(data.token_ids[example['context']].reshape(-1).tolist(),skip_special_tokens=True)
                        records[-1]['reference_pretokenized_text']=codec.bpe.decode(data.token_ids[example['target']].tolist(),skip_special_tokens=True)
            return dict(split=split,seed=seed,indices=indices.tolist(),dataset_sha=data.manifest['episodes_sha'],
                records=records,nll_per_code=float(np.mean([r['nll_per_code'] for r in records])),
                nll_per_block=float(np.mean([r['nll_per_code'] for r in records]))*model.offsets[-1],
                nll_kind='prior_MC_logmeanexp_4_estimate' if isinstance(model,AstraGRAM) else 'conditional_code_NLL',
                seconds=time.perf_counter()-start,note='Code likelihood is not text perplexity; no fabricated action metrics')
        finally:model.train(was_training)


@exclusive_job
def run(args):
    require_legacy_idle();torch.set_num_threads(1)
    from .codec import FrozenNCMCodec
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True);data=RawCodeDataset(args.data)
    if ck['manifest_identity']['codec_id']!=data.manifest['codec']['codec_id']:raise ValueError('Transfer changed codec')
    # A preserved P1 validation panel may evaluate a later compatible backbone.
    config={**ck['config'],'context_source':'codes'}
    model=make_model(ck['arm'],config,inference=ck.get('posterior_included') is False)
    model.load_state_dict(ck['model'],strict=True);model.to(args.device)
    panel=None
    if args.panel:
        previous=json.loads(args.panel.read_text())
        if previous['dataset_sha']!=data.manifest['episodes_sha']:raise ValueError('Retention panel corpus changed')
        panel=previous['indices']
    result=evaluate_raw(model,data,FrozenNCMCodec('cpu',family='lm1b'),indices=panel,split=args.split,n=args.n)
    result['checkpoint_sha']=sha256(args.checkpoint);atomic_json(args.out,result)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint',type=Path);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--panel',type=Path)
    ap.add_argument('--split',choices=['valid','test'],default='valid');ap.add_argument('--n',type=int,default=32)
    ap.add_argument('--device',choices=['cpu','mps'],default='cpu');run(ap.parse_args())
