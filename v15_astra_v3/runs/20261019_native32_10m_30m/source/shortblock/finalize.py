"""Fixed end-of-budget comparison; sealed test cannot select checkpoints."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from ..controls import make_model
from ..isolation import atomic_json, exclusive_job, sha256
from .codec import load_codec
from .data import DATA, ShortDataset
from .evaluate import evaluate


@exclusive_job
def finalize(run):
    torch.set_num_threads(1);torch.mps.set_per_process_memory_fraction(.5)
    data=ShortDataset(DATA);codec=load_codec(data.manifest['codec'])
    results={}
    deadline=json.loads((run/'plan.json').read_text())['deadline']
    for name in ('30m','10m'):
        arm=run/name
        checkpoint=arm/'gram/best_nll_inference.pt'
        if not checkpoint.exists():checkpoint=arm/'gram/latest_inference.pt'
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if ck['dataset_sha']!=data.manifest['episodes_sha']:raise ValueError('Dataset changed')
        model=make_model('code_gram',ck['config'],inference=True).to('mps')
        model.load_state_dict(ck['model'],strict=True)
        for split in ('valid','test'):
            path=arm/f'final_{split}.json'
            if path.exists():continue
            if time.time()>deadline-30:raise RuntimeError('Evaluation deadline reached')
            result=evaluate(model,data,codec,n=32,split=split)
            result.update(step=ck['step'],timestamp=time.time(),checkpoint_sha=sha256(checkpoint))
            atomic_json(path,result)
        results[name]={split:json.loads((arm/f'final_{split}.json').read_text()) for split in ('valid','test')}
        del model,ck;gc.collect();torch.mps.empty_cache()
    shared=set(p.name for p in (run/'30m/gram').glob('equal_step_*_inference.pt')) & set(
        p.name for p in (run/'10m/gram').glob('equal_step_*_inference.pt'))
    equal_step={}
    if shared and time.time()<deadline-120:
        milestone=sorted(shared)[-1]
        for name in ('30m','10m'):
            ck=torch.load(run/name/'gram'/milestone,map_location='cpu',weights_only=True)
            model=make_model('code_gram',ck['config'],inference=True).to('mps')
            model.load_state_dict(ck['model'])
            result=evaluate(model,data,codec,n=16,split='valid',chunks=2)
            result.update(step=ck['step'],timestamp=time.time())
            atomic_json(run/name/'equal_step_valid.json',result)
            equal_step[name]=result
            del model,ck;gc.collect();torch.mps.empty_cache()
    summary=dict(completed=True,results={name:{split:{k:v for k,v in value.items() if k!='records'}
        for split,value in splits.items()} for name,splits in results.items()},
        equal_step_validation={name:{k:v for k,v in value.items() if k!='records'} for name,value in equal_step.items()},
        limitations=['One seed and small reused corpus','Context sensitivity and seam repetition are diagnostic proxies, not semantic quality certification',
            'Predecessor used a different codec/target/context recipe, so its code NLL is not directly comparable'])
    atomic_json(run/'comparison.json',summary)
    (run/'FINAL_REPORT.md').write_text('# Native32 GRAM paired experiment\n\n'+
        'The matched 10M and 30M outcomes, including held-out code likelihood, shuffled-context sensitivity, repetition, '+
        'saved boundary text and end-to-end throughput are recorded in comparison.json. '+
        'Per-example continuations are in each arm\'s final_valid.json and final_test.json. '+
        'These diagnostics do not establish general language capability.\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    finalize(ap.parse_args().run)
