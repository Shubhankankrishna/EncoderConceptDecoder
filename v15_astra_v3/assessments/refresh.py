"""Refresh validation from saved checkpoints while the supervisor is idle."""
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..controls import make_model
from ..isolation import ROOT, atomic_json, exclusive_job, sha256
from ..shortblock.codec import load_codec
from ..shortblock.data import DATA, ShortDataset
from ..shortblock.evaluate import evaluate

RUN=ROOT/'runs/20261019_native32_10m_30m'
OUT=ROOT/'reports/quality_check_20261019'


def compact(result):
    return {k:v for k,v in result.items() if k not in ('records','indices')}


def unigram(data, indices):
    targets=[data.examples[int(i)]['target'] for i in data.indices['train']]
    train=np.asarray(data.codes[targets],dtype=np.int64)
    valid=np.asarray(data.codes[[data.examples[int(i)]['target'] for i in indices]],dtype=np.int64)
    offset=0;total=0.;positions=0
    for length in data.manifest['codec']['levels']:
        counts=np.bincount(train[:,offset:offset+length].reshape(-1),minlength=8192)+1
        logp=np.log(counts)-np.log(counts.sum())
        values=valid[:,offset:offset+length]
        total-=float(logp[values].sum());positions+=values.size;offset+=length
    return dict(nll_per_code=total/positions,definition='Add-one per-level code-frequency baseline, fitted on training target windows only')


@exclusive_job
def main():
    torch.set_num_threads(1);torch.mps.set_per_process_memory_fraction(.5)
    OUT.mkdir(exist_ok=True,parents=True)
    data=ShortDataset(DATA);codec=load_codec(data.manifest['codec'])
    result=dict(latest={},equal_step={},timestamp=time.time())
    for group in ('latest','equal_step'):
        for name in ('30m','10m'):
            directory=RUN/name/'gram'
            path=directory/('latest_inference.pt' if group=='latest' else 'equal_step_00001000_inference.pt')
            if not path.exists():continue
            ck=torch.load(path,map_location='cpu',weights_only=True)
            if ck['dataset_sha']!=data.manifest['episodes_sha']:raise ValueError('Checkpoint data identity changed')
            model=make_model('code_gram',ck['config'],inference=True).to('mps')
            model.load_state_dict(ck['model'],strict=True)
            observation=evaluate(model,data,codec,n=16,split='valid',chunks=4)
            state=json.loads((directory/'status.json').read_text())
            observation.update(step=ck['step'],timestamp=time.time(),checkpoint_sha=sha256(path),
                elapsed_seconds=state['elapsed_seconds'] if group=='latest' else None,
                observation_kind='read-only saved-checkpoint diagnostic; no optimizer/RNG/checkpoint-selection changes')
            atomic_json(OUT/f'{group}_{name}.json',observation)
            if group=='latest':atomic_json(directory/f"eval_{ck['step']:08d}.json",observation)
            result[group][name]=compact(observation)
            if 'frequency_baseline' not in result:result['frequency_baseline']=unigram(data,observation['indices'])
            print(json.dumps(dict(group=group,arm=name,**compact(observation))),flush=True)
            del model,ck;gc.collect();torch.mps.empty_cache()
    result['comparison_limit']='Latest snapshots have unequal data/time; only the step-1000 pair is matched in updates and sampled training examples.'
    atomic_json(OUT/'summary.json',result)


if __name__=='__main__':main()
