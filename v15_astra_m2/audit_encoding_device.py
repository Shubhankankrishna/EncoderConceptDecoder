"""Check batched MPS encoding against already committed CPU cache entries."""
import json
import time

import numpy as np
import torch

from .codec import ROOT,FrozenNCMCodec,state_hash
from .data import DATA,atomic_json
from .tasks import parse_action


def main():
    torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.6)
    progress=json.loads((DATA/'encoding_progress.json').read_text());assert progress['device']=='cpu'
    codes=np.load(DATA/'codes.partial.npy',mmap_mode='r');texts=json.loads((DATA/'texts.json').read_text())
    rng=np.random.default_rng(1951);n=min(2048,progress['completed'])
    ix=np.r_[np.arange(7),rng.choice(np.arange(7,progress['completed']),n-7,replace=False)]
    codec=FrozenNCMCodec('mps',family='lm1b');before=state_hash(codec.model)
    times=[];mismatches=0;block_mismatches=0;outputs=[]
    for start in range(0,n,32):
        ii=ix[start:start+32];ids,_=codec.tokenize([texts[i] for i in ii]);tic=time.perf_counter()
        y=codec.encode_ids(ids).cpu().numpy();times.append(time.perf_counter()-tic)
        changed=y!=codes[ii];mismatches+=int(changed.sum());block_mismatches+=int(changed.any(1).sum())
        if start==0:outputs=y[:7]
    action_text=codec.decode_text(torch.from_numpy(outputs.astype(np.int64)))
    actions_ok=all(parse_action(t)==i for i,t in enumerate(action_text))
    report=dict(examples=n,cpu_committed_pool=progress['completed'],mismatched_codes=mismatches,
                code_fraction_changed=mismatches/(n*641),changed_blocks=block_mismatches,
                blocks_per_second=n/sum(times),batch=32,device='mps',dtype='float32',
                action_roundtrips=actions_ok,codec_state_unchanged=state_hash(codec.model)==before,
                codec=codec.identity,cpu_blocks_per_second=progress['blocks_per_second'],
                scope='fixed 2048-block CPU-reference subset, not proof of universal device equality')
    report['accepted']=bool(n==2048 and mismatches==0 and actions_ok and report['codec_state_unchanged'])
    atomic_json(ROOT/'reports/encoding_device_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='codec'}),flush=True)


if __name__=='__main__':main()
