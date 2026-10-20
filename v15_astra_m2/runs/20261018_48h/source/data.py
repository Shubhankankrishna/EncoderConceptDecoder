"""Batch-build verified trajectories before training; immutable mmap code cache."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .codec import ROOT, FrozenNCMCodec, sha256
from .tasks import ACTION_TEXTS, VERSION, make_episodes, step, distances, optimal_actions

DATA=ROOT/'artifacts/data'


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,path)


def generate(max_records=140000):
    DATA.mkdir(parents=True,exist_ok=True)
    if (DATA/'source_manifest.json').exists():return
    episodes=list(make_episodes())
    np.random.default_rng(1518).shuffle(episodes)
    rows=[];texts=list(ACTION_TEXTS);indices={t:i for i,t in enumerate(texts)};kept=[]
    for e in episodes:
        if len(rows)>=max_records and e['split'] not in ('ood',):continue
        ei=len(kept);kept.append(e)
        dist=distances(e['env'],e['goal'])
        for j,r in enumerate(e['trajectory']):
            assert step(e['env'],r['current'],r['action'])==r['next']
            assert r['action'] in optimal_actions(e['env'],r['current'],dist)
            if r['text'] not in indices:
                indices[r['text']]=len(texts);texts.append(r['text'])
            rows.append((indices[r['text']],r['action'],r['state'],r['action'],
                         ('train','valid','test','ood').index(e['split']),ei,j))
    # Verify rule/topology groups never cross splits, including all trace steps.
    groups={}
    for e in kept:
        assert e['group'] not in groups or groups[e['group']]==e['split']
        groups[e['group']]=e['split']
    (DATA/'texts.json').write_text(json.dumps(texts))
    with open(DATA/'episodes.jsonl','w') as f:
        for e in kept:f.write(json.dumps(e)+'\n')
    np.savez(DATA/'samples.npz',x=np.array([r[0] for r in rows],dtype='<i4'),
             y=np.array([r[1] for r in rows],dtype='<i4'),
             state=np.array([r[2] for r in rows],dtype='u1'),
             action=np.array([r[3] for r in rows],dtype='u1'),
             split=np.array([r[4] for r in rows],dtype='u1'),
             episode=np.array([r[5] for r in rows],dtype='<i4'),
             transition=np.array([r[6] for r in rows],dtype='u1'))
    manifest=dict(version=VERSION,seed=1518,records=len(rows),unique_blocks=len(texts),
                  episodes=len(kept),groups=len(groups),group_splits=groups,
                  split_counts={s:sum(r[4]==i for r in rows) for i,s in enumerate(('train','valid','test','ood'))},
                  source='locally generated exact deterministic environments',license='repository MIT',
                  source_sha=sha256(ROOT/'tasks.py'),state_visibility='current_only',
                  action_boundary='first complete canonical sentence; one action per block',
                  target_length_input=False,context_blocks=1,
                  files={n:sha256(DATA/n) for n in ('texts.json','episodes.jsonl','samples.npz')})
    atomic_json(DATA/'source_manifest.json',manifest)
    print('generated',manifest['records'],manifest['unique_blocks'],manifest['split_counts'],flush=True)


def build(device='cpu',batch=32):
    audit=json.loads((ROOT/'reports/codec_audit.json').read_text())
    assert audit['decision']['action_codec_accepted']
    generate()
    if (DATA/'manifest.json').exists():return
    texts=json.loads((DATA/'texts.json').read_text())
    codec=FrozenNCMCodec(device,family='lm1b')
    source=json.loads((DATA/'source_manifest.json').read_text())
    progress=DATA/'encoding_progress.json'
    if progress.exists():
        p=json.loads(progress.read_text());assert p['codec_id']==codec.identity['codec_id']
        assert p['batch']==batch and p['device']==device
        done=p['completed']
        # Recovery if interrupted after the complete array was renamed, before
        # its final manifest was committed.
        array_path=DATA/('codes.partial.npy' if (DATA/'codes.partial.npy').exists() else 'codes.npy')
        codes=np.load(array_path,mmap_mode='r+')
        assert codes.shape==(len(texts),641)
        if array_path.name=='codes.npy':assert done==len(texts)
    else:
        done=0;codes=np.lib.format.open_memmap(DATA/'codes.partial.npy',mode='w+',dtype='<u2',shape=(len(texts),641))
    t=time.perf_counter();start=done
    for i in range(done,len(texts),batch):
        ids,_=codec.tokenize(texts[i:i+batch]);y=codec.encode_ids(ids).cpu().numpy()
        assert y.shape==(len(ids),641) and y.min()>=0 and y.max()<8192
        codes[i:i+len(y)]=y.astype('<u2')
        if i//batch%16==0 or i+len(y)==len(texts):
            codes.flush()
            p=dict(completed=i+len(y),total=len(texts),codec_id=codec.identity['codec_id'],device=device,batch=batch,
                   blocks_per_second=(i+len(y)-start)/max(time.perf_counter()-t,.001))
            atomic_json(progress,p);print(json.dumps(p),flush=True)
    codes.flush();del codes
    if (DATA/'codes.partial.npy').exists():os.replace(DATA/'codes.partial.npy',DATA/'codes.npy')
    source['codec']=codec.identity;source['code_sha']=sha256(DATA/'codes.npy')
    source['dtype']='<u2';source['encoding_device']=device;source['encoding_batch']=batch
    source['encoding_seconds_this_invocation']=time.perf_counter()-t
    atomic_json(DATA/'manifest.json',source)


class Dataset:
    def __init__(self,root=DATA):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.codes=np.load(self.root/'codes.npy',mmap_mode='r')
        self.rows=dict(np.load(self.root/'samples.npz'))
        self.indices={s:np.flatnonzero(self.rows['split']==i) for i,s in enumerate(('train','valid','test','ood'))}
        self.episodes=[json.loads(x) for x in (self.root/'episodes.jsonl').read_text().splitlines()]

    def batch(self,indices,device):
        r=self.rows
        return tuple(torch.from_numpy(np.array(a,dtype=np.int64)).to(device) for a in
                     (self.codes[r['x'][indices]],r['state'][indices],self.codes[r['y'][indices]]))

    def verify_hashes(self):
        for n,h in self.manifest['files'].items():assert sha256(self.root/n)==h,n
        assert sha256(self.root/'codes.npy')==self.manifest['code_sha']


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--generate-only',action='store_true');args=ap.parse_args();torch.set_num_threads(4)
    generate() if args.generate_only else build(args.device,args.batch)
