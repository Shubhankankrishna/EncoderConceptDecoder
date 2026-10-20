"""Offline native128 codec cache. Train and validation graph families stay separate."""
import json
import time

import numpy as np
import torch

from .audit import ordinary_texts
from .common import ROOT,atomic_json,heavy_job,sha
from .graphs import load_dataset,oracle_path
from .ncm import load_ncm,GENERATOR_SHA

WORDS='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo twentythree'.split()


def plan_text(path,strategy):
    return 'start at node '+WORDS[path[0]]+'. '+''.join('move to node '+WORDS[n]+'. ' for n in path[1:])+'the goal is reached.'


def main():
    torch.set_num_threads(2)
    target=ROOT/'data/ncm_cache';target.mkdir(exist_ok=True)
    if (target/'manifest.json').exists():
        load_cache();print('Existing cache verified');return
    net,codec,_=load_ncm('cpu')
    records=[]
    graphs=load_dataset()['graphs']
    quotas={'train':1024,'valid':96,'test':96,'ood':96}
    for split,limit in quotas.items():
        for i,g in enumerate([g for g in graphs if g['split']==split][:limit]):
            strategy=i%2;path=oracle_path(g,g['start'],strategy)
            records.append({'kind':'plan','split':split,'graph_id':g['id'],'path':path,'strategy':strategy,
                            'text':plan_text(path,strategy)})
    for i,text in enumerate(ordinary_texts(256)):
        records.append({'kind':'retention','split':'train' if i<192 else 'valid','path':[],'strategy':0,'text':text})
    n=len(records)
    raw=np.lib.format.open_memmap(target/'inputs.npy',mode='w+',dtype=np.float32,shape=(n,641,384))
    codes=np.lib.format.open_memmap(target/'codes.npy',mode='w+',dtype=np.int16,shape=(n,641))
    tokens=np.lib.format.open_memmap(target/'tokens.npy',mode='w+',dtype=np.int16,shape=(n,128))
    began=time.time()
    for start in range(0,n,8):
        rows=[]
        for record in records[start:start+8]:
            ids=codec.bpe.encode(record['text'])
            if record['kind']=='plan' and len(ids)>128:raise ValueError('Plan text exceeds unchanged codec window')
            ids=ids[:128];rows.append(ids+[codec.pad]*(128-len(ids)))
        ids=torch.tensor(rows)
        with torch.no_grad():
            idx=codec.encode_ids(ids)
            inp=net.teacher_inputs(idx)
        end=start+len(rows)
        codes[start:end]=idx.numpy();raw[start:end]=inp.numpy();tokens[start:end]=ids.numpy()
        if start%64==0:
            atomic_json(target/'progress.json',{'completed':end,'total':n,'seconds':time.time()-began,'timestamp':time.time()})
            print(f'cached {end}/{n}',flush=True)
    raw.flush();codes.flush();tokens.flush()
    atomic_json(target/'progress.json',{'completed':n,'total':n,'seconds':time.time()-began,'timestamp':time.time()})
    atomic_json(target/'records.json',records)
    atomic_json(target/'manifest.json',{'version':1,'graph_sha':sha(ROOT/'data/graphs.json'),
        'generator_sha':GENERATOR_SHA,'codec':codec.identity,'items':n,'quotas':quotas,
        'files':{name:sha(target/name) for name in ['inputs.npy','codes.npy','tokens.npy','records.json']},
        'inputs_dtype':'float32; identical to CPU canonical reconstruction',
        'retention_source':'FineWeb-Edu preexisting train document families only; not LM1B evaluation',
        'seconds':time.time()-began})
    print('cache complete',flush=True)


def load_cache(verify=True):
    root=ROOT/'data/ncm_cache'
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['graph_sha']!=sha(ROOT/'data/graphs.json'):raise ValueError('Graph/cache identity changed')
    if verify:
        for name,digest in manifest['files'].items():
            if sha(root/name)!=digest:raise ValueError(f'Cache corrupted: {name}')
    return {'manifest':manifest,'records':json.loads((root/'records.json').read_text()),
            **{name:np.load(root/f'{name}.npy',mmap_mode='r') for name in ('inputs','codes','tokens')}}


if __name__=='__main__':
    with heavy_job():main()
