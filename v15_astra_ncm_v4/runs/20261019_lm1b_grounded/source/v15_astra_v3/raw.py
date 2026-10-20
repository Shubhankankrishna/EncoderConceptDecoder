"""P0: provenance-bearing documents -> unique frozen codes -> causal windows.

No corpus is bundled or downloaded. A supplied JSONL must document its license,
source, family, and near-duplicate cluster. Exact/whitespace duplicates are
merged before splitting. Semantic near-duplicate discovery remains an upstream
corpus curation requirement; the supplied cluster IDs are not a proven audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from .tasks import group_split
from .isolation import LEGACY,output_path,atomic_json,sha256,require_legacy_idle,exclusive_job


def raw_windows(documents,encode,*,history=2,block_size=128,pad_id=0,forbidden_hashes=(),max_blocks=20000):
    if min(history,block_size)<1:raise ValueError('Positive block/history lengths required')
    required={'doc_id','family','duplicate_cluster','license','source','text','case_sensitive'}
    parent={};seen_ids=set();digests={};docs=[]
    def root(x):
        parent.setdefault(x,x)
        while parent[x]!=x:x=parent[x]
        return x
    def union(a,b):
        a,b=root(a),root(b);parent[max(a,b)]=min(a,b)
    for d in documents:
        if set(d)!=required:raise ValueError('Raw rows require provenance only; action/state/reward fields are forbidden')
        if any(not isinstance(d[k],str) or not d[k].strip() for k in required-{'case_sensitive'}):
            raise ValueError('Nonempty provenance and document text required')
        if d['case_sensitive'] is not False:raise ValueError('Uncased LM1B codec cannot certify case-sensitive corpora')
        if d['doc_id'] in seen_ids:raise ValueError('Duplicate document ID')
        seen_ids.add(d['doc_id'])
        digest=hashlib.sha256(' '.join(d['text'].split()).encode()).hexdigest()
        if digest in forbidden_hashes:raise ValueError('Sealed downstream text found in raw corpus')
        union('family:'+d['family'],'duplicate:'+d['duplicate_cluster'])
        if digest in digests:union('family:'+d['family'],'family:'+digests[digest]['family'])
        else:digests[digest]=d
        docs.append((d,digest))
    blocks=[];block_lookup={};examples=[];catalog=[];used=set();short=0
    for d,digest in docs:
        if digest in used:continue
        used.add(digest);group=root('family:'+d['family']);split=group_split(group)
        # Official tokenizer special tokens occur once at document boundaries;
        # splitting never inserts future tokens or crosses a document boundary.
        ids=list(encode(d['text']));indices=[];lengths=[]
        for offset in range(0,len(ids),block_size):
            row=ids[offset:offset+block_size];lengths.append(len(row))
            key=tuple(row+[pad_id]*(block_size-len(row)))
            if key not in block_lookup:
                if len(blocks)>=max_blocks:raise ValueError('Raw pilot exceeds declared unique-block cap; prepare a separate larger budget')
                block_lookup[key]=len(blocks);blocks.append(key)
            indices.append(block_lookup[key])
        catalog.append(dict(doc_id=d['doc_id'],group=group,split=split,text_sha=digest,
                            source=d['source'],license=d['license'],block_ids=indices,actual_lengths=lengths))
        if len(indices)<=history:short+=1
        for t in range(history,len(indices)):
            examples.append(dict(doc_id=d['doc_id'],group=group,split=split,context=indices[t-history:t],target=indices[t]))
    # Boilerplate may survive document deduplication. Remove every window that
    # touches an exactly repeated text-ID block across family splits.
    ownership={}
    for d in catalog:
        for i in d['block_ids']:ownership.setdefault(i,set()).add(d['split'])
    shared={i for i,splits in ownership.items() if len(splits)>1}
    before=len(examples);examples=[e for e in examples if not shared.intersection(e['context']+[e['target']])]
    return dict(block_ids=np.asarray(blocks,dtype=np.int64).reshape(-1,block_size),
                examples=examples,documents=catalog,dropped_exact_duplicates=len(docs)-len(catalog),
                cross_split_shared_blocks=len(shared),removed_shared_block_windows=before-len(examples),
                documents_too_short=short,history=history,block_size=block_size)


@exclusive_job
def build_raw(source,out,history=2,batch=4,max_blocks=20000,audit_blocks=512):
    require_legacy_idle()
    from .codec import FrozenNCMCodec,state_hash
    source=Path(source);out=output_path(out)
    if (out/'manifest.json').exists():
        data=RawCodeDataset(out)
        if data.manifest['source_sha']!=sha256(source):raise ValueError('Completed raw cache has another source')
        return data.manifest
    if source.stat().st_size>64*1024**2:raise ValueError('P0 pilot source exceeds 64 MiB; split preparation into versioned batches')
    if min(batch,max_blocks,audit_blocks)<1:raise ValueError('Positive preparation budgets required')
    docs=[json.loads(s) for s in source.read_text().splitlines()]
    codec=FrozenNCMCodec('cpu',family='lm1b')
    # Exclude every exact legacy puzzle text from a supplied raw corpus.
    legacy_text=json.loads((LEGACY/'artifacts/data/texts.json').read_text())
    def strings(x):
        if isinstance(x,str):yield x
        elif isinstance(x,dict):
            for v in x.values():yield from strings(v)
        elif isinstance(x,list):
            for v in x:yield from strings(v)
    forbidden={hashlib.sha256(' '.join(t.split()).encode()).hexdigest() for t in strings(legacy_text)}
    from .isolation import ROOT
    for known in (ROOT/'artifacts').glob('*/episodes.jsonl'):
        for line in known.read_text().splitlines():
            for r in json.loads(line).get('trajectory',[]):
                forbidden.add(hashlib.sha256(' '.join(r['text'].split()).encode()).hexdigest())
    raw=raw_windows(docs,codec.bpe.encode,history=history,pad_id=codec.pad,forbidden_hashes=forbidden,max_blocks=max_blocks)
    if set(e['split'] for e in raw['examples'])!={'train','valid','test'}:
        raise ValueError('Raw corpus needs usable windows from train, valid and test families')
    identity=dict(source_sha=sha256(source),codec_id=codec.identity['codec_id'],history=history,batch=batch,
                  max_blocks=max_blocks,audit_blocks=audit_blocks,unique_blocks=len(raw['block_ids']),
                  raw_ids_sha=hashlib.sha256(raw['block_ids'].tobytes()).hexdigest())
    out.mkdir(parents=True,exist_ok=True);progress_file=out/'preparation.json';partial=out/'codes.partial.npy'
    if progress_file.exists():
        progress=json.loads(progress_file.read_text())
        if progress['identity']!=identity:raise ValueError('Cannot resume changed raw preparation')
        # A completed-array rename may have preceded the final manifest write.
        if not partial.exists() and (out/'codes.npy').exists():(out/'codes.npy').rename(partial)
        codes=np.lib.format.open_memmap(partial,mode='r+')
        start_at=progress['encoded_blocks'];exact=progress['exact'];token_correct=progress['token_correct'];token_total=progress['token_total'];audited=progress['audited']
        if codes.shape!=(len(raw['block_ids']),641):raise ValueError('Partial array shape mismatch')
        if hashlib.sha256(np.asarray(codes[:start_at]).tobytes()).hexdigest()!=progress['prefix_sha']:
            raise ValueError('Partial encoded prefix changed')
    else:
        codes=np.lib.format.open_memmap(partial,mode='w+',dtype='<u2',shape=(len(raw['block_ids']),641))
        start_at=exact=token_correct=token_total=audited=0
    audit_set=set(np.random.default_rng(151803).choice(len(codes),min(audit_blocks,len(codes)),replace=False).tolist())
    frozen_before=state_hash(codec.model);started=time.monotonic();stopping=[False];last_report=started
    def stop_requested(*_):stopping[0]=True
    old={sig:signal.signal(sig,stop_requested) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        for start in range(start_at,len(codes),batch):
            require_legacy_idle();ids=torch.tensor(raw['block_ids'][start:start+batch]);code=codec.encode_ids(ids)
            chosen=[i for i in range(len(ids)) if start+i in audit_set]
            if chosen:
                recon=codec.decode_logits(code[chosen]).argmax(-1).cpu();original=ids[chosen];mask=original!=codec.pad
                token_correct+=int(((recon==original)&mask).sum());token_total+=int(mask.sum())
                exact+=int(((recon==original)|~mask).all(-1).sum());audited+=len(chosen)
            codes[start:start+len(ids)]=code.cpu().numpy();codes.flush();completed=start+len(ids)
            if time.monotonic()-last_report>=15 or completed==len(codes) or stopping[0]:
                progress=dict(identity=identity,encoded_blocks=completed,total_blocks=len(codes),exact=exact,
                    token_correct=token_correct,token_total=token_total,audited=audited,
                    prefix_sha=hashlib.sha256(np.asarray(codes[:completed]).tobytes()).hexdigest(),
                    blocks_per_second=(completed-start_at)/max(.001,time.monotonic()-started),timestamp=time.time())
                atomic_json(progress_file,progress);print(json.dumps(progress),flush=True);last_report=time.monotonic()
            if stopping[0]:raise SystemExit(75)
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)
    if state_hash(codec.model)!=frozen_before:raise RuntimeError('Frozen codec state mutated')
    del codes;partial.rename(out/'codes.npy')
    np.save(out/'token_ids.npy',raw['block_ids'].astype('<u2'))
    (out/'documents.json').write_text(json.dumps(raw['documents'],indent=2))
    (out/'examples.json').write_text(json.dumps(raw['examples']))
    manifest=dict(kind='raw_concept_windows_v3',source_sha=sha256(source),codec=codec.identity,
        history=history,block_size=128,unique_blocks=len(raw['block_ids']),examples=len(raw['examples']),
        preparation_identity=identity,codec_state_unchanged=True,
        cross_split_shared_blocks=raw['cross_split_shared_blocks'],removed_shared_block_windows=raw['removed_shared_block_windows'],
        near_duplicate_scope='caller-supplied family and duplicate clusters; exact/whitespace duplicates merged',
        case_sensitive_supported=False,decode_audit=dict(exact_nonpad_block_rate=exact/max(1,audited),
            nonpad_token_accuracy=token_correct/max(1,token_total),sampled_blocks=audited,selection='seeded random unique-block panel; seed 151803'),
        boundary_rule='official tokenization once per document, consecutive 128-ID blocks, PAD final block; fixed W full-block context',
        files={name:sha256(out/name) for name in ('codes.npy','token_ids.npy','documents.json','examples.json')})
    # Generic conditional likelihood learning is allowed even with imperfect
    # reconstruction. Exact symbolic fidelity is explicitly NOT certified.
    manifest['episodes_sha']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    atomic_json(out/'manifest.json',manifest);return manifest


class RawCodeDataset:
    def __init__(self,path):
        self.root=Path(path);self.manifest=json.loads((self.root/'manifest.json').read_text())
        if self.manifest.get('kind')!='raw_concept_windows_v3':raise ValueError('Not a raw pretraining cache')
        identity={k:v for k,v in self.manifest.items() if k!='episodes_sha'}
        if hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()!=self.manifest['episodes_sha']:
            raise ValueError('Raw manifest identity changed')
        for name,h in self.manifest['files'].items():
            if sha256(self.root/name)!=h:raise ValueError('Raw cache identity changed')
        self.examples=json.loads((self.root/'examples.json').read_text())
        self.codes=np.load(self.root/'codes.npy',mmap_mode='r');self.encoding=None
        self.token_ids=np.load(self.root/'token_ids.npy',mmap_mode='r') if (self.root/'token_ids.npy').exists() else None
        if self.codes.shape!=(self.manifest['unique_blocks'],641):raise ValueError('Raw code shape mismatch')
        self.indices={s:np.asarray([i for i,e in enumerate(self.examples) if e['split']==s],dtype=np.int64) for s in ('train','valid','test')}
        docs={d['doc_id']:d for d in json.loads((self.root/'documents.json').read_text())}
        groups={}
        for d in docs.values():
            if d['group'] in groups and groups[d['group']]!=d['split']:raise ValueError('Raw family crosses splits')
            groups[d['group']]=d['split']
            if group_split(d['group'])!=d['split']:raise ValueError('Raw split assignment changed')
        for e in self.examples:
            d=docs[e['doc_id']];sequence=e['context']+[e['target']];w=self.manifest['history']
            if len(sequence)!=w+1 or e['split']!=d['split'] or e['group']!=d['group']:raise ValueError('Cross-document/split window')
            if not any(sequence==d['block_ids'][i:i+w+1] for i in range(len(d['block_ids'])-w)):
                raise ValueError('Noncausal or cross-document window')

    def batch(self,indices,device='cpu',require_codes=True):
        e=[self.examples[int(i)] for i in indices]
        # There are deliberately no action, state, outcome, or strategy fields.
        return dict(x=torch.tensor(np.asarray(self.codes[[v['context'] for v in e]],dtype=np.int64),device=device),
                    y=torch.tensor(np.asarray(self.codes[[v['target'] for v in e]],dtype=np.int64),device=device))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--history',type=int,default=2);ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--max-blocks',type=int,default=20000);ap.add_argument('--audit-blocks',type=int,default=512)
    args=ap.parse_args();torch.set_num_threads(1);print(json.dumps(build_raw(args.source,args.out,args.history,args.batch,args.max_blocks,args.audit_blocks),indent=2))
