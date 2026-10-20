"""Curate the existing provenance-preserved FineWeb-Edu cache for P0.

Only prior training documents are eligible; prior dev/test hostnames are also
excluded. This is a fresh model and new hostname-grouped split of previously
cached public data, not a claim of a newly collected or untouched benchmark.
"""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from tokenizers import Tokenizer
from .isolation import ROOT,LEGACY,output_path,atomic_json,sha256
from .raw import raw_windows

SOURCE=ROOT.parent/'v7_tiny/data/fineweb_edu.jsonl.gz'
SOURCE_SHA='39b26d4184c42b497a11c89e58b3ff59dc30e340d4981614a50339b82e407dab'


def host(url):return (urlsplit(url).hostname or '').lower().removeprefix('www.')


def prepare(out,block_budget=24000,seed=151803):
    out=output_path(out)
    if out.exists():raise FileExistsError('Curated corpus is immutable')
    if sha256(SOURCE)!=SOURCE_SHA:raise ValueError('Existing FineWeb cache checksum changed')
    allowed={json.loads(s)['identity'] for s in (ROOT.parent/'v10_10M/data/text_train_meta.jsonl').read_text().splitlines()}
    excluded_hosts=set()
    for split in ('dev','test'):
        for line in (ROOT.parent/f'v10_10M/data/text_{split}_meta.jsonl').read_text().splitlines():
            excluded_hosts.add(host(json.loads(line)['source']))
    tokenizer=Tokenizer.from_file(str(LEGACY/'artifacts/codec/lm1b/bert-tokenizer/tokenizer.json'))
    candidates=[];reasons=Counter()
    with gzip.open(SOURCE,'rt') as stream:
        for line in stream:
            r=json.loads(line);h=host(r['url'])
            if r['normalized_sha256'] not in allowed:reasons['not_prior_train']+=1;continue
            if not h or h in excluded_hosts:reasons['prior_heldout_hostname']+=1;continue
            text=r['text']
            # Keep complete prose documents, not truncated code/equation pages.
            if any(s in text for s in ('```','#include','def __','public static','<script','SELECT *')):
                reasons['code_or_markup']+=1;continue
            if sum(c.isalpha() for c in text)/max(1,len(text))<.55:reasons['low_prose_fraction']+=1;continue
            ids=tokenizer.encode(text).ids
            if not 384<=len(ids)<=2048:reasons['length_outside_384_2048']+=1;continue
            key=hashlib.sha256(f'{seed}:{r["normalized_sha256"]}'.encode()).hexdigest()
            candidates.append((key,r,h,len(ids)))
    candidates.sort(key=lambda x:x[0]);docs=[];used=set();bands=defaultdict(set);sketches=[];blocks=0;per_host=Counter()
    for _,r,h,n in candidates:
        words=re.findall(r'\w+',r['text'].casefold());normalized=' '.join(words)
        digest=hashlib.sha256(normalized.encode()).hexdigest()
        if digest in used:reasons['normalized_duplicate']+=1;continue
        shingles={hashlib.blake2b(' '.join(words[i:i+5]).encode(),digest_size=8).digest() for i in range(len(words)-4)}
        sketch=tuple(sorted(shingles)[:64]);bucket_keys=[(i,sketch[i*4:(i+1)*4]) for i in range(16)]
        nearby=set().union(*(bands[k] for k in bucket_keys));near=False
        for j in nearby:
            previous=sketches[j]
            if len(set(sketch)&previous)/max(1,len(set(sketch)|previous))>=.8:near=True;break
        if near:reasons['lexical_near_duplicate']+=1;continue
        count=(n+127)//128
        if blocks+count>block_budget:continue
        if per_host[h]>=12:reasons['hostname_document_cap']+=1;continue
        used.add(digest);i=len(docs);sketches.append(set(sketch))
        for k in bucket_keys:bands[k].add(i)
        docs.append(dict(doc_id=r['id'],family='host:'+h,duplicate_cluster='normalized:'+digest,
            license='FineWeb-Edu database ODC-By; underlying webpage rights retained; see upstream dataset card',
            source=r['url'],text=r['text'],case_sensitive=False))
        per_host[h]+=1;blocks+=count
        if block_budget-blocks<3:break
    raw=raw_windows(docs,lambda text:tokenizer.encode(text).ids,max_blocks=block_budget)
    counts=Counter(e['split'] for e in raw['examples']);groups=Counter(d['split'] for d in raw['documents'])
    if min(counts.values(),default=0)<100 or set(counts)!={'train','valid','test'}:raise ValueError('Insufficient split coverage')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs))
    report=dict(source_dataset='HuggingFaceFW/fineweb-edu',source_url='https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu',
        upstream_license='ODC-By database; preserve source webpage attribution',source_path=str(SOURCE),source_sha=SOURCE_SHA,
        documents=len(docs),hostnames=len(per_host),unique_blocks=len(raw['block_ids']),windows=len(raw['examples']),
        document_splits=dict(groups),window_splits=dict(counts),seed=seed,block_budget=block_budget,
        curated_sha=sha256(out),pretokenizer_sha=sha256(LEGACY/'artifacts/codec/lm1b/bert-tokenizer/tokenizer.json'),
        excluded=dict(reasons),prior_holdout_hosts_excluded=len(excluded_hosts),
        selection='seeded hash order, only previous train identities, complete 384–2048-ID prose, max 12 docs per hostname',
        near_duplicate_method='casefolded word normalization; bottom-64 five-word shingle hashes, 16x4 LSH bands, sketch Jaccard >=0.8 rejection; not semantic dedup',
        case_scope='uncased prose prediction; no exact-case, code, mathematical-variable or accent fidelity claim',
        reuse_disclosure='Previously cached public data used by earlier local experiments; new v3 model and document-family split, not an independent external benchmark')
    atomic_json(out.with_suffix('.manifest.json'),report);return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--blocks',type=int,default=24000)
    args=ap.parse_args();print(json.dumps(prepare(args.out,args.blocks),indent=2))
