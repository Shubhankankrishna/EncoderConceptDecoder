"""32-ID targets and eight preceding blocks, with document-family isolation."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from ..raw import raw_windows
from ..tasks import group_split
from ..codec import state_hash
from ..isolation import ROOT, LEGACY, atomic_json, sha256, exclusive_job
from .codec import ShortNCMCodec

SOURCE = ROOT / 'artifacts/fineweb_p0_20261018/documents.jsonl'
DATA = ROOT / 'artifacts/raw_native32_20261019'
REPORT = ROOT / 'reports/shortblock_20261019'


def exclude_probed_test_families(raw, groups):
    excluded=set(groups)
    before=len(raw['examples'])
    raw['examples']=[e for e in raw['examples'] if not (e['split']=='test' and e['group'] in excluded)]
    raw['excluded_probed_test_groups']=sorted(excluded)
    raw['excluded_probed_test_windows']=before-len(raw['examples'])
    return raw


def windows(codec):
    documents = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    # Preserve the legacy downstream exclusion, in addition to the curated
    # corpus's prior held-out host exclusions.
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
    legacy = json.loads((LEGACY/'artifacts/data/texts.json').read_text())
    forbidden = {hashlib.sha256(' '.join(s.split()).encode()).hexdigest() for s in strings(legacy)}
    raw=raw_windows(documents, codec.bpe.encode, history=8, block_size=32,
        pad_id=codec.pad, forbidden_hashes=forbidden, max_blocks=120000)
    scope=json.loads((REPORT/'probe_scope.json').read_text())
    return exclude_probed_test_families(raw,scope['exclude_test_groups'])


def split_blocks(raw, split):
    return np.asarray(sorted({i for e in raw['examples'] if e['split'] == split
        for i in e['context'] + [e['target']]}), dtype=np.int64)


class ShortDataset:
    def __init__(self, path=DATA):
        self.root = Path(path)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        identity = {k:v for k,v in self.manifest.items() if k != 'episodes_sha'}
        if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != self.manifest['episodes_sha']:
            raise ValueError('Short dataset manifest identity changed')
        if self.manifest['kind'] != 'native32_causal_windows_v1' or self.manifest['history'] != 8:
            raise ValueError('Wrong short-context contract')
        for name, digest in self.manifest['files'].items():
            if sha256(self.root/name) != digest:
                raise ValueError(f'Cached file changed: {name}')
        self.examples = json.loads((self.root/'examples.json').read_text())
        self.codes = np.load(self.root/'codes.npy', mmap_mode='r')
        self.token_ids = np.load(self.root/'token_ids.npy', mmap_mode='r')
        self.encoding = None
        if self.codes.shape != (self.manifest['unique_blocks'], sum(self.manifest['codec']['levels'])):
            raise ValueError('Wrong native code shape')
        if self.token_ids.shape != (len(self.codes), 32):
            raise ValueError('Wrong raw text shape')
        docs = {d['doc_id']:d for d in json.loads((self.root/'documents.json').read_text())}
        owners = {}
        valid_sequences = {}
        for d in docs.values():
            if group_split(d['group']) != d['split']:
                raise ValueError('Changed document-family split')
            for i in d['block_ids']:
                owners.setdefault(i, set()).add(d['split'])
            valid_sequences[d['doc_id']] = {tuple(d['block_ids'][i:i+9]) for i in range(len(d['block_ids'])-8)}
        for e in self.examples:
            d = docs[e['doc_id']]
            sequence = e['context'] + [e['target']]
            if len(sequence) != 9 or tuple(sequence) not in valid_sequences[e['doc_id']]:
                raise ValueError('Noncausal window')
            if e['split'] != d['split'] or e['group'] != d['group']:
                raise ValueError('Cross-document/split window')
            if any(len(owners[i]) != 1 for i in sequence):
                raise ValueError('Cross-split duplicate block')
            if e['split']=='test' and e['group'] in self.manifest.get('excluded_probed_test_groups',[]):
                raise ValueError('Previously probed test family is not eligible')
        self.indices = {s:np.asarray([i for i,e in enumerate(self.examples) if e['split']==s], dtype=np.int64)
            for s in ('train', 'valid', 'test')}
        if any(not len(x) for x in self.indices.values()):
            raise ValueError('Empty split')

    def batch(self, indices, device='cpu', require_codes=True):
        examples = [self.examples[int(i)] for i in indices]
        return dict(x=torch.tensor(np.asarray(self.codes[[e['context'] for e in examples]], dtype=np.int64), device=device),
            y=torch.tensor(np.asarray(self.codes[[e['target'] for e in examples]], dtype=np.int64), device=device))


@exclusive_job
def build(adaptation, out=DATA, batch=8):
    torch.set_num_threads(2)
    out = Path(out)
    codec = ShortNCMCodec(adaptation=adaptation)
    acceptance = json.loads((REPORT/'codec_acceptance.json').read_text())
    if not acceptance['passed'] or acceptance['adaptation_sha'] != sha256(adaptation) or acceptance['codec_id']!=codec.identity['codec_id']:
        raise ValueError('Codec acceptance is missing or changed')
    if (out/'manifest.json').exists():
        result = ShortDataset(out).manifest
        codec.assert_matches(result['codec'])
        return result
    raw = windows(codec)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(source_sha=sha256(SOURCE), codec_id=codec.identity['codec_id'],
        block_size=32, history=8, raw_ids_sha=hashlib.sha256(raw['block_ids'].tobytes()).hexdigest())
    progress_path, partial = out/'preparation.json', out/'codes.partial.npy'
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress['identity'] != identity:
            raise ValueError('Changed preparation cannot resume')
        if not partial.exists() and (out/'codes.npy').exists():
            (out/'codes.npy').rename(partial)
        codes = np.lib.format.open_memmap(partial, mode='r+')
        completed = progress['encoded_blocks']
        if hashlib.sha256(np.asarray(codes[:completed]).tobytes()).hexdigest() != progress['prefix_sha']:
            raise ValueError('Partial prefix changed')
    else:
        codes = np.lib.format.open_memmap(partial, mode='w+', dtype='<u2',
            shape=(len(raw['block_ids']), sum(codec.levels)))
        completed = 0
    if codes.shape != (len(raw['block_ids']), sum(codec.levels)):
        raise ValueError('Changed code shape')
    before = state_hash(codec.model)
    stopped = [False]
    def stop(*_):
        stopped[0] = True
    previous = {s:signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    start, initial, last = time.monotonic(), completed, 0.
    try:
        while completed < len(codes):
            end = min(len(codes), completed+batch)
            ids = torch.tensor(raw['block_ids'][completed:end])
            codes[completed:end] = codec.encode_ids(ids).numpy()
            completed = end
            if time.monotonic()-last > 15 or stopped[0] or completed == len(codes):
                codes.flush()
                progress = dict(identity=identity, encoded_blocks=completed, total_blocks=len(codes),
                    prefix_sha=hashlib.sha256(np.asarray(codes[:completed]).tobytes()).hexdigest(),
                    blocks_per_second=(completed-initial)/max(.001, time.monotonic()-start), timestamp=time.time())
                atomic_json(progress_path, progress)
                print(json.dumps(progress), flush=True)
                last = time.monotonic()
            if stopped[0]:
                raise SystemExit(75)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if state_hash(codec.model) != before:
        raise ValueError('Frozen codec mutated')
    del codes
    os.replace(partial, out/'codes.npy')
    np.save(out/'token_ids.npy', raw['block_ids'].astype('<u2'))
    for name in ('examples', 'documents'):
        (out/f'{name}.json').write_text(json.dumps(raw[name]))
    manifest = dict(kind='native32_causal_windows_v1', source_sha=sha256(SOURCE), codec=codec.identity,
        history=8, observed_context_ids=256, block_size=32, unique_blocks=len(raw['block_ids']),
        examples=len(raw['examples']), preparation_identity=identity, codec_state_unchanged=True,
        split_counts={s:sum(e['split']==s for e in raw['examples']) for s in ('train','valid','test')},
        decode_audit=acceptance['adapted'], codec_acceptance_sha=sha256(REPORT/'codec_acceptance.json'),
        cross_split_shared_blocks=raw['cross_split_shared_blocks'],
        removed_shared_block_windows=raw['removed_shared_block_windows'],
        excluded_probed_test_groups=raw['excluded_probed_test_groups'],
        excluded_probed_test_windows=raw['excluded_probed_test_windows'],
        probe_scope_sha=sha256(REPORT/'probe_scope.json'),
        boundary_rule='Tokenize each document once; native 32-ID chunks; eight complete observed chunks; PAD only document tail',
        limitations='Same small curated public corpus as predecessor; supplied family/duplicate groups, not an independent external benchmark',
        files={n:sha256(out/n) for n in ('codes.npy','token_ids.npy','examples.json','documents.json')})
    manifest['episodes_sha'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    atomic_json(out/'manifest.json', manifest)
    ShortDataset(out)
    return manifest


if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--adaptation', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=DATA)
    args=ap.parse_args()
    print(json.dumps(build(args.adaptation, args.out), indent=2))
