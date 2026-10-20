"""Versioned complete observations and read-only data access for new experiments."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .tasks import ACTION_TEXTS,parse_action
from .baselines import legal_actions
from .representation import serialize_observation,parse_observation,fixed_observed_blocks
from .isolation import ROOT,output_path,atomic_json,sha256,require_legacy_idle,exclusive_job


class Dataset:
    def __init__(self,directory):
        self.root=Path(directory).resolve()
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        if sha256(self.root/'episodes.jsonl')!=self.manifest['episodes_sha']:
            raise ValueError('Dataset changed after manifest creation')
        self.episodes=[json.loads(s) for s in (self.root/'episodes.jsonl').read_text().splitlines()]
        from .audit import validate_episodes
        validate_episodes(self.episodes)
        self.rows=[];self.indices={s:[] for s in ('train','valid','test')}
        for ei,e in enumerate(self.episodes):
            for ti,r in enumerate(e['trajectory']):
                self.indices[e['split']].append(len(self.rows));self.rows.append((ei,ti))
        self.indices={s:np.asarray(ix,dtype=np.int64) for s,ix in self.indices.items()}
        self.codes=None;self.target_codes=None;self.encoding=None
        if (self.root/'encoding_manifest.json').exists():
            self.encoding=json.loads((self.root/'encoding_manifest.json').read_text())
            if self.encoding['episodes_sha']!=self.manifest['episodes_sha']:raise ValueError('Encoding dataset mismatch')
            for name,h in self.encoding['files'].items():
                if sha256(self.root/name)!=h:raise ValueError(f'Encoding corruption: {name}')
            self.codes=np.load(self.root/'observed_codes.npy',mmap_mode='r')
            self.target_codes=np.load(self.root/'action_codes.npy',mmap_mode='r')
            if self.codes.shape!=(len(self.rows),self.encoding['context_blocks'],641):raise ValueError('Wrong encoded shape')
            if self.target_codes.shape!=(7,641):raise ValueError('Wrong action code shape')
            if self.encoding['codec']['vocab_sizes']!=[8192]*12:raise ValueError('Unexpected codec vocabulary')
            if not self.encoding['action_roundtrip_passed']:raise ValueError('Action fidelity failed')

    def record(self,index):
        ei,ti=self.rows[int(index)];e=self.episodes[ei]
        return e,e['trajectory'][ti]

    def batch(self,indices,device='cpu',require_codes=False):
        records=[self.record(i) for i in indices]
        actions=[r['action'] for e,r in records]
        state=torch.tensor([r['state'] for e,r in records],dtype=torch.long,device=device)
        optimal=torch.zeros(len(records),7,dtype=torch.bool,device=device)
        legal=optimal.clone()
        for i,(e,r) in enumerate(records):
            optimal[i,r['valid_actions']]=True;legal[i,legal_actions(e['env'],r['current'])]=True
        result=dict(state=state,action=torch.tensor(actions,dtype=torch.long,device=device),
                    optimal_mask=optimal,legal_mask=legal)
        if require_codes:
            if self.codes is None:raise ValueError('Complete observations must be encoded before a code model can train')
            result['x']=torch.tensor(np.asarray(self.codes[indices],dtype=np.int64),device=device)
            result['y']=torch.tensor(np.asarray(self.target_codes[actions],dtype=np.int64),device=device)
        return result


@exclusive_job
def encode_dataset(directory,batch=4):
    require_legacy_idle()
    from .codec import FrozenNCMCodec
    directory=output_path(directory);data=Dataset(directory)
    if (directory/'encoding_manifest.json').exists():raise FileExistsError('Encoding is immutable')
    if batch<1:raise ValueError('Positive batch required')
    codec=FrozenNCMCodec('cpu',family='lm1b');blocks=data.manifest['required_observed_blocks']
    # Fail on any overlength observation BEFORE creating output arrays.
    tokens=[]
    for i in range(len(data.rows)):
        e,r=data.record(i)
        text=serialize_observation(e['env'],r['current'],e['goal'])
        if text!=r['text']:raise ValueError('Incomplete or noncanonical observation')
        tokens.append(fixed_observed_blocks(codec.bpe.encode(text),blocks=blocks,pad_id=codec.pad))
    shape=(len(tokens),blocks,641);partial=directory/'observed_codes.partial.npy'
    array=np.lib.format.open_memmap(partial,mode='w+',dtype='<u2',shape=shape)
    semantic_exact=0
    for start in range(0,len(tokens),batch):
        require_legacy_idle()
        ids=torch.tensor(tokens[start:start+batch],dtype=torch.long)
        encoded=codec.encode_ids(ids.flatten(0,1))
        decoded=codec.decode_logits(encoded).argmax(-1).cpu().reshape(len(ids),blocks*128)
        for row,token_ids in enumerate(decoded.tolist()):
            e,r=data.record(start+row)
            try:
                recovered=parse_observation(codec.bpe.decode(token_ids,skip_special_tokens=True))
                semantic_exact+=recovered==parse_observation(r['text'])
            except (ValueError,TypeError,KeyError):pass
        codes=encoded.reshape(len(ids),blocks,641).cpu().numpy()
        array[start:start+len(ids)]=codes.astype('<u2')
        array.flush()
    ids,_=codec.tokenize(list(ACTION_TEXTS));target=codec.encode_ids(ids)
    if [parse_action(t) for t in codec.decode_text(target)]!=list(range(7)):
        raise ValueError('Canonical action roundtrip failed')
    np.save(directory/'action_codes.npy',target.cpu().numpy().astype('<u2'))
    del array;partial.rename(directory/'observed_codes.npy')
    manifest=dict(codec=codec.identity,episodes_sha=data.manifest['episodes_sha'],
                  context_blocks=blocks,complete_observation=True,encoding_device='cpu',
                  code_only_accepted=semantic_exact==len(tokens),
                  observation_semantic_roundtrip_rate=semantic_exact/len(tokens),
                  code_only_note='Acceptance requires exact parsed observations for every cached row; no broader domain fidelity claim.',
                  action_roundtrip_passed=True,
                  files={p:sha256(directory/p) for p in ('observed_codes.npy','action_codes.npy')})
    atomic_json(directory/'encoding_manifest.json',manifest)
    return manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('dataset',type=Path);ap.add_argument('--batch',type=int,default=4)
    args=ap.parse_args();torch.set_num_threads(1)
    print(json.dumps(encode_dataset(args.dataset,args.batch),indent=2))
