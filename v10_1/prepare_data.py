"""Clean train/dev in a new corpus, preserving original identities and partitions.

No sealed-test records are opened. Test stays reserved in the original corpus;
it must be re-encoded with this tokenizer only when final evaluation is authorized.
"""
import argparse
from array import array
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex
from v10_10M.prepare_data import digest, key, SPECIAL
from v10_10M.data_tools.clean_math import clean_math_field
from v10_10M.data import verify_sudoku

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'v10_10M/data'
DATA=ROOT/'v10_1/data'
FAMILIES=('text','math','sudoku_easy','sudoku_hard')


def records():
    result=[];needed={}
    for split in ('train','dev'):
        for family in FAMILIES:
            with (ORIGINAL/f'{family}_{split}_meta.jsonl').open() as f:
                for line in f:
                    row=json.loads(line)
                    row.update(split=split,family=family)
                    if family=='text':needed[row['identity']]=row;continue
                    if family=='math':
                        prompt=row['prompt']
                        if not prompt.startswith('Math: ') or not prompt.endswith('\nAnswer:'):
                            raise ValueError('Unexpected math prompt framing')
                        q=clean_math_field(prompt[6:-8]);a=clean_math_field(row['answer'])
                        row.update(prompt='Math: '+q+'\nAnswer:',answer=a,text=' '+a,normalized_identity=key(q))
                    else:
                        if not verify_sudoku(row['puzzle'],row['solution'])['valid']:
                            raise ValueError('Invalid reference board')
                        row.update(prompt='Sudoku:\n'+row['puzzle']+'\nSolution:\n',text=row['solution'],normalized_identity=key(row['puzzle']))
                    result.append(row)
    # Stream the source cache but decode text only for assigned train/dev records.
    source=ROOT/'v7_tiny/data/fineweb_edu.jsonl.gz'
    with gzip.open(source,'rt') as f:
        for line in f:
            record=json.loads(line);identity=record['normalized_sha256']
            if identity in needed:
                row=needed.pop(identity);row.update(prompt='',text=record['text'],normalized_identity=identity)
                result.append(row)
    if needed:raise ValueError('Missing assigned text documents')
    # Same-question clean collisions keep dev untouched and quarantine train.
    dev={(r['family'],r['normalized_identity']) for r in result if r['split']=='dev'}
    seen=set();kept=[];quarantine=[]
    for row in result:
        identity=(row['family'],row['normalized_identity']);unique=(row['split'],*identity)
        if (row['split']=='train' and identity in dev) or unique in seen:
            quarantine.append({k:row[k] for k in ('identity','normalized_identity','family','split')});continue
        seen.add(unique);kept.append(row)
    return kept,quarantine


def build(destination=DATA):
    destination=Path(destination)
    if destination.exists():raise ValueError('Refusing existing corpus destination')
    destination.mkdir(parents=True)
    rows,quarantine=records()
    print('Assigned clean records',dict(Counter((r['split'],r['family']) for r in rows)),flush=True)
    tokenizer=Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer=pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r'[0-9]'),behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=True)])
    tokenizer.decoder=decoders.ByteLevel()
    tokenizer.train_from_iterator((r['prompt']+r['text'] for r in rows if r['split']=='train'),
        trainers.BpeTrainer(vocab_size=8192,special_tokens=SPECIAL,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False))
    tokenizer.save(str(destination/'tokenizer.json'))
    assert tokenizer.get_vocab_size()==8192 and len(tokenizer.encode('123456789').ids)==9
    files={};summary={}
    for split in ('train','dev'):
        for family in FAMILIES:
            name=f'{family}_{split}';flat=array('H');offsets=[0];starts=[];excluded=0
            with (destination/f'{name}_meta.jsonl').open('w') as f:
                for row in rows:
                    if (row['split'],row['family'])!=(split,family):continue
                    prefix=tokenizer.encode(row['prompt']).ids;body=tokenizer.encode(row['text']).ids
                    seq=[1]+prefix+body+[2]
                    if family!='text' and len(seq)>385:excluded+=1;continue
                    if tokenizer.decode(prefix+body,skip_special_tokens=False)!=row['prompt']+row['text']:
                        raise ValueError('Tokenizer roundtrip failed')
                    flat.extend(seq);offsets.append(len(flat));starts.append(1+len(prefix))
                    f.write(json.dumps({k:v for k,v in row.items() if k not in ('text','split','family')})+'\n')
            np.save(destination/f'{name}.npy',np.asarray(flat,dtype=np.uint16))
            np.save(destination/f'{name}_offsets.npy',np.asarray(offsets,dtype=np.int64))
            np.save(destination/f'{name}_answer.npy',np.asarray(starts,dtype=np.int32))
            summary[name]=dict(records=len(starts),tokens=len(flat),excluded_oversize=excluded)
            print(name,summary[name],flush=True)
    (destination/'quarantine.json').write_text(json.dumps(quarantine,indent=2)+'\n')
    for p in sorted(destination.iterdir()):files[p.name]=digest(p)
    old=json.loads((ORIGINAL/'manifest.json').read_text())
    manifest=dict(version='v10.1-clean-identity-preserved',datasets=summary,files=files,
        tokenizer_sha256=digest(destination/'tokenizer.json'),vocab_size=8192,
        original_manifest_sha256=digest(ORIGINAL/'manifest.json'),original_provenance=old,
        normalization='Safe literal bytes decoding on math only; original identity and source split preserved',
        split_policy='Original train/dev assignments preserved; normalized train/dev collisions quarantined; no hash-based reassignment',
        test_policy='No test arrays produced or sealed-test metadata read; original test reserved for later re-encoding',
        quarantined=len(quarantine),tokenizer_fit='train only',fresh_weights_required=True,
        builder_sha256=digest(Path(__file__)))
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('CLEAN DATA READY',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(DATA));a=p.parse_args();build(a.output)
