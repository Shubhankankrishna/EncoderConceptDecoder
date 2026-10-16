"""Validate actual stored records and disjoint identities before launch."""
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from .prepare_data import DATA, digest
from .data import Corpus, verify_sudoku


def main():
    manifest=json.loads((DATA/'manifest.json').read_text())
    tokenizer=Tokenizer.from_file(str(DATA/'tokenizer.json'))
    assert digest(DATA/'tokenizer.json')==manifest['tokenizer_sha256']
    result={};identities={s:set() for s in ('train','dev','test')}
    for name, spec in manifest['datasets'].items():
        split=name.rsplit('_',1)[1]
        arr=np.load(DATA/f'{name}.npy',mmap_mode='r')
        off=np.load(DATA/f'{name}_offsets.npy');ans=np.load(DATA/f'{name}_answer.npy')
        meta=[json.loads(x) for x in (DATA/f'{name}_meta.jsonl').read_text().splitlines()]
        assert len(arr)==off[-1] and len(off)==len(ans)+1==len(meta)+1
        assert arr.max()<tokenizer.get_vocab_size() and np.all(np.diff(off)>0)
        assert digest(DATA/f'{name}.npy')==spec['sha256']
        hashes=[m['identity'] for m in meta]
        assert len(hashes)==len(set(hashes)),name+' contains duplicate instances'
        identities[split].update(hashes)
        lengths=np.diff(off)
        answer_lengths=lengths-ans-1
        if name.startswith('sudoku'):
            assert np.all(answer_lengths==81)
            for row in meta:
                assert verify_sudoku(row['puzzle'],row['solution'])['valid']
        result[name]=dict(records=len(meta),length_min=int(lengths.min()),length_median=float(np.median(lengths)),
                          length_max=int(lengths.max()),answer_length_median=float(np.median(answer_lengths)),
                          group_remainders=np.bincount(ans%8,minlength=8).tolist())
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        assert not identities[a]&identities[b],f'Identity leakage: {a}/{b}'
    for family in Corpus().arrays:
        batch=Corpus().batch(4,family=family)
        assert batch['target_mask'].any()
    report=dict(status='passed',identity_overlap=0,datasets=result,
                caveat='Identity/content checks do not prove semantic or Sudoku isomorphism deduplication beyond upstream guarantees.')
    path=DATA.parent/'research/data_audit.json';path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
