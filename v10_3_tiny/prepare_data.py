"""Build the tiny corpus by re-tokenizing the clean V10.1 corpus.

Source: v10_1/data (read only). Its builder verified that the 8,192-token
tokenizer round-trips every record, so decoding the stored arrays recovers the
exact prompt and target text. A 1,024-token BPE with the same pre-tokenizer
(digits isolated, byte level) is fitted on the decoded train split only.

The math module HELD_OUT is removed from the pretraining 'math' family and
written as a separate 'newmath' family: it is the new phase for the sleep and
continual-learning tests. Identities, train/dev assignments and answer
boundaries are preserved. No sealed-test data exist in the source corpus.
"""
from array import array
import hashlib
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex
from v10_10M.prepare_data import SPECIAL

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'v10_1/data'
DATA = ROOT / 'v10_3_tiny/data'
FAMILIES = ('text', 'math', 'sudoku_easy', 'sudoku_hard')
HELD_OUT = 'algebra__linear_1d'
VOCAB, CONTEXT = 1024, 256


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def decoded_records(old):
    for split in ('train', 'dev'):
        for family in FAMILIES:
            base = SOURCE / f'{family}_{split}'
            arr, offsets, starts = np.load(str(base) + '.npy', mmap_mode='r'), np.load(str(base) + '_offsets.npy'), np.load(str(base) + '_answer.npy')
            meta = [json.loads(x) for x in (SOURCE / f'{family}_{split}_meta.jsonl').read_text().splitlines()]
            assert len(meta) == len(starts)
            for i, row in enumerate(meta):
                seq = [int(t) for t in arr[offsets[i]:offsets[i + 1]]]
                assert seq[0] == 1 and seq[-1] == 2
                a = int(starts[i])
                prompt = old.decode(seq[1:a], skip_special_tokens=False)
                text = old.decode(seq[a:-1], skip_special_tokens=False)
                target = family
                if family == 'math' and row['source'] == HELD_OUT: target = 'newmath'
                yield dict(split=split, family=target, prompt=prompt, text=text, meta=row)


def main():
    if DATA.exists(): raise SystemExit('Refusing existing ' + str(DATA))
    old = Tokenizer.from_file(str(SOURCE / 'tokenizer.json'))
    rows = list(decoded_records(old))
    print('decoded', len(rows), flush=True)
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(r'[0-9]'), behavior='isolated'),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
    tok.decoder = decoders.ByteLevel()
    fit = (r['prompt'] + r['text'] for r in rows if r['split'] == 'train' and r['family'] != 'newmath')
    tok.train_from_iterator(fit, trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIAL,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    assert tok.get_vocab_size() == VOCAB and len(tok.encode('123456789').ids) == 9
    assert [tok.id_to_token(i) for i in range(3)] == [old.id_to_token(i) for i in range(3)], 'pad/bos/eos ids must match'
    DATA.mkdir(parents=True); tok.save(str(DATA / 'tokenizer.json'))
    summary = {}
    for split in ('train', 'dev'):
        for family in FAMILIES + ('newmath',):
            sel = [r for r in rows if r['split'] == split and r['family'] == family]
            flat, offsets, starts, excluded = array('H'), [0], [], 0
            with (DATA / f'{family}_{split}_meta.jsonl').open('w') as f:
                for r in sel:
                    prefix, body = tok.encode(r['prompt']).ids, tok.encode(r['text']).ids
                    if tok.decode(prefix + body, skip_special_tokens=False) != r['prompt'] + r['text']:
                        raise ValueError('round trip failed')
                    seq = [1] + prefix + body + [2]
                    if family != 'text' and len(seq) > CONTEXT + 1: excluded += 1; continue
                    flat.extend(seq); offsets.append(len(flat)); starts.append(1 + len(prefix))
                    f.write(json.dumps(r['meta']) + '\n')
            name = f'{family}_{split}'
            np.save(DATA / f'{name}.npy', np.asarray(flat, dtype=np.uint16))
            np.save(DATA / f'{name}_offsets.npy', np.asarray(offsets, dtype=np.int64))
            np.save(DATA / f'{name}_answer.npy', np.asarray(starts, dtype=np.int32))
            summary[name] = dict(records=len(starts), tokens=len(flat), excluded_oversize=excluded)
            print(name, summary[name], flush=True)
    files = {p.name: digest(p) for p in sorted(DATA.iterdir())}
    manifest = dict(version='v10.3-tiny', datasets=summary, files=files, vocab_size=VOCAB, context=CONTEXT,
                    held_out_math_module=HELD_OUT, source_manifest_sha256=digest(SOURCE / 'manifest.json'),
                    tokenizer_fit='decoded train split, excluding the held-out module', builder_sha256=digest(Path(__file__)))
    (DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('READY', flush=True)


if __name__ == '__main__':
    main()
