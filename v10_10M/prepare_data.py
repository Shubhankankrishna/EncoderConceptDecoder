"""Prepare independently sourced text/math/Sudoku, with provenance and sealed tests.

Only data bytes are fetched; no remote dataset Python is executed.
"""
import csv
import gzip
import hashlib
import io
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, Regex

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data'
MATH_REV = 'ee37b6c8f92cceb4e32abae57ed6fd35a0863845'
SUDOKU_REV = '58942f96baeb572ca3127e2a9e9c70f330783d6b'
MODULES = ('arithmetic__add_sub_multiple', 'arithmetic__mul', 'algebra__linear_1d', 'comparison__sort')
SPECIAL = ['<pad>', '<bos>', '<eos>', '<unk>']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def key(s):
    return hashlib.sha256(re.sub(r'\s+', ' ', s.strip()).encode()).hexdigest()


def download(url, path):
    if path.exists():
        return path
    tmp = path.with_suffix('.partial')
    with requests.get(url, stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for b in r.iter_content(1 << 20):
                f.write(b)
    tmp.replace(path)
    return path


def sudoku_records(split, count):
    # The upstream corpus is already shuffled and separates equivalent puzzles
    # across official train/test. Preserve those partitions before any transform.
    path = DATA/'raw'/f'sudoku_{split}_{count}.jsonl'
    if path.exists():
        return [json.loads(x) for x in path.read_text().splitlines()]
    url = f'https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/{SUDOKU_REV}/{split}.csv'
    result = []
    with requests.get(url, stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        lines = (s.decode('utf-8') for s in r.iter_lines() if s)
        reader = csv.reader(lines)
        next(reader)
        for source, puzzle, solution, rating in reader:
            puzzle = puzzle.replace('.', '0')
            if len(puzzle) != 81 or len(solution) != 81:
                raise ValueError('Unexpected Sudoku schema')
            result.append(dict(source=source, puzzle=puzzle, solution=solution, rating=int(rating)))
            if len(result) >= count:
                break
    if len(result) != count:
        raise ValueError('Incomplete Sudoku download')
    path.write_text(''.join(json.dumps(x)+'\n' for x in result))
    return result


def text_records():
    source = ROOT.parent/'v7_tiny/data/fineweb_edu.jsonl.gz'
    if not source.exists():
        raise FileNotFoundError('Expected provenance-preserved local FineWeb-Edu cache: '+str(source))
    # Preserve original educational-text partitions, including lexical near-dedup.
    mappings = {}
    for split in ('train', 'dev', 'test'):
        meta = json.loads((source.parent/f'edu_{split}_documents.json').read_text())
        for h in meta['hashes']:
            mappings[h] = split
    result, seen = [], set()
    with gzip.open(source, 'rt') as f:
        for line in f:
            x = json.loads(line)
            if x['normalized_sha256'] in mappings and x['normalized_sha256'] not in seen:
                seen.add(x['normalized_sha256'])
                result.append(dict(text=x['text'], prompt='', split=mappings[x['normalized_sha256']],
                                   family='text', source=x['url'], identity=x['normalized_sha256']))
    return result, dict(path=str(source), sha256=digest(source), original_manifest=json.loads((source.parent/'manifest.json').read_text()))


def main():
    DATA.mkdir(exist_ok=True)
    (DATA/'raw').mkdir(exist_ok=True)
    records, text_provenance = text_records()
    print(f'Text: {len(records)} source documents', flush=True)
    jobs = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        sud_train = pool.submit(sudoku_records, 'train', 50000)
        sud_test = pool.submit(sudoku_records, 'test', 2048)
        for module in MODULES:
            for split in ('train', 'test'):
                url = f'https://huggingface.co/datasets/deepmind/math_dataset/resolve/{MATH_REV}/{module}/math_dataset-{split}.parquet'
                path = DATA/'raw'/f'{module}_{split}.parquet'
                jobs.append((module, split, url, pool.submit(download, url, path)))
        provenance = []
        seen_math = set()
        # Process official test first, excluding duplicated questions from train.
        for module, split, url, future in sorted(jobs, key=lambda z: z[1] != 'test'):
            path = future.result()
            table = pq.read_table(path)
            n = len(table)
            rng = np.random.default_rng(1011)
            rows = rng.choice(n, min(n, 60000 if split == 'train' else 2000), replace=False)
            sample = table.take(rows).to_pylist()
            for row in sample:
                q, a = row['question'], row['answer']
                if isinstance(q, bytes): q = q.decode()
                if isinstance(a, bytes): a = a.decode()
                identity = key(q)
                if identity in seen_math:
                    continue
                seen_math.add(identity)
                dest = 'train' if split == 'train' else ('dev' if int(identity[:8], 16)%2 else 'test')
                records.append(dict(prompt=f'Math: {q}\nAnswer:', text=f' {a}', family='math',
                                    split=dest, source=module, identity=identity, answer=a))
            provenance.append(dict(url=url, sha256=digest(path), rows=n, sampled=len(sample)))
            print(f'Math: {module}/{split}, {len(sample)} rows', flush=True)
        for split, source_rows in [('train', sud_train.result()), ('test', sud_test.result())]:
            for row in source_rows:
                identity = key(row['puzzle'])
                dest = 'train' if split == 'train' else ('dev' if int(identity[:8], 16)%2 else 'test')
                # Native easy/hard mixture; no invented compression-aligned tasks.
                easy = row['source'].startswith(('puzzles0_', 'puzzles1_', 'puzzles2_'))
                records.append(dict(prompt='Sudoku:\n'+row['puzzle']+'\nSolution:\n', text=row['solution'],
                                    family='sudoku_easy' if easy else 'sudoku_hard', split=dest,
                                    identity=identity, **row))
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    # Isolated digits preserve the 81-cell board and multi-digit arithmetic.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r'[0-9]'), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=8192, special_tokens=SPECIAL,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator((r['prompt']+r['text'] for r in records if r['split']=='train'), trainer)
    tokenizer.save(str(DATA/'tokenizer.json'))
    assert len(tokenizer.encode('123456789').ids) == 9
    summary = {}
    for split in ('train', 'dev', 'test'):
        for family in ('text', 'math', 'sudoku_easy', 'sudoku_hard'):
            rows = [r for r in records if r['split']==split and r['family']==family]
            flat, offsets, starts, meta = [], [0], [], []
            for r in rows:
                prompt = tokenizer.encode(r['prompt']).ids
                body = tokenizer.encode(r['text']).ids
                tok = [1]+prompt+body+[2]
                # No answer truncation. Text windows are cropped by the sampler.
                if family != 'text' and len(tok) > 385:
                    continue
                flat.extend(tok); offsets.append(len(flat)); starts.append(1+len(prompt) if prompt else 1)
                meta.append({k:v for k,v in r.items() if k not in ('text', 'split', 'family')})
            name = f'{family}_{split}'
            np.save(DATA/f'{name}.npy', np.asarray(flat, np.uint16))
            np.save(DATA/f'{name}_offsets.npy', np.asarray(offsets, np.int64))
            np.save(DATA/f'{name}_answer.npy', np.asarray(starts, np.int32))
            (DATA/f'{name}_meta.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in meta))
            summary[name] = dict(records=len(meta), tokens=len(flat), excluded_oversize=len(rows)-len(meta),
                                 sha256=digest(DATA/f'{name}.npy'),
                                 meta_sha256=digest(DATA/f'{name}_meta.jsonl'))
            print(name, summary[name], flush=True)
    manifest = dict(version=1, text=text_provenance, math=provenance,
                    sudoku_revision=SUDOKU_REV, sudoku_train_sample=50000, sudoku_test_sample=2048,
                    tokenizer_sha256=digest(DATA/'tokenizer.json'), vocab_size=tokenizer.get_vocab_size(),
                    datasets=summary, split_policy='Preserve official or existing source split; official math/Sudoku test deterministically divided into development/sealed test before training; no test tokenizer fitting',
                    text_reuse='Previously cached public FineWeb-Edu; new randomly initialized V10 model. Not a fresh external test corpus.',
                    near_duplicates='FineWeb inherits prior lexical near-dedup; math exact normalized question dedup only; Sudoku inherits upstream equivalence-separated train/test. Dev/test equivalence not independently audited.')
    (DATA/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('DATA READY', flush=True)


if __name__ == '__main__':
    main()
