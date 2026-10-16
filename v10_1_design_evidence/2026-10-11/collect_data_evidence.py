"""Read only active train/dev data; never reads sealed test contents or modifies a run."""
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from v10_10M.data import verify_sudoku
from v10_10M.data_tools.clean_math import clean_math_field

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'v10_10M/data'
DEST = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def key(s):
    return hashlib.sha256(re.sub(r'\s+', ' ', s.strip()).encode()).hexdigest()

result = {'captured_unix': time.time(), 'scope': 'train/dev only; no sealed test contents',
          'manifest_sha256': sha(DATA / 'manifest.json'), 'files': {}}
for rel in ['v10/proposed_v10.md', 'v10/V10_RCDL_Architecture_Reviewed.md',
            'v10_10M/model.py', 'v10_10M/train.py', 'v10_10M/memory.py',
            'v10_10M/data.py', 'v10_10M/data_tools/prepare_v2.py']:
    result['files'][rel] = sha(ROOT / rel)
h = hashlib.sha256()
for path in sorted((ROOT / 'v10_10M').glob('*.py')):
    h.update(path.name.encode()); h.update(path.read_bytes())
result['active_source_sha256'] = h.hexdigest()
attachment = Path('/Users/shubhankankrishna/.codex/attachments/42b00510-47d9-46d7-a0cc-4a12f65d0ae8/Pasted text.txt')
result['attachment_sha256'] = sha(attachment)
tokenizer = Tokenizer.from_file(str(DATA / 'tokenizer.json'))
identities = {}
clean_ids = {}
result['families'] = {}
for family in ['text', 'math', 'sudoku_easy', 'sudoku_hard']:
    for split in ['train', 'dev']:
        name = f'{family}_{split}'
        offsets = np.load(DATA / (name + '_offsets.npy'), mmap_mode='r')
        starts = np.load(DATA / (name + '_answer.npy'), mmap_mode='r')
        lengths = np.diff(offsets)
        stat = {'records': len(starts), 'tokens': int(offsets[-1]),
                'record_length_p0_p50_p95_p100': np.percentile(lengths, [0,50,95,100]).tolist(),
                'supervised_length_p0_p50_p95_p100': np.percentile(lengths - starts, [0,50,95,100]).tolist()}
        ids = set(); normalized = set(); sources = Counter(); wrapper = 0
        invalid = Counter(); clue_counts = []; answer_lengths = []; token_samples = []
        with (DATA / (name + '_meta.jsonl')).open() as f:
            for n,line in enumerate(f):
                row = json.loads(line); ids.add(row['identity']); sources[row.get('source','')] += 1
                if family == 'math':
                    answer = row['answer']; wrapper += int(answer.strip().startswith(("b'", 'b"')))
                    q = row['prompt'][len('Math: '):-len('\nAnswer:')]
                    normalized.add(key(clean_math_field(q)))
                    cleaned = clean_math_field(answer); answer_lengths.append(len(cleaned))
                    if n % 100 == 0:
                        token_samples.append([len(tokenizer.encode(' '+answer).ids), len(tokenizer.encode(' '+cleaned).ids)])
                if family.startswith('sudoku'):
                    check = verify_sudoku(row['puzzle'], row['solution'])
                    if not check['valid']: invalid[check['reason']] += 1
                    clue_counts.append(sum(c != '0' for c in row['puzzle']))
        identities[name] = ids; clean_ids[name] = normalized
        stat['duplicate_identity_count'] = len(starts) - len(ids)
        if family == 'math':
            stat.update(wrapped_answers=wrapper, modules=dict(sources),
                        duplicate_clean_questions=len(starts)-len(normalized),
                        clean_answer_character_quantiles=np.percentile(answer_lengths,[0,50,95,100]).tolist(),
                        sampled_answer_tokens_wrapped_clean_mean=np.mean(token_samples,axis=0).tolist(),
                        answer_token_comparison_sample_n=len(token_samples))
        if family.startswith('sudoku'):
            stat.update(invalid_ground_truth=dict(invalid), clues_p0_p50_p95_p100=np.percentile(clue_counts,[0,50,95,100]).tolist())
        result['families'][name] = stat
result['exact_train_dev_identity_overlap'] = {f: len(identities[f+'_train'] & identities[f+'_dev']) for f in ['text','math','sudoku_easy','sudoku_hard']}
result['clean_math_train_dev_question_overlap'] = len(clean_ids['math_train'] & clean_ids['math_dev'])
result['limitations'] = ['Exact identity checks do not establish semantic or Sudoku-isomorphism separation.',
                        'Answer-token comparison uses the active tokenizer on a deterministic every-100th-record sample, excludes EOS.']
(DEST / 'data_evidence.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
