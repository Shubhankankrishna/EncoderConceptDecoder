"""Does the tiny corpus carry structure aligned to the architecture's 8-token groups?

  python -B -m v10_3_tiny.audit_data      -> v10_3_tiny/data_audit.json

V9's GraphNav responses had a designed four-token step structure that matched
V9's four-token concept chunks (review_v9.md:113), so its hierarchy result could
not separate the architecture from the data. The V10 concept path pools fixed
8-token groups (release at 8g+7) and its training head forecasts the next 8
tokens. The audit asks, per family and split:

  1. Where does the answer start relative to the group grid? (answer index mod 8;
     uniform means no alignment; a single residue means a fixed alignment)
  2. Do structural tokens recur with period 8? (for the 12 most frequent token
     ids, the largest share of their occurrences in any one residue mod 8;
     1/8 = no alignment). Positions 0-3 are excluded: every symbolic record
     starts with the same fixed header (<bos>, family word, ':', newline), which
     is an absolute-position constant, not a period.
  3. How long are answers, relative to the 8-token forecast window?
  4. For text: what fraction of training windows are random crops (begin offset
     uniform, so the grid falls at random content positions)?

Read-only on data; refuses nothing because it writes one new file.
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parent / 'data'
OUT = Path(__file__).resolve().parent / 'data_audit.json'
FAMILIES = ('text', 'math', 'newmath', 'sudoku_easy', 'sudoku_hard')
CONTEXT = 256


def chi2_uniform(counts):
    counts = np.asarray(counts, float); e = counts.sum() / len(counts)
    return float(((counts - e) ** 2 / e).sum()) if e > 0 else 0.


def audit(family, split, rng, sample=20000):
    base = DATA / f'{family}_{split}'
    arr = np.load(str(base) + '.npy', mmap_mode='r'); offsets = np.load(str(base) + '_offsets.npy'); starts = np.load(str(base) + '_answer.npy')
    n = len(starts); idx = rng.choice(n, min(n, sample), replace=False)
    lengths = np.diff(offsets)[: n]
    out = dict(records=int(n))
    if family == 'text':
        cropped = (lengths > CONTEXT + 1).mean()
        out.update(fraction_longer_than_window=float(cropped),
                   note='Corpus.batch crops such documents at a uniform random begin offset, so group boundaries fall at random content positions')
    else:
        a = starts[idx].astype(int); answer_len = (lengths[idx] - a).astype(int)
        res = np.bincount(a % 8, minlength=8)
        out.update(answer_start_mod8=res.tolist(), answer_start_mod8_max_share=float(res.max() / res.sum()),
                   answer_start_mod8_chi2_df7=chi2_uniform(res), distinct_answer_starts=int(len(set(a.tolist()))),
                   answer_tokens_quantiles=dict(zip(('p10', 'p50', 'p90', 'max'), np.quantile(answer_len, [.1, .5, .9, 1.]).astype(int).tolist())),
                   fraction_answer_within_8=float((answer_len <= 8).mean()),
                   sequence_length_mod8=np.bincount(lengths[idx] % 8, minlength=8).tolist())
    # Periodicity of frequent token ids at the positions the model actually sees.
    pos_by_tok = {}
    tot = Counter()
    for i in idx[:4000]:
        seq = np.asarray(arr[offsets[i]:offsets[i + 1]])
        if family == 'text' and len(seq) > CONTEXT + 1:
            b = int(rng.integers(0, len(seq) - CONTEXT)); seq = seq[b:b + CONTEXT + 1]
        tot.update(seq[4:].tolist())
        for p, t in enumerate(seq.tolist()):
            if p < 4: continue
            pos_by_tok.setdefault(t, np.zeros(8, int))[p % 8] += 1
    top = [t for t, _ in tot.most_common(12)]
    shares = {int(t): float(pos_by_tok[t].max() / pos_by_tok[t].sum()) for t in top}
    out.update(frequent_token_max_residue_share=shares, frequent_token_worst=max(shares.values()))
    return out


def main():
    rng = np.random.default_rng(8)
    result = {f: {s: audit(f, s, rng) for s in ('train', 'dev')} for f in FAMILIES}
    result['thresholds'] = dict(uniform_residue_share=1 / 8, chi2_df7_p001=24.32)
    OUT.write_text(json.dumps(result, indent=1) + '\n')
    for f in FAMILIES:
        for s in ('train', 'dev'):
            r = result[f][s]
            print(f, s, {k: v for k, v in r.items() if k not in ('frequent_token_max_residue_share', 'note')})


if __name__ == '__main__':
    main()
