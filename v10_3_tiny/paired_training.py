"""Paired training-stream comparison: every seed-1 arm saw the identical batch at
each step (step-based schedule, same seed), so per-step token-loss differences
are paired. Reports mean (arm - reference) token loss by family in step bins.

  python -B -m v10_3_tiny.paired_training --runs C0_s1 A_s1 X8_s1 X4_s1 --reference C0_s1
"""
import argparse, json
from pathlib import Path
import numpy as np

RUNS = Path(__file__).resolve().parent / 'runs'
BINS = [(1, 1000), (1001, 2000), (2001, 3000), (3001, 4000), (4001, 5000), (5001, 6000)]


def load(name):
    return {r['step']: r for r in map(json.loads, (RUNS / name / 'train.jsonl').open())}


def main(a):
    ref = load(a.reference); out = {}
    for name in a.runs:
        if name == a.reference: continue
        run = load(name); res = {}
        for fam in ('text', 'math', 'sudoku_easy', 'sudoku_hard'):
            row = {}
            for lo, hi in BINS:
                d = [run[s]['token'] - ref[s]['token'] for s in range(lo, hi + 1)
                     if s in run and s in ref and ref[s]['family'] == fam and run[s]['family'] == fam]
                assert all(run[s]['family'] == ref[s]['family'] for s in range(lo, hi + 1) if s in run and s in ref), 'streams differ'
                if d:
                    d = np.array(d); row[f'{lo}-{hi}'] = dict(n=len(d), mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))))
            res[fam] = row
        out[f'{name} - {a.reference}'] = res
    Path(a.output).write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out.items():
        print(k)
        for fam, row in v.items():
            print(f'  {fam:12s}', '  '.join(f"{b}: {x['mean']:+.3f}±{x['se']:.3f}" for b, x in row.items()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--runs', nargs='+', required=True); p.add_argument('--reference', required=True)
    p.add_argument('--output', default=str(RUNS.parent / 'paired_training.json'))
    main(p.parse_args())
