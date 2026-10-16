"""Summaries from train.jsonl for every run (no model is loaded).

  python -B -m v11_tiny.training_summary      (writes v11_tiny/training_summary.json)

- Stream identity: per seed, the (family, document) hash sequence must be identical across all six arms.
- Paired training-stream gaps: every arm saw the identical batch at each step, and the stream is one
  pass over fresh documents, so the per-step token-loss difference on the same batch is a low-variance,
  effectively held-out comparison (as in v10_3_tiny/paired_training.py). Reported over the last 1,000
  updates per family, with a block-bootstrap CI over 50-step blocks.
- Thermal: median step time in the first and last 500 updates (runs were co-scheduled three at a time).
- Telemetry: final sigma^2, mean v2 boundaries and bridge rows per update, max driver memory.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'
ARMS = ('C0', 'ORD', 'ORDE', 'X8', 'NOAUX', 'K1NB', 'K1', 'K1SC', 'NOAUXSC', 'K1NBSC')


def read(name):
    p = RUNS / name / 'train.jsonl'
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else None


def block_ci(diff, block=50, draws=2000, seed=0):
    n = len(diff) // block
    if n < 2: return None
    blocks = np.array(diff[:n * block]).reshape(n, block).mean(1)
    rng = np.random.default_rng(seed); boot = blocks[rng.integers(0, n, (draws, n))].mean(1)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def main():
    logs = {f'{a}_s{s}': read(f'{a}_s{s}') for a in ARMS for s in (1, 2, 3)}
    out = dict(stream_identity={}, paired_gaps={}, runs={})
    for s in (1, 2, 3):
        streams = {a: [r['stream'] for r in logs[f'{a}_s{s}']] for a in ARMS if logs[f'{a}_s{s}']}
        ref = next(iter(streams.values()), None)
        out['stream_identity'][f's{s}'] = dict(arms=sorted(streams), identical=all(v == ref for v in streams.values()),
                                               lengths={a: len(v) for a, v in streams.items()})
    for name, rec in logs.items():
        if not rec: continue
        st = [r['step_seconds'] for r in rec]
        last = rec[-1]
        out['runs'][name] = dict(steps=len(rec), first500_median_step=float(np.median(st[:500])), last500_median_step=float(np.median(st[-500:])),
                                 elapsed=last['elapsed'], max_driver_gb=max((r.get('mps_driver_gb') or 0) for r in rec),
                                 sigma1_sq=last.get('sigma1_sq'), sigma2_sq=last.get('sigma2_sq'),
                                 mean_v2_boundaries=float(np.mean([r.get('v2_boundaries') or 0 for r in rec])) if 'v2_boundaries' in last else None,
                                 mean_bridge_rows=float(np.mean([r.get('bridge_rows') or 0 for r in rec])) if 'bridge_rows' in last else None,
                                 candidate_steps=sum(bool(r.get('candidate_executed')) for r in rec),
                                 final_participation_ratio=[r['tau_participation_ratio'] for r in rec if 'tau_participation_ratio' in r][-5:] or None)
    pairs = [('K1', 'X8'), ('K1', 'C0'), ('K1', 'ORD'), ('K1', 'K1NB'), ('K1', 'NOAUX'), ('K1NB', 'C0'), ('NOAUX', 'C0'),
             ('X8', 'C0'), ('ORD', 'C0'), ('K1NB', 'X8'), ('K1SC', 'K1NB'), ('K1', 'K1SC'), ('K1SC', 'X8'), ('K1SC', 'C0'), ('K1', 'ORDE'), ('C0', 'ORDE'), ('ORDE', 'ORD'), ('NOAUXSC', 'C0'), ('K1NBSC', 'C0'), ('K1NBSC', 'NOAUXSC'), ('K1SC', 'K1NBSC'), ('NOAUX', 'NOAUXSC'), ('K1NB', 'K1NBSC')]
    for a, b in pairs:
        for s in (1, 2, 3):
            ra, rb = logs[f'{a}_s{s}'], logs[f'{b}_s{s}']
            if not ra or not rb or len(ra) != len(rb): continue
            assert all(x['stream'] == y['stream'] for x, y in zip(ra, rb))
            row = {}
            for fam in ('text', 'math', 'sudoku_easy', 'sudoku_hard'):
                d = [x['token'] - y['token'] for x, y in zip(ra[-1000:], rb[-1000:]) if x['family'] == fam]
                row[fam] = dict(mean=float(np.mean(d)) if d else None, ci=block_ci(d, block=10 if fam != 'text' else 50), updates=len(d))
            out['paired_gaps'][f'{a}-{b}_s{s}'] = row
    (ROOT / 'training_summary.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['stream_identity']))
    for k, v in out['paired_gaps'].items():
        print(k, {f: (round(x['mean'], 4) if x['mean'] is not None else None) for f, x in v.items()})


if __name__ == '__main__':
    main()
