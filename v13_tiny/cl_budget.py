"""Continual-stage budget (DEVIATIONS item 32, fixed before any continual timing existed).

    python3 cl_budget.py --probe runs/post/probe --deadline "2026-10-15 04:00"   -> prints the steps per phase

The incoming steps per phase n (the same for every method) is the largest n in [40, 100] for which seq, v13 and
replay together are predicted to finish by the deadline, from B = 8 probe timings taken on the idle machine:
    per incoming batch: seq t_s, replay 1.25 t_s (20% replay batches plus margin), v13 t_v (wake + sleep);
    per phase and method: an overhead for rows (cached after the first method), gates and saving;
    once per method: model load and pool start (5 min, conservative).
"""
import argparse
import datetime
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', default='runs/post/probe'); ap.add_argument('--deadline', default='2026-10-15 04:00')
    a = ap.parse_args()
    pr = HERE / a.probe
    seq = [json.loads(l) for l in open(pr / 'seq' / 'log.jsonl')]
    v13 = [json.loads(l) for l in open(pr / 'v13' / 'log.jsonl')]
    t_s = sum(r['s_train'] for r in seq) / sum(r['steps'] for r in seq)
    t_v = sum(r['s_wake'] + r['s_sleep'] for r in v13) / sum(r['steps'] for r in v13)
    g_v = max(r['s_gates'] for r in v13)
    per_n = 6 * (t_s + 1.25 * t_s + t_v)                       # seconds per unit of n over all phases and methods
    fixed = 3 * 300 + 6 * (g_v + 60) + 6 * 3 * 30 + 900         # loads, gates, saving, rows once (15 min)
    left = datetime.datetime.strptime(a.deadline, '%Y-%m-%d %H:%M').timestamp() - datetime.datetime.now().timestamp()
    n = int(max(40, min(100, (left - fixed) // per_n)))
    print(json.dumps(dict(t_seq=round(t_s, 2), t_v13=round(t_v, 2), gates_s=round(g_v, 1), seconds_left=int(left),
                          predicted_hours=round((fixed + n * per_n) / 3600, 2), steps=n)), flush=True)
    (pr / 'budget.json').write_text(json.dumps(dict(steps=n, t_seq=t_s, t_v13=t_v, left=left)))
    print(n)


if __name__ == '__main__':
    main()
