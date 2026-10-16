"""V14's continual default from V13-tiny's continual matrix (v14_Architecture.md §9.4), by a rule fixed on 15 Oct
02:30 IST before that matrix existed (DEVIATIONS.md item 11). Reads ../v13_tiny/runs/eval/cl/cl_results.json, writes
runs/cl/choice.json with the outcome, the numbers it used and the configuration of `cl.py --method v14`.

Definitions (Dataset_v13 §10.3 and REPORT §5 item 5, as §9.4 quotes them; mean acquisition gain = mean over S1..S6 of
the gain on each phase's new columns):
  helps      forgetting below sequential LoRA's, S0 retention before the return phase (S5) above it with the paired
             95% interval of (v13 - seq) excluding zero, and a mean acquisition gain at least half of seq's;
  advantage  mean acquisition gain within 0.03 of replay's and S0 retention before return at least 0.05 above replay's.
Precedence: O6 (V13's matrix missing) > O1 (advantage) > O4 (mean acquisition gain below half of seq's) > O2 (helps
and final average below replay's) > O3 (does not help). Helps without the advantage bar and not below replay on final
average (a case §9.4 does not list) maps to O1's default. O5 (S3 counterparts forget under every method: the S3c
column at S6 below its value at S3 for seq, replay and v13) is reported beside the outcome; by §9.4 it changes no
default (code gates are an arm, not run here).
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = {
    'O1': dict(base='v13', anchor=False, replay=0.0, pos_batches=12, rank=4),
    'O2': dict(base='v13', anchor=True, replay=0.2, pos_batches=12, rank=4),
    'O3': dict(base='replay', anchor=True, replay=0.2, pos_batches=12, rank=4),
    'O4': dict(base='v13', anchor=False, replay=0.0, pos_batches=24, rank=8),
    'O6': dict(base='v13', anchor=False, replay=0.0, pos_batches=12, rank=4),
}


def mean_gain(m):
    return sum(v['gain'] for v in m['acquisition'].values()) / len(m['acquisition'])


def decide(res):
    if not all(k in res and 'metrics' in res[k] for k in ('seq', 'replay', 'v13')):
        return 'O6', dict(reason='V13-tiny matrix incomplete')
    v, s, r = (res[k]['metrics'] for k in ('v13', 'seq', 'replay'))
    pv = res.get('paired', {}).get('v13-seq', {})
    ci = pv.get('s0_before_return_ci', [0, 0])
    gv, gs, gr = mean_gain(v), mean_gain(s), mean_gain(r)
    helps = (v['forgetting'] < s['forgetting'] and v['retention_before_return'] > s['retention_before_return']
             and ci[0] > 0 and gv >= 0.5 * gs)
    advantage = abs(gv - gr) <= 0.03 and v['retention_before_return'] - r['retention_before_return'] >= 0.05
    o5 = all(res[k]['metrics']['s3_counterparts']['S6'] < res[k]['metrics']['s3_counterparts']['S3']
             for k in ('seq', 'replay', 'v13'))
    nums = dict(forgetting=dict(v13=v['forgetting'], seq=s['forgetting'], replay=r['forgetting']),
                retention_before_return=dict(v13=v['retention_before_return'], seq=s['retention_before_return'],
                                             replay=r['retention_before_return']),
                s0_before_return_ci_v13_minus_seq=ci, mean_acquisition_gain=dict(v13=gv, seq=gs, replay=gr),
                final_average=dict(v13=v['final_average'], seq=s['final_average'], replay=r['final_average']),
                helps=helps, advantage=advantage, O5=o5)
    if advantage: return 'O1', nums
    if gv < 0.5 * gs: return 'O4', nums
    if helps and v['final_average'] < r['final_average']: return 'O2', nums
    if helps: return 'O1', dict(nums, note='helps, not below replay on final average, advantage bar not met: O1 default')
    return 'O3', nums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v13', default='../v13_tiny/runs/eval/cl/cl_results.json')
    ap.add_argument('--out', default='runs/cl/choice.json')
    a = ap.parse_args()
    p = (HERE / a.v13)
    res = json.loads(p.read_text()) if p.exists() else {}
    outcome, nums = decide(res)
    out = dict(outcome=outcome, config=CONFIG[outcome], numbers=nums, source=str(p))
    (HERE / a.out).parent.mkdir(parents=True, exist_ok=True)
    (HERE / a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
