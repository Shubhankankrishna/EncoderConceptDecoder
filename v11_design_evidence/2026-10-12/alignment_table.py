"""Decoder alignment and release table for V11 rows 0-17, derived from the inherited code.

  python -B v11_design_evidence/2026-10-12/alignment_table.py   (from the repo root)

Uses v10_10M.model.Decoder.align (aligned forecast route) and
v10_2.model.history_geometry (concept-history reader mask) unchanged. Each
forecast slot carries a unique tag 100*(g+1)+s so its landing rows are visible.
Reads code only; writes alignment_table.json next to this script.
"""
import json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v10_10M.model import Decoder
from v10_2.model import history_geometry

OUT = Path(__file__).with_name('alignment_table.json')


def table(length, t):
    g = length // 8                                    # completed groups only
    d = 4
    forecast = torch.zeros(1, g, 2, d)
    for i in range(g):
        for s in range(2):
            forecast[0, i, s] = 100 * (i + 1) + s       # tag: group i's forecast (of group i+1), slot s
    guidance = Decoder.align(forecast, t)[0, :, 0] if g else torch.zeros(t)
    mask, _, _ = history_geometry(t, max(g, 0), torch.tensor([length]), 2, 'cpu')
    mask = mask[0, 0]                                  # [t, 2g + 1 sentinel]
    rows = []
    for r in range(t):
        tag = int(guidance[r])
        keys = sorted({k // 2 for k in range(2 * g) if bool(mask[r, k])})
        rows.append(dict(row=r, observed=f'x_0..x_{r}', valid=r < length, predicts=f'x_{r+1}',
                         target_group=(r + 1) // 8, target_half=((r + 1) % 8) // 4,
                         guidance=None if tag == 0 else dict(forecast_of_group=tag // 100, from_group=tag // 100 - 1, slot=tag % 100),
                         reader_keys_groups=keys, sentinel=bool(mask[r, 2 * g])))
    return rows


def main():
    cases = {'length_18': table(18, 18), 'bos_only_length_1': table(1, 1),
             'tail_length_19_T_20': table(19, 20), 'padded_length_12_T_18': table(12, 18)}
    checks = []
    for name, rows in cases.items():
        for r in rows:
            gd = r['guidance']
            if gd is not None:   # causal: the forecast's source group must be released by this row
                checks.append(8 * gd['from_group'] + 7 <= r['row'])
                checks.append(gd['forecast_of_group'] == r['target_group'] and gd['slot'] == r['target_half'])
            for k in r['reader_keys_groups']:
                checks.append(8 * k + 7 <= r['row'])
    result = dict(cases=cases, all_checks_pass=all(checks), n_checks=len(checks))
    OUT.write_text(json.dumps(result, indent=1) + '\n')
    print('rows 0-17 (length 18):')
    for r in cases['length_18']:
        print(r['row'], r['predicts'], 'target group', r['target_group'], 'half', r['target_half'], '| guidance', r['guidance'], '| reader keys', r['reader_keys_groups'], '| sentinel', r['sentinel'])
    print('tail length 19 groups formed:', 19 // 8, ' checks:', result['all_checks_pass'], len(checks))


if __name__ == '__main__':
    main()
