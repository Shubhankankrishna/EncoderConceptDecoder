"""Distraction vs alignment: re-score math/newmath with r in {0, 8, 16} filler tokens after BOS (r = 8 and 16
keep the group alignment of r = 0) and Sudoku with header shifts 4 and 8 (v10_3_tiny/grid_shift8.py logic),
for every final checkpoint. A residue (alignment) effect is periodic in r; distraction grows with filler.

  python -B -m v11_tiny.offsets_extra     (writes v11_tiny/offsets_extra.json; run with the GPU idle)
"""
import json
from pathlib import Path
import torch
from v10_3_tiny import evaluate as E
from v10_3_tiny.config import TinyConfig
from .model import build
from . import evaluate as V

ROOT = Path(__file__).resolve().parent; RUNS = ROOT / 'runs'


def filler16():
    arr, offsets, _, _ = E.load_family('text')
    return [int(t) for t in arr[offsets[7]:offsets[8]] if int(t) > 3][100:116]


@torch.no_grad()
def score(model, scored, device, r_values, filler):
    out = {}
    for fam in ('math', 'newmath'):
        res = {}
        for r in r_values:
            docs = []
            for d in scored[fam]:
                n = int(d['lengths'][0]); seq = torch.cat((d['ids'][0, :n], d['targets'][0, n - 1:n]))
                new = torch.cat((seq[:1], torch.tensor(filler[:r], dtype=torch.long), seq[1:])); m = len(new) - 1
                t = ((m - 1) // 8 + 1) * 8
                x = torch.zeros(1, t, dtype=torch.long); y = torch.zeros_like(x); mask = torch.zeros_like(x, dtype=torch.bool)
                x[0, :m], y[0, :m] = new[:-1], new[1:]; src = torch.nonzero(d['target_mask'][0]).flatten(); mask[0, src + r] = True
                docs.append(dict(ids=x, targets=y, target_mask=mask, lengths=torch.tensor([m])))
            num = den = 0.
            for s in range(0, len(docs), E.BATCH):
                c = docs[s:s + E.BATCH]; ce, _ = E.batch_ce(model, E.to(E.stack(c), device)); num += sum(ce); den += sum(int(z['target_mask'].sum()) for z in c)
            res[r] = num / den
        out[fam] = res
    return out


def main():
    scored, _ = E.panels(); filler = filler16(); result = {}
    for run in sorted(p for p in RUNS.iterdir() if p.is_dir() and (p / 'final.pt').exists() and p.name != 'solo'):
        ck = torch.load(run / 'final.pt', map_location='cpu', weights_only=False)
        cfg = TinyConfig(**{k: (tuple(v) if k == 'engram_rows' else v) for k, v in ck['config'].items()})
        model = build(ck['arm'], cfg); model.load_state_dict(ck['model']); model.place('mps').eval()
        res = score(model, scored, 'mps', (0, 1, 4, 7, 8, 9, 16), filler) if run.name.endswith('_s1') else {'math': {}, 'newmath': {}}   # r-curve: seed 1 only
        res['sudoku_shift4'] = E.grid_shift(model, scored, 'mps', shift=4)
        res['sudoku_shift8'] = E.grid_shift(model, scored, 'mps', shift=8)
        result[run.name] = res
        print(run.name, {f: {r: round(v, 3) for r, v in res[f].items()} for f in ('math', 'newmath')},
              {k: round(res[k]['sudoku_easy']['degradation'], 3) for k in ('sudoku_shift4', 'sudoku_shift8')}, flush=True)
        (ROOT / 'offsets_extra.json').write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
