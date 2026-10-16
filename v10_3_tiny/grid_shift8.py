"""Shift-8 control for the Sudoku grid-shift test (see ARCHITECTURE.md data audit).

A shift of 4 moves every cell half a group; a shift of 8 moves it exactly one
group, keeping the cell-to-group alignment while applying the same kind of
out-of-distribution header edit. If an arm's extra shift-4 degradation is
alignment, its shift-8 degradation should fall back toward C0's.

  python -B -m v10_3_tiny.grid_shift8 --runs C0_s1 A_s1 X8_s1 X4_s1
"""
import argparse, json
from pathlib import Path
import torch
from . import evaluate as E
from .sleep import load

RUNS = Path(__file__).resolve().parent / 'runs'

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--runs', nargs='+', required=True); p.add_argument('--device', default='mps')
    a = p.parse_args(); torch.set_num_threads(1)
    scored, _ = E.panels(); scored = {f: scored[f] for f in ('sudoku_easy', 'sudoku_hard')}; out = {}
    for name in a.runs:
        model, ck = load(RUNS / name, a.device)
        out[name] = {str(s): E.grid_shift(model, scored, a.device, shift=s) for s in (4, 8)}
        print(name, {s: {f: round(v['degradation'], 4) for f, v in r.items()} for s, r in out[name].items()}, flush=True)
    (RUNS.parent / 'grid_shift8.json').write_text(json.dumps(out, indent=1) + '\n')
