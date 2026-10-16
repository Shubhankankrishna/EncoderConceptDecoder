"""Observational chunk-phase comparison on the same held-out episodes.

Action queries are even positions; with k=6 their concept plan lags 1,3 or 5
tokens. Phase effects are observational because path length and layout differ.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import analyze_vs as A
import run_vs as R

def run(name):
    torch.set_num_threads(1)
    import speed_v3;speed_v3.enable_block_attention(60)
    m,args=A.load(name)
    units=A.pin_units(m,R.pin_eval_sets()["heldout"])
    units={k:v for k,v in units.items() if "lag" in k}
    report={k:dict(mean=float(np.nanmean(v)),n_episodes=int(np.isfinite(v).sum())) for k,v in units.items()}
    out=Path(R.RUNS)/name
    (out/"phase.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez(out/"phase_units.npz",**units)
    print(json.dumps(report,indent=2))

if __name__=="__main__":run(sys.argv[1])
