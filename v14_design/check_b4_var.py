"""Post-hoc follow-up to check_projectors.py's b4 (not pre-registered): per family, the normalised V13 targets' per-coordinate
variance against the codebook's nearest-word distance. Run from the repo root: taskpolicy -b python3 v14_design/check_b4_var.py"""
import sys, json, torch
from pathlib import Path
sys.path.insert(0, 'v14_design'); sys.path.insert(0, 'v13_tiny')
import checks as C
from check_projectors import collect
torch.set_num_threads(2)
tk = C.TK.Tok(); m, cb, _ = C.FE.load(C.V13 / 'runs' / 'base' / 'final.pt', torch.device('cpu'), tk)
out = {}
for f in C.FAMS:
    d = collect(m, cb, C.rows_for(f, 8, tk), m.c.k)
    M = torch.cat([x['m'] for x in d]).float()
    z = cb.normalise(M).reshape(-1, cb.S, cb.ds).double()
    d2 = (z[:, :, None, :] - cb.code.double()[None]).pow(2).sum(-1)
    sig2 = (d2.min(-1).values.mean(0) / cb.ds)                      # per segment
    var = z.var(0).mean(-1)                                          # mean per-coordinate variance, per segment
    out[f] = dict(groups=len(M), var_per_coord=[round(float(v), 3) for v in var], sigma2=[round(float(v), 3) for v in sig2],
                  unexplained_share=[round(float(s / v), 3) for s, v in zip(sig2, var)])
    print(f, out[f], flush=True)
json.dump(out, open(str(Path(__file__).resolve().parent / 'checks' / 'b4_var.json'), 'w'), indent=1)
