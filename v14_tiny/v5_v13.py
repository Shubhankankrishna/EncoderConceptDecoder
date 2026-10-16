"""V5's V13-tiny side, run in its own process by diagnostics_v14.py (V13-tiny's model.py and ncp.py share module names
with V14-tiny's): the PQ code of m_g for every real group of the given rows at V13-tiny's snap_010, snap_030, snap_050
and snap_070, and the fraction of segment codes that change between snap_010 and snap_050 (compared by V5) and between
snap_030 and snap_070 (reported). Prints one JSON line.

    python3 v5_v13.py --v13 ../v13_tiny --rows runs/eval/diag/dev_rows.pkl [--device mps]
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v13', required=True); ap.add_argument('--rows', required=True); ap.add_argument('--device', default='mps')
    a = ap.parse_args()
    v13 = Path(a.v13).resolve(); here = Path(__file__).resolve().parent
    rows = pickle.loads(Path(a.rows).resolve().read_bytes())
    sys.path = [str(v13)] + [p for p in sys.path if p and Path(p).resolve() != here]
    os.chdir(v13)
    import numpy as np
    import torch
    import model as M
    import ncp as N
    import tok as TK
    dev = torch.device(a.device); tk = TK.Tok()
    codes = {}
    for ck in ('snap_010', 'snap_030', 'snap_050', 'snap_070'):
        z = torch.load(v13 / 'runs/base' / f'{ck}.pt', map_location='cpu', weights_only=False)
        m = M.V13(M.Cfg(V=tk.vocab)); m.load_state_dict(z['model']); m = m.to(dev).eval()
        cb = N.PQCodebook(m.c.d, m.c.pq_S, m.c.pq_N); cb.load_state_dict(z['codebook']); cb = cb.to(dev)
        for f, rs in rows.items():
            out = []
            with torch.no_grad():
                for i in range(0, len(rs), 8):
                    ids = torch.from_numpy(np.stack([r['ids'] for r in rs[i:i + 8]])).long().to(dev)
                    doc = torch.from_numpy(np.stack([r['doc'] for r in rs[i:i + 8]])).long().to(dev)
                    mm = m(ids, doc, return_all=True)['m']
                    real = doc[:, ::m.c.k] > 0
                    out.append(cb.assign(cb.normalise(mm))[real].cpu())
            codes[(ck, f)] = torch.cat(out)
        del m, cb, z
    res = {}
    for x, y in (('snap_010', 'snap_050'), ('snap_030', 'snap_070')):
        per = {f: round(float((codes[(x, f)] != codes[(y, f)]).float().mean()), 4) for f in rows}
        c1 = torch.cat([codes[(x, f)] for f in rows]); c2 = torch.cat([codes[(y, f)] for f in rows])
        per['pooled'] = round(float((c1 != c2).float().mean()), 4); per['groups'] = len(c1)
        res[f'{x}-{y}'] = per
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
