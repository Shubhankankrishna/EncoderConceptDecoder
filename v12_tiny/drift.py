"""Codebook drift (advisor #13): the NCP head learns against a moving EMA k-means codebook. For the same dev
targets (summaries from the later checkpoint's model), the fraction whose code changes between the earlier and the
later checkpoint's codebook (normaliser + codewords), per family and per codebook segment.

  python -B drift.py RUN CKPT_A CKPT_B     -> runs/RUN/drift_A_B.json
"""
import json, sys
from pathlib import Path
import torch
from evaluate import load_run
from losses import group_targets
from info_test import family_batches
from losses import group_valid
import train

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def main(run, ca, cb_, device='mps'):
    mA, arm = load_run(run, device, ca); mB, _ = load_run(run, device, cb_)
    kind = train.ARMS[arm]['ncp']; res = {}
    for fam in ('pinpad', 'math', 'text'):
        ch, n = 0, 0; chs = torch.zeros(mB.codebook.S)
        for b in family_batches(fam, device, 'dev'):
            out = mB(b['ids']); raw = group_targets(out, mB.k, kind); G = raw.shape[1]
            v = group_valid(b, G, mB.k); t = raw[v]
            ia = mA.codebook.assign(mA.codebook.normalise(t)); ib = mB.codebook.assign(mB.codebook.normalise(t))
            chs += (ia != ib).float().sum(0).cpu(); n += t.shape[0]
        res[fam] = dict(rows=n, frac_changed_per_segment=(chs / max(n, 1)).tolist(), frac_changed=float(chs.sum() / max(n * mB.codebook.S, 1)))
        print(fam, res[fam], flush=True)
    (ROOT / 'runs' / run / f'drift_{ca}_{cb_}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
