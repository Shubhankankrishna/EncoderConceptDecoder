"""Per-family dev NLL for any run/checkpoint (forward-only): pinpad act/pos/end, text, math, newmath.

  python -B devnll.py RUN CKPT [CKPT ...]   -> runs/RUN/devnll_CKPT.json
"""
import json, sys
from pathlib import Path
import torch
from data import Split
from evaluate import load_run, dev_nll, bpe_dev_nll

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def main(run, ckpts, device='mps'):
    for c in ckpts:
        model, _ = load_run(run, device, c); model.eval()
        r = dict(run=run, ckpt=c, pinpad=dev_nll(model, Split('dev'), device),
                 **{f: bpe_dev_nll(model, f, device) for f in ('text', 'math', 'newmath')})
        (ROOT / 'runs' / run / f'devnll_{c}.json').write_text(json.dumps(r, indent=1)); print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
