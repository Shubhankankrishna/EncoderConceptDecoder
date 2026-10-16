"""Chance baseline for switch F1 (advisor #20). For a trained metacontroller, keep each episode's number of predicted
switches but place them uniformly at random over that episode's valid units (20 draws); score pair and colour F1
with the same tolerance and clock as `meta.switch_metrics`. CPU only.

  python -B switch_chance.py RUN TAG      -> adds 'chance' to runs/RUN/meta/TAG.json (original eval untouched)
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from data import Split
from evaluate import load_run
from meta import MetaController, step_loss, switch_metrics

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def main(run, tag, draws=20, n=1024, thr=0.5):
    torch.set_num_threads(1)
    model, _ = load_run(run, 'cpu')
    cp = tag.split('_a')[0]
    if model.kind == 'ECD': model.ctrl_point = cp
    elif cp.startswith('L'): model.ctrl_layer = int(cp[1:])
    mc = MetaController(model.d); mc.load_state_dict(torch.load(ROOT / 'runs' / run / 'meta' / f'{tag}.pt', map_location='cpu'))
    dm = Split('dev_meta'); rng = np.random.default_rng(0)
    real = {k: [] for k in ('pair', 'colour')}; chance = {k: [] for k in ('pair', 'colour')}
    for s in range(0, n, 128):
        b = dm.batch(np.arange(s, min(s + 128, n)), lam_obs=0.0, device='cpu', pad_to=176)
        _, _, info = step_loss(model, mc, b, 0.0, sample=False)
        mask, beta = info['mask'].cpu(), info['beta'].detach().cpu()
        for key, name in (('abs_act', 'pair'), ('tgt_col', 'colour')):
            real[name].append(switch_metrics(info, b, key=key)['f1'])
        for _ in range(draws):
            fake = torch.zeros_like(beta)
            for i in range(beta.shape[0]):
                valid = [j for j in range(1, beta.shape[1]) if mask[i, j]]
                k = int(((beta[i] > thr) & mask[i])[1:].sum())
                if k and valid: fake[i, rng.choice(valid, size=min(k, len(valid)), replace=False)] = 1.0
            fi = dict(info, beta=fake)
            for key, name in (('abs_act', 'pair'), ('tgt_col', 'colour')):
                chance[name].append(switch_metrics(fi, b, key=key)['f1'])
    res = {f'{k}_f1': float(np.mean(v)) for k, v in real.items()}
    res.update({f'{k}_f1_chance': float(np.mean(v)) for k, v in chance.items()})
    p = ROOT / 'runs' / run / 'meta' / f'{tag}.json'
    d = json.loads(p.read_text()); d['chance'] = res; p.write_text(json.dumps(d, indent=1))
    print(json.dumps(dict(run=run, tag=tag, **res)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
