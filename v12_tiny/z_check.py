"""Is the metacontroller's held z a task-level code? (advisor #21) For a trained point:
(a) cosine between z at unit 0 (forced) and the z held after the observed switch;
(b) linear probes from the held z (unit 3) to the episode's first pair and second pair (pair after the first true switch),
    trained on half of dev_meta, scored on the other half; chance = majority class.
CPU only.   python -B z_check.py RUN TAG"""
import json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from data import Split
from evaluate import load_run
from meta import MetaController, step_loss
from probes import fit_probe

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def main(run, tag, n=1024):
    torch.set_num_threads(3)
    model, _ = load_run(run, 'cpu'); cp = tag.split('_a')[0]
    if model.kind == 'ECD': model.ctrl_point = cp
    elif cp.startswith('L'): model.ctrl_layer = int(cp[1:])
    mc = MetaController(model.d); mc.load_state_dict(torch.load(ROOT / 'runs' / run / 'meta' / f'{tag}.pt', map_location='cpu'))
    Z, Y1, Y2, cos = [], [], [], []
    dm = Split('dev_meta')
    for s in range(0, n, 128):
        b = dm.batch(np.arange(s, min(s + 128, n)), lam_obs=0.0, device='cpu', pad_to=176)
        _, _, info = step_loss(model, mc, b, 0.0, sample=False)
        z = info['z'].cpu(); hdr = b['header']; aa = b['abs_act']
        for i in range(z.shape[0]):
            tr = torch.arange(int(hdr[i]), int(b['lengths'][i]) - 1, 2)
            seq = aa[i, (tr + 1).clamp(max=aa.shape[1] - 1)]
            sw = [j for j in range(1, len(seq)) if seq[j] != seq[j - 1] and seq[j] >= 0 and seq[j - 1] >= 0]
            if len(seq) < 4 or not sw: continue
            Z.append(z[i, 3]); Y1.append(int(seq[0])); Y2.append(int(seq[sw[0]]))
            cos.append(float(F.cosine_similarity(z[i, 0], z[i, 3], 0)))
    torch.set_grad_enabled(True)
    Z = torch.stack(Z); Y1 = torch.tensor(Y1); Y2 = torch.tensor(Y2); h = len(Z) // 2
    res = dict(run=run, tag=tag, episodes=len(Z), cos_z0_z3=float(np.mean(cos)),
               probe_first_pair=fit_probe(Z[:h], Y1[:h], Z[h:], Y1[h:], 4), chance_first=float(np.bincount(Y1[h:].numpy()).max() / (len(Z) - h)),
               probe_second_pair=fit_probe(Z[:h], Y2[:h], Z[h:], Y2[h:], 4), chance_second=float(np.bincount(Y2[h:].numpy()).max() / (len(Z) - h)))
    (ROOT / 'runs' / run / 'meta' / f'zcheck_{tag}.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
