"""Read-only probe: what does the second concept visit add, and would the
decoder gain anything from the visit-1 state or the change between visits?

For the frozen step-61,384 checkpoint, per family:
  relative change   RMS(U2 - U1) / RMS(U2) over valid groups
  cosine            mean cosine similarity of U1 and U2 per slot
  predictive value  FVU of ridge readouts of the next pooled state P3[g+1]
                    from U1, U2, the delta U2 - U1, and [U1, U2] concatenated,
                    fitted on the calibration draw and scored on the panel
Same panels and seeds as probe_latent_baselines.py. Dev split, no training;
output refuses to overwrite.
"""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture
from v10_1.prepare_data import DATA

HERE = Path(__file__).resolve().parent
OUT = HERE / 'probe_visit_delta.json'
CKPT = ROOT / 'v10_1/runs/pretrain_16h/stopped_step_0061384.pt'
LARGE = dict(text=64, math=64, sudoku_easy=32, sudoku_hard=32)
CALIB = dict(text=32, math=32, sudoku_easy=16, sudoku_hard=16)
LAMBDAS = (0.1, 1.0, 10.0, 100.0)


def draws(seed, counts):
    corpus = Corpus('dev', seed=seed, data_dir=DATA)
    return {f: [corpus.batch(1, family=f) for _ in range(counts[f])] for f in corpus.arrays}


@torch.no_grad()
def collect(model, batches):
    rows = dict(u1=[], u2=[], y=[], all_u1=[], all_u2=[])
    for b in batches:
        out = model(b['ids'], b['lengths'])
        valid, pairs = out['valid_groups'][0], out['valid_groups'][0, 1:]
        u1, u2 = out['states'][0][0], out['states'][-1][0]
        rows['all_u1'].append(u1[valid].reshape(-1, u1.shape[-1])); rows['all_u2'].append(u2[valid].reshape(-1, u2.shape[-1]))
        if int(pairs.sum()) == 0: continue
        rows['u1'].append(u1[:-1][pairs].flatten(-2)); rows['u2'].append(u2[:-1][pairs].flatten(-2))
        rows['y'].append(out['pooled'][0, 1:][pairs].flatten(-2))
    return {k: torch.cat(v).double().numpy() for k, v in rows.items()}


def ridge(x, y, lam):
    mx, my = x.mean(0), y.mean(0); xc = x - mx
    w = np.linalg.solve(xc.T @ xc + lam * len(x) * np.eye(x.shape[1]), xc.T @ (y - my))
    return lambda z: (z - mx) @ w + my


def fit_score(cx, cy, tx, ty):
    half = len(cx) // 2
    lam = min(LAMBDAS, key=lambda l: float(((ridge(cx[:half], cy[:half], l)(cx[half:]) - cy[half:]) ** 2).mean()))
    return float(((ridge(cx, cy, lam)(tx) - ty) ** 2).mean() / ty.var(0).mean()), lam


def main():
    if OUT.exists(): raise SystemExit('Refusing to overwrite ' + str(OUT))
    torch.set_num_threads(2); started = time.time()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = V101(ModelConfig(**ck['model_config']), Architecture(**ck['architecture']))
    model.load_state_dict(ck['model']); model.place('cpu').eval(); step = int(ck['step']); del ck
    calib, large = draws(20261013, CALIB), draws(20261012, LARGE)
    result = dict(checkpoint=str(CKPT.relative_to(ROOT)), step=step, families={})
    for fam in LARGE:
        c, t = collect(model, calib[fam]), collect(model, large[fam])
        a1, a2 = t['all_u1'], t['all_u2']
        cos = (a1 * a2).sum(1) / (np.linalg.norm(a1, axis=1) * np.linalg.norm(a2, axis=1))
        entry = dict(slots=len(a1), relative_change_rms=float(np.sqrt(((a2 - a1) ** 2).mean() / (a2 ** 2).mean())),
                     cosine_mean=float(cos.mean()), fvu={}, lambda_chosen={})
        feats = dict(u1=lambda r: r['u1'], u2=lambda r: r['u2'], delta=lambda r: r['u2'] - r['u1'],
                     u1_and_u2=lambda r: np.concatenate([r['u1'], r['u2']], 1))
        for name, f in feats.items():
            entry['fvu'][name], entry['lambda_chosen'][name] = fit_score(f(c), c['y'], f(t), t['y'])
        result['families'][fam] = entry
    result['seconds'] = time.time() - started
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    for fam, e in result['families'].items():
        print(fam, 'rel change %.3f cos %.3f' % (e['relative_change_rms'], e['cosine_mean']), {k: round(v, 3) for k, v in e['fvu'].items()})


if __name__ == '__main__':
    main()
