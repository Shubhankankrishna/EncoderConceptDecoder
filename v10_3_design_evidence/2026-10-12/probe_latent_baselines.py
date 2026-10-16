"""Read-only probe: does V10.1's concept forecast beat trivial baselines?

The training "latent" loss is the MSE between the PQ forecast of group g and the
stopped, online pooled E3 state of group g+1. Its absolute value can rise while
prediction improves, because the target's spread changes during training. This
probe scores the forecast against baselines on the same targets:

  mean        per-coordinate mean of targets from a disjoint calibration draw
  persistence predict P3[g+1] by P3[g]
  ridge(P3)   linear map P3[g] -> P3[g+1], fitted on the calibration draw
  ridge(U2)   linear map from the final concept state U[g] -> P3[g+1]
  ridge(F2)   linear map from the PQ forecast F[g] -> P3[g+1]

and reports every MSE divided by the target variance on the scored panel
(fraction of variance unexplained). Two checkpoints of the same run are scored.
Dev split only; no training, no memory writes; output refuses to overwrite.
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
OUT = HERE / 'probe_latent_baselines.json'
RUN = ROOT / 'v10_1/runs/pretrain_16h'
CHECKPOINTS = ('before_throughput_step_0003171.pt', 'stopped_step_0061384.pt')
LARGE = dict(text=64, math=64, sudoku_easy=32, sudoku_hard=32)
CALIB = dict(text=32, math=32, sudoku_easy=16, sudoku_hard=16)
LAMBDAS = (0.1, 1.0, 10.0, 100.0)
BOOT = 2000


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def draws(seed, counts):
    corpus = Corpus('dev', seed=seed, data_dir=DATA)
    return {f: [corpus.batch(1, family=f) for _ in range(counts[f])] for f in corpus.arrays}


@torch.no_grad()
def collect(model, batches):
    """Per document: arrays of (P3[g], U2[g], F1[g], F2[g], P3[g+1]) over valid pairs."""
    docs = []
    for b in batches:
        out = model(b['ids'], b['lengths'])
        pairs = out['valid_groups'][0, 1:]
        if int(pairs.sum()) == 0: continue
        sel = lambda z: z[0, :-1][pairs].flatten(-2).double().numpy()
        docs.append(dict(p3=sel(out['pooled']), u2=sel(out['states'][-1]), f1=sel(out['forecasts'][0]),
                         f2=sel(out['forecasts'][-1]), y=out['pooled'][0, 1:][pairs].flatten(-2).double().numpy()))
    return docs


def ridge(x, y, lam):
    mx, my = x.mean(0), y.mean(0)
    xc = x - mx
    w = np.linalg.solve(xc.T @ xc + lam * len(x) * np.eye(x.shape[1]), xc.T @ (y - my))
    return lambda z: (z - mx) @ w + my


def per_doc_mse(docs, predict):
    # Matches latent_errors: mean over slots and coordinates, then mean over pairs.
    return np.array([((predict(d) - d['y']) ** 2).mean(1).sum() for d in docs]), np.array([len(d['y']) for d in docs])


def main():
    if OUT.exists(): raise SystemExit('Refusing to overwrite ' + str(OUT))
    torch.set_num_threads(2); started = time.time(); rng = np.random.default_rng(5)
    calib_batches, large_batches = draws(20261013, CALIB), draws(20261012, LARGE)
    result = dict(scope='Frozen V10.1 checkpoints; dev split; calibration seed 20261013, scored panel seed 20261012.',
                  checkpoints={})
    for name in CHECKPOINTS:
        path = RUN / name
        ck = torch.load(path, map_location='cpu', weights_only=False)
        model = V101(ModelConfig(**ck['model_config']), Architecture(**ck['architecture']))
        model.load_state_dict(ck['model']); model.place('cpu').eval(); step = int(ck['step']); del ck
        entry = dict(file=str(path.relative_to(ROOT)), sha256=sha(path), step=step,
                     pooler_outn_gain_rms=float(model.pooler.outn.weight.square().mean().sqrt()),
                     core_exit_gain_rms=float(model.core.exit_norm.weight.square().mean().sqrt()), families={})
        for fam in LARGE:
            cal, test = collect(model, calib_batches[fam]), collect(model, large_batches[fam])
            cy = np.concatenate([d['y'] for d in cal]); ty = np.concatenate([d['y'] for d in test])
            var = float(ty.var(0).mean())
            predictors = {
                'forecast_visit1': lambda d: d['f1'], 'forecast_visit2': lambda d: d['f2'],
                'calibration_mean': lambda d, m=cy.mean(0): np.broadcast_to(m, d['y'].shape),
                'persistence_p3': lambda d: d['p3'], 'zero': lambda d: np.zeros_like(d['y'])}
            fitted = {}
            for src in ('p3', 'u2', 'f2'):
                cx = np.concatenate([d[src] for d in cal]); half = len(cx) // 2
                # Choose lambda on a split of the calibration draw only.
                best = min(LAMBDAS, key=lambda lam: float(((ridge(cx[:half], cy[:half], lam)(cx[half:]) - cy[half:]) ** 2).mean()))
                f = ridge(cx, cy, best); fitted[src] = best
                predictors[f'ridge_{src}'] = lambda d, f=f, src=src: f(d[src])
            fam_out = dict(documents=len(test), pairs=int(len(ty)), target_variance=var,
                           target_rms=float(np.sqrt((ty ** 2).mean())), ridge_lambda=fitted, mse={}, fraction_unexplained={})
            sums = {}
            for k, p in predictors.items():
                s, n = per_doc_mse(test, p); sums[k] = s
                fam_out['mse'][k] = float(s.sum() / n.sum()); fam_out['fraction_unexplained'][k] = fam_out['mse'][k] / var
            n = per_doc_mse(test, predictors['zero'])[1]
            idx = rng.integers(0, len(test), (BOOT, len(test)))
            fam_out['forecast2_minus_baseline_ci95'] = {
                k: [float(np.quantile(((sums['forecast_visit2'] - sums[k])[idx]).sum(1) / n[idx].sum(1), q)) for q in (.025, .975)]
                for k in ('calibration_mean', 'persistence_p3', 'ridge_p3')}
            entry['families'][fam] = fam_out
        result['checkpoints'][str(step)] = entry
        del model
    result['seconds'] = time.time() - started
    result['script_sha256'] = sha(Path(__file__))
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    for step, e in result['checkpoints'].items():
        print('step', step, 'pooler gain rms %.3f' % e['pooler_outn_gain_rms'])
        for fam, f in e['families'].items():
            print(' ', fam, 'var %.3f' % f['target_variance'], {k: round(v, 3) for k, v in f['fraction_unexplained'].items()})


if __name__ == '__main__':
    main()
