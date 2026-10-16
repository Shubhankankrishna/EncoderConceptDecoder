"""Read-only companion to probe_latent_baselines.py: why is the raw PQ forecast
worse than a linear map of itself?

Two mechanisms predict different things. If the soft-PQ convex combination
shrinks the forecast toward the codebook centroid, var(F2)/var(target) is small
and a per-coordinate affine fit (scale and shift only) recovers most of the
ridge gain. If F lives in a rotated subspace, the affine fit does not help and
only a full linear map does. Same checkpoint, panels and seeds as the baseline
probe; step 61,384 only. Output refuses to overwrite.
"""
from pathlib import Path
import hashlib, json, math, sys, time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture
from v10_1.prepare_data import DATA

HERE = Path(__file__).resolve().parent
OUT = HERE / 'probe_forecast_spread.json'
CKPT = ROOT / 'v10_1/runs/pretrain_16h/stopped_step_0061384.pt'
LARGE = dict(text=64, math=64, sudoku_easy=32, sudoku_hard=32)
CALIB = dict(text=32, math=32, sudoku_easy=16, sudoku_hard=16)


def draws(seed, counts):
    corpus = Corpus('dev', seed=seed, data_dir=DATA)
    return {f: [corpus.batch(1, family=f) for _ in range(counts[f])] for f in corpus.arrays}


@torch.no_grad()
def collect(model, batches):
    f2, u2, y, ent, top = [], [], [], [], []
    for b in batches:
        out = model(b['ids'], b['lengths'])
        pairs = out['valid_groups'][0, 1:]
        if int(pairs.sum()) == 0: continue
        f2.append(out['forecast'][0, :-1][pairs].flatten(-2)); u2.append(out['u'][0, :-1][pairs].flatten(-2))
        y.append(out['pooled'][0, 1:][pairs].flatten(-2))
        p = out['probs'][0][out['valid_groups'][0]]          # [groups, slots, segments, words]
        ent.append(-(p * p.clamp_min(1e-12).log()).sum(-1).flatten(0, 1)); top.append(p.max(-1).values.flatten(0, 1))
    cat = lambda z: torch.cat(z).double().numpy()
    return cat(f2), cat(u2), cat(y), cat(ent), cat(top)


def main():
    if OUT.exists(): raise SystemExit('Refusing to overwrite ' + str(OUT))
    torch.set_num_threads(2); started = time.time()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = V101(ModelConfig(**ck['model_config']), Architecture(**ck['architecture']))
    model.load_state_dict(ck['model']); model.place('cpu').eval(); step = int(ck['step']); del ck
    cb = model.head.codebook.detach().double()               # [segments, words, 16]
    result = dict(checkpoint=str(CKPT.relative_to(ROOT)), step=step, pq_words=model.c.pq_words,
                  max_entropy_nats=math.log(model.c.pq_words),
                  codebook_rms=float(cb.square().mean().sqrt()),
                  codebook_centroid_rms=float(cb.mean(1).square().mean().sqrt()), families={})
    calib, large = draws(20261013, CALIB), draws(20261012, LARGE)
    for fam in LARGE:
        cf, _, cy, _, _ = collect(model, calib[fam])
        f2, u2, y, ent, top = collect(model, large[fam])
        var = y.var(0)
        # Per-coordinate affine fit on calibration: y ~ a*f + c, least squares per coordinate.
        a = ((cf - cf.mean(0)) * (cy - cy.mean(0))).sum(0) / ((cf - cf.mean(0)) ** 2).sum(0).clip(1e-12)
        c = cy.mean(0) - a * cf.mean(0)
        affine = a * f2 + c
        corr = np.array([np.corrcoef(f2[:, j], y[:, j])[0, 1] for j in range(y.shape[1])])
        result['families'][fam] = dict(
            pairs=len(y), target_variance=float(var.mean()),
            forecast_to_target_variance=float(f2.var(0).mean() / var.mean()),
            concept_state_to_target_variance=float(u2.var(0).mean() / var.mean()),
            forecast_raw_fvu=float(((f2 - y) ** 2).mean() / var.mean()),
            forecast_affine_fvu=float(((affine - y) ** 2).mean() / var.mean()),
            affine_scale_median=float(np.median(a)),
            per_coordinate_correlation_median=float(np.median(corr)),
            mean_offset_sq_over_variance=float(((f2.mean(0) - y.mean(0)) ** 2).mean() / var.mean()),
            pq_entropy_mean_nats=float(ent.mean()), pq_top_probability_mean=float(top.mean()))
    result['seconds'] = time.time() - started
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
