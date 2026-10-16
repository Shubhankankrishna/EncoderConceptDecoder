"""Read-only probe: do V10.1's K=8 candidate codes differ, and is the hindsight
winner predictable from the prefix?

Uses the trainer's own path (chunk_batch -> candidate_forecasts ->
per_document_chunk_ce + 0.1 * next-pool MSE) so utilities match what selected
winners during training. For several eligible boundaries per document it records
all eight utilities and reports:

  spread         max - min utility over codes (nats of chunk CE, plus 0.1 MSE)
  gap            second best - best
  regret         utility of the prefix-selected code - best
  code effect    share of utility variance (after removing each boundary's mean)
                 explained by a per-code constant: high means a code helps or
                 hurts everywhere, i.e. it is a global bias, not a strategy
  predictability held-out multinomial logistic regression from the code-free
                 pooled state P3 at the boundary to the winner, versus the
                 majority-class rate and the model's own selector

Dev split only, frozen step-61,384 checkpoint, output refuses to overwrite.
"""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture, chunk_batch, per_document_chunk_ce
from v10_1.prepare_data import DATA

HERE = Path(__file__).resolve().parent
OUT = HERE / 'probe_candidate_utility.json'
CKPT = ROOT / 'v10_1/runs/pretrain_16h/stopped_step_0061384.pt'
SCORED = dict(text=64, math=64, sudoku_easy=32, sudoku_hard=32)      # seed 20261012, as in the other probes
TRAIN = dict(text=192, math=192, sudoku_easy=96, sudoku_hard=96)     # seed 20261014, classifier fitting only
PER_DOC = 6


def draws(seed, counts):
    corpus = Corpus('dev', seed=seed, data_dir=DATA)
    return {f: [corpus.batch(1, family=f) for _ in range(counts[f])] for f in corpus.arrays}


@torch.no_grad()
def boundaries_for(model, batches, rng):
    rows = []
    for b in batches:
        x, l, y, m = b['ids'], b['lengths'], b['targets'], b['target_mask']
        out = model(x, l)
        # All eligible boundaries of this document, subsampled.
        options = [a for a in range(out['valid_groups'].shape[1]) if _eligible(x, l, y, m, out, a)]
        if not options: continue
        for at in rng.choice(options, min(PER_DOC, len(options)), replace=False):
            at_t = torch.tensor([int(at)])
            chunk = chunk_batch(x, l, y, m, out['valid_groups'], boundaries=at_t)
            sub = {k: out[k] for k in ('pooled', 'sources', 'codes', 'valid_groups')}
            pred = model.candidate_forecasts(sub, l, at_t)                         # [1, K, 2, d]
            k = model.c.codes
            has_next = (at_t + 1) < out['valid_groups'].sum(-1)
            target = out['pooled'][0, min(int(at) + 1, out['pooled'].shape[1] - 1)]
            mse = (pred[0] - target).square().mean((-1, -2)) * has_next.double().float()
            expanded = {kk: chunk[kk].repeat_interleave(k, 0) for kk in ('previous', 'labels', 'mask')}
            ce = per_document_chunk_ce(model, pred.flatten(0, 1), expanded)
            utility = (ce + .1 * mse).double().numpy()
            sel_logits = out['selector'][0, int(at)]
            rows.append(dict(utility=utility, selected=int(out['codes'][0, int(at)]),
                             selector_probs=sel_logits.softmax(-1).double().numpy(),
                             p3=out['pooled'][0, int(at)].flatten().double().numpy()))
    return rows


def _eligible(x, l, y, m, out, a):
    try:
        return chunk_batch(x, l, y, m, out['valid_groups'], boundaries=torch.tensor([a])) is not None
    except ValueError:
        return False


def summarize(rows):
    u = np.stack([r['utility'] for r in rows]); s = np.sort(u, 1)
    win = u.argmin(1); sel = np.array([r['selected'] for r in rows])
    centred = u - u.mean(1, keepdims=True)
    code_effect = centred.mean(0)
    share = float((code_effect[None] ** 2).sum() * len(u) / (centred ** 2).sum())
    probs = np.stack([r['selector_probs'] for r in rows])
    return dict(boundaries=len(u), utility_mean=float(u.mean()),
                spread_median=float(np.median(s[:, -1] - s[:, 0])), spread_p90=float(np.quantile(s[:, -1] - s[:, 0], .9)),
                gap_median=float(np.median(s[:, 1] - s[:, 0])),
                regret_mean=float((u[np.arange(len(u)), sel] - s[:, 0]).mean()),
                within_boundary_utility_std_median=float(np.median(u.std(1))),
                code_main_effect_share=share, code_main_effect=code_effect.tolist(),
                winner_counts=np.bincount(win, minlength=u.shape[1]).tolist(),
                selected_counts=np.bincount(sel, minlength=u.shape[1]).tolist(),
                selector_accuracy=float((sel == win).mean()),
                selector_prob_of_winner_mean=float(probs[np.arange(len(u)), win].mean()),
                selector_winner_ce=float(-np.log(probs[np.arange(len(u)), win].clip(1e-12)).mean()))


def main():
    if OUT.exists(): raise SystemExit('Refusing to overwrite ' + str(OUT))
    torch.set_num_threads(2); started = time.time(); rng = np.random.default_rng(17)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = V101(ModelConfig(**ck['model_config']), Architecture(**ck['architecture']))
    model.load_state_dict(ck['model']); model.place('cpu').eval(); step = int(ck['step']); del ck
    scored, train = draws(20261012, SCORED), draws(20261014, TRAIN)
    result = dict(checkpoint=str(CKPT.relative_to(ROOT)), step=step, boundaries_per_document=PER_DOC,
                  utility='chunk CE of the candidate forecast + 0.1 * MSE to the next pooled state (trainer definition)',
                  families={})
    for fam in SCORED:
        test_rows, train_rows = boundaries_for(model, scored[fam], rng), boundaries_for(model, train[fam], rng)
        entry = summarize(test_rows)
        xtr = np.stack([r['p3'] for r in train_rows]); ytr = np.array([r['utility'].argmin() for r in train_rows])
        xte = np.stack([r['p3'] for r in test_rows]); yte = np.array([r['utility'].argmin() for r in test_rows])
        scaler = StandardScaler().fit(xtr)
        accs = {}
        for c in (0.01, 0.1, 1.0):
            clf = LogisticRegression(C=c, max_iter=2000).fit(scaler.transform(xtr), ytr)
            accs[str(c)] = float((clf.predict(scaler.transform(xte)) == yte).mean())
        majority = int(np.bincount(ytr, minlength=model.c.codes).argmax())
        entry.update(classifier_train_boundaries=len(xtr), classifier_heldout_accuracy_by_C=accs,
                     majority_class_heldout_accuracy=float((yte == majority).mean()), chance=1 / model.c.codes)
        result['families'][fam] = entry
    result['seconds'] = time.time() - started
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    for fam, e in result['families'].items():
        print(fam, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k != 'code_main_effect'})


if __name__ == '__main__':
    main()
