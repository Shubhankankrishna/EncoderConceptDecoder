"""Read-only probe: can Local-Support-Learning GMM gates separate data
distributions inside the frozen V10.1 model?

LSL (Ben-Kish et al., 2026) gates a low-rank adapter per weight matrix with
  open(x) = log Phi_pos(x) > log Phi_neg(x)
where Phi_pos is a diagonal GMM fitted on the new phase's inputs to that matrix
and Phi_neg on a small generic reference sample. The paper reports results on
1.5B-7B models and notes that retention depends on representation quality. This
probe measures the gate at width 128.

Sites are the inputs of attention (qkv) and FFN (up) matrices:
  e2.<i>.attn / e2.<i>.ffn   Reader output of two-stream block i, per visit
  core.<i>.attn / .ffn       RMSNorm output inside concept block i, per visit
  dec.<i>.attn / .ffn        RMSNorm output inside decoder block i
Protocol "reference": Phi_neg on text (the generic reference); each symbolic
family in turn is the new phase. Open rates are reported on held-out text and on
the other symbolic families, which neither GMM saw (the paper's OOD prior tasks).
Protocol "leave_one_out": Phi_neg on the other three families.
Both report raw per-token decisions and the paper's EMA smoothing (alpha=0.1,
reset per document). Also reports whether visit-1 and visit-2 activations of a
shared-weight block are separable. Dev split, frozen checkpoint, no training;
output refuses to overwrite.

GMMs use data-sampled initialisation and one OpenMP thread: k-means
initialisation crashed (SIGSEGV) alongside PyTorch's OpenMP runtime on macOS.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture
from v10_1.prepare_data import DATA

HERE = Path(__file__).resolve().parent
OUT = HERE / 'probe_gmm_gates.json'
CKPT = ROOT / 'v10_1/runs/pretrain_16h/stopped_step_0061384.pt'
DOCS = dict(text=64, math=192, sudoku_easy=32, sudoku_hard=32)
FAMILIES = tuple(DOCS)
COMPONENTS, ALPHA, CAP = 4, 0.1, 4000


def collect(model, corpus):
    """acts[site][family] -> list of per-document arrays [rows, 128] in sequence order."""
    acts, calls = {}, {}
    def hook(name):
        def fn(module, inputs):
            n = calls.get(name, 0); calls[name] = n + 1
            key = f'{name}.v{n + 1}' if name.split('.')[0] in ('e2', 'core') else name
            hook.current.setdefault(key, []).append(inputs[0][0].detach().double().numpy())
        return fn
    handles = []
    for i, blk in enumerate(model.e2):
        handles += [blk.attn.register_forward_pre_hook(hook(f'e2.{i}.attn')), blk.ffn.register_forward_pre_hook(hook(f'e2.{i}.ffn'))]
    for i, blk in enumerate(model.core.layers):
        handles += [blk.attn.register_forward_pre_hook(hook(f'core.{i}.attn')), blk.ffn.register_forward_pre_hook(hook(f'core.{i}.ffn'))]
    for i, blk in enumerate(model.decoder.layers):
        handles += [blk.attn.register_forward_pre_hook(hook(f'dec.{i}.attn')), blk.ffn.register_forward_pre_hook(hook(f'dec.{i}.ffn'))]
    with torch.no_grad():
        for fam in FAMILIES:
            for _ in range(DOCS[fam]):
                b = corpus.batch(1, family=fam); calls.clear(); hook.current = {}
                out = model(b['ids'], b['lengths'])
                length = int(b['lengths'][0]); groups = int(out['valid_groups'][0].sum())
                for key, mats in hook.current.items():
                    x = mats[0]
                    x = x[:2 * groups] if key.startswith('core') else x[:length]
                    if len(x): acts.setdefault(key, {}).setdefault(fam, []).append(x)
    for h in handles: h.remove()
    return acts


def split_docs(docs, rng):
    order = rng.permutation(len(docs)); half = len(docs) // 2
    return [docs[i] for i in order[:half]], [docs[i] for i in order[half:]]


def fit(rows, rng):
    rows = np.concatenate(rows)
    if len(rows) > CAP: rows = rows[rng.choice(len(rows), CAP, replace=False)]
    return GaussianMixture(COMPONENTS, covariance_type='diag', reg_covar=1e-4, init_params='random_from_data', n_init=2, random_state=0).fit(rows)


def rates(pos, neg, docs):
    raw, smooth = [], []
    for x in docs:
        g = (pos.score_samples(x) > neg.score_samples(x)).astype(float); raw.append(g)
        s, out = 0., []
        for v in g:
            s = ALPHA * v + (1 - ALPHA) * s; out.append(s > .5)
        smooth.append(np.array(out, float))
    return float(np.concatenate(raw).mean()), float(np.concatenate(smooth).mean())


def main():
    if OUT.exists(): raise SystemExit('Refusing to overwrite ' + str(OUT))
    torch.set_num_threads(2); started = time.time(); rng = np.random.default_rng(23)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = V101(ModelConfig(**ck['model_config']), Architecture(**ck['architecture']))
    model.load_state_dict(ck['model']); model.place('cpu').eval(); step = int(ck['step']); del ck
    acts = collect(model, Corpus('dev', seed=20261012, data_dir=DATA))
    result = dict(checkpoint=str(CKPT.relative_to(ROOT)), step=step, gmm_components=COMPONENTS,
                  covariance='diag', smoothing_alpha=ALPHA, fit_cap_rows=CAP, documents=DOCS,
                  rows={k: {f: int(sum(len(x) for x in v[f])) for f in v} for k, v in acts.items()},
                  reference={}, leave_one_out={}, visit_separability={})
    for site, by_fam in acts.items():
        halves = {f: split_docs(by_fam[f], rng) for f in FAMILIES}
        # Protocol 1: text is the generic reference; each symbolic family is a new phase.
        neg = fit(halves['text'][0], rng); res = {}
        for new in FAMILIES[1:]:
            pos = fit(halves[new][0], rng)
            entry = dict(hit_heldout_new=rates(pos, neg, halves[new][1]), open_heldout_text=rates(pos, neg, halves['text'][1]))
            for other in FAMILIES[1:]:
                if other != new: entry[f'open_unseen_{other}'] = rates(pos, neg, halves[other][1])
            res[new] = entry
        result['reference'][site] = res
        # Protocol 2: reference is the mix of the other three families.
        res = {}
        for new in FAMILIES:
            neg2 = fit([x for f in FAMILIES if f != new for x in halves[f][0]], rng); pos = fit(halves[new][0], rng)
            res[new] = dict(hit_heldout_new=rates(pos, neg2, halves[new][1]),
                            **{f'open_heldout_{f}': rates(pos, neg2, halves[f][1]) for f in FAMILIES if f != new})
        result['leave_one_out'][site] = res
    # Shared-weight sites: are visit-1 and visit-2 inputs separable by the same GMM test?
    for site in sorted({k.rsplit('.v', 1)[0] for k in acts if '.v' in k}):
        sep = {}
        for fam in FAMILIES:
            a, b = split_docs(acts[f'{site}.v1'][fam], np.random.default_rng(1)), split_docs(acts[f'{site}.v2'][fam], np.random.default_rng(1))
            g1, g2 = fit(a[0], rng), fit(b[0], rng)
            v1 = np.concatenate(a[1]); v2 = np.concatenate(b[1])
            acc = .5 * ((g1.score_samples(v1) > g2.score_samples(v1)).mean() + (g2.score_samples(v2) > g1.score_samples(v2)).mean())
            sep[fam] = float(acc)
        result['visit_separability'][site] = sep
    result['seconds'] = time.time() - started
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    for site, res in result['reference'].items():
        print(site, {n: {k: [round(v, 2) for v in vals] for k, vals in e.items()} for n, e in res.items()})
    print('visit separability', {s: {f: round(v, 2) for f, v in d.items()} for s, d in result['visit_separability'].items()})


if __name__ == '__main__':
    main()
