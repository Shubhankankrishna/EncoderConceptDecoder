"""Read-only CPU probe of the stopped V10.1 checkpoint for the V10.2 review.

Measures (1) decoder concept-route router weights and contribution magnitudes,
(2) mask-and-renormalize versus value-zero ablations, (3) RBF query geometry and
kernel neighbourhoods, and (4) a calibrated mean-substitution control.

It never trains, never writes memory, and never modifies an existing file.
Output: probe_v10_1_interfaces.json beside this script (refuses to overwrite).
"""
from pathlib import Path
import hashlib, json, math, sys, time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture, token_loss
from v10_1.prepare_data import DATA

HERE = Path(__file__).resolve().parent
OUT = HERE / 'probe_v10_1_interfaces.json'
CKPT = ROOT / 'v10_1/runs/pretrain_16h/stopped_step_0061384.pt'
FAMILIES = ('text', 'math', 'sudoku_easy', 'sudoku_hard')
LARGE_N = dict(text=64, math=64, sudoku_easy=32, sudoku_hard=32)
LARGE_SEED, CALIB_SEED, BOOT = 20261012, 20261013, 4000


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def draws(seed, counts):
    corpus = Corpus('dev', seed=seed, data_dir=DATA)
    return {f: [corpus.batch(1, family=f) for _ in range(counts[f])] for f in corpus.arrays}


class MeanSubstitution:
    """Replace decoder concept inputs (forecast and continuous u) by fixed means."""
    def __init__(self, model, forecast_mean, u_mean):
        self.decoder, self.f, self.u = model.decoder, forecast_mean, u_mean
        self.original = model.decoder.forward
    def __enter__(self):
        def patched(early, final, forecast, mask, rope, u, lengths, *a, **k):
            return self.original(early, final, self.f.expand_as(forecast), mask, rope,
                                 self.u.expand_as(u), lengths, *a, **k)
        self.decoder.forward = patched
    def __exit__(self, *exc):
        self.decoder.forward = self.original


@torch.no_grad()
def main():
    if OUT.exists(): raise SystemExit('Refusing to overwrite ' + str(OUT))
    torch.set_num_threads(2); torch.manual_seed(0); started = time.time()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    c = ModelConfig(**ck['model_config']) if isinstance(ck.get('model_config'), dict) else ModelConfig()
    model = V101(c, Architecture(**ck['architecture'])); model.load_state_dict(ck['model'])
    model.place('cpu').eval(); step = int(ck['step']); del ck
    dec = model.decoder

    # Calibration means for the mean-substitution control come from a disjoint draw.
    calib = draws(CALIB_SEED, dict(text=32, math=32, sudoku_easy=16, sudoku_hard=16))
    means = {}
    for fam, batches in calib.items():
        fs, us = [], []
        for b in batches:
            out = model(b['ids'], b['lengths']); v = out['valid_groups']
            fs.append(out['forecast'][v].reshape(-1, 2, c.width)); us.append(out['u'][v].reshape(-1, 2, c.width))
        means[fam] = (torch.cat(fs).mean(0), torch.cat(us).mean(0))

    panels = dict(routine=draws(1811, dict(text=2, math=2, sudoku_easy=2, sudoku_hard=2)),
                  large=draws(LARGE_SEED, LARGE_N))
    variants = {
        'both_off_renormalized': dict(ablate=True),
        'pq_off': dict(ablate_pq=True),
        'continuous_off_renormalized': dict(ablate_continuous=True),
        'continuous_zero_value': dict(zero_continuous=True),
        'both_off_zero_value': dict(ablate_pq=True, zero_continuous=True),
        'one_visit': dict(visits=1),
    }
    result = dict(checkpoint=str(CKPT.relative_to(ROOT)), checkpoint_sha256=sha(CKPT), step=step,
                  device='cpu', threads=2, panels={}, router={}, query_geometry={},
                  scope='Interventions on one frozen V10.1 checkpoint; dev split only; no training, '
                        'no memory writes, no optimizer state used.')
    queries_by_fam = {}
    for pname, panel in panels.items():
        result['panels'][pname] = {}
        for fam, batches in panel.items():
            docs = []
            route_w = torch.zeros(len(dec.layers), 3); route_rows = 0
            contrib = {k: torch.zeros(len(dec.layers)) for k in ('e1', 'e3', 'concept')}
            for b in batches:
                x, l, y, m = b['ids'], b['lengths'], b['targets'], b['target_mask']; n = int(m.sum())
                if n == 0: continue
                out = model(x, l); rec = dict(n=n, nll=float(token_loss(out, y, m)))
                for name, flags in variants.items():
                    off = model(x, l, codes=out['codes'], **flags); rec[name] = float(token_loss(off, y, m))
                fmean, umean = means[fam]
                with MeanSubstitution(model, fmean, umean):
                    rec['mean_substitution'] = float(token_loss(model(x, l, codes=out['codes']), y, m))
                docs.append(rec)
                # Router weights and per-source routed contribution RMS on legal rows (i>=7, i<length).
                pos = torch.arange(x.shape[1]); legal = (pos[None] >= 7) & (pos[None] < l[:, None])
                if legal.any():
                    early, h, _, _ = model.encode(x, l, None)
                    enc = [p(nm(z)) for p, nm, z in zip(dec.src_proj, dec.src_norm, (early, h))]
                    cont = dec.align(dec.concept_projection(out['u']), x.shape[1])
                    for i, w in enumerate(out['decoder_routes']):
                        wl = w[legal]; route_w[i] += wl.sum(0)
                        s = dec.scales[i]
                        for j, (key, src) in enumerate(zip(('e1', 'e3', 'concept'), enc + [cont])):
                            contrib[key][i] += float(((s * wl[:, j:j + 1] * src[legal]) ** 2).mean(-1).sqrt().sum())
                    route_rows += int(legal.sum())
                if pname == 'large':
                    v = out['valid_groups']
                    queries_by_fam.setdefault(fam, []).append((out['queries'][v], out['codes'][v]))
            tot = sum(d['n'] for d in docs)
            agg = {k: sum(d[k] * d['n'] for d in docs) / tot for k in docs[0] if k != 'n'}
            deltas = {k: agg[k] - agg['nll'] for k in agg if k != 'nll'}
            # Document-level paired bootstrap for token-weighted deltas.
            rng = np.random.default_rng(7); n_arr = np.array([d['n'] for d in docs], float)
            ci = {}
            for k in deltas:
                diff = np.array([(d[k] - d['nll']) * d['n'] for d in docs])
                idx = rng.integers(0, len(docs), (BOOT, len(docs)))
                boot = diff[idx].sum(1) / n_arr[idx].sum(1)
                ci[k] = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
            result['panels'][pname][fam] = dict(documents=len(docs), supervised_tokens=tot, nll=agg['nll'],
                                                delta_nll=deltas, bootstrap95=ci if len(docs) > 2 else None)
            if route_rows:
                result['router'].setdefault(pname, {})[fam] = dict(
                    legal_rows=route_rows,
                    mean_weight_e1_e3_concept=(route_w / route_rows).tolist(),
                    routed_contribution_rms_e1_e3_concept={k: (v / route_rows).tolist() for k, v in contrib.items()})
    result['router']['init_concept_weight'] = float(torch.tensor([0., 0., -2.]).softmax(0)[2])
    result['router']['scale_rms_per_layer'] = dec.scales.square().mean(-1).sqrt().tolist()
    result['router']['fusion_rms'] = float(dec.fusion.square().mean().sqrt())

    gain = model.core.exit_norm.weight
    result['query_geometry']['exit_norm_gain_rms'] = float(gain.square().mean().sqrt())
    for fam, items in queries_by_fam.items():
        q = torch.cat([a for a, _ in items]); k = torch.cat([b for _, b in items])
        doc = torch.cat([torch.full((len(a),), i) for i, (a, _) in enumerate(items)])
        norms = q.norm(dim=-1); d2 = torch.cdist(q, q).square()
        same = (k[:, None] == k[None]) & (doc[:, None] != doc[None])
        entry = dict(queries=len(q), radius_median=float(norms.median()), radius_min=float(norms.min()),
                     radius_max=float(norms.max()), cross_document_same_code_pairs=int(same.sum()))
        if same.any():
            dist = d2[same].sqrt()
            entry['same_code_distance_quantiles_05_50_95_max'] = [float(dist.quantile(p)) for p in (.05, .5, .95)] + [float(dist.max())]
            entry['kernel'] = {}
            for sigma in (0.25, 0.5, 1.0):
                w = torch.exp(-d2 / (2 * sigma ** 2)) * same
                ws = w[same]
                neff = (w.sum(1) ** 2 / w.square().sum(1).clamp_min(1e-30))[same.any(1)]
                entry['kernel'][str(sigma)] = dict(
                    weight_quantiles_05_50_95=[float(ws.quantile(p)) for p in (.05, .5, .95)],
                    floor_exp_minus_2r2_over_sigma2=float(math.exp(-2 * float(norms.median()) ** 2 / sigma ** 2)),
                    median_effective_neighbours=float(neff.median()),
                    median_same_code_bank_size=float(same.sum(1)[same.any(1)].float().median()))
        result['query_geometry'][fam] = entry
    result['seconds'] = time.time() - started
    result['script_sha256'] = sha(Path(__file__))
    OUT.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
