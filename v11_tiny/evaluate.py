"""Evaluation of one trained arm of the v11.md §6.6 objective test (dev split only).

  python -B -m v11_tiny.evaluate --run-dir v11_tiny/runs/K1_s1

Panels are v10_3_tiny.evaluate.panels() (deterministic, identical across arms), so per-document
rows can be paired. C0 and X8 use the inherited V10.3 evaluation functions unchanged (comparable
to V10_3_tiny_REPORT.md); the V11 arms use the functions here; ORD gets base NLL, alignment,
grid shift and generation. Every arm additionally gets:
  - alignment sensitivity: math/newmath re-scored with r = 0..7 dev-text tokens inserted after BOS
  - batched greedy generation, limit = answer tokens + 64, truncation counted separately
Writes evaluation.json in the run directory; refuses to overwrite.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from v10_10M.data import verify_sudoku
from v10_1.evaluate import score_math
from v10_3_tiny import evaluate as E
from v10_3_tiny.config import TinyConfig
from .model import build, V11Tiny, Ordinary

DATA = E.DATA
GENERATE = dict(math=32, newmath=64, sudoku_easy=8, sudoku_hard=8)        # math: per pretraining module
EOS, BOS = 2, 1


def to(b, device): return E.to(b, device)


def rows_of(docs, base, extra=None):
    out = []
    for i, (d, c) in enumerate(zip(docs, base)):
        r = dict(index=d['index'], source=d['source'], n=int(d['target_mask'].sum()), ce=c)
        if extra:
            for k, v in extra.items(): r[k] = v[i]
        out.append(r)
    return out


def summarize(rows, fam):
    tot = sum(r['n'] for r in rows)
    s = dict(documents=len(rows), tokens=tot, nll=sum(r['ce'] for r in rows) / tot)
    keys = [k for k in rows[0] if k not in ('index', 'source', 'n', 'ce')]
    if keys: s['delta'] = {k: (sum(r[k] for r in rows) - sum(r['ce'] for r in rows)) / tot for k in keys}
    if fam == 'math':
        s['by_module'] = {}
        for m in sorted({r['source'] for r in rows}):
            sel = [r for r in rows if r['source'] == m]; t = sum(r['n'] for r in sel)
            s['by_module'][m] = dict(documents=len(sel), nll=sum(r['ce'] for r in sel) / t,
                                     **({'delta': {k: (sum(r[k] for r in sel) - sum(r['ce'] for r in sel)) / t for k in keys}} if keys else {}))
    return s


@torch.no_grad()
def calibration_means(model, calib, device):
    fs, us = [], []
    for b in calib:
        o = model(*[to(b, device)[k] for k in ('ids', 'lengths')]); v = o['valid_groups'][0]
        fs.append(o['forecast'][0][v]); us.append(o['u'][0][v])
    return torch.cat(fs).mean(0), torch.cat(us).mean(0)


V11_INTERVENTIONS = dict(all_concept_off=dict(ablate=True), aligned_off=dict(ablate_aligned=True), reader_off=dict(ablate_reader=True),
                         reader_state_off=dict(zero_state=True), reader_forecast_off=dict(zero_forecast_memory=True),
                         visit1_for_visit2=dict(use_visit1=True))


@torch.no_grad()
def nll_v11(model, scored, calib, device):
    result = {}
    for fam, docs in scored.items():
        means = calibration_means(model, calib[fam], device)
        docs = [d for d in docs if int(d['target_mask'].sum())]; rows = []
        for start in range(0, len(docs), E.BATCH):
            chunk = docs[start:start + E.BATCH]; b = to(E.stack(chunk), device)
            base, out = E.batch_ce(model, b); extra = {}
            for name, flags in V11_INTERVENTIONS.items():
                extra[name] = E.batch_ce(model, b, **flags)[0]
            extra['mean_substitution'] = E.batch_ce(model, b, mean_substitution=means)[0]
            # Shuffle: each document reads another document's concept states at the same group index
            # (same norm, wrong content); groups the partner lacks fall back to the calibration mean.
            if len(chunk) > 1:
                perm = torch.roll(torch.arange(len(chunk), device=device), 1)
                valid = out['valid_groups'][perm][..., None, None]
                f_s = torch.where(valid, out['forecast'][perm], means[0]); u_s = torch.where(valid, out['u'][perm], means[1])
                extra['shuffle'] = E.batch_ce(model, b, mean_substitution=(f_s, u_s))[0]
            else:
                extra['shuffle'] = base
            rows += rows_of(chunk, base, extra)
        result[fam] = dict(summary=summarize(rows, fam), documents=rows)
    return result


@torch.no_grad()
def nll_plain(model, scored, device):
    result = {}
    for fam, docs in scored.items():
        docs = [d for d in docs if int(d['target_mask'].sum())]; rows = []
        for start in range(0, len(docs), E.BATCH):
            chunk = docs[start:start + E.BATCH]; base, _ = E.batch_ce(model, to(E.stack(chunk), device))
            rows += rows_of(chunk, base)
        result[fam] = dict(summary=summarize(rows, fam), documents=rows)
    return result


@torch.no_grad()
def latent_v11(model, scored, calib, device):
    """Next-concept forecast quality in the model's normalized target space (gate G3) and in raw pooled
    space (comparable to V10.3), plus visit-2 gain."""
    sd = model.target_var.clamp_min(.01).sqrt(); mu = model.target_mu
    s1 = float(model.log_sigma1.clamp_min(math.log(.01)).exp()) if hasattr(model, 'log_sigma1') else None
    s2 = float(model.log_sigma2.clamp_min(math.log(.01)).exp()) if hasattr(model, 'log_sigma2') else None
    def collect(docs):
        acc = {k: [] for k in ('f1', 'f2', 'u', 'p', 'y', 'praw', 'yraw', 'f2raw')}
        for b in docs:
            b = to(b, device); o = model(b['ids'], b['lengths']); pairs = o['valid_groups'][0, 1:]
            if int(pairs.sum()) == 0: continue
            tau = model.normalized_target(o['pooled'])[0]
            f1 = model.head(o['u1'])[0][0, :-1][pairs]; f2 = o['forecast'][0, :-1][pairs]
            vals = dict(f1=f1, f2=f2, u=o['u'][0, :-1][pairs], p=tau[:-1][pairs], y=tau[1:][pairs],
                        praw=o['pooled'][0, :-1][pairs], yraw=o['pooled'][0, 1:][pairs], f2raw=f2 * sd + mu)
            for k, v in vals.items(): acc[k].append(v.flatten(-2).cpu().double().numpy())
        return {k: np.concatenate(v) for k, v in acc.items()}
    out = {}
    for fam in scored:
        c, t = collect(calib[fam]), collect(scored[fam])
        def space(pred_keys, ykey, inkey, ukey, fkey):
            var = float(t[ykey].var(0).mean()); score = lambda p: float(((p - t[ykey]) ** 2).mean() / var)
            r = dict(target_variance=var, calibration_mean=score(np.broadcast_to(c[ykey].mean(0), t[ykey].shape)),
                     persistence=score(t[inkey]), ridge_input=score(E.fit_ridge(c[inkey], c[ykey])(t[inkey])),
                     ridge_state=score(E.fit_ridge(c[ukey], c[ykey])(t[ukey])),
                     ridge_forecast=score(E.fit_ridge(c[fkey], c[ykey])(t[fkey])))
            for k in pred_keys: r['direct_' + k] = score(t[k])
            return r
        norm = space(('f1', 'f2'), 'y', 'p', 'u', 'f2')
        norm['ridge_forecast_visit1'] = float(((E.fit_ridge(c['f1'], c['y'])(t['f1']) - t['y']) ** 2).mean() / norm['target_variance'])
        raw = space(('f2raw',), 'yraw', 'praw', 'u', 'f2raw')
        D = t['y'].shape[1]
        def gnll(f, s): return None if s is None else float(((f - t['y']) ** 2).sum(1).mean() / (2 * D * s) + .5 * math.log(2 * math.pi * s))
        out[fam] = dict(pairs=len(t['y']), normalized=norm, raw=raw,
                        forecast_variance_ratio=float(t['f2'].var(0).mean() / t['y'].var(0).mean()),
                        visit2_gain=dict(nll_visit1=gnll(t['f1'], s1), nll_visit2=gnll(t['f2'], s2), sigma1_sq=s1, sigma2_sq=s2,
                                         fvu_visit1=norm['direct_f1'], fvu_visit2=norm['direct_f2'],
                                         fvu_gain=norm['direct_f1'] - norm['direct_f2']),
                        participation_ratio=participation(t['p']))
    return out


@torch.no_grad()
def pooled_participation(model, scored, device):
    """Participation ratio (effective rank) of raw pooled slot states on the full scored panel, per family."""
    out = {}
    for fam, docs in scored.items():
        xs = []
        for start in range(0, len(docs), E.BATCH):
            b = to(E.stack(docs[start:start + E.BATCH]), device); o = model(b['ids'], b['lengths'])
            xs.append(o['pooled'][o['valid_groups']].flatten(1).cpu().double().numpy())
        x = np.concatenate(xs)
        out[fam] = dict(groups=len(x), participation_ratio=participation(x),
                        median_coordinate_variance=float(np.median(x.var(0))))
    return out


@torch.no_grad()
def calibration_stats(model, calib, device):
    xs = []
    for fam, docs in calib.items():
        for b in docs:
            b = to(b, device); o = model(b['ids'], b['lengths']); xs.append(o['pooled'][0][o['valid_groups'][0]])
    x = torch.cat(xs); return x.mean(0), x.var(0, unbiased=False)


def participation(x):
    ev = np.clip(np.linalg.eigvalsh(np.cov(x.T)), 0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))


def filler_tokens():
    arr, offsets, _, _ = E.load_family('text')
    seq = [int(t) for t in arr[offsets[7]:offsets[8]] if int(t) > 3]
    return seq[100:108]


@torch.no_grad()
def alignment(model, scored, device):
    """Re-score math/newmath with r = 0..7 filler tokens inserted after BOS; scored tokens unchanged."""
    filler = torch.tensor(filler_tokens()); out = {}
    for fam in ('math', 'newmath'):
        per_r = []
        for r in range(8):
            docs = []
            for d in scored[fam]:
                n = int(d['lengths'][0]); seq = torch.cat((d['ids'][0, :n], d['targets'][0, n - 1:n]))
                assert int(seq[0]) == BOS
                new = torch.cat((seq[:1], filler[:r], seq[1:])); m = len(new) - 1
                t = ((m - 1) // 8 + 1) * 8
                x = torch.zeros(1, t, dtype=torch.long); y = torch.zeros_like(x); mask = torch.zeros_like(x, dtype=torch.bool)
                x[0, :m], y[0, :m] = new[:-1], new[1:]
                src = torch.nonzero(d['target_mask'][0]).flatten()
                mask[0, src + r] = True
                assert int(mask.sum()) == int(d['target_mask'].sum()) and torch.equal(y[0, src + r], d['targets'][0, src])
                docs.append(dict(ids=x, targets=y, target_mask=mask, lengths=torch.tensor([m]), index=d['index'], source=d['source']))
            num = den = 0.
            for start in range(0, len(docs), E.BATCH):
                chunk = docs[start:start + E.BATCH]; ce, _ = E.batch_ce(model, to(E.stack(chunk), device))
                num += sum(ce); den += sum(int(c['target_mask'].sum()) for c in chunk)
            per_r.append(num / den)
        out[fam] = dict(nll_by_offset=per_r, sensitivity=max(per_r) - min(per_r))
    return out


@torch.no_grad()
def generate_batch(model, prefixes, limits, device):
    """Greedy, full recompute per step (no cache), right-padded batch with per-row lengths."""
    n = len(prefixes); seqs = [list(p) for p in prefixes]; done = [False] * n; new = [[] for _ in range(n)]
    for _ in range(max(limits)):
        active = [i for i in range(n) if not done[i]]
        if not active: break
        t = max(len(seqs[i]) for i in active)
        ids = torch.zeros(len(active), t, dtype=torch.long)
        for j, i in enumerate(active): ids[j, :len(seqs[i])] = torch.tensor(seqs[i])
        lengths = torch.tensor([len(seqs[i]) for i in active])
        logits = model(ids.to(device), lengths.to(device))['logits']
        nxt = logits[torch.arange(len(active)), (lengths - 1).to(device)].argmax(-1).cpu().tolist()
        for j, i in enumerate(active):
            seqs[i].append(nxt[j]); new[i].append(nxt[j])
            if nxt[j] == EOS or len(new[i]) >= limits[i] or len(seqs[i]) >= 256: done[i] = True
    return new


def generation(model, device, tok):
    rng = np.random.default_rng(1927); result = {}
    for fam, n in GENERATE.items():
        src = 'math' if fam == 'math' else fam
        arr, offsets, starts, meta = E.load_family(src)
        if fam == 'math':
            groups = {}
            for i, r in enumerate(meta): groups.setdefault(r['source'], []).append(i)
            chosen = [int(i) for g in sorted(groups) for i in rng.choice(groups[g], n, replace=False)]
        else:
            chosen = [int(i) for i in rng.choice(len(meta), n, replace=False)]
        prefixes = [[int(t) for t in arr[offsets[i]:offsets[i] + starts[i]]] for i in chosen]
        limits = [int(offsets[i + 1] - offsets[i] - starts[i]) + 64 for i in chosen]
        preds = []
        for s in range(0, len(chosen), 32):
            preds += generate_batch(model, prefixes[s:s + 32], limits[s:s + 32], device)
        rows = []
        for i, pred, lim, pre in zip(chosen, preds, limits, prefixes):
            eos = bool(pred and pred[-1] == EOS); text = tok.decode(pred[:-1] if eos else pred, skip_special_tokens=False)
            if 'math' in fam: ok = score_math(text, meta[i]['answer'], eos)['correct']
            else: ok = eos and verify_sudoku(meta[i]['puzzle'], text)['valid']
            rows.append(dict(index=i, source=meta[i].get('source', fam), correct=bool(ok), eos=eos,
                             truncated=not eos, hit_context=len(pre) + len(pred) >= 256, generated=len(pred), limit=lim))
        by = {}
        for r in rows: by.setdefault(r['source'], []).append(r)
        result[fam] = dict(n=len(rows), correct=sum(r['correct'] for r in rows), truncated=sum(r['truncated'] for r in rows),
                           by_module={k: dict(n=len(v), correct=sum(r['correct'] for r in v), truncated=sum(r['truncated'] for r in v))
                                      for k, v in by.items()}, rows=rows)
    return result


def scales(model):
    out = {}
    if hasattr(model, 'head'): out['codebook_rms'] = float(model.head.codebook.detach().square().mean().sqrt())
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'fusion_map'):
        out['fusion_map_rms'] = float(model.decoder.fusion_map.weight.detach().square().mean().sqrt())
    if isinstance(model, V11Tiny):
        out.update(target_var_mean=float(model.target_var.mean()), target_mu_rms=float(model.target_mu.square().mean().sqrt()))
        for k in ('log_sigma1', 'log_sigma2'):
            if hasattr(model, k): out[k.replace('log_', '') + '_sq'] = float(getattr(model, k).detach().clamp_min(math.log(.01)).exp())
        w = [float(model.encoder_concept_crc.scale.detach().abs().mean())]
        out['ec_crc_scale_mean_abs'] = w[0]; out['anchor_scale'] = float(model.core.anchor_scale.detach())
    return out


def main(a):
    run = Path(a.run_dir); target = run / a.output
    if target.exists(): raise SystemExit('Refusing to overwrite ' + str(target))
    ck = torch.load(run / a.checkpoint, map_location='cpu', weights_only=False)
    torch.set_num_threads(a.threads); started = time.time(); arm = ck['arm']
    cfg = TinyConfig(**{k: (tuple(v) if k == 'engram_rows' else v) for k, v in ck['config'].items()})
    model = build(arm, cfg); model.load_state_dict(ck['model']); model.place(a.device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    tok = Tokenizer.from_file(str(DATA / 'tokenizer.json'))
    scored, calib = E.panels()
    if a.quick:
        scored = {f: v[:6] for f, v in scored.items()}; calib = {f: v[:6] for f, v in calib.items()}
        GENERATE.update(math=2, newmath=4, sudoku_easy=1, sudoku_hard=1)
    result = dict(arm=arm, checkpoint=str(run / a.checkpoint), step=ck['step'], tokens=ck['tokens'], scales=scales(model))
    if arm in ('C0', 'X8'):
        result['nll'] = E.nll_and_interventions(model, arm, scored, calib, a.device)
        for fam, v in result['nll'].items(): v['summary'] = summarize(v['documents'], fam)
    elif arm in ('ORD', 'ORDE'):
        result['nll'] = nll_plain(model, scored, a.device)
    else:
        result['nll'] = nll_v11(model, scored, calib, a.device)
    print('nll', {f: round(v['summary']['nll'], 4) for f, v in result['nll'].items()}, round(time.time() - started), flush=True)
    result['alignment'] = alignment(model, scored, a.device); print('alignment', {f: round(v['sensitivity'], 4) for f, v in result['alignment'].items()}, flush=True)
    result['grid_shift'] = E.grid_shift(model, scored, a.device)
    if arm == 'X8':
        result['fvu_v103'] = E.fvu(model, arm, scored, calib, a.device)
        result['exploration'] = E.exploration(model, arm, scored, a.device)
    if arm in ('X8', 'NOAUX', 'K1NB', 'K1'):
        result['pooled_participation'] = pooled_participation(model, scored, a.device)
    if arm in ('NOAUX', 'K1NB', 'K1'):
        if arm == 'NOAUX':
            # NOAUX never updates its buffers (no latent loss); use calibration-panel statistics instead and label it.
            m, v = calibration_stats(model, calib, a.device); model.target_mu.copy_(m); model.target_var.copy_(v)
            result['latent_normalization'] = 'calibration_normalized (NOAUX has no trained target buffers)'
        else:
            result['latent_normalization'] = 'model buffers (EMA over training batches)'
        result['latent'] = latent_v11(model, scored, calib, a.device); print('latent done', round(time.time() - started), flush=True)
    result['generation'] = generation(model, a.device, tok)
    result['seconds'] = time.time() - started
    target.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({f: round(v['summary']['nll'], 4) for f, v in result['nll'].items()}),
          json.dumps({f: (v['correct'], v['truncated']) for f, v in result['generation'].items()}), round(result['seconds']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run-dir', required=True); p.add_argument('--checkpoint', default='final.pt')
    p.add_argument('--device', default='mps'); p.add_argument('--output', default='evaluation.json'); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--quick', action='store_true')
    main(p.parse_args())
