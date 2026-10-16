"""Deep evaluation of one trained tiny arm on fixed dev panels.

  python -B -m v10_3_tiny.evaluate --run-dir v10_3_tiny/runs/X8_s1

Panels are deterministic (seeded index choice, seeded text crops) and identical
across arms, so per-document results can be paired. Writes evaluation.json in
the run directory and refuses to overwrite it.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from v10_10M.data import verify_sudoku
from v10_1.evaluate import score_math
from v10_1.model import chunk_batch, per_document_chunk_ce
from v10_2.model import future_ce
from .model import build, C0
from .config import TinyConfig

DATA = Path(__file__).resolve().parent / 'data'
CONTEXT = 256
PANEL = dict(text=256, math=128, newmath=128, sudoku_easy=64, sudoku_hard=64)     # math: per pretraining module
CALIB = dict(text=64, math=32, newmath=32, sudoku_easy=32, sudoku_hard=32)
GENERATE = dict(math=32, newmath=64, sudoku_easy=8, sudoku_hard=8)                # math: per module


def load_family(family, split='dev'):
    base = DATA / f'{family}_{split}'
    meta = [json.loads(x) for x in (DATA / f'{family}_{split}_meta.jsonl').read_text().splitlines()]
    return np.load(str(base) + '.npy', mmap_mode='r'), np.load(str(base) + '_offsets.npy'), np.load(str(base) + '_answer.npy'), meta


def example(family, store, i, rng):
    arr, offsets, starts, meta = store
    seq = np.array(arr[offsets[i]:offsets[i + 1]], dtype=np.int64); answer = int(starts[i])
    if len(seq) > CONTEXT + 1:
        begin = int(rng.integers(0, len(seq) - CONTEXT)); seq = seq[begin:begin + CONTEXT + 1]; answer = 1
    n = len(seq) - 1; t = min(CONTEXT, ((n - 1) // 8 + 1) * 8)
    x = np.zeros((1, t), np.int64); y = np.zeros_like(x); m = np.zeros_like(x, dtype=bool)
    x[0, :n], y[0, :n] = seq[:-1], seq[1:]; m[0, max(0, answer - 1):n] = True
    return dict(ids=torch.tensor(x), targets=torch.tensor(y), target_mask=torch.tensor(m), lengths=torch.tensor([n]),
                index=i, source=meta[i].get('source', family))


def panels():
    """Scored and calibration panels; disjoint document indices within each family."""
    rng = np.random.default_rng(20261012); scored, calib = {}, {}
    for fam in PANEL:
        src = 'math' if fam == 'math' else fam
        store = load_family(src)
        meta = store[3]
        if fam == 'math':
            idx = []; cal = []
            for module in sorted({r['source'] for r in meta}):
                pool = [i for i, r in enumerate(meta) if r['source'] == module]; pick = rng.permutation(pool)
                idx += list(pick[:PANEL[fam]]); cal += list(pick[PANEL[fam]:PANEL[fam] + CALIB[fam]])
        else:
            pick = rng.permutation(len(meta)); idx = list(pick[:PANEL[fam]]); cal = list(pick[PANEL[fam]:PANEL[fam] + CALIB[fam]])
        scored[fam] = [example(fam, store, int(i), rng) for i in idx]
        calib[fam] = [example(fam, store, int(i), rng) for i in cal]
    return scored, calib


def to(b, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def doc_ce(model, b, **flags):
    out = model(b['ids'], b['lengths'], **flags)
    ce = F.cross_entropy(out['logits'][0], b['targets'][0], reduction='none')
    return float((ce * b['target_mask'][0]).sum()), out


BATCH = 16   # documents per forward; right padding with lengths, as in training


def stack(docs):
    t = max(d['ids'].shape[1] for d in docs)
    out = {k: torch.zeros(len(docs), t, dtype=docs[0][k].dtype) for k in ('ids', 'targets', 'target_mask')}
    for i, d in enumerate(docs):
        for k in out: out[k][i, :d[k].shape[1]] = d[k][0]
    out['lengths'] = torch.cat([d['lengths'] for d in docs])
    return out


def batch_ce(model, b, **flags):
    out = model(b['ids'], b['lengths'], **flags)
    ce = F.cross_entropy(out['logits'].flatten(0, 1), b['targets'].flatten(), reduction='none').reshape(b['targets'].shape)
    return (ce * b['target_mask']).sum(-1).tolist(), out


class MeanSubstitution:
    """Replace the decoder's concept inputs (forecast F and state U) by calibration means."""
    def __init__(self, model, f, u):
        self.d, self.f, self.u = model.decoder, f, u; self.original = model.decoder.forward
    def __enter__(self):
        def patched(early, final, forecast, mask, rope, u, lengths, *a, **k):
            return self.original(early, final, self.f.expand_as(forecast), mask, rope, self.u.expand_as(u), lengths, *a, **k)
        self.d.forward = patched
    def __exit__(self, *exc):
        self.d.forward = self.original


def interventions(arm):
    if arm == 'A':
        return dict(both_off_renormalized=dict(ablate=True), both_off_value_zero=dict(ablate_pq=True, zero_continuous=True),
                    pq_off=dict(ablate_pq=True), continuous_off_renormalized=dict(ablate_continuous=True),
                    continuous_value_zero=dict(zero_continuous=True), one_visit=dict(visits=1))
    return dict(all_concept_off=dict(ablate=True), aligned_off=dict(ablate_aligned=True), reader_off=dict(ablate_reader=True),
                reader_pq_off=dict(ablate_cross_pq=True), reader_state_off=dict(zero_continuous=True), one_visit=dict(visits=1))


@torch.no_grad()
def nll_and_interventions(model, arm, scored, calib, device):
    result = {}
    for fam, docs in scored.items():
        means = None
        if arm != 'C0':
            fs, us = [], []
            for b in calib[fam]:
                o = model(*[to(b, device)[k] for k in ('ids', 'lengths')]); v = o['valid_groups'][0]
                assert o['correction'] is None, 'calibration means must come from the empty-bank model'
                fs.append(o['forecast'][0][v]); us.append(o['u'][0][v])
            means = (torch.cat(fs).mean(0), torch.cat(us).mean(0))
        rows = []
        docs = [d for d in docs if int(d['target_mask'].sum())]
        for start in range(0, len(docs), BATCH):
            chunk = docs[start:start + BATCH]; b = to(stack(chunk), device)
            base, out = batch_ce(model, b)
            part = [dict(index=d['index'], source=d['source'], n=int(d['target_mask'].sum()), ce=c) for d, c in zip(chunk, base)]
            if arm != 'C0':
                for name, flags in interventions(arm).items():
                    for r, c in zip(part, batch_ce(model, b, codes=out['codes'], **flags)[0]): r[name] = c
                with MeanSubstitution(model, *means):
                    for r, c in zip(part, batch_ce(model, b, codes=out['codes'])[0]): r['mean_substitution'] = c
            rows += part
        tot = sum(r['n'] for r in rows)
        summary = dict(documents=len(rows), tokens=tot, nll=sum(r['ce'] for r in rows) / tot)
        if arm != 'C0':
            summary['delta'] = {k: (sum(r[k] for r in rows) - sum(r['ce'] for r in rows)) / tot
                                for k in rows[0] if k not in ('index', 'source', 'n', 'ce')}
        if fam == 'math':
            summary['by_module'] = {}
            for module in sorted({r['source'] for r in rows}):
                sel = [r for r in rows if r['source'] == module]; t = sum(r['n'] for r in sel)
                summary['by_module'][module] = dict(documents=len(sel), nll=sum(r['ce'] for r in sel) / t)
        result[fam] = dict(summary=summary, documents=rows)
    return result


@torch.no_grad()
def grid_shift(model, scored, device, shift=4):
    """Sudoku has a constant layout (data_audit.json): re-score each board with
    `shift` extra header-newline tokens so every cell moves half a group. All
    arms see the same out-of-distribution edit; compare their degradation."""
    out = {}
    for fam in ('sudoku_easy', 'sudoku_hard'):
        base_sum = shifted_sum = tokens = 0.
        for b in scored[fam]:
            n = int(b['lengths'][0]); seq = torch.cat((b['ids'][0, :n], b['targets'][0, n - 1:n]))
            answer = int(torch.nonzero(b['target_mask'][0])[0]) + 1
            assert int(seq[3]) == int(seq[answer - 1]), 'header newline expected at position 3 and before the answer'
            new = torch.cat((seq[:4], seq[3:4].repeat(shift), seq[4:])); m = len(new) - 1
            t = min(CONTEXT, ((m - 1) // 8 + 1) * 8)
            x = torch.zeros(1, t, dtype=torch.long); y = torch.zeros_like(x); mask = torch.zeros_like(x, dtype=torch.bool)
            x[0, :m], y[0, :m] = new[:-1], new[1:]; mask[0, answer + shift - 1:m] = True
            shifted = dict(ids=x, targets=y, target_mask=mask, lengths=torch.tensor([m]))
            k = int(b['target_mask'].sum()); assert k == int(mask.sum())
            base_sum += doc_ce(model, to(b, device))[0]; shifted_sum += doc_ce(model, to(shifted, device))[0]; tokens += k
        out[fam] = dict(boards=len(scored[fam]), shift=shift, nll=base_sum / tokens, nll_shifted=shifted_sum / tokens,
                        degradation=(shifted_sum - base_sum) / tokens)
    return out


def ridge(x, y, lam):
    mx, my = x.mean(0), y.mean(0); xc = x - mx
    w = np.linalg.solve(xc.T @ xc + lam * len(x) * np.eye(x.shape[1]), xc.T @ (y - my))
    return lambda z: (z - mx) @ w + my


def fit_ridge(cx, cy):
    half = len(cx) // 2
    lam = min((0.1, 1., 10., 100.), key=lambda l: float(((ridge(cx[:half], cy[:half], l)(cx[half:]) - cy[half:]) ** 2).mean()))
    return ridge(cx, cy, lam)


@torch.no_grad()
def fvu(model, arm, scored, calib, device):
    def collect(docs):
        acc = dict(f2=[], u2=[], p3=[], y=[], dec=[])
        for b in docs:
            b = to(b, device); o = model(b['ids'], b['lengths']); pairs = o['valid_groups'][0, 1:]
            if int(pairs.sum()) == 0: continue
            f2 = o['forecasts'][-1][0, :-1][pairs]
            dec = model.decoder.fusion_map(f2) if hasattr(model.decoder, 'fusion_map') else model.decoder.fusion * f2
            # The E3 rows the aligned forecast is added to: slot s of group g lands on rows 8g+7+4s .. +3.
            rows_e3 = []
            h = model.encode(b['ids'], b['lengths'])[1][0]
            for g in torch.nonzero(pairs).flatten().tolist():
                for s in range(2):
                    r = min(8 * g + 7 + 4 * s, h.shape[0] - 1); rows_e3.append(h[r])
            acc.setdefault('e3', []).append(torch.stack(rows_e3).reshape(-1, 2 * h.shape[-1]).cpu().double().numpy())
            for k, v in (('f2', f2), ('u2', o['states'][-1][0, :-1][pairs]), ('p3', o['pooled'][0, :-1][pairs]),
                         ('y', o['pooled'][0, 1:][pairs]), ('dec', dec)):
                acc[k].append(v.flatten(-2).cpu().double().numpy())
        return {k: np.concatenate(v) for k, v in acc.items()}
    result = {}
    for fam in scored:
        c, t = collect(calib[fam]), collect(scored[fam]); var = float(t['y'].var(0).mean())
        score = lambda p: float(((p - t['y']) ** 2).mean() / var)
        a = ((c['f2'] - c['f2'].mean(0)) * (c['y'] - c['y'].mean(0))).sum(0) / ((c['f2'] - c['f2'].mean(0)) ** 2).sum(0).clip(1e-12)
        b0 = c['y'].mean(0) - a * c['f2'].mean(0)
        result[fam] = dict(pairs=len(t['y']), target_variance=var,
                           forecast_variance_ratio=float(t['f2'].var(0).mean() / var),
                           decoder_forecast_input_variance_ratio=float(t['dec'].var(0).mean() / var),
                           decoder_forecast_input_to_e3_variance_ratio=float(t['dec'].var(0).mean() / t['e3'].var(0).mean()),
                           fvu=dict(calibration_mean=score(np.broadcast_to(c['y'].mean(0), t['y'].shape)),
                                    forecast=score(t['f2']), forecast_affine=score(a * t['f2'] + b0),
                                    ridge_forecast=score(fit_ridge(c['f2'], c['y'])(t['f2'])),
                                    ridge_state=score(fit_ridge(c['u2'], c['y'])(t['u2'])),
                                    ridge_input=score(fit_ridge(c['p3'], c['y'])(t['p3']))))
    return result


@torch.no_grad()
def exploration(model, arm, scored, device, per_doc=4, limit=dict(text=64, math=96, newmath=32, sudoku_easy=32, sudoku_hard=32)):
    rng = np.random.default_rng(31); result = {}
    for fam, docs in scored.items():
        recs = []
        # Interleave sources so a slice is not dominated by one math module.
        by = {}
        for b in docs: by.setdefault(b['source'], []).append(b)
        mixed = [d for group in zip(*by.values()) for d in group] if len(by) > 1 else docs
        for b in mixed[:limit[fam]]:
            b = to(b, device); out = model(b['ids'], b['lengths'])
            x, l, y, m = b['ids'], b['lengths'], b['targets'], b['target_mask']
            options = []
            for at in range(out['valid_groups'].shape[1]):
                try:
                    if chunk_batch(x, l, y, m, out['valid_groups'], boundaries=torch.tensor([at], device=x.device)) is not None: options.append(at)
                except ValueError: pass
            for at in rng.choice(options, min(per_doc, len(options)), replace=False) if options else []:
                at_t = torch.tensor([int(at)], device=x.device)
                chunk = chunk_batch(x, l, y, m, out['valid_groups'], boundaries=at_t)
                sub = {k: out[k] for k in ('pooled', 'sources', 'codes', 'valid_groups')}
                pred = model.candidate_forecasts(sub, l, at_t); k = model.c.codes
                expanded = {kk: chunk[kk].repeat_interleave(k, 0) for kk in ('previous', 'labels', 'mask')}
                n = float(chunk['mask'].sum())
                if arm == 'A':
                    target = out['pooled'][0, min(int(at) + 1, out['pooled'].shape[1] - 1)]
                    has_next = float(int(at) + 1 < int(out['valid_groups'].sum()))
                    utility = per_document_chunk_ce(model, pred.flatten(0, 1), expanded) + .1 * has_next * (pred[0] - target).square().mean((-1, -2))
                    ll = -utility * n
                else:
                    _, ce = future_ce(model, pred.flatten(0, 1), expanded, details=True)
                    ll = -(ce * expanded['mask']).sum(-1)
                log_prior = F.log_softmax(out['selector'][0, int(at)], -1)
                recs.append(dict(ll=(ll / n).cpu().double().numpy(), n=n, log_prior=log_prior.cpu().double().numpy(),
                                 committed=int(out['codes'][0, int(at)])))
        if not recs: continue
        u = -np.stack([r['ll'] for r in recs]); s = np.sort(u, 1); win = u.argmin(1)
        lp = np.stack([r['log_prior'] for r in recs]); com = np.array([r['committed'] for r in recs]); n = np.array([r['n'] for r in recs])
        centred = u - u.mean(1, keepdims=True); effect = centred.mean(0)
        counts = np.bincount(win, minlength=u.shape[1]); p = counts / counts.sum()
        joint = lp + (-u) * n[:, None]
        mix = (np.logaddexp.reduce(joint, 1) / n).mean()
        uniform = (np.logaddexp.reduce(-u * n[:, None] - math.log(u.shape[1]), 1) / n).mean()
        result[fam] = dict(boundaries=len(recs), spread_median=float(np.median(s[:, -1] - s[:, 0])),
                           gap_median=float(np.median(s[:, 1] - s[:, 0])),
                           regret_mean=float((u[np.arange(len(u)), com] - s[:, 0]).mean()),
                           code_main_effect_share=float((effect ** 2).sum() * len(u) / max((centred ** 2).sum(), 1e-12)),
                           winner_counts=counts.tolist(), committed_counts=np.bincount(com, minlength=u.shape[1]).tolist(),
                           selector_accuracy=float((com == win).mean()),
                           selector_winner_ce=float(-lp[np.arange(len(u)), win].mean()),
                           winner_entropy=float(-(p[p > 0] * np.log(p[p > 0])).sum()),
                           prior_entropy=float(-(np.exp(lp) * lp).sum(1).mean()),
                           loglik_per_token=dict(mixture_learned_prior=float(mix), mixture_uniform=float(uniform),
                                                 committed=float(-u[np.arange(len(u)), com].mean()), best_in_hindsight=float(-s[:, 0].mean())))
    return result


@torch.no_grad()
def greedy(model, prefix, max_new, device, concept=True):
    ids = torch.tensor([prefix], device=device); committed = None
    for _ in range(max_new):
        out = model(ids, codes=committed) if concept else model(ids)
        if concept: committed = out['codes']
        nxt = out['logits'][:, -1].argmax(-1); ids = torch.cat((ids, nxt[:, None]), -1)
        if int(nxt) == 2: break
    return ids[0, len(prefix):].cpu().tolist()


def generation(model, arm, device, tok):
    rng = np.random.default_rng(1927); result = {}
    for fam, n in GENERATE.items():
        src = 'math' if fam == 'math' else fam
        arr, offsets, starts, meta = load_family(src)
        if fam == 'math':
            groups = {}
            for i, r in enumerate(meta): groups.setdefault(r['source'], []).append(i)
            chosen = [int(i) for g in sorted(groups) for i in rng.choice(groups[g], n, replace=False)]
        else:
            chosen = [int(i) for i in rng.choice(len(meta), n, replace=False)]
        rows = []
        for i in chosen:
            prefix = [int(t) for t in arr[offsets[i]:offsets[i] + starts[i]]]
            pred = greedy(model, prefix, 24 if 'math' in fam else 96, device, concept=not isinstance(model, C0))
            eos = bool(pred and pred[-1] == 2); text = tok.decode(pred[:-1] if eos else pred, skip_special_tokens=False)
            if 'math' in fam: ok = score_math(text, meta[i]['answer'], eos)['correct']
            else:
                check = verify_sudoku(meta[i]['puzzle'], text); ok = eos and check['valid']
            rows.append(dict(index=i, source=meta[i].get('source', fam), correct=bool(ok)))
        by = {}
        for r in rows: by.setdefault(r['source'], []).append(r['correct'])
        result[fam] = dict(n=len(rows), correct=sum(r['correct'] for r in rows),
                           by_module={k: dict(n=len(v), correct=int(sum(v))) for k, v in by.items()}, rows=rows)
    return result


def scales(model):
    out = {}
    if hasattr(model, 'head'): out['codebook_rms'] = float(model.head.codebook.detach().square().mean().sqrt())
    if getattr(model.decoder, 'fusion', None) is not None: out['fusion_rms'] = float(model.decoder.fusion.detach().square().mean().sqrt())
    if hasattr(model.decoder, 'fusion_map'): out['fusion_map_rms'] = float(model.decoder.fusion_map.weight.detach().square().mean().sqrt())
    return out


def main(a):
    run = Path(a.run_dir); target = run / a.output
    if target.exists(): raise SystemExit('Refusing to overwrite ' + str(target))
    ck = torch.load(run / a.checkpoint, map_location='cpu', weights_only=False)
    torch.set_num_threads(a.threads); started = time.time()
    model = build(ck['arm'], TinyConfig(**{k: (tuple(v) if k == 'engram_rows' else v) for k, v in ck['config'].items()}))
    model.load_state_dict(ck['model']); model.place(a.device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    tok = Tokenizer.from_file(str(DATA / 'tokenizer.json'))
    scored, calib = panels(); arm = ck['arm']
    result = dict(arm=arm, checkpoint=str(run / a.checkpoint), step=ck['step'], tokens=ck['tokens'], scales=scales(model))
    result['nll'] = nll_and_interventions(model, arm, scored, calib, a.device); print('nll done', time.time() - started, flush=True)
    result['grid_shift'] = grid_shift(model, scored, a.device); print('grid shift', result['grid_shift'], flush=True)
    if arm != 'C0':
        result['fvu'] = fvu(model, arm, scored, calib, a.device); print('fvu done', time.time() - started, flush=True)
        result['exploration'] = exploration(model, arm, scored, a.device); print('exploration done', time.time() - started, flush=True)
    result['generation'] = generation(model, arm, a.device, tok)
    result['seconds'] = time.time() - started
    target.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({f: v['summary']['nll'] for f, v in result['nll'].items()}), json.dumps({f: v['correct'] for f, v in result['generation'].items()}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run-dir', required=True); p.add_argument('--checkpoint', default='final.pt')
    p.add_argument('--device', default='mps'); p.add_argument('--output', default='evaluation.json'); p.add_argument('--threads', type=int, default=2)
    main(p.parse_args())
