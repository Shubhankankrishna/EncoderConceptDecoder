"""Correctness tests for the tiny V10.3 arms. Run: python -B -m v10_3_tiny.tests

Each test prints PASS/FAIL with its measured quantity; the process exits
non-zero if any test fails. Results are also written to tests_result.json.
"""
import json, math, sys, time
from pathlib import Path
import torch
from torch.nn import functional as F
from v10_10M.data import Corpus
from v10_2.model import history_geometry
from .config import TinyConfig
from .model import build, counts, C0, V103Tiny
from .losses import compute, mixture
from .train import optimizers
from v10_1.model import chunk_batch

DATA = Path(__file__).resolve().parent / 'data'
OUT = Path(__file__).resolve().parent / 'tests_result.json'
ARMS = ('C0', 'A', 'X8', 'X4')
results = []


def record(name, ok, **detail):
    results.append(dict(test=name, passed=bool(ok), **detail))
    print(('PASS ' if ok else 'FAIL ') + name, json.dumps(detail), flush=True)


def fresh(arm, seed=0):
    torch.manual_seed(seed); m = build(arm); m.place('cpu'); return m


class Float64Norm:
    """Test-only: v10_10M RMSNorm computes its scale in float32 even for float64
    inputs, which limits finite differences to float32 resolution. Inside this
    context the norm runs in the input dtype; production code is unchanged."""
    def __enter__(self):
        import v10_10M.model as base
        self.base, self.original = base, base.RMSNorm.forward
        base.RMSNorm.forward = lambda s, x: x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * s.weight
    def __exit__(self, *exc):
        self.base.RMSNorm.forward = self.original


def batch(family, n=2, context=256, seed=3):
    return Corpus('dev', seed=seed, data_dir=DATA).batch(n, context, family=family)


def test_counts():
    for arm in ARMS:
        c = counts(fresh(arm)); record(f'parameter budget {arm}', c['dense'] <= 600_000, **c)


def test_roundtrip():
    for arm in ARMS:
        m = fresh(arm, 1); m2 = fresh(arm, 2)
        m2.load_state_dict(m.state_dict(), strict=True)
        same = all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
        record(f'state_dict strict round trip {arm}', same, tensors=len(m.state_dict()))


def test_optimizer_groups():
    for arm in ARMS:
        m = fresh(arm); dense_opt, _, _ = optimizers(arm, m, 1e-3)
        names = {id(p): n for n, p in m.named_parameters()}
        decayed = {names[id(p)] for g in dense_opt.param_groups if g['weight_decay'] > 0 for p in g['params']}
        if arm == 'C0':
            record('optimizer groups C0', 'head.codebook' not in names.values(), decayed=len(decayed)); continue
        expect = arm == 'A'
        record(f'codebook decayed is {expect} for {arm}', ('head.codebook' in decayed) == expect,
               decayed_matrices=len(decayed))


def test_causality():
    torch.manual_seed(5)
    for arm in ARMS:
        m = fresh(arm).eval(); worst = 0.
        for fam in ('text', 'math', 'sudoku_easy'):
            b = batch(fam, 1); x, l = b['ids'], b['lengths']; t = int(l[0])
            with torch.no_grad():
                base = m(x, l)['logits']
                for i in sorted({1, 6, 7, 8, 9, 15, 16, 17, t // 2, t - 2}):
                    if i >= t - 1 or i < 0: continue
                    y = x.clone(); y[0, i + 1:t] = torch.randint(3, 1024, (t - i - 1,))
                    other = m(y, l)['logits']
                    worst = max(worst, float((base[0, :i + 1] - other[0, :i + 1]).abs().max()))
        record(f'future perturbation leaves earlier logits unchanged {arm}', worst < 1e-5, max_abs_diff=worst)


def test_release_mask():
    bad = 0; rows = 0
    for length in (1, 7, 8, 9, 15, 16, 17, 33):
        t = 40; g = t // 8
        mask, _, _ = history_geometry(t, g, torch.tensor([length]), 14, 'cpu')
        m = mask[0, 0]
        for i in range(t):
            real = m[i, :-1]; sentinel = m[i, -1]
            ends = torch.arange(g).repeat_interleave(2) * 8 + 7
            legal = (ends <= i) & (ends < length) & (i < length)
            rows += 1
            if not torch.equal(real, legal) or bool(sentinel) == bool(legal.any()): bad += 1
    record('concept-history release mask and sentinel', bad == 0, rows=rows, mismatches=bad)


def test_candidate_consistency():
    for arm in ('X8', 'X4', 'A'):
        m = fresh(arm).eval(); b = batch('text', 2, 96)
        with torch.no_grad():
            out = m(b['ids'], b['lengths'])
            chunk = chunk_batch(b['ids'], b['lengths'], b['targets'], b['target_mask'], out['valid_groups'], torch.Generator().manual_seed(0))
            rows, at = chunk['rows'], chunk['boundaries']
            sub = {k: out[k][rows] for k in ('pooled', 'sources', 'codes', 'valid_groups')}
            pred = m.candidate_forecasts(sub, b['lengths'][rows], at)
            committed = out['codes'][rows, at]
            diff = float((pred[torch.arange(len(rows)), committed] - out['forecast'][rows, at]).abs().max())
        record(f'candidate branch with committed code equals main forecast {arm}', diff < 1e-5, max_abs_diff=diff)


def test_mixture_gradient():
    # Analytic check of the normaliser: dL/dlogit = (pi - r)/n, dL/dll = -r/n, per document mean.
    torch.manual_seed(0); n, k = 3, 8
    logits = torch.randn(n, k, dtype=torch.float64, requires_grad=True)
    ll = (-torch.rand(n, k, dtype=torch.float64) * 20).requires_grad_(True)
    n_doc = torch.tensor([8., 5., 3.], dtype=torch.float64)
    loss = -(torch.logsumexp(F.log_softmax(logits, -1) + ll, -1) / n_doc).mean()
    loss.backward()
    with torch.no_grad():
        pi = logits.softmax(-1); r = (F.log_softmax(logits, -1) + ll).softmax(-1)
        e1 = float((logits.grad - (pi - r) / n_doc[:, None] / n).abs().max())
        e2 = float((ll.grad + r / n_doc[:, None] / n).abs().max())
    record('mixture loss analytic gradient', max(e1, e2) < 1e-12, logit_error=e1, ll_error=e2)
    # End-to-end directional finite difference through selector, codes, core, PQ and readout.
    norm = Float64Norm(); norm.__enter__()
    m = fresh('X8', 7).double().train(); b = batch('text', 2, 96)
    torch.manual_seed(1)
    params = [p for p in m.parameters() if p.requires_grad and p is not m.engram.table.weight]
    direction = [torch.randn_like(p) for p in params]
    def value():
        out = m(b['ids'], b['lengths'])
        chunk = chunk_batch(b['ids'], b['lengths'], b['targets'], b['target_mask'], out['valid_groups'], boundaries=None,
                            generator=torch.Generator().manual_seed(4))
        return mixture(m, out, chunk, b['lengths'])[0]
    m.zero_grad(); value().backward()
    analytic = sum(float((p.grad * d).sum()) for p, d in zip(params, direction) if p.grad is not None)
    eps = 1e-5
    with torch.no_grad():
        for p, d in zip(params, direction): p.add_(eps * d)
        plus = float(value())
        for p, d in zip(params, direction): p.sub_(2 * eps * d)
        minus = float(value())
        for p, d in zip(params, direction): p.add_(eps * d)
    numeric = (plus - minus) / (2 * eps)
    rel = abs(numeric - analytic) / max(abs(numeric), 1e-8)
    record('mixture loss directional finite difference (float64)', rel < 1e-6, analytic=analytic, numeric=numeric, relative_error=rel)
    # Every code receives gradient on a candidate update (not only the winner).
    m.zero_grad(); value().backward()
    g = m.core.codes.weight.grad.abs().sum(-1)
    norm.__exit__()
    record('all K code embeddings receive mixture gradient', bool((g > 0).all()), per_code_grad=g.tolist())


def test_rbf_derivative():
    norm = Float64Norm(); norm.__enter__()
    m = fresh('X8', 9).double().eval(); b = batch('text', 1, 64)
    for p in m.parameters(): p.requires_grad_(False)
    with torch.no_grad(): out = m(b['ids'], b['lengths'])
    q = out['queries']; torch.manual_seed(2)
    base = (torch.randn_like(q) * .05).requires_grad_(True)
    def loss(c):
        o = m(b['ids'], b['lengths'], codes=out['codes'], correction_override=c)
        return F.cross_entropy(o['logits'][0, :int(b['lengths'][0])], b['targets'][0, :int(b['lengths'][0])])
    l = loss(base); l.backward(); d = torch.randn_like(q); analytic = float((base.grad * d).sum())
    eps = 1e-5
    with torch.no_grad(): numeric = (float(loss(base + eps * d)) - float(loss(base - eps * d))) / (2 * eps)
    norm.__exit__()
    rel = abs(numeric - analytic) / max(abs(numeric), 1e-8)
    record('RBF correction derivative through aligned, reader-key and reader-value routes (float64)', rel < 1e-6,
           analytic=analytic, numeric=numeric, relative_error=rel)


def test_mps_parity():
    if not torch.backends.mps.is_available():
        record('cpu/mps parity', True, skipped='mps unavailable'); return
    for arm in ('C0', 'X8'):
        cpu = fresh(arm, 11).train(); mps = fresh(arm, 11); mps.place('mps').train()
        b = batch('text', 2, 96)
        def run(model, device):
            bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(bb['ids'], bb['lengths'])
            loss = compute(arm, model, out, bb, arm != 'C0', torch.Generator().manual_seed(0))['total']
            model.zero_grad(); loss.backward()
            g = torch.cat([p.grad.flatten().cpu() for n, p in model.named_parameters() if p.grad is not None and 'engram.table' not in n])
            return float(loss.detach()), g
        lc, gc = run(cpu, 'cpu'); lm, gm = run(mps, 'mps')
        rel = float((gc - gm).norm() / gc.norm())
        record(f'cpu/mps forward and gradient parity {arm}', abs(lc - lm) < 1e-3 and rel < 1e-3,
               loss_cpu=lc, loss_mps=lm, gradient_relative_difference=rel)


def main():
    started = time.time()
    for t in (test_counts, test_roundtrip, test_optimizer_groups, test_release_mask, test_causality,
              test_candidate_consistency, test_mixture_gradient, test_rbf_derivative, test_mps_parity):
        try: t()
        except Exception as e: record(t.__name__, False, error=repr(e))
    OUT.write_text(json.dumps(dict(seconds=time.time() - started, torch=torch.__version__, results=results), indent=2) + '\n')
    failed = [r['test'] for r in results if not r['passed']]
    print(f'{len(results) - len(failed)}/{len(results)} passed', flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
