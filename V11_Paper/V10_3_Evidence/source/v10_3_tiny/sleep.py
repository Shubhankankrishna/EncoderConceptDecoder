"""Sleep-layer tests for tiny V10.3 on the held-out algebra module (newmath).

  python -B -m v10_3_tiny.sleep --run-dir v10_3_tiny/runs/X8_s1 --out-dir v10_3_tiny/runs/sleep_X8_s1

Every condition starts from the same frozen pretrained checkpoint; the ones
that train see the identical newmath stream and identical adapter init.

  base          no change
  plain         rank-4 LoRA on every attention (qkv, out) and FFN (up, down)
                matrix of E2, the concept core and the decoder
  gated         the same adapters; each sub-layer's adapter output is multiplied
                per row by the Local-Support-Learning gate
                    open(x) = log Phi_pos(x) > log Phi_neg(x)
                where x is the frozen model's own input to that sub-layer. A
                separate adapter-off pass computes every gate, so an upstream
                adapter cannot move a downstream gate. Phi_neg: diagonal GMM on
                inputs from all pretraining families (text, the three arithmetic
                modules, both Sudoku families; equal row cap per family).
                Phi_pos: diagonal GMM on newmath train inputs. Separate GMMs per
                visit for E2 and the core. Token-rate sites (E2, decoder) use the
                paper's per-document EMA (alpha 0.1); group-rate core sites use
                raw decisions.
  rbf           frozen model plus an RBF correction bank written by
                v10_2.memory.learn_episode on newmath train documents. (eta, sigma)
                is chosen on held-out TRAIN documents: lowest newmath loss with at
                most 0.01 nats/token harm on text+math (choose_bank). On the
                untrained smoke checkpoint the V10 default eta 0.1 wrote values of
                RMS ~1e-5; with sigma 1 the kernel is nearly global (query norm ~1).
  consolidated  (--consolidate) fresh gated adapters distilled from the
                rbf-corrected model on the same documents; the bank is then reset

Scored on evaluate.panels() (dev split): per-document NLL, math and newmath exact
match, gate open rates per site and family. Writes sleep.json after every
condition and refuses to overwrite an existing output directory.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')   # sklearn GMM + torch OpenMP crashed on macOS with k-means init
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.mixture import GaussianMixture
from tokenizers import Tokenizer
from v10_10M.data import Corpus
from v10_1.evaluate import score_math
from v10_2.memory import RBFMemory, learn_episode, model_era
from v10_10M.train import atomic_json
from .config import TinyConfig
from .model import build
from .losses import compute
from .compare import paired
from . import evaluate as E

DATA = E.DATA
RANK, EMA, COMPONENTS, CAP_POS, CAP_NEG = 4, .1, 4, 4000, 1500
NEG_DOCS = dict(text=24, math=96, sudoku_easy=12, sudoku_hard=12)
POS_DOCS = 128
GEN = dict(math=32, newmath=64)       # math: per module


class LoRA(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base, self.gate, self.enabled = base, None, True
        dev = base.weight.device
        self.lora_a = nn.Parameter(torch.randn(rank, base.in_features, device=dev) / math.sqrt(base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=dev))

    def forward(self, x):
        y = self.base(x)
        if not self.enabled: return y
        delta = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return y + (delta if self.gate is None else delta * self.gate)


def sites(model):
    found = []
    for prefix, blocks, visits in (('e2', model.e2, 2), ('core', model.core.layers, 2), ('dec', model.decoder.layers, 1)):
        for i, blk in enumerate(blocks):
            found += [(f'{prefix}.{i}.attn', blk.attn, ('qkv', 'out'), visits), (f'{prefix}.{i}.ffn', blk.ffn, ('up', 'down'), visits)]
    return found


def ema(g):
    t = g.shape[-1]; i = torch.arange(t, device=g.device)
    lag = (i[:, None] - i[None, :]).clamp_min(0).to(g.dtype)
    w = EMA * (1 - EMA) ** lag * (i[:, None] >= i[None, :]).to(g.dtype)
    return ((g @ w.T) > .5).to(g.dtype)


def density(gmm, device):
    logw = torch.tensor(np.log(gmm.weights_), dtype=torch.float32, device=device)
    mu = torch.tensor(gmm.means_, dtype=torch.float32, device=device)
    var = torch.tensor(gmm.covariances_, dtype=torch.float32, device=device)
    const = logw - .5 * torch.log(2 * math.pi * var).sum(-1)
    def logp(x):
        z = x.float().unsqueeze(-2) - mu
        return torch.logsumexp(const - .5 * (z.square() / var).sum(-1), -1)
    return logp


class Controller:
    """Per-forward bookkeeping of sub-layer calls: collect inputs, record gates, apply gates."""
    def __init__(self, model):
        self.sites, self.mode, self.calls, self.store, self.collected, self.gmm = sites(model), 'idle', {}, {}, {}, {}
        for name, module, children, visits in self.sites:
            module.register_forward_pre_hook(self.hook(name, children, visits))

    def begin(self, mode):
        self.mode, self.calls = mode, {}

    def hook(self, name, children, visits):
        def fn(module, inputs):
            gate = None
            if self.mode != 'idle':
                n = self.calls.get(name, 0); self.calls[name] = n + 1
                if n >= visits: raise RuntimeError(f'{name} called {n + 1} times in one forward')
                key = f'{name}.v{n + 1}' if visits == 2 else name
                if self.mode == 'collect': self.collected[key] = inputs[0].detach()
                elif self.mode == 'record': self.store[key] = self.decide(key, inputs[0])
                elif self.mode == 'apply': gate = self.store[key]
            for c in children:
                child = getattr(module, c)
                if isinstance(child, LoRA): child.gate = gate
        return fn

    def decide(self, key, x):
        pos, neg = self.gmm[key]
        g = (pos(x) > neg(x)).to(x.dtype)
        if not key.startswith('core'): g = ema(g)
        return g.unsqueeze(-1)


def set_enabled(model, flag):
    for m in model.modules():
        if isinstance(m, LoRA): m.enabled = flag


def forward(model, ctrl, condition, ids, lengths=None, **kw):
    if condition in ('gated', 'consolidated'):
        with torch.no_grad():
            set_enabled(model, False); ctrl.begin('record'); model(ids, lengths, **kw)
        set_enabled(model, True); ctrl.begin('apply')
    else:
        ctrl.begin('idle')
    out = model(ids, lengths, **kw); ctrl.begin('idle')
    return out


def load(run, device):
    ck = torch.load(Path(run) / 'final.pt', map_location='cpu', weights_only=False)
    model = build(ck['arm'], TinyConfig(**{k: (tuple(v) if k == 'engram_rows' else v) for k, v in ck['config'].items()}))
    model.load_state_dict(ck['model']); model.place(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, ck


def newmath_corpus(split, seed):
    c = Corpus(split, seed=seed, data_dir=DATA)
    base = DATA / f'newmath_{split}'
    c.arrays['newmath'] = (np.load(str(base) + '.npy', mmap_mode='r'), np.load(str(base) + '_offsets.npy'), np.load(str(base) + '_answer.npy'))
    return c


def cap(rows, n, rng):
    return rows[rng.choice(len(rows), n, replace=False)] if len(rows) > n else rows


@torch.no_grad()
def collect(model, ctrl, docs, device):
    rows = {}
    for fam, batch in docs.items():
        for b in batch:
            b = E.to(b, device); ctrl.begin('collect'); out = model(b['ids'], b['lengths']); ctrl.begin('idle')
            length, groups = int(b['lengths'][0]), int(out['valid_groups'][0].sum())
            for key, x in ctrl.collected.items():
                x = x[0, :2 * groups] if key.startswith('core') else x[0, :length]
                if len(x): rows.setdefault(key, {}).setdefault(fam, []).append(x.cpu().double().numpy())
    return rows


def fit_gates(model, ctrl, device, seed):
    rng = np.random.default_rng(seed); corpus = newmath_corpus('train', seed)
    neg = collect(model, ctrl, {f: [corpus.batch(1, E.CONTEXT, family=f) for _ in range(n)] for f, n in NEG_DOCS.items()}, device)
    pos = collect(model, ctrl, {'newmath': [corpus.batch(1, E.CONTEXT, family='newmath') for _ in range(POS_DOCS)]}, device)
    fitted, report = {}, {}
    for key in pos:
        neg_rows = np.concatenate([cap(np.concatenate(v), CAP_NEG, rng) for v in neg[key].values()])
        pos_rows = cap(np.concatenate(pos[key]['newmath']), CAP_POS, rng)
        fit = lambda r: GaussianMixture(COMPONENTS, covariance_type='diag', reg_covar=1e-4, init_params='random_from_data',
                                        n_init=2, random_state=0).fit(r)
        gp, gn = fit(pos_rows), fit(neg_rows); fitted[key] = (gp, gn)
        tp, tn = density(gp, device), density(gn, device)
        both = np.concatenate([pos_rows, neg_rows]); xt = torch.tensor(both, dtype=torch.float32, device=device)
        ours = (tp(xt) > tn(xt)).cpu().numpy(); theirs = gp.score_samples(both) > gn.score_samples(both)
        report[key] = dict(pos_rows=len(pos_rows), neg_rows=len(neg_rows), neg_families={f: int(sum(len(a) for a in v)) for f, v in neg[key].items()},
                           fit_open_pos=float(ours[:len(pos_rows)].mean()), fit_open_neg=float(ours[len(pos_rows):].mean()),
                           torch_sklearn_agreement=float((ours == theirs).mean()))
    return fitted, report


def install(ctrl, fitted, device):
    ctrl.gmm = {k: (density(p, device), density(n, device)) for k, (p, n) in fitted.items()}


def attach(model, seed):
    torch.manual_seed(seed); mods = []
    for _, module, children, _ in sites(model):
        for c in children:
            w = LoRA(getattr(module, c), RANK); setattr(module, c, w); mods.append(w)
    return [p for m in mods for p in (m.lora_a, m.lora_b)]


def schedule(step, steps):
    return min(1., (step + 1) / 50) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / steps)))


def adapt(model, ctrl, condition, arm, params, a, device, teacher=None):
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(.9, .95), weight_decay=0.)
    corpus = newmath_corpus('train', a.seed + 1000); log = []
    for step in range(a.steps):
        for g in opt.param_groups: g['lr'] = a.lr * schedule(step, a.steps)
        raw = corpus.batch(a.batch_size, E.CONTEXT, family='newmath')
        b = {k: raw[k].to(device) for k in ('ids', 'lengths', 'targets', 'target_mask')}
        if teacher is not None:
            with torch.no_grad(): target = teacher(b)
        out = forward(model, ctrl, condition, b['ids'], b['lengths'])
        if teacher is None:
            loss = compute(arm, model, out, b, False, torch.Generator().manual_seed(a.seed * 7919 + step))['total']
        else:
            m = b['target_mask']
            kl = F.kl_div(F.log_softmax(out['logits'][m], -1), target[m], reduction='batchmean', log_target=True)
            loss = kl
        opt.zero_grad(set_to_none=True); loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.)); opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            log.append(dict(step=step, loss=float(loss), grad_norm=norm)); print(condition, log[-1], flush=True)
    for p in params: p.requires_grad_(False)
    return log


@torch.no_grad()
def score(model, ctrl, condition, scored, device, bank=None):
    result, rates = {}, {}
    kw = {} if bank is None else dict(bank=bank)
    for fam, docs in scored.items():
        rows = []
        for b in docs:
            b = E.to(b, device); n = int(b['target_mask'].sum())
            if n == 0: continue
            out = forward(model, ctrl, condition, b['ids'], b['lengths'], **kw)
            ce = F.cross_entropy(out['logits'][0], b['targets'][0], reduction='none')
            rows.append(dict(index=b['index'], source=b['source'], n=n, ce=float((ce * b['target_mask'][0]).sum())))
            if condition in ('gated', 'consolidated'):
                length, groups = int(b['lengths'][0]), int(out['valid_groups'][0].sum())
                label = b['source'] if fam == 'math' else fam
                answer_rows = b['target_mask'][0]
                for key, g in ctrl.store.items():
                    if not key.startswith('core'):
                        ga = g[0, :, 0][answer_rows]
                        r = rates.setdefault(key + '@answer', {}).setdefault(label, [0., 0]); r[0] += float(ga.sum()); r[1] += len(ga)
                    g = g[0, :2 * groups, 0] if key.startswith('core') else g[0, :length, 0]
                    if len(g):
                        r = rates.setdefault(key, {}).setdefault(label, [0., 0]); r[0] += float(g.sum()); r[1] += len(g)
        tot = sum(r['n'] for r in rows)
        summary = dict(documents=len(rows), nll=sum(r['ce'] for r in rows) / tot)
        if fam == 'math':
            summary['by_module'] = {s: sum(r['ce'] for r in rows if r['source'] == s) / sum(r['n'] for r in rows if r['source'] == s)
                                    for s in sorted({r['source'] for r in rows})}
        result[fam] = dict(summary=summary, documents=rows)
    open_rates = {k: {f: v[0] / v[1] for f, v in fams.items()} for k, fams in rates.items()}
    return result, open_rates


@torch.no_grad()
def generate(model, ctrl, condition, device, tok, bank=None):
    rng = np.random.default_rng(2718); result = {}
    kw = {} if bank is None else dict(bank=bank)
    for fam, n in GEN.items():
        arr, offsets, starts, meta = E.load_family(fam)
        groups = {}
        for i, r in enumerate(meta): groups.setdefault(r['source'], []).append(i)
        chosen = [int(i) for g in sorted(groups) for i in rng.choice(groups[g], n, replace=False)]
        rows = []
        for i in chosen:
            ids = torch.tensor([[int(t) for t in arr[offsets[i]:offsets[i] + starts[i]]]], device=device); start = ids.shape[1]; committed = None
            for _ in range(24):
                out = forward(model, ctrl, condition, ids, None, codes=committed, **kw); committed = out['codes']
                nxt = out['logits'][:, -1].argmax(-1); ids = torch.cat((ids, nxt[:, None]), -1)
                if int(nxt) == 2: break
            pred = ids[0, start:].cpu().tolist(); eos = bool(pred and pred[-1] == 2)
            text = tok.decode(pred[:-1] if eos else pred, skip_special_tokens=False)
            rows.append(dict(index=i, source=meta[i]['source'], correct=bool(score_math(text, meta[i]['answer'], eos)['correct'])))
        by = {}
        for r in rows: by.setdefault(r['source'], []).append(r['correct'])
        result[fam] = {k: dict(n=len(v), correct=int(sum(v))) for k, v in by.items()}
        result[fam]['rows'] = rows
    return result


def bind(model, ck, sigma):
    tok_sha = hashlib.sha256((DATA / 'tokenizer.json').read_bytes()).hexdigest()
    pre_sha = hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest()
    era = model_era(model, tok_sha, pre_sha, sigma); model.bind_memory_era(era, sigma); return era


def write_bank(model, ck, a, device, eta, sigma, episodes, seed):
    era = bind(model, ck, sigma)
    bank = RBFMemory(era, width=2 * model.c.width, capacity=8192, sigma=sigma, codes=model.c.codes)
    corpus = newmath_corpus('train', seed); reports = []
    for i in range(episodes):
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in corpus.batch(1, E.CONTEXT, family='newmath').items()}
        r = learn_episode(model, bank, b['ids'], b['lengths'], b['targets'], b['target_mask'], f'newmath-{i}', era, eta)
        reports.append(dict(writes=r['writes'], eta=r['eta'], prewrite_loss=r['prewrite_loss']))
    first, last = reports[:50], reports[-50:]
    summary = dict(episodes=len(reports), rows=int(len(bank.keys)), eta=eta, sigma=sigma,
                   prewrite_loss_first50=float(np.mean([r['prewrite_loss'] for r in first])),
                   prewrite_loss_last50=float(np.mean([r['prewrite_loss'] for r in last])),
                   effective_eta_median=float(np.median([r['eta'] for r in reports])),
                   value_rms=float(bank.values.square().mean().sqrt()) if len(bank.values) else 0.)
    return bank, summary


@torch.no_grad()
def mean_loss(model, docs, device, bank=None):
    num = den = 0.
    for b in docs:
        b = E.to(b, device); out = model(b['ids'], b['lengths'], **({} if bank is None else dict(bank=bank)))
        ce = F.cross_entropy(out['logits'][0], b['targets'][0], reduction='none'); m = b['target_mask'][0]
        num += float((ce * m).sum()); den += float(m.sum())
    return num / den


def choose_bank(model, ck, a, device):
    """Predeclared calibration on TRAIN-split documents disjoint from the write
    stream (dev untouched): minimise held-out newmath loss subject to at most
    0.01 nats/token harm on held-out pretraining-family documents."""
    held = newmath_corpus('train', a.seed + 2000)
    new = [held.batch(1, E.CONTEXT, family='newmath') for _ in range(48)]
    old = [held.batch(1, E.CONTEXT, family='text') for _ in range(24)] + [held.batch(1, E.CONTEXT, family='math') for _ in range(24)]
    base_new, base_old = mean_loss(model, new, device), mean_loss(model, old, device)
    grid = []
    for sigma in a.sigma_grid:
        for eta in a.eta_grid:
            bank, summary = write_bank(model, ck, a, device, eta, sigma, a.calibration_episodes, a.seed + 3000)
            snap = bank.snapshot()
            grid.append(dict(eta=eta, sigma=sigma, new_delta=mean_loss(model, new, device, snap) - base_new,
                             old_delta=mean_loss(model, old, device, snap) - base_old, value_rms=summary['value_rms'],
                             effective_eta_median=summary['effective_eta_median']))
            print('bank grid', grid[-1], flush=True)
    ok = [g for g in grid if g['old_delta'] <= .01]
    best = min(ok or grid, key=lambda g: g['new_delta'])
    return best, dict(grid=grid, base_new=base_new, base_old=base_old, chosen=best, constraint_met=bool(ok))


def contrasts(results, base_name='base'):
    rng = np.random.default_rng(0); out = {}
    names = [n for n in results if n != base_name]
    pairs = [(base_name, n) for n in names] + ([('plain', 'gated')] if {'plain', 'gated'} <= set(results) else [])
    for x, y in pairs:
        entry = {}
        for fam in results[x]['nll']:
            dx, dy = results[x]['nll'][fam]['documents'], results[y]['nll'][fam]['documents']
            entry[fam] = paired(dx, dy, rng)
            if fam == 'math':
                for s in sorted({r['source'] for r in dx}):
                    entry[s] = paired([r for r in dx if r['source'] == s], [r for r in dy if r['source'] == s], rng)
        out[f'{y} - {x}'] = entry
    return out


def g4(results, cs):
    if not {'plain', 'gated'} <= set(results): return None
    d = cs['gated - plain']; checks = {}
    for key in ('text', 'arithmetic__mul', 'comparison__sort'):
        nll_margin = -d[key]['delta']
        acc = None
        if key != 'text':
            gp, pp = results['gated']['generation']['math'][key], results['plain']['generation']['math'][key]
            acc = 100 * (gp['correct'] - pp['correct']) / gp['n']
        checks[key] = dict(nll_improvement_vs_plain=nll_margin, accuracy_points_vs_plain=acc,
                           passed=bool(nll_margin >= .01 or (acc is not None and acc >= 2)))
    return dict(rule='gated beats plain on text, mul and sort by >= 0.01 nats/token or >= 2 accuracy points (ARCHITECTURE.md G4)',
                checks=checks, passed=all(c['passed'] for c in checks.values()),
                related_skill_add_sub=dict(nll_gated_minus_plain=d['arithmetic__add_sub_multiple']['delta']),
                new_skill=dict(newmath_nll_gated_minus_plain=d['newmath']['delta'],
                               newmath_exact=dict(plain=results['plain']['generation']['newmath'], gated=results['gated']['generation']['newmath'])))


def main(a):
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(a.threads); device = a.device; started = time.time()
    tok = Tokenizer.from_file(str(DATA / 'tokenizer.json'))
    scored, _ = E.panels(); results = {}; record = dict(args=vars(a), conditions=results)
    def save():
        slim = {n: {k: v for k, v in r.items()} for n, r in results.items()}
        record['contrasts'] = contrasts(results) if 'base' in results and len(results) > 1 else {}
        record['G4'] = g4(results, record['contrasts']) if record['contrasts'] else None
        record['seconds'] = time.time() - started
        atomic_json(out_dir / 'sleep.json', dict(record, conditions=slim))
    def finish(name, model, ctrl, condition, bank=None, **extra):
        nll, rates = score(model, ctrl, condition, scored, device, bank)
        gen = generate(model, ctrl, condition, device, tok, bank)
        results[name] = dict(nll=nll, generation=gen, gate_open_rates=rates, **extra)
        brief = {f: round(v['summary']['nll'], 4) for f, v in nll.items()}
        print(name, brief, {f: {k: v['correct'] for k, v in g.items() if k != 'rows'} for f, g in gen.items()},
              round(time.time() - started), flush=True)
        save()

    model, ck = load(a.run_dir, device); arm = ck['arm']; record['arm'] = arm; record['checkpoint_step'] = ck['step']
    ctrl = Controller(model)
    finish('base', model, ctrl, 'base')
    fitted, gate_report = fit_gates(model, ctrl, device, a.seed); record['gate_fit'] = gate_report; save()
    print('gates fitted', round(time.time() - started), flush=True)

    model, _ = load(a.run_dir, device); ctrl = Controller(model); params = attach(model, a.seed)
    log = adapt(model, ctrl, 'plain', arm, params, a, device)
    finish('plain', model, ctrl, 'plain', train_log=log, adapter_parameters=sum(p.numel() for p in params))

    model, _ = load(a.run_dir, device); ctrl = Controller(model); install(ctrl, fitted, device); params = attach(model, a.seed)
    log = adapt(model, ctrl, 'gated', arm, params, a, device)
    finish('gated', model, ctrl, 'gated', train_log=log, adapter_parameters=sum(p.numel() for p in params))

    if a.episodes:
        model, _ = load(a.run_dir, device); ctrl = Controller(model)
        best, calibration = choose_bank(model, ck, a, device); record['bank_calibration'] = calibration; save()
        bank, summary = write_bank(model, ck, a, device, best['eta'], best['sigma'], a.episodes, a.seed + 1000)
        finish('rbf', model, ctrl, 'base', bank=bank.snapshot(), bank_summary=summary)
        if a.consolidate:
            snap = bank.snapshot(); ctrl.gmm = {}; install(ctrl, fitted, device)
            def teacher(b):
                set_enabled(model, False); ctrl.begin('idle')
                return F.log_softmax(model(b['ids'], b['lengths'], bank=snap)['logits'][b['target_mask']], -1)
            params = attach(model, a.seed)
            for p in params: p.requires_grad_(False)
            def teacher_wrapped(b):
                was = [p.requires_grad for p in params]
                for p in params: p.requires_grad_(False)
                t = teacher(b); set_enabled(model, True)
                for p, w in zip(params, was): p.requires_grad_(w)
                return t
            # Distil per masked row: the student sees the same rows the teacher scored.
            def teacher_full(b):
                t = teacher_wrapped(b); full = torch.zeros(*b['target_mask'].shape, t.shape[-1], device=t.device); full[b['target_mask']] = t
                return full
            log = adapt(model, ctrl, 'consolidated', arm, params, a, device, teacher=teacher_full)
            bank = None   # reset: the consolidated model is scored without any bank
            finish('consolidated', model, ctrl, 'consolidated', train_log=log)
    print('DONE', round(time.time() - started), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True); p.add_argument('--out-dir', required=True)
    p.add_argument('--device', default='mps'); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=5); p.add_argument('--steps', type=int, default=600)
    p.add_argument('--batch-size', type=int, default=16); p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--episodes', type=int, default=512); p.add_argument('--calibration-episodes', type=int, default=96)
    p.add_argument('--eta-grid', type=float, nargs='+', default=[.1, 10., 100., 1000.])
    p.add_argument('--sigma-grid', type=float, nargs='+', default=[1., .3, .1]); p.add_argument('--consolidate', action='store_true')
    main(p.parse_args())
