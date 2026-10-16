"""S6-base (draft §10.3-§10.7): phase training for LSL-scaled / always-open / VQ-only, G6.0, per-phase measurements.

Evaluation uses full recomputation for every model (pre- and post-phase alike), so paired comparisons share one
engine; at zero banks it reproduces frozen generate() (contract 14 suite).
"""
import argparse, json, math, random, signal, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import v9paths
from v9paths import DATA, RUNS, SMOKE, digest, read_json, write_json, files_hash, CODE
from model import chunk_info, shift, freeze, count, Core
from runtime import DEVICE, sync, seed_all, atomic_json, Log, lr_at, nll, balanced, load_system, implementation_hash as frozen_hash
from data import Mixer, fixed_text, PAD, DONE, T, parse_prompt
from evaluate import text_eval
import lsl, v9data
from v9data import dispatch_verify, is_colourcount, phase_mixer

S6_FILES = [CODE / n for n in ('s6.py', 'lsl.py', 'v9data.py', 'v9paths.py')]
FOUNDATION_SUITES = (('standard', 1024), ('lookup', 512), ('compose', 512), ('pointer', 512), ('hard10', 512))
PHASE_DEV = dict(colourcount=('cc', 'colourcount_dev'), shifted=('v8', 'dev_shifted'), bind=('v8', 'dev_bind'))
LOOKUP_LIKE = ('lookup', 'compose', 'bind')
NB = 4 if SMOKE else 82          # ~1M tokens of 64 x 192 batches for each GMM fit (§10.4)
CAP = 32 if SMOKE else None      # per-suite item cap in smoke mode only


# ------------------------------------------------------------------ evaluation engine (System, full recompute)
@torch.no_grad()
def last_logits(s, x, docs, pos):
    f = s.foundation
    if s.is_flat:
        h = f.emb(x)
        for b in f.blocks:
            h = b(h, docs)
        hid = f.nf(h)
    else:
        h = f.encode(x, docs); c = f.pool(h); cd, _ = chunk_info(docs); chat, a, _ = s.plans(c, cd)
        z = h + shift(chat, docs) + f.gamma * shift(a, docs)
        for b in f.dec:
            z = b(z, docs)
        hid = f.nf(z)
    return hid[torch.arange(len(pos), device=x.device), pos] @ f.emb.weight.t()


@torch.no_grad()
def greedy(s, prompts, max_new):
    B = len(prompts); P = len(prompts[0]); limit = min(T, P + max_new)
    x = torch.full((B, limit + 4), PAD, dtype=torch.long, device=DEVICE); docs = torch.full_like(x, -1)
    x[:, :P] = torch.as_tensor(np.stack(prompts), device=DEVICE); docs[:, :P] = 0
    one_chunk = torch.tensor([is_colourcount(p) or parse_prompt(p)[0] in LOOKUP_LIKE for p in prompts], device=DEVICE)
    done = torch.zeros(B, dtype=torch.bool, device=DEVICE); length = torch.full((B,), limit, device=DEVICE)
    for pos in range(P, limit):
        tok = last_logits(s, x[:, :pos + 1], docs[:, :pos + 1], torch.full((B,), pos - 1, device=DEVICE)).argmax(-1)
        x[:, pos] = torch.where(done, torch.full_like(tok, PAD), tok); docs[:, pos] = torch.where(done, docs[:, pos], torch.zeros_like(docs[:, pos]))
        if (pos - P) % 4 == 3:
            new = (one_chunk | (x[:, pos - 3] == DONE)) & ~done
            length = torch.where(new, torch.full_like(length, pos + 1), length); done |= new
            if bool(done.all()):
                break
    xx = x.cpu().numpy(); ln = length.cpu().numpy()
    return [dispatch_verify(xx[i, :P], xx[i, P:int(ln[i])])['success'] for i in range(B)], xx, ln


def suite_arrays(name):
    if name in ('colourcount_dev', 'colourcount_calibration', 'colourcount_test') or name.startswith('calibration_'):
        return v9data.load_npz(name)
    return np.load(DATA / f'{name}.npz')


def symbolic(s, name, n=None, batch=64):   # 256 -> 64 on MPS (I21, same cause as I19)
    """Greedy success per item, in item order; items are batched by prompt length (colour-count prompts vary)."""
    a = suite_arrays(name); n = min(CAP or n or len(a['x']), n or len(a['x']), len(a['x'])); lens = a['prompt_len'][:n]; out = np.zeros(n, bool)
    for P in sorted(set(int(v) for v in lens)):
        idx = np.flatnonzero(lens == P)
        for i in range(0, len(idx), batch):
            j = idx[i:i + batch]; ok, _, _ = greedy(s, [r[:P] for r in a['x'][j]], a['x'].shape[1] - P); out[j] = ok
    return dict(success=[bool(v) for v in out], rate=float(out.mean()), n=n)


def text(s, family, src='edu', n=256):
    r = text_eval(s, 'dev', family=family, source=src, n=CAP or n)
    return dict(numerators=r['numerators'], denominators=r['denominators'], nll=r['nll'])


def calibration_batches(names, rows=64):
    """Old-task inputs from the dedicated calibration split (never development or sealed)."""
    out = {}
    for name in names:
        if name in ('text_base', 'text_continual'):
            tb, _ = fixed_text('train', rows, seed=97600, family='base' if name == 'text_base' else 'continual')
            out[name] = [dict(x=tb['x'].to(DEVICE), docs=tb['docs'].to(DEVICE), m=tb['m'].to(DEVICE))]
        else:
            a = v9data.load_npz('colourcount_calibration' if name == 'colourcount' else f'calibration_{name}')
            out[name] = [dict(x=torch.tensor(a['x'][i:i + rows], device=DEVICE), docs=torch.tensor(a['docs'][i:i + rows], device=DEVICE), m=torch.tensor(a['m'][i:i + rows], device=DEVICE)) for i in range(0, min(256, len(a['x'])), rows)]
    return out


@torch.no_grad()
def function_kl(pre, post, batches):
    """Token-level KL(pre || post) on old-task inputs (paper App. H), mean over valid next-token positions."""
    tot = 0.; n = 0.
    for b in batches:
        lp = pre(b['x'], b['docs']).float().log_softmax(-1); lq = post(b['x'], b['docs']).float().log_softmax(-1)
        kl = (lp.exp() * (lp - lq)).sum(-1); valid = (b['docs'] >= 0).float()
        tot += float((kl * valid).sum()); n += float(valid.sum())
    return tot / max(n, 1)


# ------------------------------------------------------------------ training
def phase_batches(mode, seed, n):
    mix = phase_mixer(mode, seed); return [mix.batch(DEVICE) for _ in range(n)]


def train_phase(args):
    torch.set_num_threads(4); seed_all(args.seed); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    first = args.phase == 1
    if first:
        s, _ = load_system(args.source, DEVICE); freeze(s)
        meta = dict(anchor=args.source, anchor_sha256=digest(args.source), arm=args.arm, rho=.3, forces=[], modes=[], neg=None, pos=[])
        if args.arm != 'vq':
            lsl.convert(s, .3)
            # Phi_neg once per anchor, on the foundation mixture (~1M tokens), as in V8 (§10.4).
            mix = Mixer(99100 + args.seed, 64, 'foundation'); batches = [mix.batch(DEVICE) for _ in range(NB)]
            meta['neg'] = lsl.fit_negative(s, batches)
    else:
        s, meta = lsl.load(args.source, DEVICE) if read_json(Path(args.source).parent / 'phase.json')['arm'] != 'vq' else load_vq(args.source)
    if meta['arm'] != 'vq':
        banks = lsl.add_phase(s, True); params = [p for b in banks for p in b.parameters()]
    else:
        params = [s.foundation.Q] + list(s.foundation.heads.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=3e-4, betas=(.9, .95), eps=1e-8, weight_decay=0.)
    mix = phase_mixer(args.mode, 99200 + 10 * args.seed + args.phase)
    log = Log(out / 'log.jsonl'); begin = time.monotonic(); last = None
    config = dict(vars(args), device=DEVICE, files_sha256=files_hash(S6_FILES), frozen_implementation_sha256=frozen_hash(), trainable_parameters=sum(p.numel() for p in params),
                  created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_json(out / 'config.json', config)
    for index in range(args.steps):
        b = mix.batch(DEVICE)
        for g in opt.param_groups:
            g['lr'] = lr_at(index, args.steps, 3e-4, 100)
        opt.zero_grad(set_to_none=True)
        logits = s(b['x'], b['docs'])
        loss = balanced(nll(logits, b['x'], b['m']), b['m'][:, :-1], b['domain'])[0]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite phase loss')
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); opt.step()
        if index == 0 or (index + 1) % 100 == 0:
            last = dict(step=index + 1, planned_steps=args.steps, loss=float(loss), gradient_norm=float(norm), elapsed_s=time.monotonic() - begin); log(**last); atomic_json(out / 'live.json', last)
    for p in params:
        p.requires_grad_(False)
    meta = dict(meta); meta['modes'] = meta['modes'] + [args.mode]
    if meta['arm'] == 'lsl':
        # Phi_pos on M + adapter after training (paper Alg. 1), the newest bank held open while capturing.
        meta['pos'] = meta['pos'] + [lsl.fit_positive(s, lsl.phases(s) - 1, phase_batches(args.mode, 99300 + args.seed + 10 * args.phase, NB))]
        lsl.set_force(s, lsl.phases(s) - 1, None); meta['forces'] = meta['forces'] + [None]
    elif meta['arm'] == 'open':
        meta['forces'] = meta['forces'] + [True]
    if meta['arm'] == 'vq':
        save_vq(out / 'checkpoint.pt', s, meta)
    else:
        lsl.save(out / 'checkpoint.pt', s, meta)
    write_json(out / 'phase.json', dict(arm=meta['arm'], phase=args.phase, mode=args.mode, modes=meta['modes']))
    write_json(out / 'train_summary.json', dict(**config, status='completed', completed_steps=args.steps, wall_s=time.monotonic() - begin, last=last, lifetime=mix.seen_tasks))


def save_vq(path, s, meta):
    from runtime import atomic_save
    atomic_save(path, dict(lsl=meta, model=s.state_dict(), metadata=s.metadata()))


def load_vq(path):
    ck = torch.load(path, map_location='cpu', weights_only=False); s, _ = load_system(ck['lsl']['anchor'], 'cpu'); s.load_state_dict(ck['model']); freeze(s)
    return s.to(DEVICE), ck['lsl']


def load_any(path):
    if Path(path).name == 'checkpoint.pt' and (Path(path).parent / 'phase.json').exists():
        ph = read_json(Path(path).parent / 'phase.json')
        return load_vq(path)[0] if ph['arm'] == 'vq' else lsl.load(path, DEVICE)[0]
    s, _ = load_system(path, DEVICE); freeze(s); return s


# ------------------------------------------------------------------ G6.0 (before a phase, a pre-registration proxy)
def g60(model_path, mode, earlier, seed):
    """Fit Phi_pos on phase data with the new adapter at zero; smoothed (rho = .3) hit rates on calibration inputs of every
    earlier phase and every foundation suite. Pass: mean <= 5% and no suite > 15%; 5-10% weakly separable; else not."""
    s = load_any(model_path)
    if not isinstance(lsl.gated_blocks(s)[0], lsl.LSLBlock):
        lsl.convert(s, .3)
        mix = Mixer(99100 + seed, 64, 'foundation'); lsl.fit_negative(s, [mix.batch(DEVICE) for _ in range(NB)])
    proxy = [lsl.Bank().to(DEVICE) for _ in lsl.gated_blocks(s)]
    lsl.fit_positive(s, None, phase_batches(mode, 99400 + seed, NB), target=proxy)
    suites = ['standard', 'lookup', 'compose', 'pointer', 'hard10', 'text_base'] + [{'text': 'text_continual'}.get(m, m) for m in earlier]
    cal = calibration_batches(suites); rates = {name: lsl.hit_rates(s, proxy, bs) for name, bs in cal.items()}
    means = {k: v['mean'] for k, v in rates.items()}; mean = float(np.mean(list(means.values()))); mx = max(means.values())
    label = 'separable' if mean <= .05 and mx <= .15 else 'weakly separable' if mean <= .10 else 'non-separable'
    return dict(mode=mode, suites=rates, suite_means=means, mean=mean, max_suite=mx, label=label, passed=label == 'separable')


# ------------------------------------------------------------------ per-phase measurement
def measure_model(path, new_modes, telemetry=False):
    """Every dev measurement a phase decision needs, as per-item vectors (paired across models)."""
    s = load_any(path); s.eval(); out = dict(path=str(path), symbolic={}, text={})
    for name, n in FOUNDATION_SUITES:
        out['symbolic'][name] = symbolic(s, f'dev_{name}', n)
    for src in ('edu', 'ts'):
        out['text']['base_' + src] = text(s, 'base', src)
    for mode in new_modes:
        if mode == 'text':
            out['text']['continual'] = text(s, 'continual', 'edu')
        else:
            out['symbolic'][mode] = symbolic(s, PHASE_DEV[mode][1], 512)
    return out


def telemetry(path, new_mode, earlier):
    """Gate telemetry of the newest bank (LSL): old- and new-task calibration inputs, per matrix with tails, joint all-open."""
    s = load_any(path); p = lsl.phases(s) - 1
    banks = [blk.lsl_banks[p] for blk in lsl.gated_blocks(s)]
    suites = ['standard', 'lookup', 'compose', 'pointer', 'hard10', 'text_base'] + [{'text': 'text_continual'}.get(m, m) for m in earlier + [new_mode]]
    cal = calibration_batches(suites)
    return {name: lsl.hit_rates(s, banks, bs) for name, bs in cal.items()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd')
    t = sub.add_parser('train'); t.add_argument('--source', required=True); t.add_argument('--out', required=True); t.add_argument('--arm', choices=['lsl', 'open', 'vq'], required=True)
    t.add_argument('--mode', required=True); t.add_argument('--phase', type=int, required=True); t.add_argument('--steps', type=int, default=2000); t.add_argument('--seed', type=int, required=True)
    t.add_argument('--resume', action='store_true', help='a phase is short and fully seeded: an interrupted phase restarts from its source checkpoint')
    g = sub.add_parser('g60'); g.add_argument('--source', required=True); g.add_argument('--mode', required=True); g.add_argument('--earlier', default=''); g.add_argument('--seed', type=int, required=True); g.add_argument('--out', required=True)
    m = sub.add_parser('measure'); m.add_argument('--source', required=True); m.add_argument('--modes', default=''); m.add_argument('--out', required=True); m.add_argument('--pre', default=None)
    m.add_argument('--telemetry-mode', default=None); m.add_argument('--earlier', default='')
    a = ap.parse_args(); torch.set_num_threads(4)
    if a.cmd == 'train':
        train_phase(a)
    elif a.cmd == 'g60':
        write_json(a.out, g60(a.source, a.mode, [v for v in a.earlier.split(',') if v], a.seed))
    else:
        modes = [v for v in a.modes.split(',') if v]; r = measure_model(a.source, modes)
        if a.pre:
            pre = load_any(a.pre); post = load_any(a.source)
            r['function_kl'] = function_kl(pre, post, [b for bs in calibration_batches(['standard', 'lookup', 'compose', 'pointer', 'hard10', 'text_base']).values() for b in bs])
        if a.telemetry_mode:
            earlier = [v for v in a.earlier.split(',') if v]
            r['telemetry'] = telemetry(a.source, a.telemetry_mode, earlier)
            # Same-weight installation test: the newest bank forced open on the same checkpoint (§10.7).
            s = load_any(a.source); lsl.set_force(s, lsl.phases(s) - 1, True); s.eval()
            r['forced_open_new'] = text(s, 'continual', 'edu') if a.telemetry_mode == 'text' else symbolic(s, PHASE_DEV[a.telemetry_mode][1], 512)
        write_json(a.out, r)
