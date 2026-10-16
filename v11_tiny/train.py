"""Step-based pretraining for one arm of the v11.md §6.6 objective test.

  python -B -m v11_tiny.train --arm K1 --seed 1 --run-dir v11_tiny/runs/K1_s1

Every arm uses the inherited V10.3-tiny schedule and data stream: batch 16, context 256,
AdamW (0.9, 0.95) lr 1e-3, warmup 200 updates then cosine to 10% over --max-steps, dense
clip 1.0, CPU SparseAdam Engram at 5x LR with its own clip, one Corpus('train', seed)
batch per update. C0 and X8 run the inherited objective code (v10_3_tiny.losses.compute,
candidate updates every 4th step, the same generator formula); `tests_trainer.py` checks that
this trainer reproduces v10_3_tiny.train step for step.

Throughput: a background thread prepares the next batch (Corpus call order is unchanged
and progress is a function of the step index, so the stream is identical) and its Engram
hash indices. AdamW `foreach` is a flag (numerics differ only at rounding level).
"""
import argparse, hashlib, json, math, queue, random, threading, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from v10_10M.data import Corpus
from v10_10M.train import canonical_ids, atomic_json
from v10_3_tiny.train import quick_eval
from v10_3_tiny.losses import compute as compute_v103
from .model import build, counts, V11Tiny, ARMS
from .losses import compute as compute_v11, LAMBDA

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'v10_3_tiny' / 'data'
NO_DECAY = {'X8': {'head.codebook'}, 'NOAUX': {'head.codebook'}, 'K1NB': {'head.codebook'}, 'K1': {'head.codebook'}}
DEVIATIONS = [
    'K = 1 for every V11 arm: no strategy codes, no prior, no visit-2 code injection (2,040 parameters fewer than V10.3 deployed).',
    'L_v2 is computed by recomputing visit 2 for all rows from stopped committed inputs (u1, anchor, E->C sources) and '
    'taking the S sampled rows; v11.md §5.2 attends to a stopped committed KV instead, so here the shared core weights also '
    'receive L_v2 gradient through earlier rows keys/values.',
    'No random-prefix training: the objective test uses the inherited V10.3-tiny stream so the X8/C0 comparison is clean; '
    'alignment is tested by offset scoring in the evaluator for every arm.',
]


def optimizers(arm, model, lr, foreach):
    exempt = NO_DECAY.get(arm, set())
    dense = [(n, p) for n, p in model.named_parameters() if n != 'engram.table.weight']
    decay = [p for n, p in dense if p.ndim >= 2 and n not in exempt]
    no_decay = [p for n, p in dense if p.ndim < 2 or n in exempt]
    adam = torch.optim.AdamW([dict(params=decay, weight_decay=.1), dict(params=no_decay, weight_decay=0.)],
                             lr=lr, betas=(.9, .95), foreach=foreach)
    table = torch.optim.SparseAdam([model.engram.table.weight], lr=lr * 5, betas=(.9, .99)) if hasattr(model, 'engram') else None
    return adam, table, [p for _, p in dense]


def source_hash():
    h = hashlib.sha256()
    for p in sorted(ROOT.glob('*.py')) + [ROOT.parent / f for f in ('v10_10M/model.py', 'v10_1/model.py', 'v10_2/model.py',
                                                                     'v10_10M/data.py', 'v10_3_tiny/model.py', 'v10_3_tiny/losses.py')]:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


class Prefetch:
    """Producer thread: the Corpus is only ever called from this thread, in step order."""

    def __init__(self, corpus, a, engram, depth=3):
        self.q = queue.Queue(depth); self.stop = False
        def run():
            for step in range(a.max_steps):
                raw = corpus.batch(a.batch_size, a.context, progress=step / a.max_steps)
                idx = engram.indices(raw['ids']) if engram is not None else None
                while not self.stop:
                    try: self.q.put((raw, idx), timeout=1.); break
                    except queue.Full: pass
                if self.stop: return
        self.t = threading.Thread(target=run, daemon=True); self.t.start()

    def get(self): return self.q.get()


def stream_hash(raw):
    return hashlib.sha256((raw['family'] + ',' + ','.join(map(str, raw['chosen']))).encode()).hexdigest()[:16]


def scalar(v):
    if v is None: return None
    if torch.is_tensor(v): return float(v.detach())
    return v


@torch.no_grad()
def buffer_stats(model, out):
    if not isinstance(model, V11Tiny) or not bool(model.target_ready): return {}
    valid = out['valid_groups']
    tau = model.normalized_target(out['pooled'])[valid].flatten(1)
    if len(tau) < 4: return dict(target_var_mean=float(model.target_var.mean()))
    cov = torch.cov(tau.T.float()); ev = torch.linalg.eigvalsh(cov.cpu()).clamp_min(0)
    return dict(target_var_mean=float(model.target_var.mean()), target_mu_rms=float(model.target_mu.square().mean().sqrt()),
                tau_participation_ratio=float(ev.sum() ** 2 / ev.square().sum().clamp_min(1e-12)),
                raw_pooled_var_median=float(out['pooled'][valid].var(0).median()))


def main(a):
    run = Path(a.run_dir); run.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    tok = Tokenizer.from_file(str(DATA / 'tokenizer.json'))
    model = build(a.arm)
    if hasattr(model, 'engram'): model.engram.canonical.copy_(canonical_ids(tok, model.c.vocab_size))
    model.place(a.device).train()
    adam, table, dense = optimizers(a.arm, model, a.lr, a.foreach)
    corpus = Corpus('train', seed=a.seed, data_dir=DATA)
    dev = Corpus('dev', seed=4242, data_dir=DATA)
    panel = {f: [dev.batch(8, a.context, family=f) for _ in range(n)] for f, n in
             dict(text=4, math=4, sudoku_easy=2, sudoku_hard=2).items()}
    manifest = dict(arm=a.arm, config=model.c.to_dict(), parameters=counts(model), args=vars(a), source_sha256=source_hash(),
                    data_manifest_sha256=hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest(),
                    torch=torch.__version__, start_unix=time.time(), lambda_=LAMBDA if a.arm in ('K1NB', 'K1') else None,
                    S=a.S if a.arm in ('K1NB', 'K1') else None,
                    objective={'C0': 'v10_3_tiny.losses (token CE)', 'X8': 'v10_3_tiny.losses (V10.3 recipe, candidates every 4th step)',
                               'ORD': 'token CE', 'ORDE': 'token CE (ordinary + Engram)', 'NOAUX': 'token CE + detached probe', 'K1NB': 'v11 latent objective, no bridge',
                               'K1': 'v11 latent objective + bridge'}[a.arm],
                    deviations_from_v11_md=DEVIATIONS if a.arm in ('NOAUX', 'K1NB', 'K1') else None,
                    trainer_notes=['Same trainer for all six arms; with --foreach off it reproduces v10_3_tiny.train bitwise for C0 '
                                   'and X8 over 12 steps (tests_trainer). Production runs use --foreach (AdamW multi-tensor), '
                                   'which differs from the inherited trainer only at rounding level.' if a.foreach else
                                   'foreach off: bitwise-identical to v10_3_tiny.train for C0/X8.',
                                   'Runs share the GPU three at a time; wall clock depends on co-scheduled arms. Equal-wall-time '
                                   'comparisons use solo smoke step times, not this run\'s elapsed time.'],
                    optimizer_rule='decay ndim>=2 except ' + (','.join(sorted(NO_DECAY.get(a.arm, []))) or 'nothing'))
    atomic_json(run / 'manifest.json', manifest)
    step = tokens = 0; fam_tokens = Counter(); start = time.monotonic(); evals = []
    log = (run / 'train.jsonl').open('w', buffering=1)
    def save(tag):
        torch.save(dict(arm=a.arm, config=model.c.to_dict(), model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        step=step, tokens=tokens, family_tokens=dict(fam_tokens), elapsed=time.monotonic() - start,
                        source_sha256=manifest['source_sha256']), run / f'{tag}.pt')
    ev = quick_eval(model, a.arm, a.device, panel); evals.append(dict(step=0, elapsed=0., nll=ev))
    feed = Prefetch(corpus, a, getattr(model, 'engram', None))
    try:
        while step < a.max_steps:
            tick = time.monotonic(); progress = step / a.max_steps
            scale = min(1., (step + 1) / 200) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
            for g in adam.param_groups: g['lr'] = a.lr * scale
            if table is not None:
                for g in table.param_groups: g['lr'] = a.lr * 5 * scale
            raw, idx = feed.get(); waited = time.monotonic() - tick
            b = {k: raw[k].to(a.device) for k in ('ids', 'lengths', 'targets', 'target_mask')}
            generator = torch.Generator().manual_seed(a.seed * 100003 + step)
            adam.zero_grad(set_to_none=True)
            if table is not None: table.zero_grad(set_to_none=True)
            candidate = False
            if a.arm in ('C0', 'X8'):
                candidate = a.arm != 'C0' and step % a.candidate_every == 0
                out = model(b['ids'], b['lengths'], idx)
                parts = compute_v103(a.arm, model, out, b, candidate, generator)
            else:
                out = model(b['ids'], b['lengths'], idx)
                if isinstance(model, V11Tiny) and not bool(model.target_ready):
                    model.update_target_buffers(out['pooled'], out['valid_groups'])     # init before the first loss
                parts = compute_v11(a.arm, model, out, b, step, 200, generator, a.S)
            parts['total'].backward()
            if a.separate_clip:
                # Training-only modules get their own clip groups; the deployed trunk's clip never sees them.
                named = [(n, p_) for n, p_ in model.named_parameters() if n != 'engram.table.weight']
                groups = dict(trunk=[p_ for n, p_ in named if not n.startswith(('bridge.', 'probe.'))],
                              bridge=[p_ for n, p_ in named if n.startswith('bridge.')],
                              probe=[p_ for n, p_ in named if n.startswith('probe.')])
                norms = {k: float(torch.nn.utils.clip_grad_norm_(v, 1.0)) if v else None for k, v in groups.items()}
                dense_norm = norms['trunk']
            else:
                norms = None
                dense_norm = float(torch.nn.utils.clip_grad_norm_(dense, 1.0))
            sparse_norm = sparse_post = None
            if table is not None:
                tg = model.engram.table.weight.grad
                if tg is not None:
                    tg = tg.coalesce(); sparse_norm = float(tg.values().norm())
                    if sparse_norm > 1.: tg.values().mul_(1. / sparse_norm)
                    model.engram.table.weight.grad = tg
                    sparse_post = float(model.engram.table.weight.grad.coalesce().values().norm())
            adam.step()
            if table is not None: table.step()
            if isinstance(model, V11Tiny) and a.arm != 'NOAUX':
                model.update_target_buffers(out['pooled'], out['valid_groups'])       # after backward (v11.md §3.3)
            step += 1
            n_in = int(raw['lengths'].sum()); tokens += n_in; scored = int(raw['target_mask'].sum())
            fam_tokens[raw['family']] += scored
            seconds = time.monotonic() - tick
            rec = dict(step=step, family=raw['family'], stream=stream_hash(raw), elapsed=time.monotonic() - start,
                       step_seconds=seconds, data_wait_seconds=waited, input_tokens=n_in, scored_tokens=scored,
                       tokens_per_second=n_in / seconds, lr=a.lr * scale, dense_grad_norm=dense_norm,
                       engram_grad_norm=sparse_norm, engram_grad_norm_post=sparse_post, candidate_executed=candidate,
                       clip_groups=norms)
            for k, v in parts.items():
                if k == 'winners': rec[k] = v
                elif k == 'future_visits': rec[k] = [scalar(x) for x in v]
                elif torch.is_tensor(v) and v.numel() == 1 or v is None or isinstance(v, (int, float, bool)): rec[k] = scalar(v)
            if isinstance(model, V11Tiny) and a.arm != 'NOAUX':
                rec['sigma1_sq'] = float(model.log_sigma1.detach().clamp_min(math.log(.01)).exp())
                rec['sigma2_sq'] = float(model.log_sigma2.detach().clamp_min(math.log(.01)).exp())
            if step % 50 == 0 or step == 1:
                rec.update(buffer_stats(model, out))
                with torch.no_grad():
                    if hasattr(model, 'head'): rec['codebook_rms'] = float(model.head.codebook.square().mean().sqrt())
                    if hasattr(model, 'decoder') and hasattr(model.decoder, 'fusion_map'):
                        rec['fusion_map_rms'] = float(model.decoder.fusion_map.weight.square().mean().sqrt())
                if a.device == 'mps': rec['mps_driver_gb'] = torch.mps.driver_allocated_memory() / 2 ** 30
            if not math.isfinite(rec['total']): raise FloatingPointError('nonfinite loss at step %d' % step)
            log.write(json.dumps(rec) + '\n')
            if step % a.eval_every == 0 and step < a.max_steps:
                ev = quick_eval(model, a.arm, a.device, panel); evals.append(dict(step=step, elapsed=time.monotonic() - start, nll=ev))
                atomic_json(run / 'quick_eval.json', evals); save('latest')
                print(json.dumps(dict(step=step, tokens=tokens, sps=round(step / (time.monotonic() - start), 2),
                                      **{f'dev_{k}': round(v, 4) for k, v in ev.items()})), flush=True)
    finally:
        feed.stop = True
    ev = quick_eval(model, a.arm, a.device, panel); evals.append(dict(step=step, elapsed=time.monotonic() - start, nll=ev))
    atomic_json(run / 'quick_eval.json', evals); save('final')
    manifest.update(end_unix=time.time(), steps=step, tokens=tokens, family_supervised_tokens=dict(fam_tokens),
                    active_seconds=time.monotonic() - start)
    atomic_json(run / 'manifest.json', manifest)
    print('DONE', a.arm, step, tokens, round(time.monotonic() - start), flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=ARMS, required=True); p.add_argument('--run-dir', required=True)
    p.add_argument('--device', choices=['cpu', 'mps'], default='mps'); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16); p.add_argument('--context', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--candidate-every', type=int, default=4)
    p.add_argument('--threads', type=int, default=2); p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--max-steps', type=int, default=6000); p.add_argument('--S', type=int, default=2)
    p.add_argument('--foreach', action='store_true')
    p.add_argument('--separate-clip', action='store_true', help='diagnostic: clip trunk, bridge and probe separately at 1.0')
    return p


if __name__ == '__main__':
    main(parser().parse_args())
