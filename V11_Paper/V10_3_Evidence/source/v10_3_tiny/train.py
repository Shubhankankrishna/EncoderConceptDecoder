"""Wall-time-bounded pretraining for one tiny arm.

  python -B -m v10_3_tiny.train --arm X8 --minutes 25 --run-dir v10_3_tiny/runs/X8_s1 --seed 1

Same schedule for every arm: AdamW (0.9, 0.95), warmup 200 updates, cosine to
10% over the wall budget, dense clip 1.0, CPU SparseAdam Engram at 5x LR with its
own clip. Candidate (exploration) updates every --candidate-every updates for
A and X arms. Optimizer grouping differs by arm on purpose: A and C0 use V10.1's
rule (decay every dense parameter with ndim >= 2, which includes the PQ
codebook); X arms exclude head.codebook from decay (V10.3 condition N3).
"""
import argparse, hashlib, json, math, os, random, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from v10_10M.data import Corpus
from v10_10M.train import canonical_ids, atomic_json
from .config import TinyConfig
from .model import build, counts
from .losses import compute

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
NO_DECAY_BY_NAME = {'X8': {'head.codebook'}, 'X4': {'head.codebook'}}


def optimizers(arm, model, lr):
    exempt = NO_DECAY_BY_NAME.get(arm, set())
    dense = [(n, p) for n, p in model.named_parameters() if n != 'engram.table.weight']
    decay = [p for n, p in dense if p.ndim >= 2 and n not in exempt]
    no_decay = [p for n, p in dense if p.ndim < 2 or n in exempt]
    adam = torch.optim.AdamW([dict(params=decay, weight_decay=.1), dict(params=no_decay, weight_decay=0.)],
                             lr=lr, betas=(.9, .95), foreach=False)
    table = torch.optim.SparseAdam([model.engram.table.weight], lr=lr * 5, betas=(.9, .99))
    return adam, table, [p for _, p in dense]


def source_hash():
    h = hashlib.sha256()
    for p in sorted(ROOT.glob('*.py')) + [ROOT.parent / f for f in ('v10_10M/model.py', 'v10_1/model.py', 'v10_2/model.py', 'v10_10M/data.py')]:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


@torch.no_grad()
def quick_eval(model, arm, device, panel):
    was = model.training; model.eval(); out = {}
    for fam, batches in panel.items():
        num = den = 0.
        for b in batches:
            bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            o = model(bb['ids'], bb['lengths'])
            n = float(bb['target_mask'].sum())
            ce = torch.nn.functional.cross_entropy(o['logits'].flatten(0, 1), bb['targets'].masked_fill(~bb['target_mask'], -100).flatten(),
                                                   ignore_index=-100, reduction='sum')
            num += float(ce); den += n
        out[fam] = num / max(den, 1)
    model.train(was); return out


def main(a):
    run = Path(a.run_dir); run.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    tok = Tokenizer.from_file(str(DATA / 'tokenizer.json'))
    model = build(a.arm)
    model.engram.canonical.copy_(canonical_ids(tok, model.c.vocab_size)); model.place(a.device).train()
    adam, table, dense = optimizers(a.arm, model, a.lr)
    corpus = Corpus('train', seed=a.seed, data_dir=DATA)
    dev = Corpus('dev', seed=4242, data_dir=DATA)
    panel = {f: [dev.batch(8, a.context, family=f) for _ in range(n)] for f, n in
             dict(text=4, math=4, sudoku_easy=2, sudoku_hard=2).items()}
    manifest = dict(arm=a.arm, config=model.c.to_dict(), parameters=counts(model), args=vars(a), source_sha256=source_hash(),
                    data_manifest_sha256=hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest(),
                    torch=torch.__version__, start_unix=time.time(),
                    optimizer_rule='V10.1 rule' if a.arm in ('A', 'C0') else 'V10.1 rule with head.codebook exempt from decay')
    atomic_json(run / 'manifest.json', manifest)
    step = tokens = 0; fam_tokens = Counter(); start = time.monotonic(); budget = a.minutes * 60
    last_eval = start; evals = []
    log = (run / 'train.jsonl').open('w', buffering=1)
    def save(tag):
        torch.save(dict(arm=a.arm, config=model.c.to_dict(), model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        step=step, tokens=tokens, family_tokens=dict(fam_tokens), elapsed=time.monotonic() - start,
                        source_sha256=manifest['source_sha256']), run / f'{tag}.pt')
    ev = quick_eval(model, a.arm, a.device, panel); evals.append(dict(step=0, elapsed=0., nll=ev))
    # Progress is step-based so every arm with the same seed sees the identical
    # family/document stream and LR schedule (equal-update comparison); the wall
    # budget is only a safety stop.
    while step < a.max_steps and time.monotonic() - start < budget:
        tick = time.monotonic(); progress = step / a.max_steps
        scale = min(1., (step + 1) / 200) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
        for g in adam.param_groups: g['lr'] = a.lr * scale
        for g in table.param_groups: g['lr'] = a.lr * 5 * scale
        raw = corpus.batch(a.batch_size, a.context, progress=progress)
        b = {k: raw[k].to(a.device) for k in ('ids', 'lengths', 'targets', 'target_mask')}
        candidate = a.arm != 'C0' and step % a.candidate_every == 0
        adam.zero_grad(set_to_none=True); table.zero_grad(set_to_none=True)
        out = model(b['ids'], b['lengths'])
        parts = compute(a.arm, model, out, b, candidate, torch.Generator().manual_seed(a.seed * 100003 + step))
        parts['total'].backward()
        dense_norm = float(torch.nn.utils.clip_grad_norm_(dense, 1.0))
        tg = model.engram.table.weight.grad
        sparse_norm = 0.
        if tg is not None:
            tg = tg.coalesce(); sparse_norm = float(tg.values().norm())
            if sparse_norm > 1.: tg.values().mul_(1. / sparse_norm)
            model.engram.table.weight.grad = tg
            sparse_post = float(model.engram.table.weight.grad.coalesce().values().norm())
        adam.step(); table.step(); step += 1
        n_in = int(raw['lengths'].sum()); tokens += n_in; fam_tokens[raw['family']] += int(raw['target_mask'].sum())
        seconds = time.monotonic() - tick
        rec = dict(step=step, family=raw['family'], elapsed=time.monotonic() - start, step_seconds=seconds,
                   tokens_per_second=n_in / seconds, lr=a.lr * scale, dense_grad_norm=dense_norm, engram_grad_norm=sparse_norm,
                   candidate=candidate, engram_grad_norm_post=sparse_post if tg is not None else 0.)
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                if hasattr(model, 'head'): rec['codebook_rms'] = float(model.head.codebook.square().mean().sqrt())
                if getattr(model.decoder, 'fusion', None) is not None: rec['fusion_rms'] = float(model.decoder.fusion.square().mean().sqrt())
                if hasattr(model.decoder, 'fusion_map'): rec['fusion_map_rms'] = float(model.decoder.fusion_map.weight.square().mean().sqrt())
                if hasattr(model, 'core'): rec['code_projection_rms'] = float(model.core.code_projection.weight.square().mean().sqrt())
        for k, v in parts.items():
            if torch.is_tensor(v) and v.numel() == 1: rec[k] = float(v)
            elif isinstance(v, (int, float, bool)): rec[k] = v
            elif k == 'winners': rec[k] = v
        if not all(math.isfinite(rec[k]) for k in ('total',)): raise FloatingPointError('nonfinite loss at step %d' % step)
        if a.device == 'mps' and step % 50 == 0:
            rec['mps_driver_gb'] = torch.mps.driver_allocated_memory() / 2 ** 30
        log.write(json.dumps(rec) + '\n')
        if time.monotonic() - last_eval >= a.eval_seconds:
            ev = quick_eval(model, a.arm, a.device, panel); evals.append(dict(step=step, elapsed=time.monotonic() - start, nll=ev))
            atomic_json(run / 'quick_eval.json', evals); last_eval = time.monotonic(); save('latest')
            print(json.dumps(dict(step=step, tokens=tokens, **{f'dev_{k}': round(v, 4) for k, v in ev.items()})), flush=True)
    ev = quick_eval(model, a.arm, a.device, panel); evals.append(dict(step=step, elapsed=time.monotonic() - start, nll=ev))
    atomic_json(run / 'quick_eval.json', evals); save('final')
    manifest.update(end_unix=time.time(), steps=step, tokens=tokens, family_supervised_tokens=dict(fam_tokens),
                    active_seconds=time.monotonic() - start)
    atomic_json(run / 'manifest.json', manifest)
    print('DONE', a.arm, step, tokens, flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['C0', 'A', 'X8', 'X4'], required=True); p.add_argument('--run-dir', required=True)
    p.add_argument('--minutes', type=float, default=120); p.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    p.add_argument('--seed', type=int, default=1); p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--context', type=int, default=256); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--candidate-every', type=int, default=4); p.add_argument('--threads', type=int, default=4)
    p.add_argument('--eval-seconds', type=float, default=300); p.add_argument('--max-steps', type=int, required=True)
    return p


if __name__ == '__main__':
    main(parser().parse_args())
