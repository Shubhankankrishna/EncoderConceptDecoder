"""Solo Metal throughput benchmark of the 10M reference shapes (v11.md §6.2). Nothing is kept.

  python -B -m v11_tiny.bench_10m            (writes v11_tiny/bench_10m.json; run with the GPU otherwise idle)

Reference V11 (K1 objective, K = 1) and its C0 at width 128, vocab 8,192, E1/E2/E3 4/4/2, core 4,
decoder 4, PQ 8x32, Engram 4 x ~156k x 32 on CPU; context 384, batch 8, S = 3; data v10_1/data
(read only, train split). Per arm: 200 timed updates after 20 warm-up updates (unsynchronized, the
production loop), then 40 updates with torch.mps.synchronize() between phases for a time split.
"""
import json, math, statistics as st, time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from v10_10M.data import Corpus
from v10_10M.train import canonical_ids
from v10_3_tiny.config import TinyConfig
from v10_3_tiny.model import C0
from v10_3_tiny.losses import compute as compute_v103
from .model import V11Tiny, counts
from .losses import compute as compute_v11
from .train import optimizers

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'v10_1' / 'data'
REF = dict(vocab_size=8192, width=128, e1_layers=4, e2_layers=4, e3_layers=2, concept_layers=4, decoder_layers=4,
           code_width=32, gate_rank=32, pq_segments=8, pq_words=32, engram_rows=(156241, 156253, 156257, 156259),
           engram_dim=32, context=384)
B, T, S, WARM, TIMED, SPLIT = 8, 384, 3, 20, 200, 40
DEVICE = 'mps'


def sync_dev():
    if DEVICE == 'mps': torch.mps.synchronize()


def build(arm):
    c = TinyConfig(**REF)
    return C0(c) if arm == 'C0' else V11Tiny(arm, c)


def bench(arm):
    torch.manual_seed(1); np.random.seed(1)
    tok = Tokenizer.from_file(str(DATA / 'tokenizer.json'))
    model = build(arm); model.engram.canonical.copy_(canonical_ids(tok, model.c.vocab_size)); model.place(DEVICE).train()
    adam, table, dense = optimizers(arm if arm != 'C0' else 'C0', model, 3e-4, foreach=True)
    corpus = Corpus('train', seed=1, data_dir=DATA)
    raws = [corpus.batch(B, T, progress=0) for _ in range(WARM + TIMED + SPLIT)]
    idx = [model.engram.indices(r['ids']) for r in raws]
    rec, mem = [], 0.
    def one(i, sync):
        t = {}; t0 = time.perf_counter()
        r = raws[i]; b = {k: r[k].to(DEVICE) for k in ('ids', 'lengths', 'targets', 'target_mask')}
        adam.zero_grad(set_to_none=True); table.zero_grad(set_to_none=True)
        out = model(b['ids'], b['lengths'], idx[i])
        g = torch.Generator().manual_seed(i)
        parts = compute_v103('C0', model, out, b, False, g) if arm == 'C0' else compute_v11(arm, model, out, b, 500, 200, g, S)
        if sync: sync_dev(); t['forward_loss'] = time.perf_counter() - t0; t1 = time.perf_counter()
        parts['total'].backward()
        if sync: sync_dev(); t['backward'] = time.perf_counter() - t1; t1 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(dense, 1.0)
        tg = model.engram.table.weight.grad
        if tg is not None:
            tg = tg.coalesce(); nrm = float(tg.values().norm())
            if nrm > 1.: tg.values().mul_(1. / nrm)
            model.engram.table.weight.grad = tg
        adam.step(); table.step()
        if arm != 'C0': model.update_target_buffers(out['pooled'], out['valid_groups'])
        float(parts['total'].detach())
        if sync: sync_dev(); t['optimizer'] = time.perf_counter() - t1
        t['step'] = time.perf_counter() - t0
        t['input_tokens'] = int(r['lengths'].sum()); t['scored_tokens'] = int(r['target_mask'].sum()); t['family'] = r['family']
        return t
    for i in range(WARM): one(i, False)
    t_start = time.perf_counter()
    for i in range(WARM, WARM + TIMED):
        rec.append(one(i, False)); mem = max(mem, torch.mps.driver_allocated_memory() / 2 ** 30 if DEVICE == 'mps' else 0.)
    wall = time.perf_counter() - t_start
    split = [one(i, True) for i in range(WARM + TIMED, WARM + TIMED + SPLIT)]
    # Engram host work: hashing (done by the prefetch thread in training) and the CPU gather.
    t0 = time.perf_counter()
    for r in raws[:40]: model.engram.indices(r['ids'])
    hash_s = (time.perf_counter() - t0) / 40
    steps = [x['step'] for x in rec]
    text = [x['step'] for x in rec if x['family'] == 'text']
    res = dict(arm=arm, parameters=counts(model), timed_updates=TIMED, wall_seconds=wall,
               step_p50=st.median(steps), step_p95=float(np.percentile(steps, 95)), text_step_p50=st.median(text) if text else None,
               input_tokens_per_second=sum(x['input_tokens'] for x in rec) / wall,
               scored_tokens_per_second=sum(x['scored_tokens'] for x in rec) / wall,
               first50_vs_last50=(st.median(steps[:50]), st.median(steps[-50:])), peak_driver_gb=mem,
               split_medians={k: st.median(x[k] for x in split) for k in ('forward_loss', 'backward', 'optimizer', 'step')},
               engram_hash_seconds_per_batch=hash_s, families=dict(zip(*np.unique([x['family'] for x in rec], return_counts=True))))
    res['families'] = {k: int(v) for k, v in res['families'].items()}
    hours16 = res['input_tokens_per_second'] * 16 * 3600
    res['tokens_in_16h_at_this_rate'] = hours16; res['tokens_per_deployed_param_16h'] = hours16 / res['parameters']['dense_deployed']
    del model, adam, table
    if DEVICE == 'mps': torch.mps.empty_cache()
    return res


if __name__ == '__main__':
    import sys
    if '--smoke' in sys.argv:
        DEVICE, WARM, TIMED, SPLIT = 'cpu', 1, 2, 1
    out = dict(prior_v10_10M_tokens_per_second=3855, batch=B, context=T, S=S, results=[])
    target = ROOT / ('bench_10m_smoke.json' if '--smoke' in sys.argv else 'bench_10m.json')
    for arm in ('C0', 'K1', 'NOAUX', 'K1NB'):
        r = bench(arm); out['results'].append(r)
        print(json.dumps({k: r[k] for k in ('arm', 'step_p50', 'step_p95', 'input_tokens_per_second', 'peak_driver_gb', 'split_medians', 'first50_vs_last50')}), flush=True)
        target.write_text(json.dumps(out, indent=1, default=float) + '\n')
