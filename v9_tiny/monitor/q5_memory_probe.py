"""Isolated allocation/lifetime probe; never writes production run outputs."""
import gc
import json
import os
from pathlib import Path
import sys
import time

V9 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V9 / 'code'))
import torch
import csearch
import hgen
import hncp
import train_s4

mode = sys.argv[1]
device = 'mps' if mode == 'mps_workers' else mode
torch.set_num_threads(4)
hs, _ = hncp.load(str(V9 / 'runs/S2_K8_dense_s911/checkpoint.pt'), device)
eng = hgen.Engine(hs)
items = train_s4.fresh_problems(512, 178100)[:64 if device == 'mps' else 4]
ps = csearch.problems_from(items, 20, 179100)
search = csearch.Search(eng, dict(H=1, children='grounded', leaf='rollout', budget=32 if device == 'mps' else 2, noise=True))
t0 = time.monotonic()


def sample(label, **extra):
    d = dict(label=label, elapsed_s=time.monotonic() - t0, pid=os.getpid(), **extra)
    if device == 'mps':
        torch.mps.synchronize()
        d.update(tensor_gb=torch.mps.current_allocated_memory()/1e9,
                 driver_gb=torch.mps.driver_allocated_memory()/1e9)
    print(json.dumps(d), flush=True)


if device == 'mps':
    original = eng.decode_chunk
    last = [0.]
    def decode(*args, **kw):
        result = original(*args, **kw)
        if time.monotonic() - last[0] > 30:
            last[0] = time.monotonic()
            sample('decode', batch=len(args[0]), width=args[0].shape[1], stats=search.stats)
        return result
    eng.decode_chunk = decode

gc.collect()
sample('start')
if mode == 'mps_workers':
    ps, stats = train_s4.solve_in_workers(eng, ps, search.cfg, 64, max_steps=2)
    assert len(ps) == 64 and all(p.done for p in ps)
    sample('workers_complete', problems=len(ps), stats=stats,
           all_done=all(p.done for p in ps), verifier_calls=sum(p.verifier_calls for p in ps))
else:
    search.step(ps)
sample('step_returned', stats=search.stats, live_nodes=sum(isinstance(o, csearch.Node) for o in gc.get_objects()))
collected = gc.collect()
sample('after_cycle_collection', collected=collected,
       live_nodes=sum(isinstance(o, csearch.Node) for o in gc.get_objects()))
if device == 'mps':
    torch.mps.empty_cache()
    sample('after_empty_cache')
