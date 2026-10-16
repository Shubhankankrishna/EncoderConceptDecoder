"""Scratch-only full evaluation to diagnose recurrent MPS evaluation batching.

Does not edit locked code or write production results. A singleton guard prevents
multiple probes from competing for MPS. Progress and MPS memory are durable.
"""
import argparse
import datetime
import fcntl
import functools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

V9 = Path(__file__).resolve().parents[1]
MON = V9 / 'monitor'
sys.path.insert(0, str(V9 / 'code'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()
    assert args.batch in (8, 16, 32, 64)
    with (MON / 'i22_probe.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        st = json.loads((V9 / 'runs/status_Q3.json').read_text())
        assert st['status'] == 'stopped_by_user', st
        os.environ['V8_DEVICE'] = 'mps'
        import measure_s2 as M
        import torch
        torch.set_num_threads(2)
        scratch = Path(tempfile.mkdtemp(prefix='v9_i22_memory_probe_'))
        state = dict(status='running', pid=os.getpid(), batch=args.batch,
                     scratch=str(scratch), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     events=[], purpose='scratch validation only; never copied to production results')
        start = time.time()

        def record(label, **extra):
            torch.mps.synchronize()
            e = dict(stage=label, elapsed_s=round(time.time() - start, 2),
                     driver_gb=round(torch.mps.driver_allocated_memory() / 1e9, 3),
                     tensor_gb=round(torch.mps.current_allocated_memory() / 1e9, 3), **extra)
            state['events'].append(e)
            state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            tmp = MON / 'i22_probe.json.tmp'
            tmp.write_text(json.dumps(state, indent=2))
            os.replace(tmp, MON / 'i22_probe.json')
            print(json.dumps(e), flush=True)

        old_batch = M.eval_batch
        M.eval_batch = lambda eng: args.batch if eng.hs.core_kind == 'recurrent' else old_batch(eng)
        for name in ('nll_suite', 'greedy_suite', 'self_states', 'gold_states',
                     'wrong_states', 'decode_codes', 'candidate_concepts', 'g23',
                     'drift_suite', 'engaged_depth'):
            original = getattr(M, name)

            def wrap(fn, name):
                @functools.wraps(fn)
                def call(*a, **kw):
                    detail = str(a[1]) if name == 'greedy_suite' else ''
                    record(name + ':start', detail=detail)
                    result = fn(*a, **kw)
                    record(name + ':done', detail=detail)
                    return result
                return call

            setattr(M, name, wrap(original, name))
        try:
            record('start')
            result = M.measure(str(V9 / 'runs/S2_K1_rec_s911/checkpoint.pt'),
                               scratch / 'measure.json',
                               str(V9 / 'runs/S1_A_delayed_s902/checkpoint.pt'))
            state.update(status='completed', seconds=result['seconds'],
                         determinism=result['determinism'], output=str(scratch / 'measure.json'))
            record('finished')
        except Exception:
            state.update(status='failed', error=traceback.format_exc())
            record('failed')
            raise


if __name__ == '__main__':
    main()
