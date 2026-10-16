"""cl_test with a per-cell wall-time cap (proposed DEVIATIONS item 36; run only with the user's approval).

    taskpolicy -b python3 build_cl_test_capped.py --cap 300 --workers 4

Same cells, same seeds and the same sampling loop as evaluate.build_panel / _gen_cell, so every item produced is
identical to what the uncapped build would produce for that cell; a cell stops early only when its wall-time cap is
reached, and keeps the items it has. Writes data/panels/cl_test.pkl (the format cl_eval.py reads) and
data/panels/cl_test_caps.json (items per cell, capped or full). Progress is checkpointed after every cell."""
import argparse, json, os, pickle, sys, time
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import data as D, evaluate as E, tasks as T


def gen_capped(args):
    task, rung, n, mode, phase, want, seed, cap = args
    rng = np.random.default_rng(D.seed_of('panel', seed, task, rung, mode, phase, want))
    out, seen, tries, t0 = [], set(), 0, time.time()
    while len(out) < n and tries < n * 30 and time.time() - t0 < cap:
        tries += 1
        e = T.generate(task, rung, rng, mode=mode, phase=phase, want=want)
        if e is None or e['key'] in seen: continue
        seen.add(e['key']); out.append(e)
    return (task, rung, mode, phase, want), out, dict(n=len(out), tries=tries, s=round(time.time() - t0, 1),
                                                     capped=len(out) < n and tries < n * 30)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cap', type=float, default=300); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    P = E.PANELS; part = P / 'cl_test_partial.pkl'
    res, info = (pickle.loads(part.read_bytes()) if part.exists() else ({}, {}))
    cells = [c for c in E.cl_cells(24, 'test') if str((c[0], c[1], c[3], c[4], c[5])) not in info]
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for key, items, st in pool.imap_unordered(gen_capped, [c + (13, a.cap) for c in cells], chunksize=1):
            res[key] = items; info[str(key)] = st
            part.write_bytes(pickle.dumps((res, info)))
            print(f'{len(info)} cells  {key}  {st}', flush=True)
    (P / 'cl_test.pkl').write_bytes(pickle.dumps(res))
    (P / 'cl_test_caps.json').write_text(json.dumps(info, indent=1))
    full = sum(1 for v in info.values() if v['n'] == 24)
    print(f'cl_test (capped at {a.cap}s/cell): {sum(len(v) for v in res.values())} items, {full}/{len(info)} cells full, '
          f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
