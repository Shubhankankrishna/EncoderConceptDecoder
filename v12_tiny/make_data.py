"""Generate the pinpad splits once (seeds disjoint per split; the test split is sealed and never read here).

  python -B make_data.py            -> data/{train,meta,dev,dev_meta,post_dev,test}.npz
train     : pretraining tasks (Table A1), expert eps = 0.05 (paper Fig. A2: mild suboptimality helps the base)
meta      : pretraining tasks, eps = 0 (paper C.4.4: clean demos for metacontroller discovery)
dev       : pretraining tasks, eps = 0.05, for NLL / probes
dev_meta  : pretraining tasks, eps = 0, for metacontroller evaluation (switch F1, rate–distortion)
post_dev  : the post-training task, eps = 0 (diagnostic only; never trained on)
test      : sealed (pretraining tasks + post task); written, never opened by any script in this package
"""
import sys, time
from multiprocessing import Pool
from pathlib import Path
import pinpad as P

OUT = Path(__file__).resolve().parent / 'data'
SPLITS = dict(train=(500_000, 11, 0.05, P.PRETRAIN_TASKS), meta=(60_000, 22, 0.0, P.PRETRAIN_TASKS),
              dev=(4_000, 33, 0.05, P.PRETRAIN_TASKS), dev_meta=(4_000, 44, 0.0, P.PRETRAIN_TASKS),
              post_dev=(1_000, 55, 0.0, [P.POST_TASK]), test=(4_000, 99, 0.0, P.PRETRAIN_TASKS + [P.POST_TASK]))


def job(args):
    n, seed, eps, tasks = args
    return P.generate(n, seed, eps, tasks)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    for name, (n, seed, eps, tasks) in SPLITS.items():
        t = time.time(); chunks = 8 if n >= 8000 else 1
        with Pool(4) as pool:
            parts = pool.map(job, [(n // chunks, seed * 1000 + i, eps, tasks) for i in range(chunks)])
        eps_list = [e for p in parts for e in p]
        P.pack(eps_list, OUT / f'{name}.npz')
        print(name, len(eps_list), f'{time.time() - t:.0f}s', flush=True)
