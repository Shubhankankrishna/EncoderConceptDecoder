"""CPU check of the gradient-share read's memory growth (V14 DEVIATIONS item 18). Runs train.grad_share three times on
a real snapshot with the cyclic garbage collector disabled, printing the process RSS after each call, then one
gc.collect() (DEBUG_SAVEALL, so the unreachable objects can be counted by type) and the RSS after it.

    taskpolicy -b python3 -u runs/plumbing/gs_leak_check.py
"""
import collections
import gc
import os
import subprocess
import sys
from pathlib import Path

V14_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V14_DIR)); os.chdir(V14_DIR)
import torch

import model as M
import tok as TK
import train as TR
import v14 as V


def rss_mb():
    return int(subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], capture_output=True, text=True).stdout) / 1024


def main():
    torch.set_num_threads(2)
    dev = torch.device('cpu'); tk = TK.Tok()
    z = torch.load('runs/base/snap_010.pt', map_location='cpu', weights_only=False)
    model = M.V14(M.Cfg(V=tk.vocab)); model.load_state_dict(z['model']); model.set_ckpt(True); model.train()
    twin = V.make_twin(model)
    twin.load_state_dict({k: v for k, v in z['twin'].items() if not k.startswith('engram.table')}, strict=False)
    cb = V.Codebook(model.c.d, model.c.pq_S, model.c.pq_N); cb.load_state_dict(z['codebook'])
    maps = V.ResidueMaps(len(V.FAMILIES), 2 * model.c.d, model.c.d); maps.load_state_dict(z['maps'])
    consts = V.sigreg_consts(dev)
    hb, hfam = TR.held_batch(tk, 4)
    gc.collect(); gc.disable()
    print(f'start rss {rss_mb():.0f} MB', flush=True)
    for i in range(3):
        TR.grad_share(model, twin, cb, maps, hb, hfam, dev, 0.01, consts, 1500 + 500 * i)
        print(f'after call {i + 1}: rss {rss_mb():.0f} MB (gc disabled)', flush=True)
    gc.set_debug(gc.DEBUG_SAVEALL)
    found = gc.collect()
    kinds = collections.Counter(type(o).__name__ for o in gc.garbage)
    tens = [o for o in gc.garbage if torch.is_tensor(o)]
    print(f'gc.collect: {found} unreachable objects; tensors among them {len(tens)} '
          f'({sum(t.numel() * t.element_size() for t in tens) / 2 ** 20:.0f} MB); top types {kinds.most_common(8)}', flush=True)
    gc.set_debug(0); gc.garbage.clear(); del tens; gc.collect()
    print(f'after collect: rss {rss_mb():.0f} MB', flush=True)


if __name__ == '__main__':
    main()
