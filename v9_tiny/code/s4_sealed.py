"""Sealed greedy evaluation of one S4 arm (L3 procedure, I18): selector-argmax plans (EI, REINFORCE, reference) or
code-free plans (RFT, GRPO), on the fresh sealed hard10 (512) and hard14 (256) manifests."""
import argparse
import numpy as np
import torch
import v9paths
from v9paths import SMOKE, write_json
import s4_eval, s5_run, baselines

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--mode', required=True); ap.add_argument('--adapter'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); torch.set_num_threads(4); split = 'dev' if SMOKE else 'v9test'
    hs, eng = s4_eval.load_arm(a.source, a.adapter); out = dict(split=split, mode=a.mode, source=a.source, adapter=a.adapter)
    for suite, n in (('hard10', 16 if SMOKE else 512), ('hard14', 8 if SMOKE else 256)):
        items = s5_run.suite_items(split, suite, n); succ = []
        for i in range(0, len(items), 64):
            chunk = items[i:i + 64]
            if a.mode == 'codes':
                g = eng.greedy([it.prompt for it in chunk], max_new=384 - chunk[0].P); succ += [v['success'] for v in g['verification']]
            else:
                succ += [ok for _, _, ok in baselines.sample_answers(eng, chunk, temp=0., code_policy='free', hop_limit=40 if suite == 'hard14' else 20)]
        out[suite] = dict(success=[bool(v) for v in succ], rate=float(np.mean(succ)), n=len(succ))
    write_json(a.out, out)
