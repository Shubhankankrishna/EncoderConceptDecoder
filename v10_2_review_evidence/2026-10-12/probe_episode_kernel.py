"""Read-only companion probe: kernel spectrum of single adaptation episodes.

The V10.2 protocol (§6.3) commits three support examples per episode. A write
changes that episode's replayed corrections by -eta*K_episode*G, so the
finite-step amplification is lambda_max of the episode's own code-partitioned
Gram matrix. This samples 3-document episodes from the same frozen V10.1
checkpoint and dev panel as the companion probes. Output refuses to overwrite.
"""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture
from v10_1.prepare_data import DATA

HERE = Path(__file__).resolve().parent
OUT = HERE / 'probe_episode_kernel.json'
CKPT = ROOT / 'v10_1/runs/pretrain_16h/stopped_step_0061384.pt'
LARGE_N = dict(text=64, math=64, sudoku_easy=32, sudoku_hard=32)
EPISODES, SUPPORT = 200, 3


@torch.no_grad()
def main():
    if OUT.exists(): raise SystemExit('Refusing to overwrite ' + str(OUT))
    torch.set_num_threads(2); started = time.time(); rng = np.random.default_rng(11)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = V101(ModelConfig(**ck['model_config']), Architecture(**ck['architecture']))
    model.load_state_dict(ck['model']); model.place('cpu').eval(); del ck
    corpus = Corpus('dev', seed=20261012, data_dir=DATA)
    panel = {f: [corpus.batch(1, family=f) for _ in range(LARGE_N[f])] for f in corpus.arrays}
    result = dict(checkpoint=str(CKPT.relative_to(ROOT)), episodes_per_family=EPISODES, support_documents=SUPPORT,
                  scope='Same-code Gram of the queries in one 3-document episode (largest code partition).', families={})
    for fam, batches in panel.items():
        docs = []
        for b in batches:
            out = model(b['ids'], b['lengths']); v = out['valid_groups']
            docs.append((out['queries'][v].double(), out['codes'][v]))
        stats = {str(s): [] for s in (0.25, 0.5, 1.0)}; sizes = []
        for _ in range(EPISODES):
            pick = rng.choice(len(docs), SUPPORT, replace=False)
            q = torch.cat([docs[i][0] for i in pick]); k = torch.cat([docs[i][1] for i in pick])
            sizes.append(len(q)); d2 = torch.cdist(q, q).square()
            for s in stats:
                best = 0.
                for code in k.unique():
                    sel = k == code
                    if int(sel.sum()) == 0: continue
                    kern = torch.exp(-d2[sel][:, sel] / (2 * float(s) ** 2))
                    best = max(best, float(torch.linalg.eigvalsh(kern)[-1]))
                stats[s].append(best)
        result['families'][fam] = dict(
            median_episode_entries=float(np.median(sizes)),
            lambda_max_median_p95={s: [float(np.median(v)), float(np.quantile(v, .95))] for s, v in stats.items()})
    result['seconds'] = time.time() - started
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['families'], indent=1))


if __name__ == '__main__':
    main()
