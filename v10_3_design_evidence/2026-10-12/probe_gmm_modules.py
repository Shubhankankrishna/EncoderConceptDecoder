"""Read-only probe: do GMM gates separate phases within one data family?

probe_gmm_gates.py treated whole families (text, math, Sudoku) as phases. A
realistic continual-learning phase is narrower. Here each of the four math
modules is a phase. For each held-out module, Phi_pos is fitted on half of its
documents and Phi_neg on a reference covering everything else the model trained
on (the other three modules, text and both Sudoku families). Reported: hit rate
on the module's held-out documents and open rate on the other modules' held-out
documents, raw per row (no smoothing; see the companion probe). Sites: E2 and
concept-core attention/FFN inputs, per visit, plus decoder inputs. Reuses the
companion's GMM settings. Dev split, frozen checkpoint; output refuses to overwrite.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture
from v10_1.prepare_data import DATA
from probe_gmm_gates import fit, split_docs

OUT = HERE / 'probe_gmm_modules.json'
CKPT = ROOT / 'v10_1/runs/pretrain_16h/stopped_step_0061384.pt'
MATH_DOCS, OTHER_DOCS = 600, dict(text=32, sudoku_easy=16, sudoku_hard=16)


def run(model, batch, store, label):
    calls, current = {}, {}
    def hook(name):
        def fn(module, inputs):
            n = calls.get(name, 0); calls[name] = n + 1
            key = f'{name}.v{n + 1}' if name.split('.')[0] in ('e2', 'core') else name
            current.setdefault(key, inputs[0][0].detach().double().numpy())
        return fn
    handles = []
    for prefix, blocks in (('e2', model.e2), ('core', model.core.layers), ('dec', model.decoder.layers)):
        for i, blk in enumerate(blocks):
            handles += [blk.attn.register_forward_pre_hook(hook(f'{prefix}.{i}.attn')),
                        blk.ffn.register_forward_pre_hook(hook(f'{prefix}.{i}.ffn'))]
    with torch.no_grad():
        out = model(batch['ids'], batch['lengths'])
    for h in handles: h.remove()
    length, groups = int(batch['lengths'][0]), int(out['valid_groups'][0].sum())
    for key, x in current.items():
        x = x[:2 * groups] if key.startswith('core') else x[:length]
        if len(x): store.setdefault(key, {}).setdefault(label, []).append(x)


def main():
    if OUT.exists(): raise SystemExit('Refusing to overwrite ' + str(OUT))
    torch.set_num_threads(2); started = time.time(); rng = np.random.default_rng(29)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = V101(ModelConfig(**ck['model_config']), Architecture(**ck['architecture']))
    model.load_state_dict(ck['model']); model.place('cpu').eval(); step = int(ck['step']); del ck
    meta = [json.loads(x) for x in (Path(DATA) / 'math_dev_meta.jsonl').read_text().splitlines()]
    corpus = Corpus('dev', seed=20261015, data_dir=DATA)
    store = {}
    for _ in range(MATH_DOCS):
        b = corpus.batch(1, family='math'); run(model, b, store, meta[b['chosen'][0]]['source'])
    for fam, n in OTHER_DOCS.items():
        for _ in range(n): run(model, corpus.batch(1, family=fam), store, fam)
    modules = sorted({r['source'] for r in meta})
    result = dict(checkpoint=str(CKPT.relative_to(ROOT)), step=step, modules=modules,
                  documents={m: len(next(iter(store.values()))[m]) for m in modules + list(OTHER_DOCS)},
                  sites={})
    for site, by in store.items():
        halves = {k: split_docs(v, rng) for k, v in by.items()}
        res = {}
        for new in modules:
            neg = fit([x for k in halves if k != new for x in halves[k][0]], rng); pos = fit(halves[new][0], rng)
            rate = lambda docs: float(np.concatenate([(pos.score_samples(x) > neg.score_samples(x)) for x in docs]).mean())
            res[new] = dict(hit_heldout_new=rate(halves[new][1]),
                            **{f'open_{m}': rate(halves[m][1]) for m in modules if m != new},
                            open_text=rate(halves['text'][1]))
        result['sites'][site] = res
    result['seconds'] = time.time() - started
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    for site, res in result['sites'].items():
        print(site, {m[:12]: {k[:16]: round(v, 2) for k, v in e.items()} for m, e in res.items()})


if __name__ == '__main__':
    main()
