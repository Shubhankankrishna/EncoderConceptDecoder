"""v11.md §4.4 control 3: where does each loss's gradient land? Final V11 checkpoints, frozen copies.

  python -B -m v11_tiny.grad_share        (writes v11_tiny/grad_share.json)

For 20 training-split batches (a fixed Corpus seed not used by any run: 777), per-loss gradient L2 norms
on module groups {embedding+Engram+E1/E2/E3 ('encoder'), pooler, E->C CRC + core ('core'), PQ head,
decoder, bridge}, with all losses at their final weights (ramp = 1). Also a frozen-model proxy for the
constant-anchor control: the bridge KL with u1 replaced by its batch mean over valid groups.
"""
import json
from pathlib import Path
import numpy as np
import torch
from v10_10M.data import Corpus
from v10_3_tiny.config import TinyConfig
from .model import build
from .losses import compute, bridge_losses, LAMBDA

ROOT = Path(__file__).resolve().parent; RUNS = ROOT / 'runs'; DATA = ROOT.parent / 'v10_3_tiny' / 'data'
GROUPS = dict(encoder=('embedding.', 'engram.', 'e1.', 'e2.', 'exit_reader.', 'e3.'), pooler=('pooler.',),
              core=('encoder_concept_crc.', 'core.'), head=('head.',), decoder=('decoder.',), bridge=('bridge.',))


def group_norms(model, loss):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and n != 'engram.table.weight']
    g = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
    out = {k: 0. for k in GROUPS}
    for (n, _), gi in zip(named, g):
        if gi is None: continue
        for k, pre in GROUPS.items():
            if n.startswith(pre): out[k] += float(gi.square().sum())
    return {k: v ** .5 for k, v in out.items()}


def main():
    corpus = Corpus('train', seed=777, data_dir=DATA)
    batches = [corpus.batch(16, 256, progress=.5) for _ in range(20)]
    result = {}
    for run in sorted(RUNS.glob('K1*_s[0-9]')) + sorted(RUNS.glob('NOAUX*_s[0-9]')):   # includes the SC variants
        if not (run / 'final.pt').exists(): continue
        ck = torch.load(run / 'final.pt', map_location='cpu', weights_only=False); arm = ck['arm']
        cfg = TinyConfig(**{k: (tuple(v) if k == 'engram_rows' else v) for k, v in ck['config'].items()})
        model = build(arm, cfg); model.load_state_dict(ck['model']); model.train()
        rows, anchor = [], []
        for i, b in enumerate(batches):
            out = model(b['ids'], b['lengths'])
            parts = compute(arm, model, out, b, 10 ** 6, 200, torch.Generator().manual_seed(i), 2)
            row = dict(family=b['family'], ntp=group_norms(model, parts['token']))
            for k, w in (('v1', LAMBDA['c'] / 3), ('v2', 2 * LAMBDA['c'] / 3), ('bridge_h', LAMBDA['h']), ('bridge_kl', LAMBDA['kl'])):
                if parts.get(k) is not None: row[k] = group_norms(model, w * parts[k])
            rows.append(row)
            if arm == 'K1':
                with torch.no_grad():
                    v = out['valid_groups']
                    if int(v.sum()):
                        mean = out['u1'][v].mean(0)
                        _, kl_real, n = bridge_losses(model, out, b['ids'], b['lengths'])
                        out2 = dict(out); out2['u1'] = torch.where(v[..., None, None], mean.expand_as(out['u1']), out['u1'])
                        _, kl_const, _ = bridge_losses(model, out2, b['ids'], b['lengths'])
                        if kl_real is not None: anchor.append(dict(family=b['family'], kl_real=float(kl_real), kl_mean_anchor=float(kl_const)))
        summary = {}
        for loss in ('ntp', 'v1', 'v2', 'bridge_h', 'bridge_kl'):
            vals = [r[loss] for r in rows if loss in r]
            if vals: summary[loss] = {k: float(np.mean([v[k] for v in vals])) for k in GROUPS}
        by_family = {}
        for fam in sorted({a['family'] for a in anchor}):
            sel = [a for a in anchor if a['family'] == fam]
            by_family[fam] = dict(kl_real=float(np.mean([a['kl_real'] for a in sel])), kl_mean_anchor=float(np.mean([a['kl_mean_anchor'] for a in sel])))
        result[run.name] = dict(mean_group_grad_norms=summary, anchor_proxy=by_family or None,
                                batches=[r['family'] for r in rows])
        print(run.name, json.dumps(summary)[:600], json.dumps(by_family), flush=True)
    (ROOT / 'grad_share.json').write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    torch.set_num_threads(4); main()
