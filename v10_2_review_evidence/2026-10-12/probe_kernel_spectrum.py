"""Read-only companion probe: spectrum of the code-partitioned RBF kernel.

Uses the same frozen V10.1 checkpoint and the same 64/64/32/32 dev panel as
probe_v10_1_interfaces.py (seed 20261012). For each family and sigma it builds
the kernel Gram matrix of all panel queries within each code partition and
reports its largest eigenvalue. A gradient write with step eta changes the
replayed corrections by -eta*K*G, so finite-step stability scales with
eta*lambda_max(K). Output refuses to overwrite.
"""
from pathlib import Path
import hashlib, json, sys, time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v10_10M.config import ModelConfig
from v10_10M.data import Corpus
from v10_1.model import V101, Architecture
from v10_1.prepare_data import DATA

HERE = Path(__file__).resolve().parent
OUT = HERE / 'probe_kernel_spectrum.json'
CKPT = ROOT / 'v10_1/runs/pretrain_16h/stopped_step_0061384.pt'
LARGE_N = dict(text=64, math=64, sudoku_easy=32, sudoku_hard=32)


@torch.no_grad()
def main():
    if OUT.exists(): raise SystemExit('Refusing to overwrite ' + str(OUT))
    torch.set_num_threads(2); started = time.time()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = V101(ModelConfig(**ck['model_config']), Architecture(**ck['architecture']))
    model.load_state_dict(ck['model']); model.place('cpu').eval(); del ck
    corpus = Corpus('dev', seed=20261012, data_dir=DATA)
    panel = {f: [corpus.batch(1, family=f) for _ in range(LARGE_N[f])] for f in corpus.arrays}
    result = dict(checkpoint=str(CKPT.relative_to(ROOT)), families={},
                  scope='Kernel of dev-panel queries as a proxy for a bank written from the same distribution; '
                        'includes same-document pairs, unlike the cross-document quantiles in the companion probe.')
    for fam, batches in panel.items():
        qs, ks = [], []
        for b in batches:
            out = model(b['ids'], b['lengths']); v = out['valid_groups']
            qs.append(out['queries'][v].double()); ks.append(out['codes'][v])
        q, k = torch.cat(qs), torch.cat(ks); d2 = torch.cdist(q, q).square()
        fam_out = {}
        for sigma in (0.25, 0.5, 1.0):
            per_code = {}
            for code in k.unique().tolist():
                sel = k == code
                if int(sel.sum()) < 2: continue
                kern = torch.exp(-d2[sel][:, sel] / (2 * sigma ** 2))
                per_code[str(code)] = dict(entries=int(sel.sum()), lambda_max=float(torch.linalg.eigvalsh(kern)[-1]),
                                           mean_row_sum=float(kern.sum(1).mean()))
            worst = max(per_code.values(), key=lambda x: x['lambda_max'])
            fam_out[str(sigma)] = dict(largest_partition_lambda_max=worst['lambda_max'], entries_in_that_partition=worst['entries'],
                                       per_code=per_code)
        result['families'][fam] = dict(queries=len(q), codes_used=int(k.unique().numel()), sigma=fam_out)
    result['seconds'] = time.time() - started
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    for fam, e in result['families'].items():
        print(fam, e['queries'], {s: (round(v['largest_partition_lambda_max'], 1), v['entries_in_that_partition']) for s, v in e['sigma'].items()})


if __name__ == '__main__':
    main()
