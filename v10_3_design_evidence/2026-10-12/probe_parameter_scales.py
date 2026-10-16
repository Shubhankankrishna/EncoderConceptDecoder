"""Read-only: RMS of scale-bearing V10.1 parameters at two checkpoints, and
whether each sits in the weight-decayed optimizer group (v10_10M/train.py
build_optimizers decays every dense parameter with ndim >= 2). Also records the
entropy of the hindsight-winner distribution from probe_candidate_utility.json,
the best constant prior for the selector. Output refuses to overwrite.
"""
from pathlib import Path
import hashlib, json, math
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'probe_parameter_scales.json'
RUN = ROOT / 'v10_1/runs/pretrain_16h'
NAMES = ('head.codebook', 'core.codes.weight', 'core.code_projection.weight', 'decoder.fusion',
         'decoder.concept_projection.weight', 'pooler.outn.weight', 'core.exit_norm.weight')


def main():
    if OUT.exists(): raise SystemExit('Refusing to overwrite ' + str(OUT))
    result = dict(decay_rule='weight_decay 0.1 if ndim >= 2 else 0 (v10_10M/train.py build_optimizers)', checkpoints={})
    for f in ('before_throughput_step_0003171.pt', 'stopped_step_0061384.pt'):
        ck = torch.load(RUN / f, map_location='cpu', weights_only=False)
        result['checkpoints'][str(ck['step'])] = {
            n: dict(rms=float(ck['model'][n].double().square().mean().sqrt()), shape=list(ck['model'][n].shape),
                    decayed=ck['model'][n].ndim >= 2) for n in NAMES}
    util = json.loads((HERE / 'probe_candidate_utility.json').read_text())
    result['winner_entropy_vs_selector_ce'] = {}
    for fam, e in util['families'].items():
        w = e['winner_counts']; n = sum(w)
        result['winner_entropy_vs_selector_ce'][fam] = dict(
            winner_entropy=-sum(c / n * math.log(c / n) for c in w if c), selector_ce=e['selector_winner_ce'])
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    OUT.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
