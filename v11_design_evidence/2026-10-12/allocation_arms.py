"""Allocation arms and the 120/60/20 stress case, by instantiation plus analytic MACs.

  python -B v11_design_evidence/2026-10-12/allocation_arms.py   (from the repo root)

Uses summarize() and macs_per_token() from param_budget.py. Writes allocation_arms.json.
"""
import contextlib, io, json, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from param_budget import summarize, macs_per_token  # noqa: E402
from v10_3_tiny.config import TinyConfig  # noqa: E402  (same attribute names; validate() allows <4 decoder blocks)

OUT = Path(__file__).with_name('allocation_arms.json')
# Reference dimensions on the relaxed config class; the reference row must reproduce param_budget.json exactly.
BASE = TinyConfig(vocab_size=8192, width=128, e1_layers=4, e2_layers=4, e3_layers=2, concept_layers=4, decoder_layers=4,
                  code_width=32, gate_rank=32, pq_segments=8, pq_words=32,
                  engram_rows=(156241, 156253, 156257, 156259), engram_dim=32, context=512)
ARMS = {
    'reference_balanced_4/4/2_c4_d4': BASE,
    'E3_4_historical_4/4/4_c4_d4': replace(BASE, e3_layers=4),
    'decoder_8_4/4/2_c4_d8': replace(BASE, decoder_layers=8),
    'concept_heavy_4/4/2_c8_d4': replace(BASE, concept_layers=8),
    'stress_ratio_param_matched_4/4/3_c5_d2': replace(BASE, e1_layers=4, e2_layers=4, e3_layers=3, concept_layers=5, decoder_layers=2),
    'stress_ratio_at_cap_6/6/3_c7_d2': replace(BASE, e1_layers=6, e2_layers=6, e3_layers=3, concept_layers=7, decoder_layers=2),
}


def stress_symbolic(d=128, V=8192, enc=120, con=60, dec=20, e2_values=(0, 24, 40, 48, 60, 120)):
    """Params and executed token-rate block-equivalents for 120/60/20 with E2 = b blocks (each run twice).
    Concept blocks run 2 visits on T/4 slot rows -> 2 * 60 / 4 = 30 token-block equivalents."""
    blk = 12 * d * d
    params = (enc + con + dec) * blk + V * d
    rows = []
    for b in e2_values:
        executed = (enc - b) + 2 * b + dec + 2 * con / 4
        rows.append(dict(e2_blocks=b, executed_token_block_equivalents=executed,
                         ratio_vs_200_layer_ordinary=executed / (enc + con + dec)))
    return dict(width=d, unique_blocks=enc + con + dec, dense_params_approx=params, rows=rows,
                formula='executed = (E - b) + 2b + D + 2C/4 = E + b + D + C/2 token-block equivalents; params = (E + C + D) * 12 d^2 + V d')


def main():
    out = {}
    for name, c in ARMS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            s = summarize(name, c)
        m = macs_per_token(c, T=384, S=3, ordinary=list(range(14, 40)))
        out[name] = dict(dense_incl_training_only=s['v11']['dense_including_training_only'], deployed=s['v11']['deployed'],
                         under_10M=s['v11']['under_10M'], c0_dense=s['c0_same_backbone']['dense'],
                         executed_blocks=s['executed_blocks'], deployed_macs_per_token=m['deployed'],
                         training_macs_per_token=m['training_total'], c0_macs_per_token=m['c0'],
                         compute_matched_ordinary_layers=m['compute_matched_ordinary_layers'],
                         lora_rank4_per_phase=s['lora_rank4_per_phase'])
    out['stress_120_60_20_symbolic'] = stress_symbolic()
    ref = json.loads(Path(__file__).with_name('param_budget.json').read_text())['reference_10M']['v11']
    assert out['reference_balanced_4/4/2_c4_d4']['dense_incl_training_only'] == ref['dense_including_training_only']
    OUT.write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out.items():
        print(k, json.dumps(v)[:400])


if __name__ == '__main__':
    main()
