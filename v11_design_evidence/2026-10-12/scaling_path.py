"""Assumption-based scaling arithmetic for V11 (no hardware measured; nothing trained).

  python -B v11_design_evidence/2026-10-12/scaling_path.py   (from the repo root)

Formulas (per block, width d):
  dense block params           12 d^2                         (attention 4 d^2, 4x FFN 8 d^2)
  MoE block params             4 d^2 + n_e * 3 d h + d n_e    (SwiGLU experts of hidden h, router)
  MoE active params            4 d^2 + k * 3 d h + d n_e      (top-k)
  executed token-rate block-equivalents per token
                               X = (E - b) + 2 b + D + 2 C / 4 = E + b + D + C/2
                               (E unique encoder blocks of which b are recurrent E2, run twice;
                                C concept blocks, 2 visits on T/4 slot rows; D decoder blocks)
  active executed params / token   N_x = X * active_block + V d   (tied head counted once)
  training FLOPs               6 * N_x * tokens   (+ attention terms, omitted: stated per row)
  MoE all-to-all bytes / token = 2 * k * d * bytes * (number of executed MoE block applications at token rate)
  serving KV bytes / token     = 2 d_kv * bytes * (full-attention applications + sliding applications * W / T_serve)
                                 with concept applications at 1/4 rate.
Assumptions are labelled in each row; tokens use 20 per active executed parameter (a dense
compute-optimal heuristic from outside the provided papers), only to size FLOPs.
Writes scaling_path.json.
"""
import json
from pathlib import Path

OUT = Path(__file__).with_name('scaling_path.json')


def rung(name, d, E, b, C, D, V, moe=None, target_total=None, T_serve=32768, window=None, bytes_=2, kv_frac=1.0):
    window = window or max(64, d // 2)
    attn = 4 * d * d
    if moe:
        h, k = moe['h'], moe['k']
        n_e = moe.get('n_e')
        if n_e is None:
            per_block = target_total / (E + C + D)
            n_e = max(k, round((per_block - attn) / (3 * d * h + d)))
        block_total = attn + n_e * 3 * d * h + d * n_e
        block_active = attn + k * 3 * d * h + d * n_e
    else:
        n_e = k = h = None
        block_total = block_active = 12 * d * d
    total = (E + C + D) * block_total + V * d
    X = E + b + D + C / 2
    n_x = X * block_active + V * d
    tokens = 20 * n_x
    flops = 6 * n_x * tokens
    executed_token_apps = E + b + D                    # token-rate applications
    full_frac = 0.25                                   # inherited rule: every 4th encoder block full; decoder full
    enc_apps = E + b
    full_apps = enc_apps * full_frac + D
    slide_apps = enc_apps * (1 - full_frac)
    concept_apps = 2 * C / 4                           # per token, KV rows at slot rate
    kv_per_token = 2 * d * kv_frac * bytes_ * (full_apps + slide_apps * window / T_serve + concept_apps)
    a2a = (2 * k * d * bytes_ * (executed_token_apps + concept_apps)) if moe else 0
    return dict(name=name, width=d, unique_blocks=dict(encoder=E, e2_recurrent=b, concept=C, decoder=D), vocab=V,
                moe=dict(n_experts=n_e, top_k=k, expert_hidden=h) if moe else None,
                total_params=total, active_executed_params_per_token=n_x, executed_block_equivalents=X,
                assumed_tokens=tokens, training_flops=flops,
                serving=dict(decode_flops_per_token=2 * n_x, kv_bytes_per_token_at_T=kv_per_token, T_serve=T_serve,
                             concept_attention='O(G^2), G = T/8 groups (2 slot rows each)'),
                moe_all_to_all_bytes_per_token=a2a)


def main():
    rows = [
        rung('R1 reference (Air; instantiated count is in param_budget.json)', 128, 10, 4, 4, 4, 8192, T_serve=384, window=64),
        rung('R2 dense ~60M', 384, 15, 6, 6, 4, 32768, T_serve=4096, window=256),
        rung('R3 dense ~0.5B', 1024, 24, 10, 10, 6, 32768, T_serve=8192, window=512),
        rung('R4 dense ~3B', 2048, 34, 14, 16, 8, 65536, T_serve=16384, window=1024),
        rung('R5 MoE ~25B total', 2048, 34, 14, 16, 8, 65536, moe=dict(n_e=64, k=4, h=1024), T_serve=16384, window=1024),
        rung('H1 hypothetical 1T total (120/60/20, E2 = 48)', 7168, 120, 48, 60, 20, 163840,
             moe=dict(h=2048, k=8), target_total=1.0e12, kv_frac=0.125, window=4096),
        rung('H3 hypothetical 3T total (120/60/20, E2 = 48)', 8192, 120, 48, 60, 20, 163840,
             moe=dict(h=2048, k=8), target_total=3.0e12, kv_frac=0.125, window=4096),
        rung('H1n 1T total, narrower (d 4096, 120/60/20, E2 = 48)', 4096, 120, 48, 60, 20, 163840,
             moe=dict(h=1536, k=8), target_total=1.0e12, kv_frac=0.125, window=4096),
        rung('H3n 3T total, narrower (d 5120, 120/60/20, E2 = 48)', 5120, 120, 48, 60, 20, 163840,
             moe=dict(h=1536, k=8), target_total=3.0e12, kv_frac=0.125, window=4096),
    ]
    OUT.write_text(json.dumps(rows, indent=1) + '\n')
    for r in rows:
        print(f"{r['name'][:48]:48s} total {r['total_params']/1e9:9.3f}B  active/token {r['active_executed_params_per_token']/1e9:8.3f}B  "
              f"X {r['executed_block_equivalents']:6.1f}  tokens {r['assumed_tokens']/1e12:8.3f}T  FLOPs {r['training_flops']:.2e}  "
              f"KV/token {r['serving']['kv_bytes_per_token_at_T']/1e3:8.1f}kB  a2a/token {r['moe_all_to_all_bytes_per_token']/1e6:6.2f}MB  experts {r['moe'] and r['moe']['n_experts']}")


if __name__ == '__main__':
    main()
