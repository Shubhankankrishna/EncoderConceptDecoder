"""Analytic params / MACs for the five concept-core designs (CPU only, no torch).

Conventions match code/model.py:macs_per_token: MACs per TOKEN; the core runs once per 4-token chunk (M = T/4 = 256
chunk positions), so per-chunk costs are divided by 4. NON_CORE = encoder + decoder + LM head of the current model.
C7 share = core / (NON_CORE + core), scored at the design's stated INFERENCE R (advisor #22).
"""
NON_CORE = 16.292e6 + 7.327e6 + 3.146e6      # encoder, decoder, head (model.macs_per_token, d 384)
ROW0 = dict(params=2.059064e6, macs_train=1.258e6, macs_eval=1.991e6)   # ConceptCoreV2 at R̄ 2.5 / eval R 4
D_ENC, T = 384, 1024
M = T // 4                                    # chunk positions
PQ_S, PQ_N, PQ_DS = 48, 128, 8                # PQ in c_hat space (384-d) — kept by default


def layer(d=256, H=4, dh=64, kv=4, dff=1024, gr_rank=32, gr_branches=2, gate=True, moe=None, extra_keys=0):
    """One core layer at chunk rate. Returns (params, macs_per_chunk_position)."""
    q_o = 2 * d * H * dh
    k_v = 2 * d * kv * dh
    g = d * H * dh if gate else 0
    if moe is None:
        ffn_p = ffn_m = 3 * d * dff
    else:                                     # moe = (E, d_expert, k, shared_pool_layers)
        E, de, k, share = moe
        ffn_p = 3 * d * de * E / share + d * E  # pool amortised over the layers sharing it, + router
        ffn_m = 3 * d * de * k + d * E
    gr_read = 2 * (gr_branches * d * gr_rank + gr_rank * gr_branches * d)   # attn + ffn read, low-rank gate
    gr_write = 2 * (gr_branches * gr_branches * d)
    scores = 2 * (M / 2 + extra_keys) * H * dh                              # causal QK^T + AV (+ extra static keys)
    p = q_o + k_v + g + ffn_p + gr_read + gr_write + 4 * d
    m = q_o + k_v + g + ffn_m + gr_read + gr_write + scores
    return p, m


def core(passes, n_layers=4, d=256, head_evals=1, per_pass_extra=0.0, once_extra=0.0, **kw):
    """Per-token MACs and params of a core that runs `passes` cycles of `n_layers` layers."""
    lp, lm = layer(d=d, **kw)
    io = 2 * D_ENC * d                                   # in-proj (X) + out-proj (u), once per chunk
    head_p = d * D_ENC + PQ_S * PQ_N * PQ_DS + PQ_S      # W_q 256->384, codebook, log_tau
    head_m = d * D_ENC + 2 * PQ_S * PQ_N * PQ_DS         # W_q + distances + c_pred einsum
    per_chunk = passes * (n_layers * lm + per_pass_extra) + io + head_evals * head_m + once_extra
    params = n_layers * lp + io + head_p
    return params, per_chunk / 4


def share(macs):
    return macs / (NON_CORE + macs)


def report(name, params, macs_eval, macs_train, extra=""):
    print(f"{name:28s} params {params/1e6:5.2f}M  eval {macs_eval/1e6:5.2f}M/tok ({share(macs_eval)*100:4.1f}%)  "
          f"train {macs_train/1e6:5.2f}M/tok  train-total x{(NON_CORE+macs_train)/(NON_CORE+ROW0['macs_train']):.2f} {extra}")


if __name__ == "__main__":
    report("ROW0 ConceptCoreV2", ROW0["params"], ROW0["macs_eval"], ROW0["macs_train"])
    lp, lm = layer()
    print(f"one d256 layer: params {lp/1e6:.3f}M, MACs/chunk-pos {lm/1e6:.3f}M;  4-layer pass = {4*lm/4/1e6:.3f}M/token")
    for R in (1, 2, 4, 6, 8, 12, 16):
        p, m = core(R)
        print(f"  plain 4x256 core, R={R:2d}: {m/1e6:5.2f}M/tok  share {share(m)*100:4.1f}%")
