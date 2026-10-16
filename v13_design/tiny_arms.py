"""Stage-1 tiny arms for v13_NOTES.md §12: matched depth, parameters, FLOPs per token and reach.

    python3 v13_design/tiny_arms.py

Baselines share V13's embedding, Engram (after the first layer), gated attention with the query and key norms and
per-head temperatures of v13_NOTES.md §3.3, and SwiGLU FFN (hidden 256), and are matched to V13-tiny's non-embedding,
non-table parameter count by choosing their depth. Norms and temperatures are not counted (as in accounting.py). FLOPs follow accounting.py's
convention (2 x matmul parameters executed per token) plus, separately, attention-score FLOPs at T = 1,024
(QK^T and AV, forward only, causal average over the visible keys).
"""
from accounting import TIERS, count, attn_params, gr_params

T = 1024
tiny = TIERS[0]
d, h, nb = tiny.d, tiny.heads * tiny.head_dim, tiny.n_branch
ffn = 3 * d * tiny.ffn_hidden
v13 = count(tiny)
target = v13['non_embedding_non_table']
d_mem = 16 * tiny.engram_dim


def scores(n_keys):
    """Attention-score FLOPs per token for one layer that sees n_keys keys on average."""
    return 2 * 2 * n_keys * h


def arm_tf_full(window=None):
    """Plain pre-norm transformer: Engram after layer 1 (one stream), n layers of gated attention + SwiGLU."""
    per_layer = attn_params(tiny, 'swa') + ffn
    engram = d_mem * d * 2 + 4 * d
    n = round((target - engram) / per_layer)
    params = engram + n * per_layer
    keys = (T + 1) / 2 if window is None else min(window, (T + 1) / 2)
    reach = 'unbounded' if window is None else n * (window - 1)
    return dict(layers=n, params=params, flops=2 * params, scores=n * scores(keys), reach=reach)


def arm_tf_confined():
    """V13 minus the concept path: E1's three S layers + Engram as the L stream (two branches, GR), then confined
    decoder layers (queries from the decoder state, keys/values from L^enc in a window of 16) + SwiGLU."""
    s_layer = attn_params(tiny, 'swa') + ffn + 2 * gr_params(tiny, nb, nb)
    engram = d_mem * d + nb * d_mem * d + 4 * d
    exit_read = gr_params(tiny, nb, 1)
    dec_layer = attn_params(tiny, 'swa') + ffn
    fixed = 3 * s_layer + engram + exit_read
    n = round((target - fixed) / dec_layer)
    params = fixed + n * dec_layer
    return dict(layers=f'3 S + {n} decoder', params=params, flops=2 * params,
                scores=(3 + n) * scores(tiny.w_d), reach=v13['reach']['R_D_confined'])


def arm_v13(own_swa=False):
    """V13 tiny: four full-attention F executions (E1, E2 x 2 loops, E3), concept self-attention over T/4 concepts at
    concept rate, decoder concept read over T/4 concepts, local layers at w = 16."""
    n_local = 3 + 3 + tiny.n_d                                  # E1 S, E3 S', decoder local
    sc = 4 * scores((T + 1) / 2) + n_local * scores(16) + 3 * scores(16)          # F layers, locals, E2 GDN-2 ~ chunk 16
    sc += tiny.n_c * scores((T / 4 + 1) / 2) / 4 + tiny.n_d * scores((T / 4 + 1) / 2)
    sc += tiny.n_c * scores(tiny.w_c) / 4                       # concept -> L window
    reach = v13['reach']['R_D_own_swa' if own_swa else 'R_D_confined']
    return dict(layers='E1 4, E2 4 x 2, E3 4, C 2, D 3', params=target, flops=v13['total']['flops'] - 2 * d * tiny.vocab,
                scores=sc, reach=reach)


def main():
    head = 2 * d * tiny.vocab
    rows = [('V13-confined (λ ramp)', arm_v13()), ('V13-confined (λ = 0)', arm_v13()),
            ('V13-own-SWA', arm_v13(own_swa=True)), ('TF-confined', arm_tf_confined()),
            ('TF-full', arm_tf_full()), ('TF-SWA-360 (stage 2)', arm_tf_full(window=16))]
    print(f'V13-tiny non-embedding non-table parameters: {target:,}')
    print(f'{"arm":24s} {"layers":30s} {"params":>10s} {"matmul FLOPs/tok":>17s} {"score FLOPs/tok":>16s} {"total":>10s} {"reach":>10s}')
    for name, r in rows:
        tot = r['flops'] + head + r['scores']
        print(f'{name:24s} {str(r["layers"]):30s} {r["params"]:>10,} {r["flops"] + head:>17,.0f} {r["scores"]:>16,.0f} '
              f'{tot:>10,.0f} {str(r["reach"]):>10s}')


if __name__ == '__main__':
    main()
