"""Analytic params / FLOPs / KV for the concept module (per TOKEN; the core runs once per k=4 tokens)."""
def layer(d, f, H, kvh, dh, n_streams, r=32):
    proj = d*H*dh + 2*d*kvh*dh + H*dh*d + d*H*dh            # q, k|v, o, output gate
    ffn = 3*d*f; gates = 2*(n_streams*(2*d*r) + d)            # GR read (low rank) + write gates, both sublayers
    return proj + ffn + gates, proj + ffn                      # params, matmul MACs per position
def core(d, f, H, kvh, dh, T, passes, layers_per_pass=1, n_streams=3, k=4):
    P, mac = layer(d, f, H, kvh, dh, n_streams); M = T // k
    attn = 2 * (M / 2) * H * dh                                 # QK^T + AV, causal average, per position
    mac_tok = passes * layers_per_pass * (mac + attn) / k
    kv_tok = passes * layers_per_pass * 2 * kvh * dh / k        # cached numbers per token
    return layers_per_pass * P, 6 * mac_tok, kv_tok
def row(name, P, fl, kv): print(f"{name:52s} params {P/1e6:8.2f}M | train FLOP/token {fl/1e6:9.1f}M | KV numbers/token {kv:7.0f}")
for tag, (d, f, H, kvh, dh, T) in {'M2  (d=512, T=1024)': (512, 1408, 8, 2, 64, 1024),
                                   'NCP-8.9B width (d=4096, T=8192)': (4096, 11008, 32, 8, 128, 8192)}.items():
    print(f"--- {tag}")
    row('NCP concept module: 8 untied layers', *core(d, f, H, kvh, dh, T, 1, 8, 1))
    row('v2: 1 shared layer, train E[R]=2.5', *core(d, f, H, kvh, dh, T, 2.5, 1, 2))
    row('v2: inference R=4', *core(d, f, H, kvh, dh, T, 4, 1, 2))
    row('v2: inference R=8 (test-time depth)', *core(d, f, H, kvh, dh, T, 8, 1, 2))
    Pz, flz, kvz = core(d, f, H, kvh, dh, T, 2.5, 1, 3); row('v2.1 (XM, K=4): training core cost x4', Pz + 4*d, 4*flz, kvz)
    row('scale variant: 2 layers/pass, train E[R]=2.5', *core(d, f, H, kvh, dh, T, 2.5, 2, 2))
