"""HRM Thinker v3 (d 256): params, MACs and a throughput range for every design option.

CPU only; never touches the GPU or any run directory. NOTE: rows here use core ff 768 (pre-trim); the final design uses 736 -- exact count in v3_ref.py (18,986,871). The encoder and decoder are built from the real v1 classes
(run_m2_25M/code) with a tiny Engram table (the table is added analytically). The v3 concept core is counted
analytically (it does not exist yet). Parameter convention = run_m2_25M/code/model.py:param_table:
compute = encoder non-embedding + concept module without codebook + decoder (embedding, Engram table, codebook excluded).

Throughput model (two points, crude): seconds/token = a * train-forward MMACs + b * sequential chunk-rate layer applications,
fitted on v1 main (28.0M MAC, 2.5 applications, ~1,730 tok/s) and v2 (27.9M MAC-equiv, 40 applications, ~1,250 tok/s).
MAC-linear alone over-predicted v2 by ~30%; this model is a range, not a promise. Run: nice -n 19 python3 v3_budget.py
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "run_m2_25M", "code"))
import torch  # noqa: E402
from hrm_encoder import EncoderConfig, Encoder  # noqa: E402
from hrm_decoder import DecoderConfig, Decoder  # noqa: E402

V, T, CHUNK = 8192, 1024, 4
M = T // CHUNK
D, H, KV, DH, R_GR = 256, 4, 2, 64, 32
n = lambda m: sum(p.numel() for p in m.parameters())
# two-point throughput fit (see docstring)
_A = None


def fit():
    import numpy as np
    A = np.array([[28.0, 2.5], [27.9, 40.0]]); y = np.array([1 / 1730, 1 / 1250])
    return np.linalg.solve(A, y)          # (a per MMAC, b per application)


def encoder(pattern="SSSFSSSF", d_ff=704, dc=256, engram_reads=1):
    c = EncoderConfig(vocab_size=V, d_model=D, n_heads=H, n_kv_heads=KV, head_dim=DH, d_ff=d_ff, d_concept=dc,
                      layer_pattern=pattern, engram_rows=11, engram_dim=48)
    e = Encoder(c)
    tiny = e.engram.table.weight.numel()
    eng_proj = n(e.engram) - tiny
    total = n(e) - tiny - e.embed.weight.numel() + (engram_reads - 1) * eng_proj
    # MACs / token
    attn_w = D * H * DH * 3 + 2 * D * KV * DH
    ffn = 3 * D * d_ff
    gr_s, gr_f = 2 * (2 * D * R_GR) + 2 * 2 * D, 2 * (2 * 2 * D * R_GR) + 2 * D
    ns, nf = pattern.count("S"), pattern.count("F")
    mac = ns * (attn_w + ffn + gr_s + 2 * 8 * H * DH) + nf * (attn_w + ffn + gr_f + 2 * (T / 2) * H * DH)
    mac += engram_reads * (2 * 8 * 48 * D + 2 * D * R_GR) + 4 * D * dc
    return dict(params=total, layer_S=n(e.layers[0]), layer_F=n(e.layers[pattern.index("F")]), engram_proj=eng_proj,
                pool=n(e.pool), macs=mac, layers=len(pattern))


def decoder(d_ff=704, dc=256, n_layers=4):
    c = DecoderConfig(d_model=D, n_heads=H, n_kv_heads=KV, head_dim=DH, d_ff=d_ff, n_layers=n_layers, window=8,
                      layout="confined", plan_in="add", d_concept=dc)
    dec = Decoder(c)
    attn_w = D * H * DH * 3 + 2 * D * KV * DH
    layer_m = attn_w + 3 * D * d_ff + 2 * (2 * D * R_GR) + 2 * D + 2 * 8 * H * DH
    mac = n_layers * layer_m + 2 * dc * D / CHUNK
    return dict(params=n(dec), layer=n(dec.layers[0]), plan=n(dec.plan), macs=mac, head_macs=D * V)


def core(dc=256, n_layers=4, d_ff=704, shared_ff=0, shared_mode="parallel", mod="table", R=3, heads=None,
         pq=None, shortcut=True):
    """v3 concept core: n_layers distinct FULL-attention layers (gated attention, NoPE, tau, q/k/v convs as v1's F layer)
    wrapped in GR read {C, S} / GR write {S}, each with its own SwiGLU d_ff; optional shared SwiGLU (hidden shared_ff)
    used by every layer ('parallel' = one extra branch per layer, 'once' = once per pass after the stack).
    mod: LoopFormer (t, dt) conditioning. 'paper' = per-layer Linear(d, 4d) adaLN modulator + 2 Fourier MLPs (d);
    'table' = learned table over the 6 reachable (t, dt) pairs -> per-layer 4 d-vectors (alpha_attn, alpha_ffn,
    gamma_attn, gamma_ffn); 'none'."""
    h = heads or dc // DH
    kvh = 2
    attn = dc * h * DH * 3 + 2 * dc * kvh * DH + (h * DH + 2 * kvh * DH) * 4 + 2 * DH + h   # q,o,gate,k,v,convs,norms,tau
    gr = 2 * (2 * (dc * R_GR + R_GR * dc + dc) + 2 * dc)       # read attn + read ffn over {C, S}
    wr = 2 * (dc + 1)
    ffn = 3 * dc * d_ff
    per_layer = attn + gr + wr + ffn
    shared = 3 * dc * shared_ff if shared_ff else 0
    shared_wr = (n_layers * (dc + 1) if shared_mode == "parallel" else (dc + 1)) if shared_ff else 0
    if mod == "paper":
        modp = n_layers * (dc * 4 * dc + 4 * dc) + 2 * (256 * dc + dc + dc * dc + dc)
    elif mod == "table":
        modp = 6 * n_layers * 4 * dc
    else:
        modp = 0
    pq_s, pq_n, pq_ds = pq or ((48, 128, 8) if dc == 384 else (32, 128, 8))
    head = dc * dc + pq_s                                   # NCP W_q + per-segment log_tau
    out_norm = dc
    params = n_layers * per_layer + shared + shared_wr + modp + head + out_norm
    codebook = pq_s * pq_n * pq_ds
    # MACs per chunk position for ONE pass
    attn_m = dc * h * DH * 3 + 2 * dc * kvh * DH + 2 * (M / 2) * h * DH
    layer_m = attn_m + ffn + 2 * (2 * 2 * dc * R_GR) + 2 * dc
    pass_m = n_layers * layer_m + (n_layers if shared_mode == "parallel" else 1) * shared
    head_m = dc * dc + 2 * pq_s * pq_n * pq_ds
    passes_train = R + (1.5 if shortcut else 0)            # full trajectory + shortcut S ~ U{1, 2}
    apps_train = n_layers * passes_train
    return dict(params=params, per_layer=per_layer, ffn=ffn, attn=attn, shared=shared + shared_wr, mod=modp,
                head=head, codebook=codebook, pq=(pq_s, pq_n, pq_ds),
                macs_eval=(R * pass_m + head_m) / CHUNK, macs_train=(passes_train * pass_m + 2 * head_m) / CHUNK,
                apps_eval=n_layers * R, apps_train=apps_train)


def build(name, enc=None, dec=None, cor=None, ls_via_decoder=False):
    enc, dec, cor = enc or {}, dec or {}, cor or {}
    dc = cor.get("dc", 256)
    e = encoder(dc=dc, **enc); d = decoder(dc=dc, **dec); c = core(**cor)
    compute = e["params"] + c["params"] + d["params"]
    fwd_eval = e["macs"] + c["macs_eval"] + d["macs"] + d["head_macs"]
    fwd_train = e["macs"] + c["macs_train"] + d["macs"] + d["head_macs"]
    if ls_via_decoder:                                      # second decoder + head pass for the shortcut NTP
        fwd_train += d["macs"] + d["head_macs"]
    a, b = fit()
    s_tok = a * fwd_train / 1e6 + b * c["apps_train"]
    tok_s = 1 / s_tok
    return dict(name=name, compute=compute, enc=e["params"], core=c["params"], dec=d["params"],
                share=(e["params"] / compute, c["params"] / compute, d["params"] / compute),
                fwd_eval=fwd_eval, fwd_train=fwd_train, core_mac_share_eval=c["macs_eval"] / fwd_eval,
                tok_s=(0.8 * tok_s, tok_s * 1.1), parts=dict(encoder=e, decoder=d, core=c))


if __name__ == "__main__":
    torch.manual_seed(0)
    rows = [
        build("R0 v2-like enc8 + core dc256 4L ff704, no shared, no mod"),
        build("R1 enc8 + core dc256 4L ff704 + shared1024 (parallel) + mod table", cor=dict(shared_ff=1024)),
        build("R2 enc8 + core dc256 4L ff704 + shared1024 + mod PAPER", cor=dict(shared_ff=1024, mod="paper")),
        build("R3 enc8 + core dc384 4L ff704 + shared1024 + table", cor=dict(dc=384, shared_ff=1024)),
        build("R4 enc8 + core dc384 4L ff1056 + shared1056 + table", cor=dict(dc=384, d_ff=1056, shared_ff=1056)),
        build("R5 enc10 SSSFSSSFSF + core dc256 + shared1024", enc=dict(pattern="SSSFSSSFSF"), cor=dict(shared_ff=1024)),
        build("R6 enc12 (SSSF)x3 + core dc256 + shared1024", enc=dict(pattern="SSSFSSSFSSSF"), cor=dict(shared_ff=1024)),
        build("R7 enc8 2 engram reads + core dc256 + shared1024", enc=dict(engram_reads=2), cor=dict(shared_ff=1024)),
        build("R8 = R1 but shared FFN once per pass", cor=dict(shared_ff=1024, shared_mode="once")),
        build("R9 = R1 with L_S through the decoder", cor=dict(shared_ff=1024), ls_via_decoder=True),
        build("R10 enc8 + core dc256 4L ff1024 + shared1024", cor=dict(d_ff=1024, shared_ff=1024)),
        build("R11 enc8 ff768 + core dc256 4L ff768 + shared1024 + dec ff768", enc=dict(d_ff=768), dec=dict(d_ff=768),
              cor=dict(d_ff=768, shared_ff=1024)),
    ]
    for r in rows:
        s = r["share"]
        print(f"{r['name'][:62]:62s} compute {r['compute']/1e6:6.2f}M | enc {r['enc']/1e6:5.2f} core {r['core']/1e6:5.2f} "
              f"dec {r['dec']/1e6:5.2f} ({s[0]*100:3.0f}/{s[1]*100:3.0f}/{s[2]*100:3.0f}%) | fwd eval {r['fwd_eval']/1e6:5.2f}M "
              f"train {r['fwd_train']/1e6:5.2f}M core-MAC {r['core_mac_share_eval']*100:3.0f}% | ~{r['tok_s'][0]:.0f}-{r['tok_s'][1]:.0f} tok/s")
    c = rows[1]["parts"]
    print("parts R1:", {k: {kk: (round(vv / 1e6, 3) if isinstance(vv, (int, float)) and vv > 1000 else vv) for kk, vv in v.items()}
                        for k, v in c.items()})
    print("mod cost paper vs table (dc 256):", core(mod="paper")["mod"], core(mod="table")["mod"],
          "| dc 384:", core(dc=384, mod="paper")["mod"], core(dc=384, mod="table")["mod"])
    a, b = fit(); print(f"fit: a={a:.3e} s/tok per MMAC, b={b:.3e} s/tok per application")
    json.dump(rows, open(os.path.join(HERE, "v3_budget.json"), "w"), indent=1, default=str)
