"""HRM THINKER-25M: encoder v1.0 (frozen) -> concept module v2 (looped core + NCP) -> decoder v1 (confined).

The wiring is e2e.HRMThinker with configs() replaced by PROMPT.md section 2.1 (d = 384). The decoder receives only
tok, u, c_pred, doc and E. This module adds nothing to the architecture; it only exposes the evaluation
interventions of section 5.2 (core off, plan swap, u off, c off, loop depth R) on the same wiring.
"""
import math
import torch
import torch.nn as nn

import compat  # noqa: F401  (renames EncoderLayer._apply so Module.to(device) works; numerics unchanged)
from hrm_encoder import EncoderConfig, Encoder, engram_indices
from concept_core_v2 import ConceptCoreV2
from hrm_decoder import DecoderConfig, Decoder, param_groups
from e2e import HRMThinker

S_SEG, N_CODES = 48, 128          # PQ vocabulary: 48 segments x 128 codes x 8 dims (d_c = 384)


def configs(d=384, V=8192, layout="confined", n_heads=6, n_kv=2, head_dim=64, d_ff=1056, **enc_over):
    """PROMPT.md 2.1: EncoderConfig defaults except width/heads/ffn/vocab; decoder confined, 4 layers, w = 8, add."""
    ecfg = EncoderConfig(vocab_size=V, d_model=d, n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim, d_ff=d_ff,
                         **enc_over)
    dcfg = DecoderConfig(d_model=d, n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim, d_ff=d_ff, n_layers=4,
                         window=8, layout=layout, plan_in="add")
    return ecfg, dcfg


def build_model(d=384, V=8192, layout="confined", S=S_SEG, N=N_CODES, n_heads=6, n_kv=2, head_dim=64, d_ff=1056,
                **enc_over):
    ecfg, dcfg = configs(d, V, layout, n_heads=n_heads, n_kv=n_kv, head_dim=head_dim, d_ff=d_ff, **enc_over)
    return HRMThinker(ecfg, dcfg, S=S, N=N), ecfg, dcfg


def forward(m, ids, doc, eidx, R, core_off=False, plan_swap=False, use_u=True, use_c=True, return_passes=False):
    """Same wiring as HRMThinker.forward, plus the plan-swap intervention (u and c_pred of another row,
    roll(1, dim=0)). Returns logits (B,T,V) and aux (c_hat, cdoc, NCP logits, u, c_pred, h, P, encoder aux)."""
    tok, c_hat, eaux = m.enc(ids, doc, eidx)
    cdoc = doc[:, ::4]
    u, lc, c_pred = m.core(c_hat, cdoc, R)
    if plan_swap:
        u_dec, c_dec = u.roll(1, dims=0), c_pred.roll(1, dims=0)
    else:
        u_dec, c_dec = u, c_pred
    logits, daux = m.dec(tok, u_dec, c_dec, doc, m.enc.embed.weight, core_off=core_off, use_u=use_u, use_c=use_c)
    return logits, dict(c_hat=c_hat, cdoc=cdoc, logits_c=lc, u=u, c_pred=c_pred, h=daux["h"], P=daux["P"], enc=eaux)


def param_table(m):
    """Section 2.2 table, read off the live model."""
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    enc_all = n(m.enc)
    table = m.enc.engram.table.weight.numel()
    emb = m.enc.embed.weight.numel()
    core = n(m.core)
    code = m.core.vocab.e.numel()
    dec = n(m.dec)
    # PROMPT 2.2 convention: compute = encoder non-embedding + concept module without codebook + decoder
    # (the tied embedding and the codebook are lookups); stored = everything.
    compute = (enc_all - table - emb) + (core - code) + dec
    return {"encoder_non_embedding": enc_all - table - emb, "engram_table": table, "token_embedding": emb,
            "concept_module": core, "codebook": code, "decoder": dec, "compute_params": compute,
            "stored_total": compute + table + emb + code}


def macs_per_token(ecfg, dcfg, T=1024, R_mean=2.5):
    """Multiply-accumulates per token (weights + attention scores); training FLOPs = 6 x MACs."""
    d, V = ecfg.d_model, ecfg.vocab_size
    H, Hk, dh = ecfg.n_heads, ecfg.n_kv_heads, ecfg.head_dim
    attn_w = d * H * dh * 3 + 2 * d * Hk * dh          # q, o, gate, k, v
    ffn = 3 * d * ecfg.d_ff
    gr_s, gr_f = 2 * (2 * d * ecfg.gr_rank) + 2 * 2 * d, 2 * (2 * 2 * d * ecfg.gr_rank) + 2 * d
    n_s, n_f = ecfg.layer_pattern.count("S"), ecfg.layer_pattern.count("F")
    swa_scores = 2 * ecfg.window * H * dh
    full_scores = 2 * (T / 2) * H * dh
    enc = n_s * (attn_w + ffn + gr_s + swa_scores) + n_f * (attn_w + ffn + gr_f + full_scores)
    de = ecfg.n_tables * ecfg.engram_dim
    enc += 2 * de * d + 2 * d * ecfg.gr_rank + 4 * d * d        # engram W_K, W_V, read; CRC pool (4d -> d)
    M = T // 4
    core_layer = attn_w + ffn + 2 * (2 * 2 * d * ecfg.gr_rank) + 2 * d + 2 * (M / 2) * H * dh
    core = R_mean * core_layer / 4 + d * d / 4                   # once per 4 tokens; NCP head W_q
    dec_layer = attn_w + ffn + 2 * (2 * d * dcfg.gr_rank) + 2 * d + 2 * dcfg.window * H * dh
    dec = dcfg.n_layers * dec_layer + 2 * d * d / 4
    head = d * V
    return dict(encoder=enc, core=core, decoder=dec, head=head, total=enc + core + dec + head)


if __name__ == "__main__":
    torch.manual_seed(0)
    m, ecfg, dcfg = build_model()
    tab = param_table(m)
    expect = {"encoder_non_embedding": 15.51e6, "engram_table": 50.33e6, "token_embedding": 3.15e6,
              "concept_module": 2.06e6, "codebook": 0.049e6, "decoder": 7.53e6, "compute_params": 25.05e6,
              "stored_total": 78.58e6}
    ok = True
    print(f"{'part':24s} {'live':>12s} {'PROMPT 2.2':>12s} {'rel diff':>9s}")
    for k, v in tab.items():
        rel = v / expect[k] - 1
        ok &= abs(rel) <= 0.01
        print(f"{k:24s} {v/1e6:11.3f}M {expect[k]/1e6:11.3f}M {rel*100:8.2f}%")
    print("EncoderConfig.reach() =", ecfg.reach(), "| DecoderConfig.reach() =", dcfg.reach())
    mac = macs_per_token(ecfg, dcfg)
    print("MACs/token: " + ", ".join(f"{k} {v/1e6:.2f}M" for k, v in mac.items()) +
          f" -> training FLOP/token = 6 x MACs = {6*mac['total']/1e6:.1f}M")
    assert ecfg.reach() == 63 and dcfg.reach() == 70
    print("CONFIG CHECK", "PASS (all parts within 1% of PROMPT 2.2)" if ok else "FAIL")
