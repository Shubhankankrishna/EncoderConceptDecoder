"""HRM Thinker v2 (d 256, Engram 80M, concept module D6, decoder + NextLat): params and MACs per token.

CPU only, never touches runs/ or the GPU. Encoder and decoder are instantiated from the real code at d 256 with a tiny
Engram table (the table is added analytically), the D6 core is counted analytically (it is not implemented yet), and the
non-core MACs reuse code/model.py:macs_per_token. Parameter convention = code/model.py:param_table ("25M" of today):
compute = encoder non-embedding + concept module without codebook + decoder; embedding, Engram table, codebook excluded.
Writes model_v2/v2_budget.json.   Run: python3 model_v2/v2_budget.py
"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import torch  # noqa: E402
from hrm_encoder import EncoderConfig, Encoder  # noqa: E402
from hrm_decoder import DecoderConfig, Decoder, NextLat  # noqa: E402
from model import macs_per_token  # noqa: E402

V, T, CHUNK = 8192, 1024, 4
M = T // CHUNK                                   # 256 chunk positions
D, H, KV, DH = 256, 4, 2, 64                     # model width, heads, kv heads, head dim
DC = 384                                         # concept width (PQ space) -- kept (step 2)
PQ_S, PQ_N, PQ_DS = 48, 128, 8                   # 48 segments x 128 codes x 8 dims = 384, 336 bits / chunk
ENG_ROWS, ENG_DIM, ENG_TABLES = 208333, 48, 8    # 208,333 is prime -> 8 x 208,333 x 48 = 79.99M
TODAY = dict(compute=25.052781e6, macs_train=28.023e6, tok_s=(1650, 1815))   # measured main run (d 384)
n = lambda m: sum(p.numel() for p in m.parameters())


def encoder(d_ff):
    c = EncoderConfig(vocab_size=V, d_model=D, n_heads=H, n_kv_heads=KV, head_dim=DH, d_ff=d_ff, d_concept=DC,
                      engram_rows=11, engram_dim=ENG_DIM)
    e = Encoder(c)
    tiny = e.engram.table.weight.numel()
    p = dict(total_non_embedding=n(e) - tiny - e.embed.weight.numel(), layer_S=n(e.layers[0]), layer_F=n(e.layers[3]),
             layer_S_attn=n(e.layers[0].attn), layer_S_ffn=n(e.layers[0].ffn),
             layer_S_gr=n(e.layers[0].read_attn) + n(e.layers[0].read_ffn) + n(e.layers[0].write_attn) + n(e.layers[0].write_ffn),
             layer_F_gr=n(e.layers[3].read_attn) + n(e.layers[3].read_ffn) + n(e.layers[3].write_attn) + n(e.layers[3].write_ffn),
             engram_proj=n(e.engram) - tiny, pool=n(e.pool), crc_router=n(e.crc_router), tok_norm=n(e.tok_norm),
             embed=e.embed.weight.numel(), engram_table=ENG_TABLES * ENG_ROWS * ENG_DIM)
    return c, p


def decoder(d_ff):
    c = DecoderConfig(d_model=D, n_heads=H, n_kv_heads=KV, head_dim=DH, d_ff=d_ff, n_layers=4, window=8,
                      layout="confined", plan_in="add", d_concept=DC)
    dec = Decoder(c)
    # u arrives at d 256 from the core (not in PQ space): W_u is 256 -> 256, not dc -> 256
    wu_fix = (D - DC) * D
    p = dict(total=n(dec) + wu_fix, layer=n(dec.layers[0]), attn=sum(n(getattr(dec.layers[0], k)) for k in ("q", "k", "v", "o", "gate", "q_norm", "k_norm")),
             ffn=n(dec.layers[0].ffn), plan=n(dec.plan) + wu_fix, out_norm=n(dec.out_norm))
    return c, p


def d6_core(E, d_expert=512, top_k=2, R_eval=6, K_train_mean=10):
    """D6 LOOPFORMER-MoE-DS at d 256: 4 distinct layers, full causal attention 4 x 64 (GQA kv 2), sigmoid output gate,
    GR read over [X~, S] (rank 32), shared pool of E SwiGLU experts x d_expert, top-k, per-layer routers, zero-init
    block outputs; S_0 = W_s c_hat, X~ = W_in c_hat (384 -> 256 each); LN_h carry; NCP head W_q 256 -> 384."""
    d, r = D, 32
    attn = 2 * d * H * DH + 2 * d * KV * DH + d * H * DH + 2 * DH          # q, o, k, v, gate, q/k norms
    gr = 2 * (2 * (d * r + r * d + d) + 2 * d)                          # attn + ffn read, 2 branches (+ U bias, gamma)
    wr = 2 * (d + 1)                                                    # attn + ffn write scalars (GRWrite, 1 stream)
    router = d * E
    per_layer = attn + gr + wr + router
    pool = E * 3 * d * d_expert
    io = 2 * DC * d                                                     # W_in, W_s
    carry = 2 * d                                                       # LN_h gain + readout norm gain
    head = d * DC + PQ_S                                                # W_q 256 -> 384, log_tau
    codebook = PQ_S * PQ_N * PQ_DS
    params = 4 * per_layer + pool + io + carry + head
    # MACs per chunk position
    attn_m = 2 * d * H * DH + 2 * d * KV * DH + d * H * DH + 2 * (M / 2) * H * DH   # weights + causal scores
    ffn_m = top_k * 3 * d * d_expert + router
    layer_m = attn_m + ffn_m + 2 * (2 * d * r + r * 2 * d) + 4 * d
    head_m = d * DC + 2 * PQ_S * PQ_N * PQ_DS
    eval_chunk = R_eval * 4 * layer_m + io + head_m
    train_chunk = K_train_mean * 4 * layer_m + io + K_train_mean * head_m          # per-pass NCP head
    return dict(E=E, d_expert=d_expert, top_k=top_k, per_layer=per_layer, pool=pool, io=io, head=head,
                codebook=codebook, params=params, macs_eval=eval_chunk / CHUNK, macs_train_fwd=train_chunk / CHUNK,
                R_eval=R_eval, K_train_mean=K_train_mean)


def nextlat(hidden):
    nl = NextLat(D, hidden)
    psi = n(nl)
    psi_m = 2 * D * hidden + hidden * hidden + hidden * D                # 3-layer MLP (LN negligible)
    kl_m = D * V                                                        # prior logits h_hat @ E^T (posterior = main logits, reused)
    return dict(hidden=hidden, psi_params=psi, psi_macs=psi_m, kl_head_macs=kl_m,
                train_equiv_with_kl=3 * psi_m + 2 * kl_m,             # psi fwd+bwd; head fwd + input-grad only (E frozen)
                train_equiv_no_kl=3 * psi_m)


def build(enc_ff, dec_ff, E, nl_hidden=2 * D, lam_kl=1.0):
    ec, pe = encoder(enc_ff)
    dc, pd = decoder(dec_ff)
    core = d6_core(E)
    nl = nextlat(nl_hidden)
    mac = macs_per_token(ec, dc, T=T)
    # macs_per_token counts the CRC pool as 4d x d; ours is 4d x dc
    enc_m = mac["encoder"] - 4 * D * D + 4 * D * DC
    # macs_per_token uses the ENCODER d_ff for the decoder FFN too; correct it to the decoder's own d_ff
    dec_m = mac["decoder"] + dc.n_layers * 3 * D * (dec_ff - enc_ff)
    mac["decoder"] = dec_m
    fwd_eval = enc_m + mac["decoder"] + mac["head"] + core["macs_eval"]
    fwd_train = enc_m + mac["decoder"] + mac["head"] + core["macs_train_fwd"]
    nl_train = nl["train_equiv_with_kl"] if lam_kl else nl["train_equiv_no_kl"]
    train_equiv = 3 * fwd_train + nl_train
    compute = pe["total_non_embedding"] + core["params"] + pd["total"]
    tok_s = [round(t * 3 * TODAY["macs_train"] / train_equiv) for t in TODAY["tok_s"]]
    return dict(
        cfg=dict(enc_d_ff=enc_ff, dec_d_ff=dec_ff, E=E, nl_hidden=nl_hidden, lam_kl=lam_kl),
        params=dict(encoder=pe["total_non_embedding"], core=core["params"], decoder=pd["total"], compute=compute,
                    share=dict(encoder=pe["total_non_embedding"] / compute, core=core["params"] / compute,
                               decoder=pd["total"] / compute),
                    embed=pe["embed"], engram_table=pe["engram_table"], codebook=core["codebook"],
                    nextlat_psi_training_only=nl["psi_params"],
                    stored=compute + pe["embed"] + pe["engram_table"] + core["codebook"]),
        parts=dict(encoder=pe, decoder=pd, core=core, nextlat=nl),
        macs=dict(encoder=enc_m, core_eval=core["macs_eval"], core_train=core["macs_train_fwd"], decoder=mac["decoder"],
                  head=mac["head"], fwd_eval=fwd_eval, fwd_train=fwd_train, core_share_eval=core["macs_eval"] / fwd_eval,
                  nextlat_train_equiv=nl_train, train_equiv=train_equiv,
                  train_vs_today=train_equiv / (3 * TODAY["macs_train"])),
        est_tok_s=tok_s, est_tokens_24h=[t * 86400 for t in tok_s])


if __name__ == "__main__":
    torch.manual_seed(0)
    rows = {}
    for name, kw in [("A_D6_as_scored_E8", dict(enc_ff=704, dec_ff=704, E=8)),
                     ("B_E12", dict(enc_ff=704, dec_ff=704, E=12)),
                     ("C_E16", dict(enc_ff=704, dec_ff=704, E=16)),
                     ("D_E16_dec1024", dict(enc_ff=704, dec_ff=1024, E=16)),
                     ("E_E12_dec1024", dict(enc_ff=704, dec_ff=1024, E=12)),
                     ("F_E8_dec1024", dict(enc_ff=704, dec_ff=1024, E=8))]:
        r = build(**kw); rows[name] = r
        p, m = r["params"], r["macs"]
        print(f"{name:20s} compute {p['compute']/1e6:6.2f}M  enc {p['encoder']/1e6:5.2f} ({p['share']['encoder']*100:4.1f}%)  "
              f"core {p['core']/1e6:5.2f} ({p['share']['core']*100:4.1f}%)  dec {p['decoder']/1e6:5.2f} ({p['share']['decoder']*100:4.1f}%) | "
              f"fwd eval {m['fwd_eval']/1e6:5.2f}M/tok core {m['core_share_eval']*100:4.1f}% | train x{m['train_vs_today']:.2f} of today "
              f"-> ~{r['est_tok_s'][0]}-{r['est_tok_s'][1]} tok/s | stored {p['stored']/1e6:6.1f}M")
    # recommended row C: shares by params / active params per token / forward MACs, and the eval-R sweep
    c = rows["C_E16"]; core = c["parts"]["core"]
    act_core = 4 * (core["per_layer"] + core["top_k"] * 3 * D * core["d_expert"]) + core["io"] + core["head"]
    act = dict(encoder=c["params"]["encoder"], core=act_core, decoder=c["params"]["decoder"])
    tot = sum(act.values())
    c["active"] = dict(act, total=tot, share={k: v / tot for k, v in act.items()})
    m = c["macs"]; fe = m["fwd_eval"]
    c["mac_share"] = dict(encoder=m["encoder"] / fe, core=m["core_eval"] / fe, decoder=m["decoder"] / fe, head=m["head"] / fe)
    sweep = {}
    for R in (1, 2, 4, 6, 8, 12, 16):
        ce = d6_core(16, R_eval=R)["macs_eval"]; f = fe - m["core_eval"] + ce
        sweep[R] = dict(core_macs=ce, share=ce / f)
    c["r_sweep"] = sweep
    print("row C active params:", {k: round(v / 1e6, 2) for k, v in act.items()}, "shares", {k: round(v * 100, 1) for k, v in c["active"]["share"].items()})
    print("row C fwd-MAC shares:", {k: round(v * 100, 1) for k, v in c["mac_share"].items()})
    print("row C core share vs eval R:", {R: f"{v['share']*100:.1f}%" for R, v in sweep.items()})
    json.dump(rows, open(os.path.join(HERE, "v2_budget.json"), "w"), indent=1)
    print("wrote model_v2/v2_budget.json")
