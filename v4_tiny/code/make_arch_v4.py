"""Write run_m2_25M/webapp/data/arch_v4.json: the draft_v4 architecture with exact counts from model_v4 (full size).
Same key shape as data/arch_v3d.json so the dashboard's v4 pages follow the v3 pages. Run on CPU:
  python make_arch_v4.py"""
import json
import os
import torch

import model_v4 as M4

OUT = "/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/data/arch_v4.json"
V3 = "/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/data/arch_v3d.json"
n = lambda mod: sum(p.numel() for p in mod.parameters())
ROWS_FULL = 208333


def counts(**over):
    m = M4.build_model("full", engram_rows=11, **over)
    t = M4.param_table(m)
    table = 8 * ROWS_FULL * 48
    e, c, d = m.enc, m.core, m.dec
    S = next(l for l in e.layers if l.kind == "S"); F = next(l for l in e.layers if l.kind == "F")
    P = dict(
        enc_layer_S=n(S), enc_layer_S_attn=n(S.attn), enc_layer_S_ffn=n(S.ffn), enc_layer_S_gr=n(S) - n(S.attn) - n(S.ffn),
        enc_layer_F=n(F), enc_layer_F_attn=n(F.attn), enc_layer_F_ffn=n(F.ffn), enc_layer_F_gr=n(F) - n(F.attn) - n(F.ffn),
        engram_read1=n(e.engram) - e.engram.table.weight.numel(), engram_read2=n(e.engram2), engram_table=table,
        enc_shared=n(e.shared) if e.shared is not None else 0, enc_shared_gains=e.shared_gain.numel() if e.shared is not None else 0,
        crc_router=n(e.crc_router), tok_norm=n(e.tok_norm), pooler=n(e.pooler),
        core_layer=n(c.layers[0]), core_attn=n(c.layers[0].attn), core_own_ffn=n(c.layers[0].ffn),
        core_gr=n(c.layers[0]) - n(c.layers[0].attn) - n(c.layers[0].ffn), core_shared=n(c.shared),
        ncp_head=n(c.Wq) + c.log_tau.numel() + 1, core_out_norm=n(c.out_norm),
        coarse_codebook=c.E0.numel(), pq_codebook=c.e.numel(),
        dec_layer=n(d.layers[0]), dec_plan=n(d.plan), dec_read=n(d.eng[0]), dec_reads=n(d.eng), dec_out_norm=n(d.out_norm),
        token_embedding=e.embed.weight.numel())
    pool = e.pooler
    if hasattr(pool, "Wk"):
        P.update(pool_Wk=pool.Wk.weight.numel(), pool_Wv=pool.Wv.weight.numel(), pool_Wo=pool.Wo.weight.numel(), pool_q=pool.q.numel(),
                 pool_p=pool.p.numel(), pool_b=pool.b.numel(), pool_ffn=(n(pool.ffn) + 1) if pool.ffn is not None else 0)
    T = dict(encoder_non_embedding=t["encoder"], concept_module=t["core"], decoder=t["decoder"], compute_params=t["compute"],
             engram_table=table, token_embedding=t["embedding"], codebook=t["codebooks"],
             stored_total=t["compute"] + table + t["embedding"] + t["codebooks"])
    return P, T


def main():
    torch.manual_seed(0)
    P, T = counts()
    v3 = json.load(open(V3))
    variants = {}
    for name, over in (("k8", dict(k=8)), ("k4", dict(k=4)), ("gconv", dict(pool="gconv")), ("attn_only", dict(pool="attn"))):
        _, Tv = counts(**over); variants[name] = Tv["compute_params"]
    rec = T["compute_params"] - P["dec_reads"] - P["pool_ffn"]
    A = dict(
        source="draft_v4.md (revision 2, 2026-10-01) · exact counts from v4_tiny/code/model_v4.py (full config) · tests in v4_tiny/",
        status="Proposed · implemented and tested at tiny scale on CPU (v4_tiny) · not trained at full scale",
        revision="r2",
        ecfg=dict(vocab_size=8192, d_model=256, n_heads=4, n_heads_full=6, n_kv_heads=2, head_dim=64, d_ff=1024, chunk=6, d_concept=256,
                  window=8, rope_dims=16, layer_pattern="SSSFSSSF", engram_orders=[2, 3], engram_heads=4, engram_rows=ROWS_FULL, engram_dim=48,
                  conv_kernel=4, gr_rank=32, enc_loop=2, enc_shared_ff=384, T=1020, block=60),
        core=dict(n_layers=4, heads=6, kv=2, own_ff=736, shared_ff=1024, M=2, supervision="terminal-only"),
        pooler=dict(kind="attention + FFN", heads=4, head_dim=64, ffn=384, init="exact masked mean"),
        pq=dict(S=32, N=256, ds=8, bits=256), coarse=dict(K=1024, d=256, bits=10),
        reach_enc=63, reach_dec=70, table=T, params=P,
        variants=dict(compute=variants, recommended_main_run=rec,
                      recommended_note="decoder Engram reads off and pooler FFN off in the main run (both kept as arms) — GO_NOGO.md"),
        k_compare=[  # draft §3.5.3, measured on the val probe items and 3,000 training rows
            ["concepts per row (T)", "256 (1,024)", "170 (1,020)", "128 (1,024)"],
            ["concepts in a median 38-token document", "9", "6", "4"],
            ["real tokens after document padding", "98.1 %", "96.9 %", "96.1 %"],
            ["tokens predicted with the null plan", "3.5 %", "5.9 %", "8.2 %"],
            ["mean plan staleness (tokens)", "1.5", "2.5", "3.5"],
            ["recall-family query inside the plan · min margin", "100 % · 4", "100 % · 2", "100 % · 0"],
            ["hop: start name / hop count inside the plan", "100 % / 100 %", "100 % / 100 %", "100 % / 75 %"],
            ["bind: constant inside the plan", "50 %", "35 %", "22 %"],
            ["core cost vs v3's core (attention / position-wise)", "1", "0.21 / 0.31", "0.12 / 0.23"],
            ["micro-step vs v3's profile (estimate)", "0.946 (v3)", "0.79–0.85", "0.76–0.82"]],
        v3_pool=dict(slot_norm=[7.69, 7.64, 8.34, 12.08], diag_mean=[0.248, 0.285, 0.330, 0.431], cos_slot4=[0.30, 0.49, 0.73, 1.0], step=11348),
        throughput=[["part", "share of v3's profile", "v4 multiplier", "why"],
                    ["encoder", "0.419", "× 1.16–1.23", "F loop +13–20 %, shared FFN +3 %"],
                    ["core", "0.364", "× 0.18–0.27", "2 pass-rows of 5; 170 vs 256 concepts"],
                    ["decoder", "0.168", "× ≈ 1.10", "two Engram reads"],
                    ["losses", "0.050", "× ≈ 1.0", "small coarse head"],
                    ["micro-step", "1.00 (v3 avg 0.946)", "≈ 0.79–0.85", "≈ 1.1–1.2× v3 (A/B decides)"]],
        losses=[dict(name="NTP(u)", weight=1, formula="CE of token t+1 at t over valid targets (same doc, not PAD)", reaches=["decoder", "core", "encoder", "Engram"]),
                dict(name="CE_coarse", weight=0.05, formula="CE of the next concept's coarse code under −‖q − E⁰_k‖²/τ₀, pair_ok pairs", reaches=["head", "core", "encoder via C"]),
                dict(name="CE_fine", weight=0.1, formula="segment-mean CE of the next fine codes, frame q − E⁰[k_true] (teacher-forced, loss only)", reaches=["head", "core", "encoder via C"]),
                dict(name="VQ_coarse", weight=1, formula="‖sg ĉ − E⁰_k‖² / 32 over complete chunks", reaches=["coarse codebook"]),
                dict(name="VQ_fine", weight=1, formula="segment-mean ‖sg r − e_n‖², r = sg ĉ − sg E⁰_k", reaches=["PQ codebook"])],
        gates=[["G1 · nothing dead", "100 / 200 / 300", "every zero-init branch moved; coarse ppl > 25 % of K", "v3's OPENED check, extended"],
               ["G2 · language", "1,217 / 1,792 / 2,376", "val within v3 + 0.05 nats", "v3: 3.319 / 3.121 / 3.005"],
               ["G3 · recall onset", "1,792", "recall1 ≥ 0.25", "v3 had 0.594"],
               ["G4 · probes alive", "4,050", "carry3 ≥ 0.40 and multi4 > 0", "v3: carry3 0.617, multi4 0.086"],
               ["G5 · second pass works", "24 h", "R2 − R1 multi4 ≥ 2 SE", "the reason M = 2 was kept"],
               ["k = 6 gate (§3.5.5)", "4,050", "vs k = 4: val ≤ +0.03, onset ≤ +300 steps, carry3 ±2 SE, tok/s ≥ 1.03×", "fallback: k = 4, new pooler"]],
        arms=[["k = 4 control", "same v4, 4 tokens per concept", "mandatory", "the compression factor (§3.5.5)"],
              ["k = 8", "8 tokens per concept, T 1,024", "strongly recommended", "is compression 8 affordable?"],
              ["no F loop", "enc_loop = 1", "strongly recommended", "is the encoder loop worth its cost?"],
              ["decoder reads on (if off by default)", "two decoder Engram reads", "mandatory (tiny-scale finding)", "memorisation vs generalisation"],
              ["pooler FFN on / gated conv", "pooler variants", "if time", "selection and composition"],
              ["posterior-code dropout (B)", "plan reads the true next coarse code with p = 0.5", "owner's choice", "RL door B"],
              ["PQ 64×128×4 · pass-1 NCP · temporal int.", "codebook / supervision variants", "if time", "named questions"]],
        v1=v3["v1"], v3=dict(compute=v3["table"]["compute_params"], encoder=v3["table"]["encoder_non_embedding"], core=v3["table"]["concept_module"],
                            decoder=v3["table"]["decoder"], width=256, val_latest=2.486, step_latest=13701))
    json.dump(A, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("wrote", OUT, T["compute_params"], "recommended", rec, variants)


if __name__ == "__main__":
    main()
