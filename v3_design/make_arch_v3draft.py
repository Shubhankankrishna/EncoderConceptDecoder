"""Build webapp/data/arch_v3d.json (the draft_v3.md architecture pages: overview, 2D figures, 3D model).

Every parameter count is instantiated from v3_ref.HRMThinkerV3 (the acceptance harness); MACs come from v3_budget's
analytical counters at the final widths (encoder ff 1024 + 2 Engram reads, core own ff 736 + shared 1024, decoder ff 1024).
Throughput rows and gates are quoted from draft_v3.md §7.2 / §9. CPU only; writes only into run_m2_25M/webapp/data.
Run: nice -n 19 python3 make_arch_v3draft.py
"""
import json, os, shutil
import torch
import v3_ref as R
import v3_budget as B

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB = os.path.join(ROOT, "run_m2_25M", "webapp", "data")

torch.manual_seed(0)
m = R.HRMThinkerV3(engram_rows=11)                      # tiny Engram table; the real 80M one is added analytically
n = lambda mod: sum(p.numel() for p in mod.parameters())
e, c, d = m.enc, m.core, m.dec
tiny = e.engram.table.weight.numel()
t = R.param_table(m)
t["engram_table"] = 8 * 208333 * 48
t["stored"] = t["compute"] + t["engram_table"] + t["embedding"] + t["codebook"]
l0, lc, ld = e.layers[0], c.layers[0], d.layers[0]
P = dict(
    enc_layer_S=n(l0), enc_layer_S_attn=n(l0.attn), enc_layer_S_ffn=n(l0.ffn), enc_layer_S_gr=n(l0) - n(l0.attn) - n(l0.ffn),
    enc_layer_F=n(e.layers[3]), enc_layer_F_attn=n(e.layers[3].attn), enc_layer_F_ffn=n(e.layers[3].ffn),
    enc_layer_F_gr=n(e.layers[3]) - n(e.layers[3].attn) - n(e.layers[3].ffn),
    engram_read1=n(e.engram) - tiny, engram_read2=n(e.engram2) - tiny,      # engram2 shares the table: count it once
    engram_table=t["engram_table"], crc_router=n(e.crc_router), pool=n(e.pool), tok_norm=n(e.tok_norm),
    core_layer=n(lc), core_attn=n(lc.attn), core_own_ffn=n(lc.ffn), core_gr=n(lc.read_attn) + n(lc.read_ffn),
    core_writes=n(lc.write_attn) + n(lc.write_ffn), core_shared=n(c.shared), core_table=c.table.numel(),
    ncp_head=n(c.Wq) + c.log_tau.numel(), core_out_norm=n(c.out_norm), codebook=t["codebook"],
    dec_layer=n(ld), dec_layer_ffn=n(ld.ffn), dec_plan=n(d.plan), dec_out_norm=n(d.out_norm), token_embedding=t["embedding"],
)
assert P["engram_read1"] == 215042 and P["engram_read2"] == 214018
assert t["compute"] == 18986871 and t["stored"] == 101116663, t

enc_m = B.encoder(d_ff=1024, engram_reads=2)
dec_m = B.decoder(d_ff=1024)
core_m = B.core(d_ff=736, shared_ff=1024)
macs = dict(encoder=enc_m["macs"], core_eval=core_m["macs_eval"], core_train=core_m["macs_train"], decoder=dec_m["macs"],
            head=dec_m["head_macs"])
macs["fwd_eval"] = macs["encoder"] + macs["core_eval"] + macs["decoder"] + macs["head"]
macs["fwd_train"] = macs["encoder"] + macs["core_train"] + macs["decoder"] + macs["head"]
macs["note"] = "analytical (v3_budget.py); F-layer 6-head score MACs not included (~+2 %)"

cfg = m.ecfg
A2 = json.load(open(os.path.join(WEB, "arch_v2.json")))
out = dict(
    source="draft_v3.md (2026-09-29) · counts from v3_design/v3_ref.py · MACs from v3_design/v3_budget.py",
    status="Draft r2: revised after the external deep review (draft_v3_review.md) · not implemented · not trained",
    revision="r2",
    ecfg=dict(vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_heads=cfg.n_heads, n_heads_full=6, n_kv_heads=cfg.n_kv_heads,
              head_dim=cfg.head_dim, d_ff=cfg.d_ff, chunk=cfg.chunk, d_concept=cfg.d_concept, window=cfg.window,
              rope_dims=cfg.rope_dims, layer_pattern=cfg.layer_pattern, engram_orders=list(cfg.engram_orders),
              engram_heads=cfg.engram_heads, engram_rows=208333, engram_dim=cfg.engram_dim, conv_kernel=cfg.conv_kernel,
              gr_rank=cfg.gr_rank, engram_reads=[0, 4]),
    core=dict(n_layers=4, heads=6, kv=2, own_ff=736, shared_ff=1024, pairs=[[a, b] for (a, b) in R.PAIRS],
              schedules={k: [[a, b] for (a, b) in v] for k, v in R.SCHED.items()}, sites=["γ_C attn", "γ_S attn", "α attn", "γ_C ffn", "γ_S ffn", "α own", "α shared"],
              main="3", shortcut=["1", "2a", "2b"], shortcut_p={"1": 0.5, "2a": 0.25, "2b": 0.25}),
    pq=dict(S=32, N=128, ds=8, bits=224),
    reach_enc=63, reach_dec=70,
    table=dict(encoder_non_embedding=t["encoder"], concept_module=t["core"], decoder=t["decoder"], compute_params=t["compute"],
               engram_table=t["engram_table"], token_embedding=t["embedding"], codebook=t["codebook"], stored_total=t["stored"]),
    params=P, macs=macs,
    throughput=[
        ["default (M 3 + shortcut, shared FFN parallel; slots incl. PAD)", "22.6M / 25.5M", 18, "1,290–1,770", "211–290M", "11.1–15.2"],
        ["lever: shared FFN once per pass", "20.9M / 22.9M", 18, "1,410–1,940", "231–318M", "12.1–16.7"],
        ["lever: t-only arm (no shortcut)", "22.6M / 22.7M", 12, "1,520–2,090", "249–342M", "13.0–17.9"],
        ["lever: main M 2 + shortcut", "20.7M / ≈22.7M", 12, "≈1,520–2,090", "≈249–342M", "≈13.0–17.9"],
        ["rejected: shortcut NTP through decoder", "22.6M / 32.0M", 18, "1,070–1,470", "175–241M", "9.2–12.6"],
        ["v1 measured (d 384, 25.05M)", "— / 28.0M", 2.5, "1,650–1,815", "132.7M in 24 h", "5.3"],
        ["v2 live (18.83M nominal, 15.69M active)", "— / 27.9M-eq", 40, "1,100–1,400", "≈200M", "≈10.6"],
    ],
    losses=[
        dict(name="L_NTP(u₃)", weight=1, formula="CE of main-trajectory logits on valid targets (same doc, not PAD)", reaches=["decoder", "core", "encoder L", "encoder G via C", "table · shared"]),
        dict(name="L_NCP(u₃)", weight=0.1, formula="per-segment CE of next chunk's PQ code, 32 segments, clean pairs", reaches=["core", "encoder G via C", "table · shared"]),
        dict(name="L_NCP(u_s)", weight=0.05, formula="same head on the shortcut endpoint (M 1 / 2a / 2b)", reaches=["core", "encoder G via C", "table · shared"]),
        dict(name="L_scons", weight=0.1, formula="MSE(rms S_s, sg rms S₃) over nonempty chunks · gain-free (cannot shrink out_norm γ)", reaches=["core", "encoder G via C", "table · shared"]),
        dict(name="L_VQ", weight=1, formula="per-segment squared error on detached c_hat, complete chunks", reaches=["codebook"]),
    ],
    gates=[
        ["E1 recall onset (primary)", "recall1 ≥ 0.25 by step 1,792 (far below v1's 0.82)", "v1 0.336 at 1,217 · v2 ≈2,250", "< 0.05 → notify, arm D"],
        ["E2 val per language", "each ≤ v1 + 0.10 at 1,217 / 1,792 / 2,376", "v1 3.227 · 3.032 · 2.913", "> +0.20 twice → notify"],
        ["E3 stability", "gn ≤ 1, no NONFINITE / guard", "v1 gn ≤ 0.38", "monitor policy"],
        ["E4 throughput", "≥ 1,290 slots/s (floor 1,200)", "v1 1,650–1,815", "< 1,200 twice → notify"],
        ["E5 encoder G contribution", "report only (not a bypass test)", "init G share 0.787", "read with F_off, plan/core effects"],
        ["M1 full recipe vs 1 pass", "recall1 +10 pts, val −0.03, paired per-item stats", "full 256-item eval", "recipe fails; ladder says why"],
        ["Ladder A1 / A3 / T3 / Full", "A3 vs A1 repetition · T3 vs A3 table · Full vs T3 shortcut", "per example and per hour", "none beats A1 → simpler model"],
        ["M2 loop geometry", "diagnostic: angles, ρ, norms (v1 1.25°, ρ 0.99)", "neither necessary nor sufficient", "renorm only if no gain + v1 signature"],
        ["M3 gradient audit", "diagnostic: weighted sums, norm of sum, cosine", "fixed batches, all families", "anti-aligned → halve λ on resume"],
        ["M4 elasticity", "report M 1 / 2a / 2b / 3", "LoopFormer: 2b > 2a", "report only"],
        ["M5 intervention effects", "core / u / c_pred / plan-swap / engram / shared / table / F off", "fixed-checkpoint, not upper bounds", "report only"],
    ],
    arms=[
        ["A1 plain one-pass", "= fixed M 1: no table, no shortcut (18,943,863)", "v1's", "baseline", "recommended"],
        ["A3 plain three-pass", "3 passes, no table, no shortcut (18,943,863)", "v1's", "A3 vs A1: does repetition help?", "recommended"],
        ["T3 t-conditioned (TMLT)", "rows (t, ⅓) only, no shortcut (18,965,367)", "v1's", "T3 vs A3: does the table help?", "recommended"],
        ["Full (the default)", "(t, Δ) table + shortcut (18,986,871)", "full", "Full vs T3: is the elastic bundle worth +50 % core work?", "default run's first 2,400 steps"],
        ["own-FFN 992, no shared", "matrices matched; α_sh dropped (18,980,727)", "full", "does the Mobius hybrid help, at equal params and time?", "after the ladder"],
        ["carry renorm", "zcRMSNorm(S) at pass boundaries", "full", "does renorm help a failed loop?", "no gain + v1 geometry"],
        ["answer (D)", "none", "full + L_answer", "earlier recall onset?", "E1 warns / fails"],
    ],
    v1=dict(**{k: A2["v1"][k] for k in ("compute", "encoder", "core", "decoder", "engram_table", "macs_train")}, width=384,
            core_desc="1 shared layer × R U{1..4}", val_24h=2.755),
    v2=dict(compute=A2["table"]["compute_params"], encoder=A2["table"]["encoder_non_embedding"], core=A2["table"]["concept_module"],
            decoder=A2["table"]["decoder"], macs_train=A2["macs"]["fwd_train"], width=256,
            core_desc="D6: 4 layers × K 8–12, 16 experts, detached windows", val_2942=3.125, recall_2942=0.539),
)
with open(os.path.join(WEB, "arch_v3d.json"), "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
shutil.copyfile(os.path.join(ROOT, "draft_v3.md"), os.path.join(WEB, "draft_v3.md"))
print("wrote arch_v3d.json", t["compute"], {k: round(v / 1e6, 2) for k, v in macs.items() if isinstance(v, float)})
