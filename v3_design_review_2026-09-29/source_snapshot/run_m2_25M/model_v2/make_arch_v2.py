"""Emit model_v2/arch_v2.json: the v2 architecture in the webapp's data/arch.json shape (params / table / macs / ecfg /
dcfg / reach / S / N) plus the blocks arch.json has no slot for (core, nextlat, losses, nodes, edges, layout3d).
Numbers come from v2_budget.build() for the recommended row C. CPU only.   Run: python3 model_v2/make_arch_v2.py
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v2_budget import build, D, H, KV, DH, DC, PQ_S, PQ_N, PQ_DS, ENG_ROWS, ENG_DIM, ENG_TABLES, V, T, M  # noqa: E402

r = build(enc_ff=704, dec_ff=704, E=16)
pe, pd, pc, pn = (r["parts"][k] for k in ("encoder", "decoder", "core", "nextlat"))
P = r["params"]
expert = 3 * D * pc["d_expert"]

ecfg = dict(vocab_size=V, d_model=D, n_heads=H, n_kv_heads=KV, head_dim=DH, d_ff=704, chunk=4, d_concept=DC, window=8,
            rope_dims=16, rope_base=10000.0, layer_pattern="SSSFSSSF", engram_after=0, engram_orders=[2, 3],
            engram_heads=4, engram_rows=ENG_ROWS, engram_dim=ENG_DIM, conv_kernel=4, gr_rank=32, gr_read_bias=2.0,
            crc_bias=[-1.0, -1.0, 1.0], eps=1e-6)
dcfg = dict(d_model=D, n_heads=H, n_kv_heads=KV, head_dim=DH, d_ff=704, n_layers=4, window=8, rope_dims=16,
            rope_base=10000.0, chunk=4, d_concept=DC, d_u=D, gr_rank=32, gr_read_bias=2.0, layout="confined",
            plan_in="add", enc_reach=63, eps=1e-6)
core = dict(design="D6 LOOPFORMER-MoE-DS (E = 16 sizing)", d=D, n_layers=4, n_heads=H, n_kv_heads=KV, head_dim=DH,
            attention="full causal over the 256 chunk positions, doc-masked, NoPE, sigmoid output gate, no q/k/v convs",
            gr_read="Qwen-style elementwise gated read over [X~, S], rank 32", carry="LN_h(S) with its own gain",
            S0="W_s c_hat (384 -> 256)", X_in="W_in c_hat (384 -> 256), re-read every pass",
            ffn=dict(kind="shared expert pool", E=pc["E"], d_expert=pc["d_expert"], top_k=pc["top_k"], active_width=pc["top_k"] * pc["d_expert"],
                     routers="one per layer (256 -> E)", aux=dict(load_balance=0.01, z_loss=0.001, scope="per window")),
            schedule=dict(K_train="U{8..12}", window=4, remainder="merged into the last window", carry_between_windows="detached",
                          backward="per window; c_hat is a leaf whose grad accumulates across windows"),
            R_eval=6, R_sweep=[1, 2, 4, 6, 8, 12, 16], d_c=DC, pq=dict(S=PQ_S, N=PQ_N, ds=PQ_DS, bits_per_chunk=PQ_S * 7),
            zero_init="block output projections W_o, W_2 of every expert", pass_index_conditioning=False)
nextlat = dict(placement="decoder final hidden h_t = out_norm(D_4) (pre-logit), as in the paper (Sec. 3.2)",
               horizon_d=1, psi="LN[h_t ; rms(E x_{t+1})] -> Linear 512->512 -> GELU -> Linear 512->512 -> GELU -> Linear 512->256 (zero-init) ; h_hat = h_t + f",
               psi_hidden=pn["hidden"], psi_params=pn["psi_params"], lambda_h=1.0, lambda_KL=1.0,
               stop_grad=["target h_{t+1}", "output head E in the KL"], mask="pairs crossing a document boundary",
               posterior_logits="reuse main logits[:, 1:] detached", training_only=True)
losses = [
    dict(name="NTP", weight=1.0, formula="CE(logits_t, x_{t+1})", reaches=["decoder", "core (final window via u_K, c_pred)", "encoder"]),
    dict(name="NCP (per pass)", weight=0.1, formula="mean over all K passes of CE over the 48 PQ segments of chunk j+1 from u_r", reaches=["core (every window)", "encoder (via c_hat)"]),
    dict(name="consistency (LoopFormer, final window)", weight=0.1, formula="mean_{r in last window, r<K} ||sg(u_K) - u_r||^2 / d", reaches=["core (final window)"]),
    dict(name="VQ", weight=1.0, formula="||sg(c_hat) - e_n||^2 (codebook only)", reaches=["codebook"]),
    dict(name="MoE load balance", weight=0.01, formula="E * sum_e f_e * p_e, per layer, per window", reaches=["core routers + experts"]),
    dict(name="MoE router z-loss", weight=0.001, formula="mean logsumexp(router logits)^2", reaches=["core routers"]),
    dict(name="NextLat next-h", weight=1.0, formula="SmoothL1(sg h_{t+1}, h_t + f_psi(LN[h_t; rms E x_{t+1}]))", reaches=["psi", "decoder", "core (final window via u_K, c_pred)", "encoder"]),
    dict(name="NextLat KL", weight=1.0, formula="KL(softmax(sg h_{t+1} E^T) || softmax(h_hat E^T)), E frozen", reaches=["psi", "decoder", "core (final window via u_K, c_pred)", "encoder"]),
]

L = lambda i: f"enc{i}"
nodes = [
    dict(id="tokens", zone="enc", label="tokens", out="B x 1024 (ids)"),
    dict(id="emb", zone="enc", label="Token embedding E (tied)", shape="8192 x 256", params=pe["embed"], counted=False),
    dict(id="engram_tables", zone="enc", label="Engram tables", shape=f"{ENG_TABLES} x {ENG_ROWS} x {ENG_DIM}", params=pe["engram_table"], counted=False),
    dict(id="engram", zone="enc", label="Engram read (after S0)", params=pe["engram_proj"], reads=["L"], writes=["L", "G"]),
] + [dict(id=L(i), zone="enc", label=f"{k}{i}", kind=k, params=pe["layer_S"] if k == "S" else pe["layer_F"],
          reads=["L"] if k == "S" else ["L", "G"], writes=["L", "G"] if k == "S" else ["G"]) for i, k in enumerate("SSSFSSSF")] + [
    dict(id="crc", zone="enc", label="CRC depth-mix", params=pe["crc_router"], out="B x 1024 x 256"),
    dict(id="pool", zone="enc", label="Chunk pool 4 x 256 -> 384, sphere sqrt(384)", params=pe["pool"], out="c_hat B x 256 x 384"),
    dict(id="tok_norm", zone="enc", label="tok = zcRMSNorm(L)", params=pe["tok_norm"], out="B x 1024 x 256"),
    dict(id="core_in", zone="core", label="W_in / W_s 384 -> 256", params=pc["io"], out="X~, S_0: B x 256 x 256"),
] + [dict(id=f"core{l}", zone="core", label=f"Core layer {l + 1}", params=pc["per_layer"], reads=["X~", "S"], writes=["S"],
          detail="gated full attention 4x64 GQA kv2 + router -> top-2 of the shared pool") for l in range(4)] + [
    dict(id="expert_pool", zone="core", label=f"Shared expert pool {pc['E']} x SwiGLU 256-512-256", params=pc["pool"], per_expert=expert),
    dict(id="carry", zone="core", label="carry LN_h(S), detached between windows"),
    dict(id="ncp_head", zone="core", label="NCP head W_q 256 -> 384 + PQ distance head", params=pc["head"]),
    dict(id="codebook", zone="core", label="PQ codebook 48 x 128 x 8", params=pc["codebook"], counted=False),
    dict(id="plan", zone="dec", label="Plan P_t = W_u u + W_c c_pred + o_r", params=pd["plan"]),
] + [dict(id=f"dec{l}", zone="dec", label=f"Confined layer {l + 1}", params=pd["layer"],
          detail="q from private D; k, v from tok only (window 8); gated SwiGLU 704") for l in range(4)] + [
    dict(id="out_norm", zone="dec", label="h = zcRMSNorm(D)", params=pd["out_norm"], out="B x 1024 x 256"),
    dict(id="head", zone="dec", label="logits = h E^T (tied)", out="B x 1024 x 8192"),
    dict(id="nextlat", zone="dec", label="NextLat psi (training only)", params=pn["psi_params"], counted=False, training_only=True),
]
edges = [
    ("tokens", "emb", "ids"), ("emb", "enc0", "e0 = rms(E x)"), ("engram_tables", "engram", "8 rows / token"),
    ("enc0", "engram", "L"), ("engram", "enc1", "L, G"), ("enc1", "enc2", "L, G"), ("enc2", "enc3", "L, G"), ("enc3", "enc4", "L, G"),
    ("enc4", "enc5", "L, G"), ("enc5", "enc6", "L, G"), ("enc6", "enc7", "L, G"), ("enc7", "crc", "e0, L, G"), ("crc", "pool", "mix"),
    ("pool", "core_in", "c_hat"), ("enc7", "tok_norm", "L"), ("core_in", "core0", "X~, S_0"), ("core0", "core1", "S"),
    ("core1", "core2", "S"), ("core2", "core3", "S"), ("core3", "carry", "S"), ("carry", "core0", "S (next pass)"),
    ("expert_pool", "core0", "top-2"), ("expert_pool", "core1", "top-2"), ("expert_pool", "core2", "top-2"), ("expert_pool", "core3", "top-2"),
    ("carry", "ncp_head", "u_r (every pass)"), ("codebook", "ncp_head", "sg(e)"), ("carry", "plan", "u_K (final pass)"),
    ("ncp_head", "plan", "c_pred"), ("tok_norm", "plan", "tok"), ("plan", "dec0", "D0 = tok + P"), ("tok_norm", "dec0", "k, v"),
    ("dec0", "dec1", "D"), ("dec1", "dec2", "D"), ("dec2", "dec3", "D"), ("dec3", "out_norm", "D"), ("out_norm", "head", "h"),
    ("emb", "head", "E (tied)"), ("out_norm", "nextlat", "h_t, sg h_{t+1}"), ("emb", "nextlat", "sg E x_{t+1}"),
]
edges = [dict(src=a, dst=b, tensor=t, training_only=(b == "nextlat")) for a, b, t in edges]
layout3d = dict(zones=dict(enc=dict(x=-10, color="#3b78c2", title="Encoder", sub=f"{P['encoder']/1e6:.2f}M + {pe['engram_table']/1e6:.1f}M memory"),
                           core=dict(x=0, color="#2e9c8c", title="Concept module v2 (D6)", sub=f"{P['core']/1e6:.2f}M · 16 experts · R 6"),
                           dec=dict(x=10, color="#e07b39", title="Decoder (confined) + NextLat", sub=f"{P['decoder']/1e6:.2f}M + NextLat {pn['psi_params']/1e6:.2f}M (train only)")),
                note="same zone positions and colours as webapp/js/view3d.js buildModel(); the core zone gains a pass ring (K passes, windows of 4, detach marks) and an expert shelf")

params = dict(enc_layer_S=pe["layer_S"], enc_layer_F=pe["layer_F"], enc_layer_S_attn=pe["layer_S_attn"], enc_layer_S_ffn=pe["layer_S_ffn"],
              enc_layer_S_gr=pe["layer_S_gr"], enc_layer_F_gr=pe["layer_F_gr"], engram_table=pe["engram_table"], engram_proj=pe["engram_proj"],
              crc_router=pe["crc_router"], pool=pe["pool"], tok_norm=pe["tok_norm"], embed=pe["embed"],
              core_total=pc["params"], core_layer_dense=pc["per_layer"], core_expert=expert, core_pool=pc["pool"], core_io=pc["io"],
              ncp_head=pc["head"], codebook=pc["codebook"],
              dec_total=pd["total"], dec_plan=pd["plan"], dec_layer=pd["layer"], dec_attn=pd["attn"], dec_ffn=pd["ffn"], nextlat_psi=pn["psi_params"])
table = dict(encoder_non_embedding=P["encoder"], engram_table=P["engram_table"], token_embedding=P["embed"], concept_module=P["core"],
             codebook=P["codebook"], decoder=P["decoder"], compute_params=P["compute"], stored_total=P["stored"],
             nextlat_psi_training_only=P["nextlat_psi_training_only"])
out = dict(version="HRM Thinker v2 · 19M · d 256 · Engram 80M · D6 core (E 16) · decoder + NextLat",
           convention="compute = encoder non-embedding + concept module without codebook + decoder (code/model.py:param_table)",
           params=params, table=table, macs=r["macs"], ecfg=ecfg, dcfg=dcfg, core=core, nextlat=nextlat, losses=losses,
           reach_enc=63, reach_dec=70, S=PQ_S, N=PQ_N, T=T, M=M, nodes=nodes, edges=edges, layout3d=layout3d,
           est_tok_s_upper_bound=r["est_tok_s"])
budget = json.load(open(os.path.join(HERE, "v2_budget.json")))
out["rows"] = {k: dict(cfg=v["cfg"], params={x: v["params"][x] for x in ("encoder", "core", "decoder", "compute", "stored")},
                       share=v["params"]["share"], fwd_eval=v["macs"]["fwd_eval"], core_share=v["macs"]["core_share_eval"],
                       train_vs_today=v["macs"]["train_vs_today"], tok_s=v["est_tok_s"]) for k, v in budget.items()}
out["active"] = budget["C_E16"]["active"]; out["mac_share"] = budget["C_E16"]["mac_share"]; out["r_sweep"] = budget["C_E16"]["r_sweep"]
out["v1"] = dict(compute=25052781, encoder=15510573, core=2059064, decoder=7532296, engram_table=50331264, stored=78578925,
                 macs_train=28.023e6, macs_eval_R4=28.76e6, tok_s=[1650, 1815], val_best=2.7496)
json.dump(out, open(os.path.join(HERE, "arch_v2.json"), "w"), indent=1)
json.dump(out, open(os.path.join(HERE, "..", "webapp", "data", "arch_v2.json"), "w"))   # read by the v2 pages of the dashboard
print("wrote model_v2/arch_v2.json |", len(nodes), "nodes,", len(edges), "edges | compute", P["compute"], "stored", P["stored"])
