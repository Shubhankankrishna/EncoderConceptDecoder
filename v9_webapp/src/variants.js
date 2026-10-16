/* V9 Atlas · variant descriptors. One entry per architecture arm that the queue trains or tests; the 2D figure, the 3D scene,
   the gallery and the Live tab all read this table, so a variant is defined once.

   Descriptor schema
     id, label, sub      short names
     stage, queue, tier  S1..S6 · Q1..Q6 · "mandatory" | "full" (full = the optional column, not in the 24 h queue)
     body                "hier" (E3 -> pool -> concept core -> D3) | "flat" (one 10-block Transformer)
     core                "dense" (C4) | "recurrent" (HRM 2x(2L+1H)) | null for flat
     K                   null (no codes, S1) | 1 | 4 | 8
     params              { total, trainable, note }
     train               part id -> "T" trains / is active · "F" frozen · "N" absent   (parts: PARTS below)
     losses              which loss boxes are drawn: slot1 / slot2 / slot3 ids
     adapter             null | { kind: "lora" | "lsl", rank, targets, where, gate }
     info                per-variant overrides of the part inspector text
     runs                regexp (string) matching the run folders that train this arm, for the live join
     source              draft section
   Numbers come from draft_v9.md; anything that is a design choice rather than a measurement says so. */
(function () {
"use strict";
const PARTS = ["tokens", "emb", "enc", "H", "pool", "cin", "core", "bh0", "codes", "bhk", "pq", "Q", "hyp", "sel", "bestm", "plan", "teacher",
  "lwta", "lsel", "ldist", "lncp", "lpi", "lv", "dsum", "dec", "head", "ltok", "vtau", "lora", "gate", "flatblk"];
function tm(o) { const m = {}; PARTS.forEach(p => { m[p] = "N"; }); Object.keys(o).forEach(s => o[s].split(/\s+/).filter(Boolean).forEach(p => { m[p] = s; })); return m; }

/* ---- training maps */
const S1_TRAIN = "tokens emb enc H pool cin core bh0 pq Q plan dsum dec head ltok lncp";
const T_S1 = tm({ T: S1_TRAIN });
const T_S1_A0 = tm({ T: S1_TRAIN.replace(" lncp", "") });
const T_FLAT = tm({ T: "tokens emb flatblk head ltok" });
// S2: the whole foundation, the PQ heads, codebook Q and the port are frozen; the new core, codes and selector train (draft §6.1)
const S2_COMMON = "cin core bh0 plan ltok lwta lsel ldist";
const S2_FROZEN = "tokens emb enc H pool pq Q dsum dec head teacher";
const T_S2_K1 = tm({ T: "cin core bh0 plan ltok lwta ldist hyp", F: S2_FROZEN });          // K = 1: one concept, no codes, no selector, L_sel = 0
const T_S2_K8 = tm({ T: S2_COMMON + " codes bhk hyp sel bestm", F: S2_FROZEN });
// S3: from the promoted S2 model
const T_S3A_DEC = tm({ T: "dec dsum ltok", F: "tokens emb enc H pool cin core bh0 codes bhk pq Q hyp sel bestm plan head" });
const T_S3A_ENC = tm({ T: "emb enc dec dsum ltok", F: "tokens H pool cin core bh0 codes bhk pq Q hyp sel bestm plan head" });
const T_S3B = tm({ T: "tokens emb enc H pool cin core bh0 codes bhk pq Q hyp sel bestm plan dsum dec head ltok lwta lsel ldist", F: "teacher" });
// S4: everything frozen except the selector and the two small heads (primary) or the decoder LoRA (token baselines)
const T_S4_EI = tm({ T: "sel vtau lpi lv", F: "tokens emb enc H pool cin core bh0 codes bhk pq Q hyp bestm plan dsum dec head" });
const T_S4_TOK = tm({ T: "lora ltok", F: "tokens emb enc H pool cin core bh0 pq Q plan dsum dec head" });
// S6 (LSL family): adapters train, the base is frozen
const S6_BASE_FROZEN = "tokens emb enc H pool cin core bh0 pq Q plan dsum dec head";
const T_S6_LSL = tm({ T: "lora gate ltok", F: S6_BASE_FROZEN });
const T_S6_OPEN = tm({ T: "lora ltok", F: S6_BASE_FROZEN });
const T_S6_VQ = tm({ T: "Q pq ltok", F: "tokens emb enc H pool cin core bh0 plan dsum dec head" });
const T_S6_FULL = tm({ T: S6_BASE_FROZEN + " ltok" });

/* Inspector text for parts that only some variants have (merged into the 2D INFO table by app.js). */
const BASE_INFO = {
  flatblk: { t: "Flat Transformer block", cat: ["Flat model", "--enc"], d: "One ordinary causal Transformer block over tokens. All ten blocks see tokens only: there is no pooling, no concept stream and no plan, so a chunk boundary means nothing to this model. Depth is matched to the hierarchy (3 + 4 + 3 = 10).",
    kv: [["blocks", "10"], ["width", "d = 128, 4 heads"], ["FFN", "400 ×9, 384 on the last"], ["position", "RoPE, pre-RMSNorm"]], src: "§5.1 · model.py Flat" },
  lncp: { t: "L_NCP · next-concept prediction", cat: ["Loss", "--loss"], d: "Mean-squared error between the predicted next pooled concept and the stop-gradient true one, on scored target chunks.", eq: "L_NCP = balanced mean ‖ĉ_j − sg(C_j+1)‖²", src: "§5.1" },
  lora: { t: "LoRA adapter", cat: ["Adapter", "--dec"], d: "Low-rank update on the attention projections of the block it sits on. The base weights stay frozen; only the adapter trains.", eq: "W' = W + B·A      B ∈ R^(d×r), A ∈ R^(r×d)", src: "§8.2 · §10.6" },
  gate: { t: "LSL gate", cat: ["Adapter", "--loss"], d: "A per-matrix switch: the adapter is applied only where this phase's inputs are denser than generic inputs. The gate reads the matrix's own input after a JL projection.", eq: "g(x) = 1[ Φpos(x) > Φneg(x) ]\ns_t = .3·g_t + .7·s_(t−1) ;  adapter on iff s_t > .5", src: "§10.1" },
  vtau: { t: "Value and terminal heads", cat: ["Selector", "--sel"], d: "Two 128 → 1 linear heads on u⁽⁰⁾.", src: "§8.2" },
  lpi: { t: "S4 policy loss", cat: ["Loss", "--loss"], d: "Trains the selector from search or verifier feedback.", src: "§8.2" },
  lv: { t: "S4 value and terminal loss", cat: ["Loss", "--loss"], d: "Trains the value and terminal heads from verifier outcomes.", src: "§8.2" },
};

const NO_LOSS = { s1: null, s2: null, s3: null };
const V = [];
function add(o) { o.parts = o.train; V.push(o); return o; }

/* ================================================================ S1 · Q1 */
add({ id: "A_delayed", label: "A_delayed", sub: "hierarchy · delayed NCP (the S2 anchor)", stage: "S1", queue: "Q1", tier: "mandatory", body: "hier", core: "dense", K: null,
  params: { total: 2448640, trainable: 2448640, note: "V8-D shape: E3 / C4 / D3" }, train: T_S1, losses: { s1: "lncp" },
  alpha: "α(u) = 0 until update 6,000, ramps to 0.1 by 8,000", runs: "^S1_A_delayed_s\\d+$", source: "§5.1–§5.2",
  info: { lncp: { t: "L_NCP · delayed α", d: "One MSE next-concept prediction through the PQ heads, switched on late (α = 0 until update 6,000, linear to 0.1 by 8,000). This arm is the S2 teacher and the NCP side of H5.", eq: "L = L_token + α(u)·L_NCP + L_VQ" } } });
add({ id: "A_alpha0", label: "A_alpha0", sub: "hierarchy · NCP off (control)", stage: "S1", queue: "Q1", tier: "mandatory", body: "hier", core: "dense", K: null,
  params: { total: 2448640, trainable: 2448640, note: "same shape as A_delayed" }, train: T_S1_A0, losses: { s1: "lncp" }, alpha: "α = 0 for all 24,000 updates", runs: "^S1_A_alpha0_s\\d+$", source: "§5.2",
  info: { lncp: { t: "L_NCP · off (α = 0)", d: "Same network, no NCP term. The concept stream, pooling and plan path are still in the graph, so any difference from A_delayed comes from the NCP loss alone. This is the paired control for H5.", eq: "L = L_token + L_VQ        (α = 0)" } } });
add({ id: "A_flat", label: "A_flat", sub: "matched 10-block Transformer (control)", stage: "S1", queue: "Q1", tier: "mandatory", body: "flat", core: null, K: null,
  params: { total: 2450048, trainable: 2450048, note: "10 blocks, FF400 ×9 + FF384" }, train: T_FLAT, losses: {}, runs: "^S1_A_flat_s\\d+$", source: "§5.1–§5.2",
  info: {} });
add({ id: "A_const_hncp", label: "A_const_hncp", sub: "K = 4 codes in S1, constant α = 1", stage: "S1", queue: "Q1", tier: "full", body: "hier", core: "dense", K: 4,
  params: { total: 2448640 + 2628, trainable: 2448640 + 2628, note: "adds K = 4 codes: e_k 64, W_z 2,048, W_π 516 (computed from the §6.1 formulas)" }, train: tm({ T: S1_TRAIN + " codes bhk hyp sel bestm" }), losses: { s1: "lncp" },
  alpha: "α = 1 constant (P-blur-1 test)", runs: null, source: "§5.2", note: "Not in the 24 h queue. The only test of P-blur-1." });
add({ id: "A_delayed_prefixlm", label: "A_delayed_prefixlm", sub: "prompt read bidirectionally (PrefixLM)", stage: "S1", queue: "Q1", tier: "full", body: "hier", core: "dense", K: null,
  params: { total: 2448640, trainable: 2448640, note: "same weights, different attention mask" }, train: T_S1, losses: { s1: "lncp" }, prefixlm: true, alpha: "delayed, as A_delayed", runs: null, source: "§5.2",
  note: "Not in the 24 h queue. Adds the PrefixLM half of HRM-Text's response-only + PrefixLM recipe." });

/* ================================================================ S2 · Q3 */
const S2_INFO = { teacher: { t: "Frozen S1 teacher", d: "The accepted A_delayed checkpoint. It supplies C_j+1 and its own (u, A, ĉ) for distillation. The student core is initialised from it (dense) or freshly initialised (recurrent)." } };
add({ id: "S2_V8repro", label: "S2 V8-repro", sub: "K = 1 · closed loop off · implementation check", stage: "S2", queue: "Q3", tier: "mandatory", body: "hier", core: "dense", K: 1,
  params: { total: 2448640, trainable: 869504, note: "V8 REFIT replays: trainable dense core 869,504" }, train: T_S2_K1, losses: { s2: "lwta", s3: "ldist" },
  runs: "^S2_V8repro_dense_s\\d+$|^S2_replay_dense", source: "§6.1, §13.2", closedLoop: false, info: S2_INFO, note: "6,000 updates, seed 811, V8's own teacher: checks that the V9 code reproduces V8." });
add({ id: "S2_K1_dense", label: "S2 K1 dense", sub: "one committed concept = V8 objective (control)", stage: "S2", queue: "Q3", tier: "mandatory", body: "hier", core: "dense", K: 1,
  params: { total: 2448640, trainable: 869504, note: "dense C4 core trainable; no new parameters at K = 1" }, train: T_S2_K1, losses: { s2: "lwta", s3: "ldist" },
  runs: "^S2_K1_dense_s\\d+$", source: "§6.1–§6.3", closedLoop: true, info: S2_INFO });
add({ id: "S2_K8_dense", label: "S2 K8 dense", sub: "8 committed concepts + selector (main claim)", stage: "S2", queue: "Q3", tier: "mandatory", body: "hier", core: "dense", K: 8,
  params: { total: 2448640 + 3208, trainable: 869504 + 3208, note: "+3,208: e_k 128, W_z 2,048, W_π 1,032" }, train: T_S2_K8, losses: { s2: "lwta", s2b: "lsel", s3: "ldist" },
  runs: "^S2_K8_dense_s\\d+$", source: "§6.1–§6.3", closedLoop: true, info: S2_INFO });
add({ id: "S2_K1_rec", label: "S2 K1 recurrent", sub: "HRM-recurrent core, one code (control)", stage: "S2", queue: "Q3", tier: "mandatory", body: "hier", core: "recurrent", K: 1,
  params: { total: 2055424, trainable: 476288, note: "recurrent core 476,288 vs dense 869,504" }, train: T_S2_K1, losses: { s2: "lwta", s3: "ldist" },
  runs: "^S2_K1_rec_s\\d+$", source: "§6.1", closedLoop: true, info: S2_INFO, note: "Q3b: the recurrent tier (runs after Q2)." });
add({ id: "S2_K8_rec", label: "S2 K8 recurrent", sub: "HRM-recurrent core, 8 codes", stage: "S2", queue: "Q3", tier: "mandatory", body: "hier", core: "recurrent", K: 8,
  params: { total: 2055424 + 3208, trainable: 476288 + 3208, note: "recurrent core + 3,208 code parameters" }, train: T_S2_K8, losses: { s2: "lwta", s2b: "lsel", s3: "ldist" },
  runs: "^S2_K8_rec_s\\d+$", source: "§6.1, §6.2", closedLoop: true, info: S2_INFO, note: "Q3b: B_H is the last fH visit (visit 6), sharing weights with visit 3." });

/* ================================================================ S3 · Q4 */
const S3_INFO = { teacher: { t: "Original frozen teacher", d: "S3b distils the code-free path toward the original S1 teacher's outputs (δ = 0.1, a chosen weight), computed by the teacher's own encoder, not the adapted one." } };
add({ id: "S3a_dec", label: "S3a-dec", sub: "decoder adapts, everything else frozen", stage: "S3", queue: "Q4", tier: "mandatory", body: "hier", core: "dense", K: 8,
  params: { total: 2448640 + 3208, trainable: null, note: "decoder blocks + final norm + γ train (count not stated in the draft)" }, train: T_S3A_DEC, losses: { s2: "ltok" },
  runs: "^S3_s\\d+_dec$", source: "§7.2", closedLoop: true, info: S3_INFO, note: "The tied embedding stays frozen, so the concept coordinates do not move. H11 predicts it keeps code authority." });
add({ id: "S3b", label: "S3b", sub: "joint: everything trains (your sequence as stated)", stage: "S3", queue: "Q4", tier: "mandatory", body: "hier", core: "dense", K: 8,
  params: { total: 2448640 + 3208, trainable: 2448640 + 3208, note: "all weights; WTA, selector and δ = 0.1 distillation stay on" }, train: T_S3B, losses: { s2: "lwta", s2b: "lsel", s3: "ldist" },
  runs: "^S3_s\\d+_joint$", source: "§7.2", closedLoop: true, info: S3_INFO, note: "H11 predicts it loses ≥ 20% of its code authority (ETA co-training)." });
add({ id: "S3a_enc", label: "S3a-enc", sub: "encoder + decoder adapt (optional)", stage: "S3", queue: "Q4", tier: "full", body: "hier", core: "dense", K: 8,
  params: { total: 2448640 + 3208, trainable: null, note: "adds encoder blocks, enc_norm and the tied embedding" }, train: T_S3A_ENC, losses: { s2: "ltok" },
  runs: null, source: "§7.2", closedLoop: true, info: S3_INFO, note: "Not in the 24 h queue. Input drift is reported as the change in encoder-pooled concepts." });

/* ================================================================ S4 · Q5 */
add({ id: "S4_EI", label: "S4 expert iteration", sub: "selector + V + τ heads (primary)", stage: "S4", queue: "Q5", tier: "mandatory", body: "hier", core: "dense", K: 8,
  params: { total: 2448640 + 3208, trainable: 1290, note: "W_π 1,032 + two 128→1 heads of 129 each" }, train: T_S4_EI, losses: { s2: "lpi", s2b: "lv" },
  runs: "^S4_s\\d+_EI", source: "§8.2", closedLoop: true,
  info: { lpi: { t: "L_π · visit-count targets", d: "Cross-entropy from the MCTS visit distribution to the selector, over 4,096 fresh problems per iteration.", eq: "L_π = Σ_s CE( N(s,·)/ΣN , π_θ(·|s) )" },
          lv: { t: "L_V and L_τ · outcome and terminal heads", d: "V_ψ regresses the verifier outcome of the played trajectory; τ_ψ predicts whether the next chunk is DONE. Both are 128 → 1 heads on u⁽⁰⁾, versioned per iteration.", eq: "L_V = Σ (V_ψ(s) − o(s))²\nL_τ = Σ BCE(τ_ψ(s), next = DONE)" },
          vtau: { t: "Value and terminal heads", d: "Two 128 → 1 linear heads on u⁽⁰⁾. They replace the verifier at interior search nodes and are what S5 reads.", kv: [["V_ψ", "129 parameters"], ["τ_ψ", "129 parameters"]], eq: "V_ψ = σ(w_v·u⁽⁰⁾ + b)    τ_ψ = σ(w_τ·u⁽⁰⁾ + b)" } },
  note: "Capacity is disclosed: 1,290 trainable parameters against 24,576 for the token baselines." });
add({ id: "S4_RFT", label: "S4 token RFT", sub: "decoder LoRA r16 · rejection-sampling FT (baseline)", stage: "S4", queue: "Q5", tier: "mandatory", body: "hier", core: "dense", K: 8,
  params: { total: 2448640 + 3208, trainable: 24576, note: "3 blocks × 2 matrices × 16 × (128+128)" }, train: T_S4_TOK, losses: { s2: "ltok" },
  adapter: { kind: "lora", rank: 16, targets: "q, v", where: ["dec"], gate: false }, codeFree: true, runs: "^S4_s\\d+_rft", source: "§8.2",
  info: { ltok: { t: "L_token on verified successes", d: "Sample complete answers at T = 1, keep the verified successes, fine-tune the LoRA on them. Code-free plan from the same checkpoint's core." } }, note: "Code-free mode: the plan is (ĉ⁽⁰⁾, A⁽⁰⁾), so the code path is unused." });
add({ id: "S4_GRPO", label: "S4 token GRPO", sub: "same adapter, policy gradient (baseline)", stage: "S4", queue: "Q5", tier: "mandatory", body: "hier", core: "dense", K: 8,
  params: { total: 2448640 + 3208, trainable: 24576, note: "same rank-16 q,v LoRA as RFT" }, train: T_S4_TOK, losses: { s2: "ltok" },
  adapter: { kind: "lora", rank: 16, targets: "q, v", where: ["dec"], gate: false }, codeFree: true, runs: "^S4_s\\d+_grpo", source: "§8.2",
  info: { ltok: { t: "L_GRPO · group-relative policy gradient", d: "The standard method, on the same adapter. ETA found GRPO fails on sparse compositional rewards; it is included because it is the standard." } }, note: "Code-free mode." });
add({ id: "S4_REINFORCE", label: "S4 REINFORCE", sub: "policy gradient on the selector (reading b)", stage: "S4", queue: "Q5", tier: "mandatory", body: "hier", core: "dense", K: 8,
  params: { total: 2448640 + 3208, trainable: 1032, note: "selector only" }, train: tm({ T: "sel lpi", F: "tokens emb enc H pool cin core bh0 codes bhk pq Q hyp bestm plan dsum dec head" }), losses: { s2: "lpi" },
  runs: "^S4_s\\d+_reinforce", source: "§8.2",
  info: { lpi: { t: "REINFORCE on the selector", d: "Policy gradient with the verifier reward, on the selector's 1,032 parameters only.", eq: "∇ = E[ (o − b) ∇ log π_θ(k|s) ]" } } });

/* ================================================================ S6 · Q2 */
const LSL_GATE = "g(x) = 1[ Φpos(x) > Φneg(x) ]   (diagonal GMMs on JL-projected inputs)\ns_t = .3·g_t + .7·s_(t−1) ;  adapter on iff s_t > .5";
const S6_INFO = { lora: { t: "LoRA adapter", d: "A new adapter per phase on the frozen S1 anchor." }, gate: { t: "LSL gate", d: "Per-matrix gate: the adapter switches on only where the phase's inputs are denser than generic inputs.", eq: LSL_GATE } };
add({ id: "S6_LSL_v8", label: "LSL-v8", sub: "rank 4 · q,v · JL 32 · reproduces V8", stage: "S6", queue: "Q2", tier: "full", body: "hier", core: "dense", K: null,
  params: { total: 2448640, trainable: null, note: "rank-4 q,v adapters on 10 gated blocks; Φpos 4 / Φneg 8" }, train: T_S6_LSL, losses: {},
  adapter: { kind: "lsl", rank: 4, targets: "q, v", where: ["enc", "core", "dec"], gate: { jl: 32, pos: 4, neg: 8, smooth: false } }, runs: null, source: "§10.6", info: S6_INFO,
  note: "Full column (not in the 24 h queue). 10 gated blocks (E3 + C4 + D3), one gate per block shared by q and v, no smoothing (ρ = 1)." });
add({ id: "S6_LSL_scaled", label: "LSL-scaled", sub: "rank 16 · q,k,v,o + FFN · JL 64", stage: "S6", queue: "Q2", tier: "mandatory", body: "hier", core: "dense", K: null,
  params: { total: 2448640, trainable: null, note: "rank-16 adapters; Φpos 16 / Φneg 32" }, train: T_S6_LSL, losses: {},
  adapter: { kind: "lsl", rank: 16, targets: "q, k, v, o, FFN", where: ["enc", "core", "dec"], gate: { jl: 64, pos: 16, neg: 32, smooth: true } }, runs: "^S6_s\\d+_lsl_P\\d$", source: "§10.6", info: S6_INFO,
  note: "q/k/v share a gate; o, w12 and w3 each gate on their own input (needs per-matrix gates)." });
add({ id: "S6_always_open", label: "Always-open LoRA", sub: "same adapter, no gate", stage: "S6", queue: "Q2", tier: "mandatory", body: "hier", core: "dense", K: null,
  params: { total: 2448640, trainable: null, note: "as LSL-scaled, gate removed" }, train: T_S6_OPEN, losses: {},
  adapter: { kind: "lsl", rank: 16, targets: "q, k, v, o, FFN", where: ["enc", "core", "dec"], gate: null }, runs: "^S6_s\\d+_open_P\\d$", source: "§10.6", info: S6_INFO, note: "The only test of whether gating helps at all." });
add({ id: "S6_VQ_only", label: "VQ-only", sub: "adapt the codebook Q and PQ heads", stage: "S6", queue: "Q2", tier: "mandatory", body: "hier", core: "dense", K: null,
  params: { total: 2448640, trainable: null, note: "codebook Q + PQ heads; frozen everywhere else in V9" }, train: T_S6_VQ, losses: {}, adapter: null, runs: "^S6_s\\d+_vq_P\\d$", source: "§10.6",
  info: {}, note: "The NCP paper's “+VQ” route. This arm alone unfreezes Q and the PQ heads." });
add({ id: "S6_full_ft", label: "Full fine-tune", sub: "all weights (forgetting reference)", stage: "S6", queue: "Q2", tier: "full", body: "hier", core: "dense", K: null,
  params: { total: 2448640, trainable: 2448640, note: "everything trains" }, train: T_S6_FULL, losses: {}, adapter: null, runs: null, source: "§10.6", info: {} });

/* ================================================================ S5 · Q6 : inference procedures on the S4 model (no new weights) */
const SEARCH = [
  { id: "S5_RH1", label: "RH(1)", sub: "receding horizon, one hop, decoder-free", R: 1, tier: "mandatory", method: "rh" },
  { id: "S5_RHH", label: "RH(H*)", sub: "deeper horizon, only if S2 supports it", R: 1, tier: "mandatory", method: "rh", H: "H*" },
  { id: "S5_G1", label: "G(1)", sub: "one tree, every hop grounded", R: 1, tier: "mandatory", method: "g" },
  { id: "S5_TT", label: "Token tree", sub: "same hops and weights, decoded children (primary competitor)", R: 1, tier: "mandatory", method: "token_tree", token: true },
  { id: "S5_BoN", label: "Best-of-N", sub: "verifier best-of-N token sampling", R: 1, tier: "mandatory", method: "best_of_n", token: true, bon: true },
  { id: "S5_AB", label: "Answer loop", sub: "AB-MCTS-A: GEN | CONT with GradCuit or a random kernel (decides H14)", R: 1, tier: "mandatory", method: "answer_loop" },
  { id: "S5_G24", label: "G(2) · G(4)", sub: "exploratory, first seed, dev hard10 at 4×", R: 2, tier: "full", method: "g" },
  { id: "S5_IMAG", label: "IMAG (R = ∞)", sub: "decode only the final path (your proposal), exploratory", R: 0, tier: "full", method: "g" }];
SEARCH.forEach(x => { x.stage = "S5"; x.queue = "Q6"; x.kind = "search"; x.params = { total: 2448640 + 3208, trainable: 0, note: "no new weights: a search procedure on the S4 model" }; x.runs = null; x.source = "§9.5–§9.6"; });

const STAGE_NAMES = { S1: "S1 · Foundation", S2: "S2 · H-NCP module", S3: "S3 · Refit", S4: "S4 · Post-training", S5: "S5 · Search", S6: "S6 · Continual learning" };
const STAGE_QUEUE = { S1: "Q1", S2: "Q3", S3: "Q4", S4: "Q5", S5: "Q6", S6: "Q2" };
const byId = id => V.find(v => v.id === id) || SEARCH.find(v => v.id === id);

/* the shown "all parts" reference: the S2 objective at the K chosen in the toolbar */
const ALL = { id: "ALL", label: "All parts", sub: "reference: every part of the S2 objective", stage: "S2", queue: "", tier: "reference", body: "hier", core: "dense", K: 8, params: { total: 2448640 + 3208, trainable: 869504 + 3208, note: "S2 K = 8" },
  train: tm({ T: "tokens emb enc H pool cin core bh0 codes bhk pq Q hyp sel bestm plan teacher lwta lsel ldist dsum dec head ltok" }), losses: { s2: "lwta", s2b: "lsel", s3: "ldist" }, closedLoop: true, runs: null, source: "§5–§6", info: {} };

window.V9V = { BASE_INFO, PARTS, VARIANTS: V, SEARCH, ALL, byId, STAGE_NAMES, STAGE_QUEUE, tm };
})();
