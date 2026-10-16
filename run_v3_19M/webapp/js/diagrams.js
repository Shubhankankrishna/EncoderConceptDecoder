/* Architecture figures. Every size / parameter count comes from arch.json (generated from code/model.py). */
PAL.full = { f: "#fbe6d4", s: "#e07b39" };

const SRC = {
  enc: "code/hrm_encoder.py · class Encoder", layer: "code/hrm_encoder.py · class EncoderLayer",
  attn: "code/hrm_encoder.py · class GatedAttention", gr: "code/hrm_encoder.py · class GRRead / GRWrite",
  ffn: "code/hrm_encoder.py · class SwiGLU", conv: "code/hrm_encoder.py · class ShortConv",
  engram: "code/hrm_encoder.py · class Engram, engram_indices()", crc: "code/hrm_encoder.py · Encoder.forward (crc_router, pool)",
  core: "code/concept_core_v2.py · class ConceptCoreV2", loop: "code/concept_core_v2.py · class LoopLayer",
  vocab: "code/concept_core_v2.py · class ConceptVocab", head: "code/concept_core_v2.py · ConceptCoreV2.head",
  plan: "code/hrm_decoder.py · class PlanBuilder, plan_index()", dlayer: "code/hrm_decoder.py · class ConfinedLayer",
  dec: "code/hrm_decoder.py · class Decoder", stack: "code/hrm_decoder.py · class StackLayer",
  model: "code/model.py · build_model(), forward()", train: "code/train.py",
};

function legend(g, x, y, items) {
  let cx = x;
  for (const it of items) {
    if (it.line) {
      el("line", { x1: cx, y1: y, x2: cx + 22, y2: y, stroke: it.line, "stroke-width": 2.2, "stroke-dasharray": it.dash || null }, g);
      cx += 28;
    } else {
      el("rect", { x: cx, y: y - 7, width: 16, height: 14, rx: 3, fill: PAL[it.k].f, stroke: PAL[it.k].s }, g);
      cx += 21;
    }
    const t = txt(g, cx, y, it.t, { size: 11.5, anchor: "start", color: "#444" });
    cx += t.getComputedTextLength ? Math.max(40, it.t.length * 6.3) + 16 : 90;
  }
}

/* =========================================================================================== FIG: overview */
function figOverview(A) {
  const P = A.params, T = A.table, e = A.ecfg;
  const svg = makeSvg(1180, 860), g = el("g", {}, svg);
  legend(g, 30, 26, [{ line: COL.L, t: "L stream (local)" }, { line: COL.G, t: "G stream (global)" }, { line: COL.plan, t: "concept / plan" },
    { k: "attn", t: "SWA layer" }, { k: "full", t: "FULL layer" }, { k: "mem", t: "memory / lookup" }, { k: "norm", t: "norm" }, { k: "loss", t: "loss" }]);
  note(g, 1150, 26, "thin arrow = read · thick arrow = write", { size: 11, anchor: "end" });

  // ---------------- encoder
  frame(g, { x: 30, y: 100, w: 390, h: 640 });
  txt(g, 406, 117, `Encoder v1.0 · ${fmtP(T.encoder_non_embedding)} + ${fmtP(T.engram_table)} Engram table`, { size: 13.5, weight: 700, anchor: "end" });
  const xL = 85, xG = 375, xE = 55, bx = 110, bw = 240;
  box(g, { x: 110, y: 790, w: 240, h: 32, k: "io", label: "Token ids  (B, T = 1024)", tip: "<b>Input</b><br>byte-level BPE, V = 8,192 (digits split), rows packed to T = 1024 with document ids; documents padded to multiples of 4." });
  box(g, { x: 110, y: 738, w: 240, h: 32, k: "emb", label: `Token embedding ${fmtI(e.vocab_size)} × ${e.d_model}`, tip: `<b>Embedding E</b> · ${fmtI(T.token_embedding)} params<br>e0 = rms(E[x]). Tied with the output head (logits = h Eᵀ).`, src: SRC.enc });
  arrow(g, [[230, 790], [230, 771]]);
  arrow(g, [[230, 738], [230, 724], [xL, 724], [xL, 165]], { c: COL.L, m: "ahL", head: true, r: 5 });
  arrow(g, [[230, 724], [xG, 724], [xG, 288], [350, 288]], { c: COL.G, m: "ahG" });
  arrow(g, [[230, 724], [xE, 724], [xE, 280], [110, 280]], { c: COL.grey, m: "ahg", dash: "4 3" });
  note(g, xL + 5, 712, "L", { color: COL.L, size: 12 }); note(g, xG + 5, 712, "G", { color: COL.G, size: 12 }); note(g, xE - 12, 712, "e0", { size: 11 });
  const rows = [["S", 0], ["E"], ["S", 1], ["S", 2], ["F", 3], ["S", 4], ["S", 5], ["S", 6], ["F", 7]];
  rows.forEach((r, i) => {
    const cy = 672 - 42 * i, y = cy - 15;
    let tip, lab, sub, k;
    if (r[0] === "S") {
      k = "attn"; lab = `Layer ${r[1]} · S`; sub = `SWA w = ${e.window} · ${fmtP(P.enc_layer_S)}`;
      tip = `<b>S layer ${r[1]}</b> · ${fmtI(P.enc_layer_S)} params<br>reads {L}, writes {L, G}<br>gated attention: sliding window ${e.window}, partial RoPE ${e.rope_dims}/${e.head_dim}, q/k/v short conv k = ${e.conv_kernel}, GQA ${e.n_heads}:${e.n_kv_heads}<br>SwiGLU ${e.d_model} → ${e.d_ff} → ${e.d_model}`;
    } else if (r[0] === "F") {
      k = "full"; lab = `Layer ${r[1]} · F`; sub = `FULL, NoPE + τ_h · ${fmtP(P.enc_layer_F)}`;
      tip = `<b>F layer ${r[1]}</b> · ${fmtI(P.enc_layer_F)} params<br>reads {L, G}, writes {G} only (attention and FFN)<br>full causal attention within the document, no positional encoding, learned per-head temperature τ_h`;
    } else {
      k = "mem"; lab = "Engram (after layer 0)"; sub = `orders 2, 3 · 8 tables · ${fmtP(P.engram_table)} lookup`;
      tip = `<b>Engram</b> · table ${fmtI(P.engram_table)} (lookup) + ${fmtI(P.engram_proj)} compute<br>hashed n-gram memory (orders 2, 3; 4 heads each) → gated value added to L and G`;
    }
    box(g, { x: bx, y, w: bw, h: 30, k, label: lab, sub, size: 12, tip, src: r[0] === "E" ? SRC.engram : SRC.layer });
    // reads (into box) and writes (to rails)
    if (r[0] !== "F") {
      arrow(g, [[xL, cy - 6], [bx, cy - 6]], { c: COL.L, m: "ahL", w: 1 });
      arrow(g, [[bx, cy + 6], [xL + 2, cy + 6]], { c: COL.L, m: "ahL", w: 1.6 });
      arrow(g, [[bx + bw, cy + 4], [xG - 2, cy + 4]], { c: COL.G, m: "ahG", w: 1.6 });
    } else {
      arrow(g, [[xL, cy - 6], [bx, cy - 6]], { c: COL.L, m: "ahL", w: 1 });
      arrow(g, [[xG, cy - 6], [bx + bw, cy - 6]], { c: COL.G, m: "ahG", w: 1 });
      arrow(g, [[bx + bw, cy + 6], [xG - 2, cy + 6]], { c: COL.G, m: "ahG", w: 1.6 });
    }
  });
  box(g, { x: bx, y: 273, w: bw, h: 30, k: "gate", label: "CRC depth-mix over {e0, L, G}", size: 12,
    tip: `<b>CRC depth-mix</b> · ${fmtI(P.crc_router)} params<br>mix = softmax(W rms(G)); m_t = mix₀·e0 + mix₁·rms(L) + mix₂·rms(G)<br>init bias (−1, −1, +1) → 0.107 / 0.107 / 0.787`, src: SRC.crc });
  arrow(g, [[xL, 296], [bx, 296]], { c: COL.L, m: "ahL", w: 1 });
  box(g, { x: bx, y: 225, w: bw, h: 30, k: "conc", label: "Concat-pool 4 tokens → 384", size: 12,
    tip: `<b>Pooling</b> · W_pool ${e.chunk * e.d_model} → ${e.d_model} (${fmtI(P.pool)} params), initialised to mean pooling`, src: SRC.crc });
  arrow(g, [[230, 273], [230, 256]]);
  box(g, { x: bx + 40, y: 177, w: bw - 40, h: 30, k: "conc", label: "ĉ = √384 · c / ‖c‖", sub: "(B, 256, 384) on the sphere", size: 11.5,
    tip: `<b>Concepts ĉ</b><br>one per 4 tokens (M = T/4 = 256), radius √${e.d_model} = ${Math.sqrt(e.d_model).toFixed(2)}`, src: SRC.crc });
  arrow(g, [[250, 225], [250, 208]]);
  box(g, { x: 40, y: 136, w: 150, h: 30, k: "norm", label: "tok = zcRMSNorm(L)", size: 11.5,
    tip: `<b>Token stream to the decoder</b><br>(B, 1024, 384). Reach ${A.reach_enc} tokens: the only encoder output the decoder may read.`, src: SRC.enc });
  // tok to decoder
  arrow(g, [[115, 136], [115, 64], [830, 64], [830, 590]], { c: COL.L, m: "ahL", w: 1.8 });
  dim(g, 480, 56, `tok (B, 1024, 384) · reach ${A.reach_enc}`, { color: COL.L, anchor: "middle" });

  // ---------------- core
  frame(g, { x: 450, y: 100, w: 320, h: 640, title: `Concept module v2 · ${fmtP(T.concept_module)}` });
  arrow(g, [[350, 192], [480, 192]], { c: COL.plan, m: "ahP", w: 1.8 });
  box(g, { x: 480, y: 177, w: 260, h: 30, k: "conc", label: "X = ĉ (re-read every pass)", size: 12, tip: "<b>X</b> = ĉ, the concept sequence; read by every loop pass, gradient flows back into the encoder." });
  box(g, { x: 480, y: 246, w: 76, h: 26, k: "io", label: "S⁰ = 0", size: 12, tip: "Loop state starts at zero (the A1 warm start was not adopted)." });
  box(g, { x: 480, y: 292, w: 260, h: 62, k: "full", label: "Shared FULL layer", sub: `reads {X, S}, writes S · ${fmtP(P.core_layer)}`, size: 13,
    tip: `<b>LoopLayer</b> · ${fmtI(P.core_layer)} params (one set of weights)<br>GR read {X, S} → gated attention (NoPE, τ_h, q/k/v conv) → write S<br>GR read {X, S} → SwiGLU → write S<br>causal over concepts, same document`, src: SRC.loop });
  arrow(g, [[610, 207], [610, 291]], { c: COL.plan, m: "ahP" });
  arrow(g, [[518, 272], [518, 291]]);
  arrow(g, [[740, 338], [758, 338], [758, 306], [741, 306]], { c: COL.ink, w: 1.4 });
  txt(g, 764, 322, "×R", { size: 15, weight: 700, anchor: "start" });
  note(g, 610, 372, "R ~ U{1..4} in training · R = 4 at evaluation", { anchor: "middle", size: 10.5 });
  box(g, { x: 480, y: 392, w: 260, h: 30, k: "norm", label: "u = zcRMSNorm(Sᴿ)", size: 12, tip: "<b>u</b> (B, 256, 384): post-loop state per chunk, sent to the decoder's plan." });
  arrow(g, [[610, 354], [610, 391]]);
  box(g, { x: 480, y: 452, w: 260, h: 44, k: "conc", label: "NCP head · q = W_q u", sub: "logits = −‖q_s − e_s,n‖² / τ_s", size: 12,
    tip: `<b>Next-concept head</b> · W_q ${fmtI(P.ncp_Wq)} + τ ${P.ncp_tau}<br>q split into ${A.S} segments × ${e.d_model / A.S} dims; distance to each segment's ${A.N} codes`, src: SRC.head });
  arrow(g, [[610, 422], [610, 451]]);
  box(g, { x: 600, y: 540, w: 140, h: 40, k: "mem", label: "PQ vocabulary", sub: `${A.S} × ${A.N} × ${e.d_model / A.S} · ${fmtP(P.codebook)}`, size: 12,
    tip: `<b>Codebook</b> ${A.S} segments × ${A.N} codes × ${e.d_model / A.S} dims = ${fmtI(P.codebook)}<br>k-means++ init from the first batch; dead codes re-seeded every 100 optimizer steps; trained by L_VQ only`, src: SRC.vocab });
  arrow(g, [[670, 540], [670, 497]], { dash: "4 3", c: COL.grey, m: "ahg" });
  dim(g, 676, 520, "sg(e)");
  box(g, { x: 480, y: 540, w: 110, h: 40, k: "conc", label: "ĉ_pred", sub: "Σₙ πₙ · sg(eₙ)", size: 12,
    tip: "<b>ĉ_pred</b>: the model's prediction of the NEXT chunk's concept, π = softmax over each segment's codes." });
  arrow(g, [[535, 496], [535, 539]]);
  box(g, { x: 480, y: 668, w: 120, h: 40, k: "loss", label: "L_NCP", sub: "CE vs codes of ĉ_{j+1}", size: 12, tip: "<b>NCP loss</b>: cross-entropy of each segment's logits against the code indices of the next concept (masked across documents). Weight 0.1." });
  box(g, { x: 620, y: 668, w: 120, h: 40, k: "loss", label: "L_VQ", sub: "‖sg(ĉ) − e‖²", size: 12, tip: "<b>VQ loss</b>: squared distance from ĉ (stop-gradient) to its nearest code per segment; updates the codebook only." });
  arrow(g, [[480, 474], [464, 474], [464, 688], [479, 688]], { c: "#d0667a", m: "ahR", dash: "4 3" });
  arrow(g, [[670, 580], [670, 667]], { c: "#d0667a", m: "ahR", dash: "4 3" });
  // outputs to decoder
  arrow(g, [[740, 407], [786, 407], [786, 628], [860, 628]], { c: COL.plan, m: "ahP", w: 1.8 });
  arrow(g, [[535, 580], [535, 616], [796, 616], [796, 652], [860, 652]], { c: COL.plan, m: "ahP", w: 1.8 });
  dim(g, 790, 396, "u", { color: COL.plan }); dim(g, 700, 610, "ĉ_pred", { color: COL.plan });

  // ---------------- decoder
  frame(g, { x: 800, y: 100, w: 360, h: 640 });
  txt(g, 1146, 117, `Decoder v1 · confined · ${fmtP(T.decoder)}`, { size: 13.5, weight: 700, anchor: "end" });
  box(g, { x: 860, y: 615, w: 280, h: 50, k: "plan", label: "Plan P_t", sub: "W_u u_j* + W_c ĉ_pred,j* + o_r(t)", size: 12.5,
    tip: `<b>Plan builder</b> · ${fmtI(P.dec_plan)} params<br>j*(t) = ⌊(t+1)/4⌋ − 1: newest chunk whose last token ≤ t<br>r(t) = (t+1) mod 4 (which token of the chunk is predicted)<br>learned null plan when j* &lt; 0 or chunk j* is in another document`, src: SRC.plan });
  circ(g, 1000, 585, 11, { label: "+", tip: "D⁰ = tok + P (ungated add)" });
  arrow(g, [[1000, 615], [1000, 597]], { c: COL.plan, m: "ahP" });
  arrow(g, [[830, 585], [988, 585]], { c: COL.L, m: "ahL" });
  // layer block
  el("rect", { x: 866, y: 352, width: 280, height: 200, rx: 9, fill: "#fff", stroke: "#cfcfca" }, g);
  el("rect", { x: 863, y: 355, width: 280, height: 200, rx: 9, fill: "#fff", stroke: "#cfcfca" }, g);
  el("rect", { x: 860, y: 358, width: 280, height: 200, rx: 9, fill: "#fff", stroke: "#b8b8b4" }, g);
  box(g, { x: 876, y: 522, w: 248, h: 24, k: "gate", label: "GR read {D}", size: 11.5, tip: "Gated-residual read of the position's private stream D.", src: SRC.gr });
  box(g, { x: 876, y: 474, w: 248, h: 36, k: "attn", label: "Gated attention", sub: "Q = W_q x · K, V = W tok · SWA w = 8", size: 11.5,
    tip: `<b>Confined attention</b> · ${fmtI(P.dec_layer_attn)} params<br>queries from D (plan-aware), keys and values from tok ONLY, window 8, partial RoPE 16/64, QK zcRMSNorm, sigmoid output gate. Nothing reads D at another position.`, src: SRC.dlayer });
  box(g, { x: 876, y: 440, w: 248, h: 22, k: "gate", label: "GR write → D", size: 11 });
  box(g, { x: 876, y: 396, w: 248, h: 32, k: "ffn", label: "GR read {D} → SwiGLU", sub: "384 → 1056 → 384", size: 11.5, tip: `SwiGLU FFN · ${fmtI(P.dec_layer_ffn)} params`, src: SRC.ffn });
  box(g, { x: 876, y: 366, w: 248, h: 22, k: "gate", label: "GR write → D", size: 11 });
  arrow(g, [[1000, 574], [1000, 559]]);
  arrow(g, [[830, 492], [876, 492]], { c: COL.L, m: "ahL", w: 1.8 });
  dim(g, 834, 482, "K,V", { color: COL.L });
  bracket(g, 1150, 358, 558, "×4");
  note(g, 1000, 344, `${fmtP(P.dec_layer)} per layer · reach ${A.reach_dec} at any depth`, { anchor: "middle", size: 10.5 });
  box(g, { x: 900, y: 298, w: 200, h: 28, k: "norm", label: "zcRMSNorm(D)", size: 12 });
  arrow(g, [[1000, 358], [1000, 327]]);
  box(g, { x: 880, y: 238, w: 240, h: 38, k: "emb", label: "Tied output head · Eᵀ", sub: "shares the 8,192 × 384 embedding", size: 12, tip: "logits = zcRMSNorm(D) · Eᵀ, no separate head matrix." });
  arrow(g, [[1000, 298], [1000, 277]]);
  box(g, { x: 900, y: 186, w: 200, h: 28, k: "io", label: "logits (B, 1024, 8192)", size: 11.5 });
  arrow(g, [[1000, 238], [1000, 215]]);
  box(g, { x: 880, y: 128, w: 240, h: 38, k: "loss", label: "L_NTP", sub: "next token · cross-doc & PAD targets masked", size: 12, tip: "Cross-entropy on x_{t+1}; targets in another document or PAD are ignored (EOS is kept)." });
  arrow(g, [[1000, 186], [1000, 167]], { c: "#d0667a", m: "ahR", dash: "4 3" });
  // tie
  arrow(g, [[350, 754], [1172, 754], [1172, 257], [1121, 257]], { c: PAL.emb.s, m: "ahg", dash: "5 4", w: 1.2 });
  dim(g, 760, 766, "weight tying: output head = embedding Eᵀ", { color: PAL.emb.s, anchor: "middle" });
  txt(g, 590, 846, "Loss = L_NTP + 0.1 · L_NCP + L_VQ", { size: 14, weight: 700 });
  return svg;
}

/* =========================================================================================== FIG: encoder layers */
function figEncoderLayers(A) {
  const P = A.params, e = A.ecfg;
  const svg = makeSvg(1180, 760), g = el("g", {}, svg);
  function panel(x0, kind) {
    const S = kind === "S";
    frame(g, { x: x0, y: 40, w: 340, h: 700, title: S ? `S layer (×6) · ${fmtP(P.enc_layer_S)}` : `F layer (×2) · ${fmtP(P.enc_layer_F)}`, dash: false, fill: "#fcfcfb" });
    const xL = x0 + 38, xG = x0 + 302, bx = x0 + 70, bw = 200, mid = bx + bw / 2;
    arrow(g, [[xL, 720], [xL, 80]], { c: COL.L, m: "ahL", w: 2 }); arrow(g, [[xG, 720], [xG, 80]], { c: COL.G, m: "ahG", w: 2 });
    txt(g, xL, 730, "L", { color: COL.L, weight: 700 }); txt(g, xG, 730, "G", { color: COL.G, weight: 700 });
    const readTip = S ? "reads {L}" : "reads {L, G}";
    const rd = (cy, lab) => {
      box(g, { x: bx, y: cy - 16, w: bw, h: 32, k: "gate", label: `GR read ${S ? "{L}" : "{L, G}"}`, sub: "x = mean σ(U V n + b)(1+γ) n", size: 12,
        tip: `<b>Gated-residual read</b>, ${readTip}<br>n_i = rms(R_i), g_i = σ(U_i V_i n_i + b_i), rank ${e.gr_rank}; b = ${e.gr_read_bias} → g = 0.881 at init<br>x = mean_i g_i (1 + γ_i) n_i`, src: SRC.gr });
      arrow(g, [[xL, cy], [bx, cy]], { c: COL.L, m: "ahL", w: 1.1 });
      if (!S) arrow(g, [[xG, cy], [bx + bw, cy]], { c: COL.G, m: "ahG", w: 1.1 });
    };
    const wr = cy => {
      box(g, { x: bx, y: cy - 14, w: bw, h: 28, k: "gate", label: `GR write → ${S ? "L, G" : "G"}`, sub: "s_i = 2σ(u_i·x + a_i)", size: 11.5,
        tip: "<b>Gated-residual write</b>: per-stream scale s_i = 2σ(u_i·x + a_i), initialised to exactly 1", src: SRC.gr });
      circ(g, xG, cy, 8, { label: "+" }); arrow(g, [[bx + bw, cy], [xG - 9, cy]], { c: COL.G, m: "ahG", w: 1.7 });
      if (S) { circ(g, xL, cy, 8, { label: "+" }); arrow(g, [[bx, cy], [xL + 9, cy]], { c: COL.L, m: "ahL", w: 1.7 }); }
    };
    rd(660);
    box(g, { x: bx, y: 548, w: bw, h: 56, k: S ? "attn" : "full", label: S ? "Gated attention · SWA" : "Gated attention · FULL", sub: S ? `window ${e.window} · partial RoPE ${e.rope_dims}/${e.head_dim}` : "causal · NoPE · per-head τ_h", size: 12.5,
      tip: `<b>Gated attention</b> · ${fmtI(S ? P.enc_layer_S_attn : P.enc_attn_F)} params<br>see the middle panel for the anatomy`, src: SRC.attn });
    arrow(g, [[mid, 644], [mid, 605]]);
    wr(510); arrow(g, [[mid, 548], [mid, 525]]);
    rd(420); arrow(g, [[mid, 496], [mid, 437]], { c: COL.grey, m: "ahg", dash: "3 3" });
    note(g, mid + 6, 468, "next read sees the updated streams", { size: 10 });
    box(g, { x: bx, y: 300, w: bw, h: 56, k: "ffn", label: "SwiGLU FFN", sub: `${e.d_model} → ${e.d_ff} → ${e.d_model}`, size: 12.5,
      tip: `<b>SwiGLU</b> · ${fmtI(P.enc_layer_S_ffn)} params<br>W₂(SiLU(W₁x) ⊙ W₃x)`, src: SRC.ffn });
    arrow(g, [[mid, 404], [mid, 357]]);
    wr(262); arrow(g, [[mid, 300], [mid, 277]]);
    const lines = S ? ["S: local role.", "Attention and FFN both write L and G.", "Reads only L, so L never sees", "anything written by a FULL layer."]
      : ["F: global role.", "Reads L and G, writes G only", "(attention and FFN).", "G therefore has unbounded reach but", "feeds only the concept pooling."];
    lines.forEach((s, i) => note(g, x0 + 54, 128 + i * 16, s, { size: 11.5, color: i === 0 ? "#333" : COL.grey }));
    note(g, x0 + 54, 222, `GR params per layer: ${fmtI(S ? P.enc_layer_S_gr : P.enc_layer_F_gr)}`, { size: 10.5 });
  }
  panel(20, "S"); panel(820, "F");
  // middle: attention anatomy
  const x0 = 380, mid = 600;
  frame(g, { x: x0, y: 40, w: 420, h: 700, title: `Gated attention anatomy · ${fmtI(P.enc_layer_S_attn)} (+${e.n_heads} τ in F)`, dash: false, fill: "#fcfcfb" });
  box(g, { x: 530, y: 690, w: 140, h: 28, k: "io", label: "x (d = 384)", size: 12 });
  const cols = [[450, "W_q", `384 → ${e.n_heads}×${e.head_dim}`, 147456], [560, "W_k", `384 → ${e.n_kv_heads}×${e.head_dim}`, 49152], [670, "W_v", `384 → ${e.n_kv_heads}×${e.head_dim}`, 49152]];
  cols.forEach(([x, n, s, p]) => {
    box(g, { x, y: 624, w: 100, h: 36, k: "attn", label: n, sub: s, size: 12, tip: `<b>${n}</b> ${s}, no bias · ${fmtI(p)} params` });
    arrow(g, [[mid, 690], [mid, 676], [x + 50, 676], [x + 50, 661]], { r: 4 });
    box(g, { x, y: 570, w: 100, h: 34, k: "gate", label: "ShortConv k=4", sub: "causal, doc-aware", size: 11,
      tip: `<b>Short convolution</b> (depthwise, causal, kernel ${e.conv_kernel}, residual form, zero-init = identity). Positions in another document contribute 0.`, src: SRC.conv });
    arrow(g, [[x + 50, 624], [x + 50, 605]]);
  });
  [[450, "q"], [560, "k"]].forEach(([x, n]) => {
    box(g, { x, y: 518, w: 100, h: 30, k: "norm", label: `zcRMSNorm(${n})`, size: 11, tip: "Per-head zero-centred RMSNorm: y = (1 + γ) x / rms(x), γ init 0." });
    arrow(g, [[x + 50, 570], [x + 50, 549]]);
  });
  arrow(g, [[720, 570], [720, 440]]);
  box(g, { x: 450, y: 452, w: 210, h: 46, k: "white", label: "S: partial RoPE (dims 0–15)", sub: "F: NoPE, q scaled by τ_h", size: 11.5,
    tip: `<b>Positions</b><br>S layers: rotary embedding on the first ${e.rope_dims} of ${e.head_dim} dims of q and k (base ${e.rope_base})<br>F layers: no positional encoding; per-head temperature τ_h = exp(log τ_h), init 1` });
  arrow(g, [[500, 518], [500, 499]]); arrow(g, [[610, 518], [610, 499]]);
  box(g, { x: 450, y: 372, w: 300, h: 54, k: "attn", label: `SDPA · GQA ${e.n_heads} q heads : ${e.n_kv_heads} kv heads`, sub: "S: causal ∧ same doc ∧ t − j < 8   ·   F: causal ∧ same doc", size: 12,
    tip: `Scaled dot-product attention, head dim ${e.head_dim}; each kv head serves ${e.n_heads / e.n_kv_heads} query heads. Boolean masks built per batch from document ids.` });
  arrow(g, [[555, 452], [555, 427]]); arrow(g, [[720, 440], [720, 427]]);
  box(g, { x: 450, y: 300, w: 300, h: 40, k: "gate", label: "y ⊙ σ(W_gate x)", sub: `W_gate 384 → ${e.n_heads * e.head_dim} · 147K`, size: 12, tip: "Sigmoid output gate per channel, computed from the layer input x." });
  arrow(g, [[600, 372], [600, 341]]);
  arrow(g, [[760, 690], [770, 690], [770, 320], [751, 320]], { c: COL.grey, m: "ahg", dash: "4 3" });
  dim(g, 774, 500, "x");
  box(g, { x: 500, y: 238, w: 200, h: 36, k: "attn", label: "W_o", sub: `${e.n_heads * e.head_dim} → 384 · 147K`, size: 12 });
  arrow(g, [[600, 300], [600, 275]]);
  box(g, { x: 530, y: 180, w: 140, h: 28, k: "io", label: "→ GR write", size: 12 });
  arrow(g, [[600, 238], [600, 209]]);
  note(g, 400, 100, "Parameter check (S):", { size: 11.5, color: "#333" });
  note(g, 400, 118, "q 147,456 + k 49,152 + v 49,152 + o 147,456 + gate 147,456", { size: 10.5, mono: true });
  note(g, 400, 134, "+ convs (384+128+128)·4 = 2,560 + q/k norms 128 = 543,360", { size: 10.5, mono: true });
  return svg;
}

/* =========================================================================================== FIG: pattern and reach */
function figReach(A) {
  const e = A.ecfg, svg = makeSvg(1180, 560), g = el("g", {}, svg);
  txt(g, 30, 30, "Layer pattern  S S S F S S S F  with Engram after layer 0", { size: 15, weight: 700, anchor: "start" });
  const seq = [["S", "0"], ["E", "Engram"], ["S", "1"], ["S", "2"], ["F", "3"], ["S", "4"], ["S", "5"], ["S", "6"], ["F", "7"]];
  seq.forEach(([k, n], i) => {
    const x = 30 + i * 124;
    box(g, { x, y: 56, w: 110, h: 56, k: k === "S" ? "attn" : k === "F" ? "full" : "mem", label: k === "E" ? "Engram" : `Layer ${n}`,
      sub: k === "S" ? "S · +10 reach" : k === "F" ? "F · G only" : "max(10, 2) + 3", size: 12.5 });
    if (i) arrow(g, [[x - 14, 84], [x - 1, 84]]);
  });
  txt(g, 30, 160, "Reach of the L stream (tokens back that can influence position t)", { size: 15, weight: 700, anchor: "start" });
  note(g, 30, 182, `per SWA layer: (w − 1) + (k − 1) = (${e.window} − 1) + (${e.conv_kernel} − 1) = 10   ·   Engram: max(reach so far, max order − 1) + (k − 1)`, { size: 12, color: "#555" });
  const steps = [["after layer 0", 10], ["after Engram", 13], ["after layer 1", 23], ["after layer 2", 33], ["after layer 4", 43], ["after layer 5", 53], ["after layer 6 (= tok)", 63],
    ["decoder confined (+ w − 1)", 70], ["stack control (+ 4 × 10)", 103]];
  const x0 = 250, sc = 7.6;
  steps.forEach(([lab, v], i) => {
    const y = 206 + i * 34;
    note(g, x0 - 12, y + 12, lab, { anchor: "end", size: 12, color: "#333" });
    const k = i === 7 ? "plan" : i === 8 ? "full" : i === 6 ? "norm" : "attn";
    el("rect", { x: x0, y, width: v * sc, height: 24, rx: 4, fill: PAL[k].f, stroke: PAL[k].s }, g);
    txt(g, x0 + v * sc + 8, y + 12, String(v), { anchor: "start", size: 12.5, weight: 700 });
  });
  line(g, x0 + 70 * sc, 200, x0 + 70 * sc, 520, { c: COL.plan, dash: "4 3", w: 1.2 });
  note(g, x0 + 70 * sc + 8, 214, "beyond 70 tokens: only the", { size: 11.5, color: COL.plan }); note(g, x0 + 70 * sc + 8, 230, "concept stream can help", { size: 11.5, color: COL.plan });
  note(g, 30, 548, `Measured in code: EncoderConfig.reach() = ${A.reach_enc}, DecoderConfig.reach() = ${A.reach_dec} (confined); exact tests T1–T9, D1–D5 perturb one token and check which positions change.`, { size: 11 });
  return svg;
}

/* =========================================================================================== FIG: engram */
function figEngram(A) {
  const e = A.ecfg, P = A.params, svg = makeSvg(1180, 520), g = el("g", {}, svg);
  const nt = e.engram_orders.length * e.engram_heads;
  txt(g, 30, 28, "Hashed n-gram lookup", { size: 15, weight: 700, anchor: "start" });
  const r1 = [["io", "token ids x_t", "V = 8,192"], ["io", "canonical ids", "6,572 classes"], ["white", "n-gram keys", `orders ${e.engram_orders.join(", ")} · other doc → id 0`],
    ["gate", "hash × 4 heads", "mult-XOR, 61-bit, mod p"], ["mem", `${nt} tables`, `${fmtI(e.engram_rows)} × ${e.engram_dim} each`], ["conc", "e_t = concat", `${nt} × ${e.engram_dim} = ${nt * e.engram_dim}`]];
  r1.forEach(([k, a, b], i) => {
    const x = 30 + i * 190;
    box(g, { x, y: 48, w: 165, h: 52, k, label: a, sub: b, size: 12.5,
      tip: i === 3 ? "x ← ((x ⊕ gram_i) · mult_{table,i}) mod 2⁶¹, mult odd; row = x mod 131,071 (prime, coprime with 2¹³) + table offset" :
        i === 4 ? `<b>Engram table</b> ${fmtI(P.engram_table)} params (a lookup: 8 rows read per token, not counted as compute)` :
          i === 1 ? "Canonical id = decode the token, strip one leading space, NFKC, lowercase; equal strings share an id (8,192 → 6,572). code/build_data.py · build_canonical_map()" : null, src: SRC.engram });
    if (i) arrow(g, [[x - 25, 74], [x - 1, 74]]);
  });
  // table visual
  for (let t = 0; t < nt; t++) {
    const x = 790 + t * 20;
    el("rect", { x, y: 120, width: 14, height: 90, rx: 2, fill: PAL.mem.f, stroke: PAL.mem.s }, g);
    el("rect", { x, y: 130 + (t * 37) % 70, width: 14, height: 5, fill: "#b8860b" }, g);
  }
  note(g, 790, 224, `${nt} tables (orders 2 and 3 × 4 heads); one row read per table per token`, { size: 11 });
  txt(g, 30, 270, "Gated injection into both streams", { size: 15, weight: 700, anchor: "start" });
  box(g, { x: 30, y: 300, w: 150, h: 44, k: "conc", label: "e_t", sub: "384", size: 12.5 });
  box(g, { x: 230, y: 326, w: 160, h: 30, k: "attn", label: "k = W_K e_t", size: 12, tip: "W_K 384 → 384" });
  box(g, { x: 230, y: 290, w: 160, h: 30, k: "attn", label: "v = W_V e_t", size: 12, tip: "W_V 384 → 384, zero-initialised so the Engram output is exactly 0 at init" });
  arrow(g, [[180, 314], [205, 314], [205, 305], [229, 305]], { r: 3 }); arrow(g, [[205, 314], [205, 341], [229, 341]], { r: 3 });
  box(g, { x: 30, y: 400, w: 150, h: 40, k: "gate", label: "h = GR read {L}", size: 12 });
  arrow(g, [[105, 490], [105, 441]], { c: COL.L, m: "ahL", w: 1.6 }); txt(g, 105, 500, "L", { color: COL.L, weight: 700 });
  box(g, { x: 440, y: 386, w: 260, h: 44, k: "gate", label: "α = σ( rms(h) · rms(k) / √384 )", sub: "per-token scalar gate", size: 12,
    tip: "Context-dependent gate: the lookup is used only when the looked-up key agrees with the current hidden state." });
  arrow(g, [[180, 420], [439, 420]]); arrow(g, [[390, 341], [415, 341], [415, 400], [439, 400]], { r: 4 });
  box(g, { x: 440, y: 300, w: 260, h: 44, k: "ffn", label: "y = SiLU( ShortConv(α · v) )", sub: "kernel 4, random init, doc-aware", size: 12 });
  arrow(g, [[390, 305], [439, 305]]); arrow(g, [[570, 386], [570, 345]]);
  box(g, { x: 760, y: 300, w: 180, h: 44, k: "gate", label: "GR write → L, G", sub: "s = 2σ(u·h + a)", size: 12 });
  arrow(g, [[700, 322], [759, 322]]);
  circ(g, 1000, 300, 9, { label: "+" }); circ(g, 1080, 344, 9, { label: "+" });
  arrow(g, [[940, 312], [990, 300]], { c: COL.L, m: "ahL" }); arrow(g, [[940, 332], [1070, 344]], { c: COL.G, m: "ahG" });
  arrow(g, [[1000, 480], [1000, 310]], { c: COL.L, m: "ahL", w: 1.8 }); arrow(g, [[1000, 290], [1000, 250]], { c: COL.L, m: "ahL", w: 1.8 });
  arrow(g, [[1080, 480], [1080, 354]], { c: COL.G, m: "ahG", w: 1.8 }); arrow(g, [[1080, 334], [1080, 250]], { c: COL.G, m: "ahG", w: 1.8 });
  txt(g, 1000, 494, "L", { color: COL.L, weight: 700 }); txt(g, 1080, 494, "G", { color: COL.G, weight: 700 });
  note(g, 440, 470, `Compute part: W_K + W_V + GR read/write + conv = ${fmtI(P.engram_proj)} params.`, { size: 11.5, color: "#444" });
  note(g, 440, 488, "Live gate mean α and ‖W_V‖ are shown in the Live training tab.", { size: 11.5 });
  return svg;
}

/* =========================================================================================== FIG: crc + pool */
function figCRC(A) {
  const e = A.ecfg, P = A.params, svg = makeSvg(1180, 440), g = el("g", {}, svg);
  txt(g, 30, 28, "Depth-mix (CRC) per token, then 4 tokens → 1 concept on the sphere", { size: 15, weight: 700, anchor: "start" });
  for (let t = 0; t < 4; t++) {
    const x = 60 + t * 150;
    [[COL.grey, "e0"], [COL.L, "rms L"], [COL.G, "rms G"]].forEach(([c, n], j) => {
      el("rect", { x: x + j * 36, y: 330, width: 30, height: 60, rx: 4, fill: "#fff", stroke: c, "stroke-width": 1.6 }, g);
      txt(g, x + j * 36 + 15, 360, n.replace("rms ", ""), { size: 11, color: c, weight: 700 });
    });
    note(g, x + 50, 405, `token 4j+${t}`, { anchor: "middle", size: 11 });
    box(g, { x: x - 4, y: 240, w: 112, h: 44, k: "gate", label: "m_t = Σ mix·{…}", sub: "mix = softmax(W rms G)", size: 11,
      tip: `<b>CRC router</b> (${fmtI(P.crc_router)} params, shared): mix_t = softmax(W · rms(G_t) + b), b init (−1, −1, +1) → (0.107, 0.107, 0.787)`, src: SRC.crc });
    arrow(g, [[x + 50, 330], [x + 50, 285]]);
    arrow(g, [[x + 50, 240], [x + 50, 214], [380, 214], [380, 196]], { r: 5 });
  }
  box(g, { x: 250, y: 150, w: 260, h: 44, k: "conc", label: "concat → W_pool", sub: `${e.chunk} × ${e.d_model} = ${e.chunk * e.d_model} → ${e.d_model} · ${fmtP(P.pool)}`, size: 12.5,
    tip: "Initialised to exact mean pooling (identity blocks / 4), then learned.", src: SRC.crc });
  box(g, { x: 250, y: 70, w: 260, h: 44, k: "conc", label: "ĉ_j = √384 · c_j / ‖c_j‖", sub: `radius ${Math.sqrt(e.d_model).toFixed(2)} · to the concept module`, size: 12.5 });
  arrow(g, [[380, 150], [380, 115]]);
  arrow(g, [[380, 70], [380, 46], [520, 46]], { c: COL.plan, m: "ahP", w: 1.6 });
  // sphere sketch
  const cx = 860, cy = 200, R = 110;
  el("circle", { cx, cy, r: R, fill: "#f3fbf9", stroke: PAL.conc.s }, g);
  el("ellipse", { cx, cy, rx: R, ry: 34, fill: "none", stroke: PAL.conc.s, "stroke-dasharray": "4 4" }, g);
  const pts = [[-40, -60], [30, -85], [70, -20], [-80, 10], [10, 40], [60, 60], [-30, 85], [95, 30]];
  pts.forEach(([dx, dy], i) => { el("circle", { cx: cx + dx, cy: cy + dy, r: 4.5, fill: i % 2 ? COL.plan : "#1f7a6e" }, g); });
  arrow(g, [[cx, cy], [cx + 70, cy - 20]], { c: COL.plan, m: "ahP" });
  note(g, cx, cy + R + 22, "every concept has the same norm √384;", { anchor: "middle", size: 11.5 }); note(g, cx, cy + R + 38, "the codebook quantises its 48 segments of 8 dims", { anchor: "middle", size: 11.5 });
  note(g, 1000, 90, "Live mix (latest eval)", { size: 11.5, color: "#333" });
  el("g", { id: "crc-live" }, g);
  return svg;
}

/* =========================================================================================== FIG: concept core */
function figCore(A) {
  const e = A.ecfg, P = A.params, ds = e.d_model / A.S, svg = makeSvg(1180, 640), g = el("g", {}, svg);
  txt(g, 30, 28, "Loop: one shared FULL layer applied R times over the concept sequence", { size: 15, weight: 700, anchor: "start" });
  arrow(g, [[40, 70], [700, 70]], { c: COL.plan, m: "ahP", w: 2, head: false });
  txt(g, 40, 56, "X = ĉ  (B, 256, 384) — re-read by every pass", { size: 12, anchor: "start", color: COL.plan, weight: 600 });
  box(g, { x: 30, y: 150, w: 70, h: 40, k: "io", label: "S⁰ = 0", size: 12.5 });
  for (let r = 0; r < 4; r++) {
    const x = 140 + r * 150;
    box(g, { x, y: 130, w: 110, h: 80, k: "full", label: `pass ${r + 1}`, sub: "shared weights", size: 13,
      tip: `<b>LoopLayer</b> (${fmtI(P.core_layer)} params, the SAME weights every pass)<br>x = GR read {X, S}; S += s·GatedAttn_FULL(x)<br>x = GR read {X, S}; S += s·SwiGLU(x)`, src: SRC.loop });
    arrow(g, [[x + 55, 70], [x + 55, 129]], { c: COL.plan, m: "ahP" });
    arrow(g, [[x - 40, 170], [x - 1, 170]]);
    dim(g, x - 36, 160, `S${"⁰¹²³"[r]}`);
    if (r) el("path", { d: `M${x - 95},${218} q47,22 95,0`, fill: "none", stroke: "#bbb", "stroke-dasharray": "3 3" }, g);
  }
  note(g, 440, 240, "dashed: tied weights", { anchor: "middle", size: 10.5 });
  box(g, { x: 740, y: 150, w: 150, h: 40, k: "norm", label: "u = zcRMSNorm(Sᴿ)", size: 12 });
  arrow(g, [[700, 170], [739, 170]]); dim(g, 704, 160, "S⁴");
  note(g, 140, 270, "R ~ U{1..4} per training step (one draw for the whole batch), R = 4 at evaluation. The screen S3 trained with R = 1.", { size: 11.5, color: "#444" });
  note(g, 140, 288, "Attention over concepts is causal and restricted to the same document (256 concepts per row).", { size: 11.5 });
  arrow(g, [[890, 170], [960, 170]], { c: COL.plan, m: "ahP", w: 1.6 }); txt(g, 1030, 170, "to decoder plan", { size: 12, color: COL.plan });

  txt(g, 30, 340, "Next-concept prediction (NCP) with a product-quantised vocabulary", { size: 15, weight: 700, anchor: "start" });
  box(g, { x: 30, y: 380, w: 130, h: 44, k: "norm", label: "u_j", sub: "384", size: 12.5 });
  box(g, { x: 200, y: 380, w: 150, h: 44, k: "conc", label: "q = W_q u_j", sub: `split ${A.S} × ${ds}`, size: 12.5, tip: `W_q 384 × 384 = ${fmtI(P.ncp_Wq)}` });
  arrow(g, [[160, 402], [199, 402]]);
  // codebook grid
  const gx = 420, gy = 360, cw = 7, ch = 3.2;
  for (let s = 0; s < A.S; s++) for (let n = 0; n < 32; n++) {
    const v = (Math.sin(s * 12.9898 + n * 78.233) * 43758.5453) % 1;
    el("rect", { x: gx + s * cw, y: gy + n * ch, width: cw - 1, height: ch - 0.6, fill: `rgba(207,169,53,${0.25 + Math.abs(v) * 0.6})` }, g);
  }
  el("rect", { x: gx - 3, y: gy - 3, width: A.S * cw + 5, height: 32 * ch + 5, fill: "none", stroke: PAL.mem.s, rx: 4 }, g).setAttribute("data-tip",
    `<b>PQ vocabulary</b>: ${A.S} segments × ${A.N} codes × ${ds} dims (${fmtI(P.codebook)} params); 32 of 128 code rows drawn`);
  note(g, gx, gy + 32 * ch + 16, `${A.S} segments (columns) × ${A.N} codes each (32 drawn) × ${ds} dims`, { size: 11 });
  box(g, { x: 800, y: 372, w: 170, h: 60, k: "conc", label: "logits_s,n", sub: "−‖q_s − e_s,n‖² / τ_s", size: 12.5,
    tip: `τ_s per segment, init 2·384/48 = 16 (current values in the Live tab)`, src: SRC.head });
  arrow(g, [[350, 402], [415, 402]]); arrow(g, [[762, 402], [799, 402]]);
  box(g, { x: 1010, y: 372, w: 150, h: 60, k: "conc", label: "ĉ_pred,j", sub: "concat_s Σ_n π_s,n sg(e_s,n)", size: 11.5 });
  arrow(g, [[970, 402], [1009, 402]]); dim(g, 974, 392, "softmax");
  box(g, { x: 780, y: 500, w: 210, h: 50, k: "loss", label: "L_NCP (× 0.1)", sub: "CE(logits_j , codes(ĉ_{j+1}))", size: 12.5,
    tip: "Target: the code index of each segment of the NEXT concept. Terms where chunk j+1 is in another document are masked." });
  arrow(g, [[885, 432], [885, 499]], { c: "#d0667a", m: "ahR", dash: "4 3" });
  box(g, { x: 420, y: 520, w: 260, h: 50, k: "loss", label: "L_VQ", sub: "‖sg(ĉ) − e_{n*}‖² → updates codebook only", size: 12.5 });
  arrow(g, [[590, 520], [590, 486]], { c: "#d0667a", m: "ahR", dash: "4 3" });
  note(g, 30, 610, "Codebook maintenance: k-means++ init from the first optimizer batch's concepts; every 100 optimizer steps, codes with zero hits since the last check are re-seeded from current concepts (after optimizer.step()).", { size: 11.5, color: "#444" });
  note(g, 30, 628, "Healthy signs: NCP CE below the code entropy (the next concept is predictable beyond its frequency), codes used ≥ 95 % per segment, variance explained rising.", { size: 11.5 });
  return svg;
}

/* =========================================================================================== FIG: decoder confined vs stack */
function figDecoder(A) {
  const svg = makeSvg(1180, 600), g = el("g", {}, svg);
  function panel(x0, confined) {
    frame(g, { x: x0, y: 30, w: 560, h: 540, title: confined ? `Confined (used) · reach ${A.reach_dec} at any depth` : "Stack (control, screen S2) · reach 103", dash: false, fill: "#fcfcfb" });
    const n = 12, sx = 36, ox = x0 + 104, tokY = 480, layerY = l => 420 - l * 80;
    for (let i = 0; i < n; i++) {
      const x = ox + i * sx, last = i === n - 1;
      el("circle", { cx: x, cy: tokY, r: 8, fill: "#e8f0fa", stroke: COL.L }, g);
      el("rect", { x: x - 7, y: tokY + 26, width: 14, height: 14, rx: 3, fill: last ? PAL.plan.f : "#f2f2f0", stroke: last ? PAL.plan.s : "#ccc" }, g);
      for (let l = 0; l < 4; l++) {
        el("rect", { x: x - 9, y: layerY(l) - 9, width: 18, height: 18, rx: 4, fill: last ? (confined ? "#fff" : PAL.attn.f) : "#f4f4f2", stroke: last ? "#555" : "#d4d4d0" }, g);
      }
      if (i % 2 === 1 || last) note(g, x, tokY + 54, last ? "t" : `t−${n - 1 - i}`, { anchor: "middle", size: 10 });
    }
    note(g, ox - 30, tokY, "tok", { anchor: "end", size: 11, color: COL.L });
    note(g, ox - 30, tokY + 33, "plan", { anchor: "end", size: 11, color: COL.plan });
    for (let l = 0; l < 4; l++) note(g, ox - 30, layerY(l), confined ? `D, layer ${l + 1}` : `h, layer ${l + 1}`, { anchor: "end", size: 11 });
    const xt = ox + (n - 1) * sx;
    if (confined) {
      for (let l = 0; l < 4; l++) for (let k = 0; k < 8; k++) {
        const xk = ox + (n - 1 - k) * sx;
        el("line", { x1: xt, y1: layerY(l) + 9, x2: xk, y2: tokY - 8, stroke: COL.L, "stroke-width": 0.8, opacity: 0.35 + 0.1 * (3 - l) / 3 }, g);
      }
      arrow(g, [[xt + 14, tokY + 33], [xt + 22, tokY + 33], [xt + 22, layerY(0) + 4], [xt + 10, layerY(0) + 4]], { c: COL.plan, m: "ahP", w: 1.5 });
      for (let l = 0; l < 3; l++) arrow(g, [[xt, layerY(l) - 9], [xt, layerY(l + 1) + 10]], { w: 1.5 });
      note(g, x0 + 20, 70, "Every layer: queries from the position's own D, keys/values from tok only (window 8).", { size: 11.5, color: "#333" });
      note(g, x0 + 20, 86, "D is never read at another position → position t sees only its own plan P_t.", { size: 11.5 });
      note(g, x0 + 20, 102, "Reach = 63 (tok) + 7 = 70, independent of depth.", { size: 11.5 });
    } else {
      for (let l = 0; l < 4; l++) for (let k = 0; k < 8; k++) {
        const xk = ox + (n - 1 - k) * sx, y2 = l === 0 ? tokY - 8 : layerY(l - 1) - 9;
        el("line", { x1: xt, y1: layerY(l) + 9, x2: xk, y2, stroke: "#5f93cc", "stroke-width": 0.8, opacity: 0.5 }, g);
      }
      el("path", { d: `M${xt},${layerY(3)} L${ox},${tokY - 12} L${xt},${tokY - 12} z`, fill: "rgba(224,123,57,.07)", stroke: "none" }, g);
      note(g, x0 + 20, 70, "One stream h = tok + P, SWA over h at every layer (+ q/k/v convs).", { size: 11.5, color: "#333" });
      note(g, x0 + 20, 86, "Later positions read earlier positions' plans; the receptive field", { size: 11.5 });
      note(g, x0 + 20, 102, "grows by 10 per layer: 63 + 4 × 10 = 103.", { size: 11.5 });
    }
  }
  panel(20, true); panel(600, false);
  note(g, 590, 590, "Both layouts have the same parameter count and differ in one line: where keys and values come from. Screen result at step 265: confined 4.341 vs stack 4.352 val nats.", { anchor: "middle", size: 11.5, color: "#444" });
  return svg;
}

/* =========================================================================================== FIG: plan timing */
function figPlan(A) {
  const svg = makeSvg(1180, 470), g = el("g", {}, svg);
  txt(g, 30, 28, "Which concept does position t read?  j*(t) = ⌊(t+1)/4⌋ − 1,  r(t) = (t+1) mod 4", { size: 15, weight: 700, anchor: "start" });
  const n = 16, sx = 66, ox = 90, ty = 196;
  for (let c = 0; c < 4; c++) {
    el("rect", { x: ox + c * 4 * sx - 28, y: ty - 26, width: 4 * sx - 10, height: 52, rx: 8, fill: c % 2 ? "#f4faf9" : "#fbfbfa", stroke: PAL.conc.s, "stroke-dasharray": "4 3" }, g);
    txt(g, ox + c * 4 * sx + 1.5 * sx - 5, ty + 112, `chunk ${c}  →  ĉ_${c}, u_${c}, ĉ_pred,${c}`, { size: 11.5, color: PAL.conc.s, weight: 600 });
  }
  for (let t = 0; t < n; t++) {
    const x = ox + t * sx;
    box(g, { x: x - 22, y: ty - 16, w: 44, h: 32, k: "io", label: `x${t}`.replace(/\d+/, d => d.split("").map(c => "₀₁₂₃₄₅₆₇₈₉"[c]).join("")), size: 13 });
    const j = Math.floor((t + 1) / 4) - 1, r = (t + 1) % 4;
    txt(g, x, ty + 50, `t=${t}`, { size: 11, color: "#555" });
    txt(g, x, ty + 70, j < 0 ? "null" : `j*=${j}`, { size: 11.5, weight: 600, color: j < 0 ? "#b44" : COL.plan });
    txt(g, x, ty + 88, `r=${r}`, { size: 11, color: "#777" });
    if (j >= 0 && (t === 3 || t === 6 || t === 7 || t === 12 || t === 14)) {
      const xs = ox + (4 * j + 1.5) * sx;
      el("path", { d: `M${xs},${ty - 28} C${xs},${ty - 110} ${x},${ty - 110} ${x},${ty - 18}`, fill: "none", stroke: COL.plan, "stroke-width": 1.3, "marker-end": mk(g, "ahP") }, g);
    }
  }
  note(g, 30, 370, "Position t predicts x_{t+1}. It reads the newest chunk that is complete at t (its last token 4j*+3 ≤ t), so u_j* and ĉ_pred,j* never contain the target;", { size: 12, color: "#333" });
  note(g, 30, 388, "ĉ_pred,j* is the model's guess of chunk j*+1, the chunk that contains x_{t+1}. The offset embedding o_r(t) says which of its 4 tokens is being predicted.", { size: 12, color: "#333" });
  note(g, 30, 410, "The first 3 positions of every document (j* < 0, or chunk j* in the previous document) get the learned null plan instead. Documents are padded to multiples of 4.", { size: 12 });
  note(g, 30, 428, "Plan-swap intervention (evaluation): u and ĉ_pred of another row are used; a drop to chance on long-range recall shows the decoder relies on the concept stream.", { size: 12 });
  note(g, 30, 450, "Tested exactly in D2/D3: perturbing u of chunk j changes only positions 4j+3 … 4j+6 in the confined decoder.", { size: 11.5 });
  return svg;
}

/* =========================================================================================== FIG: objective & data (HTML) */
function figObjective(A) {
  const d = document.createElement("div");
  d.innerHTML = `
  <div class="grid g2" style="color:#1d1d1f">
   <div>
    <h3 style="margin-top:0">Objective</h3>
    <div class="kvt" style="color:#333">
     <span>total</span><span><b>L = L_NTP + 0.1 · L_NCP + L_VQ</b></span>
     <span>L_NTP</span><span>next-token cross-entropy; targets in another document or PAD ignored, EOS kept</span>
     <span>L_NCP</span><span>cross-entropy over ${A.S} segments × ${A.N} codes of the next concept</span>
     <span>L_VQ</span><span>squared distance to the nearest code (codebook only)</span>
    </div>
    <h3>Optimiser and schedule</h3>
    <div class="kvt" style="color:#333">
     <span>optimiser</span><span>AdamW β = (0.9, 0.95), weight decay 0.1 on matrices and norm gains</span>
     <span>learning rate</span><span>peak 6e-4, linear warm-up 300 steps, cosine to 10 %, clip 1.0</span>
     <span>batch</span><span>2 rows × accumulate 16 × 1,024 tokens = 32,768 tokens / step</span>
     <span>loop depth</span><span>R ~ U{1..4} per step, R = 4 at evaluation</span>
     <span>precision</span><span>float32 on Apple MPS (M2, 16 GB), no compile</span>
     <span>main run</span><span>4,050 steps ≈ 132.7M tokens, 24 h training-time budget</span>
    </div>
   </div>
   <div>
    <h3 style="margin-top:0">Training mixture (by tokens)</h3>
    <div class="pbar" style="margin-bottom:6px">
     <div style="width:45%;background:#d6e6f7">language 45 %</div><div style="width:22%;background:#fbdcc8">symbolic 22 %</div>
     <div style="width:10%;background:#e6e0f5">code 10</div><div style="width:13%;background:#fff2bf">retrieval 13</div><div style="width:10%;background:#d2efea">long 10</div>
    </div>
    <div style="font-size:13px;color:#555">language = 45 % TinyStories + 55 % WikiText-103 · 264.4M training tokens · BPE 8,192 · T = 1,024</div>
    <h3>Long-range probe families (evidence ≥ 128 tokens back)</h3>
    <div class="kvt" style="color:#333;font-size:13px">
     <span>lr_recall1</span><span>recall one single-token value bound to a 3-syllable name</span>
     <span>lr_carry3</span><span>carry 3 values across the gap (per-offset accuracy)</span>
     <span>lr_multi4</span><span>4 bindings, one queried</span>
     <span>lr_hop</span><span>multi-hop chains, depth 2–4 (OOD 5–6)</span>
     <span>lr_bind</span><span>variable binding, digit answers</span>
    </div>
    <div style="font-size:12.5px;color:#666;margin-top:6px">In-distribution gaps 128–420 tokens, OOD 500–820. All beyond the decoder's 70-token reach, so only the concept stream can carry the answer.</div>
    <h3>Evaluation interventions</h3>
    <div style="font-size:13px;color:#333">intact · core off (null plan everywhere) · plan swap (another row's u, ĉ_pred) · u off · ĉ_pred off · R = 1…4 · truncation curves r ∈ {72, 128, 256, 512, 1024}</div>
   </div>
  </div>`;
  return d;
}

const FIGS = [
  { id: "overview", t: "Whole model", fn: figOverview, cap: "Up through the encoder, down through the concept module, up through the confined decoder. Thin arrows are reads, thick arrows are writes; only tok (reach 63) and the per-position plan reach the decoder." },
  { id: "enc", t: "Encoder layers", fn: figEncoderLayers, cap: "Both layer types share the gated attention block (middle) and the gated-residual read/write grammar; they differ only in which streams they read and write." },
  { id: "reach", t: "Pattern & reach", fn: figReach, cap: "Reach is exact (tested by single-token perturbation): the L stream sees 63 tokens back, the confined decoder 70." },
  { id: "engram", t: "Engram memory", fn: figEngram, cap: "A hashed n-gram memory after layer 0. 50.3M parameters are a lookup table (8 rows read per token); only 0.32M are compute." },
  { id: "crc", t: "Concept pooling", fn: figCRC, cap: "Each token mixes its embedding, local and global states; 4 consecutive tokens are pooled into one concept on the sphere." },
  { id: "core", t: "Concept module & NCP", fn: figCore, cap: "One shared FULL layer looped R times (reads X and S, writes S), then a distance head predicts the next concept's product-quantised codes." },
  { id: "dec", t: "Confined decoder", fn: figDecoder, cap: "In the confined layout the plan of position t reaches only position t; the stack (control) mixes neighbouring positions' plans." },
  { id: "plan", t: "Plan timing", fn: figPlan, cap: "Which chunk each position reads, and why the target is never inside it." },
  { id: "obj", t: "Objective & data", fn: figObjective, cap: "(from code/train.py, data/MANIFEST.json, code/probes.py)." },
];
