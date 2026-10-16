/* v2 architecture figures (d 256 · Engram 80M · concept module D6 with 16 experts · confined decoder + NextLat).
   Every size comes from data/arch_v2.json (model_v2/make_arch_v2.py → model_v2/v2_budget.py, which builds the real classes). */
const SRC2 = {
  spec: "model_v2/MODEL_V2_19M.md", budget: "model_v2/v2_budget.py", enc: "code/hrm_encoder.py (unchanged design, d 256)",
  engram: "code/hrm_encoder.py · Engram (rows 208,333)", core: "concept_v2/DESIGN_D6.md · MODEL_V2_19M.md §6 (new module concept_core_d6.py)",
  dec: "code/hrm_decoder.py · Decoder, PlanBuilder (W_u 256 → 256)", nl: "code/hrm_decoder.py · class NextLat · MODEL_V2_19M.md §7",
};
const V2C = { core: "#2e9c8c", exp: "#7a5bd0", nl: "#7a5bd0", det: "#d0667a" };

function legend2(g, x, y, items) {
  let cx = x;
  for (const it of items) {
    if (it.line) { el("line", { x1: cx, y1: y, x2: cx + 22, y2: y, stroke: it.line, "stroke-width": 2.2, "stroke-dasharray": it.dash || null }, g); cx += 28; }
    else { el("rect", { x: cx, y: y - 7, width: 16, height: 14, rx: 3, fill: (PAL[it.k] || PAL.white).f, stroke: (PAL[it.k] || PAL.white).s, "stroke-dasharray": it.dash || null }, g); cx += 21; }
    txt(g, cx, y, it.t, { size: 11.5, anchor: "start", color: "#444" });
    cx += Math.max(40, it.t.length * 6.3) + 16;
  }
}

/* =========================================================================================== 1 whole model */
function f2Overview(A) {
  const P = A.params, T = A.table, e = A.ecfg, c = A.core;
  const svg = makeSvg(1220, 900), g = el("g", {}, svg);
  legend2(g, 30, 20, [{ line: COL.L, t: "L / tok stream" }, { line: COL.G, t: "G stream" }, { line: COL.plan, t: "concept / plan" }, { line: V2C.nl, t: "training only", dash: "5 4" },
    { k: "attn", t: "SWA layer" }, { k: "full", t: "full attention" }, { k: "ffn", t: "experts" }, { k: "mem", t: "memory" }, { k: "loss", t: "loss" }]);

  // ---------------- encoder
  frame(g, { x: 30, y: 60, w: 360, h: 800, title: `Encoder · ${fmtP(T.encoder_non_embedding)}`, note: `+ ${fmtP(T.engram_table)} Engram` });
  const bx = 90, bw = 240;
  box(g, { x: bx, y: 812, w: bw, h: 30, k: "io", label: "Token ids (B, 1024)", tip: "<b>Input</b><br>V = 8,192, T = 1,024, packed with document ids." });
  box(g, { x: bx, y: 766, w: bw, h: 32, k: "emb", label: `Embedding ${fmtI(e.vocab_size)} × ${e.d_model}`, sub: "tied with the head", size: 12, tip: `<b>Token embedding E</b> · ${fmtI(T.token_embedding)} params (not counted)<br>e0 = rms(E[x]); logits = h Eᵀ.`, src: SRC2.enc });
  arrow(g, [[210, 812], [210, 799]]);
  const rows = [["S", 0], ["E"], ["S", 1], ["S", 2], ["F", 3], ["S", 4], ["S", 5], ["S", 6], ["F", 7]];
  rows.forEach((r, i) => {
    const y = 716 - 50 * i;
    const k = r[0] === "S" ? "attn" : r[0] === "F" ? "full" : "mem";
    const lab = r[0] === "E" ? "Engram (after S0)" : `${r[0]}${r[1]} · ${r[0] === "S" ? "sliding window 8" : "full attention"}`;
    const sub = r[0] === "E" ? `8 × ${fmtI(e.engram_rows)} × ${e.engram_dim} = ${fmtP(P.engram_table)}` : `${fmtP(r[0] === "S" ? P.enc_layer_S : P.enc_layer_F)} · reads ${r[0] === "S" ? "{L}" : "{L,G}"} · writes ${r[0] === "S" ? "{L,G}" : "{G}"}`;
    box(g, { x: bx, y, w: bw, h: 36, k, label: lab, sub, size: 12, src: r[0] === "E" ? SRC2.engram : SRC2.enc,
      tip: r[0] === "E" ? `<b>Engram</b> · ${fmtI(P.engram_table)} lookups + ${fmtI(P.engram_proj)} compute<br>orders 2, 3 × 4 hash heads; rows 131,071 → <b>208,333</b> (prime) in v2; gate α = σ(rms(h)·rms(k)/√256)`
        : `<b>${r[0]} layer ${r[1]}</b> · ${fmtI(r[0] === "S" ? P.enc_layer_S : P.enc_layer_F)} params<br>d 256, 4 heads × 64, GQA 4:2, SwiGLU 256 → 704 → 256`});
    if (i) arrow(g, [[210, y + 50], [210, y + 37]], { c: COL.G, m: "ahG", w: 1.2 });
  });
  arrow(g, [[210, 766], [210, 753]], { c: COL.L, m: "ahL" });
  box(g, { x: bx, y: 256, w: bw, h: 30, k: "gate", label: "CRC depth-mix {e0, L, G}", size: 12, tip: `<b>CRC router</b> · ${P.crc_router} params` });
  box(g, { x: bx, y: 206, w: bw, h: 34, k: "conc", label: "Pool 4 × 256 → 384 · on sphere √384", size: 11.5, sub: `ĉ (B, 256, 384) · ${fmtP(P.pool)}`, tip: `<b>Chunk pooling</b> · ${fmtI(P.pool)} params<br>d_c stays 384 so the PQ codebook (48 × 128 × 8) is unchanged.` });
  arrow(g, [[210, 316], [210, 287]], { c: COL.G, m: "ahG" }); arrow(g, [[210, 256], [210, 241]], { c: COL.plan, m: "ahP" });
  box(g, { x: bx, y: 120, w: bw, h: 30, k: "norm", label: "tok = zcRMSNorm(L) · reach 63", size: 12, tip: "<b>tok</b>: the decoder's only view of the tokens (keys and values)." });
  arrow(g, [[60, 740], [60, 135], [bx, 135]], { c: COL.L, m: "ahL", w: 1.6 });
  note(g, 64, 760, "L", { color: COL.L, size: 12 });

  // ---------------- concept module
  frame(g, { x: 430, y: 60, w: 380, h: 800, title: `Concept module v2 (D6) · ${fmtP(T.concept_module)}`, stroke: "#9fd3ca" });
  const cx = 480, cw = 280;
  box(g, { x: cx, y: 110, w: cw, h: 34, k: "conc", label: "X̃ = W_in ĉ   ·   S₀ = W_s ĉ", sub: "384 → 256 each", size: 12, tip: `<b>Inputs</b> · ${fmtI(P.core_io)} params<br>X̃ is re-read by every sublayer of every pass; S₀ is a distinct projection (no dead start).`, src: SRC2.core });
  arrow(g, [[330, 223], [405, 223], [405, 127], [cx, 127]], { c: COL.plan, m: "ahP", w: 1.8 });
  dim(g, 340, 214, "ĉ");
  for (let l = 0; l < 4; l++) {
    const y = 180 + l * 70;
    box(g, { x: cx, y, w: cw, h: 50, k: "full", label: `Layer ${l + 1} · full causal attention 4×64`, sub: `GR read [X̃, S] · router → top-2 of ${c.ffn.E}`, size: 12,
      tip: `<b>Core layer ${l + 1}</b> (distinct weights) · ${fmtI(P.core_layer_dense)} params + shared experts<br>GR read over [X̃, S] (rank 32) → gated full attention over the 256 chunk positions (doc mask, NoPE) → S += s·attn → GR read → router 256 → ${c.ffn.E} → top-2 experts → S += Σ g_k expert_k`, src: SRC2.core });
    if (l) arrow(g, [[cx + cw / 2, y - 20], [cx + cw / 2, y]], { c: COL.plan, m: "ahP" });
    arrow(g, [[cx + cw + 36, y + 25], [cx + cw, y + 25]], { c: "#b08a5a", m: "ah", w: 1 });
  }
  arrow(g, [[cx + cw / 2, 144], [cx + cw / 2, 180]], { c: COL.plan, m: "ahP" });
  // expert shelf
  box(g, { x: cx + cw + 38, y: 180, w: 28, h: 260, k: "ffn", label: "", tip: `<b>Shared expert pool</b> · ${c.ffn.E} SwiGLU experts 256 → ${c.ffn.d_expert} → 256 · ${fmtI(P.core_pool)} params<br>shared by the 4 layers; top-2 → active width ${c.ffn.active_width}.` });
  for (let k = 0; k < 16; k++) el("rect", { x: cx + cw + 42, y: 186 + k * 15.6, width: 20, height: 11, rx: 2, fill: [2, 9].includes(k) ? "#dc8b56" : "#fde3d0", stroke: "#dc8b56", "stroke-width": 0.6 }, g);
  txt(g, cx + cw + 52, 452, `${c.ffn.E} experts`, { size: 10.5, color: "#a0602e" });
  // loop arrow
  arrow(g, [[cx, 440], [cx - 28, 440], [cx - 28, 205], [cx, 205]], { c: COL.plan, m: "ahP", w: 2 });
  txt(g, cx - 38, 322, "× K passes", { size: 12, weight: 700, color: COL.plan, anchor: "end" });
  note(g, cx - 38, 338, "8–12 train", { anchor: "end" }); note(g, cx - 38, 352, "6 at eval", { anchor: "end" });
  box(g, { x: cx, y: 470, w: cw, h: 30, k: "norm", label: "carry LN_h(S) · detached every 4 passes", size: 11.5, tip: "<b>Carry</b>: bounded by LN_h; the carry is detached between windows of 4 passes (D3), each window graded." });
  arrow(g, [[cx + cw / 2, 440], [cx + cw / 2, 470]], { c: COL.plan, m: "ahP" });
  box(g, { x: cx, y: 530, w: cw, h: 34, k: "conc", label: "NCP head · every pass", sub: "W_q 256 → 384 · distance to codes", size: 12, tip: `<b>NCP head</b> · ${fmtI(P.ncp_head)} params<br>every pass r predicts the true next chunk's 48 PQ codes from u_r.` });
  arrow(g, [[cx + cw / 2, 500], [cx + cw / 2, 530]], { c: COL.plan, m: "ahP" });
  box(g, { x: cx, y: 590, w: cw, h: 30, k: "mem", label: `PQ codebook ${A.S} × ${A.N} × 8 (336 bits)`, size: 11.5, tip: `<b>PQ codebook</b> · ${fmtI(P.codebook)} (not counted), trained by L_VQ only; unchanged from v1.` });
  arrow(g, [[cx + cw / 2, 590], [cx + cw / 2, 565]], { c: "#d1a92f", m: "ah", w: 1 });
  box(g, { x: cx + 20, y: 660, w: 110, h: 30, k: "loss", label: "0.1 · L_NCP", size: 11.5, tip: "mean over all K passes (grounded per-pass target)" });
  box(g, { x: cx + 150, y: 660, w: 110, h: 30, k: "loss", label: "L_VQ", size: 11.5, tip: "codebook only" });
  box(g, { x: cx + 20, y: 710, w: 240, h: 30, k: "loss", label: "0.1 · consistency (final window)", size: 11.5, tip: "LoopFormer: ‖sg(u_K) − u_r‖² / d over the passes of the final window only" });
  box(g, { x: cx + 20, y: 760, w: 240, h: 30, k: "loss", label: "0.01 · LB + 0.001 · z (per window)", size: 11.5, tip: "MoE load balance and router z-loss, accumulated per window" });
  arrow(g, [[cx, 547], [cx - 14, 547], [cx - 14, 675], [cx + 20, 675]], { c: "#d0667a", m: "ahR", w: 1 });

  // ---------------- decoder
  frame(g, { x: 850, y: 60, w: 340, h: 800, title: `Decoder (confined) · ${fmtP(T.decoder)}`, stroke: "#f1c6a6" });
  const dx = 900, dw = 240;
  box(g, { x: dx, y: 740, w: dw, h: 40, k: "plan", label: "Plan P = W_u u_K + W_c ĉ_pred + o_r", sub: "j*(t) = ⌊(t+1)/4⌋ − 1", size: 11.5, tip: `<b>Plan builder</b> · ${fmtI(P.dec_plan)} params<br>W_u 256 → 256 (u arrives at 256), W_c 384 → 256.`, src: SRC2.dec });
  arrow(g, [[cx + cw, 485], [830, 485], [830, 760], [dx, 760]], { c: COL.plan, m: "ahP", w: 1.8 });
  dim(g, 836, 500, "u_K, ĉ_pred");
  for (let l = 0; l < 4; l++) {
    const y = 660 - l * 70;
    box(g, { x: dx, y, w: dw, h: 46, k: "attn", label: `Confined layer ${l + 1}`, sub: "q from own D · k, v from tok only (w 8)", size: 12, tip: `<b>Confined layer ${l + 1}</b> · ${fmtI(P.dec_layer)} params<br>SwiGLU 256 → 704 → 256; no position reads another's plan.`, src: SRC2.dec });
    arrow(g, [[dx + dw / 2, y + 46 + (l ? 24 : 34)], [dx + dw / 2, y + 46]], { c: COL.plan, m: "ahP" });
    arrow(g, [[1170, y + 23], [dx + dw, y + 23]], { c: COL.L, m: "ahL", w: 1 });
  }
  arrow(g, [[330, 135], [400, 135], [400, 46], [1170, 46], [1170, 683]], { c: COL.L, m: "ahL", w: 1.6, head: false });
  dim(g, 560, 55, "tok (reach 63) → keys and values of every decoder layer");
  box(g, { x: dx, y: 390, w: dw, h: 28, k: "norm", label: "h = zcRMSNorm(D)", size: 12 });
  arrow(g, [[dx + dw / 2, 450], [dx + dw / 2, 419]], { c: COL.plan, m: "ahP" });
  box(g, { x: dx, y: 330, w: dw, h: 32, k: "emb", label: "logits = h Eᵀ (tied)", size: 12 });
  arrow(g, [[dx + dw / 2, 390], [dx + dw / 2, 363]]);
  box(g, { x: dx, y: 270, w: 110, h: 30, k: "loss", label: "L_NTP", size: 12 });
  arrow(g, [[dx + 55, 330], [dx + 55, 301]], { c: "#d0667a", m: "ahR" });
  box(g, { x: dx + 20, y: 150, w: 200, h: 46, k: "gate", label: "NextLat ψ (training only)", sub: "ĥ_{t+1} = h_t + f(LN[h_t; E x_{t+1}])", size: 11.5, dash: "5 4",
    tip: `<b>NextLat</b> · ${fmtI(P.nextlat_psi)} params, training only<br>on the decoder's final hidden state h (where the paper puts it); SmoothL1 to sg(h_{t+1}) + KL through the frozen head.`, src: SRC2.nl });
  arrow(g, [[dx + dw - 20, 390], [dx + dw - 20, 197]], { c: V2C.nl, m: "ah", dash: "5 4", w: 1.3 });
  box(g, { x: dx + 20, y: 214, w: 200, h: 28, k: "loss", label: "1.0 · L_next-h + 1.0 · L_KL", size: 11.5 });
  return svg;
}

/* =========================================================================================== 2 encoder */
function f2Encoder(A) {
  const P = A.params, e = A.ecfg;
  const svg = makeSvg(1180, 520), g = el("g", {}, svg);
  frame(g, { x: 20, y: 20, w: 520, h: 480, title: "S layer (×6) · reads {L} · writes {L, G}", stroke: "#b9d1ef" });
  frame(g, { x: 640, y: 20, w: 520, h: 480, title: "F layer (×2) · reads {L, G} · writes {G} only", stroke: "#f1c6a6" });
  [[20, "S", P.enc_layer_S, P.enc_layer_S_gr], [640, "F", P.enc_layer_F, P.enc_layer_F_gr]].forEach(([x0, k, tot, gr]) => {
    const x = x0 + 60, w = 400;
    box(g, { x, y: 430, w, h: 34, k: "gate", label: k === "S" ? "GR read of {L}" : "GR read of {L, G}", sub: `σ(U V rms(R) + 2) · rank 32 · ${fmtI(gr)} params (both reads + writes)`, size: 12 });
    box(g, { x, y: 350, w, h: 50, k: k === "S" ? "attn" : "full", label: k === "S" ? `Gated SWA, w ${e.window}, partial RoPE ${e.rope_dims}/${e.head_dim}` : "Gated full causal attention, NoPE + τ_h",
      sub: `q 256 → 4×64 · k, v 256 → 2×64 · out gate σ · ${fmtI(P.enc_layer_S_attn)} params`, size: 12, tip: k === "S" ? "q/k/v causal depthwise conv k 4 (document-aware), q/k zcRMSNorm" : "per-head temperature τ_h; within the document" });
    box(g, { x, y: 270, w, h: 34, k: "gate", label: k === "S" ? "write L and G (per-stream scale)" : "write G only", size: 12 });
    box(g, { x, y: 190, w, h: 34, k: "gate", label: "GR read (again)", size: 12 });
    box(g, { x, y: 110, w, h: 44, k: "ffn", label: `SwiGLU 256 → ${e.d_ff} → 256`, sub: `${fmtI(P.enc_layer_S_ffn)} params`, size: 12 });
    box(g, { x, y: 60, w, h: 30, k: "gate", label: k === "S" ? "write L and G" : "write G only", size: 12 });
    [[430, 401], [350, 305], [270, 225], [190, 155], [110, 91]].forEach(([a, b]) => arrow(g, [[x + w / 2, a], [x + w / 2, b]], { c: k === "S" ? COL.L : COL.G, m: k === "S" ? "ahL" : "ahG" }));
    txt(g, x0 + 260, 486, `layer total ${fmtI(tot)} params`, { size: 12, weight: 600, color: "#555" });
  });
  txt(g, 590, 250, "same\ngrammar", { size: 12, color: COL.grey });
  return svg;
}

/* =========================================================================================== 3 engram */
function f2Engram(A) {
  const P = A.params, e = A.ecfg, v1 = A.v1;
  const svg = makeSvg(1180, 440), g = el("g", {}, svg);
  box(g, { x: 30, y: 190, w: 170, h: 50, k: "io", label: "canonical ids", sub: "x_{t−2}, x_{t−1}, x_t", size: 12 });
  box(g, { x: 250, y: 150, w: 190, h: 130, k: "white", label: "hash 2-grams, 3-grams", sub: "× 4 heads = 8 row ids", size: 12, tip: "multiplicative-XOR hash mod 2⁶¹, then mod rows (prime, coprime with 2¹³). Another document → boundary id 0." });
  arrow(g, [[200, 215], [250, 215]]);
  for (let t = 0; t < 8; t++) {
    const y = 40 + t * 46;
    box(g, { x: 500, y, w: 190, h: 36, k: "mem", label: `table ${t + 1} · ${t < 4 ? "2-gram" : "3-gram"} head ${t % 4 + 1}`, sub: `${fmtI(e.engram_rows)} × ${e.engram_dim}`, size: 11 });
    arrow(g, [[440, 215], [470, 215], [470, y + 18], [500, y + 18]], { c: COL.grey, m: "ahg", w: 0.8, r: 3 });
    arrow(g, [[690, y + 18], [740, 215]], { c: "#d1a92f", m: "ah", w: 0.8, head: false });
  }
  box(g, { x: 740, y: 180, w: 150, h: 70, k: "mem", label: "concat 8 × 48", sub: "e_t (384)", size: 12 });
  box(g, { x: 930, y: 130, w: 220, h: 40, k: "gate", label: "k = W_K e · v = W_V e", sub: "384 → 256 · W_V zero-init", size: 11.5 });
  box(g, { x: 930, y: 190, w: 220, h: 40, k: "gate", label: "α = σ(rms(h)·rms(k)/√256)", size: 11.5 });
  box(g, { x: 930, y: 250, w: 220, h: 40, k: "attn", label: "y = SiLU(conv(α v)) → L, G", size: 11.5 });
  arrow(g, [[890, 215], [930, 150]]); arrow(g, [[1040, 170], [1040, 190]]); arrow(g, [[1040, 230], [1040, 250]]);
  txt(g, 1040, 330, `v1: 8 × 131,071 × 48 = ${fmtP(v1.engram_table)}`, { size: 12, color: COL.grey });
  txt(g, 1040, 350, `v2: 8 × ${fmtI(e.engram_rows)} × 48 = ${fmtP(P.engram_table)}`, { size: 13, weight: 700 });
  txt(g, 1040, 372, "only the row count changes (fewer collisions)", { size: 11, color: COL.grey });
  txt(g, 1040, 392, `compute ${fmtI(P.engram_proj)} params · +0.48 GB optimizer state`, { size: 11, color: COL.grey });
  return svg;
}

/* =========================================================================================== 4 concepts */
function f2Concepts(A) {
  const P = A.params;
  const svg = makeSvg(1180, 330), g = el("g", {}, svg);
  for (let i = 0; i < 8; i++) {
    const x = 40 + i * 70;
    box(g, { x, y: 220, w: 58, h: 40, k: i < 4 ? "attn" : "white", label: `t${i}`, size: 12 });
    arrow(g, [[x + 29, 220], [x + 29, 190]], { c: COL.grey, m: "ahg", w: 0.9 });
  }
  box(g, { x: 40, y: 150, w: 268, h: 40, k: "gate", label: "mix_t = softmax(W rms G) · {e0, rms L, rms G}", size: 11 });
  box(g, { x: 320, y: 150, w: 268, h: 40, k: "gate", label: "mix per token", size: 11 });
  box(g, { x: 40, y: 80, w: 268, h: 40, k: "conc", label: "concat 4 × 256 = 1,024 → W_pool → 384", size: 11.5, tip: `<b>W_pool</b> 1,024 → 384 · ${fmtI(P.pool)} params · orthonormal mean-pool init` });
  box(g, { x: 320, y: 80, w: 268, h: 40, k: "conc", label: "next chunk", size: 11.5 });
  arrow(g, [[174, 150], [174, 121]], { c: COL.plan, m: "ahP" }); arrow(g, [[454, 150], [454, 121]], { c: COL.plan, m: "ahP" });
  circ(g, 174, 40, 22, { f: "#d7f2ed", s: "#3a9d8f", label: "ĉ₀" }); circ(g, 454, 40, 22, { f: "#d7f2ed", s: "#3a9d8f", label: "ĉ₁" });
  arrow(g, [[174, 80], [174, 63]], { c: COL.plan, m: "ahP" }); arrow(g, [[454, 80], [454, 63]], { c: COL.plan, m: "ahP" });
  const x = 680;
  txt(g, x, 60, "Why d_c stays 384 while d = 256", { size: 14, weight: 700, anchor: "start" });
  ["The PQ codebook needs d_c divisible by its 48 segments; 256 is not.",
   "Keeping 384 leaves a measured, working component untouched:",
   "   48 × 128 codes × 8 dims = 336 bits per chunk, codes_min 0.99, recall1 0.992 in v1.",
   "A concept compresses 4 × 256 = 1,024 token dims into 384 (2.7 ×).",
   "ĉ = √384 · c / ‖c‖: every concept on the same sphere; only its direction matters.",
   "Alternative (ablation): 32 × 128 × 8 = 224 bits at d_c 256."].forEach((s, i) => txt(g, x, 90 + i * 24, s, { size: 12, anchor: "start", color: i === 2 ? "#666" : COL.ink }));
  return svg;
}

/* =========================================================================================== 5 core structure */
function f2Core(A) {
  const P = A.params, c = A.core;
  const svg = makeSvg(1180, 600), g = el("g", {}, svg);
  frame(g, { x: 20, y: 20, w: 700, h: 560, title: "One core layer (4 distinct layers per pass)", stroke: "#9fd3ca" });
  const x = 90, w = 420;
  box(g, { x: 60, y: 510, w: 140, h: 40, k: "conc", label: "X̃ (clean input)", size: 12, tip: "re-read every sublayer of every pass" });
  box(g, { x: 400, y: 510, w: 140, h: 40, k: "plan", label: "S (carry)", size: 12, tip: "the running state; S₀ = W_s ĉ" });
  box(g, { x, y: 430, w, h: 44, k: "gate", label: "GR read: mean_i σ(U_i V_i rms(R_i) + 2)(1+γ_i) rms(R_i)", sub: "R ∈ {X̃, S} · rank 32", size: 11.5, tip: "Qwen-style elementwise gated read; the two branches differ from step 0" });
  arrow(g, [[130, 510], [130, 475]], { c: COL.plan, m: "ahP" }); arrow(g, [[470, 510], [470, 475]], { c: COL.plan, m: "ahP" });
  box(g, { x, y: 350, w, h: 50, k: "full", label: "gated full causal attention · 4 heads × 64, GQA kv 2", sub: "doc mask · NoPE · sigmoid output gate · no q/k/v convs · W_o zero-init", size: 11.5, tip: `attention part ${fmtI(262272)} params` });
  arrow(g, [[x + w / 2, 430], [x + w / 2, 401]]);
  box(g, { x, y: 290, w, h: 30, k: "plan", label: "S ← S + s · attn", size: 12 });
  arrow(g, [[x + w / 2, 350], [x + w / 2, 321]]);
  box(g, { x, y: 230, w, h: 30, k: "gate", label: "GR read of {X̃, S} again", size: 12 });
  arrow(g, [[x + w / 2, 290], [x + w / 2, 261]]);
  box(g, { x, y: 170, w, h: 34, k: "gate", label: `router 256 → ${c.ffn.E} (per layer) → top-2, gates g_k`, size: 12, tip: "routing depends on the state S, so different passes can pick different experts, with no pass index" });
  arrow(g, [[x + w / 2, 230], [x + w / 2, 205]]);
  box(g, { x, y: 100, w, h: 40, k: "ffn", label: "Σ_k g_k · expert_k(x)   (2 of 16)", sub: "SwiGLU 256 → 512 → 256 · W_2 zero-init", size: 12 });
  arrow(g, [[x + w / 2, 170], [x + w / 2, 141]]);
  box(g, { x, y: 56, w, h: 28, k: "plan", label: "S ← S + s · experts   → next layer", size: 12 });
  arrow(g, [[x + w / 2, 100], [x + w / 2, 85]]);
  // pool
  frame(g, { x: 560, y: 60, w: 140, h: 440, title: "pool", stroke: "#e7b891" });
  for (let k = 0; k < c.ffn.E; k++) box(g, { x: 580, y: 96 + k * 24.5, w: 100, h: 19, k: "ffn", label: `expert ${k + 1}`, size: 10, rx: 4, fill: [3, 11].includes(k) ? "#f5b98f" : null });
  arrow(g, [[580, 120], [511, 120]], { c: "#dc8b56", m: "ah", w: 1 });
  // right: facts
  const fx = 760;
  txt(g, fx, 50, "Concept module v2 = D6 LOOPFORMER-MoE-DS", { size: 15, weight: 700, anchor: "start" });
  [["params", fmtI(P.core_total)], ["per layer (dense part)", fmtI(P.core_layer_dense)], ["expert pool", `${c.ffn.E} × ${fmtI(P.core_expert)} = ${fmtI(P.core_pool)}`],
   ["active FFN width", `${c.ffn.active_width} (top-2 × 512)`], ["inputs W_in, W_s", fmtI(P.core_io)], ["NCP head", fmtI(P.ncp_head)],
   ["passes", "K ~ U{8..12} train · R 6 eval"], ["carry", "LN_h(S), detached every 4 passes"], ["pass index input", "none (routing gives variation)"],
   ["identity at init", "W_o, W_2 zero-init (one per path)"], ["MoE on MPS", "capacity 1.25 batched dispatch"], ["new metric", "dropped-token fraction per layer per pass"]]
    .forEach(([a, b], i) => { txt(g, fx, 86 + i * 26, a, { size: 12, anchor: "start", color: "#666" }); txt(g, fx + 170, 86 + i * 26, b, { size: 12, anchor: "start", weight: 600 }); });
  txt(g, fx, 420, "Why it should work (vs v1's idle loop)", { size: 13, weight: 700, anchor: "start" });
  ["1. identity at init, bounded carry (LN_h)", "2. many passes (8–12) in training", "3. every pass graded on the true next code", "4. detached windows: each must improve its input", "5. state-routed experts: a new computation per pass"]
    .forEach((s, i) => txt(g, fx, 446 + i * 22, s, { size: 12, anchor: "start" }));
  return svg;
}

/* =========================================================================================== 6 pass schedule */
function f2Schedule(A) {
  const svg = makeSvg(1180, 470), g = el("g", {}, svg);
  const ks = [8, 10, 12], x0 = 150, pw = 72;  // rows start at y 80
  const wins = K => { const n = Math.floor(K / 4), out = []; let s = 1; for (let i = 0; i < n; i++) { const len = i === n - 1 ? K - 4 * i : 4; out.push([s, s + len - 1]); s += len; } return out; };
  ks.forEach((K, row) => {
    const y = 84 + row * 116;
    txt(g, 30, y + 22, `K = ${K}`, { size: 14, weight: 700, anchor: "start" });
    const W = wins(K);
    W.forEach(([a, b], wi) => {
      const last = wi === W.length - 1;
      el("rect", { x: x0 + (a - 1) * pw - 6, y: y - 8, width: (b - a + 1) * pw + 12 - 12, height: 60, rx: 10, fill: last ? "#f2e8fc" : "#eef7f5", stroke: last ? "#a07cc5" : "#9fd3ca" }, g);
      txt(g, x0 + (a - 1) * pw + 4, y - 16, last ? `window ${wi + 1} · + consistency → u_K` : `window ${wi + 1}`, { size: 11, anchor: "start", color: last ? "#7a5bd0" : "#2e9c8c", weight: 600 });
      if (!last) { const xd = x0 + b * pw - 9; el("line", { x1: xd, y1: y - 12, x2: xd, y2: y + 56, stroke: "#d0667a", "stroke-width": 2, "stroke-dasharray": "4 3" }, g); txt(g, xd, y + 68, "detach", { size: 10, color: "#d0667a" }); }
    });
    for (let r = 1; r <= K; r++) {
      const x = x0 + (r - 1) * pw;
      box(g, { x, y, w: pw - 18, h: 28, k: "full", label: `r${r}`, size: 11, rx: 6, tip: `pass ${r}: 4 layers; its readout u_${r} is graded on the true next-chunk code (NCP)` });
      el("circle", { cx: x + (pw - 18) / 2, cy: y + 42, r: 4, fill: "#d0667a" }, g);
      if (r === 6) { el("path", { d: `M${x + (pw - 18) / 2},${y - 46} v18`, stroke: "#111", "stroke-width": 2, "marker-end": mk(g, "ah") }, g); if (!row) txt(g, x + 27, y - 54, "eval R 6", { size: 11, weight: 700 }); }
    }
  });
  const ly = 430;
  el("circle", { cx: 160, cy: ly, r: 4, fill: "#d0667a" }, g); txt(g, 172, ly, "NCP loss on every pass (0.1 × mean)", { size: 12, anchor: "start" });
  el("line", { x1: 450, y1: ly - 8, x2: 450, y2: ly + 8, stroke: "#d0667a", "stroke-width": 2, "stroke-dasharray": "4 3" }, g); txt(g, 462, ly, "carry detached, window backward done", { size: 12, anchor: "start" });
  el("rect", { x: 760, y: ly - 8, width: 22, height: 16, rx: 4, fill: "#f2e8fc", stroke: "#a07cc5" }, g); txt(g, 790, ly, "final window: LoopFormer consistency, NTP and NextLat enter here", { size: 12, anchor: "start" });
  return svg;
}

/* =========================================================================================== 7 looped experts */
function f2MoE(A) {
  const c = A.core, E = c.ffn.E;
  const svg = makeSvg(1180, 520), g = el("g", {}, svg);
  txt(g, 30, 30, "Illustration (not measured): which 2 of 16 experts one chunk position uses, pass by pass, in layer 2", { size: 13, weight: 600, anchor: "start" });
  const x0 = 120, y0 = 60, cw = 44, ch = 26, R = 12;
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let e = 0; e < E; e++) txt(g, x0 - 12, y0 + e * ch + ch / 2, `expert ${e + 1}`, { size: 10.5, anchor: "end", color: "#666" });
  for (let r = 0; r < R; r++) {
    txt(g, x0 + r * cw + cw / 2, y0 + E * ch + 16, `r${r + 1}`, { size: 11, color: "#555" });
    const a = Math.floor(rnd() * E); let b = Math.floor(rnd() * E); if (b === a) b = (a + 5) % E;
    for (let e = 0; e < E; e++) {
      const on = e === a || e === b, w = e === a ? 0.62 : 0.38;
      el("rect", { x: x0 + r * cw + 2, y: y0 + e * ch + 2, width: cw - 4, height: ch - 4, rx: 4, fill: on ? `rgba(220,139,86,${0.35 + w})` : "#f6f2ee", stroke: on ? "#c2703a" : "#ece4dc" }, g);
    }
  }
  [4, 8].forEach(r => el("line", { x1: x0 + r * cw, y1: y0 - 6, x2: x0 + r * cw, y2: y0 + E * ch + 4, stroke: "#d0667a", "stroke-width": 2, "stroke-dasharray": "4 3" }, g));
  const fx = 700;
  txt(g, fx, 80, "Why experts in a looped core", { size: 14, weight: 700, anchor: "start" });
  ["A dense layer applied again computes the same map; only the input changes.",
   "A router reads the state S, so a later pass can pick different experts:",
   "a different computation without a pass index (defined for any R).",
   "The looped-MoE paper (Sparse Layers are Critical to Scaling Looped LMs)",
   "measured later passes choosing different experts and scaling through sparsity.",
   `${E} experts → ${E * (E - 1) / 2} expert pairs to route into (8 experts: 28).`,
   "Active width stays 1,024: more experts add capacity, not compute."].forEach((s, i) => txt(g, fx, 110 + i * 24, s, { size: 12, anchor: "start" }));
  txt(g, fx, 300, "Checks that decide it", { size: 13, weight: 700, anchor: "start" });
  ["E11/E16: routing divergence across passes and windows (> 0)", "E15: no pool collapse (load entropy, dropped tokens < 2 %)", "A4: a dense-FFN twin must be worse, or experts did not matter"]
    .forEach((s, i) => txt(g, fx, 326 + i * 22, s, { size: 12, anchor: "start" }));
  return svg;
}

/* =========================================================================================== 8 decoder + plan */
function f2Decoder(A) {
  const P = A.params;
  const svg = makeSvg(1180, 460), g = el("g", {}, svg);
  txt(g, 30, 28, "Which plan each position reads: j*(t) = ⌊(t+1)/4⌋ − 1 (the newest chunk ending at or before t)", { size: 13, weight: 600, anchor: "start" });
  const x0 = 60, cw = 52;
  for (let t = 0; t < 20; t++) {
    const j = Math.floor((t + 1) / 4) - 1, x = x0 + t * cw;
    box(g, { x, y: 60, w: cw - 6, h: 30, k: Math.floor(t / 4) % 2 ? "attn" : "white", label: `x${t}`, size: 11, rx: 5 });
    box(g, { x, y: 120, w: cw - 6, h: 30, k: j < 0 ? "io" : "plan", label: j < 0 ? "null" : `u${j}`, size: 11, rx: 5, tip: `position ${t} predicts x${t + 1} from plan ${j < 0 ? "null" : "chunk " + j}` });
    if (t >= 6 && (t + 1) % 4 === 0) { el("path", { d: `M${x - 6},${105} v60`, stroke: "#7a5bd0", "stroke-width": 2, "stroke-dasharray": "4 3" }, g); }
  }
  txt(g, 60, 185, "Dashed purple line = a plan boundary: 1 NextLat transition in 4 (t = 4j+6 → 4j+7) crosses it, so ψ must predict the state built on the NEXT plan from h_t and x_{t+1}.", { size: 12, anchor: "start", color: "#5a4a8a" });
  txt(g, 60, 205, "The target is stop-gradded, so that gradient enters the OLD plan u_j: the plan learns to anticipate what the next plan changes.", { size: 12, anchor: "start", color: "#5a4a8a" });
  const y = 250;
  box(g, { x: 60, y, w: 250, h: 40, k: "plan", label: "D₀ = tok + P_t", sub: `plan ${fmtP(P.dec_plan)} · W_u 256→256, W_c 384→256`, size: 11.5 });
  box(g, { x: 350, y, w: 300, h: 40, k: "attn", label: "× 4 confined layers", sub: `${fmtP(P.dec_layer)} each · q from D, k/v from tok, w 8`, size: 11.5 });
  box(g, { x: 690, y, w: 190, h: 40, k: "norm", label: "h = zcRMSNorm(D)", size: 12 });
  box(g, { x: 920, y, w: 220, h: 40, k: "emb", label: "logits = h Eᵀ → L_NTP", size: 12 });
  arrow(g, [[310, y + 20], [350, y + 20]]); arrow(g, [[650, y + 20], [690, y + 20]]); arrow(g, [[880, y + 20], [920, y + 20]]);
  ["Confinement is unchanged from v1: nothing reads D at another position, so reach = 63 + (8 − 1) = 70 tokens at any depth.",
   "Anything older than 70 tokens can reach position t only through its plan, i.e. through the concept module.",
   `v2 keeps the layout and the FFN ratio (704 / 256 = 2.75 = 1056 / 384): ${fmtP(P.dec_total)} params (v1 ${fmtP(A.v1.decoder)}).`]
    .forEach((s, i) => txt(g, 60, 340 + i * 26, s, { size: 12.5, anchor: "start" }));
  return svg;
}

/* =========================================================================================== 9 NextLat */
function f2NextLat(A) {
  const n = A.nextlat, P = A.params;
  const svg = makeSvg(1180, 500), g = el("g", {}, svg);
  box(g, { x: 60, y: 380, w: 150, h: 40, k: "norm", label: "h_t", sub: "decoder final state", size: 12 });
  box(g, { x: 60, y: 60, w: 150, h: 40, k: "norm", label: "h_{t+1}", sub: "next position", size: 12 });
  box(g, { x: 260, y: 380, w: 170, h: 40, k: "emb", label: "rms(sg E x_{t+1})", sub: "the next token (action)", size: 11.5 });
  box(g, { x: 260, y: 300, w: 380, h: 40, k: "gate", label: "LN[h_t ; e_{t+1}]  (512)", size: 12, dash: "5 4" });
  box(g, { x: 260, y: 220, w: 380, h: 50, k: "gate", label: "f_ψ: 512 → 512 → 512 → 256 (GELU, last layer zero-init)", sub: `${fmtI(P.nextlat_psi)} params · training only`, size: 11.5, dash: "5 4" });
  box(g, { x: 260, y: 150, w: 380, h: 40, k: "plan", label: "ĥ_{t+1} = h_t + f_ψ(…)", size: 12 });
  arrow(g, [[135, 380], [135, 360], [330, 360], [330, 341]]); arrow(g, [[345, 380], [345, 341]]); arrow(g, [[450, 300], [450, 271]]); arrow(g, [[450, 220], [450, 191]]);
  arrow(g, [[135, 380], [135, 170], [260, 170]], { c: COL.grey, m: "ahg", dash: "3 3" }); note(g, 142, 280, "residual");
  box(g, { x: 700, y: 110, w: 230, h: 40, k: "loss", label: "SmoothL1(sg h_{t+1}, ĥ_{t+1})", sub: `× λ_h = ${n.lambda_h}`, size: 11.5 });
  box(g, { x: 700, y: 180, w: 230, h: 46, k: "loss", label: "KL(p(·|sg h_{t+1}) ‖ p(·|ĥ_{t+1}))", sub: `head frozen · × λ_KL = ${n.lambda_KL}`, size: 11 });
  arrow(g, [[640, 165], [700, 130]], { c: "#d0667a", m: "ahR" }); arrow(g, [[640, 175], [700, 200]], { c: "#d0667a", m: "ahR" });
  arrow(g, [[210, 80], [960, 80], [960, 130], [931, 130]], { c: COL.grey, m: "ahg", dash: "6 4" }); dim(g, 520, 72, "stop-gradient target");
  const fx = 700, fy = 280;
  txt(g, fx, fy, "Why here (first principles)", { size: 14, weight: 700, anchor: "start" });
  ["Placement = the decoder's pre-logit state, exactly as in the paper.",
   "Theorem: next-token + transition consistency ⇒ h_t is a belief state.",
   "In a confined decoder h_t sees only 70 tokens + its own plan, so the",
   "only way to carry the older history is the plan: NextLat's pressure",
   "lands on the concept stream (gradient h_t → D → P → u, ĉ_pred → core).",
   "d = 1 · ψ 512 · posterior = main logits (reused) · doc-boundary mask.",
   "Measured neutral in v1 toy tests (D-006) → keep/kill rule at the screen."].forEach((s, i) => txt(g, fx, fy + 26 + i * 22, s, { size: 12, anchor: "start", color: i === 6 ? "#a8841a" : COL.ink }));
  return svg;
}

/* =========================================================================================== 10 losses */
function f2Losses(A) {
  const L = A.losses;
  const cols = ["encoder", "core early windows", "core final window", "experts / routers", "decoder", "ψ", "codebook"];
  const M = {
    "NTP": [1, 0, 1, 1, 1, 0, 0], "NCP (per pass)": [1, 1, 1, 1, 0, 0, 0], "consistency (LoopFormer, final window)": [1, 0, 1, 1, 0, 0, 0], "VQ": [0, 0, 0, 0, 0, 0, 1],
    "MoE load balance": [0, 1, 1, 1, 0, 0, 0], "MoE router z-loss": [0, 1, 1, 1, 0, 0, 0], "NextLat next-h": [1, 0, 1, 1, 1, 1, 0], "NextLat KL": [1, 0, 1, 1, 1, 1, 0] };
  const svg = makeSvg(1180, 90 + L.length * 42), g = el("g", {}, svg);
  const x0 = 360, cw = 112;
  txt(g, 30, 40, "loss · weight", { size: 12, weight: 700, anchor: "start" });
  cols.forEach((c, i) => txt(g, x0 + i * cw + cw / 2, 40, c, { size: 11.5, weight: 700 }));
  L.forEach((l, r) => {
    const y = 70 + r * 42;
    box(g, { x: 30, y, w: 310, h: 32, k: "loss", label: `${l.name} · ${l.weight}`, size: 11.5, tip: `<b>${l.name}</b> × ${l.weight}<br>${l.formula}<br>reaches: ${l.reaches.join(", ")}` });
    (M[l.name] || []).forEach((on, i) => el("circle", { cx: x0 + i * cw + cw / 2, cy: y + 16, r: on ? 8 : 3, fill: on ? (i === 5 ? "#7a5bd0" : "#d0667a") : "#dcdcdc" }, g));
  });
  return svg;
}

/* =========================================================================================== 11 budget */
function f2Budget(A) {
  const rows = A.rows, v1 = A.v1;
  const svg = makeSvg(1180, 520), g = el("g", {}, svg);
  txt(g, 30, 30, "Compute parameters by module (convention of v1's “25M”: embedding, Engram table, codebook excluded)", { size: 13, weight: 600, anchor: "start" });
  const list = [["v1 (today)", { encoder: v1.encoder, core: v1.core, decoder: v1.decoder, compute: v1.compute }],
    ["A · D6 as scored (E 8)", rows.A_D6_as_scored_E8.params], ["B · E 12", rows.B_E12.params], ["C · E 16 (chosen)", rows.C_E16.params], ["D · E 16 + decoder 1024", rows.D_E16_dec1024.params]];
  const sx = v => v / 26e6 * 700, x0 = 230;
  list.forEach(([n, p], i) => {
    const y = 60 + i * 44; let x = x0;
    txt(g, x0 - 12, y + 13, n, { size: 12, anchor: "end", weight: i === 3 ? 700 : 400 });
    [["encoder", "#d6e6f7", "#3b78c2"], ["core", "#d2efea", "#2e9c8c"], ["decoder", "#fbdcc8", "#e07b39"]].forEach(([k, f, s]) => {
      const w = sx(p[k]); el("rect", { x, y, width: w, height: 26, fill: f, stroke: s, rx: 3 }, g);
      if (w > 40) txt(g, x + w / 2, y + 13, `${(p[k] / 1e6).toFixed(2)}M`, { size: 10.5 }); x += w;
    });
    txt(g, x + 8, y + 13, `${(p.compute / 1e6).toFixed(2)}M`, { size: 12, anchor: "start", weight: 700 });
  });
  el("line", { x1: x0 + sx(19e6), y1: 52, x2: x0 + sx(19e6), y2: 280, stroke: "#7a5bd0", "stroke-dasharray": "4 3" }, g); txt(g, x0 + sx(19e6), 292, "19M target", { size: 11, color: "#7a5bd0" });
  // R sweep
  const sw = A.r_sweep, Rs = Object.keys(sw).map(Number), gx = 230, gy = 480, gw = 600, gh = 150;
  txt(g, 30, 330, "Core share of forward MACs vs eval R (row C)", { size: 13, weight: 600, anchor: "start" });
  el("line", { x1: gx, y1: gy, x2: gx + gw, y2: gy, stroke: "#bbb" }, g); el("line", { x1: gx, y1: gy, x2: gx, y2: gy - gh, stroke: "#bbb" }, g);
  const px = R => gx + (R / 16) * gw, py = s => gy - s * gh / 0.6;
  el("path", { d: Rs.map((R, i) => `${i ? "L" : "M"}${px(R)},${py(sw[R].share)}`).join(" "), fill: "none", stroke: "#2e9c8c", "stroke-width": 2.2 }, g);
  Rs.forEach(R => { el("circle", { cx: px(R), cy: py(sw[R].share), r: R === 6 ? 6 : 3.5, fill: R === 6 ? "#2e9c8c" : "#fff", stroke: "#2e9c8c", "stroke-width": 1.5 }, g); txt(g, px(R), gy + 14, R, { size: 10.5, color: "#666" }); txt(g, px(R), py(sw[R].share) - 12, `${(sw[R].share * 100).toFixed(0)}%`, { size: 10.5 }); });
  [[0.25, "old cap 25 %"], [0.4, "v2 cap 40 %"]].forEach(([s, t]) => { el("line", { x1: gx, y1: py(s), x2: gx + gw, y2: py(s), stroke: s === 0.4 ? "#7a5bd0" : "#d0667a", "stroke-dasharray": "4 3" }, g); txt(g, gx + gw + 8, py(s), t, { size: 11, anchor: "start", color: s === 0.4 ? "#7a5bd0" : "#d0667a" }); });
  ["R 6 lies inside window 2", "for every K ∈ {8..12}:", "a supervised, carry-", "dependent depth."].forEach((t, i) => txt(g, 945, 360 + i * 18, t, { size: 12, anchor: "start" }));
  txt(g, 945, 440, "The cap moved, not the depth.", { size: 12, anchor: "start", weight: 600 });
  return svg;
}

const FIGS2 = [
  { id: "v2overview", t: "Whole model", fn: f2Overview, cap: "Encoder (left, bottom → top), concept module (middle, top → bottom, looped K times with a shared pool of 16 experts), confined decoder (right, bottom → top) with NextLat on its final hidden state. Dashed purple = training only." },
  { id: "v2enc", t: "Encoder layers (d 256)", fn: f2Encoder, cap: "Same design as v1 at width 256: two layer types, one gated-residual grammar; S layers write both streams, F layers only the global one." },
  { id: "v2engram", t: "Engram memory 80M", fn: f2Engram, cap: "The only change from v1 is the row count (131,071 → 208,333, prime): 80.0M lookups at the same compute." },
  { id: "v2crc", t: "Concepts (d_c 384)", fn: f2Concepts, cap: "Tokens are mixed and pooled 4 → 1 into 384-dim concepts on a sphere; the PQ layer is untouched." },
  { id: "v2core", t: "Concept module D6", fn: f2Core, cap: "Four distinct full-attention layers read the clean input X̃ and the carry S through gated reads; each layer's router picks 2 of 16 shared experts from the state." },
  { id: "v2sched", t: "Pass schedule", fn: f2Schedule, cap: "Training: K ~ U{8..12} passes in windows of 4 (remainder merged), carry detached between windows, every pass graded; the final window adds LoopFormer's consistency and receives NTP and NextLat. Eval: R 6." },
  { id: "v2moe", t: "Looped experts", fn: f2MoE, cap: "Illustration of the mechanism the looped-MoE paper measured: the same position picks different experts on later passes because the router reads the state." },
  { id: "v2dec", t: "Confined decoder & plan", fn: f2Decoder, cap: "Unchanged confinement (reach 70). The plan switches every 4 positions; those switches are where NextLat pushes the plan to anticipate the next one." },
  { id: "v2nextlat", t: "NextLat", fn: f2NextLat, cap: "Training-only latent dynamics on the decoder's final state (Teoh et al.): predict the next hidden state from the current one and the next token." },
  { id: "v2loss", t: "Losses & gradients", fn: f2Losses, cap: "L_total = NTP + 0.1 NCP(every pass) + 0.1 consistency + VQ + 0.01 LB + 0.001 z + 1.0 NextLat-h + 1.0 NextLat-KL. Dots = which parts each term trains." },
  { id: "v2budget", t: "Budget & proportions", fn: f2Budget, cap: "Row C (18.83M) keeps the decoder's ratio and spends the spare parameters on experts (top-2 keeps compute constant). From model_v2/v2_budget.py." },
];
const FIG2_KEY = {
  v2overview: "v2 rebalances the model: the concept module is now the largest part, and NextLat makes the decoder depend on its plan.",
  v2enc: "Same encoder design as v1, one third narrower.", v2engram: "80M numbers of n-gram memory, looked up, not computed.",
  v2crc: "Concepts stay 384-dimensional so the working PQ codebook is unchanged.", v2core: "Four distinct layers per pass, experts chosen by the state.",
  v2sched: "Detached windows force every group of passes to improve on its input.", v2moe: "Later passes, different experts: variation without a pass counter.",
  v2dec: "Each position sees 70 tokens plus its own plan, nothing more.", v2nextlat: "Predict your own next state: a belief-state pressure that lands on the plan.",
  v2loss: "Eight terms, one sum; each trains a known set of modules.", v2budget: "18.83M compute parameters, split 39 / 42 / 19 %.",
};
const FIG2_3D = { v2overview: "all", v2enc: "enc", v2engram: "enc", v2crc: "enc", v2core: "core", v2sched: "core", v2moe: "core", v2dec: "dec", v2nextlat: "dec", v2loss: "all", v2budget: "all" };
