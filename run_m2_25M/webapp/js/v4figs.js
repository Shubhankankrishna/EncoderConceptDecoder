/* v4 draft architecture figures (draft_v4.md r2 · d 256 · 19,898,011 compute params · k = 6 tokens per concept ·
   attention + FFN pooler · F layers applied twice · shared FFN in the encoder · core 4 layers × 2 passes, no table ·
   coarse 1,024 + PQ 32 × 256 × 8 · decoder Engram reads). Same drawing vocabulary as js/v3dfigs.js (svg.js helpers,
   legend3d, V3C). Every size comes from data/arch_v4.json (v4_tiny/code/make_arch_v4.py → model_v4.py). Design only;
   tiny-experiment measurements are labelled as such. */
const SRC4 = {
  spec: "draft_v4.md", enc: "draft_v4.md §3.1–§3.4 · model_v4.py EncoderV4", loop: "draft_v4.md §3.3 (option B)", shared: "draft_v4.md §3.4",
  pool: "draft_v4.md §3.5.4 · model_v4.py PoolAttnFFN", k: "draft_v4.md §3.5.3 (measured on val probes + 3,000 train rows)",
  core: "draft_v4.md §4.1–§4.2 · model_v4.py ConceptCoreV4", code: "draft_v4.md §4.6, §6.2", dec: "draft_v4.md §5.2–§5.3 · model_v4.py DecoderV4",
  obj: "draft_v4.md §6.1", budget: "draft_v4.md §7 · make_arch_v4.py", gates: "draft_v4.md §8.3, §9.2", tiny: "v4_tiny/REPORT.md",
};
const V4C = { acc: "#b5476b", core: "#2e9c8c", sh: "#dc8b56", off: "#b5476b" };

/* =========================================================================================== 1 whole model */
function f4Overview(A) {
  const P = A.params, T = A.table, e = A.ecfg, c = A.core;
  const svg = makeSvg(1220, 925), g = el("g", {}, svg);
  legend3d(g, 30, 20, [{ line: COL.L, t: "L / tok stream" }, { line: COL.G, t: "G stream" }, { line: COL.plan, t: "concept / plan" }, { line: "#d1a92f", t: "Engram rows" },
    { k: "attn", t: "SWA layer" }, { k: "full", t: "full attention" }, { k: "ffn", t: "FFN" }, { k: "mem", t: "memory" }, { k: "loss", t: "loss" }]);

  // ---------------- encoder
  frame(g, { x: 30, y: 60, w: 380, h: 820, title: `Encoder · ${fmtP(T.encoder_non_embedding)}`, note: `+ ${fmtP(T.engram_table)} Engram` });
  const bx = 80, bw = 240;
  box(g, { x: bx, y: 832, w: bw, h: 30, k: "io", label: `Token ids (B, ${e.T})`, tip: `<b>Input</b><br>V = 8,192, T = ${e.T} = 170 × 6; every document padded to a multiple of 6.` });
  box(g, { x: bx, y: 786, w: bw, h: 32, k: "emb", label: `Embedding ${fmtI(e.vocab_size)} × ${e.d_model}`, sub: "tied with the head", size: 12, tip: `<b>Token embedding E</b> · ${fmtI(T.token_embedding)} params (not counted)` });
  arrow(g, [[200, 832], [200, 819]]);
  const rows = [["S", 0], ["E", 1], ["S", 1], ["S", 2], ["F", 3], ["S", 4], ["E", 2], ["S", 5], ["S", 6], ["F", 7]];
  rows.forEach((r, i) => {
    const y = 738 - 47 * i;
    const k = r[0] === "S" ? "attn" : r[0] === "F" ? "full" : "mem";
    const lab = r[0] === "E" ? `Engram read ${r[1]}${r[1] === 1 ? " (conv)" : " (no conv)"}` : `${r[0]}${r[1]} · ${r[0] === "S" ? "SWA w 8 · 4 heads" : "full attn · 6 heads · ×2"}`;
    const sub = r[0] === "E" ? `${fmtP(r[1] === 1 ? P.engram_read1 : P.engram_read2)} · rows gathered once` : `${fmtP(r[0] === "S" ? P.enc_layer_S : P.enc_layer_F)} · + shared FFN branch`;
    box(g, { x: bx, y, w: bw, h: 34, k, label: lab, sub, size: 11.5, src: r[0] === "F" ? SRC4.loop : SRC4.enc,
      tip: r[0] === "F" ? `<b>F layer ${r[1]}, applied twice</b> · ${fmtI(P.enc_layer_F)} params (no new ones)<br>(L, G) ← F(L, G) → G′; then (L, G′) ← F(L, G′). F writes G only, so reach(L) stays 63 (tested). Bypass: enc_loop = 1.`
        : r[0] === "S" ? `<b>S layer ${r[1]}</b> · ${fmtI(P.enc_layer_S)} params<br>reads {L}, writes {L, G}; own SwiGLU 1024 + shared SwiGLU ${e.enc_shared_ff} × g_ℓ in parallel`
        : `<b>Engram read ${r[1]}</b> · ${fmtI(r[1] === 1 ? P.engram_read1 : P.engram_read2)} params<br>${r[1] === 1 ? "after layer 0, with conv" : "after layer 4, no conv, W_v zero-init"}` });
    if (r[0] === "F") { arrow(g, [[bx + bw + 4, y + 26], [bx + bw + 22, y + 26], [bx + bw + 22, y + 8], [bx + bw + 4, y + 8]], { c: V4C.acc, m: "ah", w: 1.6 }); note(g, bx + bw + 26, y + 17, "×2", { size: 12, color: V4C.acc }); }
    if (i) arrow(g, [[200, y + 47], [200, y + 35]], { c: COL.G, m: "ahG", w: 1.2 });
  });
  arrow(g, [[200, 786], [200, 773]], { c: COL.L, m: "ahL" });
  box(g, { x: 362, y: 342, w: 26, h: 396, k: "ffn", label: "", tip: `<b>Encoder shared SwiGLU 256 → ${e.enc_shared_ff} → 256</b> · ${fmtI(P.enc_shared)} params + ${P.enc_shared_gains} gains<br>beside every layer's own FFN, same read and write gate; W₂ zero-init, g_ℓ = 1 → exactly v3's encoder at step 0.`, src: SRC4.shared });
  txt(g, 375, 540, "shared FFN 384", { size: 10.5, color: "#a0602e", weight: 600 }).setAttribute("transform", `rotate(-90 375 540)`);
  box(g, { x: 36, y: 600, w: 22, h: 150, k: "mem", label: "", tip: `<b>One Engram table</b> · 8 × ${fmtI(e.engram_rows)} × ${e.engram_dim} = ${fmtI(T.engram_table)} lookups, gathered once, read four times (2 encoder, 2 decoder)` });
  txt(g, 47, 588, "80M", { size: 10.5, weight: 700, color: "#a8841a" });
  arrow(g, [[58, 691], [bx, 691]], { c: "#d1a92f", m: "ah", w: 1 }); arrow(g, [[47, 600], [47, 456], [bx, 456]], { c: "#d1a92f", m: "ah", w: 1 });
  box(g, { x: bx, y: 254, w: bw, h: 30, k: "gate", label: "router mix {e0, L, G} per token", size: 11.5, tip: `<b>CRC router</b> · ${P.crc_router} params` });
  box(g, { x: bx, y: 196, w: bw, h: 42, k: "conc", label: "Pooler · 6 tokens → 1 concept", sub: `4-head attention + SwiGLU 384 · ${fmtP(P.pooler)}`, size: 11.5,
    tip: `<b>Attention + FFN pooler</b> · ${fmtI(P.pooler)} params<br>attention selects (shared W_K, W_V across slots; pads masked), the FFN composes; init = exact masked mean (tested). ĉ on the sphere √256.`, src: SRC4.pool });
  arrow(g, [[200, 308], [200, 285]], { c: COL.G, m: "ahG" }); arrow(g, [[200, 254], [200, 239]], { c: COL.plan, m: "ahP" });
  box(g, { x: bx, y: 120, w: bw, h: 30, k: "norm", label: "tok = zcRMSNorm(L) · reach 63", size: 12 });
  arrow(g, [[69, 772], [69, 135], [bx, 135]], { c: COL.L, m: "ahL", w: 1.4 });

  // ---------------- concept module
  frame(g, { x: 440, y: 60, w: 380, h: 820, title: `Concept module v4 · ${fmtP(T.concept_module)}`, stroke: "#9fd3ca" });
  const cx = 490, cw = 270;
  box(g, { x: cx, y: 108, w: 125, h: 36, k: "conc", label: "C = ĉ (170)", sub: "1 per 6 tokens", size: 12, tip: "<b>C</b>: 170 concepts per row (T/6), re-read by every sublayer of both passes." });
  box(g, { x: cx + 145, y: 108, w: 125, h: 36, k: "io", label: "S₀ = 0", sub: "working memory", size: 12 });
  arrow(g, [[320, 217], [420, 217], [420, 126], [cx, 126]], { c: COL.plan, m: "ahP", w: 1.8 });
  for (let l = 0; l < 4; l++) {
    const y = 180 + l * 66;
    box(g, { x: cx, y, w: cw, h: 48, k: "full", label: `Layer ${l + 1} · full attention 6×64`, sub: `read {C,S} · own FFN ${c.own_ff} ‖ shared ${c.shared_ff}`, size: 11.5,
      tip: `<b>Core layer ${l + 1}</b> (distinct) · ${fmtI(P.core_layer)} params<br>GR read {C, S} → gated full causal attention over 170 concepts → S += write · attn → GR read → own SwiGLU ${c.own_ff} + shared SwiGLU ${c.shared_ff}. No (t, Δ) table.`, src: SRC4.core });
    if (l) arrow(g, [[cx + cw / 2, y - 18], [cx + cw / 2, y]], { c: COL.plan, m: "ahP" });
    arrow(g, [[cx + cw + 30, y + 24], [cx + cw, y + 24]], { c: V3C.sh, m: "ah", w: 1 });
  }
  arrow(g, [[cx + cw / 2, 144], [cx + cw / 2, 180]], { c: COL.plan, m: "ahP" });
  box(g, { x: cx + cw + 32, y: 180, w: 26, h: 246, k: "ffn", label: "", tip: `<b>Shared SwiGLU 256 → ${c.shared_ff} → 256</b> · ${fmtI(P.core_shared)} params (kept full size: removing it cost v3 multi4 −0.129).`, src: SRC4.core });
  txt(g, cx + cw + 45, 303, "shared FFN", { size: 10.5, color: "#a0602e", weight: 600 }).setAttribute("transform", `rotate(-90 ${cx + cw + 45} 303)`);
  arrow(g, [[cx + 40, 424], [cx + 40, 446], [cx - 30, 446], [cx - 30, 164], [cx + 40, 164], [cx + 40, 178]], { c: COL.plan, m: "ahP", w: 2 });
  txt(g, cx - 36, 150, "× M = 2 passes", { size: 12, weight: 700, color: COL.plan, anchor: "start" });
  note(g, cx + 60, 440, "plain passes · no (t, Δ) table · no shortcut", { size: 10.5 });
  box(g, { x: cx, y: 470, w: cw, h: 30, k: "norm", label: "u = zcRMSNorm(S₂) · terminal only", size: 11.5, tip: "<b>Readout</b>: only S₂ is supervised (pass-1 NCP is an arm with a kill criterion)." });
  arrow(g, [[cx + cw / 2, 446], [cx + cw / 2, 470]], { c: COL.plan, m: "ahP" });
  box(g, { x: cx, y: 528, w: cw, h: 36, k: "conc", label: "Head · q = W_q u", sub: "coarse logits −‖q − E⁰‖²/τ₀ · fine PQ logits", size: 11.5, tip: `<b>Head</b> · ${fmtI(P.ncp_head)} params<br>ĉ_pred = c̄ + PQexp(q − c̄), c̄ = Σ π_k E⁰_k (no future information).`, src: SRC4.code });
  arrow(g, [[cx + cw / 2, 500], [cx + cw / 2, 528]], { c: COL.plan, m: "ahP" });
  box(g, { x: cx, y: 590, w: 128, h: 34, k: "mem", label: `coarse ${A.coarse.K} × 256`, sub: "10 bits · 'abstract action'", size: 10.5, tip: `<b>Coarse codebook E⁰</b> · ${fmtI(P.coarse_codebook)} (not counted)<br>nearest centroid to ĉ; a cluster / action label, not a reconstruction step.` });
  box(g, { x: cx + 142, y: 590, w: 128, h: 34, k: "mem", label: `PQ ${A.pq.S} × ${A.pq.N} × ${A.pq.ds}`, sub: "on the residual", size: 10.5, tip: `<b>Fine PQ codebook</b> · ${fmtI(P.pq_codebook)} (not counted)<br>PQ of ĉ − E⁰_k: 1 bit per dimension.` });
  arrow(g, [[cx + 64, 590], [cx + 64, 565]], { c: "#d1a92f", m: "ah", w: 1 }); arrow(g, [[cx + 206, 590], [cx + 206, 565]], { c: "#d1a92f", m: "ah", w: 1 });
  box(g, { x: cx - 20, y: 660, w: 150, h: 30, k: "loss", label: "0.05 · CE_coarse", size: 11 });
  box(g, { x: cx + 140, y: 660, w: 150, h: 30, k: "loss", label: "0.1 · CE_fine", size: 11, tip: "fine logits in the frame q − E⁰[k_true] (teacher-forced, loss only)" });
  box(g, { x: cx - 20, y: 710, w: 150, h: 30, k: "loss", label: "VQ_coarse", size: 11, tip: "E⁰ only" });
  box(g, { x: cx + 140, y: 710, w: 150, h: 30, k: "loss", label: "VQ_fine", size: 11, tip: "PQ only" });
  arrow(g, [[cx + 20, 564], [cx + 20, 659]], { c: "#d0667a", m: "ahR", w: 1 });
  note(g, cx - 20, 776, "Two passes kept: pass 2 bought v3 multi4 +0.09; pass 3 only +0.01", { size: 10.5, color: "#2e6f66" });
  note(g, cx - 20, 796, "Table removed: +0.008 nats, every probe within noise (v3 24 h)", { size: 10.5, color: "#2e6f66" });

  // ---------------- decoder
  frame(g, { x: 850, y: 60, w: 340, h: 820, title: `Decoder (confined) · ${fmtP(T.decoder)}`, stroke: "#f1c6a6" });
  const dx = 900, dw = 240;
  box(g, { x: dx, y: 760, w: dw, h: 40, k: "plan", label: "Plan: shift 1 chunk, repeat × 6", sub: "P = W_u u + W_c ĉ_pred + o_r", size: 11.5, tip: `<b>Plan builder</b> · ${fmtI(P.dec_plan)} params<br>j*(t) = ⌊(t+1)/6⌋ − 1; null plan for the first 5 tokens of each document.`, src: SRC4.dec });
  arrow(g, [[cx + cw, 485], [836, 485], [836, 780], [dx, 780]], { c: COL.plan, m: "ahP", w: 1.8 });
  dim(g, 840, 500, "u, ĉ_pred");
  for (let l = 0; l < 4; l++) {
    const y = 682 - l * 92;
    box(g, { x: dx, y, w: dw, h: 42, k: "attn", label: `Confined layer ${l + 1}`, sub: "q from D · k, v from tok (w 8)", size: 11.5, tip: `<b>Confined layer ${l + 1}</b> · ${fmtI(P.dec_layer)} params` });
    arrow(g, [[dx + dw / 2, y + 42 + (l ? 50 : 36)], [dx + dw / 2, y + 42]], { c: COL.plan, m: "ahP" });
    arrow(g, [[1170, y + 21], [dx + dw, y + 21]], { c: COL.L, m: "ahL", w: 1 });
    if (l === 0 || l === 2) {
      box(g, { x: dx + 30, y: y - 34, w: dw - 60, h: 26, k: "mem", label: `Engram read (decoder) · ${fmtP(P.dec_read)}`, size: 10.5, dash: "5 4",
        tip: `<b>Decoder Engram read after layer ${l}</b> · ${fmtI(P.dec_read)} params<br>reads the encoder's gathered rows e_t (no second gather), W_V zero-init, position-wise (reach stays 70).<br><b>Tiny-scale finding:</b> a memorisation channel; recommended off in the main run, kept as a mandatory arm.` });
      arrow(g, [[870, y - 21], [dx + 30, y - 21]], { c: "#d1a92f", m: "ah", w: 1 });
    }
  }
  arrow(g, [[47, 750], [47, 896], [870, 896], [870, 477]], { c: "#d1a92f", m: "ah", w: 1, head: false, dash: "4 3" });
  note(g, 470, 909, "same e_t rows → decoder Engram reads (recommended off in the main run; mandatory arm)", { size: 10.5, color: "#a8841a" });
  arrow(g, [[320, 135], [430, 135], [430, 46], [1170, 46], [1170, 703]], { c: COL.L, m: "ahL", w: 1.6, head: false });
  dim(g, 560, 55, "tok (reach 63) → keys and values of every decoder layer");
  box(g, { x: dx, y: 260, w: dw, h: 26, k: "norm", label: "h = zcRMSNorm(D)", size: 12 });
  arrow(g, [[dx + dw / 2, 320], [dx + dw / 2, 287]], { c: COL.plan, m: "ahP" });
  box(g, { x: dx, y: 206, w: dw, h: 30, k: "emb", label: "logits = h Eᵀ (tied)", size: 12 });
  arrow(g, [[dx + dw / 2, 260], [dx + dw / 2, 237]]);
  box(g, { x: dx, y: 150, w: 110, h: 30, k: "loss", label: "NTP(u)", size: 12 });
  arrow(g, [[dx + 55, 206], [dx + 55, 181]], { c: "#d0667a", m: "ahR" });
  return svg;
}

/* =========================================================================================== 2 encoder */
function f4Encoder(A) {
  const P = A.params, e = A.ecfg;
  const svg = makeSvg(1180, 560), g = el("g", {}, svg);
  frame(g, { x: 20, y: 20, w: 540, h: 520, title: "S layer (×6) · own + shared FFN", stroke: "#b9d1ef" });
  frame(g, { x: 620, y: 20, w: 540, h: 520, title: "F layers 3, 7 · run twice", stroke: "#f1c6a6" });
  const x = 70, w = 440;
  box(g, { x, y: 470, w, h: 32, k: "gate", label: "x = GR read of {L}", size: 12 });
  box(g, { x, y: 402, w, h: 44, k: "attn", label: `Gated SWA, w ${e.window}, partial RoPE · 4 heads`, sub: `${fmtI(P.enc_layer_S_attn)} params`, size: 12 });
  box(g, { x, y: 342, w, h: 32, k: "gate", label: "write L and G", size: 12 });
  box(g, { x, y: 282, w, h: 32, k: "gate", label: "x′ = GR read again", size: 12 });
  box(g, { x, y: 196, w: 200, h: 52, k: "ffn", label: `own SwiGLU ${e.d_ff}`, sub: `${fmtI(P.enc_layer_S_ffn)} params`, size: 12 });
  box(g, { x: x + 240, y: 196, w: 200, h: 52, k: "ffn", label: `shared SwiGLU ${e.enc_shared_ff} × g_ℓ`, sub: "one set, all 8 layers", size: 12, fill: "#f5b98f",
    tip: `<b>Encoder shared FFN</b> · ${fmtI(P.enc_shared)} params + 8 gains (g_ℓ = 1)<br>W₂ zero-init → bit-identical to v3's encoder at step 0 (test 11); alive at step 1 (test 5). Tiny scale: switching it off costs +0.014 / +0.024 nats.`, src: SRC4.shared });
  box(g, { x, y: 120, w, h: 34, k: "gate", label: "y = FFN_ℓ(x′) + g_ℓ · FFN_shared(x′) → write L and G", size: 11.5 });
  [[470, 447], [402, 375], [342, 315], [282, 249], [196, 155]].forEach(([a, b]) => arrow(g, [[x + w / 2, a], [x + w / 2, b]], { c: COL.L, m: "ahL" }));
  txt(g, 290, 70, `layer ${fmtI(P.enc_layer_S)} · shared branch ${fmtI(P.enc_shared)} (once)`, { size: 12, weight: 600, color: "#555" });
  // F loop
  const fx = 680, fw = 420;
  box(g, { x: fx, y: 440, w: fw, h: 40, k: "io", label: "(L, G) in", size: 12 });
  box(g, { x: fx, y: 360, w: fw, h: 46, k: "full", label: "F layer · first application", sub: "reads {L, G} · writes G only → G′", size: 12 });
  box(g, { x: fx, y: 270, w: fw, h: 46, k: "full", label: "F layer · second application (same weights)", sub: "reads {L, G′} · writes G only → G″", size: 12, stroke: V4C.acc });
  box(g, { x: fx, y: 190, w: fw, h: 40, k: "io", label: "(L unchanged, G″) out", size: 12 });
  arrow(g, [[fx + fw / 2, 440], [fx + fw / 2, 407]], { c: COL.G, m: "ahG" }); arrow(g, [[fx + fw / 2, 360], [fx + fw / 2, 317]], { c: COL.G, m: "ahG" }); arrow(g, [[fx + fw / 2, 270], [fx + fw / 2, 231]], { c: COL.G, m: "ahG" });
  arrow(g, [[fx + fw + 8, 296], [fx + fw + 30, 296], [fx + fw + 30, 383], [fx + fw + 8, 383]], { c: V4C.acc, m: "ah", w: 1.8, both: false });
  txt(g, fx + fw + 36, 340, "×2", { size: 14, weight: 700, color: V4C.acc, anchor: "start" });
  ["Why only the F layers: S layers write L, so looping any of them grows reach(L) 63 → 123", "and the decoder's reach 70 → 130, past the 128-token probe floor (tested: mutant caught).",
   "F layers write only G, which the decoder never reads: reach stays exactly 63 (test 3).", `No new parameters; ≈ +13–20 % step time (estimate). Bypass enc_loop = 1 = "F loop off".`,
   "Tiny scale: F loop off costs +0.022 nats at k = 4, only +0.002 at k = 6 (single seed)."]
    .forEach((s, i) => txt(g, fx - 30, 70 + i * 18, s, { size: 11, anchor: "start", color: i === 4 ? "#a0405a" : COL.ink }));
  return svg;
}

/* =========================================================================================== 3 engram, four reads */
function f4Engram(A) {
  const P = A.params, e = A.ecfg;
  const svg = makeSvg(1180, 470), g = el("g", {}, svg);
  box(g, { x: 30, y: 200, w: 150, h: 50, k: "io", label: "canonical ids", sub: "x_{t−2}, x_{t−1}, x_t", size: 12 });
  box(g, { x: 210, y: 170, w: 160, h: 110, k: "white", label: "hash 2-, 3-grams", sub: "× 4 heads = 8 row ids", size: 12 });
  arrow(g, [[180, 225], [210, 225]]);
  box(g, { x: 400, y: 150, w: 150, h: 150, k: "mem", label: "one table", sub: `8 × ${fmtI(e.engram_rows)} × ${e.engram_dim}`, size: 12, tip: `<b>${fmtI(P.engram_table)} lookups</b> (not compute)` });
  arrow(g, [[370, 225], [400, 225]]);
  box(g, { x: 580, y: 190, w: 120, h: 70, k: "mem", label: "e_t (384)", sub: "gathered ONCE", size: 12 });
  arrow(g, [[550, 225], [580, 225]], { c: "#d1a92f", m: "ah" });
  const reads = [["encoder read 1", "after layer 0 · conv · → L, G", P.engram_read1, "+0.14 to +0.17 nats (both encoder reads, tiny scale)", 20, "#e7cf7a", false],
                 ["encoder read 2", "after layer 4 · no conv · W_v = 0", P.engram_read2, "", 120, "#e7cf7a", false],
                 ["decoder read A", "after decoder layer 0 · gate reads D · → D", P.dec_read, "proposed off in the main run (memorisation)", 240, "#e9a7b9", true],
                 ["decoder read B", "after decoder layer 2 · gate reads D · → D", P.dec_read, "", 340, "#e9a7b9", true]];
  reads.forEach(([t, s, p, n, y, st, dsh]) => {
    box(g, { x: 760, y, w: 380, h: 80, k: dsh ? "white" : "mem", label: `${t} · ${fmtP(p)}`, sub: s, size: 12, dash: dsh ? "5 4" : null, stroke: st,
      tip: `<b>${t}</b> · ${fmtI(p)} params<br>h = GRRead(stream); k = W_K e; v = W_V e (zero-init); α = σ(⟨rms h, rms k⟩/√256); stream += write(h) · SiLU(α v)` });
    if (n) note(g, 770, y + 74, n, { size: 10.5, color: dsh ? "#a33a50" : "#2e6f66" });
    arrow(g, [[700, 225], [730, 225], [730, y + 40], [760, y + 40]], { c: "#d1a92f", m: "ah", w: 1, dash: dsh ? "4 3" : null });
  });
  txt(g, 30, 340, "One table, four reads (draft §3.2, §5.2)", { size: 13, weight: 700, anchor: "start" });
  ["The n-gram key does not change with depth, so the rows are gathered once and reused.",
   "The decoder reads add a raw trigram-like feature next to the output (the owner's 'spit out common tokens').",
   "Reach safety: e_t depends only on ids t−2 … t, and each read is position-wise (D1 tested at p … p+2).",
   "Tiny-scale result: +0.47 to +0.70 nats on memorised text, −0.14 to −0.17 on unseen text.",
   "→ the decoder reads memorise; the encoder reads on the same table generalise."]
    .forEach((s, i) => txt(g, 30, 366 + i * 19, s, { size: 11.5, anchor: "start", color: i >= 3 ? "#a33a50" : COL.ink }));
  return svg;
}

/* =========================================================================================== 4 pooler */
function f4Pool(A) {
  const P = A.params, vp = A.v3_pool;
  const svg = makeSvg(1180, 640), g = el("g", {}, svg);
  txt(g, 30, 28, "6 tokens → 1 concept: attention selects, a small FFN composes (draft §3.5.4)", { size: 14, weight: 700, anchor: "start" });
  const words = ["The", "code", "of", "kamiro", "is", "lion"], wgt = [0.08, 0.10, 0.07, 0.38, 0.09, 0.28];
  words.forEach((w, i) => {
    const x = 40 + i * 92;
    box(g, { x, y: 470, w: 80, h: 36, k: i === 3 || i === 5 ? "conc" : "white", label: w, size: 12, tip: "illustrative weights: content tokens (names, values) can take most of a head's weight, at any slot" });
    el("rect", { x: x + 10, y: 440 - wgt[i] * 160, width: 60, height: wgt[i] * 160, rx: 4, fill: "rgba(181,71,107,.35)", stroke: V4C.acc }, g);
    txt(g, x + 40, 452, "α " + wgt[i].toFixed(2), { size: 10.5, color: "#7a2f4a" });
    arrow(g, [[x + 40, 470], [x + 40, 446]], { c: COL.grey, m: "ahg", w: 0.8 });
  });
  note(g, 40, 528, "router mix m_i per token, + position vector p_r (zero-init) · pads masked · empty chunk → uniform", { size: 11 });
  box(g, { x: 40, y: 300, w: 548, h: 44, k: "gate", label: "4 heads: s_{h,i} = q_h · W_K(rms m_i + p_r) / 8 + b_{h,r}", sub: `W_K, W_V, W_O 256×256 shared across slots · q, p, b zero-init · ${fmtI(P.pool_Wk + P.pool_Wv + P.pool_Wo + P.pool_q + P.pool_p + P.pool_b)} params`, size: 11.5 });
  box(g, { x: 40, y: 230, w: 548, h: 40, k: "conc", label: "c₀ = W_O [Σ_i α_{h,i} W_V(m_i + p_r)]_h", sub: "init W_V = W_O = I → exact masked mean (test 4)", size: 11.5 });
  box(g, { x: 40, y: 160, w: 548, h: 40, k: "ffn", label: "c₁ = c₀ + g · SwiGLU₃₈₄(rms c₀)", sub: `W₂ zero-init, g = 1 · ${fmtI(P.pool_ffn)} params`, size: 11.5, tip: "Tiny scale: the FFN writes (gain 0.56–0.71) but switching it off is neutral (Δ ≈ 0): recommended as an arm, not default." });
  box(g, { x: 190, y: 92, w: 250, h: 40, k: "conc", label: "ĉ = √256 · c₁ / ‖c₁‖", sub: "one concept (170 per row)", size: 11.5 });
  [[372, 345], [300, 271], [230, 201], [160, 133]].forEach(([a, b]) => arrow(g, [[306, a], [306, b]], { c: COL.plan, m: "ahP" }));
  // v3's trained pool
  const x0 = 650;
  txt(g, x0, 70, `Why not v3's pool: it starts as the average, trains into a per-slot map (step ${fmtI(vp.step)})`, { size: 12, weight: 700, anchor: "start" });
  vp.slot_norm.forEach((v, i) => {
    const h = v / 13 * 120;
    el("rect", { x: x0 + 20 + i * 80, y: 220 - h, width: 50, height: h, rx: 4, fill: i === 3 ? "#f5b98f" : "#dbe8f9", stroke: i === 3 ? "#dc8b56" : "#5b8fd0" }, g);
    txt(g, x0 + 45 + i * 80, 210 - h, v.toFixed(2), { size: 11, weight: 600 });
    txt(g, x0 + 45 + i * 80, 236, `slot ${i + 1}`, { size: 11, color: "#666" });
  });
  el("line", { x1: x0 + 10, y1: 220 - 4 / 13 * 120, x2: x0 + 340, y2: 220 - 4 / 13 * 120, stroke: "#999", "stroke-dasharray": "4 3" }, g);
  note(g, x0 + 345, 220 - 4 / 13 * 120, "init 4.0", { size: 10.5 });
  ["‖W_i‖_F per slot: the newest token counts ≈ 1.6× the oldest,", "and the SAME weights apply to every chunk: slot 2 is weighted", "identically whether it holds 'the' or 'Paris'."]
    .forEach((s, i) => txt(g, x0, 266 + i * 18, s, { size: 11.5, anchor: "start" }));
  // candidates
  const ty = 350, cx = [x0, x0 + 180, x0 + 300];
  ["candidate", "params (k 6)", "verdict"].forEach((h, i) => txt(g, cx[i], ty, h, { size: 11.5, weight: 700, anchor: "start" }));
  [["v3 linear / plain conv", "393,216", "per-slot, content-blind"], ["concat-FFN", "458,752", "no sharing across slots"], ["attention only", "198,424", "selects, cannot compose"],
   ["attention + FFN (default)", fmtI(P.pooler), "selects and composes"], ["gated conv (owner)", "363,033", "selects; ≈ 3–5× slower to learn (toy)"]]
    .forEach((r, j) => r.forEach((c, i) => txt(g, cx[i], ty + 24 + j * 22, c, { size: 11.5, anchor: "start", weight: j === 3 ? 700 : 400 })));
  txt(g, x0, ty + 150, "Tiny scale (d 64, 5M tokens): attention, linear and gated conv tie on language", { size: 11.5, anchor: "start", color: "#a0405a" });
  txt(g, x0, ty + 168, "(paired +0.017 ± 0.018); the linear pool collapses the coarse code (lowest perplexity).", { size: 11.5, anchor: "start", color: "#a0405a" });
  txt(g, x0, ty + 186, "The pooler weights by content, but at that budget it selects structure, not facts.", { size: 11.5, anchor: "start", color: "#a0405a" });
  return svg;
}

/* =========================================================================================== 5 k = 4 / 6 / 8 */
function f4K(A) {
  const svg = makeSvg(1180, 470), g = el("g", {}, svg);
  txt(g, 30, 28, "How many tokens per concept? Measured, not assumed (draft §3.5.3)", { size: 14, weight: 700, anchor: "start" });
  const cx = [30, 470, 660, 850];
  ["", "k = 4 (v3)", "k = 6 (v4 default)", "k = 8 (arm)"].forEach((h, i) => txt(g, cx[i], 62, h, { size: 12.5, weight: 700, anchor: "start", color: i === 2 ? V4C.acc : COL.ink }));
  A.k_compare.forEach((r, j) => {
    const y = 90 + j * 28;
    el("rect", { x: 22, y: y - 13, width: 1136, height: 26, rx: 6, fill: j % 2 ? "#fafbfc" : "#fff", stroke: "#eceff3" }, g);
    r.forEach((c, i) => txt(g, cx[i], y, c, { size: 11.5, anchor: "start", weight: i === 2 ? 650 : 400, color: /75 %|· 0$/.test(c) && i === 3 ? "#a0405a" : COL.ink }));
  });
  ["The recall question is always inside a complete chunk the plan includes, at every k; at k = 8 a quarter of hop items have the hop count outside it.",
   "After 2 passes and no shortcut the core is only ≈ 8–12 % of a step, so k = 8 buys ≈ 3–4 % speed over k = 6: its case is the horizon (T/8) and compression, not speed.",
   "Tiny scale (d 64): k6 − k4 = +0.015 / +0.012 / +0.020 / +0.028 / +0.035 nats at 1–5M tokens, 1.07× faster: the gap reaches the +0.03 gate by 4–5M."]
    .forEach((s, i) => txt(g, 30, 396 + i * 22, s, { size: 11.5, anchor: "start", color: i === 2 ? "#a0405a" : COL.ink }));
  return svg;
}

/* =========================================================================================== 6 core */
function f4Core(A) {
  const P = A.params, c = A.core;
  const svg = makeSvg(1180, 560), g = el("g", {}, svg);
  frame(g, { x: 20, y: 20, w: 620, h: 520, title: "One core layer ℓ (4 distinct layers per pass, 2 passes)", stroke: "#9fd3ca" });
  const x = 70, w = 520;
  box(g, { x: 70, y: 470, w: 160, h: 40, k: "conc", label: "C (170 concepts)", size: 12 });
  box(g, { x: 430, y: 470, w: 160, h: 40, k: "plan", label: "S (memory)", size: 12, tip: "S₀ = 0; pass 2 reads S₁, which already holds what pass 1 retrieved" });
  box(g, { x, y: 396, w, h: 40, k: "gate", label: "x = GRRead{C, S}  (no pass-dependent gains)", size: 12 });
  arrow(g, [[150, 470], [150, 437]], { c: COL.plan, m: "ahP" }); arrow(g, [[510, 470], [510, 437]], { c: COL.plan, m: "ahP" });
  box(g, { x, y: 322, w, h: 46, k: "full", label: "gated full causal attention · 6 × 64, GQA kv 2", sub: `doc mask over 170 concepts · ${fmtI(P.core_attn)} params`, size: 12 });
  arrow(g, [[x + w / 2, 396], [x + w / 2, 369]]);
  box(g, { x, y: 266, w, h: 30, k: "plan", label: "S ← S + write(x) · attn", size: 12 });
  arrow(g, [[x + w / 2, 322], [x + w / 2, 297]]);
  box(g, { x, y: 210, w, h: 30, k: "gate", label: "x′ = GRRead{C, S} again", size: 12 });
  arrow(g, [[x + w / 2, 266], [x + w / 2, 241]]);
  box(g, { x, y: 134, w: 240, h: 48, k: "ffn", label: `own SwiGLU ${c.own_ff}`, sub: `${fmtI(P.core_own_ffn)} params`, size: 12 });
  box(g, { x: x + 280, y: 134, w: 240, h: 48, k: "ffn", label: `shared SwiGLU ${c.shared_ff}`, sub: "same weights in all layers", size: 12, fill: "#f5b98f" });
  arrow(g, [[x + 120, 210], [x + 120, 183]]); arrow(g, [[x + 400, 210], [x + 400, 183]]);
  box(g, { x, y: 66, w, h: 34, k: "plan", label: "S ← S + write(x′) · [own(x′) + shared(x′)]", size: 12 });
  arrow(g, [[x + 120, 134], [x + 120, 101]]); arrow(g, [[x + 400, 134], [x + 400, 101]]);
  const fx = 680;
  txt(g, fx, 50, "Concept module v4", { size: 15, weight: 700, anchor: "start" });
  [["params (compute)", fmtI(A.table.concept_module)], ["per layer", fmtI(P.core_layer)], ["· attention 6 × 64", fmtI(P.core_attn)], ["· own FFN 736", fmtI(P.core_own_ffn)],
   ["· 2 GR reads + writes", fmtI(P.core_gr)], ["shared FFN 1024", fmtI(P.core_shared)], ["head + τ₀ + norm", fmtI(P.ncp_head + P.core_out_norm)], ["(t, Δ) table", "removed (−43,008)"],
   ["passes", "M = 2, plain"], ["supervision", "terminal-only (u = zcRMSNorm(S₂))"], ["shortcut", "removed (was +42 % core compute)"], ["core cost vs v3", "≈ 0.21 attention · 0.31 position-wise"]]
    .forEach(([a, b], i) => { txt(g, fx, 84 + i * 25, a, { size: 12, anchor: "start", color: "#666" }); txt(g, fx + 180, 84 + i * 25, b, { size: 12, anchor: "start", weight: 600 }); });
  txt(g, fx, 400, "Why 4 × 2 (evidence from v3's 24 h switch-offs)", { size: 13, weight: 700, anchor: "start" });
  ["pass 2: multi4 +0.09 · pass 3: +0.01 → keep 2", "shared FFN off: multi4 −0.129 → keep full size", "(t, Δ) table off: +0.008 nats, probes within noise", "L_scons on S₁ would pull S₂ → S₁: dropped",
   "Tiny scale: 1 pass instead of 2 costs +0.008 / +0.014"]
    .forEach((s, i) => txt(g, fx, 424 + i * 21, s, { size: 12, anchor: "start", color: i === 4 ? "#a0405a" : COL.ink }));
  return svg;
}

/* =========================================================================================== 7 two-level codebook */
function f4Code(A) {
  const P = A.params;
  const svg = makeSvg(1180, 480), g = el("g", {}, svg);
  txt(g, 30, 28, "Two levels: a coarse 'abstract action' code and fine PQ codes on the residual (draft §4.6, §6.2)", { size: 14, weight: 700, anchor: "start" });
  circ(g, 90, 140, 30, { f: "#d7f2ed", s: "#3a9d8f", label: "ĉ_j" });
  box(g, { x: 170, y: 110, w: 220, h: 60, k: "mem", label: `coarse: k_j = argmin ‖ĉ − E⁰_k‖`, sub: `K = ${A.coarse.K} × 256 · 10 bits`, size: 11.5, tip: `${fmtI(P.coarse_codebook)} values, VQ-trained (codebook-only gradient), restarted first` });
  box(g, { x: 430, y: 110, w: 220, h: 60, k: "white", label: "residual r_j = ĉ_j − E⁰_{k_j}", size: 11.5 });
  box(g, { x: 690, y: 110, w: 220, h: 60, k: "mem", label: `PQ ${A.pq.S} × ${A.pq.N} × ${A.pq.ds} on r_j`, sub: "1 bit per dimension", size: 11.5, tip: `${fmtI(P.pq_codebook)} values; restarted from residuals of the UPDATED E⁰ (test 10)` });
  arrow(g, [[120, 140], [170, 140]], { c: COL.plan, m: "ahP" }); arrow(g, [[390, 140], [430, 140]]); arrow(g, [[650, 140], [690, 140]]);
  txt(g, 30, 220, "Head and the two frames (teacher-forced in the loss, never in the plan)", { size: 13, weight: 700, anchor: "start" });
  box(g, { x: 30, y: 240, w: 360, h: 50, k: "conc", label: "coarse logits = −‖q − E⁰_k‖² / τ₀", sub: "CE_coarse × 0.05 on the next concept's k_{j+1}", size: 11.5 });
  box(g, { x: 410, y: 240, w: 360, h: 50, k: "loss", label: "fine logits in the frame q − E⁰[k_{j+1}]", sub: "CE_fine × 0.1 · the TRUE next coarse code, loss only", size: 11.5 });
  box(g, { x: 790, y: 240, w: 360, h: 50, k: "plan", label: "plan: ĉ_pred = c̄ + PQexp(q − c̄)", sub: "c̄ = Σ_k π_k E⁰_k · no future information", size: 11.5 });
  ["Why a coarse level: a policy choosing coarse codes makes one 10-bit decision per 6 tokens: K^{T/6} = 2^{1.7T} sequences vs V^T = 2^{13T} for tokens.",
   "Reconstruction comes from the fine level: VE bound ≈ 0.75 at 1 bit/dim; the coarse level adds ≈ 5 % at full scale (1,024 centroids in 256 dims).",
   "Tests: NCP never trains a codebook, VQ trains only codebooks (test 6); the fine CE sees E⁰ only through the true next code (test 7); restart order (test 10).",
   "Tiny scale: fine-level VE on the residual 0.715 / 0.726 (k4 / k6, bound 0.75); coarse perplexity 36–62 % of K and coarse CE 1.07–1.53 vs ln K 4.16 across the five arms."]
    .forEach((s, i) => txt(g, 30, 340 + i * 26, s, { size: 11.5, anchor: "start", color: i === 3 ? "#a0405a" : COL.ink }));
  return svg;
}

/* =========================================================================================== 8 decoder plan */
function f4Decoder(A) {
  const P = A.params;
  const svg = makeSvg(1180, 470), g = el("g", {}, svg);
  txt(g, 30, 28, "Shift one chunk, repeat over 6 tokens: j*(t) = ⌊(t+1)/6⌋ − 1 (draft §5.3)", { size: 13.5, weight: 700, anchor: "start" });
  const x0 = 40, cw = 46;
  for (let t = 0; t < 24; t++) {
    const j = Math.floor((t + 1) / 6) - 1, x = x0 + t * cw;
    box(g, { x, y: 60, w: cw - 5, h: 30, k: Math.floor(t / 6) % 2 ? "attn" : "white", label: `x${t}`, size: 10.5, rx: 5 });
    box(g, { x, y: 120, w: cw - 5, h: 30, k: j < 0 ? "io" : "plan", label: j < 0 ? "null" : `P${j}`, size: 10.5, rx: 5, tip: `position ${t} predicts x${t + 1} from ${j < 0 ? "the null plan" : "plan " + j + " (u_" + j + ", ĉ_pred," + j + " = predicted concept of chunk " + (j + 1) + ")"}` });
  }
  note(g, 40, 168, "The plan made at chunk j (u_j and the predicted concept of chunk j + 1) is added at all 6 positions that write chunk j + 1, with an offset o_r for the position inside the chunk.", { size: 11.5, color: "#2e6f66" });
  note(g, 40, 188, "Null plan for the first 5 tokens of each document (was 3 at k = 4). Tested exactly at k = 4/6/8: the plan of chunk j changes only positions 6j+5 … 6j+10 (D2).", { size: 11.5, color: "#2e6f66" });
  const y = 230;
  box(g, { x: 40, y, w: 240, h: 40, k: "plan", label: "D₀ = tok + P_t", sub: `plan ${fmtP(P.dec_plan)} · offsets 6 × 256`, size: 11.5 });
  box(g, { x: 310, y, w: 300, h: 40, k: "attn", label: "× 4 confined layers", sub: `${fmtP(P.dec_layer)} each · q from D, k/v from tok, w 8`, size: 11.5 });
  box(g, { x: 640, y, w: 220, h: 40, k: "mem", label: "+ Engram reads after 0 and 2", sub: "proposed off in the main run", size: 11, dash: "5 4" });
  box(g, { x: 890, y, w: 250, h: 40, k: "emb", label: "logits = zcRMSNorm(D) Eᵀ → NTP", size: 11.5 });
  arrow(g, [[280, y + 20], [310, y + 20]]); arrow(g, [[610, y + 20], [640, y + 20]]); arrow(g, [[860, y + 20], [890, y + 20]]);
  ["Confinement unchanged: nothing reads D at another position, so reach = 63 + 7 = 70 at any depth, with or without the Engram reads (D1 tested).",
   "Anything older than 70 tokens reaches position t only through its plan, i.e. through the concept module (test 1: perturbations travel past 70 only via the core).",
   "With both read W_V = 0 the decoder is bit-identical to v3's decoder with a 6-chunk plan (test 15).",
   `Decoder ${fmtP(A.table.decoder)} with the reads; ${fmtP(A.table.decoder - P.dec_reads)} without them (the recommended main-run default).`]
    .forEach((s, i) => txt(g, 40, 312 + i * 24, s, { size: 12, anchor: "start" }));
  return svg;
}

/* =========================================================================================== 9 losses */
function f4Losses(A) {
  const L = A.losses;
  const cols = ["decoder", "core", "encoder L (tok)", "encoder G / F / pooler (C)", "coarse E⁰", "PQ"];
  const M = { "NTP(u)": [1, 1, 1, 1, 0, 0], CE_coarse: [0, 1, 0, 1, 0, 0], CE_fine: [0, 1, 0, 1, 0, 0], VQ_coarse: [0, 0, 0, 0, 1, 0], VQ_fine: [0, 0, 0, 0, 0, 1] };
  const svg = makeSvg(1180, 420), g = el("g", {}, svg);
  const x0 = 360, cw = 130;
  txt(g, 30, 40, "loss · weight", { size: 12, weight: 700, anchor: "start" });
  cols.forEach((c, i) => txt(g, x0 + i * cw + cw / 2, 40, c, { size: 11.5, weight: 700 }));
  L.forEach((l, r) => {
    const y = 70 + r * 44;
    box(g, { x: 30, y, w: 310, h: 32, k: "loss", label: `${l.name} · ${l.weight}`, size: 11.5, tip: `<b>${l.name}</b> × ${l.weight}<br>${l.formula}<br>trains: ${l.reaches.join(", ")}`, src: SRC4.obj });
    (M[l.name] || []).forEach((on, i) => el("circle", { cx: x0 + i * cw + cw / 2, cy: y + 16, r: on ? 8 : 3, fill: on ? "#d0667a" : "#dcdcdc" }, g));
  });
  txt(g, 30, 312, "L = NTP(u) + 0.05 CE_coarse + 0.1 CE_fine + VQ_coarse + VQ_fine     (u = zcRMSNorm(S₂); terminal-only)", { size: 13, weight: 700, anchor: "start" });
  ["Removed from v3: NCP(u_s) and L_scons (no shortcut trajectory). VQ_coarse ≈ 7 dominates the printed loss, so the dashboard headlines NTP.",
   "The encoder's global half (F layers, loop, pooler) is trained only through C: NTP via the plan, and both NCP terms (gradient-reach map, draft §6.4).",
   "Tested: NCP gives zero gradient to both codebooks; VQ gives gradient only to them; VQ_coarse never touches PQ (test 6)."]
    .forEach((s, i) => txt(g, 30, 340 + i * 22, s, { size: 12, anchor: "start" }));
  return svg;
}

/* =========================================================================================== 10 budget */
function f4Budget(A) {
  const T = A.table, P = A.params, v1 = A.v1, v3 = A.v3;
  const svg = makeSvg(1180, 560), g = el("g", {}, svg);
  txt(g, 30, 30, "Compute parameters by module (embedding, Engram table and codebooks excluded)", { size: 13, weight: 600, anchor: "start" });
  const rec = A.variants.recommended_main_run;
  const list = [["v1 (d 384)", v1], ["v3 (d 256, running)", v3], ["v4 draft (d 256)", { encoder: T.encoder_non_embedding, core: T.concept_module, decoder: T.decoder, compute: T.compute_params }],
    ["v4 recommended main run", { encoder: T.encoder_non_embedding - P.pool_ffn, core: T.concept_module, decoder: T.decoder - P.dec_reads, compute: rec }]];
  const sx = v => v / 26e6 * 700, x0 = 230;
  list.forEach(([n, p], i) => {
    const y = 60 + i * 46; let x = x0;
    txt(g, x0 - 12, y + 13, n, { size: 12, anchor: "end", weight: i >= 2 ? 700 : 400, color: i === 3 ? V4C.acc : COL.ink });
    [["encoder", "#d6e6f7", "#3b78c2"], ["core", "#d2efea", "#2e9c8c"], ["decoder", "#fbdcc8", "#e07b39"]].forEach(([k, f, s]) => {
      const w = sx(p[k]); el("rect", { x, y, width: w, height: 26, fill: f, stroke: s, rx: 3 }, g);
      if (w > 40) txt(g, x + w / 2, y + 13, `${(p[k] / 1e6).toFixed(2)}M · ${(100 * p[k] / p.compute).toFixed(0)}%`, { size: 10.5 }); x += w;
    });
    txt(g, x + 8, y + 13, `${(p.compute / 1e6).toFixed(2)}M`, { size: 12, anchor: "start", weight: 700 });
  });
  el("line", { x1: x0 + sx(20e6), y1: 52, x2: x0 + sx(20e6), y2: 246, stroke: V4C.acc, "stroke-dasharray": "4 3" }, g); txt(g, x0 + sx(20e6), 258, "20M cap (owner)", { size: 11, color: V4C.acc });
  txt(g, 30, 296, "Where v4's parameters went relative to v3 (exact, model_v4.py)", { size: 13, weight: 600, anchor: "start" });
  [["encoder shared FFN 384 + 8 gains", "+294,920"], ["attention + FFN pooler replaces the linear pool", "+231,193"], ["decoder Engram reads (2)", "+427,522"],
   ["plan offsets 6 × 256", "+512"], ["(t, Δ) table removed", "−43,008"], ["coarse temperature τ₀", "+1"], ["total vs v3 18,986,871", `+${fmtI(T.compute_params - v3.compute)} = ${fmtI(T.compute_params)}`]]
    .forEach(([a, b], i) => { txt(g, 30, 322 + i * 22, a, { size: 12, anchor: "start" }); txt(g, 460, 322 + i * 22, b, { size: 12, anchor: "start", weight: 650 }); });
  const tx = 620;
  txt(g, tx, 296, "Step time vs v3's profile (estimate; interleaved A/B decides)", { size: 13, weight: 600, anchor: "start" });
  const cx = [tx, tx + 90, tx + 230, tx + 330];
  A.throughput.forEach((r, j) => r.forEach((c, i) => txt(g, cx[i], 322 + j * 22, c, { size: 11, anchor: "start", weight: j === 0 || j === A.throughput.length - 1 ? 700 : 400 })));
  txt(g, tx, 470, "Tiny-scale CPU ratio (k6 / k4) 1.07× does not transfer to the M2 GPU.", { size: 11.5, anchor: "start", color: "#a0405a" });
  return svg;
}

/* =========================================================================================== 11 gates & arms */
function f4Gates(A) {
  const svg = makeSvg(1180, 640), g = el("g", {}, svg);
  txt(g, 30, 28, "How v4 will be judged: early gates against v3 at matched tokens, and controlled arms", { size: 14, weight: 700, anchor: "start" });
  const cx = [30, 260, 420, 860];
  ["gate", "at v3's step", "criterion", "reference"].forEach((h, i) => txt(g, cx[i], 60, h, { size: 12, weight: 700, anchor: "start" }));
  A.gates.forEach((r, j) => {
    const y = 86 + j * 30;
    el("rect", { x: 22, y: y - 13, width: 1136, height: 26, rx: 6, fill: j === 2 || j === 5 ? "#f8e9ef" : j % 2 ? "#fafbfc" : "#fff", stroke: "#eceff3" }, g);
    r.forEach((c, i) => txt(g, cx[i], y, c, { size: 11, anchor: "start", weight: i === 0 ? 700 : 400 }));
  });
  const ay = 300;
  txt(g, 30, ay, "Arms (each ≈ 10 h to step 4,050, after the main run; matched tokens)", { size: 13, weight: 700, anchor: "start" });
  const ax = [30, 380, 700, 900];
  ["arm", "change", "priority", "question"].forEach((h, i) => txt(g, ax[i], ay + 26, h, { size: 11.5, weight: 700, anchor: "start" }));
  A.arms.forEach((r, j) => r.forEach((c, i) => txt(g, ax[i], ay + 50 + j * 24, c, { size: 11.5, anchor: "start", weight: i === 0 || /mandatory/.test(c) ? 700 : 400, color: /mandatory/.test(c) ? V4C.acc : COL.ink })));
  txt(g, 30, 600, "No gate stops the run (v3 policy): gates inform the owner. G3 (recall onset, ≈ 5 h after launch) is the first reading of whether v4's concept path retrieves.", { size: 12, anchor: "start", color: "#555" });
  return svg;
}

const FIGS4 = [
  { id: "v4whole", t: "Whole model", fn: f4Overview, cap: "Encoder (left) with the F layers applied twice, a shared FFN beside every layer and one 80M Engram table; an attention + FFN pooler makes one concept per 6 tokens. Concept module (middle): 4 distinct layers × 2 plain passes, terminal-only, with a coarse + fine codebook head. Confined decoder (right) reading the shifted plan and, as proposed, two direct Engram reads." },
  { id: "v4enc", t: "Encoder: loop & shared FFN", fn: f4Encoder, cap: "v3's encoder kept. Two additions: a shared SwiGLU 384 beside every layer's own FFN (zero-init, so step 0 equals v3), and a second in-place application of the two F layers, which write only G, so the decoder's reach stays 63." },
  { id: "v4engram", t: "Engram: one table, four reads", fn: f4Engram, cap: "The rows are gathered once and read by two encoder reads (as v3) and two new decoder reads. At tiny scale the decoder reads acted as a memorisation channel, so the recommendation is to keep them off in the main run and test them as an arm." },
  { id: "v4pool", t: "Pooler: 6 tokens → 1 concept", fn: f4Pool, cap: "Four attention heads with weights shared across slots choose which tokens a concept keeps; a small FFN composes the selections. v3's pool, by contrast, trained into a fixed per-slot map. Candidate poolers priced at full size; the owner's gated convolution is the cheapest that can select." },
  { id: "v4k", t: "4, 6 or 8 tokens per concept", fn: f4K, cap: "Measured on the validation probe items and 3,000 training rows: the probe question is always inside the plan; padding and null-plan share grow with k; k = 8 costs hop validity for little extra speed. k = 6 default, k = 4 control, k = 8 arm." },
  { id: "v4core", t: "Concept layer & two passes", fn: f4Core, cap: "Four distinct 6-head layers reading C and S, own FFN 736 plus the shared FFN 1024, applied twice from S₀ = 0. The (t, Δ) table and the shortcut trajectory are removed on v3's 24 h evidence; supervision is terminal-only." },
  { id: "v4code", t: "Two-level codebook", fn: f4Code, cap: "A 1,024-way coarse code (the abstract action, 10 bits) and PQ codes on the residual (1 bit per dimension). The fine loss uses the true next coarse code in its frame; the plan uses the soft expectation, so nothing from the future reaches the decoder." },
  { id: "v4dec", t: "Decoder & shifted plan", fn: f4Decoder, cap: "NCP's mechanism with k = 6: the plan made at chunk j is shifted one chunk and repeated over the 6 positions that write chunk j + 1. The decoder still sees only 70 tokens plus that plan." },
  { id: "v4loss", t: "Losses & gradient reach", fn: f4Losses, cap: "NTP, coarse and fine next-concept cross-entropies, and two codebook-only VQ terms. Dots show which parts each term trains." },
  { id: "v4budget", t: "Budget & throughput", fn: f4Budget, cap: "19,898,011 compute parameters (19,175,576 in the recommended main run without the decoder reads and pooler FFN), under the 20M cap. Throughput is an estimate from v3's profile; a measured A/B decides before launch." },
  { id: "v4gates", t: "Gates & control arms", fn: f4Gates, cap: "Early gates against v3 at matched tokens (G1–G5) and the k = 6 gate against the mandatory k = 4 control. Recall onset at step 1,792 is the first full-scale test of the one claim the tiny experiment could not settle." },
];
const FIG4_KEY = {
  v4whole: "Three parts, 19.90M compute: a refined v3 encoder with a content-aware pooler, a 4 × 2 concept module, a confined decoder.",
  v4enc: "Global layers twice, a shared FFN everywhere; the decoder's reach is untouched.", v4engram: "One 80M table, four reads; the decoder pair memorised at tiny scale.",
  v4pool: "Attention chooses by content; v3's pool chose by position.", v4k: "Six tokens per concept, with the measured costs.",
  v4core: "Two plain passes, no table, terminal-only.", v4code: "A small action code and fine detail on the residual.",
  v4dec: "Shift one chunk, repeat six times.", v4loss: "Five terms; codebooks trained only by VQ.", v4budget: "Under 20M; faster core pays for the encoder loop.",
  v4gates: "Recall onset is the first real test; k = 4 is the control.",
};
const FIG4_3D = { v4whole: "all", v4enc: "enc", v4engram: "enc", v4pool: "enc", v4k: "enc", v4core: "core", v4code: "core", v4dec: "dec", v4loss: "all", v4budget: "all", v4gates: "all" };
