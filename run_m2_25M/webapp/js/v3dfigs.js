/* v3 draft architecture figures (draft_v3.md · d 256 · 18,986,871 compute params · Engram 80M read twice ·
   4-layer concept core with LoopFormer (t, Δt) table, shared FFN, M ≤ 3 · confined decoder, no NextLat).
   Same drawing vocabulary as js/v2figs.js. Every size comes from data/arch_v3d.json (v3_design/make_arch_v3draft.py →
   v3_ref.py, which instantiates the real v1 classes). Design only: nothing here is a v3 measurement. */
const SRC3D = {
  spec: "draft_v3.md", ref: "v3_design/v3_ref.py", enc: "draft_v3.md §3 · hrm_encoder.py at d 256 (F heads 6, ff 1024)",
  engram: "draft_v3.md §3.2 · check_engram2.py (EngramRead2)", crc: "draft_v3.md §3.3 · §4.6",
  core: "draft_v3.md §4.1 · v3_ref.py CoreLayerV3", traj: "draft_v3.md §4.2 · v3_ref.py SCHED", table: "draft_v3.md §4.3 · v3_ref.py PAIRS",
  shared: "draft_v3.md §4.4", dec: "draft_v3.md §5 · hrm_decoder.py (confined, ff 1024)", obj: "draft_v3.md §6", budget: "draft_v3.md §7",
  gates: "draft_v3.md §9",
};
const V3C = { core: "#2e9c8c", sh: "#dc8b56", tab: "#7a5bd0", sc: "#a07cc5", det: "#d0667a" };
const THIRD = x => ["0", "⅓", "⅔", "1"][x];

function legend3d(g, x, y, items) {
  let cx = x;
  for (const it of items) {
    if (it.line) { el("line", { x1: cx, y1: y, x2: cx + 22, y2: y, stroke: it.line, "stroke-width": 2.2, "stroke-dasharray": it.dash || null }, g); cx += 28; }
    else { el("rect", { x: cx, y: y - 7, width: 16, height: 14, rx: 3, fill: (PAL[it.k] || PAL.white).f, stroke: (PAL[it.k] || PAL.white).s, "stroke-dasharray": it.dash || null }, g); cx += 21; }
    txt(g, cx, y, it.t, { size: 11.5, anchor: "start", color: "#444" });
    cx += Math.max(40, it.t.length * 6.3) + 16;
  }
}

/* =========================================================================================== 1 whole model */
function f3dOverview(A) {
  const P = A.params, T = A.table, e = A.ecfg, c = A.core;
  const svg = makeSvg(1220, 900), g = el("g", {}, svg);
  legend3d(g, 30, 20, [{ line: COL.L, t: "L / tok stream" }, { line: COL.G, t: "G stream" }, { line: COL.plan, t: "concept / plan" }, { line: V3C.sc, t: "shortcut (training only)", dash: "5 4" },
    { k: "attn", t: "SWA layer" }, { k: "full", t: "full attention" }, { k: "ffn", t: "FFN" }, { k: "mem", t: "memory" }, { k: "loss", t: "loss" }]);

  // ---------------- encoder
  frame(g, { x: 30, y: 60, w: 360, h: 800, title: `Encoder · ${fmtP(T.encoder_non_embedding)}`, note: `+ ${fmtP(T.engram_table)} Engram` });
  const bx = 90, bw = 240;
  box(g, { x: bx, y: 812, w: bw, h: 30, k: "io", label: "Token ids (B, 1024)", tip: "<b>Input</b><br>V = 8,192, T = 1,024, packed with document ids; every document padded to a multiple of 4." });
  box(g, { x: bx, y: 766, w: bw, h: 32, k: "emb", label: `Embedding ${fmtI(e.vocab_size)} × ${e.d_model}`, sub: "tied with the head", size: 12, tip: `<b>Token embedding E</b> · ${fmtI(T.token_embedding)} params (not counted)<br>e0 = rms(E[x]); logits = h Eᵀ.`, src: SRC3D.enc });
  arrow(g, [[210, 812], [210, 799]]);
  const rows = [["S", 0], ["E", 1], ["S", 1], ["S", 2], ["F", 3], ["S", 4], ["E", 2], ["S", 5], ["S", 6], ["F", 7]];
  rows.forEach((r, i) => {
    const y = 718 - 45 * i;
    const k = r[0] === "S" ? "attn" : r[0] === "F" ? "full" : "mem";
    const lab = r[0] === "E" ? `Engram read ${r[1]}${r[1] === 1 ? " (conv)" : " (no conv, W_v = 0)"}` : `${r[0]}${r[1]} · ${r[0] === "S" ? "SWA w 8 · 4 heads" : "full attention · 6 heads"}`;
    const sub = r[0] === "E" ? `${fmtP(r[1] === 1 ? P.engram_read1 : P.engram_read2)} · shared 80M table` : `${fmtP(r[0] === "S" ? P.enc_layer_S : P.enc_layer_F)} · reads ${r[0] === "S" ? "{L}" : "{L,G}"} · writes ${r[0] === "S" ? "{L,G}" : "{G}"}`;
    box(g, { x: bx, y, w: bw, h: 33, k, label: lab, sub, size: 11.5, src: r[0] === "E" ? SRC3D.engram : SRC3D.enc,
      tip: r[0] === "E" ? `<b>Engram read ${r[1]}</b> · ${fmtI(r[1] === 1 ? P.engram_read1 : P.engram_read2)} params<br>${r[1] === 1 ? "after layer 0; gate sees 10 tokens of L; silu(conv(α v)) → L, G" : "after layer 4; same rows, own W_k, W_v (zero-init), gate reads L only (43 tokens); no conv keeps L-reach 63"}`
        : `<b>${r[0]} layer ${r[1]}</b> · ${fmtI(r[0] === "S" ? P.enc_layer_S : P.enc_layer_F)} params<br>d 256, ${r[0] === "S" ? "4" : "6"} heads × 64, GQA kv 2, SwiGLU 256 → ${e.d_ff} → 256`});
    if (i) arrow(g, [[210, y + 45], [210, y + 34]], { c: COL.G, m: "ahG", w: 1.2 });
  });
  arrow(g, [[210, 766], [210, 752]], { c: COL.L, m: "ahL" });
  box(g, { x: 350, y: 640, w: 30, h: 128, k: "mem", label: "", tip: `<b>One Engram table</b> · 8 × ${fmtI(e.engram_rows)} × ${e.engram_dim} = ${fmtI(T.engram_table)} lookups, gathered once, read twice` });
  txt(g, 365, 628, "80M", { size: 10.5, weight: 700, color: "#a8841a" });
  arrow(g, [[350, 690], [331, 690]], { c: "#d1a92f", m: "ah", w: 1 }); arrow(g, [[365, 640], [365, 460], [331, 460]], { c: "#d1a92f", m: "ah", w: 1 });
  box(g, { x: bx, y: 250, w: bw, h: 30, k: "gate", label: "CRC depth-mix {e0, L, G}", size: 12, tip: `<b>CRC router</b> · ${P.crc_router} params · init 0.107 / 0.107 / 0.787<br>E5 (report only): a low G share does not mean the global path is bypassed; the core's full attention can still retrieve local concepts. The F/G branch is tested by <code>F_off</code>.` });
  box(g, { x: bx, y: 200, w: bw, h: 34, k: "conc", label: "Pool 4 × 256 → 256 · sphere √256 = 16", size: 11.5, sub: `C = ĉ (B, 256, 256) · ${fmtP(P.pool)}`, tip: `<b>Chunk pooling</b> · ${fmtI(P.pool)} params<br>4:1 compression (v1's); at d_c = d the init is plain mean pooling.`, src: SRC3D.crc });
  arrow(g, [[210, 313], [210, 281]], { c: COL.G, m: "ahG" }); arrow(g, [[210, 250], [210, 235]], { c: COL.plan, m: "ahP" });
  box(g, { x: bx, y: 120, w: bw, h: 30, k: "norm", label: "tok = zcRMSNorm(L) · reach 63", size: 12, tip: "<b>tok</b>: the decoder's only view of the tokens (keys and values). F layers never write L." });
  arrow(g, [[60, 740], [60, 135], [bx, 135]], { c: COL.L, m: "ahL", w: 1.6 });
  note(g, 64, 760, "L", { color: COL.L, size: 12 });

  // ---------------- concept module
  frame(g, { x: 430, y: 60, w: 380, h: 800, title: `Concept module v3 · ${fmtP(T.concept_module)}`, stroke: "#9fd3ca" });
  const cx = 480, cw = 270;
  box(g, { x: cx, y: 108, w: 125, h: 36, k: "conc", label: "C = ĉ", sub: "evidence · re-read", size: 12, tip: "<b>C</b>: fixed for the input, re-read by every sublayer of every pass, differentiable back to the encoder.", src: SRC3D.core });
  box(g, { x: cx + 145, y: 108, w: 125, h: 36, k: "io", label: "S₀ = 0", sub: "working memory", size: 12, tip: "<b>S</b>: the scratchpad. S₀ = 0 means nothing has been written yet; the first read still sees C." });
  arrow(g, [[330, 217], [405, 217], [405, 126], [cx, 126]], { c: COL.plan, m: "ahP", w: 1.8 });
  dim(g, 336, 208, "C");
  for (let l = 0; l < 4; l++) {
    const y = 180 + l * 66;
    box(g, { x: cx, y, w: cw, h: 48, k: "full", label: `Layer ${l + 1} · full attention 6×64`, sub: `read {C,S} · own FFN ${c.own_ff} ‖ shared ${c.shared_ff}`, size: 11.5,
      tip: `<b>Core layer ${l + 1}</b> (distinct weights) · ${fmtI(P.core_layer)} params<br>GR read {C, S} × (1+γ_C), (1+γ_S) → gated full causal attention over 256 concepts (NoPE, τ_h) → S += s(1+α)·attn → GR read {C, S} → [own SwiGLU ${c.own_ff} ‖ shared SwiGLU ${c.shared_ff}] → S += s·[(1+α_own) own + (1+α_sh) shared]`, src: SRC3D.core });
    if (l) arrow(g, [[cx + cw / 2, y - 18], [cx + cw / 2, y]], { c: COL.plan, m: "ahP" });
    arrow(g, [[cx + cw + 30, y + 24], [cx + cw, y + 24]], { c: V3C.sh, m: "ah", w: 1 });
    arrow(g, [[cx - 22, y + 14], [cx, y + 14]], { c: V3C.tab, m: "ah", w: 1 });
  }
  arrow(g, [[cx + cw / 2, 144], [cx + cw / 2, 180]], { c: COL.plan, m: "ahP" });
  box(g, { x: cx + cw + 32, y: 180, w: 26, h: 246, k: "ffn", label: "", tip: `<b>Shared SwiGLU 256 → ${c.shared_ff} → 256</b> · ${fmtI(P.core_shared)} params<br>one set of weights used by all 4 layers in every pass (Mobius-style store); output projection zero-init.`, src: SRC3D.shared });
  txt(g, cx + cw + 45, 303, "shared FFN", { size: 10.5, color: "#a0602e", weight: 600 }).setAttribute("transform", `rotate(-90 ${cx + cw + 45} 303)`);
  box(g, { x: cx - 50, y: 180, w: 26, h: 246, k: "gate", label: "", tip: `<b>(t, Δ) table</b> · ${fmtI(P.core_table)} params · zero-init, no weight decay<br>6 pairs × 4 layers × 7 vectors × 256: per-stream read gains γ_C, γ_S and write gains α. Φ(S; C, t, Δ).`, src: SRC3D.table });
  txt(g, cx - 37, 303, "(t, Δ) table", { size: 10.5, color: V3C.tab, weight: 600 }).setAttribute("transform", `rotate(-90 ${cx - 37} 303)`);
  // loop arrow
  arrow(g, [[cx + 40, 424], [cx + 40, 446], [cx - 64, 446], [cx - 64, 164], [cx + 40, 164], [cx + 40, 178]], { c: COL.plan, m: "ahP", w: 2 });
  txt(g, cx - 70, 150, "× M = 3 passes", { size: 12, weight: 700, color: COL.plan, anchor: "start" });
  note(g, cx + 60, 440, "t = 0 → ⅓ → ⅔ · Δ = ⅓ each", { size: 10.5 });
  box(g, { x: cx, y: 470, w: cw, h: 30, k: "norm", label: "u₃ = zcRMSNorm(S₃) · raw S carried", size: 11.5, tip: "<b>Readout</b>: S is carried raw between passes (no carry norm); only the readout is normalised. One out_norm shared by both trajectories." });
  arrow(g, [[cx + cw / 2, 446], [cx + cw / 2, 470]], { c: COL.plan, m: "ahP" });
  box(g, { x: cx, y: 530, w: cw, h: 34, k: "conc", label: "NCP head · endpoints only", sub: "W_q 256 → 256 · distance to codes", size: 12, tip: `<b>NCP head</b> · ${fmtI(P.ncp_head)} params<br>predicts the next chunk's 32 PQ codes from u₃ (and from the shortcut endpoint u_s); c_pred = Σ π sg(e).` });
  arrow(g, [[cx + cw / 2, 500], [cx + cw / 2, 530]], { c: COL.plan, m: "ahP" });
  box(g, { x: cx, y: 590, w: cw, h: 30, k: "mem", label: `PQ codebook ${A.pq.S} × ${A.pq.N} × ${A.pq.ds} (${A.pq.bits} bits)`, size: 11.5, tip: `<b>PQ codebook</b> · ${fmtI(P.codebook)} (not counted), trained by L_VQ only on detached ĉ.` });
  arrow(g, [[cx + cw / 2, 590], [cx + cw / 2, 565]], { c: "#d1a92f", m: "ah", w: 1 });
  box(g, { x: cx - 30, y: 650, w: cw + 60, h: 40, k: "white", label: "shortcut S ~ U{1, 2}: M 1 (0,1) · 2a (⅓,⅔) · 2b (⅔,⅓)", sub: "from S₀ = 0 · no decoder · training only", size: 11, dash: "5 4",
    tip: "<b>Shortcut trajectory</b> (LoopFormer): a second trajectory with a smaller budget, supervised by NCP(u_s) and by consistency with sg(u₃). Makes M = 1, 2 usable at inference (elasticity)." });
  box(g, { x: cx - 30, y: 710, w: 100, h: 30, k: "loss", label: "0.1 · NCP(u₃)", size: 11 });
  box(g, { x: cx + 80, y: 710, w: 110, h: 30, k: "loss", label: "0.05 · NCP(u_s)", size: 11 });
  box(g, { x: cx + 200, y: 710, w: 100, h: 30, k: "loss", label: "L_VQ", size: 11, tip: "codebook only" });
  box(g, { x: cx - 30, y: 760, w: cw + 60, h: 30, k: "loss", label: "0.1 · L_scons(rms S_s, sg rms S₃)", size: 11.5, tip: "shortcut consistency on the gain-free normalised state: cannot shrink out_norm's gain (v3_ref T4c)" });
  arrow(g, [[cx + 20, 564], [cx + 20, 709]], { c: "#d0667a", m: "ahR", w: 1 });
  arrow(g, [[cx + cw - 30, 690], [cx + cw - 30, 759]], { c: V3C.sc, m: "ah", w: 1, dash: "4 3" });
  note(g, cx - 30, 815, "Recurrence is tested by the arm ladder: 1-pass → 3-pass → t-conditioned → full recipe", { size: 10.5, color: "#5a4a8a" });

  // ---------------- decoder
  frame(g, { x: 850, y: 60, w: 340, h: 800, title: `Decoder (confined) · ${fmtP(T.decoder)}`, stroke: "#f1c6a6" });
  const dx = 900, dw = 240;
  box(g, { x: dx, y: 740, w: dw, h: 40, k: "plan", label: "Plan P = W_u u₃ + W_c c_pred + o_r", sub: "j*(t) = ⌊(t+1)/4⌋ − 1", size: 11.5, tip: `<b>Plan builder</b> · ${fmtI(P.dec_plan)} params<br>W_u, W_c 256 → 256; null plan for the first 3 positions of each document.`, src: SRC3D.dec });
  arrow(g, [[cx + cw, 485], [830, 485], [830, 760], [dx, 760]], { c: COL.plan, m: "ahP", w: 1.8 });
  dim(g, 836, 500, "u₃, c_pred");
  for (let l = 0; l < 4; l++) {
    const y = 660 - l * 70;
    box(g, { x: dx, y, w: dw, h: 46, k: "attn", label: `Confined layer ${l + 1}`, sub: "q from own D · k, v from tok only (w 8)", size: 12, tip: `<b>Confined layer ${l + 1}</b> · ${fmtI(P.dec_layer)} params<br>4 heads × 64, SwiGLU 256 → 1024 → 256; no position reads another's plan.`, src: SRC3D.dec });
    arrow(g, [[dx + dw / 2, y + 46 + (l ? 24 : 34)], [dx + dw / 2, y + 46]], { c: COL.plan, m: "ahP" });
    arrow(g, [[1170, y + 23], [dx + dw, y + 23]], { c: COL.L, m: "ahL", w: 1 });
  }
  arrow(g, [[330, 135], [400, 135], [400, 46], [1170, 46], [1170, 683]], { c: COL.L, m: "ahL", w: 1.6, head: false });
  dim(g, 560, 55, "tok (reach 63) → keys and values of every decoder layer");
  box(g, { x: dx, y: 390, w: dw, h: 28, k: "norm", label: "h = zcRMSNorm(D)", size: 12 });
  arrow(g, [[dx + dw / 2, 450], [dx + dw / 2, 419]], { c: COL.plan, m: "ahP" });
  box(g, { x: dx, y: 330, w: dw, h: 32, k: "emb", label: "logits = h Eᵀ (tied)", size: 12 });
  arrow(g, [[dx + dw / 2, 390], [dx + dw / 2, 363]]);
  box(g, { x: dx, y: 270, w: 110, h: 30, k: "loss", label: "L_NTP(u₃)", size: 12 });
  arrow(g, [[dx + 55, 330], [dx + 55, 301]], { c: "#d0667a", m: "ahR" });
  box(g, { x: dx + 20, y: 160, w: 200, h: 56, k: "white", label: "NextLat: removed", sub: "at plan boundaries it duplicates NCP", size: 11.5, dash: "5 4",
    tip: "<b>NextLat off by default</b> (draft §5.3)<br>v2: nl_h / nl_kl flat while NTP fell; gradient cosine with NTP −0.28 to −0.54 on W_u. Saves 0.66M params and ≈7 % of training compute." });
  return svg;
}

/* =========================================================================================== 2 encoder */
function f3dEncoder(A) {
  const P = A.params, e = A.ecfg;
  const svg = makeSvg(1180, 520), g = el("g", {}, svg);
  frame(g, { x: 20, y: 20, w: 520, h: 480, title: "S layer (×6) · reads {L} · writes {L, G}", stroke: "#b9d1ef" });
  frame(g, { x: 640, y: 20, w: 520, h: 480, title: "F layer (×2) · reads {L, G} · writes {G} only", stroke: "#f1c6a6" });
  [[20, "S", P.enc_layer_S, P.enc_layer_S_gr, P.enc_layer_S_attn], [640, "F", P.enc_layer_F, P.enc_layer_F_gr, P.enc_layer_F_attn]].forEach(([x0, k, tot, gr, at]) => {
    const x = x0 + 60, w = 400;
    box(g, { x, y: 430, w, h: 34, k: "gate", label: k === "S" ? "GR read of {L}" : "GR read of {L, G}", sub: `σ(U V rms(R) + 2)(1+γ) rms(R) · rank 32 · ${fmtI(gr)} params (reads + writes)`, size: 12 });
    box(g, { x, y: 350, w, h: 50, k: k === "S" ? "attn" : "full", label: k === "S" ? `Gated SWA, w ${e.window}, partial RoPE ${e.rope_dims}/${e.head_dim} · 4 heads` : "Gated full causal attention, NoPE + τ_h · 6 heads",
      sub: `q 256 → ${k === "S" ? "4" : "6"}×64 · k, v 256 → 2×64 · out gate σ · ${fmtI(at)} params`, size: 12, tip: k === "S" ? "q/k/v causal depthwise conv k 4 (document-aware), QK zcRMSNorm" : "F-layer heads 4 → 6 (v1's geometry) where facts enter G: +197k params, ≈ +2 % MACs; GQA repeat 3" });
    box(g, { x, y: 270, w, h: 34, k: "gate", label: k === "S" ? "write L and G (per-stream scale 2σ(u·x+a))" : "write G only", size: 12 });
    box(g, { x, y: 190, w, h: 34, k: "gate", label: "GR read (again)", size: 12 });
    box(g, { x, y: 110, w, h: 44, k: "ffn", label: `SwiGLU 256 → ${e.d_ff} → 256 (ratio 4)`, sub: `${fmtI(P.enc_layer_S_ffn)} params · v1 2.75, v2 704`, size: 12 });
    box(g, { x, y: 60, w, h: 30, k: "gate", label: k === "S" ? "write L and G" : "write G only", size: 12 });
    [[430, 401], [350, 305], [270, 225], [190, 155], [110, 91]].forEach(([a, b]) => arrow(g, [[x + w / 2, a], [x + w / 2, b]], { c: k === "S" ? COL.L : COL.G, m: k === "S" ? "ahL" : "ahG" }));
    txt(g, x0 + 260, 486, `layer total ${fmtI(tot)} params`, { size: 12, weight: 600, color: "#555" });
  });
  txt(g, 590, 250, "same\ngrammar", { size: 12, color: COL.grey });
  return svg;
}

/* =========================================================================================== 3 engram */
function f3dEngram(A) {
  const P = A.params, e = A.ecfg;
  const svg = makeSvg(1180, 470), g = el("g", {}, svg);
  box(g, { x: 30, y: 200, w: 160, h: 50, k: "io", label: "canonical ids", sub: "x_{t−2}, x_{t−1}, x_t", size: 12 });
  box(g, { x: 225, y: 160, w: 170, h: 130, k: "white", label: "hash 2-grams, 3-grams", sub: "× 4 heads = 8 row ids", size: 12, tip: "multiplicative-XOR hash mod 2⁶¹, then mod rows (prime). Another document → boundary id 0. The key is depth-invariant." });
  arrow(g, [[190, 225], [225, 225]]);
  for (let t = 0; t < 8; t++) {
    const y = 40 + t * 48;
    box(g, { x: 440, y, w: 180, h: 38, k: "mem", label: `table ${t + 1} · ${t < 4 ? "2-gram" : "3-gram"} head ${t % 4 + 1}`, sub: `${fmtI(e.engram_rows)} × ${e.engram_dim}`, size: 11 });
    arrow(g, [[395, 225], [415, 225], [415, y + 19], [440, y + 19]], { c: COL.grey, m: "ahg", w: 0.8, r: 3 });
    arrow(g, [[620, y + 19], [670, 225]], { c: "#d1a92f", m: "ah", w: 0.8, head: false });
  }
  box(g, { x: 670, y: 190, w: 120, h: 70, k: "mem", label: "e_t = 8 × 48", sub: "gathered ONCE", size: 12, tip: "The rows are gathered once per token and shared by both reads." });
  const r = (y, n, conv, ctx, p) => {
    frame(g, { x: 830, y, w: 330, h: 190, title: `read ${n} · after layer ${n === 1 ? 0 : 4}`, stroke: n === 1 ? "#e7cf7a" : "#c9b4ec" });
    box(g, { x: 850, y: y + 36, w: 290, h: 34, k: "gate", label: `k = W_K e · v = W_V e${n === 2 ? " (W_V zero-init)" : ""}`, sub: "384 → 256", size: 11.5 });
    box(g, { x: 850, y: y + 80, w: 290, h: 34, k: "gate", label: `α = σ(rms(h_L)·rms(k)/√256) · gate sees ${ctx}`, size: 11 });
    box(g, { x: 850, y: y + 124, w: 290, h: 34, k: n === 1 ? "attn" : "white", label: conv ? "y = SiLU(conv₄(α v)) → L, G" : "y = SiLU(α v) → L, G  (no conv)", size: 11.5 });
    txt(g, 995, y + 176, `${fmtI(p)} params`, { size: 11, color: COL.grey });
    arrow(g, [[790, 225], [812, 225], [812, y + 53], [850, y + 53]], { c: "#d1a92f", m: "ah", w: 1 });
  };
  r(20, 1, true, "10 tokens of L", P.engram_read1); r(250, 2, false, "43 tokens of L", P.engram_read2);
  txt(g, 30, 360, "Why one table read twice (draft §3.2)", { size: 13, weight: 700, anchor: "start" });
  ["The n-gram key is the same at every depth: 4 tables would store one map 4 times (6.4 trigrams / row).",
   "Depth changes what the value is FOR, so each read has its own W_k, W_v, gate and write (Mobius principle).",
   "Read 2's gate reads L only: a G-reading gate would make L depend on FULL writes (reach → 1,024).",
   "No conv on read 2 keeps L-reach 63 / decoder reach 70 (with a conv: 66 / 73), verified in float64.",
   `W_V₂ = 0 ⇒ bit-identical to v1's one-read encoder at step 0 · total ${fmtP(P.engram_table)} lookups (8 × ${fmtI(e.engram_rows)} × 48).`]
    .forEach((s, i) => txt(g, 30, 386 + i * 18, s, { size: 11.5, anchor: "start" }));
  return svg;
}

/* =========================================================================================== 4 concepts */
function f3dConcepts(A) {
  const P = A.params, q = A.pq;
  const svg = makeSvg(1180, 330), g = el("g", {}, svg);
  for (let i = 0; i < 8; i++) {
    const x = 40 + i * 70;
    box(g, { x, y: 220, w: 58, h: 40, k: i < 4 ? "attn" : "white", label: `t${i}`, size: 12 });
    arrow(g, [[x + 29, 220], [x + 29, 190]], { c: COL.grey, m: "ahg", w: 0.9 });
  }
  box(g, { x: 40, y: 150, w: 268, h: 40, k: "gate", label: "mix_t = softmax(W rms G) · {e0, rms L, rms G}", size: 11 });
  box(g, { x: 320, y: 150, w: 268, h: 40, k: "gate", label: "mix per token", size: 11 });
  box(g, { x: 40, y: 80, w: 268, h: 40, k: "conc", label: "concat 4 × 256 = 1,024 → W_pool → 256", size: 11.5, tip: `<b>W_pool</b> 1,024 → 256 · ${fmtI(P.pool)} params · init = mean pooling` });
  box(g, { x: 320, y: 80, w: 268, h: 40, k: "conc", label: "next chunk", size: 11.5 });
  arrow(g, [[174, 150], [174, 121]], { c: COL.plan, m: "ahP" }); arrow(g, [[454, 150], [454, 121]], { c: COL.plan, m: "ahP" });
  circ(g, 174, 40, 22, { f: "#d7f2ed", s: "#3a9d8f", label: "ĉ₀" }); circ(g, 454, 40, 22, { f: "#d7f2ed", s: "#3a9d8f", label: "ĉ₁" });
  arrow(g, [[174, 80], [174, 63]], { c: COL.plan, m: "ahP" }); arrow(g, [[454, 80], [454, 63]], { c: COL.plan, m: "ahP" });
  const x = 680;
  txt(g, x, 50, "d_c = d = 256 and PQ 32 × 128 × 8", { size: 14, weight: 700, anchor: "start" });
  ["Compression 4 : 1, exactly v1's (v2 used 1,024 → 384 = 2.7 : 1).",
   "ĉ = √256 · c / ‖c‖: every concept on the sphere of radius 16.",
   `PQ follows d_c: ${q.S} segments × ${q.N} codes × ${q.ds} dims = ${q.bits} bits (v1/v2: 48 × 128 × 8).`,
   "Per-segment metrics stay comparable: mean square 1 per dim at any d_c;",
   "   segments 8-dim with 128 codes; τ init log 16 = v1's; codes_min, VE are ratios.",
   "Rejected: 32 × 256 × 8 (finer than v1 ever had), 64 × 128 × 4 (near grid).",
   "Target hygiene: NCP pairs only if the target chunk is complete and same-doc."].forEach((s, i) => txt(g, x, 82 + i * 24, s, { size: 12, anchor: "start", color: i === 4 ? "#666" : COL.ink }));
  return svg;
}

/* =========================================================================================== 5 core layer */
function f3dCore(A) {
  const P = A.params, c = A.core;
  const svg = makeSvg(1180, 620), g = el("g", {}, svg);
  frame(g, { x: 20, y: 20, w: 720, h: 580, title: "One core layer ℓ (4 distinct layers per pass) · Φ(S; C, t, Δ)", stroke: "#9fd3ca" });
  const x = 90, w = 420;
  box(g, { x: 60, y: 530, w: 140, h: 40, k: "conc", label: "C (evidence)", size: 12, tip: "re-read every sublayer of every pass" });
  box(g, { x: 400, y: 530, w: 140, h: 40, k: "plan", label: "S (memory)", size: 12, tip: "raw carried state; S₀ = 0" });
  box(g, { x, y: 450, w, h: 44, k: "gate", label: "x = ½[g_C(1+γ_C)(1+γᶜₚ) rms C + g_S(1+γ_S)(1+γˢₚ) rms S]", sub: "gated read · rank 32 · per-stream (t, Δ) gains", size: 11, tip: "The per-stream read gains let a pass weigh evidence vs memory differently at t = 0 and late in the trajectory." });
  arrow(g, [[130, 530], [130, 495]], { c: COL.plan, m: "ahP" }); arrow(g, [[470, 530], [470, 495]], { c: COL.plan, m: "ahP" });
  box(g, { x, y: 370, w, h: 50, k: "full", label: "gated full causal attention · 6 heads × 64, GQA kv 2", sub: "doc mask over 256 concepts · NoPE · τ_h · QK zcRMSNorm · q/k/v conv k4 (16 tokens)", size: 11.5, tip: `attention ${fmtI(P.core_attn)} params · same class as the encoder's F layer` });
  arrow(g, [[x + w / 2, 450], [x + w / 2, 421]]);
  box(g, { x, y: 310, w, h: 30, k: "plan", label: "S ← S + 2σ(u·x+a)·(1+α_attn,p)·attn", size: 12 });
  arrow(g, [[x + w / 2, 370], [x + w / 2, 341]]);
  box(g, { x, y: 250, w, h: 30, k: "gate", label: "x′ = read {C, S} again (γ_C,ffn, γ_S,ffn)", size: 12 });
  arrow(g, [[x + w / 2, 310], [x + w / 2, 281]]);
  box(g, { x, y: 170, w: 200, h: 46, k: "ffn", label: `own SwiGLU ${c.own_ff}`, sub: `${fmtI(P.core_own_ffn)} params`, size: 12 });
  box(g, { x: x + 220, y: 170, w: 200, h: 46, k: "ffn", label: `shared SwiGLU ${c.shared_ff}`, sub: "same weights in all layers", size: 12, fill: "#f5b98f", tip: `shared ${fmtI(P.core_shared)} params · w2 zero-init: writes 0 at step 0` });
  arrow(g, [[x + 100, 250], [x + 100, 217]]); arrow(g, [[x + 320, 250], [x + 320, 217]]);
  box(g, { x, y: 90, w, h: 34, k: "plan", label: "S ← S + s·[(1+α_own,p) own(x′) + (1+α_sh,p) shared(x′)]", size: 11.5 });
  arrow(g, [[x + 100, 170], [x + 100, 125]]); arrow(g, [[x + 320, 170], [x + 320, 125]]);
  arrow(g, [[x + w / 2, 90], [x + w / 2, 62]]); note(g, x + w / 2 + 8, 70, "→ layer ℓ + 1", { size: 11 });
  // table column
  frame(g, { x: 560, y: 60, w: 160, h: 460, title: "(t, Δ) row p", stroke: "#c9b4ec" });
  c.sites.forEach((s, i) => box(g, { x: 575, y: 100 + i * 56, w: 130, h: 40, k: "gate", label: s, sub: "256-vector · 0 at init", size: 11, rx: 6 }));
  [[575, 120, 511, 472], [575, 176, 511, 478], [575, 232, 511, 325], [575, 288, 511, 265], [575, 344, 511, 268], [575, 400, 511, 107], [575, 456, 511, 113]]
    .forEach(([a, b, c2, d]) => arrow(g, [[a, b], [c2 + 20, d]], { c: V3C.tab, m: "ah", w: 0.7, dash: "3 3" }));
  // right: facts
  const fx = 780;
  txt(g, fx, 50, "Concept module v3", { size: 15, weight: 700, anchor: "start" });
  [["params (compute)", fmtI(A.table.concept_module)], ["per layer", fmtI(P.core_layer)], ["· attention 6 × 64", fmtI(P.core_attn)], ["· own FFN 736", fmtI(P.core_own_ffn)],
   ["· 2 GR reads + writes", fmtI(P.core_gr + P.core_writes)], ["shared FFN 1024", fmtI(P.core_shared)], ["(t, Δ) table", fmtI(P.core_table)], ["NCP head + norm", fmtI(P.ncp_head + P.core_out_norm)],
   ["passes", "M = 3 main + shortcut 1 / 2"], ["carry", "raw S (no LN_h)"], ["MoE", "none (dense)"], ["identity at init", "table 0 · shared w2 0"]]
    .forEach(([a, b], i) => { txt(g, fx, 84 + i * 25, a, { size: 12, anchor: "start", color: "#666" }); txt(g, fx + 170, 84 + i * 25, b, { size: 12, anchor: "start", weight: 600 }); });
  txt(g, fx, 408, "Why this anatomy", { size: 13, weight: 700, anchor: "start" });
  ["6 heads: every failing v2 probe is a retrieval", "own 736 ‖ shared 1024 = 1,760 hidden per layer", "parallel shared branch costs MACs, not", "   sequential applications (what sank v2)", "dense: under a param cap MoE idles weights", "   (v2: 18.83M nominal, 15.69M active)"]
    .forEach((s, i) => txt(g, fx, 432 + i * 22, s, { size: 12, anchor: "start", color: s.startsWith("   ") ? "#666" : COL.ink }));
  return svg;
}

/* =========================================================================================== 6 trajectories */
function f3dTraj(A) {
  const svg = makeSvg(1180, 500), g = el("g", {}, svg), sc = A.core.schedules;
  txt(g, 30, 28, "S_i = Φ(S_{i−1}; C, t_{i−1}, Δ_i),  S₀ = 0,  Σ Δ_i = 1 — every trajectory lands on t = 1", { size: 14, weight: 700, anchor: "start" });
  const x0 = 250, W = 720, px = t => x0 + t * W / 3;
  [0, 1, 2, 3].forEach(t => { el("line", { x1: px(t), y1: 56, x2: px(t), y2: 410, stroke: "#e2e5ea", "stroke-dasharray": "3 4" }, g); txt(g, px(t), 425, `t = ${THIRD(t)}`, { size: 12, color: "#666" }); });
  const rows = [["3", "main M = 3", "every micro-batch · drives NTP + NCP", COL.plan, 1], ["2a", "shortcut 2a", "25 % · (⅓, ⅔)", V3C.sc, 0], ["2b", "shortcut 2b", "25 % · (⅔, ⅓) · LoopFormer predicts 2b > 2a", V3C.sc, 0], ["1", "shortcut M = 1", "50 % · one pass does all the work", V3C.sc, 0]];
  rows.forEach(([k, n, s, col, main], r) => {
    const y = 90 + r * 84;
    txt(g, 30, y, n, { size: 13, weight: 700, anchor: "start", color: main ? COL.plan : "#7a5bd0" }); note(g, 30, y + 18, s, { size: 11 });
    let t = 0;
    sc[k].forEach(([a, b], i) => {
      const xa = px(a), xb = px(a + b);
      box(g, { x: xa + 6, y: y - 16, w: xb - xa - 12, h: 32, k: main ? "full" : "gate", label: `pass ${i + 1} · (t ${THIRD(a)}, Δ ${THIRD(b)})`, size: 11.5, dash: main ? null : "5 4",
        tip: `<b>pass ${i + 1}</b>: all 4 layers with table row (${THIRD(a)}, ${THIRD(b)}). ${a === 0 && b === 1 ? "Told it owes the whole job." : b === 1 ? "Told it owes a third." : "Told it owes two thirds."}`, src: SRC3D.traj });
      t = a + b;
    });
    el("circle", { cx: px(3) + 16, cy: y, r: 6, fill: main ? COL.plan : "#b9a3ec" }, g);
    txt(g, px(3) + 28, y, main ? "u₃ → decoder, NCP" : "u_s → NCP, L_scons", { size: 11.5, anchor: "start", color: main ? COL.plan : "#7a5bd0" });
  });
  arrow(g, [[px(3) + 16, 174], [px(3) + 16, 96]], { c: V3C.sc, m: "ah", dash: "4 3", w: 1.1 }); note(g, px(3) + 24, 140, "sg(u₃) target", { size: 10.5 });
  txt(g, 30, 452, "Exactly 6 reachable (t, Δ) pairs on the ⅓ grid → the table is a lookup of the fully trained function. Cost 3 + 1.5 = 4.5 passes in training (4.25 reusing 2a's exact first step); 3 passes at eval.", { size: 12, anchor: "start" });
  txt(g, 30, 474, "Terminal-only supervision: intermediate states are never pulled toward the answer. Full backprop through both trajectories (no detach, no truncation).", { size: 12, anchor: "start", color: "#555" });
  return svg;
}

/* =========================================================================================== 7 (t, Δ) table */
function f3dTable(A) {
  const c = A.core, svg = makeSvg(1180, 470), g = el("g", {}, svg);
  const pr = c.pairs, use = { "0,1": "3 · 2a", "1,1": "3", "2,1": "3 · 2b", "0,2": "2b", "1,2": "2a", "0,3": "1" };
  const freq = { "0,1": "100 %", "1,1": "100 %", "2,1": "100 %", "0,2": "25 %", "1,2": "25 %", "0,3": "50 %" };
  txt(g, 30, 28, "The (t, Δ) table: 6 pairs × 4 layers × 7 sites × 256 = 43,008 parameters, zero-init, no weight decay", { size: 14, weight: 700, anchor: "start" });
  const x0 = 200, y0 = 70, cw = 26, ch = 44;
  pr.forEach(([a, b], r) => {
    const y = y0 + r * (ch + 8), key = `${a},${b}`, main = b === 1;
    txt(g, x0 - 12, y + ch / 2 - 7, `(t ${THIRD(a)}, Δ ${THIRD(b)})`, { size: 12, anchor: "end", weight: 700, color: main ? COL.plan : "#7a5bd0" });
    note(g, x0 - 12, y + ch / 2 + 9, `${use[key]} · ${freq[key]}`, { size: 10.5, anchor: "end" });
    for (let l = 0; l < 4; l++) for (let s = 0; s < 7; s++) {
      const x = x0 + l * (7 * cw + 14) + s * cw;
      const v = Math.abs(Math.sin(r * 3.1 + l * 1.7 + s * 0.9));
      el("rect", { x, y, width: cw - 3, height: ch, rx: 3, fill: `rgba(122,91,208,${0.08 + 0.35 * v * (main ? 1 : 0.6)})`, stroke: "#c9b4ec", "stroke-width": 0.6 }, g);
    }
  });
  for (let l = 0; l < 4; l++) txt(g, x0 + l * (7 * cw + 14) + 3.5 * cw, y0 - 14, `layer ${l + 1}`, { size: 11.5, weight: 600 });
  c.sites.forEach((s, i) => txt(g, x0 + i * cw + cw / 2 - 1, y0 + 6 * (ch + 8) + 10, ["γC", "γS", "α", "γC", "γS", "αo", "αs"][i], { size: 9.5, color: "#666" }));
  note(g, x0, y0 + 6 * (ch + 8) + 28, "attn read (γ_C, γ_S) · attn write α · ffn read (γ_C, γ_S) · own-FFN write α · shared-FFN write α   (shading illustrative, not trained values)", { size: 11 });
  const fx = 905;
  txt(g, fx, 80, "Why a table, not the paper's MLP", { size: 13, weight: 700, anchor: "start" });
  ["All 6 pairs are visited in training:", "a lookup IS the trained function.", "43k vs ≈ 1.3M for Fourier + MLP.", "", "Init: (1+γ) = (1+α) = 1 ⇒", "bit-identical to the unconditioned", "core at step 0 (not AdaLN-Zero,", "which writes 0 with S₀ = 0).", "", "No decay: an all-zero table IS", "v1's fixed point; decay pulls", "back toward it."]
    .forEach((s, i) => txt(g, fx, 104 + i * 20, s, { size: 12, anchor: "start" }));
  return svg;
}

/* =========================================================================================== 8 stagnation */
function f3dStag(A) {
  const svg = makeSvg(1180, 480), g = el("g", {}, svg);
  txt(g, 30, 28, "Why v1's loop idled (leading hypothesis: a fixed point in direction space) and how v3 tests it", { size: 14, weight: 700, anchor: "start" });
  // v1 measured norms
  const gx = 60, gy = 250, gw = 420, gh = 170, ns = [9.8, 18.5, 27.1, 35.6, 44.1, 52.6, 61.0, 69.4], ang = [6.7, 1.25, 0.55, 0.35, 0.22, 0.16, 0.11];
  el("line", { x1: gx, y1: gy, x2: gx + gw, y2: gy, stroke: "#bbb" }, g); el("line", { x1: gx, y1: gy, x2: gx, y2: gy - gh, stroke: "#bbb" }, g);
  const px = i => gx + 20 + i * (gw - 40) / 7, py = v => gy - v / 72 * gh;
  el("path", { d: ns.map((v, i) => `${i ? "L" : "M"}${px(i)},${py(v)}`).join(" "), fill: "none", stroke: "#d0667a", "stroke-width": 2.2 }, g);
  ns.forEach((v, i) => { el("circle", { cx: px(i), cy: py(v), r: 3.5, fill: "#fff", stroke: "#d0667a", "stroke-width": 1.5 }, g); txt(g, px(i), gy + 14, i + 1, { size: 10.5, color: "#666" }); });
  txt(g, gx, 62, "v1 measured ‖S‖ per pass (linear ⇒ aligned writes)", { size: 12, weight: 600, anchor: "start" });
  txt(g, gx, 80, `update angles ${ang.map(a => a + "°").join(" → ")}`, { size: 11, anchor: "start", color: "#a0405a" });
  txt(g, gx + gw / 2, gy + 32, "pass", { size: 11, color: "#666" });
  ["Hypothesis: the GR read normalises S, so if S's direction settles,", "every pass sees nearly the same input and writes nearly the same vector.", "The (t, Δ) table removes that route to invariance. It lets passes differ;", "it does not force them to do useful work: the arms decide."]
    .forEach((s, i) => txt(g, gx, 310 + i * 20, s, { size: 12, anchor: "start" }));
  // signatures table
  const tx = 560, cols = ["typical of", "‖ΔS_i‖", "angles", "ρ = ‖S_M‖ / Σ‖ΔS_i‖", "arms"];
  const rows = [["v1's fixed point", "constant", "→ 0", "→ 1", "no gain"], ["idle passes (data-limited)", "passes 2–3 → 0", "—", "—", "no gain"], ["useful iteration", "comparable", "any", "any", "gain"]];
  const cx = [tx, tx + 190, tx + 305, tx + 380, tx + 515];
  cols.forEach((c, i) => txt(g, cx[i], 70, c, { size: 12, weight: 700, anchor: "start" }));
  rows.forEach((r, j) => r.forEach((c, i) => {
    const y = 104 + j * 40;
    if (!i) box(g, { x: tx - 8, y: y - 15, w: 590, h: 30, k: j === 2 ? "conc" : j === 0 ? "loss" : "white", label: "", rx: 6 });
    txt(g, cx[i], y, c, { size: 12, anchor: "start", weight: i === 4 ? 700 : 400 });
  }));
  txt(g, tx, 250, "Geometry is a diagnostic, not a gate (r2)", { size: 13, weight: 700, anchor: "start" });
  ["Not necessary: aligned updates (1,0), (1,0.05), (1,0.05) give angle ≈ 0, ρ = 0.9997,", "   yet cross a 0.03 decision line only on pass 3.",
   "Not sufficient: pair-specific gains can rotate S without doing useful work.",
   "Test: ladder A1 (1 pass) · A3 (3 plain) · T3 (t-conditioned) · Full, paired stats.", "   Carry renorm only if no recurrence gain AND v1's signature (angle < 10°, ρ > 0.9).",
   "The design permits iteration; it does not require it."]
    .forEach((s, i) => txt(g, tx, 276 + i * 21, s, { size: 12, anchor: "start", color: s.startsWith("   ") ? "#555" : COL.ink }));
  return svg;
}

/* =========================================================================================== 9 decoder + plan */
function f3dDecoder(A) {
  const P = A.params;
  const svg = makeSvg(1180, 440), g = el("g", {}, svg);
  txt(g, 30, 28, "Which plan each position reads: j*(t) = ⌊(t+1)/4⌋ − 1 (the newest chunk ending at or before t)", { size: 13, weight: 600, anchor: "start" });
  const x0 = 60, cw = 52;
  for (let t = 0; t < 20; t++) {
    const j = Math.floor((t + 1) / 4) - 1, x = x0 + t * cw;
    box(g, { x, y: 60, w: cw - 6, h: 30, k: Math.floor(t / 4) % 2 ? "attn" : "white", label: `x${t}`, size: 11, rx: 5 });
    box(g, { x, y: 120, w: cw - 6, h: 30, k: j < 0 ? "io" : "plan", label: j < 0 ? "null" : `u${j}`, size: 11, rx: 5, tip: `position ${t} predicts x${t + 1} from plan ${j < 0 ? "null" : "chunk " + j} (u₃[${j}] + c_pred[${j}], which looks at chunk ${j + 1})` });
  }
  txt(g, 60, 185, "c_pred[j*] is the core's prediction of chunk j* + 1, the chunk that contains x_{t+1}: the look-ahead half of the plan (turning it off cost v1 16 carry points).", { size: 12, anchor: "start", color: "#2e6f66" });
  const y = 230;
  box(g, { x: 60, y, w: 250, h: 40, k: "plan", label: "D₀ = tok + P_t", sub: `plan ${fmtP(P.dec_plan)} · W_u, W_c 256 → 256`, size: 11.5 });
  box(g, { x: 350, y, w: 300, h: 40, k: "attn", label: "× 4 confined layers", sub: `${fmtP(P.dec_layer)} each · 4 heads · q from D, k/v from tok, w 8`, size: 11.5 });
  box(g, { x: 690, y, w: 190, h: 40, k: "norm", label: "h = zcRMSNorm(D)", size: 12 });
  box(g, { x: 920, y, w: 220, h: 40, k: "emb", label: "logits = h Eᵀ → L_NTP", size: 12 });
  arrow(g, [[310, y + 20], [350, y + 20]]); arrow(g, [[650, y + 20], [690, y + 20]]); arrow(g, [[880, y + 20], [920, y + 20]]);
  ["Confinement unchanged from v1: nothing reads D at another position, so token-path reach = 63 + 7 = 70 at any depth.",
   "Anything older than 70 tokens reaches position t only through its plan, i.e. through the concept module.",
   `Changes from v1: SwiGLU 1024 (ratio 4, same as the encoder) and 4 heads (window-8 attention gains little from more). ${fmtP(A.table.decoder)} params.`,
   "NextLat removed: at plan boundaries (t ≡ 2 mod 4) it asks for exactly what c_pred already predicts; elsewhere it is redundant.",
   "Width-256 risk named: logits rank ≤ 256 over 8,192 tokens; E row norms, output entropy and ‖P‖/‖tok‖ logged hourly."]
    .forEach((s, i) => txt(g, 60, 310 + i * 24, s, { size: 12.5, anchor: "start" }));
  return svg;
}

/* =========================================================================================== 10 losses */
function f3dLosses(A) {
  const L = A.losses;
  const cols = ["decoder", "core", "encoder L (tok)", "encoder G / F / pool (C)", "codebook", "table · shared FFN"];
  const M = { "L_NTP(u₃)": [1, 1, 1, 1, 0, 1], "L_NCP(u₃)": [0, 1, 0, 1, 0, 1], "L_NCP(u_s)": [0, 1, 0, 1, 0, 1], "L_scons": [0, 1, 0, 1, 0, 1], "L_VQ": [0, 0, 0, 0, 1, 0] };
  const svg = makeSvg(1180, 420), g = el("g", {}, svg);
  const x0 = 360, cw = 130;
  txt(g, 30, 40, "loss · weight", { size: 12, weight: 700, anchor: "start" });
  cols.forEach((c, i) => txt(g, x0 + i * cw + cw / 2, 40, c, { size: 11.5, weight: 700 }));
  L.forEach((l, r) => {
    const y = 70 + r * 44;
    box(g, { x: 30, y, w: 310, h: 32, k: "loss", label: `${l.name} · ${l.weight}`, size: 11.5, dash: /u_s|scons/.test(l.name) ? "5 4" : null, tip: `<b>${l.name}</b> × ${l.weight}<br>${l.formula}<br>reaches: ${l.reaches.join(", ")}`, src: SRC3D.obj });
    (M[l.name] || []).forEach((on, i) => el("circle", { cx: x0 + i * cw + cw / 2, cy: y + 16, r: on ? 8 : 3, fill: on ? (/u_s|scons/.test(l.name) ? "#a07cc5" : "#d0667a") : "#dcdcdc" }, g));
  });
  txt(g, 30, 312, "L = L_NTP(u₃) + 0.1 L_NCP(u₃) + 0.05 L_NCP(u_s) + 0.1 L_scons(rms S_s, sg rms S₃) + L_VQ", { size: 13, weight: 700, anchor: "start" });
  ["v1's three terms keep v1's weights (owner: \"v1 objective functions are good\"); dashed purple = the shortcut trajectory's terms.",
   "Removed from v2: NextLat-h, NextLat-KL, MoE LB, router z, per-pass NCP, final-window consistency. Denominators over the whole optimizer batch.",
   "The encoder's global half is trained only through C: three sources in v3 (NTP via plan, main NCP, shortcut NCP + L_scons); v1 had two."]
    .forEach((s, i) => txt(g, 30, 340 + i * 22, s, { size: 12, anchor: "start" }));
  return svg;
}

/* =========================================================================================== 11 budget */
function f3dBudget(A) {
  const T = A.table, v1 = A.v1, v2 = A.v2, M = A.macs;
  const svg = makeSvg(1180, 560), g = el("g", {}, svg);
  txt(g, 30, 30, "Compute parameters by module (v1's convention: embedding, Engram table, codebook excluded)", { size: 13, weight: 600, anchor: "start" });
  const list = [["v1 (d 384)", v1], ["v2 (d 256, D6)", v2], ["v3 draft (d 256)", { encoder: T.encoder_non_embedding, core: T.concept_module, decoder: T.decoder, compute: T.compute_params }]];
  const sx = v => v / 26e6 * 700, x0 = 200;
  list.forEach(([n, p], i) => {
    const y = 60 + i * 46; let x = x0;
    txt(g, x0 - 12, y + 13, n, { size: 12, anchor: "end", weight: i === 2 ? 700 : 400 });
    [["encoder", "#d6e6f7", "#3b78c2"], ["core", "#d2efea", "#2e9c8c"], ["decoder", "#fbdcc8", "#e07b39"]].forEach(([k, f, s]) => {
      const w = sx(p[k]); el("rect", { x, y, width: w, height: 26, fill: f, stroke: s, rx: 3 }, g);
      if (w > 40) txt(g, x + w / 2, y + 13, `${(p[k] / 1e6).toFixed(2)}M · ${(100 * p[k] / p.compute).toFixed(0)}%`, { size: 10.5 }); x += w;
    });
    txt(g, x + 8, y + 13, `${(p.compute / 1e6).toFixed(2)}M`, { size: 12, anchor: "start", weight: 700 });
  });
  el("line", { x1: x0 + sx(19e6), y1: 52, x2: x0 + sx(19e6), y2: 200, stroke: "#7a5bd0", "stroke-dasharray": "4 3" }, g); txt(g, x0 + sx(19e6), 212, "19M cap (owner: \"within 19 million\")", { size: 11, color: "#7a5bd0" });
  // MACs bar
  txt(g, 30, 250, "Forward MACs per token (analytical, v3_budget.py; 6-head F-layer scores ≈ +2 % not included)", { size: 13, weight: 600, anchor: "start" });
  const parts = [["encoder", M.encoder, "#d6e6f7"], ["core (4.5 passes)", M.core_train, "#d2efea"], ["decoder", M.decoder, "#fbdcc8"], ["head", M.head, "#e6e0f5"]];
  let x = 200; const tot = M.fwd_train, sw = 700 / tot;
  parts.forEach(([n, v, f]) => { el("rect", { x, y: 270, width: v * sw, height: 28, fill: f, stroke: "#aab", rx: 3 }, g); txt(g, x + v * sw / 2, 284, `${n} ${(v / 1e6).toFixed(1)}M`, { size: 10.5 }); x += v * sw; });
  txt(g, x + 8, 284, `train ${(tot / 1e6).toFixed(1)}M · eval ${(M.fwd_eval / 1e6).toFixed(1)}M`, { size: 12, anchor: "start", weight: 700 });
  txt(g, 30, 320, `The core is ${(100 * M.core_train / tot).toFixed(0)} % of training MACs against ${(100 * T.concept_module / T.compute_params).toFixed(0)} % of parameters: the honest cost of recurrence.`, { size: 12, anchor: "start" });
  // throughput table
  const ty = 356, cx = [30, 360, 520, 600, 740, 880];
  ["configuration", "fwd MACs eval / train", "core apps / fwd", "slots/s", "net slots 48 h", "slots / param"].forEach((h, i) => txt(g, cx[i], ty, h, { size: 11.5, weight: 700, anchor: "start" }));
  A.throughput.forEach((r, j) => r.forEach((c, i) => txt(g, cx[i], ty + 24 + j * 22, c, { size: 11.5, anchor: "start", weight: j === 0 ? 700 : 400, color: j >= 5 ? "#777" : j === 4 ? "#a0405a" : COL.ink })));
  return svg;
}

/* =========================================================================================== 12 gates & arms */
function f3dGates(A) {
  const svg = makeSvg(1180, 700), g = el("g", {}, svg);
  txt(g, 30, 28, "How v3 will be judged: early gates, milestone diagnostics, and a controlled arm ladder", { size: 14, weight: 700, anchor: "start" });
  const cx = [30, 280, 640, 900];
  ["gate", "pass", "reference", "fail → action"].forEach((h, i) => txt(g, cx[i], 60, h, { size: 12, weight: 700, anchor: "start" }));
  A.gates.forEach((r, j) => {
    const y = 86 + j * 30;
    el("rect", { x: 22, y: y - 13, width: 1136, height: 26, rx: 6, fill: j === 0 || j === 6 ? "#e9f6f3" : j % 2 ? "#fafbfc" : "#fff", stroke: "#eceff3" }, g);
    r.forEach((c, i) => txt(g, cx[i], y, c, { size: 11.5, anchor: "start", weight: i === 0 ? 700 : 400 }));
  });
  const ay = 440;
  txt(g, 30, ay, "Arms (sequential, N ≈ 2,400 steps, one shared per-step LR trace; matched steps and matched wall time)", { size: 13, weight: 700, anchor: "start" });
  const ax = [30, 220, 560, 660, 990];
  ["arm", "architecture change", "objective", "question", "runs when"].forEach((h, i) => txt(g, ax[i], ay + 26, h, { size: 11.5, weight: 700, anchor: "start" }));
  A.arms.forEach((r, j) => r.forEach((c, i) => txt(g, ax[i], ay + 50 + j * 22, c, { size: 11.5, anchor: "start", weight: i === 0 || (i === 4 && c === "recommended") ? 700 : 400, color: i === 4 && c === "recommended" ? COL.plan : COL.ink })));
  return svg;
}

const FIGS3D = [
  { id: "v3whole", t: "Whole model", fn: f3dOverview, cap: "Encoder (left, bottom → top) with one 80M Engram table read twice; concept module (middle) of 4 distinct full-attention layers with a shared FFN and a (t, Δ) table, looped M = 3 times from S₀ = 0, plus a training-only shortcut trajectory; confined decoder (right) without NextLat." },
  { id: "v3enc", t: "Encoder layers (d 256)", fn: f3dEncoder, cap: "v1's grammar at width 256. The headroom goes to FFN ratio 4 (256 → 1024) in every layer and 6 heads in the two F layers, where facts enter G." },
  { id: "v3engram", t: "Engram: one table, two reads", fn: f3dEngram, cap: "The n-gram key does not change with depth, so one 80M table is gathered once; read 2 after layer 4 has its own projections (W_V zero-init), a gate that reads L only, and no conv, keeping reach 63." },
  { id: "v3crc", t: "Concepts (d_c 256) & PQ", fn: f3dConcepts, cap: "Four tokens are mixed and pooled into one 256-dim concept on the sphere of radius 16; the PQ codebook follows d_c to 32 × 128 × 8, keeping v1's per-segment geometry." },
  { id: "v3core", t: "Concept layer & Φ", fn: f3dCore, cap: "One of 4 distinct layers: gated reads of evidence C and memory S with per-stream (t, Δ) gains, 6-head full attention, then the own FFN (736) in parallel with the shared FFN (1024). Seven gains per (pair, layer) come from the table." },
  { id: "v3traj", t: "Trajectories (M ≤ 3)", fn: f3dTraj, cap: "LoopFormer's Φ(h; t, Δt) on the grid of ⅓: the main trajectory (⅓, ⅓, ⅓) feeds the decoder; a shortcut with 1 or 2 passes is supervised by NCP and by consistency with the stop-gradient main endpoint." },
  { id: "v3table", t: "The (t, Δ) table", fn: f3dTable, cap: "Exactly six (t, Δ) pairs are reachable, so a zero-initialised lookup table replaces the paper's Fourier + MLP modulator at 3 % of its cost. Shortcut-only rows are sampled per micro-batch, so they get gradient on ≈ 99 % of optimizer steps, but with much less weight. Shading is illustrative, not trained values." },
  { id: "v3stag", t: "Why v1 idled · signatures", fn: f3dStag, cap: "v1's writes were nearly identical in direction every pass (norm grows linearly, angles → 0.11°). A fixed point is the leading hypothesis; the table is a tested remedy. Geometry is neither necessary nor sufficient for useful work, so it only explains results; the arm ladder decides." },
  { id: "v3dec", t: "Confined decoder & plan", fn: f3dDecoder, cap: "v1's confined decoder with FFN 1024 and 4 heads: each position sees 70 tokens plus its own target-aligned plan (u₃ and c_pred of the newest complete chunk). NextLat is removed." },
  { id: "v3loss", t: "Losses & gradient reach", fn: f3dLosses, cap: "v1's objective plus the two shortcut terms. Dots show which parts each term trains; the gain-free L_scons cannot shrink the output-norm gain the decoder uses." },
  { id: "v3budget", t: "Budget & throughput", fn: f3dBudget, cap: "18,986,871 compute parameters (50.8 / 25.7 / 23.5 %) against v1's 25.05M and v2's 18.83M. Throughput is an unvalidated two-point fit in slots/s (the counter includes PAD, 5.7 %); a measured pre-launch gate decides. 48 h ≈ 199–273M non-pad tokens, 0.75–1.03 epochs." },
  { id: "v3gates", t: "Gates & control arms", fn: f3dGates, cap: "E1 (recall onset) is primary. M1 compares the full recipe with one pass; recurrence itself is attributed by the ladder A1 / A3 / T3 / Full with paired statistics. Geometry, the gradient audit and E5 are diagnostics; interventions are fixed-checkpoint effects. No gate stops the run: failures notify, the owner decides." },
];
const FIG3D_KEY = {
  v3whole: "Three parts, 18.99M compute: a v1-grammar encoder, a 4-layer concept module looped at most 3 times, and a confined decoder.",
  v3enc: "Same encoder design, wider FFNs, 6 heads where global attention happens.", v3engram: "80M numbers of n-gram memory, one table, read at two depths.",
  v3crc: "Concepts at 256 dims; PQ geometry per segment unchanged.", v3core: "Evidence C and memory S, read with pass-dependent gains.",
  v3traj: "Every trajectory knows where it is (t) and how much it owes (Δ).", v3table: "Six pairs, a zero-initialised lookup, no weight decay.",
  v3stag: "The table is a hypothesis about v1's fixed point; the arms test it.", v3dec: "70 tokens plus its own plan, nothing more.",
  v3loss: "Five terms: v1's three and two for the shortcut.", v3budget: "Within 19M; about v1's speed.", v3gates: "Recall onset first; recurrence attributed by a controlled ladder.",
};
const FIG3D_3D = { v3whole: "all", v3enc: "enc", v3engram: "enc", v3crc: "enc", v3core: "core", v3traj: "core", v3table: "core", v3stag: "core", v3dec: "dec", v3loss: "all", v3budget: "all", v3gates: "all" };
