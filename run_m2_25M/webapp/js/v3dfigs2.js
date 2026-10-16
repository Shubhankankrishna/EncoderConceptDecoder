/* v3 (draft r2) extra 2D figures: the training step as implemented (run_v3_19M/code/train.py) and the measured throughput
   work (run_v3_19M/DECISIONS.md V3-005/006). Registered into FIGS3D before js/v3d.js renders the figure list. */
const PERF_V3 = {                                                  // measured on the M2 (MPS, fp32), see DECISIONS V3-006
  profile: [["encoder fwd", 17.5, "#d6e6f7", "#3b78c2"], ["core fwd (main + shortcut batched)", 15.2, "#d2efea", "#2e9c8c"],
            ["head + decoder fwd", 7.0, "#fbdcc8", "#e07b39"], ["losses", 2.1, "#fcdde3", "#d0667a"], ["backward", 56.7, "#ebe5fa", "#8f7cc0"]],
  micro_ms: 1154, opt_ms: 151,
  ab: [["block-local attention, block 64", "+6.1 %", "adopted", "paired 0.943, IQRs disjoint; driver 7.0 GiB"],
       ["block-local attention, block 128", "+4.9 %", "not needed", "below the 5 % rule; 64 is better"],
       ["step-wide mask memo", "+0.9 %", "noise", "bit-identical either way"],
       ["micro-batch 4 × 8", "thrash", "rejected", "swap 4.6 → 6.4 GB, warm-up not done in 8 min"],
       ["torch.compile, core layers only", "+17.7 %", "superseded", "on top of block64; k_norm kept eager (compiler bug)"],
       ["torch.compile, all layers", "+54.4 %", "adopted", "3,108 vs 1,961 tok/s; 25 recompiles, all in warm-up (122 s)"]],
  seq_note: "sequential 8-step runs were unusable: the same base config measured 2,059 then 1,478 tok/s (thermal)",
};

function f3dStep(A) {
  const svg = makeSvg(1180, 640), g = el("g", {}, svg);
  txt(g, 30, 26, "One optimizer step as implemented (32,768 tokens = 16 micro-batches × 2 rows × 1,024)", { size: 14, weight: 700, anchor: "start" });
  // 1. fetch
  frame(g, { x: 20, y: 46, w: 1140, h: 86, title: "① fetch the whole optimizer batch first", note: "loader prefetch = accum + 4" });
  for (let i = 0; i < 16; i++) box(g, { x: 34 + i * 58, y: 84, w: 50, h: 34, k: i % 2 ? "io" : "white", label: `mb ${i + 1}`, size: 10.5,
    tip: `<b>micro-batch ${i + 1}</b> · 2 rows × 1,024 tokens (v1's data order, seed 0)` });
  box(g, { x: 980, y: 80, w: 166, h: 42, k: "gate", label: "count N (ids, doc only)", sub: "targets · pairs · chunks", size: 11,
    tip: "Full-batch denominators: N_targets, N_pairs (nonempty current, complete target, same doc), N_complete, N_nonempty — before any forward, so every micro-batch divides by the same N (no /accum)." });
  // 2. per micro-batch
  frame(g, { x: 20, y: 148, w: 1140, h: 300, title: "② for each micro-batch: forward (main + shortcut together) → Σ λ·sum / N → backward", note: "shortcut drawn per micro-batch from its own RNG" });
  box(g, { x: 40, y: 200, w: 150, h: 60, k: "attn", label: "encoder", sub: "SSSFSSSF · 2 Engram reads", size: 12, tip: "One Engram gather feeds both reads; window-8 layers use block-local attention (block 64)." });
  arrow(g, [[190, 230], [228, 230]], { c: COL.G });
  box(g, { x: 230, y: 208, w: 70, h: 44, k: "conc", label: "C", sub: "256 × 256", size: 12 });
  arrow(g, [[300, 230], [338, 230]], { c: COL.plan });
  frame(g, { x: 340, y: 186, w: 430, h: 150, fill: "#f7fbfa", stroke: "#bfe3db" });
  txt(g, 555, 202, "concept core: rows [main ; shortcut] share each pass", { size: 11.5, weight: 700, color: "#2e7d6f" });
  const passes = [["pass 1", "(0,⅓) | (0,1) / (0,⅔) / reuse"], ["pass 2", "(⅓,⅓) | – / (⅔,⅓) / (⅓,⅔)"], ["pass 3", "(⅔,⅓) main only"]];
  passes.forEach(([p, s], i) => box(g, { x: 356 + i * 138, y: 218, w: 126, h: 54, k: i < 2 ? "conc" : "white", label: p, sub: s, size: 11,
    tip: `<b>${p}</b>: 4 layers; table row per block of rows. Shortcut 1 (p 0.5) rides pass 1; 2b (p 0.25) rides passes 1–2; 2a (p 0.25) reuses main S₁ and rides pass 2.` }));
  arrow(g, [[482, 245], [492, 245]]); arrow(g, [[620, 245], [630, 245]]);
  note(g, 356, 292, "≤ 12 sequential layer applications per micro-batch (18 before batching)", { size: 11 });
  note(g, 356, 310, "shortcut p: 1 → 0.5 · 2a → 0.25 · 2b → 0.25", { size: 11 });
  arrow(g, [[770, 230], [808, 230]], { c: COL.plan });
  box(g, { x: 810, y: 200, w: 150, h: 60, k: "ffn", label: "decoder (u₃, c_pred)", sub: "confined · tied head", size: 11.5 });
  const L = [["NTP(u₃)", "÷ N_targets", 1], ["NCP(u₃)", "÷ N_pairs", 0.1], ["NCP(u_s)", "÷ N_pairs", 0.05], ["L_scons", "÷ N_nonempty", 0.1], ["VQ", "÷ N_complete", 1]];
  L.forEach(([n, d, w], i) => box(g, { x: 40 + i * 190, y: 366, w: 176, h: 46, k: "loss", label: `${w} · ${n}`, sub: d, size: 11.5,
    tip: `${n}: summed over its valid items in this micro-batch, divided by the whole optimizer batch's count, weight ${w}.` }));
  box(g, { x: 990, y: 366, w: 156, h: 46, k: "gate", label: "backward()", sub: "one per micro-batch", size: 12 });
  note(g, 40, 432, "gradients accumulate across the 16 micro-batches = exactly the full-batch mean (tested against one 32-row batch)", { size: 11 });
  // 3. optimizer
  frame(g, { x: 20, y: 464, w: 1140, h: 160, title: "③ after the 16th micro-batch", note: "per-step state is in every checkpoint" });
  const S = [["clip ‖g‖ ≤ 1.0", "global, incl. codebook", "norm"], ["AdamW step", "(0.9, 0.95), wd 0.1\nno decay: table, biases, embeddings, codebook", "gate"],
             ["LR trace", "warm-up 300 steps → cosine on\nelapsed training time (48 h)", "mem"], ["every 100 steps", "dead codes re-seeded from\ncomplete chunks, Adam rows reset", "conc"],
             ["evaluations", "hourly + at all 23 of v1's eval steps\n24 h milestone (val) · 48 h final (test)", "io"]];
  S.forEach(([a, b, k], i) => box(g, { x: 40 + i * 224, y: 504, w: 206, h: 96, k, label: a, sub: b, size: 12 }));
  for (let i = 0; i < 4; i++) arrow(g, [[246 + i * 224, 552], [262 + i * 224, 552]]);
  return svg;
}

function f3dPerf(A) {
  const P = PERF_V3, svg = makeSvg(1180, 640), g = el("g", {}, svg);
  txt(g, 30, 26, "Throughput on the M2: where one micro-step's time goes, and what was changed (measured, MPS fp32)", { size: 14, weight: 700, anchor: "start" });
  // profile bar
  let x = 40; const W = 1100;
  P.profile.forEach(([n, pct, f, s]) => {
    const w = W * pct / 100; el("rect", { x, y: 60, width: w, height: 40, fill: f, stroke: s, rx: 4 }, g);
    txt(g, x + w / 2, 80, w > 120 ? `${n} ${pct} %` : `${pct} %`, { size: 11.5, weight: 600 });
    if (w <= 120) txt(g, x + w / 2, 114, n, { size: 10.5, color: "#5a5a60" });
    x += w;
  });
  note(g, 40, 136, `one micro-step ≈ ${P.micro_ms} ms (B2, eager, full attention) + optimizer ≈ ${P.opt_ms} ms per step · backward ≈ 1.35 × forward (low: the forward has non-matmul overhead)`, { size: 11.5 });
  // block-local attention illustration
  frame(g, { x: 20, y: 160, w: 560, h: 330, title: "block-local attention for the window-8 layers", note: "10 layers: 6 encoder S + 4 decoder" });
  const cell = 12, n = 16, ox = 50, oy = 214;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const need = j <= i && i - j < 3, inBlock = Math.floor(i / 4) * 4 - 2 <= j && j < Math.floor(i / 4) * 4 + 4;
    el("rect", { x: ox + j * cell, y: oy + i * cell, width: cell - 1, height: cell - 1, fill: need ? "#2e9c8c" : j <= i ? "#e8ecf1" : "#f7f8fa" }, g);
    el("rect", { x: ox + 230 + j * cell, y: oy + i * cell, width: cell - 1, height: cell - 1, fill: need ? "#2e9c8c" : inBlock && j >= 0 ? "#cfe9e3" : "#ffffff", stroke: inBlock ? "#9fd3c7" : "#f0f2f5", "stroke-width": 0.5 }, g);
  }
  txt(g, ox + 96, oy - 12, "v1: full T × T mask (1024²)", { size: 11.5, weight: 600 });
  txt(g, ox + 326, oy - 12, "v3: blocks of 64 + 7 keys", { size: 11.5, weight: 600 });
  note(g, 44, 424, "teal = keys a query may use (causal, last 8, same document). Scores computed: 1024 × 71 instead of 1024 × 1024.", { size: 11 });
  note(g, 44, 440, "Exact: masked entries are 0 after softmax in both forms (parity 1e-14 fwd, 8e-16 grad in float64; 4e-6 on MPS).", { size: 11 });
  // A/B table
  frame(g, { x: 600, y: 160, w: 560, h: 330, title: "interleaved A/B (configs alternate every step)", note: "adopt: ≥ 5 % paired + parity" });
  const cx = [616, 850, 930, 1010];
  ["change", "tok/s", "verdict", ""].forEach((h, i) => txt(g, cx[i], 200, h, { size: 11.5, weight: 700, anchor: "start" }));
  P.ab.forEach((r, j) => {
    const y = 226 + j * 40;
    el("rect", { x: 608, y: y - 14, width: 544, height: 34, rx: 6, fill: r[2] === "adopted" ? "#e9f6f3" : j % 2 ? "#fafbfc" : "#fff", stroke: "#eceff3" }, g);
    txt(g, cx[0], y, r[0], { size: 11.5, anchor: "start", weight: 600 });
    txt(g, cx[1], y, r[1], { size: 11.5, anchor: "start", weight: 700, color: r[1].startsWith("+") ? "#1f7a5c" : COL.grey });
    txt(g, cx[2], y, r[2], { size: 11, anchor: "start", color: r[2] === "adopted" ? "#1f7a5c" : COL.grey });
    txt(g, cx[0], y + 13, r[3], { size: 9.5, anchor: "start", color: COL.grey });
  });
  note(g, 44, 520, "Why interleaved: " + P.seq_note + "; alternating configurations every optimizer step (ABBA) makes drift hit both equally.", { size: 11.5 });
  note(g, 44, 540, "Earlier, exact and cheap: shortcut batched with the main passes (18 → 12 sequential core applications), one Engram gather, the C-half of every core read cached per forward.", { size: 11.5 });
  note(g, 44, 560, "Not pursued: fused own + shared W1/W3 (core forward is 15 % of the step; gain ≈ 2 %), bf16 (slower on MPS, V-006).", { size: 11.5 });
  return svg;
}

if (typeof FIGS3D !== "undefined") {
  const at = FIGS3D.findIndex(f => f.id === "v3budget");
  FIGS3D.splice(at + 1, 0,
    { id: "v3step", t: "One training step", fn: f3dStep, cap: "What the trainer actually does per optimizer step: fetch all 16 micro-batches and count the full-batch denominators; per micro-batch run the encoder, the core with the sampled shortcut batched into the main passes, the decoder, five losses each divided by its whole-batch count, and one backward; then clip, AdamW, the LR trace, codebook restarts every 100 steps, and the evaluation schedule." },
    { id: "v3perf", t: "Throughput work", fn: f3dPerf, cap: "Measured on the M2: the backward pass is 57 % of a micro-step; block-local attention for the ten window-8 layers computes 1024 × 71 scores instead of 1024² and is +6.1 % in a paired, interleaved A/B (sequential runs were unusable because of thermal drift). torch.compile of every layer (with the key-norm call kept eager to avoid an inductor-on-MPS gradient bug) adds +54 %: ≈ 3,100 tok/s, 2.4× the draft's estimate." });
  FIG3D_KEY.v3step = "Fetch all, count once, then 16 × (forward, weighted sum / N, backward), then one update.";
  FIG3D_KEY.v3perf = "Backward dominates; the window-8 attention was doing 14× too much work.";
  FIG3D_3D.v3step = "all"; FIG3D_3D.v3perf = "enc";
}
