/* Chapter 3 (S1) figures: ecdFlow, ncp, alpha, armsMatrix, s1results, s1tests */
(function () {
const { H, S, ui, loop, stepper, clamp, lerp, Svg } = G;
const ARM = { alpha0: "#2a78d6", delayed: "#eb6834", flat: "#1baf7a" };   // validated categorical slots 1 to 3 (dataviz skill)
const D = () => window.GUIDE_DATA || { s1: {}, s2: {} };
const pct = v => (v * 100).toFixed(1) + "%";

/* ------------------------------------------------------------ ecdFlow: the whole S1 network, one step at a time */
G.reg("ecdFlow", fig => {
  const sv = fig.svg(900, 440, { label: "Data flow through the encoder, pooling, concept stack, guess head and decoder" });
  const XR = 470, cxL = i => 80 + i * 100, cxR = i => XR + 80 + i * 100, tokX = (cx, j) => cx + (j - 1.5) * 24;
  const gs = {}; ["tok", "enc", "pool", "con", "head", "hint", "dec"].forEach(k => { gs[k] = sv.g({}); });
  const TY = 396, EY = [352, 334, 316], CY = [210, 192, 174, 156];
  sv.text(30, 10, "READS: encoder, then concept stack", { size: 12, mono: true, fill: "var(--ink3)" });
  sv.text(XR + 30, 10, "WRITES: decoder", { size: 12, mono: true, fill: "var(--ink3)" });
  sv.line(XR - 18, 10, XR - 18, 430, { stroke: "var(--line2)", sw: 1, dash: "4 5" });
  /* tokens (both towers) */
  const TOKS = ["A", "blue", "C", "SEP", "C", "orange", "F", "SEP", "blue", "TO", "C", "SEP", "orange", "TO", "F", "SEP"];
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      [cxL(i), cxR(i)].forEach(cx => { sv.rect(tokX(cx, j) - 10, TY - 10, 20, 20, { fill: "var(--panel)", stroke: "var(--line2)", sw: 1.2, rx: 4 }, gs.tok); });
    }
    sv.text(cxL(i), TY + 26, "chunk " + i, { size: 11, anchor: "middle", fill: "var(--ink3)" }, gs.tok);
    sv.text(cxR(i), TY + 26, "chunk " + i, { size: 11, anchor: "middle", fill: "var(--ink3)" }, gs.tok);
  }
  /* encoder */
  EY.forEach((y, k) => { sv.rect(30, y - 7, 400, 14, { fill: "var(--enc-s)", stroke: "var(--enc)", sw: 1.4, rx: 5 }, gs.enc); });
  sv.text(36, EY[2], "Encoder · 3 blocks · reads every token", { size: 11, weight: 600, base: "central" }, gs.enc);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) sv.line(tokX(cxL(i), j), TY - 12, tokX(cxL(i), j), EY[0] + 8, { stroke: "var(--enc)", sw: 1, opacity: .6 }, gs.enc);
  /* pooling */
  for (let i = 0; i < 4; i++) {
    const c = cxL(i); sv.path(`M ${c - 46} 304 L ${c + 46} 304 L ${c + 14} 254 L ${c - 14} 254 Z`, { fill: "var(--con-s)", stroke: "var(--con)", sw: 1.2 }, gs.pool);
    sv.rect(c - 17, 228, 34, 24, { fill: "var(--con)", stroke: "none", rx: 6, opacity: .9 }, gs.pool);
    sv.text(c, 240, "C" + i, { size: 11.5, weight: 700, anchor: "middle", fill: "var(--bg)" }, gs.pool);
  }
  sv.text(10, 279, "pool", { size: 11, fill: "var(--con)", weight: 600 }, gs.pool); sv.text(10, 292, "4 → 1", { size: 10.5, fill: "var(--ink3)" }, gs.pool);
  /* concept stack */
  CY.forEach(y => sv.rect(52, y - 7, 356, 14, { fill: "var(--con-s)", stroke: "var(--con)", sw: 1.4, rx: 5 }, gs.con));
  sv.text(58, CY[3], "Concept stack · 4 blocks · reads concepts only", { size: 11, weight: 600 }, gs.con);
  for (let i = 0; i < 4; i++) sv.line(cxL(i), 226, cxL(i), CY[0] + 8, { stroke: "var(--con)", sw: 1.4 }, gs.con);
  /* head + predicted concept */
  for (let i = 0; i < 4; i++) {
    const c = cxL(i); sv.line(c, CY[3] - 8, c, 128, { stroke: "var(--con)", sw: 1.4 }, gs.head);
    sv.rect(c - 38, 106, 76, 22, { fill: "var(--code-s)", stroke: "var(--code)", sw: 1.4, rx: 7 }, gs.head); sv.text(c, 117, "guess head", { size: 11, weight: 600, anchor: "middle" }, gs.head);
    sv.line(c, 106, c, 86, { stroke: "var(--code)", sw: 1.4 }, gs.head);
    sv.rect(c - 17, 62, 34, 24, { fill: "var(--code)", stroke: "none", rx: 6 }, gs.head); sv.text(c, 74, "ĉ" + i, { size: 11.5, weight: 700, anchor: "middle", fill: "var(--bg)" }, gs.head);
  }
  sv.text(10, 74, "guess of\nthe NEXT\nconcept", { size: 10, fill: "var(--code)", weight: 600 }, gs.head);
  /* hint arrows to the decoder */
  const hintPts = [];
  for (let i = 0; i < 4; i++) {
    const x1 = cxL(i), lvl = 46 - i * 0, tx = i < 3 ? cxR(i + 1) - 6 : cxR(3) + 44;
    const yb = 50 - i * 7, d = `M ${x1} 60 V ${yb} H ${tx} V 368`;
    sv.path(d, { stroke: "var(--code)", sw: i === 1 ? 2.4 : 1.3, opacity: i === 1 ? 1 : .55, arrow: "code", dash: i === 1 ? null : "5 4" }, gs.hint);
    hintPts.push([[x1, 60], [x1, yb], [tx, yb], [tx, 368]]);
    sv.circle(tx, 372, 8, { fill: "var(--panel)", stroke: "var(--code)", sw: 1.6 }, gs.hint); sv.text(tx, 372, "+", { size: 12, weight: 700, anchor: "middle", fill: "var(--code)" }, gs.hint);
  }
  sv.text(XR + 20, 70, "the guess from chunk j is added to\nthe writer's input for chunk j + 1", { size: 12, fill: "var(--code)", weight: 600 }, gs.hint);
  sv.path("M 432 330 C 450 330 452 392 482 392", { stroke: "var(--enc)", sw: 1.5, dash: "5 4", arrow: "enc", fill: "none" }, gs.hint);
  sv.text(436, 306, "token features H\n(real tokens still go in)", { size: 10.5, fill: "var(--enc)" }, gs.hint);
  /* decoder */
  EY.forEach(y => sv.rect(XR + 30, y - 7, 400, 14, { fill: "var(--dec-s)", stroke: "var(--dec)", sw: 1.4, rx: 5 }, gs.dec));
  sv.text(XR + 36, EY[2], "Decoder · 3 blocks · writes tokens", { size: 11, weight: 600 }, gs.dec);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    sv.line(tokX(cxR(i), j), EY[0] + 8, tokX(cxR(i), j), 360, { stroke: "var(--dec)", sw: 1, opacity: .5 }, gs.dec);
    sv.line(tokX(cxR(i), j), EY[2] - 8, tokX(cxR(i), j), 276, { stroke: "var(--dec)", sw: 1, opacity: .6 }, gs.dec);
    sv.rect(tokX(cxR(i), j) - 10, 252, 20, 22, { fill: "var(--dec-s)", stroke: "var(--dec)", sw: 1.2, rx: 4 }, gs.dec);
  }
  sv.text(XR + 36, 230, "the next tokens, one by one", { size: 12, fill: "var(--dec)", weight: 600 }, gs.dec);
  /* pulses */
  const poly = (a, b) => [[a, b[0], a, b[1]]];
  const PULSE = [
    [],
    Array.from({ length: 8 }, (_, k) => [[tokX(cxL(k >> 1), (k & 1) * 2 + 1), TY - 10], [tokX(cxL(k >> 1), (k & 1) * 2 + 1), EY[2] - 8]]),
    [0, 1, 2, 3].map(i => [[cxL(i), 306], [cxL(i), 254], [cxL(i), 232]]),
    [0, 1, 2, 3].map(i => [[cxL(i), 224], [cxL(i), CY[0] + 7], [cxL(i), CY[3] - 7]]),
    [0, 1, 2, 3].map(i => [[cxL(i), CY[3] - 8], [cxL(i), 128], [cxL(i), 88]]),
    hintPts,
    [0, 1, 2, 3].map(i => [[cxR(i), 360], [cxR(i), EY[2] - 8], [cxR(i), 276]]),
  ];
  const dots = Array.from({ length: 8 }, () => sv.circle(-20, -20, 4.2, { fill: "var(--accent)", stroke: "var(--panel)", sw: 1.5 }));
  const order = ["tok", "enc", "pool", "con", "head", "hint", "dec"];
  const STEPS = [
    { label: "1 Tokens", title: "The tokens go in.", text: "Sixteen tokens, four chunks. Here a chunk is a road in the puzzle or a hop in the answer. The left tower reads. The right tower writes." },
    { label: "2 Encoder", title: "The encoder reads every token.", text: "Three blocks of look back and think. Each token ends up as a richer vector that knows about the tokens before it." },
    { label: "3 Pool", title: "Pooling squeezes four vectors into one concept.", text: "The four vectors of a chunk are averaged. Sixteen positions become four concepts: one summary note per chunk." },
    { label: "4 Concept stack", title: "The concept stack thinks about steps.", text: "Four blocks that read only the concepts. There are four times fewer positions here than tokens, so this part is cheap, and it reasons one step at a time." },
    { label: "5 Guess", title: "A head guesses the next concept.", text: "From the concept stack's output, a small head produces ĉ, a guess of the next chunk's concept. This guess is the thing S1 trains with the NCP loss." },
    { label: "6 Hint", title: "The guess is handed to the writer, one chunk late.", text: "The guess made at chunk j is added to the decoder's input for chunk j + 1, so it can steer what gets written next. The decoder still reads the real tokens too." },
    { label: "7 Write", title: "The decoder writes the next tokens.", text: "Three blocks of look back and think, now with the hint added. The output is the probability of every token, and one is picked." },
  ];
  let stp = 0;
  stepper(fig, STEPS, i => { stp = i; order.forEach((k, ix) => { gs[k].style.transition = "opacity .35s"; gs[k].style.opacity = ix < i ? .7 : ix === i ? 1 : .1; }); }, { ms: 4200 });
  const along = (pts, u) => { let L = 0; const seg = []; for (let k = 1; k < pts.length; k++) { const d = Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]); seg.push(d); L += d; } let a = u * L; for (let k = 0; k < seg.length; k++) { if (a <= seg[k] || k === seg.length - 1) { const f = seg[k] ? Math.min(1, a / seg[k]) : 0; return [lerp(pts[k][0], pts[k + 1][0], f), lerp(pts[k][1], pts[k + 1][1], f)]; } a -= seg[k]; } return pts[pts.length - 1]; };
  loop(fig, t => { const P = PULSE[stp] || []; dots.forEach((d, k) => { if (!P[k]) { d.setAttribute("cx", -20); return; } const [x, y] = along(P[k], (t * 0.45 + k * 0.13) % 1); d.setAttribute("cx", x); d.setAttribute("cy", y); }); });
});

/* ------------------------------------------------------------ ncp: guessing a vector by blending stickers */
G.reg("ncp", fig => {
  const W = 900, Hh = 400, sv = fig.svg(W, Hh, { label: "A guessed concept is a blend of codebook entries" });
  const PX = 20, PY = 40, PW = 520, PH = 330;
  const E = [[70, 60], [210, 40], [370, 70], [480, 140], [430, 270], [270, 290], [110, 250], [60, 150]];   // eight codebook entries ("stickers") in a 2-D picture of a 16-D space
  let belief = [330, 150], truth = [200, 190], sure = 70, mode = "belief";
  const seg = ui.seg([["belief", "Click moves the model's belief"], ["truth", "Click moves the true next concept"]], mode, v => { mode = v; }, "click mode");
  const sl = ui.range("sureness", 20, 160, 2, 70, v => { sure = v; draw(); }, v => v < 50 ? "very sure" : v < 100 ? "fairly sure" : "unsure");
  const r = ui.readout(); fig.$ui.append(seg, sl, r);
  sv.rect(PX, PY, PW, PH, { fill: "var(--panel2)", stroke: "var(--line2)", rx: 12 });
  sv.text(PX + 8, PY - 14, "A picture of concept space (really 16 numbers per slot; shown in 2-D)", { size: 12, fill: "var(--ink3)" });
  const hull = sv.path("", { stroke: "var(--ink3)", sw: 1.2, dash: "5 4", fill: "var(--bg2)", opacity: .7 }), dyn = sv.g({}), bars = sv.g({});
  hull.setAttribute("d", "M " + [0, 1, 2, 3, 4, 5, 6, 7].map(i => `${PX + E[i][0]} ${PY + E[i][1]}`).join(" L ") + " Z");
  const hit = sv.rect(PX, PY, PW, PH, { fill: "transparent", stroke: "none", rx: 12, attrs: { style: "cursor:crosshair" } });
  hit.addEventListener("click", e => { const b = sv.svg.getBoundingClientRect(), x = (e.clientX - b.left) / b.width * W - PX, y = (e.clientY - b.top) / b.height * Hh - PY; const p = [clamp(x, 5, PW - 5), clamp(y, 5, PH - 5)]; if (mode === "belief") belief = p; else truth = p; draw(); });
  function probs() { const a = E.map(e => Math.exp(-((e[0] - belief[0]) ** 2 + (e[1] - belief[1]) ** 2) / (2 * sure * sure))); const z = a.reduce((x, y) => x + y, 0); return a.map(v => v / z); }
  function draw() {
    dyn.innerHTML = ""; bars.innerHTML = ""; const p = probs();
    const g = [0, 0]; E.forEach((e, i) => { g[0] += p[i] * e[0]; g[1] += p[i] * e[1]; });
    E.forEach((e, i) => { S("line", { x1: PX + belief[0], y1: PY + belief[1], x2: PX + e[0], y2: PY + e[1], stroke: "var(--enc)", "stroke-width": 0.5 + 6 * p[i], opacity: .15 + .6 * p[i], "stroke-linecap": "round" }, dyn); });
    E.forEach((e, i) => { S("circle", { cx: PX + e[0], cy: PY + e[1], r: 9, fill: "var(--code-s)", stroke: "var(--code)", "stroke-width": 1.8 }, dyn); const t = S("text", { x: PX + e[0], y: PY + e[1] + 0.5, "text-anchor": "middle", "dominant-baseline": "central", "font-size": 10, "font-weight": 700, fill: "var(--ink)" }, dyn); t.textContent = i + 1; });
    S("circle", { cx: PX + belief[0], cy: PY + belief[1], r: 6, fill: "var(--enc)", stroke: "var(--panel)", "stroke-width": 2 }, dyn);
    S("path", { d: `M ${PX + truth[0]} ${PY + truth[1] - 12} l 4 9 l 9 1 l -7 6 l 2 9 l -8 -5 l -8 5 l 2 -9 l -7 -6 l 9 -1 z`, fill: "var(--sel)", stroke: "var(--panel)", "stroke-width": 1.6 }, dyn);
    S("line", { x1: PX + g[0], y1: PY + g[1], x2: PX + truth[0], y2: PY + truth[1], stroke: "var(--loss)", "stroke-width": 2, "stroke-dasharray": "4 3" }, dyn);
    S("circle", { cx: PX + g[0], cy: PY + g[1], r: 8, fill: "var(--loss)", stroke: "var(--panel)", "stroke-width": 2.5 }, dyn);
    const t1 = S("text", { x: PX + g[0] + 12, y: PY + g[1] - 12, "font-size": 12, "font-weight": 650, fill: "var(--loss)" }, dyn); t1.textContent = "the guess";
    const err = ((g[0] - truth[0]) ** 2 + (g[1] - truth[1]) ** 2) / 1000;
    /* probability bars */
    const BX = 580; sv.text(BX, 28, "How much weight each sticker gets", { size: 12.5, weight: 650 }, bars);
    p.forEach((v, i) => { const y = 48 + i * 36; sv.text(BX, y + 11, "sticker " + (i + 1), { size: 12, fill: "var(--ink2)" }, bars); sv.rect(BX + 70, y, 220, 22, { fill: "var(--bg2)", stroke: "none", rx: 5 }, bars); sv.rect(BX + 70, y, Math.max(2, 220 * v), 22, { fill: "var(--code)", stroke: "none", rx: 5 }, bars); sv.text(BX + 76 + 220 * v, y + 11, (v * 100).toFixed(0) + "%", { size: 11.5, mono: true, fill: "var(--ink2)" }, bars); });
    const top = Math.max(...p);
    r.set(`loss (squared distance) <b>${err.toFixed(2)}</b> · ${top < 0.6 ? "the guess is a <b>blend</b>, so it sits between stickers" : "the model is sure, so the guess sits on one sticker"}`);
    fig.$live.hidden = false; fig.$live.innerHTML = "<b>The blue dot</b> is where the model believes the next concept is. It gives each sticker a weight by closeness. <b>The red dot</b> is the weighted average of the stickers. <b>The orange star</b> is the true next concept. The training loss is the distance from red to orange.";
  }
  draw();
});

/* ------------------------------------------------------------ alpha: the delayed schedule, with the real logged NCP loss */
G.reg("alpha", fig => {
  const W = 900, Hh = 462, sv = fig.svg(W, Hh, { label: "NCP weight over training and the logged NCP loss of a delayed run and a no-NCP run" });
  const cur = D().s1.curves || {}, X0 = 70, X1 = 860, MAXU = 24000;
  const px = u => X0 + (X1 - X0) * u / MAXU;
  let sched = "delayed";
  const alphaOf = u => sched === "off" ? 0 : sched === "const" ? 1 : u < 6000 ? 0 : u < 8000 ? 0.1 * (u - 6000) / 2000 : 0.1;
  /* top: schedule */
  const T0 = 36, T1 = 118; let amax = 0.1; const pyA = a => T1 - (T1 - T0) * a / amax;
  const topG = sv.g({});
  function drawTop() {
    topG.innerHTML = ""; amax = sched === "const" ? 1 : 0.1;
    sv.text(X0, 16, "NCP weight α (how much the guessing task counts)", { size: 13, weight: 650 }, topG);
    (sched === "const" ? [0, 0.5, 1] : [0, 0.05, 0.1]).forEach(a => { sv.line(X0, pyA(a), X1, pyA(a), { stroke: "var(--line)", sw: 1 }, topG); sv.text(X0 - 8, pyA(a), String(a), { size: 11, mono: true, anchor: "end", fill: "var(--ink3)" }, topG); });
    if (sched === "delayed") { sv.rect(px(6000), T0 - 6, px(8000) - px(6000), T1 - T0 + 12, { fill: "var(--code-s)", stroke: "none", rx: 0, opacity: .8 }, topG); sv.path(`M ${px(0)} ${pyA(0)} H ${px(6000)} L ${px(8000)} ${pyA(0.1)} H ${px(MAXU)}`, { stroke: ARM.delayed, sw: 2.6 }, topG); sv.text(px(3000), pyA(0) - 12, "off", { size: 12, anchor: "middle", fill: "var(--ink2)" }, topG); sv.text(px(7000), T0 - 12, "ramp", { size: 12, anchor: "middle", fill: "var(--code)", weight: 600 }, topG); sv.text(px(16000), pyA(0.1) - 12, "on at 0.1", { size: 12, anchor: "middle", fill: "var(--ink2)" }, topG); }
    else if (sched === "const") { sv.path(`M ${px(0)} ${pyA(1)} H ${px(MAXU)}`, { stroke: "#7a5cc7", sw: 2.6 }, topG); sv.text(px(12000), pyA(1) + 16, "on at 1.0 from update 0 (A_const and A_const_hncp)", { size: 12, anchor: "middle", fill: "var(--ink2)" }, topG); }
    else { sv.path(`M ${px(0)} ${pyA(0)} H ${px(MAXU)}`, { stroke: ARM.alpha0, sw: 2.6 }, topG); sv.text(px(12000), pyA(0) - 12, "off for all 24,000 updates (A_alpha0)", { size: 12, anchor: "middle", fill: "var(--ink2)" }, topG); }
  }
  /* bottom: real curves */
  const B0 = 190, B1 = 380, lo = 0.4, hi = 1.05; const pyN = v => B1 - (B1 - B0) * (v - lo) / (hi - lo);
  sv.text(X0, 168, "The NCP loss that was actually logged (real runs, seed 902 solid, seed 901 dashed)", { size: 13, weight: 650 });
  [0.4, 0.6, 0.8, 1.0].forEach(v => { sv.line(X0, pyN(v), X1, pyN(v), { stroke: "var(--line)", sw: 1 }); sv.text(X0 - 8, pyN(v), v.toFixed(1), { size: 11, mono: true, anchor: "end", fill: "var(--ink3)" }); });
  [0, 6000, 12000, 18000, 24000].forEach(u => sv.text(px(u), B1 + 18, u === 0 ? "0" : (u / 1000) + "k", { size: 11, mono: true, anchor: "middle", fill: "var(--ink3)" }));
  sv.text((X0 + X1) / 2, B1 + 38, "training update", { size: 12, anchor: "middle", fill: "var(--ink2)" });
  sv.rect(px(6000), B0 - 4, px(8000) - px(6000), B1 - B0 + 8, { fill: "var(--code-s)", stroke: "none", rx: 0, opacity: .6 });
  const lines = [["delayed_s902", ARM.delayed, null, "A_delayed s902"], ["delayed_s901", ARM.delayed, "6 4", "A_delayed s901"], ["alpha0_s902", ARM.alpha0, null, "A_alpha0 s902"], ["alpha0_s901", ARM.alpha0, "6 4", "A_alpha0 s901"]];
  lines.forEach(([k, c, dash]) => { const pts = cur[k] || []; if (!pts.length) return; sv.path("M" + pts.map(p => `${px(p.step).toFixed(1)} ${pyN(clamp(p.ncp, lo, hi)).toFixed(1)}`).join(" L"), { stroke: c, sw: 2.2, dash }); });
  lines.slice(0, 3).forEach(([k, c, dash, name], i) => { if (i === 1) return; sv.line(X0 + 8 + (i === 2 ? 250 : 0), 446, X0 + 36 + (i === 2 ? 250 : 0), 446, { stroke: c, sw: 2.4, dash }); sv.text(X0 + 42 + (i === 2 ? 250 : 0), 446, i === 0 ? "trained on NCP (A_delayed)" : "never trained on NCP (A_alpha0)", { size: 11.5, fill: "var(--ink2)" }); });
  const mk = sv.line(0, 36, 0, B1, { stroke: "var(--ink)", sw: 1.2, dash: "3 3" }), mk2 = sv.circle(0, 0, 0, {}); const lab = sv.text(0, 150, "", { size: 12, mono: true, halo: true });
  let u = 0, run = null;
  const sl = ui.range("update", 0, 24000, 200, 0, v => { u = v; fig.playing = false; pb.textContent = "Play"; draw(); }, v => v.toLocaleString("en-US"));
  const pb = ui.btn("Play", () => { fig.playing = !fig.playing; pb.textContent = fig.playing ? "Pause" : "Play"; if (fig.playing) G.kick(); });
  const r = ui.readout(); const sch = ui.seg([["delayed", "Delayed (main arm)"], ["const", "Constant 1.0 (V6-style)"], ["off", "Off (A_alpha0)"]], "delayed", v => { sched = v; drawTop(); draw(); }, "schedule"); fig.$ui.append(sch, sl, pb, r); fig.playing = !G.REDUCED; pb.textContent = fig.playing ? "Pause" : "Play";
  const at = (k, uu) => { const pts = cur[k] || []; let best = null; pts.forEach(p => { if (!best || Math.abs(p.step - uu) < Math.abs(best.step - uu)) best = p; }); return best ? best.ncp : null; };
  function draw() {
    const x = px(u); mk.setAttribute("x1", x); mk.setAttribute("x2", x); lab.setAttribute("x", Math.min(x + 6, X1 - 150)); lab.textContent = "update " + Math.round(u).toLocaleString("en-US");
    const a = alphaOf(u), d = at("delayed_s902", Math.max(u, 200)), z = at("alpha0_s902", Math.max(u, 200));
    r.set(`α = <b>${a.toFixed(sched === "const" ? 1 : 3)}</b>${d != null ? ` · NCP loss: A_delayed <b>${d.toFixed(2)}</b>, A_alpha0 <b>${z.toFixed(2)}</b>` : ""}`);
  }
  loop(fig, (t, dt) => { u += dt * 2400; if (u > MAXU) u = 0; sl.set(u); draw(); });
  drawTop(); draw();
});

/* ------------------------------------------------------------ armsMatrix: which switches differ between arms */
G.reg("armsMatrix", fig => {
  const COLS = [["hier", "Reads in chunks"], ["ncp", "Next-concept loss weight"], ["codes", "Codes K"], ["mask", "Prompt attention"], ["len", "Updates"], ["seeds", "Seeds"]];
  const ROWS = [
    { id: "A_delayed", tier: "mand", hier: "yes", ncp: "off until 6,000, ramp to 0.1 by 8,000", codes: "none", mask: "backwards only", len: "24,000", seeds: "901, 902", note: "The anchor. Also the teacher for S2." },
    { id: "A_alpha0", tier: "mand", hier: "yes", ncp: "0 always", codes: "none", mask: "backwards only", len: "24,000", seeds: "901, 902", note: "Same network, guessing task never counted." },
    { id: "A_flat", tier: "mand", hier: "no", ncp: "0 always", codes: "none", mask: "backwards only", len: "24,000", seeds: "901, 902", note: "Plain 10-block model, same size." },
    { id: "A_delayed_prefixlm", tier: "full", hier: "yes", ncp: "off until 6,000, ramp to 0.1 by 8,000", codes: "none", mask: "prompt read both ways", len: "24,000", seeds: "2 new", note: "Is it better to read the puzzle in both directions?" },
    { id: "A_const", tier: "full", hier: "yes", ncp: "1.0 from update 0", codes: "none", mask: "backwards only", len: "24,000", seeds: "1 new", note: "Recreates the old V6 setting where NCP went badly." },
    { id: "A_const_hncp", tier: "full", hier: "yes", ncp: "1.0 from update 0", codes: "4", mask: "backwards only", len: "24,000", seeds: "1 new", note: "Puts S2's several-guesses idea inside S1." },
    { id: "A_delayed_x4", tier: "full", hier: "yes", ncp: "off until 6,000, ramp to 0.1 by 8,000", codes: "none", mask: "backwards only", len: "96,000", seeds: "1 new", note: "Four times longer. Does NCP pay off later?" },
    { id: "A_alpha0_x4", tier: "full", hier: "yes", ncp: "0 always", codes: "none", mask: "backwards only", len: "96,000", seeds: "1 new", note: "Matched control for the long run." },
  ];
  const PAIRS = [
    ["A_delayed", "A_alpha0", "Does the next-concept guessing help?"], ["A_alpha0", "A_flat", "Does the chunk hierarchy help?"], ["A_const", "A_delayed", "Does the timing of NCP matter?"],
    ["A_const_hncp", "A_const", "Do several codes fix the old failure?"], ["A_delayed_prefixlm", "A_delayed", "Does reading the prompt both ways help?"], ["A_delayed_x4", "A_delayed", "Does NCP pay off with longer training?"],
  ];
  let pi = 0, show = "all";
  const key = (r, c) => c === "ncp" && r.id === "A_flat" ? "0 always" : r[c];
  const wrap = H("div", { class: "tablewrap", style: "margin:4px 12px 8px" }), cap = H("div", { class: "fig-cap-live" });
  fig.$ui.append(ui.seg(PAIRS.map((p, i) => [String(i), p[2].replace("?", "")]), "0", v => { pi = +v; draw(); }, "comparison"));
  fig.$stage.append(wrap); fig.$live.hidden = false;
  function draw() {
    const [a, b, q] = PAIRS[pi], ra = ROWS.find(r => r.id === a), rb = ROWS.find(r => r.id === b), diff = COLS.filter(([c]) => c !== "seeds" && key(ra, c) !== key(rb, c)).map(c => c[0]);
    const t = H("table", { class: "t" }, H("thead", {}, H("tr", {}, H("th", {}, "Arm"), COLS.map(([, l]) => H("th", {}, l)), H("th", {}, "Status"))));
    const tb = H("tbody"); ROWS.forEach(r => {
      const on = r.id === a || r.id === b; const tr = H("tr", { style: on ? "background:var(--accent-s)" : (show === "mand" && r.tier !== "mand" ? "opacity:.5" : "") }, H("td", {}, H("b", { class: "mono" }, r.id)), COLS.map(([c]) => H("td", { style: on && diff.includes(c) ? "background:var(--code-s);font-weight:650" : "" }, r[c])), H("td", {}, H("span", { class: "chip " + (r.tier === "mand" ? "mand" : "full") }, r.tier === "mand" ? "mandatory" : "full column")));
      tb.append(tr);
    }); t.append(tb); wrap.innerHTML = ""; wrap.append(t);
    const names = diff.map(c => COLS.find(x => x[0] === c)[1].toLowerCase());
    fig.$live.innerHTML = `<b>${q}</b> Compare <b class="mono">${a}</b> with <b class="mono">${b}</b>. ${diff.length === 1 ? `Exactly one thing differs: <b>${names[0]}</b>. Any difference in results can be traced to it.` : diff.length === 0 ? "Nothing differs." : `These differ: <b>${names.join(", ")}</b>.`} <span class="muted">The shaded cells are the switches that differ.</span>`;
  }
  draw();
});

/* ------------------------------------------------------------ maskDemo: causal versus PrefixLM attention */
G.reg("maskDemo", fig => {
  const W = 900, Hh = 410, sv = fig.svg(W, Hh, { label: "Which tokens each token may look at, causal versus PrefixLM" });
  let pre = false; fig.$ui.append(ui.seg([["c", "Backwards only (every other arm)"], ["p", "PrefixLM (A_delayed_prefixlm)"]], "c", v => { pre = v === "p"; draw(); }, "attention mask"));
  const N = 16, CS = 21, X0 = 110, Y0 = 56, dyn = sv.g({});
  function draw() {
    dyn.innerHTML = "";
    sv.text(X0, 20, "token being looked at (column)", { size: 12, fill: "var(--ink3)" }, dyn); sv.text(14, Y0 + N * CS / 2, "token that\nis looking\n(row)", { size: 12, fill: "var(--ink3)" }, dyn);
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) { const ok = c <= r || (pre && r < 8 && c < 8), extra = pre && r < 8 && c < 8 && c > r; sv.rect(X0 + c * CS, Y0 + r * CS, CS - 1.5, CS - 1.5, { fill: extra ? "var(--code)" : ok ? "var(--con)" : "var(--bg2)", stroke: "none", rx: 3, opacity: ok ? .9 : 1 }, dyn); }
    for (let k = 0; k <= 4; k++) { sv.line(X0 + k * 4 * CS - .8, Y0 - 4, X0 + k * 4 * CS - .8, Y0 + N * CS, { stroke: "var(--ink3)", sw: 1, opacity: .5 }, dyn); sv.line(X0 - 4, Y0 + k * 4 * CS - .8, X0 + N * CS, Y0 + k * 4 * CS - .8, { stroke: "var(--ink3)", sw: 1, opacity: .5 }, dyn); }
    ["road", "road", "hop 1", "hop 2"].forEach((t, k) => sv.text(X0 + k * 4 * CS + 2 * CS, Y0 + N * CS + 16, t, { size: 11, anchor: "middle", fill: "var(--ink3)" }, dyn));
    sv.rect(X0 - 8, Y0 - 2, 8 * CS + 8, 8 * CS + 4, { fill: "none", stroke: "var(--enc)", sw: 2, rx: 4, dash: "5 3" }, dyn); sv.text(X0 + 4 * CS, Y0 + 8 * CS + 14 + 0, "", { size: 1 }, dyn);
    const RX = 500; sv.rect(RX, 60, 14, 14, { fill: "var(--con)", stroke: "none", rx: 3 }, dyn); sv.text(RX + 24, 67, "this token may look at that earlier token", { size: 12.5 }, dyn);
    sv.rect(RX, 90, 14, 14, { fill: pre ? "var(--code)" : "var(--bg2)", stroke: "none", rx: 3 }, dyn); sv.text(RX + 24, 97, pre ? "new: a prompt token may also look ahead inside the prompt" : "(PrefixLM adds cells here)", { size: 12.5 }, dyn);
    sv.rect(RX, 120, 14, 14, { fill: "var(--bg2)", stroke: "none", rx: 3 }, dyn); sv.text(RX + 24, 127, "blocked: the future", { size: 12.5 }, dyn);
    sv.rect(RX, 150, 14, 14, { fill: "none", stroke: "var(--enc)", sw: 2, rx: 3, dash: "3 2" }, dyn); sv.text(RX + 24, 157, "the prompt: two roads, 8 tokens", { size: 12.5 }, dyn);
    sv.text(RX, 214, pre ? "Every prompt chunk now sees the whole puzzle,\nso its concept can summarise all of it.\nThe answer rows are unchanged: the answer\nmay never peek ahead." : "Every token sees only itself and earlier tokens.\nThat is the normal setup for a language model,\nand it is what the answer must obey.", { size: 12.5, fill: "var(--ink2)" }, dyn);
    sv.text(RX, 300, "Text rows always stay backwards-only, so text\nloss keeps its meaning. A test checks that NCP\nstill scores no target that lies inside the prompt.", { size: 12, fill: "var(--ink3)" }, dyn);
    fig.$live.hidden = false; fig.$live.innerHTML = pre ? "<b>PrefixLM</b> is borrowed from a recent paper where it added a few points on question answering at 1 billion parameters. Nobody knows how much of that carries over to four-token puzzle chunks at 2.4 million, so it is one separate S1 arm and not a default." : "<b>Backwards only</b> is how every arm trains by default. Switch to PrefixLM to see the one change made in the full-column arm.";
  }
  draw();
});

/* ------------------------------------------------------------ s1results: sealed results, per arm and seed */
G.reg("s1results", fig => {
  const T = D().s1.table || {}, W = 900, Hh = 430, sv = fig.svg(W, Hh, { label: "Sealed S1 results per arm and seed" });
  const METRICS = { standard: "Standard puzzles solved", hard10: "Hard10 puzzles solved", lookup: "Lookup solved", nll_edu: "Text loss, web text (lower is better)", nll_ts: "Text loss, stories (lower is better)" };
  const ARMS = [["alpha0", "A_alpha0"], ["delayed", "A_delayed"], ["flat", "A_flat"]];
  let metric = "standard";
  fig.$ui.append(ui.seg(Object.entries(METRICS).map(([k, v]) => [k, v.split(" (")[0]]), metric, v => { metric = v; draw(); }, "metric"));
  function draw() {
    sv.clear(); const isText = metric.startsWith("nll"), X0 = 80, X1 = 860, Y0 = 350, Y1 = 50;
    let lo = 0, hi = 1; if (isText) { const vals = []; Object.keys(T).forEach(k => vals.push(T[k][metric])); lo = Math.floor(Math.min(...vals) * 50) / 50 - 0.02; hi = Math.ceil(Math.max(...vals) * 50) / 50 + 0.02; }
    const py = v => Y0 - (Y0 - Y1) * (v - lo) / (hi - lo);
    sv.text(X0, 24, METRICS[metric], { size: 14, weight: 650 });
    const ticks = isText ? 5 : 5; for (let i = 0; i <= ticks; i++) { const v = lo + (hi - lo) * i / ticks; sv.line(X0, py(v), X1, py(v), { stroke: "var(--line)", sw: 1 }); sv.text(X0 - 8, py(v), isText ? v.toFixed(2) : Math.round(v * 100) + "%", { size: 11.5, mono: true, anchor: "end", fill: "var(--ink3)" }); }
    if (metric === "standard") { sv.line(X0, py(0.8), X1, py(0.8), { stroke: "var(--good)", sw: 1.4, dash: "6 4" }); }
    const gw = (X1 - X0) / 3;
    ARMS.forEach(([a, name], gi) => {
      const gx = X0 + gi * gw + gw / 2; sv.text(gx, Y0 + 36, name, { size: 13.5, weight: 650, anchor: "middle", mono: true });
      [["901", -58, .5], ["902", 6, 1]].forEach(([s, dx, op]) => {
        const k = `A_${a}_s${s}`, v = (T[k] || {})[metric]; if (v == null) return;
        const x = gx + dx; if (!isText) { const h = Y0 - py(v); sv.rect(x, py(v), 52, Math.max(h, 1.5), { fill: ARM[a], stroke: ARM[a], rx: 5, opacity: op, sw: 1.5, dash: s === "901" ? "5 3" : null }); }
        else sv.circle(x + 26, py(v), 9, { fill: ARM[a], stroke: "var(--panel)", sw: 2.5, opacity: op });
        sv.text(x + 26, isText ? py(v) - 17 : py(v) - 10, isText ? v.toFixed(3) : (v * 100).toFixed(1) + "%", { size: 12.5, weight: 650, anchor: "middle", mono: true });
        sv.text(x + 26, Y0 + 16, "seed " + s, { size: 11, anchor: "middle", fill: "var(--ink3)" });
      });
      const h = T[`A_${a}_s811_historical`]; if (h && h[metric] != null) { sv.path(`M ${gx + 80} ${py(h[metric]) - 8} l 8 8 l -8 8 l -8 -8 z`, { fill: "var(--panel)", stroke: "var(--ink)", sw: 1.8 }); sv.text(gx + 80, py(h[metric]) - 17, isText ? h[metric].toFixed(3) : (h[metric] * 100).toFixed(0) + "%", { size: 11, anchor: "middle", mono: true, fill: "var(--ink2)" }); }
    });
    sv.text(X0, Hh - 14, "light dashed bar = seed 901 · solid bar = seed 902 · diamond = V8's seed 811 (reference only, not counted in any test)" + (metric === "standard" ? " · green dashed line = G1 teacher bar, 80% (judged on dev puzzles)" : ""), { size: 11.5, fill: "var(--ink3)" });
    fig.$live.hidden = false; fig.$live.innerHTML = metric === "standard" ? "<b>Look at the two seeds of every arm.</b> The recipe is identical inside each group of bars, yet seed 901 and seed 902 end up far apart. Flat scored 0.0% on seed 902." : isText ? "<b>The plain model is a little better at text.</b> Lower is better. The flat model has the lowest loss on both seeds, on both kinds of text." : "<b>The same pattern shows up on harder puzzles.</b> Scores are lower, and the seeds still disagree.";
  }
  draw();
});

/* ------------------------------------------------------------ s1tests: the three pre-registered hypotheses */
G.reg("s1tests", fig => {
  const t = D().s1.tests; if (!t) { fig.$stage.textContent = "Results are not available in this build."; return; }
  const P = x => (x * 100).toFixed(1);
  const CARDS = [
    ["H3", "A_delayed beats A_flat on standard puzzles", `Pooled over both seeds: ${t.H3.pooled.delta > 0 ? "+" : ""}${P(t.H3.pooled.delta)} points, 95% interval ${P(t.H3.pooled.interval[0])} to ${P(t.H3.pooled.interval[1])}.`, "Supported", `Mostly because the flat model collapsed: it scored 13.7% on seed 901 and 0.0% on seed 902. A_delayed itself scored 16.8% and 83.2%.`, t.H3.supported],
    ["H4", "A_flat has lower text loss than A_delayed", `Flat is lower by ${Math.abs(t.H4.pooled.delta).toFixed(3)} nats, interval ${Math.abs(t.H4.pooled.interval[1]).toFixed(3)} to ${Math.abs(t.H4.pooled.interval[0]).toFixed(3)}.`, "Supported", "On both seeds and both text types, the plain model reads text slightly better. The chunk model gives something up on text.", t.H4.supported],
    ["H5", "A_delayed beats A_alpha0 on standard puzzles", `Pooled: ${P(t.H5.pooled.delta)} points, interval ${P(t.H5.pooled.interval[0])} to ${P(t.H5.pooled.interval[1])}. Seed 902 alone: +1.0 point, interval −0.4 to +2.4.`, "Not supported", "On seed 902, delayed and no-NCP are statistically tied. On seed 901 the delayed arm did far worse. The data do not show that the next-concept task helped.", t.H5.supported],
  ];
  const wrap = H("div", { class: "gates", style: "padding:6px 14px 12px" });
  CARDS.forEach(([id, claim, nums, verdict, why, ok]) => wrap.append(H("div", { class: "gate" }, H("div", { class: "id" }, id + " · prediction made before any run"), H("h4", {}, claim), H("p", {}, nums), H("div", { class: "chips", style: "margin:0 0 6px" }, H("span", { class: "chip " + (ok ? "done" : "full") }, verdict)), H("p", {}, why))));
  fig.$stage.append(wrap);
});
})();
