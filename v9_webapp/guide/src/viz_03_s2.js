/* Chapter 4 (S2) figures: meanModes, hncpFlow, selector, closedLoop, deadCodes, codesDecode, s2status */
(function () {
const { H, S, ui, loop, stepper, clamp, lerp, rng, gauss, along } = G;
const D = () => window.GUIDE_DATA || { s1: {}, s2: {} };
const ROADC = ["#2a78d6", "#eb6834", "#1baf7a"];

/* ------------------------------------------------------------ meanModes: why one regression lands on the average */
G.reg("meanModes", fig => {
  const W = 900, Hh = 440, sv = fig.svg(W, Hh, { label: "Hypotheses moving toward three possible next concepts" });
  const PX = 20, PY = 44, PW = 560, PH = 360;
  const MODES = [{ x: 120, y: 95, n: "blue road" }, { x: 450, y: 125, n: "orange road" }, { x: 285, y: 305, n: "green road" }];
  let K = 1, eps = 0.05, skew = false, seedN = 1, hyp = [], trails = [], win = [], steps = 0, rf;
  const probs = () => skew ? [0.6, 0.3, 0.1] : [1 / 3, 1 / 3, 1 / 3];
  const mean = () => { const p = probs(); return [MODES.reduce((a, m, i) => a + p[i] * m.x, 0), MODES.reduce((a, m, i) => a + p[i] * m.y, 0)]; };
  const kSl = ui.range("hypotheses K", 1, 6, 1, 1, v => { K = v; reset(); }, v => String(v));
  const eSl = ui.range("what losers learn", 0, 1, 0.05, 0.05, v => { eps = v; }, v => v === 0 ? "nothing" : v >= 1 ? "same as winner" : v.toFixed(2));
  const sk = ui.seg([["eq", "Options equally likely"], ["sk", "Blue most likely"]], "eq", v => { skew = v === "sk"; reset(); }, "option probabilities");
  const pb = ui.btn("Pause", () => { fig.playing = !fig.playing; pb.textContent = fig.playing ? "Pause" : "Play"; if (fig.playing) G.kick(); });
  fig.$ui.append(kSl, eSl, sk, pb, ui.btn("Restart", () => { seedN++; reset(); }));
  fig.playing = !G.REDUCED; pb.textContent = fig.playing ? "Pause" : "Play";
  sv.rect(PX, PY, PW, PH, { fill: "var(--panel2)", stroke: "var(--line2)", rx: 12 });
  sv.text(PX + 6, PY - 18, "Concept space, shown in 2-D. Dots are possible next concepts.", { size: 12.5, fill: "var(--ink3)" });
  const dyn = sv.g({}), right = sv.g({});
  MODES.forEach((m, i) => { sv.circle(PX + m.x, PY + m.y, 17, { fill: "none", stroke: ROADC[i], sw: 2, opacity: .7 }); sv.circle(PX + m.x, PY + m.y, 5, { fill: ROADC[i] }); sv.text(PX + m.x, PY + m.y - 28, m.n, { size: 12, anchor: "middle", fill: "var(--ink2)", weight: 600 }); });
  function reset() {
    rf = rng(seedN * 131 + K * 7 + (skew ? 3 : 0)); hyp = Array.from({ length: K }, () => [PW / 2 + (rf() - .5) * 90, PH / 2 + (rf() - .5) * 90]); trails = hyp.map(h => [h.slice()]); win = new Array(K).fill(1 / K); steps = 0; draw();
  }
  function train(n) {
    const p = probs();
    for (let s = 0; s < n; s++) {
      let u = rf(), mi = 0, a = 0; for (let i = 0; i < 3; i++) { a += p[i]; if (u <= a) { mi = i; break; } }
      const t = [MODES[mi].x + gauss(rf) * 12, MODES[mi].y + gauss(rf) * 12];
      let w = 0, bd = 1e18; hyp.forEach((h, k) => { const d = (h[0] - t[0]) ** 2 + (h[1] - t[1]) ** 2; if (d < bd) { bd = d; w = k; } });
      hyp.forEach((h, k) => { const lr = 0.035 * (k === w ? 1 : eps); h[0] += lr * (t[0] - h[0]); h[1] += lr * (t[1] - h[1]); win[k] = win[k] * 0.995 + (k === w ? 0.005 : 0); });
      steps++;
    }
    hyp.forEach((h, k) => { const tr = trails[k]; tr.push(h.slice()); if (tr.length > 70) tr.shift(); });
  }
  function draw() {
    dyn.innerHTML = ""; right.innerHTML = "";
    const mu = mean();
    trails.forEach((tr, k) => { if (tr.length > 1) S("polyline", { points: tr.map(p => `${PX + p[0]},${PY + p[1]}`).join(" "), fill: "none", stroke: "var(--code)", "stroke-width": 1.6, opacity: .35 }, dyn); });
    hyp.forEach((h, k) => { S("rect", { x: PX + h[0] - 10, y: PY + h[1] - 10, width: 20, height: 20, rx: 6, fill: "var(--code)", stroke: "var(--panel)", "stroke-width": 2 }, dyn); const t = S("text", { x: PX + h[0], y: PY + h[1] + 0.5, "text-anchor": "middle", "dominant-baseline": "central", "font-size": 11, "font-weight": 700, fill: "var(--bg)" }, dyn); t.textContent = k + 1; });
    S("path", { d: `M ${PX + mu[0] - 8} ${PY + mu[1] - 8} l 16 16 m 0 -16 l -16 16`, stroke: "var(--loss)", "stroke-width": 3, "stroke-linecap": "round" }, dyn);
    const lab = S("text", { x: PX + mu[0] + 14, y: PY + mu[1] + 22, "font-size": 12, "font-weight": 650, fill: "var(--loss)" }, dyn); lab.textContent = "average of the options";
    /* coverage metric: distance from each option centre to its nearest hypothesis */
    const cov = MODES.map(m => Math.min(...hyp.map(h => Math.hypot(h[0] - m.x, h[1] - m.y)))), avg = cov.reduce((a, b) => a + b, 0) / 3;
    const RX = 620; sv.text(RX, 58, "How often each guess is the closest", { size: 13, weight: 650 }, right);
    win.forEach((v, k) => { const y = 78 + k * 34; sv.text(RX, y + 11, "guess " + (k + 1), { size: 12, fill: "var(--ink2)" }, right); sv.rect(RX + 60, y, 190, 22, { fill: "var(--bg2)", stroke: "none", rx: 5 }, right); sv.rect(RX + 60, y, Math.max(2, 190 * v), 22, { fill: "var(--code)", stroke: "none", rx: 5 }, right); sv.text(RX + 66 + 190 * v, y + 11, Math.round(v * 100) + "%", { size: 11.5, mono: true, fill: "var(--ink2)" }, right); });
    sv.text(RX, 320, "Typical distance from a real option\nto the nearest guess:", { size: 12.5, fill: "var(--ink2)" }, right);
    sv.text(RX, 366, avg.toFixed(0), { size: 30, weight: 700, mono: true, fill: avg < 40 ? "var(--good)" : "var(--loss)" }, right);
    sv.text(RX + 52, 368, avg < 40 ? "close: every option is covered" : "far: some option has no guess near it", { size: 12, fill: "var(--ink2)" }, right);
    fig.$live.hidden = false;
    fig.$live.innerHTML = K === 1 ? "<b>One guess has no choice.</b> It is pulled toward every option in turn, so it settles on the average, the red cross. That point is not any real option. Every road the model could take is far from it." : eps >= 1 ? "<b>When losers learn as much as winners, nobody specialises.</b> All guesses drift to the same average. Lower the slider to let only the closest guess learn." : K < 3 ? "<b>Two guesses cannot cover three options.</b> One guess ends up on a single option and the other settles between the remaining two. More guesses are needed." : "<b>Winner-take-all training.</b> For each example only the closest guess gets a full lesson. The guesses spread out, and each commits to one real option.";
  }
  loop(fig, (t, dt) => { if (steps > 3000) return; train(12); draw(); });
  reset();
});

/* ------------------------------------------------------------ hncpFlow: one block, K codes */
G.reg("hncpFlow", fig => {
  const sv = fig.svg(900, 430, { label: "The last concept block run once per code, then guesses, selector and winner" });
  const KK = 4, ys = [74, 134, 194, 254], Y0 = 346, hues = ["#d9a227", "#e07b39", "#c84d72", "#7357c9"];
  const gs = {}; ["stream", "codes", "block", "heads", "sel", "train"].forEach(k => { gs[k] = sv.g({}); });
  const vec = (g, x, y, color, seed, w) => { w = w || 54; sv.rect(x - w / 2, y - 14, w, 28, { fill: "var(--panel)", stroke: "var(--line2)", sw: 1, rx: 6 }, g); for (let b = 0; b < 8; b++) sv.rect(x - w / 2 + 5 + b * ((w - 10) / 8), y - 10, (w - 10) / 8 - 1.5, 20, { fill: color, stroke: "none", rx: 1.2, opacity: .3 + (((b * 47 + seed * 31) % 65) / 100) }, g); };
  /* stream */
  vec(gs.stream, 56, 194, "var(--con)", 3); sv.text(56, 224, "stream x", { size: 12, weight: 650, anchor: "middle" }, gs.stream); sv.text(56, 240, "(the block's\nplain input)", { size: 10.5, anchor: "middle", fill: "var(--ink3)" }, gs.stream);
  /* codes */
  ys.forEach((y, k) => { vec(gs.codes, 146, y, hues[k], k + 1, 46); sv.text(146, y - 24, "code " + (k + 1), { size: 11, anchor: "middle", fill: "var(--ink2)" }, gs.codes); sv.line(83, 194, 214 - 12, y, { stroke: "var(--con)", sw: 1.2, opacity: .6 }, gs.codes); sv.line(171, y, 202, y, { stroke: hues[k], sw: 1.4 }, gs.codes); sv.circle(214, y, 11, { fill: "var(--panel)", stroke: "var(--ink3)", sw: 1.4 }, gs.codes); sv.text(214, y, "+", { size: 15, weight: 700, anchor: "middle" }, gs.codes); });
  sv.text(146, 14, "K learned code vectors\n(4 drawn, 8 in S2)", { size: 10.5, anchor: "middle", fill: "var(--ink3)" }, gs.codes);
  /* block */
  sv.rect(250, 38, 160, 244, { fill: "var(--con-s)", stroke: "var(--con)", sw: 1.8, rx: 14 }, gs.block); sv.text(330, 56, "last concept block", { size: 13, weight: 650, anchor: "middle" }, gs.block);
  ys.forEach((y, k) => { sv.line(225, y, 250, y, { stroke: "var(--ink3)", sw: 1.4, arrow: "ink3" }, gs.block); sv.line(410, y, 440, y, { stroke: hues[k], sw: 1.6 }, gs.block); });
  sv.text(330, 262, "same weights for every row\nkeys and values come\nfrom the plain stream", { size: 10.5, anchor: "middle", fill: "var(--ink2)" }, gs.block);
  sv.rect(250, 316, 160, 60, { fill: "var(--con-s)", stroke: "var(--con)", sw: 1.4, rx: 12, dash: "5 3" }, gs.block); sv.text(330, 346, "same block, no code", { size: 12, weight: 600, anchor: "middle" }, gs.block);
  sv.path(`M 83 194 C 100 194 100 ${Y0} 250 ${Y0}`, { stroke: "var(--con)", sw: 1.2, opacity: .6, arrow: "con" }, gs.block);
  ys.forEach((y, k) => vec(gs.block, 466, y, hues[k], k + 7, 46)); sv.text(466, 106 - 52, "u(k)", { size: 12, weight: 650, anchor: "middle" }, gs.block);
  vec(gs.block, 466, Y0, "var(--con)", 11, 46); sv.line(410, Y0, 440, Y0, { stroke: "var(--con)", sw: 1.6 }, gs.block); sv.text(466, Y0 + 26, "u(0)", { size: 12, weight: 650, anchor: "middle" }, gs.block);
  /* heads */
  sv.rect(520, 38, 96, 244, { fill: "var(--frozen-s)", stroke: "var(--frozen)", sw: 1.6, rx: 12, dash: "5 3" }, gs.heads); sv.text(568, 56, "frozen\nsticker heads", { size: 11.5, weight: 650, anchor: "middle" }, gs.heads);
  ys.forEach((y, k) => { sv.line(492, y, 520, y, { stroke: hues[k], sw: 1.4, arrow: "ink3" }, gs.heads); sv.line(616, y, 648, y, { stroke: hues[k], sw: 1.4 }, gs.heads); vec(gs.heads, 676, y, hues[k], k + 13, 46); });
  sv.text(676, 30, "guesses", { size: 12, weight: 650, anchor: "middle" }, gs.heads); sv.text(676, 106 - 54 + 14, "ĉ(k)", { size: 11, anchor: "middle", fill: "var(--ink3)" }, gs.heads);
  /* selector */
  sv.rect(520, 316, 96, 60, { fill: "var(--sel-s)", stroke: "var(--sel)", sw: 1.6, rx: 12 }, gs.sel); sv.text(568, 346, "selector", { size: 12.5, weight: 650, anchor: "middle" }, gs.sel); sv.line(492, Y0, 520, Y0, { stroke: "var(--con)", sw: 1.4, arrow: "con" }, gs.sel);
  const pis = [.42, .28, .2, .1]; pis.forEach((p, k) => { sv.rect(660 + k * 32, Y0 + 28 - 70 * p, 22, 70 * p, { fill: "var(--sel)", stroke: "none", rx: 3 }, gs.sel); sv.text(671 + k * 32, Y0 + 40, String(k + 1), { size: 10.5, anchor: "middle", fill: "var(--ink3)" }, gs.sel); });
  sv.line(616, Y0, 650, Y0, { stroke: "var(--sel)", sw: 1.4, arrow: "sel" }, gs.sel); sv.text(722, Y0 - 52, "odds of each code π", { size: 11.5, weight: 650, anchor: "middle" }, gs.sel);
  /* training: distance to truth */
  const dist = [.55, .15, .8, .4];
  ys.forEach((y, k) => { sv.line(702, y, 732, y, { stroke: "var(--ink3)", sw: 1.2 }, gs.train); sv.rect(732, y - 9, 120 * dist[k] + 4, 18, { fill: k === 1 ? "var(--good)" : "var(--bg2)", stroke: k === 1 ? "none" : "var(--line2)", rx: 4 }, gs.train); if (k === 1) sv.path(`M 862 ${y} l 5 8 l 9 1 l -7 6 l 2 9 l -9 -5 l -9 5 l 2 -9 l -7 -6 l 9 -1 z`, { fill: "var(--code)", stroke: "none" }, gs.train); });
  sv.text(792, 30, "distance to the\ntrue next concept", { size: 11.5, weight: 650, anchor: "middle" }, gs.train); sv.text(792, 288, "shortest wins: code 2", { size: 12, anchor: "middle", fill: "var(--good)", weight: 650 }, gs.train);
  const order = ["stream", "codes", "block", "heads", "sel", "train"];
  const STEPS = [
    { label: "1 Stream", title: "The stream arrives.", text: "The last concept block is about to process one position. Its normal input is a vector, the stream x." },
    { label: "2 Add a code", title: "Add a different code each time.", text: "We keep K learned code vectors. For code k, a small matrix turns it into a nudge that is added to the stream. K different nudged copies are made." },
    { label: "3 One block, K times", title: "Run the same block on every copy.", text: "The block has the same weights every time. Its keys and values still come from the plain stream, so one pass over the sequence serves all K codes. Out come K different vectors, one per code. A plain row with no code is kept too." },
    { label: "4 Guesses", title: "Frozen heads turn each vector into a guess.", text: "The sticker heads from S1 stay frozen. They convert each vector into a guess of the next concept. Now we have K committed guesses instead of one blur." },
    { label: "5 Selector", title: "A selector predicts the odds.", text: "A small linear layer reads the plain row and outputs odds for the K codes. At test time it is the only way to choose without seeing the answer." },
    { label: "6 Learn", title: "Training rewards the closest guess.", text: "We compare every guess with the true next concept. The closest one gets the full lesson (winner-take-all) and the selector learns to predict which code tends to win." },
  ];
  let stp = 0; const PU = [[], [[[83, 194], [146, 194]]], ys.map((y, k) => [[225, y], [440, y]]), ys.map((y, k) => [[492, y], [648, y]]), [[[492, Y0], [650, Y0]]], ys.map((y, k) => [[702, y], [732, y]])];
  stepper(fig, STEPS, i => { stp = i; order.forEach((k, ix) => { gs[k].style.transition = "opacity .35s"; gs[k].style.opacity = ix < i ? .72 : ix === i ? 1 : .1; }); if (i >= 2) { gs.stream.style.opacity = .72; gs.block.style.opacity = i === 2 ? 1 : .72; } }, { ms: 4800 });
  const dots = Array.from({ length: 6 }, () => sv.circle(-20, -20, 4.4, { fill: "var(--accent)", stroke: "var(--panel)", sw: 1.5 }));
  loop(fig, t => { const P = PU[stp] || []; dots.forEach((d, k) => { if (!P[k]) { d.setAttribute("cx", -20); return; } const [x, y] = along(P[k], (t * 0.5 + k * 0.1) % 1); d.setAttribute("cx", x); d.setAttribute("cy", y); }); });
});

/* ------------------------------------------------------------ selector: odds, hindsight and best-of-m */
G.reg("selector", fig => {
  const W = 900, Hh = 500, sv = fig.svg(W, Hh, { label: "Distances, selector odds and who receives the decoder's training signal for m equals 1, 2 or K" });
  const KK = 6; let m = 2, seedN = 3, d, pi, share, drawn = [];
  const mseg = ui.seg([["1", "m = 1"], ["2", "m = 2"], ["6", "m = K"]], "2", v => { m = +v; drawn = []; calc(); draw(); }, "codes drawn per step");
  fig.$ui.append(mseg, ui.btn("New example", () => { seedN++; drawn = []; fresh(); }), ui.btn("Draw one", () => { drawOne(); }, "pri"));
  const dyn = sv.g({});
  function fresh() { const r = rng(seedN * 977); d = Array.from({ length: KK }, () => 0.15 + 0.85 * r()); const raw = Array.from({ length: KK }, () => 0.2 + r() ** 2 * 2); const z = raw.reduce((a, b) => a + b, 0); pi = raw.map(v => v / z); calc(); draw(); }
  function pickM(r) { const avail = pi.map((p, k) => [p, k]), got = []; for (let i = 0; i < Math.min(m, KK); i++) { const z = avail.reduce((a, b) => a + b[0], 0); let u = r() * z, j = 0; for (; j < avail.length - 1; j++) { u -= avail[j][0]; if (u <= 0) break; } got.push(avail[j][1]); avail.splice(j, 1); } return got; }
  function calc() { const r = rng(4242 + m), c = new Array(KK).fill(0), N = 5000; for (let i = 0; i < N; i++) { const s = pickM(r); let b = s[0]; s.forEach(k => { if (d[k] < d[b]) b = k; }); c[b]++; } share = c.map(v => v / N); }
  function drawOne() { const r = rng(Math.floor(Math.random() * 1e9)); drawn = pickM(r); draw(); }
  function draw() {
    dyn.innerHTML = ""; const best = d.indexOf(Math.min(...d)), X0 = 190, CW = 100, rows = [["Distance to the truth", "lower is closer", d, "var(--enc)", 1], ["Selector odds π", "its prediction of which code wins", pi, "var(--sel)", Math.max(...pi) * 1.15], ["Share of the decoder's lesson", "who gets the token gradient for this m", share, "var(--code)", 1]];
    rows.forEach(([t, sub, vals, col, mx], ri) => {
      const y0 = 50 + ri * 150, hh = 88; sv.text(20, y0 + 18, t, { size: 13.5, weight: 650 }, dyn); sv.text(20, y0 + 38, sub, { size: 11.5, fill: "var(--ink3)" }, dyn);
      sv.line(X0 - 8, y0 + hh, X0 + CW * KK, y0 + hh, { stroke: "var(--line2)", sw: 1.2 }, dyn);
      vals.forEach((v, k) => { const x = X0 + k * CW + 20, h = Math.max(2, hh * v / mx * 0.92), isD = drawn.includes(k); sv.rect(x, y0 + hh - h, 60, h, { fill: col, stroke: isD ? "var(--ink)" : "none", sw: 2.5, rx: 5, opacity: ri === 0 && k === best ? 1 : .8 }, dyn); sv.text(x + 30, y0 + hh - h - 9, ri === 0 ? v.toFixed(2) : Math.round(v * 100) + "%", { size: 12, mono: true, anchor: "middle", fill: "var(--ink2)" }, dyn); if (ri === 0 && k === best) sv.path(`M ${x + 30} ${y0 + hh + 24} l 4 8 l 9 1 l -7 6 l 2 9 l -8 -5 l -8 5 l 2 -9 l -7 -6 l 9 -1 z`, { fill: "var(--code)" }, dyn); sv.text(x + 30, y0 + hh + 14, "code " + (k + 1), { size: 11.5, anchor: "middle", fill: "var(--ink3)" }, dyn); });
    });
    const picked = drawn.length ? drawn.reduce((a, k) => d[k] < d[a] ? k : a, drawn[0]) : null;
    fig.$live.hidden = false;
    fig.$live.innerHTML = (drawn.length ? `<b>This draw:</b> the selector offered codes ${drawn.map(k => k + 1).join(" and ")}. The closest of them, <b>code ${picked + 1}</b>, steers the decoder and receives the lesson. ` : "") + (m === 1 ? "<b>m = 1:</b> the code that gets the lesson is whatever the selector happened to draw. A code is pushed toward whatever happened to occur, even if it represents another option. That is the same pull toward the average that K = 1 has." : m === 2 ? "<b>m = 2:</b> two codes are drawn using the selector's odds and the closer one wins the lesson. The star (the truly closest code) gets it often, but the training still looks like how the model will be used. This is the setting used in the main arm." : "<b>m = K:</b> every code is offered, so the truly closest code always gets the lesson. Training is easiest, but at test time there is no answer to look at, so the selector's prior was never practised. An earlier project version did exactly this and its prior was confidently wrong.");
  }
  fresh();
});

/* ------------------------------------------------------------ closedLoop: reading its own guesses */
G.reg("closedLoop", fig => {
  const W = 900, Hh = 380, sv = fig.svg(W, Hh, { label: "Teacher forcing versus closed-loop spans of the concept core's inputs" });
  const NC = 8, X0 = 150, CWd = 88, bx = j => X0 + j * CWd; let N = 3, j0 = 2, on = true;
  fig.$ui.append(ui.check("closed loop on for this sequence", true, v => { on = v; draw(); }), ui.range("span length N", 1, 4, 1, 3, v => { N = v; draw(); }, v => String(v)), ui.range("starts after chunk", 0, 4, 1, 2, v => { j0 = v; draw(); }, v => String(v)));
  const dyn = sv.g({}); fig.playing = !G.REDUCED; const dots = [];
  function draw() {
    dyn.innerHTML = ""; dots.length = 0;
    sv.text(14, 50, "What the core reads", { size: 13, weight: 650 }, dyn); sv.text(14, 70, "(its inputs)", { size: 11.5, fill: "var(--ink3)" }, dyn);
    sv.text(14, 190, "The core's guess", { size: 13, weight: 650 }, dyn); sv.text(14, 210, "of the next concept", { size: 11.5, fill: "var(--ink3)" }, dyn);
    sv.text(14, 316, "What it is graded on", { size: 13, weight: 650 }, dyn); sv.text(14, 336, "always the true concept", { size: 11.5, fill: "var(--ink3)" }, dyn);
    for (let j = 0; j < NC; j++) {
      const inLoop = on && j > j0 && j <= j0 + N, x = bx(j) + 60;
      sv.rect(x - 34, 40, 68, 38, { fill: inLoop ? "var(--code-s)" : "var(--enc-s)", stroke: inLoop ? "var(--code)" : "var(--enc)", sw: inLoop ? 2.4 : 1.4, rx: 9 }, dyn);
      sv.text(x, 59, (inLoop ? "own " : "real ") + "C" + j, { size: 12, weight: 650, anchor: "middle", fill: inLoop ? "var(--code)" : "var(--enc)" }, dyn);
      sv.rect(x - 34, 172, 68, 38, { fill: "var(--con-s)", stroke: "var(--con)", sw: 1.4, rx: 9 }, dyn); sv.text(x, 191, "ĉ" + j, { size: 12.5, weight: 650, anchor: "middle", fill: "var(--con)" }, dyn);
      sv.line(x, 78, x, 172, { stroke: "var(--con)", sw: 1.4, arrow: "con" }, dyn);
      sv.rect(x - 34, 300, 68, 38, { fill: "var(--panel)", stroke: "var(--line2)", sw: 1.4, rx: 9 }, dyn); sv.text(x, 319, "true C" + (j + 1), { size: 11.5, anchor: "middle", fill: "var(--ink2)" }, dyn);
      sv.path(`M ${x} 212 L ${x} 298`, { stroke: "var(--loss)", sw: 1.4, dash: "4 3", arrow: "loss" }, dyn);
      if (inLoop) { const px = bx(j - 1) + 60; const pts = [[px + 34, 192], [px + 66 - 8, 130], [x, 110], [x, 80]]; sv.path(`M ${px + 34} 192 C ${px + 60} 192 ${x - 20} 100 ${x} 80`, { stroke: "var(--code)", sw: 2.4, arrow: "code" }, dyn); dots.push([[px + 34, 192], [px + 60, 180], [x - 20, 100], [x, 82]]); }
    }
    sv.text(W / 2, 362, on ? `chunks ${j0 + 1} to ${Math.min(NC - 1, j0 + N)} read the model's own earlier guess instead of the real concept` : "ordinary training: every input is the real concept of its chunk", { size: 12.5, anchor: "middle", fill: "var(--ink2)" }, dyn);
    fig.$live.hidden = false; fig.$live.innerHTML = "<b>Why practise on your own output?</b> At search time the model will read its own guesses, never real concepts. In ordinary training it has only ever read real ones. So in a quarter of training sequences (p = 0.25) a short span of 1 to 4 chunks is replaced by the model's own committed guess, with no gradient through that hand-off. The grading target never changes. <span class=\"muted\">One limit: the guess fed back is the winning one chosen using the answer, so wrong branches that search will visit are not practised here.</span>";
  }
  loop(fig, t => { }); draw();
});

/* ------------------------------------------------------------ deadCodes: restart rule (illustration) */
G.reg("deadCodes", fig => {
  const W = 900, Hh = 360, sv = fig.svg(W, Hh, { label: "Win share of eight codes over time, one code dying and being restarted" });
  const K = 8, thr = 1 / (10 * K); let share, upd = 0, log = "", killed = -1, lastEp = 0;
  const reset = () => { share = Array.from({ length: K }, () => 1 / K); upd = 0; killed = -1; log = ""; lastEp = 0; draw(); };
  fig.$ui.append(ui.btn("Kill a code", () => { killed = [1, 3, 5, 6][Math.floor(Math.random() * 4)]; log = `A code stops winning (code ${killed + 1})`; }), ui.btn("Reset", reset));
  fig.playing = !G.REDUCED; const pb = G.playBtn(fig); fig.$ui.append(pb);
  const dyn = sv.g({}), X0 = 90, BW = 90;
  function tick(dt) {
    upd += dt * 400; const target = share.map((v, k) => k === killed ? 0.002 : 1 / K + Math.sin(upd / 300 + k * 1.7) * 0.02);
    let tot = 0; share = share.map((v, k) => { const nv = lerp(v, target[k], 0.02); tot += nv; return nv; }); share = share.map(v => v / tot);
    const ep = Math.floor(upd / 500); if (ep !== lastEp) { lastEp = ep; if (killed >= 0 && share[killed] < thr * 1.5) { const top = share.indexOf(Math.max(...share)); share[killed] = share[top]; log = `Update ${ep * 500}: code ${killed + 1} was dead, so it restarts as a copy of code ${top + 1} plus a little noise`; killed = -1; } }
    draw();
  }
  function draw() {
    dyn.innerHTML = ""; sv.text(20, 26, "Share of the last 500 updates in which each code was the closest", { size: 13, weight: 650 }, dyn);
    const Y0 = 270, mx = 0.3, py = v => Y0 - 200 * v / mx;
    sv.line(X0 - 10, Y0, X0 + K * BW, Y0, { stroke: "var(--line2)", sw: 1.2 }, dyn);
    sv.line(X0 - 10, py(thr), X0 + K * BW, py(thr), { stroke: "var(--bad)", sw: 1.4, dash: "6 4" }, dyn); sv.text(X0 + K * BW + 4, py(thr), "dead below\n1 / (10 K)", { size: 11, fill: "var(--bad)", base: "central" }, dyn);
    share.forEach((v, k) => { const dead = v < thr * 1.2; sv.rect(X0 + k * BW + 12, py(v), BW - 24, Math.max(2, Y0 - py(v)), { fill: dead ? "var(--bad)" : "var(--code)", stroke: "none", rx: 5, opacity: dead ? 1 : .85 }, dyn); sv.text(X0 + k * BW + BW / 2, Y0 + 18, "code " + (k + 1), { size: 11.5, anchor: "middle", fill: "var(--ink3)" }, dyn); sv.text(X0 + k * BW + BW / 2, py(v) - 9, (v * 100).toFixed(1) + "%", { size: 11, mono: true, anchor: "middle", fill: "var(--ink2)" }, dyn); });
    sv.text(20, 330, log || "Press Kill a code to see the rule fire. Winner-take-all training can leave a code that never wins, and it then never learns.", { size: 12.5, fill: "var(--ink2)" }, dyn);
    fig.$live.hidden = false; fig.$live.innerHTML = "<b>Illustration, not a measurement.</b> Every 500 updates (and only in the first 80% of training), a code whose win share fell below 1 in 10 K is dead. It is restarted as a copy of the most-winning code plus a little noise, with its optimiser memory cleared. The project logs per-code win share, selector entropy and the distances between guesses, and checks that at least K/2 codes stay alive.";
  }
  loop(fig, (t, dt) => tick(dt)); reset();
});

/* ------------------------------------------------------------ codesDecode: what does each code actually write? (G2.4) */
G.reg("codesDecode", fig => {
  const W = 900, Hh = 340, sv = fig.svg(W, Hh, { label: "Eight codes and the road each one decodes to" });
  const SCEN = {
    good: { l: "Codes that branch", dec: [0, 1, 0, 2, 1, 2, 0, 1], t: "The 8 codes decode to all three roads, and every decoded road is a real one. Searching over codes would explore real alternatives." },
    collapse: { l: "Codes that collapse", dec: [1, 1, 1, 1, 1, 1, 1, 1], t: "Every code writes the same road. The codes carry no choice, so there is nothing to branch over. Gate G2.4 would fail." },
    messy: { l: "Codes that sometimes break", dec: [0, 3, 1, 3, 2, 0, 3, 1], t: "Some codes write a road that does not exist from this town (illegal). Branching exists, but a search would waste effort on impossible moves." },
  };
  let key = "good";
  fig.$ui.append(ui.seg(Object.entries(SCEN).map(([k, v]) => [k, v.l]), key, v => { key = v; draw(); }, "scenario"));
  const dyn = sv.g({});
  function draw() {
    dyn.innerHTML = ""; const sc = SCEN[key]; sv.text(20, 26, "A state where blue and orange both make progress. Green leads away from the goal.", { size: 13, weight: 650 }, dyn);
    sc.dec.forEach((d, k) => { const x = 40 + k * 106; sv.rect(x, 60, 92, 40, { fill: "var(--code-s)", stroke: "var(--code)", sw: 1.4, rx: 9 }, dyn); sv.text(x + 46, 80, "code " + (k + 1), { size: 12.5, weight: 650, anchor: "middle" }, dyn); sv.line(x + 46, 100, x + 46, 140, { stroke: "var(--ink3)", sw: 1.4, arrow: "ink3" }, dyn); sv.text(x + 46, 120, "writes", { size: 10.5, anchor: "middle", fill: "var(--ink3)", halo: true }, dyn);
      const ill = d === 3, c = ill ? "#c63d3d" : ROADC[d]; sv.rect(x, 142, 92, 40, { fill: ill ? "var(--bad-s)" : `color-mix(in srgb, ${c} 20%, var(--panel))`, stroke: c, sw: 1.8, rx: 9, dash: ill ? "4 3" : null }, dyn); sv.text(x + 46, 162, ill ? "illegal" : ["blue", "orange", "green"][d], { size: 13, weight: 650, anchor: "middle" }, dyn); });
    const legal = sc.dec.filter(d => d < 3), distinct = new Set(legal).size, progress = legal.filter(d => d < 2), distinctProgress = new Set(progress).size, dup = 1 - new Set(sc.dec).size / sc.dec.length;
    const rows = [["Distinct legal roads", `${distinct} of 3`], ["Distinct roads that make progress", `${distinctProgress} of 2`], ["Illegal writes", `${sc.dec.filter(d => d === 3).length} of 8`], ["Branching check (needs at least 2 distinct legal)", distinct >= 2 ? "pass" : "fail"]];
    rows.forEach(([a, b], i) => { sv.text(40, 224 + i * 26, a, { size: 13, fill: "var(--ink2)" }, dyn); sv.text(520, 224 + i * 26, b, { size: 13.5, weight: 700, mono: true, fill: b === "fail" ? "var(--bad)" : b === "pass" ? "var(--good)" : "var(--ink)" }, dyn); });
    sv.text(W - 20, Hh - 12, "illustration with made-up outcomes", { size: 11, anchor: "end", fill: "var(--ink3)" }, dyn);
    fig.$live.hidden = false; fig.$live.innerHTML = "<b>" + sc.l + ".</b> " + sc.t;
  }
  draw();
});

/* ------------------------------------------------------------ s2status: development diagnostics from the real run files */
G.reg("s2status", fig => {
  const S2 = D().s2 || {}, gen = D().generated || "";
  const ARMS = [["V8repro", "S2_V8repro_dense", "V8 reproduction · K = 1, no closed loop", "#e87ba4"], ["K1", "S2_K1_dense", "K = 1 · closed loop", "#008300"], ["K8", "S2_K8_dense", "K = 8 · closed loop", "#4a3aa7"]];
  const pct = v => v == null ? "–" : (v * 100).toFixed(1) + "%";
  const wrap = H("div", { class: "tablewrap", style: "margin:6px 12px 10px" });
  const t = H("table", { class: "t" }, H("thead", {}, H("tr", {}, H("th", {}, "Arm"), H("th", {}, "Seed"), H("th", { class: "n" }, "Standard"), H("th", { class: "n" }, "Hard10"), H("th", { class: "n" }, "Hard14"), H("th", {}, "State"))));
  const tb = H("tbody");
  ARMS.forEach(([k, pre, label]) => ["911", "912"].forEach(s => {
    const r = S2[`${pre}_s${s}`]; if (!r) return; const m = r.measure, run = !r.wall_s;
    tb.append(H("tr", {}, H("td", {}, H("b", { class: "mono" }, "S2 " + k), H("div", { class: "muted", style: "font-size:12px" }, label)), H("td", { class: "mono" }, s), ...["standard", "hard10", "hard14"].map(c => H("td", { class: "n" }, m && m[c] ? pct(m[c].rate) : "–", m && m[c] ? H("div", { class: "muted", style: "font-size:11px" }, "n = " + m[c].n) : null)), H("td", {}, run ? H("span", { class: "chip run" }, `training ${r.steps.toLocaleString("en-US")} / 12,000`) : m ? H("span", { class: "chip done" }, "measured") : H("span", { class: "chip wait" }, "measuring"))));
  }));
  t.append(tb); wrap.append(t); fig.$stage.append(wrap);
  fig.$stage.append(H("div", { class: "note" }, `Snapshot of the run files taken ${gen}. Greedy decoding, development diagnostics, not sealed results. Seed 911 learns from S1's seed-902 teacher and seed 912 from V8's seed-811 teacher, so compare arms within a seed, not across seeds. Hard14 has only 256 puzzles, so a single point is about 0.4 percentage points.`));
  /* charts */
  const sv = new G.Svg(fig.$stage, 900, 366, { label: "Training curves of the S2 arms" }); const dyn = sv.g({});
  const MET = { symbolic: ["Symbolic loss (puzzle tokens)", [0.04, 0.3]], ncp: ["NCP loss (guess of the next concept)", [0.4, 0.9]], text: ["Text loss", [2.3, 2.6]], ent: ["K8 selector entropy (max = ln 8 = 2.08)", [1.85, 2.1]], pick: ["K8: share where the decoder's code is the best code (1 in 4 by chance)", [0.2, 0.4]] };
  let met = "symbolic";
  fig.$ui.append(ui.seg(Object.entries(MET).map(([k, v]) => [k, { symbolic: "Symbolic loss", ncp: "NCP loss", text: "Text loss", ent: "Selector entropy", pick: "Best code picked" }[k]]), met, v => { met = v; draw(); }, "metric"));
  function draw() {
    dyn.innerHTML = ""; const [title, [lo, hi]] = MET[met], X0 = 70, X1 = 870, Y0 = 270, Y1 = 46, px = s => X0 + (X1 - X0) * s / 12000, py = v => Y0 - (Y0 - Y1) * (v - lo) / (hi - lo);
    sv.text(20, 22, title, { size: 13.5, weight: 650 }, dyn);
    for (let i = 0; i <= 4; i++) { const v = lo + (hi - lo) * i / 4; sv.line(X0, py(v), X1, py(v), { stroke: "var(--line)", sw: 1 }, dyn); sv.text(X0 - 8, py(v), v.toFixed(met === "text" ? 2 : 2), { size: 11, mono: true, anchor: "end", fill: "var(--ink3)" }, dyn); }
    [0, 3000, 6000, 9000, 12000].forEach(s => sv.text(px(s), Y0 + 18, s === 0 ? "0" : (s / 1000) + "k", { size: 11, mono: true, anchor: "middle", fill: "var(--ink3)" }, dyn)); sv.text((X0 + X1) / 2, Y0 + 38, "training update", { size: 12, anchor: "middle", fill: "var(--ink2)" }, dyn);
    if (met === "ent") { sv.line(X0, py(Math.log(8)), X1, py(Math.log(8)), { stroke: "var(--ink3)", sw: 1.4, dash: "6 4" }, dyn); sv.text(X1, py(Math.log(8)) - 9, "uniform choice over 8 codes", { size: 11.5, anchor: "end", fill: "var(--ink3)" }, dyn); }
    if (met === "pick") { sv.line(X0, py(0.25), X1, py(0.25), { stroke: "var(--ink3)", sw: 1.4, dash: "6 4" }, dyn); sv.text(X1, py(0.25) - 9, "chance with m = 2 of 8: 0.25", { size: 11.5, anchor: "end", fill: "var(--ink3)" }, dyn); }
    const series = [];
    if (met === "ent" || met === "pick") { ["S2_K8_dense_s911", "S2_K8_dense_s912"].forEach((n, i) => { const r = S2[n]; if (r && r.k8) series.push([n.replace("S2_K8_dense_", "K8 seed "), r.k8.map(p => [p.step, met === "ent" ? p.selector_entropy : p.kdec_is_kstar]), "#4a3aa7", i ? "6 4" : null]); }); }
    else ARMS.forEach(([k, pre, , c]) => ["911", "912"].forEach((s, i) => { const r = S2[`${pre}_s${s}`]; if (r && r.curve && r.curve.length) series.push([`${k} seed ${s}`, r.curve.map(p => [p.step, p[met]]), c, i ? "6 4" : null]); }));
    series.forEach(([name, pts, c, dash]) => { const ok = pts.filter(p => p[1] != null); if (ok.length) sv.path("M" + ok.map(p => `${px(p[0]).toFixed(1)} ${py(clamp(p[1], lo, hi)).toFixed(1)}`).join(" L"), { stroke: c, sw: 2.2, dash }, dyn); });
    series.forEach(([name, , c, dash], i) => { const x = 74 + (i % 3) * 270, y = 326 + Math.floor(i / 3) * 20; sv.line(x, y, x + 26, y, { stroke: c, sw: 2.4, dash }, dyn); sv.text(x + 32, y, name, { size: 11.5, fill: "var(--ink2)" }, dyn); });
    const lastOf = pts => { const ok = pts.filter(p => p[1] != null); const tail = ok.slice(-2); return tail.length ? tail.reduce((a, p) => a + p[1], 0) / tail.length : null; };
    const finals = series.map(([name, pts]) => `${name}: <b>${lastOf(pts) == null ? "–" : lastOf(pts).toFixed(3)}</b>`).join(" · ");
    fig.$live.hidden = false;
    fig.$live.innerHTML = (met === "ent" ? "<b>How sure is the selector?</b> Its entropy is a measure of how spread out its odds are. At the maximum it would treat all eight codes as equally likely. " : met === "pick" ? "<b>Does the decoder's chosen code match the best code?</b> With two codes drawn from eight, matching by luck gives 0.25. " : "<b>Lower is better.</b> ") + `Latest values (average of the last 1,200 updates shown): ${finals}. <span class="muted">These are training-time diagnostics. The gates, not this chart, decide anything.</span>`;
  }
  draw();
});
})();
