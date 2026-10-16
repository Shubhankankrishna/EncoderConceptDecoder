/* Start + Basics figures: overview, tokens, nextTok, loss, descent, attention, block */
(function () {
const { H, S, ui, loop, playBtn, stepper, clamp, lerp, ease } = G;

/* ------------------------------------------------------------ overview: the six experiments */
G.reg("overview", fig => {
  const sv = fig.svg(920, 366, { label: "Map of the six experiments S1 to S6" });
  const C = [
    { k: "S1", t: "Foundation", d: "Read in chunks and\nguess the next concept", c: "enc", q: "Q1", st: "finished", sc: "good" },
    { k: "S2", t: "H-NCP", d: "Keep several sharp\nguesses, not one blur", c: "con", q: "Q3", st: "running", sc: "accent" },
    { k: "S3", t: "Refit", d: "Let the word writer\nadapt to sharp guesses", c: "dec", q: "Q4", st: "designed", sc: "dec" },
    { k: "S4", t: "Post-training", d: "Learn from a right\nor wrong checker", c: "sel", q: "Q5", st: "designed", sc: "dec" },
    { k: "S5", t: "Search", d: "Plan steps ahead\nwithout writing words", c: "code", q: "Q6", st: "designed", sc: "dec" },
  ];
  const W = 150, GAP = 34, Y = 18, HH = 200;
  const xs = C.map((_, i) => 10 + i * (W + GAP));
  const dots = [];
  C.forEach((c, i) => {
    const g = sv.g({ tabindex: 0, role: "link", "aria-label": `${c.k} ${c.t}, go to chapter`, style: "cursor:pointer" });
    sv.rect(xs[i], Y, W, HH, { fill: "var(--panel)", stroke: `var(--${c.c})`, sw: 1.6, rx: 14 }, g);
    sv.rect(xs[i], Y, W, 56, { fill: `var(--${c.c})`, stroke: "none", rx: 14 }, g);
    sv.rect(xs[i], Y + 40, W, 16, { fill: `var(--${c.c})`, stroke: "none", rx: 0 }, g);
    sv.text(xs[i] + 14, Y + 20, c.k, { size: 22, weight: 700, fill: "var(--bg)" }, g);
    sv.text(xs[i] + 14, Y + 43, c.t, { size: 14, weight: 600, fill: "var(--bg)" }, g);
    sv.text(xs[i] + 14, Y + 92, c.d, { size: 12.5, fill: "var(--ink2)" }, g);
    sv.text(xs[i] + 14, Y + 150, "queue " + c.q, { size: 11.5, mono: true, fill: "var(--ink3)" }, g);
    sv.rect(xs[i] + 12, Y + 162, 82, 22, { fill: c.sc === "good" ? "var(--good-s)" : c.sc === "accent" ? "var(--accent-s)" : "var(--dec-s)", stroke: "none", rx: 11 }, g);
    sv.text(xs[i] + 53, Y + 173.5, c.st, { size: 11.5, weight: 600, anchor: "middle", fill: c.sc === "good" ? "var(--good)" : c.sc === "accent" ? "var(--accent)" : "var(--dec)" }, g);
    const go = () => G.goto(c.k.toLowerCase());
    g.addEventListener("click", go); g.addEventListener("keydown", e => { if (e.key === "Enter") go(); });
    if (i < 4) {
      const x1 = xs[i] + W + 3, x2 = xs[i + 1] - 3, y = Y + 100;
      sv.line(x1, y, x2, y, { stroke: "var(--ink3)", sw: 1.8, arrow: "ink3" });
      dots.push({ x1, x2, y, el: sv.circle(x1, y, 3.4, { fill: "var(--accent)" }) });
    }
  });
  /* S6 branch */
  const y6 = 262, g6 = sv.g({ tabindex: 0, role: "link", "aria-label": "S6 Continual learning, go to chapter", style: "cursor:pointer" });
  sv.rect(10, y6, 560, 84, { fill: "var(--panel)", stroke: "var(--loss)", sw: 1.6, rx: 14 }, g6);
  sv.rect(10, y6, 78, 84, { fill: "var(--loss)", stroke: "none", rx: 14 }, g6);
  sv.rect(60, y6, 28, 84, { fill: "var(--loss)", stroke: "none", rx: 0 }, g6);
  sv.text(49, y6 + 34, "S6", { size: 24, weight: 700, anchor: "middle", fill: "var(--bg)" }, g6);
  sv.text(49, y6 + 58, "queue Q2", { size: 9.5, mono: true, anchor: "middle", fill: "var(--bg)" }, g6);
  sv.text(104, y6 + 24, "Continual learning", { size: 15, weight: 600 }, g6);
  sv.text(104, y6 + 47, "Learn new skills one after another\nwithout forgetting the old ones", { size: 12.5, fill: "var(--ink2)" }, g6);
  sv.rect(432, y6 + 52, 82, 22, { fill: "var(--dec-s)", stroke: "none", rx: 11 }, g6);
  sv.text(473, y6 + 63.5, "designed", { size: 11.5, weight: 600, anchor: "middle", fill: "var(--dec)" }, g6);
  g6.addEventListener("click", () => G.goto("s6")); g6.addEventListener("keydown", e => { if (e.key === "Enter") G.goto("s6"); });
  sv.path(`M ${xs[0] + 75} ${Y + HH + 4} L ${xs[0] + 75} ${y6 - 4}`, { stroke: "var(--ink3)", sw: 1.8, arrow: "ink3", dash: "5 4" });
  sv.text(xs[0] + 86, (Y + HH + y6) / 2, "starts from the S1 model, does not wait for S2 to S5", { size: 12, fill: "var(--ink2)" });
  sv.text(600, y6 + 8, "Real run order on the one GPU", { size: 11.5, weight: 600, fill: "var(--ink3)" });
  sv.text(600, y6 + 32, "Q1 → Q3a → Q2 → Q3b →\nQ4 → Q5 → Q6 → QF", { size: 13, mono: true });
  sv.text(600, y6 + 68, "Q3a and Q3b are the two halves of S2.", { size: 11.5, fill: "var(--ink2)" });
  fig.playing = !G.REDUCED; fig.$ui.append(playBtn(fig));
  loop(fig, t => dots.forEach((d, i) => { const u = (t * 0.32 + i * 0.23) % 1; d.el.setAttribute("cx", d.x1 + (d.x2 - d.x1) * u); d.el.setAttribute("opacity", Math.sin(u * Math.PI)); }));
});

/* ------------------------------------------------------------ tokens */
G.reg("tokens", fig => {
  const inp = H("input", { type: "text", class: "txt", value: "The cat sat on the mat.", maxlength: 90, "aria-label": "Text to cut into tokens" });
  fig.$ui.append(H("label", { class: "rng" }, "Your text", inp));
  const box = H("div", { class: "chipbox", "aria-live": "polite" }), sum = H("div", { class: "note" });
  fig.$stage.append(box, sum);
  const id = s => { let h = 2166136261; for (const ch of s) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619); } return (h >>> 0) % 1990 + 58; };
  function cut(text) {
    const parts = text.match(/[A-Za-z]+|\d+|[^\sA-Za-z\d]/g) || [], out = [];
    parts.forEach(w => { if (/^[A-Za-z]+$/.test(w) && w.length > 6) { const k = Math.ceil(w.length / 2); out.push(w.slice(0, k), "·" + w.slice(k)); } else out.push(w); });
    return out;
  }
  function draw() {
    const toks = cut(inp.value); box.innerHTML = "";
    toks.forEach(t => box.append(H("span", { class: "tk" }, H("b", {}, t), H("i", {}, "id " + id(t.replace("·", ""))))));
    sum.textContent = `${inp.value.length} characters became ${toks.length} tokens. The model only ever sees the ${toks.length} ID numbers.`;
  }
  inp.addEventListener("input", draw); draw();
});

/* ------------------------------------------------------------ nextTok */
G.reg("nextTok", fig => {
  const CTX = {
    cat: { p: "The cat sat on the", o: [["mat", .45], ["floor", .19], ["sofa", .12], ["roof", .08], ["bed", .07], ["dog", .05], ["moon", .03], ["idea", .01]], label: "Sentence" },
    sum: { p: "2 + 2 =", o: [["4", .9], ["5", .03], ["3", .02], ["22", .02], ["four", .02], ["0", .01]], label: "Arithmetic" },
    fork: { p: "At the fork, turn", o: [["left", .48], ["right", .47], ["around", .03], ["up", .02]], label: "Fork" },
  };
  let key = "cat", T = 1, counts = {}, total = 0;
  const sv = fig.svg(720, 300, { label: "Bar chart of next-token probabilities" });
  const seg = ui.seg(Object.entries(CTX).map(([k, v]) => [k, v.label]), key, k => { key = k; counts = {}; total = 0; draw(); });
  const tr = ui.range("boldness", 0.2, 2, 0.05, 1, v => { T = v; draw(); }, v => v.toFixed(2));
  const r = ui.readout();
  fig.$ui.append(seg, tr, ui.btn("Draw 1", () => sample(1), "pri"), ui.btn("Draw 100", () => sample(100)), ui.btn("Reset", () => { counts = {}; total = 0; draw(); }), r);
  const dist = () => { const o = CTX[key].o.map(([w, p]) => [w, Math.pow(p, 1 / T)]); const z = o.reduce((a, b) => a + b[1], 0); return o.map(([w, p]) => [w, p / z]); };
  function sample(n) { const d = dist(); for (let i = 0; i < n; i++) { let u = Math.random(), acc = 0; for (const [w, p] of d) { acc += p; if (u <= acc) { counts[w] = (counts[w] || 0) + 1; total++; break; } } } draw(); }
  function draw() {
    sv.clear(); const d = dist(), rowH = 31, x0 = 130, bw = 380, mx = Math.max(...d.map(a => a[1]));
    sv.text(14, 22, CTX[key].p, { size: 16, weight: 600 }); sv.rect(14 + CTX[key].p.length * 8.6 + 6, 12, 40, 20, { fill: "var(--code-s)", stroke: "var(--code)", rx: 5 });
    sv.text(14 + CTX[key].p.length * 8.6 + 26, 22, "?", { size: 15, weight: 700, anchor: "middle", fill: "var(--code)" });
    sv.text(x0 + bw + 70, 46, "your draws", { size: 11.5, fill: "var(--ink3)", mono: true });
    d.forEach(([w, p], i) => {
      const y = 62 + i * rowH, best = p === mx;
      sv.text(x0 - 10, y + 10, w, { size: 14, anchor: "end", weight: best ? 650 : 400 });
      sv.rect(x0, y, bw, 20, { fill: "var(--bg2)", stroke: "none", rx: 5 });
      sv.rect(x0, y, Math.max(2, bw * p / mx * 0.86), 20, { fill: best ? "var(--con)" : "var(--enc)", stroke: "none", rx: 5, opacity: best ? 1 : .8 });
      sv.text(x0 + Math.max(2, bw * p / mx * 0.86) + 8, y + 10, (p * 100).toFixed(p < .1 ? 1 : 0) + "%", { size: 12.5, mono: true, fill: "var(--ink2)" });
      const c = counts[w] || 0; if (c) { sv.text(x0 + bw + 70, y + 10, "●".repeat(Math.min(c, 12)) + (c > 12 ? "+" : "") + "  " + c, { size: 11, mono: true, fill: "var(--accent)" }); }
    });
    r.set(`${total} draws · most likely token marked in teal`);
  }
  draw();
});

/* ------------------------------------------------------------ loss */
G.reg("loss", fig => {
  const W = 720, Hh = 300, sv = fig.svg(W, Hh, { label: "Plot of loss equals minus log probability" });
  const X0 = 70, X1 = 650, Y0 = 250, Y1 = 30, yMax = 5.5;
  const px = p => X0 + (X1 - X0) * p, py = l => Y0 - (Y0 - Y1) * Math.min(l, yMax) / yMax;
  let p = 0.12;
  const sl = ui.range("probability given to the true token", 1, 99, 1, 12, v => { p = v / 100; draw(); }, v => v + "%");
  const r = ui.readout(); fig.$ui.append(sl, r);
  const dyn = [];
  [0, 1, 2, 3, 4, 5].forEach(l => { sv.line(X0, py(l), X1, py(l), { stroke: "var(--line)", sw: 1 }); sv.text(X0 - 10, py(l), String(l), { size: 12, anchor: "end", fill: "var(--ink3)", mono: true }); });
  [0, .25, .5, .75, 1].forEach(v => sv.text(px(v), Y0 + 18, Math.round(v * 100) + "%", { size: 12, anchor: "middle", fill: "var(--ink3)", mono: true }));
  sv.text(X0 - 48, (Y0 + Y1) / 2, "loss (surprise)", { size: 12, fill: "var(--ink2)", anchor: "middle" }).setAttribute("transform", `rotate(-90 ${X0 - 48} ${(Y0 + Y1) / 2})`);
  sv.text((X0 + X1) / 2, Hh - 8, "probability the model gave to the real next token", { size: 12.5, fill: "var(--ink2)", anchor: "middle" });
  let d = ""; for (let i = 0; i <= 200; i++) { const q = 0.004 + (1 - 0.004) * i / 200; d += (i ? "L" : "M") + px(q).toFixed(1) + " " + py(-Math.log(q)).toFixed(1); }
  sv.path(d, { stroke: "var(--loss)", sw: 2.4 });
  const stem = sv.line(0, 0, 0, 0, { stroke: "var(--ink3)", sw: 1.2, dash: "4 4" }), dot = sv.circle(0, 0, 7, { fill: "var(--loss)", stroke: "var(--panel)", sw: 2.5 }), lab = sv.text(0, 0, "", { size: 13, weight: 650, halo: true });
  function draw() {
    const l = -Math.log(p), x = px(p), y = py(l);
    stem.setAttribute("x1", x); stem.setAttribute("x2", x); stem.setAttribute("y1", Y0); stem.setAttribute("y2", y);
    dot.setAttribute("cx", x); dot.setAttribute("cy", y);
    lab.setAttribute("x", Math.min(x + 12, X1 - 90)); lab.setAttribute("y", Math.max(y - 16, Y1 + 8)); lab.textContent = "loss " + l.toFixed(2);
    r.set(`confidence <b>${Math.round(p * 100)}%</b> → loss <b>${l.toFixed(2)}</b>` + (p < 0.03 ? " · a confident wrong answer" : p > 0.9 ? " · barely surprised" : ""));
  }
  draw();
});

/* ------------------------------------------------------------ descent */
G.reg("descent", fig => {
  const W = 720, Hh = 300, sv = fig.svg(W, Hh, { label: "A ball rolling down a loss curve" });
  const X0 = 40, X1 = 680, Y0 = 262, Y1 = 28, XM = 6;
  const L = x => 0.2 * (x - 3.2) ** 2 + 0.5 * Math.sin(2.3 * x) + 1.2, dL = x => 0.4 * (x - 3.2) + 1.15 * Math.cos(2.3 * x);
  let lo = 1e9, hi = -1e9; for (let i = 0; i <= 300; i++) { const v = L(XM * i / 300); lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const px = x => X0 + (X1 - X0) * x / XM, py = v => Y0 - (Y0 - Y1) * (v - lo + 0.1) / (hi - lo + 0.3);
  let d = ""; for (let i = 0; i <= 300; i++) { const x = XM * i / 300; d += (i ? "L" : "M") + px(x).toFixed(1) + " " + py(L(x)).toFixed(1); }
  sv.rect(X0, Y0, X1 - X0, 3, { fill: "var(--line2)", stroke: "none", rx: 1 });
  sv.path(d + `L${X1} ${Y0} L${X0} ${Y0} Z`, { fill: "var(--enc-s)", stroke: "none", opacity: .6 });
  sv.path(d, { stroke: "var(--enc)", sw: 2.4 });
  sv.text(X0, 14, "loss", { size: 12, fill: "var(--ink3)" }); sv.text(X1, Y0 + 18, "one parameter's value", { size: 12, fill: "var(--ink3)", anchor: "end" });
  const hit = sv.rect(X0, Y1 - 10, X1 - X0, Y0 - Y1 + 10, { fill: "transparent", stroke: "none", rx: 0, attrs: { style: "cursor:crosshair" } });
  const trail = sv.g({}), ball = sv.circle(0, 0, 9, { fill: "var(--sel)", stroke: "var(--panel)", sw: 3 });
  const hint = sv.text(W / 2, 150, "", { size: 13, fill: "var(--ink2)", anchor: "middle" });
  let x = 0.6, steps = 0, running = false, acc = 0, lr = 0.15, hist = [];
  const r = ui.readout();
  const sl = ui.range("learning rate", 0.02, 1, 0.01, 0.15, v => { lr = v; }, v => v.toFixed(2));
  const place = nx => { x = clamp(nx, 0, XM); steps = 0; hist = []; trail.innerHTML = ""; running = false; sync(); upd(); };
  fig.$ui.append(sl, ui.btn("Step", () => { running = false; step(); }, "pri"), ui.btn("Run", () => { running = true; acc = 0; G.kick(); }), ui.btn("Reset", () => place(0.6)), r);
  fig.playing = true;
  function step() {
    const g = dL(x); hist.push(x); x = clamp(x - lr * g, 0, XM); steps++;
    S("circle", { cx: px(hist[hist.length - 1]), cy: py(L(hist[hist.length - 1])), r: 3.4, fill: "var(--sel)", opacity: .45 }, trail); sync(); upd();
  }
  function upd() { ball.setAttribute("cx", px(x)); ball.setAttribute("cy", py(L(x))); }
  function sync() {
    const g = Math.abs(dL(x)); const settled = g < 0.03 && steps > 0;
    r.set(`steps <b>${steps}</b> · loss <b>${L(x).toFixed(2)}</b> · slope <b>${g.toFixed(2)}</b>${settled ? " · settled" : ""}`);
    hint.textContent = steps === 0 ? "click anywhere on the curve to move the ball" : (lr > 0.7 && steps > 3 && !settled ? "big steps overshoot the valley and bounce" : "");
    if (settled) running = false;
  }
  hit.addEventListener("click", e => { const b = sv.svg.getBoundingClientRect(); place(((e.clientX - b.left) / b.width * W - X0) / (X1 - X0) * XM); });
  loop(fig, (t, dt) => { if (!running) return; acc += dt; if (acc > 0.16) { acc = 0; step(); if (steps > 90) running = false; } });
  upd(); sync();
});

/* ------------------------------------------------------------ attention */
G.reg("attention", fig => {
  const words = ["The", "cat", "that", "chased", "the", "mouse", "was", "tired"];
  const A = [[1], [.45, .55], [.1, .7, .2], [0, .3, .3, .4].map((v, i) => [0, .45, .3, .25][i]), [0, .15, .1, .4, .35], [0, .1, 0, .3, .5, .1], [0, .6, .05, .1, 0, .15, .1], [0, .4, 0, 0, 0, .1, .4, .1]];
  const sv = fig.svg(760, 300, { label: "Attention arcs from the chosen word to earlier words" });
  let sel = 6;
  const r = ui.readout(); fig.$ui.append(r);
  const xs = words.map((_, i) => 60 + i * 92), Y = 214;
  const arcs = sv.g({}), boxes = [];
  words.forEach((w, i) => {
    const g = sv.g({ tabindex: 0, role: "button", "aria-label": "Show attention for " + w, style: "cursor:pointer" });
    const b = sv.rect(xs[i] - 38, Y, 76, 40, { fill: "var(--panel)", stroke: "var(--line2)", sw: 1.4, rx: 10 }, g);
    sv.text(xs[i], Y + 20, w, { size: 15, weight: 600, anchor: "middle" }, g);
    const wt = sv.text(xs[i], Y + 58, "", { size: 11.5, anchor: "middle", mono: true, fill: "var(--ink2)" }, g);
    boxes.push({ b, wt }); g.addEventListener("click", () => { sel = i; draw(); }); g.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sel = i; draw(); } });
  });
  sv.text(380, 24, "causal attention: a word can only look at itself and the words before it", { size: 13.5, anchor: "middle", fill: "var(--ink2)" });
  function draw() {
    arcs.innerHTML = ""; const row = A[sel];
    boxes.forEach(({ b, wt }, i) => {
      const fut = i > sel; b.style.stroke = i === sel ? "var(--accent)" : "var(--line2)"; b.style.fill = fut ? "var(--bg2)" : i === sel ? "var(--accent-s)" : "var(--panel)"; b.setAttribute("stroke-width", i === sel ? 2.4 : 1.4);
      b.parentNode.style.opacity = fut ? .45 : 1; wt.textContent = !fut && row[i] ? Math.round(row[i] * 100) + "%" : fut ? "hidden" : "";
    });
    row.forEach((w, j) => {
      if (!w || j === sel) return;
      const x1 = xs[sel], x2 = xs[j], mid = (x1 + x2) / 2, h = 40 + Math.abs(x1 - x2) * 0.2;
      S("path", { d: `M ${x1} ${Y} Q ${mid} ${Y - h * 1.7} ${x2} ${Y}`, fill: "none", "stroke-width": 1.5 + w * 14, "stroke-linecap": "round", opacity: .25 + w * 1.1, style: "stroke:var(--con)" }, arcs);
    });
    const top = row.map((w, j) => [w, j]).filter(a => a[1] !== sel).sort((a, b) => b[0] - a[0])[0];
    r.set(`<b>${words[sel]}</b> pays most attention to <b>${top ? words[top[1]] : "itself"}</b>`);
  }
  draw();
});

/* ------------------------------------------------------------ block */
G.reg("block", fig => {
  const sv = fig.svg(760, 340, { label: "One transformer block: attention then feed-forward, each added back to the input" });
  const cols = [140, 240, 340, 440];
  const stepsTxt = [
    { label: "1 Vectors in", title: "Vectors in.", text: "Each position arrives as a vector of 128 numbers. Here are four positions, drawn as colour bars." },
    { label: "2 Attention", title: "Attention: look back.", text: "Each position gathers a weighted blend of the earlier positions' vectors. Only backwards, never forwards." },
    { label: "3 Add back", title: "Add back.", text: "The result is added to the vector the position already had, so nothing it knew is thrown away." },
    { label: "4 Feed-forward", title: "Feed-forward: think.", text: "Each position separately runs its vector through a small network, and again adds the result back." },
  ];
  const glyph = (x, y, hue, seed) => { const g = sv.g({}); sv.rect(x - 21, y - 18, 42, 36, { fill: "var(--panel)", stroke: "var(--line2)", sw: 1, rx: 7 }, g); for (let k = 0; k < 6; k++) sv.rect(x - 16 + k * 5.6, y - 13, 4.4, 26, { fill: `var(--${hue})`, stroke: "none", rx: 1.6, opacity: .3 + ((k * 41 + seed * 17) % 65) / 100 }, g); return g; };
  const gin = cols.map((x, i) => glyph(x, 318, "dec", i)), gout = cols.map((x, i) => glyph(x, 32, "dec", i + 7));
  const wire = (x, y1, y2) => sv.line(x, y1, x, y2, { stroke: "var(--ink3)", sw: 1.6 });
  cols.forEach(x => { wire(x, 300, 276); wire(x, 212, 200); wire(x, 178, 168); wire(x, 120, 106); wire(x, 84, 50); });
  const att = sv.g({}), ff = sv.g({}), bar1 = sv.g({}), bar2 = sv.g({}), links = sv.g({});
  sv.rect(100, 212, 400, 64, { fill: "var(--enc-s)", stroke: "var(--enc)", sw: 1.6, rx: 12 }, att); sv.text(300, 228, "Attention · look back", { size: 15, weight: 650, anchor: "middle" }, att);
  sv.rect(100, 120, 400, 48, { fill: "var(--con-s)", stroke: "var(--con)", sw: 1.6, rx: 12 }, ff); sv.text(300, 144, "Feed-forward · think", { size: 15, weight: 650, anchor: "middle" }, ff);
  sv.rect(100, 178, 400, 22, { fill: "var(--bg2)", stroke: "var(--line2)", sw: 1.2, rx: 8 }, bar1); sv.text(300, 189, "+ add back", { size: 12.5, weight: 600, anchor: "middle", fill: "var(--ink2)" }, bar1);
  sv.rect(100, 84, 400, 22, { fill: "var(--bg2)", stroke: "var(--line2)", sw: 1.2, rx: 8 }, bar2); sv.text(300, 95, "+ add back", { size: 12.5, weight: 600, anchor: "middle", fill: "var(--ink2)" }, bar2);
  const by1 = sv.path("M 470 318 H 545 V 184 H 504", { stroke: "var(--ink3)", sw: 1.8, dash: "6 4", arrow: "ink3" });
  const by2 = sv.path("M 504 194 H 575 V 95 H 504", { stroke: "var(--ink3)", sw: 1.8, dash: "6 4", arrow: "ink3" });
  sv.text(552, 262, "shortcut", { size: 11.5, fill: "var(--ink3)" }); sv.text(582, 140, "shortcut", { size: 11.5, fill: "var(--ink3)" });
  [[0, 1], [1, 3], [2, 3], [0, 3]].forEach(([a, b]) => { const x1 = cols[b], x2 = cols[a]; sv.path(`M ${x1} 272 Q ${(x1 + x2) / 2} ${272 - 24 - Math.abs(x1 - x2) * 0.1} ${x2} 272`, { stroke: "var(--enc)", sw: 2, arrow: "enc" }, links); });
  sv.text(14, 36, "× 10 blocks\nin our models", { size: 12, fill: "var(--ink3)" });
  sv.text(630, 318, "vectors in", { size: 12, fill: "var(--ink3)" }); sv.text(630, 32, "vectors out", { size: 12, fill: "var(--ink3)" });
  const dots = cols.map(x => sv.circle(x, 318, 5, { fill: "var(--accent)" }));
  const hi = i => {
    gin.forEach(g => g.style.opacity = i === 0 ? 1 : .55);
    att.style.opacity = i === 1 ? 1 : .35; links.style.opacity = i === 1 ? 1 : 0;
    bar1.style.opacity = by1.style.opacity = i === 2 ? 1 : .35; ff.style.opacity = i === 3 ? 1 : .35; bar2.style.opacity = by2.style.opacity = i === 3 ? 1 : .35;
  };
  stepper(fig, stepsTxt, hi, { ms: 3600 });
  loop(fig, t => dots.forEach((d, k) => { const u = (t * 0.22 + k * 0.08) % 1; d.setAttribute("cy", 318 - 290 * u); }));
});
})();
