/* Chapter 7 (S5) figures: puct, costCalc, rhTimeline, compound, answerLoop */
(function () {
const { H, S, ui, loop, stepper, clamp, lerp, rng, along } = G;

/* ------------------------------------------------------------ puct: real PUCT on a toy tree */
G.reg("puct", fig => {
  const W = 900, Hh = 430, sv = fig.svg(W, Hh, { label: "A search tree with four candidate codes, visit counts and PUCT scores" });
  const TRUE = [0.30, 0.65, 0.50, 0.20], PRI = [0.45, 0.20, 0.25, 0.10], NAMES = ["code 1", "code 2", "code 3", "code 4"], C = 1.5;
  let N, Wv, rf, last = null, sims = 0;
  const reset = () => { N = [0, 0, 0, 0]; Wv = [0, 0, 0, 0]; rf = rng(77); last = null; sims = 0; draw(); };
  const score = (a, tot) => (N[a] ? Wv[a] / N[a] : 0) + C * PRI[a] * Math.sqrt(tot) / (1 + N[a]);
  function sim() {
    const tot = N.reduce((a, b) => a + b, 0), sc = [0, 1, 2, 3].map(a => score(a, tot)); let a = 0; sc.forEach((v, i) => { if (v > sc[a] + 1e-12) a = i; });
    const o = rf() < TRUE[a] ? 1 : 0; N[a]++; Wv[a] += o; sims++; last = { a, o, sc, tot };
  }
  fig.$ui.append(ui.btn("Run 1 simulation", () => { sim(); draw(); }, "pri"), ui.btn("Run 10", () => { for (let i = 0; i < 10; i++) sim(); draw(); }), ui.btn("Run 100", () => { for (let i = 0; i < 100; i++) sim(); draw(); }), ui.btn("Reset", reset));
  const dyn = sv.g({});
  function draw() {
    dyn.innerHTML = ""; const tot = N.reduce((a, b) => a + b, 0), mx = Math.max(1, ...N), xs = [120, 330, 540, 750];
    sv.rect(W / 2 - 70, 24, 140, 44, { fill: "var(--ink)", stroke: "none", rx: 12, opacity: .9 }, dyn); sv.text(W / 2, 46, "current position", { size: 13, weight: 650, anchor: "middle", fill: "var(--bg)" }, dyn);
    xs.forEach((x, a) => {
      const sel = last && last.a === a; sv.path(`M ${W / 2} 68 C ${W / 2} 110 ${x} 100 ${x} 134`, { stroke: sel ? "var(--accent)" : "var(--ink3)", sw: 1.5 + 9 * N[a] / mx, opacity: N[a] ? .55 : .3 }, dyn);
      sv.rect(x - 78, 134, 156, 262, { fill: "var(--panel)", stroke: sel ? "var(--accent)" : "var(--line2)", sw: sel ? 2.8 : 1.4, rx: 14 }, dyn);
      sv.text(x, 156, NAMES[a], { size: 14.5, weight: 650, anchor: "middle" }, dyn);
      const rows = [["prior P", PRI[a].toFixed(2), "var(--sel)", PRI[a] / 0.5], ["visits N", String(N[a]), "var(--code)", N[a] / mx], ["average result Q", N[a] ? (Wv[a] / N[a]).toFixed(2) : "–", "var(--good)", N[a] ? Wv[a] / N[a] : 0]];
      rows.forEach(([l, v, c, f], i) => { const y = 176 + i * 52; sv.text(x - 66, y + 4, l, { size: 11.5, fill: "var(--ink3)" }, dyn); sv.rect(x - 66, y + 14, 132, 12, { fill: "var(--bg2)", stroke: "none", rx: 4 }, dyn); sv.rect(x - 66, y + 14, Math.max(2, 132 * clamp(f, 0, 1)), 12, { fill: c, stroke: "none", rx: 4 }, dyn); sv.text(x + 66, y + 4, v, { size: 13, weight: 650, mono: true, anchor: "end" }, dyn); });
      sv.text(x, 342, "PUCT score", { size: 11.5, anchor: "middle", fill: "var(--ink3)" }, dyn); sv.text(x, 366, score(a, tot).toFixed(2), { size: 20, weight: 700, mono: true, anchor: "middle", fill: sel ? "var(--accent)" : "var(--ink)" }, dyn);
      if (N[a] === mx && tot > 0) sv.text(x, 388, "most visited", { size: 11.5, weight: 700, anchor: "middle", fill: "var(--good)" }, dyn);
    });
    sv.text(W - 16, 20, `simulations: ${sims}`, { size: 12.5, mono: true, anchor: "end", fill: "var(--ink2)" }, dyn);
    fig.$live.hidden = false;
    fig.$live.innerHTML = last ? `<b>Simulation ${sims}:</b> the search picked <b>${NAMES[last.a]}</b> because it had the highest score (${last.sc[last.a].toFixed(2)}). It tried that branch, got a ${last.o ? "right" : "wrong"} result, and updated that branch's average.` : "<b>Four candidate codes.</b> The selector's prior favours code 1, but each branch hides a different true success rate (30%, 65%, 50%, 20%). The search does not know them. It finds out by trying. Run some simulations.";
    if (sims >= 100) fig.$live.innerHTML += ` After ${sims} simulations the prior's favourite is no longer the most visited. <span class="muted">Ties at the very first simulation go to the lowest-numbered code, a property of the formula that the project notes.</span>`;
  }
  reset();
});

/* ------------------------------------------------------------ costCalc */
G.reg("costCalc", fig => {
  const W = 900, Hh = 360, sv = fig.svg(W, Hh, { label: "Block-position cost of different search styles" });
  let B = 16; const HOPS = 6;
  fig.$ui.append(ui.range("tries per real step", 2, 64, 2, 16, v => { B = v; draw(); }, v => String(v)));
  const dyn = sv.g({});
  function draw() {
    dyn.innerHTML = ""; const ROWS = [
      ["RH(1) · decoder-free tries", HOPS * (5 * B + 28), "var(--con)", `${HOPS} × (${B} × 5 + 28)`],
      ["IMAG · imagine all, decode once", 5 * B * HOPS + 24 * HOPS, "var(--con)", `${HOPS * B} × 5 + ${HOPS} × 24`],
      ["G(1) · every try decoded", HOPS * 29 * B, "var(--enc)", `${HOPS} × ${B} × 29`],
      ["Token tree · every try decoded", HOPS * 28 * B, "var(--dec)", `${HOPS} × ${B} × 28`],
    ];
    const mx = Math.max(...ROWS.map(r => r[1])), X0 = 270, X1 = 780;
    sv.text(20, 24, `Work for a ${HOPS}-hop answer, in "block-positions" (one block looking at one position)`, { size: 13.5, weight: 650 }, dyn);
    ROWS.forEach(([n, v, c, f], i) => { const y = 54 + i * 62; sv.text(X0 - 12, y + 12, n, { size: 13, weight: 650, anchor: "end" }, dyn); sv.text(X0 - 12, y + 30, f, { size: 11, mono: true, anchor: "end", fill: "var(--ink3)" }, dyn); sv.rect(X0, y, X1 - X0, 26, { fill: "var(--bg2)", stroke: "none", rx: 6 }, dyn); sv.rect(X0, y, Math.max(3, (X1 - X0) * v / mx), 26, { fill: c, stroke: "none", rx: 6 }, dyn); sv.text(X0 + (X1 - X0) * v / mx + 8, y + 13, v.toLocaleString("en-US"), { size: 13, mono: true, weight: 650 }, dyn); });
    sv.rect(20, 306, 860, 40, { fill: "var(--bg2)", stroke: "var(--line2)", rx: 10 }, dyn);
    sv.text(34, 326, `Per try, a decoder-free child costs 5 and a token child costs 28: about ${(28 / 5).toFixed(1)}× cheaper. At larger scale the gap would grow to roughly 15×.`, { size: 12.5, fill: "var(--ink2)" }, dyn);
    fig.$live.hidden = false; fig.$live.innerHTML = "<b>This is arithmetic from the design, not a measurement.</b> On the real laptop GPU at this tiny width, each step is probably limited by launch overhead rather than arithmetic, so the real time ratio could be close to 1×. For that reason every comparison in S5 uses measured wall-clock time, and a speed test comes first.";
  }
  draw();
});

/* ------------------------------------------------------------ rhTimeline: when does the decoder run? */
G.reg("rhTimeline", fig => {
  const W = 900, Hh = 410, sv = fig.svg(W, Hh, { label: "When the decoder runs for each search style across eight hops" });
  let R = 2; const HOPS = 8;
  fig.$ui.append(ui.range("G(R): decode every R steps", 1, 4, 1, 2, v => { R = v; draw(); }, v => String(v)));
  const dyn = sv.g({});
  const ROWS = () => [
    { n: "RH(1)", d: "imagine one step, then decode it, every hop", dec: h => 1, img: h => 1 },
    { n: `G(${R})`, d: `imagine ${R} step${R > 1 ? "s" : ""}, then decode them`, dec: h => (h + 1) % R === 0 || h === HOPS - 1 ? 1 : 0, img: h => 1 },
    { n: "IMAG", d: "imagine everything, decode once at the end", dec: h => h === HOPS - 1 ? 1 : 0, img: h => 1, final: true },
    { n: "Token tree", d: "every try is decoded", dec: h => 1, img: h => 0, all: true },
  ];
  function draw() {
    dyn.innerHTML = ""; const X0 = 220, CW = 80;
    for (let h = 0; h < HOPS; h++) sv.text(X0 + h * CW + CW / 2, 26, "hop " + (h + 1), { size: 12, anchor: "middle", fill: "var(--ink3)" }, dyn);
    ROWS().forEach((r, i) => {
      const y = 44 + i * 82; sv.text(14, y + 22, r.n, { size: 14.5, weight: 700 }, dyn); sv.text(14, y + 42, r.d, { size: 11, fill: "var(--ink3)" }, dyn);
      let decs = 0, imgs = 0;
      for (let h = 0; h < HOPS; h++) {
        const x = X0 + h * CW; if (r.img(h)) { for (let k = 0; k < 3; k++) sv.circle(x + 18 + k * 14, y + 14, 5, { fill: "var(--con)" }, dyn); imgs++; }
        const dd = r.dec(h); if (dd) { sv.rect(x + 8, y + 28, CW - 16, 28, { fill: r.all ? "var(--dec)" : "var(--enc)", stroke: "none", rx: 6 }, dyn); sv.text(x + CW / 2, y + 42, "decode", { size: 11, weight: 650, anchor: "middle", fill: "var(--bg)" }, dyn); decs++; }
      }
      sv.text(X0 + HOPS * CW + 12, y + 28, `${decs} decode${decs === 1 ? "" : "s"}`, { size: 13, weight: 700, mono: true }, dyn);
    });
    sv.circle(X0, 392, 5, { fill: "var(--con)" }, dyn); sv.text(X0 + 10, 392, "imagined tries (cheap)", { size: 11.5, fill: "var(--ink2)" }, dyn); sv.rect(X0 + 190, 385, 22, 14, { fill: "var(--enc)", rx: 3, stroke: "none" }, dyn); sv.text(X0 + 218, 392, "decoder writes tokens and re-reads them", { size: 11.5, fill: "var(--ink2)" }, dyn);
    fig.$live.hidden = false; fig.$live.innerHTML = "<b>The decoder is the expensive part.</b> RH(1) runs it once per real step and never inside the lookahead. G(R) decodes every R steps. IMAG decodes only at the very end, which is cheapest and trusts the imagination the most. The token tree runs it on every try. <span class=\"muted\">Schematic: tries per step are drawn as three dots, and actual budgets are set by wall-clock time.</span>";
  }
  draw();
});

/* ------------------------------------------------------------ compound: why imagined depth must stay short */
G.reg("compound", fig => {
  const W = 900, Hh = 340, sv = fig.svg(W, Hh, { label: "Chance that an imagined path stays accurate for N steps" });
  let p = 0.95; fig.$ui.append(ui.range("accuracy of one imagined step", 0.8, 0.995, 0.005, 0.95, v => { p = v; draw(); }, v => (v * 100).toFixed(1) + "%"));
  const dyn = sv.g({});
  function draw() {
    dyn.innerHTML = ""; const X0 = 70, X1 = 860, Y0 = 270, Y1 = 40, NMAX = 20, px = n => X0 + (X1 - X0) * (n - 0.5) / NMAX, py = v => Y0 - (Y0 - Y1) * v;
    [0, .25, .5, .75, 1].forEach(v => { sv.line(X0, py(v), X1, py(v), { stroke: "var(--line)", sw: 1 }, dyn); sv.text(X0 - 8, py(v), Math.round(v * 100) + "%", { size: 11, mono: true, anchor: "end", fill: "var(--ink3)" }, dyn); });
    sv.line(X0, py(0.9), X1, py(0.9), { stroke: "var(--good)", sw: 1.6, dash: "6 4" }, dyn); sv.text(X1, py(0.9) - 9, "the bar: 90% agreement", { size: 11.5, anchor: "end", fill: "var(--good)" }, dyn);
    let H = 0; for (let n = 1; n <= NMAX; n++) { if (Math.pow(p, n) >= 0.9) H = n; else break; }
    for (let n = 1; n <= NMAX; n++) { const v = Math.pow(p, n), ok = n <= H; sv.rect(px(n) - 14, py(v), 28, Y0 - py(v), { fill: ok ? "var(--good)" : "var(--frozen)", stroke: "none", rx: 4, opacity: ok ? .9 : .7 }, dyn); sv.text(px(n), Y0 + 16, String(n), { size: 11, mono: true, anchor: "middle", fill: "var(--ink3)" }, dyn); if (n === 1 || n === 5 || n === 10 || n === 20) sv.text(px(n), py(v) - 8, Math.round(v * 100) + "%", { size: 11, mono: true, anchor: "middle", fill: "var(--ink2)" }, dyn); }
    sv.text((X0 + X1) / 2, Y0 + 38, "number of imagined steps in a row (N)", { size: 12, anchor: "middle", fill: "var(--ink2)" }, dyn);
    fig.$live.hidden = false; fig.$live.innerHTML = `<b>With each imagined step ${(p * 100).toFixed(1)}% accurate, the lookahead may reach H = ${H}.</b> A tiny error per step compounds: 95% per step leaves only ${(Math.pow(0.95, 20) * 100).toFixed(0)}% after 20 steps. That is why receding-horizon search keeps the imagined depth short and re-reads the real position after every committed step. <span class="muted">Simplified model with independent errors. Gate G2.3 measures the real thing.</span>`;
  }
  draw();
});

/* ------------------------------------------------------------ answerLoop: AB-MCTS-A */
G.reg("answerLoop", fig => {
  const sv = fig.svg(900, 360, { label: "The outer answer loop: fresh attempts or refinements of a failed answer" });
  const gs = {}; ["root", "gen", "fail", "pick", "cont", "ok"].forEach(k => { gs[k] = sv.g({}); });
  const box = (g, x, y, w, h, t, c, st) => { sv.rect(x, y, w, h, { fill: st === "ok" ? "var(--good-s)" : st === "bad" ? "var(--bad-s)" : "var(--panel)", stroke: st === "ok" ? "var(--good)" : st === "bad" ? "var(--bad)" : `var(--${c})`, sw: 1.8, rx: 12 }, g); sv.text(x + w / 2, y + h / 2, t, { size: 12.5, weight: 650, anchor: "middle" }, g); };
  box(gs.root, 20, 150, 120, 60, "the puzzle", "ink3");
  box(gs.gen, 190, 70, 170, 60, "attempt 1\n(fresh search, new noise)", "enc"); sv.line(140, 170, 190, 110, { stroke: "var(--ink3)", sw: 1.6, arrow: "ink3" }, gs.gen);
  box(gs.fail, 410, 70, 150, 60, "checker: wrong", "bad", "bad"); sv.line(360, 100, 410, 100, { stroke: "var(--ink3)", sw: 1.6, arrow: "ink3" }, gs.fail);
  box(gs.pick, 410, 160, 150, 70, "pick the next move\nby Thompson sampling", "code"); sv.line(485, 130, 485, 160, { stroke: "var(--ink3)", sw: 1.6, arrow: "ink3" }, gs.pick);
  box(gs.cont, 620, 160, 250, 70, "refine the failed answer:\nnudge one code at a random chunk,\nthen write again", "dec"); sv.line(560, 195, 620, 195, { stroke: "var(--dec)", sw: 1.8, arrow: "dec" }, gs.cont); sv.text(590, 180, "CONT", { size: 11, weight: 700, anchor: "middle", fill: "var(--dec)" }, gs.cont);
  box(gs.gen, 190, 250, 170, 60, "or: attempt 2\n(a whole new search)", "enc"); sv.path("M 485 230 C 485 280 400 280 360 280", { stroke: "var(--enc)", sw: 1.8, arrow: "enc" }, gs.gen); sv.text(440, 300, "GEN", { size: 11, weight: 700, anchor: "middle", fill: "var(--enc)" }, gs.gen);
  box(gs.ok, 620, 270, 250, 56, "checker: right, stop", "good", "ok"); sv.path("M 745 230 L 745 270", { stroke: "var(--good)", sw: 1.8, arrow: "good" }, gs.ok);
  const order = ["root", "gen", "fail", "pick", "cont", "ok"];
  const STEPS = [
    { label: "1 Start", title: "Begin with the puzzle.", text: "The inner search (RH, G(R) or the token tree, whichever did best on development puzzles) is the engine. This outer loop wraps it." },
    { label: "2 Attempt", title: "Make a full attempt.", text: "The inner search runs to the end and writes a complete answer." },
    { label: "3 Check", title: "The checker says wrong.", text: "The only feedback is one bit. If the answer is right, we stop. If not, and budget remains, we choose what to do next." },
    { label: "4 Choose", title: "Decide between two moves.", text: "A bandit rule (Thompson sampling) balances starting fresh against refining what we have, learning from how each kind of move has worked so far." },
    { label: "5 Refine", title: "CONT: refine a failed answer.", text: "Nudge one code at a randomly chosen committed chunk, with GradCuit or a random direction of the same size, then write again from that chunk. S5 runs both nudge types to see what the reward adds." },
    { label: "6 Stop", title: "Stop at the first right answer.", text: "Anything the loop reports as solved has been written out and passed the checker. Search never declares success on its own." },
  ];
  stepper(fig, STEPS, i => { order.forEach((k, ix) => { gs[k].style.transition = "opacity .3s"; gs[k].style.opacity = ix <= i ? 1 : .15; }); }, { ms: 4600 });
});
})();
