/* Chapter 6 (S4) figures: eiLoop, visitsTarget, paramsBar, gradReward */
(function () {
const { H, S, ui, loop, stepper, clamp, lerp, along } = G;

/* ------------------------------------------------------------ eiLoop: expert iteration */
G.reg("eiLoop", fig => {
  const sv = fig.svg(900, 430, { label: "The expert iteration loop: problems, search, play, verify, learn" });
  const BX = [20, 196, 372, 548, 724], BW = 156, BY = 110, BH = 120;
  const NODES = [
    ["Fresh puzzles", "enc", "From the generator, never seen before: half standard, half hard10."],
    ["Search", "code", "At each step, try codes in concept space for a budget of 32 tries."],
    ["Play one step", "dec", "Take the most-visited code, write 4 tokens, read them back."],
    ["Verify", "good", "When the answer is complete, the checker says right or wrong."],
    ["Learn", "sel", "Train the selector and two tiny judges on what just happened."],
  ];
  const gs = NODES.map(() => sv.g({}));
  NODES.forEach(([t, c, d], i) => {
    sv.rect(BX[i], BY, BW, BH, { fill: "var(--panel)", stroke: `var(--${c})`, sw: 2, rx: 14 }, gs[i]); sv.rect(BX[i], BY, BW, 34, { fill: `var(--${c})`, stroke: "none", rx: 14 }, gs[i]); sv.rect(BX[i], BY + 20, BW, 14, { fill: `var(--${c})`, stroke: "none", rx: 0 }, gs[i]);
    sv.text(BX[i] + BW / 2, BY + 17, (i + 1) + ". " + t, { size: 13.5, weight: 650, anchor: "middle", fill: "var(--bg)" }, gs[i]);
    const wrapTxt = (str, n) => { let line = "", out = []; str.split(" ").forEach(w => { if ((line + " " + w).trim().length > n) { out.push(line); line = w; } else line = (line + " " + w).trim(); }); out.push(line); return out.join("\n"); };
    sv.text(BX[i] + 10, BY + 80, wrapTxt(d, 22), { size: 11.5, fill: "var(--ink2)" }, gs[i]);
    if (i < 4) sv.line(BX[i] + BW + 3, BY + BH / 2, BX[i + 1] - 3, BY + BH / 2, { stroke: "var(--ink3)", sw: 1.8, arrow: "ink3" });
  });
  sv.path(`M ${BX[2] + BW / 2} ${BY - 4} C ${BX[2] + BW / 2} ${BY - 56} ${BX[1] + BW / 2} ${BY - 56} ${BX[1] + BW / 2} ${BY - 4}`, { stroke: "var(--ink3)", sw: 1.8, dash: "5 4", arrow: "ink3" });
  sv.text((BX[1] + BX[2] + BW) / 2, BY - 64, "repeat each step until STOP (at most 20 hops)", { size: 12, anchor: "middle", fill: "var(--ink2)" });
  sv.path(`M ${BX[4] + BW / 2} ${BY + BH + 4} C ${BX[4] + BW / 2} ${BY + BH + 90} ${BX[1] + BW / 2} ${BY + BH + 90} ${BX[1] + BW / 2} ${BY + BH + 4}`, { stroke: "var(--sel)", sw: 2.2, arrow: "sel" });
  sv.text((BX[1] + BX[4] + BW) / 2, BY + BH + 92, "a better selector and better judges guide the next round · 4 rounds in all", { size: 12.5, weight: 600, anchor: "middle", fill: "var(--sel)" });
  sv.rect(20, 366, 860, 50, { fill: "var(--bg2)", stroke: "var(--line2)", rx: 10 });
  sv.text(34, 391, "Round 0 has no judges yet, so every try is written out and played to the end by the real decoder (512 puzzles).\nRounds 1 to 4 use the judges instead (4,096 puzzles each), which is far cheaper.", { size: 12, fill: "var(--ink2)" });
  const STEPS = [
    { label: "1 Puzzles", title: "Start with fresh puzzles.", text: "They come from the same generator as always and are checked against every evaluation puzzle, so none of them are test puzzles." },
    { label: "2 Search", title: "Search for a good choice.", text: "From the current position, the model tries codes in concept space. Each try is cheap because the decoder does not run. A try that looks good gets tried more often. After 32 tries, we know how often each code was visited." },
    { label: "3 Play", title: "Commit to the most-visited code.", text: "That one code is decoded into four real tokens, which are read back in. The search then starts again from this new, real position." },
    { label: "4 Verify", title: "The checker settles it.", text: "When the answer ends, the verifier says right or wrong. That single bit is the only feedback in the whole loop." },
    { label: "5 Learn", title: "Teach the instincts what the search found.", text: "The selector is trained to produce the search's visit counts directly. The two tiny judges are trained to predict whether the answer will be right and whether stopping is legal. Then the loop restarts with these improved parts." },
  ];
  let stp = 0; const PU = [[], [[[BX[0] + BW + 3, 170], [BX[1] - 3, 170]]], [[[BX[1] + BW + 3, 170], [BX[2] - 3, 170]]], [[[BX[2] + BW + 3, 170], [BX[3] - 3, 170]]], [[[BX[3] + BW + 3, 170], [BX[4] - 3, 170]]]];
  const dot = sv.circle(-20, -20, 5, { fill: "var(--accent)", stroke: "var(--panel)", sw: 1.5 });
  stepper(fig, STEPS, i => { stp = i; gs.forEach((g, k) => { g.style.transition = "opacity .3s"; g.style.opacity = k === i ? 1 : .45; }); }, { ms: 4800 });
  loop(fig, t => { const P = PU[stp] || []; if (!P[0]) { dot.setAttribute("cx", -20); return; } const [x, y] = along(P[0], (t * 0.6) % 1); dot.setAttribute("cx", x); dot.setAttribute("cy", y); });
});

/* ------------------------------------------------------------ visitsTarget: search counts become the selector's target */
G.reg("visitsTarget", fig => {
  const W = 900, Hh = 380, sv = fig.svg(W, Hh, { label: "Search visit counts and the selector converging to them" });
  const N = [14, 3, 1, 8, 0, 2, 1, 1], tot = N.reduce((a, b) => a + b, 0), target = N.map(v => v / tot);
  let logits = new Array(8).fill(0), steps = 0, lr = 0.35;
  const sm = z => { const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };
  const r = ui.readout(); const pb = G.playBtn(fig);
  fig.$ui.append(ui.btn("One training step", () => { fig.playing = false; pb.textContent = "Play"; step(); }, "pri"), pb, ui.btn("Reset", () => { logits = new Array(8).fill(0); steps = 0; draw(); }), r);
  const dyn = sv.g({});
  function step() { const p = sm(logits); logits = logits.map((z, i) => z - lr * (p[i] - target[i])); steps++; draw(); }
  function draw() {
    dyn.innerHTML = ""; const p = sm(logits), X0 = 70, BWd = 78, Y0 = 150;
    sv.text(20, 22, "At one position, search visited each code this many times (32 tries in all, 2 went to STOP):", { size: 13, weight: 650 }, dyn);
    N.forEach((v, i) => { const x = X0 + i * (BWd + 16); sv.rect(x, Y0 - 100 * v / 14 * 0.95, BWd, Math.max(2, 100 * v / 14 * 0.95), { fill: "var(--code)", stroke: "none", rx: 5 }, dyn); sv.text(x + BWd / 2, Y0 - 100 * v / 14 * 0.95 - 8, String(v), { size: 12.5, mono: true, anchor: "middle", weight: 650 }, dyn); sv.text(x + BWd / 2, Y0 + 16, "code " + (i + 1), { size: 11.5, anchor: "middle", fill: "var(--ink3)" }, dyn); });
    sv.line(X0 - 8, Y0, X0 + 8 * (BWd + 16), Y0, { stroke: "var(--line2)", sw: 1.2 }, dyn);
    sv.text(20, 218, "The selector's odds, which we train to match those visit shares:", { size: 13, weight: 650 }, dyn);
    const Y1 = 342;
    N.forEach((v, i) => { const x = X0 + i * (BWd + 16), h = 100 * p[i] / 0.5 * 0.95, ht = 100 * target[i] / 0.5 * 0.95; sv.rect(x, Y1 - h, BWd, Math.max(2, h), { fill: "var(--sel)", stroke: "none", rx: 5 }, dyn); sv.line(x - 4, Y1 - ht, x + BWd + 4, Y1 - ht, { stroke: "var(--ink)", sw: 2, dash: "4 3" }, dyn); sv.text(x + BWd / 2, Y1 + 16, Math.round(p[i] * 100) + "%", { size: 11.5, mono: true, anchor: "middle", fill: "var(--ink2)" }, dyn); });
    sv.line(X0 - 8, Y1, X0 + 8 * (BWd + 16), Y1, { stroke: "var(--line2)", sw: 1.2 }, dyn); sv.text(X0 + 8 * (BWd + 16) + 6, Y1 - 60, "dashed line:\nthe target", { size: 11, fill: "var(--ink3)" }, dyn);
    const kl = target.reduce((a, t, i) => a + (t > 0 ? t * Math.log(t / p[i]) : 0), 0);
    r.set(`training steps <b>${steps}</b> · gap to target (KL) <b>${kl.toFixed(3)}</b>`);
    fig.$live.hidden = false; fig.$live.innerHTML = steps === 0 ? "<b>Before learning,</b> the selector treats every code the same. Press a step to nudge its odds toward the search's visit shares." : kl < 0.05 ? "<b>The selector has absorbed what the search found.</b> It would now suggest code 1 first without searching. This is the core of expert iteration: the search is slow but good, and training copies it into the quick instinct." : "<b>The odds move toward the target.</b> The loss is cross-entropy between the target and the selector's odds, and its gradient is simply the difference between them.";
  }
  loop(fig, (t, dt) => { if (steps < 120) { if (Math.floor(t * 8) !== Math.floor((t - dt) * 8)) step(); } }); draw();
});

/* ------------------------------------------------------------ paramsBar: how much each method can change */
G.reg("paramsBar", fig => {
  const sv = fig.svg(900, 270, { label: "Trainable parameters of the S4 methods", narrow: true });
  const ROWS = [["Selector REINFORCE", 1032, "var(--sel)", "selector only"], ["Expert iteration (primary)", 1290, "var(--sel)", "selector 1,032 + two judges of 129"], ["Token RFT", 24576, "var(--dec)", "rank-16 adapters on 3 decoder blocks"], ["Token GRPO", 24576, "var(--dec)", "the same adapters"]];
  const X0 = 250, X1 = 800, mx = 26000, px = v => X0 + (X1 - X0) * v / mx;
  ROWS.forEach(([n, v, c, sub], i) => { const y = 36 + i * 50; sv.text(X0 - 12, y + 10, n, { size: 13.5, weight: 650, anchor: "end" }); sv.text(X0 - 12, y + 28, sub, { size: 11, anchor: "end", fill: "var(--ink3)" }); sv.rect(X0, y, X1 - X0, 22, { fill: "var(--bg2)", stroke: "none", rx: 5 }); sv.rect(X0, y, Math.max(3, px(v) - X0), 22, { fill: c, stroke: "none", rx: 5 }); sv.text(px(v) + 8, y + 11, v.toLocaleString("en-US"), { size: 12.5, mono: true, weight: 650 }); });
  fig.$live.hidden = false; fig.$live.innerHTML = "<b>The comparison is not equal on capacity, and the project says so.</b> The token baselines can change about 19 times more numbers than expert iteration. That favours the baselines on capacity, and it favours the primary method on design, since it works on exactly the choice we care about. Every result table lists the trainable parameters of each arm.";
});

/* ------------------------------------------------------------ gradReward: why the reward is o − 1 */
G.reg("gradReward", fig => {
  const W = 900, Hh = 330, sv = fig.svg(W, Hh, { label: "Log-probability of a rejected answer when nudging a code with two reward rules" });
  const X0 = 90, X1 = 560, Y0 = 270, Y1 = 50; let rule = "good", n = 0;
  const px = s => X0 + (X1 - X0) * s / 10, py = v => Y0 - (Y0 - Y1) * (v + 6) / 6;
  fig.$ui.append(ui.seg([["good", "Reward = o − 1 (what S4 uses)"], ["bad", "Reward = o (a mistake)"]], rule, v => { rule = v; n = 0; draw(); }, "reward rule"), ui.btn("Take 10 nudges", () => { n = 10; draw(); }, "pri"), ui.btn("Reset", () => { n = 0; draw(); }));
  const dyn = sv.g({});
  function draw() {
    dyn.innerHTML = "";
    sv.text(X0, 28, "Log-probability the model gives to its own rejected answer", { size: 13, weight: 650 }, dyn);
    [0, -2, -4, -6].forEach(v => { sv.line(X0, py(v), X1, py(v), { stroke: "var(--line)", sw: 1 }, dyn); sv.text(X0 - 8, py(v), String(v), { size: 11, mono: true, anchor: "end", fill: "var(--ink3)" }, dyn); });
    for (let s = 0; s <= 10; s += 2) sv.text(px(s), Y0 + 18, String(s), { size: 11, mono: true, anchor: "middle", fill: "var(--ink3)" }, dyn); sv.text((X0 + X1) / 2, Y0 + 40, "number of nudges to the code (at most 10)", { size: 12, anchor: "middle", fill: "var(--ink2)" }, dyn);
    const val = s => rule === "good" ? -0.8 - 0.46 * s : -0.8;
    let d = ""; for (let s = 0; s <= n; s++) d += (s ? "L" : "M") + px(s).toFixed(1) + " " + py(val(s)).toFixed(1); sv.path(d, { stroke: rule === "good" ? "var(--good)" : "var(--bad)", sw: 3 }, dyn);
    for (let s = 0; s <= n; s++) sv.circle(px(s), py(val(s)), 4.2, { fill: rule === "good" ? "var(--good)" : "var(--bad)", stroke: "var(--panel)", sw: 1.6 }, dyn);
    sv.text(610, 60, rule === "good" ? "Rejected answer: reward = −1" : "Rejected answer: reward = 0", { size: 14, weight: 650, fill: rule === "good" ? "var(--good)" : "var(--bad)" }, dyn);
    sv.text(610, 96, "loss = − reward × log-probability\nof the current answer", { size: 12.5, mono: true, fill: "var(--ink2)" }, dyn);
    sv.text(610, 150, rule === "good" ? "The gradient is non-zero, so each nudge\nmakes the rejected answer less likely.\nDecoding again may now give a different\nanswer, which the checker re-scores." : "Multiplying by zero wipes out the gradient.\nNothing moves, however many nudges are\ntaken. The method would silently do nothing.", { size: 13, fill: "var(--ink2)" }, dyn);
    sv.text(610, 262, "toy numbers, to show the mechanism only", { size: 11, fill: "var(--ink3)" }, dyn);
    fig.$live.hidden = false; fig.$live.innerHTML = "<b>GradCuit</b> nudges one code vector at test time. A test checks, on a failed answer, that the gradient is non-zero and that one step lowers the rejected answer's probability. It also checks that a version using the verdict itself as the reward is caught as a no-op.";
  }
  draw();
});
})();
