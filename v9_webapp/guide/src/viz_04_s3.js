/* Chapter 5 (S3) figures: freezeMap (also used by S4 and S6), coordDrift, lambdaMix */
(function () {
const { H, S, ui, loop, clamp, lerp } = G;
const V9V = window.V9V;

/* ------------------------------------------------------------ freezeMap: which parts learn in each arm (reads the variant table) */
const GROUPS = [
  ["emb", "Token\nembedding", ["emb"], "enc"], ["enc", "Encoder", ["enc"], "enc"], ["pool", "Pooling", ["pool"], "con"], ["core", "Concept\ncore", ["core", "bh0", "cin"], "con"],
  ["codes", "Codes +\nselector", ["codes", "bhk", "hyp", "sel", "bestm"], "code"], ["pq", "Sticker heads\n+ codebook", ["pq", "Q"], "code"], ["dec", "Decoder", ["dec", "dsum", "head"], "dec"],
  ["teacher", "S1 teacher", ["teacher"], "frozen"], ["lora", "Adapters", ["lora"], "dec"], ["gate", "Gates", ["gate"], "loss"], ["vtau", "Value + stop\nheads", ["vtau"], "sel"],
];
G.reg("freezeMap", fig => {
  const ids = (fig.dataset.variants || "").split(",").filter(Boolean), vs = ids.map(i => V9V.byId(i)).filter(Boolean);
  if (!vs.length) { fig.$stage.textContent = "No arms listed."; return; }
  const sv = fig.svg(900, 330, { label: "Which parts of the model learn, are frozen, or are absent in each arm" }); let cur = vs[0];
  fig.$ui.append(ui.seg(vs.map(v => [v.id, v.label]), cur.id, id => { cur = vs.find(v => v.id === id); draw(); }, "arm"));
  const dyn = sv.g({});
  const stateOf = (V, parts) => { const t = V.train || {}; const st = parts.map(p => t[p] || "N"); return st.includes("T") ? "T" : st.includes("F") ? "F" : "N"; };
  const present = GROUPS.filter(g => vs.some(v => stateOf(v, g[2]) !== "N"));
  function draw() {
    dyn.innerHTML = ""; const main = present.filter(g => !["teacher", "lora", "gate", "vtau"].includes(g[0])), extra = present.filter(g => ["teacher", "lora", "gate", "vtau"].includes(g[0]));
    const bw = 112, gap = 14, X0 = 14;
    main.forEach((g, i) => {
      const x = X0 + i * (bw + gap), st = stateOf(cur, g[2]), col = `var(--${g[3]})`;
      sv.rect(x, 60, bw, 100, { fill: st === "T" ? col : st === "F" ? "var(--frozen-s)" : "var(--panel)", stroke: st === "N" ? "var(--line2)" : st === "F" ? "var(--frozen)" : col, sw: st === "T" ? 0 : 1.8, rx: 12, dash: st === "N" ? "5 4" : st === "F" ? "6 3" : null, opacity: st === "N" ? .6 : 1 }, dyn);
      sv.text(x + bw / 2, 96, g[1], { size: 13, weight: 650, anchor: "middle", fill: st === "T" ? "var(--bg)" : st === "N" ? "var(--ink3)" : "var(--ink)" }, dyn);
      sv.text(x + bw / 2, 136, st === "T" ? "LEARNS" : st === "F" ? "frozen" : "not in this arm", { size: 11.5, weight: 700, anchor: "middle", mono: true, fill: st === "T" ? "var(--bg)" : "var(--ink3)" }, dyn);
      if (i < main.length - 1) sv.line(x + bw + 2, 110, x + bw + gap - 2, 110, { stroke: "var(--ink3)", sw: 1.4, arrow: "ink3" }, dyn);
    });
    extra.forEach((g, i) => {
      const x = X0 + i * (bw + gap), st = stateOf(cur, g[2]), col = `var(--${g[3]})`;
      sv.rect(x, 200, bw, 66, { fill: st === "T" ? col : st === "F" ? "var(--frozen-s)" : "var(--panel)", stroke: st === "N" ? "var(--line2)" : st === "F" ? "var(--frozen)" : col, sw: st === "T" ? 0 : 1.8, rx: 12, dash: st === "N" ? "5 4" : st === "F" ? "6 3" : null, opacity: st === "N" ? .6 : 1 }, dyn);
      sv.text(x + bw / 2, 224, g[1], { size: 12, weight: 650, anchor: "middle", fill: st === "T" ? "var(--bg)" : st === "N" ? "var(--ink3)" : "var(--ink)" }, dyn);
      sv.text(x + bw / 2, 248, st === "T" ? "LEARNS" : st === "F" ? "frozen" : "not in this arm", { size: 11, weight: 700, anchor: "middle", mono: true, fill: st === "T" ? "var(--bg)" : "var(--ink3)" }, dyn);
    });
    sv.text(14, 30, cur.label + " · " + (cur.sub || ""), { size: 14, weight: 650 }, dyn);
    const P = cur.params || {}, f = n => n == null ? "not stated in the design" : n.toLocaleString("en-US");
    sv.text(14, 304, `Parameters in use: ${f(P.total)} · learning: ${f(P.trainable)}${cur.K > 1 ? " · K = " + cur.K : ""}${cur.codeFree ? " · code-free plan" : ""}`, { size: 12.5, fill: "var(--ink2)", mono: true }, dyn);
    fig.$live.hidden = false; fig.$live.innerHTML = `<b>${cur.label}.</b> ${cur.note || cur.sub || ""} ${cur.tier === "full" ? '<span class="chip full">full column</span>' : '<span class="chip mand">mandatory</span>'}`;
  }
  draw();
});

/* ------------------------------------------------------------ coordDrift: moving the encoder moves the map */
G.reg("coordDrift", fig => {
  const W = 900, Hh = 340, sv = fig.svg(W, Hh, { label: "A grid of concept coordinates before and after the encoder trains" });
  let amt = 0.0; const sl = ui.range("how much the encoder trains", 0, 1, 0.02, 0.0, v => { amt = v; draw(); }, v => v < 0.05 ? "not at all" : Math.round(v * 100) + "%");
  const r = ui.readout(); fig.$ui.append(sl, ui.btn("Train it a lot", () => { sl.set(1); amt = 1; draw(); }), r);
  const dyn = sv.g({}), GX = 7, GY = 5, PW = 330, PH = 230, OX = [40, 520], OY = 60;
  const disp = (x, y) => [amt * 30 * Math.sin(1.9 * y / PH * 3 + 0.4) + amt * 12, amt * 26 * Math.cos(1.5 * x / PW * 3 + 0.8)];
  const KEY = [[0.2, 0.25, "var(--enc)", "A"], [0.62, 0.3, "var(--sel)", "B"], [0.45, 0.78, "var(--con)", "C"]];
  function draw() {
    dyn.innerHTML = "";
    [["Before: the concept core learned these coordinates", 0], ["After: the encoder has changed", 1]].forEach(([t, side]) => {
      sv.text(OX[side], 36, t, { size: 13, weight: 650 }, dyn); sv.rect(OX[side], OY, PW, PH, { fill: "var(--panel2)", stroke: "var(--line2)", rx: 12 }, dyn);
      const pos = (u, v) => { const x = u * PW, y = v * PH, d = side ? disp(x, y) : [0, 0]; return [OX[side] + x + d[0], OY + y + d[1]]; };
      for (let i = 0; i <= GX; i++) { let d = ""; for (let j = 0; j <= 20; j++) { const p = pos(i / GX, j / 20); d += (j ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); } sv.path(d, { stroke: "var(--line2)", sw: 1 }, dyn); }
      for (let j = 0; j <= GY; j++) { let d = ""; for (let i = 0; i <= 28; i++) { const p = pos(i / 28, j / GY); d += (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); } sv.path(d, { stroke: "var(--line2)", sw: 1 }, dyn); }
      KEY.forEach(([u, v, c, n]) => { const p = pos(u, v); sv.circle(p[0], p[1], 11, { fill: c, stroke: "var(--panel)", sw: 2.5 }, dyn); sv.text(p[0], p[1], n, { size: 11.5, weight: 700, anchor: "middle", fill: "var(--bg)" }, dyn);
        if (side) { const q = [OX[1] + u * PW, OY + v * PH]; sv.circle(q[0], q[1], 11, { fill: "none", stroke: c, sw: 1.6, opacity: .8 }, dyn); sv.line(q[0], q[1], p[0], p[1], { stroke: c, sw: 1.2, dash: "3 3", opacity: .7 }, dyn); } });
    });
    sv.path("M 385 175 H 505", { stroke: "var(--ink3)", sw: 2, arrow: "ink3" }, dyn); sv.text(445, 160, "train the\nencoder", { size: 11.5, anchor: "middle", fill: "var(--ink3)" }, dyn);
    const avg = KEY.reduce((a, [u, v]) => { const d = disp(u * PW, v * PH); return a + Math.hypot(d[0], d[1]); }, 0) / 3;
    r.set(`average shift of A, B, C: <b>${avg.toFixed(0)}</b> units`);
    fig.$live.hidden = false; fig.$live.innerHTML = amt < 0.05 ? "<b>The concept core was trained against these coordinates.</b> It learned that, say, concept B lives at one place. Slide the encoder training up." : "<b>The solid dots are where the concepts now are. The rings are where the concept core still expects them.</b> Even if the core itself is frozen, it is now reading a map that has moved. That is why 'freeze the core' does not mean 'freeze the interface', and why S3 changes the decoder first and the encoder only in a separate, optional arm.";
  }
  draw();
});

/* ------------------------------------------------------------ lambdaMix: the transplant check */
G.reg("lambdaMix", fig => {
  const W = 900, Hh = 330, sv = fig.svg(W, Hh, { label: "Mixing the teacher's plan with the student's plan" });
  let lam = 0.3; const sl = ui.range("λ (how much of the student)", 0, 1, 0.05, 0.3, v => { lam = v; draw(); }, v => v.toFixed(2));
  const seg = ui.seg([["0", "λ = 0"], ["0.3", "0.3"], ["0.6", "0.6"], ["1", "λ = 1"]], "0.3", v => { lam = +v; sl.set(lam); draw(); }, "preset");
  fig.$ui.append(seg, sl); const dyn = sv.g({});
  const O = [150, 250], uT = [210, -130], uS = [110, -170];
  function draw() {
    dyn.innerHTML = ""; const mix = [lerp(uT[0], uS[0], lam), lerp(uT[1], uS[1], lam)];
    sv.rect(14, 14, 520, 300, { fill: "var(--panel2)", stroke: "var(--line2)", rx: 12 }, dyn);
    sv.line(O[0], O[1], O[0] + uT[0], O[1] + uT[1], { stroke: "var(--frozen)", sw: 3.2, arrow: "frozen" }, dyn); sv.text(O[0] + uT[0] + 8, O[1] + uT[1] - 10, "teacher's plan", { size: 12.5, weight: 650, fill: "var(--ink2)" }, dyn);
    sv.line(O[0], O[1], O[0] + uS[0], O[1] + uS[1], { stroke: "var(--con)", sw: 3.2, arrow: "con" }, dyn); sv.text(O[0] + uS[0] - 70, O[1] + uS[1] - 14, "student's plan (no code)", { size: 12.5, weight: 650, fill: "var(--con)" }, dyn);
    sv.path(`M ${O[0] + uT[0]} ${O[1] + uT[1]} L ${O[0] + uS[0]} ${O[1] + uS[1]}`, { stroke: "var(--ink3)", sw: 1.2, dash: "4 4" }, dyn);
    sv.line(O[0], O[1], O[0] + mix[0], O[1] + mix[1], { stroke: "var(--sel)", sw: 4.4, arrow: "sel" }, dyn); sv.circle(O[0] + mix[0], O[1] + mix[1], 6, { fill: "var(--sel)", stroke: "var(--panel)", sw: 2 }, dyn);
    sv.text(O[0] + mix[0] + 12, O[1] + mix[1] + 22, "the plan the decoder reads", { size: 12.5, weight: 650, fill: "var(--sel)" }, dyn);
    [0, 0.3, 0.6, 1].forEach(l => { const p = [O[0] + lerp(uT[0], uS[0], l), O[1] + lerp(uT[1], uS[1], l)]; sv.circle(p[0], p[1], 3.2, { fill: "var(--ink)" }, dyn); sv.text(p[0] + 8, p[1] + 14, "λ=" + l, { size: 10.5, mono: true, fill: "var(--ink3)" }, dyn); });
    sv.text(560, 60, "u = (1 − λ) · u_teacher + λ · u_student", { size: 14, weight: 650, mono: true }, dyn);
    sv.text(560, 100, lam === 0 ? "λ = 0: the decoder reads exactly the\nteacher's plan. This must reproduce the\nteacher's output bit for bit (a test)." : lam === 1 ? "λ = 1: the decoder reads exactly the\nstudent's plan." : `λ = ${lam.toFixed(2)}: a blend, ${Math.round((1 - lam) * 100)}% teacher and ${Math.round(lam * 100)}% student.`, { size: 13, fill: "var(--ink2)" }, dyn);
    sv.text(560, 190, "The check uses λ = 0, 0.3, 0.6 and 1.\nIt measures how gracefully the student's\ncode-free path can stand in for the teacher.", { size: 12.5, fill: "var(--ink3)" }, dyn);
    fig.$live.hidden = false; fig.$live.innerHTML = "<b>Before any new training,</b> S3 checks the student against the teacher by mixing their plans. The code-free part of the student is the one being checked, since the teacher has no codes.";
  }
  draw();
});
})();
