/* Chapter 8 (S6) figures: v8history, lora, gmmGate, phases, credit */
(function () {
const { H, S, ui, loop, clamp, lerp, rng, gauss } = G;

/* ------------------------------------------------------------ v8history: what went wrong last time (real saved numbers) */
G.reg("v8history", fig => {
  const sv = fig.svg(900, 380, { label: "V8 continual-learning results: new skill gained and old skills lost, and the gate-installation gap" });
  sv.text(20, 26, "Phase 2 (the new skill 'bind'): change in puzzles solved, in percentage points", { size: 13.5, weight: 650 });
  const R = [["New skill: bind", 25.2, "var(--good)"], ["Old skill: navigation (standard)", -81.5, "var(--bad)"], ["Old skill: hard10 navigation", -30.7, "var(--bad)"], ["Old skill: lookup", -28.8, "var(--bad)"], ["Old skill: pointer", -18.8, "var(--bad)"]];
  const X0 = 360, mid = 560, sc = 2.2;
  sv.line(mid, 44, mid, 222, { stroke: "var(--ink3)", sw: 1.4 });
  R.forEach(([n, v, c], i) => { const y = 50 + i * 34, w = Math.abs(v) * sc; sv.text(X0 - 12, y + 12, n, { size: 12.5, anchor: "end", fill: "var(--ink2)" }); sv.rect(v >= 0 ? mid : mid - w, y, w, 24, { fill: c, stroke: "none", rx: 5 }); sv.text(v >= 0 ? mid + w + 8 : mid - w - 8, y + 12, (v > 0 ? "+" : "−") + Math.abs(v).toFixed(1), { size: 12.5, mono: true, weight: 700, anchor: v >= 0 ? "start" : "end" }); });
  sv.text(20, 262, "Phase 3 (a shifted map layout): puzzles solved out of 512", { size: 13.5, weight: 650 });
  [["Adapter forced open", 171, "var(--good)"], ["Adapter switched by the fitted gate", 1, "var(--bad)"]].forEach(([n, v, c], i) => { const y = 280 + i * 36; sv.text(X0 - 12, y + 12, n, { size: 12.5, anchor: "end", fill: "var(--ink2)" }); sv.rect(X0, y, 440 * v / 512 * 2.5 > 440 ? 440 : Math.max(3, 440 * v / 512 * 2.5), 24, { fill: c, stroke: "none", rx: 5 }); sv.text(X0 + Math.max(3, 440 * v / 512 * 2.5) + 8, y + 12, String(v), { size: 12.5, mono: true, weight: 700 }); });
  sv.text(20, 366, "Source: V8's saved phase decisions (D_lsl_phase 1 to 3). Phase 1 also left stories text slightly worse (+0.0071 nats).", { size: 11.5, fill: "var(--ink3)" });
  fig.$live.hidden = false; fig.$live.innerHTML = "<b>Two failures, in plain words.</b> In phase 2 the model learned the new skill and lost most of the old ones. In phase 3 the skill worked when the adapter was forced on and nearly vanished when the gate decided. The tiny setup used there was far below the paper's scale, and the safe-guard control (adapter always on) was never run, so these results tell us what to measure better, not that the idea fails.";
});

/* ------------------------------------------------------------ lora: a thin add-on to a frozen matrix */
G.reg("lora", fig => {
  const W = 900, Hh = 380, sv = fig.svg(W, Hh, { label: "A frozen weight matrix plus a low-rank adapter" });
  const N = 16, CS = 11; let r = 4;
  const rs = ui.range("rank r", 1, 16, 1, 4, v => { r = v; draw(); }, v => String(v));
  fig.$ui.append(rs, ui.btn("r = 4 (V8 and LSL-v8)", () => { r = 4; rs.set(4); draw(); }), ui.btn("r = 16 (LSL-scaled)", () => { r = 16; rs.set(16); draw(); }));
  const dyn = sv.g({}); const rf = rng(5); const Bm = Array.from({ length: N }, () => Array.from({ length: 16 }, () => gauss(rf))), Am = Array.from({ length: 16 }, () => Array.from({ length: N }, () => gauss(rf)));
  const cell = (g, x, y, v, mx, col) => sv.rect(x, y, CS - 1, CS - 1, { fill: v >= 0 ? "var(--dec)" : "var(--sel)", stroke: "none", rx: 1.5, opacity: col ? clamp(Math.abs(v) / mx, 0.05, 1) : .9 }, g);
  function draw() {
    dyn.innerHTML = ""; const OX = 40, OY = 70;
    const dW = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => { let s = 0; for (let k = 0; k < r; k++) s += Bm[i][k] * Am[k][j]; return s / Math.sqrt(r); }));
    const mx = Math.max(...dW.flat().map(Math.abs));
    sv.text(OX + 88, 48, "frozen matrix W", { size: 13, weight: 650, anchor: "middle" }, dyn);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) sv.rect(OX + j * CS, OY + i * CS, CS - 1, CS - 1, { fill: "var(--frozen)", stroke: "none", rx: 1.5, opacity: .35 + 0.4 * (((i * 7 + j * 13) % 9) / 9) }, dyn);
    sv.text(OX + 210, OY + 88, "+", { size: 30, weight: 300, anchor: "middle" }, dyn);
    const BX = OX + 250; sv.text(BX + 5 * r / 2 + 8, 48, "B", { size: 13, weight: 650, anchor: "middle" }, dyn); for (let i = 0; i < N; i++) for (let k = 0; k < r; k++) cell(dyn, BX + k * CS, OY + i * CS, Bm[i][k], 2.4, true);
    const AX = BX + r * CS + 40; sv.text(AX + 88, OY - 22, "A", { size: 13, weight: 650, anchor: "middle" }, dyn); for (let k = 0; k < r; k++) for (let j = 0; j < N; j++) cell(dyn, AX + j * CS, OY + k * CS, Am[k][j], 2.4, true);
    sv.text(AX + 195, OY + 88, "=", { size: 30, weight: 300, anchor: "middle" }, dyn);
    const PX = AX + 235; sv.text(PX + 88, 48, "the change B·A", { size: 13, weight: 650, anchor: "middle" }, dyn); for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) cell(dyn, PX + j * CS, OY + i * CS, dW[i][j], mx, true);
    const full = 128 * 128, ad = r * 256;
    sv.rect(40, 290, 820, 62, { fill: "var(--bg2)", stroke: "var(--line2)", rx: 10 }, dyn);
    sv.text(56, 312, `Shown at 16 × 16. The real matrices are 128 × 128: ${full.toLocaleString("en-US")} numbers, all frozen.`, { size: 12.5, fill: "var(--ink2)" }, dyn);
    sv.text(56, 336, `The adapter has r × (128 + 128) = ${ad.toLocaleString("en-US")} numbers per matrix (${(100 * ad / full).toFixed(0)}% of the matrix), and its change has at most ${r} independent patterns.`, { size: 12.5, weight: 650 }, dyn);
    fig.$live.hidden = false; fig.$live.innerHTML = "<b>The base stays untouched.</b> Only B and A learn. At rank 1 the change is one stripe pattern repeated, and higher ranks allow richer changes. V8 used rank 4 on two attention matrices. S6's scaled arm uses rank 16 on all attention and feed-forward matrices.";
  }
  draw();
});

/* ------------------------------------------------------------ gmmGate: when does the adapter switch on? */
G.reg("gmmGate", fig => {
  const W = 900, Hh = 430, sv = fig.svg(W, Hh, { label: "A gate built from two density models: new-task data versus everything else" });
  let ov = 0; const sl = ui.range("how much the new task overlaps the old one", 0, 1, 0.02, 0, v => { ov = v; draw(); }, v => v < 0.1 ? "separate" : v > 0.85 ? "same inputs" : Math.round(v * 100) + "%");
  fig.$ui.append(sl, ui.btn("Separate tasks", () => { ov = 0; sl.set(0); draw(); }), ui.btn("Shared format (like bind)", () => { ov = 0.9; sl.set(0.9); draw(); }));
  const PX = 20, PY = 36, PW = 560, PH = 370, dyn = sv.g({}), bars = sv.g({});
  const rf = rng(21), GEN = Array.from({ length: 220 }, () => [gauss(rf) * 2.2, gauss(rf) * 1.5]), OLD = Array.from({ length: 90 }, () => [-2.0 + gauss(rf) * .55, 0.5 + gauss(rf) * .5]), NEW0 = Array.from({ length: 90 }, () => [gauss(rf) * .55, gauss(rf) * .5]);
  const sx = x => PX + PW * (x + 6) / 12, sy = y => PY + PH * (1 - (y + 4) / 8);
  const fit = pts => { const n = pts.length, m = [0, 1].map(d => pts.reduce((a, p) => a + p[d], 0) / n), v = [0, 1].map(d => pts.reduce((a, p) => a + (p[d] - m[d]) ** 2, 0) / n + 1e-3); return { m, v }; };
  const lp = (g, p) => -0.5 * [0, 1].reduce((a, d) => a + Math.log(2 * Math.PI * g.v[d]) + (p[d] - g.m[d]) ** 2 / g.v[d], 0);
  function draw() {
    dyn.innerHTML = ""; bars.innerHTML = "";
    const centre = lerp(3.6, -2.0, ov), NEW = NEW0.map(p => [centre + p[0], 0.5 + p[1]]), pos = fit(NEW), neg = fit(GEN.concat(OLD)), open = p => lp(pos, p) > lp(neg, p);
    sv.rect(PX, PY, PW, PH, { fill: "var(--panel2)", stroke: "var(--line2)", rx: 12 }, dyn);
    for (let i = 0; i < 56; i++) for (let j = 0; j < 36; j++) { const x = -6 + 12 * (i + .5) / 56, y = -4 + 8 * (j + .5) / 36; if (open([x, y])) sv.rect(sx(x) - PW / 112, sy(y) - PH / 72, PW / 56 + .5, PH / 36 + .5, { fill: "var(--sel)", stroke: "none", rx: 0, opacity: .16 }, dyn); }
    GEN.forEach(p => sv.circle(sx(p[0]), sy(p[1]), 2.6, { fill: "var(--frozen)", opacity: .6 }, dyn)); OLD.forEach(p => sv.circle(sx(p[0]), sy(p[1]), 3.4, { fill: "var(--enc)", opacity: .9 }, dyn)); NEW.forEach(p => sv.circle(sx(p[0]), sy(p[1]), 3.4, { fill: "var(--sel)", opacity: .95 }, dyn));
    sv.text(PX + 8, PY - 12, "Inputs to one matrix, squeezed into 2-D. The shaded area is where the gate opens.", { size: 12.5, fill: "var(--ink3)" }, dyn);
    const rate = a => a.filter(open).length / a.length, ro = rate(OLD), rn = rate(NEW), rg = rate(GEN);
    const RX = 620; sv.text(RX, 56, "Where does the gate open?", { size: 13.5, weight: 650 }, bars);
    [["New task (should be high)", rn, "var(--sel)"], ["Old task (should be low)", ro, "var(--enc)"], ["Generic data (should be low)", rg, "var(--frozen)"]].forEach(([n, v, c], i) => { const y = 84 + i * 70; sv.text(RX, y, n, { size: 12.5, fill: "var(--ink2)" }, bars); sv.rect(RX, y + 12, 200, 24, { fill: "var(--bg2)", stroke: "none", rx: 6 }, bars); sv.rect(RX, y + 12, Math.max(2, 200 * v), 24, { fill: c, stroke: "none", rx: 6 }, bars); sv.text(RX + 210, y + 24, Math.round(v * 100) + "%", { size: 13, mono: true, weight: 700 }, bars); });
    sv.text(RX, 330, ro > 0.3 ? "The old task now opens this adapter.\nWherever the gate opens, the newest\nupdate wins, so the old skill gets\noverwritten there." : "The gate stays closed on old inputs,\nso the old skill is left alone.", { size: 12.5, fill: ro > 0.3 ? "var(--bad)" : "var(--good)", weight: 650 }, bars);
    fig.$live.hidden = false; fig.$live.innerHTML = "<b>The rule:</b> g(x) = 1 when the new task's density at x is higher than the generic density. One density model (Φpos) is fitted to the new task's inputs, another (Φneg) to everything else, including the old tasks. In S6, 16 and 32 components are used per matrix, after a random projection. Here each is a single Gaussian for clarity. <span class=\"muted\">The gate can only protect old skills where the inputs are different enough to tell apart. That is what S6 measures before it trusts LSL.</span>";
  }
  draw();
});

/* ------------------------------------------------------------ phases: three tasks in a row */
G.reg("phases", fig => {
  const sv = fig.svg(900, 400, { label: "Three phases of new tasks added to the frozen S1 model, each with a predicted separability" });
  const P = [
    { t: "P1 · new text", d: "Domain-shifted text.", exp: "expected separable", note: "Likely separable from puzzle data, but overlaps ordinary text. Repeats V8's phase 1, whose preservation was not established.", c: "enc" },
    { t: "P2 · a new puzzle kind", d: "Count a colour in a list (new generator), or the shifted-layout puzzle, whichever passes the separability check.", exp: "must pass G6.0", note: "The positive test of LSL. The colour-count task has a header no old task uses, so it is easy to tell apart.", c: "con" },
    { t: "P3 · bind", d: "Follow four colours in order.", exp: "predicted to overlap", note: "Bind shares the road-list format with navigation, lookup and pointer. The design predicts LSL will forget about as much as always-open does.", c: "loss" },
  ];
  sv.rect(20, 292, 860, 70, { fill: "var(--frozen-s)", stroke: "var(--frozen)", sw: 1.6, rx: 14, dash: "6 3" }); sv.text(450, 316, "The S1 model, frozen (seed 902 or V8's seed 811)", { size: 14, weight: 650, anchor: "middle" }); sv.text(450, 342, "encoder + concept stack + decoder · never changed, except in the VQ-only and full fine-tune arms", { size: 12, anchor: "middle", fill: "var(--ink2)" });
  const gs = P.map((p, i) => {
    const g = sv.g({}), x = 20 + i * 296;
    sv.rect(x, 24, 268, 200, { fill: "var(--panel)", stroke: `var(--${p.c})`, sw: 2, rx: 14 }, g); sv.rect(x, 24, 268, 40, { fill: `var(--${p.c})`, stroke: "none", rx: 14 }, g); sv.rect(x, 48, 268, 16, { fill: `var(--${p.c})`, stroke: "none", rx: 0 }, g);
    sv.text(x + 14, 44, p.t, { size: 14.5, weight: 700, fill: "var(--bg)" }, g);
    const wrapTxt = (str, n) => { let line = "", out = []; str.split(" ").forEach(w => { if ((line + " " + w).trim().length > n) { out.push(line); line = w; } else line = (line + " " + w).trim(); }); out.push(line); return out.join("\n"); };
    sv.text(x + 14, 100, wrapTxt(p.d, 34), { size: 12.5, fill: "var(--ink)" }, g);
    sv.rect(x + 14, 148, 150, 22, { fill: `var(--${p.c === "enc" ? "enc-s" : p.c === "con" ? "con-s" : "loss-s"})`, stroke: "none", rx: 11 }, g); sv.text(x + 89, 159, p.exp, { size: 11.5, weight: 700, anchor: "middle", fill: `var(--${p.c})` }, g);
    sv.text(x + 14, 196, wrapTxt(p.note, 40), { size: 11, fill: "var(--ink2)" }, g);
    sv.path(`M ${x + 134} 226 L ${x + 134} 288`, { stroke: `var(--${p.c})`, sw: 2, arrow: p.c, dash: "5 3" }, g);
    sv.rect(x + 34 + 0, 252, 200, 28, { fill: `var(--${p.c === "enc" ? "enc-s" : p.c === "con" ? "con-s" : "loss-s"})`, stroke: `var(--${p.c})`, sw: 1.6, rx: 8 }, g); sv.text(x + 134, 266, `adapter bank ${i + 1} + gate`, { size: 12, weight: 650, anchor: "middle" }, g);
    return g;
  });
  sv.text(450, 386, "2,000 updates per phase · learning rate 3e-4 · 2 seeds (931, 932) · 3 arms × 3 phases × 2 seeds = 18 phase-runs", { size: 12, anchor: "middle", fill: "var(--ink3)" });
  fig.$live.hidden = false; fig.$live.innerHTML = "<b>Before each phase</b> a quick check (G6.0) fits the new task's density model and counts how often the gate would open on every earlier task. Fewer than 5% on average, and no task above 15%, means <em>separable</em>. Up to 10% is weakly separable. Above that, non-separable. The phase is run and labelled either way.";
});

/* ------------------------------------------------------------ credit: the four conditions */
G.reg("credit", fig => {
  const C = [
    ["It really learns the new skill", "Installs at least +10 points on puzzles, or a text-loss improvement whose interval excludes zero.", "A method that learns nothing can look perfect on retention."],
    ["It keeps the old skills", "No retained suite loses more than 2 points, and text gets worse by no more than 0.005 nats.", "This is stricter than V8's own phase-1 loss of 0.0071."],
    ["It beats the always-open adapter", "The retained-suite loss is smaller than with the same adapter and no gate, by an interval that excludes zero.", "Without this, the gate may be doing nothing useful."],
    ["It installs about as well as VQ-only", "Within 5 points of the arm that only adapts the codebook.", "VQ-only is expected to retain best and install least."],
  ];
  const st = [true, true, true, true]; const box = H("div", { class: "gates", style: "padding:8px 14px 4px" }), verdict = H("div", { class: "verdict", style: "margin:4px 14px 12px", "aria-live": "polite" });
  fig.$stage.append(box, verdict);
  function draw() {
    box.innerHTML = ""; C.forEach(([h, d, why], i) => { const b = H("button", { type: "button", class: "gate", style: `text-align:left;cursor:pointer;font:inherit;border-color:${st[i] ? "var(--good)" : "var(--bad)"};background:${st[i] ? "var(--good-s)" : "var(--bad-s)"}`, onclick: () => { st[i] = !st[i]; draw(); } }, H("div", { class: "id" }, `Condition ${i + 1} · ${st[i] ? "met" : "not met"} (click to flip)`), H("h4", {}, h), H("p", {}, d), H("p", { class: "muted", style: "font-size:13px" }, why)); box.append(b); });
    const all = st.every(Boolean); verdict.className = "verdict " + (all ? "ok" : "no"); verdict.textContent = all ? "All four hold on this phase, so LSL is credited with retention on it." : `${st.filter(v => !v).length} of 4 not met, so LSL gets no retention credit on this phase, however good it looks on the others.`;
  }
  draw();
});
})();
