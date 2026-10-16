/* Chapter 2 figures: graphnav (interactive puzzle + token view) and chunks (tokens squeezed into concepts) */
(function () {
const { H, S, ui, loop, clamp, rng, Svg } = G;
const ROAD = [{ n: "blue", c: "#2a78d6", dash: null }, { n: "orange", c: "#eb6834", dash: "8 4" }, { n: "green", c: "#1baf7a", dash: "2.5 4.5" }];
const NAMES = "ABCDEFGHIJ";

/* same generator logic as the frozen data.py: standard = 2 to 4 colour runs of 2 to 4 hops; hard = shortest path of at least 4 hops */
function makePuzzle(seed, hard) {
  const r = rng(seed * 7919 + 13), ri = (a, b) => a + Math.floor(r() * (b - a)), n = 10;
  for (let tries = 0; tries < 800; tries++) {
    const f = [0, 1, 2].map(() => Array.from({ length: n }, (_, v) => { let d = ri(0, n - 1); if (d >= v) d++; return d; }));
    const start = ri(0, n);
    let walk = [], goal;
    if (hard) {
      const dist = { [start]: 0 }, path = { [start]: [] }, q = [start];
      for (const v of q) for (let c = 0; c < 3; c++) { const u = f[c][v]; if (!(u in dist)) { dist[u] = dist[v] + 1; path[u] = path[v].concat([[c, u]]); q.push(u); } }
      const far = Object.keys(dist).map(Number).filter(g => dist[g] >= 4); if (!far.length) continue;
      goal = far[ri(0, far.length)]; walk = path[goal];
    } else {
      let cur = start, prev = -1; const vis = [start];
      for (let run = ri(2, 5), k = 0; k < run; k++) {
        let c = prev >= 0 ? ri(0, 2) : ri(0, 3); if (prev >= 0 && c >= prev) c++; prev = c;
        for (let d = ri(2, 5), m = 0; m < d; m++) { cur = f[c][cur]; walk.push([c, cur]); vis.push(cur); }
      }
      goal = cur; if (vis.slice(0, -1).includes(goal)) continue;
    }
    const edges = []; for (let v = 0; v < n; v++) for (let c = 0; c < 3; c++) edges.push([v, c, f[c][v]]);
    for (let i = edges.length - 1; i > 0; i--) { const j = ri(0, i + 1); [edges[i], edges[j]] = [edges[j], edges[i]]; }
    return { f, start, goal, walk, edges, hard };
  }
}
const tk = (txt, cls, sub) => H("span", { class: "tk " + (cls || "") }, H("b", {}, txt), sub ? H("i", {}, sub) : null);
const special = t => tk(t, "ph");
const town = v => tk(NAMES[v]);
const road = c => tk(ROAD[c].n, "c" + c);
const grp = (toks, cap) => H("div", {}, cap ? H("div", { class: "tkcap" }, cap) : null, H("div", { class: "tkgrp" }, toks));

G.reg("graphnav", fig => {
  let seed = 3, hard = false, P, cur, hops, finished, msg, showAll = false;
  fig.$stage.classList.add("split2");
  const left = H("div"), right = H("div");
  fig.$stage.append(left, right);
  const sv = new Svg(left, 460, 400, { label: "Ten towns joined by blue, orange and green one-way roads", narrow: true });
  [0, 1, 2].forEach(c => { const m = S("marker", { id: `${sv.id}-r${c}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto" }, sv.defs); S("path", { d: "M0 0 L10 5 L0 10 z", fill: ROAD[c].c }, m); });
  const CX = 230, CY = 205, R = 150, pos = i => { const a = -Math.PI / 2 + i * 2 * Math.PI / 10; return [CX + R * Math.cos(a), CY + R * Math.sin(a)]; };
  const edgeG = sv.g({}), nodeG = sv.g({});
  const status = H("div", { class: "verdict", "aria-live": "polite" });
  const btns = ROAD.map((rd, c) => { const b = H("button", { class: "btn", type: "button" }, H("span", { class: "dot", style: `background:${rd.c}` }), "Take " + rd.n + " road"); b.addEventListener("click", () => step(c)); b.addEventListener("mouseenter", () => draw(c)); b.addEventListener("mouseleave", () => draw()); return b; });
  const done = ui.btn("Say DONE", sayDone, "pri");
  fig.$ui.append(ui.seg([["std", "Standard"], ["hard", "Hard"]], "std", v => { hard = v === "hard"; fresh(); }, "difficulty"), ui.btn("New puzzle", () => { seed++; fresh(); }), ...btns, done, ui.btn("Undo", () => { if (finished) finished = false; else hops.pop(); recompute(); msg = null; draw(); }), ui.btn("Show a solution", () => { hops = P.walk.map(([c, to]) => ({ c, to })); finished = false; recompute(); msg = ["", "A shortest-style solution is filled in. Now say DONE to finish."]; draw(); }));
  function fresh() { P = makePuzzle(seed, hard); hops = []; finished = false; msg = null; showAll = false; recompute(); draw(); }
  function recompute() { cur = P.start; hops.forEach(h => { cur = h.to; }); }
  function step(c) { if (finished) return; const to = P.f[c][cur]; hops.push({ c, to }); cur = to; msg = null; draw(); }
  function sayDone() {
    if (finished) return;
    if (hops.length && cur === P.goal) { finished = true; msg = ["ok", `Accepted. ${hops.length} hops, DONE said at the goal. The verifier is satisfied.`]; }
    else msg = ["no", hops.length ? "Rejected: DONE is only legal when you stand on the goal town. Keep going." : "Rejected: you cannot say DONE before taking at least one road."];
    draw();
  }
  function quad(a, b, c) { const [x1, y1] = pos(a), [x2, y2] = pos(b); const dx = x2 - x1, dy = y2 - y1, d = Math.hypot(dx, dy) || 1, k = [0.26, -0.26, 0.09][c] * d; const mx = (x1 + x2) / 2 - dy / d * k, my = (y1 + y2) / 2 + dx / d * k; return { x1, y1, x2, y2, mx, my }; }
  function draw(hover) {
    edgeG.innerHTML = ""; nodeG.innerHTML = "";
    P.edges.forEach(([v, c, to]) => { if (v === cur && !finished) return; const q = quad(v, to, c); S("path", { d: `M${q.x1} ${q.y1} Q${q.mx} ${q.my} ${q.x2} ${q.y2}`, fill: "none", stroke: ROAD[c].c, "stroke-width": 1.1, opacity: .17 }, edgeG); });
    let at = P.start; hops.forEach(h => { const q = quad(at, h.to, h.c); S("path", { d: `M${q.x1} ${q.y1} Q${q.mx} ${q.my} ${q.x2} ${q.y2}`, fill: "none", stroke: ROAD[h.c].c, "stroke-width": 6, opacity: .42, "stroke-linecap": "round" }, edgeG); at = h.to; });
    if (!finished) for (let c = 0; c < 3; c++) { const to = P.f[c][cur], q = quad(cur, to, c), on = hover === c; const sh = 21 / Math.max(1, Math.hypot(q.x2 - q.mx, q.y2 - q.my)); const ex = q.x2 + (q.mx - q.x2) * sh, ey = q.y2 + (q.my - q.y2) * sh;
      S("path", { d: `M${q.x1} ${q.y1} Q${q.mx} ${q.my} ${ex} ${ey}`, fill: "none", stroke: ROAD[c].c, "stroke-width": on ? 4.6 : 2.8, "stroke-dasharray": ROAD[c].dash || "none", "stroke-linecap": "round", "marker-end": `url(#${sv.id}-r${c})` }, edgeG); }
    for (let v = 0; v < 10; v++) {
      const [x, y] = pos(v), isCur = v === cur, isGoal = v === P.goal, isStart = v === P.start;
      S("circle", { cx: x, cy: y, r: 19, fill: isCur ? "var(--accent)" : isGoal ? "var(--code-s)" : "var(--panel)", stroke: isGoal ? "var(--code)" : isStart ? "var(--good)" : "var(--line2)", "stroke-width": isGoal || isStart || isCur ? 3 : 1.4 }, nodeG);
      const t = S("text", { x, y, "text-anchor": "middle", "dominant-baseline": "central", "font-size": 15, "font-weight": 700, fill: isCur ? "var(--bg)" : "var(--ink)" }, nodeG); t.textContent = NAMES[v];
      if (isStart || isGoal) { const lab = S("text", { x, y: y + (y > CY ? 35 : -33), "text-anchor": "middle", "font-size": 11, "font-weight": 650, fill: isGoal ? "var(--code)" : "var(--good)" }, nodeG); lab.textContent = isGoal && isStart ? "START + GOAL" : isGoal ? "GOAL" : "START"; }
    }
    S("text", { x: 14, y: 20, "font-size": 13, fill: "var(--ink2)" }, nodeG).textContent = finished ? "Finished." : `You are in town ${NAMES[cur]}. The goal is ${NAMES[P.goal]}.`;
    btns.forEach(b => { b.disabled = finished; }); done.disabled = finished;
    panel();
  }
  function panel() {
    right.innerHTML = "";
    right.append(H("div", { class: "tkcap" }, "What the model reads: the puzzle"), grp([special("GNAV"), town(P.goal), town(P.start), special("SEP")], "instruction chunk: navigate, goal, start"));
    const roads = P.edges, shown = showAll ? roads : roads.slice(0, 4);
    const list = H("div", { class: showAll ? "scrollbox" : "" });
    shown.forEach(([v, c, to], i) => list.append(H("div", { style: "margin-bottom:5px" }, grp([town(v), road(c), town(to), special("SEP")], i < 1 || showAll ? `road chunk ${i + 1}: from, colour, to` : null))));
    right.append(list, H("button", { class: "btn", type: "button", style: "margin:2px 0 8px", onclick: () => { showAll = !showAll; panel(); } }, showAll ? "Show fewer roads" : `Show all 30 road chunks (${roads.length - 4} more)`));
    right.append(H("div", { class: "tkcap" }, "What the model writes: your answer so far"));
    const ans = H("div", { style: "display:flex;flex-wrap:wrap;gap:6px" });
    hops.forEach((h, i) => ans.append(grp([road(h.c), special("TO"), town(h.to), special("SEP")], "hop " + (i + 1))));
    if (finished) ans.append(grp([special("DONE"), special("SEP"), special("SEP"), special("SEP")], "stop"));
    if (!hops.length) ans.append(H("span", { class: "muted", style: "font-size:14px" }, "nothing yet"));
    right.append(ans, status);
    status.className = "verdict" + (msg && msg[0] ? " " + msg[0] : ""); status.textContent = msg ? msg[1] : `${hops.length} hop${hops.length === 1 ? "" : "s"} so far. Every hop is one group of four tokens.`; right.append(status);
  }
  fresh();
});

/* ------------------------------------------------------------ chunks: squeeze tokens into concepts */
G.reg("chunks", fig => {
  const W = 900, Hh = 330, sv = fig.svg(W, Hh, { label: "Four tokens squeezed into one concept vector" });
  const CH = [
    { cap: "a road in the puzzle", t: ["A", "blue", "C", "SEP"], k: [0, 1, 0, 3] },
    { cap: "another road", t: ["C", "orange", "F", "SEP"], k: [0, 2, 0, 3] },
    { cap: "answer: hop 1", t: ["blue", "TO", "C", "SEP"], k: [1, 3, 0, 3] },
    { cap: "answer: hop 2", t: ["orange", "TO", "F", "SEP"], k: [2, 3, 0, 3] },
  ];
  const cx = i => 135 + i * 210, TW = 46, SP = 50, YT = 100, YC = 235;
  const fillOf = k => k === 1 ? "var(--enc-s)" : k === 2 ? "color-mix(in srgb, #eb6834 22%, var(--panel))" : k === 3 ? "var(--code-s)" : "var(--panel2)";
  const strokeOf = (txt) => ["blue"].includes(txt) ? "#2a78d6" : txt === "orange" ? "#eb6834" : ["SEP", "TO"].includes(txt) ? "var(--code)" : "var(--line2)";
  const toks = [], cons = [], links = [];
  CH.forEach((c, i) => {
    sv.text(cx(i), 30, c.cap, { size: 13, anchor: "middle", fill: "var(--ink2)", weight: 600 });
    const grpEl = sv.rect(cx(i) - 104, 48, 208, 100, { fill: "var(--panel2)", stroke: "var(--line2)", rx: 14, dash: "5 4", sw: 1.2 });
    c.t.forEach((txt, j) => {
      const g = sv.g({});
      sv.rect(-TW / 2, -20, TW, 40, { fill: "var(--panel)", stroke: strokeOf(txt), sw: 1.6, rx: 8 }, g);
      sv.text(0, 0, txt, { size: txt.length > 5 ? 10.5 : 12.5, weight: 600, anchor: "middle", mono: true }, g);
      toks.push({ g, i, j });
    });
    const cg = sv.g({}); sv.rect(cx(i) - 38, YC - 34, 76, 68, { fill: "var(--con-s)", stroke: "var(--con)", sw: 1.8, rx: 14 }, cg);
    for (let b = 0; b < 8; b++) sv.rect(cx(i) - 28 + b * 7, YC - 24, 5, 48, { fill: "var(--con)", stroke: "none", rx: 1.5, opacity: .25 + (((b * 53 + i * 29) % 70) / 100) }, cg);
    cons.push(cg); sv.text(cx(i), YC + 52, "concept " + i, { size: 12.5, anchor: "middle", weight: 650, fill: "var(--con)" });
    links.push(sv.path(`M ${cx(i)} 150 L ${cx(i)} ${YC - 36}`, { stroke: "var(--con)", sw: 2, arrow: "con" }));
  });
  sv.text(W / 2, Hh - 12, "each concept = the average of its four token vectors, rescaled", { size: 12.5, anchor: "middle", fill: "var(--ink3)" });
  let t = 0, phase = 0, dir = 1;
  const sl = ui.range("squeeze", 0, 1, 0.01, 0, v => { t = phase = v; fig.playing = false; pb.textContent = "Play"; draw(); }, v => Math.round(v * 100) + "%");
  const pb = ui.btn("Play", () => { fig.playing = !fig.playing; pb.textContent = fig.playing ? "Pause" : "Play"; if (fig.playing) G.kick(); });
  fig.$ui.append(sl, pb); fig.playing = !G.REDUCED; pb.textContent = fig.playing ? "Pause" : "Play"; if (G.REDUCED) { t = phase = 1; sl.set(1); }
  function draw() {
    toks.forEach(({ g, i, j }) => { const x = cx(i) + (j - 1.5) * SP * (1 - 0.8 * t); g.setAttribute("transform", `translate(${x} ${YT + 8 * t})`); g.style.opacity = 1 - 0.5 * t; });
    cons.forEach((c, i) => { c.style.opacity = clamp((t - 0.15) / 0.6, 0, 1); links[i].style.opacity = clamp((t - 0.1) / 0.5, 0, 1); });
  }
  loop(fig, (tt, dt) => { phase += dir * dt * 0.3; if (phase >= 1.3) dir = -1; if (phase <= -0.3) dir = 1; t = clamp(phase, 0, 1); draw(); sl.set(t); });
  draw();
});
})();
