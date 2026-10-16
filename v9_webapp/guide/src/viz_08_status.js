/* Chapter 9 figures: queue (status cards from the run files) and zones (how an interval becomes a verdict) */
(function () {
const { H, S, ui, clamp } = G;
const D = () => window.GUIDE_DATA || { s1: {}, s2: {} };

G.reg("queue", fig => {
  const S2 = D().s2 || {}, gen = D().generated || "";
  const cells = ["S2_V8repro_dense_s911", "S2_V8repro_dense_s912", "S2_K1_dense_s911", "S2_K1_dense_s912", "S2_K8_dense_s911", "S2_K8_dense_s912"];
  const measured = cells.filter(n => S2[n] && S2[n].measure).length, training = cells.map(n => S2[n]).find(r => r && !r.wall_s);
  const CARDS = [
    ["Q1", "S1 foundation", "done", "finished", "Six runs of 24,000 updates plus checks, then the one sealed evaluation. About 17.7 hours while other work shared the machine (the design said about 12).", "mandatory"],
    ["Q3a", "S2 dense tier", measured === 6 ? "done" : "run", measured === 6 ? "measured, awaiting gates" : "running", `${measured} of 6 training-and-measurement cells done${training ? `; one is training (update ${training.steps.toLocaleString("en-US")} of 12,000)` : ""}. Then a V8 replay, the S2 gates, and the decision about which model is promoted to S3 to S5.`, "mandatory"],
    ["Q2", "S6 continual learning (base)", "wait", "queued next", "Starts with a timing pilot, then 18 phase-runs. About 6 hours planned, provisional until the pilot.", "mandatory"],
    ["Q3b", "S2 recurrent tier", "wait", "queued", "The same comparisons with the smaller recurrent core. About 4.6 hours planned.", "mandatory"],
    ["Q4", "S3 refit", "wait", "waiting for S2", "S3a-dec and S3b on two seeds. About 3 hours planned.", "mandatory"],
    ["Q5", "S4 post-training", "wait", "waiting for S3", "Expert iteration and three baselines per promoted seed. About 20 hours planned, provisional.", "mandatory"],
    ["Q6", "S5 search", "wait", "waiting for S4", "Five methods, three budgets. Between about 3 and 38 hours depending on whether search can batch puzzles. Sized after a speed test.", "mandatory"],
    ["QF", "Final analysis", "wait", "last", "Pools every result, applies the multiple-testing correction within each stage, and writes the final report.", "mandatory"],
  ];
  const wrap = H("div", { class: "arms", style: "padding:6px 14px 10px" });
  CARDS.forEach(([id, t, cls, st, d]) => wrap.append(H("div", { class: "arm" }, H("div", { class: "chips" }, H("span", { class: "chip " + cls }, st)), H("h4", {}, id + " · " + t), H("p", {}, d))));
  fig.$stage.append(wrap, H("div", { class: "note" }, `Run order on the single GPU: Q1, Q3a, Q2, Q3b, Q4, Q5, Q6, QF. Hours are the design's estimates, not promises. Status read from the run files at ${gen}.`));
});

G.reg("zones", fig => {
  const sv = fig.svg(900, 300, { label: "A confidence interval compared with pass, partial and fail zones" });
  let est = 38, unc = 8;
  fig.$ui.append(ui.range("measured swap sensitivity", 0, 60, 1, 38, v => { est = v; draw(); }, v => v + "%"), ui.range("uncertainty (± at 95%)", 1, 25, 1, 8, v => { unc = v; draw(); }, v => "±" + v));
  const dyn = sv.g({}), X0 = 60, X1 = 840, px = v => X0 + (X1 - X0) * v / 60;
  function draw() {
    dyn.innerHTML = "";
    sv.rect(px(0), 80, px(15) - px(0), 70, { fill: "var(--bad-s)", stroke: "none", rx: 0 }, dyn); sv.rect(px(15), 80, px(30) - px(15), 70, { fill: "var(--warn-s)", stroke: "none", rx: 0 }, dyn); sv.rect(px(30), 80, px(60) - px(30), 70, { fill: "var(--good-s)", stroke: "none", rx: 0 }, dyn);
    sv.text((px(0) + px(15)) / 2, 100, "FAIL zone", { size: 12.5, weight: 700, anchor: "middle", fill: "var(--bad)" }, dyn); sv.text((px(15) + px(30)) / 2, 100, "PARTIAL band", { size: 12.5, weight: 700, anchor: "middle", fill: "var(--warn)" }, dyn); sv.text((px(30) + px(60)) / 2, 100, "PASS zone", { size: 12.5, weight: 700, anchor: "middle", fill: "var(--good)" }, dyn);
    sv.text((px(0) + px(15)) / 2, 124, "below 15%", { size: 11.5, anchor: "middle", fill: "var(--ink2)" }, dyn); sv.text((px(15) + px(30)) / 2, 124, "15% to 30%", { size: 11.5, anchor: "middle", fill: "var(--ink2)" }, dyn); sv.text((px(30) + px(60)) / 2, 124, "30% and above", { size: 11.5, anchor: "middle", fill: "var(--ink2)" }, dyn);
    for (let v = 0; v <= 60; v += 10) { sv.line(px(v), 150, px(v), 156, { stroke: "var(--ink3)", sw: 1 }, dyn); sv.text(px(v), 170, v + "%", { size: 11, mono: true, anchor: "middle", fill: "var(--ink3)" }, dyn); }
    const lo = clamp(est - unc, 0, 60), hi = clamp(est + unc, 0, 60);
    sv.line(px(lo), 52, px(hi), 52, { stroke: "var(--ink)", sw: 4 }, dyn); sv.line(px(lo), 44, px(lo), 60, { stroke: "var(--ink)", sw: 3 }, dyn); sv.line(px(hi), 44, px(hi), 60, { stroke: "var(--ink)", sw: 3 }, dyn); sv.circle(px(est), 52, 7, { fill: "var(--accent)", stroke: "var(--panel)", sw: 2.5 }, dyn);
    sv.text(px(lo), 34, lo.toFixed(0) + "%", { size: 11.5, mono: true, anchor: "middle" }, dyn); sv.text(px(hi), 34, hi.toFixed(0) + "%", { size: 11.5, mono: true, anchor: "middle" }, dyn);
    const v = lo >= 30 ? "PASS" : hi < 15 ? "FAIL" : "INCONCLUSIVE", col = v === "PASS" ? "var(--good)" : v === "FAIL" ? "var(--bad)" : "var(--warn)";
    sv.rect(60, 200, 780, 70, { fill: "var(--panel)", stroke: col, sw: 2.2, rx: 12 }, dyn); sv.text(80, 224, v, { size: 20, weight: 700, fill: col }, dyn);
    sv.text(80, 252, v === "PASS" ? "The whole interval sits in the pass zone. The gate holds for this seed." : v === "FAIL" ? "The whole interval sits in the fail zone. The gate fails for this seed." : lo >= 15 && hi < 30 ? "The interval sits in the partial band. Proceed, flagged, with no capability claim." : "The interval straddles a boundary. Proceed under the partial consequence and claim nothing.", { size: 12.5, fill: "var(--ink2)" }, dyn);
    fig.$live.hidden = false; fig.$live.innerHTML = "<b>The rule from the pre-registration.</b> A gate passes only if the <em>whole</em> 95% interval is in the pass zone, and fails only if the whole interval is in the fail zone. Anything else is inconclusive. The example uses the swap-sensitivity row of gate G2.1. Each gate is judged per seed, and passing needs both seeds to pass. A point estimate alone never decides.";
  }
  draw();
});
})();
