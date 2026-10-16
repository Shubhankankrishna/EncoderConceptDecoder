/* Compare page (DESIGN.md §2, advisor calls #14 / #15): v1, v2, v3 on shared charts.
   x = optimizer step (same tokens in the same order through v3's epoch 0, step <= 8,556) or wall-clock training hours
   (what each run achieved with the same time). Data: data/ref_v1v2.json + data/v3_live.json; no new files. */
(function () {
  const { chart, interp, fmt, fI } = LC;
  const RUNS = { v1: { name: "v1 · 25M (d 384)", color: "#3b78c2", dash: "6 4" }, v2: { name: "v2 · 19M MoE", color: "#7a5bd0", dash: "6 4" },
                 v3: { name: "v3 · 19M", color: "#168b76", dash: null } };
  const EPOCH0 = 8556;
  let REF = null, V3 = null, xmode = "step", show = { v1: true, v2: true, v3: true }, atS = "latest", timer = null, lastKey = null;

  function runs() {                                                   // unified {evals, train} per run
    const H = V3?.history || {}, wh = new Map((H.wall_h || []).map(([s, h]) => [s, h]));
    const tr3 = (H.ntp || []).map(([s, v]) => ({ step: s, ntp: v, wall_h: wh.get(s) }))
      .map((r, i) => ({ ...r, tok_s: (H.tok_s || [])[i]?.[1] }));
    return { v1: { evals: REF.v1.evals, train: REF.v1.train }, v2: { evals: REF.v2.evals, train: REF.v2.train },
             v3: { evals: V3?.evals || [], train: tr3 } };
  }
  const X = r => xmode === "step" ? r.step : r.wall_h;
  const pts = (rows, key) => rows.filter(r => r[key] != null && X(r) != null).map(r => [X(r), r[key]]);

  function series(R, key, src = "evals") {
    return Object.keys(RUNS).filter(k => show[k]).map(k => ({ name: RUNS[k].name, color: RUNS[k].color, dash: RUNS[k].dash,
      pts: pts(R[k][src], key), dots: src === "evals" && k === "v3", w: k === "v3" ? 2.6 : 1.8 }));
  }
  function opts(extra) {
    const o = xmode === "step"
      ? { xname: "step", vlines: [{ x: EPOCH0, label: "v3 epoch 0 ends", color: "#c98311" }, { x: 1217, label: "1,217" }, { x: 1792, label: "1,792" }, { x: 2376, label: "2,376" }] }
      : { xname: "h", xlabel: "training hours", xfmt: v => (+v).toFixed(1), vlines: [{ x: 24, label: "24 h", color: "#c98311" }] };
    return Object.assign(o, { x0: 0 }, extra);
  }
  const card = (id, title, sub, body, wide) => `<div class="card v3t-card${wide ? " wide" : ""}" id="${id}"><div class="v3t-h"><h4>${title}</h4><span class="muted">${sub}</span></div>${body}</div>`;

  function atTable(R) {
    const v3e = R.v3.evals, last3 = v3e.at(-1);
    const S = atS === "latest" ? (last3 ? last3.step : 1217) : +atS;
    const keys = [["val", "val (mean)", 3], ["ts", "TinyStories", 3], ["wk", "WikiText", 3], ["recall1", "recall1", 3], ["carry3", "carry3", 3],
                  ["multi4", "multi4", 3], ["hop", "hop", 3], ["bind", "bind", 3], ["swap", "plan-swap recall1", 3], ["VE", "codebook VE", 3], ["wall_h", "hours to reach", 1]];
    const val = (k, key) => {
      const rows = R[k].evals; if (!rows.length) return null;
      if (k === "v3") { const e = rows.filter(r => r.step <= S + 5).at(-1); return e && Math.abs(e.step - S) <= Math.max(60, 0.1 * S) ? e[key] : interp(rows, S, key); }
      return interp(rows, S, key);
    };
    const cols = Object.keys(RUNS);
    const better = (key, a, b) => a == null || b == null || Math.abs(a - b) < 5e-4 ? "" : (["val", "ts", "wk", "wall_h"].includes(key) ? a < b : a > b) ? "ok" : "ba";
    return `<div class="cmp-at"><div class="cmp-at-h"><b>At step</b> ${["1217", "1792", "2376", "4050", "latest"].map(s => `<button class="sn-page ${String(atS) === s ? "on" : ""}" data-at="${s}">${s === "latest" ? "v3 latest" : fI(+s)}</button>`).join("")}
      <span class="muted">step ${fI(S)}${S > EPOCH0 ? " · beyond v3's epoch 0: data no longer matched" : " · same tokens in the same order"}</span></div>
      <table class="t"><thead><tr><th>metric</th>${cols.map(k => `<th style="color:${RUNS[k].color}">${RUNS[k].name}</th>`).join("")}<th>v3 − v1</th></tr></thead><tbody>${
      keys.map(([key, n, d]) => { const v = cols.map(k => val(k, key)); const dv = v[2] != null && v[0] != null ? v[2] - v[0] : null;
        return `<tr><td>${n}</td>${v.map((x, i) => `<td class="${i === 2 ? better(key, x, v[0]) : ""}">${fmt(x, d)}</td>`).join("")}<td>${dv == null ? "–" : (dv >= 0 ? "+" : "") + fmt(dv, d)}</td></tr>`; }).join("")
      }</tbody></table><div class="muted" style="font-size:12px;margin-top:6px">v1 / v2 interpolated between their evaluations; v3 = its evaluation at that step when one exists. Green = v3 better than v1, red = worse. “–” = the run had not reached that step.</div></div>`;
  }

  function render() {
    const R = runs(), root = document.getElementById("cmp-root");
    const xmax3 = xmode === "step" ? Math.max(...R.v3.evals.map(e => e.step), V3?.step || 0, 0) : Math.max(V3?.wall_h || 0, 0);
    const xmax = Math.max(xmode === "step" ? 4100 : 24.5, xmax3 * 1.05);
    const O = e => opts(Object.assign({ x1: xmax }, e));
    root.innerHTML = `<div class="cmp-bar card">
        <div class="sn-group"><span class="sn-l">x-axis</span><button class="sn-page ${xmode === "step" ? "on" : ""}" data-x="step">same step (same tokens)</button><button class="sn-page ${xmode === "hours" ? "on" : ""}" data-x="hours">same hours</button></div>
        <div class="sn-group"><span class="sn-l">runs</span>${Object.keys(RUNS).map(k => `<button class="sn-chip ${show[k] ? "on" : ""} sn-${k}" data-run="${k}">${RUNS[k].name}</button>`).join("")}</div>
        <div class="muted cmp-note">${xmode === "step" ? "Same step = same training data seen so far (all runs: seed 0, v1's row order). Valid through v3's first epoch." : "Same hours = training time excluding pauses and out-of-process evaluations. v3 runs ≈ 2.4× more steps per hour than v1."}</div></div>
      ${atTable(R)}
      <div class="v3t-grid" style="margin-top:14px">
      ${card("cm-val", "Validation loss (mean of TinyStories and WikiText)", "nats · lower is better · quick eval on 128 rows per language", chart("cm-val", O({ W: 1140, H: 300, series: series(R, "val") })), true)}
      ${card("cm-ts", "TinyStories", "validation nats", chart("cm-ts", O({ series: series(R, "ts") })))}
      ${card("cm-wk", "WikiText", "validation nats", chart("cm-wk", O({ series: series(R, "wk") })))}
      ${card("cm-rec", "Long-range recall1", "exact match, 128 items · the concept route", chart("cm-rec", O({ series: series(R, "recall1"), y0: 0, y1: 1, refs: xmode === "step" ? [{ y: 0.25, label: "E1 line", color: "#c98311" }] : [] })))}
      ${card("cm-carry", "Harder long-range tasks", "carry3 (solid) · bind (dotted), exact match", chart("cm-carry", O({ series: series(R, "carry3").concat(series(R, "bind").map(s => ({ ...s, name: s.name + " bind", dash: "1 3", dots: false }))), y0: 0 })))}
      ${card("cm-swap", "Plan-swap recall1 (should stay ≈ 0)", "recall with another row's plan: > 0 would mean the answer leaks around the concept route", chart("cm-swap", O({ series: series(R, "swap"), y0: 0 })))}
      ${card("cm-ve", "Codebook variance explained", "VE of the PQ codes on real chunks", chart("cm-ve", O({ series: series(R, "VE"), y0: 0, y1: 1 })))}
      ${card("cm-ntp", "Training NTP loss", "logged windows · v3's is a full-batch mean over valid targets", chart("cm-ntp", O({ series: series(R, "ntp", "train") })))}
      ${card("cm-tok", "Throughput", "tokens (slots) per second", chart("cm-tok", O({ series: series(R, "tok_s", "train"), tfmt: v => fI(v), y0: 0 })))}
      </div>`;
  }

  async function load() {
    if (!REF) REF = await (await fetch("data/ref_v1v2.json")).json();
    try { V3 = await fetchLive("data/v3_live.json"); } catch (e) { V3 = null; }
    const key = V3 ? V3.step + ":" + (V3.evals || []).length : "none";
    if (key !== lastKey || !document.querySelector("#cmp-root .cmp-bar")) { lastKey = key; render(); }
  }
  document.addEventListener("click", e => {
    if (!e.target.closest("#cmp-root")) return;
    const x = e.target.closest("[data-x]"), r = e.target.closest("[data-run]"), a = e.target.closest("[data-at]");
    if (x) xmode = x.dataset.x; else if (r) show[r.dataset.run] = !show[r.dataset.run]; else if (a) atS = a.dataset.at; else return;
    render();
  });
  window.addEventListener("tabshown", e => {
    if (e.detail !== "compare") { if (timer) { clearInterval(timer); timer = null; } return; }
    load(); if (!timer) timer = setInterval(load, 60000);
  });
  if ((document.querySelector("section.tab.on") || {}).id === "tab-compare") load();
})();
