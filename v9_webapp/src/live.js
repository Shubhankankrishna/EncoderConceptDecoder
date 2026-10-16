/* V9 Atlas · Live telemetry page.
   Source: serve.py's /api/telemetry (polled), else the snapshot baked in at build time (window.V9_SNAPSHOT), else offline.
   Everything shown is a development diagnostic: step-log losses and the 2k-update dev checks. Times are IST. */
(function () {
"use strict";
const $ = s => document.querySelector(s), $$ = s => Array.from(document.querySelectorAll(s));
const VV = window.V9V;
const L = window.V9L = { data: null, source: "offline", fetched: 0 };
const IST = "Asia/Kolkata";
const fmtT = (ms, o) => new Intl.DateTimeFormat("en-GB", Object.assign({ timeZone: IST, hour: "2-digit", minute: "2-digit", hour12: false }, o || {})).format(new Date(ms));
const fmtDT = ms => new Intl.DateTimeFormat("en-GB", { timeZone: IST, day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false }).format(new Date(ms));
const dur = s => { if (s == null || !isFinite(s)) return "–"; s = Math.max(0, s); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60); return h ? `${h} h ${String(m).padStart(2, "0")} m` : `${m} m`; };
const pct = (v, d = 1) => v == null ? "–" : (100 * v).toFixed(d) + "%";
const f3 = v => v == null ? "–" : v.toFixed(3);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const utcMs = s => s ? Date.parse(s) : null;

/* ================================================================ state */
const ST = { from: 2000, metric: "text", smooth: 3, logy: false, hidden: new Set(), suite: "standard", stepMode: "latest", step: 12000, sig: "", active: false, url: "" };
try { ST.url = localStorage.getItem("v9-telemetry-url") || ""; } catch (e) {}
const METRICS = [["text", "text loss", "text"], ["symbolic", "symbolic loss", "symbolic"], ["loss", "total", "loss"], ["graph_node_accuracy", "graph acc.", "graph_node_accuracy"], ["gradient_norm", "grad norm", "gradient_norm"], ["lr", "lr", "lr"]];
const ARM_COL = { A_delayed: "--enc", A_alpha0: "--sel", A_flat: "--loss" };
function parseRun(n) { const m = /^(S\d)_(.+?)_s(\d+)(?:_(.*))?$/.exec(n); return m ? { stage: m[1], arm: m[2] + (m[4] ? "_" + m[4] : ""), seed: +m[3] } : { stage: (/^(S\d)_/.exec(n) || [])[1] || "?", arm: n.replace(/^S\d_/, ""), seed: 0 }; }
function runStyle(n) {
  const p = parseRun(n), col = ARM_COL[p.arm] || ({ S1: "--code", S2: "--con", S3: "--dec", S4: "--sel", S6: "--loss" })[p.stage] || "--ink2";
  const seeds = Object.keys(L.data.runs).map(parseRun).filter(q => q.arm === p.arm).map(q => q.seed).sort(), idx = Math.max(0, seeds.indexOf(p.seed));
  return { color: `var(${col})`, dash: p.arm === "A_flat" ? "6 4" : "", op: idx ? .62 : 1, w: idx ? 1.7 : 2.2, label: `${p.arm} · s${p.seed}`, p };
}
const S1runs = () => Object.keys(L.data.runs).filter(n => /^S1_/.test(n)).sort((a, b) => (parseRun(a).arm === "A_flat") - (parseRun(b).arm === "A_flat") || a.localeCompare(b));
const col = (r, k) => (r.series.cols && r.series.cols[k]) || [];
const lastOf = a => { for (let i = a.length - 1; i >= 0; i--) if (a[i] != null) return a[i]; return null; };
function medianSpeed(r) {          // seconds per update over the last ten log intervals
  const st = col(r, "step"), el = col(r, "elapsed_s"), v = [];
  for (let i = Math.max(1, st.length - 10); i < st.length; i++) if (st[i] > st[i - 1]) v.push((el[i] - el[i - 1]) / (st[i] - st[i - 1]));
  v.sort((a, b) => a - b); return v.length ? v[Math.floor(v.length / 2)] : null;
}
const isRunning = r => !r.completed && r.series.n > 0;
function smoothArr(y, w) { if (!w) return y; return y.map((_, i) => { let s = 0, c = 0; for (let k = Math.max(0, i - w); k <= i; k++) if (y[k] != null) { s += y[k]; c++; } return c ? s / c : null; }); }

/* ================================================================ fetch */
const LOCALHOST = /^(127\.0\.0\.1|localhost|\[::1\])$/.test(location.hostname) || location.protocol === "file:";
const hosted = () => !LOCALHOST && !ST.url;           // a published copy: the sandbox cannot reach the viewer's own machine
function endpoint() {
  if (ST.url) return ST.url.replace(/\/$/, "") + "/api/telemetry";
  if (location.protocol === "http:" || location.protocol === "https:") return "/api/telemetry";
  return "http://127.0.0.1:8765/api/telemetry";
}
async function poll() {
  if (hosted()) { if (window.V9_SNAPSHOT && L.source === "offline") { L.data = window.V9_SNAPSHOT; L.source = "snapshot"; } render(); return; }
  const t0 = performance.now();
  try {
    const ctl = new AbortController(), to = setTimeout(() => ctl.abort(), 6000);
    const r = await fetch(endpoint(), { cache: "no-store", signal: ctl.signal }); clearTimeout(to);
    if (!r.ok) throw new Error(r.status);
    const d = await r.json(); if (d.error) throw new Error(d.error);
    L.data = d; L.source = "live"; L.fetched = Date.now(); L.latency = performance.now() - t0;
  } catch (e) {
    if (window.V9_SNAPSHOT) { if (L.source !== "live" || Date.now() - L.fetched > 20000) { L.data = window.V9_SNAPSHOT; L.source = "snapshot"; } }
    else if (L.source !== "live") L.source = "offline";
    else if (Date.now() - L.fetched > 30000) L.source = "stale";
  }
  render();
}

/* ================================================================ charts */
function niceTicks(lo, hi, n) {
  if (!(hi > lo)) return [lo]; const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p, st = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p, out = [];
  for (let v = Math.ceil(lo / st) * st; v <= hi + st * 1e-6; v += st) out.push(+v.toPrecision(10)); return out;
}
function lineChart(host, series, o) {
  o = o || {}; const W = Math.max(420, Math.round(host.clientWidth || 920)), H = o.h || 300, m = { l: 54, r: 18, t: 14, b: 44 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
  const vis = series.filter(s => !s.hide && s.x.length);
  const tf = v => (o.logy ? Math.log10(v) : v), ok = v => v != null && isFinite(v) && (!o.logy || v > 0);
  let xs = vis.flatMap(s => s.x), inX = x => !o.xdom || (x >= o.xdom[0] && x <= o.xdom[1]), ys = vis.flatMap(s => s.y.filter((v, i) => ok(v) && inX(s.x[i]))).concat(o.refs ? o.refs.map(r => r.y) : []);
  if (!xs.length || !ys.length) { host.innerHTML = `<div class="notice" style="margin:14px">No data yet for this chart.</div>`; return; }
  const x0 = o.xdom ? o.xdom[0] : Math.min(...xs), x1 = o.xdom ? o.xdom[1] : Math.max(...xs);
  let y0 = o.ydom ? o.ydom[0] : Math.min(...ys), y1 = o.ydom ? o.ydom[1] : Math.max(...ys);
  if (!o.ydom) { const pad = (y1 - y0 || Math.abs(y1) || 1) * .06; y0 = o.logy ? y0 * .9 : y0 - pad; y1 = o.logy ? y1 * 1.1 : y1 + pad; if (o.zero) y0 = Math.min(y0, 0); }
  const X = x => m.l + (x - x0) / (x1 - x0 || 1) * iw, Y = y => m.t + ih - (tf(y) - tf(y0)) / (tf(y1) - tf(y0) || 1) * ih;
  let g = "";
  const yt = o.logy ? [1, 2, 3, 5, 7, 10].flatMap(k => [0.01, 0.1, 1, 10].map(p => k * p)).filter(v => v >= y0 && v <= y1) : niceTicks(y0, y1, 5);
  yt.forEach(v => { g += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${o.yfmt ? o.yfmt(v) : +v.toPrecision(3)}</text>`; });
  niceTicks(x0, x1, 8).forEach(v => { g += `<line class="grid" x1="${X(v)}" x2="${X(v)}" y1="${m.t}" y2="${m.t + ih}"/><text x="${X(v)}" y="${H - 22}" text-anchor="middle">${o.xfmt ? o.xfmt(v) : (v >= 1000 ? v / 1000 + "k" : v)}</text>`; });
  g += `<line class="axis" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${m.t + ih}"/><line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${m.t + ih}" y2="${m.t + ih}"/>`;
  if (o.xlabel) g += `<text x="${m.l + iw / 2}" y="${H - 4}" text-anchor="middle">${o.xlabel}</text>`;
  (o.refs || []).forEach(r => { const rc = r.color || "var(--good)"; g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(r.y)}" y2="${Y(r.y)}" style="stroke:${rc};stroke-dasharray:5 4"/><text x="${W - m.r - 4}" y="${Y(r.y) - 5}" text-anchor="end" style="fill:${rc};paint-order:stroke;stroke:var(--panel);stroke-width:4px;stroke-linejoin:round">${r.label}</text>`; });
  (o.vlines || []).forEach(v => { g += `<line x1="${X(v.x)}" x2="${X(v.x)}" y1="${m.t}" y2="${m.t + ih}" style="stroke:var(--line2);stroke-dasharray:2 3"/><text x="${X(v.x) + 4}" y="${m.t + 10}" style="font-size:10px">${v.label}</text>`; });
  const cid = "clip-" + (host.id || "c"); g += `<clipPath id="${cid}"><rect x="${m.l}" y="${m.t - 4}" width="${iw + 6}" height="${ih + 8}"/></clipPath><g clip-path="url(#${cid})">`;
  vis.forEach(s => {
    let d = "", pen = false;
    s.x.forEach((x, i) => { const y = s.y[i]; if (!ok(y)) { pen = false; return; } d += (pen ? "L" : "M") + X(x).toFixed(1) + "," + Y(y).toFixed(1); pen = true; });
    if (s.line !== false) g += `<path d="${d}" fill="none" style="stroke:${s.color};stroke-width:${s.w || 2};opacity:${s.op == null ? 1 : s.op};stroke-dasharray:${s.dash || "none"};stroke-linejoin:round"/>`;
    if (s.pts) s.x.forEach((x, i) => { const y = s.y[i]; if (!ok(y)) return; const iv = s.iv && s.iv[i]; if (iv) g += `<line x1="${X(x)}" x2="${X(x)}" y1="${Y(iv[0])}" y2="${Y(iv[1])}" style="stroke:${s.color};opacity:.35;stroke-width:2"/>`; g += `<circle cx="${X(x)}" cy="${Y(y)}" r="4.2" style="fill:${s.color};stroke:var(--panel);stroke-width:1.5;opacity:${s.op == null ? 1 : s.op}"/>`; });
    if (s.endDot) { const i = s.x.length - 1; if (ok(s.y[i])) g += `<circle cx="${X(s.x[i])}" cy="${Y(s.y[i])}" r="3.4" style="fill:${s.color}"/>`; }
  });
  g += "</g>";
  host.style.position = "relative";
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.title || "chart")}">${g}<line class="hv" id="hv" y1="${m.t}" y2="${m.t + ih}" style="display:none"/><rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent" id="hit"/></svg><div class="chart-tip" style="display:none"></div>`;
  const svg = host.querySelector("svg"), hv = host.querySelector("#hv"), tip = host.querySelector(".chart-tip"), hit = host.querySelector("#hit");
  hit.addEventListener("pointermove", e => {
    const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * W, xv = x0 + (px - m.l) / iw * (x1 - x0); let best = null, rows = [];
    vis.forEach(s => { let lo = 0, hi = s.x.length - 1; while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (s.x[mid] < xv) lo = mid; else hi = mid; } const i = Math.abs(s.x[lo] - xv) <= Math.abs(s.x[hi] - xv) ? lo : hi;
      if (ok(s.y[i]) && Math.abs(s.x[i] - xv) < (x1 - x0) * .06) { rows.push([s, s.x[i], s.y[i]]); if (!best || Math.abs(s.x[i] - xv) < Math.abs(best - xv)) best = s.x[i]; } });
    if (!rows.length) { hv.style.display = "none"; tip.style.display = "none"; return; }
    hv.setAttribute("x1", X(best)); hv.setAttribute("x2", X(best)); hv.style.display = "";
    rows.sort((a, b) => a[2] - b[2]);
    tip.innerHTML = `<b>${o.xfmt ? o.xfmt(best) : best.toLocaleString("en-US")}</b>` + rows.map(r => `<div><i style="background:${r[0].color}"></i>${esc(r[0].label)}<span>${o.vfmt ? o.vfmt(r[2]) : (+r[2].toPrecision(4))}</span></div>`).join("");
    tip.style.display = ""; const left = X(best) / W * r.width + 14; tip.style.left = (left > r.width - 230 ? left - 244 : left) + "px"; tip.style.top = "8px";
  });
  hit.addEventListener("pointerleave", () => { hv.style.display = "none"; tip.style.display = "none"; });
}

/* ================================================================ sections */
function kpi(k, v, s, bar) { return `<div class="kpi"><div class="k">${k}</div><b>${v}</b><div class="s">${s || ""}</div>${bar != null ? `<div class="bar"><i style="width:${Math.min(100, bar * 100).toFixed(1)}%"></i></div>` : ""}</div>`; }
function current() {
  const d = L.data, live = Object.entries(d.runs).filter(([, r]) => isRunning(r));
  live.sort((a, b) => (utcMs(b[1].last_update_utc) || 0) - (utcMs(a[1].last_update_utc) || 0));
  return live[0] ? { name: live[0][0], r: live[0][1] } : null;
}
const PLAN = [   // execution order from register I13, with the assistant's duration estimates (hours); not measured values
  { id: "Q1", label: "Q1 · S1 seeds", block: "Q1", match: /^S1_/, h: 17.5 }, { id: "Q3a", label: "Q3a · S2 dense", block: "Q3", match: /^S2_.*dense|^S2_replay_dense/, h: 11 },
  { id: "Q2", label: "Q2 · S6-base", block: "Q2", match: /^S6_/, h: 10 }, { id: "Q3b", label: "Q3b · S2 recurrent", block: "Q3", match: /^S2_.*rec/, h: 9 },
  { id: "Q4", label: "Q4 · S3", block: "Q4", match: /^S3_/, h: 5 }, { id: "Q5", label: "Q5 · S4", block: "Q5", match: /^S4_/, h: 6 }, { id: "Q6", label: "Q6 · S5", block: "Q6", match: /^S5_|^s\d+_/, h: 16 }, { id: "QF", label: "QF · final analysis", block: "QF", match: /^QF_|^FINAL/, h: 1 }];
const EXPECT_S1 = 6;
function q1Eta() {
  const d = L.data, cur = current(), runs = Object.entries(d.runs).filter(([n]) => /^S1_/.test(n));
  const done = runs.filter(([, r]) => r.completed).length, sp = cur ? medianSpeed(cur.r) : null;
  if (!cur || !sp || !/^S1_/.test(cur.name)) return null;       // only while an S1 run is the one training
  const left = (cur.r.config.steps || 24000) - lastOf(col(cur.r, "step")), pending = Math.max(0, EXPECT_S1 - runs.length), alone = Math.min(sp, 0.31);
  const secs = left * sp + pending * 24000 * alone + 3000;      // + about 50 min of dev eval, G1, teachers and sealed eval (assumption)
  return { secs, end: Date.now() + secs * 1000, done, pending, sp };
}
function refPace(arm, core) { const t = armTimes(arm, core); return t.train ? t.train / 12000 : null; }     // seconds per update of an already finished run of the same arm
function renderKpis() {
  const d = L.data, cur = current(), eta = q1Eta(), sw = d.system && d.system.swap, h = [], q3 = d.blocks.Q3, onS2 = cur ? /^S2_/.test(cur.name) : !!(q3 && q3.status === "running" && /^S2_/.test(q3.stage || ""));
  if (cur) {
    const p = parseRun(cur.name), step = lastOf(col(cur.r, "step")), plan = cur.r.config.steps || 24000, sp = medianSpeed(cur.r), left = sp ? (plan - step) * sp : null, sl = lastOf(col(cur.r, "slots_per_s"));
    const ref = onS2 ? refPace(p.arm.replace(/_.*$/, ""), /_rec$/.test(cur.name.replace(/_s\d+$/, "")) ? "rec" : "dense") : null;
    h.push(kpi("Running now", esc(p.arm) + (p.seed ? " · s" + p.seed : ""), `${step.toLocaleString("en-US")} / ${plan.toLocaleString("en-US")} updates · ${(100 * step / plan).toFixed(1)}%`, step / plan));
    h.push(kpi("This run ends", left ? fmtT(Date.now() + left * 1000) + " IST" : "–", left ? `in ${dur(left)} at ${sp.toFixed(2)} s/update` : ""));
    h.push(kpi("Throughput", sp ? (3600 / sp).toFixed(0) + " upd/h" : "–", sp ? `${sp.toFixed(3)} s/update` + (ref && sp > ref * 1.15 ? ` · ${(sp / ref).toFixed(1)}× slower than seed 911 (${ref.toFixed(3)})` : ref ? " · on seed-911 pace" : "") + (sl ? ` · ${Math.round(sl).toLocaleString("en-US")} slots/s` : "") : ""));
    h.push(kpi("Loss now", f3(lastOf(col(cur.r, "loss"))), `text ${f3(lastOf(col(cur.r, "text")))} · symbolic ${f3(lastOf(col(cur.r, "symbolic")))}` + (lastOf(col(cur.r, "ncp")) != null ? ` · NCP ${f3(lastOf(col(cur.r, "ncp")))}` : "")));
  } else if (onS2) {
    const m = /^(S2_.+?_s\d+)_measure/.exec(q3.stage || ""), nm = m ? parseRun(m[1]) : null;
    h.push(kpi("Running now", nm ? esc(nm.arm) + " · s" + nm.seed : "Q3", m ? "measurement: greedy suites, drift, contracts" : esc(q3.stage || "")));
  } else h.push(kpi("Running now", "idle", d.blocks.Q1 ? "Q1 is " + d.blocks.Q1.status : "no run is training"));
  if (eta) h.push(kpi("Q1 finishes", fmtDT(eta.end) + " IST", `${eta.done} of ${EXPECT_S1} runs done · ${eta.pending} not started · estimate`));
  if (onS2) {
    const core = (cur ? cur.name : (q3.stage || "")).includes("_rec_") ? "rec" : "dense", e = q3Eta(core), tier = core === "dense" ? "Q3a" : "Q3b";
    if (e) {
      h.push(kpi(tier + " finishes", e.unknown ? "–" : fmtDT(e.slow) + " IST", e.unknown ? "no timing for this tier yet" : e.slow - e.fast > 600e3 ? `if today's pace holds · ${fmtDT(e.fast)} at seed-911 pace · estimate` : "at seed-911 pace · estimate"));
      h.push(kpi(tier + " cells", `${e.done} of ${e.cells} measured`, `${e.training} training · ${e.measuring} measuring · ${e.pending} not started`, e.done / e.cells));
    }
  }
  const q = d.blocks, run = Object.values(q).filter(b => b.status === "running").length, wait = Object.values(q).filter(b => b.status === "waiting").length;
  h.push(kpi("Queue", `${run} running · ${wait} waiting`, `${Object.keys(q).length} blocks launched`));
  const ts = Object.values(d.tests), okc = ts.filter(t => t.status === "passed").length;
  h.push(kpi("Tests", `${okc} / ${ts.length} passed`, "contracts and smokes under the locks"));
  if (sw) { const f = sw.used_mb / sw.total_mb; h.push(kpi("Swap", `${(sw.used_mb / 1024).toFixed(1)} / ${(sw.total_mb / 1024).toFixed(1)} GB`, f > .8 ? "high: a K8 measurement can hit the MPS memory cap" : "memory pressure is low", f)); }
  $("#lv-kpis").innerHTML = h.join("");
}
const BLOCK_NAME = { Q1: "S1 foundation seeds", Q2: "S6-base continual learning", Q3: "S2 H-NCP (dense, then recurrent)", Q4: "S3 refit (S3a-dec, S3b)", Q5: "S4 expert iteration + baselines", Q6: "S5 search evaluation", QF: "Final analysis" };
function renderQueue() {
  const d = L.data, order = ["Q1", "Q3", "Q2", "Q4", "Q5", "Q6", "QF"];
  $("#lv-q-sub").textContent = "execution order (register I13): Q1 → Q3a → Q2 → Q3b → Q4 → Q5 → Q6 → QF";
  $("#lv-queue").innerHTML = order.map(id => {
    const b = d.blocks[id], tc = d.tests["q" + id.slice(1).toLowerCase() + "_contracts"], ts = d.tests["q" + id.slice(1).toLowerCase() + "_smoke"];
    const stc = !b ? "waiting" : b.status === "running" ? "running" : /complete|done|finished/.test(b.status) ? "done" : /stop|fail/.test(b.status) ? "stopped" : "waiting";
    const lock = d.locks[id] ? `lock ${fmtDT(utcMs(d.locks[id].created_utc))}` : "no lock";
    return `<div class="qrow"><span class="id">${id}</span><span><b>${BLOCK_NAME[id]}</b></span><span class="st ${stc}">${b ? esc(b.status) : "not launched"}</span>
      <span class="sub">${b ? esc(b.stage) + (b.driver_alive ? " · driver alive" : " · driver not running") + ` · up ${dur(b.elapsed_s)}` : ""}${tc ? ` · contracts ${tc.status}${tc.n ? " (" + tc.n + ")" : ""}` : ""}${ts ? ` · smoke ${ts.status}${ts.seconds ? " " + dur(ts.seconds) : ""}` : ""} · ${lock}</span></div>`;
  }).join("");
}
function runningCore() {       // "dense" | "rec" | null: which Q3 tier is training or being measured
  const d = L.data, cur = current(), name = cur ? cur.name : (d.blocks.Q3 && d.blocks.Q3.status === "running" ? String(d.blocks.Q3.stage || "") : "");
  return /^S2_/.test(name) ? (name.includes("_rec_") ? "rec" : "dense") : null;
}
function planRows() {
  const d = L.data, now = Date.now(), cur = current(), proj = {};
  if (cur && /^S1_/.test(cur.name)) { const e = q1Eta(); if (e) proj.Q1 = e.end; }
  const core = runningCore(); if (core) { const e = q3Eta(core); if (e && !e.unknown) proj[core === "dense" ? "Q3a" : "Q3b"] = e.slow; }
  const meas = PLAN.map(p => {
    const rs = Object.entries(d.runs).filter(([n]) => p.match.test(n)); if (!rs.length) return null;
    const ts = rs.flatMap(([, r]) => col(r, "t").filter(v => v)).map(v => v * 1000); if (!ts.length) return null;
    const bk = d.blocks[p.block], run = rs.some(([, r]) => isRunning(r)) || !!(bk && bk.status === "running" && p.match.test(String(bk.stage || "")));
    return { s: Math.min(...ts), e: run ? now : Math.max(...ts), run, runs: rs };
  });
  let cursor = now; const rows = [];
  PLAN.forEach((p, i) => {
    const m = meas[i];
    if (m) { const pj = m.run ? proj[p.id] : undefined; rows.push({ p, m, s: m.s, e: pj ? Math.max(m.e, pj) : m.e, plan: false, proj: pj }); cursor = Math.max(cursor, pj || m.e); }
    else { rows.push({ p, s: cursor, e: cursor + p.h * 3600e3, plan: true }); cursor += p.h * 3600e3; }
  });
  return { rows, now, fin: rows[rows.length - 1].e };
}
function renderGantt() {
  const host = $("#lv-gantt"), W = 640, rowH = 30, H = 28 + PLAN.length * rowH + 28, x0 = 150, padR = 46, P = planRows(), rows = P.rows, now = P.now;
  const t0 = Math.min(...rows.map(r => r.s)), t1 = Math.max(...rows.map(r => r.e)), X = t => x0 + (t - t0) / (t1 - t0 || 1) * (W - x0 - padR);
  let g = `<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" style="fill:var(--panel2)"/><line x1="0" y1="0" x2="0" y2="6" style="stroke:var(--ink3);stroke-width:2"/></pattern></defs>`;
  const step = Math.max(12, Math.ceil((t1 - t0) / 3600e3 / 5 / 12) * 12) * 3600e3;
  for (let t = Math.ceil(t0 / step) * step; t < t1; t += step) g += `<line class="grid" x1="${X(t)}" x2="${X(t)}" y1="14" y2="${H - 22}"/><text x="${X(t)}" y="${H - 8}" text-anchor="middle" style="font-size:9.5px">${fmtT(t, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}</text>`;
  rows.forEach((r, i) => {
    const y = 20 + i * rowH, col_ = `var(${({ Q1: "--enc", Q3a: "--con", Q3b: "--con", Q2: "--loss", Q4: "--dec", Q5: "--sel", Q6: "--code", QF: "--ink3" })[r.p.id]})`;
    g += `<text x="0" y="${y + 16}" style="fill:var(--ink);font-size:11.5px;font-family:var(--sans)">${r.p.label}</text>`;
    if (r.plan) g += `<rect x="${X(r.s)}" y="${y + 3}" width="${Math.max(3, X(r.e) - X(r.s))}" height="20" rx="5" class="hatch" style="stroke:${col_};stroke-width:1.5"><title>planned estimate ${r.p.h} h</title></rect><text x="${X(r.e) + 4}" y="${y + 17}" style="font-size:10px">~${r.p.h} h</text>`;
    else {
      // measured: one segment per run
      const segs = r.m.runs.map(([n, rr]) => { const t = col(rr, "t").filter(v => v).map(v => v * 1000); return { n, s: Math.min(...t), e: isRunning(rr) ? now : Math.max(...t), run: isRunning(rr) }; });
      segs.forEach((sg, k) => { g += `<rect x="${X(sg.s)}" y="${y + 3}" width="${Math.max(3, X(sg.e) - X(sg.s))}" height="20" rx="4" style="fill:${col_};opacity:${k % 2 ? .62 : .95};stroke:var(--panel);stroke-width:1"><title>${esc(sg.n)} · ${dur((sg.e - sg.s) / 1000)}</title></rect>`; });
      if (r.proj) g += `<rect x="${X(now)}" y="${y + 3}" width="${Math.max(3, X(r.proj) - X(now))}" height="20" rx="4" class="hatch" style="stroke:${col_};stroke-width:1.5"><title>projected</title></rect>`;
      g += `<text x="${X(r.proj || r.e) + 4}" y="${y + 17}" style="font-size:10px">${r.proj ? "≈ " + fmtT(r.proj) : dur((r.e - r.s) / 1000)}</text>`;
    }
  });
  g += `<line x1="${X(now)}" x2="${X(now)}" y1="10" y2="${H - 22}" style="stroke:var(--accent);stroke-width:1.5"/><text x="${X(now)}" y="9" text-anchor="middle" style="fill:var(--accent);font-size:10px">now ${fmtT(now)}</text>`;
  const fin = rows[rows.length - 1].e;
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="chart-svg" role="img" aria-label="queue timeline">${g}</svg><div class="figcap">Times are IST. Solid bars are measured from each run's own log. Hatched bars are planned estimates (assistant's, from the pre-registered I13 sizing and smoke timings), not measurements. Projected end of the whole queue: <b>${fmtDT(fin)} IST</b>.</div>`;
}
function renderCurves() {
  const d = L.data, runs = S1runs(), M = METRICS.find(m => m[0] === ST.metric) || METRICS[0];
  $("#lv-legend").innerHTML = runs.map(n => { const s = runStyle(n); return `<button data-run="${n}" class="${ST.hidden.has(n) ? "off" : ""}"><i class="${s.dash ? "dash" : ""}" style="border-top-color:${s.color};opacity:${s.op}"></i>${esc(s.label)}${isRunning(d.runs[n]) ? " ●" : ""}</button>`; }).join("");
  $$("#lv-legend button").forEach(b => b.addEventListener("click", () => { ST.hidden.has(b.dataset.run) ? ST.hidden.delete(b.dataset.run) : ST.hidden.add(b.dataset.run); renderCurves(); renderDev(); }));
  const series = runs.map(n => { const r = d.runs[n], s = runStyle(n), x = col(r, "step"), y = smoothArr(col(r, M[2]), ST.smooth); return { x, y, color: s.color, dash: s.dash, op: s.op, w: s.w, label: s.label, hide: ST.hidden.has(n), endDot: isRunning(r) }; });
  const acc = /accuracy/.test(ST.metric);
  lineChart($("#lv-curves"), series, { logy: ST.logy && !acc, xlabel: "update", title: M[1], zero: false, xdom: [ST.from, Math.max(...series.flatMap(s => s.x), ST.from + 1)], ydom: acc ? [0, 1.02] : null, yfmt: acc ? v => Math.round(v * 100) + "%" : null, vfmt: acc ? v => (100 * v).toFixed(1) + "%" : v => (+v.toPrecision(4)).toString(),
    vlines: [{ x: 6000, label: "α starts" }, { x: 8000, label: "α = 0.1" }] });
}
function devSeries(n, suite) { const r = L.data.runs[n], pts = r.dev_checks.filter(c => c[suite]); return { x: pts.map(c => c.step), y: pts.map(c => c[suite].rate), iv: pts.map(c => c[suite].interval) }; }
function renderDev() {
  const runs = S1runs(), suites = ["standard", "lookup", "compose", "hard10"];
  $("#lv-suite").innerHTML = suites.map(s => `<button data-s="${s}" class="${ST.suite === s ? "on" : ""}">${s}</button>`).join("");
  $$("#lv-suite button").forEach(b => b.addEventListener("click", () => { ST.suite = b.dataset.s; renderDev(); }));
  const series = runs.map(n => { const s = runStyle(n), v = devSeries(n, ST.suite); return Object.assign(v, { color: s.color, dash: s.dash, op: s.op, w: s.w, label: s.label, hide: ST.hidden.has(n), pts: true }); });
  const refs = ST.suite === "standard" ? [{ y: .8, label: "G1 ≥ 80%" }] : ST.suite === "lookup" ? [{ y: .95, label: "G1 ≥ 95%" }] : [];
  lineChart($("#lv-dev"), series, { h: 280, xlabel: "update", ydom: [0, 1.02], yfmt: v => Math.round(v * 100) + "%", vfmt: v => (100 * v).toFixed(1) + "%", refs, title: "dev success" });
}
function renderMatched() {
  const d = L.data, runs = S1runs(), cur = current();
  const latestFlat = Math.max(...runs.map(n => lastOf(col(d.runs[n], "step")) || 0).filter(v => v), 0);
  const step = ST.stepMode === "latest" ? (cur ? lastOf(col(cur.r, "step")) : latestFlat) : ST.step;
  const sl = $("#lv-step"); sl.max = Math.max(24000, latestFlat); $("#lv-step-v").textContent = step.toLocaleString("en-US");
  const rows = runs.map(n => {
    const r = d.runs[n], st = col(r, "step"), has = st.length && st[st.length - 1] >= step - 50, w = [], idx = [];
    st.forEach((s, i) => { if (s > step - 1000 && s <= step) idx.push(i); });
    const av = k => { const c = col(r, k); const v = idx.map(i => c[i]).filter(x => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
    const dc = r.dev_checks.filter(c => c.step <= step).slice(-1)[0];
    return { n, has, text: av("text"), symbolic: av("symbolic"), gacc: av("graph_node_accuracy"), dstep: dc && dc.step, std: dc && dc.standard && dc.standard.rate, h10: dc && dc.hard10 && dc.hard10.rate, look: dc && dc.lookup && dc.lookup.rate };
  });
  const best = (k, lo) => { const v = rows.filter(r => r.has && r[k] != null).map(r => r[k]); return v.length ? (lo ? Math.min(...v) : Math.max(...v)) : null; };
  const B = { text: best("text", 1), symbolic: best("symbolic", 1), gacc: best("gacc"), std: best("std"), h10: best("h10"), look: best("look") };
  const c = (r, k, f, lo) => `<td class="${r.has && r[k] != null && r[k] === B[k] ? "best" : ""}">${r.has ? f(r[k]) : "–"}</td>`;
  $("#lv-matched").innerHTML = `<div class="muted" style="font-size:12.5px;margin-bottom:6px">Average of the 1,000 updates up to <b>${step.toLocaleString("en-US")}</b>; dev success is the latest check at or before that update. Best value per column is shaded.</div>
    <table class="m"><thead><tr><th>run</th><th>text loss ↓</th><th>symbolic ↓</th><th>graph acc.</th><th>dev std</th><th>dev hard10</th><th>dev lookup</th><th>check at</th></tr></thead><tbody>${rows.map(r => { const s = runStyle(r.n); return `<tr><td><i style="display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;background:${s.color};opacity:${s.op}"></i>${esc(s.label)}${r.has ? "" : ` <span class="muted">(not at ${step.toLocaleString("en-US")} yet)</span>`}</td>${c(r, "text", f3)}${c(r, "symbolic", f3)}${c(r, "gacc", v => pct(v, 0))}${c(r, "std", pct)}${c(r, "h10", pct)}${c(r, "look", pct)}<td>${r.has && r.dstep ? r.dstep.toLocaleString("en-US") : "–"}</td></tr>`; }).join("")}</tbody></table>
    <div class="muted" style="font-size:12px;margin-top:8px">Text loss is the H4 axis (lower is better, flat is expected to win). Symbolic loss, graph accuracy and dev success follow the reasoning task (H3). One flat seed so far means no claim about either hypothesis.</div>`;
}
function renderSpeed() {
  const runs = S1runs(), series = runs.map(n => { const r = L.data.runs[n], st = col(r, "step"), el = col(r, "elapsed_s"), s = runStyle(n), x = [], y = [];
    for (let i = 1; i < st.length; i++) if (st[i] > st[i - 1] && el[i] >= el[i - 1]) { x.push(st[i]); y.push((el[i] - el[i - 1]) / (st[i] - st[i - 1])); }
    return { x, y: smoothArr(y, 2), color: s.color, dash: s.dash, op: s.op, w: s.w, label: s.label, hide: ST.hidden.has(n) }; });
  const cur = current(); $("#lv-speed-sub").textContent = cur ? `now ${(medianSpeed(cur.r) || 0).toFixed(3)} s/update` : "";
  lineChart($("#lv-speed"), series, { h: 240, xlabel: "update", zero: true, yfmt: v => v.toFixed(2), vfmt: v => v.toFixed(3) + " s/upd", title: "seconds per update" });
}
function renderG1() {
  const d = L.data, g = d.g1, devs = Object.entries(d.runs).filter(([n, r]) => /^S1_/.test(n) && r.dev_eval);
  let h = "";
  if (g && g.seeds) h += `<div class="muted" style="font-size:12.5px;margin-bottom:8px">${esc(g.rule || "")}${g.final === false ? " · provisional until both seeds are evaluated" : ""}</div>` + Object.entries(g.seeds).map(([sd, v]) => `<div style="margin-bottom:12px"><div style="display:flex;justify-content:space-between;font-size:13px"><b>A_delayed · seed ${sd}</b><span class="st ${v.passed ? "done" : "stopped"}">${v.passed ? "G1 passed" : "G1 failed"}</span></div>
      <div style="display:grid;grid-template-columns:70px 1fr 56px;gap:8px;align-items:center;font-size:12px;margin-top:6px"><span>standard</span><div class="gbar"><i style="width:${v.standard * 100}%;background:var(${v.standard >= .8 ? "--good" : "--bad"})"></i><u style="left:80%"></u></div><span class="mono">${pct(v.standard)}</span>
      <span>lookup</span><div class="gbar"><i style="width:${v.lookup * 100}%;background:var(${v.lookup >= .95 ? "--good" : "--bad"})"></i><u style="left:95%"></u></div><span class="mono">${pct(v.lookup)}</span></div></div>`).join("");
  else h += `<div class="notice">G1 has not been evaluated yet.</div>`;
  if (devs.length) h += `<table class="m" style="margin-top:6px"><thead><tr><th>dev eval, 2,048 items</th><th>standard</th><th>lookup</th><th>compose</th><th>hard10</th></tr></thead><tbody>${devs.map(([n, r]) => { const s = r.dev_eval.symbolic || {}, st = runStyle(n); const cl = k => s[k] ? pct(s[k].rate) : "–"; return `<tr><td>${esc(st.label)}</td><td>${cl("standard")}</td><td>${cl("lookup")}</td><td>${cl("compose")}</td><td>${cl("hard10")}</td></tr>`; }).join("")}</tbody></table>`;
  $("#lv-g1").innerHTML = h;
}
function renderSys() {
  const d = L.data, sw = d.system && d.system.swap, f = sw ? sw.used_mb / sw.total_mb : 0;
  const age = L.source === "live" ? (Date.now() - L.fetched) / 1000 : null;
  $("#lv-sys").innerHTML = `${sw ? `<div style="font-size:12.5px;display:flex;justify-content:space-between"><span>swap</span><span class="mono">${sw.used_mb.toFixed(0)} / ${sw.total_mb.toFixed(0)} MB</span></div><div class="gbar"><i style="width:${f * 100}%;background:var(${f > .8 ? "--bad" : f > .6 ? "--warn" : "--good"})"></i></div>` : ""}
    <dl class="kv"><dt>load (1 / 5 / 15 min)</dt><dd>${d.system && d.system.load ? d.system.load.join(" · ") : "–"} of ${d.system ? d.system.cpus : "–"} cores</dd><dt>data source</dt><dd>${L.source === "live" ? "live server" + (L.latency ? ` · ${Math.round(L.latency)} ms` : "") : L.source === "snapshot" ? "snapshot, taken " + fmtDT(utcMs(d.server_utc)) + " IST" : L.source}</dd><dt>server time</dt><dd>${fmtDT(utcMs(d.server_utc))} IST</dd></dl>`;
  $("#lv-events").innerHTML = (d.events || []).slice(-30).reverse().map(e => `<div><span class="t">${e.utc ? fmtT(utcMs(e.utc)) : ""}</span><span>${esc(e.stage || "")} ${esc(e.event || "")}${e.block ? " · " + esc(e.block) : ""}${e.returncode != null ? " rc=" + e.returncode : ""}</span></div>`).join("");
}
function renderTests() {
  const d = L.data, ids = ["q1", "q2", "q3", "q4", "q5", "q6", "qf"], t = d.tests; let ok = 0;
  const cell = x => x ? `<span class="st ${x.status === "passed" ? "done" : "stopped"}">${x.status}</span>${x.n ? " " + x.n : ""}` : "–";
  $("#lv-tests").innerHTML = `<table class="m"><thead><tr><th>block</th><th>contracts</th><th>smoke</th><th>lock</th></tr></thead><tbody>${ids.map(i => { const c = t[i + "_contracts"], s = t[i + "_smoke"], lk = d.locks[i.toUpperCase()]; if (c && s && c.status === "passed" && s.status === "passed") ok++; return `<tr><td>${i.toUpperCase()}</td><td>${cell(c)}</td><td>${cell(s)}${s && s.seconds ? " · " + dur(s.seconds) : ""}</td><td>${lk ? fmtDT(utcMs(lk.created_utc)) : "–"}</td></tr>`; }).join("")}</tbody></table>`;
  $("#lv-tests-sub").textContent = `${ok} of ${ids.length} blocks fully verified`;
}


/* ================================================================ Q3 board and S2 curves */
const Q3_ARMS = [["V8repro", "dense", "V8-repro", "K = 1 · closed loop off"], ["K1", "dense", "K1 dense", "K = 1 · control"], ["K8", "dense", "K8 dense", "K = 8 · main claim"]];
const Q3B_ARMS = [["K1", "rec", "K1 recurrent", "K = 1 · recurrent core"], ["K8", "rec", "K8 recurrent", "K = 8 · recurrent core"]];
const Q3_COL = { V8repro: "--enc", K1: "--con", K8: "--code" };
const Q3_SEEDS = [911, 912];
const Q3_DEFAULT = { V8repro: [2700, 1116], K1: [6272, 1362], K8: [8955, 3842] };      // seed-911 train wall and measure seconds; used only until the same arm has a finished run in the payload
const Q3_M = 2, Q3_REPLAY_S = 1800;                                                       // best-of-m draws (drive_q3.ARMS); replay + register allowance in seconds (assumption)
const s2name = (arm, core, seed) => `S2_${arm}_${core}_s${seed}`;
function armTimes(arm, core) {
  let tr = null, me = null;
  for (const s of Q3_SEEDS) { const r = L.data.runs[s2name(arm, core, s)]; if (!r) continue; if (tr == null && r.completed && r.wall_s) tr = r.wall_s; if (me == null && r.measure && r.measure.seconds) me = r.measure.seconds; }
  const def = core === "dense" ? Q3_DEFAULT[arm] : null;
  return { train: tr != null ? tr : def && def[0], meas: me != null ? me : def && def[1], measured: tr != null };
}
function q3State(name) {
  const d = L.data, r = d.runs[name], blk = d.blocks.Q3 || {};
  if (!r) return { st: "pending", pct: 0 };
  const steps = (r.config && r.config.steps) || 12000, last = lastOf(col(r, "step")) || 0;
  if (!r.completed) return { st: "training", pct: last / steps, last, steps, r };
  if (!r.measure) return { st: String(blk.stage || "").startsWith(name + "_measure") ? "measuring" : "queued for measurement", pct: 1, last, steps, r };
  return { st: "done", pct: 1, last, steps, r };
}
/* Remaining time of one tier from the runs' own measured timings (not the PLAN hours): training left x pace + measurement, per cell, plus the replay allowance.
   fast = every later cell at the seed-911 pace; slow = later training scaled by how much slower the running cell is than its seed-911 twin (contention from smoke tests). */
function q3Eta(core) {
  const d = L.data, arms = core === "dense" ? Q3_ARMS : Q3B_ARMS, now = Date.now();
  let fast = 0, slow = 0, cells = 0, done = 0, training = 0, measuring = 0, pending = 0, unknown = false, ratio = 1;
  for (const seed of Q3_SEEDS) for (const [arm, c] of arms) { const x = q3State(s2name(arm, c, seed)); if (x.st === "training") { const t = armTimes(arm, c), sp = medianSpeed(x.r); if (t.train && sp) ratio = Math.min(3, Math.max(1, sp / (t.train / 12000))); } }
  for (const seed of Q3_SEEDS) for (const [arm, c] of arms) {
    cells++; const x = q3State(s2name(arm, c, seed)), t = armTimes(arm, c);
    if (x.st === "done") { done++; continue; }
    if (t.train == null || t.meas == null) { unknown = true; continue; }
    const refP = t.train / 12000;
    if (x.st === "training") { training++; const sp = medianSpeed(x.r) || refP, left = x.steps - x.last; fast += left * refP + t.meas; slow += left * sp + t.meas; }
    else if (x.st === "pending") { pending++; fast += t.train + t.meas; slow += t.train * ratio + t.meas; }
    else { measuring++; fast += t.meas; slow += t.meas; }
  }
  const rep = d.runs["S2_replay_" + core];
  if (!(rep && rep.completed)) { const sp = rep && medianSpeed(rep), left = rep ? (((rep.config && rep.config.steps) || 6000) - (lastOf(col(rep, "step")) || 0)) : 0, add = rep && sp ? left * sp + 600 : Q3_REPLAY_S; fast += add; slow += add; }
  return { fast: now + fast * 1000, slow: now + Math.max(slow, fast) * 1000, cells, done, training, measuring, pending, unknown, ratio };
}
function q3Card(arm, core, label, sub, seed) {
  const name = s2name(arm, core, seed), x = q3State(name), r = x.r, g = r && r.measure && r.measure.greedy;
  const cls = x.st === "done" ? "done" : x.st === "training" || x.st === "measuring" ? "running" : "waiting";
  const sp = r && x.st === "training" ? medianSpeed(r) : null, ref = refPace(arm, core), last = k => r && x.last ? lastOf(col(r, k)) : null;
  const slow = sp && ref && sp > ref * 1.15 ? ` <span class="warn">${(sp / ref).toFixed(1)}× slower</span>` : "";
  return `<div class="q3card ${x.st === "training" || x.st === "measuring" ? "now" : ""}"><div class="q3t"><b>${label}</b><span class="st ${cls}">${x.st}</span></div><div class="muted" style="font-size:11.5px">${sub} · seed ${seed}</div>
    <div class="kpi" style="padding:0;border:0;box-shadow:none;background:none"><div class="bar"><i style="width:${(x.pct * 100).toFixed(1)}%;background:var(${Q3_COL[arm]})"></i></div></div>
    <div class="q3m">${r ? `${(x.last || 0).toLocaleString("en-US")} / ${x.steps.toLocaleString("en-US")}` : "not started"}${sp ? ` · ${sp.toFixed(2)} s/upd` : ""}${slow}</div>
    ${r && x.last ? `<div class="q3m">text ${f3(last("text"))} · symbolic ${f3(last("symbolic"))}${last("ncp") != null ? " · NCP " + f3(last("ncp")) : ""}</div>` : ""}
    ${arm === "K8" && r && x.last ? `<div class="q3m">selector entropy ${f3(last("selector_entropy"))} <span class="muted">of ${Math.log(8).toFixed(3)}</span> · pair dist. ${f3(last("hypothesis_pair_distance"))}</div>` : ""}
    ${g && g.standard ? `<div class="q3m"><b>dev standard ${pct(g.standard.rate)}</b> · hard10 ${pct(g.hard10 && g.hard10.rate)} · hard14 ${pct(g.hard14 && g.hard14.rate)}</div>` : ""}
    ${r && r.completed && r.wall_s ? `<div class="q3m muted">trained in ${dur(r.wall_s)}${r.measure && r.measure.seconds ? " · measured in " + dur(r.measure.seconds) : ""}</div>` : ""}</div>`;
}
function q3Table() {
  const d = L.data, rows = [];
  for (const seed of Q3_SEEDS) for (const [arm, core, label] of Q3_ARMS) { const r = d.runs[s2name(arm, core, seed)], g = r && r.measure && r.measure.greedy; if (g && g.standard) rows.push({ arm, seed, label, g, r }); }
  if (!rows.length) return `<div class="notice">No Q3 measurement has finished yet.</div>`;
  const rate = (g, k) => g && g[k] ? g[k].rate : null, n = (g, k) => g && g[k] ? g[k].n : null;
  const best = {}; Q3_SEEDS.forEach(sd => ["standard", "hard10", "hard14"].forEach(k => { const v = rows.filter(r => r.seed === sd && rate(r.g, k) != null).map(r => rate(r.g, k)); if (v.length > 1) best[sd + k] = Math.max(...v); }));
  const c = (r, k) => `<td class="${best[r.seed + k] != null && rate(r.g, k) === best[r.seed + k] ? "best" : ""}">${rate(r.g, k) == null ? "–" : pct(rate(r.g, k))}</td>`;
  let body = rows.map(r => `<tr><td><i style="display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;background:var(${Q3_COL[r.arm]})"></i>${esc(r.label)} · s${r.seed}</td>${c(r, "standard")}${c(r, "hard10")}${c(r, "hard14")}<td>${r.r.wall_s ? dur(r.r.wall_s) : "–"}</td><td>${r.r.measure.seconds ? dur(r.r.measure.seconds) : "–"}</td></tr>`).join("");
  const delta = Q3_SEEDS.map(sd => { const k8 = rows.find(r => r.seed === sd && r.arm === "K8"), k1 = rows.find(r => r.seed === sd && r.arm === "K1"); if (!k8 || !k1) return ""; const f = k => rate(k8.g, k) != null && rate(k1.g, k) != null ? ((rate(k8.g, k) - rate(k1.g, k)) * 100).toFixed(1) : "–"; return `<tr class="dl"><td>K8 − K1 · s${sd} <span class="muted">(points)</span></td><td>${f("standard")}</td><td>${f("hard10")}</td><td>${f("hard14")}</td><td></td><td></td></tr>`; }).join("");
  const ns = rows[0].g; return `<table class="m"><thead><tr><th>measured cell</th><th>standard${n(ns, "standard") ? " (n " + n(ns, "standard") + ")" : ""}</th><th>hard10${n(ns, "hard10") ? " (n " + n(ns, "hard10") + ")" : ""}</th><th>hard14${n(ns, "hard14") ? " (n " + n(ns, "hard14") + ")" : ""}</th><th>train</th><th>measure</th></tr></thead><tbody>${body}${delta}</tbody></table>`;
}
const Q3_METRICS = { text: "text loss", symbolic: "symbolic loss", ncp: "NCP loss", loss: "total" };
const K8_METRICS = [["selector_entropy", "selector entropy", "entropy of the prior pi over the 8 hypotheses"], ["kdec_is_kstar", "decoder picks best", "share of positions where the decoder's code (best of m = 2 prior draws) is the oracle-best code k*"], ["hypothesis_pair_distance", "pair distance", "mean squared distance between the 8 predicted concepts"], ["d_min", "d min", "squared error of the best hypothesis"], ["d_mean", "d mean", "mean squared error over the hypotheses"], ["selector", "selector loss", "KL of the oracle responsibilities to the prior"]];
function renderQ3() {
  const d = L.data, any = Object.keys(d.runs).some(n => /^S2_/.test(n)), head = $("#lv-q3-head"), box = $("#lv-q3");
  if (!head || !box) return;
  head.hidden = box.hidden = !any; if (!any) return;
  const blk = d.blocks.Q3 || {}, e = q3Eta("dense");
  $("#lv-q3-sub").textContent = `block ${blk.status || "?"}${blk.stage ? " · " + blk.stage : ""} · ${e.done} of ${e.cells} dense cells measured`;
  const rowOf = (seed, arms) => `<div class="q3row">${arms.map(a => q3Card(a[0], a[1], a[2], a[3], seed)).join("")}</div>`;
  const rep = d.runs.S2_replay_dense, repReg = rep ? (rep.completed ? "done" : "training") : "pending", anyRec = Object.keys(d.runs).some(n => /^S2_.*_rec_/.test(n));
  const seedNote = { 911: "teacher: S1 A_delayed s902", 912: "teacher: V8 seed 811 (A_delayed_ncp), because S1 s901 failed G1" };
  $("#lv-q3-board").innerHTML = `<div class="q3tier"><b>Q3a · dense tier</b> <span class="muted">order: seed 911 (V8-repro, K1, K8), seed 912 (same), then replay and register</span></div><div class="q3seeds">${Q3_SEEDS.map(sd => `<div><div class="q3note">seed ${sd} · ${seedNote[sd]}</div>${rowOf(sd, Q3_ARMS)}</div>`).join("")}</div>
    <div class="q3note" style="margin-top:12px">then</div><div class="q3row"><div class="q3card"><div class="q3t"><b>V8 replay</b><span class="st ${repReg === "done" ? "done" : repReg === "training" ? "running" : "waiting"}">${repReg}</span></div><div class="muted" style="font-size:11.5px">K = 1 · seed 811 · 6,000 updates</div></div>
    <div class="q3card"><div class="q3t"><b>Register + promotion</b><span class="st waiting">${d.runs.S2_K8_dense_s912 && d.runs.S2_K8_dense_s912.measure ? "next" : "pending"}</span></div><div class="muted" style="font-size:11.5px">G2.1 / G2.2 / G2.3 → K8 or K1 goes to S3</div></div></div>
    ${anyRec ? `<div class="q3tier" style="margin-top:12px"><b>Q3b · recurrent tier</b></div><div class="q3seeds">${Q3_SEEDS.map(sd => `<div><div class="q3note">seed ${sd}</div>${rowOf(sd, Q3B_ARMS)}</div>`).join("")}</div>` : `<div class="q3tier muted" style="margin-top:10px">Q3b · recurrent tier: runs after Q2, not started.</div>`}`;
  $("#lv-q3-table").innerHTML = q3Table();
  // curves: symbolic is the default because text loss is flat in every arm
  const runs = Object.keys(d.runs).filter(n => /^S2_(V8repro|K1|K8)_(dense|rec)_s\d+$/.test(n)).sort();
  const keys = ["symbolic", "ncp", "text", "loss"].filter(k => runs.some(n => col(d.runs[n], k).length));
  if (!keys.includes(ST.q3metric)) ST.q3metric = keys[0];
  $("#lv-q3-metric").innerHTML = keys.map(k => `<button data-m="${k}" class="${k === ST.q3metric ? "on" : ""}">${Q3_METRICS[k]}</button>`).join("");
  $$("#lv-q3-metric button").forEach(b => b.addEventListener("click", () => { ST.q3metric = b.dataset.m; renderQ3(); }));
  const sty = n => { const m = /^S2_(\w+?)_(dense|rec)_s(\d+)$/.exec(n), seeds = runs.filter(x => x.startsWith(`S2_${m[1]}_${m[2]}_`)).map(x => +x.split("_s")[1]).sort(), i = Math.max(0, seeds.indexOf(+m[3]));
    return { color: `var(${Q3_COL[m[1]] || "--ink2"})`, dash: m[2] === "rec" ? "6 4" : "", op: i ? .62 : 1, w: i ? 1.7 : 2.2, label: `${m[1] === "V8repro" ? "V8-repro" : m[1]} ${m[2]} · s${m[3]}`, arm: m[1] }; };
  $("#lv-q3-legend").innerHTML = runs.map(n => { const s = sty(n); return `<button data-run="${n}" class="${ST.hidden.has(n) ? "off" : ""}"><i class="${s.dash ? "dash" : ""}" style="border-top-color:${s.color};opacity:${s.op}"></i>${esc(s.label)}${isRunning(d.runs[n]) ? " ●" : ""}</button>`; }).join("");
  $$("#lv-q3-legend button").forEach(b => b.addEventListener("click", () => { ST.hidden.has(b.dataset.run) ? ST.hidden.delete(b.dataset.run) : ST.hidden.add(b.dataset.run); renderQ3(); }));
  const sm = +($("#lv-q3-smooth").value || 3), ser = runs.map(n => { const r = d.runs[n], s = sty(n); return { x: col(r, "step"), y: smoothArr(col(r, ST.q3metric), sm), color: s.color, dash: s.dash, op: s.op, w: s.w, label: s.label, hide: ST.hidden.has(n), endDot: isRunning(r) }; });
  const tr = d.runs.S1_A_delayed_s902, refs = tr && ST.q3metric !== "loss" && col(tr, ST.q3metric).length ? [{ y: col(tr, ST.q3metric).filter(v => v != null).slice(-10).reduce((a, b, _, v) => a + b / v.length, 0), label: "teacher end · s911", color: "var(--ink3)" }] : [];
  lineChart($("#lv-q3-curves"), ser, { h: 280, xlabel: "update", xdom: [500, Math.max(...ser.flatMap(s => s.x), 1000)], title: ST.q3metric, vfmt: v => (+v.toPrecision(4)).toString(), refs,
    vlines: [{ x: 1200, label: "δ→0 starts" }, { x: 3600, label: "δ = 0" }] });
  // K8 hypothesis diagnostics
  const k8 = runs.filter(n => /^S2_K8_/.test(n) && col(d.runs[n], "selector_entropy").length), kh = $("#lv-q3-k8-wrap");
  if (kh) kh.hidden = !k8.length;
  if (k8.length) {
    if (!K8_METRICS.some(m => m[0] === ST.k8metric)) ST.k8metric = K8_METRICS[0][0];
    const M = K8_METRICS.find(m => m[0] === ST.k8metric);
    $("#lv-q3-k8-metric").innerHTML = K8_METRICS.map(m => `<button data-m="${m[0]}" title="${m[2]}" class="${m[0] === ST.k8metric ? "on" : ""}">${m[1]}</button>`).join("");
    $$("#lv-q3-k8-metric button").forEach(b => b.addEventListener("click", () => { ST.k8metric = b.dataset.m; renderQ3(); }));
    const ks = k8.map(n => { const r = d.runs[n], s = sty(n); return { x: col(r, "step"), y: smoothArr(col(r, M[0]), sm), color: s.color, dash: s.dash, op: s.op, w: s.w, label: s.label, endDot: isRunning(r) }; });
    const krefs = M[0] === "selector_entropy" ? [{ y: Math.log(8), label: "uniform over 8 = ln 8", color: "var(--ink3)" }] : M[0] === "kdec_is_kstar" ? [{ y: Q3_M / 8, label: "m / K = 0.25 if the prior were uniform", color: "var(--ink3)" }] : [];
    lineChart($("#lv-q3-k8"), ks, { h: 240, xlabel: "update", xdom: [500, Math.max(...ks.flatMap(s => s.x), 1000)], title: M[1], vfmt: v => (+v.toPrecision(4)).toString(), refs: krefs });
    $("#lv-q3-k8-cap").textContent = M[2] + ". Logged by train_s2 on scored positions of the K = 8 runs.";
  }
}
/* ================================================================ chrome: pill, overview, nav dot */
L.gatePill = (stage, name) => {
  if (stage !== "S1" || name !== "competence" && name !== "Competence" || !L.data || !L.data.g1 || !L.data.g1.seeds) return null;
  const v = Object.values(L.data.g1.seeds), ok = v.filter(x => x.passed).length;
  return `<span class="pill ${ok === v.length ? "gate" : "kill"}" title="development diagnostic: dev point estimates, 2,048 items">G1 dev: ${ok} of ${v.length} anchor seeds passed${L.data.g1.final === false ? " · provisional" : ""}</span>`;
};
function renderChrome() {
  const d = L.data, cur = d ? current() : null, pill = $("#live-pill"), t = $("#live-pill-t"), dot = $("#nav-dot");
  [pill, dot].forEach(e => e && e.classList.remove("on", "snap"));
  let txt, status;
  if (!d || L.source === "offline") { txt = "telemetry offline"; status = "design · telemetry offline"; }
  else if (L.source === "snapshot") { pill.classList.add("snap"); dot.classList.add("snap"); txt = "snapshot " + fmtT(utcMs(d.server_utc)) + " IST"; status = "snapshot · " + fmtDT(utcMs(d.server_utc)) + " IST"; }
  else {
    pill.classList.add("on"); dot.classList.add("on");
    const q3b = d.blocks.Q3;
    if (cur) { const step = lastOf(col(cur.r, "step")), plan = cur.r.config.steps || 24000, p = parseRun(cur.name), pl = PLAN.find(x => x.match.test(cur.name)), tag = p.seed ? ` s${p.seed}` : ""; txt = `live · ${p.arm}${tag} ${(100 * step / plan).toFixed(0)}%`; status = `${pl ? pl.id : p.stage} training · ${p.arm}${tag} · ${(100 * step / plan).toFixed(0)}%`; }
    else if (q3b && q3b.status === "running") { txt = "live · Q3 " + String(q3b.stage || "").replace(/^S2_/, ""); status = "Q3 · " + (q3b.stage || "running"); }
    else { txt = "live · queue idle"; status = "live · no run training"; }
  }
  if (L.source === "stale") txt = "telemetry stale";
  const lc = $("#lv-conn"); if (lc) { const c = L.source === "live" ? ["--good", "live · " + (L.latency ? Math.round(L.latency) + " ms" : "connected")] : L.source === "snapshot" ? ["--warn", "snapshot"] : L.source === "stale" ? ["--warn", "stale"] : ["--frozen", "offline"]; lc.innerHTML = `<i style="background:var(${c[0]})"></i>${c[1]}`; }
  t.textContent = txt; const sp = $("#status-pill"); if (sp) sp.textContent = status;
  const vp = $("#ver-pill"); if (vp) vp.textContent = "draft_v9.md · " + (!d || L.source === "offline" ? "design" : L.source === "snapshot" ? "snapshot" : cur ? "queue running" : "queue idle");
  const sq = $("#stat-q"), sql = $("#stat-q-l");
  if (sq && d && L.source !== "offline" && d.runs && Object.keys(d.runs).length) { try { const P = planRows(); sq.textContent = fmtDT(P.fin).replace(",", ""); sql.textContent = "projected end of the queue, IST (estimate; Q2–Q6 durations are planned, not measured)"; } catch (e) {} }
  const box = $("#ov-live");
  if (box) {
    if (!d || L.source === "offline") { box.hidden = true; return; }
    const eta = q1Eta(), rs = S1runs(), flat = rs.filter(n => /A_flat/.test(n)), hier = rs.filter(n => !/A_flat/.test(n)), core = runningCore(), e3 = core ? q3Eta(core) : null;
    const step = cur && /^S1_/.test(cur.name) ? lastOf(col(cur.r, "step")) : null, avgAt = (names, k) => { const v = names.map(n => { const r = d.runs[n], st = col(r, "step"), c = col(r, k), idx = st.map((s, i) => i).filter(i => st[i] > step - 1000 && st[i] <= step); const x = idx.map(i => c[i]).filter(z => z != null); return x.length ? x.reduce((a, b) => a + b, 0) / x.length : null; }).filter(z => z != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
    const hl = step && flat.length && hier.length ? `At update ${step.toLocaleString("en-US")}: text loss ${f3(avgAt(flat, "text"))} flat vs ${f3(avgAt(hier, "text"))} hierarchy (mean), symbolic ${f3(avgAt(flat, "symbolic"))} vs ${f3(avgAt(hier, "symbolic"))}. ` : "";
    box.hidden = false;
    box.innerHTML = `<div style="display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;justify-content:space-between"><div><div class="eyebrow">${L.source === "live" ? "live now" : "snapshot"} · development diagnostics</div><b>${status}</b>${eta ? ` · Q1 finishes about ${fmtDT(eta.end)} IST` : ""}${e3 && !e3.unknown ? ` · ${core === "dense" ? "Q3a" : "Q3b"} finishes about ${fmtDT(e3.slow)} IST (${e3.done} of ${e3.cells} cells measured)` : ""}<div class="muted" style="font-size:13px">${hl}Not a sealed result.</div></div><a class="btn primary" href="#live" style="text-decoration:none">Open live telemetry →</a></div>`;
  }
}
function signature() { const d = L.data; return d ? [L.source, d.server_utc && L.source !== "live" ? d.server_utc : "", Object.values(d.runs).map(r => r.series.n + ":" + r.dev_checks.length + (r.completed ? "c" : "")).join(","), Object.values(d.blocks).map(b => b.status + b.stage).join("|"), (d.events || []).length, Object.keys(d.tests).length].join("#") : "x"; }
function render() {
  renderChrome();
  const d = L.data; if (!d) { if (ST.active) $("#lv-kpis").innerHTML = `<div class="notice" style="grid-column:1/-1">No telemetry yet. Start the read-only server with <code>python serve.py</code> in <code>v9_webapp/</code> and this page connects on its own. Without it, a page built with a snapshot shows the snapshot.</div>`; return; }
  if (!ST.active) { const sg = signature(); if (sg !== L.sigAll) { L.sigAll = sg; window.dispatchEvent(new Event("v9live")); } return; }
  const note = L.source === "snapshot" ? (hosted() ? `<div class="notice snapnote" style="grid-column:1/-1">This published copy cannot reach your computer, so it shows a snapshot taken ${fmtDT(utcMs(d.server_utc))} IST. For live updates open <code>http://127.0.0.1:8765/</code> on the Mac that runs the experiment (start the server with <code>python serve.py</code> in <code>v9_webapp/</code>).</div>` : `<div class="notice snapnote" style="grid-column:1/-1">Showing a snapshot taken ${fmtDT(utcMs(d.server_utc))} IST. Run <code>python serve.py</code> in <code>v9_webapp/</code> for live updates.</div>`) : L.source === "stale" ? `<div class="notice snapnote" style="grid-column:1/-1">The telemetry server stopped answering; showing the last data received.</div>` : "";
  renderKpis(); $("#lv-kpis").insertAdjacentHTML("afterbegin", note); renderQueue(); renderSys(); renderTests();
  const sig = signature(); if (sig === ST.sig) return; ST.sig = sig;
  window.dispatchEvent(new Event("v9live"));
  renderGantt(); renderQ3(); renderQ1();
}
function renderQ1() {              // the finished Q1 panels sit in a fold; their charts are only drawn while it is open
  const f = $("#lv-q1"); if (!f) return;
  if (!L.q1touched) f.open = !(L.data.blocks.Q1 && /complete|done|finished/.test(L.data.blocks.Q1.status) && Object.keys(L.data.runs).some(n => /^S2_/.test(n)));
  const sub = $("#lv-q1-sub"); if (sub) sub.textContent = L.data.blocks.Q1 ? "block " + L.data.blocks.Q1.status + " · " + S1runs().length + " runs" : "";
  if (f.open) { renderCurves(); renderDev(); renderMatched(); renderSpeed(); renderG1(); }
}

/* ================================================================ wiring */
function buildControls() {
  $("#lv-metric").innerHTML = METRICS.map(m => `<button data-m="${m[0]}" class="${m[0] === ST.metric ? "on" : ""}">${m[1]}</button>`).join("");
  $$("#lv-metric button").forEach(b => b.addEventListener("click", () => { ST.metric = b.dataset.m; $$("#lv-metric button").forEach(x => x.classList.toggle("on", x === b)); renderCurves(); }));
  $$("#lv-from button").forEach(b => b.addEventListener("click", () => { ST.from = +b.dataset.f; $$("#lv-from button").forEach(x => x.classList.toggle("on", x === b)); renderCurves(); }));
  $("#lv-q3-smooth").addEventListener("input", () => renderQ3());
  $("#lv-smooth").addEventListener("input", e => { ST.smooth = +e.target.value; renderCurves(); });
  $("#lv-logy").addEventListener("change", e => { ST.logy = e.target.checked; renderCurves(); });
  $$("#lv-step-mode button").forEach(b => b.addEventListener("click", () => { ST.stepMode = b.dataset.m; $$("#lv-step-mode button").forEach(x => x.classList.toggle("on", x === b)); $("#lv-step-wrap").hidden = ST.stepMode !== "pick"; renderMatched(); }));
  $("#lv-step").addEventListener("input", e => { ST.step = +e.target.value; renderMatched(); });
  $("#lv-url").value = ST.url; $("#lv-url").addEventListener("change", e => { ST.url = e.target.value.trim(); try { localStorage.setItem("v9-telemetry-url", ST.url); } catch (x) {} L.source = "offline"; poll(); });
  $("#lv-refresh").addEventListener("click", () => { ST.sig = ""; poll(); });
}
let timer = null;
function schedule() { clearTimeout(timer); const auto = $("#lv-auto"); if (auto && !auto.checked) return; timer = setTimeout(async () => { if (!document.hidden) await poll(); schedule(); }, ST.active ? 5000 : 15000); }
window.addEventListener("v9tab", e => { ST.active = e.detail === "live"; if (ST.active) { ST.sig = ""; render(); } schedule(); });
window.addEventListener("v9theme", () => { ST.sig = ""; render(); });
let rz = null; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { ST.sig = ""; render(); }, 200); });
document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
buildControls();
$("#lv-q1 > summary").addEventListener("click", () => { L.q1touched = true; });       // a viewer's own choice beats the default (open until Q1 is done)
$("#lv-q1").addEventListener("toggle", () => { if ($("#lv-q1").open && L.data) renderQ1(); });
$("#lv-auto").addEventListener("change", schedule);
if (window.V9_SNAPSHOT) { L.data = window.V9_SNAPSHOT; L.source = "snapshot"; }
const first = document.querySelector("#tab-live.on"); ST.active = !!first;
render(); poll().then(schedule);
})();
