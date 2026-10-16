/* Live telemetry. /api/live every 5 s (GPU, process, heartbeat, file signatures); /api/telemetry when a log file changes. */
const PROBES = ["lr_recall1", "lr_carry3", "lr_multi4", "lr_hop", "lr_bind"];
const C = { total: "#7d8aa0", ntp: "#86ade0", ema: "#2f6fb3", ncp: "#2e9c8c", vq: "#cfa935", ts: "#8f7cc0", wk: "#e07b39", mean: "#5f7896",
  r1: "#2e9c8c", swap: "#d0667a", c3: "#3b78c2", m4: "#8f7cc0", hop: "#e07b39", bind: "#cfa935", H: "#999", gap: "#2e8b57" };
const f3 = v => v == null || !isFinite(v) ? "—" : Number(v).toFixed(3), f2 = v => v == null || !isFinite(v) ? "—" : Number(v).toFixed(2),
  f4 = v => v == null ? "—" : Number(v).toFixed(4), fI = v => v == null ? "—" : Math.round(v).toLocaleString("en-US");
const tparse = s => s ? new Date(s.replace(" ", "T")).getTime() : null;
const mean = a => a && a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
let charts = null, liveCharts = null, lastData = null, lastLive = null, lastSig = null, liveAt = 0;

function fmtH(h) { if (h == null || !isFinite(h)) return "—"; const hh = Math.floor(h), mm = Math.round((h - hh) * 60); return hh ? `${hh} h ${String(mm).padStart(2, "0")} min` : `${mm} min`; }
function ago(ms) { const s = Math.max(0, Math.round(ms / 1000)); return s < 60 ? `${s} s ago` : s < 3600 ? `${Math.floor(s / 60)} min ago` : `${(s / 3600).toFixed(1)} h ago`; }
function clock(ms) { return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }

/* ------------------------------------------------------------------ derived live state */
function runState() {
  const st = lastData?.status?.state, proc = lastLive?.proc;
  if (lastLive && !proc) return lastLive.pause_flag || st === "paused" ? "paused" : st === "finished" ? "finished" : "stopped";
  return st === "starting" ? "running" : (st || "unknown");
}
function currentStep() {
  const L = lastLive, hb = L?.heartbeat, now = Date.now();
  if (hb && hb.t && now / 1000 - hb.t < 180) return { step: hb.step, exact: true, t: hb.t * 1000 };
  const r = L?.last_row || lastData?.status; if (!r || r.step == null) return null;
  const base = r.step, t0 = tparse(r.time), sps = L?.sec_per_step || lastData?.status?.sec_per_step;
  if (runState() !== "running" || !sps || !t0) return { step: base, exact: true, t: t0 };
  const inc = Math.min(Math.floor((now - t0) / 1000 / sps), 49);
  return { step: base + Math.max(inc, 0), exact: false, t: t0 };
}
function nextEval() {
  const a = tparse(lastLive?.last_eval_anchor); if (!a) return null;
  const start = a + 60 * 60e3; return { start, done: start + 2.5 * 60e3 };
}

/* ------------------------------------------------------------------ header pill */
function setPill() {
  if (document.documentElement.dataset.ver === "v3" || document.documentElement.dataset.ver === "v4") {   // v4 pages show the live v3 run
    const P3 = window.V3PILL;                                       // set by js/v3train.js from data/v3_live.json
    const live3 = P3 && P3.age != null && P3.age < 15;
    document.getElementById("live-dot").className = "dot " + (live3 ? "running" : "v3dot");
    document.getElementById("live-text").textContent = P3 ? `v3 ${live3 ? "training" : "not live"} · step ${fI(P3.step)}${P3.val != null ? ` · val ${f3(P3.val)}` : ""}` : "v3 · no run data";
    document.getElementById("live-gpu").textContent = P3 && P3.tok_s ? `${fI(P3.tok_s)} tok/s` : "";
    document.getElementById("live-age").textContent = P3 && P3.age != null ? `updated ${P3.age < 1 ? "just now" : Math.round(P3.age) + " min ago"}` : "";
    document.getElementById("live-pill").title = "v3 run status from data/v3_live.json (written every 25 steps)";
    return;
  }
  document.getElementById("live-pill").title = "Live status (refreshes every 5 s)";
  const P2 = document.documentElement.dataset.ver === "v2" ? window.V2PILL : null;   // v2 mode: the v2 run's status (js/v2.js)
  if (P2) {
    document.getElementById("live-dot").className = "dot " + (P2.state === "starting" ? "running" : P2.state);
    document.getElementById("live-text").textContent = `v2 ${P2.state}${P2.step != null ? ` · step ${fI(P2.step)}` : ""}${P2.val != null ? ` · val ${f3(P2.val)}` : ""}`;
    const g = lastLive?.gpu?.device; document.getElementById("live-gpu").textContent = g != null ? `GPU ${g}%` : "";
    document.getElementById("live-age").textContent = liveAt ? `updated ${ago(Date.now() - liveAt)}` : "";
    return;
  }
  const s = runState(), cs = currentStep(), dot = document.getElementById("live-dot");
  dot.className = "dot " + s;
  document.getElementById("tab-live-dot").className = "tab-live" + (s === "running" ? " on" : "");
  const ev = lastData?.evals?.at(-1);
  document.getElementById("live-text").textContent = `${s}${cs ? ` · step ${cs.exact ? "" : "≈"}${fI(cs.step)}` : ""}${ev ? ` · val ${f3(ev.val_mean)}` : ""}`;
  const g = lastLive?.gpu?.device;
  document.getElementById("live-gpu").textContent = g != null ? `GPU ${g}%` : "";
  document.getElementById("live-age").textContent = liveAt ? `updated ${ago(Date.now() - liveAt)}` : "";
}

/* ------------------------------------------------------------------ live training page: hero, verdict, KPI tiles, section nav */
const LVL_RANK = { r: 3, a: 2, g: 1, n: 0 };
const worst = arr => arr.reduce((w, s) => (LVL_RANK[s.lvl] > LVL_RANK[w] ? s.lvl : w), "n");
function chartEntries() { return [...Object.entries(liveCharts || {}), ...Object.entries(charts || {})]; }
function allVerdicts() {
  const out = chartEntries().filter(([, c]) => c.status).map(([k, c]) => ({ id: "ch-" + k, t: c.o.title, lvl: c.status.lvl, why: c.status.why }));
  healthItems().filter(i => ["Training process", "Time budget", "Power"].includes(i.t)).forEach(i => out.unshift({ id: "s-checks", t: i.t, lvl: i.lvl, why: i.what }));
  return out;
}
function sectionLevel(secId) {
  const sec = document.getElementById(secId); if (!sec) return { lvl: "n", n: {} };
  let items;
  if (secId === "s-checks") items = healthItems().map(i => ({ lvl: i.lvl }));
  else items = chartEntries().filter(([, c]) => c.status && sec.contains(c.wrap)).map(([, c]) => ({ lvl: c.status.lvl }));
  const n = { g: 0, a: 0, r: 0 }; items.forEach(i => { if (n[i.lvl] != null) n[i.lvl]++; });
  return { lvl: worst(items), n, total: items.length };
}
const countChips = n => ["g", "a", "r"].filter(k => n[k]).map(k => `<span class="cc ${k}"><i></i>${n[k]} ${{ g: "healthy", a: "watch", r: "problem" }[k]}</span>`).join("");

/* ------------------------------------------------------------------ two-stage plan (D-041) */
const CONT_PLAN = { steps: 4050, lr: 6e-4, warmup: 300, lr_start: 6e-5, lr_end: 6e-5, hours: 24 };
function planInfo() {                  // server "plan" when available; otherwise derived from the chain log (older server)
  const d = lastData; if (!d) return null;
  if (d.plan) return d.plan;
  const chain = d.chain || [], idx = pat => chain.reduce((m, l, i) => (l.includes(pat) ? i : m), -1);
  const cont = chain.filter(l => / cont |chain4/.test(l)), lastC = cont.at(-1) || "";
  const st = d.status || {}, mainDone = /STOP \((max_steps|time_budget)\)/.test(st.last_stop || "");
  const started = idx("cont start") >= 0;
  const s1 = started ? "done" : mainDone ? (idx("chain done") > idx("main start") ? "done" : "final evaluation") : (runState() || "running");
  let s2 = "not armed";
  if (idx("cont done") >= 0) s2 = "done";
  else if (idx("cont training done") > idx("cont start") && started) s2 = "evaluating";
  else if (started && idx("cont failed") > idx("cont start")) s2 = "failed";
  else if (started && idx("cont paused") > idx("cont start")) s2 = "paused";
  else if (started) s2 = "running";
  else if (/cancelled/.test(lastC)) s2 = "cancelled";
  else if (/not started/.test(lastC)) s2 = "not started: " + lastC.split("not started:")[1].trim();
  else if (/cont armed/.test(lastC)) s2 = "armed";
  return { stage: started ? 2 : 1, s1, s2, main_steps: 4050, cont: CONT_PLAN, derived: true };
}
function contGrace() {                 // first ~600 continuation steps: LR re-warms, a temporary loss rise is expected
  const S = lastData?.status?.schedule, cs = currentStep();
  return !!(S && cs && cs.step >= S.offset && cs.step < S.offset + (S.warmup || 100) + 1300);
}
function renderPlan() {
  const P = planInfo(); if (!P) return "";
  const cs = currentStep(), st = lastData.status || {}, main = P.main_steps || 4050, cont = P.cont?.steps || 4050;
  const g = cs ? cs.step : 0, p1 = Math.min(100, 100 * Math.min(g, main) / main), p2 = Math.max(0, Math.min(100, 100 * (g - main) / cont));
  const lvl = s => /done/.test(s) ? "done" : /running|starting|started|evaluat|final/.test(s) ? "on" : /armed/.test(s) ? "armed" : /cancel|not started|failed|stopped|paused/.test(s) ? "off" : "";
  const eta = st.eta_h != null && P.stage === 1 && P.s1 === "running" ? new Date(Date.now() + (st.eta_h + 1.3) * 3600e3).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" }) : null;
  const lbl = s => s === "armed" ? "armed · waiting" : s;
  return `<div class="plan">
    <div class="stg ${lvl(P.s1)}"><div class="stg-h"><span class="stg-n">1</span><b>Main run</b><span class="stg-st">${lbl(P.s1)}</span></div>
      <div class="stg-bar"><div style="width:${p1}%"></div></div>
      <div class="stg-s">steps 0 → 4,050 · LR 6e-4 cosine → 6e-5 · then final evaluation (≈ 1.3 h)</div></div>
    <div class="stg-arrow">→</div>
    <div class="stg ${lvl(P.s2)}"><div class="stg-h"><span class="stg-n">2</span><b>Continuation · +24 h</b><span class="stg-st">${lbl(P.s2)}</span></div>
      <div class="stg-bar"><div style="width:${p2}%"></div></div>
      <div class="stg-s">steps 4,050 → 8,100 · LR 6e-5 → ${(P.cont?.lr || 6e-4).toExponential(0)} (${P.cont?.warmup || 300} steps) → 6e-5 · same data mix, unseen rows${eta && /armed/.test(P.s2) ? ` · starts ≈ <b>${eta}</b>` : ""}</div></div>
  </div>`;
}

function renderNow() {
  const el = document.getElementById("now"); if (!el || !lastLive) return;
  const s = runState(), cs = currentStep(), st = lastData?.status || {}, L = lastLive, now = Date.now();
  const total = st.total_steps || 4050, pct = cs ? 100 * cs.step / total : 0;
  const sps = L.sec_per_step || st.sec_per_step, lr = L.last_row, t0 = tparse(lr?.time);
  const nextLog = t0 && sps ? t0 + 50 * sps * 1000 : null, ne = nextEval();
  const remH = cs && sps ? (total - cs.step) * sps / 3600 : null;
  const evalNow = ne && now > ne.start && now < ne.done + 3 * 60e3;
  const gpu = L.gpu?.device;
  // verdict
  const V = allVerdicts(), n = { g: 0, a: 0, r: 0 }; V.forEach(v => { if (n[v.lvl] != null) n[v.lvl]++; });
  const wl = n.r ? "r" : n.a ? "a" : "g", judged = n.g + n.a + n.r;
  const vt = wl === "g" ? `All ${judged} checks healthy` : wl === "a" ? `${n.a} ${n.a > 1 ? "things" : "thing"} to watch, nothing wrong` : `${n.r} ${n.r > 1 ? "problems need" : "problem needs"} attention`;
  const vs = wl === "g" ? "Every chart and check on this page is inside its healthy range." : wl === "a" ? "Amber means worth a look, not broken. Click an item to jump to its chart." : "Red means a value is outside its safe range. Click an item to jump to its chart.";
  const flagged = V.filter(v => v.lvl === "r" || v.lvl === "a").sort((a, b) => LVL_RANK[b.lvl] - LVL_RANK[a.lvl]);
  const ticks = (lastData?.evals || []).filter(e => e.step <= total).map(e => `<i style="left:${100 * e.step / total}%" title="evaluation at step ${e.step}"></i>`).join("");
  el.innerHTML = `
    <div class="hero-main">
      <div class="hero-l">
        <div class="state-chip s-${s}"><span class="dot ${s}"></span>${s}${evalNow && s === "running" ? " · evaluating now" : ""}</div>
        <div class="stepbig">${cs ? (cs.exact ? "" : "≈ ") + fI(cs.step) : "—"} <small>/ ${fI(total)} steps</small></div>
        <div class="pbar2"><div class="fill" style="width:${pct}%"></div><div class="ticks">${ticks}</div></div>
        <div class="hero-meta">
          <span><b>${pct.toFixed(1)} %</b> done</span>
          <span><b>${cs ? (cs.step * 32768 / 1e6).toFixed(1) : "—"}M</b> of ${(total * 32768 / 1e6).toFixed(1)}M tokens</span>
          <span>${s === "running" && remH ? `finishes ≈ <b>${new Date(now + remH * 3600e3).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" })}</b>` : s === "paused" ? "clock stopped while paused" : "—"}</span>
          <span class="muted">${cs && !cs.exact ? "step estimated from speed" : L.heartbeat ? "exact (heartbeat)" : "last logged step"}</span>
        </div>
      </div>
      <div class="verdict v-${wl}">
        <div class="v-icon">${wl === "g" ? "✓" : wl === "a" ? "!" : "✕"}</div>
        <div class="v-body">
          <div class="v-t">${vt}</div><div class="v-s">${vs}</div>
          <div class="v-counts">${countChips(n)}</div>
          ${lastData?.evals?.length ? `<div class="v-ev">Latest evaluation: step ${fI(lastData.evals.at(-1).step)} at ${clock(tparse(lastData.evals.at(-1).time))}${ne && s === "running" ? ` · next ≈ ${clock(ne.start)}` : ""}</div>` : ""}
          ${flagged.length ? `<ul class="v-list">${flagged.slice(0, 5).map(v => `<li data-go="${v.id}" class="${v.lvl}"><i></i><b>${v.t}</b><span>${v.why}</span></li>`).join("")}</ul>` : ""}
        </div>
      </div>
    </div>
    ${renderPlan()}
    <div class="hero-row">
      <div class="hs"><div class="lbl">Time left</div><div class="val">${s === "running" ? fmtH(remH) : "—"}</div>
        <div class="s">used ${fmtH(st.wall_h)} of 24 h${st.budget_left_h != null && remH != null && remH > st.budget_left_h ? ' · <span class="warn">budget ends first</span>' : ""}</div></div>
      <div class="hs"><div class="lbl">Next data</div><div class="val sm">${nextLog && s === "running" ? (now < nextLog ? "losses ≈ " + clock(nextLog) : "losses due now") : "—"}</div>
        <div class="s">${ne && s === "running" ? (evalNow ? "evaluation running (≈ 2 min)" : "evaluation ≈ " + clock(ne.start)) : ""}</div></div>
      <div class="hs"><div class="lbl">GPU load</div><div class="val">${gpu != null ? gpu + " %" : "—"}</div>
        <canvas class="spark" id="spark-gpu"></canvas></div>
      <div class="hs"><div class="lbl">Trainer</div><div class="val sm">${L.proc ? `CPU ${L.proc.cpu.toFixed(0)} % · ${L.proc.rss_gb} GB` : "not running"}</div>
        <div class="s">${L.proc ? "up " + L.proc.etime + " · GPU mem " + (L.gpu?.mem_gb ?? "—") + " GB" : ""}</div></div>
      <div class="hs"><div class="lbl">Power</div><div class="val sm">${L.battery ? L.battery.pct + " % · " + L.battery.state : "—"}</div>
        <div class="s">${L.battery?.ac ? "on charger" : '<span class="warn">on battery</span> · auto-pause at 7 %'} · guard ${lastData?.system?.battery_guard ? "on" : "off"}</div></div>
      <div class="hs"><div class="lbl">Memory</div><div class="val sm">swap ${L.swap_gb ?? "—"} GB</div>
        <div class="s">${L.pause_flag ? '<span class="warn">PAUSE flag set</span>' : "no pause requested"} · disk ${lastData?.system?.disk_free_gb ?? "—"} GB free</div></div>
    </div>`;
  spark2(document.getElementById("spark-gpu"), (L.hist || []).map(h => h[1]), "#2e9c8c", 0, 100);
}

function renderKpis() {
  const el = document.getElementById("trp-kpis"); if (!el || !lastData) return;
  const ev = lastData.evals, tr = lastData.train, e1 = ev.at(-1), e0 = ev.at(-2), t1 = tr.at(-1), t0 = tr.at(-7);
  const cst = k => charts?.[k]?.status?.lvl || "n", gap = e => e.codebook.code_entropy - e.codebook.ncp_ce;
  const d = (a, b, dig = 3, lowerBetter = true) => (a == null || b == null) ? "" : `<span class="kd ${b === a ? "" : (b - a < 0) === lowerBetter ? "kg" : "kb"}">${sgn(b - a, dig)}</span>`;
  const tiles = [
    { k: "val", lbl: "Validation loss", v: f3(e1?.val_mean), u: "nats", dl: d(e0?.val_mean, e1?.val_mean), sub: "vs previous evaluation", sp: ev.map(e => e.val_mean), col: "#8f7cc0" },
    { k: "probe", lbl: "Long-range recall", v: f3(e1?.probes.lr_recall1?.exact), u: "", dl: d(e0?.probes.lr_recall1?.exact, e1?.probes.lr_recall1?.exact, 3, false), sub: `plan swap ${f3(e1?.swap_recall1)} (chance 0.008)`, sp: ev.map(e => e.probes.lr_recall1?.exact), col: "#2e9c8c" },
    { k: "gap", lbl: "Concept gap H − NCP", v: e1 ? f2(gap(e1)) : "—", u: "nats", dl: e0 && e1 ? d(gap(e0), gap(e1), 2, false) : "", sub: "how predictable the next concept is", sp: ev.map(gap), col: "#2e8b57" },
    { k: "ntp", lbl: "Training loss (NTP)", v: f3(t1?.ntp_ema), u: "nats", dl: d(t0?.ntp_ema, t1?.ntp_ema), sub: "vs 300 steps ago", sp: tr.slice(-40).map(r => r.ntp_ema), col: "#3b78c2" },
    { k: "tps", lbl: "Throughput", v: t1?.tok_s ? fI(t1.tok_s) : "—", u: "tok/s", dl: "", sub: `GPU ${lastLive?.gpu?.device ?? "—"} % busy now`, sp: tr.slice(-40).map(r => r.tok_s), col: "#cfa935" },
    { k: "codes", lbl: "Codebook alive", v: lastData.restarts.at(-1) ? (100 * lastData.restarts.at(-1).codes_used_min_seg).toFixed(0) + " %" : "—", u: "", dl: "", sub: `VE ${f3(e1?.codebook.var_explained)} · perplexity ${f2(e1?.codebook.perplexity_mean)}`, sp: lastData.restarts.slice(-30).map(r => r.codes_used_min_seg), col: "#e07b39", lo: 0.8, hi: 1 },
  ];
  el.innerHTML = tiles.map(t => `<button class="kpi k-${cst(t.k)}" data-go="ch-${t.k}" title="${charts?.[t.k]?.status?.why || ""}">
      <div class="kl"><span class="kdot"></span>${t.lbl}</div>
      <div class="kv">${t.v}<small>${t.u}</small>${t.dl}</div>
      <div class="ks">${t.sub}</div><canvas class="kspark" data-k="${t.k}"></canvas></button>`).join("");
  tiles.forEach(t => spark2(el.querySelector(`canvas[data-k="${t.k}"]`), t.sp, t.col, t.lo, t.hi));
}

function renderNav() {
  const nav = document.getElementById("trp-nav"); if (!nav) return;
  const secs = [...document.querySelectorAll("#tab-train .trp-sec")];
  nav.innerHTML = secs.map(sec => {
    const S = sectionLevel(sec.id), sum = sec.querySelector(".trp-sum");
    if (sum) sum.innerHTML = S.total ? countChips(S.n) : "";
    return `<button data-go="${sec.id}" class="${nav.dataset.active === sec.id ? "on" : ""}"><span class="nd ${S.total ? S.lvl : "n"}"></span>${sec.dataset.title}</button>`;
  }).join("");
}

function spark2(cv, vals, color, lo, hi) {
  if (!cv) return; const dpr = Math.min(devicePixelRatio || 1, 2), W = cv.clientWidth, H = cv.clientHeight; if (!W) return;
  cv.width = W * dpr; cv.height = H * dpr; const c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  const v = vals.map(x => (fin(x) ? x : null)), ok = v.filter(x => x != null); if (ok.length < 2) return;
  const a = lo ?? Math.min(...ok), b = hi ?? Math.max(...ok), r = b - a || 1;
  const P = v.map((x, i) => x == null ? null : [i / (v.length - 1) * W, H - 2 - (x - a) / r * (H - 5)]).filter(Boolean);
  const g = c.createLinearGradient(0, 0, 0, H); g.addColorStop(0, color + "40"); g.addColorStop(1, color + "00");
  c.beginPath(); P.forEach(([x, y], i) => i ? c.lineTo(x, y) : c.moveTo(x, y)); c.lineTo(P.at(-1)[0], H); c.lineTo(P[0][0], H); c.closePath(); c.fillStyle = g; c.fill();
  c.beginPath(); P.forEach(([x, y], i) => i ? c.lineTo(x, y) : c.moveTo(x, y)); c.strokeStyle = color; c.lineWidth = 1.6; c.stroke();
  const [lx, ly] = P.at(-1); c.fillStyle = color; c.beginPath(); c.arc(lx - 1.5, ly, 2.4, 0, 7); c.fill();
}

// jump links (verdict items, KPI tiles, section nav) and scroll-spy for the section nav
document.addEventListener("click", e => {
  const g = e.target.closest("[data-go]"); if (!g) return;
  const t = document.getElementById(g.dataset.go); if (!t) return;
  e.preventDefault(); t.scrollIntoView({ behavior: document.hidden || matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  if (t.classList.contains("chart")) { t.classList.remove("flash"); void t.offsetWidth; t.classList.add("flash"); }
});
window.addEventListener("scroll", () => {
  const nav = document.getElementById("trp-nav"); if (!nav || !document.getElementById("tab-train").classList.contains("on")) return;
  let cur = null; document.querySelectorAll("#tab-train .trp-sec").forEach(s => { if (s.getBoundingClientRect().top < 200) cur = s.id; });
  if (cur !== nav.dataset.active) { nav.dataset.active = cur || ""; nav.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.go === cur)); }
}, { passive: true });
function setHdrVar() { const h = document.querySelector("header.top"); if (h) document.documentElement.style.setProperty("--hdr", h.offsetHeight + "px"); }
window.addEventListener("resize", setHdrVar); setTimeout(setHdrVar, 0);

/* ------------------------------------------------------------------ health lights */
function healthItems() {
  const d = lastData; if (!d) return [];
  const ev = d.evals, last = ev.at(-1), prev = ev.at(-2), tr = d.train, rs = d.restarts.at(-1), L = lastLive || {};
  const s = runState(), items = [];
  const add = (lvl, t, v, what, bad) => items.push({ lvl, t, v, what, bad });
  add(s === "running" ? (L.gpu?.device == null || L.gpu.device > 50 ? "g" : "a") : s === "finished" ? "g" : s === "paused" ? "a" : "r", "Training process", s,
    s === "running" ? `running on the GPU (${L.gpu?.device ?? "?"} % busy)` : s === "finished" ? "training completed normally; the final evaluation runs next (≈ 1 h)" : s === "paused" ? "paused by you or the battery guard; resume with code/resume.sh" : "no trainer process found",
    "the process dying unexpectedly (the launcher retries up to 6 times)");
  if (last && prev) { const dv = last.val_mean - prev.val_mean;
    add(dv < -0.005 ? "g" : dv < 0.005 || contGrace() ? "a" : "r", "Validation loss falling", (dv >= 0 ? "+" : "") + dv.toFixed(3), `${f3(prev.val_mean)} → ${f3(last.val_mean)} nats since the previous evaluation`, "flat or rising for several hours (learning has stalled)"); }
  if (tr.length > 7) { const a = tr.at(-7).ntp_ema, b2 = tr.at(-1).ntp_ema, dv = b2 - a;
    add(dv < 0 ? "g" : dv < 0.05 || contGrace() ? "a" : "r", "Training loss trend", (dv >= 0 ? "+" : "") + dv.toFixed(3), `smoothed NTP ${f3(a)} → ${f3(b2)} over the last 300 steps`, "a sustained rise (instability)"); }
  const gn = tr.at(-1)?.gn_preclip;
  add(d.guards.length ? "a" : gn < 1 ? "g" : "a", "Stability", gn != null ? "grad " + f2(gn) : "—", d.guards.length ? `${d.guards.length} guard event(s), see Safety guards below` : "gradient norm steady under the 1.0 clip; no guard has fired", "NaN losses or spikes (they trigger an automatic reload at half the learning rate)");
  if (rs) add(rs.codes_used_min_seg >= 0.95 ? "g" : rs.codes_used_min_seg >= 0.8 ? "a" : "r", "Codebook in use (G-C1)", (100 * rs.codes_used_min_seg).toFixed(0) + " %", "codes used in the least-used of the 48 dictionaries since the last check", "below 95 % (parts of the codebook are dead)");
  if (last) { const gap = last.codebook.code_entropy - last.codebook.ncp_ce;
    add(gap > 0.2 ? "g" : gap > 0 ? "a" : "r", "Next concept predictable (G-C2)", gap.toFixed(2) + " nats", "how much better the next concept is predicted than by code frequency alone", "a gap ≤ 0 (the concept module predicts nothing useful)");
    add(last.codebook.var_explained >= 0.6 ? "g" : last.codebook.var_explained >= 0.5 ? "a" : "r", "Codebook fits the concepts (G-E2)", f3(last.codebook.var_explained), "share of concept variation captured by the nearest codes (0.6 expected)", "low values (codes summarise concepts poorly)");
    const r1 = last.probes.lr_recall1?.exact, sw = last.swap_recall1;
    add(r1 > 0.05 && sw < 0.03 ? "g" : "a", "Long-range recall via concepts", `${f3(r1)} vs ${f3(sw)}`, r1 > 0.05 && sw < 0.03 ? "recall works, and collapses to chance when the plan is swapped: the concept stream carries it" : "not yet clearly above chance (0.008)", "a high score that survives the plan swap (a leak around the concept stream)"); }
  const st = d.status;
  if (st.eta_h != null) add(st.eta_h <= st.budget_left_h ? "g" : "a", "Time budget", fmtH(st.eta_h) + " left", `${fmtH(st.budget_left_h)} of the 24 h budget remain`, "the budget ending before step 4,050");
  const tps = tr.at(-1)?.tok_s;
  if (tps) add(tps >= 1700 ? "g" : tps >= 1300 ? "a" : "r", "Speed", fI(tps) + " tok/s", "tokens per second over the last 50 steps", "below ~1,700, usually another app using the GPU (video, 3D, games)");
  const b = L.battery;
  if (b) add(b.ac || b.pct > 30 ? "g" : b.pct > 10 ? "a" : "r", "Power", `${b.pct} % ${b.ac ? "· charger" : "· battery"}`, b.ac ? "on the charger" : "on battery; the guard pauses safely at 7 %", "running out of battery (work since the last checkpoint would be lost)");
  return items;
}
function renderHealth(elId) {
  const el = document.getElementById(elId); if (!el) return;
  el.innerHTML = healthItems().map(i => `<div class="hl ${i.lvl}"><div class="lt"></div><div class="ht">${i.t}<span>${i.v}</span></div><div class="hd">${i.what}</div><div class="hb">bad would be: ${i.bad}</div></div>`).join("");
}

/* ------------------------------------------------------------------ per-chart verdicts (green / amber / red) */
const ST = (lvl, why) => ({ lvl, why });
const sgn = (v, d = 3) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d);
const fin = v => v != null && isFinite(v);
// trend of a 50-step training series: mean of the last 2 points vs 2 points `win` rows earlier
function trendSt(key, { win = 6, flat = 0.002, bad = 0.05, what = "" } = {}) {
  const r = (lastData?.train || []).filter(x => fin(x[key]));
  if (r.length < win + 2) return null;
  const a = mean(r.slice(-win - 2, -win).map(x => x[key])), b = mean(r.slice(-2).map(x => x[key])), d = b - a;
  const span = `over the last ${fI(r.at(-1).step - r.at(-win - 1).step)} steps`;
  if (d < -flat) return ST("g", `${what}falling ${sgn(d)} ${span}`);
  if (d < bad) return ST("a", `${what}flat (${sgn(d)}) ${span}; normal for short stretches`);
  if (contGrace()) return ST("a", `${what}rising ${sgn(d)} ${span}: expected while the learning rate re-warms after the continuation start`);
  return ST("r", `${what}rising ${sgn(d)} ${span}`);
}
// an hourly evaluation series that should fall
function evalFallSt(fn, { flat = 0.003, tol = 0.03, name = "" } = {}) {
  const ev = (lastData?.evals || []).filter(e => fin(fn(e)));
  if (ev.length < 2) return null;
  const a = fn(ev.at(-2)), b = fn(ev.at(-1)), d = b - a, best = Math.min(...ev.map(fn));
  if (d < -flat) return ST("g", `${name}down ${sgn(d)} since the previous evaluation (${f3(a)} → ${f3(b)})`);
  if (d < flat) return ST("a", `${name}flat since the previous evaluation (${sgn(d)})`);
  if (contGrace()) return ST("a", `${name}up ${sgn(d)}: expected while the learning rate re-warms after the continuation start`);
  return b - best < tol ? ST("a", `${name}up ${sgn(d)} since the previous evaluation, still near its best`) : ST("r", `${name}rising: ${sgn(d)} and ${f3(b - best)} above its best`);
}
// relative drift of an evaluation series over the last `k` evaluations (should stay stable)
function driftSt(fn, { k = 3, ok = 0.1, warn = 0.3, name = "value", abs = false } = {}) {
  const ev = (lastData?.evals || []).filter(e => fin(fn(e)));
  if (ev.length < 2) return null;
  const a = fn(ev.at(-Math.min(k + 1, ev.length))), b = fn(ev.at(-1)), d = abs ? Math.abs(b - a) : Math.abs(b - a) / Math.max(Math.abs(a), 1e-9);
  const txt = abs ? `moved ${d.toFixed(3)}` : `moved ${(100 * d).toFixed(1)} %`;
  return ST(d < ok ? "g" : d < warn ? "a" : "r", `${name} ${txt} over the last ${Math.min(k, ev.length - 1)} evaluations${d < ok ? " (stable)" : d < warn ? " (drifting)" : " (unstable)"}`);
}
function lrExpected(step) {        // code/train.py::lr_at: main (6e-4, warm-up 300, cosine to 10 % over 4,050) or the continuation (D-041)
  const S = lastData?.status?.schedule;
  if (S && S.lr && step >= S.offset) {
    const k = step - S.offset, s0 = S.warmup_start_frac || 0, mf = S.lr_min_frac ?? 0.1;
    const warm = s0 + (1 - s0) * Math.min(1, (k + 1) / S.warmup), prog = Math.min(k, S.total) / S.total;
    return S.lr * warm * (mf + (1 - mf) * 0.5 * (1 + Math.cos(Math.PI * prog)));
  }
  const tot = 4050, warm = Math.min(1, (step + 1) / 300), prog = Math.min(step, tot) / tot;
  return 6e-4 * warm * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * prog)));
}
function attachStatus(ch) {
  Object.entries(ch).forEach(([k, c]) => { c.wrap.id = "ch-" + k; });
  ch.ntp.o.status = () => trendSt("ntp_ema", { flat: 0.002, bad: 0.05, what: "smoothed NTP " });
  ch.ncp.o.status = () => trendSt("ncp", { flat: 0.003, bad: 0.05 });
  ch.vq.o.status = () => trendSt("vq", { flat: 0.002, bad: 0.05 });
  ch.total.o.status = () => trendSt("loss", { flat: 0.005, bad: 0.08 });
  ch.val.o.status = () => evalFallSt(e => e.val_mean, { name: "mean " });
  ch.bpb.o.status = () => evalFallSt(e => fin(e.bpb_ts) && fin(e.bpb_wk) ? (e.bpb_ts + e.bpb_wk) / 2 : null, { flat: 0.001, tol: 0.01, name: "mean bpb " });
  ch.probe.o.status = () => {
    const e = lastData?.evals?.at(-1); if (!e) return null;
    const r1 = e.probes.lr_recall1?.exact, sw = e.swap_recall1;
    if (!fin(r1)) return null;
    if (sw >= 0.03) return ST("r", `plan swap keeps ${f3(sw)} of recall: information may bypass the concept stream`);
    if (r1 > 0.05) return ST("g", `recall1 ${f3(r1)} vs ${f3(sw)} with the plan swapped: the concept stream carries it`);
    return ST("a", `recall1 ${f3(r1)} is not yet clearly above chance (0.008)`);
  };
  ch.offs.o.status = () => {
    const ev = (lastData?.evals || []).filter(e => e.probes.lr_carry3?.by_offset && e.probes.lr_multi4?.by_offset);
    if (ev.length < 3) return null;
    const m = e => mean([...e.probes.lr_carry3.by_offset, ...e.probes.lr_multi4.by_offset]);
    const a = m(ev.at(-3)), b = m(ev.at(-1)), d = b - a;
    return ST(d >= 0.01 ? "g" : d > -0.03 ? "a" : "r", `average per-token accuracy ${f3(a)} → ${f3(b)} over 2 evaluations (${d >= 0.01 ? "rising" : d > -0.03 ? "flat" : "falling"})`);
  };
  ch.gap.o.status = () => {
    const ev = lastData?.evals || []; if (ev.length < 2) return null;
    const g = e => e.codebook.code_entropy - e.codebook.ncp_ce, a = g(ev.at(-2)), b = g(ev.at(-1));
    if (b <= 0) return ST("r", `gap ${f2(b)} nats: the next concept is not predicted better than by code frequency`);
    return ST(b > 0.2 && b - a > -0.02 ? "g" : "a", `gap ${f2(b)} nats (${sgn(b - a, 2)} since the previous evaluation)`);
  };
  ch.ve.o.status = () => { const v = lastData?.evals?.at(-1)?.codebook.var_explained; return fin(v) ? ST(v >= 0.6 ? "g" : v >= 0.5 ? "a" : "r", `${f3(v)} of concept variation captured (target ≥ 0.6)`) : null; };
  ch.codes.o.status = () => { const r = lastData?.restarts?.at(-1); if (!r) return null; const v = r.codes_used_min_seg;
    return ST(v >= 0.95 ? "g" : v >= 0.8 ? "a" : "r", `${(100 * v).toFixed(0)} % of codes used in the least-used dictionary (target ≥ 95 %)`); };
  ch.dead.o.status = () => { const r = lastData?.restarts?.at(-1); if (!r) return null; const v = r.restart_dead;
    return ST(v <= 30 ? "g" : v <= 300 ? "a" : "r", `${fI(v)} of 6,144 codes re-seeded at step ${fI(r.step)}${v <= 30 ? " (codebook fully alive)" : ""}`); };
  ch.ppl.o.status = () => { const ev = lastData?.evals || []; if (!ev.length) return null; const ps = ev.map(e => e.codebook.perplexity_mean), v = ps.at(-1), mx = Math.max(...ps);
    return ST(v >= 96 && v >= 0.95 * mx ? "g" : v >= 64 ? "a" : "r", `${f2(v)} of 128 codes in effective use${v >= 0.95 * mx ? "" : ` (below its peak ${f2(mx)})`}`); };
  ch.mix.o.status = () => {
    const ev = (lastData?.evals || []).filter(e => e.crc_mix); if (ev.length < 2) return null;
    const a = ev.at(-Math.min(4, ev.length)).crc_mix, b = ev.at(-1).crc_mix, d = Math.max(...b.map((x, i) => Math.abs(x - a[i])));
    return ST(d < 0.05 ? "g" : d < 0.15 ? "a" : "r", `largest weight change ${d.toFixed(3)} over the last ${Math.min(3, ev.length - 1)} evaluations (${d < 0.05 ? "stable" : "shifting"})`);
  };
  ch.lr.o.status = () => {
    const r = lastData?.train?.at(-1); if (!r || !fin(r.lr)) return null;
    const x = lrExpected(r.step), q = r.lr / x;
    if (Math.abs(q - 1) < 0.03) return ST("g", `${r.lr.toExponential(2)} at step ${fI(r.step)}, on the planned cosine`);
    if (Math.abs(q - 0.5) < 0.03) return ST("a", `half the planned rate: the divergence guard reduced it`);
    return ST("a", `${r.lr.toExponential(2)} vs planned ${x.toExponential(2)}`);
  };
  ch.gn.o.status = () => {
    const r = (lastData?.train || []).filter(x => fin(x.gn_preclip)); if (!r.length) return null;
    const v = r.at(-1).gn_preclip, mx = Math.max(...r.slice(-12).map(x => x.gn_preclip)), gd = lastData.guards.length;
    if (gd) return ST("a", `${gd} safety-guard event(s); latest norm ${f2(v)}`);
    return ST(v < 0.8 && mx < 1 ? "g" : v < 1 ? "a" : "r", `${f2(v)} now, peak ${f2(mx)} over the last 600 steps (clip at 1.0)`);
  };
  ch.tps.o.status = () => { const v = lastData?.train?.at(-1)?.tok_s; if (!fin(v)) return null;
    return ST(v >= 1700 ? "g" : v >= 1300 ? "a" : "r", `${fI(v)} tok/s over the last 50 steps${v >= 1700 ? "" : v >= 1300 ? "; slower than usual (≥ 1,700), check other apps using the GPU" : "; far below normal"}`); };
  ch.mem.o.status = () => {
    const r = (lastData?.train || []).filter(x => fin(x.mps_driver_gib)), sw = lastLive?.swap_gb ?? (r.at(-1)?.swap_mb ?? 0) / 1024;
    if (!r.length) return null;
    const grow = r.at(-1).mps_driver_gib - Math.min(...r.slice(-20).map(x => x.mps_driver_gib));
    if (grow > 1) return ST("a", `GPU memory grew ${f2(grow)} GiB over the last 1,000 steps`);
    return ST(sw < 3 ? "g" : sw < 5 ? "a" : "r", `GPU memory flat at ${f2(r.at(-1).mps_driver_gib)} GiB; swap ${f2(sw)} GB${sw < 3 ? "" : " (high: close heavy apps)"}`);
  };
  ch.eng.o.status = () => { const v = lastData?.evals?.at(-1)?.engram_alpha; if (fin(v) && v < 0.02) return ST("a", `gate nearly closed (${f3(v)}): Engram unused`);
    return driftSt(e => e.engram_alpha, { ok: 0.05, warn: 0.15, abs: true, name: "α" }); };
  ch.tau.o.status = () => {      // learned temperatures may drift (NCP τ is meant to fall); flag only blow-ups or collapse
    const ks = [["enc.layers.3.attn.log_tau", "encoder layer 3"], ["enc.layers.7.attn.log_tau", "encoder layer 7"], ["core.layer.attn.log_tau", "loop layer"], ["core.log_tau", "NCP"]];
    const ev = (lastData?.evals || []).filter(e => e.tau); if (ev.length < 2) return null;
    const e0 = ev.at(-Math.min(4, ev.length)), e1 = ev.at(-1);
    const rows = ks.filter(([k]) => e1.tau[k] && e0.tau[k]).map(([k, n]) => { const a = mean(e0.tau[k]), b = mean(e1.tau[k]); return { n, b, d: Math.abs(b - a) / Math.max(Math.abs(a), 1e-9) }; });
    if (!rows.length) return null;
    const out = rows.find(r => !(Math.abs(r.b) > 0.05 && Math.abs(r.b) < 100));
    if (out) return ST("r", `${out.n} τ out of range (${f2(out.b)})`);
    const w = rows.reduce((m, r) => (r.d > m.d ? r : m));
    return ST(w.d < 0.3 ? "g" : w.d < 0.6 ? "a" : "r", `largest drift: ${w.n} ${(100 * w.d).toFixed(0)} % over the last ${ev.indexOf(e1) - ev.indexOf(e0)} evaluations${w.d < 0.3 ? " (slow learning, no blow-up)" : " (fast change)"}`);
  };
}
function attachLiveStatus(ch) {
  Object.entries(ch).forEach(([k, c]) => { c.wrap.id = "ch-" + k; });
  ch.gpu.o.status = () => {
    if (runState() !== "running") return { lvl: "n", label: runState(), why: "the trainer is not running" };
    const h = (lastLive?.hist || []).slice(-60).map(x => x[1]).filter(fin); if (h.length < 6) return null;
    const m = mean(h);
    return ST(m >= 85 ? "g" : m >= 60 ? "a" : "r", `${m.toFixed(0)} % average over the last ${Math.round(h.length * 5 / 60)} min${m >= 85 ? "" : " (evaluations cause short dips; long dips mean other GPU work)"}`);
  };
  ch.step.o.status = () => {
    const s = (lastData?.steps?.length ? lastData.steps : lastLive?.steps) || [];
    if (s.length < 20) return { lvl: "n", label: "Waiting", why: "per-step values start after the trainer's next restart (heartbeat)" };
    const a = mean(s.slice(-20, -10).map(r => r.ema)), b = mean(s.slice(-10).map(r => r.ema)), d = b - a;
    return ST(d < 0.02 ? "g" : d < 0.1 ? "a" : "r", `smoothed NTP ${sgn(d)} over the last 20 steps`);
  };
}

/* ------------------------------------------------------------------ charts */
function mkCharts() {
  const A = document.getElementById("tr-charts-a"), B = document.getElementById("tr-charts-b"), Cc = document.getElementById("tr-charts-c"), D = document.getElementById("tr-charts-d");
  const ch = {};
  ch.ntp = new LineChart(A, { latestIdx: 1, latestLabel: "smoothed", title: "Next-token loss (training)", measures: "how well the model predicts the next token on training data (nats).", good: "falling; flattening is normal while the learning rate is high.", series: [{ name: "NTP (50-step mean)", color: C.ntp, width: 1.2 }, { name: "NTP (smoothed)", color: C.ema, width: 2.4 }] });
  ch.ncp = new LineChart(A, { title: "Next-concept loss (training)", measures: "how well the concept module predicts the next concept's 48 codes (uniform = 4.85).", good: "falling steadily.", series: [{ name: "NCP", color: C.ncp, width: 2 }] });
  ch.vq = new LineChart(A, { title: "VQ loss (training)", measures: "distance from concepts to their nearest codes.", good: "falling after warm-up.", series: [{ name: "VQ", color: C.vq, width: 2 }] });
  ch.total = new LineChart(A, { title: "Total loss", measures: "NTP + 0.1·NCP + VQ, the quantity being minimised.", good: "falling. Not comparable with other models (VQ has its own scale).", series: [{ name: "total", color: C.total, width: 2 }] });
  ch.val = new LineChart(B, { latestLabel: "mean", title: "Validation loss", measures: "next-token loss on held-out text never trained on (nats per token).", good: "keeps falling. Dashed line = where the short screens ended.", series: [{ name: "mean", color: C.mean, width: 2.6 }, { name: "TinyStories", color: C.ts }, { name: "WikiText-103", color: C.wk }],
    hlines: [{ y: 4.3408, label: "screen S1 @265 (4.341)", color: "#aaa" }] });
  ch.probe = new LineChart(B, { latestLabel: "recall1", title: "Long-range probes (exact match)", measures: "answers hidden 128–420 tokens back, beyond the decoder's reach. Dashed = recall1 with another document's plan.", good: "solid lines rising while the dashed one stays at chance (0.008).", yMin: 0,
    series: [{ name: "recall1", color: C.r1, width: 2.6 }, { name: "recall1 (plan swap)", color: C.swap, dash: true }, { name: "carry3", color: C.c3 }, { name: "multi4", color: C.m4 }, { name: "hop", color: C.hop }, { name: "bind", color: C.bind }],
    hlines: [{ y: 0.5, label: "G-D3: multi4 ≥ 0.5", color: "#c0c0c0" }] });
  ch.offs = new LineChart(B, { latestLabel: "carry3 · 1st", title: "Partial credit on carry3 / multi4", measures: "accuracy of each answer token separately (exact match needs all three).", good: "rising: a sign the full answers will follow.", yMin: 0,
    series: [{ name: "carry3 · 1st", color: "#3b78c2" }, { name: "carry3 · 2nd", color: "#6b9bd6", dash: true }, { name: "carry3 · 3rd", color: "#9fbfe6" }, { name: "multi4 · 1st", color: "#8f7cc0" }, { name: "multi4 · 2nd", color: "#b3a3d9", dash: true }, { name: "multi4 · 3rd", color: "#d0c6ea" }] });
  ch.bpb = new LineChart(B, { latestLabel: "TinyStories", title: "Validation bits per byte", measures: "the same validation loss per byte of text (tokenizer-independent).", good: "falling.", series: [{ name: "TinyStories", color: C.ts }, { name: "WikiText-103", color: C.wk }] });
  ch.gap = new LineChart(Cc, { latestIdx: 2, latestLabel: "gap, nats", title: "Is the next concept predictable? (G-C2)", measures: "NCP loss vs the code-frequency entropy H; the green line is the gap.", good: "gap above 0 and growing.", series: [{ name: "NCP loss (eval)", color: C.ncp, width: 2.2 }, { name: "code entropy H", color: C.H, dash: true }, { name: "gap H − NCP", color: C.gap, width: 2.2 }] });
  ch.ve = new LineChart(Cc, { title: "Codebook fit (G-E2)", measures: "share of concept variation captured by the nearest codes.", good: "above 0.6.", series: [{ name: "variance explained", color: C.vq, width: 2.2 }], hlines: [{ y: 0.6, label: "0.6", color: "#bbb" }] });
  ch.codes = new LineChart(Cc, { latestLabel: "training", title: "Codes in use (G-C1)", measures: "least-used dictionary's share of codes in use.", good: "at or above 95 %.", yMax: 1, fmt: v => (100 * v).toFixed(0) + "%",
    series: [{ name: "since last check (training)", color: C.ncp, width: 2.2 }, { name: "on eval data (min)", color: C.wk, dash: true }, { name: "on eval data (mean)", color: "#aaa" }], hlines: [{ y: 0.95, label: "95 %", color: "#bbb" }] });
  ch.dead = new LineChart(Cc, { title: "Dead codes re-seeded", measures: "codes unused for 100 steps and replaced (of 6,144).", good: "near 0 after the first few hundred steps.", series: [{ name: "re-seeded", color: C.swap, dots: "big" }], yMin: 0, fmt: v => v.toFixed(0) });
  ch.ppl = new LineChart(Cc, { title: "Codebook perplexity", measures: "effective number of codes used per dictionary (of 128).", good: "high and stable.", series: [{ name: "perplexity", color: C.m4, width: 2 }], fmt: v => v.toFixed(1) });
  ch.mix = new LineChart(Cc, { latestIdx: 2, latestLabel: "G weight", title: "What goes into a concept (CRC mix)", measures: "average weight on raw embedding / local / global state.", good: "any stable mix; the global stream usually dominates.", yMin: 0, series: [{ name: "e0 (embedding)", color: "#999" }, { name: "L (local)", color: "#3b78c2" }, { name: "G (global)", color: "#e07b39" }] });
  ch.lr = new LineChart(D, { title: "Learning rate", measures: "step size of the optimizer.", good: "follows the planned warm-up and cosine (stretched to 4,050 steps at step 554).", fmt: v => v.toExponential(1), series: [{ name: "lr", color: C.ntp, width: 2 }] });
  ch.gn = new LineChart(D, { latestLabel: "all", title: "Gradient norm", measures: "size of each update before clipping at 1.0.", good: "steady, well below 1.0.", series: [{ name: "all", color: C.total, width: 2 }, { name: "without codebook", color: C.ntp, dash: true }, { name: "codebook", color: C.vq }], yMin: 0 });
  ch.tps = new LineChart(D, { title: "Throughput", measures: "tokens per second over each 50-step window (evaluations excluded).", good: "≈ 1,700–2,100 on this M2; dips mean other GPU work.", fmt: v => v.toFixed(0), series: [{ name: "tok/s", color: C.ncp, width: 2 }], yMin: 0 });
  ch.mem = new LineChart(D, { latestLabel: "MPS GiB", title: "Memory", measures: "GPU (MPS) driver memory and macOS swap, GiB.", good: "flat; swap below ~3 GB.", series: [{ name: "MPS driver", color: C.m4 }, { name: "MPS allocated", color: C.ts, dash: true }, { name: "swap", color: C.wk }], yMin: 0 });
  ch.eng = new LineChart(D, { title: "Engram gate α", measures: "average share of the looked-up n-gram vector admitted.", good: "stable; the model decides how much memory to use.", series: [{ name: "α mean", color: C.vq, width: 2 }] });
  ch.tau = new LineChart(D, { latestIdx: 2, latestLabel: "loop layer", title: "Attention temperatures τ", measures: "learned sharpness of full-attention heads and NCP segments (NCP ÷ 10).", good: "slow drift, no blow-up.", series: [{ name: "encoder layer 3", color: "#e07b39" }, { name: "encoder layer 7", color: "#f0a36b", dash: true }, { name: "loop layer", color: C.ncp }, { name: "NCP τ / 10", color: C.m4 }] });
  attachStatus(ch);
  return ch;
}
function mkLiveCharts() {
  const P = document.getElementById("tr-charts-live"), ch = {};
  ch.gpu = new LineChart(P, { latestLabel: "GPU %", title: "GPU load, last hour (live)", measures: "how busy the M2 GPU is, sampled every 5 s; grey = trainer CPU.", good: "≈ 95–100 %; short dips are evaluations and checkpoints.", xLabel: "min", yMin: 0, yMax: 100, fmt: v => v.toFixed(0), series: [{ name: "GPU %", color: C.ncp, width: 1.6 }, { name: "trainer CPU %", color: "#aaa" }] });
  ch.step = new LineChart(P, { latestIdx: 1, latestLabel: "NTP smoothed", title: "Losses every step (live)", measures: "NTP, NCP and VQ of each optimizer step, from the trainer's heartbeat.", good: "noisy but trending down.", empty: "starts after the trainer's next restart (heartbeat)",
    series: [{ name: "NTP", color: C.ntp, width: 1 }, { name: "NTP (smoothed)", color: C.ema, width: 2.2 }, { name: "NCP", color: C.ncp, width: 1 }, { name: "VQ", color: C.vq, width: 1 }] });
  attachLiveStatus(ch);
  return ch;
}
function renderLiveCharts() {
  if (!liveCharts) liveCharts = mkLiveCharts();
  const L = lastLive; if (!L) return;
  const now = L.server_time;
  liveCharts.gpu.set([L.hist.map(h => [+((h[0] - now) / 60).toFixed(2), h[1]]), L.hist.map(h => [+((h[0] - now) / 60).toFixed(2), h[2]])]);
  const s = (lastData?.steps?.length ? lastData.steps : L.steps) || [];
  liveCharts.step.set([s.map(r => [r.step, r.ntp]), s.map(r => [r.step, r.ema]), s.map(r => [r.step, r.ncp]), s.map(r => [r.step, r.vq])]);
  document.getElementById("steps-note").textContent = s.length ? `${s.length} steps from the trainer's heartbeat` : "GPU load is live now; per-step losses begin after the trainer's next restart";
}

function renderTrain() {
  const d = lastData; if (!d) return;
  if (!charts) charts = mkCharts();
  const ev = d.evals, tr = d.train, last = ev.at(-1);
  const P = k => tr.map(r => [r.step, r[k]]), E = fn => ev.map(e => [e.step, fn(e)]);
  const contV = (d.events || []).filter(e => e.kind === "cont").map(e => ({ x: e.step, label: "continuation", color: "#2e9c8c" }));
  const evs = (d.events || []).filter(e => e.kind === "resume").map(e => ({ x: e.step, label: "", color: "#ccc" })).concat(contV);
  ["total", "ntp", "ncp", "vq", "tps", "gn", "mem"].forEach(k => { charts[k].o.vlines = evs; });
  ["val", "probe", "offs", "bpb", "gap", "ve", "codes", "ppl", "mix", "eng", "tau"].forEach(k => { charts[k].o.vlines = contV; });
  charts.lr.o.vlines = [{ x: 554, label: "schedule → 4,050", color: "#c7811a" }].concat(contV);
  charts.ntp.set([P("ntp"), P("ntp_ema")]); charts.ncp.set([P("ncp")]); charts.vq.set([P("vq")]); charts.total.set([P("loss")]);
  charts.val.set([E(e => e.val_mean), E(e => e.val_ts), E(e => e.val_wk)]);
  charts.probe.set([E(e => e.probes.lr_recall1?.exact), E(e => e.swap_recall1), E(e => e.probes.lr_carry3?.exact), E(e => e.probes.lr_multi4?.exact), E(e => e.probes.lr_hop?.exact), E(e => e.probes.lr_bind?.exact)]);
  charts.offs.set([0, 1, 2].map(i => E(e => e.probes.lr_carry3?.by_offset?.[i])).concat([0, 1, 2].map(i => E(e => e.probes.lr_multi4?.by_offset?.[i]))));
  charts.bpb.set([E(e => e.bpb_ts), E(e => e.bpb_wk)]);
  charts.gap.set([E(e => e.codebook.ncp_ce), E(e => e.codebook.code_entropy), E(e => e.codebook.code_entropy - e.codebook.ncp_ce)]);
  charts.ve.set([E(e => e.codebook.var_explained)]);
  charts.codes.set([d.restarts.map(r => [r.step, r.codes_used_min_seg]), E(e => e.codebook.codes_used_frac_min), E(e => e.codebook.codes_used_frac_mean)]);
  charts.dead.set([d.restarts.map(r => [r.step, r.restart_dead])]);
  charts.ppl.set([E(e => e.codebook.perplexity_mean)]);
  charts.mix.set([0, 1, 2].map(i => E(e => e.crc_mix?.[i])));
  charts.lr.set([P("lr")]);
  charts.gn.set([P("gn_preclip"), P("gn_without_codebook"), P("gn_codebook")]);
  charts.tps.set([P("tok_s")]);
  charts.mem.set([P("mps_driver_gib"), P("mps_alloc_gib"), tr.map(r => [r.step, r.swap_mb != null && r.swap_mb >= 0 ? r.swap_mb / 1024 : null])]);
  charts.eng.set([E(e => e.engram_alpha)]);
  const tk = (e, k) => e.tau ? mean(e.tau[k]) : null;
  charts.tau.set([E(e => tk(e, "enc.layers.3.attn.log_tau")), E(e => tk(e, "enc.layers.7.attn.log_tau")), E(e => tk(e, "core.layer.attn.log_tau")), E(e => { const v = tk(e, "core.log_tau"); return v == null ? null : v / 10; })]);

  const head = ["when", "step", "tokens", "val loss", "Δ", "TinyStories", "WikiText", "recall1", "swap", "carry3", "multi4", "hop", "bind", "codes min", "VE", "NCP gap", "eval s"];
  document.getElementById("tr-evals").innerHTML = `<tr>${head.map(h => `<th>${h}</th>`).join("")}</tr>` + ev.slice().reverse().map((e, i, arr) => {
    const p = arr[i + 1], dv = p ? e.val_mean - p.val_mean : null;
    return `<tr class="${e.is_best ? "best" : ""}"><td>${(e.time || "").slice(5, 16)} <span class="muted">${e.run === "cont" ? "cont · " : ""}${e.tag}</span></td><td>${e.step}</td><td>${(e.tokens / 1e6).toFixed(1)}M</td>
    <td><b>${f4(e.val_mean)}</b></td><td class="${dv == null ? "" : dv < 0 ? "delta-good" : "delta-bad"}">${dv == null ? "—" : (dv > 0 ? "+" : "") + dv.toFixed(3)}</td><td>${f3(e.val_ts)}</td><td>${f3(e.val_wk)}</td>
    ${PROBES.map((pp, j) => (j === 0 ? `<td><b>${f3(e.probes[pp]?.exact)}</b></td><td>${f3(e.swap_recall1)}</td>` : `<td>${f3(e.probes[pp]?.exact)}</td>`)).join("")}
    <td>${f2(e.codebook.codes_used_frac_min)}</td><td>${f3(e.codebook.var_explained)}</td><td>${f2(e.codebook.code_entropy - e.codebook.ncp_ce)}</td><td>${e.eval_sec ? e.eval_sec.toFixed(0) : "—"}</td></tr>`;
  }).join("");
  const sys = d.system;
  document.getElementById("tr-ckpts").innerHTML = `<tr><th>file</th><th>saved</th><th>GiB</th><th>hard links</th></tr>` +
    sys.checkpoints.map(c => `<tr><td>${c.name}</td><td>${c.mtime.slice(5, 16)}</td><td>${c.size_gb.toFixed(2)}</td><td>${c.links}</td></tr>`).join("") +
    `<tr><td colspan="4" class="muted" style="text-align:left;white-space:normal">ckpt_last and ckpt_best are hard links to a step file (same data, no extra disk). Training resumes from ckpt_last.</td></tr>`;
  document.getElementById("tr-guards").innerHTML = d.guards.length ? d.guards.map(g => `<div class="mono">${g.time} ${g.kind} step ${g.step}</div>`).join("")
    : "None fired. The divergence guard (NaN or loss × 3) would reload the last checkpoint at half the learning rate; the memory guard would halve the micro-batch.";
  if (last) {
    const tau = last.tau || {}, ncpT = tau["core.log_tau"] || [];
    document.getElementById("tr-internals").innerHTML = `<div class="kvt">
      <span>concept ingredients (e0 / L / G)</span><span>${(last.crc_mix || []).map(x => x.toFixed(3)).join(" / ")}</span>
      <span>Engram gate α (mean)</span><span>${f3(last.engram_alpha)} · ‖W_V‖ = ${f2(last.engram_wv_norm)}</span>
      <span>τ encoder layer 3 (6 heads)</span><span class="mono">${(tau["enc.layers.3.attn.log_tau"] || []).map(x => x.toFixed(2)).join(" ")}</span>
      <span>τ encoder layer 7 (6 heads)</span><span class="mono">${(tau["enc.layers.7.attn.log_tau"] || []).map(x => x.toFixed(2)).join(" ")}</span>
      <span>τ loop layer (6 heads)</span><span class="mono">${(tau["core.layer.attn.log_tau"] || []).map(x => x.toFixed(2)).join(" ")}</span>
      <span>τ NCP (48 segments)</span><span>${ncpT.length ? `min ${f2(Math.min(...ncpT))} · mean ${f2(mean(ncpT))} · max ${f2(Math.max(...ncpT))} (started at 16)` : "—"}</span>
      <span>codebook perplexity</span><span>${f2(last.codebook.perplexity_mean)} of 128 (${last.codebook.n_concepts} eval concepts)</span></div>
      <div class="muted" style="font-size:12px;margin-top:8px">from the evaluation at step ${last.step} (${last.time})</div>`;
  }
  const pre = document.getElementById("tr-stdout"); pre.textContent = d.stdout.join("\n"); pre.scrollTop = pre.scrollHeight;
  document.getElementById("tr-chain").textContent = d.chain.join("\n") + "\n\n— launcher —\n" + d.launcher.join("\n");
}

function renderCrcLive() {
  const last = lastData?.evals?.at(-1), live = document.getElementById("crc-live");
  if (!live || !last || !last.crc_mix) return;
  live.innerHTML = "";
  [["e0", "#999"], ["L", COL.L], ["G", COL.G]].forEach(([n, c], i) => {
    const v = last.crc_mix[i];
    el("rect", { x: 1022, y: 104 + i * 26, width: 130 * v, height: 16, rx: 3, fill: c, opacity: 0.8 }, live);
    txt(live, 1016, 112 + i * 26, n, { anchor: "end", size: 11.5, color: c, weight: 700 });
    txt(live, 1026 + 130 * v, 112 + i * 26, v.toFixed(3), { anchor: "start", size: 11 });
  });
  note(live, 1000, 196, `step ${last.step}`, { size: 10.5 });
}

/* ------------------------------------------------------------------ screens */
let scCharts = null;
function renderScreens() {
  const d = lastData; if (!d) return;
  const S = d.screens, ev = d.evals, m265 = ev.find(e => e.tag === "step265");
  const s1v = S.S1 ? S.S1.val_mean : null;
  document.getElementById("sc-explain").innerHTML = [
    ["S1", "#3b78c2", "design", "Confined decoder", "The design as specified: decoder positions never read each other's plans; loop depth R ~ 1–4.", S.S1],
    ["S2", "#e07b39", "control", "Stack decoder", "One mixed decoder stream, so plans leak between positions (reach 103). Tests whether confinement costs language quality.", S.S2],
    ["S3", "#2e9c8c", "control", "No recurrence (R = 1)", "The concept module runs a single pass. Tests whether looping helps at this early stage.", S.S3],
  ].map(([id, col, role, t, x, e]) => `<div class="card arm"><div class="arm-top"><span class="arm-id" style="background:${col}">${id}</span><span class="badge">${role}</span></div>
    <h4>${t}</h4><p>${x}</p>${e ? `<div class="arm-v"><b>${f3(e.val_mean)}</b><span>val nats @ step ${e.step}</span>${s1v != null && id !== "S1" ? `<span class="badge">${(e.val_mean - s1v >= 0 ? "+" : "") + (e.val_mean - s1v).toFixed(4)} vs S1</span>` : ""}</div>` : ""}</div>`).join("");
  document.getElementById("sc-verdict").innerHTML = `<div class="verdict-row"><div class="vi">i</div><div><b>Verdict:</b> the three arms are within 0.016 nats of each other at 8.7M tokens.
    Confinement costs nothing measurable (screen gate PASS, S1 − S2 = −0.011), and recurrence neither helps nor hurts yet. The main run at step 265 is not comparable because it was still in learning-rate warm-up.</div></div>`;
  const rows = [["S1", "confined", S.S1], ["S2", "stack decoder", S.S2], ["S3", "R = 1", S.S3], ["main @265", "main run, still warming up", m265]];
  const head = ["arm", "what differs", "step", "TinyStories", "WikiText", "val mean", "Δ vs S1", ...PROBES.map(p => p.slice(3)), "swap r1", "codes used", "VE", "NCP", "H"];
  const s1 = S.S1 ? S.S1.val_mean : null;
  document.getElementById("sc-table").innerHTML = `<tr>${head.map(h => `<th>${h}</th>`).join("")}</tr>` + rows.filter(r => r[2]).map(([n, w, e]) => `<tr>
    <td><b>${n}</b></td><td style="text-align:left">${w}</td><td>${e.step}</td><td>${f4(e.val_ts)}</td><td>${f4(e.val_wk)}</td><td><b>${f4(e.val_mean)}</b></td>
    <td>${s1 != null && n !== "S1" ? (e.val_mean - s1 >= 0 ? "+" : "") + (e.val_mean - s1).toFixed(4) : "—"}</td>
    ${PROBES.map(p => `<td>${f3(e.probes[p]?.exact)}</td>`).join("")}<td>${f3(e.swap_recall1)}</td><td>${e.codes_min_seg_restart != null ? f3(e.codes_min_seg_restart) : "—"}</td>
    <td>${f3(e.codebook.var_explained)}</td><td>${f3(e.codebook.ncp_ce)}</td><td>${f3(e.codebook.code_entropy)}</td></tr>`).join("") +
    "";
  if (!scCharts) {
    const p = document.getElementById("sc-charts");
    const ser = [{ name: "S1 confined", color: "#3b78c2" }, { name: "S2 stack", color: "#e07b39" }, { name: "S3 R = 1", color: "#2e9c8c" }, { name: "main (first 300)", color: "#999", dash: true }];
    const done = () => ({ lvl: "n", label: "Completed", why: "the screen runs finished at step 265; kept for reference" });
    scCharts = { ntp: new LineChart(p, { title: "NTP during the screens", series: ser, status: done }), ncp: new LineChart(p, { title: "NCP during the screens", series: ser, status: done }), vq: new LineChart(p, { title: "VQ during the screens", series: ser, status: done }) };
  }
  const main300 = d.train.filter(r => r.step <= 300);
  for (const k of ["ntp", "ncp", "vq"]) scCharts[k].set(["S1", "S2", "S3"].map(s => (S[s]?.curve || []).map(r => [r.step, r[k]])).concat([main300.map(r => [r.step, r[k]])]));
  const last = ev.at(-1), rs = d.restarts.at(-1);
  const b = (ok, t) => ok == null ? `<span class="badge">${t || "pending"}</span>` : ok ? `<span class="badge pass">${t || "PASS"}</span>` : `<span class="badge fail">${t || "not yet"}</span>`;
  const g = [
    ["G-E0 / G-D0", "all exact tests pass", "T1–T9, D1–D5, P1–P5 at init (runs/tests/); P5 on trained weights runs after training", b(true)],
    ["screen gate", "confined within ±0.02 nats of stack at equal steps", S.S1 && S.S2 ? `S1 − S2 = ${(S.S1.val_mean - S.S2.val_mean).toFixed(4)}` : "—", S.S1 && S.S2 ? b(Math.abs(S.S1.val_mean - S.S2.val_mean) <= 0.02) : b(null)],
    ["G-E2", "codebook variance explained ≥ 0.6 (report)", last ? `VE ${f3(last.codebook.var_explained)} @ step ${last.step}` : "—", last ? b(last.codebook.var_explained >= 0.6, last.codebook.var_explained >= 0.6 ? "PASS (live)" : "below (live)") : b(null)],
    ["G-C1", "≥ 95 % of codes used in every segment", rs ? `${f3(rs.codes_used_min_seg)} since last check @ ${rs.step}; eval set ${f2(last?.codebook.codes_used_frac_min)}` : "—", rs ? b(rs.codes_used_min_seg >= 0.95, rs.codes_used_min_seg >= 0.95 ? "PASS (live)" : "below (live)") : b(null)],
    ["G-C2", "NCP loss below the code-frequency entropy", last ? `${f3(last.codebook.ncp_ce)} < ${f3(last.codebook.code_entropy)}` : "—", last ? b(last.codebook.ncp_ce < last.codebook.code_entropy, last.codebook.ncp_ce < last.codebook.code_entropy ? "PASS (live)" : "not yet") : b(null)],
    ["G-E3 / G-D2", "plan swap drops every long-range probe to chance; swap cost ~ gain beyond 70 tokens", last ? `recall1 ${f3(last.probes.lr_recall1?.exact)} → swap ${f3(last.swap_recall1)}; truncation curve at the final evaluation` : "—", b(null, "final eval")],
    ["G-C3", "R = 4 beats R = 1 on lr_hop", last ? `lr_hop ${f3(last.probes.lr_hop?.exact)} (R = 4)` : "—", b(null, "final eval")],
    ["G-D3", "lr_multi4 exact ≥ 0.5", last ? `multi4 ${f3(last.probes.lr_multi4?.exact)}` : "—", last ? b(last.probes.lr_multi4?.exact >= 0.5 ? true : null, last.probes.lr_multi4?.exact >= 0.5 ? "PASS (live)" : "not reached yet") : b(null)],
    ["G-E1", "linear probe recovers each concept's 4 tokens ≥ 99 %", "final evaluation (probes part)", b(null, "final eval")],
  ];
  const cls = st => /PASS/.test(st) ? "pass" : /not yet|not reached|below/.test(st) ? "fail" : "pend";
  document.getElementById("sc-gates").innerHTML = g.map(r => `<div class="gate ${cls(r[3])}"><div class="gate-top"><span class="gate-id">${r[0]}</span>${r[3]}</div>
    <div class="gate-c">${r[1]}</div><div class="gate-e">${r[2]}</div></div>`).join("");
  const n = { pass: 0, fail: 0, pend: 0 }; g.forEach(r => n[cls(r[3])]++);
  document.getElementById("sc-gsum").innerHTML = `<span class="cc g"><i></i>${n.pass} passing</span>${n.fail ? `<span class="cc a"><i></i>${n.fail} not yet</span>` : ""}<span class="cc"><i style="background:var(--ink3)"></i>${n.pend} at final evaluation</span>`;
}

/* ------------------------------------------------------------------ overview + guide */
function renderOverviewLive() {
  const o = document.getElementById("ov-live"); if (!o || !lastData) return;
  const ev = lastData.evals, last = ev.at(-1), cs = currentStep(), s = runState(), tr = lastData.train.at(-1), total = lastData.status?.total_steps || 4050;
  const gapv = last ? last.codebook.code_entropy - last.codebook.ncp_ce : null;
  const H = healthItems(), nb = { g: 0, a: 0, r: 0 }; H.forEach(i => { if (nb[i.lvl] != null) nb[i.lvl]++; });
  const wl = nb.r ? "r" : nb.a ? "a" : "g";
  const ebd = document.getElementById("ov-eb-dot"), ebs = document.getElementById("ov-eb-state");
  if (ebd) ebd.className = "dot " + s;
  if (ebs) ebs.textContent = { running: "training live on an Apple M2", paused: "training paused (Apple M2)", finished: "trained on an Apple M2", stopped: "training stopped (Apple M2)" }[s] || "Apple M2";
  o.innerHTML = `<div class="hl-top"><span class="hl-kicker">${s === "finished" ? "Training complete" : "Training right now"}</span><span class="state-chip s-${s}"><span class="dot ${s}"></span>${s}</span></div>
    <div class="hl-step">${cs ? (cs.exact ? "" : "≈ ") + fI(cs.step) : "—"} <small>/ ${fI(total)} steps</small></div>
    <div class="progress"><div style="width:${cs ? 100 * cs.step / total : 0}%"></div></div>
    <div class="hl-grid">
      <div class="hl-m"><div class="l">validation loss</div><div class="v">${f3(last?.val_mean)}</div><div class="s">nats · step ${last?.step ?? "—"}</div></div>
      <div class="hl-m"><div class="l">recall via concepts</div><div class="v">${f3(last?.probes?.lr_recall1?.exact)}</div><div class="s">plan swap ${f3(last?.swap_recall1)}</div></div>
      <div class="hl-m"><div class="l">concept gap H − NCP</div><div class="v">${gapv == null ? "—" : f2(gapv)}</div><div class="s">nats of predictability</div></div>
      <div class="hl-m"><div class="l">training NTP</div><div class="v">${f3(tr?.ntp_ema)}</div><div class="s">GPU ${lastLive?.gpu?.device ?? "—"} % busy</div></div>
    </div>
    ${(() => { const P = planInfo(); return P ? `<div class="hl-plan"><span class="stg-n sm">${P.stage}</span>Stage ${P.stage} of 2 · ${P.stage === 1 ? `main run ${P.s1}; continuation (+24 h) ${P.s2 === "armed" ? "armed, starts after the final evaluation" : P.s2}` : `continuation ${P.s2}`}</div>` : ""; })()}
    <div class="hl-verdict ${wl}">${wl === "g" ? "✓" : "!"} ${wl === "g" ? `All ${H.length} health checks pass` : `${nb.r ? nb.r + " problem" + (nb.r > 1 ? "s" : "") : nb.a + " to watch"}`}
      <a href="#train" data-tab-link="train">Live training →</a></div>`;
  const r1 = last?.probes?.lr_recall1?.exact, sw = last?.swap_recall1, proof = document.getElementById("ov-proof");
  if (proof && last) {
    const bar = (lbl, v, col) => `<div class="ev-row"><div class="ev-l"><span>${lbl}</span><b>${f3(v)}</b></div>
      <div class="ev-track"><div style="width:${Math.max(0.6, 100 * (v || 0))}%;background:${col}"></div><span class="chance" style="left:0.8%" title="chance ≈ 0.008"></span></div></div>`;
    proof.innerHTML = `<div class="ev-bars">${bar("recall1, normal plan", r1, "linear-gradient(90deg,#2e9c8c,#3fbf9f)")}${bar("recall1, plan swapped", sw, "#d0667a")}
      <div class="ev-note">${r1 > 0.05 && sw < 0.03 ? `<b class="good">The concept stream carries the answer.</b> ` : ""}Evaluation at step ${fI(last.step)} · chance ≈ 0.008 · 256 items.</div></div>`;
  }
  renderHealth("ov-health");
}
function renderTimeline() {
  const el = document.getElementById("gd-timeline"); if (!el || !lastData) return;
  const items = [];
  lastData.chain.forEach(l => { const m = l.match(/^(\S+ \S+) (.*)$/); if (m && !/DRY=1|battery guard started/.test(m[2])) items.push([m[1], m[2]]); });
  (lastData.events || []).forEach(e => items.push([e.time, e.kind === "resume" ? `trainer resumed at step ${e.step}` : e.kind === "cont" ? `continuation started at step ${fI(e.step)}` : e.kind === "stop:pause" ? `paused at step ${e.step}` : `${e.kind} at step ${e.step}`]));
  lastData.evals.forEach(e => items.push([e.time, `evaluation (${e.tag}) at step ${e.step}: validation ${f3(e.val_mean)}, recall1 ${f3(e.probes.lr_recall1?.exact)}`]));
  items.sort((a, b) => a[0] < b[0] ? -1 : 1);
  const seen = new Set();
  const kind = x => /^evaluation/.test(x) ? "ev" : /pause|stop|guard|fail/i.test(x) ? "ctl" : "";
  el.innerHTML = `<ul class="tl-list">${items.filter(([t, x]) => { const k = t + x; if (seen.has(k)) return false; seen.add(k); return true; }).map(([t, x]) => `<li class="${kind(x)}"><span>${t.slice(5, 16)}</span><span>${x}</span></li>`).join("")}</ul>`;
  const ul = el.querySelector(".tl-list"); if (ul) ul.scrollTop = ul.scrollHeight;
}

/* ------------------------------------------------------------------ polling */
async function pollLive() {
  try {
    const L = await (await fetch("api/live", { cache: "no-store" })).json();
    lastLive = L; liveAt = Date.now();
    const sig = JSON.stringify(L.sig);
    if (sig !== lastSig || !lastData) { lastSig = sig; await pollTelemetry(); }
    else refreshVisible(false);
  } catch (e) {
    document.getElementById("live-text").textContent = "dashboard server not reachable";
    document.getElementById("live-dot").className = "dot stopped";
  }
}
async function pollTelemetry() {
  try { lastData = await (await fetch("api/telemetry", { cache: "no-store" })).json(); refreshVisible(true); } catch (e) {}
}
function refreshVisible(full) {
  setPill(); renderCrcLive();
  const on = id => document.getElementById("tab-" + id).classList.contains("on");
  if (on("overview")) renderOverviewLive();
  if (on("train")) { renderLiveCharts(); if (full || !charts) renderTrain(); else Object.values(charts).forEach(c => c.updStatus()); renderNow(); renderKpis(); renderHealth("tr-health"); renderNav(); }
  if (on("screens") && (full || !scCharts)) renderScreens();
  if (on("guide")) renderTimeline();
}
function renderNowTick() {   // cheap 1 s update of the ticking step only
  const cs = currentStep(), e = document.querySelector("#now .stepbig");
  if (e && cs) e.innerHTML = `${cs.exact ? "" : "≈ "}${fI(cs.step)} <small>/ ${fI(lastData?.status?.total_steps || 4050)} steps</small>`;
}
function startPolling() {
  pollLive();
  setInterval(() => { if (!document.hidden || Date.now() - liveAt > 60e3) pollLive(); }, 5000);
  setInterval(() => { setPill(); if (document.getElementById("tab-train").classList.contains("on")) renderNowTick(); }, 1000);
  setInterval(pollTelemetry, 120000);
}
function onTabShown(id) { if (lastData) refreshVisible(true); }
