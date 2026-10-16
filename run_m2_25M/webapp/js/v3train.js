/* v3 live training page — built from the same components as v1's Live training page (js/telemetry.js, js/charts.js):
   hero + verdict, KPI tiles, section chips, health cards with "bad would be", LineChart cards with a status pill.
   Data (read-only):
     data/v3_live.json     written by run_v3_19M/code/train.py every 25 steps and after each evaluation
     data/ref_v1v2.json    v1 / v2 evaluations and training curves (same data order: step k = same tokens)
     :8766/api/live        optional; the v3-side server (per-step heartbeat, trainer process, GPU %, battery)
   Polls every 30 s while the tab is open and the window is visible; charts redraw only when new data arrives. */
(function () {
  const LIVE_API = /^(localhost|127\.0\.0\.1)$/.test(location.hostname) ? "http://localhost:8766/api/live" : "/v3api/live";   // phone: via webapp/phone_server.py
  const TOK_PER_STEP = 32768, SLOTS_EPOCH = 280.4e6;
  const REMOTE = !/^(localhost|127\.0\.0\.1)$/.test(location.hostname);   // opened through the phone gateway
  const C3 = "#2e9c8c", C3b = "#168b76", C1 = "#3b78c2", C2 = "#7a5bd0", CW = "#e07b39", CT = "#c2410c", CY = "#cfa935", CG = "#8a93a3";
  const PAIRC = { t01: "#168b76", t11: "#2e9c8c", t21: "#7cc7b8", t02: "#7a5bd0", t12: "#b39ddb", t03: "#e07b39" };
  const PAIRN = { t01: "(0, ⅓)", t11: "(⅓, ⅓)", t21: "(⅔, ⅓)", t02: "(0, ⅔)", t12: "(⅓, ⅔)", t03: "(0, 1)" };
  const V1_STEPS = [185, 265, 365, 554, 766, 1009, 1217, 1402, 1593, 1792, 1992, 2190, 2376, 2559, 2764, 2944, 3121, 3304, 3484, 3669, 3853, 4039, 4050];
  const GATE_STEPS = [1217, 1792, 2376];
  const RANK = { r: 3, a: 2, g: 1, n: 0 };

  let REF = null, D = null, L = null, CH = null, timer = null, keyD = null, keyL = null, keyR = null;
  let R = { steps: [], restarts: [], opened: [], evals: [] };        // run logs: data/v3_train_log.jsonl, data/v3_evals.jsonl (symlinks, read-only)
  let MS = null, MSsrc = "";                                           // milestone / final evaluation (data/v3_eval_final.json, data/v3_eval_24h.json; symlinks, 404 until written)
  const fin = v => v != null && isFinite(v);
  const f = (v, d = 3) => fin(v) ? (+v).toFixed(d) : "—";
  const fI = v => fin(v) ? Math.round(v).toLocaleString("en-US") : "—";
  const sgn = (v, d = 3) => !fin(v) ? "—" : (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d);
  const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
  const median = a => { const s = a.filter(fin).sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : null; };
  const clock = ms => new Date(ms).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
  const day = ms => new Date(ms).toLocaleString("en-GB", { weekday: "short", hour: "2-digit", minute: "2-digit" });
  const fmtH = h => { if (!fin(h)) return "—"; const hh = Math.floor(h), mm = Math.round((h - hh) * 60); return hh ? `${hh} h ${String(mm).padStart(2, "0")} min` : `${mm} min`; };
  const ST = (lvl, why, label) => ({ lvl, why, label });
  const interp = (rows, step, key) => LC.interp(rows, step, key);
  const ser = k => (D?.history?.[k] || []).filter(p => fin(p[1]));
  const evs = () => (D?.evals || []);
  const evPts = (key, src) => (src || []).filter(e => fin(e[key])).map(e => [e.step, e[key]]);
  const lastOf = k => ser(k).at(-1)?.[1];

  /* ------------------------------------------------------------------ GPU samples
     macOS's GPU counter (ioreg "Device Utilization %") sometimes reads exactly 0 between ~95 % samples, e.g. while the
     display sleeps, even though the trainer keeps stepping at full speed. While the trainer is stepping, 0 readings are
     treated as missing; throughput is the ground truth for "is the GPU working". */
  const stepping = () => { const hb = L?.heartbeat; return !!(L?.proc && hb && Date.now() / 1000 - hb.t < 120); };
  const gpuSamples = () => (L?.hist || []).map(h => [h[0], fin(h[1]) && !(h[1] === 0 && stepping()) ? h[1] : null]);
  const gpuLatest = () => { const g = gpuSamples().filter(p => p[1] != null && p[0] > (L?.server_time || L?.t || 0) - 120); return g.length ? g.at(-1)[1] : null; };

  /* ------------------------------------------------------------------ run state */
  function state() {
    const now = Date.now() / 1000, hb = L?.heartbeat, H = D.hours || 48, wh = Math.max(D.wall_h || 0, hb?.wall_h || 0);   // heartbeat wall_h is per step
    const tLast = Math.max(D.t || 0, hb?.t || 0), age = tLast ? (now - tLast) / 60 : null;
    const step = hb?.step != null && hb.step >= (D.step || 0) ? hb.step : D.step || 0;
    const inMilestone = D.milestone_hours && !D.milestone_done && wh >= D.milestone_hours - 0.1;   // train.py stops at 24 h, evaluates (≈ 2 h), exits 5, relaunches
    const proc = L ? L.proc : undefined;                                  // undefined = the v3 server is not reachable
    let s = "running", label = "training";
    if (wh >= H - 0.01) { s = proc || (age != null && age < 30) ? "running" : "finished"; label = s === "running" ? "final evaluation" : "finished"; }
    else if (L?.pause_flag) { s = "paused"; label = "paused"; }
    else if (inMilestone) { s = "running"; label = "24 h milestone evaluation"; }
    else if (proc === null && age > 8) { s = "stopped"; label = "trainer not running"; }            // a relaunch (compile ≈ 2 min) is not a stop
    else if (age != null && age > 45) { s = "stopped"; label = "no data for " + Math.round(age) + " min"; }
    else if (age != null && age > 1.2 && hb) { s = "running"; label = "evaluating"; }
    return { s, label, age, step, inMilestone, H, wh, proc };
  }
  /* Upcoming evaluations — train.py rule: at each of v1's eval steps, or when 60 min have passed since the last
     evaluation ended (the timer restarts after every evaluation, so it only fires once v1's steps run out after 4,050). */
  const EVAL_SEC = 85;                                                   // a quick evaluation takes ≈ 80–90 s
  function nextEvals(n = 2) {
    const r = stepRate(), S = state(); if (!r) return [];
    const sps = r / 3600, now = Date.now() / 1000, last = (D.evals || []).at(-1);
    let tCur = last && fin(tparse(last)) ? tparse(last) / 1000 : now, stepAt = S.step, tAt = now;
    const stepTime = st => tAt + (st - stepAt) / sps, out = [];
    for (let guard = 0; out.length < n && guard < 20; guard++) {
      const v1 = V1_STEPS.find(x => x > stepAt), tv = v1 ? stepTime(v1) : Infinity, th = tCur + 3600;
      const hourly = th < tv, t = hourly ? th : tv, st = hourly ? Math.round(stepAt + (th - tAt) * sps) : v1;
      out.push({ step: st, t: t * 1000, kind: hourly ? "hourly" : "v1 step", gate: GATE_STEPS.includes(st) });
      tCur = t + EVAL_SEC; stepAt = st; tAt = t + EVAL_SEC;               // training pauses while it evaluates
    }
    return out;
  }
  const stepRate = () => fin(D.tok_s) ? D.tok_s / TOK_PER_STEP * 3600 : null;          // steps per hour
  const atStep = (s, cur) => { const r = stepRate(); return r ? Date.now() + (s - cur) / r * 3600e3 : null; };

  /* ------------------------------------------------------------------ health checks: one list drives cards, verdict, KPIs and chart pills */
  function v1Gap(e) {                                                   // v3 − v1 at the same step, per language and mean
    const v1 = REF.v1.evals, d = (x, k) => { const r = interp(v1, e.step, k); return fin(r) && fin(x) ? x - r : null; };   // null past v1's last eval (4,050): number − null would be the number itself
    const ts = d(e.ts, "ts"), wk = d(e.wk, "wk"), m = d(e.val, "val");
    return { ts: fin(ts) ? ts : null, wk: fin(wk) ? wk : null, m: fin(m) ? m : null };
  }
  /* Loss trend = v1's rule at v3's 25-step cadence: mean of the last 4 windows (100 steps) vs the 4 windows 300 steps
     earlier; NTP uses the trainer's smoothed ntp_ema. Raw 25-step values wobble ±0.06 nats, which the old 100-step
     comparison flagged red (replay over steps 300–3,075: old rule 2 red + 32 amber, this rule 0 + 0). */
  const trendSeries = key => {
    const col = key === "ntp" ? "ntp_ema" : key, rs = R.steps.filter(r => fin(r[col]));
    return rs.length ? rs.map(r => [r.step, r[col]]) : ser(key);          // fallback: v3_live.json raw values
  };
  function trend(key, { win = 12, k = 4, flat = 0.002, bad = 0.05, what = "" } = {}) {
    const r = trendSeries(key); if (r.length < win + k) return null;
    const a = mean(r.slice(-win - k, -win).map(p => p[1])), b = mean(r.slice(-k).map(p => p[1])), d = b - a;
    const span = `over the last ${fI(r.at(-1)[0] - r.at(-win - 1)[0])} steps`;
    if (d < -flat) return ST("g", `${what}falling ${sgn(d)} ${span}`);
    if (d < bad) return ST("a", `${what}flat (${sgn(d)}) ${span}; normal for short stretches, and late in training when the learning rate is low`);
    return ST("r", `${what}rising ${sgn(d)} ${span}`);
  }
  const smoothNtp = () => trendSeries("ntp");                          // [[step, smoothed NTP]]

  function checks() {
    const S = state(), ev = evs(), e1 = ev.at(-1), e0 = ev.at(-2), G = REF.gates, out = {};
    const add = (id, lvl, t, v, what, bad, go) => { out[id] = { id, lvl, t, v, what, bad, go }; };
    // 1 process
    if (S.label === "24 h milestone evaluation") add("proc", "a", "Training process", "milestone eval", "the 24 h milestone evaluation is running (≈ 2 h); training logs pause meanwhile", "no new data long after the evaluation should have finished", "v3c-step");
    else if (S.s === "stopped") add("proc", "r", "Training process", "stopped", S.label + (S.age != null ? ` · last data ${Math.round(S.age)} min ago` : ""), "the process dying unexpectedly", "v3c-step");
    else if (S.s === "paused") add("proc", "a", "Training process", "paused", "a PAUSE flag is set in the run folder", "a pause nobody asked for", "v3c-step");
    else if (S.s === "finished") add("proc", "g", "Training process", "finished", "the 48 h run completed", "—", "v3c-step");
    else add("proc", "g", "Training process", S.label,
      S.proc ? `running for ${S.proc.etime} · CPU ${f(S.proc.cpu, 0)} % · last step ${S.age < 1 ? "just now" : Math.round(S.age) + " min ago"}` : `last data ${S.age != null ? (S.age < 1 ? "just now" : Math.round(S.age) + " min ago") : "—"}`,
      "no new step for more than 15 min outside an evaluation", "v3c-step");
    // 2 validation loss falling
    if (e1 && e0) { const d = e1.val - e0.val;
      add("val", d < -0.005 ? "g" : d < 0.005 ? "a" : "r", "Validation loss falling", sgn(d), `${f(e0.val)} → ${f(e1.val)} nats between step ${fI(e0.step)} and ${fI(e1.step)}`, "flat or rising over several evaluations (learning has stalled)", "v3c-val"); }
    else add("val", "n", "Validation loss falling", e1 ? f(e1.val) : "—", "needs two evaluations; the first ones run at v1's eval steps (185, 265, 365 …)", "flat or rising over several evaluations", "v3c-val");
    // 3 pace vs v1 (E2 thresholds at the gate steps; before them the trend decides)
    const gaps = ev.map(e => ({ step: e.step, ...v1Gap(e) })).filter(g => fin(g.m));
    const g1 = gaps.at(-1), g0 = gaps.at(-2);
    if (g1) {
      const worst = Math.max(g1.ts ?? -9, g1.wk ?? -9), gated = g1.step >= GATE_STEPS[0] - 60;
      const lvl = gated ? (worst <= G.E2_margin[0] ? "g" : worst <= G.E2_margin[1] ? "a" : "r")
                        : (g1.m <= G.E2_margin[0] ? "g" : g0 && g1.m < g0.m ? "g" : "a");
      const trendTxt = g0 ? ` (was ${sgn(g0.m)} at step ${fI(g0.step)}${g1.m < g0.m ? ", closing" : ", widening"})` : "";
      add("pace", lvl, "Keeping pace with v1", sgn(g1.m) + " nats",
        (g1.m > 0 ? `v3 is ${f(g1.m)} nats behind v1 at step ${fI(g1.step)}${trendTxt}` : `v3 is ${f(-g1.m)} nats ahead of v1 at step ${fI(g1.step)}${trendTxt}`)
          + (e1 && e1.step > g1.step ? ` · v1's evaluations end at step ${fI(g1.step)}, so this is the last same-step comparison` : "")
          + (gated ? ` · TinyStories ${sgn(g1.ts)}, WikiText ${sgn(g1.wk)} (the line is +0.10 on each)` : ""),
        gated ? "either language more than +0.20 behind v1 at a gate step (E2)" : "the gap widening; from step 1,217 either language must be within +0.10 of v1 (E2)", "v3c-gap");
    } else if (e1) add("pace", "n", "Keeping pace with v1", "—", "v1 has no evaluation at this step (v1 stopped at 4,050)", "—", "v3c-gap");
    // 4 training loss trend
    const tr = trend("ntp", { what: "smoothed NTP " }), sn = smoothNtp();
    if (tr) add("ntp", tr.lvl, "Training loss trend", sgn(sn.at(-1)?.[1] - sn.at(-13)?.[1]), tr.why, "a rise of 0.05 nats or more in the smoothed loss sustained over 300 steps (instability)", "v3c-ntp");
    // 5 stability (E3)
    const gnAll = ser("gn_preclip"), gnPost = gnAll.filter(p => p[0] > 300).map(p => p[1]), gnNow = gnAll.at(-1)?.[1];
    const nonFinite = ["loss", "ntp"].some(k => { const v = (D.history?.[k] || []).at(-1)?.[1]; return v != null && !isFinite(v); });
    if (nonFinite) add("stab", "r", "Stability", "NaN", "a loss is not finite", "NaN or exploding losses", "v3c-gn");
    else if (gnPost.length) { const mx = Math.max(...gnPost);
      add("stab", mx <= 1 ? "g" : "a", "Stability", "grad " + f(gnNow, 2), mx <= 1 ? `gradient norm stayed under the 1.0 clip since warm-up (peak ${f(mx, 2)})` : `gradient norm reached ${f(mx, 2)} after warm-up (above the 1.0 clip)`, "norms above 1.0 after warm-up (E3) or NaN losses", "v3c-gn"); }
    else if (fin(gnNow)) add("stab", "g", "Stability", "grad " + f(gnNow, 2), "warm-up (first 300 steps): norms above 1 are normal here; the E3 check starts at step 300", "NaN losses", "v3c-gn");
    // 6 speed (E4)
    const tk = ser("tok_s").map(p => p[1]), tkMed = median(tk.slice(-6)), runMed = median(tk);
    if (fin(tkMed)) { const slow = runMed && tkMed < 0.85 * runMed;
      add("speed", tkMed >= G.E4[0] ? (slow ? "a" : "g") : tkMed >= G.E4[1] ? "a" : "r", "Speed", fI(tkMed) + " tok/s",
        slow ? `slower than this run's usual ${fI(runMed)} tok/s: another app may be using the GPU` : `median of the last windows; this run's usual is ≈ ${fI(runMed)} tok/s`,
        `below ${fI(G.E4[0])} tok/s (E4 pass), and ${fI(G.E4[1])} is the floor`, "v3c-tok"); }
    // 7 long-range recall (E1)
    if (e1) { const r1 = e1.recall1, sw = e1.swap, v1r = interp(REF.v1.evals, e1.step, "recall1"), gateE = ev.find(e => e.step >= G.E1.step);
      let lvl, what;
      if (r1 > 0.05 && sw >= 0.03) { lvl = "r"; what = `plan swap keeps ${f(sw)} of recall: the answer may bypass the concept stream`; }
      else if (gateE) { lvl = gateE.recall1 >= G.E1.pass_ ? "g" : gateE.recall1 >= G.E1.fail ? "a" : "r"; what = `recall ${f(gateE.recall1)} at step ${fI(gateE.step)} (E1 pass ≥ ${G.E1.pass_}; v1 had ${f(G.E1.v1)})`; }
      else if (r1 >= G.E1.pass_) { lvl = "g"; what = `already above the E1 bar (${G.E1.pass_}) before step ${fI(G.E1.step)}`; }
      else { lvl = "n"; what = `not expected yet: v1's recall appeared at step ≈ 1,009–1,217 (v1 here: ${f(v1r)}); judged at step ${fI(G.E1.step)}`; }
      add("recall", lvl, "Long-range recall (E1)", `${f(r1)} · swap ${f(sw)}`, what, `below ${G.E1.pass_} at step ${fI(G.E1.step)}, or recall that survives the plan swap`, "v3c-rec"); }
    // 8 codebook fit vs v1
    if (e1 && fin(e1.VE)) { const v1ve = interp(REF.v1.evals, e1.step, "VE"), d = fin(v1ve) ? e1.VE - v1ve : null;
      add("cb", d == null || d >= -0.05 ? "g" : "a", "Codebook fits the concepts", f(e1.VE), `share of concept variation captured by the nearest codes${fin(v1ve) ? ` (v1 at this step ${f(v1ve)})` : ""}; least-used dictionary uses ${f(100 * e1.codes_min, 0)} % of codes`, "clearly below v1 at the same step (the codes summarise concepts poorly)", "v3c-cb"); }
    // 9 zero-initialised parts opened
    const op = Object.values(D.opened || {}).sort((a, b) => a.step - b.step).at(-1);
    if (op) add("opened", op.still_zero?.length ? (op.step >= 300 ? "r" : "a") : "g", "New parts switched on", op.still_zero?.length ? op.still_zero.join(", ") : "all moved",
      op.still_zero?.length ? `still exactly zero at step ${op.step}` : `the (t, Δ) table, shared FFN and both Engram projections all moved off zero (checked at step ${op.step})`, "any part still exactly zero at step 300", "v3-opened-card");
    // 10 memory
    const fp = lastOf("footprint_gib"), fps = ser("footprint_gib"), wh = ser("wall_h"), sw = L?.swap_gb ?? (lastOf("swap_mb") ?? 0) / 1000;
    if (fin(fp)) { const hrs = wh.length > 1 ? wh.at(-1)[1] - wh[0][1] : 0, leak = hrs > 0.3 ? (fp - fps[0][1]) / hrs : null, guard = D.config?.mem_restart_gib || 24;
      const re = leak > 0.1 ? Math.max(0, (guard - fp) / leak) : null;
      add("mem", sw > 4.6 ? "a" : "g", "Memory", `${f(fp, 1)} GiB`, `process footprint${leak != null ? ` growing ${f(leak, 2)} GiB/h` : ""}${re != null ? `; planned automatic restart at ${guard} GiB in ≈ ${f(re, 1)} h (resumes from checkpoint)` : ""} · swap ${f(sw, 1)} GB`,
        "swap above ≈ 4.6 GB (launch baseline + 2 GB), which slows the GPU", "v3c-mem"); }
    // 11 power
    const b = L?.battery;
    if (b) add("power", b.ac || b.pct > 30 ? "g" : b.pct > 10 ? "a" : "r", "Power", `${b.pct} % · ${b.ac ? "charger" : "battery"}`, b.ac ? "on the charger" : "on battery", "running out of battery (work since the last checkpoint would be lost)", "v3-now");
    return out;
  }

  /* ------------------------------------------------------------------ hero */
  function verdictItems(K) {                                           // health checks + charts that carry their own check
    const own = Object.entries(CH || {}).filter(([, c]) => c.o.own && c.status && c.status.lvl !== "n")
      .map(([k, c]) => ({ lvl: c.status.lvl, t: c.o.title, what: c.status.why, go: "v3c-" + k }));
    return [...Object.values(K), ...own];
  }
  function renderHero(K) {
    const el = document.getElementById("v3-now"), S = state(), ev = evs(), now = Date.now();
    const pct = 100 * S.wh / S.H, tokens = (D.slots || 0) * 0.943;
    const tok48 = D.total_steps ? D.total_steps * TOK_PER_STEP * 0.943 : null, ep48 = D.total_steps ? D.total_steps * TOK_PER_STEP / SLOTS_EPOCH : null;   // trainer's projection at the current speed
    const remTrain = Math.max(0, S.H - S.wh), eta = now + (remTrain + (D.milestone_done ? 0 : 2) + 3) * 3600e3;
    const items = verdictItems(K), n = { g: 0, a: 0, r: 0 }; items.forEach(i => { if (n[i.lvl] != null) n[i.lvl]++; });
    const wl = n.r ? "r" : n.a ? "a" : "g";
    const vt = wl === "g" ? `All ${n.g} checks healthy` : wl === "a" ? `${n.a} ${n.a > 1 ? "things" : "thing"} to watch, nothing wrong` : `${n.r} ${n.r > 1 ? "problems need" : "problem needs"} attention`;
    const vs = wl === "g" ? "Every check on this page is inside its healthy range." : wl === "a" ? "Amber means worth a look, not broken. Click an item to jump to its chart." : "Red means a value is outside its safe range. Click an item to jump to its chart.";
    const flagged = items.filter(i => i.lvl === "a" || i.lvl === "r").sort((a, b) => RANK[b.lvl] - RANK[a.lvl]);
    const ticks = ev.map(e => fin(e.wall_h) ? `<i style="left:${Math.min(100, 100 * e.wall_h / S.H)}%" title="evaluation at step ${fI(e.step)}"></i>` : "").join("")
      + `<i style="left:${100 * (D.milestone_hours || 24) / S.H}%;background:#c98311" title="24 h milestone evaluation"></i>`;
    const nextV1 = V1_STEPS.find(s => s > S.step), nextGate = GATE_STEPS.find(s => s > S.step), e1 = ev.at(-1);
    const NE = nextEvals(2), nextEvT = NE[0]?.t ?? null;
    const msT = !D.milestone_done ? Date.now() + Math.max(0, (D.milestone_hours || 24) - S.wh) * 3600e3 : null;
    const gpuH = gpuSamples().map(p => p[1]).filter(fin), gpuNow = gpuLatest();
    const fp = lastOf("footprint_gib"), sw = L?.swap_gb ?? (lastOf("swap_mb") ?? 0) / 1000;
    const p1 = Math.min(100, 100 * S.wh / (D.milestone_hours || 24)), p2 = Math.max(0, Math.min(100, 100 * (S.wh - (D.milestone_hours || 24)) / (S.H - (D.milestone_hours || 24))));
    const st1 = D.milestone_done ? "done" : S.inMilestone ? "evaluating" : S.s === "stopped" ? "stopped" : "running";
    const st2 = !D.milestone_done ? "next" : S.s === "finished" ? "done" : S.s === "stopped" ? "stopped" : "running";
    const lv = s => s === "done" ? "done" : s === "running" || s === "evaluating" ? "on" : s === "next" ? "armed" : "off";
    el.innerHTML = `
      <div class="hero-main">
        <div class="hero-l">
          <div class="state-chip s-${S.s}"><span class="dot ${S.s}"></span>v3 · ${S.label}</div>
          <div class="stepbig">${fI(S.step)} <small>/ ≈ ${fI(D.total_steps)} steps</small></div>
          <div class="pbar2"><div class="fill" style="width:${Math.min(100, pct)}%"></div><div class="ticks">${ticks}</div></div>
          <div class="hero-meta">
            <span><b>${f(pct, 1)} %</b> of 48 h</span>
            <span><b>${(tokens / 1e6).toFixed(1)}M</b> of ≈ <b>${fin(tok48) ? Math.round(tok48 / 1e6) + "M" : "—"}</b> tokens by 48 h${fin(ep48) ? ` (${f(ep48, 2)} epochs)` : ""}</span>
            <span>finishes ≈ <b>${day(eta)}</b></span>
            <span class="muted">updated ${S.age == null ? "—" : S.age < 1 ? "just now" : Math.round(S.age) + " min ago"}</span>
          </div>
        </div>
        <div class="verdict v-${wl}">
          <div class="v-icon">${wl === "g" ? "✓" : wl === "a" ? "!" : "✕"}</div>
          <div class="v-body">
            <div class="v-t">${vt}</div><div class="v-s">${vs}</div>
            <div class="v-counts">${chips(n)}</div>
            ${e1 ? `<div class="v-ev">Latest evaluation: step ${fI(e1.step)} · val ${f(e1.val)}${NE.length ? ` · next: ${NE.map(x => `step ${fI(x.step)} ≈ ${clock(x.t)}`).join(", then ")}` : ""}</div>` : ""}
            ${flagged.length ? `<ul class="v-list">${flagged.slice(0, 5).map(v => `<li data-go="${v.go}" class="${v.lvl}"><i></i><b>${v.t}</b><span>${v.what}</span></li>`).join("")}</ul>` : ""}
          </div>
        </div>
      </div>
      <div class="plan">
        <div class="stg ${lv(st1)}"><div class="stg-h"><span class="stg-n">1</span><b>First 24 h</b><span class="stg-st">${st1}</span></div>
          <div class="stg-bar"><div style="width:${p1}%"></div></div>
          <div class="stg-s">warm-up 300 steps to LR 6e-4, then cosine by time · quick evaluations at v1's eval steps and hourly · ends with the 24 h milestone evaluation (≈ 2 h)</div></div>
        <div class="stg-arrow">→</div>
        <div class="stg ${lv(st2)}"><div class="stg-h"><span class="stg-n">2</span><b>Second 24 h</b><span class="stg-st">${st2}</span></div>
          <div class="stg-bar"><div style="width:${p2}%"></div></div>
          <div class="stg-s">cosine continues to LR 6e-5 at 48 h · then the final evaluation, including the sealed test split</div></div>
      </div>
      <div class="hero-row">
        <div class="hs"><div class="lbl">Time left</div><div class="val">${S.s === "finished" ? "—" : fmtH(remTrain)}</div><div class="s">of training · used ${fmtH(S.wh)} of 48 h</div></div>
        <div class="hs"><div class="lbl">Next evaluations</div>${NE.length ? NE.map((x, i) => `<div class="${i ? "s" : "val sm"}">${i ? "then " : ""}step ${fI(x.step)} · ≈ ${clock(x.t)}${x.kind === "hourly" ? " (hourly)" : ""}</div>`).join("") : '<div class="val sm">—</div>'}
          <div class="s">${msT ? `24 h milestone evaluation ≈ ${day(msT)}` : "milestone done"}${nextGate ? ` · gate step ${fI(nextGate)} ≈ ${clock(atStep(nextGate, S.step))}` : ""}</div></div>
        <div class="hs"><div class="lbl">GPU load</div><div class="val">${fin(gpuNow) ? gpuNow + " %" : "—"}</div><canvas class="spark" id="v3-spark-gpu"></canvas></div>
        <div class="hs"><div class="lbl">Trainer</div><div class="val sm">${S.proc ? `CPU ${f(S.proc.cpu, 0)} % · ${S.proc.rss_gb} GB` : L ? "not running" : "—"}</div>
          <div class="s">${S.proc ? "up " + S.proc.etime : L ? "" : "process info from :8766 unavailable"} · ${fI(D.tok_s)} tok/s</div></div>
        <div class="hs"><div class="lbl">Power</div><div class="val sm">${L?.battery ? `${L.battery.pct} % · ${L.battery.state}` : "—"}</div><div class="s">${L?.battery ? (L.battery.ac ? "on charger" : '<span class="warn">on battery</span>') : ""}</div></div>
        <div class="hs"><div class="lbl">Memory</div><div class="val sm">${f(fp, 1)} GiB · swap ${f(sw, 1)} GB</div><div class="s">${K.mem ? K.mem.what.split("·")[0].replace(/^process footprint/, "footprint") : ""}</div></div>
      </div>`;
    spark2(document.getElementById("v3-spark-gpu"), gpuH.slice(-240), "#2e9c8c", 0, 100);
  }
  const tparse = e => fin(e.wall_h) && fin(D.wall_h) && fin(D.t) ? (D.t - (D.wall_h - e.wall_h) * 3600) * 1000 : null;
  const chips = n => ["g", "a", "r"].filter(k => n[k]).map(k => `<span class="cc ${k}"><i></i>${n[k]} ${{ g: "healthy", a: "watch", r: "problem" }[k]}</span>`).join("");

  /* ------------------------------------------------------------------ KPI tiles */
  function renderKpis(K) {
    const el = document.getElementById("v3-kpis"), ev = evs(), e1 = ev.at(-1), e0 = ev.at(-2);
    const d = (a, b, dig = 3, lowerBetter = true) => !fin(a) || !fin(b) ? "" : `<span class="kd ${b === a ? "" : (b - a < 0) === lowerBetter ? "kg" : "kb"}">${sgn(b - a, dig)}</span>`;
    const gapsF = ev.map(e => [e.step, v1Gap(e).m]).filter(p => fin(p[1])), ntp = ser("ntp"), tk = ser("tok_s");
    const v1ntp = interp(REF.v1.train, lastOf("ntp") != null ? ntp.at(-1)[0] : 0, "ntp");
    const lvl = k => K[k]?.lvl || "n";
    const tiles = [
      { go: "v3c-val", k: "val", lbl: "Validation loss", v: f(e1?.val), u: "nats", dl: d(e0?.val, e1?.val), sub: "vs previous evaluation", sp: ev.map(e => e.val), col: "#8f7cc0" },
      { go: "v3c-gap", k: "pace", lbl: "Gap to v1", v: sgn(gapsF.at(-1)?.[1]), u: "nats", dl: d(gapsF.at(-2)?.[1], gapsF.at(-1)?.[1]), sub: gapsF.length && e1 && e1.step > gapsF.at(-1)[0] ? `at step ${fI(gapsF.at(-1)[0])}, v1's last · below 0 = v3 ahead` : "same step · below 0 = v3 ahead", sp: gapsF.map(p => p[1]), col: C1 },
      { go: "v3c-rec", k: "recall", lbl: "Long-range recall", v: f(e1?.recall1), u: "", dl: "", sub: (() => { const r = e1 ? interp(REF.v1.evals, e1.step, "recall1") : null, L1 = REF.v1.evals.at(-1); return fin(r) ? `v1 here ${f(r)} · E1 at step 1,792` : `v1's last ${f(L1?.recall1)} (step ${fI(L1?.step)}) · E1 passed`; })(), sp: ev.map(e => e.recall1), col: C3, lo: 0, hi: 1 },
      { go: "v3c-ntp", k: "ntp", lbl: "Training loss (NTP)", v: f(lastOf("ntp")), u: "nats", dl: d(smoothNtp().at(-13)?.[1], smoothNtp().at(-1)?.[1]), sub: `smoothed · vs 300 steps ago · v1 ${f(v1ntp)}`, sp: ntp.slice(-40).map(p => p[1]), col: C1 },
      { go: "v3c-tok", k: "speed", lbl: "Throughput", v: fI(lastOf("tok_s")), u: "tok/s", dl: "", sub: `GPU ${gpuLatest() ?? "—"} % busy · v1 ran ≈ 1,700`, sp: tk.slice(-40).map(p => p[1]), col: CY },
      { go: "v3c-cb", k: "cb", lbl: "Codebook fit", v: f(e1?.VE), u: "", dl: d(e0?.VE, e1?.VE, 3, false), sub: `codes used ${e1 ? f(100 * e1.codes_min, 0) + " %" : "—"} · ppl ${f(e1?.perplexity, 0)}`, sp: ev.map(e => e.VE), col: CW },
    ];
    el.innerHTML = tiles.map(t => `<button class="kpi k-${lvl(t.k)}" data-go="${t.go}" title="${(K[t.k]?.what || "").replace(/"/g, "&quot;")}">
        <div class="kl"><span class="kdot"></span>${t.lbl}</div>
        <div class="kv">${t.v}<small>${t.u}</small>${t.dl}</div>
        <div class="ks">${t.sub}</div><canvas class="kspark" data-k="${t.k}"></canvas></button>`).join("");
    tiles.forEach(t => spark2(el.querySelector(`canvas[data-k="${t.k}"]`), t.sp, t.col, t.lo, t.hi));
  }

  /* ------------------------------------------------------------------ health cards */
  function renderHealth(K) {
    document.getElementById("v3-health").innerHTML = Object.values(K).map(i =>
      `<div class="hl ${i.lvl}" data-go="${i.go}" style="cursor:pointer"><div class="lt"></div><div class="ht">${i.t}<span>${i.v}</span></div><div class="hd">${i.what}</div><div class="hb">bad would be: ${i.bad}</div></div>`).join("");
  }

  /* ------------------------------------------------------------------ charts */
  function mk() {
    const P = id => document.getElementById(id), ch = {};
    const NOLOG = "run logs not readable (data/v3_train_log.jsonl, data/v3_evals.jsonl)";
    const K = () => checks();
    const fromCheck = id => () => { const c = K()[id]; return c ? ST(c.lvl, c.what, c.lvl === "n" ? "Not yet" : undefined) : null; };
    const diag = why => () => ST("n", why, "Diagnostic");
    const live = P("v3-ch-live"), cmp = P("v3-ch-cmp"), loss = P("v3-ch-loss"), con = P("v3-ch-concept"), sys = P("v3-ch-system");
    // Live
    ch.step = new LineChart(live, { latestIdx: 1, latestLabel: "NTP smoothed", title: "Loss every step (live)", measures: "next-token (NTP) and next-concept (NCP) loss of each optimizer step, from the trainer's heartbeat (last 600 steps).", good: "noisy, trending down.",
      empty: "live per-step feed not available (v3 server on :8766)", series: [{ name: "NTP", color: "#86ade0", width: 1 }, { name: "NTP (smoothed)", color: "#2f6fb3", width: 2.4 }, { name: "NCP", color: C3, width: 1.2 }],
      status: () => { const s = L?.steps || []; if (s.length < 40) return null; const a = mean(s.slice(-40, -20).map(r => r.ema)), b = mean(s.slice(-20).map(r => r.ema)), dd = b - a;
        return ST(dd < 0.02 ? "g" : dd < 0.1 ? "a" : "r", `smoothed NTP ${sgn(dd)} over the last 20 steps`); } });
    ch.gpu = new LineChart(live, { latestLabel: "GPU %", title: "GPU load, last hour", measures: "how busy the M2 GPU is, sampled every 5 s by the v3 dashboard server.", good: "≈ 95 %; short dips are evaluations and checkpoints.",
      xLabel: "min", yMin: 0, yMax: 100, fmt: v => v.toFixed(0), empty: "GPU samples not available (v3 server on :8766)", series: [{ name: "GPU %", color: C3, width: 1.6 }],
      status: () => {
        const raw = (L?.hist || []).slice(-60).map(x => x[1]).filter(fin); if (raw.length < 6) return null;
        const h = gpuSamples().slice(-60).map(p => p[1]).filter(fin), zeros = raw.length - h.length, tk = lastOf("tok_s");
        const glitch = zeros >= 6 ? ` · ${Math.round(100 * zeros / raw.length)} % of samples read 0 while the trainer kept stepping: a macOS counter glitch (often with the display asleep), shown as gaps` : "";
        if (!h.length) return stepping() ? ST("g", `GPU counter reads 0, but the trainer is stepping at ${fI(tk)} tok/s${glitch}`) : null;
        const m = mean(h);
        return ST(m >= 85 || (zeros >= 6 && tk >= 0.85 * (median(ser("tok_s").map(p => p[1])) || 0)) ? "g" : m >= 60 ? "a" : "r", `${m.toFixed(0)} % average over the last ${Math.round(raw.length * 5 / 60)} min${glitch || (m >= 85 ? "" : " (evaluations cause short dips; long dips mean other GPU work)")}`); } });
    // v3 vs v1
    ch.val = new LineChart(cmp, { latestLabel: "v3 val", title: "Validation loss vs v1 and v2", measures: "next-token loss on held-out text (mean of TinyStories and WikiText, nats), from v3's quick evaluation (128 rows per language).", good: "falling, and the teal line closing on the dashed blue one.",
      series: [{ name: "v3", color: C3b, width: 2.6, dots: "big" }, { name: "v1", color: C1, dash: true }, { name: "v2", color: C2, dash: true }], status: fromCheck("val") });
    ch.gap = new LineChart(cmp, { latestIdx: 2, latestLabel: "mean gap", title: "How far behind v1? (same step)", measures: "v3 minus v1 validation loss at the same step, per language. Below 0 means v3 is ahead.", good: "falling towards 0; within +0.10 at the gate steps 1,217 / 1,792 / 2,376 (E2).",
      fmt: v => (v >= 0 ? "+" : "") + v.toFixed(3), series: [{ name: "TinyStories", color: C3, dots: "big" }, { name: "WikiText", color: CT, dots: "big" }, { name: "mean", color: "#5f7896", width: 2.4 }],
      hlines: [{ y: 0, label: "level with v1", color: "#9aa3b2" }, { y: 0.1, label: "E2 pass +0.10", color: "#1f9460" }, { y: 0.2, label: "E2 limit +0.20", color: "#c98311" }], status: fromCheck("pace") });
    ch.rec = new LineChart(cmp, { latestLabel: "v3 recall", title: "Long-range recall", measures: "exact-match answers hidden 128–420 tokens back, beyond the decoder's reach. Plan swap = the same test with another document's plan (should stay ≈ 0).", good: "v3 rising above 0.25 by step 1,792 (E1) while the plan swap stays near 0.",
      yMin: 0, series: [{ name: "v3 recall", color: C3b, width: 2.6, dots: "big" }, { name: "v3 plan swap", color: CG }, { name: "v1 recall", color: C1, dash: true }, { name: "v2 recall", color: C2, dash: true }],
      hlines: [{ y: 0.25, label: "E1: 0.25 by step 1,792", color: "#c98311" }], status: fromCheck("recall") });
    ch.ntp = new LineChart(cmp, { latestLabel: "v3 NTP", title: "Training loss vs v1 and v2", measures: "next-token loss on the training batches (nats), 25-step windows for v3 and 50-step windows for v1 / v2.", good: "falling, close to or below the dashed v1 line.",
      series: [{ name: "v3", color: C3b, width: 2.4 }, { name: "v1", color: C1, dash: true }, { name: "v2", color: C2, dash: true }], status: fromCheck("ntp") });
    try {
    ch.partial = new LineChart(cmp, { empty: NOLOG, latestLabel: "carry3 · 1st", title: "Partial credit on carry3 / multi4", sub: "accuracy of each answer token separately (exact match needs all three) · rising first, before exact match",
      measures: "probes.lr_carry3.by_offset and lr_multi4.by_offset: per-position accuracy of the 3 answer tokens on held-out probe items.", good: "rising: the earliest sign the full answers will follow. v1's carry3 exact match reached 0.125 at step 2,376 and 0.273 at 2,764.",
      yMin: 0, yMax: 1, series: [{ name: "carry3 · 1st", color: "#3b78c2" }, { name: "carry3 · 2nd", color: "#6b9bd6", dash: true }, { name: "carry3 · 3rd", color: "#9fbfe6" }, { name: "multi4 · 1st", color: "#8f7cc0" }, { name: "multi4 · 2nd", color: "#b3a3d9", dash: true }, { name: "multi4 · 3rd", color: "#d0c6ea" }], status: diag("probe detail; E1 covers recall1") });
    ch.hop = new LineChart(cmp, { empty: NOLOG, latestLabel: "2 hops", title: "Multi-hop recall by number of hops", sub: "exact match on hop probes with 2, 3 and 4 links to follow",
      measures: "probes.lr_hop.exact_h2 / h3 / h4 (subsets of the 128 hop items, so each point rests on ~40 items).", good: "2 hops first, then 3; 4 hops is the hardest.",
      yMin: 0, series: [{ name: "2 hops", color: C3b, width: 2, dots: "big" }, { name: "3 hops", color: C2, dots: "big" }, { name: "4 hops", color: CW, dots: "big" }], status: diag("probe detail") });
    } catch (e) { console.error("probe charts:", e); }
    // Losses
    ch.ncp = new LineChart(loss, { latestLabel: "full depth", title: "Next-concept loss", measures: "how well the concept module predicts the next concept's codes, at full loop depth and through the shortcut.", good: "falling; the two lines close together means the shortcut works.",
      series: [{ name: "full depth (u₃)", color: C3, width: 2 }, { name: "shortcut", color: "#7cc7b8", dash: true }], status: () => trend("ncp", { flat: 0.003, bad: 0.05 }) });
    ch.vq = new LineChart(loss, { title: "VQ loss", measures: "distance from concepts to their nearest codes (trains the codebook only).", good: "flat or falling after warm-up; it rises a little while the concepts themselves are still moving.",
      series: [{ name: "VQ", color: CY, width: 2 }], status: () => trend("vq", { flat: 0.002, bad: 0.05 }) });
    ch.scons = new LineChart(loss, { title: "Shortcut consistency", measures: "L_scons = MSE(rms S_s, sg rms S₃), weight 0.1: how far the shortcut's state size is from the full loop's.", good: "small and flat.",
      fmt: v => v.toFixed(4), series: [{ name: "L_scons", color: C2, width: 2 }], status: () => { const v = lastOf("scons"); return fin(v) ? ST(v < 0.02 ? "g" : v < 0.05 ? "a" : "r", `${f(v, 4)} (it started at 0.14 and settled below 0.01)`) : null; } });
    ch.total = new LineChart(loss, { title: "Total loss", measures: "the quantity being minimised: NTP + 0.1·NCP + shortcut terms + VQ.", good: "falling. Not comparable with v1 (different terms).",
      series: [{ name: "total", color: "#7d8aa0", width: 2 }], status: () => trend("loss", { flat: 0.005, bad: 0.08 }) });
    // Concepts
    ch.cb = new LineChart(con, { latestLabel: "v3 VE", title: "Codebook fit", measures: "variance explained (VE): share of concept variation captured by the nearest codes; codes in use = least-used dictionary's share of its codes (evaluation, and every 100 training steps).", good: "at or above v1's dashed line.",
      yMin: 0, yMax: 1, series: [{ name: "v3 VE", color: CW, width: 2.2, dots: "big" }, { name: "codes in use (eval)", color: CY, dash: true }, { name: "codes in use (training)", color: "#b08a2e" }, { name: "v1 VE", color: C1, dash: true }], status: fromCheck("cb") });
    try {
    ch.dead = new LineChart(con, { empty: NOLOG, title: "Dead codes re-seeded", measures: "codes unused for 100 steps and replaced (of 4,096 = 32 dictionaries × 128), from the trainer's restart records.", good: "near 0 once the codebook settles; v1 treated more than 30 as worth a look.",
      yMin: 0, fmt: v => v.toFixed(0), series: [{ name: "re-seeded", color: "#e2728a", dots: "big" }], status: diag("codebook maintenance; no v3 threshold") });
    ch.pred = new LineChart(con, { empty: NOLOG, latestLabel: "gap, nats", title: "Is the next concept predictable?", measures: "code entropy H minus the next-concept loss (evaluation): how much better the concept module predicts the next concept's codes than code frequency alone.", good: "above 0 and growing; v1's check G-C2 used ≥ 0.2.",
      series: [{ name: "H − NCP", color: C3, width: 2.2, dots: "big" }], hlines: [{ y: 0, label: "no better than frequency", color: "#9aa3b2" }], status: diag("v1's G-C2 line (0.2) is shown in the explainer, not judged here") });
    } catch (e) { console.error("codebook charts:", e); }
    ch.mix = new LineChart(con, { latestLabel: "G share", title: "What goes into a concept", measures: "average weight of the CRC depth-mix on the raw embedding e0, the local stream L and the global stream G (E5, report only). v3 started at 0.107 / 0.107 / 0.787.", good: "any stable mix; not a bypass test.",
      yMin: 0, yMax: 1, latestIdx: 2, series: [{ name: "e0 (embedding)", color: "#999" }, { name: "L (local)", color: C1 }, { name: "G (global)", color: CW, dots: "big" }, { name: "v1 G share", color: C1, dash: true }], status: diag("E5 is reported, not judged") });
    ch.eng = new LineChart(con, { title: "Engram memory gate α", measures: "average share of the looked-up n-gram vector admitted into the encoder.", good: "open and stable; near 0 would mean the memory is unused.",
      series: [{ name: "α", color: CY, width: 2 }], status: () => { const v = lastOf("engram_alpha"); return fin(v) ? ST(v < 0.02 ? "a" : "g", v < 0.02 ? `gate nearly closed (${f(v)}): Engram unused` : `gate open at ${f(v)}: the memory is in use`) : null; } });
    try {
    // Loop (recurrence) — every pill Diagnostic: no pre-registered thresholds exist for these
    const lp = P("v3-ch-loop"), lp2 = P("v3-ch-loop2"), dg = w => diag(w);
    ch.depthadv = new LineChart(lp, { empty: NOLOG, latestLabel: "nats", title: "Do 3 passes beat 1?", sub: "next-concept loss with 1 pass minus with 3 passes, per evaluation · above 0 = the extra passes help",
      measures: "codebook.ncp_ce_M1 − codebook.ncp_ce on held-out data: the same model run with schedule \"1\" (one pass, (t, Δ) = (0, 1)) and \"3\" (three passes). Both paths are computed separately in evaluate.py.", good: "above 0 and growing: depth is paying off. v3 is a shortcut-trained model, so 1 pass is meant to be usable; 3 passes should still end up ahead.",
      fmt: v => (v >= 0 ? "+" : "") + v.toFixed(4), series: [{ name: "1 pass − 3 passes", color: C3b, width: 2.4, dots: "big" }], hlines: [{ y: 0, label: "3 passes = 1 pass", color: "#9aa3b2" }],
      status: () => { const e = R.evals.at(-1); if (!e) return null; const v = e.codebook.ncp_ce_M1 - e.codebook.ncp_ce; return ST("n", `${v >= 0 ? "3 passes ahead by" : "1 pass ahead by"} ${Math.abs(v).toFixed(4)} nats at step ${fI(e.step)}`, "Diagnostic"); } });
    ch.dS = new LineChart(lp, { empty: NOLOG, title: "How big is each pass's update?", sub: "‖ΔS‖ per pass on training batches (every 25 steps) · pass 1 builds S from zero",
      measures: "mean norm of the change each pass makes to the working state S: ΔS₁ = S₁ (S₀ = 0), ΔS₂ = S₂ − S₁, ΔS₃ = S₃ − S₂ (train.py loop_geometry, main trajectory).", good: "no target. Passes 2 and 3 shrinking towards 0 would mean they have little left to do.",
      series: [{ name: "pass 1", color: "#86ade0" }, { name: "pass 2", color: C3 }, { name: "pass 3", color: CW, width: 2 }], status: dg("per-pass update size; read with the relative chart") });
    ch.rel = new LineChart(lp, { empty: NOLOG, title: "Later passes relative to pass 1", sub: "‖ΔS₂‖ / ‖ΔS₁‖ and ‖ΔS₃‖ / ‖ΔS₁‖ · lines = training, dashed with dots = evaluation",
      measures: "train.py rel_dS (training batches) and evaluate.py rel_step (held-out probe items). 1.0 means the pass changes S as much as pass 1 did.", good: "no target; well above 0 means later passes still write substantially.",
      series: [{ name: "pass 2 / pass 1", color: C3 }, { name: "pass 3 / pass 1", color: CW, width: 2 }, { name: "pass 2 (eval)", color: C3, dots: "big", dash: true }, { name: "pass 3 (eval)", color: CW, dots: "big", dash: true }],
      hlines: [{ y: 1, label: "same size as pass 1", color: "#9aa3b2" }], status: dg("relative update size") });
    ch.Snorm = new LineChart(lp, { empty: NOLOG, title: "Working-state size after each pass", sub: "‖S₁‖, ‖S₂‖, ‖S₃‖ on training batches · S is carried raw between passes",
      measures: "mean norm of the working state S after each pass of the main trajectory (train.py loop_geometry). u₃ = zcRMSNorm(S₃) is what the decoder sees.", good: "no target; steady growth pass to pass means the passes add rather than cancel.",
      series: [{ name: "after pass 1", color: "#86ade0" }, { name: "after pass 2", color: C3 }, { name: "after pass 3", color: CW, width: 2 }], status: dg("state size") });
    ch.ang = new LineChart(lp, { empty: NOLOG, latestLabel: "degrees", title: "Are passes 2 and 3 doing different work?", sub: "angle between ΔS₂ and ΔS₃ · 0° = the same direction · line = training, dashed with dots = evaluation · dashed = v1",
      measures: "mean angle between consecutive pass updates (train.py loop_geometry angle23; evaluate.py consecutive_step_angle_deg). v1's loop settled at 1.25°.", good: "no target: small angles do not prove idle passes, and large ones do not prove useful ones (draft §4.7).",
      yMin: 0, series: [{ name: "train", color: C2, width: 1.8 }, { name: "eval", color: C2, dots: "big", dash: true }], hlines: [{ y: 1.25, label: "v1 1.25°", color: C1 }], status: dg("direction between updates") });
    ch.rho = new LineChart(lp, { empty: NOLOG, latestLabel: "ρ₃", title: "Do the updates add up or cancel? (ρ₃)", sub: "‖S₃‖ / (‖ΔS₁‖ + ‖ΔS₂‖ + ‖ΔS₃‖) · 1 = all updates point the same way · dashed = v1",
      measures: "ratio of the final state's size to the sum of the update sizes (train.py / evaluate.py rho). Near 1 means the passes push S in one direction; lower means later passes partly undo earlier ones.", good: "no target; v1 sat at 0.99.",
      fmt: v => v.toFixed(3), series: [{ name: "train", color: C3, width: 1.8 }, { name: "eval", color: C3, dots: "big", dash: true }], hlines: [{ y: 0.99, label: "v1 0.99", color: C1 }], status: dg("alignment of updates") });
    ch.grad = new LineChart(lp, { empty: NOLOG, latestLabel: "ratio", title: "Where does the training signal land?", sub: "next-token gradient at pass 1's state ÷ at pass 3's state (evaluation) · 1 = even",
      measures: "|∂L_NTP/∂S₁| / |∂L_NTP/∂S₃| on held-out probe items (evaluate.py grad_ratio_S1_over_SM).", good: "no target; far above 1 would mean early passes dominate the learning signal, far below 1 that only the last pass is being trained.",
      fmt: v => v.toFixed(3), series: [{ name: "pass 1 ÷ pass 3", color: CW, width: 2, dots: "big" }], hlines: [{ y: 1, label: "even", color: "#9aa3b2" }], status: dg("gradient split across passes") });
    ch.table = new LineChart(lp2, { empty: NOLOG, title: "(t, Δ) table: total gain per pair", sub: "‖γ_C‖ + ‖γ_S‖ + ‖α‖ per (t, Δ) pair, mean over the 4 core layers · zero at start",
      measures: "the table tells each pass where it is (t) and how much work it owes (Δ). γ_C, γ_S scale the reads of C and S; α are the write gains. Zero-initialised, no weight decay.", good: "growing. The pairs only a shortcut uses ((0, ⅔), (⅓, ⅔), (0, 1)) are trained less often but each covers a bigger step (Δ = ⅔ or 1), so they need not be the smallest.",
      yMin: 0, fmt: v => v.toFixed(3), series: Object.keys(PAIRC).map(k => ({ name: PAIRN[k], color: PAIRC[k] })), status: dg("new v3 part; no threshold was set before the run") });
    ch.alpha = new LineChart(lp2, { empty: NOLOG, title: "(t, Δ) table: write gains α per pair", sub: "‖α‖ per pair: attention write + own-FFN write + shared-FFN write, pooled",
      measures: "train.py table_norms pools table sites 2, 5, 6 (attention α, own-FFN α, shared-FFN α) into one number, so the shared FFN's own write gain cannot be separated from the logs.", good: "growing; a pair whose α stays near 0 barely changes how its pass writes.",
      yMin: 0, fmt: v => v.toFixed(3), series: Object.keys(PAIRC).map(k => ({ name: PAIRN[k], color: PAIRC[k] })), status: dg("pooled write gains") });
    ch.short = new LineChart(lp2, { empty: NOLOG, latestLabel: "nats", title: "Shortcut vs main path (training)", sub: "shortcut NCP loss minus main NCP loss, every 25 steps · 0 = the shortcut predicts as well",
      measures: "ncp_s − ncp on training batches: the same NCP head on the shortcut endpoint (1, 2a or 2b passes) and on the main 3-pass endpoint. L_scons (Losses section) pulls them together.", good: "near 0: a shorter trajectory is almost as good, which is what the shortcut design aims for.",
      fmt: v => (v >= 0 ? "+" : "") + v.toFixed(4), series: [{ name: "shortcut − main", color: C2, width: 2 }], hlines: [{ y: 0, label: "equal", color: "#9aa3b2" }], status: dg("shortcut quality") });
    ch.shares = new LineChart(lp2, { empty: NOLOG, latestLabel: "% M 1", title: "Which shortcut was trained?", sub: "share of micro-batches per shortcut trajectory in each 25-step window",
      measures: "short_counts / 400 (25 steps × 16 micro-batches). Designed sampling: 1 pass 50 %, 2a 25 %, 2b 25 %.", good: "close to 50 / 25 / 25 (random sampling wobbles a few points).",
      yMin: 0, yMax: 100, fmt: v => v.toFixed(0), series: [{ name: "1 pass", color: CW, width: 2 }, { name: "2a (⅓, ⅔)", color: C2 }, { name: "2b (⅔, ⅓)", color: C3 }], hlines: [{ y: 50, label: "50 %", color: "#9aa3b2" }, { y: 25, label: "25 %", color: "#9aa3b2" }], status: dg("sampling as designed") });
    ch.zero = new LineChart(lp2, { empty: NOLOG, latestLabel: "shared FFN", title: "Weight size of the zero-initialised parts", sub: "weight norms of parts that start at exactly 0 · steps 100 / 200 / 300, then every evaluation",
      measures: "‖W₂‖ of the shared FFN (its output projection), ‖table‖, and ‖W_v‖ of the two Engram reads. These are weight sizes only: how much of each write actually comes through the shared FFN is not measured in this run.", good: "all moved off 0 (checked at step 300) and still growing or settled.",
      latestIdx: 0, series: [{ name: "shared FFN W₂", color: "#f2a56f", width: 2.4, dots: "big" }, { name: "(t, Δ) table", color: "#b39ddb" }, { name: "Engram read 1 W_v", color: CY }, { name: "Engram read 2 W_v", color: "#b08a2e", dash: true }], status: dg("weight size, not usage") });
    ch.tau = new LineChart(lp2, { empty: NOLOG, latestLabel: "layer 4", title: "Core attention sharpness τ", sub: "mean τ per core layer (6 heads) · τ multiplies the attention logits · 1 at start · higher = sharper · dashed = encoder full-attention layers",
      measures: "evaluation telemetry: the field is named log_tau but evaluate.py stores τ = exp(log_tau). The core layers use the same GatedAttention class as the encoder's F layers.", good: "slow drift, no blow-up.",
      latestIdx: 3, fmt: v => v.toFixed(3), series: [{ name: "core layer 1", color: "#86ade0" }, { name: "core layer 2", color: C3 }, { name: "core layer 3", color: C2 }, { name: "core layer 4", color: CW, width: 2 }, { name: "encoder F3", color: "#aaa", dash: true }, { name: "encoder F7", color: "#777", dash: true }],
      hlines: [{ y: 1, label: "start", color: "#9aa3b2" }], status: dg("attention temperatures") });
    ch.ncptau = new LineChart(lp2, { empty: NOLOG, latestLabel: "τ", title: "NCP head temperature", sub: "mean τ over the 32 code segments · τ divides the code distances · 16 at start · lower = sharper",
      measures: "evaluation telemetry core.log_tau (stored as τ = exp(log_tau)). It read 15.4 at the first evaluation (step 185).", good: "falling then settling: the head grows more confident as concepts become predictable.",
      yMin: 0, fmt: v => v.toFixed(2), series: [{ name: "NCP τ", color: C3b, width: 2.2, dots: "big" }], status: dg("prediction sharpness") });
    } catch (e) { console.error("loop charts:", e); }
    // System
    ch.tok = new LineChart(sys, { latestLabel: "v3 tok/s", title: "Throughput", measures: "tokens (slots incl. PAD) per second over each log window, evaluations excluded.", good: "≈ 3,000 on this M2 for v3; above 1,290 passes E4, 1,200 is the floor.",
      yMin: 0, fmt: v => v.toFixed(0), series: [{ name: "v3", color: C3b, width: 2.2 }, { name: "v1", color: C1, dash: true }, { name: "v2", color: C2, dash: true }],
      hlines: [{ y: 1290, label: "E4 pass 1,290 (floor 1,200)", color: "#1f9460" }], status: fromCheck("speed") });
    ch.gn = new LineChart(sys, { latestLabel: "all", title: "Gradient norm", measures: "size of each update before clipping at 1.0; codebook part ×10.", good: "under 1.0 after warm-up (E3).",
      yMin: 0, series: [{ name: "all (pre-clip)", color: "#7d8aa0", width: 2 }, { name: "codebook ×10", color: CY, dash: true }], hlines: [{ y: 1, label: "clip 1.0", color: "#9aa3b2" }], status: fromCheck("stab") });
    ch.lr = new LineChart(sys, { title: "Learning rate", measures: "optimizer step size.", good: "warm-up to 6e-4 over 300 steps, then a time-based cosine to 6e-5 at 48 h.",
      fmt: v => v.toExponential(1), series: [{ name: "lr", color: "#86ade0", width: 2 }], status: () => { const v = lastOf("lr"); return fin(v) ? ST("n", `${v.toExponential(2)} at step ${fI(ser("lr").at(-1)[0])}`, "Schedule") : null; } });
    ch.mem = new LineChart(sys, { latestLabel: "footprint GiB", title: "Memory", measures: "trainer process footprint, GPU (MPS) driver memory and macOS swap. The trainer restarts itself from its checkpoint when the footprint reaches 24 GiB.", good: "slow footprint growth (a planned restart handles it); swap below ≈ 4.6 GB.",
      yMin: 0, series: [{ name: "footprint GiB", color: CW, width: 2 }, { name: "MPS driver GiB", color: C2 }, { name: "swap GB", color: CG, dash: true }], status: fromCheck("mem") });
    Object.entries(ch).forEach(([k, c]) => { c.wrap.id = "v3c-" + k; });
    ["step", "gpu", "ncp", "vq", "scons", "total", "eng"].forEach(k => { ch[k].o.own = true; });
    return ch;
  }

  function setCharts() {
    const ev = evs(), v1 = REF.v1, v2 = REF.v2, step = state().step;
    const xr = Math.max(step * 1.15, step + 120, 300);                            // reference curves: show a little of what comes next
    const clip = pts => pts.filter(p => p[0] <= xr);
    const tr = (src, k) => clip((src || []).filter(r => fin(r[k])).map(r => [r.step, r[k]]));
    const gaps = ev.map(e => ({ s: e.step, ...v1Gap(e) }));
    CH.val.set([evPts("val", ev), clip(evPts("val", v1.evals)), clip(evPts("val", v2.evals))]);
    CH.gap.set([gaps.map(g => [g.s, g.ts]), gaps.map(g => [g.s, g.wk]), gaps.map(g => [g.s, g.m])]);
    CH.rec.set([evPts("recall1", ev), evPts("swap", ev), clip(evPts("recall1", v1.evals)), clip(evPts("recall1", v2.evals))]);
    CH.ntp.set([ser("ntp"), tr(v1.train, "ntp"), tr(v2.train, "ntp")]);
    CH.ncp.set([ser("ncp"), ser("ncp_s")]);
    CH.vq.set([ser("vq")]);
    CH.scons.set([ser("scons")]);
    CH.total.set([ser("loss")]);
    CH.mix.set([0, 1, 2].map(i => ev.filter(e => e.mix).map(e => [e.step, e.mix[i]])).concat([clip(evPts("G_share", v1.evals))]));
    CH.eng.set([ser("engram_alpha")]);
    setLoop();
    CH.tok.set([ser("tok_s"), tr(v1.train, "tok_s"), tr(v2.train, "tok_s")]);
    CH.gn.set([ser("gn_preclip"), ser("gn_codebook").map(p => [p[0], 10 * p[1]])]);
    CH.lr.set([ser("lr")]);
    CH.mem.set([ser("footprint_gib"), ser("mps_driver_gib"), ser("swap_mb").map(p => [p[0], p[1] / 1000])]);
  }
  /* ------------------------------------------------------------------ loop telemetry from the run logs */
  function setLoop() {
    try {
    if (!CH) return;
    const st = R.steps.filter(r => r.geometry), ev = R.evals, g = (r, k, i) => r.geometry?.[k]?.[i];
    const S3 = k => [0, 1, 2].map(i => st.map(r => [r.step, g(r, k, i)]));
    CH.depthadv.set([ev.map(e => [e.step, e.codebook?.ncp_ce_M1 - e.codebook?.ncp_ce])]);
    CH.dS.set(S3("dS_norm"));
    CH.rel.set([st.map(r => [r.step, g(r, "rel_dS", 0)]), st.map(r => [r.step, g(r, "rel_dS", 1)]), ev.map(e => [e.step, e.loop?.rel_step?.[0]]), ev.map(e => [e.step, e.loop?.rel_step?.[1]])]);
    CH.Snorm.set(S3("S_norm"));
    CH.ang.set([st.map(r => [r.step, r.geometry.angle23]), ev.map(e => [e.step, e.loop?.angle23])]);
    CH.rho.set([st.map(r => [r.step, r.geometry.rho3]), ev.map(e => [e.step, e.loop?.rho3])]);
    CH.grad.set([ev.map(e => [e.step, e.loop?.grad_ratio])]);
    const keys = Object.keys(PAIRC), tb = R.steps.filter(r => r.table);
    CH.table.set(keys.map(k => tb.map(r => [r.step, r.table[k] ? r.table[k].a + r.table[k].gC + r.table[k].gS : null])));
    CH.alpha.set(keys.map(k => tb.map(r => [r.step, r.table[k]?.a])));
    CH.short.set([R.steps.filter(r => fin(r.ncp) && fin(r.ncp_s)).map(r => [r.step, r.ncp_s - r.ncp])]);
    const sc = R.steps.filter(r => r.short_counts), tot = r => Object.values(r.short_counts).reduce((a, b) => a + b, 0) || 1;
    CH.shares.set(["1", "2a", "2b"].map(k => sc.map(r => [r.step, 100 * (r.short_counts[k] || 0) / tot(r)])));
    const op = R.opened, tel = ev.filter(e => e.telemetry);
    const zs = (ko, kt) => op.map(o => [o.step, o[ko]]).concat(tel.map(e => [e.step, e.telemetry[kt]]));
    CH.zero.set([zs("shared_w2", "shared_w2_norm"), zs("table", "table_norm"), zs("Wv1", "engram_Wv_norm"), zs("Wv2", "engram2_Wv_norm")]);
    const tm = (e, k) => { const v = e.telemetry?.tau?.[k]; return v && v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
    CH.tau.set([0, 1, 2, 3].map(l => tel.map(e => [e.step, tm(e, `core.layers.${l}.attn.log_tau`)])).concat([3, 7].map(l => tel.map(e => [e.step, tm(e, `enc.layers.${l}.attn.log_tau`)]))));
    CH.ncptau.set([tel.map(e => [e.step, tm(e, "core.log_tau")])]);
    CH.pred.set([ev.filter(e => e.codebook).map(e => [e.step, e.codebook.code_entropy - e.codebook.ncp_ce])]);
    CH.dead.set([R.restarts.map(r => [r.step, r.restart_dead])]);
    const dv = D?.evals || [];
    CH.cb.set([evPts("VE", dv), evPts("codes_min", dv), R.restarts.map(r => [r.step, r.codes_used_min_seg]), REF ? evPts("VE", REF.v1.evals).filter(p => p[0] <= Math.max(state().step * 1.15, state().step + 120, 300)) : []]);
    const pb = (fam, i) => ev.map(e => [e.step, e.probes?.[fam]?.by_offset?.[i]]);
    CH.partial.set([pb("lr_carry3", 0), pb("lr_carry3", 1), pb("lr_carry3", 2), pb("lr_multi4", 0), pb("lr_multi4", 1), pb("lr_multi4", 2)]);
    CH.hop.set(["exact_h2", "exact_h3", "exact_h4"].map(k => ev.map(e => [e.step, e.probes?.lr_hop?.[k]])));
    renderGlance(); renderTableGrid();
  
    } catch (e) { console.error("loop telemetry:", e); }
  }
  function renderGlance() {
    try {
    const el = document.getElementById("v3-glance"); if (!el) return;
    const r = R.steps.filter(x => x.geometry).at(-1), e = R.evals.at(-1), gm = r?.geometry || {};
    const adv = e?.codebook ? e.codebook.ncp_ce_M1 - e.codebook.ncp_ce : null;
    const sh = [...R.opened.map(o => [o.step, o.shared_w2]), ...R.evals.filter(x => x.telemetry).map(x => [x.step, x.telemetry.shared_w2_norm])];
    const cell = (l, v, s) => `<div class="hs"><div class="lbl">${l}</div><div class="val sm">${v}</div><div class="s">${s}</div></div>`;
    el.innerHTML = [
      cell("Update size per pass", gm.dS_norm ? gm.dS_norm.map(x => f(x, 1)).join(" · ") : "—", "‖ΔS₁‖ · ‖ΔS₂‖ · ‖ΔS₃‖ (training)"),
      cell("Passes 2 and 3 vs pass 1", gm.rel_dS ? gm.rel_dS.map(x => f(x, 2)).join(" · ") : "—", "1.00 = as big as pass 1"),
      cell("Angle ΔS₂ ↔ ΔS₃", fin(gm.angle23) ? f(gm.angle23, 2) + "°" : "—", "v1 settled at 1.25°"),
      cell("Updates add up (ρ₃)", f(gm.rho3, 3), "1 = same direction · v1 0.99"),
      cell("3 passes vs 1 pass", fin(adv) ? `${adv >= 0 ? "+" : "−"}${Math.abs(adv).toFixed(4)} nats` : "—", fin(adv) ? (adv > 0 ? "3 passes predict better" : "no depth advantage yet") + ` · step ${fI(e.step)}` : "after the first evaluation"),
      cell("Signal at pass 1 ÷ pass 3", f(e?.loop?.grad_ratio, 3), "next-token gradient · 1 = even"),
      cell("Shared FFN output weight", sh.length ? f(sh.at(-1)[1], 2) : "—", `weight size, not usage${sh.length > 1 ? ` · 0 at start → ${f(sh[0][1], 2)} at step ${sh[0][0]}` : ""}`),
      cell("(t, Δ) table size", f(e?.telemetry?.table_norm, 2), "‖table‖ · 0 at start"),
    ].join("");
    const w = document.getElementById("v3-glance-when"); if (w) w.textContent = r ? `· training step ${fI(r.step)}${e ? `, evaluation step ${fI(e.step)}` : ""}` : "";
  
    } catch (e) { console.error("loop glance:", e); }
  }
  const USED = { t01: "main · 2a", t11: "main", t21: "main · 2b", t02: "2b", t12: "2a", t03: "1 pass" };
  function renderTableGrid() {
    try {
    const el = document.getElementById("v3-tgrid"); if (!el) return;
    const r = R.steps.filter(x => x.table).at(-1); if (!r) { el.innerHTML = '<div class="muted">Appears after the first log window.</div>'; return; }
    const keys = ["t01", "t11", "t21", "t02", "t12", "t03"], sites = [["gC", "γ_C · read of C"], ["gS", "γ_S · read of S"], ["a", "α · writes (pooled)"]];
    const mx = Math.max(1e-9, ...keys.flatMap(k => sites.map(([s]) => r.table[k]?.[s] || 0)));
    const cellc = v => `<td class="v3-tg" style="--k:${(45 * v / mx).toFixed(0)}%">${f(v, 3)}</td>`;
    el.innerHTML = `<table class="t v3-tgrid"><thead><tr><th>pair (t, Δ)</th><th>used by</th>${sites.map(([, n]) => `<th>${n}</th>`).join("")}<th>total</th></tr></thead><tbody>` +
      keys.map(k => { const t = r.table[k] || {}; return `<tr><td><b>${PAIRN[k]}</b></td><td class="muted">${USED[k]}</td>${sites.map(([s]) => cellc(t[s] || 0)).join("")}<td><b>${f((t.gC || 0) + (t.gS || 0) + (t.a || 0), 3)}</b></td></tr>`; }).join("") +
      `</tbody></table><div class="muted" style="font-size:12px;margin-top:8px">Mean norm over the 4 core layers; darker = larger. The main path uses (0, ⅓) → (⅓, ⅓) → (⅔, ⅓); the shortcuts use 2a (0, ⅓) → (⅓, ⅔), 2b (0, ⅔) → (⅔, ⅓) and 1 pass (0, 1). α pools the attention, own-FFN and shared-FFN write gains, so the shared FFN's share cannot be read from this table.</div>`;
    const w = document.getElementById("v3-tgrid-when"); if (w) w.textContent = `· training step ${fI(r.step)}`;
  
    } catch (e) { console.error("table grid:", e); }
  }
  async function fetchJsonl(url) {                                  // run logs; the last line may be mid-append, so a bad line is dropped
    const r = await fetch(url + "?t=" + Date.now(), { cache: "no-store" }); if (!r.ok) throw new Error("HTTP " + r.status);
    const out = [];
    (await r.text()).replace(/\bNaN\b|-?\bInfinity\b/g, "null").split("\n").forEach(l => { if (!l.trim()) return; try { out.push(JSON.parse(l)); } catch (e) { /* partial line */ } });
    return out;
  }
  async function pollRuns() {
    try {
      const [tl, el] = await Promise.all([fetchJsonl("data/v3_train_log.jsonl"), fetchJsonl("data/v3_evals.jsonl")]);
      const k = tl.length + ":" + (tl.at(-1)?.step ?? "") + ":" + el.length;
      if (k === keyR) return false;
      keyR = k;
      R = { steps: tl.filter(x => !x.kind), restarts: tl.filter(x => x.kind === "restart"), opened: tl.filter(x => x.kind === "opened"), evals: el };
      return true;
    } catch (e) { return false; }
  }
  /* The trainer builds v3_live.json's history / evals from in-process lists, so every memory re-exec and launcher relaunch
     starts them empty (first seen at the 21:27 memory_restart). The run logs are append-only and complete, so rebuild both
     from them with write_dash's exact mapping (train.py) whenever the logs hold more; scalars in the live file are kept. */
  const HKEYS = ["loss", "ntp", "ncp", "ncp_s", "scons", "vq", "tok_s", "gn_preclip", "gn_codebook", "angle23", "rho3",
    "lr", "epochs", "footprint_gib", "swap_mb", "mps_driver_gib", "mps_alloc_gib", "engram_alpha", "wall_h"];
  function evalRow(e) {
    const pr = e.probes || {}, cb = e.codebook || {}, lp = e.loop || {}, tel = e.telemetry || {}, mix = tel.crc_mix_e0_L_G || [];
    const o = { step: e.step, tag: e.tag, wall_h: e.wall_h, val: e.val_lang_mean, ts: e.val_tinystories?.nats, wk: e.val_wikitext?.nats,
      bpb_ts: e.val_tinystories?.bpb, bpb_wk: e.val_wikitext?.bpb };
    for (const [k, v] of Object.entries(pr)) o[k.slice(3)] = v?.exact;
    Object.assign(o, { swap: e.recall1_plan_swap?.exact, ncp: cb.ncp_ce, ncp_M1: e.ncp_M1, VE: cb.var_explained, codes_min: cb.codes_used_frac_min,
      perplexity: cb.perplexity_mean, G_share: mix[2], mix, Wv1: tel.engram_Wv_norm, Wv2: tel.engram2_Wv_norm, alpha: tel.engram_alpha_mean,
      shared_w2: tel.shared_w2_norm, table_norm: tel.table_norm, angle23: lp.angle23, rho3: lp.rho3, rel_step: lp.rel_step,
      S_norm: lp.state_norm, grad_ratio: lp.grad_ratio, eval_sec: e.eval_sec, is_best: e.is_best });
    return o;
  }
  function backfill(d) {
    if (!d) return d;
    d.history = d.history || {};
    if (R.steps.length > (d.history?.loss?.length || 0)) {
      const H = {};
      for (const k of HKEYS) H[k] = R.steps.filter(r => r[k] != null).map(r => [r.step, r[k]]);
      H.table = R.steps.filter(r => r.table && Object.keys(r.table).length).map(r => [r.step, r.table]);
      H.rel_dS = R.steps.filter(r => r.geometry && Object.keys(r.geometry).length).map(r => [r.step, r.geometry.rel_dS]);
      d.history = Object.assign({}, d.history, H);
    }
    if (R.evals.length > (d.evals?.length || 0)) {
      d.evals = R.evals.filter(e => e.val_lang_mean != null).map(evalRow);
      d.history.val = d.evals.map(e => [e.step, e.val]);
      d.history.recall = d.evals.filter(e => e.recall1 != null).map(e => [e.step, e.recall1]);
    }
    if (R.opened.length > Object.keys(d.opened || {}).length) d.opened = Object.fromEntries(R.opened.map(o => [String(o.step), o]));
    return d;
  }

  /* ------------------------------------------------------------------ verdicts (four questions, three tiers)
     Tier 1 "switched on and moving" = live telemetry; tier 2 "carries load" = switch-off tests of the milestone / final
     evaluation (final_eval.py: lang.<name>.<intervention>.delta_vs_intact, items.<split>.<intervention>.<probe>.exact,
     R sweep 1 / 2a / 2b / 3); tier 3 "worth its cost" = the controlled arms, not run. Pills: g = a criterion fixed before
     the run is met; a = met marginally, or a v1 reference is not; n = Measured (no pre-set bar) / Pending 24 h / Needs arm.
     The draft calls every switch-off test a report, not a gate, so those rows are Measured. Kept out of checks(). */
  async function pollMS() {
    for (const [u, src] of [["data/v3_eval_final.json", "final evaluation (test split)"], ["data/v3_eval_24h.json", "24 h milestone evaluation (val split)"]]) {
      try {
        const r = await fetch(u + "?t=" + Date.now(), { cache: "no-store" }); if (!r.ok) continue;
        const j = JSON.parse((await r.text()).replace(/\bNaN\b|-?\bInfinity\b/g, "null"));
        const k = src + JSON.stringify(Object.keys(j)) + (j.lang ? Object.keys(j.lang).length : 0);
        const changed = k !== pollMS.key; pollMS.key = k; MS = j; MSsrc = src; return changed;
      } catch (e) { /* not written yet, or mid-write: keep the last good parse */ }
    }
    return false;
  }
  const msLang = k => { const L2 = MS?.lang; if (!L2) return null; const v = ["tinystories", "wikitext"].map(n => L2[n]?.[k]?.delta_vs_intact).filter(fin); return v.length === 2 ? mean(v) : null; };
  const msProbe = (k, fam) => { const it = MS?.items; if (!it) return null; const sp = it.val || it.test; return sp?.[k]?.[fam]?.exact ?? null; };
  const interpW = (rows, h, key) => { const r = (rows || []).filter(e => fin(e[key]) && fin(e.wall_h)); for (let i = 1; i < r.length; i++) if (r[i].wall_h >= h && r[i - 1].wall_h <= h) { const w = (h - r[i - 1].wall_h) / ((r[i].wall_h - r[i - 1].wall_h) || 1); return r[i - 1][key] + w * (r[i][key] - r[i - 1][key]); } return null; };
  const PILL = { met: ["g", "Met"], small: ["g", "Met · small"], marg: ["a", "Marginal"], meas: ["n", "Measured"], pend: ["n", "Pending 24 h"], arm: ["n", "Needs arm"], nolog: ["n", "Not logged"], ref: ["a", "Below v1"] };
  function vcard(q, overall, rows, note) {
    return `<div class="card v3-vcard"><div class="v3-vhead"><h4>${q}</h4><span class="st-pill n v3-vover"><i></i>${overall}</span></div>
      <table class="t v3-vtab"><tbody>${rows.map(([c, ev, p]) => { const [lvl, lab] = PILL[p]; return `<tr><td class="v3-vc">${c}</td><td class="v3-ve">${ev}</td><td><span class="st-pill ${lvl}"><i></i>${lab}</span></td></tr>`; }).join("")}</tbody></table>
      ${note ? `<div class="muted v3-vnote">${note}</div>` : ""}</div>`;
  }
  function renderVerdicts() {
    const el = document.getElementById("v3-verdicts"); if (!el || !D || !REF) return;
    try {
      const ev = R.evals, e = ev.at(-1), dv = D.evals || [], d1 = dv.at(-1), st = R.steps.filter(x => x.geometry), g = st.at(-1)?.geometry || {};
      const has = !!MS?.lang, pend = has ? "meas" : "pend", when = has ? `from the ${MSsrc}` : "at the 24 h milestone evaluation (≈ 01:10, then ≈ 2 h to finish)";
      const nat = v => fin(v) ? `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(4)} nats` : "—";
      /* 1 · looping */
      const adv = ev.filter(x => x.codebook).map(x => [x.step, x.codebook.ncp_ce_M1 - x.codebook.ncp_ce]), advPos = adv.filter(p => p[0] >= 1402), advNow = adv.at(-1)?.[1];
      const inSig = fin(g.angle23) && fin(g.rho3) && g.angle23 < 10 && g.rho3 > 0.90;
      const sc = R.steps.filter(r => fin(r.ncp) && fin(r.ncp_s)).slice(-4), scGap = sc.length ? mean(sc.map(r => r.ncp_s - r.ncp)) : null;
      const r1 = msLang("R1"), r2a = msLang("R2a"), r2b = msLang("R2b"), tOff = msLang("table_off");
      const hop3 = msProbe("intact", "lr_hop"), hop1 = msProbe("R1", "lr_hop");
      const loopRows = [
        ["Passes do different work (not repeating)", `angle ΔS₂↔ΔS₃ ${f(g.angle23, 2)}° (v1 1.25°) · ρ₃ ${f(g.rho3, 3)} · passes 2 and 3 write ${g.rel_dS ? g.rel_dS.map(x => f(x, 2)).join(" / ") : "—"} of pass 1${inSig ? " · still inside the draft's near-repetitive signature (angle < 10° with ρ₃ > 0.90), though 3× v1's angle" : " · outside the draft's near-repetitive signature"}`, inSig ? "marg" : "meas"],
        ["3 passes predict the next concept better than 1", `${nat(advNow)} at step ${fI(adv.at(-1)?.[0])} · positive at ${advPos.filter(p => p[1] > 0).length} of ${advPos.length} evaluations since step 1,402 · a report item in the draft (elasticity), no pre-set bar`, "meas"],
        ["The 1-pass shortcut is usable", `shortcut − main next-concept loss ${nat(scGap)} (mean of the last 100 steps) · what the shortcut design asks for`, "meas"],
        ["Extra passes improve next-token loss", has ? `1 pass instead of 3: ${nat(r1)} · 2a: ${nat(r2a)} · 2b: ${nat(r2b)} (positive = the extra passes help; v1: 1 vs 4 passes differed by 0.001)` : `the like-for-like test (next-token loss at 1 / 2 / 3 passes) ${when}`, pend],
        ["The (t, Δ) table carries load", has ? `switching the table off: ${nat(tOff)} next-token loss` : `table-off test ${when}`, pend],
        ["Depth helps multi-hop recall", has && fin(hop3) ? `hop exact ${f(hop1)} with 1 pass vs ${f(hop3)} with 3 (v1's hop by pass: 0.126 / 0.126 / 0.117)` : `live: hop ${f(e?.probes?.lr_hop?.exact)} (2 / 3 / 4 hops ${["exact_h2", "exact_h3", "exact_h4"].map(k => f(e?.probes?.lr_hop?.[k], 2)).join(" / ")}), tracking v1 · the per-pass test ${MS && !MS.items ? "is in the milestone evaluation's items part, still running" : when}`, pend],
        ["Worth its cost", "only the controlled arms decide: one-pass arm A1 vs full, A3, T3 (draft §9.4) — designed, not run", "arm"],
      ];
      const loopOver = has && fin(r1) ? `Active · 1 pass instead of 3 costs ${nat(r1)} (v1: 0.001)` : "Active · small gain · language test at 24 h";
      /* 2 · shared FFN */
      const op = Object.values(D.opened || {}).sort((a, b) => a.step - b.step).at(-1), tel = ev.filter(x => x.telemetry), w2 = tel.map(x => [x.step, x.telemetry.shared_w2_norm]);
      const w2d = w2.length > 3 ? (w2.at(-1)[1] - w2.at(-4)[1]) / 3 : null, sOff = msLang("shared_ffn_off"), sOffR = msProbe("shared_ffn_off", "lr_recall1"), rIn = msProbe("intact", "lr_recall1");
      const shRows = [
        ["Switched on", op ? `output weight moved off exactly 0 by step ${op.step} (${op.still_zero?.includes("shared_w2") ? "still zero" : `‖W₂‖ ${f(op.shared_w2, 2)}`}) · the step-300 check fixed before the run` : "—", op && !op.still_zero?.includes("shared_w2") ? "met" : "marg"],
        ["Still being trained into use", `‖W₂‖ ${f(w2.at(-1)?.[1], 2)} now${fin(w2d) ? `, ${w2d >= 0 ? "+" : "−"}${Math.abs(w2d).toFixed(2)} per evaluation` : ""} (0 at start, ${f(R.opened[0]?.shared_w2, 2)} at step ${R.opened[0]?.step ?? 100}) · a weight size, not usage`, "meas"],
        ["Its share of each write", "not logged in this run: the table's write gain α pools attention, own-FFN and shared-FFN writes", "nolog"],
        ["Carries load (switch-off test)", has ? `shared FFN off: next-token ${nat(sOff)}${fin(sOffR) && fin(rIn) ? ` · recall ${f(rIn)} → ${f(sOffR)}` : ""} · switching a trained branch off is out of distribution: a large cost proves it is used, a small one does not prove it is useless` : `shared-FFN-off test ${when}`, pend],
        ["Worth its cost", "the parameter-matched arm (own FFN 992, no shared FFN) decides — designed, not run", "arm"],
      ];
      const shOver = has && fin(sOff) ? `Switched on · off costs ${nat(sOff)}` : "Switched on · load test at 24 h";
      /* 3 · Engram */
      const al = e?.telemetry?.engram_alpha_mean ?? lastOf("engram_alpha"), wv1 = e?.telemetry?.engram_Wv_norm, wv2 = e?.telemetry?.engram2_Wv_norm;
      const eOff = msLang("engram_off"), e2Off = msLang("engram_read2_off");
      const enRows = [
        ["Memory gate open", `α ${f(al)} of the looked-up n-gram vector admitted (v1's page treated < 0.02 as unused)`, "meas"],
        ["Both reads switched on", op ? `W_v moved off 0 by step ${op.step} (read 1 ${f(op.Wv1, 2)}, read 2 ${f(op.Wv2, 2)}) · the step-300 check` : "—", op && !(op.still_zero || []).some(x => /Wv/.test(x)) ? "met" : "marg"],
        ["The second read (new in v3) is used", `‖W_v‖ read 2 ${f(wv2, 1)} vs read 1 ${f(wv1, 1)} (ratio ${fin(wv1) && fin(wv2) ? (wv2 / wv1).toFixed(2) : "—"}) · weight size, not usage`, "meas"],
        ["Carries load (switch-off test)", has ? `Engram off: next-token ${nat(eOff)} · second read only off: ${nat(e2Off)} · v1 and v2 never had this switch, so this is the first measurement` : `Engram-off and read-2-off tests ${when} · v1 and v2 never measured this`, pend],
      ];
      const enOver = has && fin(eOff) ? `In use · off costs ${nat(eOff)}` : "In use · load test at 24 h";
      /* 4 · v3 vs v1 */
      const v1e = REF.v1.evals, v1Last = v1e.at(-1), gE = dv.filter(x => fin(v1Gap(x).m)).at(-1), g1 = gE ? { ...v1Gap(gE), step: gE.step } : {}, worst = Math.max(g1.ts ?? -9, g1.wk ?? -9);
      const past = d1 && v1Last && d1.step > v1Last.step;                   // v3 is past v1's last evaluation (step 4,050)
      const gs = dv.filter(x => x.step >= 1217).map(x => v1Gap(x).m).filter(fin), gmin = gs.length ? Math.min(...gs) : null, gmax = gs.length ? Math.max(...gs) : null;
      const vW = d1 ? interpW(v1e, d1.wall_h, "val") : null, rW = d1 ? interpW(v1e, d1.wall_h, "recall1") : null;
      const tk = median(ser("tok_s").map(p => p[1])), tk1 = median((REF.v1.train || []).map(r => r.tok_s).filter(fin));
      const vs1 = (k, lab) => { const r = interp(v1e, d1?.step, k); return fin(r) ? `${lab} ${f(d1?.[k])} vs v1 ${f(r)}` : `${lab} ${f(d1?.[k])} vs v1's last ${f(v1Last?.[k])} (step ${fI(v1Last?.step)})`; };
      const vsRows = [
        ["Early gates (fixed before the run)", "E1 recall pass (0.594 at 1,792) · E2 language pass at 1,217 / 1,792 / 2,376 · E3 stability held · E4 speed held", "met"],
        ["Language at the same tokens", fin(g1.m) ? `${sgn(g1.m)} nats at step ${fI(g1.step)} (TinyStories ${sgn(g1.ts)}, WikiText ${sgn(g1.wk)}) · ${fin(gmin) ? `between ${sgn(gmin)} and ${sgn(gmax)} since step 1,217` : ""}${past ? ` · v1's last evaluation was step ${fI(v1Last.step)}, so this is the final same-step reading` : ""}` : "no same-step v1 evaluation yet", fin(g1.m) ? (worst <= 0.10 ? "met" : "marg") : "meas"],
        ["Language at the same hours", fin(vW) ? `${sgn(d1.val - vW)} nats at ${f(d1.wall_h, 1)} h (v3 ${f(d1.val)} vs v1 ${f(vW)}) · v3 runs ${fin(tk) && fin(tk1) ? (tk / tk1).toFixed(1) + "×" : "≈ 1.9×"} v1's tokens per hour` : "—", "meas"],
        ["Long-range recall", `${vs1("recall1", "recall1")}${past ? "" : " at the same step"}${fin(rW) ? ` · at the same hours v1 had ${f(rW)}` : ""} · plan swap ${f(d1?.swap)}`, "meas"],
        ["Harder probes", `${vs1("carry3", "carry3")} · ${vs1("multi4", "multi4")} · ${vs1("hop", "hop")}`, "meas"],
        ["Codebook", `${vs1("VE", "variance explained")} · codes in use ${fin(d1?.codes_min) ? f(100 * d1.codes_min, 0) + " %" : "—"}`, "meas"],
      ];
      const vsOver = fin(g1.m) ? `Ahead at the same hours · ${f(g1.m, 2)} behind at the same tokens${past ? " (last, step " + fI(g1.step) + ")" : ""}` : "Ahead at the same hours";
      el.innerHTML = [
        vcard("Is looping working?", loopOver, loopRows, "Tier 1 says the loop is active and its passes differ; the gain so far is small and measured on next-concept loss only. The 24 h evaluation adds the like-for-like next-token test; only the arms can say whether the loop is worth its compute."),
        vcard("Is the shared (Möbius) FFN useful?", shOver, shRows, "Only its weight size is logged live. The switch-off test at 24 h is the first measurement of its contribution."),
        vcard("Is Engram useful?", enOver, enRows, "The gate is wide open and both reads grew; whether the model depends on them is the switch-off test at 24 h."),
        vcard("v3 vs v1: are we doing well?", vsOver, vsRows, "Same-token comparison holds for the first epoch only (steps ≤ ≈ 8,556) and v1's step-matched evaluations end at 4,050. At 24 h v3 will read above v1's annealed final (2.754): expected, v3 is mid-schedule."),
      ].join("");
    } catch (err) { console.error("verdicts:", err); }
  }

  function setLiveCharts() {
    const s = (L?.steps || []).filter(r => r && fin(r.step)).slice(-600);
    CH.step.set([s.map(r => [r.step, r.ntp]), s.map(r => [r.step, r.ema]), s.map(r => [r.step, r.ncp])]);
    const now = L?.server_time || L?.t || Date.now() / 1000;
    CH.gpu.set([gpuSamples().map(([t, g]) => [+((t - now) / 60).toFixed(2), g])]);
    document.getElementById("v3-live-note").textContent = s.length ? `Every optimizer step, from the trainer's heartbeat (last ${s.length} steps).` : "The per-step feed comes from the v3 dashboard server on :8766, which is not reachable right now.";
  }

  /* ------------------------------------------------------------------ records */
  function renderRecords() {
    const ev = evs(), v1 = REF.v1.evals;
    const cl = x => !fin(x) ? "" : x <= 0.10 ? "ok" : x <= 0.20 ? "wa" : "ba";
    const th = (t, tip) => `<th title="${tip}">${t}</th>`;
    document.getElementById("v3-evals").innerHTML = ev.length ? `<thead><tr>${th("Step", "optimizer step")}${th("Hours", "training hours")}${th("Val loss", "mean of TinyStories / WikiText (nats)")}${th("TinyStories", "")}${th("WikiText", "")}${th("vs v1 (TS / WK)", "v3 − v1 at the same step; green ≤ +0.10, amber ≤ +0.20")}${th("Recall", "long-range recall1, exact")}${th("v1 recall", "")}${th("Plan swap", "recall with another document's plan")}${th("carry3", "")}${th("multi4", "")}${th("hop", "")}${th("bind", "")}${th("Codebook VE", "variance explained")}${th("NCP 3 / 1", "next-concept loss at loop depth 3 / 1")}${th("Eval s", "evaluation time")}</tr></thead><tbody>` +
      ev.slice().reverse().map(e => { const g = v1Gap(e);
        return `<tr${e.is_best ? ' class="best"' : ""}><td><b>${fI(e.step)}</b></td><td>${f(e.wall_h, 1)}</td><td><b>${f(e.val)}</b></td><td>${f(e.ts)}</td><td>${f(e.wk)}</td>
          <td><span class="${cl(g.ts)}">${sgn(g.ts)}</span> / <span class="${cl(g.wk)}">${sgn(g.wk)}</span></td><td><b>${f(e.recall1)}</b></td><td class="muted">${f(interp(v1, e.step, "recall1"))}</td><td>${f(e.swap)}</td>
          <td>${f(e.carry3)}</td><td>${f(e.multi4)}</td><td>${f(e.hop)}</td><td>${f(e.bind)}</td><td>${f(e.VE)}</td><td>${f(e.ncp, 2)} / ${f(e.ncp_M1, 2)}</td><td>${f(e.eval_sec, 0)}</td></tr>`; }).join("") + "</tbody>"
      : `<tbody><tr><td class="muted">The first evaluation runs at step 185 (v1's first eval step), then at every v1 eval step and hourly.</td></tr></tbody>`;
    const op = Object.values(D.opened || {}).sort((a, b) => a.step - b.step);
    document.getElementById("v3-opened").innerHTML = op.length ? `<table class="t"><tr><th>Step</th><th>(t, Δ) table</th><th>Shared FFN out</th><th>Engram W_v1</th><th>Engram W_v2</th><th>Still zero</th></tr>` +
      op.map(o => `<tr><td>${o.step}</td><td>${f(o.table)}</td><td>${f(o.shared_w2)}</td><td>${f(o.Wv1)}</td><td>${f(o.Wv2)}</td><td>${o.still_zero?.length ? `<b style="color:var(--bad)">${o.still_zero.join(", ")}</b>` : '<span style="color:var(--good)">none ✓</span>'}</td></tr>`).join("") + "</table>"
      + `<div class="muted" style="font-size:12px;margin-top:8px">Weight norms of the parts that start at exactly zero. Checked at steps 100, 200 and 300; every one must have moved.</div>`
      : `<div class="muted">Logged at steps 100, 200 and 300.</div>`;
    const c = D.config || {};
    document.getElementById("v3-setup").innerHTML = `
      <span>model</span><span>19.0M compute parameters (encoder 9.6M · loop core 4.9M · decoder 4.5M) + 80M-entry Engram table</span>
      <span>budget</span><span>48 h of training on the Apple M2 (MPS), milestone evaluation at 24 h</span>
      <span>batch</span><span>${c.micro_bs ?? 2} × ${c.accum ?? 16} accumulation × 1,024 = 32,768 tokens per step</span>
      <span>schedule</span><span>peak LR ${c.lr ?? 6e-4}, warm-up ${c.warmup ?? 300} steps, time-based cosine to 6e-5</span>
      <span>speed-ups</span><span>block attention ${c.block_attn ?? "—"} · torch.compile ${c.compile ?? "—"} (+54 % measured before launch)</span>
      <span>comparison</span><span>same data order as v1 (seed 0): step k = same tokens. Holds for the first epoch only (steps ≤ ≈ 8,556)</span>
      <span>caveat</span><span>the 24 h numbers are mid-schedule (LR ≈ 3.3e-4) against v1's fully annealed final</span>
      <span>tokens</span><span>throughput counts slots including PAD (5.7 %); tokens ≈ 0.943 × slots</span>
      <span>48 h total</span><span>${D.total_steps ? `≈ ${Math.round(D.total_steps * TOK_PER_STEP * 0.943 / 1e6)}M tokens (${f(D.total_steps * TOK_PER_STEP / SLOTS_EPOCH, 2)} epochs, ≈ ${fI(D.total_steps)} steps at the current speed; updates as speed changes) · for scale, v1's main run was 4,050 steps = 132.7M slots in 24 h` : "—"}</span>`;
  }

  /* ------------------------------------------------------------------ section chips (scoped to this tab) */
  function renderNav() {
    const nav = document.getElementById("v3-nav"), secs = [...document.querySelectorAll("#tab-v3train .trp-sec")];
    nav.innerHTML = secs.map(sec => {
      const lv = [...sec.querySelectorAll(".chart")].map(c => ["g", "a", "r"].find(k => c.classList.contains("st-" + k))).concat([...sec.querySelectorAll(".hl")].map(h => ["g", "a", "r"].find(k => h.classList.contains(k)))).filter(Boolean);
      const n = { g: 0, a: 0, r: 0 }; lv.forEach(k => n[k]++);
      const w = n.r ? "r" : n.a ? "a" : n.g ? "g" : "n", sum = sec.querySelector(".trp-sum");
      if (sum) sum.innerHTML = chips(n);
      return `<button data-go="${sec.id}" class="${nav.dataset.active === sec.id ? "on" : ""}"><span class="nd ${w}"></span>${sec.dataset.title}</button>`;
    }).join("");
  }
  window.addEventListener("scroll", () => {
    const tab = document.getElementById("tab-v3train"), nav = document.getElementById("v3-nav"); if (!tab || !tab.classList.contains("on") || !nav) return;
    let cur = null; tab.querySelectorAll(".trp-sec").forEach(s => { if (s.getBoundingClientRect().top < 200) cur = s.id; });
    if (cur !== nav.dataset.active) { nav.dataset.active = cur || ""; nav.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.go === cur)); }
  }, { passive: true });

  /* ------------------------------------------------------------------ render and poll */
  function render(fullData) {
    if (!CH) CH = mk();
    if (fullData) { setCharts(); renderRecords(); } else { setLoop(); Object.values(CH).forEach(c => c.updStatus()); }
    setLiveCharts();
    const K = checks();
    renderHero(K); renderKpis(K); renderHealth(K);
    renderVerdicts();
    renderNav();
    pill();
  }
  function renderEmpty() {
    document.getElementById("v3-now").innerHTML = `<div class="state-chip"><span class="dot"></span>v3 · no data</div><div class="stepbig">v3 run not started</div>
      <div class="hero-meta"><span>This page fills in when <code>data/v3_live.json</code> is written by <code>run_v3_19M/code/train.py</code>.</span></div>`;
  }
  async function fetchApi() {
    const ctl = new AbortController(), to = setTimeout(() => ctl.abort(), 5000);
    try { const r = await fetch(LIVE_API, { cache: "no-store", signal: ctl.signal }); if (!r.ok) return null; return JSON.parse((await r.text()).replace(/\bNaN\b|-?\bInfinity\b/g, "null")); }
    catch (e) { return null; } finally { clearTimeout(to); }
  }
  async function poll() {
    if (document.hidden && keyD) return;                          // first load always draws; later refreshes wait for a visible window
    try { if (!REF) REF = await (await fetch("data/ref_v1v2.json", { cache: "no-store" })).json(); }
    catch (e) { const el = document.getElementById("v3-now"); if (el && !CH) el.innerHTML = '<div class="stepbig">v1 / v2 reference data not readable</div><div class="hero-meta"><span>data/ref_v1v2.json could not be loaded; retrying every 30 s.</span></div>'; return; }
    const [d, l, runsChanged, msChanged] = await Promise.all([fetchLive("data/v3_live.json").catch(() => null), fetchApi(), pollRuns(), pollMS()]);
    if (!d) return renderEmpty();
    backfill(d);
    const kd = d.step + ":" + (d.evals || []).length + ":" + d.t, kl = l ? (l.heartbeat?.step ?? "") + ":" + (l.hist || []).length + ":" + (l.hist || []).at(-1)?.[0] : "none";
    if (kd === keyD && kl === keyL) { if ((runsChanged || msChanged) && CH) { setLoop(); renderVerdicts(); renderNav(); } return; }   // nothing new: no redraw (keeps the GPU for training)
    const full = kd !== keyD; keyD = kd; keyL = kl; D = d; L = l;
    render(full);
  }
  new MutationObserver(() => { if (CH && document.getElementById("tab-v3train").classList.contains("on")) Object.values(CH).forEach(c => c.draw()); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  document.addEventListener("visibilitychange", () => { if (!document.hidden && timer) poll(); });

  /* ------------------------------------------------------------------ header pill (read by setPill in js/telemetry.js) */
  async function pill() {
    try {
      const d = await fetchLive("data/v3_live.json");
      if (!(d.evals || []).length) {                                   // live file restarted with the process: last val from the log
        const ev = R.evals.length ? R.evals : await fetchJsonl("data/v3_evals.jsonl").catch(() => []);
        d.evals = ev.filter(e => e.val_lang_mean != null).slice(-1).map(evalRow);
      }
      const hb = L?.heartbeat && L.heartbeat.step >= d.step ? L.heartbeat : null;   // per-step heartbeat when this page has it
      window.V3PILL = { step: hb ? hb.step : d.step, tok_s: d.tok_s, val: (d.evals || []).at(-1)?.val, age: (Date.now() / 1000 - (hb ? hb.t : d.t)) / 60 };
    } catch (e) { /* keep the last value */ }
    if (typeof setPill === "function" && document.documentElement.dataset.ver === "v3") setPill();
    const dot = document.getElementById("tab-live-dot");
    if (dot && window.V3PILL) dot.className = "tab-live" + (window.V3PILL.age < 15 ? " on" : "");
  }
  setInterval(() => { if (document.documentElement.dataset.ver === "v3" && !document.hidden) pill(); }, 30000);
  pill(); setTimeout(pill, 1500);                                   // the first "tabshown" can fire before this script loads
  window.addEventListener("tabshown", () => { if (document.documentElement.dataset.ver === "v3") pill(); });
  function onShown(id) {
    if (id !== "v3train") { if (timer) { clearInterval(timer); timer = null; } return; }
    keyD = keyL = keyR = null; poll(); if (!timer) timer = setInterval(poll, REMOTE ? 60000 : 30000);   // away from the Mac: every 60 s (mobile data)
  }
  window.addEventListener("tabshown", e => onShown(e.detail));
  const cur = (document.querySelector("section.tab.on") || {}).id;
  if (cur === "tab-v3train") onShown("v3train");
})();
