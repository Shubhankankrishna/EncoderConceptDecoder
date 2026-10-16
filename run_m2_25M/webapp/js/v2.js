/* v2 pages: overview, 2D figures, live training (the 3D page is js/view3d_v2.js). Data: data/arch_v2.json (model_v2/make_arch_v2.py).
   Live: data/v2_live.json, written by the future v2 trainer (contract shown on the page). Never touches the v1 run. */
(function () {
  let A2 = null, figsDone = false, liveTimer = null;
  const pct = x => (100 * x).toFixed(1) + " %";
  const M = n => (n / 1e6).toFixed(2) + "M";

  function bar(arr) {
    const tot = arr.reduce((s, a) => s + a[1], 0);
    return `<div class="pbar">${arr.map(([n, v, c]) => `<div style="width:${100 * v / tot}%;background:${c}" title="${n}: ${fmtI(Math.round(v))} (${pct(v / tot)})">${v / tot > 0.1 ? `${n} ${(100 * v / tot).toFixed(0)}%` : ""}</div>`).join("")}</div>`;
  }
  const CE = "#d6e6f7", CC = "#d2efea", CD = "#fbdcc8", CH = "#e6e0f5";

  function renderOverview() {
    const A = A2, T = A.table, P = A.params, v1 = A.v1, m = A.macs, c = A.core;
    document.getElementById("v2-eb").textContent = `${fmtI(T.compute_params)} compute parameters`;
    document.getElementById("v2-stats").innerHTML = [
      ["compute params", M(T.compute_params), `v1 ${M(v1.compute)} · same convention`],
      ["stored params", M(T.stored_total), `+ Engram ${M(T.engram_table)}, embedding, codebook`],
      ["width d", A.ecfg.d_model, "4 heads × 64 · GQA 4:2 (v1: 384)"],
      ["concept share", pct(T.concept_module / T.compute_params), `of compute params (v1: ${pct(v1.core / v1.compute)})`],
      ["forward MACs / token", M(m.fwd_eval), `eval R 6 (v1 ≈ ${M(v1.macs_eval_R4)} at R 4)`],
      ["decoder reach", A.reach_dec + " tokens", `encoder L stream ${A.reach_enc} (unchanged)`],
    ].map(([l, v, s]) => `<div class="card stat"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
    document.getElementById("v2-hero-side").innerHTML = `<div class="card" style="height:100%">
      <div class="l" style="font-size:12px;color:var(--ink3);text-transform:uppercase;letter-spacing:.08em;font-weight:650">Status</div>
      <div style="font-size:22px;font-weight:700;margin:6px 0 2px">Design · not trained</div>
      <div class="muted" style="font-size:13px">Spec: <code>model_v2/MODEL_V2_19M.md</code>. Next: implement, then 265-step screens (throughput, NextLat keep/kill, D6 checks), then the main run.</div>
      <div class="v2cmp"><span class="a">v1 run</span><span></span><span>best val ${v1.val_best} (continuation)</span>
        <span class="a">training cost</span><span></span><span>${m.train_vs_today.toFixed(2)} × v1 per token</span>
        <span class="a">tok/s</span><span></span><span>≤ ${fmtI(A.est_tok_s_upper_bound[0])}–${fmtI(A.est_tok_s_upper_bound[1])} (upper bound)</span></div></div>`;
    const card = (cls, n, t, sub, p, x, facts, old) => `<div class="story-card ${cls}"><div class="story-num">${n}</div><h4>${t} <span>${sub}</span></h4><p>${x}</p>
      <div class="facts">${facts.map(f => `<span>${f}</span>`).join("")}</div><div class="v2cmp"><span class="a">v1</span><span>→</span><span class="n">v2</span><span class="o">${old}</span><span></span><span class="n">${M(p)}</span></div></div>`;
    document.getElementById("v2-story").innerHTML =
      card("c-enc", 1, "Encoder", "same design, d 256", T.encoder_non_embedding, "Two streams (local L, reach 63; global G), SSSFSSSF, Engram after layer 0, CRC mix and 4 → 1 pooling into 384-dim concepts. One third narrower; local knowledge moves to the 80M Engram table.", ["8 layers", "d_ff 704", `Engram ${M(T.engram_table)}`, "ĉ 384-d"], M(v1.encoder)) +
      `<div class="story-arrow">→</div>` +
      card("c-core", 2, "Concept module", "D6 · looped experts", T.concept_module, "Four distinct full-attention layers, looped 8–12 times in training (6 at eval) in detached windows of 4, every pass graded on the next chunk's code; a shared pool of 16 experts chosen by the state; LoopFormer consistency in the final window.", ["4 layers × K", `${c.ffn.E} experts, top-2`, "active 1,024", "R 6 eval"], M(v1.core)) +
      `<div class="story-arrow">→</div>` +
      card("c-dec", 3, "Decoder", "confined + NextLat", T.decoder, "Unchanged confined layout: each position sees the last 70 tokens plus its own plan. NextLat (training only) predicts the next hidden state from the current one and the next token, which pushes the plan to carry the history.", ["4 layers", "reach 70", `NextLat ψ ${M(P.nextlat_psi)}`, "tied head"], M(v1.decoder));
    const act = A.active, ms = A.mac_share, sh = A.rows.C_E16.share;
    document.getElementById("v2-props").innerHTML = `
      <div class="card v2bars"><h4>Share of compute parameters</h4>
        <div class="brow"><span>v1 (d 384)</span>${bar([["encoder", v1.encoder, CE], ["core", v1.core, CC], ["decoder", v1.decoder, CD]])}</div>
        <div class="brow"><span><b>v2 (d 256)</b></span>${bar([["encoder", T.encoder_non_embedding, CE], ["core", T.concept_module, CC], ["decoder", T.decoder, CD]])}</div>
        <div class="muted" style="font-size:12px">v1 62 / 8 / 30 % → v2 ${(sh.encoder * 100).toFixed(0)} / ${(sh.core * 100).toFixed(0)} / ${(sh.decoder * 100).toFixed(0)} %. The core is no longer the ant.</div></div>
      <div class="card v2bars"><h4>Active parameters and compute (v2)</h4>
        <div class="brow"><span>active params / token</span>${bar([["encoder", act.encoder, CE], ["core", act.core, CC], ["decoder", act.decoder, CD]])}</div>
        <div class="brow"><span>forward MACs / token</span>${bar([["encoder", m.encoder, CE], ["core", m.core_eval, CC], ["decoder", m.decoder, CD], ["head", m.head, CH]])}</div>
        <div class="muted" style="font-size:12px">With 2 of 16 experts active, the core holds 42 % of the parameters but uses 30 % per token; it does 35 % of the compute at R 6.</div></div>`;
    const CH_ = [
      ["Width", "d 384 → 256", "Width costs compute everywhere; depth-by-recurrence and lookups do not. 4 heads is the minimum that keeps GQA 4:2.", "NTP at 8.7M tokens vs v1's screens (tie band 0.02)"],
      ["Concept module", "1 shared layer × R 1–4 → D6: 4 layers × K 8–12, 16 experts", "v1's loop was idle (R 1..8 equal, step angle 0.1°). D6 fixes all four audit defects and gives each pass a state-routed computation.", "E16 routing divergence, A4 dense twin, R-sweep not flat"],
      ["Proportions", "62 / 8 / 30 % → 39 / 42 / 19 %", "The core is the only long-range path and the measured bottleneck; spare parameters go to experts (constant compute).", "E15: no pool collapse at 16 experts (else row A, E 8)"],
      ["Engram", "50.3M → 80.0M (rows 208,333)", "Nearly free in compute; the v1 gate is open (α 0.77) so the table is used; more rows = fewer collisions.", "swap at the screen; eval with the Engram read off"],
      ["Concept width", "d_c 384 kept", "256 is not divisible by the 48 PQ segments; 384 keeps the measured codebook (codes_min 0.99).", "codes_min ≥ 0.95 at the screen"],
      ["Decoder", "same layout, d_ff 704", "The user asked to keep it; same FFN ratio as v1 (2.75).", "if NTP lags with good concept metrics → row D"],
      ["NextLat", "off → on (d 1, λ 1 / 1)", "In a confined decoder the plan is the only channel for history, so belief-state pressure lands on the concept stream.", "keep only if NTP within 0.02 AND a long-range metric improves"],
      ["Losses", "3 → 8 terms", "Per-pass NCP, consistency, MoE balance and NextLat each target a specific failure.", "gradient share per term at steps 50 and 265"],
    ];
    document.getElementById("v2-changes").innerHTML = `<tr><th>part</th><th>v1 → v2</th><th>why (first principles)</th><th>deciding check</th></tr>` +
      CH_.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`).join("");
    document.getElementById("v2-changes").classList.add("v2t");
    const R = A.rows, nm = { A_D6_as_scored_E8: "A · D6 as scored (E 8)", B_E12: "B · E 12", C_E16: "C · E 16 (chosen)", D_E16_dec1024: "D · E 16 + decoder 1024" };
    document.getElementById("v2-rows").innerHTML = `<table class="t"><tr><th>row</th><th>compute</th><th>enc / core / dec</th><th>core MACs</th></tr>` +
      Object.keys(nm).map(k => `<tr${k === "C_E16" ? ' style="font-weight:700"' : ""}><td>${nm[k]}</td><td>${M(R[k].params.compute)}</td><td>${(R[k].share.encoder * 100).toFixed(0)} / ${(R[k].share.core * 100).toFixed(0)} / ${(R[k].share.decoder * 100).toFixed(0)} %</td><td>${pct(R[k].core_share)}</td></tr>`).join("") +
      `</table><div class="muted" style="font-size:12px;margin-top:6px">Row C meets all three constraints: decoder unchanged, ≈ 19M, D6 mechanism unchanged (only the pool size moves). Row A is the fallback if 16 experts collapse.</div>`;
    document.getElementById("v2-loss").innerHTML = `<table class="t">${A.losses.map(l => `<tr><td>${l.name}</td><td><b>${l.weight}</b></td><td style="font-size:12px">${l.formula}</td></tr>`).join("")}</table>
      <div class="muted" style="font-size:12px;margin-top:6px">L_total = the weighted sum of all terms. NTP / NCP / VQ keep v1's weights; MoE weights are the Switch / ST-MoE values; NextLat weights are the paper's.</div>`;
    document.getElementById("v2-compute").innerHTML = bar([["encoder", m.encoder, CE], ["core", m.core_eval, CC], ["decoder", m.decoder, CD], ["head", m.head, CH]]) + `<table class="t" style="margin-top:8px">
      <tr><td>forward MACs / token (eval R 6)</td><td><b>${M(m.fwd_eval)}</b></td></tr><tr><td>core in training (K̄ 10, per-pass head)</td><td>${M(m.core_train)}</td></tr>
      <tr><td>NextLat (training only)</td><td>${M(m.nextlat_train_equiv)} MAC-equiv.</td></tr><tr><td>training compute vs v1</td><td><b>${m.train_vs_today.toFixed(2)} ×</b></td></tr>
      <tr><td>tok/s (MACs-proportional)</td><td>≤ ${fmtI(A.est_tok_s_upper_bound[0])}–${fmtI(A.est_tok_s_upper_bound[1])}</td></tr></table>
      <div class="muted" style="font-size:12px;margin-top:6px">A smaller model does not train faster here: D6 trains ~10 passes, so the cost per token ≈ v1's. The tok/s figure is an upper bound (MPS handles D6's many small kernels poorly); a 265-step throughput screen with a 1,200 tok/s floor comes first.</div>`;
    document.getElementById("v2-risks").innerHTML = `<table class="t"><tr><th>result at the screen</th><th>fallback</th></tr>
      <tr><td>NextLat: NTP worse &gt; 0.02, no long-range gain</td><td>λ_h = λ_KL = 0 (rest unchanged)</td></tr>
      <tr><td>pool collapse with 16 experts (E15)</td><td>row A: 8 experts, 15.68M</td></tr>
      <tr><td>throughput &lt; 1,200 tok/s</td><td>λ_KL = 0; then K ~ U{8..10} (a design decision)</td></tr>
      <tr><td>flat R-sweep, no routing divergence, dense twin ≈ D6</td><td>D3 (dense FFN): simpler, equal score</td></tr>
      <tr><td>NTP &gt; 0.02 worse than v1 at equal tokens, no long-range gain</td><td>width 256 is wrong → revisit at d 384</td></tr></table>`;
  }

  /* ---------------------------------------------------------------- 2D figures */
  function gotoFig2(id) {
    showTab("v2arch2d");
    document.querySelectorAll("#fig2-nav button").forEach(x => x.classList.toggle("on", x.dataset.id === id));
    const f = document.getElementById("fig2-" + id); if (f) requestAnimationFrame(() => f.scrollIntoView({ block: "start" }));
  }
  function open3D2(view) {
    showTab("v2arch3d"); window.scrollTo(0, 0);
    let n = 0; const t = setInterval(() => { const b = document.querySelector(`#v3b-bar [data-view="${view}"]`); if ((b && document.querySelector("#v3b-info h5")) || ++n > 40) { clearInterval(t); if (b) b.click(); } }, 100);
  }
  window.gotoFig2 = gotoFig2;
  function renderFigs2() {
    if (figsDone) return; figsDone = true;
    const nav = document.getElementById("fig2-nav"), box_ = document.getElementById("figs2");
    const ctx = { figs: FIGS2, go: gotoFig2, open3d: open3D2, key: FIG2_KEY, map3d: FIG2_3D };
    FIGS2.forEach((f, i) => {
      const b = document.createElement("button"); b.innerHTML = `<span class="fn">${i + 1}</span>${f.t}`; b.dataset.id = f.id; if (!i) b.classList.add("on");
      b.onclick = () => gotoFig2(f.id); nav.appendChild(b);
      const fig = document.createElement("figure"); fig.className = "fig"; fig.id = "fig2-" + f.id;
      const h = document.createElement("h3"); h.className = "figt"; h.innerHTML = `<span class="figt-n">${i + 1}</span>${f.t}`; fig.appendChild(h);
      try { fig.appendChild(f.fn(A2)); } catch (e) { const p = document.createElement("p"); p.className = "muted"; p.textContent = "figure error: " + e.message; fig.appendChild(p); console.error(e); }
      const cap = document.createElement("figcaption"); cap.innerHTML = `<b>Figure ${i + 1}.</b> ${f.cap}`; fig.appendChild(cap);
      box_.appendChild(fig);
      figTools(fig, f.id, i, FIGS2.length, ctx);
    });
    window.addEventListener("scroll", () => {
      if (!document.getElementById("tab-v2arch2d").classList.contains("on")) return;
      let cur = null; document.querySelectorAll("#figs2 figure.fig").forEach(f => { if (f.getBoundingClientRect().top < innerHeight * 0.4) cur = f.id.slice(5); });
      if (cur) document.querySelectorAll("#fig2-nav button").forEach(x => x.classList.toggle("on", x.dataset.id === cur));
    }, { passive: true });
  }

  /* ---------------------------------------------------------------- live training */
  const CHARTS = [
    ["total", "Total loss", "weighted sum of all 8 terms"], ["ntp", "NTP", "next-token cross-entropy"], ["ncp", "NCP (mean over passes)", "next-concept code CE, every pass"],
    ["ncp_pass", "NCP by pass", "one line per pass r: should fall with r"], ["cons", "Consistency (final window)", "‖sg(u_K) − u_r‖² / d"], ["vq", "VQ", "codebook loss"],
    ["nl_h", "NextLat next-h", "SmoothL1; may rise in LR cooldown (paper §E): no falling-loss verdict"], ["nl_kl", "NextLat KL", "prior vs posterior next-token distribution"], ["lb", "MoE load balance + z", "per window"],
    ["route_ent", "Expert load entropy", "E15: collapse if it falls toward 0"], ["drop", "Dropped tokens", "per layer per pass, target < 2 %"], ["route_div", "Routing divergence", "E11 / E16: across passes and windows, > 0"],
    ["val", "Validation NTP", "hourly; vs v1 at equal tokens"], ["recall", "recall1 / carry3 / multi4", "long-range probes vs plan swap"], ["toks", "Throughput", "tok/s, floor 1,200"],
  ];
  function spark(pts) {
    if (!pts || pts.length < 2) return "";
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]), x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const px = x => 4 + (x - x0) / (x1 - x0 || 1) * 292, py = y => 122 - (y - y0) / (y1 - y0 || 1) * 110;
    return `<svg viewBox="0 0 300 130"><path d="${pts.map((p, i) => `${i ? "L" : "M"}${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`).join(" ")}" fill="none" stroke="#7a5bd0" stroke-width="2"/></svg>
      <div class="muted" style="font-size:11.5px">last ${ys[ys.length - 1].toFixed(4)} at step ${xs[xs.length - 1]}</div>`;
  }
  async function pollLive() {
    let d = null;
    try { const r = await fetch("data/v2_live.json?t=" + Date.now(), { cache: "no-store" }); if (r.ok) d = await r.json(); } catch (e) {}
    const st = document.getElementById("v2t-status");
    if (!d) {
      st.innerHTML = `<div class="card v2status"><div><div class="big">No v2 run yet</div><div class="muted">The v1 continuation is still training (see the v1 → Live training page). This page activates when <code>webapp/data/v2_live.json</code> appears.</div></div><span class="v2tag a" style="margin-left:auto">waiting</span></div>`;
      document.getElementById("v2t-kpis").innerHTML = [["phase", "design"], ["next", "implementation"], ["then", "3 screens × 265 steps"]].map(([a, b]) => `<div class="card stat"><div class="l">${a}</div><div class="v" style="font-size:20px">${b}</div></div>`).join("");
      document.getElementById("v2t-charts").innerHTML = CHARTS.map(([k, t, s]) => `<div class="card v2chart"><h4>${t}</h4><div class="muted" style="font-size:12px">${s}</div><div class="ph">waiting for the v2 run</div></div>`).join("");
      return;
    }
    const age = d.t ? Math.round((Date.now() / 1000 - d.t) / 60) : null;
    st.innerHTML = `<div class="card v2status"><div><div class="big">${d.phase || "training"} · step ${fmtI(d.step)}${d.total_steps ? " / " + fmtI(d.total_steps) : ""}</div>
      <div class="muted">${d.time || ""}${age != null ? ` · ${age} min ago` : ""} · lr ${d.lr != null ? d.lr.toExponential(2) : "–"} · gn ${d.gn != null ? d.gn.toFixed(2) : "–"}</div></div>
      <span class="v2tag ${age != null && age < 20 ? "g" : "a"}" style="margin-left:auto">${age != null && age < 20 ? "live" : "stale"}</span></div>`;
    document.getElementById("v2t-dot").style.background = age != null && age < 20 ? "#2e9c8c" : "#d1a92f";
    document.getElementById("v2t-kpis").innerHTML = [["NTP", d.ntp], ["total", d.loss], ["tok/s", d.tok_s]].map(([a, b]) => `<div class="card stat"><div class="l">${a}</div><div class="v">${b != null ? (+b).toFixed(a === "tok/s" ? 0 : 4) : "–"}</div></div>`).join("");
    const H = d.history || {};
    const multi = (keys, cols) => {                               // several series in one panel (per-pass NCP)
      const ser = keys.map(k => H[k] || []).filter(s => s.length > 1); if (!ser.length) return "";
      const all = ser.flat(), xs = all.map(p => p[0]), ys = all.map(p => p[1]);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      const px = x => 4 + (x - x0) / (x1 - x0 || 1) * 292, py = y => 122 - (y - y0) / (y1 - y0 || 1) * 110;
      return `<svg viewBox="0 0 300 130">${ser.map((s, i) => `<path d="${s.map((p, j) => `${j ? "L" : "M"}${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`).join(" ")}" fill="none" stroke="${cols[i]}" stroke-width="1.8"/>`).join("")}</svg>
        <div class="muted" style="font-size:11.5px">${keys.map((k, i) => `<span style="color:${cols[i]}">■</span> ${k.replace("ncp_p", "pass ")} ${(H[k] || []).length ? (+H[k].at(-1)[1]).toFixed(3) : "–"}`).join(" · ")}</div>`;
    };
    document.getElementById("v2t-charts").innerHTML = CHARTS.map(([k, t, s]) => {
      const body = k === "ncp_pass" ? multi(["ncp_p1", "ncp_p4", "ncp_p6", "ncp_p8"], ["#c9a0dc", "#7a5bd0", "#2e9c8c", "#e07b39"]) : spark(H[k]);
      return `<div class="card v2chart"><h4>${t}</h4><div class="muted" style="font-size:12px">${s}</div>${body || `<div class="ph">not logged yet</div>`}</div>`;
    }).join("");
  }
  function renderLiveStatic() {
    document.getElementById("v2t-screens").innerHTML = `<table class="t"><tr><th>screen</th><th>pass condition</th></tr>
      <tr><td>1 · throughput &amp; memory</td><td>≥ 1,200 tok/s and swap &lt; 3 GB; else λ_KL = 0</td></tr>
      <tr><td>2 · NextLat keep / kill (twins on the D6 core)</td><td>kill if NTP worse &gt; 0.02; keep only if within 0.02 AND recall1 / carry3 / future-token probe improves</td></tr>
      <tr><td>3 · D6 checks</td><td>E15 (no collapse, drops &lt; 2 %), E16 &gt; 0, A4 dense twin worse, A2 consistency off</td></tr></table>`;
    document.getElementById("v2t-contract").textContent = JSON.stringify({ phase: "screen1 | screen2 | screen3 | main", step: 120, total_steps: 4050, time: "2026-10-01 10:00:00", t: 1790000000,
      lr: 6e-4, gn: 0.3, tok_s: 1500, loss: 5.1, ntp: 2.3,
      history: { total: [[50, 5.9]], ntp: [[50, 2.9]], ncp: [[50, 3.4]], ncp_pass: "…", cons: [[50, 0.02]], vq: [[50, 2.7]], nl_h: [[50, 0.1]], nl_kl: [[50, 0.5]], lb: [[50, 1.01]],
        route_ent: [[50, 2.7]], drop: [[50, 0.01]], route_div: [[50, 0.4]], val: [[180, 2.9]], recall: [[180, 0.9]], toks: [[50, 1500]] } }, null, 1);
  }

  /* v2 run record (the run ended 2026-09-29 23:13; 8766 now serves v3, so nothing here polls it) */
  window.V2PILL = { state: "ended", step: 3089, val: 3.1074 };
  async function renderRecord() {
    const el_ = document.getElementById("v2t-record"); if (!el_ || el_.dataset.done) return; el_.dataset.done = 1;
    let R = null; try { R = await (await fetch("data/ref_v1v2.json")).json(); } catch (e) {}
    if (!R) { el_.innerHTML = '<div class="card">reference data missing (data/ref_v1v2.json)</div>'; return; }
    const v2 = R.v2, f = v2.final, ev = v2.evals, f3_ = x => x == null ? "–" : (+x).toFixed(3);
    el_.innerHTML = `<div class="grid g6">${[["last step", "3,089", "24.05 h of 48 h"], ["val (quick)", f3_(f.val), `TS ${f3_(f.ts)} · WK ${f3_(f.wk)}`],
      ["recall1 (quick)", f3_(f.recall1), "at the 24 h milestone"], ["recall1 (test, partial 24 h eval)", f3_(f.recall1_test_24h), "OOD items not run"],
      ["throughput", "≈ 1,390", "tok/s (slots)"], ["status", "ended", "owner decision, v3 took the GPU"]]
      .map(([l, v, s_]) => `<div class="card stat"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s_}</div></div>`).join("")}</div>
      <div class="card scroll-x tcard" style="margin-top:14px"><table class="t"><thead><tr><th>step</th><th>h</th><th>val</th><th>TS</th><th>WK</th><th>recall1</th><th>carry3</th><th>multi4</th><th>plan-swap</th><th>VE</th><th>codes_min</th></tr></thead><tbody>${
      ev.slice().reverse().map(e => `<tr><td>${fmtI(e.step)}</td><td>${e.wall_h != null ? e.wall_h.toFixed(1) : "–"}</td><td><b>${f3_(e.val)}</b></td><td>${f3_(e.ts)}</td><td>${f3_(e.wk)}</td><td><b>${f3_(e.recall1)}</b></td><td>${f3_(e.carry3)}</td><td>${f3_(e.multi4)}</td><td>${f3_(e.swap)}</td><td>${f3_(e.VE)}</td><td>${f3_(e.codes_min)}</td></tr>`).join("")}</tbody></table></div>
      <p class="muted" style="font-size:12px;margin-top:8px">Source: run_v2_19M/runs/v2_48h/log.jsonl (hourly quick evals: 128 rows per language, 128 items per probe family). Decision record: run_v2_19M/DECISIONS.md V-012.</p>`;
  }

  async function ensure() { if (!A2) { A2 = await (await fetch("data/arch_v2.json")).json(); window.ARCH2 = A2; } return A2; }
  async function onShown(id) {
    if (!id.startsWith("v2")) { if (liveTimer) { clearInterval(liveTimer); liveTimer = null; } return; }
    await ensure();
    if (id === "v2overview" && !document.getElementById("v2-story").innerHTML) renderOverview();
    if (id === "v2arch2d") renderFigs2();
    if (id === "v2train") renderRecord();
  }
  window.addEventListener("tabshown", e => onShown(e.detail));
  document.addEventListener("click", e => { const a = e.target.closest("[data-fig2]"); if (a) { e.preventDefault(); ensure().then(() => { renderFigs2(); gotoFig2(a.dataset.fig2); }); } });
  const cur = (document.querySelector("section.tab.on") || {}).id;
  if (cur && cur.startsWith("tab-v2")) onShown(cur.slice(4));
})();
