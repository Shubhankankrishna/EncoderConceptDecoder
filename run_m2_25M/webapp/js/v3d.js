/* v3 draft pages (draft_v3.md): overview and 2D figures. The 3D page is js/view3d_v3d.js. Data: data/arch_v3d.json
   (v3_design/make_arch_v3draft.py). Static design content: v1/v2 numbers are labelled measurements, nothing here is a v3 result. */
(function () {
  let A3 = null, figsDone = false;
  const M = n => (n / 1e6).toFixed(2) + "M";
  const pct = x => (100 * x).toFixed(1) + " %";
  const stat = (l, v, s) => `<div class="card stat"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
  const CE = "#d6e6f7", CC = "#d2efea", CD = "#fbdcc8", CH = "#e6e0f5";
  function bar(arr) {
    const tot = arr.reduce((s, a) => s + a[1], 0);
    return `<div class="pbar">${arr.map(([n, v, c]) => `<div style="width:${100 * v / tot}%;background:${c}" title="${n}: ${fmtI(Math.round(v))} (${pct(v / tot)})">${v / tot > 0.1 ? `${n} ${(100 * v / tot).toFixed(0)}%` : ""}</div>`).join("")}</div>`;
  }

  function renderOverview() {
    const A = A3, T = A.table, P = A.params, c = A.core, mc = A.macs, v1 = A.v1, v2 = A.v2;
    document.getElementById("v3-stats").innerHTML = [
      stat("compute params", M(T.compute_params), "within the 19M cap · v1's convention"),
      stat("stored params", M(T.stored_total), `+ Engram ${M(T.engram_table)}, embedding, codebook`),
      stat("width d = d_c", A.ecfg.d_model, "S / decoder 4 heads · F / core 6 heads"),
      stat("concept share", pct(T.concept_module / T.compute_params), `v1 ${pct(v1.core / v1.compute)} · v2 ${pct(v2.core / v2.compute)}`),
      stat("recurrence", "M ≤ 3", "(t, Δ) conditioned · shortcut 1 / 2"),
      stat("decoder reach", A.reach_dec + " tokens", `encoder L stream ${A.reach_enc}`),
    ].join("");
    document.getElementById("v3-hero-side").innerHTML = `<div class="card" style="height:100%">
      <div class="l" style="font-size:12px;color:var(--ink3);text-transform:uppercase;letter-spacing:.08em;font-weight:650">Status</div>
      <div style="font-size:22px;font-weight:700;margin:6px 0 8px">Implemented · 48 h pre-training</div>
      <p style="margin:0 0 12px;font-size:13px">Spec: <code>draft_v3.md</code>. Exact counts and 10 float64 tests: <code>v3_design/v3_ref.py</code> (count reproduced independently; harness gaps C1–C7 recorded, repairs pending). The 3D pass animation illustrates the graph; it is not a trained activation.</p>
      <div class="proposal-status"><b>First test:</b> recall onset (recall1 ≥ 0.25 by step 1,792) and per-language val within v1 + 0.10 at v1's eval steps. <b>Recurrence test:</b> a controlled ladder, plain 1-pass → plain 3-pass → t-conditioned 3-pass → full recipe, with paired statistics, per example and per hour. Full vs 1-pass (M1) alone tests the recipe, not recurrence.</div>
      <div id="v3-live-banner"></div>
      <div class="v3-kv" style="margin-top:14px"><span>advisor calls</span><b>41 + review R1–R7 + r2</b><span>tests</span><b>49 / 49 acceptance + 6 / 6 speed parity</b><span>throughput</span><b>block-local attention +6.1 %</b><span>code</span><b>run_v3_19M/ (49/49 tests)</b></div></div>`;
    fetchLive("data/v3_live.json").then(async d => {
      const el_ = document.getElementById("v3-live-banner"); if (!d || !el_) return;
      let ev = (d.evals || []).at(-1);
      if (!ev) try {                                                   // live file restarted with the process: last eval from the run log
        const ls = (await (await fetch("data/v3_evals.jsonl?t=" + Date.now(), { cache: "no-store" })).text()).replace(/\bNaN\b|-?\bInfinity\b/g, "null").trim().split("\n");
        const j = JSON.parse(ls.at(-1)); ev = { val: j.val_lang_mean, recall1: j.probes?.lr_recall1?.exact };
      } catch (e) { ev = null; }
      const age = (Date.now() / 1000 - d.t) / 60;
      el_.innerHTML = `<a class="v3-live-link" href="#v3train" data-tab-link="v3train"><span class="dot" style="background:${age < 15 ? "#168b76" : "#d1a92f"}"></span>
        <b>${age < 15 ? "Training now" : "Run paused / stale"}</b> · step ${fmtI(d.step)} · ${(d.wall_h || 0).toFixed(1)} of ${d.hours} h${ev ? ` · val ${ev.val.toFixed(3)} · recall1 ${(ev.recall1 ?? 0).toFixed(3)}` : ""} · open live training →</a>`;
    }).catch(() => {});
    document.getElementById("v3-evidence").innerHTML = `<table class="t"><thead><tr><th>Observed in v1 / v2</th><th>What it establishes</th><th>v3 design response</th></tr></thead><tbody>
      <tr><td>v1 loop: ‖S‖ 9.8 → 69 linearly, update angles 6.7° → 0.11°, R1 ≈ R4</td><td>Near-repetitive updates: the loop recipe did not earn its cost (a fixed point in direction space is the leading hypothesis, not a proof)</td><td>LoopFormer (t, Δ) table as a <i>tested</i> remedy; the arm ladder separates repetition, conditioning and shortcut</td></tr>
      <tr><td>v1 core off: recall 97 % → 0.4 %</td><td>The concept route carries long-range facts</td><td>Keep C, u, c_pred, PQ, the confined decoder and target-aligned plan</td></tr>
      <tr><td>v2 MoE: routing divergence 0.04, drops 4.5–5 %, 15.69M active of 18.83M</td><td>Experts did not diversify passes and idled parameters under a cap</td><td>Dense core; sparsity lives in the 80M Engram instead</td></tr>
      <tr><td>v2 detached windows, per-pass NCP best at R2</td><td>Credit did not reach early passes; intermediate targets can harm late states</td><td>Full backprop through ≤ 3 passes; terminal-only supervision</td></tr>
      <tr><td>v2 NextLat: nl_h / nl_kl flat; cosine with NTP −0.28 to −0.54 on W_u</td><td>Auxiliary barely moved and conflicted with NTP</td><td>NextLat removed: not shown to help here (it overlaps NCP at plan boundaries but is not the same target)</td></tr>
      <tr><td>v2 val ${v2.val_2942} / recall1 ${v2.recall_2942} at step 2,942 vs v1 2.913 / 0.92 at 2,376</td><td>v2 underperforms; width, depth, losses and schedule are confounded</td><td>v1's objective, FFN ratio 4, 6 heads where retrieval happens, matched-step gates vs v1</td></tr>
    </tbody></table><p class="muted" style="font-size:12px;margin:9px 0 0">v1/v2 numbers are measurements from their runs; none are v3 results.</p>`;
    const card = (cls, n, t, sub, p, x, facts, a, b) => `<div class="story-card ${cls}"><div class="story-num">${n}</div><h4>${t} <span>${sub}</span></h4><p>${x}</p>
      <div class="facts">${facts.map(f => `<span>${f}</span>`).join("")}</div><div class="v2cmp"><span class="a">v1 · v2</span><span>→</span><span class="n">v3</span><span class="o">${a} · ${b}</span><span></span><span class="n">${M(p)}</span></div></div>`;
    document.getElementById("v3-story").innerHTML =
      card("c-enc", 1, "Encoder", "v1 grammar, d 256", T.encoder_non_embedding, `SSSFSSSF with L/G streams, gated residuals and gated attention. FFN ratio 4 everywhere, 6 heads in the F layers. One 80M Engram table is read after layer 0 and again after layer 4; CRC pools 4 tokens into each 256-d concept.`, ["8 layers", "SwiGLU 1024", `Engram ${M(T.engram_table)} × 2 reads`, "reach 63"], M(v1.encoder), M(v2.encoder)) +
      `<div class="story-arrow">→</div>` +
      card("c-core", 2, "Concept module", "4 layers × M ≤ 3", T.concept_module, `Four distinct 6-head full-attention layers read evidence C and write working memory S (S₀ = 0). Each FFN sublayer runs its own SwiGLU 736 in parallel with one shared SwiGLU 1024. A zero-init (t, Δ) table conditions every pass: Φ(S; C, t, Δt).`, ["M = 3 main", "shortcut 1 / 2a / 2b", "dense, no MoE", "PQ 32 × 128 × 8"], M(v1.core), M(v2.core)) +
      `<div class="story-arrow">→</div>` +
      card("c-dec", 3, "Decoder", "confined, plan-aware", T.decoder, `v1's confined decoder: 4 layers, queries from the private stream D, keys and values from tok only (window 8). The plan adds u₃ and c_pred of the newest complete chunk. Tied head. NextLat removed.`, ["4 layers", "SwiGLU 1024", "reach 70", "tied head"], M(v1.decoder), M(v2.decoder));
    document.getElementById("v3-depth").innerHTML = `<h4>How S is updated</h4>
      <p><code>S₀ = 0</code>. One pass runs all 4 layers: <code>S ← S + s(1+α)·Attn(read{C,S})</code>, then <code>S ← S + s[(1+α_own)·own(x) + (1+α_sh)·shared(x)]</code>. Read gains <code>(1+γ_C)</code>, <code>(1+γ_S)</code> and write gains <code>(1+α)</code> come from the (t, Δ) row of that pass.</p>
      <p>Main trajectory: <b>(⅓, ⅓, ⅓)</b> → <code>u₃ = zcRMSNorm(S₃)</code> feeds the decoder and NCP. Shortcut per micro-batch: <b>M 1</b> (50 %), <b>2a (⅓, ⅔)</b> or <b>2b (⅔, ⅓)</b> (25 % each), training only. Full backprop through both.</p>
      <p class="muted">S is carried raw (no carry norm). Loop geometry (angles, ρ) is a diagnostic, neither necessary nor sufficient. Fallback arm "carry renorm" runs only if recurrence shows no benefit <i>and</i> the geometry shows v1's signature.</p>`;
    document.getElementById("v3-objective").innerHTML = `<h4>Objective</h4><p><code>L = L_NTP(u₃) + 0.1 L_NCP(u₃) + 0.05 L_NCP(u_s) + 0.1 L_scons(rms S_s, sg rms S₃) + L_VQ</code></p>
      <table class="t">${A.losses.map(l => `<tr><td>${l.name}</td><td><b>${l.weight}</b></td><td style="font-size:12px">${l.formula}</td></tr>`).join("")}</table>
      <p class="muted" style="margin-top:8px">v1's terms keep v1's weights. Removed from v2: NextLat ×2, MoE LB and z, per-pass NCP. Denominators span the whole optimizer batch; clean target masks.</p>`;
    document.getElementById("v3-data").innerHTML = `<h4>Parameters and compute</h4>
      <div class="brow" style="margin:6px 0"><span>compute params</span>${bar([["encoder", T.encoder_non_embedding, CE], ["core", T.concept_module, CC], ["decoder", T.decoder, CD]])}</div>
      <div class="brow" style="margin:6px 0"><span>training MACs / token</span>${bar([["encoder", mc.encoder, CE], ["core", mc.core_train, CC], ["decoder", mc.decoder, CD], ["head", mc.head, CH]])}</div>
      <table class="t" style="margin-top:8px"><tr><td>encoder: 6 S × ${fmtI(P.enc_layer_S)} + 2 F × ${fmtI(P.enc_layer_F)} + Engram reads ${fmtI(P.engram_read1)} / ${fmtI(P.engram_read2)} + pool, router, norm</td><td><b>${fmtI(T.encoder_non_embedding)}</b></td></tr>
      <tr><td>core: 4 × ${fmtI(P.core_layer)} + shared ${fmtI(P.core_shared)} + table ${fmtI(P.core_table)} + NCP head ${fmtI(P.ncp_head)} + norm</td><td><b>${fmtI(T.concept_module)}</b></td></tr>
      <tr><td>decoder: 4 × ${fmtI(P.dec_layer)} + plan ${fmtI(P.dec_plan)} + norm</td><td><b>${fmtI(T.decoder)}</b></td></tr>
      <tr><td>throughput (two-point fit, unvalidated; counter counts slots incl. PAD)</td><td><b>1,290–1,770 slots/s</b></td></tr><tr><td>48 h net: slots · non-pad tokens · tokens / param · epochs</td><td>211–290M · ≈ 199–273M · 10.5–14.4 · 0.75–1.03</td></tr></table>
      <p class="muted" style="margin-top:8px">48 h time-based cosine 6e-4 → 6e-5, warm-up 300 steps, AdamW (0.9, 0.95), B2 × 16 × 1,024, v1's corpus and order.</p>`;
    document.getElementById("v3-gates").innerHTML = `<h4>Accept or reject v3</h4><ol>
      <li><b>E1 recall onset (primary):</b> recall1 ≥ 0.25 by step 1,792 (v1 0.336 at 1,217; v2 ≈ 2,250). Deliberately far below v1's 0.82: passing means no v2-like delay, not matching v1.</li>
      <li><b>E2 val per language:</b> each ≤ v1 + 0.10 at steps 1,217 / 1,792 / 2,376.</li>
      <li><b>E4 throughput:</b> ≥ 1,290 slots/s, hard floor 1,200. <b>E5</b> (encoder G contribution) is report-only: a low G share does not mean the global path is bypassed.</li>
      <li><b>M1 full recipe vs single pass:</b> recall1 +10 points (val −0.03) at ≥ 2 of 3 gate steps, with paired per-item statistics. <b>Recurrence itself</b> needs the ladder A1 / A3 / T3 / Full. Loop geometry and the gradient audit are diagnostics; interventions are fixed-checkpoint effects, not upper bounds.</li></ol>
      <p class="muted">No gate stops the run overnight: a failure notifies (with v1's numbers) and the run continues to 24 h.</p>`;
    document.getElementById("v3-experiments").innerHTML = `<table class="t"><thead><tr><th>Arm</th><th>Architecture change</th><th>Objective</th><th>Question</th><th>Runs when</th></tr></thead><tbody>
      ${A.arms.map(r => `<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td><td>${r[4] === "recommended" ? "<b>recommended</b>" : r[4]}</td></tr>`).join("")}
      </tbody></table><p class="muted" style="font-size:12px;margin:9px 0 0">All arms share one per-step LR trace (the default's, or a predeclared schedule: the owner's choice), and are compared at matched steps and matched wall time. One seed per run: indicative. Pilot ladder before the 48 h run is an open owner decision.</p>`;
  }

  function goto(id) {
    if (!document.getElementById("tab-v3arch2d").classList.contains("on")) showTab("v3arch2d");
    document.querySelectorAll("#fig3-nav button").forEach(b => b.classList.toggle("on", b.dataset.id === id));
    const f = document.getElementById("fig3-" + id); if (f) requestAnimationFrame(() => f.scrollIntoView({ block: "start" }));
  }
  function open3D(view) {
    showTab("v3arch3d"); window.scrollTo(0, 0);
    let n = 0; const timer = setInterval(() => { const b = document.querySelector(`#v3c-bar [data-view="${view}"]`);
      if ((b && document.querySelector("#v3c-info h5")) || ++n > 40) { clearInterval(timer); if (b) b.click(); } }, 100);
  }
  function renderFigs() {
    if (figsDone) return; figsDone = true;
    const nav = document.getElementById("fig3-nav"), box_ = document.getElementById("figs3");
    const ctx = { figs: FIGS3D, go: goto, open3d: open3D, key: FIG3D_KEY, map3d: FIG3D_3D };
    FIGS3D.forEach((f, i) => {
      const b = document.createElement("button"); b.dataset.id = f.id; b.innerHTML = `<span class="fn">${i + 1}</span>${f.t}`; if (!i) b.classList.add("on"); b.onclick = () => goto(f.id); nav.appendChild(b);
      const fig = document.createElement("figure"); fig.className = "fig"; fig.id = "fig3-" + f.id;
      const h = document.createElement("h3"); h.className = "figt"; h.innerHTML = `<span class="figt-n">${i + 1}</span>${f.t}`; fig.appendChild(h);
      try { fig.appendChild(f.fn(A3)); } catch (e) { const p = document.createElement("p"); p.className = "muted"; p.textContent = "figure error: " + e.message; fig.appendChild(p); console.error(e); }
      const cap = document.createElement("figcaption"); cap.innerHTML = `<b>Figure ${i + 1}.</b> ${f.cap}`; fig.appendChild(cap); box_.appendChild(fig);
      figTools(fig, f.id, i, FIGS3D.length, ctx);
    });
    window.addEventListener("scroll", () => { if (!document.getElementById("tab-v3arch2d").classList.contains("on")) return;
      let cur = null; document.querySelectorAll("#figs3 figure.fig").forEach(f => { if (f.getBoundingClientRect().top < innerHeight * .4) cur = f.id.slice(5); });
      if (cur) document.querySelectorAll("#fig3-nav button").forEach(b => b.classList.toggle("on", b.dataset.id === cur)); }, { passive: true });
  }
  async function ensure() { if (!A3) { const r = await fetch("data/arch_v3d.json"); if (!r.ok) throw new Error("v3 architecture data unavailable"); A3 = await r.json(); window.ARCH3D = A3; } return A3; }
  async function shown(id) { if (!id.startsWith("v3")) return; try { await ensure(); if (id === "v3overview" && !document.getElementById("v3-stats").innerHTML) renderOverview(); if (id === "v3arch2d") renderFigs(); } catch (e) { console.error("v3 page", e); } }
  window.addEventListener("tabshown", e => shown(e.detail));
  document.addEventListener("click", e => { const a = e.target.closest("[data-fig3]"); if (a) { e.preventDefault(); ensure().then(() => { renderFigs(); goto(a.dataset.fig3); }); } });
  const initial = (document.querySelector("section.tab.on") || {}).id; if (initial && initial.startsWith("tab-v3")) shown(initial.slice(4));
})();
