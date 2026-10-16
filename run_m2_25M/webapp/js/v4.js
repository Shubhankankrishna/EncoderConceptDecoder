/* v4 (proposed) pages: overview, 2D figures, and the tiny experiment & verdict. The 3D page is js/view3d_v4.js.
   Data: data/arch_v4.json (exact counts from v4_tiny/code/model_v4.py, make_arch_v4.py) and data/v4_tiny.json (every
   result of the CPU experiment, make_v4tiny_json.py). Same components and classes as the v3 pages and the v1 Results tab.
   Pill rule (DESIGN #23): green only for deterministic, pre-registered checks; tiny-scale learning results are grey
   "Measured" / "Single seed"; negative signs are amber; nothing is invented. */
(function () {
  let A4 = null, R4 = null, figsDone = false, tinyDone = false;
  const M = n => (n / 1e6).toFixed(2) + "M";
  const pct = x => (100 * x).toFixed(1) + " %";
  const fin = v => v != null && isFinite(v);
  const n3 = v => fin(v) ? Number(v).toFixed(3) : "—";
  const sgn = (v, d = 3) => !fin(v) ? "—" : (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d);
  const stat = (l, v, s) => `<div class="card stat"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
  const CE = "#d6e6f7", CC = "#d2efea", CD = "#fbdcc8", CH = "#e6e0f5";
  const ACC = "#b5476b";
  function bar(arr) {
    const tot = arr.reduce((s, a) => s + a[1], 0);
    return `<div class="pbar">${arr.map(([n, v, c]) => `<div style="width:${100 * v / tot}%;background:${c}" title="${n}: ${fmtI(Math.round(v))} (${pct(v / tot)})">${v / tot > 0.1 ? `${n} ${(100 * v / tot).toFixed(0)}%` : ""}</div>`).join("")}</div>`;
  }
  const sec = (id, title, sub, body) => `<section class="rs-sec" id="${id}"><div class="sh"><div><h3>${title}</h3><p>${sub}</p></div></div>${body}</section>`;
  const pill = (lvl, lab) => `<span class="st-pill ${lvl}"><i></i>${lab}</span>`;

  /* ================================================================== overview */
  function renderOverview() {
    const A = A4, T = A.table, P = A.params, c = A.core, v1 = A.v1, v3 = A.v3;
    document.getElementById("v4-eyebrow-n").textContent = M(T.compute_params);
    document.getElementById("v4-stats").innerHTML = [
      stat("compute params", M(T.compute_params), `< 20M cap · ${M(A.variants.recommended_main_run)} in the recommended main run`),
      stat("stored params", M(T.stored_total), `+ Engram ${M(T.engram_table)}, embedding, codebooks`),
      stat("tokens per concept", "6", `k = 4 control · k = 8 arm · T ${A.ecfg.T}`),
      stat("concept share", pct(T.concept_module / T.compute_params), `v3 ${pct(v3.core / v3.compute)}`),
      stat("recurrence", "M = 2", "plain passes · terminal-only · no table"),
      stat("decoder reach", A.reach_dec + " tokens", `encoder L stream ${A.reach_enc} (loop keeps it)`),
    ].join("");
    document.getElementById("v4-hero-side").innerHTML = `<div class="card" style="height:100%">
      <div class="l" style="font-size:12px;color:var(--ink3);text-transform:uppercase;letter-spacing:.08em;font-weight:650">Status</div>
      <div style="font-size:22px;font-weight:700;margin:6px 0 8px">Proposed · tested at tiny scale</div>
      <p style="margin:0 0 12px;font-size:13px">Spec: <code>draft_v4.md</code> r2. Code: <code>v4_tiny/code/model_v4.py</code> builds both the full model (counts exact to the draft) and a d 64 version that was trained on CPU. The 3D pass animation illustrates the graph; it is not a trained activation.</p>
      <div class="proposal-status"><b>Verdict:</b> implementation <b>go</b> · architecture <b>sound, one claim untested</b> (retrieval through the new concept path) · launch <b>conditional</b>, gated, after v3's final evaluation, with the decoder Engram reads and the pooler FFN moved to arms. <a href="#v4tiny" data-tab-link="v4tiny">Read the evidence →</a></div>
      <div id="v4-live-banner"></div>
      <div class="v3-kv" style="margin-top:14px"><span>tests</span><b>15 / 15 (both attention paths)</b><span>mutants caught</span><b>13 / 13</b><span>tiny arms</span><b>5 (CPU, single seed)</b><span>advisor</span><b>${R4 ? R4.advisor_consults : "29"} numbered consults</b></div></div>`;
    fetchLive("data/v3_live.json").then(d => {
      const el_ = document.getElementById("v4-live-banner"); if (!d || !el_) return;
      const age = (Date.now() / 1000 - d.t) / 60;
      el_.innerHTML = `<a class="v3-live-link" href="#v3train" data-tab-link="v3train"><span class="dot" style="background:${age < 15 ? "#168b76" : "#d1a92f"}"></span>
        <b>v3 ${age < 15 ? "training now" : "paused / stale"}</b> · step ${fmtI(d.step)} · ${(d.wall_h || 0).toFixed(1)} of ${d.hours} h · v4 waits for its final evaluation →</a>`;
    }).catch(() => {});
    document.getElementById("v4-evidence").innerHTML = `<table class="t"><thead><tr><th>Measured in v1 / v3</th><th>What it establishes</th><th>v4 design response</th></tr></thead><tbody>
      <tr><td>v3's pool (step 11,348): slot norms 7.7 / 7.6 / 8.3 / <b>12.1</b>, same weights for every chunk</td><td>The pool starts as the average and trains into a per-slot recency map; it cannot weight a token by what it is</td><td>Attention + FFN pooler with weights shared across slots; 6 tokens per concept (k = 4 control, k = 8 arm)</td></tr>
      <tr><td>v3 24 h: pass 2 multi4 +0.09, pass 3 +0.01</td><td>The value is in the second visit, not more passes</td><td>M = 2 plain passes; the third pass and the shortcut trajectory are dropped</td></tr>
      <tr><td>v3 24 h: (t, Δ) table off +0.008 nats, probes within noise</td><td>The conditioning table never earned its place</td><td>Table removed (−43,008 parameters)</td></tr>
      <tr><td>v3 24 h: shared FFN off → multi4 0.547 → 0.418</td><td>A central, layer-shared memory is used for multi-fact work</td><td>Core shared FFN kept at full size; a small shared FFN (384) added to the encoder</td></tr>
      <tr><td>v3: F layers off → multi4 0, carry3 0.89 → 0.24; concepts are 82–86 % G</td><td>G is what the concept module consumes</td><td>F layers 3 and 7 applied twice (writes only G, so reach(L) stays 63)</td></tr>
      <tr><td>v1 = v3 codebook VE 0.651 with the same 8-dim × 128-code segment</td><td>Fit is capped by the segment shape, not by N</td><td>PQ 32 × 256 × 8 (1 bit/dim) on the residual of a coarse 1,024-way code: an action label for concept-level RL</td></tr>
      <tr><td>v3: Engram off +0.60 nats, recall halved</td><td>The n-gram memory is valuable in the encoder</td><td>Two decoder reads of the same rows (owner's brief) — <b>tiny scale: a memorisation channel</b>, so recommended as an arm</td></tr>
    </tbody></table><p class="muted" style="font-size:12px;margin:9px 0 0">v1 / v3 numbers are measurements from their runs; none are v4 results. Tiny-scale results are on the Tiny experiment tab.</p>`;
    const card = (cls, n, t, sub, p, x, facts, a) => `<div class="story-card ${cls}"><div class="story-num">${n}</div><h4>${t} <span>${sub}</span></h4><p>${x}</p>
      <div class="facts">${facts.map(f => `<span>${f}</span>`).join("")}</div><div class="v2cmp"><span class="a">v3</span><span>→</span><span class="n">v4</span><span class="o">${a}</span><span></span><span class="n">${M(p)}</span></div></div>`;
    document.getElementById("v4-story").innerHTML =
      card("c-enc", 1, "Encoder", "v3's, refined", T.encoder_non_embedding, `SSSFSSSF with L/G gated-residual streams and one 80M Engram table read after layers 0 and 4. New: the F layers run twice in place, a shared SwiGLU 384 sits beside every layer's own FFN, and an attention + FFN pooler turns every 6 tokens into one 256-d concept.`, ["8 layers", "F × 2", "shared FFN 384", "pooler 6 → 1", "reach 63"], M(v3.encoder)) +
      `<div class="story-arrow">→</div>` +
      card("c-core", 2, "Concept module", "4 layers × 2", T.concept_module, `Four distinct 6-head full-attention layers read the concepts C and write the memory S (S₀ = 0), twice. Own SwiGLU 736 plus the shared SwiGLU 1024. No (t, Δ) table, no shortcut, terminal-only. A coarse 1,024-way code and PQ codes on the residual are predicted for the next concept.`, ["M = 2", "170 concepts / row", "coarse 1,024", "PQ 32 × 256 × 8"], M(v3.core)) +
      `<div class="story-arrow">→</div>` +
      card("c-dec", 3, "Decoder", "confined, plan-aware", T.decoder, `v3's confined decoder: 4 layers, keys and values from tok only (window 8). The plan of chunk j is shifted one chunk and repeated over the 6 positions that write chunk j + 1. Proposed: two direct Engram reads after layers 0 and 2.`, ["4 layers", "plan × 6", "Engram reads (arm)", "reach 70"], M(v3.decoder));
    document.getElementById("v4-depth").innerHTML = `<h4>How concepts are made and used</h4>
      <p><b>Pool.</b> For the 6 tokens of a chunk: router mix <code>m_i</code>; 4 heads score <code>q_h · W_K(rms m_i + p_r)</code>; <code>c₀ = W_O[Σ α_{h,i} W_V(m_i + p_r)]</code>; <code>c₁ = c₀ + g·SwiGLU₃₈₄(rms c₀)</code>; <code>ĉ = √256 · c₁/‖c₁‖</code>. Init is the exact masked mean (tested).</p>
      <p><b>Think.</b> <code>S₀ = 0</code>, <code>S₁ = Φ(C, S₀)</code>, <code>S₂ = Φ(C, S₁)</code>, <code>u = zcRMSNorm(S₂)</code>. Pass 2 reads S₁, which already holds what pass 1 retrieved: the structure a two-step retrieval needs.</p>
      <p><b>Write.</b> Position t predicts x<sub>t+1</sub> from <code>tok</code> (70 tokens) plus <code>P = W_u u_j + W_c ĉ_pred,j + o_r</code> with <code>j = ⌊(t+1)/6⌋ − 1</code>.</p>`;
    document.getElementById("v4-objective").innerHTML = `<h4>Objective</h4><p><code>L = NTP(u) + 0.05 CE_coarse + 0.1 CE_fine + VQ_coarse + VQ_fine</code></p>
      <table class="t">${A.losses.map(l => `<tr><td>${l.name}</td><td><b>${l.weight}</b></td><td style="font-size:12px">${l.formula}</td></tr>`).join("")}</table>
      <p class="muted" style="margin-top:8px">Removed from v3: NCP(u_s) and L_scons (no shortcut). The printed loss is dominated by VQ_coarse ≈ 7, so the dashboard headlines NTP.</p>`;
    document.getElementById("v4-data").innerHTML = `<h4>Parameters and compute</h4>
      <div class="brow" style="margin:6px 0"><span>compute params</span>${bar([["encoder", T.encoder_non_embedding, CE], ["core", T.concept_module, CC], ["decoder", T.decoder, CD]])}</div>
      <table class="t" style="margin-top:8px"><tr><td>encoder: 6 S × ${fmtI(P.enc_layer_S)} + 2 F × ${fmtI(P.enc_layer_F)} + Engram reads ${fmtI(P.engram_read1)} / ${fmtI(P.engram_read2)} + shared FFN ${fmtI(P.enc_shared)} + 8 gains + pooler ${fmtI(P.pooler)} + router, norm</td><td><b>${fmtI(T.encoder_non_embedding)}</b></td></tr>
      <tr><td>core: 4 × ${fmtI(P.core_layer)} + shared ${fmtI(P.core_shared)} + head ${fmtI(P.ncp_head)} + norm</td><td><b>${fmtI(T.concept_module)}</b></td></tr>
      <tr><td>decoder: 4 × ${fmtI(P.dec_layer)} + plan ${fmtI(P.dec_plan)} + 2 Engram reads ${fmtI(P.dec_reads)} + norm</td><td><b>${fmtI(T.decoder)}</b></td></tr>
      <tr><td>variants: k = 8 · gated-conv pooler · recommended main run (no decoder reads, no pooler FFN)</td><td>${fmtI(A.variants.compute.k8)} · ${fmtI(A.variants.compute.gconv)} · <b>${fmtI(A.variants.recommended_main_run)}</b></td></tr>
      <tr><td>throughput vs v3 (estimate from v3's profile; A/B decides)</td><td><b>≈ 1.1–1.2×</b></td></tr></table>
      <p class="muted" style="margin-top:8px">48 h time-based cosine 6e-4 → 6e-5, AdamW (0.9, 0.95), B 2 × 16 × 1,020 = 32,640 tokens per step; same corpus as v1 / v3, re-packed for k = 6.</p>`;
    document.getElementById("v4-gates").innerHTML = `<h4>Accept or reject v4</h4><ol>
      ${A.gates.map(r => `<li><b>${r[0]}</b> (step ${r[1]}): ${r[2]}. <span class="muted">${r[3]}</span></li>`).join("")}</ol>
      <p class="muted">No gate stops the run: gates inform the owner. G3 (recall onset, ≈ 5 h after launch) is the first full-scale reading of the one claim the tiny experiment could not settle.</p>`;
    document.getElementById("v4-experiments").innerHTML = `<table class="t"><thead><tr><th>Arm</th><th>Change</th><th>Priority</th><th>Question</th></tr></thead><tbody>
      ${A.arms.map(r => `<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td>${/mandatory/.test(r[2]) ? `<b style="color:${ACC}">${r[2]}</b>` : r[2]}</td><td>${r[3]}</td></tr>`).join("")}
      </tbody></table><p class="muted" style="font-size:12px;margin:9px 0 0">Each arm ≈ 10 h to step 4,050, run after the main 48 h run on its step-indexed learning-rate trace, compared at matched tokens. One seed per arm: indicative.</p>`;
  }

  /* ================================================================== 2D figures */
  function goto(id) {
    if (!document.getElementById("tab-v4arch2d").classList.contains("on")) showTab("v4arch2d");
    document.querySelectorAll("#fig4-nav button").forEach(b => b.classList.toggle("on", b.dataset.id === id));
    const f = document.getElementById("fig4-" + id); if (f) requestAnimationFrame(() => f.scrollIntoView({ block: "start" }));
  }
  function open3D(view) {
    showTab("v4arch3d"); window.scrollTo(0, 0);
    let n = 0; const timer = setInterval(() => { const b = document.querySelector(`#v4c-bar [data-view="${view}"]`);
      if ((b && document.querySelector("#v4c-info h5")) || ++n > 40) { clearInterval(timer); if (b) b.click(); } }, 100);
  }
  function renderFigs() {
    if (figsDone) return; figsDone = true;
    const nav = document.getElementById("fig4-nav"), box_ = document.getElementById("figs4");
    const ctx = { figs: FIGS4, go: goto, open3d: open3D, key: FIG4_KEY, map3d: FIG4_3D };
    FIGS4.forEach((f, i) => {
      const b = document.createElement("button"); b.dataset.id = f.id; b.innerHTML = `<span class="fn">${i + 1}</span>${f.t}`; if (!i) b.classList.add("on"); b.onclick = () => goto(f.id); nav.appendChild(b);
      const fig = document.createElement("figure"); fig.className = "fig"; fig.id = "fig4-" + f.id;
      const h = document.createElement("h3"); h.className = "figt"; h.innerHTML = `<span class="figt-n">${i + 1}</span>${f.t}`; fig.appendChild(h);
      try { fig.appendChild(f.fn(A4)); } catch (e) { const p = document.createElement("p"); p.className = "muted"; p.textContent = "figure error: " + e.message; fig.appendChild(p); console.error(e); }
      const cap = document.createElement("figcaption"); cap.innerHTML = `<b>Figure ${i + 1}.</b> ${f.cap}`; fig.appendChild(cap); box_.appendChild(fig);
      figTools(fig, f.id, i, FIGS4.length, ctx);
    });
    window.addEventListener("scroll", () => { if (!document.getElementById("tab-v4arch2d").classList.contains("on")) return;
      let cur = null; document.querySelectorAll("#figs4 figure.fig").forEach(f => { if (f.getBoundingClientRect().top < innerHeight * .4) cur = f.id.slice(5); });
      if (cur) document.querySelectorAll("#fig4-nav button").forEach(b => b.classList.toggle("on", b.dataset.id === cur)); }, { passive: true });
  }

  /* ================================================================== tiny experiment & verdict */
  const ARMC = { k4_attnffn_s0: "#3b78c2", k6_attnffn_s0: ACC, k6_attnffn_p07: "#e07b39", k6_linear_p07: "#8a93a3", k6_gconv_p07: "#2e9c8c" };
  const arm = n => R4.arms.find(a => a.name === n);
  function tile(l, v, s) { return `<div class="rs-k"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`; }

  function tVerdict() {
    const vc = (q, overall, lvl, rows, note) => `<div class="card v3-vcard"><div class="v3-vhead"><h4>${q}</h4>${pill(lvl, overall)}</div>
      <table class="t v3-vtab"><tbody>${rows.map(([c, ev, l, lab]) => `<tr><td class="v3-vc">${c}</td><td class="v3-ve">${ev}</td><td>${pill(l, lab)}</td></tr>`).join("")}</tbody></table>
      ${note ? `<div class="muted v3-vnote">${note}</div>` : ""}</div>`;
    return sec("v4t-verdict", "Verdict", "Three questions with different confidence, kept separate (GO_NOGO.md).", `<div class="grid g3" style="align-items:start">
      ${vc("1 · Is v4 implemented correctly?", "Go", "g", [
        ["Parameter ledger", "full-size model instantiates to 19,898,011 (k 6) and 19,899,043 (k 8), every sub-part equal to the draft", "g", "Met"],
        ["Acceptance tests", `${R4.tests.length} / 15 pass on block-local and full-mask attention`, "g", "Met"],
        ["Deliberate defects", `${R4.mutants.filter(m => m.status === "KILLED").length} / ${R4.mutants.length} caught by assertions`, "g", "Met"],
        ["Derived from v3", "encoder and decoder reduce bit-exactly to v3's with the new branches off", "g", "Met"]], "Deterministic tests on the same file that builds the full model: transfers with certainty.")}
      ${vc("2 · Is the architecture sound?", "No defect · one claim untested", "a", [
        ["Concept route used", "core off +0.12 to +0.25 nats; plans row-specific", "n", "Measured"],
        ["Pooler selects by content", "within-slot ratios ≠ 1 (v3's pool cannot); currently selects structure", "n", "Measured"],
        ["Codebooks healthy", "fine-level VE 0.72 vs the 0.75 bound; coarse ppl 46–62 % of K (k4 / k6)", "n", "Measured"],
        ["Retrieval through v4's path", "not reached at 5–7M tokens, d 64 (v3 needed ≈ 59M at d 256)", "a", "Untested"],
        ["Decoder Engram reads", "memorise repeated text, hurt unseen text", "r", "Negative"]], "Tiny scale, single seed: mechanism-level evidence.")}
      ${vc("3 · Launch after v3?", "Conditional", "a", [
        ["Implementation", "go now (data rebuild, trainer port, harness)", "g", "Ready"],
        ["Defaults", "decoder reads off, pooler FFN off in the main run; both as arms", "a", "Owner decides"],
        ["Gates", "G3 recall onset at step 1,792; k = 6 gate at 4,050 vs the mandatory k = 4 control", "n", "Pre-registered"],
        ["Timing", "after v3's 48 h end (≈ 02:10) and its final evaluation", "n", "Planned"]], "Go to implementation now; launch gated. The gates answer the open question within ≈ 12 h.")}
      </div>
      <div class="card" style="margin-top:14px"><div class="proposal-status"><b>Bottom line.</b> v4 is correct as built and nothing found argues against it, but its central claim — that the new concept path (6 tokens per concept, attention pooler, 2 passes, two-level codebook) still retrieves facts the way v3 does — has only been testable at full scale. Build it now, launch it after v3's final evaluation with the decoder Engram reads and the pooler FFN moved to arms, and read gate G3 at step 1,792.</div></div>`);
  }

  function tKpis() {
    const k6 = arm("k6_attnffn_s0"), k4 = arm("k4_attnffn_s0");
    const co6 = k6.decomp.switch_offs.core_off, co4 = k4.decomp.switch_offs.core_off;
    return `<div class="rs-kpis" style="margin-top:6px">
      ${tile("Full-size ledger", "19,898,011", "exact to the draft (k 6) · 19.18M recommended")}
      ${tile("Acceptance tests", `${R4.tests.length} / 15`, `+ ${R4.mutants.length} / ${R4.mutants.length} mutants caught`)}
      ${tile("Concept route", `${sgn(co4.delta, 2)}`, `core off, k4 · k6 ${sgn(co6.delta, 2)} ± ${co6.paired_se.toFixed(2)} nats`)}
      ${tile("Retrieval onset", "not reached", "both filler pools · chance ≈ 1/150")}
      ${tile("k6 vs k4 at 5M", `+${(k6.final.val - k4.final.val).toFixed(3)}`, "nats val · 1.07× faster (CPU)")}
      ${tile("Cost to v3", "≈ 2.5 %", "of its tokens · no guard event · recovered 18:10")}
    </div>`;
  }

  function tFindings() {
    const f = (c, t, x) => `<div class="rs-f ${c}"><i></i><div><b>${t}</b><p>${x}</p></div></div>`;
    return `<div class="rs-find">
      ${f("g", "v4 is built exactly as drafted", "19,898,011 parameters, every sub-part equal; 15 tests on both attention paths; 13 / 13 planted bugs caught; bit-exact reductions to v3's encoder and decoder.")}
      ${f("g", "The concept route works for language", "Switching the core off costs +0.12 to +0.25 nats, and swapping plans between rows costs as much: the plan is row-specific, even before retrieval is learned.")}
      ${f("g", "The pooler weights tokens by content", "Within the same slot, token classes get different weights — impossible for v3's per-slot pool, which in addition collapses the coarse code.")}
      ${f("a", "…but at this budget it selects structure", "Frequent and template words get more weight than names and values. The stop-word claim is not testable until retrieval pays.")}
      ${f("a", "Retrieval onset was not reached", "At 5–7M tokens with d 64, on both filler pools. v3 needed ≈ 59M at d 256: a budget limit, not a mechanism failure.")}
      ${f("r", "The decoder Engram reads memorise", "+0.47 to +0.70 nats on memorised text, −0.14 to −0.17 on unseen text, while the encoder's reads on the same table generalise.")}
      ${f("a", "Two parts earn nothing at this scale", "The pooler FFN writes but Δ ≈ 0; the F loop helps at k = 4 (+0.022) but not at k = 6 (+0.002).")}
      ${f("a", "Compression 6 costs ≈ 0.03 nats", "k6 − k4 widens to +0.035 by 5M tokens at 1.07× speed: right at the draft's gate, so the full-scale gate decides.")}
    </div>`;
  }

  function tCorrect() {
    const tests = R4.tests.map(t => { const [nm, ...rest] = t.split(" | "); return `<tr><td><b>${nm.replace(/^(\d+)\s/, "$1 · ")}</b></td><td style="font-size:12px;color:var(--ink2)">${rest.join(" | ")}</td><td>${pill("g", "Pass")}</td></tr>`; }).join("");
    const muts = R4.mutants.map(m => `<tr><td>${m.name}</td><td style="font-size:12px;color:var(--ink2)">${m.caught.split(":")[0].replace(/^test_/, "")}</td><td>${pill(m.status === "KILLED" ? "g" : "r", m.status === "KILLED" ? "Caught" : "Missed")}</td></tr>`).join("");
    return sec("v4t-correct", "Tier 1 · correctness", "Deterministic checks: these transfer to the full model with certainty.",
      `<div class="grid g2" style="align-items:start"><div class="card scroll-x"><h4>15 acceptance tests (float64, randomized model, k = 4 / 6 / 8)</h4><table class="t v3-vtab"><tbody>${tests}</tbody></table>
        <p class="rs-note">All 15 also pass with full-mask attention (${R4.tests_noblock_pass} / 15).</p></div>
      <div class="card scroll-x"><h4>Deliberate defects: each rewrites one line of the model</h4><table class="t v3-vtab"><tbody>${muts}</tbody></table>
        <p class="rs-note">A test suite that cannot fail proves nothing: every one of these defects is caught by an assertion, not by a crash.</p></div></div>`);
  }

  function tFlaw() {
    return sec("v4t-flaw", "Design flaw found during the experiment (not in v4)", "Reported in full, with what it did and did not affect.",
      `<div class="grid g2" style="align-items:start"><div class="card"><h4>What happened</h4><p style="font-size:13.5px;line-height:1.55">The probe fillers were drawn from only 3,000 training documents (6,519 sentences). Seen about ten times each, the tiny model memorised them. Evaluation items used a different filler sample. It showed up as next-token loss of 4.7 nats on evaluation probe rows vs 1.31 on rows built from the training pool (validation text: 2.9–3.0).</p>
        <p style="font-size:13.5px;line-height:1.55"><b>Fix for a rerun:</b> sample fillers fresh per item from all 457,614 documents, and evaluate on both pools.</p></div>
      <div class="card"><h4>What it touched, and the re-measurement</h4><table class="t v3-vtab"><tbody>
        <tr><td class="v3-vc">Recall</td><td class="v3-ve">re-measured with 192 items from the training pool: at chance in both pools for all 5 arms</td><td>${pill("n", "Stands")}</td></tr>
        <tr><td class="v3-vc">Decoder Engram reads</td><td class="v3-ve">re-measured on both pools: the result became a finding (memorisation channel)</td><td>${pill("a", "Revised")}</td></tr>
        <tr><td class="v3-vc">Facts content / template ratios</td><td class="v3-ve">template words repeated in every item: part may be memorisation</td><td>${pill("a", "Caveat")}</td></tr>
        <tr><td class="v3-vc">Everything on validation text</td><td class="v3-ve">tier 1, k6 vs k4, pooler comparison, codebooks, switch-offs on text, within-slot text ratios</td><td>${pill("g", "Unaffected")}</td></tr>
      </tbody></table></div></div>`);
  }

  function tCurves() {
    const S = (key, f) => R4.arms.map(a => ({ name: a.label, color: ARMC[a.name], dots: true, pts: a.evals.map(e => [e.real / 1e6, f(e)]) }));
    const ch = (id, t, sub, ser, o = {}) => `<div class="card"><h4>${t}</h4><p class="muted" style="margin:-4px 0 8px;font-size:12px">${sub}</p>${LC.chart(id, Object.assign({ series: ser, xlabel: "real tokens (M)", xname: "tokens (M)", xfmt: v => (+v).toFixed(1), W: 560, H: 250 }, o))}</div>`;
    return sec("v4t-curves", "Learning curves", "Validation, every 1M real tokens, all five arms (p = synthetic share of training rows).",
      `<div class="grid g2">${ch("v4t-val", "Validation next-token loss (nats)", "Text only. The p 0.7 arms see fewer text tokens, so their text loss is higher by construction.", S("val", e => e.val), { tfmt: v => (+v).toFixed(3) })}
      ${ch("v4t-rec", "recall1 (exact, 128 items)", "Chance ≈ 0.007 (≈ 150 value words): every arm stays at chance.", S("rec", e => e.probes.lr_recall1), { refs: [{ y: 0.0067, label: "chance", color: "#d0667a" }], y0: 0, y1: 0.05, tfmt: v => (+v).toFixed(3) })}
      ${ch("v4t-ppl", "Coarse-code perplexity (of K = 64)", "v3's linear pool (grey) collapses onto fewer coarse codes at every reading.", S("ppl", e => e.ppl0), { refs: [{ y: 16, label: "25 % of K", color: "#d1a92f" }], tfmt: v => (+v).toFixed(1) })}
      ${ch("v4t-ve", "Codebook variance explained", "Two-level codebook; the fine level alone explains ≈ 0.72 of the residual (bound 0.75).", S("ve", e => e.ve), { tfmt: v => (+v).toFixed(3) })}</div>`);
  }

  function tSwitch() {
    const rows = [["core_off", "Concept core off", "learned null plan everywhere"], ["plan_swap", "Plan swapped", "another row's plans"], ["c_off", "Predicted concept off", "plan without ĉ_pred"],
      ["M1", "1 pass instead of 2", "second pass removed"], ["loop_off", "F loop off", "F layers applied once"], ["enc_shared_off", "Encoder shared FFN off", ""], ["core_shared_off", "Core shared FFN off", ""],
      ["pool_ffn_off", "Pooler FFN off", ""], ["engram_off", "All four Engram reads off", ""], ["enc_engram_off", "Encoder Engram reads off", "decoder reads kept"], ["dec_reads_off", "Decoder Engram reads off", "encoder reads kept"]];
    const cols = ["k4_attnffn_s0", "k6_attnffn_s0", "k6_attnffn_p07", "k6_linear_p07", "k6_gconv_p07"];
    const cell = (a, k) => { const v = arm(a).decomp && arm(a).decomp.switch_offs[k]; if (!v) return `<td class="muted">—</td>`;
      const sig = Math.abs(v.delta) > 2 * v.paired_se, col = !sig ? "transparent" : v.delta > 0 ? "color-mix(in srgb, #2e9c8c 22%, transparent)" : "color-mix(in srgb, #d0667a 22%, transparent)";
      return `<td style="background:${col}">${sgn(v.delta)}<br><small class="muted">± ${v.paired_se.toFixed(3)}</small></td>`; };
    return sec("v4t-switch", "What each part carries", "Δ validation NTP when the part is switched off, paired SE over 64 rows. Green: removing it hurts (the part is used). Red: removing it helps.",
      `<div class="card scroll-x"><table class="rs-mx"><thead><tr><th>switch-off</th>${cols.map(c => `<th style="color:${ARMC[c]}">${arm(c).label}</th>`).join("")}</tr></thead><tbody>
      ${rows.map(([k, n, s]) => `<tr><td><b>${n}</b>${s ? `<br><small class="muted">${s}</small>` : ""}</td>${cols.map(c => cell(c, k)).join("")}</tr>`).join("")}</tbody></table>
      <p class="rs-note">The draft's kill line for a part is 0.01 nats (and no probe moving by 2 SE). Fixed-checkpoint effects, single seed; "—" = not measured for that arm.</p></div>`);
  }

  function tReads() {
    const cols = ["k4_attnffn_s0", "k6_attnffn_s0", "k6_attnffn_p07", "k6_linear_p07", "k6_gconv_p07"];
    const r = (lab, f) => `<tr><td><b>${lab}</b></td>${cols.map(c => { const v = f(arm(c)); return v ? `<td style="background:${v.delta > 2 * v.paired_se ? "color-mix(in srgb, #2e9c8c 22%, transparent)" : v.delta < -2 * v.paired_se ? "color-mix(in srgb, #d0667a 22%, transparent)" : "transparent"}">${sgn(v.delta)}<br><small class="muted">± ${v.paired_se.toFixed(3)}</small></td>` : "<td>—</td>"; }).join("")}</tr>`;
    return sec("v4t-reads", "The decoder Engram reads are a memorisation channel", "Δ next-token loss when both decoder reads are switched off, on three kinds of text (positive = the reads were helping).",
      `<div class="card scroll-x"><table class="rs-mx"><thead><tr><th>text</th>${cols.map(c => `<th style="color:${ARMC[c]}">${arm(c).label}</th>`).join("")}</tr></thead><tbody>
      ${r("probe rows, training filler pool (memorised)", a => a.reads && a.reads.train_fillers)}
      ${r("probe rows, unseen fillers", a => a.reads && a.reads.val_fillers)}
      ${r("validation text", a => a.decomp && a.decomp.switch_offs.dec_reads_off)}
      ${r("encoder reads off, validation text (same table)", a => a.decomp && a.decomp.switch_offs.enc_engram_off)}</tbody></table>
      <p class="rs-note">On repeated text the decoder reads carry much of the memorised continuation; on new text they inject the wrong one. The encoder's reads on the same table generalise, so hash collisions are excluded. Whether it matters at full scale depends on how much the corpus repeats (v3 sees ≈ 2 epochs): hence "off in the main run, mandatory arm, switch-off on repeated vs novel text".</p></div>`);
  }

  function tPool() {
    const cols = ["k4_attnffn_s0", "k6_attnffn_s0", "k6_attnffn_p07", "k6_gconv_p07"];
    const ratios = a => a.decomp && a.decomp.text_by_slot ? a.decomp.text_by_slot.map(x => x.ratio) : null;
    const facts = a => a.decomp && a.decomp.facts_by_slot ? a.decomp.facts_by_slot.map(x => x.ratio) : null;
    const rowv = (vals, good) => vals ? vals.map(v => `<td style="background:color-mix(in srgb, ${v > 1 ? "#e07b39" : "#2e9c8c"} ${Math.round(Math.min(60, Math.abs(Math.log(v)) * 120))}%, transparent)">${v.toFixed(2)}</td>`).join("") : "";
    const prof = a => { const p = a.final && a.final.pool && a.final.pool.slot_profile; if (!p) return ""; const k = p[0].length, mean = p[0].map((_, s) => p.reduce((t, h) => t + h[s], 0) / p.length);
      return `<div class="rs-br sm"><div class="rs-bl"><b style="color:${ARMC[a.name]}">${a.label}</b><span>mean weight per slot (uniform ${(1 / k).toFixed(2)})</span></div><div style="display:flex;gap:3px;align-items:flex-end;height:42px">${mean.map(v => `<div title="${v.toFixed(2)}" style="flex:1;height:${Math.round(100 * v / Math.max(...mean))}%;background:${ARMC[a.name]};opacity:.75;border-radius:3px 3px 0 0"></div>`).join("")}</div><div class="rs-bv">${mean.at(-1).toFixed(2)} <small>newest</small></div></div>`; };
    return sec("v4t-pool", "What the pooler learned", "Within-slot weight ratios (full chunks; same slot, so position cannot produce the effect) and the per-slot profile.",
      `<div class="grid g2" style="align-items:start"><div class="card scroll-x"><h4>Text: top-100 frequent tokens ÷ the rest, per slot</h4><table class="rs-mx"><tbody>
        ${cols.map(c => { const v = ratios(arm(c)); return v ? `<tr><td style="color:${ARMC[c]}"><b>${arm(c).label}</b></td>${rowv(v)}</tr>` : ""; }).join("")}</tbody></table>
        <h4 style="margin-top:14px">Probe facts: names and values ÷ template words, per slot</h4><table class="rs-mx"><tbody>
        ${cols.map(c => { const v = facts(arm(c)); return v ? `<tr><td style="color:${ARMC[c]}"><b>${arm(c).label}</b></td>${rowv(v)}</tr>` : ""; }).join("")}</tbody></table>
        <p class="rs-note">Values ≠ 1 inside one slot mean the pooler weights tokens by what they are (v3's pool cannot). Today it favours frequent and template tokens (orange), not names and values: the draft predicts this until retrieval pays. Facts ratios carry the filler-memorisation caveat. gconv: gate share, not attention weight.</p></div>
      <div class="card rs-bars"><h4>Slot profiles: recency on top of content</h4>${R4.arms.filter(a => a.final && a.final.pool && a.name !== "k6_gconv_p07").map(prof).join("")}
        <p class="rs-note">Position explains 57–71 % of the attention weights' variance (gconv's gates: 6–8 %, its position lives in the slot weights w_r). v3's trained pool was 1.6× on its newest slot; the attention pooler is 2–3×, but it adds the content preference v3 could not have.</p></div></div>`);
  }

  function tTier3() {
    const t = ["k6_attnffn_p07", "k6_linear_p07", "k6_gconv_p07"].map(arm);
    const rows = [1, 2, 3, 4].map(m => `<tr><td><b>${m}M</b></td>${t.map(a => { const e = a.evals.find(x => Math.round(x.real / 1e6) === m); return `<td>${e ? e.val.toFixed(3) : "—"}</td>`; }).join("")}${t.map(a => { const e = a.evals.find(x => Math.round(x.real / 1e6) === m); return `<td>${e ? e.ppl0.toFixed(1) : "—"}</td>`; }).join("")}</tr>`).join("");
    const k4 = arm("k4_attnffn_s0"), k6 = arm("k6_attnffn_s0");
    const kser = [{ name: "k6 − k4 (nats)", color: ACC, dots: true, pts: k6.evals.map((e, i) => [e.real / 1e6, e.val - k4.evals[i].val]) }];
    const fin_ = t.map(a => `<tr><td style="color:${ARMC[a.name]}"><b>${a.label}</b></td><td>${(a.final.real / 1e6).toFixed(2)}M</td><td>${a.final.val.toFixed(3)}</td><td>${Math.round(a.final.probes.lr_recall1 * 192)} / 192</td><td>${a.final.codebook.coarse_ppl.toFixed(1)}</td></tr>`).join("");
    return sec("v4t-tier3", "Tier 3 · comparing the variants", "Single seed, tiny scale: indicative only. Rules were written down before the numbers existed.",
      `<div class="grid g2" style="align-items:start"><div class="card scroll-x"><h4>Poolers at matched tokens (k 6, p 0.7, same seed and data)</h4>
        <table class="rs-mx"><thead><tr><th>tokens</th><th>attention + FFN</th><th>linear / plain conv</th><th>gated conv</th><th>ppl att.</th><th>ppl lin.</th><th>ppl gconv</th></tr></thead><tbody>${rows}</tbody></table>
        <p class="rs-note"><b>Tied</b> on language within checkpoint wobble; attention − linear at the matched finals (5.00M, paired over 64 rows) = <b>+0.017 ± 0.018</b> nats. The one consistent difference: the linear pool's coarse perplexity is lowest at every row (concepts collapse onto fewer codes).</p>
        <h4 style="margin-top:12px">Finals (gconv ran longer and decayed fully: not token-matched)</h4><table class="rs-mx"><thead><tr><th>arm</th><th>tokens</th><th>val</th><th>recall1</th><th>coarse ppl</th></tr></thead><tbody>${fin_}</tbody></table></div>
      <div class="card"><h4>Compression: k = 6 vs k = 4 (p 0.5, matched rows)</h4>${LC.chart("v4t-k", { series: kser, xlabel: "real tokens (M)", xname: "tokens (M)", xfmt: v => (+v).toFixed(1), refs: [{ y: 0.03, label: "draft gate +0.03", color: "#d1a92f" }, { y: 0, label: "", color: "#8a93a3" }], y0: -0.005, y1: 0.045, tfmt: v => (v >= 0 ? "+" : "") + (+v).toFixed(3), W: 560, H: 250 })}
        <p class="rs-note">The gap widens steadily (+0.015 → +0.035) and reaches the draft's +0.03 gate at ≈ 4–5M tokens, for 1.07× speed on CPU (does not transfer). No paired SE exists here (different packing per k). The full-scale gate at step 4,050 with the mandatory k = 4 control decides. The synthetic share: 0.7 vs 0.5 cost 0.15–0.29 nats of text and bought no recall.</p></div></div>`);
  }

  function tBrief() {
    const rows = [["Concept module (core)", "+0.21 / +0.12 · plans row-specific", "n", "Used"], ["Second pass", "+0.008 / +0.014", "n", "Used, small"], ["F-layer loop", "+0.022 / +0.002", "a", "k-dependent"],
      ["Encoder shared FFN", "+0.024 / +0.014", "n", "Used"], ["Core shared FFN", "+0.008 / +0.012", "n", "Used"], ["Attention pooler", "content-selective (not a per-slot map)", "n", "Works"],
      ["Pooler FFN", "writes, Δ ≈ 0", "a", "Not useful yet"], ["Decoder Engram reads", "−0.006 / −0.023 text; −0.14 to −0.17 unseen; +0.47 to +0.70 memorised", "r", "Memorises"],
      ["Two-level codebook", "fine VE 0.72 (bound 0.75), coarse ppl 46–62 % of K", "n", "Healthy"], ["ĉ_pred in the plan", "+0.024 / +0.015", "n", "Used"], ["6 vs 4 tokens per concept", "+0.035 nats at 5M, 1.07× faster", "a", "At the gate"]];
    return sec("v4t-brief", "Your brief, part by part (tiny scale)", "Δ val NTP when removed, k4 / k6 (p 0.5), single seed.",
      `<div class="card"><table class="t v3-vtab"><tbody>${rows.map(([a, b, l, p]) => `<tr><td class="v3-vc">${a}</td><td class="v3-ve">${b}</td><td>${pill(l, p)}</td></tr>`).join("")}</tbody></table></div>`);
  }

  function tImpl() {
    const rows = [["§5.2 decoder Engram reads", "memorisation channel", "off in the main run; mandatory arm; add a switch-off on repeated vs novel text (reverses an item of the brief — owner's call)"],
      ["§3.5.4 pooler FFN", "writes, Δ ≈ 0", "priced arm with the 0.01-nat kill line at step 4,050, or drop it (−294,913 params)"],
      ["§3.5.4 pooler alternatives", "gated conv ties, 130k cheaper; plain conv = v3's pool", "add the gated conv as a priced arm"],
      ["§3.5.1 why replace v3's pool", "per-slot pool collapses the coarse code", "add as a second argument"],
      ["§8.4 telemetry", "slot averages mix position and content", "log within-slot class ratios + position share"],
      ["§3.3 F loop", "+0.022 at k 4, +0.002 at k 6", "keep the no-loop arm"], ["§3.5.5 k = 6 gate", "gap ≈ +0.03 by 4–5M", "keep the gate as written"], ["§6 / §1.3b codebook", "fine VE 0.72 vs bound 0.75", "no change"]];
    return sec("v4t-impl", "What this changes in draft_v4", "Recommendations for the owner; the draft itself is not edited.",
      `<div class="card scroll-x"><table class="t"><thead><tr><th>Draft section</th><th>Finding</th><th>Recommendation</th></tr></thead><tbody>${rows.map(r => `<tr><td style="white-space:normal;width:22%"><b>${r[0]}</b></td><td style="white-space:normal;width:30%">${r[1]}</td><td style="white-space:normal">${r[2]}</td></tr>`).join("")}</tbody></table></div>`);
  }

  function tChecklist() {
    const rows = [["Model and acceptance harness", "done (v4_tiny/code)", "—", "g", "Done"], ["Data rebuild at k = 6 (T 1,020, block 60)", "not started", "CPU, now · 1–2 h", "n", "Next"],
      ["v4 trainer: port v3's train.py (guards, re-exec, cosine, two-level codebook, eval suite)", "not started", "CPU, now · ≈ ½ day", "n", "Next"],
      ["Two uncovered tests + full-config smoke", "not started", "CPU · 1 h", "n", "Next"], ["MPS compile / gradient checks (k_norm eager pin)", "not started", "GPU after v3 · 1 h", "n", "After v3"],
      ["profile_step + interleaved A/B on the real model", "not started", "GPU after v3 · 1–2 h", "n", "After v3"], ["Memory-growth trace (v3 re-execs at 24 GiB)", "open", "GPU · 2 h", "a", "Open"],
      ["Owner decisions §10.6: RL door A + C vs B · core 4 × 2 vs 5 × 2 · k 6 default / k 8 arm · learned boundaries as v5", "open", "owner", "a", "Owed"]];
    return sec("v4t-check", "Launch checklist", "What exists and what is left (draft §8.4–§8.5).",
      `<div class="card scroll-x"><table class="t v3-vtab"><tbody>${rows.map(r => `<tr><td class="v3-vc">${r[0]}</td><td class="v3-ve">${r[1]} · ${r[2]}</td><td>${pill(r[3], r[4])}</td></tr>`).join("")}</tbody></table>
      <p class="rs-note"><b>What in v3's 48 h evaluation would change the call:</b> second-pass gain on test (basis for M = 2) · shared-FFN switch-off on test · ĉ_pred contribution ≈ 0.01 (RL-door framing) · the v4 design rules on test + OOD · and v3's ≈ 2.5 % token shortfall (V3-018) when comparing counts.</p></div>`);
  }

  function tCost() {
    const H = R4.v3_heartbeats, ts = s => { const [h, m] = s.split(":").map(Number); return h + m / 60; };
    const mk = (n, arr, c) => ({ name: n, color: c, dots: true, pts: arr.map(([t, v]) => [ts(t), v]) });
    const ser = [mk("reference (nothing on P-cores)", H.reference, "#168b76"), mk("two arms", H.two_arms, "#e07b39"), mk("three arms", H.three_arms, "#d0667a"), mk("finals + analysis", H.tail, "#a07cc5"), mk("recovered", H.recovered, "#3b78c2")];
    return sec("v4t-cost", "What the experiment cost v3", "v3's training speed (tok/s per heartbeat) during the owner-approved CPU window, 14:25–18:04.",
      `<div class="grid g2" style="align-items:start"><div class="card">${LC.chart("v4t-hb", { series: ser, xlabel: "time of day (h)", xname: "time", xfmt: v => `${Math.floor(v)}:${String(Math.round((v % 1) * 60)).padStart(2, "0")}`, tfmt: v => Math.round(v) + " tok/s", y0: 1400, y1: 3500, W: 560, H: 250 })}</div>
      <div class="card"><div class="kvt"><span>reference</span><b>2,907–3,229 tok/s (E-cores only or idle)</b><span>two arms</span><b>2,121–2,474 (≈ −27 %)</b><span>three arms</span><b>1,871–2,472 (≈ −35 %)</b><span>worst</span><b>1,563 (five analysis jobs, 17:57)</b><span>recovered</span><b>2,900 at 18:10 · 3,358 at 18:34</b><span>tokens lost</span><b>≈ 13–14M ≈ 400–420 steps ≈ 2.5 % of the run (time-based schedule)</b><span>guards</span><b>no new event · arithmetic unaffected</b><span>v3 quality</span><b>new best val at every hourly (2.513 → 2.497 → 2.486)</b></div>
      <p class="rs-note">Recorded as V3-018 in run_v3_19M/DECISIONS.md; read v3's 48 h final with it in mind.</p></div></div>`);
  }

  function tSetup() {
    const a = arm("k6_attnffn_s0"), p = a.params;
    return sec("v4t-setup", "Setup", "Same topology as the full v4, widths divided by about 4.",
      `<div class="grid g2" style="align-items:start"><div class="card"><div class="kvt">
        <span>model</span><b>d 64, V 8,192, T 480; 8-layer SSSFSSSF encoder (F × 2, shared FFN 96); pooler 4 heads + FFN 96; core 4 × 2 (own 184 + shared 256); coarse K 64 + PQ 8 × 256 × 8; 4 confined decoder layers + 2 Engram reads</b>
        <span>compute params</span><b>${fmtI(p.compute)} (k 6) · pooler ${fmtI(p.parts.router_pooler_norm)} incl. router</b>
        <span>data</span><b>457,614 documents (38.6M tokens) from 40,000 random rows of v3's shard, re-packed per k in the same order; probes recall1 / carry3 / multi4 with 80–200-token gaps (beyond the decoder's 70)</b>
        <span>training</span><b>B 4 × 480, AdamW, lr 2e-3, warm-up 400, cosine to 0.1, k-means++ codebook init, restarts every 200 steps</b>
        <span>hardware</span><b>Apple M2 CPU (P-cores, 2 threads per arm), while v3 trained on the GPU</b></div></div>
      <div class="card"><h4>Arms</h4><table class="t">${R4.arms.map(x => `<tr><td style="color:${ARMC[x.name]}"><b>${x.label}</b></td><td>${(x.final.real / 1e6).toFixed(2)}M tokens · ${fmtI(x.final.steps)} steps</td><td>${Math.round(x.tok_s)} tok/s</td></tr>`).join("")}</table>
        <p class="rs-note">Documents: <a href="data/v4_REPORT.md" target="_blank" rel="noopener">REPORT.md</a> · <a href="data/v4_GO_NOGO.md" target="_blank" rel="noopener">GO_NOGO.md</a> · <a href="data/draft_v4.md" target="_blank" rel="noopener">draft_v4.md</a>.</p></div></div>`);
  }

  function renderTiny() {
    if (tinyDone) return; tinyDone = true;
    const root = document.getElementById("v4t-root"); if (!root) return;
    if (!R4) { root.innerHTML = `<div class="card empty-note">The tiny-experiment results are not available (data/v4_tiny.json).</div>`; return; }
    document.getElementById("v4t-meta").textContent = `${R4.arms.length} arms · ${R4.tests.length} tests · ${R4.mutants.length} mutants · ${R4.advisor_consults} advisor consults · source: ${R4.source}`;
    const parts = [tVerdict, tKpis, tFindings, tCorrect, tFlaw, tCurves, tSwitch, tReads, tPool, tTier3, tBrief, tImpl, tChecklist, tCost, tSetup];
    root.innerHTML = parts.map(f => { try { return f(); } catch (e) { console.error("v4 tiny:", f.name, e); return ""; } }).join("");
  }

  /* ================================================================== wiring */
  async function ensure() {
    if (!A4) { const r = await fetch("data/arch_v4.json"); if (!r.ok) throw new Error("v4 architecture data unavailable"); A4 = await r.json(); window.ARCH4 = A4; }
    if (!R4) { try { const r = await fetch("data/v4_tiny.json"); R4 = r.ok ? await r.json() : null; } catch (e) { R4 = null; } }
    return A4;
  }
  async function shown(id) {
    if (!id.startsWith("v4")) return;
    try { await ensure();
      if (id === "v4overview" && !document.getElementById("v4-stats").innerHTML) renderOverview();
      if (id === "v4arch2d") renderFigs();
      if (id === "v4tiny") renderTiny();
    } catch (e) { console.error("v4 page", e); }
  }
  window.addEventListener("tabshown", e => shown(e.detail));
  document.addEventListener("click", e => { const a = e.target.closest("[data-fig4]"); if (a) { e.preventDefault(); ensure().then(() => { renderFigs(); goto(a.dataset.fig4); }); } });
  const initial = (document.querySelector("section.tab.on") || {}).id; if (initial && initial.startsWith("tab-v4")) shown(initial.slice(4));
})();
