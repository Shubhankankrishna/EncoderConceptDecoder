/* v3 · 24 h evaluation tab: run_v3_19M/code/final_eval.py --split val at the 24 h milestone (data/v3_eval_24h.json, symlink,
   read-only). Same components and classes as the v1 Results tab (js/final.js: rs-kpis, rs-find, rs-bars, rs-mx, rs-syn, rs-lp,
   kvt, rs-gen); every number is read from the file. Validation split: the test split stays sealed until the 48 h evaluation. */
(function () {
  let E = null, gen = "tinystories";
  const PK = ["lr_recall1", "lr_carry3", "lr_multi4", "lr_hop", "lr_bind"];
  const PN = { lr_recall1: "recall1", lr_carry3: "carry3", lr_multi4: "multi4", lr_hop: "hop", lr_bind: "bind" };
  const IV = [
    ["intact", "Intact", "the full model, 3 passes"],
    ["c_off", "Predicted concept off", "plan without the ĉ_pred term (only u)"],
    ["u_off", "Core state off", "plan without the u term (only ĉ_pred)"],
    ["core_off", "Concept core off", "learned null plan everywhere: the decoder has only its last 70 tokens"],
    ["plan_swap", "Plan swapped", "the plan of a different document"],
    ["engram_off", "Engram off", "both Engram reads removed"],
    ["engram_read2_off", "Engram read 2 off", "the second read (after encoder layer 4) removed"],
    ["F_off", "Encoder F layers off", "the two full-attention encoder layers skipped"],
    ["table_off", "(t, Δ) table off", "every table gain set to exactly 1"],
    ["shared_ffn_off", "Shared FFN off", "the core's shared SwiGLU removed"],
  ];
  const MS = [["R1", "1 pass", "(0, 1)"], ["R2a", "2 passes · 2a", "(0, ⅓) → (⅓, ⅔)"], ["R2b", "2 passes · 2b", "(0, ⅔) → (⅔, ⅓)"], ["R3", "3 passes (trained main path)", "(0, ⅓) → (⅓, ⅓) → (⅔, ⅓)"]];
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fin = v => v != null && isFinite(v);
  const n3 = v => fin(v) ? Number(v).toFixed(3) : "—", n2 = v => fin(v) ? Number(v).toFixed(2) : "—", n4 = v => fin(v) ? Number(v).toFixed(4) : "—";
  const pct = v => fin(v) ? (100 * v).toFixed(1) + "%" : "—";
  const dchip = (d, lowerBetter = true, dig = 3) => !fin(d) ? "" :
    `<span class="rs-d ${Math.abs(d) < 5e-4 ? "z" : (d < 0) === lowerBetter ? "g" : "b"}">${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d).toFixed(dig)}</span>`;
  const heat = v => `background: color-mix(in srgb, #2e9c8c ${Math.round(8 + 80 * Math.max(0, Math.min(1, v || 0)))}%, transparent)`;
  const sec = (id, title, sub, body) => `<section class="rs-sec" id="${id}"><div class="sh"><div><h3>${title}</h3><p>${sub}</p></div></div>${body}</section>`;
  const fmtDur = s => { const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60); return h ? `${h} h ${m} min` : `${m} min`; };
  const meanL = k => (E.lang.tinystories[k].delta_vs_intact + E.lang.wikitext[k].delta_vs_intact) / 2;

  async function load() {
    try { const r = await fetch("data/v3_eval_24h.json?t=" + Date.now(), { cache: "no-store" }); E = r.ok ? JSON.parse((await r.text()).replace(/\bNaN\b|-?\bInfinity\b/g, "null")) : null; }
    catch (e) { E = null; }
    render();
  }

  function render() {
    const root = document.getElementById("v3e-root"); if (!root) return;
    if (!E || !E.lang) { root.innerHTML = `<div class="card empty-note">The 24 h evaluation is not available yet (run_v3_19M/runs/v3_48h/eval_24h.json).</div>`; return; }
    document.getElementById("v3e-meta").textContent = `Evaluated ${E.time} · step ${E.step.toLocaleString()} · ${E.split} split · main path M = ${E.eval_R} · ${fmtDur(Object.values(E.seconds || {}).reduce((a, b) => a + b, 0))} of evaluation`;
    const parts = [kpis, findings, secLang, secM, secTrunc, secProbes, secSynth, secLinear, secHealth, secAudit, secGen];
    root.innerHTML = parts.map(f => { try { return f(); } catch (e) { console.error("v3 24 h eval:", f.name, e); return ""; } }).join("");
  }

  /* ---------------------------------------------------------------- KPI tiles + findings */
  function kpis() {
    const t = E.items.val.intact, sw = E.items.val.plan_swap, cb = E.health.codebook, L = E.lang;
    const tile = (l, v, s) => `<div class="rs-k"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
    return `<div class="rs-kpis">
      ${tile("TinyStories val", n3(L.tinystories.intact.nats), `nats · ${n3(L.tinystories.intact.bpb)} bits/byte`)}
      ${tile("WikiText-103 val", n3(L.wikitext.intact.nats), `nats · ${n3(L.wikitext.intact.bpb)} bits/byte`)}
      ${tile("recall1 · val", pct(t.lr_recall1.exact), `plan swapped: ${pct(sw.lr_recall1.exact)} (chance ≈ 0.8%)`)}
      ${tile("multi4 · val", pct(t.lr_multi4.exact), `carry3 ${pct(t.lr_carry3.exact)} · hop ${pct(t.lr_hop.exact)}`)}
      ${tile("Synthetic tasks", pct(t._macro_synthetic), `macro over ${Object.keys(t).filter(k => !k.startsWith("lr_") && !k.startsWith("_")).length} families`)}
      ${tile("Concept codebook", pct(cb.codes_used_frac_min), `codes used · VE ${n3(cb.var_explained)} · perplexity ${n2(cb.perplexity_mean)} of 128`)}
    </div>`;
  }
  function findings() {
    const T = E.items.val, P = E.probes, L = E.lang;
    const maxSwap = Math.max(...PK.filter(k => k !== "lr_bind").map(k => T.plan_swap[k].exact));
    const m1 = meanL("R1"), tab = meanL("table_off"), sh = meanL("shared_ffn_off"), eng = meanL("engram_off"), core = meanL("core_off");
    const br = Object.values(P.lr_hop_bridge_by_pass).map(v => v.acc), ge1 = Object.values(P.G_E1_concept_to_token).map(v => v.acc);
    const F = [
      ["g", "The concept stream carries long-range information", `With another document's plan every long-range probe falls to chance (at most ${pct(maxSwap)} on recall1 / carry3 / multi4 / hop), from ${pct(T.intact.lr_recall1.exact)} recall1 intact.`],
      ["g", "The concept core and Engram both carry real load", `Concept core off costs +${n3(core)} nats (mean of the two languages); Engram off +${n3(eng)} (WikiText +${n3(L.wikitext.engram_off.delta_vs_intact)}).`],
      ["a", "Extra passes add almost nothing yet", `1 pass instead of 3 costs only ${dchip(m1)} nats; 2a / 2b are within ${n3(Math.max(meanL("R2a"), meanL("R2b")))} of 3 passes. hop with 1 pass ${pct(T.R1.lr_hop.exact)} vs ${pct(T.intact.lr_hop.exact)} with 3.`],
      ["a", "The (t, Δ) table and the shared FFN are nearly idle", `Switching them off costs ${dchip(tab)} and ${dchip(sh)} nats although both opened from zero and grew (table norm ${n2(E.health.telemetry.table_norm)}, shared ‖W₂‖ ${n2(E.health.telemetry.shared_w2_norm)}).`],
      ["r", "The passes do not compute the intermediate hop", `A linear probe finds the lr_hop bridge entity equally often after pass 1, 2 and 3 (${br.map(pct).join(" / ")}); concepts keep their 4 tokens only ${pct(Math.min(...ge1))}–${pct(Math.max(...ge1))} of the time.`],
    ];
    return `<div class="rs-find">${F.map(([l, t, x]) => `<div class="rs-f ${l}"><i></i><div><b>${t}</b><p>${x}</p></div></div>`).join("")}</div>`;
  }

  /* ---------------------------------------------------------------- language: switch-off tests and pass schedules */
  function bars(name, label, rows, note) {
    const d = E.lang[name], base = d.intact.nats, vals = rows.map(([k]) => d[k].nats), lo = Math.min(...vals) - 0.1, hi = Math.max(...vals);
    return `<div class="card rs-bars"><h4>${label} · val</h4>${rows.map(([k, t, x]) => `<div class="rs-br" title="${esc(x)}">
        <div class="rs-bl"><b>${t}</b><span>${x}</span></div>
        <div class="rs-bt"><div class="${k === "intact" || k === "R3" ? "in" : ""}" style="width:${(100 * (d[k].nats - lo) / (hi - lo)).toFixed(1)}%"></div></div>
        <div class="rs-bv">${n3(d[k].nats)}<small>${k === "intact" || k === "R3" ? "nats" : ""}</small>${k === "intact" || k === "R3" ? "" : dchip(d[k].nats - base)}</div></div>`).join("")}
      <div class="rs-note">${note || `bits/byte intact ${n3(d.intact.bpb)} · ${d.intact.tokens.toLocaleString()} tokens · bars start at ${n2(lo)} nats`}</div></div>`;
  }
  function secLang() {
    return sec("v3e-lang", "Language: what each part is worth", "Next-token loss on held-out validation text with one part switched off (lower is better; Δ vs intact). Fixed-checkpoint effects, not training-time value.",
      `<div class="grid g2">${bars("tinystories", "TinyStories", IV)}${bars("wikitext", "WikiText-103", IV)}</div>`);
  }
  function secM() {
    const z = (name) => { const d = E.lang[name]; return Math.min(...MS.map(([k]) => d[k].nats)); };
    const mb = (name, label) => {
      const d = E.lang[name], lo = z(name) - 0.002, hi = Math.max(...MS.map(([k]) => d[k].nats));
      return `<div class="card rs-bars"><h4>${label} · val</h4>${MS.map(([k, t, x]) => `<div class="rs-br sm"><div class="rs-bl"><b>${t}</b><span>${x}</span></div>
        <div class="rs-bt"><div class="${k === "R3" ? "in" : ""}" style="width:${(100 * (d[k].nats - lo) / Math.max(hi - lo, 1e-9)).toFixed(1)}%"></div></div>
        <div class="rs-bv">${n4(d[k].nats)}${k === "R3" ? "" : dchip(d[k].nats - d.R3.nats, true, 4)}</div></div>`).join("")}
        <div class="rs-note">Bars start at ${n3(lo)} nats (zoomed: the whole spread is ${n4(hi - z(name))} nats).</div></div>`;
    };
    const T = E.items.val, cols = [["R1", "1"], ["R2a", "2a"], ["R2b", "2b"], ["intact", "3"]];
    const tbl = `<div class="card"><h4>Long-range probes by pass schedule</h4><div class="scroll-x"><table class="rs-mx"><tr><th>probe</th>${cols.map(([k, t]) => `<th class="${k === "intact" ? "in" : ""}">${t}</th>`).join("")}</tr>
      ${PK.map(p => `<tr><td><b>${PN[p]}</b></td>${cols.map(([k]) => `<td style="${heat(T[k][p].exact)}" class="${k === "intact" ? "in" : ""}">${pct(T[k][p].exact)}</td>`).join("")}</tr>`).join("")}
      <tr><td><b>synthetic macro</b></td>${cols.map(([k]) => `<td style="${heat(T[k]._macro_synthetic)}" class="${k === "intact" ? "in" : ""}">${pct(T[k]._macro_synthetic)}</td>`).join("")}</tr></table></div>
      <div class="rs-note">512 items per probe: one item is 0.2 points. Trained with 3 main passes plus the 1 / 2a / 2b shortcuts.</div></div>`;
    return sec("v3e-m", "Pass schedules: does thinking longer help?", "The core run with 1, 2 or 3 passes on the grid of thirds at evaluation. This is v3's central question; the like-for-like answer at 24 h is: barely.",
      `<div class="grid g2">${mb("tinystories", "TinyStories")}${mb("wikitext", "WikiText-103")}</div><div style="margin-top:16px">${tbl}</div>`);
  }
  function secTrunc() {
    const rs = [72, 128, 256, 512, 1024];
    const card = (name, label) => {
      const d = E.trunc[name], vals = rs.map(r => d["r" + r].nats).concat(d.swap_r1024.nats), lo = Math.min(...vals) - 0.05, hi = Math.max(...vals), s = d.summary;
      return `<div class="card rs-bars"><h4>${label}</h4>
        ${rs.map(r => `<div class="rs-br sm"><div class="rs-bl"><b>last ${r} tokens</b></div><div class="rs-bt"><div class="${r === 1024 ? "in" : ""}" style="width:${(100 * (d["r" + r].nats - lo) / (hi - lo)).toFixed(1)}%"></div></div><div class="rs-bv">${n3(d["r" + r].nats)}${r > 72 ? dchip(d["r" + r].nats - d.r72.nats) : ""}</div></div>`).join("")}
        <div class="rs-br sm"><div class="rs-bl"><b>1024 + plan swapped</b></div><div class="rs-bt"><div class="bad" style="width:${(100 * (d.swap_r1024.nats - lo) / (hi - lo)).toFixed(1)}%"></div></div><div class="rs-bv">${n3(d.swap_r1024.nats)}${dchip(d.swap_r1024.nats - d.r1024.nats)}</div></div>
        <div class="rs-chips"><span class="chip">gain beyond 70 tokens <b>${n3(s.gain_beyond_70)}</b> nats</span><span class="chip">cost of a wrong plan <b>${n3(s.swap_cost)}</b> nats</span><span class="chip">${s.rows} rows · ${s.rows_ctx_ge_512} with ≥ 512 context</span></div></div>`;
    };
    return sec("v3e-trunc", "How far back does the model use?", "Loss on the last 16 positions of each row with the visible history truncated. The decoder alone reaches 70 tokens; anything further must come through the concepts. TinyStories rows are mostly short stories, so WikiText is the meaningful long-range test.",
      `<div class="grid g2">${card("tinystories", "TinyStories")}${card("wikitext", "WikiText-103")}</div>`);
  }

  /* ---------------------------------------------------------------- probes, synthetic tasks */
  function secProbes() {
    const S = E.items.val, cols = IV.map(([k, t]) => [k, t]);
    const table = `<div class="scroll-x"><table class="rs-mx"><tr><th>probe</th>${cols.map(([k, t]) => `<th class="${k === "intact" ? "in" : ""}">${t}</th>`).join("")}</tr>
      ${PK.map(p => `<tr><td><b>${PN[p]}</b></td>${cols.map(([k]) => { const v = S[k]?.[p]?.exact; return `<td style="${heat(v)}" class="${k === "intact" ? "in" : ""}">${pct(v)}</td>`; }).join("")}</tr>`).join("")}</table></div>`;
    const offs = ["lr_carry3", "lr_multi4"].map(p => `<div class="rs-off"><b>${PN[p]}</b>${S.intact[p].by_offset.map((v, i) => `<div class="rs-bt t" title="answer token ${i + 1}"><div style="width:${100 * v}%"></div><span>tok ${i + 1} · ${pct(v)}</span></div>`).join("")}<span class="rs-note">exact (all 3 right): ${pct(S.intact[p].exact)}</span></div>`).join("");
    const h = S.intact.lr_hop, h1 = S.R1.lr_hop;
    const hop = `<div class="rs-off"><b>hop by depth</b>${[2, 3, 4].map(k => `<div class="rs-bt t" title="${k} links"><div style="width:${100 * (h["exact_h" + k] || 0)}%"></div><span>${k} links · ${pct(h["exact_h" + k])} (1 pass ${pct(h1["exact_h" + k])})</span></div>`).join("")}<span class="rs-note">exact: ${pct(h.exact)} with 3 passes, ${pct(h1.exact)} with 1</span></div>`;
    return sec("v3e-probes", "Long-range probes", "Answers hidden 128–420 tokens back, 512 validation items per probe, scored by exact match; chance ≈ 0.8% for recall1, ≈ 0 for the others. Columns: every switch-off test.",
      `<div class="card">${table}<div class="rs-offs">${offs}${hop}</div></div>`);
  }
  function secSynth() {
    const T = E.items.val, I = T.intact, C = T.core_off;
    const fams = Object.keys(I).filter(k => !k.startsWith("lr_") && !k.startsWith("_")).sort((a, b) => I[b].exact - I[a].exact);
    return sec("v3e-synth", "Synthetic reasoning tasks", "Exact-match accuracy per family (validation items), intact vs with the concept core switched off, sorted by intact accuracy.",
      `<div class="card"><div class="rs-syn">${fams.map(f => `<div class="rs-sr"><span class="nm">${f.replace(/_/g, " ")}</span>
        <div class="rs-bt"><div style="width:${100 * I[f].exact}%"></div></div><span class="v">${pct(I[f].exact)}</span>
        <div class="rs-bt o"><div style="width:${100 * (C[f]?.exact ?? 0)}%"></div></div><span class="v">${pct(C[f]?.exact)}</span></div>`).join("")}</div>
        <div class="rs-legend"><span><i class="t"></i>intact</span><span><i class="o"></i>concept core off</span><span>macro average: intact <b>${pct(I._macro_synthetic)}</b> · core off <b>${pct(C._macro_synthetic)}</b> · plan swapped <b>${pct(T.plan_swap._macro_synthetic)}</b></span></div></div>`);
  }

  /* ---------------------------------------------------------------- linear probes, health, audit */
  function secLinear() {
    const P = E.probes, g = P.G_E1_concept_to_token;
    const row = (lbl, v, gate) => `<div class="rs-lp"><span class="nm">${lbl}</span><div class="rs-bt g"><div style="width:${100 * v.acc}%"></div><i class="base" style="left:${100 * v.majority_baseline}%" title="majority baseline ${pct(v.majority_baseline)}"></i>${gate ? `<i class="gate" style="left:${100 * gate}%" title="gate ${pct(gate)}"></i>` : ""}</div><span class="v">${pct(v.acc)}</span><span class="b">base ${pct(v.majority_baseline)}</span></div>`;
    const fams = PK.map(p => [p, P["u_query_to_answer_" + p]]).filter(([, v]) => v);
    return sec("v3e-linear", "What the representations contain (linear probes)", "A linear read-out trained on 80% and scored on 20%. Grey tick: majority baseline; red tick: v1's pre-registered G-E1 gate.",
      `<div class="grid g2">
        <div class="card"><h4>Concept → each of its 4 tokens</h4>${Object.entries(g).map(([k, v]) => row(`token ${+k.slice(3) + 1}`, v, 0.99)).join("")}
          <div class="rs-note">${g.pos0.n_train.toLocaleString()} train / ${g.pos0.n_test.toLocaleString()} held-out concepts, ≈ ${g.pos0.classes.toLocaleString()} token classes.</div></div>
        <div class="card"><h4>Core state u at the question → answer tokens</h4>${fams.map(([p, v]) => Object.entries(v).map(([k, x]) => row(`${PN[p]} · answer token ${+k.slice(3) + 1}`, x)).join("")).join("")}
          <h4 style="margin-top:12px">hop · bridge entity after pass 1 / 2 / 3</h4>${Object.entries(P.lr_hop_bridge_by_pass).map(([k, v]) => row(k.replace("pass", "after pass "), v)).join("")}
          <div class="rs-note">Flat across passes: the passes do not yet compute the intermediate hop.</div></div></div>`);
  }
  function secHealth() {
    const cb = E.health.codebook, te = E.health.telemetry, LD = E.health.loop_dynamics_lr_hop, mix = te.crc_mix_e0_L_G;
    const ld = (s, t) => { const d = LD[s]; if (!d) return ""; return `<div class="card"><h4>${t}</h4><div class="kvt">
        <span>state norm per pass</span><span>${d.state_norm.map(n2).join(" → ")}</span>
        <span>step size per pass</span><span>${d.step.map(n2).join(" → ")}</span>
        <span>angle between steps</span><span>${d.consecutive_step_angle_deg.map(v => n2(v) + "°").join(" → ")}</span>
        <span>ρ = |S| / Σ|ΔS|</span><span><b>${n3(d.rho)}</b></span>
        <span>gradient pass 1 / last</span><span>${n2(d.grad_ratio_S1_over_SM)}</span></div></div>`; };
    return sec("v3e-health", "Concept stream health", "Codebook use, what goes into a concept, and how the core state moves pass by pass on hop items.",
      `<div class="grid g3">
        <div class="card"><h4>Codebook (${cb.n_concepts.toLocaleString()} val concepts)</h4><div class="kvt">
          <span>codes used (min / mean)</span><span><b>${pct(cb.codes_used_frac_min)}</b> / ${pct(cb.codes_used_frac_mean)}</span>
          <span>perplexity (of 128)</span><span><b>${n2(cb.perplexity_mean)}</b></span>
          <span>variance explained</span><span><b>${n3(cb.var_explained)}</b></span>
          <span>next-concept loss, 3 passes</span><span><b>${n3(cb.ncp_ce)}</b> vs entropy ${n3(cb.code_entropy)}</span>
          <span>next-concept loss, 1 pass</span><span>${n3(cb.ncp_ce_M1)} (${dchip(cb.ncp_ce_M1 - cb.ncp_ce)} vs 3)</span></div></div>
        <div class="card"><h4>What goes into a concept</h4><div class="rs-stack">${[["embedding", mix[0], "#9aa3af"], ["local L", mix[1], "#3b78c2"], ["global G", mix[2], "#e07b39"]].map(([n, v, c]) => `<div style="width:${100 * v}%;background:${c}" title="${n} ${pct(v)}">${v > 0.12 ? n + " " + pct(v) : ""}</div>`).join("")}</div>
          <div class="kvt" style="margin-top:12px"><span>embedding / local / global</span><span>${mix.map(pct).join(" / ")}</span><span>Engram gate α (mean)</span><span><b>${n3(te.engram_alpha_mean)}</b></span>
          <span>‖W_V‖ read 1 / read 2</span><span>${n2(te.engram_Wv_norm)} / ${n2(te.engram2_Wv_norm)}</span><span>(t, Δ) table norm</span><span>${n2(te.table_norm)}</span><span>shared FFN ‖W₂‖</span><span>${n2(te.shared_w2_norm)}</span></div></div>
        ${ld("3", "Core dynamics · 3 passes (hop)")}
      </div><div class="grid g2" style="margin-top:16px">${ld("2a", "Core dynamics · 2a")}${ld("2b", "Core dynamics · 2b")}</div>`);
  }
  function secAudit() {
    const S = E.audit.summary, names = { encoder: "encoder", core_attn: "core attention", core_own_ffn: "core own FFN", core_reads_writes: "core reads / writes", core_shared: "shared FFN", core_table: "(t, Δ) table", ncp_head: "next-concept head", dec_plan: "decoder plan" };
    const rows = Object.entries(S).map(([k, v]) => `<tr><td><b>${names[k] || k}</b></td><td style="${heat(Math.max(0, v.cos_median))}">${n3(v.cos_median)}</td><td>${pct(v.cos_frac_negative)}</td><td>${n3(v.short_over_main_median)}</td></tr>`).join("");
    return sec("v3e-audit", "Does the shortcut training fight the main objective?", "Per-term gradient audit on fixed diagnostic batches (TinyStories, WikiText, probes, synthetic × shortcut 1 / 2a / 2b): cosine between the main gradient (NTP + NCP) and the shortcut gradient, per parameter group. A diagnostic, not a gate.",
      `<div class="card"><div class="scroll-x"><table class="rs-mx"><tr><th>parameter group</th><th>median cosine</th><th>batches with negative cosine</th><th>|g_short| / |g_main| (median)</th></tr>${rows}</table></div>
       <div class="rs-note">Positive everywhere: the shortcut terms pull in the same direction as the main loss. The decoder plan receives no shortcut gradient by design (0).</div></div>`);
  }
  function secGen() {
    const G = E.generation[gen];
    const fix = x => { try { return /[Â-ô][\u0080-¿]/.test(x) ? decodeURIComponent(escape(x)) : x; } catch (e) { return x; } };
    const cont = x => { const i = x.indexOf("<eos>"); return i < 0 ? esc(fix(x)) : esc(fix(x.slice(0, i))) + `<span class="eos">end of text</span>`; };
    const col = (k, t) => `<div class="rs-gcol"><h4>${t}</h4>${G[k].map(s => `<div class="rs-gen"><span class="p">${esc(fix(s.prompt))}</span><span class="c">${cont(s.continuation)}</span></div>`).join("")}</div>`;
    return sec("v3e-gen", "What it writes", "Continuations of held-out validation prompts (64-token prompt in grey, 128 generated tokens in ink).",
      `<div class="card"><div class="rs-tabs" data-set="v3gen"><button data-v="tinystories" class="${gen === "tinystories" ? "on" : ""}">TinyStories</button><button data-v="wikitext" class="${gen === "wikitext" ? "on" : ""}">WikiText-103</button></div>
       <div class="grid g2">${col("greedy", "Greedy")}${col("sample_T0.8", "Sampled · temperature 0.8")}</div></div>`);
  }

  document.addEventListener("click", e => {
    const tb = e.target.closest('.rs-tabs[data-set="v3gen"] [data-v]'); if (!tb) return;
    gen = tb.dataset.v; const y = window.scrollY; render(); window.scrollTo(0, y);
  });
  window.addEventListener("tabshown", e => { if (e.detail === "v3eval24") load(); });
  if ((document.querySelector("section.tab.on") || {}).id === "tab-v3eval24") load();
})();
