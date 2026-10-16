/* Results tab (D-044): the final evaluation (code/final_eval.py → runs/<run>/eval_final.json) rendered in full.
   Every number is read from the file; each panel shows its reference (intact, chance, baseline or gate). */
let FINAL = null, rsStage = "main", rsItemSplit = "test", rsGen = "tinystories";
const PROBE_KEYS = ["lr_recall1", "lr_carry3", "lr_multi4", "lr_hop", "lr_bind"];
const PROBE_NAME = { lr_recall1: "recall1", lr_carry3: "carry3", lr_multi4: "multi4", lr_hop: "hop", lr_bind: "bind" };
const INTV = [
  ["intact", "Intact", "the full model"],
  ["c_off", "Predicted concept off", "plan without the ĉ_pred term (only u)"],
  ["u_off", "Loop state off", "plan without the u term (only ĉ_pred)"],
  ["core_off", "Concept module off", "the learned null plan everywhere: the decoder has only its last 70 tokens"],
  ["plan_swap", "Plan swapped", "the plan of a different document"],
];
const RS = [1, 2, 3, 4, 6, 8];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const n3 = v => v == null || !isFinite(v) ? "—" : Number(v).toFixed(3), n2 = v => v == null || !isFinite(v) ? "—" : Number(v).toFixed(2);
const pct = v => v == null || !isFinite(v) ? "—" : (100 * v).toFixed(1) + "%";
const dchip = (d, lowerBetter = true, dig = 3) => d == null || !isFinite(d) ? "" :
  `<span class="rs-d ${Math.abs(d) < 5e-4 ? "z" : (d < 0) === lowerBetter ? "g" : "b"}">${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d).toFixed(dig)}</span>`;
const heat = v => `background: color-mix(in srgb, #2e9c8c ${Math.round(8 + 80 * Math.max(0, Math.min(1, v)))}%, transparent)`;

async function loadFinal() {
  try { FINAL = await (await fetch("api/final", { cache: "no-store" })).json(); } catch (e) { FINAL = null; }
  renderResults();
}
function curEval() { return FINAL && (FINAL[rsStage] || FINAL.main); }

function renderResults() {
  const root = document.getElementById("rs-root"); if (!root) return;
  if (!FINAL || !FINAL.main) { root.innerHTML = `<div class="card empty-note">The final evaluation of the main run is not available yet (runs/main/eval_final.json).</div>`; return; }
  const hasCont = !!FINAL.cont;
  if (rsStage === "cont" && !hasCont) rsStage = "main";
  const E = curEval(), B = rsStage === "cont" ? FINAL.main : null;
  document.getElementById("rs-stage").innerHTML = `
    <button data-st="main" class="${rsStage === "main" ? "on" : ""}">24 h model · step ${FINAL.main.step}</button>
    <button data-st="cont" class="${rsStage === "cont" ? "on" : ""}" ${hasCont ? "" : "disabled title='available when the continuation finishes (≈ Tue morning)'"}>48 h model${hasCont ? " · step " + (FINAL.cont.step + 4050) : " · pending"}</button>`;
  document.getElementById("rs-meta").textContent = `Evaluated ${E.time} · ${E.ckpt} · loop depth R = ${E.eval_R} · ${fmtDur(Object.values(E.seconds || {}).reduce((a, b) => a + b, 0))} of evaluation`;
  root.innerHTML = [kpis(E, B), findings(E), secLang(E, B), secRsweep(E), secTrunc(E), secProbes(E), secSynth(E), secLinear(E), secHealth(E), secGates(E), secGen(E), secP5()].join("");
  mountRsweepCharts(E); mountLoopCharts(E);
}
function fmtDur(s) { const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60); return h ? `${h} h ${m} min` : `${m} min`; }

/* ---------------------------------------------------------------- KPI tiles + findings */
function kpis(E, B) {
  const t = E.items.test.intact, o = E.items.ood.intact, sw = E.items.test.plan_swap, cb = E.health.codebook;
  const tile = (l, v, s, d) => `<div class="rs-k"><div class="l">${l}</div><div class="v">${v}${d || ""}</div><div class="s">${s}</div></div>`;
  const bt = B && B.items.test.intact;
  return `<div class="rs-kpis">
    ${tile("TinyStories test", n3(E.lang.tinystories.intact.nats), `nats · ${n3(E.lang.tinystories.intact.bpb)} bits/byte`, B && dchip(E.lang.tinystories.intact.nats - B.lang.tinystories.intact.nats))}
    ${tile("WikiText-103 test", n3(E.lang.wikitext.intact.nats), `nats · ${n3(E.lang.wikitext.intact.bpb)} bits/byte`, B && dchip(E.lang.wikitext.intact.nats - B.lang.wikitext.intact.nats))}
    ${tile("recall1 · test", pct(t.lr_recall1.exact), `plan swapped: ${pct(sw.lr_recall1.exact)} (chance ≈ 0.8%)`, bt && dchip(t.lr_recall1.exact - bt.lr_recall1.exact, false))}
    ${tile("carry3 · test", pct(t.lr_carry3.exact), `per-token ${t.lr_carry3.by_offset.map(v => (100 * v).toFixed(0) + "%").join(" / ")}`, bt && dchip(t.lr_carry3.exact - bt.lr_carry3.exact, false))}
    ${tile("Synthetic tasks", pct(t._macro_synthetic), `macro over ${Object.keys(t).filter(k => !k.startsWith("lr_") && !k.startsWith("_")).length} families · OOD ${pct(o._macro_synthetic)}`, bt && dchip(t._macro_synthetic - bt._macro_synthetic, false))}
    ${tile("Concept codebook", pct(cb.codes_used_frac_min), `codes used · VE ${n3(cb.var_explained)} · NCP gap ${n2(cb.code_entropy - cb.ncp_ce)} nats`)}
  </div>`;
}
function findings(E) {
  const L = E.lang, T = E.items.test, O = E.items.ood, P = E.probes.G_E1_concept_to_token;
  const maxSwap = Math.max(...PROBE_KEYS.filter(k => k !== "lr_bind").map(k => T.plan_swap[k].exact));
  const coreTS = L.tinystories.core_off.nats - L.tinystories.intact.nats, coreWK = L.wikitext.core_off.nats - L.wikitext.intact.nats;
  const rTS = L.tinystories.R4.nats - L.tinystories.R1.nats, rWK = L.wikitext.R4.nats - L.wikitext.R1.nats;   // R = 4 minus R = 1 (negative = the loop helps)
  const ge1 = Object.values(P).map(v => v.acc);
  const F = [
    ["g", "The concept stream carries long-range information", `With another document's plan every long-range probe falls to chance (at most ${pct(maxSwap)} on recall1 / carry3 / multi4 / hop), from ${pct(T.intact.lr_recall1.exact)} recall1 intact.`],
    ["g", "The model relies on its concepts for ordinary text", `Switching the concept module off costs +${n3(coreTS)} nats on TinyStories and +${n3(coreWK)} on WikiText.`],
    ["a", "Recurrence adds almost nothing yet", `Four loop passes vs one: ${dchip(rTS)} nats on TinyStories, ${dchip(rWK)} on WikiText; hop R4 ${pct(T.intact.lr_hop.exact)} vs R1 ${pct(T.R1.lr_hop.exact)} (3 items of 512, within noise).`],
    ["a", "Longer contexts are harder", `Out-of-distribution items (≈ 2× longer): recall1 ${pct(T.intact.lr_recall1.exact)} → ${pct(O.intact.lr_recall1.exact)}, carry3 ${pct(T.intact.lr_carry3.exact)} → ${pct(O.intact.lr_carry3.exact)}; synthetic ${pct(T.intact._macro_synthetic)} → ${pct(O.intact._macro_synthetic)}.`],
    ["r", "Concepts do not store their 4 tokens exactly (G-E1)", `A linear probe recovers each token ${pct(Math.min(...ge1))}–${pct(Math.max(...ge1))} of the time (majority baseline ≈ 4–5%); the pre-registered gate is 99%.`],
  ];
  return `<div class="rs-find">${F.map(([l, t, x]) => `<div class="rs-f ${l}"><i></i><div><b>${t}</b><p>${x}</p></div></div>`).join("")}</div>`;
}

/* ---------------------------------------------------------------- language */
function sec(id, title, sub, body) { return `<section class="rs-sec" id="${id}"><div class="sh"><div><h3>${title}</h3><p>${sub}</p></div></div>${body}</section>`; }
function secLang(E, B) {
  const col = (name, label) => {
    const d = E.lang[name], base = d.intact.nats, vals = INTV.map(([k]) => d[k].nats), lo = Math.min(...vals) - 0.1, hi = Math.max(...vals);
    return `<div class="card rs-bars"><h4>${label} · test</h4>${INTV.map(([k, t, x]) => `<div class="rs-br" title="${esc(x)}">
        <div class="rs-bl"><b>${t}</b><span>${x}</span></div>
        <div class="rs-bt"><div class="${k === "intact" ? "in" : ""}" style="width:${(100 * (d[k].nats - lo) / (hi - lo)).toFixed(1)}%"></div></div>
        <div class="rs-bv">${n3(d[k].nats)}<small>${k === "intact" ? "nats" : ""}</small>${k === "intact" ? (B ? dchip(d.intact.nats - B.lang[name].intact.nats) : "") : dchip(d[k].nats - base)}</div></div>`).join("")}
      <div class="rs-note">bits/byte intact ${n3(d.intact.bpb)} · ${d.intact.tokens.toLocaleString()} tokens · bars start at ${n2(lo)} nats</div></div>`;
  };
  return sec("rs-lang", "Language: what each part of the plan is worth", "Next-token loss on held-out test text with parts of the concept stream removed (lower is better; Δ vs intact).",
    `<div class="grid g2">${col("tinystories", "TinyStories")}${col("wikitext", "WikiText-103")}</div>`);
}
function secRsweep(E) {
  return sec("rs-r", "Loop depth R: does thinking longer help?", "The shared concept layer applied R times at evaluation (trained with R ~ 1–4). Axes are zoomed to the data range; the differences are tiny.",
    `<div class="grid g2"><div id="rs-rl"></div><div id="rs-rp"></div></div>`);
}
function mountRsweepCharts(E) {
  const a = document.getElementById("rs-rl"), b = document.getElementById("rs-rp"); if (!a || !b) return;
  const L = E.lang, T = E.items.test, O = E.items.ood;
  const c1 = new LineChart(a, { title: "Language loss vs R", xLabel: "R", latestLabel: "R = 8", measures: "test nats at each loop depth (R = 4 is the default).", series: [{ name: "TinyStories", color: "#8f7cc0", width: 2.2, dots: true }] });
  c1.set([RS.map(r => [r, L.tinystories["R" + r].nats])]);
  const c2 = new LineChart(a, { title: "WikiText loss vs R", xLabel: "R", latestLabel: "R = 8", series: [{ name: "WikiText-103", color: "#e07b39", width: 2.2, dots: true }] });
  c2.set([RS.map(r => [r, L.wikitext["R" + r].nats])]);
  const pr = k => RS.map(r => [r, (r === 4 ? T.intact : T["R" + r])[k].exact]);
  const c3 = new LineChart(b, { title: "Long-range probes vs R (test)", xLabel: "R", latestLabel: "recall1 @ R 8", yMin: 0, fmt: v => (100 * v).toFixed(0) + "%",
    series: [{ name: "recall1", color: "#2e9c8c", width: 2.2, dots: true }, { name: "carry3", color: "#3b78c2", dots: true }, { name: "hop", color: "#e07b39", dots: true }, { name: "multi4", color: "#8f7cc0", dots: true }] });
  c3.set([pr("lr_recall1"), pr("lr_carry3"), pr("lr_hop"), pr("lr_multi4")]);
  const po = k => RS.map(r => [r, (r === 4 ? O.intact : O["R" + r])[k].exact]);
  const c4 = new LineChart(b, { title: "Long-range probes vs R (longer, OOD)", xLabel: "R", latestLabel: "recall1 @ R 8", yMin: 0, fmt: v => (100 * v).toFixed(0) + "%",
    series: [{ name: "recall1", color: "#2e9c8c", width: 2.2, dots: true }, { name: "carry3", color: "#3b78c2", dots: true }] });
  c4.set([po("lr_recall1"), po("lr_carry3")]);
}
function secTrunc(E) {
  const rs = [72, 128, 256, 512, 1024];
  const card = (name, label) => {
    const d = E.trunc[name], vals = rs.map(r => d["r" + r].nats).concat(d.swap_r1024.nats), lo = Math.min(...vals) - 0.05, hi = Math.max(...vals), s = d.summary;
    return `<div class="card rs-bars"><h4>${label}</h4>
      ${rs.map(r => `<div class="rs-br sm"><div class="rs-bl"><b>last ${r} tokens</b></div><div class="rs-bt"><div class="${r === 1024 ? "in" : ""}" style="width:${(100 * (d["r" + r].nats - lo) / (hi - lo)).toFixed(1)}%"></div></div><div class="rs-bv">${n3(d["r" + r].nats)}${r > 72 ? dchip(d["r" + r].nats - d.r72.nats) : ""}</div></div>`).join("")}
      <div class="rs-br sm"><div class="rs-bl"><b>1024 + plan swapped</b></div><div class="rs-bt"><div class="bad" style="width:${(100 * (d.swap_r1024.nats - lo) / (hi - lo)).toFixed(1)}%"></div></div><div class="rs-bv">${n3(d.swap_r1024.nats)}${dchip(d.swap_r1024.nats - d.r1024.nats)}</div></div>
      <div class="rs-chips"><span class="chip">gain beyond 70 tokens <b>${n3(s.gain_beyond_70)}</b> nats</span><span class="chip">cost of a wrong plan <b>${n3(s.swap_cost)}</b> nats</span><span class="chip">${s.rows} rows · ${s.rows_ctx_ge_512} with ≥ 512 context</span></div></div>`;
  };
  return sec("rs-trunc", "How far back does the model use?", "Loss on the same target tokens with the visible history truncated (the decoder alone reaches 70 tokens; anything further must come through the concepts). Only 1 of 256 TinyStories rows has ≥ 512 tokens of context (stories are short), so WikiText (234 of 256) is the meaningful long-range test here.",
    `<div class="grid g2">${card("tinystories", "TinyStories")}${card("wikitext", "WikiText-103")}</div>`);
}

/* ---------------------------------------------------------------- probes and synthetic tasks */
function secProbes(E) {
  const S = E.items[rsItemSplit], cols = INTV.map(([k, t]) => [k, t]).concat([1, 2, 3, 6, 8].map(r => ["R" + r, "R = " + r]));
  const table = `<div class="scroll-x"><table class="rs-mx"><tr><th>probe</th>${cols.map(([k, t]) => `<th class="${k === "intact" ? "in" : ""}">${t}</th>`).join("")}</tr>
    ${PROBE_KEYS.map(p => `<tr><td><b>${PROBE_NAME[p]}</b></td>${cols.map(([k]) => { const v = S[k]?.[p]?.exact; return `<td style="${heat(v)}" class="${k === "intact" ? "in" : ""}">${pct(v)}</td>`; }).join("")}</tr>`).join("")}</table></div>`;
  const offs = ["lr_carry3", "lr_multi4"].map(p => `<div class="rs-off"><b>${PROBE_NAME[p]}</b>${S.intact[p].by_offset.map((v, i) => `<div class="rs-bt t" title="answer token ${i + 1}"><div style="width:${100 * v}%"></div><span>tok ${i + 1} · ${pct(v)}</span></div>`).join("")}<span class="rs-note">exact (all 3 right): ${pct(S.intact[p].exact)}</span></div>`).join("");
  return sec("rs-probes", "Long-range probes", "Answers hidden 128–420 tokens back (test) or in ≈ 2× longer documents (OOD). 512 items per probe; chance ≈ 0.8% for recall1, ≈ 0 for the others.",
    `<div class="card"><div class="rs-tabs" data-set="split"><button data-v="test" class="${rsItemSplit === "test" ? "on" : ""}">Test</button><button data-v="ood" class="${rsItemSplit === "ood" ? "on" : ""}">Out-of-distribution (longer)</button></div>${table}
     <div class="rs-offs">${offs}</div></div>`);
}
function secSynth(E) {
  const T = E.items.test.intact, O = E.items.ood.intact;
  const fams = Object.keys(O).filter(k => !k.startsWith("lr_") && !k.startsWith("_")).sort((a, b) => (T[b]?.exact ?? -1) - (T[a]?.exact ?? -1));
  return sec("rs-synth", "Synthetic reasoning tasks", "Exact-match accuracy per family (200 items each), test vs longer OOD items, sorted by test accuracy.",
    `<div class="card"><div class="rs-syn">${fams.map(f => `<div class="rs-sr"><span class="nm">${f.replace(/_/g, " ")}</span>
      <div class="rs-bt"><div style="width:${100 * (T[f]?.exact ?? 0)}%"></div></div><span class="v">${T[f] ? pct(T[f].exact) : "—"}</span>
      <div class="rs-bt o"><div style="width:${100 * (O[f]?.exact ?? 0)}%"></div></div><span class="v">${pct(O[f]?.exact)}</span></div>`).join("")}</div>
      <div class="rs-legend"><span><i class="t"></i>test</span><span><i class="o"></i>out-of-distribution</span><span>macro average: test <b>${pct(T._macro_synthetic)}</b> · OOD <b>${pct(O._macro_synthetic)}</b></span><span class="muted">variable binding newnames exists only in OOD</span></div></div>`);
}

/* ---------------------------------------------------------------- linear probes and health */
function secLinear(E) {
  const P = E.probes, g = P.G_E1_concept_to_token;
  const row = (lbl, v, gate) => `<div class="rs-lp"><span class="nm">${lbl}</span><div class="rs-bt g"><div style="width:${100 * v.acc}%"></div><i class="base" style="left:${100 * v.majority_baseline}%" title="majority baseline ${pct(v.majority_baseline)}"></i>${gate ? `<i class="gate" style="left:${100 * gate}%" title="gate ${pct(gate)}"></i>` : ""}</div><span class="v">${pct(v.acc)}</span><span class="b">base ${pct(v.majority_baseline)}</span></div>`;
  const fams = PROBE_KEYS.map(p => [p, P["u_query_to_answer_" + p]]).filter(([, v]) => v);
  return sec("rs-linear", "What the representations contain (linear probes)", "A linear read-out trained on 80% and scored on 20%. Grey tick: majority baseline; red tick: pre-registered gate.",
    `<div class="grid g2">
      <div class="card"><h4>G-E1 · concept → each of its 4 tokens</h4>${Object.entries(g).map(([k, v]) => row(`token ${+k.slice(3) + 1}`, v, 0.99)).join("")}
        <div class="rs-note">${g.pos0.n_train.toLocaleString()} train / ${g.pos0.n_test.toLocaleString()} test concepts, ≈ ${g.pos0.classes.toLocaleString()} token classes. Gate G-E1 (≥ 99%): not met.</div></div>
      <div class="card"><h4>Loop state u at the question → answer tokens</h4>${fams.map(([p, v]) => Object.entries(v).map(([k, x]) => row(`${PROBE_NAME[p]} · answer token ${+k.slice(3) + 1}`, x)).join("")).join("")}
        <h4 style="margin-top:12px">hop · bridge entity after pass 1 / 2 / 4</h4>${Object.entries(P.lr_hop_bridge_by_pass).map(([k, v]) => row(k.replace("pass", "after pass "), v)).join("")}
        <div class="rs-note">Flat across passes: the loop does not yet compute the intermediate hop.</div></div></div>`);
}
function secHealth(E) {
  const cb = E.health.codebook, te = E.health.telemetry, ld = E.health.loop_dynamics_lr_hop, mix = te.crc_mix_e0_L_G;
  return sec("rs-health", "Concept stream health", "Codebook use, what goes into a concept, and how the loop state moves pass by pass (hop items, R = 8).",
    `<div class="grid g3">
      <div class="card"><h4>Codebook (${cb.n_concepts.toLocaleString()} test concepts)</h4><div class="kvt">
        <span>codes used (min / mean)</span><span><b>${pct(cb.codes_used_frac_min)}</b> / ${pct(cb.codes_used_frac_mean)}</span>
        <span>perplexity (of 128)</span><span><b>${n2(cb.perplexity_mean)}</b></span>
        <span>variance explained</span><span><b>${n3(cb.var_explained)}</b> (gate ≥ 0.6)</span>
        <span>next-concept loss vs H</span><span><b>${n3(cb.ncp_ce)}</b> vs ${n3(cb.code_entropy)} → gap ${n2(cb.code_entropy - cb.ncp_ce)}</span></div>
        <div class="rs-note">Test-set values; the live page's hourly numbers use validation rows.</div></div>
      <div class="card"><h4>What goes into a concept</h4><div class="rs-stack">${[["embedding", mix[0], "#9aa3af"], ["local L", mix[1], "#3b78c2"], ["global G", mix[2], "#e07b39"]].map(([n, v, c]) => `<div style="width:${100 * v}%;background:${c}" title="${n} ${pct(v)}">${v > 0.12 ? n + " " + pct(v) : ""}</div>`).join("")}</div>
        <div class="kvt" style="margin-top:12px"><span>embedding / local / global</span><span>${mix.map(v => pct(v)).join(" / ")}</span><span>Engram gate α (mean)</span><span><b>${n3(te.engram_alpha_mean)}</b></span><span>‖W_V‖ Engram</span><span>${n2(te.engram_Wv_norm)}</span></div></div>
      <div class="card"><h4>Loop dynamics (hop, 8 passes)</h4><div class="kvt"><span>step size pass 1 → 8</span><span>${n2(ld.step[0])} → ${n2(ld.step.at(-1))}</span><span>turn between passes</span><span>${n2(ld.consecutive_step_angle_deg[0])}° → ${n2(ld.consecutive_step_angle_deg.at(-1))}°</span><span>gradient ratio pass 1 / 4</span><span>${n2(ld.grad_ratio_pass1_pass4)}</span></div>
        <div class="rs-note">Each pass adds a nearly identical step in almost the same direction: the loop moves the state but barely changes its direction after pass 1.</div></div>
    </div><div class="grid g2" style="margin-top:16px"><div id="rs-ld1"></div><div id="rs-ld2"></div></div>`);
}
function mountLoopCharts(E) {
  const ld = E.health.loop_dynamics_lr_hop, a = document.getElementById("rs-ld1"), b = document.getElementById("rs-ld2"); if (!a || !b) return;
  const x = ld.step.map((_, i) => i + 1);
  new LineChart(a, { title: "State norm and step size per pass", xLabel: "pass", latestLabel: "norm @ pass 8", fmt: v => v.toFixed(1), series: [{ name: "state norm", color: "#2e9c8c", width: 2.2, dots: true }, { name: "step size", color: "#3b78c2", dots: true }] })
    .set([x.map((p, i) => [p, ld.state_norm[i]]), x.map((p, i) => [p, ld.step[i]])]);
  new LineChart(b, { title: "Direction change per pass (degrees)", xLabel: "pass", latestLabel: "@ pass 8", yMin: 0, fmt: v => v.toFixed(2), series: [{ name: "angle between consecutive steps", color: "#e07b39", width: 2.2, dots: true }, { name: "turn", color: "#8f7cc0", dots: true }] })
    .set([x.map((p, i) => [p, ld.consecutive_step_angle_deg[i]]), x.map((p, i) => [p, ld.turn_deg[i]])]);
}

/* ---------------------------------------------------------------- gates, generations, P5 */
function secGates(E) {
  const T = E.items.test, cb = E.health.codebook, g = E.probes.G_E1_concept_to_token, tr = E.trunc.tinystories.summary;
  const p5 = FINAL[rsStage + "_p5"] || "", p5ok = (p5.match(/PASS/g) || []).length, p5n = (p5.match(/^(PASS|FAIL)/gm) || []).length;
  const maxSwap = Math.max(...PROBE_KEYS.filter(k => k !== "lr_bind").map(k => T.plan_swap[k].exact));
  const G = [
    ["G-E0 / G-D0", "exact tests pass (P5 on trained weights)", `${p5ok} of ${p5n} P5 checks PASS on ${E.ckpt}`, p5n && p5ok === p5n ? "pass" : "fail"],
    ["G-E1", "linear probe recovers each concept's 4 tokens ≥ 99%", `${pct(Math.min(...Object.values(g).map(v => v.acc)))}–${pct(Math.max(...Object.values(g).map(v => v.acc)))}`, "fail"],
    ["G-E2", "codebook variance explained ≥ 0.6", `VE ${n3(cb.var_explained)}`, cb.var_explained >= 0.6 ? "pass" : "fail"],
    ["G-C1", "≥ 95% of codes used in every segment", `${pct(cb.codes_used_frac_min)} (least-used segment)`, cb.codes_used_frac_min >= 0.95 ? "pass" : "fail"],
    ["G-C2", "next-concept loss below the code-frequency entropy", `${n3(cb.ncp_ce)} < ${n3(cb.code_entropy)}`, cb.ncp_ce < cb.code_entropy ? "pass" : "fail"],
    ["G-E3 / G-D2", "plan swap drops every long-range probe to chance; swap cost ≈ gain beyond 70", `probes under swap ≤ ${pct(maxSwap)} ✓ · swap cost ${n3(tr.swap_cost)} vs gain ${n3(tr.gain_beyond_70)} nats (a wrong plan hurts far more than the long context helps)`, "part"],
    (() => { const a = T.intact.lr_hop, b = T.R1.lr_hop, se = Math.sqrt(a.exact * (1 - a.exact) / a.n + b.exact * (1 - b.exact) / b.n), z = se ? (a.exact - b.exact) / se : 0;
      return ["G-C3", "R = 4 beats R = 1 on hop", `hop R4 ${pct(a.exact)} vs R1 ${pct(b.exact)}: ${Math.round((a.exact - b.exact) * a.n)} items of ${a.n}, z = ${z.toFixed(2)}${Math.abs(z) < 2 ? " (not distinguishable from noise)" : ""}`,
        Math.abs(z) < 2 ? "noise" : z > 0 ? "pass" : "fail"]; })(),
    ["G-D3", "multi4 exact ≥ 0.5", `multi4 ${pct(T.intact.lr_multi4.exact)}`, T.intact.lr_multi4.exact >= 0.5 ? "pass" : "fail"],
  ];
  const lab = { pass: "PASS", fail: "NOT MET", part: "PARTLY", noise: "WITHIN NOISE" };
  return sec("rs-gates", "Pre-registered gates: final verdict", "Criteria fixed in PROMPT.md §6 before training, judged on the final evaluation.",
    `<div class="gates">${G.map(([id, c, e, s]) => `<div class="gate ${s === "pass" ? "pass" : s === "part" || s === "noise" ? "fail" : "rs-no"}"><div class="gate-top"><span class="gate-id">${id}</span><span class="badge ${s === "pass" ? "pass" : s === "part" || s === "noise" ? "fail" : "no"}">${lab[s]}</span></div><div class="gate-c">${c}</div><div class="gate-e">${e}</div></div>`).join("")}</div>`);
}
function secGen(E) {
  const G = E.generation[rsGen];
  const fix = x => { try { return /[\u00c2-\u00f4][\u0080-\u00bf]/.test(x) ? decodeURIComponent(escape(x)) : x; } catch (e) { return x; } };   // UTF-8 read as Latin-1
  const cont = x => { const i = x.indexOf("<eos>"); return i < 0 ? esc(fix(x)) : esc(fix(x.slice(0, i))) + `<span class="eos">end of text</span>`; };
  const col = (k, t) => `<div class="rs-gcol"><h4>${t}</h4>${G[k].map(s => `<div class="rs-gen"><span class="p">${esc(fix(s.prompt))}</span><span class="c">${cont(s.continuation)}</span></div>`).join("")}</div>`;
  return sec("rs-gen", "What it writes", "Continuations of held-out test prompts (64-token prompt in grey, 128 generated tokens in ink).",
    `<div class="card"><div class="rs-tabs" data-set="gen"><button data-v="tinystories" class="${rsGen === "tinystories" ? "on" : ""}">TinyStories</button><button data-v="wikitext" class="${rsGen === "wikitext" ? "on" : ""}">WikiText-103</button></div>
     <div class="grid g2">${col("greedy", "Greedy")}${col("sample_T0.8", "Sampled · temperature 0.8")}</div></div>`);
}
function secP5() {
  const lines = (FINAL[rsStage + "_p5"] || "").split("\n").filter(Boolean);
  const sc = FINAL.screens_p5 || {};
  return sec("rs-p5", "Exact structural tests on the trained weights (P5)", "Causality, the decoder's 70-token reach with the concept module off, and document isolation, re-checked bit-exactly after training.",
    `<div class="card rs-p5">${lines.map(l => `<div class="${l.startsWith("PASS") ? "ok" : "bad"}">${esc(l)}</div>`).join("") || '<div class="muted">not available</div>'}
     <div class="rs-note">Screens: ${Object.entries(sc).map(([k, v]) => `${k} ${v ? (v.match(/PASS/g) || []).length + " PASS" : "—"}`).join(" · ")}</div></div>`);
}

/* ---------------------------------------------------------------- wiring */
document.addEventListener("click", e => {
  const st = e.target.closest("#rs-stage [data-st]"); if (st && !st.disabled) { rsStage = st.dataset.st; renderResults(); return; }
  const tb = e.target.closest(".rs-tabs [data-v]"); if (!tb) return;
  const set = tb.parentElement.dataset.set;
  if (set === "split") rsItemSplit = tb.dataset.v; else if (set === "gen") rsGen = tb.dataset.v;
  const y = window.scrollY; renderResults(); window.scrollTo(0, y);
});
window.addEventListener("tabshown", e => { if (e.detail === "results") loadFinal(); });
