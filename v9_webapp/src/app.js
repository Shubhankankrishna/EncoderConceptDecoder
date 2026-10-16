/* V9 Atlas · 2D figures and page logic. Every number is taken from draft_v9.md; toy simulations are labelled in the page. */
(function () {
"use strict";
const NS = "http://www.w3.org/2000/svg";
const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs || {}) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function txt(p, x, y, s, o = {}) {
  const t = el("text", { x, y, "text-anchor": o.anchor || "middle", "font-size": o.size || 12, "font-weight": o.weight || 500, class: o.cls || null, style: o.color ? `fill:${o.color}` : null, "dominant-baseline": o.base || "middle" }, p);
  t.textContent = s; return t;
}
function svgRoot(host, w, h) { host.innerHTML = ""; const s = el("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" }, host); el("defs", {}, s); return s; }
function marker(svg, id, color) {
  const d = svg.querySelector("defs");
  const m = el("marker", { id, viewBox: "0 0 10 10", refX: 8.5, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, d);
  el("path", { d: "M0,0 L10,5 L0,10 z", style: `fill:${color}` }, m);
}
function wire(p, pts, cls, mk) {
  const d = pts.map((q, i) => (i ? "L" : "M") + q[0] + "," + q[1]).join(" ");
  return el("path", { d, class: "wire " + cls, "marker-end": mk ? `url(#${mk})` : null }, p);
}
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

/* ---------------------------------------------------------------- tooltip */
const tip = $("#tip");
document.addEventListener("mousemove", e => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (!t) { tip.classList.remove("on"); return; }
  tip.innerHTML = t.getAttribute("data-tip");
  const x = Math.min(innerWidth - 320, e.clientX + 14), y = Math.min(innerHeight - 80, e.clientY + 14);
  tip.style.left = x + "px"; tip.style.top = y + "px"; tip.classList.add("on");
});

/* ---------------------------------------------------------------- theme + tabs */
$("#theme-btn").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("v9-theme", root.dataset.theme); } catch (e) {}
  window.dispatchEvent(new Event("v9theme"));
});
try { const t = localStorage.getItem("v9-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => window.dispatchEvent(new Event("v9theme")));

const TABS = ["overview", "live", "arch", "model3d", "stages", "search", "continual", "evidence"];
function showTab(id) {
  if (!TABS.includes(id)) id = "overview";
  $$("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === id));
  $$("section.tab").forEach(s => s.classList.toggle("on", s.id === "tab-" + id));
  if (location.hash.slice(1) !== id) history.replaceState(null, "", "#" + id);
  try { localStorage.setItem("v9-tab", id); } catch (e) {}
  window.dispatchEvent(new CustomEvent("v9tab", { detail: id }));
}
$$("#tabs button").forEach(b => b.addEventListener("click", () => { showTab(b.dataset.tab); scrollTo({ top: 0 }); }));
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

/* ================================================================ STAGE DATA (§5–§10) */
const STAGES = [
  { id: "S1", name: "Foundation", color: "--enc", short: "V8-D ECD trained from scratch with NTP + delayed MSE NCP on text and symbolic data.",
    gates: ["competence", "paired NCP vs α0"],
    obj: `L_S1  = L_token + α(u)·L_NCP + L_VQ
L_NCP = balanced mean ‖ĉ_j − sg(C_j+1)‖²   (scored target chunks)
α(u)  = 0 for u < 6000, linear to 0.1 by 8000, then 0.1
24,000 updates · 64 rows × 192 tokens · 264.6M tokens`,
    what: "Nothing architectural changes relative to V8-D. V9 adds the training seeds that V8 lacked, paired item-level tests, one PrefixLM arm, and two arms that test why NCP hurt in V6.",
    arms: [["A_delayed (anchor)", "811 + 2 new", "S2 teacher; NCP evidence", "A_delayed", 1], ["A_alpha0", "811 + 2 new", "NCP control", "A_alpha0", 1], ["A_flat", "811 + 2 new", "architecture control", "A_flat", 1],
      ["A_delayed_prefixlm", "2 new", "PrefixLM interaction", "A_delayed_prefixlm", 0], ["A_const (α = 1 from step 0)", "1 new", "reproduce V6 reversal; P-blur-2", null, 0], ["A_const_hncp (K = 4, α = 1)", "1 new", "the only test of P-blur-1", "A_const_hncp", 0], ["A_delayed ×4 budget", "1 new", "crossover at 96k updates", null, 0]],
    gateText: [["Competence", "≥ 80% standard dev and ≥ 95% lookup on the anchor's seeds. A seed that fails is never an S2 teacher."], ["Reporting", "NCP vs α0 and hierarchy vs flat reported with paired McNemar and bootstrap intervals, including results that disfavor NCP."]],
    cost: "~12 h mandatory · ~34 h full (Q1)", sec: "§5" },
  { id: "S2", name: "H-NCP concept module", color: "--con", short: "Separate dense or HRM-recurrent core learns K committed next concepts and a selector, against a frozen S1 teacher.",
    gates: ["G2.1 authority", "G2.2 calibration", "G2.3 closed loop"],
    obj: `k*_j   = argmin_k d_j,k ,  d_j,k = mean (ĉ_j^(k) − sg(C_j+1))²
L_WTA  = Σ w_j [ (1−ε) d_j,k* + ε/(K−1) Σ_{k≠k*} d_j,k ]     ε = .05
L_sel  = KL( softmin(−d/τ_q) ‖ π_j )                         τ_q = .1·mean d
plan   = best of m = 2 codes sampled from π                 → frozen decoder
L_S2   = L_token(plan) + 0.1·L_WTA + δ(u)·L_dist + 0.1·L_sel
K = 1  ⇒  L_S2 ≡ V8 REFIT objective (contract test, 1e-6)`,
    what: "Your “separate recurrent concept module with Z injection” stage. Z becomes a discrete code k with a learned 16-d embedding. Explorative modelling becomes winner-take-all over K codes with best-of-m plans. The core is also trained on its own predictions (p_cl = .25, N ~ U{1..4}), because decoder-free search needs that.",
    arms: [["V8 replay (K = 1, closed loop off)", "811", "implementation check", "S2_V8repro", 1], ["S2-K1-dense", "2", "mandatory control", "S2_K1_dense", 1], ["S2-K8-dense", "2", "main claim", "S2_K8_dense", 1], ["S2-K8-rec", "2", "HRM-recurrent core", "S2_K8_rec", 1], ["S2-K1-rec", "2", "recurrent control", "S2_K1_rec", 1],
      ["S2-K8-m1 / -mK", "1 each", "prior vs hindsight plans", null, 0], ["S2-K8-open", "1", "no closed loop (isolates R4)", null, 0], ["S2-K4 / S2-K16", "1 each", "K sweep", null, 0]],
    gateText: [["G2.1 Authority", "(a) hindsight-code graph NLL ≤ K = 1 − .05 nats. (b) Swapping the code at the last prompt chunk changes the first colour in ≥ 30% of standard items."],
      ["G2.2 Calibration", "(a) Single-sample prior NLL within +.05 of K = 1 on graph and text. (b) Selector top-2 covers k* ≥ 90%. (c) ECE ≤ .05 and ≥ K/2 live codes."],
      ["G2.3 Closed loop", "(a) N-step drift (N = 1..5) ≤ K = 1's. (b) Selector-argmax hops decoded against 73 candidates are legal ≥ 95% at N = 1. (c) Greedy success ≥ K = 1 − 1 point."]],
    cost: "~12.2 h mandatory · ~22 h full (Q3)", sec: "§6" },
  { id: "S3", name: "Refit + continued pretraining", color: "--dec", short: "Let encoder and decoder adapt to committed plans, gated on keeping code authority (ETA: co-training can erase it).",
    gates: ["G3 authority kept"],
    obj: `λ-curve: u = (1−λ)·u_T + λ·u_S ,  λ ∈ {0, .3, .6, 1}   (λ = 0 bit-exact)
S3a: encoder + decoder train 6,000 updates; core, codes, selector frozen
S3b: everything trains 6,000 updates; WTA, selector, distillation stay on`,
    what: "Your “refit and keep pretraining until stable” stage. It runs two ways side by side. The prediction is that S3a keeps authority and S3b loses it, as ETA found when it co-trained a base with its abstraction module.",
    arms: [["S3a-dec (decoder only)", "2", "adapt the decoder to committed plans", "S3a_dec", 1], ["S3b joint", "2", "your sequence as stated", "S3b", 1], ["S3a-enc (encoder + decoder)", "2", "optional", "S3a_enc", 0]],
    gateText: [["G3", "Swap rate ≥ 30% and ≥ 0.8 × S2's; G2.2(a) holds; sealed standard/hard10 no worse than S2 by > 1 point; text NLL no worse by > .01. Otherwise S2's frozen model goes forward."]],
    cost: "~3 h mandatory · ~4.5 h full (Q4)", sec: "§7" },
  { id: "S4", name: "Post-training: expert iteration", color: "--sel", short: "Search finds verified code sequences; the selector and a value head are distilled from them. GradCuit becomes a refine kernel.",
    gates: ["beats token RFT / GRPO"],
    obj: `for i in 1..4:  run concept MCTS on 4,096 fresh problems (standard + hard10)
  L_π = Σ_s CE( N(s,·)/ΣN , π_θ(·|s) )          visit-count targets
  L_V = Σ_s ( V_ψ(s) − o(s) )²                  o = verifier outcome of the played trajectory
  L_τ = Σ_s BCE( τ_ψ(s), next chunk is DONE )   terminal head
frozen: encoder, decoder, Q, W_A, core, hypothesis block
iteration 0: no heads yet → R = 1, every node decoded and verified`,
    what: "SFT is folded into pretraining, because the BFS demonstrations are already SFT. “RL with Z” has three readings: test-time GradCuit (now a refine kernel inside search), policy gradient on the selector (secondary arm), and expert iteration (primary).",
    arms: [["Expert iteration over codes", "2", "primary · 1,290 trainable params", "S4_EI", 1], ["Token RFT (decoder LoRA r16)", "2", "baseline · 24,576 params", "S4_RFT", 1], ["Token GRPO (same adapter)", "2", "standard method", "S4_GRPO", 1], ["REINFORCE on selector", "2", "reading (b)", "S4_REINFORCE", 1]],
    gateText: [["Primary metric", "Sealed hard10 and hard14 success after post-training, greedy and with search, at equal verifier calls and wall-clock. Trainable parameters are listed per arm."], ["GradCuit kernel", "e = e_k + δ with RMS(δ) ≤ 0.5·RMS(e_k), 3 steps. Always paired with a random-direction kernel of the same norm, and never reported as a standalone method."]],
    cost: "~20 h mandatory · ~28 h full (Q5)", sec: "§8" },
  { id: "S5", name: "Decoder-free concept MCTS", color: "--code", short: "PUCT over codes with the decoder removed between re-grounding points. A curve over R ∈ {1, 2, 4, ∞}.",
    gates: ["beats token tree search"],
    obj: `PUCT:  a* = argmax_a Q(s,a) + 1.5·π(a|s)·√N(s) / (1 + N(s,a))
child:  c̃_j+1 = ĉ_j^(k)  → core step → u^(0), π, V_ψ, τ_ψ     (≈5 block-positions)
every R committed chunks: decode 4 tokens/chunk, re-encode, replace rolled concepts
outer:  AB-MCTS-A over complete answers — GEN (new search) | CONT (refine kernel)`,
    what: "Your inference-time idea. Prompt encoded once. Each child is one code choice and one incremental core step, with no encoder or decoder pass. The chosen path is decoded at the end, or every R steps.",
    arms: [["Concept MCTS: RH(1), RH(H*), G(1)", "—", "your proposal's family; IMAG (R = ∞) is exploratory", "S5_RH1", 1], ["Token tree search, same hops", "—", "primary competitor", "S5_TT", 1], ["Verifier best-of-N", "—", "strongest simple baseline", "S5_BoN", 1], ["AB-MCTS-A answers", "—", "answer-level family, decides H14", "S5_AB", 1]],
    gateText: [["Value head (R5)", "AUROC ≥ .80 on complete candidates, then ≥ .70 on intermediate nodes. Otherwise leaves are decoded and verified."], ["Kill rule", "If G2.3(b) legality at N = 1 is < 80%, S5 runs at R = 1 only."]],
    cost: "3.1 h if search batches, ~38 h if not (Q6)", sec: "§9" },
  { id: "S6", name: "Continual learning (LSL)", color: "--loss", short: "Independent of S2. Measures separability first, then compares LSL against always-open, VQ-only and full fine-tuning.",
    gates: ["G6.0 separability", "LSL > always-open"],
    obj: `g(x) = 1[ Φpos(x) > Φneg(x) ]     diagonal GMMs on JL-projected matrix inputs
s_t  = .3·g_t + .7·s_t−1 ;  adapter on iff s_t > .5
G6.0: mean smoothed hit rate on old-task tokens ≤ 5%, no suite > 15%`,
    what: "Your continual-learning test with the LOCAL SUPPORT LEARNING paper. V8 tested LSL only where its own theory predicts failure. V9 measures the gate on old-task inputs during and after every phase.",
    arms: [["LSL-scaled", "2", "rank 16, JL 64, 16/32 comps", "S6_LSL_scaled", 1], ["Always-open LoRA", "2", "does gating help?", "S6_always_open", 1], ["VQ / selector-only", "2", "NCP paper +VQ", "S6_VQ_only", 1], ["LSL-v8", "2", "reproduce V8 config", "S6_LSL_v8", 0], ["Full fine-tune", "2", "forgetting reference", "S6_full_ft", 0], ["Replay 10%", "2", "ceiling only", null, 0]],
    gateText: [["Credit rule", "LSL is credited with retention on a phase only if its retained-suite loss beats always-open by a paired interval excluding 0, and its install is no worse than VQ-only's."], ["Negative control", "P3 (bind) is confirmed if LSL − always-open stays within ±5 points on retained navigation."]],
    cost: "~6 h mandatory · ~12 h full (Q2)", sec: "§10" },
];

/* ---------------------------------------------------------------- pipeline cards */
(function pipeline() {
  const host = $("#pipeline");
  STAGES.forEach(s => {
    const b = document.createElement("button");
    b.className = "stage-card";
    b.innerHTML = `<div class="bar" style="background:var(${s.color})"></div><span class="sn">${s.id} · ${s.sec}</span><b>${s.name}</b><p>${s.short}</p><div class="gatechips">${s.gates.map(g => `<span class="pill gate">${g}</span>`).join("")}</div>`;
    b.addEventListener("click", () => { openStage(s.id); showTab("stages"); scrollTo({ top: 0 }); });
    host.appendChild(b);
  });
})();

/* ================================================================ MODES FIGURE (overview) */
(function modes() {
  const host = $("#fig-modes");
  const W = 560, H = 320;
  const MODES = [{ x: 90, y: 95, p: .45, c: "c1", lab: "c1 → N5 · progress" }, { x: 330, y: 85, p: .40, c: "c2", lab: "c2 → N3 · progress" }, { x: 215, y: 255, p: .15, c: "c3", lab: "c3 → N0 · back" }];
  const MCOL = ["--enc", "--dec", "--sel"];
  const mean = MODES.reduce((a, m) => ({ x: a.x + m.p * m.x, y: a.y + m.p * m.y }), { x: 0, y: 0 });
  let rng = mulberry(7), codes, mse, wins, tick, anim = null;
  const K = 4;
  function reset() {
    rng = mulberry(7); tick = 0;
    codes = Array.from({ length: K }, (_, i) => ({ x: mean.x + 18 * Math.cos(i * 1.7), y: mean.y + 18 * Math.sin(i * 1.7) }));
    mse = { x: 215, y: 175 }; wins = new Array(K).fill(1 / K);
    draw();
  }
  function gauss() { let u = 0, v = 0; while (!u) u = rng(); while (!v) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function sample() { let r = rng(), i = 0; while (r > MODES[i].p && i < 2) { r -= MODES[i].p; i++; } return { x: MODES[i].x + 10 * gauss(), y: MODES[i].y + 10 * gauss() }; }
  function step() {
    const s = sample(), lr = .07;
    let best = 0, bd = 1e9;
    codes.forEach((c, i) => { const d = (c.x - s.x) ** 2 + (c.y - s.y) ** 2; if (d < bd) { bd = d; best = i; } });
    codes.forEach((c, i) => { const w = i === best ? .95 : .05 / (K - 1); c.x += lr * w * (s.x - c.x) * 1.0; c.y += lr * w * (s.y - c.y); });
    wins = wins.map((w, i) => .985 * w + .015 * (i === best ? 1 : 0));
    mse.x += .03 * (s.x - mse.x); mse.y += .03 * (s.y - mse.y);
    tick++;
    if (tick % 120 === 0) {          // dead-hypothesis restart (§6.4): split the most-winning code
      const lo = wins.indexOf(Math.min(...wins)), hi = wins.indexOf(Math.max(...wins));
      if (wins[lo] < 1 / (10 * K) + .02) { codes[lo] = { x: codes[hi].x + 6 * gauss(), y: codes[hi].y + 6 * gauss() }; wins[lo] = wins[hi] / 2; wins[hi] /= 2; }
    }
  }
  function draw() {
    const svg = svgRoot(host, W, H);
    const g = el("g", {}, svg);
    // sample clouds
    const r2 = mulberry(3);
    MODES.forEach((m, i) => {
      el("circle", { cx: m.x, cy: m.y, r: 30, style: `fill:var(${MCOL[i]});opacity:.10` }, g);
      el("circle", { cx: m.x, cy: m.y, r: 30, style: `fill:none;stroke:var(${MCOL[i]});stroke-dasharray:3 3;opacity:.7` }, g);
      for (let k = 0; k < 14; k++) { const a = r2() * 6.28, d = r2() * 22; el("circle", { cx: m.x + d * Math.cos(a), cy: m.y + d * Math.sin(a), r: 1.8, style: `fill:var(${MCOL[i]});opacity:.55` }, g); }
      txt(g, m.x, m.y + (m.y > 200 ? 44 : -54), m.lab, { size: 11.5, weight: 600, color: `var(${MCOL[i]})` });
      txt(g, m.x, m.y + (m.y > 200 ? 58 : -40), `p = ${m.p.toFixed(2)}`, { size: 10.5, cls: "t3 t-mono" });
    });
    // mean of modes
    el("line", { x1: mean.x - 7, y1: mean.y - 7, x2: mean.x + 7, y2: mean.y + 7, style: "stroke:var(--loss);stroke-width:2.5" }, g);
    el("line", { x1: mean.x - 7, y1: mean.y + 7, x2: mean.x + 7, y2: mean.y - 7, style: "stroke:var(--loss);stroke-width:2.5" }, g);
    txt(g, mean.x + 12, mean.y + 2, "mean of modes · no legal hop", { size: 11, anchor: "start", color: "var(--loss)", weight: 600 });
    // MSE predictor
    el("circle", { cx: mse.x, cy: mse.y, r: 6, style: "fill:var(--panel);stroke:var(--loss);stroke-width:2" }, g);
    txt(g, mse.x - 10, mse.y + 1, "MSE NCP", { size: 10.5, anchor: "end", color: "var(--loss)" });
    // codes
    codes.forEach((c, i) => {
      el("rect", { x: c.x - 6, y: c.y - 6, width: 12, height: 12, transform: `rotate(45 ${c.x} ${c.y})`, style: "fill:var(--code);stroke:var(--panel);stroke-width:1.5" }, g);
      txt(g, c.x + 10, c.y - 9, "k" + (i + 1), { size: 10.5, anchor: "start", weight: 700, cls: "t-mono", color: "var(--code)" });
    });
    // selector bars
    const bx = 418; txt(g, bx + 50, 22, "selector π (win share)", { size: 11, weight: 600, cls: "t2" });
    wins.forEach((w, i) => {
      const y = 44 + i * 30;
      txt(g, bx, y + 8, "k" + (i + 1), { size: 11, anchor: "start", cls: "t-mono", weight: 600 });
      el("rect", { x: bx + 22, y, width: 76, height: 14, rx: 4, style: "fill:var(--bg2)" }, g);
      el("rect", { x: bx + 22, y, width: Math.max(1, 76 * w), height: 14, rx: 4, style: "fill:var(--sel)" }, g);
      txt(g, bx + 102, y + 8, (100 * w).toFixed(0) + "%", { size: 10.5, anchor: "start", cls: "t3 t-mono" });
    });
    txt(g, bx + 50, 180, `updates: ${tick}`, { size: 11, cls: "t3 t-mono" });
    const legend = [["◆", "--code", "4 committed codes"], ["○", "--loss", "one MSE prediction"]];
    legend.forEach((l, i) => { txt(g, bx, 216 + i * 20, l[0], { size: 12, anchor: "start", color: `var(${l[1]})` }); txt(g, bx + 16, 216 + i * 20, l[2], { size: 11, anchor: "start", cls: "t2" }); });
    // current node
    el("circle", { cx: 215, cy: 175, r: 4, style: "fill:var(--ink3)" }, g); txt(g, 222, 186, "current state", { size: 10, anchor: "start", cls: "t3" });
  }
  function run() {
    if (anim) return;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const target = tick + 600;
    if (reduce) { while (tick < target) step(); draw(); return; }
    const f = () => { for (let i = 0; i < 6 && tick < target; i++) step(); draw(); anim = tick < target ? requestAnimationFrame(f) : null; };
    anim = requestAnimationFrame(f);
  }
  $("#mm-train").addEventListener("click", run);
  $("#mm-reset").addEventListener("click", () => { if (anim) cancelAnimationFrame(anim); anim = null; reset(); });
  reset();
})();
function mulberry(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

/* ================================================================ ARCHITECTURE 2D */
const INFO = {
  tokens: { t: "Token ids", cat: ["Input", "--tok"], d: "Four tokens make one chunk. On GraphNav a chunk is one prompt edge or one response hop, so one concept is one reasoning step.", kv: [["chunk", "k = 4 tokens"], ["hop", "[COL, TO, NODE, SEP]"], ["colours", "N_COL = 3"]], src: "§4.2 · data.py:145–151" },
  emb: { t: "Token embedding", cat: ["Encoder", "--enc"], d: "Embedding matrix tied with the output head.", kv: [["width", "d = 128"]], src: "§5.1 · model.py" },
  enc: { t: "Encoder E3", cat: ["Encoder", "--enc"], d: "Three causal Transformer blocks over tokens. Unchanged from V8-D.", eq: "H_t = Encoder_E3(embed(x))_t", kv: [["blocks", "3"], ["heads", "4"], ["FFN", "384"], ["position", "RoPE, pre-RMSNorm"]], src: "§5.1" },
  H: { t: "Token features H_t", cat: ["Encoder", "--enc"], d: "These feed both the pooling step and, directly, the decoder input. The decoder still reads true tokens; the concept stream adds a plan on top.", src: "§5.1" },
  pool: { t: "Pool k = 4", cat: ["Concepts", "--con"], d: "Mean of the four token features in a chunk, then RMS-normalized. This pooled concept is also the NCP target for the previous position.", eq: "C_j = RMS(mean(H_4j … H_4j+3))", src: "§5.1" },
  cin: { t: "Concept stream C_≤j", cat: ["Concepts", "--con"], d: "Causal over concepts and document-isolated. In closed-loop spans (§6.5) the student reads its own committed predictions here instead of encoder-pooled concepts.", kv: [["closed loop", "p_cl = .25, N ~ U{1..4}"], ["fed back", "detached ĉ^(k*) (soft; snap is an arm)"]], src: "§6.5" },
  core: { t: "Student core (first L−1 blocks)", cat: ["Concept module", "--con"], d: "Two variants. Dense: a C4 copy initialized from the teacher (V8 dense refit trained 869,504 parameters). Recurrent: the V8 HRM-pattern core, 2×(2L+1H), with 2,055,424 active parameters. This is the separate recurrent module from your stage 2.", eq: "V_j = blocks_1..L−1(C_≤j)_j", src: "§6.1 · V8 Core" },
  bh0: { t: "Hypothesis block B_H, code-free path", cat: ["Concept module", "--con"], d: "The last block runs once with no code. Its output u⁽⁰⁾ feeds the selector, the value and terminal heads (S4/S5), and distillation to the teacher. It is the zero-control anchor: the code-free plan reproduces the S1 model.", eq: "u_j^(0) = Norm(B_H(query = V_j, kv = V_≤j))", src: "§6.2" },
  codes: { t: "Code embeddings e_k (the discrete Z)", cat: ["Codes", "--code"], d: "Z as a discrete code k ∈ {1..K} with a learned 16-d embedding, written into the query through W_z (128 × 16, std .02). Codes change only the readout at position j; later positions attend to the code-free stream, so one forward pass serves all K hypotheses.", kv: [["e_k", "R^16, N(0,1) init"], ["write RMS", "≈ .08 (~4% of V_j, verify at init)"], ["dead codes", "split the top winner every 500 updates"]], src: "§6.2 · §6.4" },
  bhk: { t: "Hypothesis block B_H, K code paths", cat: ["Codes", "--code"], d: "Same block and weights, queried K times, each with a different code added to the query. Keys and values always come from the code-free stream. Costs +K single-position evaluations per chunk, about +29% block-positions at K = 8.", eq: "u_j^(k) = Norm(B_H(query = V_j + W_z e_k, kv = V_≤j))", src: "§6.2" },
  pq: { t: "PQ heads W_s (shared across hypotheses)", cat: ["Concepts", "--con"], d: "Each hypothesis keeps the NCP paper's soft-PQ expectation inside it, so the departure from the paper is exactly one axis (K). Heads are shared so codes differ only through u⁽ᵏ⁾, which is what makes K = 1 equivalence hold.", eq: "ĉ_j^(k) = concat_s Σ_n softmax(W_s u^(k) + b_s)_n Q_s,n\nA_j^(k) = W_A RMS(u^(k))", kv: [["segments", "8"], ["codes / segment", "32"], ["dims / segment", "16"]], src: "§6.2 · NCP paper eq. 9" },
  Q: { t: "Codebook Q and port W_A", cat: ["Codebook", "--code"], d: "Frozen and shared with the teacher in S2, as in the V8 refit.", kv: [["Q", "8 × 32 × 16"]], src: "§6.1" },
  hyp: { t: "K committed next concepts ĉ⁽ᵏ⁾", cat: ["Codes", "--code"], d: "Each code commits to one possible next concept. Winner-take-all training moves each one onto a mode of the true next-concept distribution, instead of the mean that squared-error NCP converges to.", src: "§2.3 · §6.3" },
  sel: { t: "Selector π", cat: ["Selector", "--sel"], d: "A linear head on u⁽⁰⁾ predicting which code will win. It is the MCTS prior in S5 and the policy trained by expert iteration in S4. Its target is the softmin posterior, so near-ties don't become label noise.", eq: "π_j = softmax(W_π u_j^(0) / T_π),  T_π = 1", src: "§6.2 · §6.3" },
  bestm: { t: "Best of m sampled codes", cat: ["Selector", "--sel"], d: "The decoder trains on plans chosen the way inference will choose them. m = 1 is a pure prior sample. m = 2 is V5's Forward-XM setting, the only configuration that produced a deployable prior (+.025). m = K is full hindsight, which was V5 V3's failure (+.345).", eq: "k_dec = argmin_{k ∈ S_j} d_j,k ,  S_j = m codes ~ π_j", kv: [["default", "m = 2"], ["arms", "m ∈ {1, 2, K}"]], src: "§6.3 · R3" },
  plan: { t: "Plan → decoder", cat: ["Plan", "--con"], d: "The chosen code's concept and port vector, shifted so the plan from chunk j first appears at position 4j+3 and steers chunk j+1. At prompt chunks with no scored NCP pair, the code-free plan is used.", eq: "plan_j = (ĉ_j^(k_dec), A_j^(k_dec))", src: "§6.2 · §4.2" },
  teacher: { t: "Frozen S1 teacher", cat: ["Frozen", "--frozen"], d: "One accepted A_delayed checkpoint. It supplies the true next pooled concept C_j+1 and its own (u, A, ĉ) for distillation of the code-free path.", src: "§6.1" },
  lwta: { t: "L_WTA · relaxed winner-take-all", cat: ["Loss", "--loss"], d: "Only the closest hypothesis is trained strongly. The others get ε/(K−1), which keeps them alive.", eq: "L_WTA = Σ_j w_j[(1−ε)d_j,k* + ε/(K−1) Σ_{k≠k*} d_j,k]\nε = .05, α = 0.1", src: "§6.3" },
  lsel: { t: "L_sel · selector loss", cat: ["Loss", "--loss"], d: "KL from the softmin posterior over hypothesis errors to the selector.", eq: "q_j,k = softmax_k(−d_j,k / τ_q),  τ_q = .1·mean_k d\nL_sel = KL(sg(q_j) ‖ π_j),  λ_sel = 0.1", src: "§6.3" },
  ldist: { t: "L_dist · distillation", cat: ["Loss", "--loss"], d: "Pins the code-free path to the teacher, so a zero control exists. Weight δ(u) follows the V8 REFIT schedule: 1 for the first 10% of updates, then linear to 0 by 30%.", eq: "L_dist = mean[‖u⁰−u_T‖² + ‖A⁰−A_T‖² + ‖ĉ⁰−ĉ_T‖²]/3", src: "§6.3 · train.py:55" },
  dsum: { t: "Decoder input", cat: ["Decoder", "--dec"], d: "Token features plus the shifted plan and port. In S2 the decoder is frozen and was trained only on expectation plans, so G2.1 checks whether it obeys sharper committed plans.", eq: "D_t = H_t + shift(ĉ)_t + γ ⊙ shift(A)_t ,  γ init 0.1", src: "§5.1 · G2.1" },
  dec: { t: "Decoder D3", cat: ["Decoder", "--dec"], d: "Three causal blocks over tokens. In S5, the decoder runs only at re-grounding points and on the final chosen path.", kv: [["blocks", "3"], ["heads", "4"], ["FFN", "384"]], src: "§5.1 · §9" },
  head: { t: "Tied output head", cat: ["Decoder", "--dec"], d: "Logits through the tied embedding.", src: "§5.1" },
  ltok: { t: "L_token", cat: ["Loss", "--loss"], d: "Next-token loss under the best-of-m plan. Symbolic rows are trained response-only, which is already SFT on verified demonstrations.", src: "§6.3 · data.py:155" },
};
/* ---------------------------------------------------------------- variants: shared state for the 2D and 3D views */
const VV = window.V9V;
Object.assign(INFO, VV.BASE_INFO);
let curV = VV.ALL, archK = 8, archSel = "codes";
const stt = id => (curV.train ? (curV.train[id] || "N") : "T");
const Kof = () => (curV.id === "ALL" ? archK : (curV.K || 0));
function pinfo(id) {
  const base = INFO[id], ov = curV.info && curV.info[id];
  if (!base && !ov) return null;
  const m = Object.assign({}, base || {}, ov || {});
  if (!m.cat && base) m.cat = base.cat;
  if (curV.core === "recurrent" && (id === "core" || id === "bh0") && !(ov && ov.d)) {
    m.d = id === "core" ? "V8's HRM-pattern recurrent core. Two L visits, one H visit, two more L visits (L, L, H, L, L) build V_j with shared weights fL and fH, 476,288 parameters in all. The schedule is L, L, H, L, L, H." : "For the recurrent core, B_H is the core's own final application: the last fH visit (visit 6 of L, L, H, L, L, H). Its weights are shared with visit 3, so no extra block is added. At K = 8 the code write enters the query of this visit.";
  }
  return m;
}
const CAPSV = v => {
  const bits = [];
  if (v.id === "ALL") return "All parts are shown. Dashed rose wires carry losses only. The losses drawn are the S2 objective. Pick a variant above to see what that arm actually trains.";
  bits.push("Solid boxes train or are active in this arm. Dashed grey boxes are frozen. Faded boxes do not exist in it. Dashed rose wires carry losses only.");
  if (v.note) bits.push(v.note);
  return bits.join(" ");
};
function stageTint(s) { return ({ S1: "--enc", S2: "--con", S3: "--dec", S4: "--sel", S5: "--code", S6: "--loss" })[s] || "--ink3"; }
function fmtN(n) { return n == null ? "not stated" : n.toLocaleString("en-US"); }

/* ---------------------------------------------------------------- 2D figure */
function lossSlots() {
  const L = curV.losses || {};
  return { A: L.s1 || (L.s2 && L.s2 !== "ltok" ? L.s2 : null), B: L.s2b || null, C: L.s3 || null };
}
function tagLoRA(g, x, y, w, text) {
  el("rect", { x, y, width: w, height: 14, rx: 5, style: "fill:var(--dec-s);stroke:var(--dec);stroke-dasharray:3 2;stroke-width:1" }, g);
  txt(g, x + w / 2, y + 7.5, text, { size: 8.5, weight: 700, color: "var(--dec)", cls: "t-mono" });
}
function drawArch() {
  if (curV.kind === "search") return drawSearchFlow();
  if (curV.body === "flat") return drawFlat();
  const host = $("#fig-arch"), K = Kof(), rec = curV.core === "recurrent", ad = curV.adapter, gated = !!(ad && ad.kind === "lsl");
  const W = 1180, H = gated ? 800 : 720, S1 = curV.id !== "ALL" && !curV.K, slots = lossSlots();
  const svg = svgRoot(host, W, H);
  [["mE", "var(--enc)"], ["mC", "var(--con)"], ["mD", "var(--dec)"], ["mK", "var(--code)"], ["mL", "var(--loss)"], ["mS", "var(--sel)"], ["mI", "var(--ink3)"]].forEach(m => marker(svg, m[0], m[1]));
  const flowCls = $("#arch-flow").checked ? " flowing" : "";
  [[20, 40, 280, "Encoder · E3", "--enc"], [320, 40, 560, "Concept module · " + (rec ? "recurrent HRM core" : "dense core") + (K > 1 ? " + H-NCP" : K === 1 ? " · K = 1" : ""), "--con"], [900, 40, 260, "Decoder · D3", "--dec"]].forEach(f => {
    el("rect", { x: f[0], y: f[1], width: f[2], height: 640, rx: 14, class: "frame" }, svg);
    el("rect", { x: f[0], y: f[1], width: f[2], height: 5, rx: 2, style: `fill:var(${f[4]})` }, svg);
    txt(svg, f[0] + 14, f[1] + 24, f[3], { size: 14, weight: 700, anchor: "start" });
  });
  const wires = el("g", {}, svg), blocks = el("g", {}, svg);
  const both = (a, b) => stt(a) !== "N" && stt(b) !== "N";
  function box(id, x, y, w, h, kind, l1, l2, tag) {
    let k = kind, cls = "";
    const s = stt(id); if (s === "F") k = "frz"; if (s === "N") cls = " dim";
    const g = el("g", { class: `blk k-${k}${cls}${archSel === id ? " sel" : ""}`, tabindex: 0, role: "button", "aria-label": (pinfo(id) || {}).t || id }, blocks);
    el("rect", { x, y, width: w, height: h, rx: 9 }, g);
    const cx = x + w / 2 - (tag ? 34 : 0);
    txt(g, cx, y + (l2 ? h / 2 - 7 : h / 2), l1, { size: 12.5, weight: 650 });
    if (l2) txt(g, cx, y + h / 2 + 9, l2, { size: 10.5, cls: "t2" });
    if (tag) tagLoRA(g, x + w - 72, y + h / 2 - 7, 66, tag);
    const go = () => { archSel = id; showInsp(id); drawArch(); };
    g.addEventListener("click", go); g.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
    return g;
  }
  const tagFor = where => (ad && ad.where.includes(where) ? (ad.kind === "lora" ? `LoRA r${ad.rank}` : ad.gate ? `r${ad.rank} ⊕ gate` : `r${ad.rank} open`) : null);
  // ---- encoder
  box("tokens", 50, 620, 220, 36, "io", "Token ids x", "4 per chunk · one chunk = one hop");
  box("emb", 50, 566, 220, 36, "enc", "Embedding · d 128", "tied with the head");
  ["E1", "E2", "E3"].forEach((n, i) => { box("enc", 50, 506 - 54 * i, 220, 40, "enc", `Encoder block ${n}`, "4 heads · FF384 · RoPE", tagFor("enc")); });
  box("H", 50, 340, 220, 36, "enc", "Token features H_t");
  box("pool", 50, 262, 220, 46, "con", "Pool k = 4", "C_j = RMS(mean H_4j…4j+3)");
  [[620, 602], [566, 546], [506, 492], [452, 438], [398, 376], [340, 308]].forEach(a => wire(wires, [[160, a[0]], [160, a[1] + 1]], "w-enc", "mE"));
  if (curV.prefixlm) txt(svg, 160, 700, "PrefixLM mask: the prompt is read bidirectionally", { size: 11, color: "var(--enc)", weight: 650 });
  // ---- concept module
  const L = 340, M = 550, R = 760;
  box("cin", L, 620, 190, 36, "con", "Concepts C_≤j", curV.closedLoop === false ? "closed loop off" : "or own predictions (closed loop)");
  if (!rec) {
    ["1", "2", "3"].forEach((n, i) => { box("core", L, 556 - 54 * i, 190, 40, "con", n === "3" ? "Core block 3 → V_j" : `Core block ${n}`, i === 0 ? "dense C4 copy" : null, tagFor("core")); });
    [[620, 597], [556, 543], [502, 489], [448, 427]].forEach(a => wire(wires, [[435, a[0]], [435, a[1]]], "w-con" + flowCls, "mC"));
  } else {
    const names = ["L visit 1", "L visit 2", "H visit 3 · fH", "L visit 4", "L visit 5 → V_j"];
    names.forEach((nm, i) => { const y = 580 - 29 * i, g = box("core", L + 16, y, 160, 24, i === 2 ? "sel" : "con", nm); if (i === 2) g.querySelector("rect").style.strokeWidth = "2.2"; });
    wire(wires, [[L + 96, 620], [L + 96, 605]], "w-con" + flowCls, "mC");
    [0, 1, 2, 3].forEach(i => wire(wires, [[L + 96, 580 - 29 * i], [L + 96, 580 - 29 * i - 5]], "w-con" + flowCls, "mC"));
    wire(wires, [[L + 16, 464], [L + 6, 464], [L + 6, 592], [L + 16, 592]], "w-con", "mC");
      }
  box("bh0", L, 380, 190, 46, "con", S1 ? (rec ? "H visit 6 · last block" : "Core block 4 (last)") : rec ? "B_H = fH visit 6 · code-free" : "B_H · code-free", S1 ? "→ u, then ĉ via PQ heads" : rec ? "→ u⁽⁰⁾ (shares visit-3 weights)" : "→ u⁽⁰⁾");
  box("sel", L, 296, 190, 46, "sel", K ? `Selector π over ${K}` : "no selector", K ? "softmax(W_π u⁽⁰⁾)" : "");
  box("codes", M, 448, 190, 40, "code", K > 1 ? `Code embeddings e_1…e_${K}` : K === 1 ? "No code (K = 1)" : "No codes", K > 1 ? "discrete Z · 16-d" : K === 1 ? "e_1 not instantiated" : "added in S2");
  box("bhk", M, 380, 190, 46, "code", K > 1 ? `B_H + W_z e_k  × ${K}` : "u⁽¹⁾ ≡ u⁽⁰⁾", K > 1 ? "same weights, K queries" : "same path as code-free");
  box("pq", M, 316, 190, 40, "con", stt("pq") === "F" ? "PQ heads W_s (frozen)" : "PQ heads W_s (shared)", "→ ĉ⁽ᵏ⁾, A⁽ᵏ⁾");
  const fan = el("g", { class: `blk k-code${archSel === "hyp" ? " sel" : ""}`, tabindex: 0, role: "button", "aria-label": "K committed next concepts" }, blocks);
  if (stt("hyp") === "F") fan.setAttribute("class", "blk k-frz" + (archSel === "hyp" ? " sel" : ""));
  if (stt("hyp") === "N") fan.classList.add("dim");
  el("rect", { x: M, y: 236, width: 190, height: 52, rx: 9 }, fan);
  const n = Math.max(1, K), cw = Math.min(30, (176 - (n - 1) * 4) / n);
  for (let i = 0; i < n; i++) {
    const x = M + 95 - (n * cw + (n - 1) * 4) / 2 + i * (cw + 4);
    el("rect", { x, y: 244, width: cw, height: 18, rx: 4, style: `fill:var(--code);opacity:${i === 1 % n ? 1 : .45};stroke:none` }, fan);
  }
  txt(fan, M + 95, 276, K > 1 ? `${K} committed next concepts ĉ⁽ᵏ⁾` : K === 1 ? "one next concept = V8 NCP" : "one prediction ĉ, no hypotheses", { size: 11, weight: 600 });
  fan.addEventListener("click", () => { archSel = "hyp"; showInsp("hyp"); drawArch(); });
  box("Q", R, 316, 106, 40, "code", "Codebook Q", "8×32×16");
  if (stt("teacher") !== "N") box("teacher", R, 604, 106, 52, "frz", "S1 teacher", "frozen");
  if (slots.A) box(slots.A, R, 238, 106, 48, "loss", ({ lncp: "L_NCP", lwta: "L_WTA", lpi: "L_π" })[slots.A], ({ lncp: curV.alpha && /α = 0 for all/.test(curV.alpha) ? "α = 0 · off" : "α(u) · delayed", lwta: "vs sg(C_j+1)", lpi: "visit counts" })[slots.A]);
  if (slots.B) box(slots.B, R, 172, 106, 44, "loss", slots.B === "lsel" ? "L_sel" : "L_V · L_τ", slots.B === "lsel" ? "KL(q ‖ π)" : "outcome, DONE");
  if (slots.C) box(slots.C, R, 384, 106, 40, "loss", "L_dist", "u⁽⁰⁾ → teacher");
  box("bestm", 420, 172, 240, 46, "sel", K > 1 ? "Best of m = 2 sampled codes" : K === 1 ? "k_dec = 1" : "no selection", K > 1 ? "plan the way inference picks it" : K === 1 ? "π ≡ 1, L_sel ≡ 0" : "one prediction feeds the plan");
  box("plan", 420, 100, 240, 42, "con", curV.codeFree ? "Plan (ĉ⁽⁰⁾, A⁽⁰⁾) · code-free" : "Plan (ĉ⁽ᵏᵈᵉᶜ⁾, A⁽ᵏᵈᵉᶜ⁾)", "→ shift → decoder");
  if (stt("vtau") !== "N") box("vtau", R, 58, 106, 34, "sel", "V_ψ · τ_ψ heads", "129 + 129 params");
  // concept wires
  wire(wires, [[270, 285], [306, 285], [306, 638], [L, 638]], "w-con" + flowCls, "mC");
  if (rec) { wire(wires, [[L + 96, 464], [L + 96, 449]], "w-con" + flowCls, "mC"); wire(wires, [[435, 448], [435, 427]], "w-con" + flowCls, "mC"); }
  if (stt("codes") !== "N") {
    wire(wires, [[L + 190, 468], [M, 468]], "w-con", null);
    wire(wires, [[M + 95, 448], [M + 95, 427]], "w-code" + flowCls, "mK");
    wire(wires, [[M + 95, 380], [M + 95, 357]], "w-code" + flowCls, "mK");
  } else wire(wires, [[L + 190, 395], [M - 12, 395], [M - 12, 336], [M, 336]], "w-con" + flowCls, "mC");
  wire(wires, [[M + 95, 316], [M + 95, 289]], "w-code" + flowCls, "mK");
  if (stt("sel") !== "N") wire(wires, [[435, 380], [435, 343]], "w-sel", "mS");
  if (stt("sel") !== "N") wire(wires, [[435, 296], [435, 260], [480, 260], [480, 219]], "w-sel", "mS");
  wire(wires, [[M + 95, 236], [M + 95, 228], [600, 228], [600, 219]], "w-code" + flowCls, "mK");
  wire(wires, [[540, 172], [540, 143]], "w-con" + flowCls, "mC");
  wire(wires, [[M + 190, 340], [R, 340]], "w-code", "mK");
  if (slots.A === "lwta" || slots.A === "lncp") wire(wires, [[M + 190, 262], [R, 262]], "w-loss", "mL");
  if (slots.A === "lwta" && slots.B === "lsel") wire(wires, [[R + 53, 238], [R + 53, 217]], "w-loss", "mL");
  if (slots.B === "lsel") wire(wires, [[L + 190, 330], [568, 330], [568, 300], [746, 300], [746, 194], [R, 194]], "w-loss", null);
  if (slots.A === "lpi") wire(wires, [[L + 190, 330], [568, 330], [568, 300], [746, 300], [746, 262], [R, 262]], "w-loss", "mL");
  if (slots.B === "lv" && stt("vtau") !== "N") wire(wires, [[R + 53, 92], [R + 53, 171]], "w-sel", "mS");
  if (slots.C && stt("teacher") !== "N") {
    wire(wires, [[L + 190, 404], [M - 8, 404], [M - 8, 430], [R - 8, 430], [R - 8, 404], [R, 404]], "w-loss", null);
    wire(wires, [[R + 53, 604], [R + 53, 425]], "w-loss", "mL");
  }
  // ---- decoder
  const DX = 920;
  box("dsum", DX, 560, 220, 52, "dec", "D_t = H_t + shift(ĉ) + γ·shift(A)", "γ init 0.1 · shift exposes v_j at 4j+3");
  ["D1", "D2", "D3"].forEach((nm, i) => { box("dec", DX, 494 - 54 * i, 220, 40, "dec", `Decoder block ${nm}`, "4 heads · FF384", tagFor("dec")); });
  box("head", DX, 324, 220, 38, "dec", "Tied head → logits");
  box("ltok", DX, 252, 220, 40, "loss", (pinfo("ltok") || {}).t && curV.info && curV.info.ltok ? curV.info.ltok.t.split(" · ")[0] : "L_token", "response-only on symbolic rows");
  box("tokens", DX, 176, 220, 40, "io", "Tokens of chunk j+1", "steered by the plan from chunk j");
  [[560, 535], [494, 481], [440, 427], [386, 363]].forEach(a => wire(wires, [[1030, a[0]], [1030, a[1]]], "w-dec" + flowCls, "mD"));
  wire(wires, [[1030, 324], [1030, 293]], "w-loss", "mL");
  wire(wires, [[1030, 252], [1030, 217]], "w-dec", "mD");
  wire(wires, [[660, 121], [886, 121], [886, 586], [DX, 586]], "w-con" + flowCls, "mC");
  txt(svg, 892, 112, "plan", { size: 11, anchor: "start", color: "var(--con)", weight: 650 });
  wire(wires, [[50, 358], [34, 358], [34, 696], [1030, 696], [1030, 613]], "w-enc" + flowCls, "mE");
  txt(svg, 600, 690, "H_t: the decoder still reads true tokens", { size: 11, color: "var(--enc)", weight: 600 });
  if (gated) drawGateStrip(svg);
  $("#arch-cap").textContent = CAPSV(curV);
}
function drawGateStrip(svg) {
  const a = curV.adapter, gt = a.gate, y = 722, h = 58;
  el("rect", { x: 20, y: y - 14, width: 1140, height: h + 28, rx: 12, class: "frame" }, svg);
  const g = el("g", {}, svg);
  txt(svg, 34, y + 2, gt ? "LSL gate on every adapted matrix" : "No gate: the adapter is always applied", { size: 12.5, weight: 700, anchor: "start" });
  const steps = gt ? [["input x", "matrix input"], [`JL projection`, `${gt.jl}-d`], ["Φpos · Φneg", `${gt.pos} / ${gt.neg} diagonal GMMs`], ["g = 1[Φpos > Φneg]", "per token"], [gt.smooth ? "s_t = .3 g + .7 s_t−1" : "no smoothing (ρ = 1)", "temporal smoothing"], ["adapter on iff s_t > .5", "else base weights only"]] : [["input x", "matrix input"], ["adapter always on", "W' = W + B·A"], ["no Φ models", "no retention mechanism"]];
  const n = steps.length, bw = Math.min(170, (1100 - (n - 1) * 24) / n);
  steps.forEach((s, i) => {
    const x = 40 + i * (bw + 24), gg = el("g", { class: "blk k-loss", tabindex: 0, role: "button" }, g);
    el("rect", { x, y: y + 12, width: bw, height: 36, rx: 8 }, gg);
    txt(gg, x + bw / 2, y + 25, s[0], { size: 11, weight: 650 }); txt(gg, x + bw / 2, y + 39, s[1], { size: 9.5, cls: "t2" });
    gg.addEventListener("click", () => { archSel = "gate"; showInsp("gate"); });
    if (i) wire(g, [[x - 24, y + 30], [x - 2, y + 30]], "w-loss", "mL");
  });
}

/* ---------------------------------------------------------------- flat model (A_flat) */
function drawFlat() {
  const host = $("#fig-arch"), W = 1180, H = 720, svg = svgRoot(host, W, H);
  [["mE", "var(--enc)"], ["mC", "var(--con)"], ["mD", "var(--dec)"], ["mL", "var(--loss)"]].forEach(m => marker(svg, m[0], m[1]));
  const flowCls = $("#arch-flow").checked ? " flowing" : "";
  el("rect", { x: 330, y: 40, width: 520, height: 640, rx: 14, class: "frame" }, svg); el("rect", { x: 330, y: 40, width: 520, height: 5, rx: 2, style: "fill:var(--enc)" }, svg);
  txt(svg, 346, 64, "Flat Transformer · 10 blocks · token stream only", { size: 14, weight: 700, anchor: "start" });
  el("rect", { x: 20, y: 40, width: 280, height: 640, rx: 14, class: "frame", style: "stroke-dasharray:5 4" }, svg);
  txt(svg, 34, 64, "What the hierarchy adds (absent here)", { size: 13, weight: 700, anchor: "start" });
  const wires = el("g", {}, svg), blocks = el("g", {}, svg);
  function box(id, x, y, w, h, kind, l1, l2, state) {
    const s = state || stt(id); let k = kind, cls = ""; if (s === "F") k = "frz"; if (s === "N") cls = " dim";
    const g = el("g", { class: `blk k-${k}${cls}${archSel === id ? " sel" : ""}`, tabindex: 0, role: "button", "aria-label": (pinfo(id) || {}).t || id }, blocks);
    el("rect", { x, y, width: w, height: h, rx: 9 }, g);
    txt(g, x + w / 2, y + (l2 ? h / 2 - 6 : h / 2), l1, { size: 12.5, weight: 650 }); if (l2) txt(g, x + w / 2, y + h / 2 + 9, l2, { size: 10.5, cls: "t2" });
    const go = () => { archSel = id; showInsp(id); drawArch(); };
    g.addEventListener("click", go); g.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
  }
  const X = 440, BW = 300;
  box("tokens", X, 628, BW, 36, "io", "Token ids x", "same 4-token chunks, but chunks mean nothing here");
  box("emb", X, 574, BW, 36, "enc", "Embedding · d 128", "tied with the head");
  for (let i = 0; i < 10; i++) {
    const y = 522 - 43 * i, band = i < 3 ? ["enc", "≙ encoder depth E3"] : i < 7 ? ["con", "≙ concept-core depth C4"] : ["dec", "≙ decoder depth D3"];
    box("flatblk", X, y, BW, 34, band[0], `Block ${i + 1}`, `${i < 9 ? "FF400" : "FF384"} · ${band[1]}`);
    wire(wires, [[X + BW / 2, y + 43 - (i ? 0 : 0) - 0], [X + BW / 2, y + 35]], "w-ink" + flowCls, "mI");
  }
  wire(wires, [[X + BW / 2, 628], [X + BW / 2, 611]], "w-ink" + flowCls, "mI"); wire(wires, [[X + BW / 2, 574], [X + BW / 2, 557]], "w-ink" + flowCls, "mI");
  box("head", X, 78, BW, 38, "dec", "RMSNorm → tied head → logits");
  wire(wires, [[X + BW / 2, 92 + 66], [X + BW / 2, 117]], "w-ink" + flowCls, "mI");
  box("ltok", 900, 78, 240, 40, "loss", "L_token", "the only loss (no NCP, no VQ)");
  wire(wires, [[X + BW, 97], [900, 97]], "w-loss", "mL");
  // depth brackets
  [[0, 3, "3 blocks ≙ E3", "--enc"], [3, 7, "4 blocks ≙ C4", "--con"], [7, 10, "3 blocks ≙ D3", "--dec"]].forEach(b => {
    const y0 = 522 - 43 * (b[1] - 1), y1 = 522 - 43 * b[0] + 34;
    el("path", { d: `M ${X + BW + 14},${y0} q 10,0 10,10 v ${y1 - y0 - 20} q 0,10 -10,10`, style: `fill:none;stroke:var(${b[3]});stroke-width:2` }, svg);
    txt(svg, X + BW + 36, (y0 + y1) / 2, b[2], { size: 11.5, anchor: "start", weight: 650, color: `var(${b[3]})` });
  });
  txt(svg, 900, 322, "Depth is matched: 3 + 4 + 3 = 10 blocks.", { size: 12, anchor: "start", weight: 650 });
  txt(svg, 900, 342, "Parameters: 2,450,048 vs 2,448,640.", { size: 12, anchor: "start", cls: "t2" });
  txt(svg, 900, 366, "Text loss: flat is lower, about .02–.07 nats", { size: 11, anchor: "start", cls: "t2" });
  txt(svg, 900, 382, "(V7, V8). Navigation: hierarchy leads in V8.", { size: 11, anchor: "start", cls: "t2" });
  // ghosts of the missing parts
  box("pool", 50, 560, 220, 46, "con", "Pool k = 4", "C_j = RMS(mean H_4j…4j+3)", "N");
  box("cin", 50, 480, 220, 36, "con", "Concept stream C_≤j", "", "N");
  box("core", 50, 396, 220, 56, "con", "Concept core C4", "dense or HRM recurrent", "N");
  box("codes", 50, 316, 220, 44, "code", "H-NCP codes + selector", "S2 only", "N");
  box("plan", 50, 232, 220, 46, "con", "Plan → decoder input", "shift(ĉ) + γ·shift(A)", "N");
  box("lncp", 50, 148, 220, 44, "loss", "L_NCP · L_VQ", "no concept losses", "N");
  txt(svg, 160, 118, "All of these are what A_flat does without.", { size: 11, cls: "t3" });
  $("#arch-cap").textContent = "A_flat is the architecture control: a plain causal Transformer with the same depth and about the same parameter count, trained on the same data, with only the token loss. The ghost column lists what the hierarchy adds. H3 asks whether the hierarchy beats it on sealed navigation; H4 asks whether it keeps the text advantage.";
}

/* ---------------------------------------------------------------- search methods (S5): the inference loop */
function drawSearchFlow() {
  const host = $("#fig-arch"), W = 1180, H = 640, svg = svgRoot(host, W, H), m = curV.method, V = curV;
  [["mI", "var(--ink3)"], ["mC", "var(--con)"], ["mK", "var(--code)"], ["mS", "var(--sel)"], ["mD", "var(--dec)"]].forEach(a => marker(svg, a[0], a[1]));
  const wires = el("g", {}, svg), blocks = el("g", {}, svg);
  const tok = !!V.token, free = !tok;
  function bx(x, y, w, h, kind, l1, l2) { const g = el("g", { class: `blk k-${kind}` }, blocks); el("rect", { x, y, width: w, height: h, rx: 10 }, g); txt(g, x + w / 2, y + (l2 ? h / 2 - 8 : h / 2), l1, { size: 12.5, weight: 650 }); if (l2) txt(g, x + w / 2, y + h / 2 + 9, l2, { size: 10.5, cls: "t2" }); return g; }
  const arrow = (pts, cls) => wire(wires, pts, cls || "w-ink flowing", "mI");
  bx(30, 270, 150, 56, "io", "Prompt", "encoded once");
  if (m === "best_of_n") {
    bx(260, 140, 200, 56, "dec", "Sample a full answer", "T = 1, code-free plan");
    bx(260, 270, 200, 56, "dec", "Sample another", "independent draw");
    bx(260, 400, 200, 56, "dec", "…until the budget ends", "calls counted one by one");
    bx(560, 270, 190, 56, "loss", "Verifier", "binary success");
    bx(850, 270, 190, 56, "sel", "First verified answer", "else failure at 2× budget");
    [[168, 142], [298, 298], [428, 428]].forEach(a => arrow([[180, 298], [220, 298], [220, a[2] ? a[2] + 27 : 168], [260, a[2] ? a[2] + 27 : 168]]));
    [168, 298, 428].forEach(y => arrow([[460, y], [510, y], [510, 298], [560, 298]]));
    arrow([[750, 298], [850, 298]]);
  } else if (m === "answer_loop") {
    bx(260, 150, 230, 62, "con", "GEN: a fresh inner search", "RH episode, fresh root noise");
    bx(260, 360, 230, 62, "loss", "CONT: refine a failed answer", "kernel at a uniform committed chunk");
    bx(570, 255, 200, 56, "sel", "Thompson sampling", "Beta(1, 1+n) per branch");
    bx(860, 150, 210, 62, "loss", "Verifier", "binary feedback");
    bx(860, 360, 210, 62, "dec", "Kernel", "GradCuit vs random direction");
    arrow([[180, 298], [220, 298], [220, 181], [260, 181]]); arrow([[490, 181], [530, 181], [530, 283], [570, 283]]); arrow([[490, 391], [530, 391], [530, 283], [570, 283]], "w-ink");
    arrow([[770, 283], [815, 283], [815, 181], [860, 181]]); arrow([[770, 283], [815, 283], [815, 391], [860, 391]]); arrow([[860, 391], [800, 480], [375, 480], [375, 423]], "w-loss");
    txt(svg, 600, 500, "failed answers feed CONT", { size: 11, cls: "t2" });
  } else {
    const Rr = V.R, ground = tok || Rr === 1;
    bx(250, 120, 190, 56, "sel", "PUCT select", "1.5·π·√N / (1+n)");
    bx(250, 240, 190, 62, tok ? "dec" : "con", tok ? "Expand: force a colour" : "Expand: pick a code", tok ? "3 colours + STOP, then greedy decode" : "ĉ⁽ᵏ⁾ → one core step");
    bx(250, 372, 190, 62, tok ? "dec" : (ground ? "dec" : "con"), tok ? "Child = decoded hop" : ground ? "Ground: decode 4 tokens" : "Decoder-free child", tok ? "28 block-positions" : ground ? "+24 block-positions" : "≈ 5 block-positions");
    bx(540, 240, 200, 62, "loss", m === "token_tree" ? "Leaf: rollout or value" : "Leaf: V_ψ, τ_ψ or verifier", m === "token_tree" ? "greedy code-free completion" : "AUROC-gated (R5)");
    bx(830, 240, 190, 56, "sel", "Back up Q, N", "visit counts");
    bx(540, 120, 200, 56, "con", m === "rh" ? "Commit one hop" : m === "g" ? "One tree for the whole answer" : "Commit after the budget", m === "rh" ? `receding horizon H = ${V.H || 1}` : Rr === 0 ? "decode only at the end (IMAG)" : `re-ground every R = ${Rr || "∞"}`);
    bx(830, 380, 190, 56, "loss", "Verify the answer", "on grounded tokens");
    arrow([[180, 298], [215, 298], [215, 148], [250, 148]]); arrow([[345, 176], [345, 240]]); arrow([[345, 302], [345, 372]]); arrow([[440, 403], [490, 403], [490, 271], [540, 271]]);
    arrow([[740, 271], [830, 271]]); arrow([[925, 240], [925, 148], [740, 148]], "w-ink"); arrow([[640, 120], [640, 90], [345, 90], [345, 120]], "w-ink"); arrow([[925, 296], [925, 380]]);
    txt(svg, 585, 74, `${V.label}: ${V.sub}`, { size: 12, cls: "t2" });
  }
  const notes = { rh: "Decoder-free children between hops: prompt encoded once, each child is one code and one incremental core step.", g: "One tree over the whole answer; grounded at every hop when R = 1.", token_tree: "The primary competitor: same weights in code-free mode, same PUCT, decoded children, equal wall-clock.", best_of_n: "The strongest simple baseline: no search, just sampling until the verifier is satisfied.", answer_loop: "Answer-level family (AB-MCTS-A). The kernel is paired with a random direction of the same norm, so GradCuit is never reported alone." };
  $("#arch-cap").textContent = (notes[m] || "") + " Budgets are 1, 4 and 16× one greedy answer's wall-clock; a hard stop at 2× marks unfinished items as failures.";
}

/* ---------------------------------------------------------------- inspector + gallery */
function infoHTML(id) {
  const I = pinfo(id); if (!I) return "";
  let h = `<div class="cat"><i style="background:var(${(I.cat || ["", "--ink3"])[1]})"></i>${(I.cat || ["Part"])[0]}</div><h3>${I.t}</h3><p>${I.d}</p>`;
  if (I.eq) h += `<pre>${I.eq}</pre>`;
  if (I.kv) h += `<dl class="kv">${I.kv.map(k => `<dt>${k[0]}</dt><dd>${k[1]}</dd>`).join("")}</dl>`;
  if (curV.id !== "ALL" && curV.train) { const s = curV.train[id]; if (s) h += `<p><span class="pill ${s === "T" ? "gate" : "untested"}">${curV.label}: ${s === "T" ? "trains / active" : s === "F" ? "frozen" : "not present"}</span></p>`; }
  h += `<div class="src">draft_v9.md ${I.src || curV.source || ""}</div>`;
  return h;
}
function showInsp(id) { $("#arch-insp").innerHTML = infoHTML(id); }
function seg(sel, attr, fn) {
  $$(sel + " button").forEach(b => b.addEventListener("click", () => { $$(sel + " button").forEach(x => x.classList.toggle("on", x === b)); fn(b.dataset[attr]); }));
}
function mini(v) {      // thumbnail: stacked bars for the encoder, core and decoder
  const bar = (x, y, w, h, c, o) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="1.5" style="fill:var(${c});opacity:${o == null ? 1 : o}"/>`;
  const st = id => (v.train ? (v.train[id] || "N") : "T"), op = id => (st(id) === "T" ? 1 : st(id) === "F" ? .4 : .13);
  let s = `<svg viewBox="0 0 64 40" width="64" height="40" aria-hidden="true">`;
  if (v.kind === "search") return s + `<circle cx="32" cy="8" r="3.5" style="fill:var(--sel)"/>` + [14, 32, 50].map(x => `<line x1="32" y1="8" x2="${x}" y2="22" style="stroke:var(--ink3)"/><circle cx="${x}" cy="22" r="3" style="fill:var(--${v.token ? "dec" : "con"})"/>`).join("") + [8, 20, 26, 38, 44, 56].map(x => `<line x1="${x < 32 ? 14 : x < 40 ? 32 : 50}" y1="22" x2="${x}" y2="34" style="stroke:var(--ink3)"/><circle cx="${x}" cy="34" r="2.2" style="fill:var(--${v.token ? "dec" : "con"})"/>`).join("") + `</svg>`;
  if (v.body === "flat") { for (let i = 0; i < 10; i++) s += bar(18, 36 - 3.6 * (i + 1), 28, 2.8, i < 3 ? "--enc" : i < 7 ? "--con" : "--dec"); return s + `</svg>`; }
  [0, 1, 2].forEach(i => { s += bar(3, 34 - 7 * i, 16, 5, "--enc", op("enc")); s += bar(45, 34 - 7 * i, 16, 5, "--dec", op("dec")); });
  const rec = v.core === "recurrent", n = rec ? 5 : 4;
  for (let i = 0; i < n; i++) s += bar(24, 36 - (rec ? 5.2 : 6) * (i + 1) + 1, 16, rec ? 3.6 : 4.6, i === 2 && rec ? "--sel" : "--con", op("core"));
  const K = v.id === "ALL" ? 8 : v.K || 0;
  if (K > 1 && st("codes") !== "N") for (let i = 0; i < Math.min(K, 8); i++) s += `<rect x="${24 + i * 2.1}" y="2" width="1.6" height="6" style="fill:var(--code);opacity:${i === 2 ? 1 : .5}"/>`;
  if (v.adapter) s += `<rect x="1" y="2" width="62" height="3" rx="1.5" style="fill:var(--dec);opacity:.55"/>`;
  return s + `</svg>`;
}
function liveFor(v) {
  const L = window.V9L && window.V9L.data; if (!L || !v.runs) return null;
  const re = new RegExp(v.runs), runs = Object.keys(L.runs).filter(n => re.test(n));
  return runs.length ? runs.map(n => ({ name: n, r: L.runs[n] })) : null;
}
function drawGallery() {
  const host = $("#arch-gallery"); if (!host) return;
  const showFull = $("#g-full") ? $("#g-full").checked : true;
  const groups = ["S1", "S2", "S3", "S4", "S5", "S6"].map(sid => {
    const items = (sid === "S5" ? VV.SEARCH : VV.VARIANTS.filter(v => v.stage === sid)).filter(v => showFull || v.tier !== "full");
    return items.length ? `<div class="gal-group"><div class="gal-head"><i style="background:var(${stageTint(sid)})"></i>${VV.STAGE_NAMES[sid]} <span class="muted">· ${VV.STAGE_QUEUE[sid]}</span></div><div class="gal-row">${items.map(v => {
      const lv = liveFor(v), st = lv ? (lv.some(x => !x.r.completed) ? "running" : "done") : null;
      return `<button class="gal-card${curV.id === v.id ? " on" : ""}" data-v="${v.id}" title="${v.sub}">${mini(v)}<span class="gal-t"><b>${v.label}</b><span>${v.sub}</span></span><span class="gal-tags">${v.tier === "full" ? `<em class="tag-full">full column</em>` : ""}${st ? `<em class="tag-${st}">${st === "running" ? "● training" : "✓ trained"}</em>` : ""}</span></button>`;
    }).join("")}</div></div>` : "";
  });
  host.innerHTML = groups.join("");
  host.querySelectorAll(".gal-card").forEach(b => b.addEventListener("click", () => setVariant(b.dataset.v)));
}
function archHead() {
  const v = curV, el_ = $("#arch-head"); if (!el_) return;
  const pill = (t, c) => `<span class="pill ${c || "untested"}">${t}</span>`;
  const lv = liveFor(v);
  el_.innerHTML = `<div class="ah-main"><span class="ah-dot" style="background:var(${stageTint(v.stage)})"></span><h3>${v.label}</h3><span class="muted">${v.sub}</span></div>
    <div class="ah-meta">${v.id === "ALL" ? "" : pill(v.stage + " · " + v.queue, "gate") + pill(v.tier === "full" ? "full column" : "mandatory", v.tier === "full" ? "untested" : "gate")}
    <span><b class="tabular">${fmtN(v.params.total)}</b> active parameters</span><span><b class="tabular">${fmtN(v.params.trainable)}</b> trainable</span><span class="muted">${v.params.note || ""}</span>
    ${lv ? `<span class="chip"><i style="background:var(--good)"></i>${lv.length} run${lv.length > 1 ? "s" : ""} · ${lv.every(x => x.r.completed) ? "trained" : "training now"}</span>` : ""}</div>`;
}
function setVariant(id, quiet) {
  const v = id === "ALL" ? VV.ALL : VV.byId(id); if (!v) return;
  curV = v;
  archSel = v.body === "flat" ? "flatblk" : v.kind === "search" ? "" : (v.adapter && v.adapter.kind === "lsl") ? "gate" : (Kof() > 1 ? "codes" : "core");
  $$("#v3-variant, #arch-variant").forEach(s => { s.value = id; });
  $("#arch-kseg").hidden = id !== "ALL";
  drawGallery(); archHead(); drawArch(); showInsp(archSel);
  if (!quiet) { try { localStorage.setItem("v9-variant", id); } catch (e) {} window.dispatchEvent(new CustomEvent("v9variant", { detail: id })); }
}
function variantOptions() {
  const og = (sid, list) => `<optgroup label="${VV.STAGE_NAMES[sid]}">${list.map(v => `<option value="${v.id}">${v.label}${v.tier === "full" ? " (full column)" : ""}</option>`).join("")}</optgroup>`;
  return `<option value="ALL">All parts (reference)</option>` + ["S1", "S2", "S3", "S4", "S6"].map(s => og(s, VV.VARIANTS.filter(v => v.stage === s))).join("") + og("S5", VV.SEARCH);
}
$$("#v3-variant, #arch-variant").forEach(s => { s.innerHTML = variantOptions(); s.addEventListener("change", () => setVariant(s.value)); });
seg("#arch-k", "k", v => { archK = +v; drawArch(); });
$("#arch-flow").addEventListener("change", drawArch);
if ($("#g-full")) $("#g-full").addEventListener("change", drawGallery);
window.addEventListener("v9live", () => { drawGallery(); archHead(); });
setVariant("ALL", true);

/* ---------------------------------------------------------------- indexing figure */
(function indexing() {
  const CH = [
    { kind: "edge", t: ["N2", "c1", "N5", "·"], lab: "edge" }, { kind: "edge", t: ["N5", "c3", "N0", "·"], lab: "edge" },
    { kind: "edge", t: ["N7", "c2", "N2", "·"], lab: "last prompt chunk · P/4−1" },
    { kind: "hop", t: ["c1", "TO", "N5", "·"], lab: "hop 1" }, { kind: "hop", t: ["c2", "TO", "N3", "·"], lab: "hop 2" }, { kind: "done", t: ["DONE", "·", "·", "·"], lab: "terminator" }];
  let J = 2;
  const segHost = $("#idx-j");
  CH.slice(0, 5).forEach((c, j) => { const b = document.createElement("button"); b.textContent = "j = " + j; b.dataset.j = j; if (j === J) b.className = "on"; segHost.appendChild(b); });
  seg("#idx-j", "j", v => { J = +v; draw(); });
  function draw() {
    const W = 1120, H = 300, host = $("#fig-index"), svg = svgRoot(host, W, H);
    marker(svg, "iC", "var(--con)"); marker(svg, "iK", "var(--code)");
    const x0 = 70, cw = 172, tw = 38;
    txt(svg, 18, 222, "…", { size: 18, cls: "t3" });
    CH.forEach((c, j) => {
      const x = x0 + j * cw, prompt = j < 3, active = j === J, target = j === J + 1;
      el("rect", { x: x - 4, y: 182, width: cw - 12, height: 74, rx: 10, style: `fill:${active ? "var(--code-s)" : target ? "var(--dec-s)" : "transparent"};stroke:${active ? "var(--code)" : target ? "var(--dec)" : "var(--line)"};stroke-dasharray:${prompt ? "0" : "4 3"}` }, svg);
      c.t.forEach((tk, i) => {
        const tx = x + 4 + i * tw;
        el("rect", { x: tx, y: 200, width: tw - 6, height: 28, rx: 6, style: `fill:var(--panel);stroke:${prompt ? "var(--enc)" : "var(--dec)"}` }, svg);
        txt(svg, tx + (tw - 6) / 2, 214, tk, { size: 11, cls: "t-mono", weight: 600 });
        const pos = 4 * j + i;
        txt(svg, tx + (tw - 6) / 2, 240, pos === 4 * J + 3 ? "4j+3" : String(pos), { size: 9.5, cls: pos === 4 * J + 3 ? "t-mono" : "t3 t-mono", color: pos === 4 * J + 3 ? "var(--code)" : null, weight: pos === 4 * J + 3 ? 700 : 500 });
      });
      txt(svg, x + cw / 2 - 8, 272, c.lab, { size: 11, cls: "t2", weight: active ? 700 : 500 });
      // concept
      const cx = x + cw / 2 - 8;
      el("circle", { cx, cy: 128, r: 15, style: `fill:${active ? "var(--code)" : "var(--con-s)"};stroke:var(--con);stroke-width:1.5` }, svg);
      txt(svg, cx, 129, "C" + j, { size: 11, weight: 700, color: active ? "var(--panel)" : null });
      wire(svg, [[cx, 182], [cx, 145]], "w-con", "iC");
    });
    // plan arrow from concept J to chunk J+1
    if (J + 1 < CH.length) {
      const sx = x0 + J * cw + cw / 2 - 8, ex = x0 + (J + 1) * cw + 4 + 0 * tw + 16, px = x0 + J * cw + 4 + 3 * tw + 16;
      el("path", { d: `M${sx},112 C${sx},62 ${px},62 ${px},196`, class: "wire w-code", "marker-end": "url(#iK)", style: "stroke-width:2.4" }, svg);
      txt(svg, px + 30, 172, "plan exposed here → predicts the next chunk", { size: 10.5, anchor: "start", color: "var(--code)", weight: 600 });
      const msg = J < 2 ? `The plan from chunk ${J} steers chunk ${J + 1}, another prompt edge. Prompt chunks have no scored NCP pair, so the code-free plan is used there.`
        : `The code chosen at chunk ${J} first reaches logit position 4j+3 = ${4 * J + 3} and steers chunk ${J + 1} (${CH[J + 1].lab}). ` +
          (J === 2 ? "So the last prompt chunk's code is the action for hop 1." : J === 3 ? "So hop 1's chunk carries the action for hop 2." : "So hop 2's chunk carries the decision to stop.");
      txt(svg, W / 2, 30, msg, { size: 13, weight: 600 });
    }
    txt(svg, 34, 128, "concepts", { size: 11, cls: "t3", anchor: "middle" });
    txt(svg, 34, 214, "tokens", { size: 11, cls: "t3", anchor: "middle" });
  }
  draw();
})();

/* ================================================================ STAGES TAB */
const ROWS = [["Encoder + embedding", ["emb", "enc"]], ["Decoder + head", ["dec", "head"]], ["Codebook Q", ["Q"]], ["Concept core", ["core"]], ["Hypothesis block + codes e_k", ["codes", "bhk"]],
  ["PQ heads W_s", ["pq"]], ["Selector π", ["sel"]], ["Value V_ψ + terminal τ_ψ", ["vtau"]], ["LoRA adapters + GMM gates", ["lora"]]];
const COLS = [["S1", "S1", "A_delayed"], ["S2", "S2", "S2_K8_dense"], ["S3a", "S3", "S3a_dec"], ["S3b", "S3", "S3b"], ["S4", "S4", "S4_EI"], ["S5", "S5", "S4_EI", true], ["S6", "S6", "S6_LSL_scaled"]];
let curStage = "S2";
function cellState(parts, col) {          // derived from the variant table, so the matrix and the architecture view cannot disagree
  const m = VV.byId(col[2]).train, st = parts.map(p => m[p] || "N").map(x => (col[3] && x === "T" ? "F" : x));
  return st.every(x => x === "N") ? "N" : st.includes("T") ? "T" : "F";
}
function drawMatrix() {
  const t = $("#matrix");
  t.innerHTML = `<thead><tr><th></th>${COLS.map(c => `<th data-s="${c[1]}" class="${c[1] === curStage ? "on" : ""}">${c[0]}</th>`).join("")}</tr></thead><tbody>${ROWS.map(r => `<tr><th>${r[0]}</th>${COLS.map((c, i) => { const v = cellState(r[1], c); return `<td class="${v}${c[1] === curStage ? " on" : ""}">${v === "T" ? "trains" : v === "F" ? "frozen" : "—"}</td>`; }).join("")}</tr>`).join("")}</tbody>`;
  t.querySelectorAll("thead th[data-s]").forEach(th => th.addEventListener("click", () => openStage(th.dataset.s)));
}
function armRow(a) {
  const v = a[3] ? VV.byId(a[3]) : null, tier = v ? v.tier : (a[4] ? "mandatory" : "full"), lv = v ? liveFor(v) : null;
  const live = lv ? (lv.every(x => x.r.completed) ? `<span class="tag-done">✓ trained</span>` : `<span class="tag-running">● training</span>`) : "";
  const link = v ? ` <button class="linkbtn" data-open="${v.id}">architecture ›</button>` : "";
  return `<tr class="${tier === "mandatory" ? "must" : ""}"><td>${a[0]}${link}</td><td>${a[1]}</td><td>${a[2]}</td><td class="tabular">${tier === "full" ? "full column" : "mandatory"} ${live}</td></tr>`;
}
function openStage(id) {
  curStage = id; drawMatrix();
  $$("#stage-tabs button").forEach(b => b.classList.toggle("on", b.dataset.s === id));
  const s = STAGES.find(x => x.id === id);
  $("#stage-detail").innerHTML = `
    <div class="panel pad"><div class="eyebrow">${s.id} · draft_v9.md ${s.sec} · planned: ${s.cost}</div><h2 style="font-size:24px">${s.name}</h2><p style="color:var(--ink2)">${s.what}</p><pre class="eq">${s.obj}</pre>
      ${id === "S6" ? `<p class="muted" style="font-size:12.5px">S6 column shows the LSL arms. The VQ/selector-only arm trains Q, the PQ heads and the selector instead.</p>` : ""}
      ${id === "S3" ? `<p class="muted" style="font-size:12.5px">The matrix shows S3a (frozen core) and S3b (joint) as separate columns.</p>` : ""}</div>
    <div style="display:grid;gap:14px;align-content:start;grid-template-columns:minmax(0,1fr)">
      <div class="panel pad"><h3>Gates</h3><div class="gate-list">${s.gateText.map(g => `<div class="gate"><b>${g[0]}</b> ${(window.V9L && window.V9L.gatePill && window.V9L.gatePill(s.id, g[0])) || '<span class="pill untested">untested</span>'}<p>${g[1]}</p></div>`).join("")}</div></div>
      <div class="panel pad tbl-wrap"><h3>Arms</h3><table class="data"><thead><tr><th>Arm</th><th>Seeds</th><th>Role</th><th>Queue</th></tr></thead><tbody>${s.arms.map(a => armRow(a)).join("")}</tbody></table></div>
    </div>`;
  $$("#stage-detail [data-open]").forEach(b => b.addEventListener("click", () => { setVariant(b.dataset.open); showTab("arch"); scrollTo({ top: 0 }); }));
}
(function stageTabs() {
  const h = $("#stage-tabs");
  STAGES.forEach(s => { const b = document.createElement("button"); b.dataset.s = s.id; b.textContent = `${s.id} · ${s.name}`; b.addEventListener("click", () => openStage(s.id)); h.appendChild(b); });
  openStage("S2");
})();

/* ================================================================ SEARCH TAB */
(function search() {
  const B = 3, DEPTH = 4, GOAL = [0, 2, 1, 2];
  const COST_CHILD = 5, COST_GROUND = 24, COST_TOKEN = 28;
  let R = 2, nodes, cost, grounds, decodedFinal, rng;
  function prior(path) { const r = mulberry(path.reduce((a, b) => a * 7 + b + 3, 11)); const p = [r() + .2, r() + .2, r() + .2]; const d = path.length < DEPTH ? GOAL[path.length] : 0; p[d] += .5; const s = p[0] + p[1] + p[2]; return p.map(x => x / s); }
  function trueValue(path) { let m = 0; while (m < path.length && path[m] === GOAL[m]) m++; return m === path.length ? .2 + .8 * m / DEPTH : .15 * m / DEPTH; }
  function mk(parent, a) {
    const path = parent ? parent.path.concat([a]) : [];
    const nd = { id: nodes.length, parent, a, path, depth: path.length, kids: [], N: 0, W: 0, P: prior(path), grounded: false, decoded: false };
    nodes.push(nd); if (parent) parent.kids[a] = nd; return nd;
  }
  function reset() { nodes = []; cost = 0; grounds = 0; decodedFinal = false; rng = mulberry(5); mk(null, null); draw(); }
  function expandOnce() {
    let s = nodes[0];
    while (s.depth < DEPTH) {
      const tried = s.kids.filter(Boolean).length;
      if (tried < B) {
        const order = [0, 1, 2].filter(a => !s.kids[a]).sort((a, b) => s.P[b] - s.P[a]);
        const ch = mk(s, order[0]);
        cost += COST_CHILD;
        if (R > 0 && ch.depth % R === 0) { ch.grounded = true; cost += COST_GROUND; grounds++; }
        let v = trueValue(ch.path) + .08 * (rng() - .5);
        for (let q = ch; q; q = q.parent) { q.N++; q.W += v; }
        return;
      }
      let best = null, bs = -1e9;
      s.kids.forEach((k, a) => { const q = k.N ? k.W / k.N : 0; const u = q + 1.5 * s.P[a] * Math.sqrt(s.N) / (1 + k.N); if (u > bs) { bs = u; best = k; } });
      s = best;
    }
    // full-depth leaf revisited: re-evaluate
    const v = trueValue(s.path); for (let q = s; q; q = q.parent) { q.N++; q.W += v; }
  }
  function bestPath() { const p = []; let s = nodes[0]; while (s.kids.filter(Boolean).length) { s = s.kids.filter(Boolean).sort((a, b) => b.N - a.N)[0]; p.push(s); } return p; }
  function draw() {
    const W = 980, H = 470, host = $("#fig-tree"), svg = svgRoot(host, W, H);
    const leafW = (W - 60) / Math.pow(B, DEPTH);
    const pos = nd => { let idx = 0; nd.path.forEach((a, i) => { idx += a * Math.pow(B, DEPTH - 1 - i); }); const span = Math.pow(B, DEPTH - nd.depth); return { x: 30 + (idx + span / 2) * leafW, y: 40 + nd.depth * 100 }; };
    const bp = bestPath(), onBest = new Set(bp.map(n => n.id));
    // levels
    for (let d = 0; d <= DEPTH; d++) {
      const y = 40 + d * 100, grounded = R > 0 && d > 0 && d % R === 0;
      if (grounded) el("rect", { x: 10, y: y - 16, width: W - 20, height: 32, rx: 8, style: "fill:var(--enc-s);opacity:.6" }, svg);
      txt(svg, W - 14, y - 22, d === 0 ? "root · prompt encoded once" : `hop ${d}${grounded ? " · re-grounded" : ""}`, { size: 10.5, anchor: "end", cls: "t3" });
    }
    nodes.slice(1).forEach(nd => { const a = pos(nd.parent), b = pos(nd); el("path", { d: `M${a.x},${a.y} C${a.x},${(a.y + b.y) / 2} ${b.x},${(a.y + b.y) / 2} ${b.x},${b.y}`, style: `fill:none;stroke:${onBest.has(nd.id) ? "var(--code)" : "var(--line2)"};stroke-width:${onBest.has(nd.id) ? 2.6 : 1.2}` }, svg); });
    nodes.forEach(nd => {
      const p = pos(nd), r = nd.depth === 0 ? 10 : Math.min(11, 4 + Math.sqrt(nd.N) * 1.6);
      const q = nd.N ? nd.W / nd.N : 0;
      const tipTxt = nd.depth === 0 ? "Root: prompt encoded once, core caches built" : `code path ${nd.path.map(a => "k" + (a + 1)).join(" → ")}<br>visits ${nd.N} · mean value ${q.toFixed(2)} · prior ${(nd.parent.P[nd.a] * 100).toFixed(0)}%<br>${nd.grounded ? "re-grounded: decoded 4 tokens, re-encoded" : "decoder-free: code → ĉ⁽ᵏ⁾ → core step"}`;
      let shape;
      if (nd.grounded || nd.depth === 0) shape = el("rect", { x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r, rx: 3, style: `fill:${nd.depth === 0 ? "var(--ink)" : "var(--enc)"};stroke:${onBest.has(nd.id) ? "var(--code)" : "var(--panel)"};stroke-width:2` }, svg);
      else shape = el("circle", { cx: p.x, cy: p.y, r, style: `fill:var(--con);stroke:${onBest.has(nd.id) ? "var(--code)" : "var(--panel)"};stroke-width:2` }, svg);
      shape.setAttribute("data-tip", tipTxt);
      if (nd.decoded) el("circle", { cx: p.x, cy: p.y, r: r + 4, style: "fill:none;stroke:var(--dec);stroke-width:2" }, svg);
    });
    if (nodes.length === 1) txt(svg, W / 2, 230, "Press “Run 30” to expand the tree with PUCT", { size: 14, cls: "t3" });
    // meter
    const tokNodes = Math.floor(cost / COST_TOKEN);
    const leaf = bp.length === DEPTH ? bp[DEPTH - 1] : null;
    const ok = leaf && trueValue(leaf.path) > .9;
    $("#s-meter").innerHTML = `<div><b>${nodes.length - 1}</b><span>concept children expanded</span></div><div><b>${cost}</b><span>block-positions spent (${grounds} re-groundings)</span></div><div><b>${tokNodes}</b><span>token children for the same spend</span></div><div><b style="color:${leaf ? (ok ? "var(--good)" : "var(--bad)") : "var(--ink3)"}">${leaf ? (ok ? "goal path" : "wrong path") : "—"}</b><span>most-visited path${R === 0 ? (decodedFinal ? " (decoded once at end)" : "; at R = ∞ it is decoded once at the end") : ""}</span></div>`;
  }
  function finish() { if (R === 0 && !decodedFinal) { const bp = bestPath(); if (bp.length === DEPTH) { bp.forEach(n => n.decoded = true); cost += COST_GROUND * DEPTH; decodedFinal = true; } } }
  seg("#s-r", "r", v => { R = +v; reset(); });
  $("#s-step").addEventListener("click", () => { expandOnce(); draw(); });
  $("#s-run").addEventListener("click", () => { for (let i = 0; i < 30; i++) expandOnce(); finish(); draw(); });
  $("#s-reset").addEventListener("click", reset);
  reset();

  // cost bars
  function costFig() {
    const svg = svgRoot($("#fig-cost"), 320, 150);
    const rows = [["Token child", 28, "--dec", "4 tokens × (E3+D3) + core"], ["Grounded concept child", 29, "--enc", "R = 1: + B_H"], ["Decoder-free child", 5, "--con", "B_H + core step"]];
    rows.forEach((r, i) => {
      const y = 12 + i * 46, w = r[1] / 30 * 200;
      txt(svg, 0, y + 6, r[0], { size: 11.5, anchor: "start", weight: 600 });
      el("rect", { x: 0, y: y + 14, width: w, height: 16, rx: 4, style: `fill:var(${r[2]})` }, svg);
      txt(svg, w + 6, y + 22, String(r[1]), { size: 12, anchor: "start", weight: 700, cls: "t-mono" });
      txt(svg, w + 30, y + 22, r[3], { size: 10, anchor: "start", cls: "t3" });
    });
  }
  costFig();
  function ratio() {
    const E = +$("#c-e").value, D = +$("#c-d").value, C = +$("#c-c").value;
    $("#c-e-v").textContent = E; $("#c-d-v").textContent = D; $("#c-c-v").textContent = C;
    $("#c-ratio").textContent = ((4 * (E + D) + C) / (C + 1)).toFixed(1) + "×";
  }
  ["#c-e", "#c-d", "#c-c"].forEach(s => $(s).addEventListener("input", ratio)); ratio();

  // loop figure
  (function loopFig() {
    const svg = svgRoot($("#fig-loop"), 620, 400);
    marker(svg, "lA", "var(--ink3)"); marker(svg, "lE", "var(--enc)"); marker(svg, "lD", "var(--dec)");
    const B2 = (x, y, w, h, k, a, b) => { const g = el("g", { class: "blk k-" + k }, svg); el("rect", { x, y, width: w, height: h, rx: 9 }, g); txt(g, x + w / 2, y + (b ? h / 2 - 7 : h / 2), a, { size: 12, weight: 650 }); if (b) txt(g, x + w / 2, y + h / 2 + 9, b, { size: 10, cls: "t2" }); };
    B2(20, 20, 180, 46, "io", "Root", "prompt encoded once");
    B2(230, 20, 180, 46, "sel", "PUCT select", "π prior, Q, N · c = 1.5");
    B2(440, 20, 160, 46, "code", "Expand code k", "ĉ⁽ᵏ⁾ → core step");
    B2(440, 120, 160, 46, "con", "V_ψ, τ_ψ on u⁽⁰⁾", "value + terminal");
    B2(230, 120, 180, 46, "io", "Backup", "mean value");
    B2(230, 220, 180, 50, "enc", "Re-ground every R", "decode 4 tok → re-encode");
    B2(20, 220, 180, 50, "dec", "Decode chosen path", "→ verifier");
    B2(20, 320, 380, 50, "sel", "AB-MCTS-A outer loop", "GEN: new search  |  CONT: GradCuit or random-kernel refine");
    wire(svg, [[200, 43], [228, 43]], "w-ink", "lA"); wire(svg, [[410, 43], [438, 43]], "w-ink", "lA");
    wire(svg, [[520, 66], [520, 118]], "w-ink", "lA"); wire(svg, [[440, 143], [412, 143]], "w-ink", "lA");
    wire(svg, [[320, 120], [320, 68]], "w-ink", "lA");
    wire(svg, [[560, 66], [580, 66], [580, 245], [412, 245]], "w-enc", "lE"); txt(svg, 585, 200, "every R", { size: 10.5, anchor: "start", color: "var(--enc)" });
    wire(svg, [[230, 260], [215, 260], [215, 100], [300, 100], [300, 68]], "w-enc", null);
    wire(svg, [[230, 143], [110, 143], [110, 218]], "w-dec", "lD"); txt(svg, 116, 170, "budget spent or DONE", { size: 10.5, anchor: "start", color: "var(--dec)" });
    wire(svg, [[110, 270], [110, 318]], "w-ink", "lA"); txt(svg, 116, 296, "fail + budget left", { size: 10.5, anchor: "start", cls: "t3" });
    wire(svg, [[400, 345], [600, 345], [600, 4], [320, 4], [320, 18]], "w-ink", "lA");
  })();
})();

/* ================================================================ CONTINUAL TAB */
(function lsl() {
  let sc = "sep";
  const OLD = [{ name: "navigation", mx: 300, my: 170, sx: 42, sy: 30, col: "--enc" }, { name: "text", mx: 140, my: 260, sx: 55, sy: 40, col: "--tok" }];
  const NEW = { sep: { name: "colour-count", mx: 470, my: 80, sx: 32, sy: 26 }, bind: { name: "bind", mx: 322, my: 160, sx: 40, sy: 30 } };
  const N2 = (x, y, g) => Math.exp(-.5 * (((x - g.mx) / g.sx) ** 2 + ((y - g.my) / g.sy) ** 2)) / (g.sx * g.sy);
  const NEG = { comps: [{ mx: 300, my: 170, sx: 70, sy: 55, w: .45 }, { mx: 140, my: 260, sx: 80, sy: 60, w: .45 }, { mx: 330, my: 260, sx: 140, sy: 110, w: .10 }] };
  const negD = (x, y) => NEG.comps.reduce((a, c) => a + c.w * N2(x, y, c), 0);
  function points(seed, g, n) { const r = mulberry(seed), out = []; for (let i = 0; i < n; i++) { let u = 0, v = 0; while (!u) u = r(); v = r(); const z1 = Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283 * v), z2 = Math.sqrt(-2 * Math.log(u)) * Math.sin(6.283 * v); out.push([g.mx + g.sx * z1, g.my + g.sy * z2]); } return out; }
  function draw() {
    const W = 600, H = 340, svg = svgRoot($("#fig-lsl"), W, H);
    const nw = NEW[sc], posG = { mx: nw.mx, my: nw.my, sx: nw.sx * 1.2, sy: nw.sy * 1.2 };
    const open = (x, y) => .9 * N2(x, y, posG) > negD(x, y);
    const cs = 10;
    for (let x = 0; x < W; x += cs) for (let y = 0; y < H; y += cs) if (open(x + cs / 2, y + cs / 2)) el("rect", { x, y, width: cs, height: cs, style: "fill:var(--code);opacity:.17" }, svg);
    const oldPts = [];
    OLD.forEach((g, i) => { points(11 + i, g, 140).forEach(p => { oldPts.push(p); el("circle", { cx: p[0], cy: p[1], r: 2.2, style: `fill:var(${g.col});opacity:.75` }, svg); }); txt(svg, g.mx, g.my + g.sy * 2.3, g.name + " (old)", { size: 11.5, weight: 650, color: `var(${g.col})` }); });
    points(31, nw, 120).forEach(p => el("rect", { x: p[0] - 2, y: p[1] - 2, width: 4, height: 4, style: "fill:var(--loss);opacity:.8" }, svg));
    el("ellipse", { cx: posG.mx, cy: posG.my, rx: posG.sx * 2, ry: posG.sy * 2, style: "fill:none;stroke:var(--loss);stroke-width:1.6;stroke-dasharray:5 4" }, svg);
    txt(svg, nw.mx + (sc === "sep" ? 0 : 92), nw.my - nw.sy * 2.4 - (sc === "sep" ? 0 : 6), "Φpos · " + nw.name + " (new)", { size: 11.5, weight: 650, color: "var(--loss)" });
    const hits = oldPts.filter(p => open(p[0], p[1])).length, rate = hits / oldPts.length;
    const navHits = oldPts.slice(0, 140).filter(p => open(p[0], p[1])).length / 140;
    const pass = rate <= .05 && navHits <= .15;
    $("#lsl-insp").innerHTML = `<div class="cat"><i style="background:var(--loss)"></i>Gate G6.0 · illustration</div><h3>${sc === "sep" ? "P2 colour-count" : "P3 bind"}</h3>
      <div class="meter" style="grid-template-columns:1fr 1fr"><div><b>${(100 * rate).toFixed(1)}%</b><span>gate hit rate, all old-task points</span></div><div><b>${(100 * navHits).toFixed(1)}%</b><span>on navigation alone</span></div></div>
      <p style="margin-top:10px"><span class="pill ${pass ? "gate" : "kill"}">${pass ? "separable · passes G6.0" : "non-separable · fails G6.0"}</span></p>
      <p>${sc === "sep" ? "The new grammar's inputs occupy their own region, so the adapter almost never fires on old tasks. This is the case LSL's claim is about, and P2 is its positive test." : "Bind uses the same edge-list serialization as navigation. The gate opens over navigation inputs, and by LSL's own theory the newest update wins there. Predicted outcome: LSL forgets about as much as always-open (within ±5 points)."}</p>
      <pre>G6.0: mean smoothed hit rate ≤ 5%
      and no foundation suite > 15%</pre>
      <p class="muted" style="font-size:12.5px">In V8, gate rates were measured only on current-phase data (drive_12h.py:148). V9 records them on old-task inputs during and after every phase.</p><div class="src">draft_v9.md §10.3–§10.4</div>`;
  }
  seg("#lsl-sc", "sc", v => { sc = v; draw(); });
  draw();
})();

/* ================================================================ EVIDENCE TAB */
(function evidence() {
  const L = [["L1", "S5 at R = ∞ beats token tree search at some budget on hard14", 25], ["L2", "Selector-argmax legality at N = 5 ≥ 90% on standard", 35], ["L3", "Expert iteration beats token RFT at equal verifier calls (hard10)", 35],
    ["L4", "LSL-scaled beats always-open on P2 retention, install ≥ VQ-only", 40], ["L5", "Hindsight-code graph NLL ≤ K = 1 − .05 nats", 55]];
  const Hh = [["H1", "K = 1 contract test passes", 95], ["H2", "Dead-code restart keeps ≥ K/2 live codes", 80], ["H3", "Hierarchy > flat on sealed standard, ≥ 2 seeds", 70], ["H4", "Flat keeps lower text NLL by ≥ .02 (E1 persists)", 75],
    ["H5", "Delayed NCP > α0 on ≥ 2 seeds", 50], ["H6", "P-blur-1: K = 4 avoids the V6 illegal-transition failure, K = 1 reproduces it", 50], ["H7", "P-blur-2: illegal rate grows with progress out-edges under K = 1", 50],
    ["H8", "G2.2 passes at m = 2, fails at m = K", 60], ["H9", "Closed loop cuts N = 3 drift by ≥ 30%", 55], ["H10", "Recurrent core ≈ dense core on every S2 gate", 65], ["H11", "S3a keeps authority; S3b loses ≥ 20%", 50],
    ["H12", "Bind confirms the negative control (±5 points)", 65], ["H13", "VQ-only retains best and installs least", 60], ["H14", "GradCuit kernel ≈ random-direction kernel", 60], ["H15", "NCP − α0 text gap shrinks from 24k to 96k updates", 45]];
  const row = (r, low) => `<div class="conf-row"><span class="id" style="color:var(${low ? "--loss" : "--ink3"})">${r[0]}</span><span>${r[1]}</span><span class="conf-wrap"><span class="bar"><i style="width:${r[2]}%;background:var(${low ? "--loss" : "--con"})"></i></span><span class="pc">~${r[2]}%</span></span></div>`;
  $("#conf").innerHTML = L.map(r => row(r, true)).join("") + `<div style="height:10px"></div>` + Hh.map(r => row(r, false)).join("");
  const K = [["Either G2.1 sub-gate fails on both S2 seeds", "Concept branches of S4 and S5 stop; token methods only."], ["G2.3(b) legality < 80% at N = 1", "S5 runs at R = 1 only."], ["Value AUROC < .70 after iteration 0", "S5 uses verifier leaves throughout."],
    ["G6.0 fails for both P2 candidates", "LSL positive test is P1 only."], ["Both new S1 seeds fail competence", "V9 stops at S1 and reports."]];
  $("#kills").innerHTML = K.map(k => `<div class="gate"><span class="pill kill">if</span> <b>${k[0]}</b><p>→ ${k[1]}</p></div>`).join("");
  const Q = [["Q0", "Contracts, sealed manifests, pilots", 1.5, 1.5, "--ink3"], ["Q1", "S1 seeds (3 arms × 2 seeds)", 12, 12, "--enc"], ["Q2", "S6-base on the S1 anchor", 6, 6, "--loss"], ["Q3", "S2 H-NCP, Tier A + B, all gates", 12.2, 12.2, "--con"], ["Q4", "S3a-dec / S3b + G3", 3, 3, "--dec"],
    ["Q5", "S4 expert iteration + baselines", 20, 20, "--sel"], ["Q6", "S5 search evaluation", 3.1, 38, "--code"]];
  let mode = "must";
  function gantt() {
    const W = 640, H = 262, svg = svgRoot($("#fig-queue"), W, H), x0 = 190, tot = mode === "must" ? 58 : 93, sc = (W - x0 - 20) / tot;
    let t = 0;
    for (let h = 0; h <= tot; h += 10) { const x = x0 + h * sc; el("line", { x1: x, y1: 14, x2: x, y2: H - 22, style: "stroke:var(--line)" }, svg); txt(svg, x, H - 10, h + " h", { size: 10, cls: "t3 t-mono" }); }
    Q.forEach((q, i) => {
      const d = mode === "must" ? q[2] : q[3], y = 18 + i * 30;
      txt(svg, 0, y + 11, q[0], { size: 11.5, anchor: "start", weight: 700, cls: "t-mono" });
      txt(svg, 30, y + 11, q[1], { size: 11, anchor: "start", cls: "t2" });
      el("rect", { x: x0 + t * sc, y, width: Math.max(2, d * sc), height: 22, rx: 5, style: `fill:var(${q[4]})`, "data-tip": `${q[0]} · ${q[1]} · ${d} h` }, svg);
      txt(svg, x0 + (t + d) * sc + 5, y + 11, d + " h", { size: 10.5, anchor: "start", cls: "t-mono t2" });
      t += d;
    });
  }
  seg("#q-mode", "m", v => { mode = v; gantt(); }); gantt();
  const E = [["E1", "A matched flat Transformer is better on text at small scale: ~.02–.07 nats (V7, 3 seeds), ~.05 (V8, 1 seed).", "Moderate"],
    ["E2", "Hierarchy beats flat on GraphNav (V8 sealed): delayed NCP 80.4%, α0 76.5%, flat 73.7%. NCP vs α0 rests on one seed.", "Moderate / weak"],
    ["E3", "NCP's sign reversed between recipes: V5 NCP .893 vs α0 .195; V6 α0 79.7% vs NCP 0% at 9k steps.", "Strong (recipe-dependent)"],
    ["E4", "Every V6 NCP-prefix failure was an illegal transition: correct format, wrong move.", "Moderate"],
    ["E5", "NCP makes open-loop self-feedback far less damaging: +.048 graph / +.217 text vs α0 +.891 / +.599.", "Strong"],
    ["E6", "Z has authority over a frozen decoder: −.11 nats graph; Z-swap +.498.", "Strong"],
    ["E7", "A variational prior alone was confidently wrong (+.345); explorative modelling made it deployable (+.025).", "Strong (1 seed)"],
    ["E8", "Latent steering lost to a simpler search every time it had a comparator (4/4).", "Strong as a pattern"],
    ["E9", "Internal RL over Z did not beat the prior mean (V5 D); ETA-style switching saturated.", "Moderate"],
    ["E10", "Recurrence never became a hop-per-pass mechanism; V8 recurrent refit ≈ dense (80.9% vs 80.8%).", "Strong at ≤ 25M"],
    ["E11", "V8 LSL installed bind (+25.2 points) and destroyed navigation (0/2048); mechanism unmeasured.", "Strong that it happened"],
    ["E12", "Size and length generalization failed for every arm: hard14 and long18 ≈ 0.", "Strong"],
    ["E13", "Almost no run used more than one hierarchy training seed.", "Process"]];
  $("#evlist").innerHTML = E.map(e => `<div class="ev"><span class="id">${e[0]}</span><span>${e[1]}</span><span class="str" style="color:var(${/Strong/.test(e[2]) ? "--good" : /weak|Moderate/.test(e[2]) ? "--warn" : "--ink3"})">${e[2]}</span></div>`).join("");
})();

/* ---------------------------------------------------------------- initial tab */
let first = location.hash.slice(1);
if (!TABS.includes(first)) { try { first = localStorage.getItem("v9-tab") || "overview"; } catch (e) { first = "overview"; } }
showTab(first);
window.addEventListener("v9live", () => openStage(curStage));
window.V9 = { showTab, INFO, setVariant, getVariant: () => curV, pinfo, stt, Kof };
})();
