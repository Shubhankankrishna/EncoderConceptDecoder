/* Concept module v2 page: five proposed designs, scores vs today's core, 2D diagram per design.
   Data: data/concept_v2.json (built from concept_v2/scores.json, budgets.json, DESIGNS_FINAL.md). */
(function () {
  let D = null, sel = "D3";
  const ORDER = ["ROW0", "D1", "D2", "D3", "D4", "D5", "D6"];
  const COL = { ROW0: "#8a93a3", D1: "#3b78c2", D2: "#e07b39", D3: "#2e9c8c", D4: "#8f63c9", D5: "#d14f7a", D6: "#3a9a4a" };
  const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const pct = x => (x * 100).toFixed(1) + " %";
  const M = x => (x / 1e6).toFixed(2) + "M";

  async function load() {
    if (D) return;
    D = await (await fetch("data/concept_v2.json?v=51")).json();
    render();
  }

  function scoreTable() {
    const C = Object.keys(D.weights);
    let h = `<table class="cv2-table"><thead><tr><th>Design</th>${C.map(c => `<th title="${esc(D.names[c])}">${c}<span>${D.weights[c]}</span></th>`).join("")}<th>Total /100</th></tr></thead><tbody>`;
    for (const k of ORDER) {
      const d = D.designs[k];
      h += `<tr class="${k === sel ? "on" : ""}" data-k="${k}"><td><span class="cv2-dot" style="background:${COL[k]}"></span>${k === "ROW0" ? esc(d.title) : k + " · " + esc(d.title)}</td>`;
      for (const c of C) {
        const v = d.scores[c];
        h += `<td class="cv2-cell" title="${esc(d.why[c])}"><div class="cv2-bar" style="width:${v * 10}%;background:${COL[k]}"></div><b>${v}</b></td>`;
      }
      h += `<td class="cv2-tot"><div class="cv2-bar" style="width:${d.total}%;background:${COL[k]}"></div><b>${d.total.toFixed(1)}</b></td></tr>`;
    }
    return h + "</tbody></table>";
  }

  /* ---------- 2D diagram: encoder -> looped core -> decoder, with the design's mechanism drawn */
  function box(x, y, w, h, fill, label, sub) {
    return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="${fill}" stroke="rgba(0,0,0,.18)"/>` +
      `<text x="${x + w / 2}" y="${y + h / 2 - (sub ? 4 : -4)}" text-anchor="middle" class="cv2-bt">${label}</text>` +
      (sub ? `<text x="${x + w / 2}" y="${y + h / 2 + 12}" text-anchor="middle" class="cv2-bs">${sub}</text>` : "");
  }
  const arrow = (x1, y1, x2, y2, c = "currentColor", dash = "") =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" stroke-width="2" ${dash ? `stroke-dasharray="${dash}"` : ""} marker-end="url(#cv2a)"/>`;

  function diagram(k) {
    const c = COL[k], W = 860, H = 300;
    let s = `<svg viewBox="0 0 ${W} ${H}" class="cv2-svg" role="img" aria-label="diagram of ${k}"><defs><marker id="cv2a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>`;
    s += box(10, 110, 120, 70, "#dce8f7", "Encoder", "8 layers · d 384");
    s += arrow(130, 145, 170, 145) + `<text x="150" y="135" text-anchor="middle" class="cv2-s">c_hat</text>`;
    s += box(710, 110, 140, 70, "#dce8f7", "Decoder", "4 layers · plan u");
    // core frame
    s += `<rect x="170" y="30" width="500" height="240" rx="16" fill="none" stroke="${c}" stroke-width="2" stroke-dasharray="6 5"/>`;
    s += `<text x="180" y="50" class="cv2-t" style="fill:${c}">Concept core · ${k === "ROW0" ? "1 shared layer · d 384" : "4 distinct layers · d 256 · FFN 1024 · full attention"}</text>`;
    s += arrow(670, 145, 710, 145) + `<text x="690" y="135" text-anchor="middle" class="cv2-s">u</text>`;
    const layers = k === "ROW0" ? ["LoopLayer"] : ["L1", "L2", "L3", "L4"];
    const lx0 = k === "ROW0" ? 370 : 230, lw = k === "ROW0" ? 100 : 70, gap = 20;
    layers.forEach((l, i) => {
      const x = lx0 + i * (lw + gap);
      const lab = (k === "D4" || k === "D6") ? l + " attn" : l;
      s += box(x, 115, lw, 60, k === "ROW0" ? "#d9d9de" : "#e3f2ef", lab, k === "ROW0" ? "attn + FFN" : (k === "D4" || k === "D6") ? "+ experts" : "attn+FFN");
      if (i) s += arrow(x - gap, 145, x, 145);
    });
    const lx1 = lx0 + layers.length * (lw + gap) - gap;
    // loop-back arrow (S carry)
    s += `<path d="M${lx1} 160 C ${lx1 + 40} 160, ${lx1 + 40} 215, ${(lx0 + lx1) / 2} 215 C ${lx0 - 40} 215, ${lx0 - 40} 160, ${lx0} 160" fill="none" stroke="${c}" stroke-width="2.4" marker-end="url(#cv2a)"/>`;
    const carryLab = { ROW0: "S carry · R ~ U{1..4} · unbounded", D1: "S carry (clamped) · R ~ Poisson(4) in [2,8]", D2: "passes t, Δ (AdaLN) · long L = 6", D3: "LN carry · K ~ U{8..12}", D4: "S carry · R ~ Poisson · ξ step scale", D5: "LN carry · shared passes 1..R−1", D6: "LN carry · K ~ U{8..12} · experts routed on the state" }[k];
    s += `<text x="${(lx0 + lx1) / 2}" y="235" text-anchor="middle" class="cv2-s" style="fill:${c}">${carryLab}</text>`;
    s += arrow(185, 145, lx0, 145, "#999") + `<text x="${(185 + lx0) / 2}" y="195" text-anchor="middle" class="cv2-s">X re-read</text>`;
    if (lx1 < 655) s += arrow(lx1, 145, 655, 145, "#999");
    // mechanism overlays
    if (k === "D2") {
      s += `<path d="M${lx0} 85 Q ${(lx0 + lx1) / 2} 55 ${lx1} 85" fill="none" stroke="${c}" stroke-width="2" stroke-dasharray="4 4" marker-end="url(#cv2a)"/>`;
      s += `<text x="${(lx0 + lx1) / 2}" y="72" text-anchor="middle" class="cv2-s" style="fill:${c}">shortcut S ~ U{1..5} → distilled to sg(long endpoint)</text>`;
    }
    if (k === "D3") {
      [0.33, 0.66].forEach(f => { const x = lx0 - 40 + f * (lx1 - lx0 + 80); s += `<line x1="${x}" y1="200" x2="${x}" y2="228" stroke="${c}" stroke-width="3"/><text x="${x}" y="262" text-anchor="middle" class="cv2-s">detach</text>`; });
      s += `<text x="${(lx0 + lx1) / 2}" y="92" text-anchor="middle" class="cv2-s" style="fill:${c}">windows of 4 passes · loss on every pass (true next-chunk code)</text>`;
    }
    if (k === "D4") {
      for (let e = 0; e < 8; e++) s += `<rect x="${lx0 + e * 44}" y="62" width="36" height="18" rx="4" fill="#efe6fa" stroke="${c}"/><text x="${lx0 + e * 44 + 18}" y="75" text-anchor="middle" class="cv2-s">E${e + 1}</text>`;
      s += `<text x="${(lx0 + lx1) / 2}" y="98" text-anchor="middle" class="cv2-s" style="fill:${c}">shared expert pool · top-2 routing differs per pass · Delta-Res over past passes</text>`;
    }
    if (k === "D5") {
      for (let j = 0; j < 4; j++) s += `<line x1="${lx1}" y1="145" x2="${lx1 + 30}" y2="${95 + j * 33}" stroke="${c}" stroke-width="1.8"/><circle cx="${lx1 + 34}" cy="${95 + j * 33}" r="5" fill="${c}"/>`;
      s += `<text x="${lx1 + 5}" y="80" class="cv2-s" style="fill:${c}">K = 4 plans</text><text x="${lx1 - 10}" y="210" class="cv2-s" style="fill:${c}">selector</text>`;
    }
    if (k === "D6") {
      for (let e = 0; e < 8; e++) s += `<rect x="${lx0 + e * 44}" y="62" width="36" height="18" rx="4" fill="#e2f3e4" stroke="${c}"/><text x="${lx0 + e * 44 + 18}" y="75" text-anchor="middle" class="cv2-bs">E${e + 1}</text>`;
      s += `<text x="${(lx0 + lx1) / 2}" y="98" text-anchor="middle" class="cv2-s" style="fill:${c}">shared expert pool · routed on the state · windows of 4 passes, every pass graded</text>`;
      [0.33, 0.66].forEach(f => { const x = lx0 - 40 + f * (lx1 - lx0 + 80); s += `<line x1="${x}" y1="200" x2="${x}" y2="228" stroke="${c}" stroke-width="3"/><text x="${x}" y="262" text-anchor="middle" class="cv2-s">detach</text>`; });
      const bx = lx0 - 40 + 0.66 * (lx1 - lx0 + 80);
      s += `<path d="M${bx + 30} 272 L${bx + 30} 280 L${lx1 + 60} 280 L${lx1 + 60} 272" fill="none" stroke="${c}" stroke-width="2"/><text x="${(bx + lx1 + 90) / 2}" y="295" text-anchor="middle" class="cv2-s" style="fill:${c}">last window: consistency → endpoint (LoopFormer)</text>`;
    }
    if (k === "D1") s += `<text x="${(lx0 + lx1) / 2}" y="92" text-anchor="middle" class="cv2-s" style="fill:${c}">identity at init · loss at the final pass only</text>`;
    if (k === "ROW0") s += `<text x="420" y="92" text-anchor="middle" class="cv2-s" style="fill:${c}">measured: R1..R8 give the same output (step angle 6.7° → 0.1°)</text>`;
    return s + "</svg>";
  }

  function detail() {
    const d = D.designs[sel], b = d.budget, c = COL[sel];
    const C = Object.keys(D.weights);
    return `<div class="cv2-detail" style="--c:${c}">
      <div class="cv2-dh"><h3>${sel === "ROW0" ? esc(d.title) : sel + " · " + esc(d.title)}</h3><span class="cv2-mech">C2 mechanism: ${esc(d.mechanism)}</span>
        <span class="cv2-total">${d.total.toFixed(1)}<small>/100</small></span></div>
      ${diagram(sel)}
      ${d.vs_row0 ? `<p class="cv2-vs"><b>Why it should beat today's module:</b> ${esc(d.vs_row0)}</p>` : ""}
      ${d.vs_others ? `<p class="cv2-vs"><b>Why it should beat D2, D3 and D4:</b> ${esc(d.vs_others)}</p>` : ""}
      <div class="cv2-chips"><span>params ${M(b.params)}</span><span>inference R ${b.eval_R}: ${M(b.eval)} MACs/token (${pct(b.share)} of model)</span><span>training ×${b.train_x.toFixed(2)} vs today</span></div>
      <div class="cv2-cols"><div><h4>Specification</h4><ul>${(d.spec || []).map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>
      <div><h4>Score rationale</h4><ul class="cv2-why">${C.map(k => `<li><b>${k} ${d.scores[k]}/10</b> <span class="muted">${esc(D.names[k])}</span><br>${esc(d.why[k])}</li>`).join("")}</ul></div></div>
    </div>`;
  }

  function render() {
    const root = document.getElementById("cv2-root");
    root.innerHTML = `
      <div class="cv2-note">${esc(D.rules.status)}</div>
      <div class="card cv2-card"><div class="sh"><div><h3>Scores vs today's concept module</h3><p>Rubric fixed before reading the papers (weights under each criterion). Hover a cell for the reason; click a row to open the design.</p></div></div>${scoreTable()}</div>
      <div class="cv2-pick">${ORDER.map(k => `<button data-k="${k}" class="${k === sel ? "on" : ""}" style="--c:${COL[k]}">${k === "ROW0" ? "Today" : k + " · " + esc(D.designs[k].title)}</button>`).join("")}</div>
      <div id="cv2-detail">${detail()}</div>
      <div class="card cv2-card"><div class="sh"><div><h3>Shared base (all five designs)</h3></div></div><ul>${D.chassis.map(x => `<li>${esc(x)}</li>`).join("")}</ul></div>`;
    root.querySelectorAll("[data-k]").forEach(el => el.addEventListener("click", () => { sel = el.dataset.k; render(); window.dispatchEvent(new CustomEvent("cv2select", { detail: sel })); }));
    if (!render.done) { render.done = true; requestAnimationFrame(() => window.dispatchEvent(new CustomEvent("cv2select", { detail: sel }))); }
  }

  window.addEventListener("tabshown", e => { if (e.detail === "concept") load(); });
  if (location.hash === "#concept") load();
})();
