/* Shared chart component (design system, webapp DESIGN.md §3): SVG line chart with hover / touch readout, reference
   lines, bands, vertical markers, dashed comparison series. Used by the v3 live page and the Compare page.
   LC.chart(id, opts) -> HTML string;  LC.interp(rows, x, key) linear interpolation on {step, key} rows. */
window.LC = (function () {
  const CG = "#8a93a3";
  const fmt = (x, d = 3) => x == null || !isFinite(x) ? "–" : (+x).toFixed(d);
  const fI = x => x == null ? "–" : Math.round(x).toLocaleString("en-US");
  const CHARTS = {};                                                     // id -> {series, x0, x1, y0, y1, ...} for hover

  /* ------------------------------------------------------------ interpolation on reference curves */
  function interp(rows, step, key) {
    const r = (rows || []).filter(x => x[key] != null);
    if (!r.length || step < r[0].step || step > r[r.length - 1].step) return null;
    for (let i = 1; i < r.length; i++) if (r[i].step >= step) {
      const a = r[i - 1], b = r[i], w = (step - a.step) / ((b.step - a.step) || 1);
      return a[key] + w * (b[key] - a[key]);
    }
    return r[r.length - 1][key];
  }

  /* ------------------------------------------------------------ SVG line chart with hover */
  function chart(id, o) {
    const W = o.W || 560, H = o.H || 250, L = 52, R = 14, T = 14, B = 34;
    const ser = o.series.filter(s => s.pts && s.pts.length);
    const all = ser.flatMap(s => s.pts).concat((o.refs || []).map(r => [null, r.y]));
    if (!ser.length) return `<div class="lc-empty">${o.empty || "waiting for data"}</div>`;
    const xs = ser.flatMap(s => s.pts.map(p => p[0]));
    let x0 = o.x0 ?? Math.min(...xs), x1 = o.x1 ?? Math.max(...xs); if (x1 === x0) x1 = x0 + 1;
    const ys = all.map(p => p[1]).filter(v => v != null && isFinite(v));
    let y0 = o.y0 ?? Math.min(...ys), y1 = o.y1 ?? Math.max(...ys);
    if (y1 === y0) { y1 += 1; y0 -= 1; }
    const pad = (y1 - y0) * 0.06; if (o.y0 == null) y0 -= pad; if (o.y1 == null) y1 += pad;
    const px = x => L + (x - x0) / (x1 - x0) * (W - L - R), py = y => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
    const ticks = (a, b, n) => { const s = Math.pow(10, Math.floor(Math.log10((b - a) / n))); const m = [1, 2, 5, 10].find(k => (b - a) / (s * k) <= n) * s; const out = []; for (let v = Math.ceil(a / m) * m; v <= b + 1e-9; v += m) out.push(+v.toFixed(10)); return out; };
    const yt = ticks(y0, y1, 5), xt = ticks(x0, x1, 6);
    let g = `<svg viewBox="0 0 ${W} ${H}" class="lc" data-chart="${id}">`;
    g += yt.map(v => `<line x1="${L}" x2="${W - R}" y1="${py(v)}" y2="${py(v)}" class="lc-grid"/><text x="${L - 6}" y="${py(v) + 4}" class="lc-ax" text-anchor="end">${o.yfmt ? o.yfmt(v) : +v.toPrecision(4)}</text>`).join("");
    g += xt.map(v => `<text x="${px(v)}" y="${H - 14}" class="lc-ax" text-anchor="middle">${o.xfmt ? o.xfmt(v) : fI(v)}</text>`).join("");
    g += `<text x="${W - R}" y="${H - 2}" class="lc-ax" text-anchor="end">${o.xlabel || "optimizer step"}</text>`;
    (o.bands || []).forEach(b => { g += `<rect x="${L}" width="${W - L - R}" y="${py(Math.min(b.hi, y1))}" height="${Math.max(0, py(Math.max(b.lo, y0)) - py(Math.min(b.hi, y1)))}" fill="${b.color}" opacity="${b.op || 0.1}"/>`; });
    (o.refs || []).forEach((r, i) => { if (r.y >= y0 && r.y <= y1) g += `<line x1="${L}" x2="${W - R}" y1="${py(r.y)}" y2="${py(r.y)}" stroke="${r.color || CG}" stroke-dasharray="5 4" stroke-width="1.2"/><text x="${W - R - 4}" y="${py(r.y) + (i % 2 ? 13 : -4)}" class="lc-ref" fill="${r.color || CG}" text-anchor="end">${r.label}</text>`; });
    (o.vlines || []).forEach(v => { if (v.x >= x0 && v.x <= x1) { const nearR = px(v.x) > W - R - 90; g += `<line y1="${T}" y2="${H - B}" x1="${px(v.x)}" x2="${px(v.x)}" stroke="${v.color || CG}" stroke-dasharray="2 3"/><text x="${px(v.x) + (nearR ? -3 : 3)}" y="${T + 10}" class="lc-ref" fill="${v.color || CG}" text-anchor="${nearR ? "end" : "start"}">${v.label}</text>`; } });
    ser.forEach(s => {
      const pts = s.pts.filter(p => p[1] != null && isFinite(p[1]) && p[0] >= x0 && p[0] <= x1);
      if (!pts.length) return;
      g += `<path d="${pts.map((p, i) => `${i ? "L" : "M"}${px(p[0]).toFixed(1)},${py(Math.max(y0, Math.min(y1, p[1]))).toFixed(1)}`).join(" ")}" fill="none" stroke="${s.color}" stroke-width="${s.w || 2}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} opacity="${s.op || 1}"/>`;
      if (s.dots) g += pts.map(p => `<circle cx="${px(p[0])}" cy="${py(Math.max(y0, Math.min(y1, p[1])))}" r="2.6" fill="${s.color}"/>`).join("");
    });
    g += `<line class="lc-cross" y1="${T}" y2="${H - B}" x1="-10" x2="-10"/></svg>`;
    CHARTS[id] = { ser, x0, x1, L, R, W, fmt: o.tfmt || (v => fmt(v, 3)), xname: o.xname || "step", xf: o.xfmt || fI };
    const legend = ser.map(s => `<span><i style="background:${s.color};${s.dash ? "opacity:.7" : ""}"></i>${s.name}</span>`).join("");
    return `<div class="lc-wrap">${g}<div class="lc-tip" hidden></div></div><div class="lc-leg">${legend}</div>`;
  }
  function hover(e) {
    const svg = e.target.closest && e.target.closest("svg.lc"); if (!svg) return;
    const c = CHARTS[svg.dataset.chart]; if (!c) return;
    const r = svg.getBoundingClientRect(), fx = (e.clientX - r.left) / r.width * c.W;
    const x = c.x0 + (fx - c.L) / (c.W - c.L - c.R) * (c.x1 - c.x0);
    const rows = c.ser.map(s => { let best = null; s.pts.forEach(p => { if (p[1] != null && (!best || Math.abs(p[0] - x) < Math.abs(best[0] - x))) best = p; }); return best ? [s, best] : null; }).filter(Boolean);
    if (!rows.length) return;
    const tip = svg.parentElement.querySelector(".lc-tip"), cr = svg.querySelector(".lc-cross");
    cr.setAttribute("x1", fx); cr.setAttribute("x2", fx);
    tip.hidden = false;
    tip.innerHTML = `<b>${c.xname} ≈ ${c.xf(x)}</b>` + rows.map(([s, p]) => `<div><i style="background:${s.color}"></i>${s.name}: <b>${c.fmt(p[1])}</b> <span>@${c.xf(p[0])}</span></div>`).join("");
    const left = (e.clientX - r.left) > r.width * 0.6;
    tip.style.left = left ? "" : (e.clientX - r.left + 14) + "px"; tip.style.right = left ? (r.right - e.clientX + 14) + "px" : "";
    tip.style.top = Math.max(0, e.clientY - r.top - 20) + "px";
  }
  document.addEventListener("mousemove", hover);
  document.addEventListener("touchmove", e => { if (e.touches[0]) hover({ target: e.target, clientX: e.touches[0].clientX, clientY: e.touches[0].clientY }); }, { passive: true });
  document.addEventListener("mouseout", e => { const svg = e.target.closest && e.target.closest("svg.lc"); if (svg && !svg.contains(e.relatedTarget)) { const t = svg.parentElement.querySelector(".lc-tip"); if (t) t.hidden = true; } });
  return { chart, interp, fmt, fI };
})();
