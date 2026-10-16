/* Minimal canvas line chart: axes, grid, reference lines, legend, hover readout. Redraws only on data / size change. */
class LineChart {
  constructor(parent, o) {
    this.o = Object.assign({ xLabel: "step", height: 230, fmt: v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3)) }, o);
    const w = document.createElement("div"); w.className = "card chart";
    const hasGuide = !!(o.measures || o.good);
    const guide = hasGuide ? `<div class="guide" hidden>${o.measures ? `<div><b>Measures</b> ${o.measures}</div>` : ""}${o.good ? `<div><b>Healthy when</b> ${o.good}</div>` : ""}</div>` : "";
    w.innerHTML = `<div class="ch-head">
        <div class="ch-tl"><div class="ct">${o.title}${hasGuide ? `<button class="ch-i" title="What does this chart show?" aria-label="explain chart">?</button>` : ""}</div>
          <div class="ch-st"></div></div>
        <div class="ch-v"><div class="latest"></div><div class="ch-vl">${o.latestLabel || "latest"}</div></div></div>
      ${o.sub ? `<div class="cs">${o.sub}</div>` : ""}${guide}<canvas></canvas><div class="legend"></div><div class="csel"></div><div class="ctip"></div>`;
    this.latestEl = w.querySelector(".latest"); this.stEl = w.querySelector(".ch-st"); this.status = null;
    if (hasGuide) w.querySelector(".ch-i").onclick = () => { const g = w.querySelector(".guide"); g.hidden = !g.hidden; w.querySelector(".ch-i").classList.toggle("on", !g.hidden); };
    parent.appendChild(w);
    this.wrap = w; this.cv = w.querySelector("canvas"); this.tip = w.querySelector(".ctip"); this.selEl = w.querySelector(".csel");
    this.cv.style.height = this.o.height + "px";
    this.data = o.series.map(() => []);
    this.sel = null; this.drag = null;
    w.querySelector(".legend").innerHTML = o.series.map(s => `<span><i style="background:${s.color};${s.dash ? "background:repeating-linear-gradient(90deg," + s.color + " 0 4px,transparent 4px 7px)" : ""}"></i>${s.name}</span>`).join("");
    new ResizeObserver(() => this.draw()).observe(this.cv);
    // range selection: drag across the plot to compare two points; a plain click clears it
    this.cv.addEventListener("pointerdown", e => this.selStart(e));
    this.cv.addEventListener("pointermove", e => { if (this.drag) this.selMove(e); else if (e.pointerType === "mouse") this.hover(e); });
    this.cv.addEventListener("pointerup", e => this.selEnd(e));
    this.cv.addEventListener("pointercancel", () => { this.drag = null; this.draw(); });
    this.cv.addEventListener("mouseleave", () => { if (this.drag) return; this.tip.style.display = "none"; this.hx = null; this.draw(); });
    this.selEl.addEventListener("click", e => { if (e.target.closest(".csel-x")) { this.sel = null; this.draw(); this.renderSel(); } });
    this.renderSel();
  }
  set(data) {
    this.data = data;
    const d0 = (data[this.o.latestIdx || 0] || []).filter(p => p[1] != null && isFinite(p[1]));
    this.latestEl.textContent = d0.length ? (this.o.latestFmt || this.o.fmt)(d0[d0.length - 1][1]) : "—";
    this.draw();
    this.renderSel();
    this.updStatus();
  }
  // o.status(chart) -> { lvl: "g" | "a" | "r" | "n", why }: green healthy, amber watch, red problem, grey no verdict
  updStatus() {
    if (!this.o.status) return;
    let s = null;
    try { s = this.o.status(this); } catch (e) { s = null; }
    if (!s) s = { lvl: "n", why: "not enough data yet" };
    this.status = s;
    const label = { g: "Healthy", a: "Watch", r: "Problem", n: s.label || "No verdict" }[s.lvl];
    this.wrap.classList.remove("st-g", "st-a", "st-r", "st-n"); this.wrap.classList.add("st-" + s.lvl);
    this.stEl.innerHTML = `<span class="st-pill ${s.lvl}"><i></i>${label}</span><span class="st-why">${s.why}</span>`;
  }
  nearestX(mx) {
    let best = null;
    this.data.forEach(d => d.forEach(([x, y]) => { if (y == null || !isFinite(y)) return; const dx = Math.abs(this.X(x) - mx); if (!best || dx < best.dx) best = { dx, x }; }));
    return best && best.x;
  }
  mouseX(e) { return e.clientX - this.cv.getBoundingClientRect().left; }
  selStart(e) {
    if (!this.b || e.button !== 0) return;
    const x = this.nearestX(this.mouseX(e)); if (x == null) return;
    this.drag = { a: x, px: e.clientX };
    this.cv.setPointerCapture(e.pointerId);
  }
  selMove(e) {
    const x = this.nearestX(this.mouseX(e)); if (x == null) return;
    if (Math.abs(e.clientX - this.drag.px) < 4) return;
    this.sel = { a: this.drag.a, b: x };
    this.tip.style.display = "none"; this.hx = null;
    this.draw(); this.renderSel();
  }
  selEnd(e) {
    if (!this.drag) return;
    const moved = Math.abs(e.clientX - this.drag.px) >= 4;
    this.drag = null;
    if (!moved) this.sel = null;                              // click without dragging clears the selection
    else if (this.sel && this.sel.a === this.sel.b) this.sel = null;
    this.draw(); this.renderSel();
  }
  valueAt(i, x) {                                               // nearest finite point of series i to x
    let best = null;
    (this.data[i] || []).forEach(p => { if (p[1] == null || !isFinite(p[1])) return; if (!best || Math.abs(p[0] - x) < Math.abs(best[0] - x)) best = p; });
    return best;
  }
  fmtX(v) { return this.o.xLabel === "step" ? Math.round(v).toLocaleString() : (+v.toFixed(1)).toLocaleString(); }
  renderSel() {
    if (!this.sel) { this.selEl.innerHTML = `<span class="csel-hint">↔ drag across the chart to measure the change between two points</span>`; this.selEl.classList.remove("on"); return; }
    const a = Math.min(this.sel.a, this.sel.b), b = Math.max(this.sel.a, this.sel.b), xl = this.o.xLabel, f = this.o.fmt;
    const span = b - a, spanTxt = xl === "step" ? `${this.fmtX(span)} steps` : `${this.fmtX(span)} ${xl}`;
    const rows = this.o.series.map((s, i) => {
      const p0 = this.valueAt(i, a), p1 = this.valueAt(i, b);
      if (!p0 || !p1 || p0 === p1) return "";
      const d = p1[1] - p0[1], pct = p0[1] !== 0 ? 100 * d / Math.abs(p0[1]) : null;
      const at = (p, x) => (p[0] !== x ? `<span class="csel-at">@${this.fmtX(p[0])}</span>` : "");
      return `<tr><td><span style="color:${s.color}">●</span> ${s.name}</td>
        <td class="n">${f(p0[1])}${at(p0, a)}</td><td class="n">→ ${f(p1[1])}${at(p1, b)}</td>
        <td class="n d ${d < 0 ? "dn" : d > 0 ? "up" : ""}">${d < 0 ? "▼ " : d > 0 ? "▲ +" : ""}${f(d)}</td>
        <td class="n p">${pct == null ? "" : (pct > 0 ? "+" : "") + pct.toFixed(1) + " %"}</td></tr>`;
    }).join("");
    this.selEl.classList.add("on");
    this.selEl.innerHTML = `<div class="csel-h"><b>${xl} ${this.fmtX(a)} → ${xl} ${this.fmtX(b)}</b> <span>(${spanTxt})</span><button class="csel-x" title="clear selection">clear ×</button></div>
      ${rows ? `<table>${rows}</table>` : `<div class="csel-hint">no data in this range</div>`}`;
  }
  css(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim() || "#888"; }
  bounds() {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    this.data.forEach(d => d.forEach(([x, y]) => { if (y == null || !isFinite(y)) return; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }));
    (this.o.hlines || []).forEach(h => { if (this.o.hlinesInRange !== false) { y0 = Math.min(y0, h.y); y1 = Math.max(y1, h.y); } });
    if (this.o.yMin != null) y0 = this.o.yMin; if (this.o.yMax != null) y1 = Math.max(this.o.yMax, y1);
    if (!isFinite(x0)) return null;
    if (x1 === x0) x1 = x0 + 1;
    if (y1 === y0) { y1 += Math.abs(y1) * 0.1 + 0.01; y0 -= Math.abs(y0) * 0.1 + 0.01; }
    const pad = (y1 - y0) * 0.06; return { x0, x1, y0: this.o.yMin != null ? y0 : y0 - pad, y1: y1 + pad };
  }
  ticks(a, b, n) {
    const span = b - a, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0) || mag * 10;
    const out = []; for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10)); return out;
  }
  draw() {
    const cv = this.cv, dpr = Math.min(window.devicePixelRatio || 1, 2), W = cv.clientWidth, H = cv.clientHeight;
    if (!W) return;
    cv.width = W * dpr; cv.height = H * dpr;
    const c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
    const b = this.bounds(), ink3 = this.css("--ink3"), line = this.css("--line");
    const L = 50, R = 10, T = 8, B = 26, pw = W - L - R, ph = H - T - B;
    c.font = "11px -apple-system, Helvetica, sans-serif";
    if (!b) { c.fillStyle = ink3; c.textAlign = "center"; c.fillText(this.o.empty || "no data yet", W / 2, H / 2); return; }
    this.b = b; this.geom = { L, T, pw, ph };
    const X = x => L + (x - b.x0) / (b.x1 - b.x0) * pw, Y = y => T + (1 - (y - b.y0) / (b.y1 - b.y0)) * ph;
    this.X = X; this.Y = Y;
    if (this.sel) {                                             // shaded selected range with edge lines
      const xa = X(Math.min(this.sel.a, this.sel.b)), xb = X(Math.max(this.sel.a, this.sel.b)), acc = this.css("--accent");
      c.save(); c.fillStyle = acc; c.globalAlpha = 0.10; c.fillRect(xa, T, xb - xa, ph);
      c.globalAlpha = 0.7; c.strokeStyle = acc; c.lineWidth = 1.2; [xa, xb].forEach(x => { c.beginPath(); c.moveTo(x, T); c.lineTo(x, T + ph); c.stroke(); }); c.restore();
    }
    c.strokeStyle = line; c.lineWidth = 1; c.fillStyle = ink3;
    this.ticks(b.y0, b.y1, 5).forEach(v => { const y = Y(v); c.beginPath(); c.moveTo(L, y); c.lineTo(W - R, y); c.stroke(); c.textAlign = "right"; c.fillText(this.o.fmt(v), L - 6, y + 3.5); });
    this.ticks(b.x0, b.x1, 6).forEach(v => { const x = X(v); c.textAlign = "center"; c.fillText(v >= 1000 ? (v / 1000).toFixed(v % 1000 ? 1 : 0) + "k" : v, x, H - 8); });
    (this.o.hlines || []).forEach(h => {
      const y = Y(h.y); c.save(); c.strokeStyle = h.color || "#999"; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(L, y); c.lineTo(W - R, y); c.stroke(); c.restore();
      c.fillStyle = h.color || "#999"; c.textAlign = "right"; c.fillText(h.label, W - R - 2, y - 4);
    });
    (this.o.vlines || []).forEach(v => {
      if (v.x < b.x0 || v.x > b.x1) return; const x = X(v.x); c.save(); c.strokeStyle = v.color || "#bbb"; c.setLineDash([2, 3]); c.beginPath(); c.moveTo(x, T); c.lineTo(x, T + ph); c.stroke(); c.restore();
      c.fillStyle = v.color || "#999"; c.textAlign = "left"; c.fillText(v.label, x + 3, T + 10);
    });
    this.o.series.forEach((s, i) => {
      const d = this.data[i] || []; if (!d.length) return;
      if (s.fill ?? this.o.series.length === 1) {               // soft area under single-series charts
        const pts = d.filter(([, y]) => y != null && isFinite(y));
        if (pts.length > 1 && /^#[0-9a-f]{6}$/i.test(s.color)) {
          const g = c.createLinearGradient(0, T, 0, T + ph); g.addColorStop(0, s.color + "30"); g.addColorStop(1, s.color + "00");
          c.beginPath(); pts.forEach(([x, y], j) => j ? c.lineTo(X(x), Y(y)) : c.moveTo(X(x), Y(y)));
          c.lineTo(X(pts.at(-1)[0]), T + ph); c.lineTo(X(pts[0][0]), T + ph); c.closePath(); c.fillStyle = g; c.fill();
        }
      }
      c.strokeStyle = s.color; c.lineWidth = s.width || 1.8; c.setLineDash(s.dash ? [5, 4] : []);
      c.beginPath(); let started = false;
      d.forEach(([x, y]) => { if (y == null || !isFinite(y)) { started = false; return; } started ? c.lineTo(X(x), Y(y)) : c.moveTo(X(x), Y(y)); started = true; });
      c.stroke(); c.setLineDash([]);
      if (s.dots || d.length < 30) { c.fillStyle = s.color; d.forEach(([x, y]) => { if (y == null) return; c.beginPath(); c.arc(X(x), Y(y), s.dots === "big" ? 3.2 : 2.3, 0, 7); c.fill(); }); }
    });
    if (this.sel) {                                             // markers on the compared points
      [this.sel.a, this.sel.b].forEach(x => this.o.series.forEach((s, i) => {
        const p = this.valueAt(i, x); if (!p) return;
        c.fillStyle = this.css("--panel"); c.strokeStyle = s.color; c.lineWidth = 2; c.beginPath(); c.arc(X(p[0]), Y(p[1]), 3.6, 0, 7); c.fill(); c.stroke();
      }));
    }
    if (this.hx != null) { c.strokeStyle = ink3; c.setLineDash([2, 2]); c.beginPath(); c.moveTo(X(this.hx), T); c.lineTo(X(this.hx), T + ph); c.stroke(); c.setLineDash([]); }
  }
  hover(e) {
    if (!this.b) return;
    const bx = this.nearestX(this.mouseX(e));
    if (bx == null) return;
    const best = { x: bx };
    this.hx = best.x; this.draw();
    const rows = this.o.series.map((s, i) => {
      const p = (this.data[i] || []).reduce((a, q) => (a == null || Math.abs(q[0] - best.x) < Math.abs(a[0] - best.x) ? q : a), null);
      return p && Math.abs(p[0] - best.x) < 1e-9 && p[1] != null ? `<span style="color:${s.color}">●</span> ${s.name}: <b>${this.o.fmt(p[1])}</b>` : null;
    }).filter(Boolean);
    this.tip.innerHTML = `<div>${this.o.xLabel} ${best.x}</div>` + rows.join("<br>");
    this.tip.style.display = "block";
    const tw = this.tip.offsetWidth;
    this.tip.style.left = Math.min(Math.max(this.cv.offsetLeft + this.X(best.x) + 12, 0), this.wrap.clientWidth - tw - 8) + "px";
    this.tip.style.top = (this.cv.offsetTop + 6) + "px";
  }
}
