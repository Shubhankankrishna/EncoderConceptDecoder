/* Tiny SVG toolkit for the architecture figures (Raschka-style: pastel rounded boxes, thin dark arrows, grey notes). */
const NS = "http://www.w3.org/2000/svg";
const PAL = {                      // v2 (D-042): same hue families as the 3D model, cleaner tints; black text stays legible
  io:    { f: "#eef0f3", s: "#9aa3af" },
  emb:   { f: "#ebe5fa", s: "#8f7cc0" },
  attn:  { f: "#dbe8f9", s: "#5b8fd0" },
  ffn:   { f: "#fde3d0", s: "#dc8b56" },
  norm:  { f: "#e1f3dc", s: "#6fab5e" },
  gate:  { f: "#f2e8fc", s: "#a07cc5" },
  mem:   { f: "#fff1c4", s: "#d1a92f" },
  conc:  { f: "#d7f2ed", s: "#3a9d8f" },
  loss:  { f: "#fcdde3", s: "#d0667a" },
  plan:  { f: "#d7f2ed", s: "#2e9c8c" },
  white: { f: "#ffffff", s: "#c3c8cf" },
  frame: { f: "#fafbfc", s: "#d9dde3" },
};
const lighten = (hex, k) => { const n = parseInt(hex.slice(1), 16), c = [n >> 16, (n >> 8) & 255, n & 255].map(v => Math.round(v + (255 - v) * k));
  return "#" + c.map(v => v.toString(16).padStart(2, "0")).join(""); };
const STREAM = c => c === "#3b78c2" ? "L" : c === "#e07b39" ? "G" : c === "#2e9c8c" ? "plan" : c === "#d0667a" ? "loss" : "ink";   // arrow tags
const COL = { L: "#3b78c2", G: "#e07b39", plan: "#2e9c8c", ink: "#2b2b2e", grey: "#8a8a90", light: "#c9c9c4" };

const fmtP = n => n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "K" : String(n);
const fmtI = n => Number(n).toLocaleString("en-US");

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

let _svgN = 0;
function mk(g, id) { const r = g.ownerSVGElement || g; return `url(#${r.dataset.mk}${id})`; }
function makeSvg(w, h) {
  const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, xmlns: NS, role: "img" });
  svg.dataset.mk = "s" + (++_svgN) + "-";
  const defs = el("defs", {}, svg);
  for (const [id, c] of [["ah", COL.ink], ["ahL", COL.L], ["ahG", COL.G], ["ahP", COL.plan], ["ahg", COL.grey], ["ahR", "#d0667a"]]) {
    const m = el("marker", { id: svg.dataset.mk + id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    el("path", { d: "M0,1.2 L10,5 L0,8.8 L2.2,5 z", fill: c, "stroke-linejoin": "round" }, m);   // swept head
  }
  for (const k of Object.keys(PAL)) {                          // soft vertical gradient per palette key (ids prefixed per SVG)
    const gr = el("linearGradient", { id: svg.dataset.mk + "g-" + k, x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
    el("stop", { offset: "0", "stop-color": lighten(PAL[k].f, 0.55) }, gr);
    el("stop", { offset: "1", "stop-color": PAL[k].f }, gr);
  }
  return svg;
}

function txt(g, x, y, s, o = {}) {
  const t = el("text", { x, y, "font-size": o.size || 12, fill: o.color || COL.ink, "text-anchor": o.anchor || "middle",
    "font-weight": o.weight || 400, "font-style": o.italic ? "italic" : null, "dominant-baseline": o.base || "middle",
    "font-family": o.mono ? "ui-monospace, SF Mono, Menlo, monospace" : null, "letter-spacing": o.ls || null }, g);
  const lines = String(s).split("\n");
  if (lines.length === 1) { t.textContent = s; return t; }
  const lh = (o.size || 12) * 1.25, y0 = y - (lines.length - 1) * lh / 2;
  lines.forEach((ln, i) => { const sp = el("tspan", { x, y: y0 + i * lh }, t); sp.textContent = ln; });
  return t;
}

/* box: {x,y,w,h,k (palette key), label, sub, tip, src, rx, size, dash, weight} */
function box(g, o) {
  const p = PAL[o.k || "white"];
  const grp = el("g", { class: o.tip ? "bx" : null, "data-k": o.k || "white" }, g);
  el("rect", { x: o.x, y: o.y, width: o.w, height: o.h, rx: o.rx ?? 8, fill: o.fill || mk(g, "g-" + (o.k || "white")), stroke: o.stroke || p.s,
    "stroke-width": o.sw || 1.1, "stroke-dasharray": o.dash || null }, grp);
  const cx = o.x + o.w / 2, cy = o.y + o.h / 2;
  if (o.label && o.sub) {
    txt(grp, cx, cy - (o.size || 12.5) * 0.55, o.label, { size: o.size || 12.5, weight: o.weight || 600 });
    txt(grp, cx, cy + (o.size || 12.5) * 0.75, o.sub, { size: (o.size || 12.5) - 2, color: "#5a5a60" });
  } else if (o.label) {
    txt(grp, cx, cy, o.label, { size: o.size || 12.5, weight: o.weight || 600 });
  }
  if (o.tip) { grp.dataset.tip = o.tip; if (o.src) grp.dataset.src = o.src; }
  return grp;
}

/* frame: large dashed container with a title tab */
function frame(g, o) {
  const grp = el("g", { class: "frm" }, g);
  el("rect", { x: o.x, y: o.y, width: o.w, height: o.h, rx: 14, fill: o.fill || "#fafbfc", stroke: o.stroke || "#d5dae1",
    "stroke-width": 1.1, "stroke-dasharray": typeof o.dash === "string" ? o.dash : null }, grp);
  if (o.title) {                                               // title as a pill tab
    const w = o.title.length * 7.1 + 18;
    el("rect", { x: o.x + 7, y: o.y + 6, width: w, height: 21, rx: 10.5, fill: "#ffffff", stroke: o.stroke || "#d5dae1", "stroke-width": 1 }, grp);
    txt(grp, o.x + 16, o.y + 16.5, o.title, { size: 13.5, weight: 700, anchor: "start", color: o.tcolor || "#23272e" });
  }
  if (o.note) txt(grp, o.x + o.w - 12, o.y + 16, o.note, { size: 11.5, anchor: "end", color: COL.grey });
  return grp;
}

/* arrow through points [[x,y],...]; o.c color, o.m marker id, o.dash, o.w width, o.curve (rounded corners) */
function arrow(g, pts, o = {}) {
  let d = `M${pts[0][0]},${pts[0][1]}`;
  const r = o.r ?? 6;
  for (let i = 1; i < pts.length; i++) {
    if (i < pts.length - 1 && r > 0) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i], [x2, y2] = pts[i + 1];
      const d1 = Math.hypot(x1 - x0, y1 - y0), d2 = Math.hypot(x2 - x1, y2 - y1);
      const rr = Math.min(r, d1 / 2, d2 / 2);
      const ax = x1 - (x1 - x0) / d1 * rr, ay = y1 - (y1 - y0) / d1 * rr;
      const bx = x1 + (x2 - x1) / d2 * rr, by = y1 + (y2 - y1) / d2 * rr;
      d += ` L${ax},${ay} Q${x1},${y1} ${bx},${by}`;
    } else d += ` L${pts[i][0]},${pts[i][1]}`;
  }
  return el("path", { d, fill: "none", stroke: o.c || COL.ink, "stroke-width": o.w || 1.35, "stroke-dasharray": o.dash || null,
    "stroke-linecap": "round", "stroke-linejoin": "round", "data-c": STREAM(o.c || COL.ink),
    "marker-end": o.head === false ? null : mk(g, o.m || "ah"), "marker-start": o.both ? mk(g, o.m || "ah") : null }, g);
}

function line(g, x1, y1, x2, y2, o = {}) {
  return el("line", { x1, y1, x2, y2, stroke: o.c || COL.light, "stroke-width": o.w || 1, "stroke-dasharray": o.dash || null }, g);
}

function circ(g, x, y, r, o = {}) {
  const grp = el("g", { class: o.tip ? "bx" : null, "data-k": "circ" }, g);
  el("circle", { cx: x, cy: y, r, fill: o.f || "#fff", stroke: o.s || COL.ink, "stroke-width": o.sw || 1.2 }, grp);
  if (o.label) txt(grp, x, y + 0.5, o.label, { size: o.size || 13, weight: 600 });
  if (o.tip) grp.dataset.tip = o.tip;
  return grp;
}

/* vertical bracket on the right of a stack with a repeat label, e.g. "×4" */
function bracket(g, x, y1, y2, label, o = {}) {
  el("path", { d: `M${x},${y1} h6 V${y2} h-6`, fill: "none", stroke: o.c || COL.ink, "stroke-width": 1.2 }, g);
  txt(g, x + 12, (y1 + y2) / 2, label, { size: o.size || 15, weight: 700, anchor: "start", color: o.c || COL.ink });
}

function note(g, x, y, s, o = {}) {
  return txt(g, x, y, s, { size: o.size || 11, color: o.color || COL.grey, anchor: o.anchor || "start", italic: o.italic, mono: o.mono });
}

/* dimension label drawn on an arrow (small grey text with white halo) */
function dim(g, x, y, s, o = {}) {
  const t = txt(g, x, y, s, { size: o.size || 10.5, color: o.color || "#6a6a70", anchor: o.anchor || "start", mono: true });
  t.setAttribute("paint-order", "stroke"); t.setAttribute("stroke", "#fff"); t.setAttribute("stroke-width", 3);
  return t;
}

/* global tooltip for [data-tip] */
(function () {
  const tip = () => document.getElementById("tip");
  document.addEventListener("mousemove", e => {
    const t = e.target.closest && e.target.closest("[data-tip]");
    const el_ = tip(); if (!el_) return;
    if (!t) { el_.style.display = "none"; return; }
    el_.innerHTML = t.dataset.tip + (t.dataset.src ? `<div class="src">${t.dataset.src}</div>` : "");
    el_.style.display = "block";
    const w = el_.offsetWidth, h = el_.offsetHeight;
    let x = e.clientX + 14, y = e.clientY + 14;
    if (x + w > innerWidth - 8) x = e.clientX - w - 14;
    if (y + h > innerHeight - 8) y = e.clientY - h - 14;
    el_.style.left = x + "px"; el_.style.top = y + "px";
  });
})();
