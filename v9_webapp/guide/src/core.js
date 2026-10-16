/* V9 Guide · core: element helpers, SVG canvas, UI controls, the animation loop, stepper, and the page chrome (TOC, progress, theme). */
(function () {
"use strict";
const NS = "http://www.w3.org/2000/svg";
const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => [...(r || document).querySelectorAll(s)];
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
let UID = 0; const uid = p => (p || "u") + (++UID);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = t => t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gauss(r) { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

/* ---------------------------------------------------------------- elements */
function H(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  kids.flat(9).forEach(c => { if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(c)); });
  return e;
}
const PAINT = new Set(["fill", "stroke", "stop-color", "color"]);
function S(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null) continue;
    if (k === "text") e.textContent = v;
    else if (PAINT.has(k) && String(v).startsWith("var(")) e.style.setProperty(k, v);
    else e.setAttribute(k, v);
  }
  if (parent) parent.appendChild(e);
  return e;
}
const COLORS = ["ink", "ink2", "ink3", "enc", "con", "dec", "code", "sel", "loss", "good", "warn", "bad", "frozen", "accent"];

class Svg {
  constructor(host, W, Ht, o) {
    o = o || {}; this.W = W; this.H = Ht; this.id = uid("sv");
    this.svg = S("svg", { viewBox: `0 0 ${W} ${Ht}`, role: "img", "aria-label": o.label || "diagram", class: o.narrow ? "narrow" : "" });
    host.appendChild(this.svg);
    const defs = S("defs", {}, this.svg);
    COLORS.forEach(c => {
      const m = S("marker", { id: `${this.id}-a-${c}`, viewBox: "0 0 10 10", refX: 8.5, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
      S("path", { d: "M0 0 L10 5 L0 10 z", fill: `var(--${c})` }, m);
    });
    this.defs = defs;
  }
  arrow(c) { return `url(#${this.id}-a-${c || "ink3"})`; }
  g(attrs, parent) { return S("g", attrs, parent || this.svg); }
  rect(x, y, w, h, o, parent) { o = o || {}; return S("rect", Object.assign({ x, y, width: w, height: h, rx: o.rx == null ? 8 : o.rx, fill: o.fill || "var(--panel)", stroke: o.stroke || "var(--line2)", "stroke-width": o.sw || 1.2 }, o.attrs || {}, o.dash ? { "stroke-dasharray": o.dash } : {}, o.opacity != null ? { opacity: o.opacity } : {}), parent || this.svg); }
  line(x1, y1, x2, y2, o, parent) { o = o || {}; return S("line", Object.assign({ x1, y1, x2, y2, stroke: o.stroke || "var(--line2)", "stroke-width": o.sw || 1.4, "stroke-linecap": "round" }, o.dash ? { "stroke-dasharray": o.dash } : {}, o.arrow ? { "marker-end": this.arrow(o.arrow) } : {}, o.opacity != null ? { opacity: o.opacity } : {}), parent || this.svg); }
  path(d, o, parent) { o = o || {}; return S("path", Object.assign({ d, fill: o.fill || "none", stroke: o.stroke || "var(--line2)", "stroke-width": o.sw || 1.4, "stroke-linecap": "round", "stroke-linejoin": "round" }, o.dash ? { "stroke-dasharray": o.dash } : {}, o.arrow ? { "marker-end": this.arrow(o.arrow) } : {}, o.opacity != null ? { opacity: o.opacity } : {}), parent || this.svg); }
  circle(cx, cy, r, o, parent) { o = o || {}; return S("circle", Object.assign({ cx, cy, r, fill: o.fill || "var(--panel)", stroke: o.stroke || "none", "stroke-width": o.sw || 1.2 }, o.opacity != null ? { opacity: o.opacity } : {}), parent || this.svg); }
  text(x, y, str, o, parent) {
    o = o || {};
    const t = S("text", { x, y, "font-size": o.size || 13, "font-weight": o.weight || 400, "text-anchor": o.anchor || "start", "dominant-baseline": o.base || "central", fill: o.fill || "var(--ink)", class: o.mono ? "mono" : "" }, parent || this.svg);
    if (o.opacity != null) t.setAttribute("opacity", o.opacity);
    if (o.halo) { t.style.paintOrder = "stroke"; t.style.stroke = "var(--panel)"; t.style.strokeWidth = "4px"; t.style.strokeLinejoin = "round"; }
    const lines = String(str).split("\n");
    if (lines.length === 1) t.textContent = str;
    else lines.forEach((ln, i) => { const ts = S("tspan", { x, dy: i === 0 ? `${-(lines.length - 1) * 0.6}em` : "1.2em" }, t); ts.textContent = ln; });
    return t;
  }
  clear() { [...this.svg.children].forEach(c => { if (c !== this.defs) c.remove(); }); }
}

/* ---------------------------------------------------------------- UI controls (placed in a figure's top bar) */
const ui = {
  btn(label, fn, cls) { const b = H("button", { class: "btn " + (cls || ""), type: "button" }, label); b.addEventListener("click", fn); return b; },
  range(label, min, max, step, val, fn, fmt) {
    fmt = fmt || (v => v); const id = uid("rg");
    const out = H("output", { for: id }, fmt(val)), inp = H("input", { type: "range", id, min, max, step, value: val });
    inp.addEventListener("input", () => { out.textContent = fmt(+inp.value); fn(+inp.value); });
    const w = H("label", { class: "rng", for: id }, label, inp, out);
    w.set = v => { inp.value = v; out.textContent = fmt(+v); }; w.get = () => +inp.value; return w;
  },
  seg(options, val, fn, label) {
    const w = H("div", { class: "seg", role: "group", "aria-label": label || "options" });
    options.forEach(([v, l]) => { const b = H("button", { type: "button", "data-v": v, class: v === val ? "on" : "" }, l); b.addEventListener("click", () => { w.set(v); fn(v); }); w.append(b); });
    w.set = v => $$("button", w).forEach(b => b.classList.toggle("on", b.dataset.v === String(v)));
    return w;
  },
  check(label, val, fn) {
    const i = H("input", { type: "checkbox" }); i.checked = !!val; i.addEventListener("change", () => fn(i.checked));
    return H("label", { class: "chk" }, i, label);
  },
  readout() { const r = H("span", { class: "readout" }); r.set = html => { r.innerHTML = html; }; return r; },
};

/* ---------------------------------------------------------------- animation loop: runs only for visible, playing figures */
const tickers = new Set(); let raf = 0, last = 0, T0 = performance.now();
function frame(now) {
  raf = 0; const dt = Math.min(0.05, (now - last) / 1000); last = now; let any = false;
  tickers.forEach(k => { if (k.fig._vis && k.fig.playing) { any = true; try { k.fn((now - T0) / 1000, dt); } catch (e) { console.error(e); k.fig.playing = false; } } });
  if (any) raf = requestAnimationFrame(frame);
}
function kick() { if (!raf) { last = performance.now(); raf = requestAnimationFrame(frame); } }
const visIO = new IntersectionObserver(es => { es.forEach(e => { e.target._vis = e.isIntersecting; if (e.isIntersecting) kick(); }); }, { rootMargin: "80px" });
function loop(fig, fn) { tickers.add({ fig, fn }); kick(); }
function playBtn(fig, on) {
  fig.playing = !REDUCED;
  const b = H("button", { class: "btn", type: "button", "aria-pressed": String(fig.playing) });
  const sync = () => { b.textContent = fig.playing ? "Pause" : "Play"; b.setAttribute("aria-pressed", String(fig.playing)); };
  b.addEventListener("click", () => { fig.playing = !fig.playing; sync(); if (fig.playing) kick(); if (on) on(fig.playing); });
  sync(); return b;
}

/* ---------------------------------------------------------------- stepper: numbered chips, prev/next/auto-play, live caption */
function stepper(fig, steps, onStep, o) {
  o = o || {}; const ms = o.ms || 3600; let cur = -1, acc = 0;
  const chips = H("div", { class: "steprow", role: "tablist" });
  const btns = steps.map((s, i) => { const b = H("button", { type: "button", role: "tab" }, s.label || String(i + 1)); b.addEventListener("click", () => { fig.playing = false; sync(); go(i); }); chips.append(b); return b; });
  const play = H("button", { class: "btn", type: "button" });
  const sync = () => { play.textContent = fig.playing ? "Pause" : "Auto-play"; };
  play.addEventListener("click", () => { fig.playing = !fig.playing; sync(); acc = 0; if (fig.playing) kick(); });
  const prev = ui.btn("Back", () => { fig.playing = false; sync(); go((cur + steps.length - 1) % steps.length); });
  const next = ui.btn("Next", () => { fig.playing = false; sync(); go((cur + 1) % steps.length); }, "pri");
  fig.$ui.append(prev, next, play);
  fig.insertBefore(chips, fig.$stage);
  function go(i) {
    cur = i; btns.forEach((b, k) => { b.classList.toggle("on", k === i); b.setAttribute("aria-selected", String(k === i)); });
    fig.$live.hidden = false; fig.$live.innerHTML = `<b>${steps[i].title || ""}</b> ${steps[i].text || ""}`; onStep(i, steps[i]);
  }
  fig.playing = o.auto !== false && !REDUCED; sync();
  loop(fig, (t, dt) => { acc += dt * 1000; if (acc > ms) { acc = 0; go((cur + 1) % steps.length); } });
  go(o.start || 0);
  return { go, get cur() { return cur; } };
}

/* ---------------------------------------------------------------- figure registry and lazy mounting */
const REG = {};
function reg(name, fn) { REG[name] = fn; }
function mountFig(fig) {
  if (fig._m) return; fig._m = 1;
  const name = fig.dataset.viz, fn = REG[name];
  const cap = $("figcaption", fig);
  fig.$title = H("div", { class: "fig-title" }, fig.dataset.title || "", fig.dataset.tag ? H("i", {}, fig.dataset.tag) : null);
  if (!fig.dataset.title && !fig.dataset.tag) fig.$title.hidden = true;
  fig.$ui = H("div", { class: "fig-ui" }); fig.$stage = H("div", { class: "fig-stage" }); fig.$live = H("div", { class: "fig-cap-live", "aria-live": "polite", hidden: true });
  fig.insertBefore(fig.$live, cap); fig.insertBefore(fig.$stage, fig.$live); fig.insertBefore(fig.$ui, fig.$stage); fig.insertBefore(fig.$title, fig.$ui);
  fig.svg = (W, h, o) => new Svg(fig.$stage, W, h, Object.assign({ label: fig.dataset.title || name }, o || {}));
  visIO.observe(fig);
  if (!fn) { fig.$stage.textContent = "Figure not available."; console.warn("missing viz", name); return; }
  try { fn(fig); } catch (e) { console.error("viz", name, e); fig.$stage.textContent = "This figure failed to load."; }
}
const mountIO = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { mountFig(e.target); mountIO.unobserve(e.target); } }), { rootMargin: "700px 0px" });

/* ---------------------------------------------------------------- page chrome */
function buildToc() {
  const toc = $("#toc"); if (!toc) return; toc.innerHTML = "";
  $$("main.guide section.chap").forEach(sec => {
    const h2 = $("h2", sec); if (!h2) return;
    const grp = H("div", { class: "grp", "data-for": sec.id });
    grp.append(H("a", { class: "h2", href: "#" + sec.id }, H("i", {}, sec.dataset.num || ""), h2.textContent));
    $$("h3[id]", sec).forEach(h => grp.append(H("a", { class: "h3", href: "#" + h.id }, h.dataset.short || h.textContent)));
    toc.append(grp);
  });
  const chips = $("nav.chapnav"); if (chips) { chips.innerHTML = ""; $$("main.guide section.chap").forEach(sec => chips.append(H("a", { href: "#" + sec.id, "data-for": sec.id }, sec.dataset.chip || sec.dataset.num || sec.id))); }
}
function spy() {
  const secs = $$("main.guide section.chap"), hs = $$("main.guide h3[id]");
  const y = innerHeight * 0.28; let cur = secs[0], curH = null;
  secs.forEach(s => { if (s.getBoundingClientRect().top < y) cur = s; });
  hs.forEach(h => { if (h.getBoundingClientRect().top < y && cur.contains(h)) curH = h; });
  $$("#toc a, nav.chapnav a").forEach(a => {
    const isSec = a.getAttribute("href") === "#" + (cur && cur.id), isH = curH && a.getAttribute("href") === "#" + curH.id;
    a.classList.toggle("on", !!(isSec && !a.classList.contains("h3")) || !!isH);
  });
  $$("#toc .grp").forEach(g => g.classList.toggle("open", cur && g.dataset.for === cur.id));
  const on = $("nav.chapnav a.on"); if (on && on.scrollIntoView && !spy.lock) { const c = on.parentElement; const l = on.offsetLeft - c.clientWidth / 2 + on.clientWidth / 2; if (Math.abs(c.scrollLeft - l) > 40) c.scrollTo({ left: l, behavior: "smooth" }); }
  const doc = document.documentElement, p = clamp(scrollY / Math.max(1, doc.scrollHeight - innerHeight), 0, 1);
  const bar = $("#progress"); if (bar) bar.style.width = (p * 100).toFixed(2) + "%";
}
function theme() {
  const btn = $("#theme"); if (!btn) return;
  const root = document.documentElement, KEY = "v9guide-theme";
  const saved = () => { try { const v = localStorage.getItem(KEY); return v === "light" || v === "dark" ? v : null; } catch (e) { return null; } };
  const label = () => { const t = root.getAttribute("data-theme"); btn.textContent = t === "dark" ? "Theme: dark" : t === "light" ? "Theme: light" : "Theme: auto"; };
  const apply = m => { if (m === "auto") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", m); label(); document.dispatchEvent(new Event("themechange")); };
  /* never overwrite a theme the host page already set: only apply a choice this reader made earlier */
  const s0 = saved(); if (s0) apply(s0); else label();
  btn.addEventListener("click", () => { const cur = root.getAttribute("data-theme") || "auto", n = { auto: "light", light: "dark", dark: "auto" }[cur]; try { n === "auto" ? localStorage.removeItem(KEY) : localStorage.setItem(KEY, n); } catch (e) { /* storage blocked */ } apply(n); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => document.dispatchEvent(new Event("themechange")));
}
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function goto(id) { const t = document.getElementById(id); if (!t) return; const y = t.getBoundingClientRect().top + scrollY - (52 + 14); scrollTo({ top: y, behavior: REDUCED ? "auto" : "smooth" }); try { history.replaceState(null, "", "#" + id); } catch (e) { /* the viewer frame may refuse this */ } }
function boot() {
  if (boot.done) return; boot.done = true;
  /* in-page links are handled here so they also work inside a sandboxed viewer frame */
  document.addEventListener("click", e => { const a = e.target.closest && e.target.closest('a[href^="#"]'); if (!a) return; const id = a.getAttribute("href").slice(1); if (id && document.getElementById(id)) { e.preventDefault(); goto(id); } });
  buildToc(); theme();
  $$("figure.fig[data-viz]").forEach(f => mountFig(f));   /* mount every figure now so the layout is final; only the animation loop waits for visibility */
  let tick = 0; addEventListener("scroll", () => { if (!tick) tick = requestAnimationFrame(() => { tick = 0; spy(); }); }, { passive: true });
  addEventListener("resize", spy); spy();
  /* deep link: figures are already mounted, so wait for fonts and jump without smooth scrolling */
  if (location.hash.length > 1) {
    const go = () => { const t = document.getElementById(decodeURIComponent(location.hash.slice(1))); if (t) { const y = t.getBoundingClientRect().top + scrollY - 66; scrollTo({ top: y, behavior: "instant" }); spy(); } };
    (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(() => setTimeout(go, 80));
  }
}
/* test hook: advance every playing figure by n frames of dt seconds, even in a hidden tab where requestAnimationFrame is paused */
let FT = 0; function force(n, dt) { dt = dt || 0.05; for (let i = 0; i < (n || 1); i++) { FT += dt; tickers.forEach(k => { if (k.fig.playing) k.fn(FT, dt); }); } }
function along(pts, u) { let L = 0; const seg = []; for (let k = 1; k < pts.length; k++) { const d = Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]); seg.push(d); L += d; } let a = u * L; for (let k = 0; k < seg.length; k++) { if (a <= seg[k] || k === seg.length - 1) { const f = seg[k] ? Math.min(1, a / seg[k]) : 0; return [lerp(pts[k][0], pts[k + 1][0], f), lerp(pts[k][1], pts[k + 1][1], f)]; } a -= seg[k]; } return pts[pts.length - 1]; }
window.G = { goto, along, force, $, $$, H, S, Svg, ui, rng, gauss, clamp, lerp, ease, uid, REDUCED, loop, playBtn, stepper, reg, mountFig, cssv, boot, kick };
document.addEventListener("DOMContentLoaded", () => setTimeout(boot, 0));
if (document.readyState !== "loading") setTimeout(boot, 0);
})();
