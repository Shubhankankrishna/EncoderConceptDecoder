/* Interactive layer for the 2D figures (D-042): filter chips generated from each figure's tagged elements (focus mode),
   click-to-pin details, zoom / pan via the viewBox, fullscreen, SVG / PNG export, a link to the matching 3D view. */
const FIG_KEY = {
  overview: "Anything older than 70 tokens can reach the decoder only through the concept stream.",
  enc: "Two layer types, one grammar: S layers write both streams, F layers write only the global stream.",
  reach: "The decoder sees exactly the last 70 tokens; everything older must come through the concepts.",
  engram: "A 50.3M-number n-gram lookup that costs almost no compute, admitted by a learned gate.",
  crc: "Every 4 tokens become one concept: a learned mix of embedding, local and global state, on a sphere.",
  core: "One shared layer thinks R times over the concepts, then predicts the next concept's codes.",
  dec: "Confined: each position sees only its own plan plus the last 70 tokens, at any depth.",
  plan: "Position t reads the newest complete chunk, so the token it predicts is never inside its plan.",
  obj: "Next-token loss + 0.1 × next-concept loss + codebook loss, on a data mix that needs long range.",
};
const FIG_3D = { overview: "all", enc: "enc", reach: "enc", engram: "enc", crc: "enc", core: "core", dec: "dec", plan: "dec", obj: "all" };
const TAG_NAMES = {
  attn: ["Sliding-window layer", "#5b8fd0"], full: ["Full attention", "#e07b39"], mem: ["Memory / codebook", "#d1a92f"],
  conc: ["Concepts", "#3a9d8f"], plan: ["Plan", "#2e9c8c"], emb: ["Embedding / head", "#8f7cc0"], gate: ["Gates & mixing", "#a07cc5"],
  norm: ["Norms", "#6fab5e"], ffn: ["Feed-forward", "#dc8b56"], loss: ["Losses", "#d0667a"], io: ["Input / output", "#9aa3af"],
  "c:L": ["L stream arrows", "#3b78c2"], "c:G": ["G stream arrows", "#e07b39"], "c:plan": ["Concept / plan arrows", "#2e9c8c"],
};

function figTools(fig, id, idx, total, ctx) {
  ctx = ctx || { figs: FIGS, go: gotoFig, open3d: open3D, key: FIG_KEY, map3d: FIG_3D };   // v2 pages pass their own context
  const svg = fig.querySelector("svg");
  const head = document.createElement("div"); head.className = "fig-head";
  head.innerHTML = `<div class="fig-key">${ctx.key[id] || ""}</div>
    <div class="fig-tools">
      ${svg ? `<button data-t="out" title="Zoom out">−</button><button data-t="in" title="Zoom in (or ⌘ / Ctrl + scroll)">+</button><button data-t="fit" title="Reset zoom">⤢</button>
      <span class="sep"></span><button data-t="full" title="Fullscreen">⛶</button><button data-t="svg" title="Download SVG">SVG</button><button data-t="png" title="Download PNG">PNG</button><span class="sep"></span>` : ""}
      <button data-t="3d" class="wide" title="Open the 3D model at this part">View in 3D →</button>
    </div>`;
  fig.querySelector("h3.figt").after(head);
  if (svg) {
    const wrap = document.createElement("div"); wrap.className = "fig-canvas"; svg.before(wrap); wrap.appendChild(svg);
    const chips = document.createElement("div"); chips.className = "fig-chips"; wrap.before(chips);
    const detail = document.createElement("div"); detail.className = "fig-detail"; wrap.after(detail);
    buildChips(svg, chips);
    wirePin(svg, detail);
    wireZoom(svg, wrap);
  }
  const foot = document.createElement("div"); foot.className = "fig-foot";
  const prev = ctx.figs[idx - 1], next = ctx.figs[idx + 1];
  foot.innerHTML = `<span>Figure ${idx + 1} of ${total}</span><span class="fig-pn">${prev ? `<button data-go-fig="${prev.id}">← ${prev.t}</button>` : ""}${next ? `<button data-go-fig="${next.id}">${next.t} →</button>` : ""}</span>`;
  fig.appendChild(foot);
  foot.querySelectorAll("[data-go-fig]").forEach(b => b.onclick = () => ctx.go(b.dataset.goFig));
  head.querySelector(".fig-tools").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    const t = b.dataset.t;
    if (t === "3d") return ctx.open3d(ctx.map3d[id] || "all");
    if (!svg) return;
    if (t === "in" || t === "out") zoomBy(svg, t === "in" ? 1 / 1.3 : 1.3);
    else if (t === "fit") resetZoom(svg);
    else if (t === "full") { document.fullscreenElement ? document.exitFullscreen() : fig.requestFullscreen?.(); }
    else if (t === "svg" || t === "png") exportFig(svg, id, t);
  });
}

/* ---------------------------------------------------------------- focus chips */
function buildChips(svg, chips) {
  const counts = {};
  svg.querySelectorAll("[data-k]").forEach(e => { const k = e.dataset.k; if (TAG_NAMES[k]) counts[k] = (counts[k] || 0) + 1; });
  svg.querySelectorAll("path[data-c]").forEach(e => { const k = "c:" + e.dataset.c; if (TAG_NAMES[k]) counts[k] = (counts[k] || 0) + 1; });
  const keys = Object.keys(TAG_NAMES).filter(k => counts[k]);
  if (keys.length < 2) { chips.remove(); return; }
  chips.innerHTML = `<span class="fc-l">Highlight</span>` + keys.map(k => `<button data-f="${k}"><i style="background:${TAG_NAMES[k][1]}"></i>${TAG_NAMES[k][0]}</button>`).join("");
  chips.addEventListener("click", e => {
    const b = e.target.closest("[data-f]"); if (!b) return;
    const on = !b.classList.contains("on");
    chips.querySelectorAll("button").forEach(x => x.classList.remove("on"));
    svg.querySelectorAll(".hit").forEach(x => x.classList.remove("hit"));
    svg.classList.toggle("filtering", on);
    if (!on) return;
    b.classList.add("on");
    const f = b.dataset.f;
    const sel = f.startsWith("c:") ? `path[data-c="${f.slice(2)}"]` : `[data-k="${f}"]`;
    svg.querySelectorAll(sel).forEach(x => x.classList.add("hit"));
  });
}

/* ---------------------------------------------------------------- click-to-pin details */
function wirePin(svg, detail) {
  const clear = () => { svg.querySelectorAll(".sel").forEach(x => x.classList.remove("sel")); detail.classList.remove("on"); detail.innerHTML = ""; };
  svg.addEventListener("click", e => {
    if (svg.dataset.dragged === "1") return;
    const t = e.target.closest("[data-tip]");
    if (!t) return clear();
    clear(); t.classList.add("sel");
    detail.innerHTML = `<div class="fd-body">${t.dataset.tip}</div>${t.dataset.src ? `<div class="fd-src"><span>source</span><code>${t.dataset.src}</code></div>` : ""}<button class="fd-x" title="Close (Esc)">×</button>`;
    detail.classList.add("on");
    detail.querySelector(".fd-x").onclick = clear;
  });
  document.addEventListener("keydown", e => { if (e.key === "Escape" && detail.classList.contains("on")) clear(); });
}

/* ---------------------------------------------------------------- zoom / pan (viewBox) */
function vb(svg) { return svg.viewBox.baseVal; }
function wireZoom(svg, wrap) {
  const o = vb(svg); svg.dataset.vb0 = [o.x, o.y, o.width, o.height].join(" ");
  let drag = null;
  wrap.addEventListener("wheel", e => {
    if (!(e.ctrlKey || e.metaKey)) return;                     // plain scroll keeps scrolling the page
    e.preventDefault();
    const r = svg.getBoundingClientRect();
    zoomBy(svg, e.deltaY > 0 ? 1.12 : 1 / 1.12, (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
  }, { passive: false });
  svg.addEventListener("pointerdown", e => {
    if (!wrap.classList.contains("zoomed") || e.button !== 0) return;
    drag = { x: e.clientX, y: e.clientY, v: { ...vbObj(svg) }, moved: false }; svg.dataset.dragged = "0";
  });
  window.addEventListener("pointermove", e => {
    if (!drag) return;
    const r = svg.getBoundingClientRect(), dx = (e.clientX - drag.x) / r.width * drag.v.w, dy = (e.clientY - drag.y) / r.height * drag.v.h;
    if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 4) { drag.moved = true; svg.dataset.dragged = "1"; wrap.classList.add("panning"); }
    setVb(svg, drag.v.x - dx, drag.v.y - dy, drag.v.w, drag.v.h);
  });
  window.addEventListener("pointerup", () => { if (!drag) return; drag = null; wrap.classList.remove("panning"); setTimeout(() => { svg.dataset.dragged = "0"; }, 0); });
}
function vbObj(svg) { const v = vb(svg); return { x: v.x, y: v.y, w: v.width, h: v.height }; }
function setVb(svg, x, y, w, h) {
  const [x0, y0, w0, h0] = svg.dataset.vb0.split(" ").map(Number);
  w = Math.min(w, w0); h = Math.min(h, h0);
  x = Math.max(x0, Math.min(x, x0 + w0 - w)); y = Math.max(y0, Math.min(y, y0 + h0 - h));
  svg.setAttribute("viewBox", `${x} ${y} ${w} ${h}`);
  svg.parentElement.classList.toggle("zoomed", w < w0 - 0.5);
}
function zoomBy(svg, k, fx = 0.5, fy = 0.5) {
  const v = vbObj(svg), [, , w0] = svg.dataset.vb0.split(" ").map(Number);
  const w = Math.max(w0 / 6, v.w * k), h = v.h * (w / v.w);
  setVb(svg, v.x + (v.w - w) * fx, v.y + (v.h - h) * fy, w, h);
}
function resetZoom(svg) { const [x, y, w, h] = svg.dataset.vb0.split(" ").map(Number); setVb(svg, x, y, w, h); }

/* ---------------------------------------------------------------- export */
function exportFig(svg, id, kind) {
  const c = svg.cloneNode(true);
  const [x, y, w, h] = svg.dataset.vb0.split(" ").map(Number);
  c.setAttribute("viewBox", `${x} ${y} ${w} ${h}`); c.setAttribute("width", w); c.setAttribute("height", h);
  c.classList.remove("filtering"); c.querySelectorAll(".hit,.sel").forEach(e => e.classList.remove("hit", "sel"));
  const st = document.createElementNS(NS, "style");
  st.textContent = `text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}`;
  c.insertBefore(st, c.firstChild);
  const bg = document.createElementNS(NS, "rect"); Object.entries({ x, y, width: w, height: h, fill: "#ffffff" }).forEach(([k, v]) => bg.setAttribute(k, v));
  c.insertBefore(bg, st.nextSibling);
  const src = new XMLSerializer().serializeToString(c), name = `hrm-thinker-${id}`;
  const save = (href, ext) => { const a = document.createElement("a"); a.download = `${name}.${ext}`; a.href = href; a.click(); };
  if (kind === "svg") { const u = URL.createObjectURL(new Blob([src], { type: "image/svg+xml" })); save(u, "svg"); setTimeout(() => URL.revokeObjectURL(u), 1000); return; }
  const img = new Image(), u = URL.createObjectURL(new Blob([src], { type: "image/svg+xml" }));
  img.onload = () => {
    const cv = document.createElement("canvas"); cv.width = w * 2; cv.height = h * 2;
    const g = cv.getContext("2d"); g.scale(2, 2); g.drawImage(img, 0, 0, w, h); URL.revokeObjectURL(u);
    save(cv.toDataURL("image/png"), "png");
  };
  img.src = u;
}

/* ---------------------------------------------------------------- 3D link */
function open3D(view) {
  showTab("arch3d"); window.scrollTo(0, 0);
  let n = 0;
  const t = setInterval(() => {                                 // the 3D scene is built lazily on first open
    const b = document.querySelector(`#v3-bar [data-view="${view}"]`);
    if ((b && document.querySelector("#v3-info h5")) || ++n > 40) { clearInterval(t); if (b) b.click(); }
  }, 100);
}
