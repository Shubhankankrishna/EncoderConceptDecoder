/* Tabs, overview, figure navigation. */
let ARCH = null;

function showTab(id) {
  const ver = typeof metaVer === "function" ? metaVer(id) : id.startsWith("v4") ? "v4" : id.startsWith("v3") ? "v3" : id.startsWith("v2") ? "v2" : "v1";
  const cross = typeof TAB_META !== "undefined" && TAB_META[id] && TAB_META[id].ver === "all";
  document.documentElement.dataset.ver = ver;
  const title = cross ? "HRM Thinker · v1 · v2 · v3" : ver === "v4" ? "HRM Thinker v4 · 19.9M · proposed" : ver === "v3" ? "HRM Thinker v3 · 19M" : ver === "v2" ? "HRM Thinker v2 · 19M" : "HRM Thinker-25M";
  const h1 = document.querySelector(".brand h1"); if (h1) h1.textContent = title; document.title = title;
  const brand = document.querySelector("a.brand");
  if (brand) { const home = ver === "v4" ? "v4overview" : ver === "v3" ? "v3overview" : ver === "v2" ? "v2overview" : "overview"; brand.href = "#" + home; brand.dataset.tabLink = home; }
  const footer = document.querySelectorAll(".foot span");
  if (footer.length >= 2) {
    footer[0].textContent = ver === "v4" ? "HRM Thinker v4 · proposed architecture (draft_v4.md) · tested at tiny scale in v4_tiny/ · read-only dashboard" : ver === "v3" ? "HRM Thinker v3 · 48 h pre-training (run_v3_19M) · read-only dashboard" : "HRM Thinker-25M · read-only dashboard (webapp/server.py) · never touches the training run";
    footer[1].textContent = ver === "v4" ? "Data: data/arch_v4.json (exact counts from model_v4.py) · data/v4_tiny.json (CPU experiment, 2026-10-01)" : ver === "v3" ? "Live data: data/v3_live.json every 25 steps · evaluations hourly and at v1's eval steps" : "Data refresh: live every 5 s · losses every 50 steps · evaluations hourly";
  }
  document.querySelectorAll("#ver-switch [data-ver-go]").forEach(b => b.classList.toggle("on", b.dataset.verGo === ver));
  try { localStorage.setItem("hrm-tab-" + ver, id); } catch (e) {}
  if (typeof renderTaskNav === "function") renderTaskNav(id);
  else document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === id));
  document.querySelectorAll("section.tab").forEach(s => s.classList.toggle("on", s.id === "tab-" + id));
  try { localStorage.setItem("hrm-tab", id); } catch (e) {}
  history.replaceState(null, "", "#" + id);
  if (typeof setPill === "function") setPill();
  onTabShown(id);
  window.dispatchEvent(new CustomEvent("tabshown", { detail: id }));
}

function renderOverview(A) {
  const T = A.table, P = A.params, M = A.macs;
  document.getElementById("ov-stats").innerHTML = [
    ["compute params", fmtP(T.compute_params), "encoder + core + decoder"],
    ["stored params", fmtP(T.stored_total), "+ Engram table, embedding, codebook"],
    ["width d", A.ecfg.d_model, `${A.ecfg.n_heads} heads × ${A.ecfg.head_dim} · GQA ${A.ecfg.n_heads}:${A.ecfg.n_kv_heads}`],
    ["decoder reach", A.reach_dec + " tokens", `encoder L stream ${A.reach_enc}`],
    ["concepts", "1 per 4 tokens", `PQ ${A.S} × ${A.N} × ${A.ecfg.d_model / A.S}`],
    ["training FLOP / token", (6 * M.total / 1e6).toFixed(0) + "M", "6 × multiply-accumulates"],
  ].map(([l, v, s]) => `<div class="card stat"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
  const parts = [["encoder (non-emb.)", T.encoder_non_embedding, "#d6e6f7"], ["concept module", T.concept_module - T.codebook, "#d2efea"], ["decoder", T.decoder, "#fbdcc8"]];
  const stored = [["Engram table", T.engram_table, "#fff2bf"], ["embedding (tied)", T.token_embedding, "#e6e0f5"], ["codebook", T.codebook, "#dcefd6"]];
  const bar = (arr, tot) => `<div class="pbar">${arr.map(([n, v, c]) => `<div style="width:${100 * v / tot}%;background:${c}" title="${n}: ${fmtI(v)}">${v / tot > 0.12 ? n : ""}</div>`).join("")}</div>`;
  document.getElementById("ov-params").innerHTML = `
    <div style="font-size:13px;margin-bottom:4px">Compute: <b>${fmtI(T.compute_params)}</b></div>${bar(parts, T.compute_params)}
    <table class="t" style="margin:8px 0 12px">${parts.map(([n, v]) => `<tr><td>${n}</td><td>${fmtI(v)}</td><td>${(100 * v / T.compute_params).toFixed(1)} %</td></tr>`).join("")}</table>
    <div style="font-size:13px;margin-bottom:4px">Stored: <b>${fmtI(T.stored_total)}</b> (compute + lookups)</div>${bar([...parts, ...stored], T.stored_total)}
    <table class="t" style="margin-top:8px">${stored.map(([n, v]) => `<tr><td>${n}</td><td>${fmtI(v)}</td><td>${(100 * v / T.stored_total).toFixed(1)} %</td></tr>`).join("")}</table>
    <div class="muted" style="font-size:12px;margin-top:6px">Encoder: 6 S layers × ${fmtI(P.enc_layer_S)} + 2 F layers × ${fmtI(P.enc_layer_F)} + Engram compute ${fmtI(P.engram_proj)} + pooling ${fmtI(P.pool)} + router ${fmtI(P.crc_router)} + norm ${P.tok_norm}.
    Decoder: plan ${fmtI(P.dec_plan)} + 4 × ${fmtI(P.dec_layer)} + norm ${P.dec_out_norm}.</div>`;
  document.getElementById("ov-setup").innerHTML = `
    <span>objective</span><span>L_NTP + 0.1 · L_NCP + L_VQ</span>
    <span>optimiser</span><span>AdamW (0.9, 0.95), wd 0.1, clip 1.0</span>
    <span>schedule</span><span>peak 6e-4, warm-up 300, cosine → 10 %</span>
    <span>batch</span><span>2 × 16 accumulation × 1,024 = 32,768 tokens / step</span>
    <span>main run</span><span>4,050 steps ≈ 132.7M tokens, 24 h budget</span>
    <span>continuation</span><span>+4,050 steps (+24 h) from the final checkpoint: LR 6e-5 → 6e-4 (300 steps) → 6e-5, unseen rows</span>
    <span>loop depth</span><span>R ~ U{1..4} train, R = 4 eval</span>
    <span>data</span><span>264.4M tokens: language 45 %, symbolic 22 %, retrieval 13 %, code 10 %, long-range probes 10 %</span>
    <span>hardware</span><span>Apple M2, 16 GB, PyTorch MPS, float32</span>`;
  const mt = [["encoder", M.encoder, "#d6e6f7"], ["concept module", M.core, "#d2efea"], ["decoder", M.decoder, "#fbdcc8"], ["output head", M.head, "#e6e0f5"]];
  document.getElementById("ov-macs").innerHTML = bar(mt, M.total) +
    `<table class="t" style="margin-top:8px">${mt.map(([n, v]) => `<tr><td>${n}</td><td>${(v / 1e6).toFixed(2)}M</td><td>${(100 * v / M.total).toFixed(1)} %</td></tr>`).join("")}
     <tr><td><b>total</b></td><td><b>${(M.total / 1e6).toFixed(2)}M</b></td><td>× 6 = ${(6 * M.total / 1e6).toFixed(1)}M FLOP</td></tr></table>
     <div class="muted" style="font-size:12px;margin-top:6px">T = 1,024, mean R = 2.5; the concept module runs once per 4 tokens. From code/model.py · macs_per_token().</div>`;
  document.getElementById("ov-fig").appendChild(figOverview(A));
  document.getElementById("ov-journey").innerHTML = JOURNEY.map(([t, x, f]) => `<li><b>${t}</b>${x}<br><a href="#arch2d" data-fig="${f}">see the figure →</a></li>`).join("");
  document.getElementById("ov-files").innerHTML = `
    <span>model code</span><span><code>code/model.py</code>, <code>hrm_encoder.py</code>, <code>concept_core_v2.py</code>, <code>hrm_decoder.py</code></span>
    <span>training loop</span><span><code>code/train.py</code> (evaluation: <code>evaluate.py</code>)</span>
    <span>run logs</span><span><code>runs/main/</code> (train_log.jsonl, log.jsonl, stdout.log)</span>
    <span>decisions</span><span><code>DECISIONS.md</code> · resume guide <code>STATUS.md</code></span>
    <span>this dashboard</span><span><code>webapp/</code> (read-only server, D-036/D-037)</span>`;
}

function renderGuide() {
  document.getElementById("gd-read").innerHTML = GUIDE_READ;
  document.getElementById("gd-control").innerHTML = GUIDE_CONTROL;
  const cats = ["all", ...new Set(GLOSS.map(g => g[1]))];
  let cat = "all", q = "";
  const catsEl = document.getElementById("gd-cats");
  const draw = () => {
    const hits = GLOSS.filter(([t, c, x]) => (cat === "all" || c === cat) && (!q || (t + " " + x).toLowerCase().includes(q)));
    document.getElementById("gd-gloss").innerHTML = hits.map(([t, c, x]) => `<div class="g"><span class="cat c-${c}">${c}</span><b>${t}</b>${x}</div>`).join("") || '<div class="muted">No matching term.</div>';
    document.getElementById("gd-count").textContent = `${hits.length} of ${GLOSS.length} terms${cat !== "all" ? ` in “${cat}”` : ""}${q ? ` matching “${q}”` : ""}.`;
    catsEl.innerHTML = cats.map(c => `<button data-cat="${c}" class="${c === cat ? "on" : ""}">${c === "all" ? "All" : c[0].toUpperCase() + c.slice(1)}</button>`).join("");
  };
  catsEl.onclick = e => { const b = e.target.closest("[data-cat]"); if (b) { cat = b.dataset.cat; draw(); } };
  document.getElementById("gd-search").oninput = e => { q = e.target.value.trim().toLowerCase(); draw(); };
  draw();
}

function gotoFig(id) {
  showTab("arch2d");
  const nav = document.getElementById("fig-nav");
  nav.querySelectorAll("button").forEach(x => x.classList.toggle("on", x.dataset.id === id));
  const f = document.getElementById("fig-" + id);
  if (f) requestAnimationFrame(() => f.scrollIntoView({ block: "start" }));   // offset comes from figure scroll-margin-top (CSS, uses --hdr)
}

function renderFigs(A) {
  const nav = document.getElementById("fig-nav"), box_ = document.getElementById("figs");
  FIGS.forEach((f, i) => {
    const b = document.createElement("button"); b.innerHTML = `<span class="fn">${i + 1}</span>${f.t}`; b.dataset.id = f.id; if (!i) b.classList.add("on");
    b.onclick = () => gotoFig(f.id);
    nav.appendChild(b);
    const fig = document.createElement("figure"); fig.className = "fig"; fig.id = "fig-" + f.id;
    const h = document.createElement("h3"); h.className = "figt"; h.innerHTML = `<span class="figt-n">${i + 1}</span>${f.t}`; fig.appendChild(h);
    const I = FIG_INTRO[f.id];
    if (I) {
      const rf = document.createElement("div"); rf.className = "readfirst";
      rf.innerHTML = `<div><div class="rf-t">Read this first</div><ul>${I.b.map(x => `<li>${x}</li>`).join("")}</ul></div>
        <div class="keys">${I.k.map(([a, v]) => `<div><span>${a}</span><b>${v}</b></div>`).join("")}</div>`;
      fig.appendChild(rf);
    }
    fig.appendChild(f.fn(A));
    const cap = document.createElement("figcaption"); cap.innerHTML = f.cap; fig.appendChild(cap);
    box_.appendChild(fig);
    if (typeof figTools === "function") figTools(fig, f.id, i, FIGS.length);   // js/figtools.js (D-042)
  });
}

(async function init() {
  if (typeof renderTaskNav !== "function") document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => { showTab(b.dataset.tab); window.scrollTo(0, 0); });
  window.addEventListener("scroll", () => {                    // highlight the figure in view in the 2D sidebar
    if (!document.getElementById("tab-arch2d").classList.contains("on")) return;
    let cur = null; document.querySelectorAll("#figs figure.fig").forEach(f => { if (f.getBoundingClientRect().top < innerHeight * 0.4) cur = f.id.slice(4); });
    if (cur) document.querySelectorAll("#fig-nav button").forEach(x => x.classList.toggle("on", x.dataset.id === cur));
  }, { passive: true });
  const th = document.getElementById("theme-btn");
  const setTheme = t => { if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; };
  try { setTheme(localStorage.getItem("hrm-theme")); } catch (e) {}
  th.onclick = () => {
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    const cur = document.documentElement.dataset.theme || (dark ? "dark" : "light");
    const nxt = cur === "dark" ? "light" : "dark"; setTheme(nxt);
    try { localStorage.setItem("hrm-theme", nxt); } catch (e) {}
    if (lastData && charts) renderTrain();
  };
  ARCH = await (await fetch("data/arch.json")).json();
  window.ARCH = ARCH;
  renderOverview(ARCH);
  renderFigs(ARCH);
  renderGuide();
  document.addEventListener("click", e => {
    const a = e.target.closest("[data-fig]"); if (a) { e.preventDefault(); gotoFig(a.dataset.fig); return; }
    const v = e.target.closest("[data-ver-go]");
    if (v) { e.preventDefault(); let id = v.dataset.verGo === "v4" ? "v4overview" : v.dataset.verGo === "v3" ? "v3overview" : v.dataset.verGo === "v2" ? "v2overview" : "overview"; try { id = localStorage.getItem("hrm-tab-" + v.dataset.verGo) || id; } catch (x) {} showTab(id); window.scrollTo(0, 0); return; }
    const t = e.target.closest("[data-tab-link], a[href='#train']"); if (t) { e.preventDefault(); showTab(t.dataset.tabLink || "train"); window.scrollTo(0, 0); }
  });
  let start = (location.hash || "").slice(1);
  if (!start) { try { start = localStorage.getItem("hrm-tab") || ""; } catch (e) { start = ""; } }
  if (!start) start = typeof landingTab === "function" ? await landingTab("v3overview") : "overview";
  if (!document.getElementById("tab-" + start)) start = "overview";
  showTab(start);
  window.addEventListener("hashchange", () => { const h = location.hash.slice(1); if (document.getElementById("tab-" + h)) { showTab(h); window.scrollTo(0, 0); } });
  startPolling();
})();
