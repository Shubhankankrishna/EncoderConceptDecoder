/* Task-first navigation (DESIGN.md §2, advisor call #14): five task buttons in the header, a sub-navigation row with the
   pages of the current task and version chips. Every existing tab id keeps working; this is a layer over showTab(). */
const TAB_META = {
  // Live
  v3train:   { task: "live", ver: "v3", label: "v3 run · live" },
  train:     { task: "live", ver: "v1", label: "v1 run (archive)" },
  v2train:   { task: "live", ver: "v2", label: "v2 run record" },
  // Compare
  compare:   { task: "compare", ver: "all", label: "v1 · v2 · v3" },
  // Architecture (sibling groups keep the sub-page when the version changes)
  overview:   { task: "arch", ver: "v1", label: "Overview", sib: "ov" },
  arch2d:     { task: "arch", ver: "v1", label: "How it works (2D)", sib: "2d" },
  arch3d:     { task: "arch", ver: "v1", label: "3D model", sib: "3d" },
  concept:    { task: "arch", ver: "v1", label: "Concept core screens", sib: "cc" },
  v2overview: { task: "arch", ver: "v2", label: "Overview", sib: "ov" },
  v2arch2d:   { task: "arch", ver: "v2", label: "How it works (2D)", sib: "2d" },
  v2arch3d:   { task: "arch", ver: "v2", label: "3D model", sib: "3d" },
  v3overview: { task: "arch", ver: "v3", label: "Overview", sib: "ov" },
  v3arch2d:   { task: "arch", ver: "v3", label: "How it works (2D)", sib: "2d" },
  v3arch3d:   { task: "arch", ver: "v3", label: "3D model", sib: "3d" },
  // Results
  results:   { task: "results", ver: "v1", label: "v1 final evaluation" },
  analysis:  { task: "results", ver: "v1", label: "v1 analysis" },
  screens:   { task: "results", ver: "v1", label: "v1 screens & gates" },
  // Record
  record:    { task: "record", ver: "all", label: "Decisions & logs" },
  guide:     { task: "record", ver: "v1", label: "Guide & glossary" },
};
const TASKS = [
  ["live", "Live", '<path d="M3 12h4l3-8 4 16 3-8h4"/>', "v3train"],
  ["compare", "Compare", '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>', "compare"],
  ["arch", "Architecture", '<path d="M12 2 3 7v10l9 5 9-5V7z"/><path d="M3 7l9 5 9-5M12 12v10"/>', "v3overview"],
  ["results", "Results", '<path d="M12 3l1.8 4.9L19 9.7l-5.2 1.8L12 16.5l-1.8-5L5 9.7l5.2-1.8z"/>', "results"],
  ["record", "Record", '<path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 21V5M8 7h7"/>', "record"],
];
const VER_NAME = { v1: "v1 · 25M", v2: "v2 · 19M MoE", v3: "v3 · 19M" };

function metaVer(id) {
  id = String(id || "");
  const m = TAB_META[id];
  if (m && m.ver !== "all") return m.ver;
  if (m) return "v3";                                              // cross-version pages use the current (v3) styling
  return id.startsWith("v3") ? "v3" : id.startsWith("v2") ? "v2" : "v1";
}

function lastInTask(task) {
  try { const t = localStorage.getItem("hrm-task-" + task); if (t && TAB_META[t]) return t; } catch (e) {}
  return TASKS.find(t => t[0] === task)[3];
}

function renderTaskNav(id) {
  const meta = TAB_META[id] || { task: "arch", ver: metaVer(id) };
  const nav = document.getElementById("tabs");
  if (nav && !nav.dataset.tasks) {
    nav.dataset.tasks = 1;
    nav.innerHTML = TASKS.map(([k, n, icon]) => `<button data-task="${k}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">${icon}</svg>${n}${k === "live" ? ' <span class="tab-live" id="tab-live-dot"></span>' : ""}</button>`).join("");
    nav.onclick = e => { const b = e.target.closest("[data-task]"); if (b) { showTab(lastInTask(b.dataset.task)); window.scrollTo(0, 0); } };
  }
  document.querySelectorAll("#tabs [data-task]").forEach(b => b.classList.toggle("on", b.dataset.task === meta.task));
  try { localStorage.setItem("hrm-task-" + meta.task, id); } catch (e) {}
  // sub-navigation: pages of this task (+ version chips for Architecture)
  let sub = document.getElementById("subnav");
  if (!sub) { sub = document.createElement("div"); sub.id = "subnav"; sub.className = "subnav"; document.querySelector("main").prepend(sub); }
  const pages = Object.entries(TAB_META).filter(([, m]) => m.task === meta.task);
  let html = "";
  if (meta.task === "arch") {
    const vers = ["v3", "v2", "v1"];
    html += `<div class="sn-group"><span class="sn-l">version</span>${vers.map(v => `<button class="sn-chip ${v === meta.ver ? "on" : ""} sn-${v}" data-go-ver="${v}">${VER_NAME[v]}</button>`).join("")}</div>`;
    html += `<div class="sn-group">${pages.filter(([, m]) => m.ver === meta.ver).map(([k, m]) => `<button class="sn-page ${k === id ? "on" : ""}" data-go="${k}">${m.label}</button>`).join("")}</div>`;
  } else if (pages.length > 1) {
    html += `<div class="sn-group">${pages.map(([k, m]) => `<button class="sn-page ${k === id ? "on" : ""} sn-${m.ver}" data-go="${k}">${m.label}</button>`).join("")}</div>`;
  }
  sub.innerHTML = html;
  sub.hidden = !html;
  sub.onclick = e => {
    const g = e.target.closest("[data-go]"); if (g) { showTab(g.dataset.go); return; }
    const v = e.target.closest("[data-go-ver]"); if (!v) return;
    const cur = TAB_META[id], want = v.dataset.goVer;
    const sib = Object.entries(TAB_META).find(([, m]) => m.task === "arch" && m.ver === want && m.sib === cur?.sib);
    showTab(sib ? sib[0] : want === "v1" ? "overview" : want + "overview");
  };
}

/* default landing: only when nothing was chosen before (no hash, no saved tab) — the live run if one is readable */
async function landingTab(fallback) {
  try {
    const d = await Promise.race([fetchLive("data/v3_live.json"), new Promise((_, r) => setTimeout(() => r(0), 400))]);
    if (d && d.t && Date.now() / 1000 - d.t < 3 * 3600) return "v3train";
  } catch (e) {}
  return fallback;
}
