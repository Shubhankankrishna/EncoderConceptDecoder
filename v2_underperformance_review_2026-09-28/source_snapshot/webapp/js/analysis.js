/* Analysis tab (D-045): the server builds a digest of every page's data and asks Claude (local Claude Code CLI, headless,
   no tools) for a plain-language analysis; the answer is cached in webapp/data/analysis.json and rendered here. */
let AN = null, anPoll = null;
const escA = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function mdInline(s) {
  return escA(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>").replace(/(^|\s)_([^_]+)_(?=\s|[.,;:!?)]|$)/g, "$1<i>$2</i>");
}
function mdToHtml(md) {                                        // small Markdown subset: headings, lists, tables, quotes, rules, paragraphs
  const lines = md.replace(/\r/g, "").split("\n"), out = [];
  let i = 0, list = null, para = [];
  const flushP = () => { if (para.length) { out.push(`<p>${mdInline(para.join(" "))}</p>`); para = []; } };
  const closeL = () => { if (list) { out.push(`</${list}>`); list = null; } };
  while (i < lines.length) {
    const l = lines[i], t = l.trim();
    if (!t) { flushP(); closeL(); i++; continue; }
    let m;
    if ((m = t.match(/^(#{1,4})\s+(.*)$/))) { flushP(); closeL(); const lv = Math.min(4, m[1].length); const id = "an-" + m[2].toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 40);
      out.push(`<h${lv} id="${id}">${mdInline(m[2])}</h${lv}>`); i++; continue; }
    if (/^(-{3,}|\*{3,})$/.test(t)) { flushP(); closeL(); out.push("<hr>"); i++; continue; }
    if (t.startsWith("|") && i + 1 < lines.length && /^\|?\s*:?-{2,}/.test(lines[i + 1].trim())) {
      flushP(); closeL();
      const row = r => r.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
      const head = row(t); i += 2; const body = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) { body.push(row(lines[i])); i++; }
      out.push(`<div class="scroll-x"><table class="t an-t"><tr>${head.map(h => `<th>${mdInline(h)}</th>`).join("")}</tr>${body.map(r => `<tr>${r.map(c => `<td>${mdInline(c)}</td>`).join("")}</tr>`).join("")}</table></div>`);
      continue;
    }
    if ((m = t.match(/^>\s?(.*)$/))) { flushP(); closeL(); out.push(`<blockquote>${mdInline(m[1])}</blockquote>`); i++; continue; }
    if ((m = t.match(/^[-*•]\s+(.*)$/)) || (m = t.match(/^\d+[.)]\s+(.*)$/))) {
      flushP(); const kind = /^\d/.test(t) ? "ol" : "ul";
      if (list !== kind) { closeL(); out.push(`<${kind}>`); list = kind; }
      out.push(`<li>${mdInline(m[1])}</li>`); i++; continue;
    }
    closeL(); para.push(t); i++;
  }
  flushP(); closeL();
  return out.join("\n");
}

async function loadAnalysis(start) {
  try { AN = await (await fetch("api/analysis" + (start ? "?run=1" : ""), { cache: "no-store" })).json(); } catch (e) { AN = { error: "dashboard server not reachable" }; }
  renderAnalysis();
  clearTimeout(anPoll);
  if (AN.running) anPoll = setTimeout(() => loadAnalysis(false), 4000);
}
function renderAnalysis() {
  const root = document.getElementById("an-root"), bar = document.getElementById("an-bar"); if (!root || !AN) return;
  const a = AN.analysis;
  bar.innerHTML = `<button class="btn primary" id="an-run" ${AN.running ? "disabled" : ""}>${AN.running ? `<span class="an-spin"></span> Claude is analysing… ${AN.elapsed ?? 0} s` : a ? "↻ Ask Claude again" : "✦ Ask Claude for an analysis"}</button>
    <div class="an-meta"><div>${a ? `Written <b>${escA(a.generated)}</b> from the data as of that moment · ${escA(a.model)} · took ${a.seconds} s` : "No analysis yet."}</div>
      ${AN.error ? `<span class="an-err">Last attempt failed: ${escA(AN.error)}</span>` : ""}
      <span class="muted">Usually 1–3 minutes. It reads only the numbers on this dashboard; nothing touches the training.</span></div>`;
  document.getElementById("an-run").onclick = () => loadAnalysis(true);
  if (!a) {
    root.innerHTML = `<div class="card an-empty"><h4>What Claude will read</h4><p>Press the button above. Claude receives this digest of every page (architecture, screens, the hourly
      evaluations, the final evaluation, the gates and the running continuation) and writes a plain-language analysis.</p><pre class="log">${escA(AN.digest_preview || "")}</pre></div>`;
    return;
  }
  const html = mdToHtml(a.markdown);
  const heads = [...html.matchAll(/<h2 id="([^"]+)">(.*?)<\/h2>/g)].map(m => [m[1], m[2].replace(/<[^>]+>/g, "")]);
  root.innerHTML = `<div class="an-layout">
      <article class="card an-doc">${html}</article>
      <aside class="an-side">
        <div class="card"><h4>Contents</h4><nav class="an-toc">${heads.map(([id, t]) => `<a href="#${id}" data-an="${id}">${t}</a>`).join("")}</nav></div>
        <div class="card"><h4>What Claude read</h4><p class="muted" style="font-size:12.5px;margin:0 0 8px">${a.digest.length.toLocaleString()} characters of numbers from every page, as of ${escA(a.generated)}.</p>
          <details><summary>Show the digest</summary><pre class="log an-dig">${escA(a.digest)}</pre></details></div>
      </aside></div>`;
  root.querySelectorAll("[data-an]").forEach(x => x.onclick = e => { e.preventDefault(); document.getElementById(x.dataset.an)?.scrollIntoView({ block: "start" }); });
}
window.addEventListener("tabshown", e => { if (e.detail === "analysis") loadAnalysis(false); });
