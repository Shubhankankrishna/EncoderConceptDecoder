/* Record page (DESIGN.md §2): renders the decision / review / advisor documents copied to data/record/ by
   v3_design/sync_record.py. Small built-in Markdown renderer (headings, paragraphs, lists, tables, code, quotes,
   emphasis, links) so the local site needs no third-party library. */
(function () {
  let INDEX = null, cur = null;
  const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  function inline(s) {
    s = esc(s);
    s = s.replace(/`([^`]+)`/g, (m, c) => `<code>${c}</code>`);
    s = s.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>");
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, t, u) => /^https?:/.test(u) ? `<a href="${u}" target="_blank" rel="noopener">${t}</a>` : `<span class="md-link" title="${u}">${t}</span>`);
    return s;
  }
  function md(text) {
    const L = text.replace(/\r/g, "").split("\n"), out = [];
    let i = 0;
    const isTable = j => /^\s*\|.*\|\s*$/.test(L[j] || "") && /^\s*\|?\s*:?-{2,}/.test(L[j + 1] || "");
    while (i < L.length) {
      const l = L[i];
      if (/^```/.test(l)) { const buf = []; i++; while (i < L.length && !/^```/.test(L[i])) buf.push(L[i++]); i++; out.push(`<pre class="md-pre"><code>${esc(buf.join("\n"))}</code></pre>`); continue; }
      const h = l.match(/^(#{1,6})\s+(.*)$/);
      if (h) { const n = h[1].length, id = "md-" + h[2].toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 60); out.push(`<h${Math.min(6, n + 1)} id="${id}" class="md-h${n}">${inline(h[2])}</h${Math.min(6, n + 1)}>`); i++; continue; }
      if (isTable(i)) {
        const row = s => s.trim().replace(/^\||\|$/g, "").split(/(?<!\\)\|/).map(c => inline(c.trim().replace(/\\\|/g, "|")));
        const head = row(L[i]); i += 2; const body = [];
        while (i < L.length && /^\s*\|/.test(L[i])) body.push(row(L[i++]));
        out.push(`<div class="md-tw"><table class="t md-t"><thead><tr>${head.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>${body.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
        continue;
      }
      if (/^\s*([-*]|\d+\.)\s+/.test(l)) {
        const ordered = /^\s*\d+\./.test(l), items = [];
        while (i < L.length && (/^\s*([-*]|\d+\.)\s+/.test(L[i]) || (/^\s{2,}\S/.test(L[i]) && items.length))) {
          const m = L[i].match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
          if (m) items.push({ d: Math.floor(m[1].length / 2), t: m[3] }); else items[items.length - 1].t += " " + L[i].trim();
          i++;
        }
        let html = "", depth = -1; const stack = [];
        items.forEach(it => {
          while (depth < it.d) { html += ordered && depth < 0 ? "<ol>" : "<ul>"; stack.push(ordered && depth < 0 ? "</ol>" : "</ul>"); depth++; }
          while (depth > it.d) { html += stack.pop(); depth--; }
          html += `<li>${inline(it.t)}</li>`;
        });
        while (stack.length) html += stack.pop();
        out.push(html); continue;
      }
      if (/^>\s?/.test(l)) { const buf = []; while (i < L.length && /^>\s?/.test(L[i])) buf.push(L[i++].replace(/^>\s?/, "")); out.push(`<blockquote>${inline(buf.join(" "))}</blockquote>`); continue; }
      if (!l.trim()) { i++; continue; }
      const buf = [l]; i++;
      while (i < L.length && L[i].trim() && !/^(#|```|\s*\||\s*([-*]|\d+\.)\s|>)/.test(L[i])) buf.push(L[i++]);
      out.push(`<p>${inline(buf.join(" "))}</p>`);
    }
    return out.join("\n");
  }
  async function open(id) {
    cur = id;
    const doc = INDEX.docs.find(d => d.id === id), body = document.getElementById("rec-body");
    document.querySelectorAll("#rec-list [data-doc]").forEach(b => b.classList.toggle("on", b.dataset.doc === id));
    body.innerHTML = `<div class="lc-empty">loading ${doc.title}…</div>`;
    const t = await (await fetch(`data/record/${id}.md?t=${Date.now()}`, { cache: "no-store" })).text();
    const html = md(t);
    const heads = [...html.matchAll(/<h([23]) id="([^"]+)" class="md-h[12]">(.*?)<\/h[23]>/g)].slice(0, 60);
    body.innerHTML = `<div class="rec-meta"><b>${doc.title}</b><span class="muted">${doc.source} · ${(doc.bytes / 1024).toFixed(0)} KB · modified ${doc.modified}</span></div>
      ${heads.length > 3 ? `<nav class="rec-toc">${heads.map(([, lv, hid, txt]) => `<a data-jump="${hid}" class="l${lv}">${txt.replace(/<[^>]+>/g, "")}</a>`).join("")}</nav>` : ""}
      <article class="md">${html}</article>`;
    try { localStorage.setItem("hrm-rec-doc", id); } catch (e) {}
  }
  async function load() {
    const root = document.getElementById("rec-root");
    try { INDEX = await (await fetch("data/record/index.json?t=" + Date.now(), { cache: "no-store" })).json(); }
    catch (e) { root.innerHTML = `<div class="lc-empty">No record documents yet: run <code>python3 v3_design/sync_record.py</code>.</div>`; return; }
    root.innerHTML = `<div class="rec-layout"><aside class="card rec-list" id="rec-list"><div class="sn-l">documents · synced ${INDEX.synced}</div>
      ${INDEX.docs.map(d => `<button data-doc="${d.id}"><b>${d.title}</b><span>${d.what}</span></button>`).join("")}</aside><div class="card rec-body" id="rec-body"></div></div>`;
    let first = INDEX.docs[0].id; try { const s = localStorage.getItem("hrm-rec-doc"); if (s && INDEX.docs.some(d => d.id === s)) first = s; } catch (e) {}
    open(first);
  }
  document.addEventListener("click", e => {
    const d = e.target.closest("#rec-list [data-doc]"); if (d) { open(d.dataset.doc); return; }
    const j = e.target.closest("[data-jump]"); if (j) { const t = document.getElementById(j.dataset.jump); if (t) t.scrollIntoView({ block: "start" }); }
  });
  window.addEventListener("tabshown", e => { if (e.detail === "record") load(); });
  if ((document.querySelector("section.tab.on") || {}).id === "tab-record") load();
})();
