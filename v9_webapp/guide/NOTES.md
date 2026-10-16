# V9 Guide (noob-friendly blog for S1-S6) — build notes

Goal (user request, 8 Oct 2026): "make a new pages of all the Experiments S1 to S6 ... first principle thinking and step by step ...
detailed explanation along with 2d and 3d architecture design in deep like a detailed blog with animation of all the variants ...
the reader is an extreme noob."

Decisions
- One long page, chapters = Prelude, World (GraphNav), S1..S6, Status, Glossary. Sticky TOC + progress bar.
- Source in guide/src, `python3 build.py` -> ../v9_guide.html (full page, opens locally / served) and guide_body.html (artifact body, no skeleton).
- Reuse Atlas tokens (../src/style.css :root) and variant descriptors (../src/variants.js, inlined as-is).
- 2D = SVG widgets (theme via CSS vars). 3D = three@0.160.0 ES module from cdn.jsdelivr.net/npm, NO importmap, own orbit controls + labels.
- Integrity: S1 = sealed results (S1_RESULTS.json). S2 numbers = development diagnostics "as of 8 Oct 2026". S3-S6 = designed, not run.
  Mandatory queue (I13 order Q1,Q3a,Q2,Q3b,Q4,Q5,Q6,QF) vs full column labelled. No verdict on K8.
- Nothing under v9_tiny/ touched; serve.py untouched; atlas untouched. Publish guide as a SEPARATE private Artifact at the end.
- Style: no em-dash asides, no "not X but Y", no emoji, plain sentences.

Status log (update as I go)
- [x] skeleton + css + core.js + build.py (data embed from runs/ at build time) + three_guide.js (model + tree scenes)
- [x] chapters written+tested: start, basics, world, S1, S2 (viz_00..03)
- [x] written, NOT yet browser-tested: S3 (viz_04), S4 (viz_05)
- [x] S3..S6, status, glossary, all widgets browser-tested (desktop light+dark, phone-width frame no h-scroll)
- [ ] final: advisor check, rebuild with fresh data, publish private Artifact (guide_body.html), kill http.server on :8791, delete phone_test.html (done)
- test: python3 -m http.server 8791 in v9_webapp; open http://127.0.0.1:8791/v9_guide.html?v=N ; tab is 'hidden' so rAF is paused -> use G.force(n,dt) for 2D and G3.render(n) for 3D
