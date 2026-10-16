# HRM Thinker dashboard — design record (first principles)

Static site served read-only by `webapp/server.py` on :8765 (v3's live run also has the v1-style engine on :8766).
Each decision cites the advisor call that settled it (`run_v3_19M/ADVISOR_LOG.md`, rows 13+ = webapp series).

## 1. Users and the questions, in the order they are asked (call #13)
One user (the owner / researcher), often checking at night, laptop or phone, while a 48 h run trains.
1. **Is the run healthy right now?** — seconds. Status, step, speed, ETA, anomalies.
2. **Is v3 learning better than v1 / v2?** — minutes. Matched-step (same tokens) and matched-time comparisons, gates.
3. **How does it work?** — deep reading. 2D figures, 3D model, what changed between versions.
4. **What did we decide and why?** — reference. Decisions, advisor log, reviews, tests, throughput record.

Consequences: the landing page answers Q1 when a run is live; Q2 gets its own cross-version page; version is a
filter inside Q3/Q4 pages, not a top-level silo; nothing a question needs may require switching version.

## 2. Information architecture (call #14)
| task (top nav) | pages (existing tab ids kept) | version |
|---|---|---|
| Live | `v3train` (v3 run) · `train` (v1 archive) · `v2train` (v2 run record) | per page |
| Compare | `compare` (new) | all |
| Architecture | `overview/arch2d/arch3d/concept` (v1) · `v2overview/v2arch2d/v2arch3d` · `v3overview/v3arch2d/v3arch3d` | chips |
| Results | `results`, `analysis`, `screens` (v1) · `v2train` record · v3 evals (live page table; 24 h / 48 h eval files) | chips |
| Record | `record` (new: DECISIONS, ADVISOR_LOG, REVIEW, draft) · `guide` | all |
Rules: every old tab id and deep link keeps working; the task buttons always show; the version switch stays as a
secondary header control; default landing = `v3train` when `v3_live.json` is readable and no tab was chosen before.

## 3. Chart contract (call #14)
One component, `js/chart.js` → `LC.chart(id, opts)`: series `{name, color, pts, dash, dots, w}`, refs (horizontal
lines), bands, vlines, fixed or auto axes, x label / name / formatter, hover and touch readout listing every series at
the nearest x. Comparison series are dashed (v1 blue #3b78c2, v2 purple #7a5bd0); v3 solid teal #168b76. Charts are
SVG with a uniform aspect ratio (text never stretches), height follows width.

## 4. Live data contract
`data/v3_live.json` (train.py, every 25 steps and after each eval) — status, history series, evals, table norms,
geometry, opened pieces. `data/ref_v1v2.json` (v3_design/make_ref_v1v2.py) — v1/v2 evals and training curves.
Browsers reject NaN: every live fetch goes through `fetchLive()` (svg.js), which maps NaN / Infinity to null.

## 5. Change log
- #13: v2 "Live training" tab embedded :8766, which now serves v3 → replaced by a frozen v2 run record; the v2 header
  pill no longer polls. v3 header pill reads the live run; v3 footer text updated.
- #14: chart component extracted from the live page into `js/chart.js` (touch support, x name/formatter added).
- #15: task navigation layer shipped (js/nav.js: TAB_META, 5 task buttons, sub-navigation with version chips that keep the
  sub-page; old tab ids and links unchanged). Verified: 19/19 ids route to their own section with the right task button,
  chips keep the sub-page, no console errors, landing with no hash / no saved tab → v3train. Bug found: nav.js's
  `renderNav` collided with telemetry.js's global of the same name → renamed `renderTaskNav`. Live page redraws only on new
  data (30 s poll) to keep the dashboard's own GPU use off the training run; header Live dot follows the v3 run.
- Compare page (js/compare.js): same-step / same-hours switch, run toggles, “at step S” table (1,217 / 1,792 / 2,376 /
  4,050 / latest) with v3 − v1, nine shared charts; ties render neutral. Header pill fixed on first load (the first
  “tabshown” fired before v3train.js registered).
- Record page (js/record.js + v3_design/sync_record.py): seven documents rendered by a built-in Markdown renderer (no
  third-party code), contents bar, wrapped tables.
- #16: **rolled back at the owner's request** ("keep the old design and style, just polish it"). The task navigation (nav.js),
  Compare and Record pages are unwired (files kept on disk, not loaded). The original version-switch tabs are back. Kept as
  repairs: the v2 tab (it showed v3's data from :8766) is a frozen v2 run record; the v3 header pill, footer and labels
  follow the live run; the v3 Live training tab (health strip, redraw only on new data); NaN-safe live fetch; js/chart.js.
  Rule from now on: polish within the existing design and style only.
