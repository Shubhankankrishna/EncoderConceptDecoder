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
- #18 (2026-09-30): **v3 Live training rebuilt on v1's Live training components** (owner: "v1 is far better than v3").
  The v3 tab now uses the same pieces as `#train`: hero (step, 48 h progress with eval ticks and the 24 h milestone,
  finish ETA), verdict panel ("All N checks healthy" / "N things to watch", click to jump), two-stage plan strip,
  machine row (time left, next event, GPU sparkline, trainer, power, memory), six KPI tiles with sparklines and deltas,
  sticky section chips, health cards with "bad would be", and `LineChart` cards with Healthy / Watch / Problem /
  Diagnostic pills and a `?` explainer. Sections: Health checks · Live · v3 vs v1 · Losses · Concepts · System · Records.
  Thresholds are only those fixed before the run (E1–E4 in ref_v1v2.json gates, 24 GiB memory guard, swap baseline +2 GB);
  geometry, (t, Δ) table, NCP-by-depth and CRC mix are marked Diagnostic. Reference curves are clipped to ≈ 1.15 × the
  current step so v3's progress is readable. Per-step loss, trainer process, GPU % and battery come from :8766/api/live
  (read-only, CORS); the page degrades to data/v3_live.json alone if that server is down. Polls every 30 s, redraws only
  on new data, skips refreshes while the window is hidden (first load always draws). Old v3t-* markup removed from
  index.html; its CSS is left in place (unused). Verified: 17/17 tabs render, console clean, dark + light, 400 px width.
- #19 (2026-09-30): **v3 3D model decluttered** (owner: "make it professional and clear"). Label placement pass after every
  frame (js/view3d_v3d.js `updateLabelLOD`): labels are placed in priority order (selected part → plate titles → landmarks →
  details, nearer first) and any label that would overlap an already placed one is hidden; detail labels appear only when a
  module is focused or the camera is zoomed in; "Labels: all" still shows everything. Landmark labels renamed in plain
  English (Tokens in, Engram memory, Concepts C, Up to 3 passes, (t, Δ) table, Shared FFN, Plan inputs → decoder, Local
  token stream → decoder, Next-token loss); plate subtitles say what each stage does. Toolbar moved above the canvas (no
  longer covers the model), "Scratchpad" → "Concept module", pass buttons grouped under "Passes" with tooltips, "▶ Play
  passes". Live (t, Δ) table moved from an overlay on the model into the side panel as bars. Tour start button now uses
  the styled class (it rendered as a default button). Side panel opens with a "How to read this model" guide. v1 / v2 3D
  views untouched (CSS additions are scoped to #v3c-stage / #v3c-bar).
- #20 (2026-09-30): **v3 Live: recurrence telemetry** (owner: "measure the recurrence better, how much the shared
  Mobius FFN is used"). New section **Loop** (chip between Concepts and System) with a "Loop at a glance" row (update
  size per pass, passes 2/3 vs pass 1, angle ΔS₂↔ΔS₃ vs v1 1.25°, ρ₃ vs v1 0.99, 3 passes vs 1 pass, gradient at pass 1 ÷
  pass 3, shared-FFN W₂ size, table size), 14 charts (depth advantage ncp_M1 − ncp; ‖ΔS‖ and ‖S‖ per pass; relative update
  size train + eval; angle; ρ₃; gradient ratio; (t, Δ) total gain and pooled write gain α per pair; shortcut − main NCP;
  shortcut sampling shares; weight size of the zero-initialised parts; core attention τ per layer; NCP head τ) and a
  6-pair × 3-site (t, Δ) grid. "v3 vs v1" gains carry3/multi4 partial credit per answer token and hop by number of hops;
  Concepts gains dead codes re-seeded, H − NCP (predictability), the full CRC mix (e0/L/G), training codes-in-use.
  Every new pill is Diagnostic (no pre-registered thresholds); v1 values appear only as reference lines / explainer text.
  **Honesty rule**: the shared FFN's share of each write is not logged in this run (table `a` pools sites 2, 5, 6 =
  attention α, own-FFN α, shared-FFN α; `shared_w2_norm` is a weight size); the page says so in the section intro, the
  glance cell and the chart explainers. τ fields are named log_tau but evaluate.py stores exp(log_tau); attention τ
  multiplies logits (higher = sharper), NCP τ divides distances (lower = sharper) — two separate charts.
  **Data**: `data/v3_train_log.jsonl` and `data/v3_evals.jsonl` are read-only symlinks into run_v3_19M/runs/v3_48h
  (train_log.jsonl, log.jsonl); fetched with a cache-buster, parsed line by line (a half-written last line is dropped),
  redrawn only when their row count / last step changes. This replaces any :8766 dependency for these charts. Both files
  are also reachable through the phone gateway (behind its key). **Verification**: a headless Chrome check (CPU only,
  isolated profile, nice 19) found a real bug — `NOLOG` used before declaration inside `mk()`, which would have blanked the
  whole page — fixed; new chart blocks and setters are now wrapped in try/catch so a fault can only empty the new cards.
  Final headless render: 35 charts, 19 Diagnostic pills, 18 grid cells, 8 chips, no console errors. REF-load failure
  now shows a message instead of an endless "Loading…".
- #21 (2026-09-30): **loss-trend checks less twitchy** (owner: "too sensitive, not red every time"). The four loss trends (NTP, NCP, VQ, total) now use v1's rule at v3's 25-step cadence: mean of the last 100 steps vs the 100 steps ending 300 steps earlier; NTP uses the trainer's smoothed ntp_ema from the run log (fallback: raw values from v3_live.json). Thresholds unchanged (flat 0.002 / 0.003 / 0.005, rise 0.05 / 0.08). Replay over steps 300–3,075: old rule 2 red + 32 amber for NTP, new rule 0 + 0; NCP / VQ / total 24 / 10 / 27 amber → 0. The health card value and the NTP KPI tile show the same smoothed 300-step comparison.
- #22 (2026-09-30): **next two evaluations** in the v3 Live top card ("Next evaluations" cell and the verdict line): step and clock time for each, computed with train.py's own rule (evaluate at v1's eval steps, or 60 min after the last evaluation ended; the timer restarts after every evaluation, so hourly evaluations begin after step 4,050), assuming the current speed and ≈ 85 s per evaluation. The cell also shows the 24 h milestone evaluation time. Replay at 10:52: step 3,304 ≈ 11:00, then 3,484 ≈ 11:34; milestone ≈ Thu 01:10.
- #23 (2026-09-30): **Verdicts section** (owner: "a benchmark for whether looping works, whether the shared FFN and Engram
  are useful, and whether v3 is doing well vs v1"; keep the design, remove nothing). New chip "Verdicts" after Health
  checks: four cards (Is looping working? · Is the shared (Möbius) FFN useful? · Is Engram useful? · v3 vs v1), each a
  table of criterion · evidence · pill, in three tiers: switched on and moving (live telemetry), carries load (switch-off
  tests of the milestone / final evaluation: lang.<name>.<intervention>.delta_vs_intact for R1/R2a/R2b, table_off,
  shared_ffn_off, engram_off, engram_read2_off; items.<split>.<intervention>.<probe>.exact), worth its cost (the
  controlled arms; not run). **Pill rule**: green only for criteria fixed before the run (E1–E5, the step-300 opened
  check, the E2 line); amber = marginal / inside the draft's near-repetitive signature; grey = Measured, Pending 24 h,
  Not logged, Needs arm. Overall labels state the measured number, no invented bars. Kept out of checks(), so the hero
  counts do not change. **Data**: `data/v3_eval_24h.json` and `data/v3_eval_final.json` are read-only symlinks into the
  run folder (404 until written; final preferred; parsed in try/catch because final_eval.py rewrites the file after
  every part). **Audit**: all 33 charts' latest values recomputed from the raw files, no mismatch; v3_live.json and the
  run log agree (step 3,853, val 2.8269). Not viewed in a browser (extension disconnected; headless Chrome avoided —
  trainer footprint 16.9 GiB, swap near the flag). Nothing removed.
- #24 (2026-09-30): **phone layout + text polish** (owner: "not optimized for iPhone 16 Pro; texts flying away in Chrome").
  Phones (≤ 600 px) and phone landscape (height ≤ 480 px): the header is no longer sticky (it wrapped to four rows,
  ≈ 190 px); the section chips are the only sticky bar (`html { --hdr: 0 !important }` outranks the inline value
  telemetry.js writes, so chip jumps land on their headings); chips 10 × 14 px padding (≈ 40 px touch targets); tighter
  top card / KPI tiles / charts; verdict tables stack (criterion + pill, evidence below); key-value lists stack; iOS
  text-size-adjust fixed, fixed background off on phones. Applies to every tab on phones, desktop unchanged.
  Desktop fix: verdict-table evidence text had white-space: nowrap from `table.t td` and ran out of its card — now wraps
  in fixed columns (rule needed `table.t.v3-vtab` specificity); verdict summary pill wraps; watch-list reasons and KPI
  subtitles no longer cut with "…" on the v3 page; verdict cards top-aligned. Verified in Chrome (extension back):
  desktop 0 escaping / 0 truncated elements, no horizontal scroll; 402 px iframe: no horizontal scroll (398/398),
  header static, cards stacked. Nothing removed.
- #25 (2026-09-30): **away-from-home access plan** (owner: "check the dashboard outside home"). Recommended route: Tailscale (private tailnet; nothing opened to the public internet; the gateway already binds 0.0.0.0 and keeps its key gate). Installs and sign-in are the owner's. Not recommended: router port-forwarding (plain HTTP); public tunnels (cloudflared quick tunnel) only as the owner's own choice — the gateway serves the whole webapp folder, including draft_v3.md, proposed_v3.md, data/record/ and the symlinked run logs / eval files, behind an 8-character key. Mobile-data changes: phone_server.py trims /v3api/live to the last 600 steps (compact JSON; the page draws at most 600); v3train.js polls every 60 s when opened off-localhost (30 s on the Mac). Gateway restart needed for the trim (owner-run).
- #26 (2026-09-30): **bug fix — false "behind v1" red after v3 passed v1's last step** (owner spotted it). LC.interp returns null past v1's last evaluation (4,050), and JavaScript's number − null = number, so v1Gap returned v3's own loss as the gap (e.g. +2.76 mean, +3.55 WikiText at step 4,787) → the Keeping-pace check, the Gap-to-v1 tile / chart, the v3-vs-v1 verdict and the records table went red. v1Gap now returns null when v1 has no value; the check, tile and verdict card show v1's last same-step comparison (step 4,050: +0.059, TS +0.051, WK +0.066) and say so; probe and recall labels show v1's last value when there is no same-step one. Charts simply end their v1/v2 lines. Replay: steps 4,413 / 4,787 → no gap; pace check → green.
- #27 (2026-09-30): **bug fix — v3 Live tab emptied after the 21:27 memory re-exec** (owner: "dashboard is not showing data properly"). train.py builds v3_live.json's history / evals from in-process lists, so every re-exec or launcher relaunch restarts them (after 21:27: 3 training points, 0 evals). v3train.js now rebuilds history, evals and opened from the append-only run logs (data/v3_train_log.jsonl, data/v3_evals.jsonl) with write_dash's exact mapping whenever the logs hold more; scalars stay from the live file. Header pill and the v3 Design banner fall back to the last logged eval. Script tags versioned (?v=20260930r). No visual change. Verified in node on the real logs (279 training points, 30 evals, opened 100/200/300).
- #28 (2026-10-01): **new v3 tab "24 h evaluation"** (owner request: all details of the 24 h evaluation, same design style). js/v3eval.js renders data/v3_eval_24h.json (symlink) with the v1 Results tab's own components and classes (rs-kpis, rs-find, rs-bars, rs-mx, rs-syn, rs-lp, kvt, rs-gen): KPIs, findings, 10 switch-off tests per language, 1/2a/2b/3 pass schedules (language + probes), truncation, probe × intervention matrix with carry3/multi4 per-token and hop by depth, synthetic families intact vs core off, linear probes, codebook / mix / core dynamics (3, 2a, 2b), shortcut gradient audit, generations. No new CSS; no other tab changed. Verified in node on the real file (11 parts, no errors, no undefined/NaN).
- #29 (2026-10-01): **v4 (proposed) pages** (owner: "add the v4 proposed architecture to a dedicated page; same design and style as v1 and v3, 2D and 3D diagrams inspired by them; include the tiny-experiment results in detail and professionally so I can decide whether to go for it"). New version chip **v4 · proposed** (accent #b5476b) with four tabs cloned from v3's markup and classes:
  - **Overview** (`#v4overview`): stats, a status card with the verdict, "Why this design" (v1/v3 measurement → v4 decision), three part cards with v3 → v4 counts, a detail grid (pool / think / write, objective, parameters, gates G1–G5 and the k = 6 gate), and the arms table.
  - **How it works** (`#v4arch2d`, js/v4figs.js): 11 figures in v3's drawing vocabulary (svg.js helpers, legend3d) with figTools: whole model, encoder loop and shared FFN, Engram one table / four reads, pooler 6 → 1 (with v3's trained per-slot pool and the priced candidates), k = 4 / 6 / 8 measured, core 4 × 2, two-level codebook and §6.2 frames, shifted plan, losses, budget, gates and arms.
  - **3D model** (`#v4arch3d`, js/view3d_v4.js): v3's engine (view3d_v3d.js) with a rebuilt scene: 18 tokens → 3 concepts, F-layer loop rings, encoder shared FFN, router, attention pooler with 4 head markers, pooler FFN, core 4 layers with a 2-marker pass ring and no table, coarse (32 × 32) and PQ (32 × 256) codebook plates, four losses, decoder with the shifted-plan builder and two Engram read blocks fed from the same table, a 10-step guided tour, passes 2 / 1 animation.
  - **Tiny experiment & verdict** (`#v4tiny`, js/v4.js): verdict (three questions, GO_NOGO.md), KPIs, findings, tier 1 (15 tests, 13 mutants), the design flaw (what it touched and did not), learning curves (LC.chart), switch-off matrix with paired SEs, the decoder-reads memorisation table, pooler within-slot ratios and slot profiles, tier 3 (matched-token poolers, finals, k6 − k4 chart), the brief part by part, draft implications, launch checklist, cost to v3 (heartbeat chart), setup.

  **Pill rule** (as #23): green only for deterministic, pre-registered checks (ledger, tests, mutants, bit-identity); tiny-scale learning results are grey "Measured"; amber = untested / owner decides / at the gate; red = negative sign (decoder reads).

  **Data contract:**
  - `data/arch_v4.json` ← `v4_tiny/code/make_arch_v4.py` (exact counts from model_v4.py at full size, k = 8 / gconv variants, the recommended main run 19,175,576);
  - `data/v4_tiny.json` ← `v4_tiny/code/make_v4tiny_json.py` (all arms' evals, finals, decompositions, recall and reads checks, tests, mutants, v3 heartbeats);
  - read-only symlinks `data/draft_v4.md`, `data/v4_REPORT.md`, `data/v4_GO_NOGO.md`.

  **Plumbing:**
  - app.js `showTab`: v4 branch in the version detection, title, brand home, footer and version-chip default;
  - telemetry.js: the header pill shows the live v3 run on v4 pages;
  - CSS limited to the accent, the id-scoped 3D rules (#v4c, #v4c-bar, #v4c-stage) and `.v4-proposal table.t td { white-space: normal }`.

  **Verification:**
  - Chrome: 22 / 22 tabs (v1–v4) route to their own section with the right version and tab button; no console errors on load or across tab switches.
  - All 11 figures render with no figure error. Four were re-checked after edits: a bottom note was moved below the frames, frame titles were shortened, and table texts were shortened to fit.
  - 3D: all 10 tour steps set their panel; views enc / core / dec / all, explode, labels, flow, passes 2 / 1 and reset all work.
  - 402 px: no horizontal page scroll on any v4 tab; header static.

  **Numbers pass:** two corrections came out of it.
  - Final token counts are 5.00M / 7.42M (from final.json), not 4.93 / 7.35M.
  - The encoder-reads range is +0.14 to +0.17. Coarse-code ranges are labelled by the arms they describe.

  No v1 / v2 / v3 content changed. The only shared-label change is the shorter v4 chip text.
