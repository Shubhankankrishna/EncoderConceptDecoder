# Progress (read first after a context compaction)
- Papers read: 42/42 — READING PHASE COMPLETE (41 folder PDFs + user-added Sparse-Layers-Looped). Full-read vs skimmed parts disclosed per note header (bibliographies always skipped; some appendices/tables skimmed where stated: Loopie App. E, CALM Thm 2–3 proofs, EBT App. F–I, Epiplexity appendices, Latent Space §1–3/4.2/5.4–5.7, DeepSeek §3–5, AI-inside-AI App. A/C/E, GradCuit App. C, RecursiveMAS App. E/F, Explorative App. D–F, EBM-CoT appendices).
- Advisor calls: 46 (ADVISOR_LOG.md; 40 for the concept designs + 6 for the v2 model document). Plan: 1 per loop paper (+13 → 26), 1 per peripheral pair (+7 → 33), 5 designs + 5 reviews + 3 webapp + 1 final ≈ 47 (> 40).
- Disclosed exceptions to "read everything": bibliographies; Loopie appendix E (model-generated IMO proofs, checked by search).
- Files: LOOP_PAPERS_NOTES.md (per-paper blocks), DESIGNS.md (rubric, baseline audit, running synthesis), ADVISOR_LOG.md.
- Also read from outside the folder (user named them): extra/DeltaAttentionResidual, extra/Qwen4_tech_report (GR section), extra/Attention_Residuals — read at design phase.
- Monitor: self-run `bash code/opus_check.sh` every ~2 papers; swap watch (flag at 5 GB → notify only); next eval ~12:30.
- Resumed 12:20 after a usage-limit stop. Remaining: 23 papers, then design phase (5 designs + reviews + scores), then webapp page.


## RESUME (reading done, 2026-09-28 12:50)
- Advisor was 'temporarily overloaded' mid-turn (after note #25) — calls #21+ pending. Next: advisor #21 on the full reading synthesis (notes #25–#42 + new DESIGNS ingredients) BEFORE any design work.
- DONE 12:53: 3 extra residual papers read (notes X1–X3) + RESIDUAL SLOT synthesis table appended to DESIGNS.md.
- Then design phase: 5 designs (one = minimal fix of current S-space loop), each with C2 mechanism, grounding target, schedule, PQ/head layout, BPTT horizon, R distribution, params + MACs at d 256 / FFN 1024; 1 advisor call each; then 5 reviews/scores (1 call each); then webapp page; final call. Target > 40 calls.

- 13:12 DESIGN PHASE: C7 denominator + row 0 computed (core_budget.py); design-space v1 fixed (#22, #23); D1 v2 revised after #24; D2–D5 DRAFT v1 written with full numbers (pre-review — advisor overloaded at 13:08). NEXT TURN: #25 D1 v2 review, then reviews of D2–D5 drafts (1 call each → #26–#29), revisions, then scoring (5 calls), webapp (3), final → > 40.

## STATUS 2026-09-28 13:34 — COMPLETE
- 42/42 papers + 3 residual papers read (notes: LOOP_PAPERS_NOTES.md).
- 5 designs, consolidated spec with numbers: DESIGNS_FINAL.md (budgets.json via core_budget.py).
- Scores vs today (scores.json): D3 70.5 · D2 65.0 · D4 60.5 · D5 59.5 · D1 59.0 · today 41.5 — projections, not results.
- Webapp page: http://127.0.0.1:8765/?v=67#concept (js/concept.js, js/concept3d.js, data/concept_v2.json, css additions; backups of
  index.html/style.css in the scratchpad as *.pre_cv2).
- Advisor: 34 logged consultations + 3 overloaded attempts (logged, not counted); stopped short of 40 per the user's "faster, no
  over-engineering" request.
- Open items: none of the designs has been trained or evaluated; the per-pass eval sheet E1–E15 is the test plan.

## STATUS 2026-09-28 13:52 — D6 ADDED (user request: best of D2 + D3 + D4)
- D6 LOOPFORMER-MoE-DS: DESIGN_D6.md (first-principles steps 1–5) + DESIGNS_FINAL.md section; 4.92M params · 7.20M MACs/token at R 6
  (21.2 %) · training ×1.40; score 70.0 vs D3 70.5 (tie — the rubric charges for moving parts; D6's extra C2 point is an argument).
- Decision procedure between D3 and D6: E16 (routing divergence across windows = the carry gets a job), A4 (dense-FFN twin = does looped
  MoE matter), A2 (consistency off = does the LoopFormer piece add anything); E17 = LoopFormer drift in the final window.
- Page: http://127.0.0.1:8765/?v=70#concept (D6 colour green #3a9a4a).
- Advisor (concept phase): 40 logged consultations (+3 overloaded attempts logged, not counted); v2 phase: #41–#46.

## v2 model document (user request 2026-09-28 evening: D6 confirmed, d 256, Engram 80M, ~19M, NextLat in the decoder)
- Document: model_v2/MODEL_V2_19M.md (full architecture, steps 1–10 with first-principles reasoning, review §13 with a per-step
  advisor column, Appendix A = arch_v2.json for the 2D/3D pages).
- Numbers from code: model_v2/v2_budget.py → v2_budget.json (rows A–F, shares, R-sweep); model_v2/make_arch_v2.py → arch_v2.json.
- Recommended row C: encoder 7.39M · core D6 with 16 experts 7.92M · decoder 3.51M = 18.83M compute params (39/42/19 %);
  Engram 8 × 208,333 × 48 = 80.0M; d_c 384 kept (PQ 48 × 128 × 8); NextLat on out_norm(D), d 1, λ_h = λ_KL = 1 (ψ 0.66M, training only).
- Honest corrections: training compute ≈ today's (D6 K̄ 10), so tokens/24 h ≈ today's; tok/s ≤ 1,660–1,830 is an upper bound
  (throughput screen, floor 1,200); old C7 cap redefined (core 35 % of forward MACs at R 6).
- Advisor: #41–#46 (v2).
