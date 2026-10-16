# Experiment chronology

Inventory captured 2026-10-16T09:11:24.049000+00:00. These are present-day imports of preserved snapshots, not reconstructed historical Git commits.

Read the saved configuration and evidence together. Versions use different datasets, budgets, tokenizers, and seeds; this is not a cross-version leaderboard.

[Every indexed execution](docs/RUNS.md) · [Machine-readable catalog](experiments/catalog.json) · [Local artifact inventory](artifacts/manifest.jsonl) · [Working guide](docs/WORKFLOW.md)

| Stage | Purpose | Evidence status | Execution groups | Start here |
| --- | --- | --- | ---: | --- |
| Original encoder | Local/global encoder design and isolated contracts | Implementation and component evidence recorded | 0 | [Encoder Design/HRM_THINKER_ENCODER_v1_SPEC.md](Encoder%20Design/HRM_THINKER_ENCODER_v1_SPEC.md) · [Encoder Design/ARCHITECTURES_AND_PAPERS.md](Encoder%20Design/ARCHITECTURES_AND_PAPERS.md) |
| Original concept module | Recurrence, writable state, steering, and end-to-end probes | Component experiment results recorded | 5 | [Concept Design/HRM_THINKER_Concept_Module.md](Concept%20Design/HRM_THINKER_Concept_Module.md) · [Concept Design/RESULTS_E2E.txt](Concept%20Design/RESULTS_E2E.txt) |
| Original decoder | Confined versus stacked decoder comparisons | Component results recorded | 15 | [Decoder Design/HRM_THINKER_DECODER_v1_SPEC.md](Decoder%20Design/HRM_THINKER_DECODER_v1_SPEC.md) · [Decoder Design/RESULTS_DECODER.txt](Decoder%20Design/RESULTS_DECODER.txt) |
| V1 / THINKER-25M | Assemble and train the original encoder–concept–decoder model | Main final evaluation recorded; continuation paused | 6 | [run_m2_25M/STATUS.md](run_m2_25M/STATUS.md) · [run_m2_25M/DECISIONS.md](run_m2_25M/DECISIONS.md) |
| V2 / THINKER-19M | Recurrent mixture-of-experts core and NextLat auxiliary | Paused run with checkpoint evaluation and audits | 2 | [run_v2_19M/STATUS.md](run_v2_19M/STATUS.md) · [v2_underperformance_review_2026-09-28/REPORT.md](v2_underperformance_review_2026-09-28/REPORT.md) |
| V3 / THINKER-19M | Repair gradient flow and resource balance with dense recurrence | Main final evaluation recorded | 17 | [run_v3_19M/DECISIONS.md](run_v3_19M/DECISIONS.md) · [v3_design_review_2026-09-29/RESULTS.md](v3_design_review_2026-09-29/RESULTS.md) |
| V4-tiny | Revised pooling, PQ, recurrence, and matched transformer controls | Pilot and comparison results recorded | 23 | [v4_tiny/REPORT.md](v4_tiny/REPORT.md) · [v4_tiny/VS_LLM_REPORT.md](v4_tiny/VS_LLM_REPORT.md) |
| V4.1-tiny | Private token-memory retrieval and concept-dependence controls | Registered and exploratory comparisons recorded | 28 | [v41_tiny/ASSESSMENT.md](v41_tiny/ASSESSMENT.md) · [v41_tiny/REPORT.md](v41_tiny/REPORT.md) |
| V5-tiny | Foundation, latent replacement, persistent control, and repair | Multiple training and evaluation stages recorded | 39 | [v5_tiny/REPORT.md](v5_tiny/REPORT.md) · [v5_tiny/RESULT_TABLES.md](v5_tiny/RESULT_TABLES.md) |
| V6 / V6.1 | Controlled replacement and time-matched latent repair; later search proposal | V6 pipeline completed; V6.1 remains a separate proposal | 29 | [v6_tiny/REPORT.md](v6_tiny/REPORT.md) · [v6_tiny/SPEC_COMPLIANCE.md](v6_tiny/SPEC_COMPLIANCE.md) |
| V7-tiny | Independent foundation/refit/control/search queue and flat controls | Stopped by user with partial results | 107 | [v7_tiny/REPORT.md](v7_tiny/REPORT.md) · [v7_tiny/README.md](v7_tiny/README.md) |
| V8 competence pilot | Competence-first foundation and gated downstream work | Pilot completed; competence gate failed | 10 | [v8_tiny/REPORT.md](v8_tiny/REPORT.md) · [v8_tiny/AUDIT_2026-10-06.md](v8_tiny/AUDIT_2026-10-06.md) |
| V8 full-exposure revision | Full-exposure diagnostic revision with explicit cache/JSON recoveries | Extended pilot completed | 51 | [v8_tiny_12h/REPORT.md](v8_tiny_12h/REPORT.md) · [v8_tiny_12h/RECOVERY_2026-10-06.md](v8_tiny_12h/RECOVERY_2026-10-06.md) · [v8_tiny_12h/RECOVERY_JSON_2026-10-06.md](v8_tiny_12h/RECOVERY_JSON_2026-10-06.md) |
| V9-tiny | H-NCP hypotheses, concept search, expert iteration, and LSL | Q1–Q4 evidence; original later queues stopped; reduced Q5 completed | 136 | [v9_tiny/runs_q5_12h/REPORT.md](v9_tiny/runs_q5_12h/REPORT.md) · [review_v9.md](review_v9.md) |
| V10 / V10-10M | RCDL implementation, public-data pretraining, and corpus audit | Pretraining stopped for transition to V10.1 | 4 | [v10_10M/ARCHITECTURE.md](v10_10M/ARCHITECTURE.md) · [v10_10M/audit/REPORT.md](v10_10M/audit/REPORT.md) · [v10_10M/DATASETS.md](v10_10M/DATASETS.md) |
| V10.1 | Clean corpus, interface contracts, and forecast supervision | Pretraining stopped for transition; validation recorded | 9 | [v10_1/README.md](v10_1/README.md) · [v10_1/VALIDATION_REPORT.md](v10_1/VALIDATION_REPORT.md) |
| V10.2 | Released concept-history retrieval, integration, and throughput checks | Implementation and production-smoke evidence recorded | 2 | [v10_2/README.md](v10_2/README.md) · [v10_2_review.md](v10_2_review.md) |
| V10.3-tiny | Mixture-likelihood exploration, forecast map, codebook fix, and sleep | Paired training/evaluation and sleep evidence recorded | 24 | [V10_3_tiny_REPORT.md](V10_3_tiny_REPORT.md) · [v10_3_tiny/REPORT_TABLES.md](v10_3_tiny/REPORT_TABLES.md) |
| V11-tiny | Code-free two-visit core and concept-anchored prediction bridge | Multi-arm training and paired comparisons recorded | 42 | [v11_tiny/COMPARISON.md](v11_tiny/COMPARISON.md) · [v11_tiny/REPORT.md](v11_tiny/REPORT.md) |
| V12-tiny | Simpler E→C→D trunk, categorical NCP, and frozen-base controller | Base and Phase-2 evidence recorded, including aborted work | 57 | [v12_tiny/RESULTS.md](v12_tiny/RESULTS.md) · [v12_tiny/FIRST_PRINCIPLES.md](v12_tiny/FIRST_PRINCIPLES.md) · [v12_tiny/DESIGN_STEPS.md](v12_tiny/DESIGN_STEPS.md) |
| V13-tiny-v2 | Calibrated task ladders, concept diagnostics, and continual-learning tests | Final base evaluation and continual-stage evidence recorded | 22 | [v13_tiny/REPORT_TABLES.md](v13_tiny/REPORT_TABLES.md) · [v13_tiny/DEVIATIONS.md](v13_tiny/DEVIATIONS.md) · [Dataset_v13.md](Dataset_v13.md) |
| V14-tiny | EMA multi-horizon targets, throughput shrink, diagnostics, and continual learning | Base, diagnostic, and continual pipeline completed | 19 | [v14_tiny/REPORT_TABLES.md](v14_tiny/REPORT_TABLES.md) · [v14_tiny/DEVIATIONS.md](v14_tiny/DEVIATIONS.md) |
| V15 proposal | Frozen hierarchical codec, predictive plans, and concept-depth memory | Proposed and analytically reviewed; not implemented or trained | 0 | [v15_Architecture.md](v15_Architecture.md) · [v15_design/review_from_scratch.md](v15_design/review_from_scratch.md) · [v15_design/evidence_manifest.json](v15_design/evidence_manifest.json) |

## Interpretation notes

- **Original encoder:** Component tests are not whole-model training results.
- **Original concept module:** ConceptCoreV2 is a component in whole-model V1, not whole-model V2.
- **Original decoder:** The original ZIP bundle stays local; its unpacked code and evidence are published.
- **V1 / THINKER-25M:** Dashboard links can reference later work. The continuation's PAUSE marker does not invalidate the main run's final evaluation.
- **V2 / THINKER-19M:** Checkpoint-matched audit comparisons differ from final V1 test readings.
- **V3 / THINKER-19M:** The earlier proposal and actual trained model differ. Audit snapshots are not independent replications.
- **V4-tiny:** Structural contracts and pilot outcomes establish different claims; consult each run's configuration.
- **V4.1-tiny:** Post-hoc plan-off controls are exploratory; interventions can introduce distribution shifts.
- **V5-tiny:** A report containing planned RL stages does not establish that every stage succeeded or completed.
- **V6 / V6.1:** Equal-attempt results do not imply an equal-time advantage. The V6.1 draft is not proof of implementation.
- **V7-tiny:** Saved pipeline tests are engineering evidence; the full scientific queue was interrupted.
- **V8 competence pilot:** A completed pilot with a failed gate does not establish success of later planned stages.
- **V8 full-exposure revision:** Recoveries reused foundations within the original budget and are not new independent training seeds. Large result payloads stay local with compact digests.
- **V9-tiny:** S stages and Q queues differ. Downstream seeds can share one foundation. Reduced Q5 uses unequal budgets and does not establish equal-compute search superiority.
- **V10 / V10-10M:** The original corpus audit records serialized-byte wrappers and restricted math coverage; inherited losses require that context.
- **V10.1:** Validation success does not establish completion of the intended 16-hour pretraining budget.
- **V10.2:** No full scientific pretraining run is inferred from sustained validation or a production-smoke checkpoint.
- **V10.3-tiny:** Use the paired comparisons and registered controls; implementation checks alone are not performance evidence.
- **V11-tiny:** V11_Paper is an evidence bundle distinct from the v11_Papers literature folder. Frozen predecessors are not new runs.
- **V12-tiny:** Keep aborted and pilot runs separate. Inspect Phase-2 logs for each controller stage rather than assuming the whole proposal completed.
- **V13-tiny-v2:** The preregistration report can retain stale status text. Final checkpoint counts differ from the last periodic log. Single-seed evidence; inspect continual stages separately.
- **V14-tiny:** REPORT.md still says pre-registered/not yet run. Final evaluation is step 12,977; table header uses the last periodic log. Changes jointly affect targets and model size. Single seed.
- **V15 proposal:** The 16 October revision uses completed V14 results and is explicitly post-results. Analytical checks are not model training or evidence of outperforming V14.

## Preserved reports with stale status text

Historical reports retain their original bytes. The entries below supplement their headers with saved evidence; they do not edit preregistrations.

- **V13**: final base evaluation records step **12474**, **97,858,328 tokens**, reasoning score **0.2829**, text NLL **2.8004**. Source: [v13_tiny/runs/eval/base/results.json](v13_tiny/runs/eval/base/results.json). Check the run index for the separate continual stages.
- **V14**: final base evaluation records step **12977**, **101,816,716 tokens**, reasoning score **0.3725**, text NLL **2.6861**. Source: [v14_tiny/runs/eval/base/results.json](v14_tiny/runs/eval/base/results.json). Check the run index for the separate continual stages.

## Inventory limits

Execution groups are directories containing saved execution/evaluation evidence, including engineering and interrupted work. A directory is not an independent experiment or training seed. Copied source snapshots are indexed as files and excluded from execution counts. Unknown seeds and settings remain null.

Heavy files and downloaded papers are local-only. Their relative paths and checksums establish identity, not availability from GitHub. Dataset bodies and sealed examples are not embedded in this catalog.
