# Folder map

Initial project snapshot: 6,179 files, excluding `.venv` and bytecode. Counts include local result artifacts and copied snapshots; they are not counts of independent experiments.

| Location | Files | Decimal MB | Snapshot-classified files | Role |
|---|---:|---:|---:|---|
| `(root)` | 34 | 26.60 | 0 | Architecture drafts/reviews, assembly prompt, historical walkthrough and core papers. |
| `.claude` | 1 | 0.00 | 0 | Local editor/tool permissions configuration; not model code. |
| `Concept Design` | 21 | 0.47 | 0 | Original concept core, NCP/PQ/recurrence prototypes and isolated/end-to-end experiment results. |
| `Decoder Design` | 39 | 0.37 | 0 | Private-plan confined decoder specification, implementations and archived experiment bundle. |
| `Emergence in Residual Stream` | 2 | 6.62 | 0 | Temporal-abstraction/metacontroller literature. |
| `Encoder Design` | 8 | 0.26 | 0 | Original local/global encoder specifications, reference implementations, harness and design conversation. |
| `Imp Paper` | 3 | 7.22 | 0 | Small selected-paper subset; duplicates of library papers. |
| `Loop Papers` | 42 | 176.64 | 0 | Main literature library for recurrence, latent reasoning, exploration and compute scaling. |
| `V11_Paper` | 32 | 24.40 | 0 | V11 design handoff bundle, selected papers and frozen V10.3 source/result evidence. |
| `audit_v2_2026-09-28` | 19 | 1.54 | 0 | Independent historical audit of the assembled system and v2 design; evidence, plots and appendices. |
| `draft_v3_review_2026-09-29` | 96 | 7.17 | 30 | Deep draft-v3 review inventory, source snapshots, claim probes and literature extracts. |
| `run_m2_25M` | 253 | 4,158.40 | 0 | Whole-model v1 training sources, environment, tokenizer/data, checkpoints, probes and main historical dashboard. |
| `run_v2_19M` | 89 | 1,223.78 | 0 | Whole-model v2 D6/MoE concept-core and NextLat training; copied decoder/encoder primitives and dashboard. |
| `run_v3_19M` | 257 | 5,402.29 | 0 | Trainer-grade v3 dense recurrent/shortcut model, completed 48-hour experiment and source/data/results. |
| `tmp` | 10 | 1.68 | 0 | Existing intermediate paper text extracts and methodology-page renders. |
| `v10` | 5 | 1.76 | 0 | RCDL proposal, reviewed architecture, notes and RCDL paper. |
| `v10_1` | 248 | 3,320.95 | 72 | CRC/chunk-utility candidate, cleaned identity-preserved corpus, production trainer, validation, optimization controls and telemetry. |
| `v10_10M` | 224 | 2,004.60 | 17 | Implemented v10 model/data/pretraining, CPU-sparse Engram, RBF correction API, research/audit tools and SQLite telemetry UI. |
| `v10_1_design_evidence` | 11 | 0.18 | 0 | Checkpoint/data/interface/telemetry probes behind v10.1 design. |
| `v10_2` | 115 | 312.87 | 57 | RF history-reader/forecast-only model, trainer/memory/telemetry and production smoke validation; no production runs folder in snapshot. |
| `v10_2_design_evidence` | 26 | 0.42 | 0 | Recorded v10.1 evaluations and review checks motivating v10.2. |
| `v10_2_review_evidence` | 12 | 0.09 | 0 | Episode-kernel, interface and spectrum probes for the v10.2 review. |
| `v10_3_design_evidence` | 21 | 0.28 | 0 | Forecast, visit, candidate-utility, parameter and density-gate probes motivating v10.3. |
| `v10_3_tiny` | 187 | 411.25 | 0 | Tiny mixture-likelihood/model-interface comparison, data-alignment controls, RBF/gated-LoRA adaptation and consolidation. |
| `v11_Papers` | 7 | 26.50 | 0 | Additional V11-era literature, distinct from the V11_Paper bundle. |
| `v11_design_evidence` | 8 | 0.05 | 0 | V11 parameter/allocation/scaling and alignment calculations. |
| `v11_tiny` | 201 | 305.25 | 0 | Code-free/Gaussian-objective/bridge study, fair flat controls, gradient/clip diagnostics and comparison reports. |
| `v12_tiny` | 148 | 83.32 | 0 | Current independent E→C→D/categorical-NCP base, pinpad mixed corpus, probes, frozen-base controller sweep and version explorer. |
| `v2_underperformance_review_2026-09-28` | 96 | 2.62 | 68 | Frozen v2 source/logs, gradient/routing replications and checkpoint diagnostics. |
| `v3_design` | 8 | 0.08 | 0 | Earlier v3 reference, budget/accounting and design verification scripts. |
| `v3_design_review_2026-09-29` | 288 | 6.87 | 268 | Earlier v3 proposal review, frozen sources, diagnostics and explanatory UI evidence. |
| `v41_tiny` | 474 | 295.73 | 0 | v4.1 retrieval/private-plan implementation, replicated controls, counterfactual/gradient/long audits and reports. |
| `v4_tiny` | 218 | 233.95 | 0 | v4 tiny implementation, model variants, comparison with a flat transformer and run queue. |
| `v5_tiny` | 155 | 379.53 | 0 | Five-phase NCP/GRAM/XM/transplant/metacontroller/internal-RL/GradCuit experiment. |
| `v6_tiny` | 211 | 554.06 | 0 | Self-contained foundation, deterministic/recurrent refit, bounded latent repair, verifiers, checks and comparisons. |
| `v7_tiny` | 772 | 2,801.15 | 0 | Independent larger foundation/refit/control/AB search study; three foundation seeds and engineering/pipeline fixtures. |
| `v8_tiny` | 135 | 412.54 | 0 | Initial competence-first pilot; gate-failed engineering result and audit. |
| `v8_tiny_12h` | 383 | 990.00 | 105 | Separate full-exposure pilot; foundation/refit/control/LSL stages, sealed endpoints and preserved recovery snapshots. |
| `v9_tiny` | 1,281 | 2,461.54 | 73 | H-NCP refit, grounded/imagined concept search, expert iteration, GRPO comparison, LSL, staged contracts and monitored recoveries. |
| `v9_webapp` | 39 | 2.69 | 0 | V9 architecture atlas and beginner S1–S6 guide, local assets/builders and read-only telemetry server. |

## Repeating internal folder conventions

| Folder/file | Meaning |
|---|---|
| `code/` or root `.py` | Version-specific executable research code |
| `data/`, `data_v9/` | Tokenizers, packed corpora, task arrays, offsets, identities and manifests |
| `runs/<name>/` | Checkpoints, actual configs, logs, per-item evaluations and summaries |
| `tests/`, `validation/`, `research/` | Contract/smoke/profiling evidence and fixtures; not automatically trained-task success |
| `spec/` | Pre-registration, locks, frozen design snapshots and amendments |
| `source_snapshot/`, `frozen/`, `recovery_*` | Preserved source/evidence copies with their own provenance |
| `monitor/` | Queue health, recovery tools, state and notifications for historical jobs |
| `web/`, `webapp/`, `viz/` | Dashboards, explanation assets and architecture explorers |
| `telemetry_data/`, `telemetry/data/` | Collector databases, exported catalogs/status and UI QA evidence |
| Compiler caches / `.so` / `.cpp` | Generated execution artifacts, not distinct model designs |

All nested directories present in the initial inventory are listed in [directories.tsv](directories.tsv).
