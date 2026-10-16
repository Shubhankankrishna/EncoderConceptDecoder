# File and folder review coverage

This appendix lists every project file in the snapshot. “Indexed” means bytes/text were read and recorded; it does not imply line-by-line semantic review of every operational script or display asset. Deep tracing focused on model/data/training/evaluation code, experiments, and the v2 proposal. Binary coverage is specified separately.

Total inventory: 1098 files; substantive project files: 203; dependencies/caches/OS files: 895. The complete inventory, sizes, inodes, modification times and project text hashes are in [inventory.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/inventory.json).

## Folder-level conclusions

| Folder | Conclusion |
|---|---|
| Root records | Explain decisions and historical state; raw executable code and result records take precedence. |
| code | Implements the d384 model with ConceptCoreV2; no D6/MoE training module exists. Evaluation has the identified truncation/padding caveats. |
| data | Packed rows and tokenizer pass recorded hashes/alignment; final item sets are exact-disjoint from training. OOD definitions need qualification. |
| runs/screens | Early schedule-matched confined/stack/fixed-R controls are within tie band. Trained structural checks abort on observed reach below the bound. |
| runs/main | Completed final matrix supports distant recall and plan use, but not useful depth scaling or robust OOD composition. |
| runs/cont | Additional training of the same model; improving validation skills, no completed final matrix at snapshot. |
| runs/tests | Initial CPU structural evidence and MPS compatibility fix; distinct from every trained-screen assertion passing. |
| concept_v2 | Literature synthesis, candidate specifications and subjective scores; design evidence, not trained candidate comparisons. |
| model_v2 | Whole-model proposal, analytical budgets and diagrams; explicitly unimplemented/untrained. |
| webapp | Presentation and cached summaries. Source metrics remain authoritative; animated expert routing does not show observed routing. |
| .venv / __pycache__ / vendor / .DS_Store | Inventoried; not treated as original architecture or experimental evidence. |

## Every project file

### root — 5 files

| File | Bytes | Lines / records | Role and coverage |
|---|---|---|---|
| [DECISIONS.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/DECISIONS.md) | 39324 | 395 / — | Chronological deviations, calibration, operations, and experiment decisions; checked against executable code and raw logs. |
| [ENV.txt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/ENV.txt) | 792 | 13 / — | Recorded runtime and hardware environment. |
| [INVENTORY.txt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/INVENTORY.txt) | 6979 | 71 / — | Historical file inventory; not treated as current experiment evidence. |
| [PAPERS_READ.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/PAPERS_READ.md) | 5186 | 36 / — | Project literature summary; not an independent experimental result. |
| [STATUS.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/STATUS.md) | 6706 | 81 / — | Operational status summary; live logs take precedence. |

### code — 38 files

| File | Bytes | Lines / records | Role and coverage |
|---|---|---|---|
| [code/OVERNIGHT_POLICY.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/OVERNIGHT_POLICY.md) | 4346 | 48 / — | Existing runtime policy and operational thresholds; no new job launched by this review. |
| [code/battery_guard.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/battery_guard.sh) | 1222 | 20 / — | Operational launcher, pause/resume, battery or monitoring script; indexed and contextualized against run logs; not executed. |
| [code/battery_guard2.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/battery_guard2.sh) | 1460 | 22 / — | Operational launcher, pause/resume, battery or monitoring script; indexed and contextualized against run logs; not executed. |
| [code/build_data.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/build_data.py) | 21068 | 416 / — | Tokenizer fitting, mixture, packing, eval rebuild and hashes. |
| [code/calibrate.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/calibrate.py) | 4259 | 84 / — | MPS batch/memory/speed calibration. |
| [code/chain.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/chain.sh) | 2263 | 29 / — | Operational launcher, pause/resume, battery or monitoring script; indexed and contextualized against run logs; not executed. |
| [code/chain2.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/chain2.sh) | 1364 | 20 / — | Operational launcher, pause/resume, battery or monitoring script; indexed and contextualized against run logs; not executed. |
| [code/chain3.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/chain3.sh) | 1644 | 23 / — | Operational launcher, pause/resume, battery or monitoring script; indexed and contextualized against run logs; not executed. |
| [code/chain4.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/chain4.sh) | 4026 | 54 / — | Operational launcher, pause/resume, battery or monitoring script; indexed and contextualized against run logs; not executed. |
| [code/check_data.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/check_data.py) | 3677 | 66 / — | Existing data assertions; independently checked full packed rows and scored items. |
| [code/compat.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/compat.py) | 1268 | 22 / — | Workaround for EncoderLayer._apply collision with torch Module machinery. |
| [code/concept_core_v2.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/concept_core_v2.py) | 4628 | 105 / — | Implemented one-layer recurrent core, PQ targets/readout, NCP/VQ losses; not proposed D6. |
| [code/data.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/data.py) | 4053 | 108 / — | Batch construction, Engram CPU indices, target masking, packing and data cursor. |
| [code/dmix_build_reference.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/dmix_build_reference.py) | 14601 | 334 / — | Earlier D-MIX construction reference; actual manifest/code take precedence. |
| [code/e2e.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/e2e.py) | 13250 | 222 / — | Model interface, original end-to-end synthetic harness and pipeline tests. |
| [code/evaluate.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py) | 13325 | 247 / — | Language, truncation, answer-span metrics, codebook and intervention telemetry; r72 issue identified. |
| [code/extend_ckpt.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/extend_ckpt.py) | 662 | 10 / — | Continuation checkpoint preparation and preserved state. |
| [code/final_eval.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/final_eval.py) | 14953 | 305 / — | Final test conditions, probe/generation orchestration and output schema. |
| [code/harness.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/harness.py) | 9299 | 204 / — | Earlier synthetic encoder/test harness. |
| [code/hrm_decoder.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/hrm_decoder.py) | 19366 | 384 / — | Confined/stack decoder, plan indexing, optional NextLat, parameter groups and tests. |
| [code/hrm_encoder.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/hrm_encoder.py) | 19696 | 455 / — | Encoder implementation, local/global streams, Engram, chunk pooling, exact structural tests. |
| [code/main_steps.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/main_steps.py) | 1054 | 15 / — | Main step budget derivation. |
| [code/model.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/model.py) | 5752 | 106 / — | Main model configuration, interventions, parameter and MAC accounting. |
| [code/opus_check.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/opus_check.sh) | 2272 | 24 / — | Operational launcher, pause/resume, battery or monitoring script; indexed and contextualized against run logs; not executed. |
| [code/opus_watch.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/opus_watch.sh) | 1401 | 17 / — | Operational launcher, pause/resume, battery or monitoring script; indexed and contextualized against run logs; not executed. |
| [code/pause.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/pause.sh) | 2046 | 29 / — | Operational launcher, pause/resume, battery or monitoring script; indexed and contextualized against run logs; not executed. |
| [code/plots.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/plots.py) | 3782 | 58 / — | Presentation of recorded curves; no additional experiment. |
| [code/probe_multi.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/probe_multi.py) | 2668 | 34 / — | Earlier representation probes; useful diagnostic context, not final-model accuracy. |
| [code/probes.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/probes.py) | 9164 | 180 / — | Linear probes and loop dynamics; reviewed what the metrics can and cannot establish. |
| [code/report_tables.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/report_tables.py) | 6389 | 115 / — | Derived tables, including separate 256-item screen probes; raw JSON is authoritative. |
| [code/resume.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/resume.sh) | 1569 | 24 / — | Operational launcher, pause/resume, battery or monitoring script; indexed and contextualized against run logs; not executed. |
| [code/run_job.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/run_job.sh) | 795 | 18 / — | Operational launcher, pause/resume, battery or monitoring script; indexed and contextualized against run logs; not executed. |
| [code/run_job2.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/run_job2.sh) | 895 | 19 / — | Operational launcher, pause/resume, battery or monitoring script; indexed and contextualized against run logs; not executed. |
| [code/swap_at_ckpt.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/swap_at_ckpt.sh) | 826 | 12 / — | Operational launcher, pause/resume, battery or monitoring script; indexed and contextualized against run logs; not executed. |
| [code/tasks.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/tasks.py) | 25384 | 555 / — | Synthetic generators and actual ID/OOD definitions; three families have unchanged distributions. |
| [code/test_compat.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/test_compat.py) | 898 | 14 / — | Compatibility test coverage. |
| [code/test_p5.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/test_p5.py) | 4071 | 71 / — | Causality/reach/doc-isolation assertions; exact-max-reach assertion causes screen aborts. |
| [code/train.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/train.py) | 22465 | 398 / — | Optimizer, accumulation, sampled R, loss/masks, checkpoint/resume, LR and evaluation schedule. |

### data — 30 files

| File | Bytes | Lines / records | Role and coverage |
|---|---|---|---|
| [data/MANIFEST.before_eval_rebuild.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/MANIFEST.before_eval_rebuild.json) | 27816 | 1787 / — | Earlier manifest; retained as provenance, not current scored-set truth. |
| [data/MANIFEST.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/MANIFEST.json) | 29992 | 1867 / — | Actual data mixture, seeds, hashes, counts and duplicate drops. |
| [data/build.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/build.log) | 1562 | 35 / — | Data-construction and eval-rebuild provenance. |
| [data/canon.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/canon.npy) | 65664 | — / — | Canonical-ID lookup shape/range checked and traced to Engram indexing. |
| [data/eval/ood.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/eval/ood.pt) | 7669653 | — / — | Every item checked: train exact-overlap, within/cross-split duplicates, lengths, answer spans, distances and family counts. |
| [data/eval/test.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/eval/test.pt) | 3961051 | — / — | Every item checked: train exact-overlap, within/cross-split duplicates, lengths, answer spans, distances and family counts. |
| [data/eval/val.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/eval/val.pt) | 3971029 | — / — | Every item checked: train exact-overlap, within/cross-split duplicates, lengths, answer spans, distances and family counts. |
| [data/lang_eval/test_tinystories_ctx16.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/lang_eval/test_tinystories_ctx16.npy) | 2176 | — / — | Language evaluation array inspected for shape, scored tokens, document/context lengths and byte counts. |
| [data/lang_eval/test_tinystories_doc.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/lang_eval/test_tinystories_doc.npy) | 1048704 | — / — | Language evaluation array inspected for shape, scored tokens, document/context lengths and byte counts. |
| [data/lang_eval/test_tinystories_tok.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/lang_eval/test_tinystories_tok.npy) | 524416 | — / — | Language evaluation array inspected for shape, scored tokens, document/context lengths and byte counts. |
| [data/lang_eval/test_wikitext_ctx16.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/lang_eval/test_wikitext_ctx16.npy) | 2176 | — / — | Language evaluation array inspected for shape, scored tokens, document/context lengths and byte counts. |
| [data/lang_eval/test_wikitext_doc.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/lang_eval/test_wikitext_doc.npy) | 1048704 | — / — | Language evaluation array inspected for shape, scored tokens, document/context lengths and byte counts. |
| [data/lang_eval/test_wikitext_tok.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/lang_eval/test_wikitext_tok.npy) | 524416 | — / — | Language evaluation array inspected for shape, scored tokens, document/context lengths and byte counts. |
| [data/lang_eval/val_tinystories_ctx16.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/lang_eval/val_tinystories_ctx16.npy) | 2176 | — / — | Language evaluation array inspected for shape, scored tokens, document/context lengths and byte counts. |
| [data/lang_eval/val_tinystories_doc.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/lang_eval/val_tinystories_doc.npy) | 1048704 | — / — | Language evaluation array inspected for shape, scored tokens, document/context lengths and byte counts. |
| [data/lang_eval/val_tinystories_tok.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/lang_eval/val_tinystories_tok.npy) | 524416 | — / — | Language evaluation array inspected for shape, scored tokens, document/context lengths and byte counts. |
| [data/lang_eval/val_wikitext_ctx16.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/lang_eval/val_wikitext_ctx16.npy) | 2176 | — / — | Language evaluation array inspected for shape, scored tokens, document/context lengths and byte counts. |
| [data/lang_eval/val_wikitext_doc.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/lang_eval/val_wikitext_doc.npy) | 1048704 | — / — | Language evaluation array inspected for shape, scored tokens, document/context lengths and byte counts. |
| [data/lang_eval/val_wikitext_tok.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/lang_eval/val_wikitext_tok.npy) | 524416 | — / — | Language evaluation array inspected for shape, scored tokens, document/context lengths and byte counts. |
| [data/maintenance/doc.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/maintenance/doc.npy) | 51040384 | — / — | Entire packed shard streamed for token ranges, hashes, masks and chunk/document boundaries. |
| [data/maintenance/tok.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/maintenance/tok.npy) | 25520256 | — / — | Entire packed shard streamed for token ranges, hashes, masks and chunk/document boundaries. |
| [data/test/doc.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/test/doc.npy) | 50860160 | — / — | Entire packed shard streamed for token ranges, hashes, masks and chunk/document boundaries. |
| [data/test/tok.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/test/tok.npy) | 25430144 | — / — | Entire packed shard streamed for token ranges, hashes, masks and chunk/document boundaries. |
| [data/token_bytes.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/token_bytes.npy) | 65664 | — / — | Byte denominator table checked against ByteLevel symbol counts. |
| [data/tokenizer.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/tokenizer.json) | 547501 | 39990 / — | 8,192-ID byte-level BPE; hash checked, bytes/canonicalization inspected. |
| [data/train/doc.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/train/doc.npy) | 1121472640 | — / — | Entire packed shard streamed for token ranges, hashes, masks and chunk/document boundaries. |
| [data/train/synthetic_doc_sha1.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/train/synthetic_doc_sha1.npy) | 331244608 | — / — | Training document hash index used for every scored-item overlap check. |
| [data/train/tok.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/train/tok.npy) | 560736384 | — / — | Entire packed shard streamed for token ranges, hashes, masks and chunk/document boundaries. |
| [data/val/doc.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/val/doc.npy) | 50970752 | — / — | Entire packed shard streamed for token ranges, hashes, masks and chunk/document boundaries. |
| [data/val/tok.npy](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/data/val/tok.npy) | 25485440 | — / — | Entire packed shard streamed for token ranges, hashes, masks and chunk/document boundaries. |

### runs — 92 files

| File | Bytes | Lines / records | Role and coverage |
|---|---|---|---|
| [runs/GO_MAIN](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/GO_MAIN) | 0 | 0 / — | Experiment control/provenance file indexed and contextualized; no independent benchmark result. |
| [runs/MAIN_STEPS_OVERRIDE](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/MAIN_STEPS_OVERRIDE) | 5 | 1 / — | Experiment control/provenance file indexed and contextualized; no independent benchmark result. |
| [runs/battery_guard.stdout](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/battery_guard.stdout) | 0 | 0 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/battery_guard2.sh.bak_thresh15](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/battery_guard2.sh.bak_thresh15) | 1419 | 22 / — | Historical backup indexed and used as provenance; current executable source takes precedence. |
| [runs/battery_guard2.stdout](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/battery_guard2.stdout) | 0 | 0 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/calibration.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/calibration.json) | 1583 | 57 / — | Recorded B1/B2/B4 calibration data. |
| [runs/calibration.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/calibration.log) | 1456 | 22 / — | Calibration process and failure context. |
| [runs/calibration_B4_watermark.txt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/calibration_B4_watermark.txt) | 493 | 3 / — | Failed B4 experiment; distinguishes the 8.38 GiB figure from successful B2. |
| [runs/chain.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/chain.log) | 2270 | 29 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/chain.stdout](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/chain.stdout) | 0 | 0 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/chain2.stdout](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/chain2.stdout) | 0 | 0 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/chain3.stdout](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/chain3.stdout) | 0 | 0 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/chain4.pid](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/chain4.pid) | 6 | 1 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/chain4.sh.bak_lr3e-4](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/chain4.sh.bak_lr3e-4) | 3983 | 54 / — | Historical backup indexed and used as provenance; current executable source takes precedence. |
| [runs/chain4.stdout](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/chain4.stdout) | 489 | 3 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/cont/GUARDS.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/cont/GUARDS.jsonl) | 0 | 0 / 0 | All guard records parsed; empty at audit snapshot. |
| [runs/cont/ckpt_best.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/cont/ckpt_best.pt) | 943346677 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/cont/ckpt_last.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/cont/ckpt_last.pt) | 943346677 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/cont/ckpt_step001263.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/cont/ckpt_step001263.pt) | 943346677 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/cont/ckpt_step001460.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/cont/ckpt_step001460.pt) | 943346677 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/cont/ckpt_step001642.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/cont/ckpt_step001642.pt) | 943346677 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/cont/config.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/cont/config.json) | 1041 | 47 / — | Run arguments/configuration inspected to distinguish main, controls and continuation. |
| [runs/cont/init_from_main.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/cont/init_from_main.pt) | 943346485 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/cont/launcher.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/cont/launcher.log) | 42 | 1 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/cont/live.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/cont/live.json) | 296 | 1 / — | Volatile heartbeat snapshot; kept separate from completed evaluation results. |
| [runs/cont/log.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/cont/log.jsonl) | 16403 | 8 / 8 | All evaluation rows parsed; trajectory and all telemetry preserved in audit snapshot. |
| [runs/cont/stdout.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/cont/stdout.log) | 6040 | 45 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/cont/train_log.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/cont/train_log.jsonl) | 21922 | 52 / 52 | All records parsed; metrics/restart records distinguished; throughput, gradients, guard status and continuation inspected. |
| [runs/data_checks.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/data_checks.json) | 704 | 31 / — | Previously recorded data invariants; independently cross-checked. |
| [runs/data_samples.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/data_samples.md) | 80621 | 271 / — | Human-readable generated samples and family examples. |
| [runs/main/GUARDS.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/GUARDS.jsonl) | 0 | 0 / 0 | All guard records parsed; empty at audit snapshot. |
| [runs/main/ckpt_best.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/ckpt_best.pt) | 943346485 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/main/ckpt_last.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/ckpt_last.pt) | 943346485 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/main/ckpt_step003853.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/ckpt_step003853.pt) | 943346549 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/main/ckpt_step004039.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/ckpt_step004039.pt) | 943346549 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/main/ckpt_step004050.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/ckpt_step004050.pt) | 943346485 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/main/config.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/config.json) | 934 | 42 / — | Run arguments/configuration inspected to distinguish main, controls and continuation. |
| [runs/main/eval_best.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/eval_best.log) | 13 | 1 / — | Best/last checkpoint identity explains skipped duplicate evaluation. |
| [runs/main/eval_final.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/eval_final.json) | 80680 | 3136 / — | Authoritative main step4050 final matrix; copied verbatim into immutable audit snapshot. |
| [runs/main/eval_final.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/eval_final.log) | 4131 | 28 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/main/figs/curves.png](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/figs/curves.png) | 209173 | — / — | Rendered main learning curves; supporting display of the underlying logs. |
| [runs/main/launcher.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/launcher.log) | 342 | 6 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/main/log.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/log.jsonl) | 46118 | 23 / 23 | All evaluation rows parsed; trajectory and all telemetry preserved in audit snapshot. |
| [runs/main/p5_trained.txt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/p5_trained.txt) | 511 | 3 / — | All assertion output checked; main passes, screen tests abort at exact reach equality. |
| [runs/main/stdout.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/stdout.log) | 15422 | 114 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/main/train_log.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main/train_log.jsonl) | 50108 | 121 / 121 | All records parsed; metrics/restart records distinguished; throughput, gradients, guard status and continuation inspected. |
| [runs/main_steps.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/main_steps.json) | 218 | 9 / — | Experiment control/provenance file indexed and contextualized; no independent benchmark result. |
| [runs/opus_monitor.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/opus_monitor.log) | 12150 | 51 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/pause.sh.bak_pre_cont](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/pause.sh.bak_pre_cont) | 1811 | 27 / — | Historical backup indexed and used as provenance; current executable source takes precedence. |
| [runs/report_tables.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/report_tables.md) | 13260 | 229 / — | Derived report tables; traced to raw JSON and distinguished validation/test sample sizes. |
| [runs/resume.sh.bak_pre_cont](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/resume.sh.bak_pre_cont) | 1110 | 17 / — | Historical backup indexed and used as provenance; current executable source takes precedence. |
| [runs/screens/S1/GUARDS.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S1/GUARDS.jsonl) | 0 | 0 / 0 | All guard records parsed; empty at audit snapshot. |
| [runs/screens/S1/ckpt_best.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S1/ckpt_best.pt) | 943346485 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/screens/S1/ckpt_last.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S1/ckpt_last.pt) | 943346485 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/screens/S1/ckpt_step000265.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S1/ckpt_step000265.pt) | 943346485 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/screens/S1/config.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S1/config.json) | 910 | 42 / — | Run arguments/configuration inspected to distinguish main, controls and continuation. |
| [runs/screens/S1/launcher.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S1/launcher.log) | 78 | 2 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/screens/S1/log.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S1/log.jsonl) | 4317 | 2 / 2 | All evaluation rows parsed; trajectory and all telemetry preserved in audit snapshot. |
| [runs/screens/S1/p5_trained.txt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S1/p5_trained.txt) | 754 | 8 / — | All assertion output checked; main passes, screen tests abort at exact reach equality. |
| [runs/screens/S1/stdout.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S1/stdout.log) | 1289 | 10 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/screens/S1/train_log.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S1/train_log.jsonl) | 3043 | 7 / 7 | All records parsed; metrics/restart records distinguished; throughput, gradients, guard status and continuation inspected. |
| [runs/screens/S2/GUARDS.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S2/GUARDS.jsonl) | 0 | 0 / 0 | All guard records parsed; empty at audit snapshot. |
| [runs/screens/S2/ckpt_best.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S2/ckpt_best.pt) | 943481461 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/screens/S2/ckpt_last.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S2/ckpt_last.pt) | 943481461 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/screens/S2/ckpt_step000265.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S2/ckpt_step000265.pt) | 943481461 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/screens/S2/config.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S2/config.json) | 907 | 42 / — | Run arguments/configuration inspected to distinguish main, controls and continuation. |
| [runs/screens/S2/launcher.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S2/launcher.log) | 78 | 2 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/screens/S2/log.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S2/log.jsonl) | 4314 | 2 / 2 | All evaluation rows parsed; trajectory and all telemetry preserved in audit snapshot. |
| [runs/screens/S2/p5_trained.txt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S2/p5_trained.txt) | 799 | 9 / — | All assertion output checked; main passes, screen tests abort at exact reach equality. |
| [runs/screens/S2/stdout.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S2/stdout.log) | 1284 | 10 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/screens/S2/train_log.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S2/train_log.jsonl) | 3037 | 7 / 7 | All records parsed; metrics/restart records distinguished; throughput, gradients, guard status and continuation inspected. |
| [runs/screens/S3/GUARDS.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S3/GUARDS.jsonl) | 0 | 0 / 0 | All guard records parsed; empty at audit snapshot. |
| [runs/screens/S3/ckpt_best.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S3/ckpt_best.pt) | 943346421 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/screens/S3/ckpt_last.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S3/ckpt_last.pt) | 943346421 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/screens/S3/ckpt_step000265.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S3/ckpt_step000265.pt) | 943346421 | — / — | Checkpoint metadata/tensor shapes and hardlinks inspected using meta tensors; weights not re-evaluated numerically. |
| [runs/screens/S3/config.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S3/config.json) | 910 | 42 / — | Run arguments/configuration inspected to distinguish main, controls and continuation. |
| [runs/screens/S3/launcher.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S3/launcher.log) | 78 | 2 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/screens/S3/log.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S3/log.jsonl) | 4318 | 2 / 2 | All evaluation rows parsed; trajectory and all telemetry preserved in audit snapshot. |
| [runs/screens/S3/p5_trained.txt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S3/p5_trained.txt) | 754 | 8 / — | All assertion output checked; main passes, screen tests abort at exact reach equality. |
| [runs/screens/S3/stdout.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S3/stdout.log) | 1287 | 10 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/screens/S3/train_log.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/screens/S3/train_log.jsonl) | 3038 | 7 / 7 | All records parsed; metrics/restart records distinguished; throughput, gradients, guard status and continuation inspected. |
| [runs/swap.stdout](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/swap.stdout) | 0 | 0 / — | Operational/training process record indexed; failures/resumes contextualized with decisions and metrics. |
| [runs/tests/D_decoder.txt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/tests/D_decoder.txt) | 1011 | 9 / — | Recorded test output reviewed with compatibility/structural-test interpretation. |
| [runs/tests/P5_d384.txt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/tests/P5_d384.txt) | 656 | 5 / — | Recorded test output reviewed with compatibility/structural-test interpretation. |
| [runs/tests/P_pipeline.txt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/tests/P_pipeline.txt) | 868 | 6 / — | Recorded test output reviewed with compatibility/structural-test interpretation. |
| [runs/tests/T_encoder.txt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/tests/T_encoder.txt) | 201 | 3 / — | Recorded test output reviewed with compatibility/structural-test interpretation. |
| [runs/tests/T_encoder_mps.txt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/tests/T_encoder_mps.txt) | 1053 | 17 / — | Recorded test output reviewed with compatibility/structural-test interpretation. |
| [runs/tests/T_encoder_mps_via_compat.txt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/tests/T_encoder_mps_via_compat.txt) | 212 | 3 / — | Recorded test output reviewed with compatibility/structural-test interpretation. |
| [runs/tests/config_check.txt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/tests/config_check.txt) | 782 | 12 / — | Recorded test output reviewed with compatibility/structural-test interpretation. |
| [runs/train.py.bak_pre_cont](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/train.py.bak_pre_cont) | 20218 | 371 / — | Historical backup indexed and used as provenance; current executable source takes precedence. |
| [runs/train.py.bak_pre_heartbeat](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/train.py.bak_pre_heartbeat) | 19459 | 360 / — | Historical backup indexed and used as provenance; current executable source takes precedence. |
| [runs/train.py.bak_pre_pause](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/runs/train.py.bak_pre_pause) | 19220 | 356 / — | Historical backup indexed and used as provenance; current executable source takes precedence. |

### concept_v2 — 9 files

| File | Bytes | Lines / records | Role and coverage |
|---|---|---|---|
| [concept_v2/ADVISOR_LOG.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/concept_v2/ADVISOR_LOG.md) | 18975 | 56 / — | Recorded advice/review history; not independent measurements. |
| [concept_v2/DESIGNS.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/concept_v2/DESIGNS.md) | 66954 | 597 / — | Evolving candidate designs and earlier reasoning, distinguished from final proposals. |
| [concept_v2/DESIGNS_FINAL.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/concept_v2/DESIGNS_FINAL.md) | 7376 | 78 / — | Candidate definitions, tests, comparisons and selection logic. |
| [concept_v2/DESIGN_D6.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/concept_v2/DESIGN_D6.md) | 8277 | 80 / — | D6 structure/training rationale; carry-necessity claim and R6 window note challenged. |
| [concept_v2/LOOP_PAPERS_NOTES.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/concept_v2/LOOP_PAPERS_NOTES.md) | 149950 | 1254 / — | Project notes for 42 papers plus three residual references; selected originals independently checked. |
| [concept_v2/PROGRESS.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/concept_v2/PROGRESS.md) | 5054 | 44 / — | Design-work progress and provenance. |
| [concept_v2/budgets.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/concept_v2/budgets.json) | 1103 | 58 / — | Candidate analytical core budgets. |
| [concept_v2/core_budget.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/concept_v2/core_budget.py) | 3258 | 59 / — | Analytical core sizing, not an implemented/trainable D6 core. |
| [concept_v2/scores.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/concept_v2/scores.json) | 9307 | 203 / — | Subjective weighted design scores; not measured model quality. |

### model_v2 — 5 files

| File | Bytes | Lines / records | Role and coverage |
|---|---|---|---|
| [model_v2/MODEL_V2_19M.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/model_v2/MODEL_V2_19M.md) | 54891 | 653 / — | Current whole-model proposal; explicitly unimplemented/untrained; reviewed architecture, graph, budget and evidence. |
| [model_v2/arch_v2.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/model_v2/arch_v2.json) | 19797 | 1069 / — | Generated diagram definition, not model weights/results. |
| [model_v2/make_arch_v2.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/model_v2/make_arch_v2.py) | 11605 | 114 / — | Diagram/data generator; not neural-network implementation. |
| [model_v2/v2_budget.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/model_v2/v2_budget.json) | 12530 | 613 / — | Six analytical variants; row C is selected proposal. |
| [model_v2/v2_budget.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/model_v2/v2_budget.py) | 10239 | 155 / — | Parameter/compute estimator; all six parameter rows reproduced, padding arithmetic checked. |

### webapp — 24 files

| File | Bytes | Lines / records | Role and coverage |
|---|---|---|---|
| [webapp/css/style.css](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/css/style.css) | 81223 | 800 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/data/analysis.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/data/analysis.json) | 18725 | 1 / — | Existing generated analysis; not an independent model evaluation. |
| [webapp/data/arch.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/data/arch.json) | 1977 | 105 / — | Derived/cached visualization data parsed; raw experiment logs take precedence. |
| [webapp/data/arch_v2.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/data/arch_v2.json) | 16487 | 1 / — | Derived/cached visualization data parsed; raw experiment logs take precedence. |
| [webapp/data/concept_v2.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/data/concept_v2.json) | 18824 | 337 / — | Derived/cached visualization data parsed; raw experiment logs take precedence. |
| [webapp/data/live_steps.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/data/live_steps.jsonl) | 577775 | 1754 / 1754 | Derived/cached visualization data parsed; raw experiment logs take precedence. |
| [webapp/index.html](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/index.html) | 30161 | 410 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/js/analysis.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/js/analysis.js) | 5368 | 72 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/js/app.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/js/app.js) | 11247 | 144 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/js/charts.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/js/charts.js) | 13168 | 185 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/js/concept.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/js/concept.js) | 10567 | 124 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/js/concept3d.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/js/concept3d.js) | 6366 | 85 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/js/content.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/js/content.js) | 12535 | 94 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/js/diagrams.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/js/diagrams.js) | 46132 | 529 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/js/figtools.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/js/figtools.js) | 10788 | 164 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/js/final.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/js/final.js) | 25171 | 223 / — | Derived final-results/gate view; PASS-line counting does not fully represent aborted screen tests. |
| [webapp/js/svg.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/js/svg.js) | 8495 | 155 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/js/telemetry.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/js/telemetry.js) | 63514 | 644 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/js/v2.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/js/v2.js) | 19527 | 172 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/js/v2figs.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/js/v2figs.js) | 39711 | 414 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/js/view3d.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/js/view3d.js) | 50071 | 552 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/js/view3d_v2.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/js/view3d_v2.js) | 57490 | 606 / — | Dashboard/diagram presentation source indexed and traced to its data; animations are illustrative, not measured behavior. |
| [webapp/server.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.log) | 115 | 1 / — | Dashboard runtime log indexed; no new model evidence. |
| [webapp/server.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/webapp/server.py) | 34670 | 650 / — | Dashboard serving/reading and analysis endpoints; not executed during audit. |

## Supporting material outside the requested run folder

The parent PROMPT.md supplied the precommitted architecture, data, and evaluation gates. Earlier Concept Design and Decoder Design result files were used to check whether proposed explanations transfer from isolated symbolic tests to end-to-end training. Relevant method sections of the local NextLat, Sparse Layers, LoopFormer, and Diffusion curriculum PDFs were checked. This is a targeted cross-check, not an assertion that every parent-folder PDF was reread.
