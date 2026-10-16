# File and folder coverage

The scope is the active v2 implementation/run and the v1/data/design evidence needed for comparison. This ledger distinguishes semantic review from mechanical inventory. It does not claim that every vendored library or numerical weight value was individually interpreted.

Snapshot contains 99 files: 72 project files and 27 generated/cache/vendor files. Inodes, byte sizes, modification times and text hashes are in [inventory.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/inventory.json). Text source is frozen under [source_snapshot](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot).

## Every project file under run_v2_19M

| Original relative path | Bytes | Review/use |
|---|---:|---|
| [ADVISOR_LOG.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/ADVISOR_LOG.md) | 2,596 | Read implementation/profiling advice; compared claims against configuration and measurements. |
| [DECISIONS.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/DECISIONS.md) | 8,469 | Read launch, architecture, data, LR, profiling, and checkpoint-retention decisions; used as intent, not independent result proof. |
| [STATUS.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/STATUS.md) | 1,304 | Read operational status; checked against saved configuration and live logs. |
| [code/V2_MONITOR_POLICY.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/V2_MONITOR_POLICY.md) | 2,149 | Read launch/pause/resume/watch/guard policy and shell behavior. Did not execute operational controls. |
| [code/battery_guard.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/battery_guard.sh) | 1,150 | Read launch/pause/resume/watch/guard policy and shell behavior. Did not execute operational controls. |
| [code/compat.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/compat.py) | 1,268 | Byte-identical compatibility shim; reviewed behavior inherited from v1. |
| [code/concept_core_d6.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/concept_core_d6.py) | 12,101 | Deep source review: state update, shared experts, cumsum capacity, gate weights, carry norm, PQ head and masking; exercised by CPU diagnostics. |
| [code/concept_core_v2.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/concept_core_v2.py) | 4,628 | Compared byte identity with v1; reviewed reused codebook/quantization and gradient stops. |
| [code/data.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/data.py) | 4,530 | Diffed against v1: epoch reseed; verified identical epoch-0 order, tensor batching, token masks. |
| [code/evaluate.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/evaluate.py) | 14,109 | Traced token weighting, same quick-eval rows, codebook entropy/CE support and probe metrics. |
| [code/final_eval.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/final_eval.py) | 14,654 | Reviewed scheduled evaluation, R sweeps, ablations and loop diagnostics; final v2 results do not yet exist. |
| [code/hrm_decoder.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/hrm_decoder.py) | 19,366 | Compared byte identity with v1; traced confined plan use, u/c interventions, tied token head, and NextLat module. |
| [code/hrm_encoder.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/hrm_encoder.py) | 19,696 | Compared byte identity with v1; traced narrow-width config, token/concept outputs, Engram, masks and gradient entry. |
| [code/model.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/model.py) | 6,298 | Deep review of dimensions, forward modes, NextLat teacher/student paths, masks, parameter grouping/accounting. |
| [code/pause.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/pause.sh) | 851 | Read launch/pause/resume/watch/guard policy and shell behavior. Did not execute operational controls. |
| [code/probes.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/probes.py) | 9,164 | Byte-identical probe generator; reused earlier split/leakage and answer-mask review. |
| [code/prof_attn.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/prof_attn.py) | 2,131 | Read profiling workload/timing scope and launch optimization context. Not rerun on MPS while training. |
| [code/prof_enc.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/prof_enc.py) | 1,087 | Read profiling workload/timing scope and launch optimization context. Not rerun on MPS while training. |
| [code/prof_layer.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/prof_layer.py) | 1,297 | Read profiling workload/timing scope and launch optimization context. Not rerun on MPS while training. |
| [code/prof_moe.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/prof_moe.py) | 1,729 | Read profiling workload/timing scope and launch optimization context. Not rerun on MPS while training. |
| [code/prof_moe2.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/prof_moe2.py) | 1,855 | Read profiling workload/timing scope and launch optimization context. Not rerun on MPS while training. |
| [code/profile_step.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/profile_step.py) | 3,366 | Read profiling workload/timing scope and launch optimization context. Not rerun on MPS while training. |
| [code/resume.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/resume.sh) | 606 | Read launch/pause/resume/watch/guard policy and shell behavior. Did not execute operational controls. |
| [code/run_job.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/run_job.sh) | 2,655 | Read launch/pause/resume/watch/guard policy and shell behavior. Did not execute operational controls. |
| [code/speed.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/speed.py) | 1,355 | Reviewed thread-local mask cache and unchanged arithmetic; launch tests provide equivalence evidence. |
| [code/start.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/start.sh) | 1,346 | Read launch/pause/resume/watch/guard policy and shell behavior. Did not execute operational controls. |
| [code/tasks.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/tasks.py) | 25,384 | Byte-identical task generator; reused earlier data/probe semantic audit. |
| [code/test_v2.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/test_v2.py) | 8,460 | Reviewed scope and saved results; expanded backward verification to K12, real dropping, and accumulation 16. |
| [code/train.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/train.py) | 26,636 | Deep review of loss weights, windows, detach/backward bridge, accumulation, LR, RNG, checkpointing, evaluation and logging. |
| [code/v2_check.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/v2_check.sh) | 1,384 | Read launch/pause/resume/watch/guard policy and shell behavior. Did not execute operational controls. |
| [code/v2_watch.sh](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/code/v2_watch.sh) | 734 | Read launch/pause/resume/watch/guard policy and shell behavior. Did not execute operational controls. |
| [runs/battery_guard.stdout](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/runs/battery_guard.stdout) | 0 | Inspected operational text, launch/monitor status, or PID metadata; no learning-quality claims inferred from empty files. |
| [runs/chain.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/runs/chain.log) | 136 | Inspected operational text, launch/monitor status, or PID metadata; no learning-quality claims inferred from empty files. |
| [runs/dashboard.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/runs/dashboard.log) | 115 | Inspected operational text, launch/monitor status, or PID metadata; no learning-quality claims inferred from empty files. |
| [runs/monitor.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/runs/monitor.log) | 821 | Inspected operational text, launch/monitor status, or PID metadata; no learning-quality claims inferred from empty files. |
| [runs/run_job.pid](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/runs/run_job.pid) | 6 | Inspected operational text, launch/monitor status, or PID metadata; no learning-quality claims inferred from empty files. |
| [runs/run_job.stdout](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/runs/run_job.stdout) | 0 | Inspected operational text, launch/monitor status, or PID metadata; no learning-quality claims inferred from empty files. |
| [runs/tests/test_v2.txt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/runs/tests/test_v2.txt) | 888 | Read every saved test result; no failure reported; assessed coverage limitations. |
| [runs/v2_48h/GUARDS.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/runs/v2_48h/GUARDS.jsonl) | 0 | Inspected: empty at cutoff; no recorded guard event. |
| [runs/v2_48h/ckpt_best.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/runs/v2_48h/ckpt_best.pt) | 1,220,142,965 | Loaded distinct step checkpoints for CPU inference/gradient diagnostics; aliases identified by inode. No optimizer update; no binary copy into report. |
| [runs/v2_48h/ckpt_last.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/runs/v2_48h/ckpt_last.pt) | 1,220,142,965 | Loaded distinct step checkpoints for CPU inference/gradient diagnostics; aliases identified by inode. No optimizer update; no binary copy into report. |
| [runs/v2_48h/ckpt_step000138.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/runs/v2_48h/ckpt_step000138.pt) | 1,220,142,965 | Loaded distinct step checkpoints for CPU inference/gradient diagnostics; aliases identified by inode. No optimizer update; no binary copy into report. |
| [runs/v2_48h/ckpt_step000267.pt](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v2_19M/runs/v2_48h/ckpt_step000267.pt) | 1,220,142,965 | Loaded distinct step checkpoints for CPU inference/gradient diagnostics; aliases identified by inode. No optimizer update; no binary copy into report. |
| [runs/v2_48h/config.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/runs/v2_48h/config.json) | 1,120 | Parsed full launch configuration and parameter table; cross-checked code and checkpoint args. |
| [runs/v2_48h/launcher.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/runs/v2_48h/launcher.log) | 32 | Inspected operational text, launch/monitor status, or PID metadata; no learning-quality claims inferred from empty files. |
| [runs/v2_48h/live.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/runs/v2_48h/live.json) | 567 | Parsed all saved records; recomputed equal-window comparisons and preserved raw snapshot. |
| [runs/v2_48h/log.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/runs/v2_48h/log.jsonl) | 4,275 | Parsed all saved records; recomputed equal-window comparisons and preserved raw snapshot. |
| [runs/v2_48h/stdout.log](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/runs/v2_48h/stdout.log) | 2,960 | Inspected operational text, launch/monitor status, or PID metadata; no learning-quality claims inferred from empty files. |
| [runs/v2_48h/train_log.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/runs/v2_48h/train_log.jsonl) | 10,326 | Parsed all saved records; recomputed equal-window comparisons and preserved raw snapshot. |
| [webapp/css/style.css](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/css/style.css) | 82,097 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/data/arch.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/data/arch.json) | 1,977 | Parsed JSON/JSONL structure and data provenance; distinguished live data, copied design content, and historical results. |
| [webapp/data/arch_v2.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/data/arch_v2.json) | 16,487 | Parsed JSON/JSONL structure and data provenance; distinguished live data, copied design content, and historical results. |
| [webapp/data/concept_v2.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/data/concept_v2.json) | 18,824 | Parsed JSON/JSONL structure and data provenance; distinguished live data, copied design content, and historical results. |
| [webapp/data/live_steps.jsonl](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/data/live_steps.jsonl) | 159,906 | Parsed JSON/JSONL structure and data provenance; distinguished live data, copied design content, and historical results. |
| [webapp/data/v2_live.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/data/v2_live.json) | 5,048 | Parsed JSON/JSONL structure and data provenance; distinguished live data, copied design content, and historical results. |
| [webapp/index.html](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/index.html) | 30,915 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/js/analysis.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/js/analysis.js) | 5,368 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/js/app.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/js/app.js) | 12,112 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/js/charts.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/js/charts.js) | 13,168 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/js/concept.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/js/concept.js) | 10,567 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/js/concept3d.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/js/concept3d.js) | 6,366 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/js/content.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/js/content.js) | 12,535 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/js/diagrams.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/js/diagrams.js) | 46,132 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/js/figtools.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/js/figtools.js) | 10,788 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/js/final.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/js/final.js) | 25,171 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/js/svg.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/js/svg.js) | 8,495 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/js/telemetry.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/js/telemetry.js) | 74,736 | Deep review of loss/depth charts, v1 reference lines, thresholds, grace periods and potentially misleading verdicts. |
| [webapp/js/v2.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/js/v2.js) | 20,665 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/js/v2figs.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/js/v2figs.js) | 39,711 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/js/view3d.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/js/view3d.js) | 50,071 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/js/view3d_v2.js](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/js/view3d_v2.js) | 57,490 | Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning. |
| [webapp/server.py](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v2_underperformance_review_2026-09-28/source_snapshot/webapp/server.py) | 35,430 | Traced run/data selection, budget/evaluation status and live field forwarding. |

## Generated, cached, and vendor files

| Path | Category | Treatment |
|---|---|---|
| code/__pycache__/compat.cpython-312.pyc | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| code/__pycache__/concept_core_d6.cpython-312.pyc | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| code/__pycache__/concept_core_v2.cpython-312.pyc | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| code/__pycache__/data.cpython-312.pyc | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| code/__pycache__/evaluate.cpython-312.pyc | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| code/__pycache__/hrm_decoder.cpython-312.pyc | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| code/__pycache__/hrm_encoder.cpython-312.pyc | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| code/__pycache__/model.cpython-312.pyc | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| code/__pycache__/probes.cpython-312.pyc | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| code/__pycache__/speed.cpython-312.pyc | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| code/__pycache__/train.cpython-312.pyc | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/__pycache__/server.cpython-312.pyc | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/vendor/OrbitControls.js | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/vendor/jsm/controls/OrbitControls.js | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/vendor/jsm/environments/RoomEnvironment.js | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/vendor/jsm/geometries/RoundedBoxGeometry.js | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/vendor/jsm/postprocessing/EffectComposer.js | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/vendor/jsm/postprocessing/MaskPass.js | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/vendor/jsm/postprocessing/OutlinePass.js | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/vendor/jsm/postprocessing/OutputPass.js | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/vendor/jsm/postprocessing/Pass.js | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/vendor/jsm/postprocessing/RenderPass.js | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/vendor/jsm/postprocessing/ShaderPass.js | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/vendor/jsm/renderers/CSS2DRenderer.js | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/vendor/jsm/shaders/CopyShader.js | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/vendor/jsm/shaders/OutputShader.js | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |
| webapp/vendor/three.module.min.js | dependency/cache/OS | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |

## Shared data continuity, every file

All 30 files below match the earlier full audit in inode, size and modification time. This audit reused that earlier full-data inspection and read the slices needed for fresh checkpoint diagnostics.

| Shared path under run_m2_25M | Inode same | Size same | mtime same |
|---|---|---|---|
| data/MANIFEST.before_eval_rebuild.json | True | True | True |
| data/MANIFEST.json | True | True | True |
| data/build.log | True | True | True |
| data/canon.npy | True | True | True |
| data/eval/ood.pt | True | True | True |
| data/eval/test.pt | True | True | True |
| data/eval/val.pt | True | True | True |
| data/lang_eval/test_tinystories_ctx16.npy | True | True | True |
| data/lang_eval/test_tinystories_doc.npy | True | True | True |
| data/lang_eval/test_tinystories_tok.npy | True | True | True |
| data/lang_eval/test_wikitext_ctx16.npy | True | True | True |
| data/lang_eval/test_wikitext_doc.npy | True | True | True |
| data/lang_eval/test_wikitext_tok.npy | True | True | True |
| data/lang_eval/val_tinystories_ctx16.npy | True | True | True |
| data/lang_eval/val_tinystories_doc.npy | True | True | True |
| data/lang_eval/val_tinystories_tok.npy | True | True | True |
| data/lang_eval/val_wikitext_ctx16.npy | True | True | True |
| data/lang_eval/val_wikitext_doc.npy | True | True | True |
| data/lang_eval/val_wikitext_tok.npy | True | True | True |
| data/maintenance/doc.npy | True | True | True |
| data/maintenance/tok.npy | True | True | True |
| data/test/doc.npy | True | True | True |
| data/test/tok.npy | True | True | True |
| data/token_bytes.npy | True | True | True |
| data/tokenizer.json | True | True | True |
| data/train/doc.npy | True | True | True |
| data/train/synthetic_doc_sha1.npy | True | True | True |
| data/train/tok.npy | True | True | True |
| data/val/doc.npy | True | True | True |
| data/val/tok.npy | True | True | True |

## Remaining repository context

- [Earlier comprehensive file audit](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/FILE_REVIEW.md), [inventory](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/inventory.json), and [results](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/RESULTS_APPENDIX.md): v1 implementation, corpus/shards, screens, final/continuation results, model_v2 design, and concept ablations. Current v1 logs/configurations were reread into this review's snapshot; byte-identical reused modules were verified.
- `run_m2_25M/model_v2`: implementation intent and prior design evidence; the new source and actual run settings take precedence when they differ.
- `Concept Design`, `Encoder Design`, `Decoder Design`, `Loop Papers`, root design notes and PDFs: contextual architecture/research material covered by the earlier audit. They were not all re-extracted or treated as fresh experimental evidence in this run diagnosis. No literature result is used here to claim that a local training change will work.
- `.venv`, caches, `.DS_Store`, and local assistant metadata: tooling/system material rather than model results; not a review of every dependency implementation.
- The report directory contains read-only diagnostic scripts, frozen text/log copies, measurements, the figure, and written review. Scripts do not call optimizer steps or alter training files.

## Checkpoint handling

At cutoff there were two distinct step checkpoints (138 and 267) and last/best aliases. Each checkpoint file is about 1.22 GB because it includes training state, not just inference weights. The CPU reader used memory mapping and parameter assignment to avoid unnecessary full copies. The aliases were recognized rather than counted as four independent experiments. Future checkpoint rotation may remove the original paths; this report preserves observations and code, not immutable copies of the binary weights.
