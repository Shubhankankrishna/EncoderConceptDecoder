# Launch report — Astra v4

The reviewed implementation is running. Snapshot: 2026-10-19T14:16:22.497206+05:30.
Active stage: **r1_seed42**, step **250**. This is a staged research pilot,
not a demonstrated improvement in language quality.

## Review outcome

The supplied 1,096-line spec was reviewed in full against the pinned source and checkpoint.
See [the corrected contract](../REVIEWED_SPEC.md) and [implementation guide](../README.md).
Major corrections: exact graph states; discounted path-cost values; node-resolved recurrence;
explicit supervised preference strategies; matched parameter and arithmetic controls; shared
training/generation FiLM code; deliberate coarse-level supervision; exact simulator search;
and independent reporting of planner and renderer performance.

This pilot retains the official 128-token/641-code pretrained hierarchy. It supersedes the
previous native32 experiment. It is GRAM-inspired, not a claim of paper-faithful GRAM.
NCM renders verified selected plans; pretrained transfer to graph decisions is not asserted.

## Execution

- Supervisor PID: 73063.
- Active trainer PID: 73635.
- Maximum deadline: **2026-10-20T13:40:27.431073+05:30**. Finish earlier on completed stages or renderer plateau.
- Twelve structured runs: six controls × two seeds × 4,000 updates, batch 32.
- Gated renderer: rank 4 attention LoRA/FiLM, up to 12,000 updates, effective batch 4.
- 1,600 train / 200 validation / 200 test / 200 larger OOD graphs, isomorphism-deduplicated and difficulty-stratified.
- Cache: 1,312 plan texts and 256 FineWeb retention texts, prepared offline in float32.
- Source registry: 59 files. Resume checks source, data, configuration and runtime identities.
- One exclusive heavy job at a time. Checkpoints save optimizer and Python/CPU/MPS RNG state.
- The old 30M/10M run stopped cleanly at 27,866 / 43,181 updates. Checkpoint hashes remain unchanged.

## Acceptance evidence

Ten regression tests pass. Actual MPS optimization and exact next-update resume pass for
all six structured configurations. The eight-graph debugging fit achieved 100% training-graph
success; it is not held-out generalization. LoRA fitting reduced fixed-batch loss and exact
optimizer recovery reproduced the next weights with zero difference. Cached inputs match
canonical float32 features exactly, and future/target code changes cannot leak into their
own prediction inputs. The real CLIs passed clean stop/resume and the full renderer
loss/evaluation/checkpoint cycle in the correct virtual environment.

| Configuration | Parameters | Median 8-example probe update seconds | Resume weight difference |
|---|---:|---:|---:|
| r0 | 247,683 | 0.0158 | 0.0 |
| r1 | 346,564 | 0.0195 | 0.0 |
| r2 | 346,564 | 0.0215 | 0.0 |
| untied2 | 346,949 | 0.0233 | 0.0 |
| untied_compute2 | 445,445 | 0.0225 | 0.0 |
| strategy2 | 346,820 | 0.0237 | 0.0 |

These are short feasibility measurements; production batch 32 and validation add cost.

Official generator 97,065,600 parameters; frozen codec 111,404,672. Attention LoRA 221,184;
all renderer additions 528,704; two-preference recurrent planner 346,820. Total configured
physical parameters 209,345,796, with 875,524 new trainable parameters in separate stages.

Zero-adapter and nonzero-FiLM full-versus-level logit differences were 5.34e-5;
CPU/MPS generator difference 8.39e-5. Frozen generator parameters remained unchanged after
an optimizer step. Those tests prove implementation parity, not the paper's benchmark scores.

## Limits that must remain visible

The codec reconstructs the exact parsed route in 87.5% of 96 validation plan texts
(normalized non-PAD token accuracy 95.88%). The raw-text codec probe also showed imperfect
roundtrips. Consequently exact structured paths remain canonical. Plain greedy pretrained
text repeats; the released sampler helper also cuts at leading CLS in the audited examples.
Its raw output and explicit boundary-normalized variant are recorded, not silently equated.

Correct versus shuffled plan NLL, free generated parsed paths, ordinary-text retention,
R2 versus controls, and final held-out/OOD success must decide whether the approach helps.
The renderer is currently queued behind grounded controls. It only starts if both R2 seeds
reach 70% validation success and beat random by at least 10 percentage points. MCTS remains
an evaluation option; equal-cost superiority over simpler planning is not established.

## Monitoring

[Live local dashboard](http://127.0.0.1:8767/). Its read-only API passed status, host,
path and rejected-write checks. Browser accessibility and visual layout were verified.
The old dashboard at 8766 remains a history view. Current metrics display the actual saved
validation step. Detailed evidence is in the JSON and log files beside this report.
