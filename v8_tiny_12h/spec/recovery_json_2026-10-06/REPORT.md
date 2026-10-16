# V8 full-exposure 12-hour pilot

A_delayed_ncp_dev exited 1; inspect /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/A_delayed_ncp_dev/process.log

Status: **failed**. One independent training seed; this is a pilot, not seed confirmation or a scaling test.

Policy: diagnostic thresholds; no competence-based stopping. Scores and confidence thresholds are evidence readouts; running a stage does not mean that it succeeded.

The user authorized continuation after the compiled decoding parity failure. Eager static decoding passed the same tolerance on trained weights; training compilation and the completed alpha-zero checkpoint were preserved. The original clock/deadline were retained. The failed source, preflights, lock and report are archived under `spec/recovery_2026-10-06`. This is an execution revision, not an additional training seed.

## Foundation evidence

| Arm | Parameters | Standard dev | Lookup dev | Competence |
|---|---:|---:|---:|---|
| A_alpha0 | 2,448,640 | 77.2% | 100.0% | False |

## Training exposure

- A_alpha0: 24,000 updates, 576,000 GraphNav examples, completed; [summary](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/A_alpha0/train_summary.json).
- A_delayed_ncp: 24,000 updates, 576,000 GraphNav examples, completed; [summary](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/A_delayed_ncp/train_summary.json).

## Acceptance and component results


## Partial and skipped work


## Sealed endpoint measurements


The 12-hour pilot intentionally does not complete the full draft matrix: additional independent training seeds, exact fresh V6 retraining, enlarged hard-context training, reverse continual order, IRC/CRC, alternate NCP coefficients, canonical-BFS label comparison, full Z ablation grid, concept-only continual banks, post-continual search confirmation, and long-context/LLM scaling remain untested. Missing stages are not successful results. Development thresholds do not establish superiority; sealed paired results and intervals must support any claim.

Raw decisions, failures, checkpoint provenance and per-item predictions are retained under `runs/`. The earlier stopped run remains in `v8_tiny`. No early low-score gate terminated this revision.
