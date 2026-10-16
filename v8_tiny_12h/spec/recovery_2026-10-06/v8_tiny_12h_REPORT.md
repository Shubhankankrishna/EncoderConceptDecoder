# V8 full-exposure 12-hour pilot

A_alpha0_trained_contracts exited 1; inspect /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/A_alpha0_trained_contracts/process.log

Status: **failed**. One independent training seed; this is a pilot, not seed confirmation or a scaling test.

Policy: diagnostic thresholds; no competence-based stopping. Scores and confidence thresholds are evidence readouts; running a stage does not mean that it succeeded.

## Foundation evidence

| Arm | Parameters | Standard dev | Lookup dev | Competence |
|---|---:|---:|---:|---|

## Training exposure

- A_alpha0: 24,000 updates, 576,000 GraphNav examples, completed; [summary](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/A_alpha0/train_summary.json).

## Acceptance and component results


## Partial and skipped work


## Sealed endpoint measurements


The 12-hour pilot intentionally does not complete the full draft matrix: additional independent training seeds, exact fresh V6 retraining, enlarged hard-context training, reverse continual order, IRC/CRC, alternate NCP coefficients, canonical-BFS label comparison, full Z ablation grid, concept-only continual banks, post-continual search confirmation, and long-context/LLM scaling remain untested. Missing stages are not successful results. Development thresholds do not establish superiority; sealed paired results and intervals must support any claim.

Raw decisions, failures, checkpoint provenance and per-item predictions are retained under `runs/`. The earlier stopped run remains in `v8_tiny`. No early low-score gate terminated this revision.
