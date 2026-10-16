# V8 tiny experiment

engineering_simple_hierarchy fresh lookup/two-hop gate failed

Status: **completed_gate_failed**. One independent training seed; this is a pilot, not seed confirmation or a scaling test.

## Foundation evidence

| Arm | Parameters | Standard dev | Lookup dev | Competence |
|---|---:|---:|---:|---|

## Training exposure

- engineering_mem128: 1,000 updates, 0 GraphNav examples, learning_gate_passed; [summary](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/runs/engineering_mem128/train_summary.json).
- engineering_simple_hierarchy: 6,000 updates, 0 GraphNav examples, completed; [summary](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny/runs/engineering_simple_hierarchy/train_summary.json).

## Acceptance and component results


## Partial and skipped work


The 12-hour pilot intentionally does not complete the full draft matrix: additional independent training seeds, exact fresh V6 retraining, reverse continual order, IRC/CRC, alternate NCP coefficients, canonical-BFS label comparison, full Z ablation grid, concept-only continual banks, post-continual search confirmation, and long-context/LLM scaling remain untested. Missing stages are not successful results. Operational development gates do not establish superiority; sealed paired results and intervals must support any claim.

Raw decisions, failures, checkpoint provenance and per-item predictions are retained under `runs/`. Historical V6 replay is an inspected development diagnostic, not V8 confirmation.
