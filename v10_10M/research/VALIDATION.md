# Implementation validation — 11 October 2026

CPU and MPS behavioral contracts passed. These checks establish implementation behavior, not trained reasoning ability.

- Future-token perturbations do not change earlier logits; full and truncated-prefix calculations agree within FP32 tolerance at group boundaries and partial prefixes.
- Padding and separate batch rows are isolated. First concept guidance is row 7; later slot/group shifts match the source equations.
- Actual calls: E1=4, E2=8, E3=4, core=8, decoder=4. Both recurrent modules retain changed state between visits.
- All K alternatives share the committed earlier history. Batched alternatives at different per-document boundaries match individual reference calls.
- Token/concept losses backpropagate into encoder, learned code table, soft PQ head, decoder fusion and sparse Engram rows.
- Empty RBF memory gives identical logits. The first frozen-backbone write has nonzero gradients through the actual PQ head and decoder. Exact RBF sums, code partitions, differentiable queries, snapshot versions and duplicate-event rejection pass. The tiny FP32 step did not resolve a loss decrease at reported precision; no adaptation-quality result is claimed.
- All dataset hashes/shapes and identity partitions pass; every supplied Sudoku target obeys full grid and clue constraints. Two duplicate cached source records were removed before the final corpus was frozen.
- A 64-update real-data MPS smoke test covered text, math, native easy and hard Sudoku with finite state. A final-source 16-update run resumed to step 24 with the same deadline and finite loaded model.
- Real-checkpoint greedy evaluation completed for each symbolic family; all three tiny samples were wrong after smoke training, as recorded in generation_smoke.json. No solver, oracle code or output repair supplied success.
- A separate frozen real-checkpoint memory integration test appended three valid entries and persisted/restored the bank. Backbone gradients stayed absent.

Short batch-8 benchmarks favored MPS for the dominant text workload (about 4,169 vs 3,132 tokens/s on CPU). The mixed 64-update smoke sustained roughly 3,500–3,900 tokens/s in later logging windows, with approximately 4.1 GiB Metal driver allocation. This is not a thermal steady-state estimate.

The sealed test partitions were inspected only for data integrity, not scored for model selection. No full 24-hour training outcome, matched baseline comparison, continual retention result, or general Sudoku-solving capability is established yet.
