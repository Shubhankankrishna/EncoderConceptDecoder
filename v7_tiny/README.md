# V7 tiny: matched foundations and concept-space repair

This folder contains an independent implementation and an authorized serial experiment queue. It preserves the V7 sequence: train the encoder/concept/decoder foundation, freeze its surrounding model, refit the concept core, train prior-driven exploratory controls, audit the complete transplant, then compare GradCuit and AB-MCTS with retries and simpler search allocation.

The baseline is an ordinary stack of **12 distinct causal Transformer blocks**, with no concept pooling, forecast vocabulary, IRC or CRC. It has **5,366,208 parameters**. V7 has **5,365,938** before exploration, and **5,407,410** with two adapters. The difference between the foundations is 270 parameters (0.0050%). The baseline gets the same tokenizer, context, training data sequence, scored-target budget and final evaluation partitions. Its separately tuned token retries also compete with V7 search at the same wall-time budget; V7's additional adapter capacity and compute remain disclosed.

V7 uses width 192, six attention heads, SwiGLU width 576, and distinct encoder/concept/decoder depths 3/4/3. Four tokens form a concept; its forecast vocabulary has 12 segments with 64 entries. The two exploration sites use 48-dimensional controls and rank 48. The Transformer uses width 192, six heads, FFN width 464 in eleven blocks and 448 in its last block. Both use rotary positions and per-head Q/K normalization.

The tiny IRC groups end every four blocks; at these shallow stack depths there is no later block that can consume a completed group. The group branch is therefore inactive here. Its router coefficients remain included in the architectural count (1,930 parameters across the ten IRC routers). This is not evidence for the group bank's behavior at larger depths.

## Data

- Fresh FineWeb-Edu sample: 26,998 training documents, **54,716,006 training tokens**, with 1,500 separate development and 1,500 separate test documents. Exact and approximate lexical deduplication is recorded in `data/manifest.json`; semantic deduplication is not claimed.
- Legacy TinyStories is training material. Educational text and stories are mixed equally by document. Each training batch contains equal numbers of text and symbolic rows, with separate domain denominators in the objective.
- Streaming symbolic data includes graph navigation, pointer following, exact lookup and two-step composition. Random labels, shuffled headers, graphs of 10/14/18 nodes and excluded evaluation instances reduce simple memorization.
- Eight fixed symbolic suites cover standard/harder/longer navigation, pointer chains, longer pointer chains, lookup, reversed layouts and composition. Hard graphs require shortest distance at least four; long graphs require at least five. BFS constructs these data and references; it is not used by model search.

## Runtime and evidence

The queue freezes its implementation, specification and data metadata before training. It runs one training job at a time, stores finite atomic recovery checkpoints, reserves the final six hours of a 24-hour ceiling for testing, and labels skipped or partial endpoints. No existing V1–V6 training files are changed.

Training uses correctness-checked partial compilation on the Apple GPU. SDPA backward remains eager because full compilation corrupted K-normalization gradients. Before the final dataset adjustment, warm throughput was approximately 7,700 V7 and 5,600 Transformer scored targets/second at batch 32; refit and adapter stages were approximately 9,400 and 5,500. Final calibration is in `runs/calibration/benchmark.json`; sustained rates are measured in each run. Cached generation was around three times faster than the dynamic reference. These are speed measurements, not quality results.

Read `spec/PRE_REGISTRATION.md`, `spec/PAPER_REVIEW.md` and `spec/COMPLIANCE.md` for the decisions and limits. The queued plan is in `runs/plan.json`; live progress is in `runs/status.json` and each stage's `live.json`. Stage events are in `runs/queue.jsonl`. The final report will be written to **REPORT.md**, with machine-readable conclusions in **PRIMARY_RESULT.json**. Until those exist, no final quality result is available.

## File map

| Area | Purpose |
|---|---|
| `code/model.py` | Distinct stacks, bounded routes, soft product-quantized forecast, exact-zero adapters and ordinary Transformer |
| `code/data.py`, `fetch_text.py` | Fresh partitions, boundary-aware packing, streaming tasks and executable verifiers |
| `code/train.py`, `runtime.py` | Explicit objectives, ownership, budgets, finite recovery and compile safeguards |
| `code/generation.py`, `search.py` | Real-token concept refresh, independent caches, replay gradients, GradCuit and AB-MCTS-A |
| `code/evaluate.py`, `audit.py`, `reachability.py` | Sealed testing, paired uncertainty, trained invariants and privileged development diagnosis |
| `code/checks.py`, `benchmark.py` | Correctness contracts and throughput selection |
| `code/drive.py`, `report.py` | Serial queue, protocol locks, budget accounting and evidence-only reporting |
| `spec/` | Frozen design and experimental protocol |
| `tests/` | Unit-contract and end-to-end smoke evidence; these do not validate model quality |

Larger width/depth, long-context training, BF16 and distributed scaling remain separate experiments. This run is intended to falsify useful tiny-model claims, rather than certify LLM-scale behavior.
