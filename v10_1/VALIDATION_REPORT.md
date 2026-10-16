# V10.1 review and launch validation — 12 October 2026

**Decision: validation passed and the 16-hour experimental pretraining run is active. It has not yet demonstrated better reasoning than V10.** The user authorized stopping V10 and starting V10.1 for 16 hours, with an optional extension to 24 hours before post-training.

## Evidence from V10

The fresh balanced CPU evaluation uses checkpoint **83,665**, not the last live status step. Its 32-sequence text train/dev NLLs are **3.6448 / 4.5539**, versus **3.8110 / 4.5742** at checkpoint 50,689. On the fixed balanced panel, math is **3/64**, easy Sudoku **0/16**, and hard Sudoku **0/16**. All generated Sudoku boards have the expected format; only 10 easy and 12 hard outputs preserve all clues, and none satisfy every constraint. Valid formatting is not a solved puzzle.

V10 stopped gracefully at **step 84,778**, **180,249,114 input tokens**, after **12.988 active hours**. Its preserved checkpoint is `../v10_10M/runs/pretrain_24h/stopped_step_0084778.pt`; final status has no error. The original source hash remains `3cbaf426badbfc45f723431838c4f5340cbf7ce2c99f1a8438ed8e3b7152a0e9`. Original train/dev arrays and metadata were rehashed against their manifest and remain unchanged.

The fresh concept-off text delta is **+0.000725 NLL**, with descriptive paired-batch 95% interval **[−0.000133, +0.001480]**; it does not establish a robust contribution on this sample. All three math successes are sorting problems.

These results support trying better concept interfaces and repaired supervision. They do not prove CRC solves the reasoning problem. Dense norms above 1 are **pre-clipping** values; they alone do not diagnose exploding gradients. V10 never populated the RBF bank, so its pretraining telemetry says nothing about successful continual-learning retention.

Evidence: [fresh checkpoint results](validation/current_v10_2026-10-12/fresh_evaluation.json), [telemetry analysis](validation/current_v10_2026-10-12/telemetry_analysis.json), [clean stop](validation/v10_stop.json).

## What is implemented

The full model has **5,731,496 dense + 20,000,320 Engram parameters**. It preserves width 128, 4× FFNs, encoder 4/4/4 depths with two middle visits, concept depth four with two visits, and decoder depth four. Early/final pooled encoder states feed the encoder→concept CRC. The final corrected concept state feeds both the PQ head and the continuous concept→decoder CRC. The separate chunk objective gives the PQ forecast direct predictive responsibility without a continuous-state shortcut.

The first-principles conditions tested are: no future information before a concept is released; independent histories between documents and candidate branches; a gradient path through each new interface; objective-preserving recurrence and accumulation; and stable, frozen query meaning for RBF writes. Good wiring is necessary but does not guarantee abstraction or solving.

Review fixes include:

- `ablate=True` now removes **both** decoder concept routes; PQ-only removal has its own flag.
- Winner-only gradient recomputation avoids retaining eight candidate graphs and matches the original min-objective gradient across all parameters, including Engram.
- Effective-batch accumulation weights each loss by its own token/group/document denominator.
- RBF inference checks model era, frozen status and bandwidth. Zero-gradient feedback consumes its event ID; replay cannot silently become a second write.
- The production trainer's first smoke exposed duplicate status keyword fields; that bug was fixed before launch and the new trainer completed stop/resume tests.

## Tests and their limits

| Check | Result | Scope |
|---|---|---|
| CPU architecture contracts | **19/19 pass** | Full width/depth; small vocabulary/table for behavioral tests |
| Metal architecture contracts | **19/19 pass** | Same contracts; tolerance 5e-5 vs CPU 5e-6 |
| Clean-data, grammar and telemetry contracts | **6/6 pass** | Every train/dev identity; original array hashes; all clean math answer grammar |
| CPU/Metal logits | Max absolute difference **6.26e-7** | Same deterministic model and inputs |
| CPU/Metal total objective | Absolute difference **4.77e-7** | Candidate update included |
| Selected CPU/Metal gradients | Max absolute difference **2.69e-7** | Embeddings, CRC, core, PQ, continuous projection and chunk readout |
| A/B/C/D/S real-data integration | **80 updates total, all finite** | 16 updates/arm; legacy tokenizer; smoke only |
| Production CPU checkpoint resume | **Pass**, steps 1–4 | Both optimizers, next data sample, source/data checks, initial dev evaluation |
| Extend 16→24 hours | **Pass** | Total budget extended; original cosine horizon remains 16h |
| Full-size batch 8 / context 384 | **Pass on Metal** | New clean corpus; full 20M table; bounded timing below |

The 19 architecture groups include baseline A equivalence, future-token perturbations, committed prefix equivalence at group boundaries and tails, padding isolation, candidate-history equality, exact execution counts, partial next chunks/EOS, auxiliary causal inputs, nonzero end-to-end gradients, empty masks, zero GradCuit, recomputation and accumulation equivalence, both-route correction derivatives, real-model finite differences, exact Gaussian writes, capacity/snapshot/era/idempotency guards, and interrupted optimizer replay.

The real RBF finite-difference and analytical update tests pass. A tiny predicted improvement can round to the same FP32 scalar on Metal; the test records whether the reduction is numerically resolved. This is not a learned-task adaptation or retention result.

Artifacts: [CPU](validation/contracts_cpu.json), [Metal](validation/contracts_mps.json), [pipeline](validation/pipeline_contracts.json), [parity](validation/metal_parity.json), [integration](validation/integration_summary.json), [resume](validation/production_resume.json).

## Memory and speed

With V10 stopped, brief full-size Metal tests used a **45% allocator cap** and effective batch eight. Warm ordinary / candidate update times were:

| Microbatch | Ordinary seconds | Candidate seconds |
|---:|---:|---:|
| 1 | 1.483 | 3.846 |
| 2 | 1.017 | 2.379 |
| 4 | 0.762 | 1.657 |
| 8 | **0.647** | **1.019** |

Full batch eight reached approximately **3.77 GiB Metal driver memory**. RSS and driver counters overlap. The chosen launch uses batch/microbatch eight, context 384, and cap 0.45. These are brief tests, not sustained thermal throughput guarantees. The fanless Air may slow over long runs; live telemetry records that.

An earlier **15% cap while V10 was running** failed during the main forward, before candidate scoring. Microbatch-one accumulation passed that restricted case. This does not mean a full batch cannot fit with the device available, and candidate recomputation alone did not resolve that earlier main-forward limit. Both results are retained.

Evidence: [batch benchmark](validation/batch_benchmark.json), [restricted-cap failure](validation/default_batch_mps.json), [restricted-cap accumulation](validation/effective_batch8_micro1_mps.json).

## Clean data and evaluation

The fresh corpus preserves **26,998 text, 239,904 math, 12,802 easy Sudoku, and 37,198 hard Sudoku training records**, plus all existing dev identities. No normalized train/dev collisions required quarantine. No symbolic answers were truncated or excluded for length. Math training tokens fall from **10,006,501 to 8,093,453**, largely removing byte-literal wrappers. A new 8,192-token BPE is trained on train only; the model must start from random weights.

No dev/test reassignment uses a newly cleaned hash. Original identities and normalized identities are both retained. The sealed test remains reserved in the original corpus and is never sampled by the trainer; re-encoding it with the new tokenizer is deferred. Exact split preservation does not prove semantic or Sudoku-isomorphism deduplication. Text source reuse and the inherited document-uniform crop sampler remain known limitations.

Routine math generation now balances four modules. It scores exact rational/decimal values and ordered lists with EOS, retaining raw-string exact match separately. Extra prose, reordered lists, invalid fractions and missing EOS fail. Sudoku is unconstrained greedy generation, scored as a whole board with EOS. No solver, correction or constrained digit decoding is used.

Because tokenization and target serialization change, raw V10 and V10.1 NLL values are **not directly comparable**. A controlled architecture claim would require matched data/tokenizer/sampler and replicated runs; this user-requested single candidate run is exploratory. Its short smoke improvements demonstrate update flow only.

## Telemetry and next decision

Every update persists losses, denominators, gradient norms, clipping, both recurrence losses, code use, candidate execution/regret, input/supervised exposure and hardware counters. Component gradients are sampled every 20 steps. Development evaluation separates PQ, continuous and both-route ablations and strict generation. SQLite ingestion preserves raw events and actual trainer timestamps; new values are accessible through the metric catalog, even when the default dashboard has no dedicated card.

Review text dev trend, balanced complete-answer scores, separate concept-route effects, code usage and resource stability before extending to 24 hours. A lower latent or chunk loss alone is insufficient. Post-training should then test frozen-era RBF adaptation, locality, duplicate-feedback handling and retained tasks with a predeclared tolerance. No post-training or retention success is claimed by this launch.

## Launch verification

The real run started **12 October 2026 at 01:20 IST**, with a planned completion around **17:20 IST**. Trainer PID **66803**, sleep guard PID **66804**, and collector PID **66775** were verified. The dashboard is [V10.1 live telemetry](http://127.0.0.1:8767/#live); V10 history remains available on port 8766.

The production smoke completed all 16 math and 2+2 Sudoku generation trials in about **69 seconds**, without exhausting its evaluation budget. Those near-random-model outputs are a pipeline test, not performance evidence. Browser checks confirmed live charts, balanced evaluation rendering, and no JavaScript console errors; the inherited chart/data utility suite passed **10/10** tests.

At the saved verification watermark, **156 consecutive optimizer updates** had finite losses and dense/Engram gradients, with no telemetry ingestion issues. The exact-run process check initially needed the absolute run directory on the command line. A graceful checkpoint at step 80 followed by an explicit-path resume corrected that; event steps remain contiguous and all optimizer/data state was retained. The README uses the verified explicit-path command. No Python source changed after this run's frozen snapshot.

The active training source fingerprint is `44012c37f54b48a707f75ad309fcdbb81da71d6fcfddb67470abf6a1ccfd82f5`; its clean data manifest fingerprint is `def40ad02c456e93b227be6d43fb91e91eb4735449d0cda2d709a6f60f0568f2`. The only Python change after the final architecture-contract run was the telemetry adapter's schema timestamp description/handler; its pipeline contracts were rerun before launch.

See [launch verification](validation/launch_verification.json), [API smoke](validation/api_smoke.json), [run manifest](runs/pretrain_16h/manifest.json), and the [live screenshot](validation/live_training.jpg). The manifest/checkpoint are authoritative for actual configuration; screenshots and this report are timestamped observations.
