# GRAM-first experiment: startup report

Snapshot: 2026-10-18T02:56:48.682774+05:30. Deadline: **2026-10-20T02:30:43.080275+05:30**.

The detached supervisor is running. At this snapshot it is completing the batch
code cache (10,272 / 28,733 unique blocks).
Long model training has not started yet. After the cache is complete, the supervisor
runs production-loader/checkpoint/evaluation smoke tests, trains **GRAM first**, then
trains the deterministic HRM comparison. Startup acceptance gates have passed.

## Model and scope

* GRAM: **10,000,333 trainable parameters**, including the training-only posterior
  and target encoder (237,505). Deployment package: 9,762,828 parameters.
* HRM comparison: 10,001,292 parameters. Same dataset and seed, equal allocated wall
  time; different feed-forward widths. This is a one-seed pilot.
* Official frozen LM1B NCM codec: 111,404,672 additional parameters and frozen
  buffers. It is outside the 10M trainable budget.
* 140,045 verified transitions: 109,943 train, 14,031 validation, 13,090 ID test,
  and 2,981 OOD. All rule/topology groups are disjoint across splits.
* Current-state-conditioned modular planning and maze action generation. The
  remaining V15 stages (strategy modes, search, distillation, continual learning)
  are not implemented or claimed validated by this pilot.

## Executed acceptance tests

| Check | Result |
|---|---|
| Base architecture tests | 10 passed |
| GRAM tests, including finite differences and trajectory KL units | 10 passed |
| Original blueprint mathematical checks | 9 passed |
| All raw transitions, labels, indices and current inputs | Verified; no split-group overlap |
| GRAM CPU/MPS fixed-noise loss | Absolute difference 0.0 |
| Gaussian CPU/MPS gradient discrepancy | Maximum 7.45e-09 |
| Prior-only overfit, three sampling seeds | 27/28 correct on every seed |
| Physical posterior removal | Same prior outputs under identical noise |
| Frozen codec | State hash unchanged |
| Stochastic MPS optimizer/RNG resume | Next-update maximum difference 0.0 |

The overfit result is a training diagnostic, not held-out performance. Main-run
validation and final test results are still pending.

## M2 settings and tokenizer decision

Float32 MPS; batch 8; gradient accumulation 2; AdamW at 0.0003 with warmup,
clipping and a cosine schedule. The short benchmark measured
18.0 blocks/second and 2.72 GiB MPS
allocation. It does not establish sustained thermal throughput.

TinyStories failed the domain/action audit. LM1B preserved all seven action
sentences and achieved 99.36% payload-token fidelity on 2,000 probes. Exact current
state is supplied separately because the codec is not lossless. Action boundaries
use a fixed first-sentence parser; trailing decoded padding is retained in logs.

A wider MPS encoding comparison changed 283/1,312,768 code IDs in nine blocks.
The run retains CPU tokenization/decoding and an immutable batch cache. MPS is used
for the trainable models. The cache saves repeated codec work; 641 latent codes per
128 text tokens do not by themselves imply a compute speedup.

## Runtime behavior

The original 48-hour deadline is retained across restarts. Checkpoints are atomic
and normally saved every ten minutes; validation runs every thirty minutes.
The final hour is reserved for ID/OOD and closed-loop evaluation. Restarts are
bounded, resource checks stop unsafe continuation, and a stop file permits graceful
shutdown. Keep the Mac connected to power with its lid open.

See `README.md` for status, stop and resume commands, `ARCHITECTURE_REVIEW.md` for
the full review, and `runs/20261018_48h/status.json` for live phase information.
