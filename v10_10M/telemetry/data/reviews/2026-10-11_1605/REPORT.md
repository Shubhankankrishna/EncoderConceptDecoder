# V10 progress review — 11 October 2026

As of 2026-10-11T16:16:11.555833+05:30. Verdict: operationally healthy, meaningful text learning, weak complete-answer reasoning, and no demonstrated benefit from concept guidance yet. This is approximately four hours into a 24-hour run; these observations do not determine the final result.

## Evidence and scope

Inspected the live dashboard at http://127.0.0.1:8766/#live and its Evaluation view, the API overview/health, 1,297 raw training records through about 4.04 hours, all routine development evaluations, host samples, and existing audit findings. Visible charts refreshed to the current logged steps when opened; operational status and audit checkpoint labels are distinct. The collector reports no ingestion issues.

Additional checks ran on an isolated CPU copy of checkpoint **24,868**, with two threads, in 75.7 seconds. They used development data only, made no optimizer steps or RBF writes, and preserved the same 96 balanced development identities as the older checkpoint 17,607 audit. The active trainer, source files and corpus were not changed. The dashboard's canonical balanced audit remains checkpoint 17,607; the newer results below are saved separately here, rather than silently relabeling that card.

## Operational health

At the snapshot, trainer PID 52954 was verified alive at step 26,040, with 55,387,810 **input tokens**, 4.06/24 hours elapsed, and approximately 19.94 hours remaining. The scheduled deadline is 12 October, 12:12 IST. Collector health was `ok`, with no last error. Checkpoint age was about 66 seconds against a 600-second cadence.

All logged losses and dense gradient norms checked were finite. The fresh checkpoint has finite parameters and finite gradients for all probed families/objectives. Future-token and prefix-consistency checks passed on all four families. The current root Python source hash matches the frozen training manifest:

`3cbaf426badbfc45f723431838c4f5340cbf7ce2c99f1a8438ed8e3b7152a0e9`

Window statistics are descriptive interval summaries, not globally token-weighted training losses. Each dense gradient value is the final update before clipping in a 20-step log interval; sparse Engram gradients are excluded.

| Elapsed hours | Mean token loss | Mean concept loss | Mean VQ loss | Median dense norm | Dense norm p95 | Median input tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| 0.00–0.50 | 4.814 | 0.483 | 11.272 | 0.96 | 2.96 | 3,877 |
| 0.50–1.00 | 3.756 | 0.328 | 6.097 | 1.31 | 3.24 | 3,918 |
| 1.00–2.00 | 3.262 | 0.296 | 4.848 | 1.54 | 3.60 | 3,909 |
| 2.00–3.00 | 3.036 | 0.314 | 4.780 | 1.86 | 3.63 | 3,964 |
| 3.54–4.04 | 2.829 | 0.331 | 4.780 | 2.15 | 3.58 | 3,766 |

Token prediction is improving. Concept and VQ losses improved early and then largely plateaued. The recent dense norm median is about 2.15, and every sampled recent update exceeds the configured clip threshold of 1. This indicates frequent clipping, not an automatic failure. The p95 remains around 3.6; the recent maximum is 8.0. There is no evidence here of numerical divergence. Logged throughput remains around 3,800–4,000 input tokens/s; evaluation, checkpoints and this CPU review can cause temporary dips.

Host headroom is limited: the latest training snapshot reports 2.16 GiB available, MPS driver 4.34 GiB and trainer RSS 0.96 GiB. The sampled host swap grew from 3.19 to 5.03 GiB between about 15:00 and 16:14 IST. Swap use alone does not prove active thrashing. These are system-wide or overlapping unified-memory counters and must not be added. GPU utilization, temperature and power are unavailable, so this review cannot establish thermal behavior.

## Text prediction: measurable progress

Comparable routine evaluations use the same deterministic sample and three batches. The initial step-0 evaluation uses a different batch count and is excluded from the progress comparison.

| Step | Text NLL | Text perplexity | Math NLL | Easy Sudoku NLL | Hard Sudoku NLL |
|---|---:|---:|---:|---:|---:|
| 3,185 | 5.7652 | 319.0 | 1.1956 | 1.3190 | 1.3129 |
| 6,441 | 5.2175 | 184.5 | 1.0357 | 0.9042 | 0.8904 |
| 9,779 | 4.9450 | 140.5 | 0.7325 | 0.7300 | 0.7497 |
| 13,117 | 4.8488 | 127.6 | 0.6263 | 0.5951 | 0.6027 |
| 16,494 | 4.7804 | 119.2 | 0.4019 | 0.4723 | 0.5265 |
| 19,668 | 4.7208 | 112.3 | 0.4265 | 0.4315 | 0.4556 |
| 22,721 | 4.6746 | 107.2 | 0.3613 | 0.5060 | 0.4864 |
| 25,937 | 4.6688 | 106.6 | 0.3823 | 0.4096 | 0.3887 |

Text NLL fell from 5.765 to 4.669, about 19.0%. Perplexity fell from 319.0 to 106.6. All eight comparable routine text evaluations decreased.

The fresh, larger text-only check sampled 32 unique sequences per split at context 384: training NLL **4.145**, development NLL **4.612**, development perplexity **100.7**, on 11,255 development target tokens. There is a train/dev gap of 0.468 nats. One checkpoint cannot establish whether that gap is widening; repeated fixed checks are needed before declaring overfitting. Mixed-family training loss must not be compared directly with text-only development NLL.

## Reasoning: success has not been established

Fresh strict generation at checkpoint 24,868, using greedy decoding with committed prefix codes and no answer repair:

| Family | Correct / trials | Older checkpoint 17,607 |
|---|---:|---:|
| Balanced math | 1/64 | 2/64 |
| Easy Sudoku | 0/16 | 0/16 |
| Hard Sudoku | 0/16 | 0/16 |

The one math success is sorting. The difference from 2/64 is too small to establish regression; neither result supports useful reasoning accuracy.

| Math module | Fresh correct / trials |
|---|---:|
| algebra__linear_1d | 0/16 |
| arithmetic__add_sub_multiple | 0/16 |
| arithmetic__mul | 0/16 |
| comparison__sort | 1/16 |

The latest routine evaluation at step 25,937 also scores 0/8 math and 0/2 for each Sudoku difficulty. Its eight math examples cover addition/subtraction only, so it must not stand in for the balanced check.

There is partial Sudoku learning. Easy outputs have valid 81-digit formatting in 15/16 cases, preserve every given clue in 12/16 cases, and achieve 99.23% clue accuracy among formatted outputs. Hard outputs have valid formatting in 16/16, preserve all clues in 14/16, and achieve 99.51% clue accuracy. Mean valid rows/columns/boxes are 5.67/1.07/3.67 for easy and 6.13/0.94/4.75 for hard, out of nine each. This supports learning formatting, copying clues and some local structure; it does **not** count as solving a puzzle or prove a reasoning algorithm.

The existing data audit found Python bytes-literal wrappers in all 239,904 math training answers, such as `b'10\n'`, and similar wrappers in questions. Reproducing these predictable tokens reduces answer NLL without computing the correct number. The fresh paired examples show incorrect generated answers despite good teacher-forced token accuracy. This is a data serialization and objective/metric problem; these logs do not establish reward hacking. The clean-data preparation path already exists separately and has not been applied to the active frozen run.

## Concept path and Engram

On the 32-sequence development text check, **NLL without concepts − NLL with concepts = -0.000236**. The descriptive paired-batch bootstrap 95% interval is **[-0.001414, 0.000973]**. It crosses zero. There is no measured useful text prediction benefit in this sample. Latest routine deltas are similarly small: text -0.001397, math -0.003082, easy +0.001119, hard +0.000482. Positive favors concepts; a negative value means the ablated model did slightly better on that sample.

Token-loss gradients reach the concept core, but their component norms are roughly 0.004–0.007 compared with 0.76–1.06 for the decoder in the small checkpoint probe. These norms cover different parameter groups and are not directly normalized contribution estimates. Combined with the ablation, they suggest the decoder currently depends weakly on concept guidance. This is a hypothesis about learned use, not proof that the architecture is incapable or disconnected. Nonzero pooled spread alone does not establish semantic concepts.

At the latest routine evaluation, text uses seven of eight conditioning codes, whereas each Sudoku family chooses one code for approximately 95% of selections. That concentration merits tracking together with contribution tests; it alone does not prove collapse, and forcing uniform codes is not an evidence-based fix.

Removing the entire Engram residual increases text development NLL by **0.524** on eight sequences. All four small family probes show increased NLL too. This demonstrates current dependence on Engram. It is not a compute-matched comparison against a model trained without Engram, nor proof that the table stores factual knowledge as intended.

The RBF bank is intentionally empty during pretraining. Continual-learning retention, transfer and local correction benefits remain untested. The logs also lack recurrence-pass ablations, so they cannot demonstrate a benefit from the second visit.

## Recommendation

Keep the current authorized 24-hour run as a diagnostic baseline, while judging it primarily by held-out text trends and strict answer checks. Do not call it a successful reasoning or concept-learning experiment on the current evidence. More time may help, but these measurements do not justify promising that the remaining hours will fix reasoning.

For the next controlled experiment: first use a separately prepared corpus with decoded math bytes and validate prompt/answer pairs; keep a larger fixed balanced development set with exact/constraint scoring; then compare concept guidance and recurrence passes on matched samples and controlled runs. Track text train/dev trends before spending beyond the current budget. Test RBF retention/transfer only in a separate continual-learning phase. Preserve the current run as the reproducible baseline rather than changing its corpus midway.

## Saved evidence

- `telemetry_analysis.json`: raw interval summaries and routine evaluation history at the analysis watermark.
- `latest_snapshot.json`: final API overview/health, source hash and paired-identity checks.
- `checkpoint_probe.json`: finite parameters, causality, gradients and small ablations at checkpoint 24,868.
- `fresh_evaluation.json`: larger text check, paired concept/Engram ablations and 96 strict balanced generation trials at checkpoint 24,868.
- Reproduce the bounded review with `python -m v10_10M.audit.review_telemetry --output-dir <new-review-folder>` while the local telemetry API is running.
