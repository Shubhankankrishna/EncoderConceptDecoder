# V6 tiny experiment report

Updated: 2026-10-04T20:16:50+05:30

Status: **completed**. Current stage: finished.

This is a 2.064M-parameter development experiment on one foundation/core seed. Independent foundation-seed confirmation and larger-scale stability remain untested. Text test documents are disjoint from training/dev after exact token-content deduplication, but come from the previously inspected V5 validation corpus.

## Correctness and throughput

Executable contract groups: 19/19 passed.

| Stage | Warm median step | Input slots/sec | Eager/compiled gradient error |
|---|---:|---:|---:|
| A | 0.224s | 54,827 | 2.4e-07 |
| B0 | 0.174s | 70,510 | 7.4e-08 |
| B1 | 0.273s | 45,074 | 2.2e-06 |

Selected generation execution: compiled_static_cache. Warm timings exclude compilation; long-run training logs include it. Input slots include padding and unscored headers. Actual targets and unique-data exposure are recorded in each stage summary.

## Training

| Run | Status | Completed/planned steps | Active parameters | Wall hours | Text NLL / graph NLL at last logged update |
|---|---|---:|---:|---:|---|
| A_ncp_s0 | completed | 18000/18000 | 2,022,144 | 1.21 | 1.7143 / 0.2618 |
| A_no_forecast_prefix_s0 | planned_prefix_completed | 9000/18000 | 2,022,144 | 0.69 | 1.9774 / 0.1247 |
| B0_det_s0 | completed | 6000/6000 | 2,055,424 | 0.36 | 1.8240 / 0.2455 |
| B0_extra_compute_s0 | completed | 9386/9386 | 2,055,424 | 0.51 | 1.8101 / 0.2464 |
| B1_direct_b025_s0 | completed | 6000/6000 | 2,057,472 | 0.54 | selection objective; see held-out NLL |
| B1_gated_b025_s0 | completed | 6000/6000 | 2,063,904 | 0.57 | selection objective; see held-out NLL |
| B1_gated_b0_s0 | completed | 6000/6000 | 2,063,904 | 0.53 | selection objective; see held-out NLL |

## Evaluation

| Evaluation | Set | Method | Success | Items | Seconds/problem |
|---|---|---|---:|---:|---:|
| eval_anchor_dev | standard | greedy | 14.06% | 64 | 0.024 |
| eval_anchor_dev | standard | retry | 25.00% | 64 | 0.298 |
| eval_anchor_dev | hard | greedy | 3.91% | 128 | 0.023 |
| eval_anchor_dev | hard | retry | 8.59% | 128 | 0.289 |
| eval_beta0_dev | standard | greedy | 14.06% | 64 | 0.024 |
| eval_beta0_dev | standard | retry | 25.00% | 64 | 0.293 |
| eval_beta0_dev | standard | prior | 28.12% | 64 | 0.238 |
| eval_beta0_dev | standard | gradcuit | 23.44% | 64 | 0.268 |
| eval_beta0_dev | standard | gradcuit_kl | 21.88% | 64 | 0.280 |
| eval_beta0_dev | standard | shared | 20.31% | 64 | 0.269 |
| eval_beta0_dev | standard | random_matched | 20.31% | 64 | 0.271 |
| eval_beta0_dev | standard | random | 15.62% | 64 | 0.242 |
| eval_beta0_dev | hard | greedy | 3.91% | 128 | 0.024 |
| eval_beta0_dev | hard | retry | 8.59% | 128 | 0.288 |
| eval_beta0_dev | hard | prior | 10.16% | 128 | 0.240 |
| eval_beta0_dev | hard | gradcuit | 6.25% | 128 | 0.276 |
| eval_beta0_dev | hard | gradcuit_kl | 6.25% | 128 | 0.285 |
| eval_beta0_dev | hard | shared | 7.03% | 128 | 0.276 |
| eval_beta0_dev | hard | random_matched | 4.69% | 128 | 0.278 |
| eval_beta0_dev | hard | random | 3.91% | 128 | 0.249 |
| eval_direct_dev | standard | greedy | 14.06% | 64 | 0.025 |
| eval_direct_dev | standard | retry | 25.00% | 64 | 0.269 |
| eval_direct_dev | standard | prior | 25.00% | 64 | 0.231 |
| eval_direct_dev | standard | gradcuit | 17.19% | 64 | 0.263 |
| eval_direct_dev | standard | gradcuit_kl | 17.19% | 64 | 0.273 |
| eval_direct_dev | standard | shared | 18.75% | 64 | 0.263 |
| eval_direct_dev | standard | random_matched | 17.19% | 64 | 0.262 |
| eval_direct_dev | standard | random | 15.62% | 64 | 0.232 |
| eval_direct_dev | hard | greedy | 3.91% | 128 | 0.023 |
| eval_direct_dev | hard | retry | 8.59% | 128 | 0.275 |
| eval_direct_dev | hard | prior | 8.59% | 128 | 0.229 |
| eval_direct_dev | hard | gradcuit | 7.03% | 128 | 0.267 |
| eval_direct_dev | hard | gradcuit_kl | 7.03% | 128 | 0.273 |
| eval_direct_dev | hard | shared | 5.47% | 128 | 0.268 |
| eval_direct_dev | hard | random_matched | 3.91% | 128 | 0.285 |
| eval_direct_dev | hard | random | 3.91% | 128 | 0.236 |
| eval_extra_anchor_dev | standard | greedy | 6.25% | 64 | 0.023 |
| eval_extra_anchor_dev | standard | retry | 18.75% | 64 | 0.283 |
| eval_extra_anchor_dev | hard | greedy | 2.34% | 128 | 0.022 |
| eval_extra_anchor_dev | hard | retry | 7.81% | 128 | 0.284 |
| eval_forecast_prefix_dev | standard | greedy | 0.00% | 64 | 0.020 |
| eval_forecast_prefix_dev | standard | retry | 0.00% | 64 | 0.255 |
| eval_forecast_prefix_dev | hard | greedy | 0.00% | 128 | 0.021 |
| eval_forecast_prefix_dev | hard | retry | 0.00% | 128 | 0.262 |
| eval_foundation_dev | standard | greedy | 6.25% | 64 | 0.020 |
| eval_foundation_dev | standard | retry | 14.06% | 64 | 0.247 |
| eval_foundation_dev | hard | greedy | 2.34% | 128 | 0.021 |
| eval_foundation_dev | hard | retry | 5.47% | 128 | 0.257 |
| eval_main_attempt_count_test | standard | greedy | 11.33% | 256 | 0.024 |
| eval_main_attempt_count_test | standard | retry | 23.83% | 256 | 0.285 |
| eval_main_attempt_count_test | standard | prior | 23.83% | 256 | 0.252 |
| eval_main_attempt_count_test | standard | gradcuit | 20.31% | 256 | 0.273 |
| eval_main_attempt_count_test | standard | gradcuit_kl | 20.31% | 256 | 0.286 |
| eval_main_attempt_count_test | standard | shared | 17.19% | 256 | 0.271 |
| eval_main_attempt_count_test | standard | random_matched | 15.62% | 256 | 0.275 |
| eval_main_attempt_count_test | standard | random | 12.50% | 256 | 0.244 |
| eval_main_attempt_count_test | hard | greedy | 4.88% | 512 | 0.024 |
| eval_main_attempt_count_test | hard | retry | 9.18% | 512 | 0.287 |
| eval_main_attempt_count_test | hard | prior | 11.91% | 512 | 0.252 |
| eval_main_attempt_count_test | hard | gradcuit | 8.59% | 512 | 0.276 |
| eval_main_attempt_count_test | hard | gradcuit_kl | 8.59% | 512 | 0.290 |
| eval_main_attempt_count_test | hard | shared | 5.86% | 512 | 0.276 |
| eval_main_attempt_count_test | hard | random_matched | 5.66% | 512 | 0.284 |
| eval_main_attempt_count_test | hard | random | 4.88% | 512 | 0.247 |
| eval_main_dev | standard | greedy | 14.06% | 64 | 0.023 |
| eval_main_dev | standard | retry | 25.00% | 64 | 0.283 |
| eval_main_dev | standard | prior | 28.12% | 64 | 0.234 |
| eval_main_dev | standard | gradcuit | 20.31% | 64 | 0.267 |
| eval_main_dev | standard | gradcuit_kl | 20.31% | 64 | 0.283 |
| eval_main_dev | standard | shared | 18.75% | 64 | 0.269 |
| eval_main_dev | standard | random_matched | 17.19% | 64 | 0.266 |
| eval_main_dev | standard | random | 15.62% | 64 | 0.238 |
| eval_main_dev | hard | greedy | 3.91% | 128 | 0.022 |
| eval_main_dev | hard | retry | 8.59% | 128 | 0.274 |
| eval_main_dev | hard | prior | 7.81% | 128 | 0.233 |
| eval_main_dev | hard | gradcuit | 5.47% | 128 | 0.263 |
| eval_main_dev | hard | gradcuit_kl | 5.47% | 128 | 0.276 |
| eval_main_dev | hard | shared | 5.47% | 128 | 0.264 |
| eval_main_dev | hard | random_matched | 3.91% | 128 | 0.268 |
| eval_main_dev | hard | random | 3.91% | 128 | 0.233 |
| eval_no_forecast_prefix_dev | standard | greedy | 79.69% | 64 | 0.023 |
| eval_no_forecast_prefix_dev | standard | retry | 100.00% | 64 | 0.122 |
| eval_no_forecast_prefix_dev | hard | greedy | 28.91% | 128 | 0.021 |
| eval_no_forecast_prefix_dev | hard | retry | 53.91% | 128 | 0.264 |
| eval_primary_gradcuit_test | standard | greedy | 11.33% | 256 | 0.023 |
| eval_primary_gradcuit_test | standard | gradcuit_kl | 20.31% | 256 | 0.293 |
| eval_primary_gradcuit_test | hard | greedy | 4.88% | 512 | 0.025 |
| eval_primary_gradcuit_test | hard | gradcuit_kl | 8.59% | 512 | 0.295 |
| eval_primary_retry_test | standard | greedy | 11.33% | 256 | 0.023 |
| eval_primary_retry_test | standard | retry | 23.44% | 256 | 0.286 |
| eval_primary_retry_test | hard | greedy | 4.88% | 512 | 0.023 |
| eval_primary_retry_test | hard | retry | 8.98% | 512 | 0.284 |

Equal attempt counts are reported separately from the locked elapsed-time comparison. Random matched updates use online gradient magnitudes and pay for that backward computation. Oracle fitting uses known valid trajectories and is excluded from ordinary repair scores.

## Draft gates and limits

- G0: prelaunch contracts plus post-training anchor/weight audits; inspect saved evidence, including any failure.
- G1/G2: report foundation/transplant NLL and prior/suffix degradation. Small mean differences alone do not establish the requested confidence margins.
- G3/G4/G5: verified prior diversity, reachability, paired repair-versus-random/retry gains, and measured costs. Item bootstrap intervals describe this one trained foundation; they do not replace independent training-seed variation.
- G6: shared versus local controls and L=4 likelihood are diagnostics; learned temporal abstraction is not established.
- BF16/FP16, distributed workers, long-context/width scaling, elastic depth, a full ordinary-transformer control and independent foundation seeds are deferred in this time-bounded experiment. FP32 and one Apple GPU are the executed configuration.
- The no-forecast control is compared with the matching 9,000-step foundation prefix under the same original 18,000-step schedule. It is not compared as an equally trained endpoint to the 18,000-step foundation.
- Time-limited/partial stages are labeled explicitly. Numerical failure never counts as completion. No V5 checkpoint or result is overwritten.

## Evidence files

- `PLAN.json`, `PRE_REGISTRATION.md`, `SPEC_COMPLIANCE.md`: fixed recipe, comparisons and coverage.
- `runs/status.json`, `runs/queue.jsonl`, `runs/driver.out`: queue status and failures.
- `runs/*/config.json`, `log.jsonl`, `checkpoint.pt`, `train_summary.json`: training/recovery provenance.
- `runs/eval_*/summary.json`, `items.jsonl`, `split_manifest.json`: NLL, every search attempt, verifier results, costs and instance identity.
- `tests/optimized_checks.json`, `runs/audit_*.json`: executable and trained-weight contracts.

## Observed anchor screen

Passed development mean screen: **True**. Text NLL gap: -0.00417; graph NLL gap: -0.01669; standard success loss: -7.81 points.

One seed and development means; not confirmation of confidence-bound gates

## Locked elapsed-time comparison

Budget: 4.620 seconds per batch of 16. Retry reference: /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v6_tiny/runs/B0_det_s0/checkpoint.pt.

- standard: repair minus retries -3.12 points; paired item 95% interval [-8.20, +1.95]; n=256.
- hard: repair minus retries -0.39 points; paired item 95% interval [-2.73, +1.96]; n=512.

Unresolved: independent foundation-seed confirmation and specified confidence gates remain required
