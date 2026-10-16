# V7 tiny experiment report

Generated UTC: 2026-10-05 12:57:10. Observed queue status: running.

This is a bounded tiny-model study. Numerical/caching smoke checks establish implementation contracts; they do not establish model quality or scaling.

V7 foundation: **5,365,938** active parameters. Ordinary Transformer: **5,366,208** (270 more, 0.0050%). Exploration adds **41,472** parameters; deployment has **5,407,410**. A resident frozen teacher is excluded from active deployment counts.

Common foundation budget: **25,000,000** scored targets per arm and seed.

| Run | Completed scored targets | Train status | Fresh text NLL | Standard strict | Hard strict |
|---|---:|---|---:|---:|---:|
| A_alpha0_s171 (dev) | 25,004,499 | completed | 2.937 | 0.0% | 0.0% |
| A_alpha0_s272 (dev) | 25,002,474 | completed | 2.925 | 0.0% | 0.0% |
| A_alpha0_s373 (dev) | 25,003,844 | completed | 2.990 | 0.0% | 0.0% |
| A_flat_s171 (dev) | 25,004,499 | completed | 2.897 | 0.0% | 0.0% |
| A_flat_s272 (dev) | 25,002,474 | completed | 2.899 | 0.0% | 0.0% |
| A_flat_s373 (dev) | 25,003,844 | completed | 2.917 | 0.0% | 0.0% |
| A_ncp_s171 (dev) | 25,004,499 | completed | 2.964 | 0.0% | 0.0% |
| A_ncp_s272 (dev) | 25,002,474 | completed | 2.920 | 0.0% | 0.0% |
| A_ncp_s373 (dev) | 25,003,844 | completed | 2.957 | 0.0% | 0.0% |
| B0_alpha0 (dev) | 8,006,046 | completed | 2.934 | 0.0% | 0.0% |
| B0_ncp_s171 (dev) | 8,006,046 | completed | 2.961 | 0.0% | 0.0% |
| B0_ncp_s272 (dev) | 8,005,962 | completed | 2.917 | 0.0% | 0.0% |
| B0_ncp_s373 (dev) | 8,003,071 | completed | 2.953 | 0.0% | 0.0% |
| B1_alpha0 (dev) | 6,002,013 | completed | 2.934 | 0.0% | 0.0% |
| B1_ncp_s171 (dev) | 6,002,013 | completed | 2.961 | 0.0% | 0.0% |
| B1_ncp_s272 (dev) | 6,001,135 | completed | 2.917 | 0.0% | 0.0% |
| B1_ncp_s373 (dev) | 6,000,773 | completed | 2.953 | 0.0% | 0.0% |
| best_Z (dev) | 6,002,013 | completed | 2.961 | 0.0% | 0.0% |
| det_extra (dev) | 6,004,162 | completed | 2.960 | 0.0% | 0.0% |
| direct_Z (dev) | 6,002,013 | completed | 2.964 | 0.0% | 0.0% |
| full_extra (dev) | 6,004,162 | completed | 2.933 | 0.0% | 0.0% |
| mean_Z (dev) | 6,002,013 | completed | 2.961 | 0.0% | 0.0% |
| pilot_alpha0 (dev) | 3,000,621 | completed | 4.343 | 0.0% | pending% |
| pilot_alpha001 (dev) | 3,000,621 | completed | 4.345 | 0.0% | pending% |
| pilot_alpha1 (dev) | 3,000,621 | completed | 4.397 | 0.0% | pending% |
| pilot_flat_lr0.0005 (dev) | 3,000,621 | completed | 4.390 | 0.0% | pending% |
| pilot_flat_lr0.001 (dev) | 3,000,621 | completed | 4.945 | 0.0% | pending% |
| pilot_flat_lr0.002 (dev) | 3,000,621 | completed | 5.052 | 0.0% | pending% |
| pilot_ncp_lr0.0005 (dev) | 3,000,621 | completed | 4.439 | 0.0% | pending% |
| pilot_ncp_lr0.001 (dev) | 3,000,621 | completed | 4.323 | 0.0% | pending% |
| pilot_ncp_lr0.002 (dev) | 3,000,621 | completed | 4.771 | 0.0% | pending% |
| pilot_plain_alpha0 (dev) | 3,000,621 | completed | 4.301 | 0.0% | pending% |
| pilot_plain_ncp (dev) | 3,000,621 | completed | 4.306 | 0.0% | pending% |

Matched foundation differences use whole held-out document resampling for token-weighted NLL, and paired complete problems for task success. Positive task differences favor the first arm; negative NLL differences favor it.

## Refit, exploration and deployment


## GradCuit and AB-MCTS

Development lock: {}


## Limits and interpretation

NCP uses feature-mean forecast/VQ losses and bounded routing adaptations. GradCuit uses bounded concept controls and response-mean rejected log-probability descent, rather than the paper’s raw hidden prefix/sequence-sum objective. AB-MCTS-A searches realized complete answers; latent-only dynamics and prefix-tree MCTS are outside this run.

Three separately trained seeds are listed individually; problem-level bootstrap intervals are conditional on each trained model. Larger widths, long-context behavior, BF16, distributed training and LLM-scale gains are not established. Semantic deduplication and graph-isomorphism independence are not claimed.

Token budgets are matched; parameter counts are almost exact. Additional compute for the V7 recipe, tuning and inference is recorded rather than treated as free. A partial endpoint cannot support a matched-completion comparison. Inference port interventions are distribution-shift diagnostics, not retrained ablations.
