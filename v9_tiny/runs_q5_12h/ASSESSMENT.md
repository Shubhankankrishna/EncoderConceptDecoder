# Reduced Q5 assessment

The reduced queue completed in 26.6 minutes. Scientific comparison: inconclusive because GRPO collapsed after its first fit.

| Policy | Standard (512) | Hard10 (512) |
| --- | ---: | ---: |
| reference | 83.20% | 33.98% |
| EI | 85.16% | 33.01% |
| grpo | 0.00% | 0.00% |

No clear held-out development advantage for the MCTS-trained greedy policy over the original model. GRPO collapsed after round 0, so beating this GRPO result does not establish a credible MCTS advantage.

EI minus reference: +1.95 percentage points on Standard; −0.98 points on Hard10. Paired item intervals and discordant counts are in q5_12h_assessment.json. These are exploratory development estimates.

GRPO round 0 sampled 2,120 successful answers out of 4,096 before fitting. Round 1 produced 0 successes out of 4,096 and no learning signal; the adapter did not change in round 1. Its response-length and first-chunk failure classifications were independently checked. Maximum adapter parameter change after round 0: 0.0886972.

Completed files, group membership, rewards, fresh prompts and all checkpoint tensors verified. Collapse follows the first GRPO fit; round 1 has zero successes and no gradient-bearing groups. No operational crash or nonfinite checkpoint was found. The existing baseline uses an unregularized sequence-log-probability objective; the specific stability cause requires a separately documented experiment, not a silent learning-rate or algorithm change.

One seed, unequal training budgets, greedy development evaluation. No sealed test evaluation or test-time MCTS comparison.

All original and reduced completed outputs are preserved. The original Q6/QF queue remains held. No training settings were changed and no scientific rerun was started.

Independent CPU check: four of four fresh prompts also produced the malformed repeated DONE chunk [5,5,5,5], with finite first-step logits. This confirms the collapsed behavior independently of MPS.
