# Active Q5 12-hour pilot

Status: activated with explicit user approval at 02:30:10 IST on 11 October 2026. Deadline: 14:30:10 IST on 11 October 2026. Authoritative machine-readable protocol: `v9_tiny/runs_q5_12h/PROTOCOL.json`. The deadline includes remaining MCTS work, GRPO, evaluation and reporting.

## Scope

Use seed 921 only. As the user's approval specifically requests preserving the running MCTS work, allow the in-progress third round (iter2) to finish untouched, then freeze its verified heads as the EI comparison checkpoint. Retain iter0, iter1 and iter2 and all original artifacts. Do not start the two remaining original EI rounds. The two already completed rounds used 23,043.9228 seconds (6.40 hours) and 66,761 training verifier calls; the report will also include the completed third round's measured cost. The EI comparison consequently uses three completed rounds while reduced GRPO uses two; this is an explicitly unequal-training-budget pilot.

Run one GRPO baseline from the same original S2_K8_dense_s911 checkpoint, not from EI's trained selector. Remove RFT, REINFORCE and seed 922 from the reduced queue. Retain GRPO's rank-16 q/v LoRA, group size 8, temperature 1, reward/advantage rules, optimizer settings, and two training rounds.

Nominal GRPO exposure: the same 512 fresh problems from each of the first two EI rounds, one complete group of eight answers per problem: 4,096 answer-verifier calls per round, 8,192 total. Calibration fixes the largest declared exposure that fits before scientific GRPO training begins. Call reduction is not a prediction of the same speedup.

## Twelve-hour allocation

| Elapsed time | Work |
| --- | --- |
| 0:00–1:00 | Freeze/snapshot original queue and artifacts; coordinate watchdog and dependency waiters; validate revised runner; short MPS GRPO calibration after production releases MPS. |
| 1:00–9:00 | Single GRPO training job, two rounds, with a strict training deadline and time reserved inside this window for fitting and checkpoint saving. |
| 9:00–11:00 | Greedy development evaluation of reference, frozen EI, and GRPO on identical Standard and Hard10 items. Reuse the verified reference result where identical. |
| 11:00–12:00 | Paired item results, accuracy differences, actual training time/calls/exposures, artifact hashes, and reduced-run report. |

The calibration measures sampling plus a conservative fitting benchmark with all groups receiving a gradient, even if the actual calibration samples have constant rewards. Synthetic benchmark rewards and all calibration outputs are discarded as scientific results. Before any GRPO training, choose and freeze 512 problems/round if projected training fits the remaining training window with a 25% runtime margin; otherwise choose the largest of 256 or 128 that fits. Use predetermined prefixes of the same fresh problem lists, never select problems by success. Record the selected exposure before training and do not change it based on scores. If even the smallest workload does not fit, report that limitation within the deadline rather than silently extending the run.

Save optimizer/model/RNG and progress state at usable GRPO boundaries, including after each completed round. Add bounded-work/deadline handling and preserve whole GRPO groups. Stop admitting new sampling work early enough to fit, save and evaluate. If a failure or overrunning operation prevents a complete comparison, deliver a clearly incomplete report at the deadline; do not mark an unfinished job completed.

Operational memory controls: sample four complete groups at a time (32 answers), accumulate eight-answer gradient microbatches into the original 64-answer optimizer batch, and recycle MPS workers only after saving a complete sampling or optimizer boundary when driver allocation reaches 4 GB. CPU parity validation confirms the GRPO objective and updates match the original implementation within numerical precision. An interrupted fit restores its Adam state, order, model and RNG state; an interrupted sampling batch resumes from the last saved whole group. Completed scientific outputs are reused, not rewritten.

## Interpretation

Primary question: how does this reduced GRPO policy compare with the already-trained EI policy under the same greedy development evaluation? Include the original reference.

This is a one-seed exploratory pilot with unequal training verifier-call budgets; if smaller problem prefixes are used, training exposure also differs. It is not the original call-matched, two-seed Q5 and cannot establish an equal-training-compute MCTS advantage or seed robustness. Report time and calls explicitly. Keep sealed test data untouched; sealed evaluation belongs to the later revised final workflow.

## Activation and validation

Do not mutate old results or claim original Q5 completion. Use `v9_tiny/runs_q5_12h` for the separate protocol, lock, queue, checkpoints and report. Preserve and hash all 145 previously protected completed artifacts, then add the third EI round's outputs. Validate changed queue, fixed GRPO group sampling, checkpoint/resume and deadline behavior with isolated CPU contracts/smoke before MPS production starts; do not bypass preflight or refresh locks without validation. Coordinate with the existing watchdog to avoid restoring the original full Q5 or launching duplicates. The heartbeat monitor now follows this amended protocol and its deadline.

Hold the original Q6/QF dependency drivers during activation. The reduced report is Q5_12H_RESULT/REPORT, not a fabricated full Q5_RESULT. Adapt downstream Q6/QF separately to the reduced arm/seed list, with validated dependencies; their runtime is outside this Q5 twelve-hour budget.
