# v4.1 tiny experiment: evidence and first-principles assessment

**Status: completed.** Generated 2026-10-02T11:23:59.986777+00:00.

14/14 new full-budget runs complete; 82,003,490 real tokens and 48,916 optimiser steps. This includes the 11 registered architecture arms, one fresh transformer reproduction and two exploratory MPS controls. The previous 10 baseline runs are reused with hashes, not counted as new training.

Read [FIRST_PRINCIPLES.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/FIRST_PRINCIPLES.md) for the architecture, [METHODS.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/METHODS.md) for controls and reproduction, and [RESULT_TABLES.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/RESULT_TABLES.md) for every seed, intervention, oracle learning rate and statistical difference.

## Assessment

**v4.1 is a real partial improvement over v4 at this scale, but the tested
concept system has not earned a claim of learned hierarchical reasoning.**
Across three pinpad seeds, adding the retrieval package raises held-out action
accuracy by 5.55 percentage points and pair-second accuracy by 16.67 points.
It closes about 42% of the measured v4-to-transformer action gap. The transformer
still leads by 7.76 action points and 12.28 pair-second points at the matched
token budgets. Language loss improves slightly across both text seeds. These
are useful capability gains, conditional on the tasks, widths and budgets used.

The fully replicated controls support keeping the private decoder for the
next study: the decoder stack lowers action accuracy by 3.27 points on average.
Removing the repeated encoder F pass changes pair-second accuracy by +0.72
points, with 0.87 points of paired item SE and 1.90 points of training-seed SE.
It fails the registered keep criterion while saving about 18% of the estimated
pinpad forward FLOPs. Use one encoder F pass in the next variant; its trained
text guard still needs checking. Keep concept-core pass count separate from
that encoder decision: trained M=1 was not tested here.

The transformer and two stack seeds are still improving at the final budget.
The registered comparison is therefore budget-limited; the measured ranking
does not establish a converged or full-size ranking.

From first principles, this makes sense. The model needs access to the current
cell, a relevant earlier colour, the relation between colours and the target's
location. R supplies a direct read of earlier token features, relieving the
compressed concept route of carrying every distant fact. Its causal mask and
private-plan boundary work as specified. However, an information route that
exists need not carry a representation that improves decisions.

The calibrated tests expose that distinction. Replacing v4.1's example-specific
plans with one training-derived mean lowers pair-second accuracy by only 1.61
points, versus 11.28 points in v4 without R. The effect is small rather than
literally zero. On text, the same replacement costs 0.040 nats, so plans have
more visible utility there. Nonconstant vectors, active gradients and weakly
readable task information should not be mistaken for either a completely idle
core or proof of useful abstract reasoning.

The strongest warning concerns the validation gate. The learned header null
is a different numerical input from a normal body plan. A model trained without
reading any concept content still suffers a large `core_off` drop when that
null is inserted. Thus the proposed switch-off can produce a false positive
for route use. This gate needs calibration before it can justify a large
architecture decision.

The matched GPU training control is also worth following up: removing decoder
plan content during training improves action accuracy by 6.74 points and
pair-second accuracy by 11.17 points in seed 0. It also follows changed facts
better. This is a post-hoc, one-seed result, and auxiliary concept learning
remains present. It supports testing a simpler or gated plan injection; it
does not establish that all concept training is useless.

The frozen oracle provides encouraging control evidence: legal C offsets repair
25 of 52 initial errors at the largest tested learning rate without damaging
the 76 initially correct cases. This uses correct labels, full-width offsets
and one-step decisions. A learned low-dimensional controller, useful switching
and long-horizon control remain untested.
No matched transformer-residual oracle was tested, so steerability alone does
not establish a unique concept-interface advantage.

My score gives substantial credit to the precise information boundaries,
implementation checks, inexpensive capability gain and steerable latent site.
It deducts for the remaining transformer gap, weak generated-policy structure,
limited evidence for useful example-specific abstractions and the invalid
route-attribution gate. It is a judgment of the tested design, not a prediction
that a larger model must get the same score.

The next worthwhile study should replicate normal versus trained plan-off
across seeds on pinpad and text, then compare a gated plan injection. Keep R
as a demonstrated candidate. Require fact-following, calibrated plan utility
and generated task structure before treating the concept interface as learned
planning. The fully replicated stack and encoder-loop decisions are reported
in the [report](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/REPORT.md)
using the original rules.

**Updated score: 6.5/10.** This is an engineering judgment about the tested architecture, not a benchmark-derived scale or a claim about eventual full-size performance.

## 1. What was built and tested

The tiny model has width 64, an eight-layer local/global encoder, attention pooling over six-token chunks, a four-layer concept core applied twice, and decoder layers S S R S. R reads the entire causal token memory, with NoPE and four temperatures, while plans remain private to each decoder position. The pooler has no FFN and the decoder has no Engram reads. Encoder Engram reads remain.

The inherited v4 backbone starts bit-identically for the same seed. R adds precisely four parameters. The decoder stack control has identical parameters and reads private decoder states for K/V; the trained loop-off control uses each encoder F layer once. No full-size model was trained.

| Model | Compute parameters | Pinpad stored parameters | Text stored parameters | Estimated pinpad forward FLOPs/token |
|---|---:|---:|---:|---:|
| v4 / no R | 1,212,032 | 1,630,048 | 2,149,728 | 2.810M |
| v4.1 / stack | 1,212,036 | 1,630,052 | 2,149,732 | 2.869M |
| Transformer LLM | 1,206,864 | 1,211,472 | 1,731,152 | 3.779M |

The full-size v4.1 ledger is 19,175,580 compute parameters and 101,600,284 stored parameters. The smaller headline excludes embedding, Engram table and codebooks; those are not free memory. The FLOP approximation excludes some gates, lookup/pooling work, auxiliary heads and convolution work. R raises this estimate about 2.1% on pinpad and 1.5% on text. The transformer is about 32% higher on pinpad, with local window 60 versus 8. This is an equal-token/near-equal-parameter comparison, not an equal-FLOP experiment.

CPU was faster for the tiny training workload in synchronised tests. Registered training therefore used the verified CPU path; the available Apple M2 GPU ran both full 6M-token MPS controls, closed-loop inference and oracle control. Every budget and actual device/thread count is in the inventory.

## 2. Correctness is established for the tested graph

**42 distinct acceptance checks passed; the 15 new v4.1 checks were also repeated without the block kernel (57 executions total). All 29 deliberate defects were detected by assertions.** The tests check more than loss reduction: global retrieval reach, future/document/row isolation, exact private-plan reach, completed-chunk alignment, legal nulls, full-model causality, gradients and actual data rows.

The manual attention oracle agrees within 4.44e−16 in float64. Full/block normal-model logits agree within 8.88e−16. CPU/MPS logits differ by at most 3.58e−7, and all 489 parameter gradients were compared, with worst absolute difference 7.26e−8.

A fresh transformer seed-0 run reproduced all 246 stored state tensors bit-for-bit and every rollout outcome. Original source hashes (24 files) and all 50 copied reference artifacts and originals are unchanged. Cumulative real-token counts also match the baseline at every shared logged training step.

The inference-artifact audit verifies finite checkpoint tensors, saved budgets/arguments, per-unit array shapes and exact reconstruction of reported aggregates. It also recomputes counterfactual correctness from the saved action IDs and fixture labels. It caught one seed-0 Engram switch-off summary with missing raw units; those units were regenerated from its checkpoint, the summary reproduced exactly, and every existing primary unit stayed bit-identical. The before/after artifacts and repair record are retained.

The fresh transformer reproduction also inherited its reference’s summary-only M1/loop-off records. Those raw units were completed from the fresh checkpoint, reproducing both original summaries exactly and preserving all old units. The original reference files were not modified.

Two engineering failures are retained: the initial MPS float64 rollout conversion, fixed by transferring logits to CPU first; and a strict CPU2/CPU4 bit-identity assertion, which failed despite maximum six-step state difference only 7.45e−9. These do not become hidden “passes.” The latter is numerical agreement, not exact full-run identity. See [ENGINEERING_LOG.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/ENGINEERING_LOG.md).

## 3. Capability at the fixed budgets

Pinpad values are equal-weight means over episodes, then seeds. b2 is the first action toward a pair’s second colour; chance is approximately 0.35.

| Model | Completed seeds | Held-out action accuracy | Held-out b2 | Action NTP | Sampled pair integrity | Greedy pair integrity |
|---|---:|---:|---:|---:|---:|---:|
| v4 / no R | 3 | 0.6936 | 0.5200 | 0.9110 | 0.1449 | 0.1410 |
| v4.1 | 3 | 0.7491 | 0.6867 | 0.7505 | 0.1748 | 0.2496 |
| Transformer LLM | 3 | 0.8268 | 0.8094 | 0.6100 | 0.3260 | 0.4387 |
| v4.1 stack | 3 | 0.7165 | 0.6000 | 0.8336 | 0.1558 | 0.1818 |
| v4.1 trained loop off | 3 | 0.7540 | 0.6939 | 0.7532 | 0.1823 | 0.2091 |

**v4.1 − no R, action accuracy:** +0.0555 ± 0.0033 paired item SE; seed SE 0.0037 (n=3).
**v4.1 − transformer, action accuracy:** -0.0776 ± 0.0032 paired item SE; seed SE 0.0125 (n=3).

The primary decision follows the registered paired-item rule. Seed SE and ranges are separately reported because the same held-out items do not create additional independent training seeds.

Text comparison uses mean per-row NTP; the runner’s token-weighted values are also in the tables. Text v4.1 − no R is -0.0107 ± 0.0018 paired item SE; seed SE 0.0034 (n=2). The R guard is +0.02 nats. Retrieval exact match remains near zero at these budgets; it cannot establish the draft’s proposed division of labour on multi-hop language questions.

Closed-loop completion is not successful composition. The sampled policy can visit six colours while violating the required pair order. Full valid-composition/task rates remain low, even for the stronger transformer. Pair integrity is conditional on an eligible even-colour transition; the tables report coverage and an all-layout convention beside it. Teacher-forced action scores measure behaviour on expert histories; generated histories are a separate distribution.

Under the secondary convention assigning zero when no eligible pair was observed, v4.1 − transformer pair-integrity score is -0.1505 ± 0.0142 paired item SE; seed SE 0.0613 (n=3) for sampling and -0.2264 ± 0.0161 paired item SE; seed SE 0.0614 (n=3) for greedy. The deficit therefore persists when every layout contributes to the score.

![Training curves](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/learning_curves.png)

## 4. The registered decisions

| Rule | Reading |
|---|---|
| R outcome | O2 partial fix (final) |
| R text guard | passes: v4.1 − no R ≤ +0.02 |
| Stack improves action accuracy by ≥2 paired SE | False |
| Keep encoder F loop by trained b2 rule | False |

Stack effect: -0.0327 ± 0.0029 paired item SE; seed SE 0.0129 (n=3). Trained loop-off b2 effect: +0.0072 ± 0.0087 paired item SE; seed SE 0.0190 (n=3). No loop-off text arm was run; the registered loop rule therefore uses pinpad.

The formal core-route kill line uses `core_off`. The raw rule is applied as written in [registered_decisions.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/runs/registered_decisions.json). The calibrated controls below change its interpretation, not its recorded outcome. Curves and the final three held-in b2 readings are retained. These are finite-budget results; no architecture was extended beyond its registered budget to select a better checkpoint.

**Budget-limited comparison:** 3 of three transformer seeds improved held-in b2 over all of their final three evaluations. The transformer is the stronger P1 arm, so the original rule labels the R/transformer comparison budget-limited. These results describe the registered token budget; full convergence was not tested.

## 5. Core reliance is smaller than the usual switch-off suggests

`core_off` replaces every body plan with the learned header null. That is not the same as removing only example-specific content. In CPU seed 0, the learned null has RMS 0.966, while the training-bank mean valid projected content has RMS 0.199; their RMS distance is 1.001 and cosine −0.075. The replacement changes both scale and direction.

For v4.1 pinpad, replacing all valid u/c_pred with one training-derived mean changes b2 by -0.0161 ± 0.0079 paired item SE; seed SE 0.0055 (n=3). Removing their content while retaining normal nulls and six slot offsets changes b2 by -0.0194 ± 0.0083 paired item SE; seed SE 0.0049 (n=3). The header-null replacement changes it by -0.1583 ± 0.0146 paired item SE; seed SE 0.0527 (n=3).

Thus a large null switch-off cannot by itself establish strong example-specific planning. The plan vectors are not literally constant and gradient routes are active, but much of checkpoint accuracy survives constant content. On text, constant plans cost +0.0398 ± 0.0019 paired item SE; seed SE 0.0063 (n=2) nats, so the conclusion differs by task.

Applying the same calibrated control to the three v4/no-R reference checkpoints gives a constant-plan b2 effect of -0.1128 ± 0.0123 paired item SE; seed SE 0.0101 (n=3). The paired difference between the R and no-R constant-plan effects is +0.0967 ± 0.0147 paired item SE; seed SE 0.0155 (n=3). The retrieval package increases task capability while these checkpoints rely less on example-specific plans. This does not prove that auxiliary concept training had no benefit.

The gradient audit separates action NTP from coarse/fine auxiliaries. Both reach the intended modules; their norm/cosine measurements are diagnostics, not causal importance scores. See [gradient_audit.json](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/runs/v41_pin_s0_6M/gradient_audit.json).

## 6. A trained control and counterfactual facts

The exploratory GPU plan-off model retains core computation, NCP/VQ objectives, allocated parameters, normal header nulls and slot offsets. It disables u/c_pred content throughout training. Its 8,192 plan-projection weights stay exactly at initialisation, and NTP-only gradients do not reach its core/pooler. Auxiliary training still acts on encoder representations, so this is not a pure model-without-concepts or model-without-auxiliaries comparison.

It is compared with a fresh normal-plan seed-0 run on the same MPS backend, same data and budget. This removes the earlier CPU/MPS comparison confound. Both are post-hoc, one-seed controls.

**A direct false positive in the route gate:** the trained no-plan model has no inference path from core outputs to logits, yet its standard `core_off` lowers b2 from 0.7967 to 0.4783. NTP-only core gradients are absent and Wu/Wc remain at initialisation. Substituting the header null into the body produces the drop even when no concept content exists to remove. Thus that switch-off is not sufficient evidence of concept-route use. The registered formal gate is still recorded exactly as written; its intended inference is invalid in this counterexample.

| Matched MPS arm | Action accuracy | Pair-second b2 | Action NTP | Changed-fact accuracy |
|---|---:|---:|---:|---:|
| Normal v4.1 | 0.7530 | 0.6850 | 0.7436 | 0.3047 |
| Trained without plan content | 0.8204 | 0.7967 | 0.6476 | 0.6328 |

The layout counterfactual moves the target to another unvisited colour’s location. The visited even partner stays fixed; both prefixes have identical legal histories and disjoint optimal next actions. The changed layout remains a valid layout. Across the CPU v4.1/transformer pairs, changed-fact accuracy differs by -0.2786 ± 0.0280 paired item SE; seed SE 0.0750 (n=3). The conditional-on-original-correct rate is descriptive because models have different original-correct subsets.

![A measured counterfactual](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/counterfactual_case.png)

This illustration is seed 0 and selected by a stated rule; all 128 cases and all completed seeds, rather than the image, determine the numerical findings.

## 7. Abstraction and control are separate

The table reports coarse-code NMI beside an episode-wise permutation null, and causally legal completed-chunk linear probes beside a majority predictor. Codebook usage is not proof of semantic abstraction. NMI is explicitly non-causal and can be positive from finite-sample bias. A small probe improvement is evidence of some readable information, not evidence that the core performs the task’s decisions.

The frozen oracle demonstrates a different property: available control authority. On 128 fixed prefixes (52 initial errors), ten C-offset steps at LR .03 fix 25 errors (48.1%) without breaking the 76 initially correct cases. u fixes 22/52 and first-pass S1 fixes 7/52 at that learning rate. Norm-matched random updates fix none. All three learning rates and achieved norms are reported, not only the best.

The oracle knows the correct actions and optimises full-width prompt offsets. It neither trains an eight-dimensional metacontroller nor tests sparse-reward RL or learned option duration. It shows that the legal latent route can steer outputs when supplied a useful offset. No matched oracle intervention in a transformer residual stream was run, so this does not establish a unique advantage of the concept interface.

![Critical controls](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/critical_controls.png)

## 8. Why long-episode likelihood can give the opposite impression

The exact decomposition is `action NLL = action-conditioned NLL + non-action probability-mass penalty`. On long episodes, the transformer’s penalty often rises sharply after six visits, when the training tasks would have ended. END accounts for almost all of the missing action probability mass. Large log penalties are sensitive to highly confident premature stopping.

For example, transformer seed 0 after six visits has 0.668 action-conditioned NLL plus 2.287 non-action penalty, totalling 2.955 nats; v4.1 has 0.759 plus 0.525, totalling 1.284. The transformer still chooses optimal actions more often in that phase (79.5% versus 74.2%). This measured decomposition explains why lower full-vocabulary NLL alone is not stronger long-horizon action reasoning. All seeds and the exact numerical identity check are in [LONG_AUDIT.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/LONG_AUDIT.md).

## 9. Limits and the next useful experiment

These are width-64 models trained on 5–6M tokens, two text seeds and three pinpad seeds. They do not settle full-size performance. Some better-model curves remain rising at the budget. The architecture comparison differs in local windows, stored tables, auxiliary objectives, initialisation and FLOPs. The R/no-R contrast also changes positional encoding and temperatures alongside range. The stack’s local dense kernel affects throughput.

The constant-plan, fact-following, trained plan-off, NMI null and long-likelihood checks are secondary, introduced after early results and clearly labelled. The trained plan-off result needs additional seeds. Natural chunk-lag strata are observational. Oracle control has privileged labels. New-name, longer-gap, distractor-count language OOD factors, k=4/k=8, wider windows, G-to-R, depth rebalancing, trained M=1, controller learning and VQ-only adaptation were not tested here.

The next controlled comparison should replicate normal versus trained plan-off across seeds on both tasks, then test a calibrated/gated plan injection against those two arms. Keep R as a demonstrated capability candidate. Require example-specific plan utility, fact-following and generated-policy structure before interpreting the concept interface as learned hierarchical reasoning. The transformer remains an essential baseline.

## Artifacts

- [FIRST_PRINCIPLES.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/FIRST_PRINCIPLES.md) — step-by-step graph, timing, codebook and control-site derivation.
- [METHODS.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/METHODS.md) — experiment specification, statistical definitions, known confounds and commands.
- [RESULT_TABLES.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/RESULT_TABLES.md) — all completed seeds and controls.
- [LONG_AUDIT.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/LONG_AUDIT.md) — action likelihood decomposition.
- [PRE_REGISTRATION.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/PRE_REGISTRATION.md) and [SECONDARY_PROTOCOL.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/SECONDARY_PROTOCOL.md) — primary decisions and later amendments.
- [raw comparisons](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/runs/v41_compare.json), [secondary comparisons](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/runs/secondary_compare.json), [provenance audit](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/runs/artifact_validation.json), [inference-artifact audit](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/runs/analysis_artifact_audit.json).
- [implementation](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v41.py), [runner](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/run_vs.py), [test suite](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/test_v41.py).

