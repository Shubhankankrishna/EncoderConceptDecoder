"""Build a reproducible written review from the frozen logs and CPU diagnostics."""
from pathlib import Path
import json, math, statistics
A=Path(__file__).resolve().parent
ROOT=A.parent
S=A/'source_snapshot'
old=ROOT/'audit_v2_2026-09-28'
snap=json.loads((A/'snapshot.json').read_text())
d=json.loads((A/'checkpoint_000267_diagnostics.json').read_text())
g=json.loads((A/'gradient_replication.json').read_text())
r=json.loads((A/'routing_replication.json').read_text())
inv=json.loads((A/'inventory.json').read_text())

def render(s):
    return s.replace('{{A}}',str(A)).replace('{{S}}',str(S)).replace('{{ROOT}}',str(ROOT)).replace('{{OLD}}',str(old))

report=r'''# v2 underperformance: evidence, mechanisms, and a controlled recovery plan

**Evidence cutoff: 28 September 2026, 22:34:33 IST.** This is an early-training diagnosis of `run_v2_19M/runs/v2_48h`, using official evaluations at steps 138 and 267, training summaries through step 275, and heartbeat step 281. Training was about 2.2 hours into a planned 48-hour run. Later results can change the conclusion.

## 1. Direct answer

**Your concern is justified: v2 is learning language more slowly than v1, even after matching the amount of training data. However, the evidence does not support the simple explanation that repeated passes are destroying token prediction.** At fixed v2 weights, six passes improve token loss over one pass on all 16 sampled validation rows at checkpoint 138 and on 15 of 16 at checkpoint 267; both language averages improve. Further passes provide very small additional gains. Concept prediction behaves differently: its loss is generally best around pass 2 and worsens afterward.

The more defensible diagnosis is that **v2 spends substantially more work in the concept module while weakening the encoder/decoder, and its training objectives and routing policy make that work harder to use effectively.** Four findings deserve attention:

1. **NextLat competes with next-token prediction on important shared weights.** Its combined auxiliary gradient opposes the token-loss gradient in the core input projection, expert output weights, decoder plan projection, and carry gain across four separate training microbatches at step 267. This is the strongest measured optimization suspect, not yet proof that removing it improves the eventual model.
2. **MoE capacity allocation systematically favors the first batch row.** In four sampled batches, real chunks in the second row lose 17.7–29.2% of expert assignments, versus 0–3.8% in the first row. Evaluation removes this dropping. This is an actual training-policy asymmetry, not merely an alarming dashboard color.
3. **The initial-state projection receives no final language-loss gradient.** State detachment after pass 4 cuts that path; it learns only from early concept and routing losses. Later shared loop weights still receive language gradients. Numerical backward tests pass, so this is a consequence of the training design rather than a broken autograd implementation.
4. **The capacity and compute tradeoff has not paid off yet.** Encoder plus decoder non-embedding parameters fell 52.7%, while the average number of concept-layer applications rose from 2.5 to 40. At steps 201–250, v2 processes about 30.5% fewer token slots per second and has 0.520 nats worse training NTP on the same data.

**My provisional score for the current demonstrated training recipe is 5/10.** That is a diagnostic score, not a prediction of the final 48-hour model and not a rejection of looping as an architectural idea. The model is learning, the codebook is recovering, and I found no numerical divergence or general backward failure. Section 14 gives the scoring rubric.

![Four measured findings: learning curve, depth effect, gradient conflict, routing bias]({{A}}/evidence_summary.png)

## 2. What was reviewed, and which “v2” this means

There are two naming layers in this repository. The earlier trained model under `run_m2_25M` uses a class named `ConceptCoreV2`; it is **v1** in the present comparison. The newly launched whole-model **v2** is the D6/MoE implementation under `run_v2_19M`. The earlier afternoon audit described the whole-model v2 proposal before its implementation. That status is now obsolete: the evening implementation and training run exist and were inspected directly.

I inventoried all 99 files under the new run at the snapshot cutoff: 72 project files and 27 generated/cache/vendor files. The project inventory includes source, configuration, decisions, tests, launch and monitoring scripts, dashboard code/data, logs, and checkpoint aliases. All project text was captured and parsed or indexed; numerical claims were traced through the model, trainer, evaluator, data path, and dashboard. Both distinct available step checkpoints were loaded for CPU diagnostics. Hard-linked `last` and `best` aliases do not represent additional independent models.

The shared data and v1 history were compared against the earlier full audit. All 30 shared data files had unchanged inode, byte size, and modification time. I reused the earlier corpus/shard/overlap inspection instead of rereading hundreds of millions of tokens. This is strong continuity evidence, not a fresh cryptographic proof against an undetectable in-place rewrite. The new epoch-reshuffle change leaves epoch 0 identical; both runs are still being compared within that epoch.

The [file coverage appendix]({{A}}/FILE_COVERAGE.md) identifies every project file and what was done with it. The [results appendix]({{A}}/RESULTS_APPENDIX.md) preserves the numerical tables. The [frozen snapshot]({{A}}/snapshot.json), checkpoint diagnostics, and scripts make this review reproducible even if the live dashboard changes. Source links below point to the frozen copy.

**Operational boundary:** no training file, model weight, optimizer, launch setting, or process was changed. Diagnostics used single-thread CPU execution with low process priority, no optimizer steps, and no MPS work. Such diagnostics can still use shared memory and CPU; the throughput comparison therefore uses the pre-audit interval through step 250. Checkpoint weights were not duplicated into this report folder; the trainer may eventually rotate the original checkpoint paths. Saved diagnostic results will remain, but exact model reruns will then need an archived copy of those weights.

## 3. First principles: what improvement would have to mean

The model is not one scalar loss or one recurrence mechanism. It is a chain:

`tokens → encoder local features + 4-token concepts → repeated concept core → continuous plan + predicted PQ concept → confined decoder → next-token distribution`.

The encoder supplies both local token features and concept features. The decoder has restricted local access, so the concept path must supply useful broader context. The product-quantized codebook supplies an auxiliary representational target; it is not the end task itself.

For a loop to earn its cost, three things must hold: later state must contain useful information, training must assign credit to the operations producing it, and the added quality must justify the reduced number of examples seen per unit time. “The state changes,” “expert choices change,” and “the auxiliary loss falls” each test only a piece of that argument.

The implemented objective is:

`NTP + 0.1 × mean_pass(NCP) + VQ + 0.1 × consistency + 0.01 × mean_pass(LB) + 0.001 × mean_pass(z) + NextLat_h + NextLat_KL`.

NTP is next-token cross entropy. NCP predicts the next chunk's 48 PQ code indices. VQ updates the codebook against detached encoder concepts. Consistency pulls earlier states in the final window toward the detached final readout. LB and z regularize routing. NextLat predicts the next decoder hidden state and its token distribution, using the next observed token during training.

These losses have different units, masks, targets, and gradient paths. In particular, **a VQ scalar around 3 does not mean it dominates encoder learning**: its encoder input is detached. Conversely, a smaller auxiliary scalar can still produce a substantial opposing gradient on shared parameters. The correct diagnostic is the gradient and its destination, not just its displayed size. See [the actual loss construction]({{S}}/code/train.py:100), [NextLat]({{S}}/code/model.py:65), and [PQ target handling]({{S}}/code/concept_core_d6.py:180).

## 4. Is the v1 performance gap real?

### 4.1 Match training examples before interpreting the dashboard

Both runs use seed 0, microbatch 2, accumulation 16, and sequence length 1,024: **32,768 token slots per optimizer update**. These counts include padding and should not be confused with loss-bearing tokens. Their epoch-0 data order is identical. v1 logs 50-step averages; v2 logs 25-step averages. I combined pairs of v2 records to compare equal windows. These training metrics average per-microbatch token means; the validation metrics instead pool valid token losses within each language.

| Optimizer steps | v1 NTP | v2 NTP | v2 minus v1 |
|---|---:|---:|---:|
| 1–50 | 8.3179 | 8.5857 | +0.2679 |
| 51–100 | 6.0605 | 7.0348 | +0.9743 |
| 101–150 | 4.4501 | 5.4277 | +0.9776 |
| 151–200 | 3.7811 | 4.4427 | +0.6616 |
| 201–250 | 3.4822 | 4.0017 | +0.5196 |

Lower is better. The gap is real, but **it is shrinking rather than exploding**. At the end of these windows, both models have consumed 8.192 million token slots. v1 took 1.390 hours; v2 took 1.919 hours, 38.1% longer. The last comparable interval's recorded throughput is approximately 1,688 versus 1,174 slots/second.

The learning rates at step 250 are 0.0004947 and 0.0004982, only 0.7% apart. That does not explain this early gap. Their full schedules differ, so this close early LR match should not be extrapolated to the whole run.

### 4.2 Validation confirms both a sample-efficiency and a time-efficiency gap

The official quick evaluation uses the same 128 rows per language. Reported mean is the arithmetic mean of token-weighted TinyStories and WikiText losses.

| Comparison | v1 step / training hours | v1 mean NTP | v2 step / training hours | v2 mean NTP | Gap |
|---|---|---:|---|---:|---:|
| Approximately equal updates/data | 265 / 1.505 | 4.5630 | 267 / 2.094 | 5.1363 | +0.5733 |
| Approximately equal time | 365 / 2.074 | 4.1221 | 267 / 2.094 | 5.1363 | +1.0142 |
| Approximately one hour | 185 / 1.037 | 5.0891 | 138 / 1.046 | 6.3298 | +1.2407 |

At nearly equal updates, the 0.5733-nat difference corresponds to about **1.77× the geometric mean perplexity** across the two language datasets. This is substantial. It is not “77% less intelligence,” nor does it establish the final training outcome.

The v1 final/continuation lines around 2.754/2.729 on the dashboard compare a mature model with a two-hour scratch model. Those are useful eventual targets, not fair early-stage baselines. The matched comparisons above establish the gap without relying on that misleading contrast.

### 4.3 The symbolic probes cannot yet adjudicate the architecture

At step 267, v2 recall exact match is 1/128; carry, multi-token exact match, and hop are zero; bind is 1/128. v1 also had essentially unlearned probes around this stage. Its substantial recall, carry, and multi-token gains appeared much later in the recorded history. Thus the current language gap is established, while a claim that v2 has permanently lost reasoning ability is not.

## 5. Does looping itself damage the model?

I held weights and examples fixed, disabled training capacity drops as the official evaluator does, and changed only the number of passes. Each checkpoint used the same eight sampled rows per language, drawn from the first 128 validation rows. This is 16 rows per checkpoint, not the full official evaluation. Values below pool loss over valid tokens within each language.

| Passes, checkpoint 267 | TinyStories NTP | WikiText NTP | Mean NCP across sampled microbatches |
|---:|---:|---:|---:|
| 1 | 4.4356 | 6.0955 | 3.9818 |
| 2 | 4.4170 | 6.0868 | 3.9735 |
| 4 | 4.4042 | 6.0816 | 3.9957 |
| 6 | 4.3976 | 6.0795 | 4.0107 |
| 8 | 4.3955 | 6.0789 | 4.0169 |
| 10 | 4.3947 | 6.0788 | 4.0193 |
| 12 | 4.3944 | 6.0787 | 4.0204 |
| 16 | 4.3942 | 6.0785 | 4.0217 |

**Six passes beat one on 16/16 sampled rows at checkpoint 138 and 15/16 at checkpoint 267.** Both language averages improve at both checkpoints. The mean language gain from R1 to R6 at step 267 is 0.0270 nats. The extra gain from R6 to R12 is only 0.0020 nats. The latter is far too small, on this sample, to explain away the official 0.5733-nat v1 gap as merely an evaluation-depth mismatch.

NCP tells a different story. In all 11 logged per-pass summaries, the final pass is worse than the first. The best pass is the first at steps 25 and 50, and the second in the other nine records. At step 250, NCP is 4.0914 at pass 1, 4.0776 at pass 2, 4.1436 at pass 6, and 4.1901 at pass 12. Later recurrence changes the state in a way that is less useful to this prediction head while being slightly more useful to the language decoder.

This is evidence of **different objectives preferring different representations or depths**, not evidence that every aspect of the representation deteriorates. A loop can move away from easy PQ prediction and still improve contextual token prediction. It can also settle into a mostly repetitive state whose small gains do not justify its cost. The measured curves support both a real early benefit and diminishing returns.

Other fixed-weight interventions at checkpoint 267 support this interpretation:

| Intervention, R6 validation sample | TinyStories NTP | WikiText NTP |
|---|---:|---:|
| Intact | 4.3976 | 6.0795 |
| Remove concept plan entirely | 4.5342 | 6.2451 |
| Remove continuous `u` contribution | 4.5113 | 6.2256 |
| Remove predicted PQ contribution `c_pred` | 4.3866 | 6.0649 |
| Swap plans between paired rows | 4.4165 | 6.0810 |

The continuous concept path is already useful. The predicted PQ contribution currently hurts these sampled language losses slightly. That is a reason to inspect branch calibration and gradients, not sufficient evidence to delete it permanently: the mature v1 audit found task-dependent usefulness, and turning a branch off after training is an out-of-distribution intervention.

On one unseen B2 training sample at R10, resetting carry after pass 4 raises pooled token loss by 0.0034; swapping that carry between rows raises it by 0.0137. Disabling attention raises it by 0.0206; disabling experts raises it by 0.0908. Carry and expert computation are therefore not completely ignored. These are small diagnostics, not estimates of the gain from training a different architecture.

**Verdict on your hypothesis:** “The extra loop work is not earning enough benefit yet” is supported. “Later loops destroy token-prediction ability” is contradicted by these fixed-weight measurements. “Training a shorter-loop model would do better” remains plausible and needs its own training control.

Evidence: [checkpoint 138]({{A}}/checkpoint_000138_diagnostics.json), [checkpoint 267]({{A}}/checkpoint_000267_diagnostics.json), [diagnostic implementation]({{A}}/checkpoint_diagnostics.py).

## 6. The strongest optimization suspect: NextLat gradient competition

NextLat has weight 1 for its hidden-state loss and weight 1 for its KL loss. Both backpropagate through the decoder state and therefore into parts of the concept path and encoder. Detached teacher targets and detached auxiliary embedding weights do not prevent gradients through the student state.

I reconstructed the actual K10 training graph, including capacity factor 1.25 and detachment after pass 4, on the first microbatch of each of four subsequent optimizer-step data positions. No optimizer updates were made. For selected shared weights I measured:

`cosine = dot(grad_NTP, grad_NextLat) / (norm(grad_NTP) × norm(grad_NextLat))`.

A negative value means that a small raw gradient-descent step for the auxiliary objective alone would locally increase token loss on that same batch. This is a local statement; it is not a complete theory of the eventual generalization benefit of regularization.

| Shared parameter | NextLat-vs-NTP cosine over four batches | NextLat gradient norm / NTP norm |
|---|---|---|
| Core input projection `W_in` | −0.437, −0.599, −0.387, −0.435 | 0.432, 0.420, 0.678, 0.611 |
| Expert output matrix `w2` | −0.254, −0.461, −0.422, −0.384 | 0.571, 0.433, 0.761, 0.701 |
| Decoder plan projection `Wu` | −0.314, −0.502, −0.523, −0.453 | 0.496, 0.385, 0.642, 0.656 |
| Carry normalization gain | −0.391, −0.569, −0.505, −0.468 | See raw diagnostic |

On `W_in`, the auxiliary projection along the NTP direction is negative with magnitude 18.9–26.6% of the NTP gradient's squared norm. This quantifies a meaningful local disagreement. **It does not mean AdamW's actual learning speed is reduced by exactly that percentage.** Adam's state, elementwise scaling, clipping, accumulation, changing examples, and later representation benefits are not represented by this raw-gradient calculation.

The checkpoint-138 sample also showed strong opposing NextLat directions. At checkpoint 267, the KL component is particularly conspicuous. This makes an auxiliary-loss ablation a much better first experiment than assuming every loop must be removed.

The evidence is not universal across all paths: concept-feature gradients disagree in three of four batches, while local token-feature gradients have mixed signs. These diagnostics cover selected core/decoder/router parameters and encoder-output leaves, not the entire encoder parameter gradient. NCP is not uniformly harmful either: its weighted gradient aligns with NTP on `W_in` in the inspected checkpoint-267 batch, despite disagreeing at checkpoint 138. Consistency gradients are comparatively small and mixed. There is no basis here to identify consistency alone as the dominant problem.

Evidence: [four-batch gradients]({{A}}/gradient_replication.json), [reproduction script]({{A}}/gradient_replication.py), [NextLat implementation]({{S}}/code/model.py:65).

## 7. A verified routing problem: later batch rows lose expert computation

The router flattens `(batch, chunk)` into one row sequence. Expert slots are assigned using a cumulative count in that sequence. With B2, 256 chunks per row, top-2 routing, 16 experts, and capacity factor 1.25, each expert gets 80 slots. Earlier assignments claim those slots first. Later assignments to a full expert are zeroed.

That creates deterministic priority by batch position. It also includes fully padded chunks in routing and capacity accounting. Surviving gate weights are not renormalized after dropping, so losing one expert can reduce the magnitude of the entire expert residual. See [dispatch and slot assignment]({{S}}/code/concept_core_d6.py:62).

I separated real chunks, defined as containing at least one non-PAD token, from fully padded chunks. Each row below averages 40 layer/pass dispatches at K10 on a separate B2 batch at checkpoint 267.

| Sample batch | First row: real assignments dropped | Second row: real assignments dropped | Second row: real chunks losing both experts |
|---|---:|---:|---:|
| 1 | 0.31% | 17.66% | 4.19% |
| 2 | 3.83% | 29.24% | 8.18% |
| 3 | 0.00% | 22.28% | 6.47% |
| 4 | 2.24% | 23.64% | 7.23% |

“Both experts lost” refers to chunk/layer/pass events, not to a chunk losing all computation across the whole model. Attention and residual paths still operate. These percentages also must not be read as fractions of next-token labels dropped: the token loss is still computed.

Reversing the same two input rows and restoring output order provides a second check. With capacity dropping disabled, aligned logits are exactly unchanged on CPU. With factor 1.25, the maximum aligned logit difference at checkpoint 267 is 1.96. The model's train-mode computation therefore depends on which other row precedes the example in the microbatch. That difference comes from cross-example capacity competition, not an attention document-mask leak.

The global training drop metric is a blend of real and padded positions: 12.8% at step 25, 27.9% at 50, 17.8% at 250, and 16.6% at 275. It does not directly state how much useful computation was removed. One initially alarming last-quarter statistic was dominated by padding; the real-chunk separation above is the corrected evidence.

### Why the apparently healthy balancing metrics do not settle this

The logged load entropy averages experts across layers and passes. Local overloads can disappear in that average. The LB loss also couples hard assignment fractions with mean soft probabilities. As a concrete mathematical counterexample, nearly uniform soft probabilities can consistently select the same two experts: the implemented LB is then about 1.0000005 while 84.375% of assignments overflow capacity. This is a synthetic diagnostic, not a claim that this exact distribution occurs in the run.

An entropy near `ln(16)` and LB near 1 therefore cannot certify that every layer/pass retains useful assignments. The measured drops already refute that inference. Likewise, an 8–10% change in top-1 expert between passes proves route changes, not useful multistep reasoning.

### What this does and does not explain

This is a verified implementation-policy problem worth testing. However, removing drops at fixed checkpoint weights improves pooled R6 token loss on the sampled checkpoint-267 training batch by only about **0.00046 nats**, and R12 by about 0.00047. The effect was also small at checkpoint 138. Thus I cannot attribute the full 0.573 validation gap to instantaneous capacity loss. Its possible damage to learning over many updates remains unmeasured.

A dropless or larger-capacity training control, with padding excluded from capacity accounting, would test that long-run effect. Higher capacity also costs memory/time on this machine; it is not a free fix. Evidence: [real/padding routing replication]({{A}}/routing_replication.json), [routing script]({{A}}/routing_replication.py), [capacity sweep and batch-order test]({{A}}/checkpoint_000267_diagnostics.json).

## 8. Detachment changes what the loop can learn

K is sampled uniformly from 8 through 12. Windows are `[4,4]`, `[4,5]`, `[4,6]`, `[4,7]`, or `[4,4,4]`. The remainder is merged, so “window size 4” does not mean every final window has four passes. NTP and NextLat are computed only after the final pass. Earlier windows receive NCP and routing losses, then their outgoing state is detached.

For a K10 example:

`S0 = W_s(concepts) → passes 1–4 → detach S4 → passes 5–10 → decoder → NTP`.

Consequences:

1. Final NTP cannot tell the first window how to prepare the particular carried state `S4`.
2. **`W_s`, used only to initialize the first window, never receives NTP, NextLat, or final-window consistency gradients for any K in the current 8–12 range.** This was confirmed across all four diagnostic batches: those gradients were absent, while weighted NCP gradient norms were 0.00165–0.00242 and routing auxiliary gradients were nonzero.
3. Core weights are shared across passes. They do receive final token-loss gradients through their later applications. It would be incorrect to say the core as a whole is trained only on NCP.
4. Every later pass can reread `X = W_in(concepts)`. Detachment alone does not force use of carried reasoning. A solution that largely recomputes a useful state from `X` can satisfy the graph. The reset/shuffle intervention shows some carry dependence, but its magnitude after several recovery passes is modest.

This creates a plausible mismatch: early state construction is rewarded for concept prediction, while the final window is rewarded for token prediction plus NextLat, and the NCP depth curves show that their preferred states differ. It also limits what a positive isolated experiment with learned initialization says about this exact training recipe.

There is no claim here that truncated backpropagation is inherently invalid. It saves memory and can work. The question is whether the chosen auxiliary objectives adequately teach the detached boundary state for this task. This run has not demonstrated that yet.

I explicitly tested the implementation rather than inferring a bug from the design. A small float64 K12 case with three windows, capacity drops, and accumulation 16 compared the actual per-window backward against a one-graph reference **with the same detach boundaries**. Maximum relative parameter-gradient difference was `5.93e-15`; carry-norm gradients were nonzero. This validates the gradient-bridging arithmetic, not equivalence to full untruncated backpropagation.

The earlier design concern about carrying a normalization operation entirely outside the gradient path is also not present in this implementation: [carry normalization is inside `one_pass`]({{S}}/code/concept_core_d6.py:167). The current limitation is the intentional state detach at [trainer line 140]({{S}}/code/train.py:140).

Evidence: [bridge check]({{A}}/gradient_bridge_check.json), [check implementation]({{A}}/verify_gradient_bridge.py), [initial projection gradient measurements]({{A}}/gradient_replication.json).

## 9. The architecture spends less on token processing and much more on recurrence

| Quantity | v1 | v2 | Meaning |
|---|---:|---:|---|
| Encoder/decoder hidden width | 384 | 256 | Narrower token processing |
| Encoder non-embedding parameters | 15.511M | 7.394M | About half the contextual encoder capacity |
| Decoder parameters | 7.532M | 3.514M | About half the local realization capacity |
| Encoder + decoder | 23.043M | 10.909M | **52.7% reduction** |
| Core excluding codebook | 2.010M | 7.925M | More capacity moved into concept computation |
| “Compute parameters,” repository convention | 25.053M | 18.834M | Excludes lookup tables/codebook/training-only head |
| Engram table | 50.331M | 80.000M | Larger hashed lexical memory |
| Stored inference parameters | 78.579M | 100.980M | v2 is not a 19M-total-parameter model |
| Additional training-only NextLat head | Not used | 0.658M | Extra parameters and training work |
| Mean core layer applications per training example | 2.5 | 40 | 1 layer × mean R2.5 versus 4 layers × mean K10 |

The 16× ratio is a count of layer applications, **not a 16× FLOP estimate**: widths and sparse expert execution differ. Nevertheless, repetition increases execution even when parameters are shared. Calling a model “19M compute parameters” does not bound the cost of applying those parameters many times.

A larger Engram table does not automatically replace the compositional work of a wider encoder/decoder. v2 is asking a larger recurrent concept subsystem to compensate for more than half the encoder/decoder capacity. The current language curve says that tradeoff has not paid off early in training. Only a control that separates width, expert capacity, depth, and objectives can establish which part accounts for how much of the gap.

The launch profile recorded careful optimizations, including cached gated reads, faster top-2 selection, and reduced PQ-head intermediates. Their code and test evidence were inspected; I did not rerun MPS profiling during the active job. Real training is still slower than v1. It also pays for repeated NCP heads and NextLat's extra vocabulary projection. The latest snapshot shows around 6.75 GiB driver memory and 4.7 GB system swap at step 275; elevated swap was already present before this audit. Swap is system-wide, so the report cannot assign every byte or slowdown to the model rather than concurrent desktop activity and thermal conditions.

Evidence: [parameter accounting]({{S}}/code/model.py:83), [configuration]({{S}}/runs/v2_48h/config.json), [recorded profiling decisions]({{S}}/DECISIONS.md:42), [matched history]({{A}}/snapshot.json).

## 10. Codebook and data: recovery, moving targets, and inherited defects

The first evaluation made the codebook look weak. The second materially changes that interpretation.

| Fixed quick-eval codebook diagnostic | v2 step 138 | v2 step 267 | v1 step 265 |
|---|---:|---:|---:|
| Minimum segment code usage | 33.59% | 80.47% | 81.25% |
| Mean segment code usage | 54.51% | 88.41% | 90.63% |
| Mean code perplexity | 21.96 | 61.37 | 67.92 |
| Explained variance | 0.2781 | 0.5459 | 0.5539 |
| NCP CE | 3.9945 | 4.1996 | 3.8973 |
| Code entropy | 3.0523 | 4.1099 | 4.2156 |

Code usage and reconstruction have almost caught up to early v1. **Persistent codebook collapse is not supported by the newer checkpoint.** The NCP head still lags, but rising raw CE from hour one to hour two is not enough to show worsening prediction against an unchanged problem. The encoder representation, code assignments, and entropy of those assignments are changing. At step 200, v2 also restarted 249 unused codes versus 26 in v1, making the target system less mature.

The CE-minus-entropy difference narrows from roughly 0.94 to 0.09 for v2. Treat that as a contextual diagnostic only: the evaluator's entropy counts and CE masks are not precisely the same support, so this is not a rigorously matched empirical-prior baseline. The within-checkpoint per-pass NCP comparison is cleaner for the loop question because examples and targets remain fixed.

The shared corpus is not a new v2 regression. The earlier audit found split and probe-generation limitations that still matter for interpreting eventual generalization, but both runs use the same corpus. Those limitations do not explain why v2 learns the same early examples more slowly. See the [earlier dataset/results audit]({{OLD}}/RESULTS_APPENDIX.md) and [file audit]({{OLD}}/FILE_REVIEW.md).

There are smaller objective-quality issues worth carrying into experiments:

- NCP masks by adjacent document ID, not explicitly by real target tokens. The earlier full-data audit found about 3.82% fully padded NCP target pairs. This behavior is inherited from v1, so it is not by itself a new explanation for the gap.
- NextLat is new and also masks only same-document pairs. PAD-to-PAD pairs therefore remain valid. They formed 1.63% and 17.05% of the two inspected checkpoint training samples' valid NextLat pairs. These are sample values, not a corpus estimate.
- MoE capacity accounting also includes padded chunks. Padding can compete for slots and distort drop/entropy telemetry even where it cannot affect token loss directly.

These masks should be made explicit in a controlled candidate. Changing them mid-run would change the experiment; no such modification was made here.

The remaining telemetry is consistent with a model still reorganizing its representation:

- NextLat hidden loss falls from about 0.531 at step 25 to 0.042 at step 75, then returns to about 0.138 at step 275. KL similarly falls to about 0.006 before rising to 0.227. The teacher states/distributions are themselves changing, so these curves alone do not establish failure. The measured opposing gradients provide the stronger concern.
- The raw consistency loss is about 0.023 at step 275; its 0.1 weight makes its scalar contribution about 0.0023. That is small beside NTP, although gradient size rather than scalar size ultimately matters. There is no measured basis to blame this penalty alone.
- The evaluated Engram gate mean rises from 0.763 to 0.834, while its value-projection norm rises from 3.06 to 10.49. The branch is being learned and used. These values do not demonstrate that it can replace the capacity removed from contextual token processing.
- The logged encoder mixture remains approximately `[0.13, 0.09, 0.78]`; attention temperatures remain close to their initialization, and mean PQ temperatures move from roughly 15.7 toward 15.0. No abrupt saturation or temperature blowup is evident in these two evaluations. A stable gate average is not proof that every position is healthy.
- Mean preclip gradient norm is 0.753 at step 275; the latest heartbeat is 1.017. Clipping can therefore occur, but there is no recorded nonfinite event or explosion. Auxiliary-gradient competition can slow learning even when global norms look ordinary.

The results appendix retains every training summary and complete raw evaluation records, including fields not used to support a causal claim.

## 11. What has been ruled out, and what remains unknown

| Explanation | Assessment from this review |
|---|---|
| Comparing scratch v2 only with fully trained v1 creates the whole gap | **Ruled out.** Matching updates/data and time still shows a gap. |
| Different training corpus or epoch-0 shuffle | **No evidence.** Shared data unchanged by metadata checks; same early order. |
| A large early LR mismatch | **Not supported.** LR differs only 0.7% at step 250. |
| NaNs, numerical explosion, or guard-triggered instability | **Not observed** through the frozen cutoff; losses improve and guard log is empty. |
| Broken window-backward/encoder-gradient accumulation | **Not supported by exact test.** K12/drop/accum reference agrees to numerical precision. |
| Carry norm has no gradient because it is outside the graph | **False for current code.** It is inside the pass and has nonzero gradients. |
| Extra passes always worsen token predictions | **Contradicted** by the sampled fixed-weight depth sweeps. |
| R6 evaluation alone explains the poor score | **Unlikely on measured evidence.** R6→R12 gain is only about 0.002 nats at step 267. |
| Codebook is permanently collapsed | **Contradicted by recovery** between the two evaluations. |
| Routing is healthy because average entropy and LB look good | **False inference.** Real-chunk capacity drops and row-order effects are measured. |
| NextLat causes most of the final performance gap | **Plausible but unproved.** Local gradient competition is measured; a training ablation is required. |
| Shorter loops, wider token modules, or dense experts would win | **Unknown.** Each needs a matched training control. |
| v2 will finish worse than v1 after 48 hours | **Unknown.** Only about 4.6% of the planned time is represented. |

This is one seed and an early checkpoint pair. There are no equal-step v1 weights available for a fresh head-to-head replay; the matched v1 evidence comes from saved logs. The v1/v2 comparison changes several architectural and objective choices at once. I therefore do not assign invented causal percentages to the individual suspects.

## 12. What to test next, in order

These are proposed experiments, not changes made to the active run. Keep the existing run as a baseline until there is an explicit decision to branch or replace it. A diagnostic concern is not a reason to silently interrupt an authorized training job.

### Experiment A: isolate NextLat first

Start two continuations from the **same saved checkpoint including optimizer state**: the current recipe and a variant with both NextLat weights zero. Replay identical training rows, sampled K values, RNG state, and an explicit matched LR sequence. If a wall-time-based schedule is left unconstrained, a faster variant will otherwise change its LR trajectory at the same update count. Compare validation at fixed updates/token slots and at fixed elapsed time.

An initial 100–300-update screen can establish the direction of the language learning curve. It is too short to settle the eventual long-range reasoning benefit. If the screen improves, follow with a KL-only removal or smaller auxiliary-weight arm to identify the culprit rather than assuming both components are equally bad. The current helper computes both components whenever either weight is nonzero; setting only KL weight to zero does not automatically remove its forward compute cost.

Primary outcomes: NTP on the unchanged validation rows, throughput including the auxiliary work, core/decoder gradient cosines, and plan-use ablations. Track both early improvement and later probe behavior. Do not select a winner from one noisy microbatch.

### Experiment B: fix routing comparability and measure the learning effect

Compare the current capacity policy with a dropless training control, or a practical capacity-2 control if dropless does not fit. Log real and PAD chunks separately; measure drop by layer, pass, row, and sequence position, and distinguish one-expert from two-expert losses. Keep a dedicated batch-order permutation check.

Excluding padded assignments and removing deterministic row priority are separately testable changes. Merely shuffling row order spreads the disadvantage but does not remove capacity loss. Renormalizing surviving gates changes the effective expert residual and also deserves an explicit choice rather than an unrecorded patch.

The success criterion is a better validation-vs-compute curve, not just a prettier entropy number. On this M2, greater capacity can reduce speed or increase swap; retain that cost in the comparison.

### Experiment C: test shorter recurrence with useful credit assignment

A clean first comparison is a K4/R4 model with full backpropagation across its four passes versus a K4/R4 model detaching after pass 2. This isolates the credit-assignment issue at a manageable depth. Then compare the successful shorter recipe with the current K8–12 recipe under equal data and equal time budgets. Changing K and detachment simultaneously without that intermediate control confounds two explanations.

For the existing long-loop design, consider token supervision at window boundaries, or another explicit objective on the boundary state that predicts final task usefulness. This costs decoder work and changes the objective; it must be justified by results. The present “mean NCP everywhere, NTP only at the end” scheme has not shown that it teaches useful early carry construction.

Align training terminal depth with evaluation for the main comparison, then retain a depth sweep as a separate generalization diagnostic. Do not conclude from truncating the current model to R1 that a model trained at R1 would have identical quality.

### Experiment D: separate the token-capacity sacrifice from the D6 idea

Train a reduced-width v1-style control with the same encoder, decoder, Engram size, data, and auxiliary settings as the selected v2 candidate, but a simple dense concept core. This tests how much of the deficit follows from narrowing token modules. A second control restoring token width, if it fits, tests whether moving budget back from experts improves the result.

The immediate objective is not a larger hyperparameter search. It is a small chain of comparisons that can distinguish auxiliary competition, routing loss, truncated credit, and capacity allocation. A single “remove all new features” run cannot attribute an improvement to looping alone.

## 13. Dashboard changes that would make the next decision more reliable

The current dashboard exposes useful raw information, but several interpretations are too generous or too pessimistic. No dashboard code was changed in this review.

1. Add v1 traces aligned by processed slots and by time, keeping the mature final score as a separately labeled target.
2. Compare NTP rather than total loss across recipes. v2 has additional objectives; its total is not a fair language-quality comparison.
3. Label `ncp_pass` correctly: the trainer averages it over the **last optimizer step's 16 microbatches**, not over the full 25-step reporting window.
4. Show NTP and NCP versus pass together at evaluation. A worsening NCP curve alone does not prove worsening token prediction.
5. Add real-chunk capacity drops, per-layer/pass overload, both-experts-lost, and batch-row position. Average load entropy alone hides the verified issue.
6. Treat route divergence as activity, not evidence of reasoning. Treat rising NextLat hidden loss as a signal requiring context, not automatic success or collapse.
7. Report target entropy/code usage beside NCP and distinguish changing targets from a fixed-target comparison.
8. Track gradient competition periodically on a fixed small diagnostic set, rather than ranking objectives by raw scalar magnitude.

Relevant source: [v1 reference lines]({{S}}/webapp/js/telemetry.js:485), [D6/NextLat verdicts]({{S}}/webapp/js/telemetry.js:690), [last-step NCP logging]({{S}}/code/train.py:408).

## 14. Provisional score and final assessment

This score concerns the **current demonstrated recipe**, including learning efficiency and quality of evidence. It is not a standardized model benchmark or a forecast of the completed checkpoint.

| Criterion | Weight | Score / 10 | Reason |
|---|---:|---:|---|
| Early language learning and time efficiency | 30% | 4 | Clear matched-data and matched-time deficit, but improving |
| Useful benefit from added recurrence | 20% | 4 | Positive R1→R6 effect; very small later gains relative to added work |
| Objective and routing design | 20% | 5 | Measured gradient competition, row-biased drops, and missing final credit to initialization |
| Evaluation and causal evidence | 15% | 6 | Good logs and ablations; only early one-seed results and no matched training controls |
| Implementation and reproducibility | 15% | 8 | Exact backward checks, checkpoint/RNG handling, clear source and operational records |

Weighted score: **5.1/10, rounded to 5/10**. Confidence is high in the early performance gap and routing/gradient-path findings; moderate in the proposed explanation of the learning deficit; low in any prediction about the final 48-hour result.

My assessment is that **v2 currently has an inefficient allocation of computation and competing training signals, not a demonstrated universal failure of looping**. The most informative next action is a controlled NextLat ablation, followed by a routing control and a shorter-loop credit-assignment test. The report does not support throwing away recurrence solely because NCP rises with depth. It does support demanding evidence that the additional recurrence can repay its cost.

## 15. Review record and evidence bundle

The report was self-reviewed against frozen source, raw logs, diagnostic JSON, and recomputed arithmetic. The review explicitly checked alternative explanations, padding confounds, fair logging windows, gradient-path claims, and limits of fixed-weight interventions. See [the self-review record]({{A}}/SELF_REVIEW.md).

- [Full numerical results appendix]({{A}}/RESULTS_APPENDIX.md)
- [Every-file coverage appendix]({{A}}/FILE_COVERAGE.md)
- [Frozen v1/v2 logs and comparisons]({{A}}/snapshot.json)
- [Checkpoint 138 diagnostics]({{A}}/checkpoint_000138_diagnostics.json)
- [Checkpoint 267 diagnostics]({{A}}/checkpoint_000267_diagnostics.json)
- [Four-batch objective gradients]({{A}}/gradient_replication.json)
- [Four-batch real/padding routing]({{A}}/routing_replication.json)
- [Independent backward arithmetic check]({{A}}/gradient_bridge_check.json)
- [Figure as PNG]({{A}}/evidence_summary.png) · [Figure as PDF]({{A}}/evidence_summary.pdf)
'''
(A/'REPORT.md').write_text(render(report))

# Numerical appendix keeps every training summary in the frozen run, not selected endpoints alone.
lines=['# Numerical evidence appendix','',f"Frozen logs: {snap['time']}. NTP is nats per valid token. Reported training throughput counts sequence slots, including padding.",'',
       '## All v2 training summaries','',
       '| Step | NTP | NCP | VQ | NextLat h | NextLat KL | Consistency | LB | z | Drop % | Load entropy | Route change % | Grad norm | Slots/s | Swap MB |',
       '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for t in snap['runs']['v2']['train']:
    if 'ntp' not in t:continue
    vals=[str(t['step'])]+[f"{t[k]:.4f}" for k in ['ntp','ncp','vq','nl_h','nl_kl','cons','lb','z']]+[f"{100*t['drop']:.2f}",f"{t['load_ent']:.4f}",f"{100*t['route_div']:.2f}",f"{t['gn_preclip']:.4f}",f"{t['tok_s']:.1f}",f"{t['swap_mb']:.1f}"]
    lines.append('| '+' | '.join(vals)+' |')
lines+=['','Consistency, NCP, LB, and z are raw logged scalars before their loss weights. NCP is averaged over sampled passes; the next table is only the last optimizer step of each interval. VQ is codebook-only. Grad norm is the recorded preclip mean.','',
        '## Every logged per-pass NCP vector','', '| Step | Last K | NCP by pass |','|---:|---:|---|']
for t in snap['runs']['v2']['train']:
    if 'ncp_pass' in t:lines.append(f"| {t['step']} | {t['R_last']} | {', '.join(f'{v:.4f}' for v in t['ncp_pass'])} |")
lines+=['','## Official early validation, all relevant points','', '| Run | Step | Hours | TinyStories | WikiText | Mean | Recall | Carry | Multi | Hop | Bind |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for name in ['v1_main','v2']:
    for e in snap['runs'][name]['eval']:
        if name=='v1_main' and e['step']>365:continue
        ps=e['probes']
        lines.append('| '+ ' | '.join([name,str(e['step']),f"{e['wall_h']:.4f}",f"{e['val_tinystories']['nats']:.6f}",f"{e['val_wikitext']['nats']:.6f}",f"{e['val_lang_mean']:.6f}"]+[f"{ps[k]['exact']:.6f}" for k in ['lr_recall1','lr_carry3','lr_multi4','lr_hop','lr_bind']])+' |')
lines+=['','## Fixed-checkpoint language/depth and state trajectories','',
        'Eight fixed rows per language, IDs 19, 36, 37, 38, 43, 62, 89, 123. Each language score pools token-loss sums/counts. NCP and state summaries average eight B2 batches; they have a different denominator from language NTP. Eval mode has no capacity dropping.','',
        '| Checkpoint | R | TinyStories | WikiText | Mean NCP | Mean state norm | Mean update norm | Mean cosine to R1 |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
for ck in [138,267]:
    z=json.loads((A/f'checkpoint_{ck:06d}_diagnostics.json').read_text())
    for depth in [1,2,4,6,8,10,12,16]:
        ncp=statistics.mean(b['ncp'][depth-1] for b in z['language_batches'])
        vals=[statistics.mean(b['state'][depth-1][key] for b in z['language_batches']) for key in ['norm','update_norm','cosine_to_first']]
        lines.append('| '+' | '.join([str(ck),str(depth),f"{z['language_summary']['tinystories'][str(depth)]:.6f}",f"{z['language_summary']['wikitext'][str(depth)]:.6f}",f'{ncp:.6f}']+[f'{v:.6f}' for v in vals])+' |')
lines+=['','## Raw gradient agreement: every selected shared path','',
        'K10, capacity 1.25, four first microbatches at successive optimizer-step offsets from checkpoint 267. These are raw gradients, not Adam-preconditioned updates or full encoder parameter gradients. Norm/projection ratios are relative to NTP on the same parameter.','',
        '| Batch offset | Parameter | NextLat cosine | NextLat norm ratio | NextLat NTP-direction projection ratio | All-aux cosine |', '|---:|---|---:|---:|---:|---:|']
for b in g['batches']:
    for key,v in b['aggregate']['nextlat'].items():
        lines.append(f"| {b['offset']} | {key} | {v['cosine']:.6f} | {v['norm_ratio']:.6f} | {v['ntp_projection_ratio']:.6f} | {b['aggregate']['all_aux'][key]['cosine']:.6f} |")
lines+=['','### Initial-state projection gradient presence','', '| Batch offset | NTP | NextLat h | NextLat KL | Final-window consistency | Weighted NCP norm | Weighted routing norm |','|---:|---|---|---|---|---:|---:|']
for b in g['batches']:
    v=b['initial_projection_gradients']
    lines.append(f"| {b['offset']} | {v['NTP']} | {v['NextLat_h']} | {v['NextLat_KL']} | {v['consistency_weighted']} | {v['NCP_weighted']:.8f} | {v['MoE_aux_weighted']:.8f} |")
lines+=['','`None` means no autograd path, not a rounded small value. All K in the configured range detach before final NTP.','',
        '## Real versus padding routing, every sampled batch row','', '| Batch offset | Row | Real chunks | PAD chunks | Real assignments dropped % | PAD assignments dropped % | Real chunks losing both experts % |', '|---:|---:|---:|---:|---:|---:|---:|']
for b in r['batches']:
    for row in [0,1]:
        ds=[v[row] for v in b['dispatch_records']];first=ds[0]
        pad=statistics.mean(z['padding_drop'] for z in ds) if first['padding_chunks'] else None
        lines.append(f"| {b['offset']} | {row+1} | {first['real_chunks']} | {first['padding_chunks']} | {100*statistics.mean(z['real_drop'] for z in ds):.4f} | {'N/A' if pad is None else f'{100*pad:.4f}'} | {100*statistics.mean(z['real_losing_both'] for z in ds):.4f} |")
lines+=['','A real chunk contains at least one non-PAD token. Percentages average the 4 layers × 10 passes for each fixed row. They do not imply that token targets are removed.','',
        '## Additional raw results','',
        f"The complete capacity sweeps, individual-row losses, batch reversal, carry/attention/expert interventions, individual loss gradients, and next-latent PAD-pair fractions are in [checkpoint 138]({A}/checkpoint_000138_diagnostics.json) and [checkpoint 267]({A}/checkpoint_000267_diagnostics.json). No field is discarded from those records.", '',
        f"All v1 main and continuation train/eval records, all v2 records, guards, configurations, matched windows, and data continuity checks are in [snapshot.json]({A}/snapshot.json). The [earlier results appendix]({old}/RESULTS_APPENDIX.md) covers completed v1 evaluations, probes, ablations, and dataset quality. It is retained as context rather than used as an equal-training-budget score for early v2."]
(A/'RESULTS_APPENDIX.md').write_text('\n'.join(lines)+'\n')

roles={
'ADVISOR_LOG.md':'Read implementation/profiling advice; compared claims against configuration and measurements.',
'DECISIONS.md':'Read launch, architecture, data, LR, profiling, and checkpoint-retention decisions; used as intent, not independent result proof.',
'STATUS.md':'Read operational status; checked against saved configuration and live logs.',
'code/concept_core_d6.py':'Deep source review: state update, shared experts, cumsum capacity, gate weights, carry norm, PQ head and masking; exercised by CPU diagnostics.',
'code/concept_core_v2.py':'Compared byte identity with v1; reviewed reused codebook/quantization and gradient stops.',
'code/hrm_encoder.py':'Compared byte identity with v1; traced narrow-width config, token/concept outputs, Engram, masks and gradient entry.',
'code/hrm_decoder.py':'Compared byte identity with v1; traced confined plan use, u/c interventions, tied token head, and NextLat module.',
'code/model.py':'Deep review of dimensions, forward modes, NextLat teacher/student paths, masks, parameter grouping/accounting.',
'code/train.py':'Deep review of loss weights, windows, detach/backward bridge, accumulation, LR, RNG, checkpointing, evaluation and logging.',
'code/data.py':'Diffed against v1: epoch reseed; verified identical epoch-0 order, tensor batching, token masks.',
'code/evaluate.py':'Traced token weighting, same quick-eval rows, codebook entropy/CE support and probe metrics.',
'code/final_eval.py':'Reviewed scheduled evaluation, R sweeps, ablations and loop diagnostics; final v2 results do not yet exist.',
'code/test_v2.py':'Reviewed scope and saved results; expanded backward verification to K12, real dropping, and accumulation 16.',
'code/speed.py':'Reviewed thread-local mask cache and unchanged arithmetic; launch tests provide equivalence evidence.',
'code/compat.py':'Byte-identical compatibility shim; reviewed behavior inherited from v1.',
'code/tasks.py':'Byte-identical task generator; reused earlier data/probe semantic audit.',
'code/probes.py':'Byte-identical probe generator; reused earlier split/leakage and answer-mask review.',
'webapp/server.py':'Traced run/data selection, budget/evaluation status and live field forwarding.',
'webapp/js/telemetry.js':'Deep review of loss/depth charts, v1 reference lines, thresholds, grace periods and potentially misleading verdicts.',
}
def role(p):
    if p in roles:return roles[p]
    if p.endswith('.pt'):return 'Loaded distinct step checkpoints for CPU inference/gradient diagnostics; aliases identified by inode. No optimizer update; no binary copy into report.'
    if p.startswith('code/prof') or p=='code/speed.py':return 'Read profiling workload/timing scope and launch optimization context. Not rerun on MPS while training.'
    if p.startswith('code/') and (p.endswith('.sh') or p.endswith('.md')):return 'Read launch/pause/resume/watch/guard policy and shell behavior. Did not execute operational controls.'
    if p.startswith('runs/'):
        if p.endswith('config.json'):return 'Parsed full launch configuration and parameter table; cross-checked code and checkpoint args.'
        if p.endswith('GUARDS.jsonl'):return 'Inspected: empty at cutoff; no recorded guard event.'
        if p.endswith('train_log.jsonl') or p.endswith('log.jsonl') or p.endswith('live.json'):return 'Parsed all saved records; recomputed equal-window comparisons and preserved raw snapshot.'
        if p.endswith('test_v2.txt'):return 'Read every saved test result; no failure reported; assessed coverage limitations.'
        return 'Inspected operational text, launch/monitor status, or PID metadata; no learning-quality claims inferred from empty files.'
    if p.startswith('webapp/data/'):
        return 'Parsed JSON/JSONL structure and data provenance; distinguished live data, copied design content, and historical results.'
    if p.startswith('webapp/'):
        return 'Captured/indexed source and traced presentation/data role; presentation assets do not independently validate model learning.'
    return 'Inventoried and captured; see frozen source and snapshot.'
lines=['# File and folder coverage','',
       'The scope is the active v2 implementation/run and the v1/data/design evidence needed for comparison. This ledger distinguishes semantic review from mechanical inventory. It does not claim that every vendored library or numerical weight value was individually interpreted.','',
       f"Snapshot contains {len(inv)} files: {sum(x['category']=='project' for x in inv)} project files and {sum(x['category']!='project' for x in inv)} generated/cache/vendor files. Inodes, byte sizes, modification times and text hashes are in [inventory.json]({A}/inventory.json). Text source is frozen under [source_snapshot]({S}).",'',
       '## Every project file under run_v2_19M','', '| Original relative path | Bytes | Review/use |','|---|---:|---|']
for e in inv:
    if e['category']!='project':continue
    p=e['path'];saved=S/p
    link=saved if saved.exists() else ROOT/'run_v2_19M'/p
    lines.append(f"| [{p}]({link}) | {e['bytes']:,} | {role(p)} |")
lines+=['','## Generated, cached, and vendor files','', '| Path | Category | Treatment |', '|---|---|---|']
for e in inv:
    if e['category']=='project':continue
    lines.append(f"| {e['path']} | {e['category']} | Inventoried; source/consumer used where relevant; no individual semantic audit of cache/vendor internals. |")
lines+=['','## Shared data continuity, every file','',
        'All 30 files below match the earlier full audit in inode, size and modification time. This audit reused that earlier full-data inspection and read the slices needed for fresh checkpoint diagnostics.','', '| Shared path under run_m2_25M | Inode same | Size same | mtime same |','|---|---|---|---|']
for e in snap['shared_data_unchanged_since_prior_audit']:
    lines.append(f"| {e['path']} | {e['same_inode']} | {e['same_size']} | {e['same_mtime']} |")
lines+=['','## Remaining repository context','',
        f"- [Earlier comprehensive file audit]({old}/FILE_REVIEW.md), [inventory]({old}/inventory.json), and [results]({old}/RESULTS_APPENDIX.md): v1 implementation, corpus/shards, screens, final/continuation results, model_v2 design, and concept ablations. Current v1 logs/configurations were reread into this review's snapshot; byte-identical reused modules were verified.",
        '- `run_m2_25M/model_v2`: implementation intent and prior design evidence; the new source and actual run settings take precedence when they differ.',
        '- `Concept Design`, `Encoder Design`, `Decoder Design`, `Loop Papers`, root design notes and PDFs: contextual architecture/research material covered by the earlier audit. They were not all re-extracted or treated as fresh experimental evidence in this run diagnosis. No literature result is used here to claim that a local training change will work.',
        '- `.venv`, caches, `.DS_Store`, and local assistant metadata: tooling/system material rather than model results; not a review of every dependency implementation.',
        '- The report directory contains read-only diagnostic scripts, frozen text/log copies, measurements, the figure, and written review. Scripts do not call optimizer steps or alter training files.',
        '', '## Checkpoint handling', '',
        'At cutoff there were two distinct step checkpoints (138 and 267) and last/best aliases. Each checkpoint file is about 1.22 GB because it includes training state, not just inference weights. The CPU reader used memory mapping and parameter assignment to avoid unnecessary full copies. The aliases were recognized rather than counted as four independent experiments. Future checkpoint rotation may remove the original paths; this report preserves observations and code, not immutable copies of the binary weights.']
(A/'FILE_COVERAGE.md').write_text('\n'.join(lines)+'\n')
print('Wrote REPORT.md, RESULTS_APPENDIX.md and FILE_COVERAGE.md')
