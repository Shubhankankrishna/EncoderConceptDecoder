# Independent review of model v2

**Score: 6.0/10 for the trained experimental system in `run_m2_25M`. The proposed whole-model v2 also earns approximately 6/10 as a design; its actual performance is unmeasured.**

This is a judgment of progress toward the project's stated goal: useful compressed representations, a concept route that carries distant information, and recurrent computation that improves reasoning under a MacBook M2 budget. It is not a ranking against commercial language models or a measured benchmark score.

The system has demonstrated useful long-range retrieval through its intended information path. It has not demonstrated a meaningful benefit from thinking for more passes, reliable compositional generalization, or an advantage over a properly trained, compute-matched conventional baseline. The next design addresses plausible weaknesses but does not yet establish that its additional machinery solves them.

![Final test and OOD task accuracy, and accuracy across recurrent depths](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/evidence_summary.png)

Review snapshot: **28 September 2026, 18:21 IST**. The continuation run was active. This review used existing results and small CPU diagnostics; it did not retrain the model, run another MPS job, or change the running experiment. All new files are in this separate audit directory.

## 1. Which “v2” exists?

There are two distinct objects in this folder. Keeping them separate is necessary to interpret every result correctly.

| Object | What exists | What can be scored |
|---|---|---|
| Implemented encoder → `ConceptCoreV2` → confined decoder | d=384; 25.053M compute parameters, 78.579M stored; trained screens, main checkpoint, and continuation | Actual behavior, training stability, data quality, and architectural tests |
| `model_v2/MODEL_V2_19M.md` | d=256; D6 core, shared 16-expert pool, detached training windows, 80M Engram, NextLat | Design quality and testability only |

The proposed model's first paragraph explicitly says **“design document, not implemented, not trained.”** Its implementation table calls for a new `concept_core_d6.py`; that module is absent. The 20 checkpoint paths resolve to nine distinct file inodes. Their metadata describes the existing d=384 architecture, with no expert tensors. The continuation is more training of the existing model, not a D6/v2-redesign experiment. [Sources: model specification][spec], [implemented core][core], [checkpoint inspection][checkpoints].

Internal review numbers such as D6's 70.0/100, D3's 70.5/100, and the whole-model review's 6.7/10 are assessments of proposals. They are not measured performance or independent replications.

## 2. Scope and strength of the inspection

The inventory covers **1,098 files**, including **203 project files** and dependency/cache/OS files. All 156 readable project text files were read programmatically and indexed: 68,208 lines, 13,566 numeric JSON leaves, and 1,985 JSONL records. I traced the executable model, training, data generation, evaluation, and proposed-design logic; inspected all result conditions and trajectories; and checked checkpoint shapes and metadata. I also consulted the earlier encoder/concept/decoder experiments and checked the relevant methods in four original papers.

This does **not** mean that every installed dependency was reviewed, that every checkpoint weight was numerically validated, or that all 42 papers referenced by the project's literature notes were independently reread. The exact coverage and every project path are in [FILE_REVIEW.md][files]. All recorded evaluation conditions, including weak results, are in [RESULTS_APPENDIX.md][appendix]. Machine-readable findings and the CPU checks are in [audit_results.json][audit] and [diagnostics.json][diagnostics].

## 3. First principles: what must this architecture accomplish?

For an autoregressive model, a prediction at position t may use only the prefix through t. This architecture deliberately divides that prefix information into two routes:

1. **Local token route.** The encoder's local stream reaches 63 tokens backward. The confined decoder adds seven, for a structural reach of 70 tokens. Each decoder position reads its own private plan; deeper decoder layers cannot pass other positions' plans sideways.
2. **Concept route.** The encoder mixes local and globally contextualized features, pools each four-token chunk into a continuous concept, and the core causally processes those chunk representations. Older information must reach the decoder through this route.
3. **Prediction.** The decoder combines its local token features with a continuous core output `u` and a predicted concept vector `c_pred`, using the newest completed chunk allowed by the causal indexing rule.

That creates three separate questions. Does the concept route contain useful information? Does it retrieve and combine the right information? Does repeating the core improve the answer? Success on the first does not establish the second or third.

The product-quantized vocabulary provides 48 code indices, each with 128 choices, as next-concept prediction targets. However, the actual path also carries continuous `c_hat`, continuous `u`, and soft code mixtures. Therefore, 48 × 7 = 336 bits describes the hard code-index format, **not a 336-bit limit on the model's information channel**. Healthy codes alone do not establish discrete semantic concepts, interpretable reasoning, or lossless compression.

The global encoder already performs substantial contextual computation before the core. Strong retrieval does not by itself locate all reasoning inside the recurrent module.

## 4. Data audit: mostly sound, with specific limitations

I streamed every packed row, checked token ranges and chunk-aligned document boundaries, recomputed all four recorded token hashes, and checked every scored evaluation item against the training document hashes. The tokenizer hash also matches.

| Packed split | Rows of 1,024 | Non-padding tokens | Valid next-token targets | Hash / boundaries |
|---|---:|---:|---:|---|
| Train | 273,797 | 264,365,789 | 261,227,372 | Pass |
| Validation | 12,444 | 11,390,921 | 11,271,141 | Pass |
| Test | 12,417 | 11,380,483 | 11,260,921 | Pass |
| Maintenance | 12,461 | 11,348,817 | 11,231,369 | Pass |

Training contains 280,368,128 allocated token slots. Slots, non-padding tokens, and supervised targets are different quantities; throughput and training progress use slots.

The actual mixture is approximately 44.3% language, 22.3% symbolic, 10.1% code/structured text, 13.0% retrieval, and 10.4% long-range tasks. Language is TinyStories and WikiText; FineWeb and tool trajectories were not used. This is a specialized experimental curriculum, not broad general-purpose pretraining.

The scored item sets contain 6,760 validation items, 6,760 test items, and 6,960 OOD items. Each has no internal exact duplicates, and **zero scored items match a training document verbatim**. Spans are valid and items fit the 1,024-token limit. Long-range test evidence starts at least 145–157 tokens before the first answer, depending on the family; OOD distances start at 519–531. These exceed both the confined decoder's 70-token reach and the stack control's 103-token reach.

Limits that matter:

- There is one identical item in each pairwise intersection of the scored splits: expression evaluation between validation/test, and sort traces for the other two intersections. Small, but remove them for strict split independence.
- The packed auxiliary validation/test/maintenance shards contain training duplicates. Those shards are **not** the separate item sets used for the final answer-span benchmark; this does not invalidate the clean scored-item checks.
- The manifest records about 2.79M synthetic training documents but only 2.070M unique ones. Repeated small-space examples reduce diversity even without looping over the packed dataset.
- `json_build`, `sort_trace`, and `yaml_lookup` use the same generator distribution for their nominal ID and OOD splits. Treat these as held-out controls, not evidence of structural extrapolation.
- OOD probes change names, distance, and sometimes difficulty together. A failure cannot identify which change caused it. A factorial benchmark should change one factor at a time.
- Only exact synthetic-document overlap was tested. This is not a near-duplicate audit of every natural-language source.

**A training-target defect:** NCP masks document crossings but not padding. Across the full training shard, 2,549,538 otherwise-valid NCP transitions have an entirely padding next chunk: **3.82%**. The corresponding shares in packed validation/test/maintenance are about 9.1%, 9.0%, and 9.6%. VQ also sees padded chunks. NTP correctly masks padding. Padding is therefore an unnecessary, predictable auxiliary target and can affect codebook-health interpretation. Fix it explicitly before v2; NextLat should likewise exclude padded pairs, not only cross-document pairs. [Source: core loss][coreloss].

## 5. What the trained model actually achieved

### Language and use of the plan

Main final checkpoint: step 4,050, 132.710M processed token slots, 21.95 logged hours. That is about 0.47 of the available packed training rows. It is not a fully converged scaling-law experiment.

| Condition | TinyStories test, nats/token | WikiText test, nats/token |
|---|---:|---:|
| Intact, R=4 | 1.9768 | 3.6290 |
| Core off | 2.5548 | 4.0843 |
| Plan swapped between rows | 2.8489 | 3.8513 |
| Continuous `u` off | 2.4881 | 4.0586 |
| Predicted-code vector off | 2.0008 | 3.6490 |
| R=1 | 1.9780 | 3.6295 |
| R=8 | 1.9770 | 3.6291 |

The concept path is being used. Removing or replacing it causes a large degradation, and the continuous `u` branch carries most of the language benefit. The code-prediction branch has a smaller language effect, but its removal hurts multi-token tasks substantially, so it is not useless.

Core-off and plan-swap are interventions on a model trained with a functioning plan. They establish dependence; they do **not** establish the performance gain relative to a separately trained no-core model or an equally expensive ordinary transformer. Wrong plans can be more damaging than missing context. Absolute nats/token also cannot be compared casually across tokenizers and datasets.

### Long-range behavior

These are final test results, **512 items per family**, at R=4. OOD means the repository's harder/shifted split.

| Task | Test exact match | OOD exact match | Test with swapped plans |
|---|---:|---:|---:|
| Recall one distant value | **97.07%** | **80.08%** | 0.39% |
| Emit a distant three-token value | 58.01% | 16.60% | 0% |
| Retrieve four three-token values | 8.20% | 0% | 0% |
| Follow a chain of links | 9.18% | 0% | 0% |
| Track a variable binding | 2.93% | 5.27% | 4.88% |

One-value retrieval is the clearest success: 497/512 correct in-distribution and 410/512 on the shifted split, collapsing to 2/512 with swapped plans. Distant evidence is beyond the local route, and the trained model's causal/isolation tests pass. This is substantial evidence that the intended concept route works.

The failure progression is informative. One fact works; longer values weaken; competing facts and composition are much harder. Binding is not established at all. The system is currently better at retrieving a compact association than maintaining and manipulating a rich state.

The multi-value metric requires all 12 answer tokens to be correct. Per-token offset accuracy is approximately 76–82%, so 8.2% complete-item accuracy does not mean every retrieval is random. Nevertheless, complete multi-fact reliability is the project's own objective, and its 50% exact-match gate is far from met. OOD multi-value and chain-following have zero complete successes.

### Does more thinking help?

| Test metric | R=1 | R=4 | R=8 |
|---|---:|---:|---:|
| Chain-following exact match | 8.59% | 9.18% | 9.18% |
| Three-token carry exact match | 59.57% | 58.01% | 58.01% |
| Four-value exact match | 8.59% | 8.20% | 7.81% |
| OOD chain-following | 0% | 0% | 0% |

R=4 adds only three correct chain items relative to R=1, out of 512. The saved aggregate results do not contain paired per-item outcomes needed to assess the small difference properly. The strict inequality in the original G-C3 gate is satisfied, but meaningful iterative reasoning has not been established. Language is effectively flat too.

The dynamics support this interpretation: state norm grows from 9.77 to 69.41 over eight passes; consecutive update angles fall from 6.70° to 0.11°; update magnitude stays around 8.4 late in the trajectory. Earlier passes still receive gradients—pass-1/pass-4 gradient ratio is 1.14—so vanishing gradients are not demonstrated as the explanation. The recurrence largely adds aligned updates. Angles alone cannot prove that a loop is useless; the flat accuracy/loss curves provide the stronger behavioral evidence.

### All synthetic families

Each result below is based on 200 items. These are exact-match percentages.

| Family | Test | OOD |
|---|---:|---:|
| arith_transform | 20.5 | 2.0 |
| brackets | 94.5 | 0.0 |
| copy | 48.0 | 54.0 |
| expr_eval | 37.0 | 7.0 |
| graph_walk | 8.5 | 0.0 |
| json_build | 98.0 | 99.5 |
| json_lookup | 93.5 | 63.0 |
| kv_retrieval | 21.5 | 12.0 |
| kv_updates | 69.5 | 16.0 |
| needle | 70.5 | 67.5 |
| nested_scopes | 99.0 | 0.0 |
| permutation | 47.5 | 0.0 |
| pointer_chase | 5.5 | 0.0 |
| py_exec | 60.5 | 5.0 |
| relational | 30.5 | 13.0 |
| seq_transform | 97.0 | 3.0 |
| sort_trace | 100.0 | 99.5 |
| stack_ops | 84.5 | 0.0 |
| state_machine | 41.5 | 0.0 |
| variable_binding | 16.0 | 12.5 |
| variable_binding_newnames | — | 10.0 |
| yaml_lookup | 20.0 | 22.5 |
| **Reported macro** | **55.40** | **22.11** |

The very high ID scores on nested scopes, brackets, and sequence transformations do not survive their structural shift. Seven synthetic OOD families have zero exact match. Moreover, removing the three unchanged-distribution control families lowers the OOD macro to **13.95% over the remaining 19 families**. That is a diagnostic subset, not a directly matched replacement for the 21-family ID macro.

### Representations and the codebook

The final codebook uses every code in each segment on the sampled evaluation concepts. Mean perplexity is 108.49 out of 128; reconstruction variance explained is 0.6506; NCP cross-entropy is 2.6978 versus marginal code entropy 4.6856. These support non-collapse and useful predictability, subject to the padding issue.

The four concept-to-token linear probes recover 64.76%, 62.97%, 65.77%, and 69.92%, against majority baselines around 4–5%. They contain considerable lexical information but fail the precommitted 99% criterion. Linear decoding failure does not prove information is absent: probe capacity, regularization, sample coverage, and optimization matter. The split is at sampled-concept level, so this is also not a strong document-independent representation benchmark.

At query positions, linear decoding from `u` deteriorates across the three answer tokens: carry 46.6%, 9.7%, 2.9%; multi-value 58.5%, 21.5%, 6.3%. The chain bridge probe is about 12.6% at passes 1 and 2, and 11.7% at pass 4. These suggest weak accessible multi-token and intermediate-state representations. They do not prove a hard “one-token capacity” ceiling.

The saved generation samples show learned local language patterns but also repetition, topic/name drift, and implausible factual continuations. There is no instruction-following or broad factual benchmark here.

## 6. Training trajectory, controls, and continuation

The three 265-step screens each processed 8.684M token slots:

| Screen | Validation language mean | Interpretation |
|---|---:|---|
| S1, confined decoder | 4.34077 | Reference |
| S2, stack decoder | 4.35146 | Confined is 0.01068 nats better; inside ±0.02 tie band |
| S3, fixed R=1 | 4.33578 | Slightly better than S1; also inside tie band |

This supports the confined decoder being viable at early training. It does not support recurrence improving learning. Main@265 is 4.5630, but main was still warming up while the screens were nearly annealed. A step-count match without a schedule match is not a fair architecture comparison.

In main, useful recall appears only around step 1,009, strong recall develops later, three-token carry develops around step 2,190, and nonzero multi-value accuracy appears much later. Consequently, a 265-step v2 screen is suitable for throughput, numerical stability, and gross regressions; it is too early to decisively reject an auxiliary loss for failing to produce mature long-range reasoning.

Main median logged training throughput is 1,711 tokens/s, with a range of 1,435–2,148. Continuation median is 1,908 tokens/s in this snapshot. These are training-window measurements, not end-to-end wall-clock throughput including all evaluation and checkpoint costs. Main and continuation guard logs are empty at the snapshot; this does not erase the documented startup/calibration incidents.

The continuation preserves the checkpoint, optimizer/RNG state and data cursor, with a new LR warm-up/cosine schedule. It initially worsens validation loss, then recovers. At the latest saved evaluation, **local step 1,642 / global step 5,692, 17:43 IST**, validation language mean is **2.74959**, compared with main-final **2.75419**. On the same 128-item validation probes, carry rises from 59.38% to 71.88%, multi-value from 5.47% to 15.63%, and chain-following from 8.59% to 11.72%; recall is 96.88% versus 98.44%. Binding remains 4.69%.

Those are encouraging validation improvements in some skills, not final 512-item test/OOD results. The continuation has no completed final test matrix in this snapshot. All eight continuation evaluations, including the early regression, are preserved in the appendix. An earlier 99.22% recall point is not a reason to replace the latest value selectively.

## 7. Evaluation defects and what the original gates really say

**The “beyond 70 tokens” comparison is confounded.** `Evaluator.truncation` takes a 72-token suffix and scores the last 16 next-token targets. Their predictor positions are 55…70. Fifteen of these 16 positions do not retain the full 70-token backward local receptive field. Consequently, the loss difference includes removal of local information as well as distant concept information. A four-aligned suffix of at least **88 tokens** preserves that local field for every scored position. [Source: evaluator][evaluator].

The saved differences are 0.0762 nats for TinyStories and 0.0401 for WikiText; the same-position plan-swap costs are 0.9250 and 0.1870. Do not interpret either ratio as the core's causal share of distant language context. Only one TinyStories row qualifies for the ≥512-token contextual subset, versus 234 WikiText rows.

A small random CPU diagnostic with core off and RoPE disabled found a nonzero full-context/r72 difference and exactly zero full-context/r88 difference. This verifies the structural issue; its tiny random-model numerical difference is **not** an estimate of the effect on the trained checkpoint. Re-evaluation should use r88, identical scored positions, a core-off negative control, and appropriate document-length stratification.

Other interpretation corrections:

- **Teacher forcing:** all-token exact match for a single contiguous answer has a useful property: if every greedy choice is correct with the correct prefix, autoregressive greedy decoding follows that same prefix. It is therefore wrong to dismiss all these exact-match results as inherently inflated. However, partial-token accuracy is conditional on correct earlier tokens; separated queries are supplied their intervening contexts; and stopping/formatting/free-running errors are outside the benchmark.
- **Trained structural tests:** main P5 passes causality, reach, and packed-document isolation. Screen P5 files fail the equality assertion because observed influence is shorter than the theoretical maximum: 69 rather than 70, or 99 rather than 103. That is not evidence of leakage beyond the limit. Those runs abort before the document-isolation assertion, so they must not be labeled all-pass. Test “no influence beyond bound” separately from “observable influence at the farthest allowed position.”
- **Byte denominator:** individually decoding byte-level BPE IDs can introduce UTF-8 replacement characters. The stored byte lengths differ from encoded byte-symbol counts for 141/8,192 IDs. On these test sets the aggregate discrepancy is tiny—about 0.0018% for TinyStories and 0.090% for WikiText—and does not change the verdict.
- **Uncertainty:** this is primarily one training seed. Aggregate scores lack paired predictions and do not provide seed-level confidence. The 128-item validation probes are especially coarse.

| Original gate | Independent assessment |
|---|---|
| G-E0 / G-D0, exact structural tests | Main trained tests pass; screen test aborts need accurate reporting; MPS compatibility fix is documented |
| G-E1, ≥99% token reconstruction | **Fail**, 63–70% |
| G-E2, PQ variance explained ≥0.6 expected | **Pass**, 0.6506 |
| G-C1, ≥95% codes used per segment | **Pass** on the final evaluation sample |
| G-C2, NCP below marginal code entropy | **Pass** |
| G-E3 / G-D2, plan use / language distant-context attribution | Strong retrieval evidence; binding not learned; language attribution **not cleanly established** |
| G-C3, R4 > R1 on chain-following | Literal inequality passes by 3/512; substantive iteration benefit **not established** |
| G-D3, multi-value exact ≥50% | **Fail**, 8.20% test |
| Confined/stack screen within ±0.02 nats | **Pass** at this early, single-seed budget |

## 8. Proposed whole-model v2: what is well motivated?

The proposed design reduces token-rate width from 384 to 256, keeps the four-token concept rate and 384-dimensional code space, allocates more compute parameters to the core, and uses four distinct recurrent layers. A separate initialization projection, zero-initialized residual outputs, and normalized carry are reasonable ways to improve optimization and state behavior. Preserving the confined decoder makes future causal tests interpretable. Sharing a sparse expert pool can add learned capacity without evaluating every expert on every token. Explicit dropped-assignment telemetry and planned dense/consistency/NextLat controls are good experimental choices.

The budget enumeration is useful: I reproduced all six published parameter tables. Row C allocates 39.26% of compute parameters to the encoder, 42.08% to the core, and 18.66% to the decoder. The proposal correctly acknowledges that a smaller parameter count need not train faster with many more recurrent applications.

The earlier experiments should constrain expectations, however. The d=64 isolated concept test can improve at deeper inference with perfect concept inputs, including when trained with R≤4. The learned-initial-state review is promising across two seeds. But those benefits do not reliably transfer to the learned-encoder end-to-end experiments. The old decoder tests found confinement competitive and NextLat neutral to slightly worse at that scale. These support targeted ablations, not a conclusion that changing all components together must help.

## 9. Proposed v2: the six most important unresolved problems

### 9.1 The objective still permits a useless recurrence

Let one pass be `S_next = F(X, S)`. The clean encoder input X is available every pass. A valid solution is `F(X, S) = g(X)`: ignore the incoming state and recompute the same sufficient answer from X. Every pass then receives the same next-concept prediction quality. If outputs are identical across passes, final-window consistency is also exactly satisfied.

Detaching state between windows changes gradient flow; it does not eliminate this forward-computation solution. Expert routing can stay constant, or change without improving reasoning. Therefore the specification's claim that windows force useful improvement or give carry a necessary job is too strong. This is the largest conceptual deduction from the design.

Test it causally: at a window boundary, reset, shuffle, or perturb the carry while holding X fixed; measure paired task loss and accuracy. Compare to an independently trained one-pass/deeper dense control. Routing diversity, gate mass, representation similarity, and update angles are supporting telemetry, not proof of productive reasoning.

### 9.2 Decoder training and default inference depths do not match

Training samples K∈{8,9,10,11,12}; NTP and NextLat use only the final output `u_K`. Default inference uses `u_6`. Pass 6 receives NCP supervision and may receive indirect gradients through later states, but the decoder is never directly trained to consume pass-6 plans for its token objective.

Endpoint consistency may reduce this mismatch, but does not guarantee it. Include actual pass-6 decoding during training or explicitly test a task-aligned pass-6 distillation objective. “Pass 6 is supervised” is insufficient unless one specifies **which objective**.

There is also a concrete schedule error in `DESIGN_D6.md`: it says R6 lies in the final consistency window only for K=8,9. With the specified windows `[4,4]`, `[4,5]`, `[4,6]`, `[4,7]`, `[4,4,4]`, R6 is in the final window for **K=8,9,10,11**, and not K=12. [Sources: schedule/pseudocode][schedule], [D6 note][d6note].

### 9.3 Carry normalization and gradients need an executable definition

The structure says carry is bounded by its own learned normalization. The pseudocode applies `S = carry(S).detach()` only after a window backward. If that is the only call, the carry norm's learned gain gets no gradient, and normalization is not applied after every pass. A small CPU autograd diagnostic confirms the missing-gradient behavior of that literal boundary-only construction.

This is an ambiguity/defect in proposed pseudocode, not a bug already observed in trained D6 weights. Make per-pass versus per-window normalization explicit, place the learned transform on a differentiable path, and assert gradients and state norms. The accumulated-leaf encoder-gradient bridge should also be compared against an equivalent retained-graph reference on a tiny example before training.

The loss-routing table additionally marks LB/z as not reaching the encoder, although routers read states derived from X and S. Without explicit detachment those auxiliary losses can reach upstream encoder features. Document the intended graph and loss averaging across layers, passes, windows, and micro-batches.

### 9.4 Hardware estimates omit costs, and the memory baseline is misquoted

| Quantity | Existing model | Proposed row C |
|---|---:|---:|
| Compute parameters, project convention | 25.053M | 18.834M |
| Stored inference parameters | 78.579M | **100.980M** |
| Additional training-only NextLat parameters | 0 | 0.658M |
| Reported training MAC-equivalents / token | 84.069M | 83.535M |
| Row-C estimate including specified padded expert slots | — | **89.433M** |

The “19M” label excludes the 80M Engram table, embedding, and codebook. It is legitimate if consistently qualified as compute parameters. It does not mean a 19M-weight storage footprint: total stored parameters increase by about **28.5%**.

With B=2, M=256 chunks, and top-2 routing, there are 1,024 expert assignments per layer-pass. Capacity factor 1.25 gives 16×80=1,280 padded slots. The specified batched dense matmuls over those slots perform 25% more expert arithmetic than counting just 1,024 assignments. On the published accounting, this adds about 5.90M training MAC-equivalents/token, moving the ratio from **0.994× to approximately 1.064× the old model**, before dispatch overhead. A different ragged implementation could change that estimate; it must be measured.

The spec also labels **8.38 GiB** as today's B=2 allocated memory. D-019 identifies it as the failed B=4 allocation attempt. The successful B=2 calibration reported **4.57 GiB allocated / 8.61 GiB driver memory**. Different memory metrics are not interchangeable. Narrower token activations alone cannot establish that up to seven passes × four core layers, extra table optimizer state, and KL logits fit within today's footprint. [Sources: memory calibration][calibration], [budget recount][audit].

Use sustained on-device measurements for token throughput, peak allocation, driver memory, swap, and wall time. MAC ratios are not rigorous throughput upper bounds, particularly with small sequential MPS kernels. Capacity dispatch must also preserve causality/document independence when assignments compete for slots; add permutation and overflow tests before accepting the old structural guarantees.

### 9.5 Allocation changes exceed the evidence supporting them

Shrinking encoder/decoder width, enlarging Engram, expanding the core, introducing MoE, changing recurrence training, and adding NextLat simultaneously make attribution difficult.

The encoder is not established to be overprovisioned: its token-reconstruction gate fails, and prior fact-concept probes show weaker value than key identity. A local n-gram lookup table cannot automatically replace the contextual computation lost by narrowing the encoder. Nonzero Engram gates and weight norms establish use, not the marginal benefit of another 30M lookup parameters. Measure bucket hits, collision behavior, and a controlled size ablation.

Keeping `c_hat` at 384 dimensions preserves its shape, not its quality. Its encoder has changed, and core output `u` shrinks to 256. There is no theorem here that a fixed code format preserves recoverable information after upstream capacity changes.

Generic dense-model tokens-per-parameter heuristics do not establish optimal allocation for a system with sparse lookup tables, tied recurrence, mixed synthetic tasks, and short training. Treat budget rows as hypotheses.

### 9.6 Auxiliary losses can succeed without the intended behavior

NextLat targets consistency across token positions; D6 consistency targets agreement across core depth. They are mathematically distinct and can both be useful. But low prediction/consistency loss does not automatically imply the desired world model or reasoning algorithm. The theoretical sufficient-state argument assumes the required prediction and transition conditions hold; finite-weight auxiliary optimization does not ensure that.

The local historical NextLat result is neutral to slightly negative, so retain a properly matched off control and test future information/long-range behavior after enough learning. Also ensure λ_KL=0 actually skips its expensive computation; merely multiplying an already-computed loss by zero does not realize the promised savings.

Finally, “10% long-range data” does not mean 10% direct supervision of long-range answers. Much of each example is filler. Per-pass NCP over all chunks can be dominated by easy local/filler prediction. Measure answer-target density and learning by family before interpreting low auxiliary loss as improved reasoning.

## 10. How much do the papers support this exact design?

The references motivate experiments, but none measures this complete combination at this scale on this hardware.

- **Sparse Layers:** supports exploring sparse capacity in looped models. The study's expert placement/sharing, scale, and configurations differ from one 16-expert pool shared across four chunk-rate layers. Its arithmetic comparisons are not end-to-end M2 throughput measurements. [Original paper](https://arxiv.org/abs/2605.09165).
- **LoopFormer:** uses trajectory/time conditioning and explicit shortcut training. D6 borrows endpoint consistency but changes conditioning, training routes, and detachment. It is an adaptation, not a reproduction of the demonstrated recipe. [Original paper](https://arxiv.org/abs/2602.11451).
- **Diffusion curriculum:** the local paper uses normalized persistent state with freshly corrupted inputs across steps. Retaining useful memory is more necessary there than when D6 rereads clean X every pass. Truncated backpropagation alone does not reproduce that incentive. Local source: `Loop Papers/Diffusion Timestep Reasoning Google.pdf`.
- **NextLat:** offers a principled auxiliary target under explicit theoretical assumptions; it does not guarantee that this confined, undertrained model acquires a sufficient belief state. [Original paper](https://arxiv.org/abs/2511.05963).
- **Engram:** motivates separating local pattern memory from dynamic computation, with allocation as an empirical tradeoff. It does not imply that 80M is better than 50M in this particular token/data budget. [Original paper](https://arxiv.org/abs/2601.07372).

## 11. Scores and what would change them

These categories and weights are my evaluative rubric, not a statistical measurement.

| Trained experimental system | Weight | Score / 10 | Reason |
|---|---:|---:|---|
| Architecture and causal separation | 20% | 8.0 | Clear information paths; main structural tests pass |
| Representation learning | 15% | 6.5 | Healthy/predictive codes, substantial token identity; reconstruction target missed |
| Useful long-range behavior | 15% | 6.0 | Strong single-fact recall; multi-token/multi-fact reliability limited |
| Iteration and systematic generalization | 20% | 2.5 | Flat R curves; severe OOD composition failures |
| Training engineering and practicality | 15% | 7.0 | Resumable training and useful M2 throughput; efficiency advantage unproven |
| Validation strength | 15% | 6.5 | Interventions and clean scored items; one seed, weak controls, evaluation defects |
| **Weighted score** | **100%** | **6.0** | Useful research prototype with its central reasoning claim unresolved |

For the **proposed whole-model v2**, my separate design rubric is: causal architecture 8.0 (20%), iterative objective 4.5 (20%), capacity allocation 7.0 (15%), hardware practicality 5.0 (15%), supporting experimental evidence 5.5 (15%), implementation/spec coherence 5.5 (15%). Weighted result **5.95 ≈ 6.0/10**. Its empirical capability score remains **unrated until implemented and evaluated**.

Why not 3/10? The implementation learns, the codebook does not collapse, the information-path tests are meaningful, and distant retrieval has a strong causal intervention result. Why not 8/10? The supposed thinking loop has not earned its computation, the demanding generalization tests fail, and the proposed fixes still admit the behavior they aim to remove.

An evidence-based path to a higher score:

1. **Repair measurement and specification first.** Fix the truncation comparison, padding masks, split/control labels, failed-screen reporting, carry graph, R6 supervision, and budget accounting. Freeze the corrected benchmark and save per-item predictions.
2. **Establish a minimal comparison.** Train a matched ordinary/full-attention or dense-core baseline, and a small recurrence change, with the same data, schedule, seed, and budget. Report both equal-token and equal-wall-time comparisons. A larger one-pass/deeper dense core is essential to distinguish useful recurrence from simply adding capacity.
3. **Make recurrence causal and measurable.** Show a repeatable improvement with R, particularly on harder chain depths, and a loss of that benefit when intermediate carry is reset/shuffled. Include multiple seeds and paired-item uncertainty. Routing diversity is not an acceptance criterion by itself.
4. **Add expensive components one at a time.** Dense D6-like core versus MoE; consistency off/on; NextLat off/on; Engram size; encoder width. Select on validation, then evaluate the chosen configuration once on held-out test/OOD.
5. **Give slow skills enough training.** Use early screens for speed and stability. Use the existing learning curves to set later reasoning checks; do not reject a mechanism at step 265 solely because skills that previously emerged after thousands of steps are still absent.

The next decisive result would be a reproducible accuracy gain from additional passes, under matched compute, that survives controlled OOD changes and disappears when the intermediate state is disrupted. That would support the actual purpose of the redesign far more than another favorable architecture review or an attractive parameter table.

[spec]: /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/model_v2/MODEL_V2_19M.md:3
[core]: /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/concept_core_v2.py:70
[coreloss]: /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/concept_core_v2.py:101
[schedule]: /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/model_v2/MODEL_V2_19M.md:178
[d6note]: /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/concept_v2/DESIGN_D6.md:80
[evaluator]: /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/code/evaluate.py:75
[calibration]: /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/DECISIONS.md:117
[files]: /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/FILE_REVIEW.md
[appendix]: /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/RESULTS_APPENDIX.md
[audit]: /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/audit_results.json
[diagnostics]: /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/diagnostics.json
[checkpoints]: /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/audit_v2_2026-09-28/checkpoints.json
