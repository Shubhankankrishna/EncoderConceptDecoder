# Review and audit of the V9 architecture

**Review date:** 7 October 2026  
**Reviewed document:** [draft_v9.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md)  
**Reviewed SHA-256:** a0b696e46ba94debd248de5d15b06c950e238221fc690b2a94e589f2546f2613  
**Scope:** architecture, objectives, state transitions, search/RL, continual learning, experimental identification, statistics and compute planning.

## 1. Verdict

V9 has a worthwhile research direction: retain the encoder–concept–decoder foundation, give the concept interface several selectable hypotheses, learn against a frozen teacher, and investigate whether accurate latent transitions can make search cheaper. Keeping continual learning as a separate study is also sensible.

**The present draft needs revision before the complete experiment is implemented.** Several central statements are stronger than the specified mechanisms support, and some contracts contradict the referenced V8 implementation. The most consequential issues are:

1. K = 1 does not reproduce the V8 refit as specified.
2. Best-of-m decoder-plan selection still uses the true future target.
3. A discrete hypothesis index does not establish a legal, distinct or decoder-consistent graph action.
4. The transition gate does not validate the states and branches MCTS will actually visit.
5. The stated GradCuit update has no learning signal on failures if it uses the draft's 0/1 verifier outcome.
6. Re-grounding, termination and off-path value targets need executable definitions.
7. S3 uses sealed tests for selection, and its adapting encoder can invalidate the frozen concept interface.
8. S6 is declared independent of S2 but requires objects that S1 does not contain.
9. The explanation of V8's continual-learning failure exceeds the measured evidence.
10. The proposed runtime and scaling figures do not yet follow from a complete execution budget.

These are repairable issues. They do not establish that NCP, recurrence, concept search or LSL cannot work. They establish that the current specification could produce an expensive result whose cause remains ambiguous.

The overall score, with its rubric, is at the end.

## 2. Audit method and evidence boundaries

I read the complete draft, including its self-review and appendices. I traced its principal contracts into:

- [V8 model](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/model.py), especially Block, Foundation, Core, DenseCore, System.plans and adapter placement.
- [V8 training](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/train.py), especially configure, objective, trainable parameters and the distillation schedule.
- [V8 runtime](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/runtime.py), especially loss normalization and NCP masks.
- [V8 data and verifier](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/data.py), including graph generation, task instructions, response chunks and split exclusion.
- [V8 evaluation](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/evaluate.py), the final report, training summaries, refit/continual decisions and sealed results.
- [V6.1 search design](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v6_1.md), particularly immutable candidate records, regeneration and answer-tree accounting.
- The V5 NCP and alpha-zero predicted-history evaluation JSON files.

I checked relevant methods, equations, ablations and limitations in the nine cited local papers listed in section 11. I visually inspected the crucial NCP objective and GradCuit reward equations, and checked the LSL appendix text. This was targeted source verification, not a claim to have re-derived every paper or re-audited every historical V1–V7 result.

I did not train a model, launch repository tests, edit the draft, or alter saved experiment results. Calculations below are algebra and arithmetic on the specification or saved metadata. The contract tests proposed later are **not executed V9 tests**.

Severity means:

- **P1:** resolve before relying on the affected experiment or conclusion.
- **P2:** resolve to make the interpretation, comparisons or planning defensible.

## 3. The first-principles chain V9 actually needs

A planning architecture needs more than a prediction with several possible outputs.

| Necessary link | Required property | What the draft currently establishes |
|---|---|---|
| Tokens → concept state | The representation retains information needed for future behavior | A plausible learned representation; sufficiency for planning is unproven |
| Code → plan | Different useful codes cause different valid outcomes | A selectable index and an authority diagnostic |
| Plan → predicted successor | The imagined successor matches what executing the plan produces | MSE to demonstration concepts and a nearest-candidate proxy |
| Predicted successor → further decisions | Errors remain controlled on generated, searched branches | Short rollouts, largely trained with hindsight choices |
| State → value/stop | Heads predict actual outcomes on the deployed state distribution | Proposed AUROC thresholds, with incomplete labels and distribution coverage |
| Search → verified output | Imagined decisions survive decoding and re-grounding | An outline without full cache/tree reconciliation semantics |
| Continued training → retained interface | The state/action meanings remain stable after adaptation | Some authority and loss checks, but not all transition checks |
| Continual updates → retained tasks | Useful new changes are applied while harmful old-task changes are bounded | Better telemetry and controls, with important remaining confounds |

The central experiment should therefore test **execution consistency**:

~~~text
For an actual prefix p and a selected code k:

(c_hat, A_hat) = proposed plan(p, k)
y             = generate one chunk using that exact plan
c_real        = pool(encode(p followed by y))

Does c_hat predict c_real?
Does y represent a valid action?
Do later decisions computed from c_hat agree with decisions from c_real?
~~~

No equality between code identity, latent proximity and semantic action should be assumed before this measurement.

## 4. What the draft gets right

### 4.1 Preserve the useful foundation evidence

The matched V8 foundations all saw 24,000 updates and 576,000 GraphNav examples. Their saved sealed results support continuing the hierarchy investigation:

| Foundation | Parameters | Standard success | Hard10 success |
|---|---:|---:|---:|
| Delayed NCP | 2,448,640 | 80.4% | 31.4% |
| Alpha-zero hierarchy | 2,448,640 | 76.5% | 30.3% |
| Ordinary Transformer | 2,450,048 | 73.7% | 28.7% |

The ordinary Transformer had lower text NLL. These results support a task-dependent trade-off in one seed; they do not establish general superiority or scaling.

Source: [completed V8 report](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/REPORT.md).

### 4.2 Separate authority from search benefit

A latent that changes the decoder is a necessary ingredient for control. It is not evidence that optimizing that latent beats token retries. V9 explicitly recognizes this distinction and includes stronger token baselines.

### 4.3 Treat predicted-history results carefully

The quoted V5 results are correct: predicted-history graph NLL increased by about 0.048 for NCP and 0.891 for alpha-zero. The draft correctly acknowledges that the decoder still received true tokens. This is useful evidence about interface robustness, not a demonstrated decoder-free world model.

Sources: [NCP evaluation](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/runs/A_ncp_s0/eval_a_val.json), [alpha-zero evaluation](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v5_tiny/runs/A_ncp_a0_s0/eval_a_val.json).

### 4.4 Improve experimental discipline

Matched foundation exposure, additional seeds, paired outcomes, explicit missing-work reporting, trained controls, old-task gate telemetry and a variable re-grounding interval are all improvements worth keeping.

### 4.5 Keep text-search claims separate

Graph response chunks have a designed four-token structure. Arbitrary text chunks do not acquire equivalent reasoning-step semantics merely because their length is four. Deferring text MCTS is justified.

## 5. Findings requiring correction

### F01 — P1: K = 1 equivalence is false under the listed parameter and input contracts

**Draft locations:** [§6.1](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:337), [§6.3](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:396), [§6.7](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:465).

In V8 REFIT:

- configure freezes the foundation, then enables the replacement core.
- DenseCore includes a **trainable copy of W_A**, along with its blocks and norm.
- The foundation PQ prediction heads remain frozen.
- The recurrent core has trainable B_u and B_A outputs.
- The core receives encoder-pooled concepts, without V9's closed-loop substitutions.

In V9:

- W_A is explicitly frozen.
- The PQ prediction heads are explicitly trainable.
- Closed-loop substitutions are enabled even for the mandatory K = 1 arm.
- The recurrent variant adds a hypothesis block after the recurrent core.

Consequently, reducing the displayed loss to token + NCP + distillation does not produce identical gradients or AdamW updates. Even a dense K = 1 first-step loss match would not repair the different trainable parameter set. The recurrent architecture has an additional discrepancy.

Source: [configure](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/train.py:10), [DenseCore](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/model.py:276), [Core outputs](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/code/model.py:244).

**Required correction:** distinguish two controls:

1. **V8 reproduction control:** original parameter ownership, input path, architecture, loss and optimizer.
2. **V9 K = 1 control:** the V9 training protocol with one hypothesis.

Compare V9 K = 8 with the second control to study multiplicity. Compare the second with the first to measure the effect of the protocol changes. Alternatively, change V9's actual parameter contract to preserve V8 exactly; then separately control closed-loop training and any added block.

Specify the query/KV normalization, residual path and positions of B_H. Its code-free path must reproduce the original Block operation, not merely have the same tensor dimensions.

### F02 — P1: Best-of-m remains hindsight selection, and its selector is not a calibrated action policy by construction

**Draft location:** [§6.3](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:375).

The subset S is sampled from the prior, but the selected decoder code is:

~~~text
k_dec = argmin over k in S of distance(prediction_k, true_next_concept)
~~~

For m > 1, the final selection depends on the future target. Therefore the statement that decoder plans do not come from hindsight is incorrect. This can be a legitimate exploratory training loss, but it does not match single-sample deployment. Sampling candidates from the prior does not remove target-dependent selection.

For two equally likely future modes, a model can achieve a much lower best-of-two reconstruction loss than its single-sample generation loss without improving its prior at all.

There are further distinctions:

- Enumerating K learned hypotheses and minimizing reconstruction is related to multiple-choice learning. It is not identical to drawing K independent samples from the deployed selector.
- q = softmax(-distance/temperature) does not contain the prior. It is a reconstruction-based responsibility target, not automatically a Bayesian posterior.
- Winning a distance contest against the recorded demonstration is different from choosing an action that reaches the goal.
- An NLL measured under an oracle-selected code is not the model's ordinary deployment NLL.

**Required correction:**

- Name the objective “prior-subset, hindsight-selected reconstruction.”
- Make m = 1 a primary deployment-aligned comparison; retain m = 2 as an explicit exploratory alternative.
- Report oracle, single-prior-sample, selector-argmax and marginal/mixture likelihood measurements separately, specifying which are tractable.
- Distinguish selector calibration against demonstration assignments from calibration against executed action success.
- Add numerical definitions for K = 1, m capped at K, softmin temperature floors and sampling when probabilities are nearly zero.

Do not claim that un-detaching a rollout alone resolves the oracle/prior mismatch. It changes gradient flow, not the distribution of actions selected during training.

### F03 — P1: The code is discrete; its predicted outcome is still a continuous hypothesis

**Draft locations:** [thesis](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:18), [requirements](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:189), [same-tree claim](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:588).

Selecting k chooses one deterministic branch of a conditional model. It does not guarantee:

- that the branch corresponds to one valid hop;
- that its PQ expectation lies near a realizable encoded chunk;
- that different codes yield different actions;
- that all useful colours are covered;
- or that a code has a stable colour interpretation across contexts.

The draft retains a soft expectation within every hypothesis. WTA can improve mode coverage, but each winning region can still contain several modes and produce a compromise. The continuous A port can also steer the decoder differently even when two predicted concepts are close.

A K = 8 latent-action tree and a three-colour graph tree consequently need not be the same tree. Several codes may decode to one colour, some may produce malformed hops, and some colours may be missing.

**Required correction:** call these “discrete hypothesis choices with continuous plans” until semantic commitment is measured. Add per-state diagnostics for:

- distinct legal decoded actions;
- useful-action coverage and duplicate rate;
- predicted-versus-executed destination agreement;
- consistency of both components of the plan, c_hat and A_hat;
- code effects at later hops as well as at the first hop.

If exact tree equivalence is desired, define a tested mapping from code actions to colour/STOP actions. If no such mapping is imposed, describe the comparison as a comparison between two action parameterizations.

### F04 — P1: Existing authority and rollout gates can pass without establishing a usable latent transition model

**Draft location:** [§6.6](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:430).

Three weaknesses matter.

**First, colour changes need not be useful control.** A swap that changes the first colour 30% of the time may simply damage the answer. Hindsight NLL improvement can also exploit future information. Measure valid, distinct alternatives and actual generation success alongside sensitivity.

**Second, nearest-candidate projection is not execution.** Finding the nearest of 73 encoded candidates only gives a geometric interpretation. The actual frozen decoder may produce a different hop under the same c_hat/A_hat plan. Even a far-off prediction has a nearest candidate. Specify distance/margin diagnostics and directly compare the projected result with the decoded result.

**Third, the tested distribution is narrower than search.** Selector-argmax rollouts do not cover low-prior actions that PUCT explores. Hindsight-fed training does not cover the wrong branches that search must recognize and abandon. Teacher-forced decoder tokens can conceal errors in imagined histories.

At N > 1, “actual context” for candidate encoding must also be defined: a gold prefix, a reconstructed projected prefix, or a prefix generated by the decoder are different tests.

**Required correction:** test branch-conditioned transitions under selector, uniform-code and search-selected actions. Evaluate both imagined and executed paths, on independent prompts, at the horizons used by search. Include invalid prefixes, cycles, STOP and branch recovery.

A 95% one-step legality threshold is not automatically adequate for a 20–40-step search. Under a simple illustrative independent-error model, 0.95^20 is 35.8% and 0.95^40 is 12.9%. This is not a measured V9 failure rate; it shows why the horizon must be part of the requirement.

### F05 — P1: The written GradCuit refinement can be an exact no-op

**Draft locations:** [§8.2 outcome](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:549), [§8.3 update](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:571), [binary answer loop](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:634).

The draft defines verifier outcomes as o in {0,1}, invokes GradCuit only on failed complete answers and multiplies the log-probability gradient by R(x). It never defines a different reward mapping.

If R equals that outcome, every refined answer has R = 0:

~~~text
delta_next = project(delta + eta * 0 * gradient) = delta
~~~

The source paper avoids this: Appendix A.1, PDF page 14, uses **0 for accepted answers and -1 for rejected answers**, then stops when accepted. With the stated ascent convention, a rejected answer's log-probability is pushed down. This was checked against the rendered equation.

Source: [GradCuit paper](https://arxiv.org/abs/2608.02585).

**Required correction:** explicitly separate verifier outcome o from optimization reward, for example R_grad = o - 1 with stopping on o = 1. A centered advantage is another option, but it must be defined and evaluated.

Require a failed-answer fixture with a nonzero gradient and a correctly signed step, followed by regeneration. Bound checks alone are insufficient. If the derivative is naturally zero because the decoder ignores the code, report that as a measured authority failure.

Changing from inserted-prefix latents to code embeddings, imposing a norm bound and changing the optimizer are adaptations. Name them as such rather than calling the implementation a faithful reproduction of the paper.

### F06 — P1: Re-grounding needs a state and cache reconciliation algorithm

**Draft location:** [§9.2](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:597).

Replacing a rolled concept with an encoded concept changes all later causal hidden states that attended to it. Existing descendant KV caches, selector priors, value estimates and potentially tree statistics then describe a different state.

“Update the caches” leaves essential behavior unspecified:

- Which suffix is recomputed?
- Are previous plans retained as exact c_hat/A_hat pairs, or recomputed from the corrected state?
- Are descendant visits discarded, replayed or associated with the old state version?
- Are encoder, decoder and recurrent visit caches branch-local?
- Can two siblings accidentally mutate shared cache tensors?
- Where is the first affected decoder position when a plan is swapped?

The final point matters because plan j first affects logit position 4j+3. An intervention at the last prompt chunk can require recomputing the last prompt position used to predict the first response token.

**Required correction:** store immutable action/plan records and versioned state identities. Re-ground from a known grounded prefix; regenerate and re-encode sequentially; rebuild every affected suffix cache. Invalidate or explicitly replay affected descendants. Charge this work.

Compare cached execution with full replay on branches, sibling orderings and re-grounding boundaries. For recurrent cores, keep separate caches for every repeated visit.

There is also a scope mismatch: the owner's proposal says “search, decode the chosen step, then search the next step.” The R = infinity variant decodes a finally selected path. Add an explicit **receding-horizon** arm: search H imagined steps, execute one real chunk, then replan. It can remove decoding within lookahead without requiring an entire imagined solution to remain accurate.

### F07 — P1: STOP actions, terminal labels and value supervision are incomplete

**Draft locations:** [§8.2](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:545), [§9.2](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:599), [token competitor](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:660).

The token competitor has only three colour actions. A verified answer also requires DONE. A separate terminal classifier is not an interchangeable substitute for a selectable STOP action unless their relationship is specified.

The terminal target “the true next chunk is DONE” is undefined for an arbitrary off-demonstration search node. GraphNav accepts multiple legal solutions; an alternative legal path does not have a unique recorded next chunk. Using the original demonstration's next token would mislabel it.

Similarly, “the outcome of the trajectory played through s” labels nodes on that trajectory. It does not automatically label every expanded node. A search-selected trajectory is not an unbiased sample of the unmodified selector policy; the value head must identify the continuation policy it predicts.

Iteration 0 creates grounded states, while later decoder-free iterations create imagined states. A head validated on grounded complete answers may be miscalibrated on those states.

**Required correction:**

- Include STOP explicitly in both action contracts, or define and symmetrically implement an equivalent termination mechanism.
- Label grounded STOP legality from the actual generated prefix and public task rules.
- Do not fill an imagined node with a hidden reference continuation.
- Mark uncompleted/unverified trajectories as unlabelled, not failed.
- Train value targets for a named continuation policy and version it across expert iterations.
- Validate on imagined and searched states at each depth, including false-positive STOP rate, Brier/log loss and reliability, not only AUROC.
- Always decode and verify before returning success.

The complete-value AUROC requirement is also inconsistent: §9.2 says 0.80, while §12.3 uses 0.70. Specify behavior in the gap, and specify the all-positive/all-negative case where AUROC is undefined.

### F08 — P1: S3 does not preserve the claimed anchor, and its selection leaks sealed results

**Draft locations:** [distillation anchor](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:385), [S3 protocol](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:509), [G3](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:516).

There are four distinct problems.

1. **An annealed penalty is not an exact anchor.** Distillation falls to zero by 30% of S2. The shared core and heads continue changing. Nothing guarantees that code-free outputs reproduce S1 at the end. Keep the frozen teacher as a real fallback; measure code-free drift.
2. **Mixing u alone does not guarantee transplant exactness.** V9 trains the PQ heads. At lambda = 0, passing teacher u through changed student heads can still change the plan. Mix complete teacher/student plan outputs, or take an explicit original-teacher branch at zero.
3. **S3a moves the state coordinates.** Training the encoder changes pooled C and possibly the tied token embedding/output matrix, while the core and selector are frozen. “Frozen core” does not imply a frozen abstraction interface.
4. **G3 selects using sealed standard/hard10 performance.** This contradicts §13.2's rule that development results determine every selection. Once used to choose S3 versus S2, that set is validation data.

G3 also omits a full re-check of G2.3. A model may preserve the first-hop swap rate and token loss while damaging imagined transitions.

**Required correction:** use development data for promotion, retain an untouched final test set, and rerun all relevant transition/value/termination checks after continuation. Begin with a decoder-only adaptation arm that freezes encoder, codebook and concept coordinates; keep encoder adaptation as a separate arm with measured drift and rebuilt caches/targets.

Specify whether S3b distillation compares original-teacher states from the original encoder or states from the adapted encoder. Those are different targets.

### F09 — P1: Independent S6 currently depends on missing S2 objects

**Draft locations:** [independent S6](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:93), [adapter locations](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:768), [concept-control measurement](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:783).

S6 starts from S1 regardless of S2. S1's ordinary V8 foundation has PQ heads, but **no H-NCP selector, code embeddings or hypothesis block**.

Yet S6 proposes adapting the hypothesis block, a VQ/selector-only arm and remeasuring G2.1(b) code swaps. These operations are undefined on the independent S1 anchor.

**Required correction:** split the claims:

- **S6-base:** independent S1 anchor; PQ/codebook-only versus matched LoRA/LSL/full-FT/replay. Measure foundation retention.
- **S6-H:** optional accepted S2/S3 anchor; include selector/hypothesis objects and evaluate authority, transition accuracy and search retention.

The first can run if H-NCP fails. Only the second tests continual learning of the complete searchable architecture. Neither should silently stand in for the other.

### F10 — P1: Search comparisons confound action spaces, post-training and feedback budgets

**Draft locations:** [S4 baselines](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:563), [S5 competitors](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:654).

Using the same original backbone does not make the proposed S5 arms differ only in node cost:

- The concept arm has a selector trained by expert iteration.
- The code-free token prior is produced through a different policy path.
- Actions are K latent choices versus three forced colours.
- Value heads receive different representations and potentially different distributions.
- S4 token baselines use the S1 anchor while the concept arm can use an adapted S3 checkpoint.
- The nominal trainable-capacity difference is disclosed, but disclosure does not remove the causal confound.

“Equal wall-clock and equal verifier calls” also requires a concrete allocation rule. Two methods with different costs cannot generally consume exactly the same amount of both resources. A shared cap is valid, but results should identify the binding constraint.

**Required correction:** use three distinct comparisons:

1. **Mechanism:** same accepted weights and latent actions, R = 1 versus R > 1; this isolates removing grounding more cleanly.
2. **Practical competitor:** best tuned token sampling/tree search under the same wall-clock cap and a common verifier-call ceiling.
3. **Feedback efficiency:** a separately reported fixed-call comparison, including differing runtime.

For S4, include a baseline from the same accepted starting checkpoint or clearly label the comparison as an entire-pipeline comparison. Report training/search data, trainable parameters, total learning compute, generated tokens, completed candidates, duplicates and cost overruns. Paired random proposal/allocation controls remain necessary.

## 6. Interpretation and measurement findings

### F11 — P2: “MSE blurring caused the earlier failure” remains a hypothesis

**Draft location:** [§2.3](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:133).

The conditional-mean argument is correct for an isolated squared-error predictor of a fixed target distribution, within its representational constraints. It does not establish that the joint NTP + NCP + VQ optimizer learns that exact conditional mean, or that its mean must decode to an illegal action.

A continuous decoder can map an intermediate vector to a valid output. Conversely, several discrete hypotheses can all be bad. The causal explanation needs direct evidence.

The supervision policy also matters: standard GraphNav uses generated coloured walks, while the hard generator uses a particular BFS path. Multimodality of valid solutions and multimodality of the **training labels conditional on the same prefix** are not the same object.

Improve P-blur tests by:

- measuring repeated continuations or the generator's conditional branching distribution;
- separately recording label ambiguity, gradient competition, forecast error and execution legality;
- comparing per-item illegal-transition rates with a denominator of all items;
- counting missing terminators and other failures separately.

The proposed “illegal transitions as a share of failures” can improve merely because a model creates more formatting failures. Also, failure of one prediction does not uniquely falsify the whole mechanism if the intended hypotheses never learned or were not deployable.

### F12 — P2: Some gates can reject the very multimodality the design intends to model

**Draft location:** [G2.2](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:441).

Suppose a fixed prefix has three equally likely demonstrated next modes, represented perfectly by three hypotheses. Even a perfectly calibrated selector can place at most 2/3 of the realized winners in its top two. The required 90% top-two coverage is then unattainable without extra information.

This does not prove the actual dataset has that ambiguity. It proves the threshold is not justified without measuring it.

Likewise:

- Winning-code frequency is sensitive to code permutations and restart rules.
- Top-label ECE can hide poor calibration on rare branches.
- Low code usage can be legitimate on deterministic contexts.
- A swap test can reward destructive sensitivity.

Use entropy/ambiguity-conditioned coverage, held-out assignment log loss, calibration bins with counts, legal-action coverage and downstream success. Compare with an appropriate attainable ceiling.

Centralize gate definitions. The transition requirement says 95% one-step legality, while the kill rule specifies a fallback only below 80%; behavior from 80% to 95% is not defined. State whether an inconclusive interval is missing evidence, a diagnostic failure or a reason to change the next experiment.

Honor the earlier lesson about sufficient exposure: complete the declared learning budget before interpreting a low score as a capability result. A failed evidence threshold does not by itself identify insufficient capacity, defective architecture or insufficient optimization.

### F13 — P2: V8's LSL failure cannot be explained away from serialization alone

**Draft locations:** [§2.6](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:174), [§10.2](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:728), [P3 rule](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:764).

The draft first correctly says the mechanism was not measured, then calls the failure predicted by theory and uninformative about LSL.

The paper's overlap argument concerns distributions of **activations at each weight matrix**. Shared input serialization is evidence of possible overlap, not a measurement of that overlap. Task headers, contextual representations and layer-specific gates may distinguish the tasks.

Even actual overlap does not mathematically imply complete forgetting or equality with always-open LoRA. The update may be harmless on part of the overlap, and gates may still protect other positions or matrices.

Therefore:

- V8 is negative evidence about the specific tiny LSL configuration and tested task sequence.
- It is not a controlled estimate of gating's benefit because the always-open control was missing.
- It does not refute the general LSL framework.
- It also does not confirm the proposed overlap explanation.

If LSL retains more than always-open by five points, that does not invalidate the paper's overlap policy. It invalidates or weakens this experiment's overlap/effect prediction.

Source: [LSL Appendix E](https://arxiv.org/abs/2610.02126).

There is a smaller factual correction: the draft's P1 row says V8 retention “held.” Both saved adjacent and cumulative retention decisions were false. The accurate wording is “point estimates remained close and new-text loss improved; formal preservation was not established.” See [phase-one decision](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v8_tiny_12h/runs/D_lsl_phase1/decision.json).

### F14 — P2: S6 needs new-task recall, same-weight installation tests and clearer data-access claims

**Draft locations:** [G6.0](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:738), [S6 acceptance](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:781).

Low gate activation on old tasks alone is insufficient: an always-closed gate passes that condition perfectly. Require useful activation/installation on the new task too.

Add:

- gate activation on old and new data, separated into prompt, response, node-selection and formatting positions;
- per-matrix and tail statistics, not just means;
- frozen-adapter, same-weight forced-open versus fitted-gate evaluations after every phase;
- free-generation as well as teacher-forced gate measurements;
- correct resets for document boundaries, branches and recurrent visits;
- retained-task performance and function change alongside gate statistics.

The same-weight test is especially important because V8 phase three went from 171/512 development shifted successes with the new adapter forced open to 1/512 after gate installation. A separate always-open trajectory helps, but cannot alone identify the installation loss of one checkpoint.

The proposed acceptance rule can credit a weak result: both LSL and VQ-only might learn essentially nothing, while LSL forgets slightly less than always-open. Require a declared minimum new-skill improvement and an acceptable old-task loss, not only a relative ordering.

Using old-task inputs to choose P2 and tune gates is useful research, but it is access to old-task data. Distinguish “no replay during adapter optimization” from “no access to prior data.” Use dedicated calibration/development samples and reserve final tests.

Finally, colour-count is a new task with a new header and response grammar. Give it its own verifier, split manifest and parser dispatch; the frozen V8 parser will not automatically support it. Header separability is a useful positive control, not evidence that arbitrary new skills can be isolated.

### F15 — P2: Statistical independence and test reuse need repair

**Draft locations:** [S1 reuse](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:283), [statistics](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/draft_v9.md:898).

V8 seed 811 and its sealed results informed V9's design. Reusing the checkpoint is efficient, but that evidence should be labelled historical/development. It does not become newly confirmatory because its checksum is verified.

A pooled item-level interval also does not capture uncertainty over training seeds. Thousands of items do not substitute for many independently trained models.

Recommended reporting:

- show every seed separately, including failed foundation teachers;
- identify whether two student runs use two independent foundation teachers or just two refit seeds from one teacher;
- report mean seed differences and their spread separately from conditional item-level intervals;
- avoid strong population-over-training claims from two or three seeds;
- preselect the primary budget and re-grounding interval, or correct for the tested budgets, R values, suites and methods;
- do not interpret an interval containing zero as equivalence;
- define equivalence/non-inferiority margins and required interval containment explicitly.

Generate fresh final evaluation manifests with the same immutable generator, excluded from new training and all tuning. Historical V8 test items can remain a reproducibility panel. This changes the evaluation sample, not the generator semantics.

The “wins at some budget point” prediction needs multiplicity control. Holm correction within a vaguely named family is insufficient until the actual list of hypotheses is fixed.

### F16 — P2: Some promised comparisons and lineage records are missing from the plan

- The 96k crossover claim compares NCP minus alpha-zero at 96k, but the arm table budgets only the NCP extension. Add a matched alpha-zero extension or remove that claim.
- The mandatory S2 queue contains only dense K1/K8. It can finish without testing the owner's recurrent replacement. Add a recurrent pair if recurrence is a central claim; otherwise label the minimum study as dense H-NCP.
- The recurrent K = 1 control has one seed while K = 8 has two. That does not satisfy the stated rule for decision-bearing comparisons.
- Multiple prior-coverage and closed-loop ablations have one seed. Keep them exploratory rather than using them for final architectural decisions.
- The draft references CF3, CF11, CF12, CF14 and a scratchpad NOTES.md without a resolvable source in the inspected repository file inventory. Copy the relevant evidence ledger into a stable referenced artifact or replace these references with exact result files.
- The recorded statements about earlier multi-seed experiments and all historical failure mechanisms should not receive stronger confidence than their accessible artifacts support.

## 7. Scalability and compute audit

### 7.1 Transformer-like scaling is plausible, but the proposal adds new bottlenecks

Dense attention/FFN blocks and fixed-factor pooling remain compatible with standard width/depth scaling. That is a reasonable architectural property to preserve.

The new costs require separate measurement:

- K-fold hypothesis evaluations;
- sequential feedback spans during training;
- tree size and branch-specific KV storage;
- decoder/encoder replay during grounding;
- value-model training and scoring;
- full-answer regeneration and backward passes for GradCuit;
- growing continual adapter/gate banks.

The draft's 28/5 block-position ratio is a useful rough arithmetic comparison, not a FLOP or latency proof. It omits details such as normalizations, heads, hypothesis evaluation placement, cache copies, root costs and synchronization. A new child also needs its code-free hypothesis/selector state prepared for subsequent expansion; charge that computation exactly once wherever it is scheduled.

For an 8-code batch, all code queries may be evaluated together. For other branching patterns they may be recomputed. State the actual caching and batching policy.

The final plan-decoding cost and any failed attempts must remain inside the budget. A method that searches cheaply but frequently produces an unverifiable final trajectory may lose the expected advantage.

### 7.2 The ~40M row is closer to 28M under the written shape

For the V8 block, with width d and FF = 3d:

~~~text
block parameters = 4d^2 + 3d(3d) + 2d = 13d^2 + 2d

foundation parameters =
    2048d                         token embedding
  + L(13d^2 + 2d)                 E + C + D blocks
  + (256d + 256)                  PQ prediction heads
  + 32d                           PQ codebook
  + d^2                           continuous port
  + 4d                            outer norms and gamma
~~~

Keeping the listed vocabulary and PQ configuration gives:

| Shape | Calculated foundation parameters |
|---|---:|
| d128, 10 total blocks | 2,448,640 |
| d256, 10 total blocks | 9,189,632 |
| d384, 14 total blocks | 27,894,016 |

The first row reproduces the actual V8 count. The last is not approximately 40M. These are foundation counts before H-NCP additions; any intended co-scaling of vocabulary, codebooks or other dimensions must be stated.

The recurrent S2 count likewise cannot simply reuse 2,055,424 after adding B_H and changing the readouts. Measure active, trainable, resident-teacher and cache memory separately.

### 7.3 The Q6 runtime cannot be accepted without measured greedy latency and an execution matrix

The written matrix is four methods, two suites, 512 items and budget multipliers 1+2+4+8+16+32 = 63.

If t_g is the relevant average greedy-answer wall-clock in seconds, one seed's maximum allocated runtime is:

~~~text
T_Q6 = 4 × 2 × 512 × 63 × t_g
     = 258,048 t_g seconds
     = 71.68 t_g hours.
~~~

An eight-hour allocation corresponds to about 0.112 seconds per greedy answer for one seed, or 0.056 seconds for two seeds, before extra variants and overhead. Early success, shared nested-budget traces or efficient batching may reduce actual cost, but those assumptions must be explicit. This arithmetic does not prove eight hours is impossible; it shows what must be measured to justify it.

The four R values, GradCuit/random controls and two-seed rule are not fully enumerated in Q6's four-method count. Q5 also leaves B_train unspecified. Its search data generation can dominate head training.

For S6, rank-16 adapters on all projections and larger GMMs should not inherit rank-4 q/v timings without a pilot. Break out training, negative fitting, positive fitting and each old-task evaluation.

On a single M2, parallel queue notation should mean controlled interleaving unless resource profiling justifies simultaneous jobs. Preserve explicit testing reserves.

### 7.4 Fixed tokens across sizes is only one scaling experiment

At 264.6M non-padding tokens, the larger shapes receive fewer tokens per parameter. A text gap changing with size under that budget mixes capacity and optimization/exposure effects.

Report both equal-data and equal-total-compute comparisons where practical, including teacher training and refits. Test whether concept action coverage and rollout consistency scale, rather than treating lower token loss as validation of searchability.

## 8. Minimal revised sequence

This is a proposed repair order, not authorization to start training.

1. **Lock contracts and evidence ownership.** Resolve F01, F05, STOP semantics, plan identity, split usage and missing reward/value definitions.
2. **Reproduce foundations.** Train the matched new seeds with the full declared exposure. Preserve historical results separately.
3. **Run V8 reproduction, V9 K1 and V9 K8.** Keep a small controlled matrix before adding every secondary arm.
4. **Measure action execution.** Decode every code for a limited set of actual prefixes. Test legal diversity, plan-to-action agreement and imagined-versus-grounded successors.
5. **Test short imagined lookahead.** Start with fixed horizons and R = 1/2, including the execute-one-step/replan protocol. Add longer rollouts only as separately measured conditions.
6. **Adapt the decoder, then test encoder adaptation.** Re-check transition and calibration contracts after each change. Select using development results.
7. **Bootstrap expert iteration from verified outcomes.** Retain terminal failures and unfinished trajectories correctly. If all outcomes are zero, report the absence of a useful signal rather than claiming policy improvement.
8. **Compare search under a complete cost ledger.** Mechanism comparisons first, practical token baselines second, separate feedback-efficiency analysis.
9. **Run S6-base independently.** Include same-weight gate installation and old-task telemetry.
10. **Run S6-H only if an H-NCP anchor exists.** This is the experiment that addresses continual learning of the complete V9 system.
11. **Evaluate locked final checkpoints on fresh tests.** Report skipped, partial and inconclusive work explicitly.

This sequence retains NCP, recurrent replacement, exploratory codes, GradCuit, MCTS and LSL. It changes what each experiment is allowed to establish.

## 9. Required validation before a long run

These are acceptance specifications for future implementation, not tests executed in this review.

| Check | Failure it must detect |
|---|---|
| Original V8 refit reproduction | Changed head/port ownership, normalization, optimizer groups or loss reductions |
| V9 K1 versus K8 contract | Unintended differences beyond the explicitly varied hypothesis mechanism |
| Inference causality and oracle exclusion | Future targets affecting deployed code choice, transitions or value inputs |
| Reward sign fixture | A binary failed answer producing a no-op GradCuit update |
| Plan replay identity | Recomputing a different plan during grounding or final decoding |
| Cached versus full replay | Stale suffixes, incorrect shift boundary, shared mutable sibling caches |
| Recurrent visit accounting | Reusing one visit's cache/state for a different repeated application |
| STOP and strict verification | Returning an answer without the required complete terminator or on a false learned terminal |
| Counterfactual label validity | Assigning original-demonstration labels to alternative search prefixes |
| Deployment-aligned transition audit | Legal nearest projection masking an illegal decoded hop |
| Same-weight gate installation | Skill lost only when the fitted gate replaces forced-open operation |
| Split and selection audit | Sealed outcomes used for promotion, budget selection or phase choice |
| Complete cost reconstruction | Uncharged failed generations, GradCuit backward passes, grounding or verifier work |
| Frozen fallback/provenance | A claimed teacher/zero-control anchor that has silently changed |

CPU/MPS bit-identical search trees should not be assumed when close floating-point scores determine argmax choices. Define deterministic tie-breaking, fixed-platform reproducibility and numerical tolerances separately. The existing compiled/eager decoding incident makes trained-weight replay checks especially important.

## 10. Review of the draft's own first-principles review

The self-review asks useful questions, but several answers need correction:

- **“Only K changes.”** Not true relative to V8 under the present head, port and feedback contracts.
- **“Committed plan.”** A selected hypothesis is committed as an index; semantic validity remains a measurement.
- **“Best-of-m fixes the prior mismatch.”** It changes the mismatch and may help empirically; it still uses hindsight for m > 1.
- **“Distillation gives an exact zero-control anchor.”** A finite, annealed penalty provides no exact guarantee.
- **“Frozen-core continuation is safe.”** The encoder can move the coordinates that the frozen core consumes.
- **“The two searches explore the same tree.”** Requires an action mapping and STOP contract that are absent.
- **“V8 was the predicted LSL failure case.”** Activation overlap and the mechanism were not measured.
- **“GradCuit cannot be RL because no shared weights change.”** Too categorical. It uses a reward-weighted policy-gradient-style update to instance-specific variables. The useful distinction is persistent policy learning versus test-time latent optimization.
- **“Same weights isolate node cost.”** Different action representations, learned priors and training histories still matter.

The revised self-review should cite measurable contracts for each answer and label what is established, inferred or unknown.

## 11. Paper and evidence audit

| Source checked | Supported connection | Limit on the transfer to V9 |
|---|---|---|
| [NCP-ArchPreview](https://arxiv.org/abs/2609.10715), local PDF equations 9 and 22; §5.1 | Soft-PQ expectation, MSE next-concept objective, and adapting codebooks/prediction heads are faithful objective-level references | The paper does not validate V9's hypothesis action semantics or latent MCTS. Its +VQ adaptation is not a proof of multi-phase retention |
| [Explorative Modeling](https://arxiv.org/abs/2607.27372), §3 and §6–7 | Best-of-candidate objectives motivate exploring alternatives | Enumerated heads plus a fitted selector are an adaptation. The paper treats autoregressive language modeling as a difficult case; prior deployment is not automatically solved |
| [GradCuit](https://arxiv.org/abs/2608.02585), Appendix A.1 and random-direction ablation | Frozen-weight latent optimization and reward-weighted sequence gradients | The failure reward mapping is essential. Code-embedding optimization is a new interface |
| [Local Support Learning](https://arxiv.org/abs/2610.02126), Algorithms 1/2/4 and Appendices E/F/H | Forced-open adapter training, post-fit local gates, smoothing and function-change diagnostics | The theory concerns activation support; serialized grammar does not prove overlap or inevitable forgetting |
| [AB-MCTS local paper](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/AB-MCTS.pdf>), §3.4 | GEN/CONT allocation and posterior-based answer refinement | Inner PUCT and outer answer refinement have different state/reward semantics. Borrowing the outer algorithm does not specify their coupling |
| [ETA local paper](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Emergence in Residual Stream/Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning.pdf>), rate–distortion ablation | A frozen pretrained base helped meaningful abstraction in the reported setting | This is evidence for a control, not a universal impossibility result for carefully continued joint training |
| [GRAM local paper](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Loop Papers/Generative Recursive Reasoning.pdf>), §2.3 and Appendix A.2 | Stochastic recursive trajectories and latent outcome prediction motivate useful experiments | It does not establish that a value head trained on grounded V9 states is reliable on imagined ones |
| [HRM-Text local paper](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Loop Papers/HRM_Text.pdf>), Table 3 | The cited Transformer MMLU progression, 40.55 → 47.72 → 53.15, supports separating response-only loss and PrefixLM | V8 already has response-only symbolic loss; transferring the total paper gain would double-count that change |
| [Free Transformer local paper](</Users/shubhankankrishna/Desktop/EncoderConceptDecoder/Loop Papers/Free Transformer.pdf>), latent model and free-bits study | Latent variables can carry useful future information and need control against training shortcuts | Its variational objective and latent sampling do not establish V9's WTA/selector calibration |

The V5–V8 code shares the cited NCP loss form, but the full architectures, schedules, continuous ports, optimizers, task mixtures and scales differ from the paper. Replace “faithfully tested the published architecture” implications with the narrower objective-level statement.

Historical steering losses are a sound reason for strong baselines. Where their confidence intervals include zero, describe them as lower point estimates with no established advantage, not repeated statistically established defeats.

## 12. Final score

This is a score for the **current design document's coherence and readiness**, not a measured score for a V9 model.

| Dimension | Score / 10 | Weight | Reason |
|---|---:|---:|---|
| Problem framing and use of evidence | 8 | 20% | Good questions, useful controls and substantial experiment history |
| Architectural contracts | 5 | 25% | K1, anchor, state/action and cache semantics need repair |
| Objectives, search and RL | 5 | 25% | Plausible direction; hindsight, reward and value/STOP issues are consequential |
| Experimental and statistical design | 6 | 15% | Better seeds and pairing; selection/test reuse and missing comparisons remain |
| Continual-learning design | 6 | 10% | Improved telemetry; incorrect causal interpretation and missing independent objects |
| Scalability and execution planning | 6 | 5% | Scalable components, but incomplete runtime accounting and a wrong size estimate |

The weighted result is 5.9/10, rounded to 6/10. The strongest reason to continue is that the draft asks a concrete, valuable question about executing plans through a concept interface. The strongest reason to revise first is that its current gates can confuse reconstruction, sensitivity, legality and actual planning ability.

**Overall score for draft_v9.md as written: 6/10.**

