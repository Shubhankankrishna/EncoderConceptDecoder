**V15 Astra: first-principles architecture review — 18 October 2026**

**Verdict:** The reviewed architecture is coherent enough for controlled experiments. The current implementation has credible evidence of correct masking, frozen-codec integration, and short-unroll GRAM training. It does **not yet establish an efficient reasoning architecture**, a benefit from stochastic trajectories, or readiness for the complete eight-stage system. The highest-value next work is stronger controls and task design, followed by a few specific specification repairs. Adding strategy modes, gradient search, MCTS, and continual learning together would make the result harder to interpret.

This review examines the original 614-line [V15 Astra design](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V15_Astra.md), the superseding 1,767-line [engineering specification](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V15_Astra_Full_Training_Blueprint_Reviewed/V15_Astra_Engineer_Training_Spec.md), their companion reviews/checks, the actual `v15_astra_m2` implementation, existing integration-test evidence, primary papers, and new CPU-only train/validation diagnostics. Historical V1–V14 performance claims quoted in the original design were not independently rerun. New measurements below are frozen at **12:41 IST, 18 October**, with validation checkpoint **48,225**; training continues separately.

Evidence and reproduction: [diagnostics JSON](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v15_astra_review_20261018/diagnostics.json), [diagnostic script](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v15_astra_review_20261018/review_diagnostics.py), [tokenizer collisions](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v15_astra_review_20261018/pretokenizer_collisions.json), [figure](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v15_astra_review_20261018/review_evidence.png). The diagnostics use train/validation only, do no neural training/inference, and confirm unchanged live experiment source hashes.

**1. Define the problem before choosing components.**

For the present tasks, the agent observes a fully specified state `s`, chooses one of at most four legal actions, and seeks a goal. Its practical objective is high probability of reaching that goal within a time budget:

\[
\max_\theta\;\Pr_\theta(\text{goal reached within }H\text{ actions}),
\qquad \text{subject to training, inference, and memory budgets}.
\]

The implemented training objective instead predicts the canonical code hierarchy of one action chosen from a shortest-path trace. That is supervised imitation of an expert next-action policy. It can be a useful route to the practical objective, but is not itself trajectory optimization or a learned world model.

A complete architecture must answer four separate questions: Does the representation retain the information needed to act? Does repeated computation improve decisions? Does stochastic computation discover useful alternatives? Does the improvement justify total cost? Each added mechanism needs evidence for its own question. Paper names and individually valid equations do not establish that their combination is necessary.

**2. Establish which version of V15 is under review.**

| Property | Original V15 Astra | Reviewed blueprint | Running M2 pilot |
|---|---|---|---|
| Codec | Newly trained, 16 text tokens, four levels, 23 codes | Frozen official NCM, checkpoint-dependent schema | Frozen LM1B NCM, 128 text IDs, 12 levels, 641 codes |
| Model budget | ~9.68M stored scalars including its small codec/adapters | Recount after actual checkpoint integration | 10,000,333 trainable GRAM parameters **plus** large frozen codec |
| Reasoning placement | Recur separately at each code level, with previous reconstruction | Recur before the complete next-block head | Two slow × two fast cycles once per action block |
| Context | Observed token-level prefix | Pooled observed codes; optional typed state | One observed code block + 88 current-state integers |
| Stochasticity | Four mode embeddings; GRAM a separate alternative | Target-conditioned posterior and inference prior | Real prior/posterior, diagonal Gaussian, full BPTT |
| Later stages | Offsets, small memory, bounded two-slot sleep | Strategy, gradient search, MCTS, distillation, CL | Not implemented in the active experiment |

The current model is a documented Astra adaptation, not the original architecture with a faster tokenizer substituted. In particular, generated coarse codes refine the **output head**, not a fresh HRM reasoning rollout at every level. A future comparison must not silently combine parameter counts or computational claims from these different designs. The supersession is explicit at [specification line 34](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V15_Astra_Full_Training_Blueprint_Reviewed/V15_Astra_Engineer_Training_Spec.md:34).

**3. Check information preservation before learning.**

Let `O` be the full observation and `C=T(O)` its frozen encoding. For code-only action selection to equal the best observation-based policy, `C` must preserve decision-relevant distinctions. A simple impossibility test is enough: if two observations have identical `C` but require disjoint optimal action sets, no code-only policy can always choose correctly for both.

There are two concrete limits in the present setup:

* **The maze text omits walls.** Walls enter only through the typed state vector in [tasks.py:70](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v15_astra_m2/tasks.py:70). In the training grid records, **3,173 identical-text prompt groups** have no common optimal action across their underlying states. These groups cover **62,408 rows**. This is missing input information before NCM, not a quantizer bug. A text-only ablation on these prompts would be structurally disadvantaged; first serialize the walls to make that comparison meaningful.
* **The chosen pre-tokenizer destroys case and some accent distinctions.** The pinned tokenizer maps `Foo = 1` and `foo = 1` to identical IDs; it also maps `résumé` and `resume` identically. Deterministic NCM therefore receives identical input for these strings. This checkpoint cannot support a claim of exact, case-sensitive code reconstruction without changing the representation contract. The current lowercase action grammar avoids that particular problem.

The existing codec audit gives LM1B **99.36% payload-token fidelity**, but only the seven supported action sentences have exhaustive action round-trip acceptance. Padding reconstructs as extraneous text, so parsing stops after the first complete canonical sentence. The strict numeric-list score also counts spaced minus signs and suffix numbers; it is not a clean semantic error rate. These facts support the narrow action interface, not arbitrary text or symbolic-program fidelity. [Codec audit](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v15_astra_m2/reports/codec_audit.json).

**Decision:** Keeping exact observed state is legitimate and sensible for this pilot. State-conditioned results must remain separate from claims that NCM learned or preserved the reasoning state. To test NCM's input value, compare a trained state-only control with the same state+code model; to test code-only reasoning, first provide a complete textual observation.

**4. Ask whether the representation saves the computation that matters.**

NCM generates a hierarchy of refinement levels and predicts positions within each level in parallel. The published method also uses cumulative reconstructed features as generator conditioning. Astra substitutes its own recurrent/context/head design and learned level summaries, so its speed and quality must be measured independently. [NCM paper, generation and architecture](https://arxiv.org/html/2610.14162).

In this pilot, **641 categorical outputs are generated to communicate one of seven actions**. Their nominal fixed-width payload is `641 × 13 = 8,333` bits; the complete seven-action label has at most `log2(7) = 2.807` bits of entropy. This comparison is not a compression lower-bound claim about all NCM text. It exposes the mismatch between a general text interface and this tiny action vocabulary.

The code tables use **6,291,456 parameters, 62.9% of all trainable GRAM parameters**. They are legitimate counted parameters, but most of the budget is not recurrent computation. The two fast/slow modules together contain 1,090,560 parameters. A direct seven-class head is therefore an essential control, not a replacement assumed to win in advance.

Reusing NCM avoids codec pretraining; batching and caching avoid re-encoding training samples repeatedly. Those are real savings. Frozen encoder/decoder execution, code-head softmaxes, and 12 sequential head evaluations still cost time. The complete system includes **111,404,672 frozen codec parameters**, plus frozen buffers, in addition to the 10M trainable model. It is a 10M-parameter **training component**, not a 10M-parameter complete deployed system.

**5. Interpret the loss through the actual target entropy.**

New examination of the seven canonical action hierarchies found:

| Quantity | Measured value | Implication |
|---|---:|---|
| Code positions identical across all actions | 493 / 641 = 76.91% | Most positions can be correct without selecting the right action |
| Different action prefixes after levels 1 / 2 / 3 / 4 | 1 / 3 / 6 / 7 | The first four true levels identify the action |
| Positions in levels 5–12 | 592 / 641 = 92.36% | Their labels are determined by the earlier true action prefix |
| Median code agreement between different action templates | 90.02% | High code agreement can accompany the wrong action |
| Training action entropy | 2.635 bits | The label distribution is small |
| Training action entropy conditional on task domain | 1.915 bits | Observing the domain removes still more uncertainty |

A theoretical lookup predictor that knows training frequencies and the teacher-forced earlier target levels, but receives **no problem input**, can attain approximately **0.003243 nats/code** on that training distribution under the parallel-within-level factorization. This is an analytically computed empirical optimum, not a trained validation baseline. It illustrates how small the loss scale can be without solving the task.

The latest prior-MC validation NLL is **0.001239 nats/code**, or **0.794 nats for the complete hierarchy**. It accompanies **82.03% optimal-action accuracy**, **92.97% legality**, and **64.84% exact canonical-code agreement**. None of those quantities can substitute for the others.

Teacher forcing is legal here; this finding is **not target leakage**. Later levels are allowed to use earlier true levels during training. The issue is interpretation and exposure shift: at deployment, early mistakes become conditioning for the rest of the hierarchy. Independent choices within a level can also form combinations absent from the seven canonical hierarchies.

**Decision:** Headline decoded action quality and complete-episode success. Keep summed NLL and level-specific NLL as diagnostics. A code perplexity near one is not evidence of nearly solved reasoning or text modeling.

**6. Check the probability and causality contracts.**

For fixed recurrent state `h` and observed memory `M`, the output distribution is

\[
p(Y\mid M,h)=\prod_{\ell=1}^{12}\prod_j p(Y_{\ell j}\mid M,h,Y_{<\ell}).
\]

This is normalized because each categorical distribution sums to one and the level dependency is acyclic. It assumes conditional independence of positions within the current level; shared blank-query attention does not remove that probabilistic assumption. It is a coherent model, not a proof that the assumption is adequate for every language or reasoning task.

The implementation uses blank position/level queries, masks out current/future target summaries, and keeps observed context separate. Tests check target perturbations, earlier-level sensitivity, and agreement between parallel teacher-forced scoring and sequential level scoring. The target-informed posterior is excluded physically from deployment checkpoints. These are substantive strengths. [Model implementation](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v15_astra_m2/model.py:130), [executed base tests](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v15_astra_m2/reports/unit_tests.log), [executed GRAM tests](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v15_astra_m2/reports/gram_unit_tests.log).

The code loss is not text perplexity. With a fixed canonical encoder and stochastic frozen decoder, a joint score `log p(C*(text)|X) + log p_D(text|C*)` can lower-bound text log likelihood. Code CE omits the reconstruction term. Greedy decoding plus first-sentence parsing instead defines an induced action policy. Its action probability sums over all code hierarchies mapping to that action, not just one canonical code sequence.

**7. Determine what recurrence contributes.**

Sharing fast/slow modules can trade extra computation for fewer unique parameters. Repeating a transformation does not automatically create an algorithm, ensure convergence, or make additional iterations useful. Current `2 × 2` recurrence is four fast calls and two slow calls, with states reset at each decision. It has no adaptive halting or persistent episode memory. Original HRM's training schedule uses a last-state gradient approximation and deep supervision; this short full-BPTT adaptation is a different experiment. [HRM paper](https://arxiv.org/html/2506.21734v3).

Two design bottlenecks deserve tests: each observed 641-code block is pooled into only 12 vectors, and a direct context-to-output attention path can support predictions even if the recurrent state contributes little. These are legal interfaces; neither establishes that recurrence is useful. Zeroing a trained recurrent state is an intervention with distribution shift, so the decisive control is a separately trained nonrecurrent model under matched budgets.

The best immediate controls are a small state-to-action classifier, a nonrecurrent code predictor, deterministic recurrence, and stochastic recurrence. Include required-path-length strata and cases that defeat local distance heuristics. Do not infer reasoning depth from the count of internal slots or NCM levels.

**8. Evaluate GRAM's stochasticity as a hypothesis.**

The implementation defines an inference prior `p(tau|X)` and a target-conditioned training posterior `q(tau|X,Y)`. Its short-unroll objective is

\[
\mathcal L=\frac{1}{641}\left\{
\mathbb E_q[-\log p(Y\mid X,\tau)]
+\sum_r\mathbb E_{q(\tau_{<r})}\mathrm{KL}(q_r\Vert p_r)
\right\}.
\]

The conditional prior and posterior must be evaluated at the same posterior-history state; the code does this. Reparameterized sampling allows gradients through Gaussian means/scales. Beta is one, likelihood and KL use consistent units, and full/truncated paths have separate tests. The paper's practical last-transition surrogate is explicitly different from its full trajectory ELBO. [GRAM paper, objective and truncation](https://arxiv.org/html/2605.19376v1), [implementation](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v15_astra_m2/gram.py:77).

But a valid ELBO does not imply that stochastic reasoning is useful. For the joint data/posterior distribution, the identity

\[
\mathbb E_{X,Y}\mathrm{KL}(q(\tau\mid X,Y)\Vert p(\tau\mid X))
=I_q(Y;\tau\mid X)+\mathbb E_X\mathrm{KL}(q(\tau\mid X)\Vert p(\tau\mid X))
\]

shows that KL penalizes information the posterior adds about the answer beyond the observed input. For a fully observed task with a small action alphabet, the model can often lower its objective by solving the decision through deterministic context and making its stochastic branch uninformative about the label. There are **6,144 Gaussian coordinates per two-transition trajectory**; their penalty is summed. That dimensionality can make optimization delicate, but alone does not prove an excessive penalty or prescribe a beta value.

Current evidence is consistent with an underused latent branch: raw KL is near numerical zero; at the latest 32-example diagnostic, prior, prior mean, and posterior all score **68.75%**, and the second prior draw adds no new distinct optimal action. This does not prove all internal trajectories or logits are identical. Noise can also vary paths leading to the same action. More seeds and fixed-checkpoint logit/behavior interventions are needed before calling collapse definitive.

The slightly negative logged KL, about `−1.2e−4` per trajectory, cannot be a true Gaussian divergence. The existing FP32 expression subtracts nearly equal terms and is consistent with cancellation near equality. For a future version use a stable variance-ratio expression, with `t=log(sd_q)-log(sd_p)`:

\[
\mathrm{KL}_i=\tfrac12\{\operatorname{expm1}(2t)-2t+[(\mu_q-\mu_p)/sd_p]^2\},
\]

and test very small differences against FP64, using a series near zero if needed. Do not silently clamp the active training loss or treat negative roundoff as an improvement.

**One specification repair:** [Stage-2 pseudocode, lines 647–662](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V15_Astra_Full_Training_Blueprint_Reviewed/V15_Astra_Engineer_Training_Spec.md:647) sums a reconstruction loss after every supervision phase. With `N_sup>1`, this is not the single-terminal-observation ELBO derived immediately above it. Define the exact full-trajectory routine with one terminal reconstruction term, and name repeated intermediate reconstruction losses an auxiliary deep-supervision objective. Keeping all graphs does not by itself resolve that objective difference. Current `N_sup=1` behavior is unaffected.

**9. Test whether observed performance exceeds simple reasoning rules.**

I evaluated a fixed, untuned heuristic on exactly the validation indices chosen by the trainer. For a maze, choose a legal neighbor minimizing Manhattan distance to the goal. For modular arithmetic, choose a legal successor minimizing circular distance to the goal. Ties use the existing action order. It uses known transitions and current state; it never uses BFS distances, target actions, or future answers to choose an action.

| Policy on the same 128 examples | Overall optimal action | Grid, 104 examples | Modular, 24 examples |
|---|---:|---:|---:|
| Training-majority baseline | 33.59% | — | — |
| Uniform legal action, expected | 43.95% | 44.47% | 41.67% |
| Fixed distance heuristic | **84.38%** | **88.46%** | **66.67%** |
| GRAM at step 48,225 | 82.03% | 88.46% | 54.17% |

Uniformly random tie-breaking among the heuristic's best moves gives 85.55% expected accuracy on this panel. Across all **14,031 validation transitions**, the fixed heuristic reaches **86.13%**, including 91.46% grid and 68.27% modular. GRAM has not been evaluated on that full panel here, so those full-panel percentages are not a matched comparison.

The heuristic has hard-coded environment knowledge and legality enforcement, while the model emits unconstrained actions. It is a practical baseline, not a matched neural ablation. The difference on 128 examples is small and no paired significance claim is justified from the currently saved aggregate model metrics. Nevertheless, the current evidence does not show a decision advantage over this inexpensive rule. My earlier assessment based on majority/random baselines was incomplete.

**Decision:** Continue the registered GRAM/HRM comparison, but require stronger controls before claiming that recurrent stochastic reasoning is paying for itself. Add a neural policy with legality filtering as a separately labeled evaluation, and retain raw-generation legality as a diagnostic.

**10. Make long-horizon and comparison claims identifiable.**

Only **81 of 3,546 validation episodes** require at least ten optimal actions, and **none requires twenty**. The fixed 128-transition panel contains just **two states needing at least ten remaining actions**. A metric called “solved by 20 steps” is not evidence of solving tasks that require twenty steps. The existing rollout code appropriately records minimum required lengths, but chooses only 24 episodes per split. Long-horizon claims need deliberately sampled minimum-length buckets and enough independent groups.

Further limitations of the present comparison:

* GRAM uses FFN width 432; HRM uses 544 to approximately match total trainable parameters. GRAM also spends 237,505 parameters on training-only posterior components. A difference compares two budgeted systems; it does not isolate noise as the only causal change.
* Equal allocated wall time does not imply equal FLOPs, updates, thermal conditions, or inference cost. Sequential runs on a fanless laptop can see different sustained conditions. Report actual optimizer time, exposure, throughput, and end-to-end latency.
* One seed is a pilot. Same seed does not make different-shaped networks identically initialized. Use multiple independent paired data streams and an additional same-backbone deterministic control for stronger conclusions.
* The 128 examples are reused to select the best checkpoint. That is valid development practice, but the selected score is optimistic and is not final test evidence.
* `optimal_action_rate` is example-weighted. The saved group-bootstrap field estimates the **unweighted mean of group means**, a different quantity. At this checkpoint those points are 82.03% and 81.36%; the latter's interval must not be attached to the former as if they were identical. For an example-weighted interval, bootstrap whole groups while recomputing the ratio of total successes to total examples. [Evaluation code](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v15_astra_m2/evaluate.py:16).

Group-disjoint splits, independently checked transitions, no optimal-action/distance inputs, and sealed test/OOD model selection are strong foundations. The missing controls concern what the results demonstrate, not an accusation of label leakage.

**11. Add global strategies only if local stochasticity proves insufficient.**

The proposed mixture `p(Y|X)=sum_k pi_k(X) p_k(Y|X)` is normalized. A single mode must persist over the scope assigned to it, and a router may use only available observations. Four learned vectors can still produce identical actions or split by task domain instead of learning different strategies.

The blueprint correctly warns that inserting noisy estimates of per-mode ELBOs into `logsumexp` is not automatically a valid expected lower bound. A clean alternative is a joint variational distribution over mode and trajectory:

\[
\mathcal B=\sum_k r_k\big(\mathbb E_{q_k}\log p(Y|X,k,\tau)-\mathrm{KL}(q_k\Vert p_k)\big)-\mathrm{KL}(r\Vert\pi).
\]

Use an explicit variational router or an appropriately independent estimation scheme; selecting responsibilities from the same noisy estimates changes the estimator. Hard winner selection is another objective. Neither deserves an exact-mixture likelihood label by default.

If one mode is retained across a multi-block action, scoring and generation must retain that same latent choice; independently mixing each block would define a different model. Require equal-cost independent-sampling controls and meaningful differences in **verified action sequences**, not just hidden-vector distances. Current seven-action targets are a weak setting for learning elaborate global strategies.

**12. Treat test-time gradient search as a costly optimizer, not a source of knowledge.**

The relevant identity is

\[
\nabla_z\mathbb E_{\xi,C\sim p(\cdot|z,\xi)}R(C)
=\mathbb E[R(C)\nabla_z\log p(C|z,\xi)],
\]

where base Gaussian noise `xi` is independent of `z`, all prior computations remain differentiable in `z`, and the code trajectory is sampled from the distribution being scored. The verifier can be discrete. Replaying a sample must preserve its noise and previous generated codes. Changing to greedy codes, rescoring canonical re-encodings, or cutting the `z` path changes or invalidates the claimed estimator.

A leave-one-out reward baseline is valid for independent samples; a current-batch mean includes the sample's own reward and changes the finite-sample expectation. A frozen decoder depending only on codes contributes **zero direct `z` score**, even if decoder outputs are sampled. The blueprint's suggestion that sampling alone might require an extra decoder score needs that dependency qualification.

GradCuit supplies evidence for latent insertion with direct attention paths and reward-weighted gradients in its tested backbones; Astra's discrete-code system needs its own experiments. [GradCuit paper](https://arxiv.org/abs/2608.02585).

Sparse all-zero rewards give no useful direction. Exact shortest-path improvement rewards are especially problematic as evidence of learned reasoning: the reward computation may already solve the problem. Count that solver cost and include it as a direct baseline. A Euclidean trust region limits `z` movement, but does not guarantee small change in the output policy; measure an appropriate policy-distance diagnostic if claiming trust-region behavior. Keep an unmodified candidate, regenerate after updates, and compare with sampling and random/sham updates at equal time and verifier calls.

**13. Repair the value contract before implementing MCTS.**

The blueprint's action-level search, terminal reward counted once, horizon-aware node keys, legal transitions, and candidate-conditional empirical priors are well motivated. A complete decoded action is a tree edge; a code refinement level is not.

However, [the value definition around line 1023](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V15_Astra_Full_Training_Blueprint_Reviewed/V15_Astra_Engineer_Training_Spec.md:1023) names `V(s,h,z)`, while search and training records contain a remaining horizon. In a finite-horizon task, the same state can have success probability zero with one action left and nonzero with three. Require the API and input representation to include **remaining horizon** and define the policy under which success is estimated. Recommended contract: `V(state, horizon_left, policy_version, strategy_context)` with pinned hidden-state recomputation semantics.

Mixing random-policy and search-policy outcome labels without defining their mixture or conditioning can train an ambiguous value target. That may still be a useful heuristic, but is not automatically either `V^policy` or optimal value. Calibration must match intended deployment.

For these three/four-action environments, exact legal-action enumeration is extremely cheap. Sampling 641-code blocks to discover that action set can be wasteful and can permanently omit a necessary branch if proposals are fixed. Compare enumeration+ranking and ordinary BFS/A* as system baselines; larger generative action spaces are a more persuasive use case. PUCT with incomplete candidates is not guaranteed to match exhaustive search merely by increasing simulations unless all necessary actions can enter the tree.

**14. Distillation and continual learning need exact scope and capacity contracts.**

Search-to-canonical-code cross-entropy is a reasonable imitation surrogate. It is not exact KL between action policies: many code sequences can parse to the same action. Verify that the student improves with search and temporary state removed, and account for teacher generation cost. A simple amortization condition is `future uses × per-use savings > teacher generation + distillation cost`; it depends on the workload.

A second mathematical distinction matters for sleep. Full autoregressive hierarchy KL decomposes as

\[
\mathrm{KL}(p_T(Y|X)\Vert p_S(Y|X))
=\sum_\ell\mathbb E_{Y_{<\ell}\sim p_T}\mathrm{KL}(p_T(Y_\ell|X,Y_{<\ell})\Vert p_S(Y_\ell|X,Y_{<\ell})).
\]

Matching conditional distributions only on **data** prefixes is a local conditional-distillation objective, not that full joint KL. The specification allows surrogates, but must pin the prefix distribution as well as how GRAM trajectories/modes are marginalized. Identical teacher/student prefixes are necessary, not sufficient for full-KL equivalence. [Sleep objective](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V15_Astra_Full_Training_Blueprint_Reviewed/V15_Astra_Engineer_Training_Spec.md:1336).

Gated low-rank updates cannot distinguish old and new tasks when the gate sees indistinguishable activations but the tasks require incompatible changes. A closed gate can protect old tasks by refusing to learn. Measure acquisition and retention together. LSL motivates support-aware gates, while RCDL derives particular functional updates for growing retrieval layers; arbitrary bounded retrieval does not inherit those properties. [LSL](https://arxiv.org/abs/2610.02126), [RCDL](https://arxiv.org/abs/2610.03858).

**Concrete capacity gap:** the revised CL loop calls `adapters.append(...)` each phase, without a stored-adapter/GMM cap or mandatory compaction when consolidation fails. Limiting simultaneously active gates is not a bound on stored parameters. The original design's two-slot replacement rule was more explicit. Choose either a genuinely fixed-slot consolidation/eviction policy with rollback, or explicitly growing capacity with measured storage and latency. The title “bounded consolidation” does not settle this. [CL loop, line 1368](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/V15_Astra_Full_Training_Blueprint_Reviewed/V15_Astra_Engineer_Training_Spec.md:1368).

Also specify whether gates read frozen-base or adapted activations. Frozen-base gating may require an extra base computation path; budget it. Retrieval keys/values tied to mutable hidden features need versioning or rebuilding after policy changes. Sleep is a proposed consolidation method with empirical work behind it, not a guarantee that fixed capacity can absorb unlimited knowledge. [Sleep paper](https://arxiv.org/abs/2606.03979).

**15. Keep M2 optimization subordinate to a meaningful experiment.**

The current system choices are practical: cached uint16 codes, tied factorized code tables, short full-BPTT recurrence, FP32 MPS training, CPU reference codec, accumulated microbatches, atomic resumable checkpoints, and deadline/resource checks. At the snapshot the MPS driver allocation is about **2.83 GiB**, throughput about **22 blocks/s**, and available system memory about **2.34 GiB**. Those are snapshots, not total peak-RAM or universal speed claims.

The larger MPS codec audit found 283 hard-code differences across 1,312,768 positions despite very small pooling errors. Retaining CPU canonical encoding is defensible because quantization can amplify tiny numerical perturbations into different labels. A future MPS-codec variant could be versioned and evaluated, but should not silently share a CPU code identity. [Device audit](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v15_astra_m2/reports/encoding_device_audit.json).

Before low-level kernel tuning, measure the components that dominate useful decisions per second: code-head projection/softmax, repeated pooling of earlier levels, frozen decode/encode, recurrence, and transfers. Cache reusable observed context and previous-level summaries where parity tests permit. Investigate restricted-action scoring or a direct action head as explicit controls. Chunking logits, mixed precision, or compilation are follow-up experiments only if measured bottlenecks justify them; changing numerical behavior near VQ boundaries needs a new audit.

For the original unused design, the grouped RLS depth update is a decaying-ridge update, and its live-state count excludes training tapes. Its near-10T example is parameter arithmetic, not evidence of economical scaling. Neither component is present in the current frozen-official-codec experiment; adding them now would require a new architecture version and experiment.

**16. Implementation decisions and acceptance criteria.**

| Priority | Finding | Concrete action before expanding scope |
|---|---|---|
| High: research validity | Current decision score does not beat the tested simple heuristic; no trained nonrecurrent/action-head control | Add practical heuristics and matched neural controls; compare complete episodes |
| High: representation | Case loss and incomplete maze text prevent broad code-only claims | Keep state-conditioned claim; provide complete text for code-only tests; re-audit any new domain |
| High: measurement | Most code targets are redundant and long tasks are rare | Report action/episode metrics; construct required-length and heuristic-failure strata |
| High: mechanism | Useful stochasticity and recurrence are unproven | Same-backbone deterministic control, prior-mean/noise diagnostics, multiple seeds |
| High: future Stage 5 | Value API omits explicit remaining horizon and clear rollout-policy semantics | Define horizon-conditioned value and calibration dataset before MCTS |
| High: future Stage 7 | Adapter/GMM storage can grow without bound | Specify fixed slots/eviction/consolidation or declare growth |
| Medium: math | Multi-phase reconstruction sum differs from derived terminal ELBO | Split exact terminal objective from auxiliary deep supervision |
| Medium: math | Sleep conditional KL lacks an explicit prefix sampling measure | Label data-prefix surrogate or sample teacher prefixes for joint KL |
| Medium: numerics/reporting | Tiny negative Gaussian KL; two different accuracy estimands | Stable near-equality KL test in next version; align each CI with its reported mean |
| Medium: architecture lineage | Original per-level recurrence differs from revised once-per-block recurrence | Keep versions and cost accounting distinct |

The next experiment should separate questions with a small factorial design: **state-only vs state+NCM input; direct-action vs full NCM-code output; one-pass vs deterministic recurrence vs GRAM**. A complete factorial is expensive; first compare direct-action state-only, nonrecurrent NCM, HRM NCM, and GRAM NCM on the same verified tasks, then add the specific ablation needed to explain any gain. Use both a fixed end-to-end budget comparison and a same-backbone mechanism comparison. Favor puzzles that demonstrably require detours, rule composition, and ten-plus actions. Preserve the original validation/test split contract and create a separately versioned hard benchmark.

Promote GRAM only after a held-out advantage in episode success or useful distinct solutions at matched cost. Add global strategies only if they beat ordinary samples; add latent optimization only if it beats sampling/sham controls; add MCTS only if it beats enumeration or best-of-N under the actual simulator budget; add distillation only with a stronger verified teacher; add continual learning only after reliable base competence and a fixed capacity protocol.

**Disposition of the live run:** continue the already registered GRAM-first/HRM-second pilot under its original 48-hour deadline. This review introduces no training changes or new resource-heavy jobs. Its results will be useful as a baseline comparison even if GRAM's extra machinery does not help. Stages 3–7 are not signed off for an automatic implementation cascade. The original design documents are preserved; this dated report records the corrections and acceptance criteria.

To reproduce the new measurements from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 v15_astra_m2/.venv/bin/python v15_astra_review_20261018/review_diagnostics.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 v15_astra_m2/.venv/bin/python v15_astra_review_20261018/plot_review.py
```

Rerunning the first command refreshes the live-metric snapshot; the report above refers to the saved 12:41 IST snapshot, not future checkpoints. The structural code measurements and fixed heuristic should reproduce exactly while the dataset hashes match. The new heuristic is a review-time development diagnostic, not a baseline registered before the ongoing training run.
