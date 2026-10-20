# V15 Astra — First-Principles Review

**Reviewed:** [V15_Astra.md](V15_Astra.md), revision 17 October 2026. **Reviewer:** the same assistant, performing a separate second pass after drafting; this is not an external or blinded peer review. **Scope:** mathematical coherence, architecture interfaces, evidence, costs and experiment design. No Astra training or implementation results exist.

**Verdict: 8.0/10 as a research specification.** It is coherent enough to implement through gated experiments. It is not evidence that the architecture beats a Transformer, that latent search works at 10M parameters, or that the proposed system scales successfully to 10T.

The most valuable change is the actual generative path: frozen codec → predicted code hierarchy → retained frozen decoder. This removes the ambiguity of older designs in which a concept forecast could be optimized yet contribute little to token generation. The price is that codec fidelity now becomes a hard prerequisite.

## 1. Review rubric

Scores are subjective, with equal weights; they assess the written design rather than model quality.

| Dimension | Score /10 | Reason |
|---|---:|---|
| Evidence and attribution | 9 | All nine supplied papers are mapped to concrete choices; adaptations and historical limitations are disclosed |
| Information and representation | 8 | High-rate codec and exact-block gates address details; small-codec fidelity remains unproven |
| Probability and causal interfaces | 9 | Complete hierarchy mixture and decoder define a coherent model; likelihood bound and generation metrics are distinguished |
| Optimization and gradient specification | 7 | Discrete search estimator and truncation are explicit, but credit assignment and estimator variance remain serious empirical risks |
| Systems and cost realism | 7 | Complete parameter/state counts and branching costs are provided; no actual kernels, memory or latency measurements exist |
| Continual learning and sleep | 8 | Bounded state, verified targets, replay, rollback and empty-memory evaluation are specified; capacity and interference remain unresolved |
| Scaling | 8 | Dimension rules and a concrete sparse near-10T ledger are consistent; this is not a demonstrated training plan |
| Experimental falsifiability | 8 | Strong controls and stop gates; the total experiment program is substantial and needs actual data/budget registration |
| **Mean** | **8.0** | **64/80** |

## 2. Does the proposed information flow make sense?

Yes. Stage 1A learns a stable representation and its decoder. Stage 1B models a distribution over that representation conditioned on observed text. Its HRM module computes that distribution iteratively. At inference, generated coarse codes constrain finer codes, and the decoder converts the resulting complete representation into text.

This arrangement is closer to the NCM mechanism than training an auxiliary future-code head beside a powerful token path. The decoder cannot bypass the codes, so the forecast has a necessary operational role. However, necessity is not usefulness relative to a simpler model. A trained Transformer prior over the same codec is required to establish whether HRM improves the result.

The chosen hierarchy allocates 216 nominal bits to 16 tokens from a 2,048-symbol alphabet, compared with 176 fixed-width raw bits. This is an intentional fidelity-first configuration. Describing it as storage compression would be wrong. It can still provide a shorter sequential generation schedule, but HRM can consume that advantage through repeated computation.

The codec's fine level has 16 slots and 1,024 choices per slot. Earlier levels contribute shared coarse information. The total alphabet capacity is large enough to avoid an obvious counting impossibility, but the residual quantizer's structure may still make exact reconstruction difficult. In particular, mean pooling, repetition-based upsampling and a small decoder may struggle with rare entities, arbitrary identifiers and long digit strings. The first substantial experiment should therefore be reconstruction, not an elaborate sleep pipeline.

**Required result:** high held-out exact-block fidelity by domain and exact-task symbol type. If it fails, change rate/capacity or training. Do not compensate by training the prior longer against already lossy labels.

## 3. Is the probabilistic model coherent?

The main specification defines each mode as a product of normalized categorical conditionals, then mixes complete hierarchy probabilities. This is correct. The outside code-count normalization changes the optimization scale without changing the probability represented inside the logarithm.

The mixture must remain global across levels for one block. Mixing root and fine heads with independent prior weights generally produces a different model. Conditioning on already generated coarse codes changes the posterior probability of a mode. The analytical checker verifies normalization and posterior conditioning in a tiny enumerated example.

The frozen decoder produces a normalized conditional token-block distribution. Summing over code hierarchies therefore defines a text-block distribution. Evaluating only the canonical encoding gives a lower bound on log probability and consequently an **upper bound on negative log-likelihood**. It is not exact token-autoregressive perplexity. The second pass corrected the less precise phrase “text likelihood upper bound” to avoid reversing the quantity being bounded.

Greedy decoder outputs and sampled decoder outputs have different induced distributions. Astra acknowledges this rather than assigning the model likelihood to every decoding heuristic. Search over sampled code trajectories followed by a deterministic decoder has a valid score-function gradient; it does not need a derivative through decoder argmax.

**Residual risk:** teacher-forced likelihood can improve while generated fine levels visit unfamiliar coarse prefixes. The current clean objective does not solve exposure shift. This is appropriately tested before adding a separately labeled denoising or retargeting algorithm.

## 4. Is future information kept out of prediction?

The written contracts are consistent:

- E(x) and current/future code labels are training targets only.
- A level sees the observed prefix, its query positions, the exploration input and earlier levels.
- Bidirectional attention among query states is legal because those states contain no current-level ground-truth codes.
- D sees code-derived vectors, not raw target text or unquantized h.
- True answer length is not a predictor input; EOS/PAD are predicted outcomes.
- Depth states reset for each codec invocation; H/U reset for each level/block.

This is particularly important when batching levels or packing documents: an implementation can violate these contracts even if every individual module looks sensible. The design correctly lists adversarial future-perturbation and packing tests as unfinished implementation requirements.

The context stem's causal attention enables some cache reuse, but window truncation can invalidate equality with a recomputed reference. The document explicitly chooses recomputation as the correctness baseline rather than promising a fast cache with unspecified semantics.

## 5. Does early z receive a useful gradient?

A path exists in the written computation. The mode embedding enters before recurrence and remains in X_l, which is reinjected into each fast update. Thus truncating earlier recurrent states does not automatically disconnect every route to the mode. This fixes an important weakness of a one-time initial latent followed by an early detach.

A live gradient is not evidence of useful specialization. The model can learn to ignore e_k; modes can become redundant; or one mode can dominate training responsibilities. Uniform mode priors remove selector complexity but do not prevent collapse. Independent mode initialization breaks exact initialization symmetry, while behavioral diversity and matched sampling tests are still required.

Truncated training computes a biased approximation to the full loss gradient. An exact finite-mixture forward value does not remove that bias. Full-gradient short runs and gradient reachability diagnostics are essential before drawing conclusions about recurrence or modes.

The search procedure uses actual sampled code-action log probabilities. Re-encoding the resulting text to canonical codes before the search backward pass would score different actions and invalidate the estimator. The document separates those roles: sampled actions for search, canonical codes for supervised sleep.

**Largest search risk:** sparse or noisy rewards and long code trajectories can make a theoretically valid score estimator too noisy to help. Norm clipping and a trust region constrain updates but do not create a useful direction. Ordinary sampling, random perturbations and extra recurrence are strong necessary controls. A fallible self-verifier can also reinforce mistakes.

## 6. Does the memory system have clear semantics?

Yes, with important capacity limits.

| State | Lifetime | Meaning | Main failure mode |
|---|---|---|---|
| Codec Depth Memory M/P | One E/D invocation | Reuse information across sublayer depth | Numerical instability or cost without benefit |
| HRM H/U | One level prediction | Iterative computation | Redundant passes, unstable or truncated credit |
| δ | One request/candidate | Search control | Untrained neighborhood, reward overfitting |
| Window memory M/S | Declared stream until reset | Reuse successful offsets from related prefixes | Interference, weak keys, stale offsets |
| Accepted LoRA | Across validated promotions | Durable adaptation | Forgetting, capacity saturation |
| Replay records | Bounded retention policy | Verified evidence for later learning | Coverage gaps and eviction of rare skills |

The fast memory's fixed feature coordinates are useful, but semantic adequacy is uncertain. Mean/final frozen token embeddings projected to sixteen dimensions can alias different tasks. Polynomial expansion does not restore information already lost in that projection. The baseline should therefore be compared with bounded nearest-neighbor retrieval and with no memory.

The second pass tightened a related issue: the memory key is the original observed request prefix and mode, not the generated answer. Using answer information in a later key construction would create a training/deployment mismatch.

Another distinction is correctly preserved: adapter updates can change which offsets work even when the key encoder stays frozen. Resetting fast memory after promotion is conservative and coherent. Its benefit must survive through the learned adapter and replay, not through accidentally retained state.

## 7. Is sleep more than replay with a new name?

The design gives it a testable additional function: imitate useful behavior produced by a fast, expensive teacher so a zero-offset, empty-memory student can reproduce it more cheaply. It uses verified canonical targets, teacher/student conditional distributions and some student-generated histories. That goes beyond merely retaining a buffer.

It also honestly differs from the full Sleep paper. Two adapter slots do not implement an indefinitely growing continuum memory system or its learned curriculum-generation policy. Only one slot is deployed at a time, and the candidate replaces rather than accumulates onto the accepted slot. This bounds capacity but eventually forces tradeoffs.

The teacher must be a frozen snapshot. A student's changing weights cannot also serve as its stationary target. The review clarified that a recorded successful δ applies to its winning mode; other modes use the unshifted teacher. Applying one mode's successful offset indiscriminately to all modes would fabricate an unverified target policy.

The old-data anchor uses training/replay contexts, separate from validation. The draft's phrase “held-out-for-optimization” was removed because optimizing an anchor on those contexts means they are not held out from optimization. Actual promotion panels remain disjoint.

Finally, the 80%-amortization gate is meaningful only when teacher improvement is positive and measured with uncertainty. If search has no demonstrated gain, there is no gain to amortize. This qualification was added during review.

**Residual risk:** a small replay buffer and KL anchor cannot guarantee global retention. Acquisition, related-skill forgetting and the full task matrix remain the outcome measures. V14 already showed why low forgetting alone can reward a model that learns too little.

## 8. Are the counts and scaling claims credible?

The supplied ledger counts token embeddings, encoder/decoder blocks, codebooks, reconstruction maps, depth projections, context blocks, self/cross-attention, FFNs, mode/position inputs, code heads and both adapter slots. It includes EMA codebooks even though they are not gradient-trained. This is much more meaningful than reporting only dense trainable prior parameters.

The two prototype totals are **9,679,106** and **7,553,282**. Optional fast-memory state is additional, as are optimizer state and activations. The 0.2988-MiB memory figure refers only to one FP32 matrix and its momentum, not the whole model or even the complete peak memory-update footprint.

For the illustrative sparse shape, approximately 9.991T stored base parameters and 172.78B selected-expert accounting can both be correct because most experts are inactive at a particular position/call. The latter is not the union of weights touched by a whole batch or repeated trajectory. The ledger field was renamed to avoid implying a global “active unique” upper bound.

The near-10T configuration is therefore a dimensional feasibility example. It does not address all necessary engineering: hardware allocation, communication efficiency, stable training schedules, sufficient data, checkpoint reliability, expert routing or useful recurrence depth. Additional search projections/adapters must be counted if enabled. The document discloses these limits.

**Principal systems concern:** four code levels × eight recurrent calls × exploration branches may erase the intended generation speed advantage. Shared context projections help but do not remove the main recurrent arithmetic. Only measured quality-versus-cost curves can decide whether the combination is worthwhile.

## 9. What was checked, and what was not

[check_design.py](v15_astra_design/check_design.py) verifies seven analytical properties:

1. A tiny hierarchy mixture sums to one and conditions mode weights correctly.
2. Responsibility-weighted gradients match a scalar finite-difference check.
3. Scalar RLS with forgetting matches a direct decaying-ridge solution.
4. The quadratic feature map implements its stated polynomial kernel.
5. A discrete-action score-function gradient matches a simple exact expectation derivative.
6. Both prototype parameter totals lie in the requested 7–10M range.
7. The specified codec payload is 216 bits per block.

These are useful algebraic checks, not neural-model tests. They do not validate matrix RLS in mixed precision, a concrete HRM implementation, data masks, exact resumption, learning dynamics, search effectiveness, latency, memory peaks or any benchmark score. Those requirements remain explicit in the main document.

The literature sources are fingerprinted and links in the new documents are checked for local existence. Existing architecture/experiment files are preserved. No training processes were started or interrupted for this review.

## 10. Recommendation

Proceed with the defined experiment sequence, starting with codec fidelity and a simple next-concept prior. Keep the full architecture as the research target, but require each later component to beat its simpler control before paying for a larger combined run.

The user's central reasoning is good: a pretrained latent vocabulary makes HRM a plausible next-concept generator, and the same latent control interface can support inference search and later consolidation. The unsupported leap would be assuming that those individually plausible mechanisms automatically yield efficient reasoning, durable continual learning or trillion-parameter scaling. Astra now states the experiments needed to bridge that gap.
