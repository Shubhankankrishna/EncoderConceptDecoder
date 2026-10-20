# V15 Astra Implementation Specification: Independent First-Principles Review

**Review date:** 17 October 2026  
**Target:** `V15_Astra_Engineer_Training_Spec.md`, version 1.0  
**Review basis:** user's eight-stage roadmap, supplied `Generative Recursive Reasoning` PDF, existing `V15_Astra.md` (an older self-trained-codec design), primary public NCM/GRAM/XM/GradCuit/LSL/RCDL literature and hand-audited mathematical interfaces.  
**Important scope:** the complete composite model has **not** been implemented, trained or empirically validated in this review. No official NCM weights are in this sandbox. Static consistency checks and small analytic computations **are not** integration tests with the published checkpoint.

## 1. Executive verdict

The document gives a reproducible, sufficiently detailed **build specification for starting engineering** of the eight-stage program. It has independent mathematics for key loss/gradient/search constructions, strict frozen-codec policies, explicit data schemas and software tests. It **cannot promise** that every aspect will work without modification: the most consequential open prerequisites are official tokenizer interface verification and whether NCM's pretrained codes preserve precise symbolic actions/states.

**Suggested implementation readiness:** 8/10 as an *engineering blueprint* for a research team, **not** 8/10 demonstrated scientific performance. Runtime-proven readiness remains low until the official NCM adapter, data cache and next-block free-generator pass the staged gates.

## 2. Checks actually performed

| Review layer | Result | Evidence / limitation |
|---|---|---|
| Stage enumeration and freeze plan | **PASS** | Exactly stages 0 through 7; explicit parameter trainability matrix |
| Code hierarchy arithmetic | **PASS** | 12 lengths sum to 641; offsets tested for contiguity |
| Output code probability math | **PASS (toy)** | 2-level NLL sum/per-code/level surrogate distinctly calculated |
| Gaussian KL | **PASS (toy)** | Closed form matched PyTorch distributions to ≈4.4×10^-16 |
| Categorical mixture and responsibilities | **PASS (toy)** | logsumexp categorical normalization and variational soft mode weights |
| Discrete reward-gradient identity | **PASS (toy)** | Score-function gradient matched exact derivative and finite differences |
| Terminal-reward backup | **PASS (toy)** | Single reward on 2-action path gives return 1, not 2 |
| LoRA initial base preservation | **PASS (toy)** | Zero B matrix yields exactly zero adapter change |
| GMM support direction | **PASS (toy)** | Density-ratio gate behaves as intended on constructed 1D supports |
| Root PUCT visit accounting | **PASS (toy)** | 20 counted root actions lead to 20 total root visits |
| Markdown structural static checks | **15/15 PASS** | Balanced code fences/math blocks, no nonprintable characters, sections/stages/papers, rollback and leakage warnings |
| End-to-end official NCM checkpoint | **NOT RUN** | Checkpoint/source checkout not present |
| Actual HRM/GRAM model gradients/masking | **NOT RUN** | Architecture is specified; not implemented in this deliverable |
| Actual verifier-grounded MCTS improvement | **NOT RUN** | Only mathematical mini-checks, no GRAM policy training |
| Search/CL effectiveness | **NOT RUN** | Cannot infer from architecture alone |

`V15_Astra_Mathematical_Reference_Checks.py` implements the nine mathematical toy checks. `V15_Astra_Math_Checks_Result.json` records actual run output. `V15_Astra_Static_Review.json` records mechanical specification checks.

## 3. First-principles review pass 1: Is the representation contract sound?

**Finding:** **Conditionally yes.** Using official NCM's frozen tokenizer is conceptually coherent and saves the cost of retraining a large codec. But the generator's ability to reason over exact symbolic states is **upper bounded by codec fidelity**. A root-level code alone may not preserve the digits/identifiers needed by a planner; the complete hierarchy must reconstruct critical fields. Text→codec→decoded text can be lossy, and two distinct actions may collide into the same code hierarchy.

**Strong design decisions:** all 12 NCM code levels kept; tokenizer bundle/version hashed; code cache compressed as uint16; task group split before block slicing; state/action verification outside learned decoder; explicit BPE padding and action-to-block-span metadata; no unverified released API functions hard-coded into normative implementation.

**Blocking source verification:** Real TinyStories/LM1B configs and code paths must be loaded and tested with the pinned official repository; `BPE text IDs`, `PAD/EOS` and partial decoding are not guaranteed by the generic `FrozenNCMCodec` pseudocode. `D_NCM(Q_NCM(X))` exact-numeric audit must precede generator training.

**Outcome:** Stage 0 may reject the TinyStories pretrained tokenizer for precise mathematical language. That is a valid outcome; use an alternative released pretrained tokenizer or an explicit current-state side channel rather than pretending codec reconstruction is lossless.

## 4. Pass 2: Are predictive losses and GRAM's ELBO mathematically consistent?

**Findings:** Properly differentiated. Stage 1 defines summed code NLL for a normalized level-parallel conditional categorical distribution. It distinguishes a level-balanced training surrogate from proper likelihood. Stage 2 distinguishes full trajectory ELBO from GRAM Eq. (14)'s last-transition-only surrogate, explains diagonal-vs-paper-isotropic Gaussian and conditions the training posterior on target codes but forbids target access to the prior.

Stage 3's strategy modes have a genuinely normalized **formal** mixture `p(Y|X)=Σπ_k p_k(Y|X)`, with GRAM trajectories inside each `p_k`. The document then uses variational per-mode lower bounds, derives an E-step responsibility formula and explicitly warns that inserting a noisy one-sample `B_k` into logsumexp is not automatically a strict likelihood bound. Hard winner-only XM is called a **different** objective.

**Residual risk:** Full softmax code prediction involves 641 categorical sites of 4,096 classes, making log-likelihood optimization heavy. A low teacher-forced NLL can coexist with poor free running because sampling an incorrect early code changes later code distributions. Evaluate free generated block success and per-horizon closed-loop success before spending more compute.

**Engineer recommendation:** Build a tiny 2-level output-head test first and overfit 32 training examples. Then test NCM's full 12-level masked factorization before GRAM. Confirm the first/root target level is learned, not supplied from target ground truth.

## 5. Pass 3: Is GradCuit-style steering a valid differentiable mechanism?

**Finding:** The proposed estimator is mathematically plausible for **discrete sampled code actions** if the GRAM prior can be reparameterized with input-independent `xi` and all **selected logprobs** remain differentiable w.r.t. continuous `z`. In that case the expected reward gradient is a score-function estimator. The specification includes leave-one-out (not sample-mean) baselines, finite-difference tests, a bounded trust region, weight freezing and equal-budget sham/random controls.

**Not guaranteed:** The published GradCuit paper was validated using continuous latents inserted at intermediate Transformer layers and reward-weighted token-logprob gradients. Our GRAM + VQ-code reinterpretation changes both latent structure and discrete outputs. Call it **GradCuit-inspired** until a faithful paper-equivalent comparison is implemented. Reward must be independently verified; zero-success batches carry little or no gradient signal.

**Code-review change applied:** Removed a redundant baseline calculation in illustrative pseudocode, explicitly added anchor/clipping terms, and kept sampled code ID differentiation separate from logprob differentiation.

## 6. Pass 4: Is MCTS a real search tree or only nested sampling?

**Finding:** It becomes genuine tree search **only** when parsed actions are canonically defined, externally verified transitions update the state, and PUCT visits/values are backed up consistently. A future NCM text block by itself is neither a legal action nor an authoritative state transition.

**Strong design decisions:** MCTS tree edges are complete actions; 12 NCM levels are only the way to *emit* a block. The action prior is an explicitly **candidate-conditional empirical estimate** from sampled legal actions, not falsely claimed exact marginal likelihood. Node keys include remaining horizon/environment version; terminal edge reward counts exactly once; benchmark includes equal-wall-time best-of-N.

**Code-review change applied:** Replaced a flawed pedagogical root-expansion loop with a normative pre-expansion loop. Each counted nonterminal-root simulation now increments one root edge, with an explicit root-visit assertion.

**Remaining ambiguity for actual code:** `value(node.state,node.history)` must either internally recompute a policy/state encoding/latent summary in deterministic prescribed fashion, or the `ValueModel` API must accept `h,z` returned from a specified prior sample. Choose ONE and pin it, otherwise value targets train on a distribution different from MCTS evaluations.

## 7. Pass 5: Can CL genuinely acquire and retain knowledge?

**Finding:** The CL design is falsifiable and reversible but not proven. The specification correctly distinguishes LSL's activation-support gating from ordinary LoRA, separates a simple retrieval correction from RCDL's functional-gradient learning rules, and keeps the pretrained NCM codec frozen initially. Sleep/consolidation is promoted only after both new-task acquisition and old-task retention pass.

**Major limitation:** Support distributions overlap; a gate that closes on new inputs can retain old tasks while failing to learn. RCDL's growing memory and even bounded normal retrieval cost storage and latency. Sequential evaluation must compare to LoRA+replay and full fine-tune+replay, with an accuracy matrix and learning speed on novel tasks. Good retention by itself is not sufficient.

## 8. Pass 6: Systems/scale plausibility

- `12×4096×128=6,291,456` scalars for tied level code tables only. In FP32 with grad and 2 Adam moments they use about **100.66 MB**, before HRM, frozen tokenizer and activations.
- One complete 641×4096 FP32 output-logit tensor per example is **10.50 MB**. At batch 2 and K=4 candidates, naïvely materialized logits are about **84 MB**, before autograd tape.
- One million frozen-code blocks stored as 641 uint16 IDs need **1.282 GB decimal** plus metadata.
- More GRAM recursions/modes, GradCuit gradients and MCTS simulations can make inference slower despite use of NCM's pretrained tokenizer. The source NCM generator's published speedups do not carry over automatically.
- A 30-CPU-minute run per arm is a **plumbing and throughput pilot**, not evidence of mature generalization. Run multiple seeds at appropriately scaled training budgets before research claims.

## 9. Defect log from review and fixes implemented

| ID | Finding | Severity | Disposition |
|---|---|---:|---|
| D01 | Old V15 spec's new 4-level self-trained codec conflicts with requested frozen official NCM | Critical | **Fixed:** entire program uses published pretrained 12-level codec; old spec explicitly superseded |
| D02 | Risk of treating NCM level depth as MCTS time | Critical | **Fixed:** independent `t/b/ell/r` clocks, MCTS action edges |
| D03 | Predicting target root omitted when borrowing NCM generator settings | Critical | **Fixed:** first target level explicitly predicted; no target root leakage |
| D04 | Confusing exact GRAM ELBO with last-step truncated surrogate | High | **Fixed:** full and truncated BPTT separate methods and tests |
| D05 | Using hard XM mode selection as exact mixture log-likelihood | High | **Fixed:** variational mixture derived and hard alternative labeled surrogate |
| D06 | Reward gradient through sampled discrete codes or exact verifier | High | **Fixed:** score-function selected logprob, fixed noise, baseline and finite differences |
| D07 | Optimizing z code contained redundant baseline calculation | Medium | **Fixed:** single leave-one-out advantage expression, explicit anchor/clipping |
| D08 | Root MCTS simulation expansion skipped visit backup | High | **Fixed:** pre-expand root and assert visits == simulations |
| D09 | Terminal reward could be counted twice in search | Critical | **Fixed:** terminal incoming-edge reward with zero terminal bootstrap |
| D10 | Conflating canonical action code likelihood with total action probability | High | **Fixed:** explicitly intractable marginal, empirical proposal priors and canonical distillation surrogate |
| D11 | Distillation action serialization may span multiple blocks | High | **Fixed:** product/sum over canonical action block likelihoods added |
| D12 | Claimed faithful RCDL for arbitrary normalized retrieval | High | **Fixed:** clear baseline retrieval versus RCDL functional-gradient modules |
| D13 | New-task acquisition not checked for gated adapters | High | **Fixed:** both acquisition+retention and sleep rollback gates |
| D14 | Lack of verified source API execution in current sandbox | Critical (runtime readiness) | **Open:** official checkpoint+matching repo must be supplied and tested |
| D15 | Extrapolation to real-world agent competence | High (research claims) | **Bounded:** exact simulator tasks first, no extrapolation from text accuracy |

## 10. Open implementation decisions that require actual empirical evidence

1. **Tokenizer choice:** TinyStories versus another official pretrained NCM tokenizer based on critical-field reconstruction. Do not mix code IDs across checkpoints.
2. **State-conditioned versus code-only policy:** Is current verified structured state a model input? If yes, evaluate separately from pure NCM-code prediction.
3. **Action-to-code collisions:** If legal actions lose identity under the frozen codec, either change serialization, choose a different released codec or add an explicit separate action head.
4. **HRM context compression:** 12 pooled vectors per past block may lose exact digits; compare against raw-code cross-attention.
5. **GRAM Gaussian parameterization:** Paper's isotropic Gaussian versus proposed diagonal noise. Register both.
6. **Recursion gradient schedule:** Full vs faithful last-transition-only truncated GRAM; compare gradient fidelity on tiny unroll.
7. **Root-level generation:** Confirm conditional first-code prediction and PAD/EOS handling in the new output head.
8. **Value representation:** Context-only or `(state,h,z)` with precise sampling and caching semantics; avoid train/deploy distribution mismatch.
9. **Search priors:** Empirical sampled-action frequencies versus trained canonical action head if sample inefficiency is high.
10. **Continual CL support estimation:** Frozen-base activations vs LSL-style refit on adapted activations, and support gate calibration across phases.

## 11. Final sign-off / next engineering instruction

**Sign off for Stage-0 implementation:** Yes, provided the team treats this as an engineering specification with **explicit acceptance tests**, not as functioning code or a guaranteed training recipe. **Sign off for Stage-1/2 research after Stage-0 gate:** Conditional. **Stage-3 through Stage-7 production sign-off:** Not before earlier stage gates pass and source API details are validated.

**Execute in order:** official NCM checksum/API verification → exact text/state reconstruction audit → canonical verified code dataset → deterministic HRM baseline → GRAM prior/posterior, prior-only free evaluation → `z` diversification → GradCuit-style latent search → flat MCTS/value → verified search distillation → reversible continual learning.
