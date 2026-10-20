# First-principles review of the Astra-R v0.2 proposal

Reviewed source: `V15_Astra_MultiStrategy_Dataset_GRAM_ActionEncoder_Z_Spec (1).md`, all 985 lines. Date: 18 October 2026. The source document is preserved. This review describes the implementation decisions in `v15_astra_v3`; it does not reinterpret the live GRAM pilot as an Astra-R experiment.

**Verdict:** The separation between raw prediction and verified action learning is sound. The action/candidate firewall and persistent-mode hypothesis are worth implementing as testable interfaces. The full architecture is not yet justified as a training recipe. Correct the checkpoint dimensions, keep objective units explicit, require a substantially larger multi-plan training set, and use empirical gates before coupling the modules.

## 1. Start with what the data actually tells us

A raw document supplies an ordered sequence. It does not identify the executable action set, physical state, causal transition, or reward. Therefore P1 can legitimately predict the next concept block; it cannot supervise an action encoder from narration alone. The draft correctly rejects that shortcut.

**Implemented:** `raw.py` requires provenance and document/duplicate-family identities, merges exact/whitespace duplicate families before splitting, creates causal within-document windows, and stores unique code blocks separately from window indices. It rejects state/action/reward fields and declared case-sensitive corpora. The raw runner receives `x,y` only. Known exact puzzle texts are excluded. Corpus-wide near-duplicate discovery and paraphrased benchmark contamination remain upstream curation work; the code does not claim to solve them.

**Qualification:** P1 is appropriate for the proposed general text foundation, but it is not mathematically necessary for a small structured planner. Keep from-scratch controls to test whether the pretraining expenditure helps. No real raw corpus was chosen or downloaded here. The prepared raw CLI is not evidence that P0 or P1 has completed.

## 2. Check the representation against the actual checkpoint

Sections 2.4, 4, 5.2 and 9 use 4,096 classes. The pinned local LM1B configuration has **8,192 at every one of the 12 levels**. The index range must be 0–8,191. The 641-position output represents 5,251,072 scalar logits across a block, not 2,625,536. Full simultaneous FP32 materialization would be about 21.0 MB per example; that arithmetic is not a measured allocation peak. The original audited generator has width 192, while the new draft's width 128 is a proposed new profile, not its existing width.

**Implemented:** keep the validated ~10M, width-192 profile; validate model/cache hierarchy compatibility; pin the complete codec identity. The new normative specification corrects the dimensions. A width-128 variant would need its own parameter ledger and matched baseline.

The official pre-tokenizer still collapses case/accent distinctions. Complete wall text removes one information-loss source but does not establish a lossless neural codec. Full observation round-trip tests are built into future cache construction. Code-only symbolic experiments cannot pass the gate merely because all seven action phrases decode correctly.

## 3. Decide which objective is being optimized

The terminal trajectory ELBO is appropriate when the likelihood and all transition KL terms use consistent units and matching posterior histories. Intermediate reconstruction is extra supervision. The draft now states that distinction correctly.

**Implemented:** stable Gaussian KL tested against FP64; a single terminal reconstruction across multiple phases; separately labeled auxiliary supervision; removable posterior; target-free public generation; cached preceding-level summaries with parity tests. For a direct seven-action likelihood, use `action CE + trajectory KL` in action units. Dividing that KL by 641 would silently change the objective. The grounded branch implements a **single-step** categorical action ELBO using the transferred target-code posterior as a training-only inference network.

An episode likelihood additionally requires a complete sequential factorization. `episode_mode_elbo` provides the tested reduction, not a fictitious finished trainer: sum each mode's valid step bounds and charge the router KL once per episode. Responsibilities derived from the same noisy likelihood sample do not automatically yield the claimed bound. A proper episode-conditioned variational router and teacher-history construction remain future work.

## 4. Ask whether the dataset can identify a strategy

Many problems with one target each do not demonstrate multiple solutions for the same problem. Conversely, two successful action strings may differ only by commuting independent moves. A strategy claim needs meaningful alternative routes, exact replay, identical underlying task populations, and equal exposure budgets.

**Implemented:** bounded shortest-path enumeration; replay verification; conservative route certification using disjoint branches around a wall; paired D-SINGLE/D-MULTI references; uniform-problem, then plan, then step sampling. Labels are returned separately from actor inputs. The stored problem population is identical between panels. Long-path fixtures preserve original topology splits and never replace the old data.

**Observed limit:** only 11 of the prepared 72 mazes pass this strict two-route test. Their split counts are 1 train / 7 validation / 3 test. Six enumerations hit the 32-path cap, so the audit does not establish uniqueness for rejected tasks. This is a useful test fixture and a failed readiness gate for strategy training. Do not train B3 on one problem and call it generalization. A future dedicated generator must satisfy per-split route-class quotas and include rotated/reflected counterparts in the same family group. Corridor length alone must not be treated as branching complexity.

## 5. Separate selecting an action from assessing a candidate

An actor predicting action `a_t` cannot receive that same teacher action. A critic predicting the outcome of candidate `a` must receive it. Past actions are available information. In a fully observed Markov maze, history is not required by the environment dynamics; its benefit must be measured rather than assumed.

**Implemented:** a `GroundedActor` interface with observed state/context/history only, an optional seven-ID history embedding, and a vectorized `CandidateCritic` operating after shared actor features. Tests reject future-action arguments and show that legitimate history can affect the actor. The generic code head stays in the backbone so raw prediction can be measured after adaptation.

The initial critic has a separate candidate table. Sharing a trainable table with an otherwise frozen actor would change that actor's policy during critic fitting, invalidating its supposedly frozen continuation labels. Shared-table joint adaptation is possible later, but must create a new policy version and refresh calibration data. A lookup table is sufficient for seven fixed actions; no claim of compositional semantics follows from it.

## 6. Make the critic's target identifiable

`Q(s,a,H)` is not a unique number until the continuation policy is specified. H includes the candidate step. If H=1, success is exactly whether that step reaches the goal; there is no remaining rollout. Illegal moves, legal-but-losing moves, and successful alternatives need different labels.

**Implemented:** policy/representation/rollout/strategy contracts, explicit integer horizons, candidate replay with transition-call accounting, binary-outcome fitting with detached actor features, and tests for positive/negative candidates. The scalar value head separately defines nonterminal H=0 as zero. A critic trained only on successful teacher actions is not a calibrated policy value function; the future dataset must include actual failed continuation outcomes. We have not generated or fitted a real critic calibration corpus.

For these tiny known environments, enumerate legal actions and compare exact BFS/heuristic decisions. A learned critic cannot be credited for information supplied by a free oracle. Value accuracy and episode success per total cost both matter.

## 7. Test global z as a persistent intervention

GRAM's local random correction, an action embedding, and an episode strategy can have different roles. Naming three tensors does not ensure the model uses them differently. Multi-plan supervision also does not force a global mode to become useful: history or local noise may already explain the alternatives.

**Implemented:** root-only mode selection, fixed root-conditioned tokens, episode/policy identity checks, independent branch copies, and gradient-bearing conditioning in the grounded actor. There is no actor API that silently resamples the root strategy at each action. Tests establish lifecycle correctness, not learned route control.

**Required experiment:** hold task and local-noise draws fixed, switch the mode, measure complete successful path classes; compare learned, shuffled and random modes against the same number of ordinary GRAM samples. Mode labels may permute across seeds. A change in first action, logit, or hidden-vector distance is insufficient. A strategy whose benefit disappears under matched verifier/time budgets should be removed.

## 8. Preserve general prediction when transferring

The draft's Option A is preferable for the stated objective: retain a generic concept generator and add grounded functionality when verified labels exist. The base checkpoint, codec identity and raw validation panel need preservation so forgetting is observable.

**Implemented:** `evaluate_raw.py` stores free continuations and per-example prior likelihood on fixed indices, and can reuse the same corpus panel for a compatible later backbone. The grounded wrapper preserves the code generator. The raw and grounded training identities are distinct. These are the necessary interfaces; no transfer checkpoint or retention result exists yet. An orchestrated transfer trainer is intentionally not presented as complete.

## 9. Apply the M2 budget to the whole system

The frozen codec is much larger than the ~10M trainable generator. NCM saves sequential generation stages in its tested setting, but its full code head can be wasteful for seven actions. Caching does not make preprocessing or decoding free. The direct action policy is an important system baseline.

**Implemented:** mmap uint16 caches, deduplicated raw blocks, one CPU thread, guarded FP32 MPS settings, explicit step/wall caps, atomic resumable checkpoints, same-core and parameter-matched controls, and an exclusive heavy-job lock. Unit tests run only tiny CPU fixtures. No new MPS profiling, codec encoding, or production training was started. The existing source/data/plan hashes and live process identities are checked separately.

## 10. Adoption decision

Adopt now: corrected representation dimensions; complete observations; raw-data causal boundaries; terminal ELBO and measurement fixes; verified multi-plan fixture tools; actor/candidate separation; horizon/policy-conditioned critic interfaces; persistent strategy lifecycle; bounded future-stage primitives.

Do not promote yet: a completed raw foundation, useful strategy modes, critic-guided planning, full episode-mixture training, GradCuit, MCTS, search distillation, or continual-learning gains. The prerequisite datasets, calibrated checkpoints and matched comparisons do not exist. This is a research-readiness boundary, not an interruption of the live run.

Primary source checks: [GRAM v4](https://arxiv.org/abs/2605.19376v4) motivates stochastic recursive trajectories and variational training; [NCM v1](https://arxiv.org/abs/2610.14162v1) and its [official source](https://github.com/HazyResearch/ncm) motivate the frozen multiscale-code interface. Neither validates this combined action/critic/global-strategy architecture. Checkpoint-specific engineering facts above come from the pinned local assets, not from generic paper claims.
