# V15 Astra v3: implementation contract after the first-principles review

**Launch amendment, 18 October:** After this review implementation, the user explicitly authorized replacing the older experiment with a 24-hour v3 GRAM run. The older run stopped and checkpointed at 75,878. P0/P1 is now being activated; see `runs/20261018_24h_raw_gram/status.json` and its registered `plan.json` for actual lifecycle state and timing. The preservation/no-launch statements below describe the earlier review phase. Grounded strategy/search/continual-learning stages remain gated. The launch raises the declared raw-cache cap to 24,000 blocks, uses a 512-block seeded reconstruction audit, and adds an absolute deadline with periodic validation.

18 October 2026. This is a separate revision, not a patch to the registered 48-hour experiment. The original blueprint, `v15_astra_m2`, its split identities, source hashes, checkpoints, and deadline remain authoritative for that experiment. This document governs only `v15_astra_v3`.

## 1. Research claim and execution boundary

The next grounded experiment asks whether recurrence, stochastic transitions, and NCM representations improve **verified decisions and complete episodes at measured cost**. Lower code loss alone is insufficient. Raw-text P1 separately asks whether adjacent document blocks can be predicted and freely decoded. Neither experiment establishes a causal world model by itself.

All training, frozen-codec construction, and evaluation command entry points refuse to run while the old supervisor or trainer exists. New heavy jobs share an exclusive file lock. Output paths resolve symlinks and must stay inside this revision. CPU unit fixtures do not load the frozen codec or use MPS. There is no automatic launch, scheduler, migration, or change to the old run.

## 2. Pinned representation and information flow

The installed official LM1B NCM checkpoint has 12 levels `(1,8,16,24,32,40,52,64,76,92,108,128)`, **641 codes, 8,192 choices per level, and 128 pre-tokenizer IDs per text block**. The 4,096 vocabulary in the new Astra-R draft is incorrect for these assets. Source revision, checkpoint checksum, pre-tokenizer files, dtype, device contract, and decoder mode contribute to the cache identity. Canonical encoding stays on CPU; no fallback download is allowed.

For grid tasks, canonical JSON contains domain, dimension, every wall, current state, and goal. Modular observations contain domain, modulus, both operators, current state, and goal. Typed state remains a separately declared 88-integer observation. The serializer does not contain distances, optimal actions, future states, teacher route classes, or target lengths.

The prepared grid dataset uses four fixed 128-ID observed blocks. Its longest observation has 197 IDs, so this is conservative and includes padding. A separately versioned two-block dataset is a reasonable later efficiency ablation; the current artifact is immutable. Overflow raises an error instead of truncating. Multi-block position encoding makes block order visible.

The LM1B tokenizer loses case/accent distinctions. `Foo`/`foo` and `résumé`/`resume` still collide. Complete serialization fixes omitted observations, **not** quantization loss. The future encoding command decodes every observation and verifies its parsed state. Code-only symbolic training requires all cached observations to round-trip semantically. State+code experiments retain their typed-state qualification. Raw language prediction is allowed with measured reconstruction errors; it does not receive an exact symbolic-fidelity certificate.

## 3. Core models and matching

The default shared core uses width 192, six attention heads, FF width 432, two context layers, two output layers, 16 recurrent slots, two fast updates per slow update, and two slow updates. Code embeddings are factorized to width 64 and tied to output tables. The source paper's isotropic option remains available as an explicitly configured control; diagonal short full-BPTT GRAM is the Astra adaptation, not an exact paper reproduction.

| Arm | Trainable parameters | Deployment parameters | Purpose |
|---|---:|---:|---|
| State → seven-action policy | 837,319 | 837,319 | Practical codec-free control |
| Nonrecurrent code policy | 8,443,788 | 8,443,788 | Separately trained one-pass control |
| Same-core HRM | 9,614,220 | 9,614,220 | Deterministic recurrence with FF=432 |
| GRAM | 10,000,333 | 9,762,828 | Same core plus prior and training posterior |
| Parameter-matched HRM | 10,001,292 | 10,001,292 | FF=544 practical budget comparison |

The code arms include 6,291,456 code-table parameters. GRAM has 237,505 training-only parameters. Frozen NCM parameters are **additional**, not counted in this table. Code-policy state-only and codes-only ablations retain unused portions of the common architecture, and the ledger counts them; the direct action control has no inactive code tables. Same-core matching and total-parameter matching answer different questions. Shared equal-shaped tensors are explicitly copied from a seeded HRM reference for the GRAM arm. Data streams are independently seeded and resume from saved RNG states.

## 4. Likelihood, gradients, and recurrence

For one target block, the head defines

`log p(Y | X,h) = sum_level sum_position log p(Y_level,position | X,h,Y_previous_levels)`.

Every level, including the root, is predicted. Parallel teacher forcing uses masks that prohibit current/future target-level summaries. Generation uses previously generated summaries only. Context is encoded once per generated block; each completed output level is summarized once and cached. CPU tests compare cached and reference generation exactly and compare conditional logits numerically.

For GRAM, `q` sees the target only during training. Both `q_r` and `p_r` are evaluated on the same posterior-history state. Local corrections are Gaussian; the prior-only generation API accepts no target. The terminal loss is `(NLL_terminal + sum_r KL(q_r || p_r)) / 641` with beta exactly one. Multiple phases extend the same trajectory; they do not duplicate terminal observations. Optional intermediate reconstruction is separately named `deep_supervision_augmented`. Truncated last-transition training remains a separate surrogate and cannot silently combine with multi-phase full-ELBO settings.

Near-equality Gaussian KL uses relative scale differences, `log1p`, and a small-argument series for `expm1(2t)-2t`. It is not clamped to zero. Tests compare values and gradients against FP64. The inner training loop avoids parameter-validation synchronizations where positive scales are guaranteed by construction; finite-gradient/checkpoint guards remain active.

Prior likelihood uses a labeled finite four-sample log-mean-exp estimate. It is not exact marginal likelihood or text perplexity. Noise diagnostics report fixed-prefix logit sensitivity, state sensitivity, decoded disagreement, prior-mean performance, and extra verified optimal actions. None alone proves a useful stochastic mechanism.

## 5. Raw-text P0/P1 lane

Raw rows contain document ID, family, duplicate cluster, source, license declaration, text, and a case-sensitivity declaration. They contain no action/state/reward labels. Exact and whitespace duplicates connect families before splitting; declared near-duplicate clusters also connect families. Semantic deduplication remains a corpus-curation obligation, not something the current code claims to solve. Known exact puzzle texts are rejected; paraphrased contamination still requires a corpus audit.

Tokenize once per document with the pinned pre-tokenizer. Consecutive 128-ID chunks preserve within-document order; only the final chunk is padded. Use W=2 observed blocks to predict the next block. Cache each unique input block once and store example-to-block indices. Validate adjacency, document identity, split identity, checksums, and shape at load. The initial builder caps source JSONL at 64 MiB and unique blocks at 20,000; larger preparation needs an explicitly revised budget.

P1 trains the same concept generator with `context_source=codes`, no action branch and no global strategy. The runner uses full terminal ELBO for GRAM. Raw evaluation stores fixed panel indices, per-example prior NLL, free decoded continuations, and reference codec reconstructions. Reuse that panel for retention after compatible adaptation. A grounded-only checkpoint is not evidence of P1 pretraining. No real raw corpus has been selected, encoded, or trained in this revision.

## 6. Verified grounded data and measurement

The new grid fixture contains 72 distinct maps and 991 independently replayed transitions. Each split has eight episodes in each required-distance bucket: 5–9, 10–19, and 20+. All original topology identities are excluded. Independent forward BFS checks shortest lengths against the reverse-distance teacher. Thirteen maps use a corridor recipe; long corridors are not automatically difficult planning problems. Recipe and heuristic-failure labels are retained. Fifty-three initial states defeat the fixed legal-distance next-action heuristic.

`multiplan.py` enumerates at most 32 shortest paths per problem and certifies a second route only when two internally disjoint divergence branches enclose a wall. This conservative definition rejects cosmetic spelling changes and commuting moves in open space. It is not an exhaustive count of all homotopy classes. The resulting 11-problem fixture has only **one training problem**, seven validation problems, and three test problems. It cannot support a strategy-learning claim. D-SINGLE and D-MULTI use the same problem IDs, with one fixed route versus uniform retained-route sampling; compare equal optimizer updates, not total expanded rows.

Evaluation saves every decision, reference, complete optimal action set, group, remaining distance, legality, and paired heuristic result. Report action success and complete episodes before code agreement. Separate constant/variable code-position accuracy. Confidence intervals resample whole groups and recompute the same ratio of total successes to examples used by the headline. Equal-group averages are separately named. Paired differences use the same sampled examples or episodes. Missing length strata remain missing, not zero.

Fixed legal-distance, uniform-legal expected success, direct-action raw/legality-filtered proposals, and exact BFS are explicit controls. BFS is labeled a known-transition solver and reports its own expansion/transition counts and runtime. Code-policy legality filtering requires a separately budgeted candidate protocol; it is not silently borrowed from the action head. Validation selects development choices; test scoring is a separate command, never called by training.

## 7. Optional grounded actor, critic, and persistent strategy

`GroundedActor` adds a seven-class head to a preserved code backbone. The actor accepts observed code/state context, **past** action IDs, and an optional already-created strategy. It has no candidate or future-action input. History pooling preserves observed ordering and ignores PAD. A training-only method can use canonical target codes in the transferred GRAM posterior; its terminal likelihood is one categorical action, so **KL is not divided by 641**. This is a single-step action ELBO, not an episode-level strategy objective.

`CandidateCritic` evaluates candidate IDs after one common feature computation. It requires a `ValueContract` fixing policy, representation, rollout policy, strategy semantics, and maximum horizon. Horizon includes the candidate action; zero-horizon candidate evaluation is rejected. Exact candidate rollout records include positive and negative outcomes and simulator-call counts. Critic fitting detaches actor features. Its action table is separate from the actor's history table so fitting the critic cannot silently change the policy whose outcomes it predicts. Tied-table joint adaptation would need a new policy identity and new labels.

`StrategyRouter.begin` reads root context once, chooses a mode, and stores root-conditioned tokens with episode and policy identities. The same tokens travel through every step. Branch copies own their tensor storage; a mismatched episode/policy is rejected. The router does not receive route-class labels at inference. Mode interventions must hold task and local-noise seeds fixed and measure **successful full paths**, not hidden-vector distances.

`episode_mode_elbo` sums mode-specific step bounds over valid steps and charges `KL(q_mode || prior_mode)` **once per episode**. Responsibilities must come from an explicit variational router or independent pilot estimates. Noisy per-mode estimates cannot be optimized by an unqualified log-sum-exp shortcut. The full sequential posterior/router trainer and transfer scheduler are not implemented or activated: their experimental gate has not passed.

## 8. Later-stage corrections

* Value: `V(features, horizon_left, contract_id)` binds remaining horizon and rollout-policy semantics; nonterminal H=0 is exactly zero. Tree keys also include history and strategy context.
* Test-time optimization: the leave-one-out score estimator requires independent on-policy sampled trajectories. A frozen code-only decoder has no direct strategy score term. Greedy decoding is not a valid substitute for sampling in that estimator. No optimizer/search loop is enabled.
* Distillation: record whether prefixes came from data, teacher sampling, or student sampling. Data-prefix conditional KL is a surrogate, not full joint hierarchical KL. Teacher gradients are stopped.
* Continual learning: exactly two stored adapter/gate bundles, bounded stored tensor bytes, candidate validation, promotion and rollback. Starting a new candidate consumes the previous rollback slot. Retention and acquisition both enter acceptance. Verified training retrieval records have entry/payload-byte caps and policy versions; promotion/rollback clears stale memory. These bounds do not include optimizer tapes or Python object overhead. No full consolidation learner is claimed.

## 9. M2 execution and stage gates

Use batch 4, accumulation 2, FP32, one CPU thread, explicit step and wall limits, a 1.2-GiB available-memory stop, and an initial MPS allocator fraction of 0.5. These are conservative starting settings, **not validated v3 MPS performance measurements**. Atomic checkpoints include optimizer, CPU/MPS RNG, NumPy RNG, data and source identities, and accumulated time. Resume refuses changed experiment settings. SIGINT/SIGTERM and a run-local STOP file request a clean checkpoint. Validation tests establish exact resume on small CPU fixtures only.

Across all positions a block has 641×8,192 = 5,251,072 output logits (21,004,288 bytes if all materialized in FP32); levelwise generation has a different peak. Codec work, recurrent work, transfers, softmax, and decoder latency must be profiled separately after the live run ends. No unmeasured speedup is claimed for the summary cache.

Progression requires: P0 fidelity/provenance → P1 prior-only raw generation and controls → sufficiently large verified grounded population → matched deterministic/GRAM controls → frozen-policy calibrated critic → persistent-mode behavioral advantage. Search, test-time optimization, distillation, and continual learning remain later gated experiments. The code implements corrected primitives, not an automatic cascade through unvalidated stages.
