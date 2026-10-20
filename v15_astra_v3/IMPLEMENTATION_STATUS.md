# Review-to-implementation traceability

Review-phase status: implemented and CPU-tested infrastructure; no empirical model result at the time of that review.

**18 October launch amendment:** The user has since authorized a 24-hour raw-text GRAM run, replacing the old experiment. See `runs/20261018_24h_raw_gram/status.json` for current state and `reports/tests_launch_20261018.log` plus `reports/preflight_raw_20261018/report.json` for launch validation. The added production path includes resumable batch preparation, an absolute deadline, periodic prior-only validation and resource checks. Grounded critic/strategy/search stages remain disabled.

| Review issue | Implementation | Evidence / limit |
|---|---|---|
| Preserve live experiment | `isolation.py`, all heavy CLI guards, output confinement | Before/after hashes and PID/progress verification; no live-file patch |
| Incomplete maze text | `representation.py`, `audit.py`, `dataset.py` | Every wall included; 991 transitions verified; no truncation |
| Tokenizer collisions | Domain acceptance contract; pinned local tokenizer audit | Case/accent collisions reproduced; unsupported fidelity claims blocked |
| Code targets are redundant | `metrics.py`, `evaluate.py`, direct-action control | Variable/constant accuracy separated; decision/episode metrics lead |
| Weak controls | `controls.py`, `baselines.py` | Real nonrecurrent module; same-core and parameter-matched arms; legal heuristic/BFS |
| Unproven stochastic utility | `diagnostics.py`, prior-mean and second-sample decision records | Sensitivity tests; no claimed performance gain |
| Tiny negative KL | `objectives.gaussian_kl` | FP32 values and gradients checked against FP64 near equality |
| Multi-phase ELBO mismatch | `gram.forward(phases, auxiliary_weight)` | One terminal NLL; auxiliary reconstruction separately named |
| Wrong CI estimand | `metrics.cluster_interval`, paired differences | Tests distinguish example weighting from equal-group weighting |
| Missing long-task strata | `benchmark.py` | 72 group-disjoint maps, balanced required-distance panels |
| Recomputed prior-level summaries | `model.generate_from_state` | Cached/reference output parity; one summary per completed level |
| Unclear raw supervision | `raw.py`, raw mode in `train.py`, `evaluate_raw.py` | No action labels; within-document windows; synthetic unit cache only |
| 4096 vs actual 8192 | Pinned codec/config checks and revised specification | Actual parameter ledger; no weights/codec mutation |
| Single-reference route bias | `multiplan.py`, `PlanSampler` | Matched problem IDs; replayed obstacle-separated routes; one train problem is insufficient |
| Actor reads future candidate | `grounding.GroundedActor`, separate critic API | Future-action argument rejected; past history supported |
| Critic lacks horizon/policy | `CandidateCritic`, `candidate_outcome`, `value.py` | Explicit contracts and positive/negative unit rollouts; no fitted critic |
| Global z resets per action | `EpisodeStrategy`, `StrategyRouter` | Identity checks and independent branch storage; no trained mode benefit |
| Per-step router KL overcount | `objectives.episode_mode_elbo` | Sum time, charge router KL once; full router trainer still gated |
| No raw retention lane | `evaluate_raw.py` | Fixed indices and free text preserved; no actual P1/transfer checkpoint |
| Incorrect score baseline | `objectives.reward_score_proxy` | Exact Bernoulli enumeration of leave-one-out estimator gradient |
| Distillation prefix ambiguity | `conditional_distillation` | Prefix measure required, teacher gradient stopped |
| Unbounded adapters/GMM | `continual.AdapterSlots` | Exactly two bundles, stored tensor-byte cap, promotion/rollback/checkpoint tests |
| Stale or unbounded retrieval | `BoundedVerifiedMemory` | Train-only verified records, payload-byte/entry caps, policy-version invalidation |
| Unsafe resumability | `train.py` | Four tiny CPU updates: uninterrupted and crash/resumed weights and RNG agree exactly |

The full strategy-mixture trainer, automated P1→grounding transfer, calibrated real-policy critic dataset, test-time optimizer, MCTS, and continual-learning trainer are **not** complete. The supplied proposals explicitly require earlier empirical acceptance before these stages. Correct reusable primitives and contracts are implemented; unvalidated end-to-end claims are not.
