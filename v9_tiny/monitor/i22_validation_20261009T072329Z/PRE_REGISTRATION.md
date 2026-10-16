# V9 implementation pre-registration

The pre-registration of V9 is `draft_v9.md` §12 (predictions) and §12.3 (the gate register), as revised on 7 October 2026 (draft SHA-256 `f8c83c4427b1cd507f994ae7ba45b99ad19a1fdf028328c53c1d8d595930e3d9`). This file records implementation-level declarations and amendments. Each entry is dated. **Entries I1–I10 were written on 6 October 2026 at 22:13 UTC (7 October, 03:43 IST), before any V9 training run and before any V9 result existed.**

## I1. Staged contract gating (amends draft §13.1's ordering)

§13.1 says every contract except items 1 (replay), 9, 12 and 14 passes before the first V9 training run. V9 instead gates each queue block on the contracts its own code needs:

| Block | Code that trains or evaluates | Contracts that must pass first |
|---|---|---|
| Q1 (S1) | frozen V8 code only (`frozen/code`, byte-identical to `v8_tiny_12h/code`) | 11 (fresh sealed manifests, the S1 part), 13 (test hygiene), testbed integrity, V8's own contract suites (`checks.py`, `revision_checks.py`), eager-vs-compiled parity; 10 only if a PrefixLM arm runs |
| Q2 (S6-base) | V9 LSL code | 14; 11 (colour-count generator, parser and verifier) |
| Q3 (S2) | V9 H-NCP code | 1–5, 8, 9 |
| Q4 (S3) | V9 S3 code | 2, 9 (re-run with the S3 code) |
| Q5 (S4) | V9 S4 code | 6, 8 |
| Q6 (S5) | V9 search code | 7, 12, 15 |

Reason: Q1 runs no V9 model code, so the rule's purpose (no untested code trains) is met for Q1. Each later block's driver checks only its own list.

## I2. Layout

The frozen testbed is at `v9_tiny/frozen/code` and `v9_tiny/frozen/data`, not `v9_tiny/code/frozen` (draft §4.1). With this layout, V8's relative paths resolve unchanged, and V8's `implementation_hash()` covers exactly the frozen files: `fcc93d876201…`, equal to V8's final implementation hash. `v9_tiny/v5_tiny` is a symlink to `../v5_tiny` for the TinyStories array. File hashes are in `spec/FROZEN.json`.

## I3. Fresh sealed manifests and sealed text

- **Symbolic.** All nine V8 suites were drawn again with V8's item counts from the frozen generator, using seeds 97300–97308. The files are `frozen/data/v9test_*.npz`, with the hashes in `spec/FRESH_SEALED.json`.
- **Disjointness.** The items are disjoint by canonical hash from V8's dev and test sets and from all 768,000 symbolic items that V8's seed-811 runs trained on, which are recorded in the checkpoints' mixer state. Zero collisions occurred.
- **Exclusion.** They were appended to the training exclusion list before any V9 training.
- **Text.** No fresh text documents exist. V9's sealed text is V8's test split, which V8 evaluated. No V9 model trains on it.
- **Development.** V9 reuses V8's development sets.

## I4. Seeds

The new seeds are 901 and 902. Under frozen `train.py` the mixer seed is 81000 + seed and the generator seed is 82000 + seed. Neither range collides with V8's evaluation-generator seeds, its probe seed 89999 or seed 811.

All three S1 arms use the same two seeds, so at a given seed they see identical batches; that is what makes the S1 comparisons paired. V8's seed 811 stays historical (draft §5.2).

## I5. Q1 scope in this launch

Only the mandatory arms run: A_delayed, A_alpha0 and A_flat, two new seeds each, with 24,000 updates and V8's exact recipe. The commands reproduce `v8_tiny_12h/runs/A_*/config.json`, which a contract checks. Training is compiled, as in V8, because the eager-vs-compiled gradient parity check passed.

Not launched now: A_delayed_prefixlm, A_const, A_const_hncp, and the ×4 arms. They are in the full column (§5.2).

## I6. Q1 wall-clock

Q1 trains while V9's later-stage code is written and tested on the same machine; those tests run on CPU where possible. Updates and exposure are fixed, so no result depends on Q1's wall-clock, which is reported as measured under development load.

## I7. G1

G1 is computed from the development `rate` fields by point estimate: standard ≥ .80 and lookup ≥ .95 (register G1). It is not V8's `evaluate.competence()`, which also required interval bounds and first-hop legality.

The S2 teacher assignment follows §13.2 and is written to `runs/S2_TEACHERS.json`.

## I8. S1 sealed procedure

The procedure is declared in `spec/S1_SEALED_PROCEDURE.json`: suites, n, decoding, text documents, the H3/H4/H5 tests, pooling across seeds, the Holm family and the decision rules. Each S1 checkpoint is evaluated once on the sealed sets, after Q1 training completes.

H3's "hierarchy" is the anchor arm, A_delayed. H4's text NLL pools both families' 512 documents.

## I9. Per-checkpoint checks, as in V8

After each Q1 run, the frozen `revision_checks.py` (data integrity and trained-weight cache parity, including hard14 and long18 prompts) and the frozen `evaluate.py` development evaluation run before the next run starts.

## I10. Seed-811 provenance

SHA-256 of the three V8 seed-811 checkpoints and their summaries is in `spec/SEED811.json`. The checkpoints are rechecked before reuse.


## I11. S2 implementation decisions (written 2026-10-07 03:47 UTC, before any S2 run on a real teacher)

Context at writing time: Q1 dev results exist for A_delayed s901 (failed G1: standard .193, lookup .272) and s902 (passed: .842, 1.0). Under I7 and §13.2 the S2 teachers are then s902 and seed 811; the driver applies that rule mechanically. No S2 result has been seen.

- **Pooling.** G2.1(b), G2.2(b), G2.3 and G2.4 pool standard and hard10 dev items. G2.2(a): graph = standard dev; text = edu and ts dev documents pooled.
- **States.** One state per item. D_self is restricted to states reached by a legal history. 'first' states are prompt-only.
- **Closed loop.** j0 is drawn among scored source chunks (NCP weight > 0). The rollout length N ~ U{1..4} stays inside the document. Feedback k* is detached.
- **Dead-code restart.** Every 500 updates, only before 80% of steps; dead means win share < 1/(10K).
- **Engine.** Full recomputation without caches. At K = 1 it equals the frozen `generate()` token for token (contract).
- **Training.** Compiled training (heads and decode compiled), after eager/compiled parity of outputs and probe gradients. The full objective's loss agrees within 1e-3 relative; argmin ties can flip under float noise, so its gradients are not compared.
- **Measurement details.** ECE intervals use a cluster bootstrap with 2,000 resamples. Imagined trajectories score the argmax sequence along the imagined path.
- **Seed rule.** A seed split proceeds with the passing seed's model (§12.3 rule 4); the third seed is deferred.
- **Replays.** The V8-schedule replays (K = 1, p_cl = 0, 6,000 updates, seed 811, V8 teacher) are an implementation check against the saved B_refit dev summaries, not a gate.

## I12. S6 implementation decisions (same date, before any S6 run)

- **Engine.** Every S6 model is evaluated by full recomputation, so pre- and post-phase comparisons share one engine. At zero banks it equals frozen `generate()` (contract).
- **LSL-scaled gating.** The q/k/v gate reads n1(x). The o, w12 and w3 gates read their own matrix inputs. JL projection 64, Φ_neg 32 and Φ_pos 16 diagonal GMMs (V8's EM protocol), ρ = .3.
- **GMM fitting.** Φ_neg is fitted once per anchor on about 1M foundation-mixture tokens. Φ_pos is fitted after training, on phase data, with the newest bank held open (paper Alg. 1).
- **Phases.** Phase training uses 2,000 updates, lr 3e-4, response- or text-only loss, and the same budget for all three arms.
- **Calibration split.** A dedicated calibration split (seeds 97500+, content-hash disjoint from every evaluation set) supplies old-task inputs for G6.0 and telemetry. Colour-count dev, calibration and test use seeds 97400–97402.
- **G6.0.** Φ_pos is fitted with the new adapter at zero. The phase is labelled separable if the mean smoothed hit rate is ≤ 5% and no suite exceeds 15%, weakly separable if the mean is ≤ 10%, and non-separable otherwise. P2 is chosen as in drive_q2.
- **Function drift.** Function KL (pre ‖ post) is measured on calibration inputs.
- **Margins the register leaves open.** G6.4 and G6.5 on P1 (text) use a .01-nat margin, since the register's "5 points" is symbolic. G6.3 pools item-level differences over the retained symbolic suites; text retention enters G6.2 only. G6.5 is one-sided: a fitted gate better than forced open passes.
- **Arms.** The mandatory arms are LSL-scaled, always-open and VQ-only (3 arms × 3 phases × 2 seeds, §10.7 budget). LSL-v8 and full fine-tune are in the full column. Seeds 931 and 932; anchors are the S2 teachers from §13.2 (the S1 A_delayed seeds that passed G1).
- **Timing pilot.** A timing pilot (§10.7 budget) is recorded in tests/s6_pilot.json before the Q2 lock.

## I13. Queue order on one GPU (user decision, 2026-10-07 04:30 UTC, before any S2 or S6 run)

The full mandatory queue does not fit in 24 h on one MPS GPU: Q1 + Q3 + Q2 need about 38–42 h, and Q4–Q6 are not yet implemented. The user chose "Tier A first".

The order is:
1. Q1.
2. Q3a: the S2 dense tier (V8repro, K1 and K8, both seeds), the dense replay, the dense register and S2_PROMOTION, then Q3A_RESULT.json.
3. Q2 (S6-base).
4. Q3b: the recurrent tier, its replay and register, and any §6.8 contingency arms, then Q3_RESULT.json.
5. Q4 onward.

No arm, exposure, seed or threshold changes. Q2 and Q3 do not run concurrently, because concurrent runs on one GPU save no time and would contaminate the throughput figures.

## I14. S3 implementation decisions (written 2026-10-07 05:22 UTC, before any S3 run)

- **Optimisation.** Learning rate 3e-4 (V8's lowest fine-tuning rate, its CL stage), warmup 200, cosine to 10%, V8's AdamW groups and clipping at 1.0. The draft gives no S3 rate.
- **Seeds and data order.** S3 seeds are 921 and 922, one per promoted S2 seed. Mixer seed 84000+seed and span generator 85000+seed, identical across arms at a seed (same data order, §7.2).
- **Objective.** The S2 objective is reused with distillation off; closed loop p_cl = .25; m, α, ε and λ_sel are taken from the S2 run. There are no dead-code restarts in S3. S3b adds δ = .1 constant distillation to the original teacher's code-free outputs, computed from the teacher's own encoder on clean tokens, against the student's code-free outputs on its own clean concepts.
- **Encoder gradients.** In S3a-enc and S3b the encoder pass runs with autograd (flag `context_grad`). The S2 objective is otherwise unchanged; S2 keeps the encoder pass under no_grad, as before.
- **λ curve.** λ ∈ {0, .3, .6, 1}, through System.plans with the student's code-free path, using the full-recompute engine (cached generation bypasses System.plans). Evaluated on dev standard 1,024, lookup 512, and text edu/ts 256 each.
- **G3.** G3 compares, per seed, against the promoted S2 model of that seed:
  - G2.1(b): S3 zone pass and point ≥ .8× S2's;
  - G2.2: zone pass;
  - G2.3-H at H = 1 and at the S2 lookahead depth H*, plus the G2.4 branching, coverage and predicted-vs-executed rows: zone no worse than S2's (pass > partial/inconclusive > fail);
  - dev standard success: paired lower bound ≥ −2 points;
  - deployable text NLL: paired upper bound ≤ +.01 nats, with argmax plans for K = 8 and code-free for K = 1;
  - hard10 and code-free drift are reported only.
- **Arm choice.** An arm replaces S2 only if both seeds pass; preference order S3a-dec, then S3b. H11's paired S3b test belongs to the final corrected-family analysis. Q4 runs the mandatory arms S3a-dec and S3b; S3a-enc is in the full column.

**I12 amendment (2026-10-07 05:30 UTC).** The GPU is occupied by Q1 and then Q3a until Q2 starts, so the §10.7 timing pilot runs as Q2's first stage and is recorded in tests/s6_pilot.json. It reports timings only and changes no phase length, arm or threshold.

## I15. S4/S5 search implementation and sizing rules (written 2026-10-07 05:25 UTC, before any search code or S4/S5 run)

- **Execution model.** Search uses the same full-recompute engine as S2 measurement, with no incremental caches. Contract 7 therefore reduces to three checks:
  - decoding under a stored plan record equals direct decoding under the same plan;
  - re-grounding invalidates the descendants of the re-grounded node;
  - results do not depend on sibling ordering or frontier batching.

  Every method uses the same engine, so wall-clock comparisons are like-for-like. Nodes per second are measured for this engine.
- **Batching.** Search runs in lockstep over B problems: one PUCT selection per tree, then one batched head call for all K children of all frontier nodes, then one batched decode for grounded children. B is a parameter; the §9.5 pilot measures nodes per second at frontier sizes {1, 16, 64}.
- **Q5 sizing rule.** The Q5 driver's first stage is the nodes/s pilot. B_train stays 32 if a decoder-free iteration on 4,096 problems is projected at ≤ 3 h and iteration 0 on 512 problems at ≤ 4 h. Otherwise:
  1. B_train drops to 16;
  2. then the problems per iteration halve (4,096 → 2,048);
  3. iteration 0 keeps 512 problems.

  The applied sizes are written to runs/S4_SIZING.json before iteration 0.
- **Q6 sizing rule.** The Q6 driver measures t_g and nodes/s first. The primary matrix (§9.6) runs in full. The secondary budgets {2, 8, 32}× and the S1-anchor token variants run only if the measured projection for the primary matrix plus the mechanism and answer-loop runs leaves time inside the Q6 window; the window is fixed when Q6 is queued.
- **Ties and noise.** Ties and Dirichlet noise follow §9.3. Noise is drawn from CPU generators seeded per (problem, episode), so batching does not change draws.

## I16. S4 implementation decisions (written 2026-10-07 05:48 UTC, before any S4 run)

- **Problems.** Fresh graph problems from the frozen generator (half standard, half hard10), with every evaluation hash excluded. Seeds are 86000 + 100·seed + iteration; search seeds are 87000 + 100·seed + 10·iteration. Hop limit 20.
- **Iteration 0.** Grounded children, verified greedy-rollout leaves, STOP prior 1/(K+1), the S2/S3 selector, Dirichlet root noise.
- **Iterations 1–4.**
  - Decoder-free children (grounded if RH(1) is not allowed; then labelled exploratory).
  - V_ψ leaves only if three G4 rows hold: complete passes or is flagged; intermediate passes; every imagined depth ≤ H passes.
  - Otherwise the whole iteration falls back to iteration-0 mode at 512 problems. This implements "leaves fall back to verified grounded completion … at iteration-0 scale" per iteration rather than per depth.
  - τ_ψ is used at imagined nodes only if G4-τ passes.
- **Labels and fits.**
  - G4 rows use a 20% held-out split by problem (index % 5 == 0).
  - Selector: CE to root visit distributions (temperature 1). V_ψ: squared error to o on executed roots. τ_ψ: BCE on executed roots plus rollout states.
  - Linear heads are fitted with Adam (lr 1e-2, 300 full-batch epochs, wd 1e-4).
  - Imagined G4 states pair depth-d imagined states along the executed code sequence with the executed state at t+d.
- **Baselines.**
  - Problem sets and verifier calls are matched per round to the expert-iteration run of the same seed.
  - RFT and GRPO use rank-16 q/v LoRA on the three decoder blocks (24,576 parameters) with code-free plans and T = 1 sampling. GRPO uses groups of 8 with group-normalised advantages, and groups with constant reward are skipped. REINFORCE trains the selector with a mean-reward baseline.
  - Each round: 2 epochs of Adam at lr 1e-3, batch 64, clipping at 1.0.
  - Without an expert-iteration schedule (concept branch not run), the default schedule is 5 rounds of 512 then 4 × 4,096 problems with 16 samples per problem.
- **GradCuit.**
  - The committed chunk is chosen uniformly; later chunks are regenerated with selector-argmax codes.
  - The random-direction kernel uses step norm lr·√16 (Adam's nominal step), projected to the same bound.
  - The lr pilot uses up to 256 failed dev hard10 answers of the S4 model. lr is chosen per bound setting by success rate, ties to the smaller lr.
- **Selection metrics.** Dev greedy success on hard10 (512) and standard (512) per arm. The token arms are decoded code-free.

## I17. S5 implementation decisions (written 2026-10-07 05:59 UTC, before any S5 run and before any sealed S5 item is read)

- **Budgets.**
  - b × t_g, where t_g is the promoted model's batched greedy wall-clock per item, measured on the same split and suite (batches of 64) with the full-recompute engine. All methods run lockstep over the same batches of 64 items.
  - Search methods size expansions per step adaptively from measured time per expansion and the expected remaining steps, i.e. the mean executed hop count of successful greedy answers. One-tree methods size the per-item expansions from an 8-item probe tree.
  - A hard stop at 2× the budget marks unfinished items as failures. Overruns are reported per batch.
- **RH(1) and RH(H\*).** Decoder-free children with V_ψ leaves when V_ψ passed its S4 rows; otherwise grounded children with rollout leaves (§9.3's fallback). No root noise at evaluation.
- **G(R) and IMAG.** One tree per item. Nodes at depth multiples of R are grounded by replaying pending chunks under their stored plan records, and records stay immutable in later decodes. G(1) grounds every child. IMAG grounds only the final path. With rollout leaves, G(R ≥ 2) becomes G(1).
- **Token tree.** Prior is the code-free decoder softmax over the 3 colour tokens plus DONE at the hop's first position; children force the first token and complete the hop greedily. The leaf rule (verified code-free rollout vs a value head on the decoder's final-layer state, fitted on S4's executed trajectories) is chosen on 256 dev hard10 items at 4×.
- **Token decoder.** The S4 token arm (RFT or GRPO) with the higher dev greedy hard10 rate, if above the promoted decoder's; otherwise the promoted decoder.
- **Best-of-N.** T = 1 code-free samples, one verifier call each, until the budget is spent.
- **Answer loop.**
  - Inner method: the best of RH(1), RH(H\*) and G(1) on 256 dev hard10 items at 4×. Each inner episode uses budget 8 expansions per step.
  - GEN vs CONT is chosen per item by Thompson sampling on Beta(1, 1 + attempts).
  - CONT uses the §8.3 kernel at the S4 GradCuit pilot's chosen lr for the .5 bound. GradCuit and random direction are run on the same items.
- **Mechanism (H16).** Grounded RH(1) on sealed hard10 at 16×. The node-matched decoder-free rerun uses a fixed expansions-per-step equal to grounded RH(1)'s nodes per item divided by mean hops.
- **Sealed procedure.** A per-seed S5_SEALED_PROCEDURE file (methods, configurations, dev choices, sealed manifest hashes) is written before any sealed item is read.
- **Exploratory block.** First seed only; G(R) only where the combined G2.3-H_R zone is not fail; IMAG unless the register says not run.
- **Not in the matrix.** The secondary budgets {2, 8, 32}× and the S1-anchor token variants are not run unless time permits; that is decided only after the primary matrix's measured runtime.

**I17 note (written 2026-10-07 06:17 UTC).** With the draft-literal PUCT score c·P·√N(s)/(1+N(s,a)), every score ties at N(s) = 0, so the first simulation takes action 0 by the tie rule. At B = 1 expansion per step, RH therefore executes code 0; at B = 2 it executes the higher-prior code. The 1× RH cells, where adaptive sizing may give B = 1–2, reflect this property of the formula and not a malfunction. Batching invariance (I15) is tested on CPU. On MPS, padding widths can change float results, so bit-identical trees across batch sizes are not claimed.

## I18. Final analysis and the S4 sealed procedure (written 2026-10-07 06:17 UTC, before any sealed S4 read)

- **S4 sealed procedure.** Every S4 arm per promoted seed (reference, EI, RFT, GRPO, REINFORCE) is decoded greedily: selector-argmax plans for EI, REINFORCE and the reference (code-free for the reference if K8 was not promoted), code-free plans for RFT and GRPO. Suites: sealed hard10 (512) and hard14 (256). The procedure file spec/S4_SEALED_PROCEDURE.json is written before any sealed S4 item is read.
- **Families.** Holm is applied within each stage family: S3 {H11 drop}, S4 {L3} (L3 leaves the family and is reported as exploratory if RH(1) was not allowed), S5 {L1, H14, H16}, S6 {H12}. S1 {H3, H4, H5} is already corrected in S1_RESULTS.json.
- **Tests.**
  - Difference tests (L1, L3) use a two-sided paired sign-flip p over items, averaged over seeds on the same items. Support needs Holm-adjusted p < .05 and a positive pooled delta.
  - H11 drop: per item, d = S3b sensitivity − .8 × S2 sensitivity (first-hop swap states), with a one-sided sign-flip p for mean(d) < 0.
  - H16 non-inferiority: one-sided p for mean(d) > −.02.
  - H14 bound: one-sided p for mean(d) < +.02.
  - H12 equivalence: TOST at ±.05.
  - Each test uses 10,000 resamples, and is reported with per-seed McNemar exact tests and paired bootstrap intervals plus the pooled interval.
- **Missing runs.** A missing run is reported as missing; it never passes or fails.

## I19. Evaluation batch size 256 → 64 on MPS (written 2026-10-07 17:29 UTC, after Q1 completed and after S2_V8repro_dense_s911 trained; no S2 measurement has completed)

- **What happened.** Q3's first measurement (S2_V8repro_dense_s911_measure, production mode) failed after 6.5 minutes with an MPS out-of-memory error ("other allocations 17.58 GiB, max allowed 20.13 GiB") while decoding hard14. The smoke tests had not caught this: they run measure_s2 in quick mode, where decode batches are 64 and no suite is large enough to fill a batch of 256, so they never exercised production batching.
- **Cause (measured, scratch process, no locked file touched).** The MPS driver retains memory for every new (batch, sequence length) tensor shape the decoder produces. In a fresh process, 17 new SDPA shapes at batch 256 added 4.28 GB, and repeating the same shapes added 0.00 GB. During greedy decoding on the S2 model, driver memory grew 3.71 → 5.35 GB over two batches of 256, with PyTorch's own allocation flat at 0.01 GB. At batch 64 the same decoding plateaued at about 3.5 GB over a full 2,048-item suite. `torch.mps.empty_cache()` recovered about 1 GB of 17 and is not a fix. Masks are not the cause (a float-mask variant grew identically).
- **Change.** measure_s2.py: `greedy_suite` and `self_states` default batch 64, and decode_codes, candidate_concepts and g23 use batch 64 in production (previously 256 and 128). s4_eval.py: greedy dev evaluation chunks of 128 → 64. Batch size does not change what is measured; no arm, exposure, seed, threshold, margin, suite size or item is changed.
- **Consequence for locks.** measure_s2.py is in the lock lists of Q2, Q3, Q4, Q5 and Q6, and s4_eval.py in those of Q5, Q6 and QF. Their contracts and smokes are rerun under new locks before any block launches. Q3's training run S2_V8repro_dense_s911 is complete and is reused (verified by queue_base.train); its measurement is rerun.
- **Residual risk, stated.** Batch size can change float rounding on MPS, so greedy decodes may differ in rare ties from a batch-256 run. No batch-256 result was produced for any S2 measurement, so no comparison is affected. S5 search cells (s5_run.py, BATCH = 64) and train_s4 (frontier 64) already use 64-item batches, but their shape variety under shrinking frontiers is unmeasured on this machine; each S5 cell runs in its own process.
- **Validation (scratch, not a result).** A full production-mode measurement of the trained S2_V8repro_dense_s911 at batch 64 completed in a scratch process in 958 s (2026-10-07 17:46 UTC), reaching every section including hard14. It was not produced under a Q3 lock and is not copied into runs/; the Q3 driver reruns the measurement.

## I20. Evaluation batch 16 for K > 1 decoders on MPS (written 2026-10-08 00:20 UTC; Q3 had completed S2_V8repro_dense_s911, S2_K1_dense_s911 and S2_K8_dense_s911 training and the first two measurements)

- **What happened.** S2_K8_dense_s911_measure failed at 23:25 UTC (MPS out of memory, "other allocations 17.88 GiB, max allowed 20.13 GiB") and the Q3 driver stopped with status failed. The two K = 1 measurements passed at batch 64 (I19).
- **Cause (measured, scratch process, no locked file touched).** I19 validated batch 64 on a K = 1 checkpoint only. A K > 1 decoder scores K code rows per sequence, so the same batch is K times wider. Tracing the K8 checkpoint in a scratch process: driver memory 1.26 GB after the NLL suites, 5.11 GB after greedy standard and hard10, then 5.11 → 19.72 GB over the 256-item hard14 greedy suite, and the next stage (D_self rollouts) crossed the cap. At batch 16 the same hard14 decoding plateaued at 2.03 GB (batch 8: 1.37 GB), at about 20 s per 16 items.
- **Change.** measure_s2.py: new `eval_batch(eng)` returns 64 when K = 1 and 16 when K > 1; `greedy_suite`, `self_states` and the production `qb` (decode_codes, candidate_concepts, g23) use it. K = 1 behaviour is byte-for-byte unchanged, so the completed V8repro and K1 measurements stay valid. No arm, exposure, seed, threshold, margin, suite size or item is changed.
- **Consequence for locks.** measure_s2.py is in the lock lists of Q2, Q3, Q4, Q5 and Q6 (not QF). The waiting Q2, Q4, Q5 and Q6 drivers were stopped (they had not started work); contracts and smokes are rerun for Q3, Q2, Q4, Q5 and Q6 under new locks before any of them launches. QF's lock is unchanged and its driver keeps waiting. Training runs are reused (queue_base.train verifies them); only the K8 measurement is redone.
- **Residual risk, stated.** Batch size can change float rounding on MPS, so a K > 1 greedy decode may differ in rare ties from a batch-64 run; no K > 1 measurement existed at any batch size, so no comparison is affected. The recurrent core (Q3b) and the S5/S6 searches are not yet validated at production shape beyond the same shape-growth argument.
- **Validation.** A full production-mode measurement of S2_K8_dense_s911 in a scratch process under the edited code (started 00:18 UTC); result recorded below when it finishes. It is not a result and is not copied into runs/.

## I21. S6 evaluation batch 256 → 64 on MPS (written 2026-10-08 11:35 UTC; Q3 tier A had completed at 11:04 UTC, and Q2 had started and failed in its first stage)

- **What happened.** Q2's first stage, the S6 timing pilot, failed at 11:17 UTC, 12 minutes after it started, with an MPS out-of-memory error ("MPS allocated: 12.22 GiB, other allocations: 7.91 GiB, max allowed: 20.13 GiB") in the first production-shape evaluation, `s6.symbolic` on dev standard (1,024 items). The Q2 driver stopped with status failed. No S6 training phase, measurement or result exists. The Q3 recurrent tier and Q4 to QF were waiting on Q2 and are unchanged.
- **Cause.** The same mechanism measured in I19: the MPS driver retains memory for every new (batch, sequence length) tensor shape that full-recompute greedy decoding produces. I19 and I20 lowered the evaluation batch in `measure_s2.py`; `s6.symbolic` has its own default of 256, which those changes did not touch, and the Q2 smoke test could not catch it because it runs capped suites.
- **Change.** `s6.py`: the default `batch` of `symbolic` goes from 256 to 64. No arm, exposure, seed, threshold, margin, suite size, item or evaluation procedure is changed, and batch size does not change what is measured.
- **Consequence for locks.** `s6.py` is in the Q2 lock list only. Q2's contracts and smoke test are rerun under a new LOCK_Q2 before the block is launched again. The locks of Q3, Q4, Q5, Q6 and QF do not include `s6.py` and are unchanged.
- **Residual risk, stated.** Batch size can change float rounding on MPS, so a greedy decode may differ in rare ties from a batch-256 run. No S6 measurement existed at any batch size, so no comparison is affected. Other stages of the pilot (text evaluation, G6.0, telemetry) had not yet run when it failed and are validated by the same scratch run below.
- **Validation.** A production-mode run of `s6_pilot.py` in a scratch process under the edited code, writing to a scratch directory. It is not a result and is not copied into `tests/` or `runs/`. Outcome recorded below when it finishes.
