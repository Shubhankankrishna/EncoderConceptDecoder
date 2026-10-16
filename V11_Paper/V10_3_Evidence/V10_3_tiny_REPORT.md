# V10.3 tiny RCDL — test report

**12 October 2026.** Every number is traceable to [v10_3_tiny/REPORT_TABLES.md](v10_3_tiny/REPORT_TABLES.md), which `v10_3_tiny/report.py` renders from the saved JSON, or to the output of the script named beside it. The prose tables were transcribed from those files and checked against them.

## 0. Verdict

**The tiny V10.3 is not ready to scale to 10M by its own predeclared rule.** It passes G1 and G2 and fails G3.

1. **The V10.3 changes fix the V10.1 recipe's damage (G1 passes).** X8 beats A on dev text by −0.010 nats [−0.015, −0.006] and on math by −0.209 [−0.235, −0.181]. On the identical training stream, A's math loss falls further behind C0 as training goes on (+0.245 in the last 1,000 updates). The codebook-decay fix works: A's codebook shrinks 27% while X's changes by about 1%.
2. **The concept path still costs more than it returns (G2 passes only on the point estimate).** X8 is worse than C0 on text in both seeds: +0.018 [+0.013, +0.023] in seed 1, +0.005 [−0.000, +0.010] in seed 2. It is also worse on math in both seeds (+0.029, +0.119). It has 60% more dense parameters and takes 1.6× longer per step. The written rule (point estimate ≤ 0.02) is met, but the seed-1 interval reaches above the margin, and same-device seed variation in X8's text NLL (0.026) is larger than the margin. This is a technical pass that reads as a cost.
3. **The mechanisms do not yet do what they are for (G3 fails).**
   - The forecast predicts the next group *worse* than the core's own input does.
   - The second concept visit is inert (≤ 0.011 nats across X8 and X4).
   - The choice of code barely matters: on the pretraining families, mixture, uniform and committed log-likelihoods agree within 0.002 nats/token. The exploration objective is correct but has almost nothing to explore.
4. **The only place the concept arms beat C0, Sudoku, is not evidence for concepts.** Sudoku has a fixed layout.
   - X8's seed-1 advantage (−0.064) disappears when the board is shifted by half a group or a whole group, and it does not replicate: in seed 2, X8 is *worse* than C0 (+0.022).
   - Two of the four concept-arm runs show the alignment signature: they degrade less when the shift preserves group alignment than when it breaks it. Neither C0 run does.
   - This is the kind of data–architecture entanglement found in V9; §3 and §6 give the details.
5. **Sleep layer: G4 passes.** Gated LoRA learns the held-out algebra module as well as plain LoRA (newmath −1.762 vs −1.760) and beats it on all three declared retention sets by far more than the 0.01-nat margin: text +0.004 vs +0.127, sorting 0.000 vs +0.352, multiplication +1.34 vs +2.24. By the predeclared rule, the sleep layer enters the 10M specification. But protection follows input-distribution distance only: the related add/subtract skill forgets as much as under plain LoRA. The calibrated RBF bank, with frozen weights, gets 88% of LoRA's new-task gain and forgets far less multiplication (+0.135 vs +1.342 for gated LoRA). Consolidating it into gated adapters and resetting the bank keeps that gain (newmath −1.58) with zero forgetting on text, Sudoku and sorting. No method protects the closely related add/subtract skill (+0.26 to +0.46) (§8).

## 1. What was built

`v10_3_tiny/` is a sub-600k-dense-parameter implementation of V10.3 (design: [V10_3_RCDL_Architecture.md](V10_3_RCDL_Architecture.md) §1–§2). It reuses the V10/V10.1/V10.2 module classes unchanged and supplies its own configuration. The rules and the go/no-go criteria were written before any result in [v10_3_tiny/ARCHITECTURE.md](v10_3_tiny/ARCHITECTURE.md); each later addition there is timestamped and states what had been seen at that time.

| Arm | What it is | Dense | Deployed dense | Engram (sparse) |
|---|---|---:|---:|---:|
| C0 | Encoder, Engram and decoder with E1/E3 CRC; no concept path | 311,704 | 311,704 | 1,050,080 |
| A | Unmodified V10.1 model and losses (latent MSE, codeword fitting, chunk head, hard-argmin exploration; codebook decayed) | 484,782 | 443,062 | 1,050,080 |
| X8 | V10.3: V10.2 RF model (forecast-only token head, concept-history reader), full learned forecast map, codebook exempt from decay, mixture-likelihood exploration with the selector as prior; K = 8 | 503,428 | 474,364 | 1,050,080 |
| X4 | X8 with K = 4 | 502,912 | 473,848 | 1,050,080 |

Width 56, 4 heads, vocabulary 1,024 (BPE re-fit on the train split, digits isolated, held-out module excluded), context 256. E1 has 2 blocks with the Engram after block 1. E2 is one two-stream block run twice. E3 has 1 block. The concept core has 2 blocks run in 2 visits. The decoder has 2 blocks.

The sleep layer adds rank-4 LoRA on every attention and FFN matrix of E2, the core and the decoder. The gated variant uses the Local-Support-Learning GMM gate (per visit, from the frozen model's own sub-layer inputs). The RBF correction bank is V10.2's `learn_episode`, and consolidation distils the bank into fresh gated adapters.

## 2. Correctness tests — 26/26 pass (`v10_3_tiny/tests.py`, `tests_result.json`)

| Test | Result |
|---|---|
| Dense budget ≤ 600k, all four arms | 311,704 / 484,782 / 503,428 / 502,912 |
| Strict `state_dict` round trip, all arms (C0 and Decoder103 included) | identical tensors |
| Optimizer groups: codebook decayed for A, exempt for X | as specified (72 vs 79 decayed matrices) |
| Concept-history release mask and zero sentinel, lengths 1/7/8/9/15/16/17/33 | 0 mismatches in 320 rows |
| Future-token perturbation leaves earlier logits unchanged (text, math, Sudoku; cut points include 7/8/9/15/16/17) | max difference 0.0, all arms |
| Candidate branch with the committed code equals the main forecast | ≤ 1.3×10⁻⁷ |
| Mixture loss: analytic gradient of the per-document normaliser | 3×10⁻¹⁷ |
| Mixture loss: directional finite difference through selector, codes, core, PQ and readout (float64) | relative error 4×10⁻⁷ |
| Every one of the K codes receives mixture gradient on a candidate update | yes, all 8 |
| RBF correction derivative through the aligned, reader-key and reader-value routes (float64) | relative error 3×10⁻⁸ |
| CPU/Metal forward and gradient parity (C0, X8) | loss identical; gradients within 5×10⁻⁷ |

Further checks:
- **Allocator:** over 200 smoke updates, Metal driver memory stayed flat (A 1.65→1.67 GB, X8 1.68→1.69 GB).
- **Sleep machinery** (smoke checkpoint):
  - torch and sklearn GMM decisions agree 100%;
  - zero-B adapters, and closed gates, reproduce the base model;
  - gated adapters move the logits less than plain ones but still more than zero;
  - the EMA gate is causal.
- **Evaluator:** batched NLL equals single-document NLL to 2×10⁻⁷ relative.

## 3. Is the data biased toward the architecture? (`audit_data.py`, `data_audit.json`)

V9's GraphNav responses had a designed four-token step structure that matched V9's four-token concept chunks (review_v9.md:113). The V10 concept path pools fixed 8-token groups, so the audit asked whether any family carries structure aligned to those groups.

- **Text:** no alignment. 97% of documents are cropped at a uniform random offset, and no frequent token favours one residue mod 8 (largest share 0.139; uniform is 0.125).
- **Math and newmath:** no alignment. Answer starts are spread over all residues (math χ² 13–16 with df 7; the p = 0.001 cutoff is 24.3). Answer lengths are unrelated to 8, and there is no padding to multiples of 8.
- **Sudoku:** fixed layout. Every record is 171 tokens and the answer starts at position 89. Eight-token groups do not match rows (9) or boxes (3×3), so this is not matched structure, but every board maps cells to groups identically.

The same layout was present in V10.1 and V10.2. It was handled with a predeclared control: re-score every board with four extra header tokens (half a group) and compare how much each arm degrades. Results are in §6.

## 4. Protocol

- **Schedule and data stream:** every arm trained 6,000 updates with the same seed and a step-based schedule, so the document stream and learning rate were identical across arms. Batch 16, context 256; AdamW 1e-3, warmup 200, cosine to 10%, clip 1.0; Engram trained with CPU SparseAdam at 5× the learning rate.
- **Devices:** the four seed-1 arms trained concurrently, X8 and A on CPU and X4 and C0 on Metal. Seed-2 runs (C0 and X8) trained concurrently on CPU.
- **Solo throughput** on Metal (200-update smoke, median s/step): C0 0.160, A 0.262, X8 0.261, X4 0.262. C0 runs 39% faster; it is compared at equal updates.
- **Not run:** the Engram on/off throughput benchmark. The machine was never idle again after launch.
- **Evaluation panels:** identical across arms, dev split only: 256 text windows, 128 problems per pretraining math module, 128 newmath problems, 64 boards per Sudoku family. Intervals are 95% paired document bootstraps.

## 5. Results at equal updates (seed 1, dev split)

| Run | Text | Math | add/sub | mul | sort | Newmath (unseen) | Sudoku easy | Sudoku hard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | **3.364** | **1.088** | 1.358 | 1.852 | **0.671** | 2.949 | 0.911 | 0.909 |
| A (V10.1) | 3.392 | 1.325 | 1.359 | **1.735** | 1.123 | **2.679** | 0.888 | 0.886 |
| X8 (V10.3) | 3.382 | 1.117 | 1.335 | 1.780 | 0.758 | 3.068 | **0.847** | **0.844** |
| X4 | 3.384 | 1.185 | **1.320** | 1.852 | 0.841 | 3.190 | 0.860 | 0.852 |

Paired document differences (95% bootstrap intervals):

| | Text | add/sub | mul | sort | Newmath | Sudoku easy |
|---|---|---|---|---|---|---|
| X8 − A | −0.010 [−0.015, −0.006] | −0.023 [−0.051, +0.002] | +0.046 [+0.027, +0.064] | −0.365 [−0.392, −0.339] | +0.389 [+0.304, +0.474] | −0.041 [−0.053, −0.029] |
| X4 − A | −0.008 [−0.014, −0.003] | −0.038 [−0.066, −0.012] | +0.117 [+0.094, +0.141] | −0.282 [−0.308, −0.258] | +0.512 [+0.418, +0.604] | −0.028 [−0.041, −0.014] |
| X8 − C0 | +0.018 [+0.013, +0.023] | −0.022 [−0.051, +0.006] | −0.072 [−0.092, −0.051] | +0.087 [+0.064, +0.111] | +0.118 [+0.045, +0.195] | −0.064 [−0.076, −0.052] |
| A − C0 | +0.028 [+0.023, +0.034] | +0.001 [−0.028, +0.029] | −0.117 [−0.138, −0.097] | +0.452 [+0.422, +0.484] | −0.271 [−0.350, −0.195] | −0.023 [−0.037, −0.010] |
| X4 − X8 | +0.002 [−0.003, +0.007] | −0.015 [−0.041, +0.011] | +0.072 [+0.051, +0.092] | +0.083 [+0.061, +0.105] | +0.123 [+0.037, +0.208] | +0.013 [+0.001, +0.024] |

- **Math is mostly about sorting.** Most of X8's math advantage over A comes from sorting (−0.365). Both concept recipes lose to C0 on sorting (A +0.452, X8 +0.087, X4 +0.170) and beat it on multiplication (A −0.117, X8 −0.072). Sorting has by far the longest answers: median 20 tokens, about 2.5 groups, against 10 for multiplication and 4 for add/subtract (dev split). These are observations; no explanation has been tested.
- **Newmath, the never-trained algebra module, is an anomaly.** A is clearly best (2.679), ahead of C0 (2.949), X8 (3.068) and X4 (3.190). Exploration spread on newmath is about 5× larger than on text (X4: 0.076 vs 0.014 nats/token), so code choice matters more out of distribution. Whether A's latent and chunk objectives explain its better zero-shot NLL is untested.
- **Exact match is at the floor** (0–1 of 96 math problems, 0 of 64 newmath, 0 of 16 boards). That is why the predeclared G1 math clause uses a paired interval: X8 1/96 vs A 0/96, interval [0, +0.031].
- **K = 8 beats K = 4** on math (+0.069 for K = 4) and on newmath (+0.123); text is level (+0.002 [−0.003, +0.007]).
- **Was the budget too short?** No. By step 6,000, text NLL changes ≤0.01 per 500 updates and the learning rate is at its 10% floor. The gaps are resolved with intervals that exclude zero. The paired training-stream gaps (next table) are stable or widening, not closing.

**Paired training-stream gaps.** Every arm saw the identical batch at each step, and under one epoch per family, so this is effectively fresh-data loss (`paired_training.py`). Stream identity holds by construction (same seed, same call sequence). The script checks family identity at every step, not document identity.

| Mean token-loss gap | Steps 1–1000 | 2001–3000 | 4001–5000 | 5001–6000 |
|---|---:|---:|---:|---:|
| X8 − C0, text | +0.000 | +0.006 | +0.012 | +0.013 |
| X8 − C0, math | +0.018 | +0.034 | +0.042 | +0.034 |
| A − C0, text | +0.068 | +0.017 | +0.021 | +0.024 |
| A − C0, math | +0.028 | +0.056 | +0.213 | **+0.245** |
| X8 − A, math | −0.010 | −0.022 | −0.170 | **−0.210** |
| X8 − C0, Sudoku easy | +0.009 | +0.016 | −0.063 | −0.063 |

Standard errors are ≤0.008 nats in every cell.

**Does concept prediction improve? Your V10.1 chart.** Yes, in V10.3. In A (the V10.1 recipe), the latent concept loss falls and then rises late on text (0.315 → 0.334 → 0.347) and on math (0.597 → 0.623 → 0.627), which reproduces the V10.1 observation. In X8, the concept-prediction losses fall at every stage on every family:
- forecast future-token CE: text 6.13 → 5.65, math 3.24 → 2.03, Sudoku 2.89 → 2.09;
- mixture loss on candidate updates: text 6.13 → 5.76, math 3.41 → 2.03.

The limit is in §7: the falling forecast loss does not mean the forecast knows more than the core's input already did.

## 6. Data–architecture alignment: the Sudoku control

Shifting the Sudoku board by 4 tokens (half a group) and by 8 tokens (one group, so cells keep their group alignment) makes the same kind of out-of-distribution header edit to every arm (`grid_shift8.py`, `grid_shift8.json`).

| NLL increase | Shift 4, easy / hard | Shift 8, easy / hard | Intact X − C0 (easy) |
|---|---|---|---|
| C0 | 0.169 / 0.171 | 0.209 / 0.213 | — |
| A | 0.180 / 0.179 | 0.273 / 0.279 | −0.023 |
| X8 | 0.223 / 0.236 | 0.296 / 0.302 | −0.064 |
| X4 | 0.218 / 0.215 | **0.123 / 0.127** | −0.051 |
| C0, seed 2 | 0.032 / 0.027 | 0.134 / 0.116 | — |
| X8, seed 2 | 0.229 / 0.225 | **0.173 / 0.167** | +0.022 (against C0 seed 2) |

- C0 has no groups, yet degrades under either shift, and how much varies 5× between seeds (0.169 vs 0.032 at shift 4). Every model learns the fixed layout from the BOS anchor to a seed-dependent degree. Both C0 runs degrade *more* at shift 8 than at shift 4, which is the plain effect of a longer out-of-distribution prefix.
- **X4 shows the alignment signature.** At a half-group shift its extra degradation (+0.049) roughly cancels its Sudoku advantage. At a whole-group shift it degrades *less* than C0 and keeps its advantage (0.983 vs 1.120).
- **X8 (seed 1) is fragile to any header edit.** It degrades more than C0 at both shifts, so after either edit its advantage is gone: shifted-4 1.071 vs C0 1.081, shifted-8 1.143 vs C0 1.120.
- **X8 seed 2 shows the alignment signature as well** (0.229 at shift 4, 0.173 at shift 8), but it has no intact advantage to lose.
- **Conclusion.** Two of four concept-arm runs degrade less when group alignment is preserved, and no C0 run does. So the 8-token grid gives concept models a layout handle on Sudoku that a group-free model lacks. Whether that handle helps, by how much, and in which seed is not stable: X8 is −0.064 against C0 in seed 1 and +0.022 in seed 2. **Sudoku results are not evidence of concept-level reasoning.** V10.1 and V10.2 Sudoku claims carry the same caveat.
- Text and math have no such alignment (§3), and G1/G2 rest on them.

## 7. What the concept path does (interventions, forecast information, exploration)

**Switching routes off (X8, Δ NLL in nats/token; 95% intervals for every cell are in REPORT_TABLES.md):**

| | Text | Math | Sudoku easy |
|---|---|---|---|
| Whole concept path off | +0.004 | **+0.100** [+0.088, +0.113] | +0.131 |
| Aligned forecast off | +0.002 | +0.021 | +0.026 |
| Concept-history reader off | +0.001 | +0.056 | +0.080 |
| Reader PQ route off | +0.000 | +0.004 | +0.008 |
| Reader state route off | +0.002 | +0.053 | +0.089 |
| One concept visit instead of two | +0.000 | +0.002 | +0.007 |
| Calibration-mean substitution | +0.002 | +0.077 | +0.126 |

- **Usage:** X8 uses its concept path on math and Sudoku (A barely does: ≤0.012 anywhere) and hardly at all on text. Mean substitution costs almost as much as switching the path off, so the decoder reads the *content* of the concept state, mostly through the reader's continuous-state route. The PQ route carries little.
- **Second visit (Loop of 2):** inert at this scale (≤0.009 nats for X8; ≤0.011 for X4).
- **Net effect:** X8 is still worse than C0 on math, even though it relies on the path. C0 has identical encoder and decoder blocks, so the path, together with its auxiliary losses, is a net cost to the shared layers. An X arm with the auxiliary weights at zero would test this; it was not run.

**Forecast information (FVU of the next group's pooled state; ridge readouts fitted on a disjoint calibration panel):**

| X8 | Text | Math | Sudoku easy |
|---|---:|---:|---:|
| ridge(forecast F) | 0.958 | 0.694 | 0.595 |
| ridge(core state U) | 0.954 | 0.655 | 0.493 |
| ridge(core input P3) | 0.954 | **0.637** | 0.512 |

The forecast is a *worse* linear predictor of the next group than the core's own input. The core does not add predictive information; it loses some. A's forecast does slightly better on math (ridge(F) 0.835 against ridge(P3) 0.907), but A is trained toward the pooled state and is the weaker model overall.

- **N2 fails (0.0017 against the 0.25 threshold).** As predeclared, this ratio compares two different spaces for X, and the functional test is `aligned_off`: +0.002 text, +0.021 math. The forecast reaches the decoder but carries little.
- **N3 passes.** X8's codebook RMS is 1.002 against 1.013 at step 1; A's falls to 0.735.

**Exploration (dev boundaries; utility is each arm's own training utility):**

| X8 | Text | Math | Sudoku easy |
|---|---:|---:|---:|
| Median spread, best to worst code (nats/token) | 0.021 | 0.028 | 0.011 |
| Code main-effect share of the variance | 0.1% | 0.4% | 5.8% |
| Selector accuracy (committed = hindsight winner) | 38% | 18% | 13% |
| Mixture LL − committed LL | +0.0017 | +0.0004 | −0.0006 |
| Selector CE on hindsight winner vs H(winners) | 2.32 vs 1.70 | 2.32 vs 1.76 | 2.82 vs 1.91 |

In training, the responsibility entropy equals the prior entropy to three decimals (X8 1.163 vs 1.168), so the posterior over codes never moves away from the prior. **N5 fails**, for a stated reason:
- The mixture objective maximises likelihood, not winner prediction.
- With a spread of about 0.01–0.03 nats/token and a code main effect of 0.1–6% (X8), the hindsight winner is noise.
- A peaked prior then necessarily has higher winner-CE than H(winners).

This explains why the predeclared proxy was the wrong test; it does not reverse the fail. The underlying fact is the same as in V10.1: **at this scale the K codes do not produce meaningfully different futures**, so exploration, mixture or argmin, has nothing to select.

## 8. Sleep layer (X8_s1, held-out algebra module)

Protocol (`sleep.py`):
- All conditions start from the frozen X8_s1 checkpoint and see the identical newmath stream: 600 updates, batch 16, learning rate 2e-3, identical adapter initialisation.
- Adapters are rank-4 LoRA on every attention and FFN matrix of E2, the concept core and the decoder: 17,920 parameters (3.6% of the dense model).
- Gated adapters use the LSL gate open(x) = log Φpos(x) > log Φneg(x).
  - x is the frozen model's own input to each sub-layer, computed in an adapter-off pass.
  - Φneg is fitted on all four pretraining families and Φpos on newmath train rows; both are 4-component diagonal GMMs, separate per visit.
  - Token-rate sites use the paper's EMA; core sites use raw decisions.
- The gate fit opens on 80–98% of newmath fitting rows and 6–19% of pretraining rows; torch and sklearn decisions agree 100%.

| Dev NLL | Text | add/sub | mul | sort | Newmath | Sudoku easy / hard |
|---|---:|---:|---:|---:|---:|---:|
| Base (no adaptation) | 3.382 | 1.335 | 1.780 | 0.758 | 3.068 | 0.847 / 0.844 |
| Plain LoRA | 3.509 | 1.735 | 4.017 | 1.110 | 1.308 | 0.893 / 0.893 |
| Gated LoRA | 3.386 | 1.725 | 3.122 | 0.757 | 1.306 | 0.847 / 0.844 |
| RBF bank (calibrated; weights frozen) | 3.382 | 1.598 | 1.916 | 0.789 | 1.518 | 0.847 / 0.844 |
| Consolidated (bank distilled into gated adapters, then bank reset) | 3.382 | 1.799 | 1.962 | 0.759 | 1.487 | 0.847 / 0.844 |

**Protection follows input distance.** The table lists the retention families in order of how often the gate opens on their answer rows (E2 and decoder sites):

| Family | Answer-row open rate | Plain forgetting | Gated forgetting | Gated − plain [95% CI] |
|---|---|---:|---:|---|
| Text | 0.00–0.09 | +0.127 | +0.004 | −0.122 [−0.128, −0.117] |
| Sudoku easy | 0.00 | +0.046 | 0.000 | −0.046 [−0.052, −0.039] |
| Sorting | 0.00–0.10 | +0.352 | 0.000 | −0.352 [−0.407, −0.301] |
| Multiplication | 0.54–0.95 | +2.237 | +1.342 | −0.895 [−0.986, −0.809] |
| Add/subtract (related skill) | 0.47–1.00 | +0.400 | +0.390 | −0.010 [−0.019, −0.001] |
| Newmath (the new task) | 1.00 | −1.760 | −1.762 | −0.002 [−0.005, +0.001] |

- **Both predeclared guards hold.** Plain LoRA forgets ≥0.01 on every retention set, so G4 is testable. Gated LoRA's newmath gain equals plain's, so the gate is not over-closed.
- **G4 passes:** gated beats plain on text, multiplication and sorting by far more than 0.01 nats.
- **The GMM gate detects feature distance; it does not guarantee retention.** Where the old task's inputs look like algebra (arithmetic answers are digit strings, as algebra answers are), the gate opens and the adapters overwrite the skill. The related add/subtract skill gets nothing from gating, and multiplication still loses 1.34 nats.
- **For the 10M specification:** gated LoRA is the retention mechanism for new phases whose inputs differ from pretraining. For a related skill it behaves like plain LoRA, and it needs a complement: replay, the RBF bank, or a gate on a representation that separates tasks rather than surface form. That is an open question for design document §3.
- **Exact match** stays at the floor: newmath 1/64 for both plain and gated (base 0/64); add/subtract 1/32 for both.

**RBF sleep-time correction (frozen weights).** The bank is V10.2's `learn_episode`: one exact gradient write per newmath training document into an RBF memory keyed by the core's final state. The step η and kernel width σ were chosen by a predeclared rule on *train-split* documents disjoint from the write stream: lowest newmath loss with at most 0.01 nats/token harm on old-task (text + math) documents. All 12 cells (96 calibration episodes each):

| σ \ η | 0.1 (V10 default) | 10 | 100 | 1000 |
|---|---|---|---|---|
| 1.0 | −0.044 / 0.000 | −1.213 / **+0.013** | −1.220 / **+0.013** | −1.220 / **+0.013** |
| 0.3 | −0.029 / 0.000 | −1.034 / +0.001 | **−1.045 / +0.001** (chosen) | −1.045 / +0.001 |
| 0.1 | −0.014 / 0.000 | −0.813 / 0.000 | −0.824 / 0.000 | −0.824 / 0.000 |

Each cell is Δ newmath / Δ old tasks, in nats/token.

- Above η ≈ 10, the per-write cap (which limits the largest written row to RMS 0.1), not η, sets the step: effective η ≈ 8.5–8.9.
- σ = 1 is too broad. Queries have norm about 1, so every key gets kernel weight 0.14–1, and the bank harms old tasks by more than the limit.
- σ = 0.1 is harmless but learns less.
- The chosen bank was rewritten from 512 episodes (2,072 rows). Pre-write loss on the stream fell from 2.28 (first 50 episodes) to 1.47 (last 50).
- **The calibration constraint was flawed.** The old-task loss is token-weighted over 24 text and 24 math documents. Text documents carry about 255 scored tokens each and math about 8, so the screen was about 97% text and effectively never saw arithmetic. That is why calibration reported +0.001 old-task harm while dev math forgetting is +0.088 (add/subtract +0.262). The chosen cell would probably not have passed a per-module screen. σ = 0.1's dev behaviour is unknown, and the grid was not rerun.

On dev, with no weight change, the bank:
- gains −1.550 [−1.655, −1.443] on newmath, 88% of plain LoRA's gain;
- leaves text and Sudoku unchanged to within 0.0005 nats;
- forgets far less related arithmetic than either LoRA: add/subtract +0.262 (gated +0.390), multiplication +0.135 (gated +1.342), sorting +0.031 (gated 0.000).

Exact match stays at 0/64 on newmath. The bank is keyed by the concept state and reads only through the concept routes, so it is the one sleep mechanism whose interference is bounded by the concept path's own reach. It is a positive result for the RBF design, provided η and σ are calibrated.

**Consolidation (the sleep step).**
- **Method:** fresh gated adapters (same gate, same initialisation) were distilled from the bank-corrected frozen model on the same newmath stream. The loss is KL to the teacher's answer-token distribution; it fell from 1.35 to 0.016 over 600 updates. The bank was then discarded and the model scored with no bank. Gate open rates are identical to gated LoRA's, because the gate depends only on frozen features.

| Δ vs base (nats/token) | Text | add/sub | mul | sort | Newmath | Sudoku |
|---|---:|---:|---:|---:|---:|---:|
| Plain LoRA | +0.127 | +0.400 | +2.237 | +0.352 | −1.760 | +0.046 |
| Gated LoRA | +0.004 | +0.390 | +1.342 | 0.000 | −1.762 | 0.000 |
| RBF bank (no weight change) | 0.000 | +0.262 | +0.135 | +0.031 | −1.550 | 0.000 |
| Consolidated (bank removed) | 0.000 | +0.464 | +0.182 | +0.001 | −1.581 | 0.000 |

- **Consolidation keeps the bank's gain without the bank.** Against the RBF bank, newmath is −0.031 [−0.051, −0.011], sorting −0.030 and multiplication +0.047.
- **Against gated LoRA trained on labels,** consolidation:
  - learns less of the new task: newmath +0.181 [+0.141, +0.225];
  - forgets far less multiplication: −1.160 [−1.307, −1.008];
  - forgets more add/subtract: +0.074 [+0.041, +0.107].
- **No mechanism protects the closely related add/subtract skill.** Its forgetting is +0.26 to +0.46 under all four methods.
- **On this one test, the sleep pipeline works end to end.** A bank write with frozen weights, then distillation into gated adapters, then a bank reset gives 90% of plain LoRA's new-task gain. Forgetting stays at zero for text, Sudoku and sorting, and multiplication forgetting is +0.18 against +2.24. The related-skill interference that remains is the open problem for §3.
- **Exact match** stays at 0/64 on newmath for the bank and for consolidation. LoRA trained on labels reached 1/64.

## 9. Second seed (same-device pair)

Seed 2 trained C0 and X8 concurrently on CPU, with the same step-based schedule and a different seed. That makes it the one same-device pair for G2. X8 was chosen by the predeclared quick-eval rule (`seed2_choice.txt`), which agreed with the full-evaluation choice in `compare.py`.

| X8 − C0 (dev, 95% CI) | Text | Math | sort | Newmath | Sudoku easy |
|---|---|---|---|---|---|
| Seed 1 (X8 CPU, C0 Metal) | +0.018 [+0.013, +0.023] | +0.029 [+0.011, +0.046] | +0.087 | +0.118 | −0.064 |
| Seed 2 (both CPU) | +0.005 [−0.000, +0.010] | +0.119 [+0.098, +0.139] | +0.208 | +0.774 | +0.022 |

| Same arm, seed 2 − seed 1 | Text | Math | Newmath |
|---|---|---|---|
| C0 | −0.013 [−0.022, −0.005] | −0.044 | −0.475 |
| X8 | −0.026 [−0.034, −0.019] | +0.046 | +0.181 |

Last 1,000 updates of the paired training stream (seed 2): X8 − C0 is +0.006 on text, +0.115 on math and +0.028 on Sudoku easy, against +0.013, +0.034 and −0.063 in seed 1.

- **Replicates:** X8 is behind C0 on text, math, sorting and newmath in both seeds.
- **Does not replicate:** X8's Sudoku advantage.
- **Seed variance:** X8 moved 0.026 nats on text between seeds on the same device (pure seed variance); C0 moved 0.013, which mixes seed and device. The clean figure exceeds the seed-1 X8 − C0 gap, so G2 cannot be decided at a 0.02 resolution from one seed. The direction, X8 behind C0, is consistent.
- **Mechanisms replicate.** X8 seed 2:
  - whole concept path off costs +0.004 on text and +0.085 on math;
  - one visit instead of two costs +0.001;
  - math ridge(F) 0.619 vs ridge(P3) 0.540: the forecast is again a worse predictor than the core's input;
  - codebook RMS 0.998.

## 10. What this means for the sub-10M build

The predeclared rule says do not promote: G3 fails. What follows takes each element you are bullish on, says what the tiny test-bed showed, and says what §3 of the design document has to answer before a 10M run is worth its cost.

| Element | Tiny result | Open question for V10.3 §3 |
|---|---|---|
| Encoder → concept → decoder | The concept path is used for math and Sudoku (switching it off costs X8 +0.100 and +0.131) but is a net cost against C0 on text and math, and its Sudoku gain is layout-dependent | Why does a path the decoder relies on still lose to no path? Next ablation: X with the auxiliary losses at zero. Does the path cost, or do its objectives? The RBF bank changes newmath loss by 1.55 nats *through the concept routes alone, with every weight frozen*. So the decoder does depend heavily on the concept state when that state carries what it needs. The question is what objective puts it there during pretraining |
| Concept prediction | The prediction loss now falls on every family (it rose in V10.1), but the forecast is a worse linear predictor of the next group than the core's input | What does the core add? A predictor that cannot beat its input carries no new information. Options: predict a target the input cannot (a longer horizon, the answer span); or make the decoder's use of the forecast the objective |
| Loop of 2 | The second visit is inert (≤ 0.011 nats across X8 and X4); the V10.3 design probe of the 10M V10.1 checkpoint found the same (visit 2 changes the state 12–17%, cosine 0.985–0.993) | Keep it only if something makes visit 2 differ: input, objective or halting |
| K = 8 vs K = 4 | K = 8 is better on math (−0.069) and newmath (−0.123); text level | Keep K = 8 |
| Explorative modelling | The mixture objective is sound (gradient-tested; responsibilities well defined) but has nothing to select: code main effect 0.1–6%, and mixture, uniform and committed likelihoods agree within 0.002 nats/token. Spread is 4–5× larger on the unseen module | Codes have to produce different futures before exploration can matter. The out-of-distribution spread suggests exploring during adaptation (sleep), not pretraining |
| Gated LoRA | G4 passes: learns the new module as well as plain LoRA, with near-zero forgetting where inputs differ (text, Sudoku, sorting). No protection for the related add/subtract skill; multiplication still loses 1.34 nats | Gate on a representation that separates tasks, not on surface form. Or pair the gate with the RBF bank or replay for related skills |
| RBF sleep-time correction | With frozen weights, the calibrated bank cuts dev newmath loss by 1.55 nats; text and Sudoku are unchanged and math forgets +0.088 (add/subtract +0.262), far less than either LoRA (§8). At V10's default step (η = 0.1) the gain is only −0.04 (value RMS 8×10⁻⁴); the per-write value cap, not η, sets the step once η ≥ 10. With σ = 1 the kernel is nearly global (query norm ~1) and harms old tasks by more than 0.01 | Calibrate η and σ as part of the design, not as defaults, and per family rather than token-weighted. Check the value RMS of V10.1's 10M bank |
| Sleep consolidation (RBF bank → gated adapters, bank reset) | Works end to end: newmath −1.58 (90% of plain LoRA) with zero forgetting on text, Sudoku and sorting, and multiplication +0.18 against plain LoRA's +2.24. Add/subtract still forgets +0.46 | The related-skill case is the remaining problem: no tested method protects it. Candidates: replay of old-task rows during distillation, or gate on the concept state rather than sub-layer inputs |
| Codebook without decay; learned forecast map | Both work as intended (codebook RMS kept; map RMS grew 2.2×) | Keep |

**Specific fixes to carry forward regardless:**
- Exempt the codebook from weight decay.
- Log exploration only on candidate updates. Zeros on other updates produced the misleading V10.1 chart, and the same trap appeared here.
- Shift-control any symbolic family with a fixed layout before citing it.
- Use per-module math results: aggregate math hides that sorting and multiplication move in opposite directions.

## 11. Limits

- **One seed for most comparisons.** §9 adds one same-device seed pair for X8 against C0. Same-device seed variance in text NLL is 0.026 for X8, larger than G2's margin. G1 (X8 against A) has one seed.
- **Devices.** Seed-1 X8 and A trained on CPU, X4 and C0 on Metal; G1 is same-device. CPU and Metal agree per step to 5×10⁻⁷, but trajectories diverge.
- **Scale.** About 0.5M dense parameters, 19M input tokens, 6,000 updates. Tiny numbers falsify mechanisms; they do not predict 10M magnitudes. Exact match is at the floor, so math conclusions rest on NLL.
- **Not run:**
  - C0 at equal wall time; it would have about 1.6× the updates and only widen G2's gap;
  - the Engram on/off throughput benchmark;
  - an X arm with the auxiliary losses at zero.
- **Exploration diagnostics** use 24–256 boundaries per family.
- **RBF calibration** screened old-task harm token-weighted, which in practice meant text only (§8).
- **Newmath** is a single held-out module whose prompt vocabulary is disjoint from the pretraining families. That is the favourable case for GMM gating.

## 12. Files

- `v10_3_tiny/`: `config.py`, `model.py`, `losses.py`, `train.py`, `evaluate.py`, `compare.py`, `sleep.py`, `audit_data.py`, `paired_training.py`, `grid_shift8.py`, `report.py`, `tests.py`.
- `ARCHITECTURE.md`: predeclared rules, timestamped.
- Outputs: `tests_result.json`, `data_audit.json`, `comparison.json`, `paired_training*.json`, `grid_shift8.json`, `REPORT_TABLES.md`.
- `runs/<arm>_s<seed>/`: `manifest.json`, `train.jsonl`, `quick_eval.json`, `final.pt`, `evaluation.json`.
- `runs/sleep_X8_s1/sleep.json`.

## 13. Process

- **Timeline:**
  - about 15:15: request;
  - by 15:46: package, data re-tokenization, the 26 tests, and the smoke and throughput runs;
  - 15:46–17:05: the four seed-1 arms trained concurrently;
  - 17:05–18:20: evaluations, the sleep tests and the second seeds. The session paused for about 55 minutes between roughly 17:25 and 18:20; the background jobs completed during the pause;
  - about 18:30: report finished.
- **Dropped,** each with a timestamped entry in `v10_3_tiny/ARCHITECTURE.md`:
  - C0 at equal wall time: no spare memory;
  - the Engram on/off throughput benchmark: the machine was never idle;
  - an X arm with the auxiliary losses at zero: identified only once results were in.
- **Rule handling:** every predeclared rule was applied as written. Where a rule turned out to be the wrong test (N2's variance ratio for X, N5's winner cross-entropy under a mixture objective, the token-weighted RBF screen), the report says so next to the result and does not change the verdict.
- **Advisor consultations:** 12 for the tiny test-bed, against the 40+ requested. Wall-clock time was the constraint.
