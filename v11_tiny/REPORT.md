# V11 tiny objective test — report

**Status:** final. Methods (§1–§5) were written before the full evaluations landed. Every result is read from `comparison.json`, `training_summary.json`, `offsets_extra.json`, `grad_share.json` and `bench_10m.json`. Advisor consultations are logged in `ADVISOR_LOG.md`.

## Bottom line

1. **Every pre-registered gate fails on K1** (G1, G2, G3, G4, alignment). G5 passes only against a plain model with no Engram; against the fair control ORDE, K1 is worse by +0.039 on text and +0.056 on math (seed 1).
2. **Two training artefacts were found and removed mid-run.** Training-only modules (bridge, then probe) sat inside the trunk's gradient clip. With both removed (seed 1):
   - the code-free topology and the latent objective are within about 0.01 of C0 on text (+0.005 to +0.011) and 0.03 on sorting;
   - they still fail the registered G2 margins by those amounts, which are below C0's own seed spread;
   - the latent objective brings the concept route alive but adds nothing on NLL.
3. **The bridge is the only term that moves anything, and it moves it the wrong way.** It costs about +0.015 on text and +0.41 on math (mostly sorting, plus +0.10 on multiplication), compresses the pooled states (text PR about 6) and cuts the decoder's use of the concept path. Its gradient into the core is 2.6–2.9× NTP's. The design's prediction of a text gain through the bridge is falsified.
4. **The V10.3 recipe (X8)** costs about 0.13 on math against C0 at two seeds, matching V10.3's Metal seed (+0.119; its CPU seed was +0.029). It shows a period-8 Sudoku layout dependence. So does the trained V11 concept path in the separate-clip arms (one seed each).
5. **10M-shape benchmark on the Air's GPU** (3-minute solo bursts): K1 3,298 tokens/s, against the 3,855 tokens/s sustained V10 prior. A 16-hour run would see about 190M tokens, about 35 per deployed parameter.

**Recommendation:** don't launch the 16-hour V11 run as designed. Drop or retune the bridge, apply the clip and head fixes, and repeat this tiny test first. Most of these results come from one or two seeds, so they are directional.

This is the "smallest worthwhile experiment" of `v11.md` §6.6. It asks whether V11's change of objective makes the concept state more useful than V10.3's. The test bed is the V10.3 tiny shape (width 56; ~0.5M dense parameters) on the inherited V10.3-tiny data and stream.

## 1. What was run

| Arm | What it is | Dense (deployed) | Engram table |
|---|---|---:|---:|
| C0 | concept-free control on the same backbone (inherited `v10_3_tiny.model.C0`) | 311,704 | 1,050,080 |
| ORD | plain transformer, 11 full-causal blocks, tied head, **no Engram** | 475,664 | — |
| ORDE | ORD + the inherited Engram after block 1 (added mid-run, seed 1 only; see §5) | 483,224 | 1,050,080 |
| X8 | V10.3 recipe (inherited `V103Tiny`, K = 8, forecast token CE + mixture every 4th step) | 474,364 | 1,050,080 |
| NOAUX | V11 topology (code-free two-visit core, K = 1), NTP only (+ detached probe) | 472,324 | 1,050,080 |
| K1NB | NOAUX + latent next-concept objective (`L_v1`, `L_v2` at K = 1, `L_vq`) | 472,324 | 1,050,080 |
| K1 | K1NB + concept-anchored NextLat bridge (training-only) | 472,324 | 1,050,080 |
| K1SC | K1 with separate gradient-clip groups for trunk / bridge / probe (diagnostic, seed 1 only; see §5) | 472,324 | 1,050,080 |
| NOAUXSC | NOAUX with the probe in its own clip group (diagnostic, seed 1 only; see §5) | 472,324 | 1,050,080 |
| K1NBSC | K1NB with the probe in its own clip group (diagnostic, seed 1 only; see §5) | 472,324 | 1,050,080 |

- **Seeds:** 1 and 2 for the six pre-registered arms. C0 also has seed 3, the third seed for a promotion claim. K1SC, ORDE, NOAUXSC and K1NBSC were added during the run as seed-1 diagnostics (17 trained runs in all, each 6,000 updates).
- **Schedule (inherited):** 6,000 updates, batch 16, context 256, AdamW (0.9, 0.95) lr 1e-3, warmup 200 then cosine to 10%, dense clip 1.0, CPU SparseAdam Engram at 5× LR. One `Corpus('train', seed)` batch per update, so every arm with the same seed saw the identical sequence of documents (checked by a per-step stream hash, §4).
- **Device:** Metal for every run, so no run was split across CPU and Metal as in V10.3's seed 1.

## 2. Throughput choices

- **Concurrency:** three runs at a time on Metal, two CPU threads each. Measured over 200-update smokes:

  | Processes | Aggregate steps/s |
  |---:|---:|
  | 1 | 3.4 |
  | 3 | 6.0 |
  | 4 | 4.8 (CPU-bound on the 4 performance cores) |

- **Data prefetch:** a background thread prepares the next batch and its Engram hash indices. The Corpus call order is unchanged.
- **Optimizer:** `AdamW(foreach=True)`. It differs from the inherited trainer only at rounding level. With `foreach` off, the new trainer reproduces `v10_3_tiny.train` bitwise for C0 and X8 over 12 updates.
- **Wall-clock caveat:** runs were co-scheduled, so elapsed time depends on which arms shared the GPU. Equal-wall-time statements use solo smoke step ratios (§7), not elapsed time.
- **Drift, not established as thermal:**
  - Some co-scheduled runs slowed by 8–20% from their first to last 500 updates (K1_s2 0.52 → 0.62 s; ORD slower with constant partners).
  - Others sped up as their partners finished (C0, K1NB).
  - The drift is confounded with concurrency changes. `pmset -g therm` recorded no thermal or performance warning at any of five checkpoints (`runs/thermal.json`), and the 3-minute solo benchmark windows show no drift (K1 0.711 → 0.703 s).
  - Per-run first/last medians are in `training_summary.json`.

## 3. Pre-registered reading (written before the full evaluations)

- **K1SC vs K1NB:**
  - K1SC ≈ K1NB → K1's cost was the clip artefact (§5), and the bridge itself is neutral at this scale.
  - K1SC still worse than K1NB → the bridge's gradient into the trunk (2–3× NTP's on pooler and core, `grad_share`) is itself the cost. The λ screen of v11.md §4.6 is then the next experiment.
- **K1NB vs NOAUX** decides the latent objective. Caveat: at `λ_c = 0.1` it delivers about 0.4% of NTP's gradient to the encoder, but about 20% on the PQ head and the whole codebook fit.
- **NOAUX's forecast-route deltas** (`aligned_off`, `reader_forecast_off`): if they are ≈ 0 while K1NB's are not, NOAUX's math cost is a dead forecast route (head and codebook trained by NTP only), not a topology effect.
- **One-change chain:** C0 → NOAUX → K1NB → K1 (and K1SC). X8 differs from every V11 arm in two ways: its objective and its K = 8 codes at the anchor. It is the recipe reference, not a one-change control.
- **ORD vs ORDE:** ORD lacks the 1.05M-row Engram that every other arm carries outside its dense count. So G5 (vs ORD) is **not** a plain-model test; K1 − ORDE is.

## 4. Correctness

- `tests.py`: **59/59 pass** (`tests_result.json`). Coverage:
  - parameter counts for every arm;
  - every stop-gradient contract by `autograd.grad`;
  - K = 1 Gaussian closed form, its σ stationarity and the σ floor;
  - boundary-sampler determinism;
  - `visits=1` and `use_visit1` semantics;
  - buffers frozen in `eval()`;
  - bridge/probe removal leaving logits identical;
  - each intervention changing the logits;
  - unknown forward kwargs raising;
  - future-token perturbation at cuts 1/7/8/9/15/16/17/33 (exactly 0.0) for ORD, ORDE, NOAUX and K1;
  - prefix-length and right-padding invariance (≤ 5×10⁻⁷);
  - bridge edge cases;
  - CPU/Metal loss and gradient parity (≤ 5×10⁻⁷).
- **Stream identity** (`training_summary.json`): per-step stream hashes are identical across all 10 seed-1 arms, all 6 seed-2 arms, and C0_s3, 6,000 updates each.
- **Trainer parity:** with `foreach` off, 12 updates of C0 and X8 match `v10_3_tiny.train` exactly (zero loss and gradient-norm difference).
- **Not tested:** exact checkpoint resume. Runs were short and uninterrupted.
- **Methods note:** while the runs were live, two progress lines were attributed to the wrong run in a status message (K1 seed 1 vs X8 seed 1). Every number in this report is read from `quick_eval.json`, `train.jsonl` or `evaluation.json`.

## 5. Found during the run: the inherited global clip and the bridge

The bridge's own parameters carry a large gradient. At the final K1 checkpoint, its KL term alone has norm 3.0 on bridge parameters, against NTP's 1.2 on the encoder. The inherited trainer clips **all** dense parameters jointly at 1.0, so in K1 the bridge inflates the global norm. Median pre-clip norm on text batches, per 1,000 updates (seed 1):

| Run | 0–1k | 1–2k | 2–3k | 3–4k | 4–5k | 5–6k |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.58 | 0.66 | 0.69 | 0.72 | 0.74 | 0.77 |
| X8 | 0.59 | 0.71 | 0.75 | 0.79 | 0.81 | 0.84 |
| K1NB | 0.74 | 0.85 | 0.86 | 0.89 | 0.93 | 0.96 |
| K1 | 1.00 | 1.73 | 2.10 | 3.10 | 4.00 | 4.40 |
| K1SC (trunk group only) | 0.55 | 0.67 | 0.71 | 0.75 | 0.77 | 0.79 |

**Why this matters under AdamW: reweighting, not shielding.** Adam normalizes each parameter's update by its running gradient scale, so a per-step scalar on the whole group nearly cancels. Two consequences:

- The within-step composition of the trunk's gradient (NTP versus bridge) is identical in K1 and K1SC.
- What the clip changes is the effective weight of each step, about 1/max(1, pre-clip norm). That weight differs by family.

Median effective step weight over the last 1,000 updates (seed 1; K1SC uses its trunk group):

| Run | Text | Math | Sudoku easy | Sudoku hard |
|---|---:|---:|---:|---:|
| C0 | 1.00 | 0.37 | 0.33 | 0.30 |
| X8 | 1.00 | 0.38 | 0.55 | 0.53 |
| NOAUX | 0.75 | 0.14 | 0.42 | 0.40 |
| K1NB | 1.00 | 0.36 | 0.49 | 0.42 |
| K1 | **0.23** | 0.25 | 0.59 | 0.55 |
| K1SC | 1.00 | 0.48 | 0.71 | 0.69 |

Two points follow:

1. **The inherited clip already reweights the mixture in every arm, C0 included.** Math batches are clipped in about 99% of steps, so a math step counts about ⅓ as much as a text step. Arms differ only in how much.
2. **K1 down-weighted text about 4× relative to K1SC**, so it effectively upweighted math about 2× and Sudoku about 3.5×. That matches the K1 − K1SC sign pattern exactly (training stream, seed 1: text +0.030, math −0.143, Sudoku −0.136). **K1 − K1SC measures the reweighting, not the bridge.**

The same flaw hit **NOAUX** through its detached probe. The probe's CE stays at about 6 nats, because NOAUX's concept route is dead, so its gradient is large. Its pre-clip math norm was 7.0 against C0's 2.7, and 65% of its text steps were clipped. **NOAUXSC** and **K1NBSC** (separate clip groups, seed 1) were therefore added to give an artefact-free chain C0 → NOAUXSC → K1NBSC → K1SC.

Over updates 200–500, all the separate-clip arms sit in C0's regime (text / math pre-clip trunk norm):

| Run | Text | Math |
|---|---:|---:|
| C0 | 0.55 | 1.36 |
| X8 | 0.56 | 1.61 |
| NOAUXSC | 0.53 | 1.29 |
| K1NBSC | 0.54 | 1.50 |
| K1SC | 0.55 | 1.47 |
| NOAUX | 1.13 | 10.95 |

**Pre-registered at consultations 13–14, before either arm finished:**

- NOAUXSC ≈ C0 on math → NOAUX's deficit was the artefact, and the code-free topology is free when unused.
- NOAUXSC still worse → an unused concept path costs even when the decoder ignores it.
- K1NBSC − NOAUXSC is the clean latent-objective test.
- K1SC − K1NBSC is the clean bridge test.
- K1NB − K1NBSC and NOAUX − NOAUXSC are the artefact costs.

§7 records which branch each landed on.

Patch for v11.md: *training-only modules (bridge, probe) get their own clip group, and the deployed trunk's clip never sees them.*

**Methods note:** every paired training-stream comparison asserts per-step stream equality, and `training_summary.py` skips pairs of unequal length. An ad-hoc check run before K1SC had finished, with offset windows, produced nonsense gaps (±1 nat). Misaligned pairs look like that.

(Results sections follow.)
## 6. Results: the pre-registered chain (inherited clip, seeds 1 and 2)

All numbers are dev panels from `evaluation.json` and `comparison.json`, in nats per scored token. Δ = NLL(first) − NLL(second), with a 95% paired document bootstrap (2,000 draws), per seed and never pooled.

### 6.1 Gate verdicts (pinned on K1 before the run)

| Gate | Criterion (v11.md §7.2) | Seed 1 | Seed 2 | Verdict |
|---|---|---|---|---|
| G1: latent objective beats V10.3 | K1 − X8 math CI hi ≤ −0.02, ≥ 2 modules, text hi ≤ +0.01 | math +0.096 [+0.083, +0.110], text +0.042 | math +0.030 [+0.015, +0.046], text +0.044 | **fail** |
| G2: non-inferior to C0 | K1 − C0 text hi ≤ +0.01; every math module hi ≤ +0.02 | text +0.056; sort +0.365 | text +0.051; sort +0.225 | **fail** |
| G2b: topology vs objectives | informational, no threshold | NOAUX − C0 math +0.369; K1 − NOAUX math −0.134 | +0.421; −0.267 | n/a. **Both numbers are the probe-in-clip artefact** (NOAUX − NOAUXSC math +0.378, §7); with it removed, NOAUXSC − C0 math is −0.012 |
| G3: concept state used | concept-off math lo ≥ +0.02 **and** ridge(forecast) ≤ ridge(input) − 0.02 | off +0.049 [+0.041, +0.058] ✓; forecast 0.529 vs input 0.493 ✗ | off +0.036 ✓; forecast 0.633 vs input 0.490 ✗ | **fail** (forecast clause) |
| G4: bridge earns its place | K1 − K1NB text hi ≤ −0.005 or math hi ≤ −0.01, no family > +0.005 | text +0.054, math +0.184 | text +0.050, math −0.127, Sudoku −0.08 | **fail** |
| G5: vs ordinary | K1 − ORD text hi ≤ +0.01 | text −0.308 | text −0.318 | passes, but **not a plain-model test**: ORD has no Engram. Against ORDE (seed 1): K1 − ORDE text **+0.039 [+0.032, +0.046]**, which would fail |
| Alignment | each concept arm − C0, max−min over r = 0..7 ≤ 0.01 | X8 +0.07/+0.36; K1 −0.01/+0.01 | K1 −0.03/+0.06 | fails as registered; **uninformative** (§6.6) |

Seed 3 exists only for C0 (it supplies the noise floor below). No gate verdict uses it.

### 6.2 Absolute dev NLL

| Run | Text | Math | add/sub | mul | sort | newmath | Sudoku easy | Sudoku hard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0_s1 | 3.365 | 1.055 | 1.350 | 1.854 | 0.617 | 3.018 | 0.896 | 0.893 |
| C0_s2 | 3.351 | 1.047 | 1.389 | 1.848 | 0.600 | 2.584 | 0.941 | 0.941 |
| C0_s3 | 3.358 | 1.182 | 1.369 | 1.767 | 0.867 | 2.766 | 0.899 | 0.896 |
| ORD_s1 | 3.729 | 1.334 | 1.376 | 1.817 | 1.096 | 2.872 | 1.066 | 1.066 |
| ORD_s2 | 3.720 | 1.379 | 1.375 | 1.835 | 1.161 | 3.030 | 1.198 | 1.199 |
| ORDE_s1 | 3.382 | 1.235 | 1.331 | 1.790 | 0.951 | 3.049 | 0.783 | 0.783 |
| X8_s1 | 3.379 | 1.194 | 1.368 | 1.770 | 0.886 | 2.833 | 0.898 | 0.877 |
| X8_s2 | 3.358 | 1.171 | 1.386 | 1.805 | 0.827 | 2.938 | 0.936 | 0.934 |
| NOAUX_s1 | 3.357 | 1.424 | 1.377 | 1.936 | 1.188 | 2.975 | 0.971 | 0.965 |
| NOAUX_s2 | 3.330 | 1.469 | 1.421 | 1.890 | 1.276 | 2.716 | 1.049 | 1.033 |
| K1NB_s1 | 3.367 | 1.107 | 1.342 | 1.771 | 0.745 | 3.021 | 0.886 | 0.897 |
| K1NB_s2 | 3.351 | 1.328 | 1.368 | 1.862 | 1.065 | 2.949 | 0.970 | 0.957 |
| K1_s1 | 3.421 | 1.290 | 1.360 | 1.908 | 0.982 | 3.055 | 0.797 | 0.798 |
| K1_s2 | 3.402 | 1.201 | 1.401 | 1.910 | 0.825 | 3.136 | 0.888 | 0.884 |

**X8 reproduces V10.3** within V10.3's own seed spread:

| X8 | Here | V10.3 seed 1 | V10.3 seed 2 |
|---|---:|---:|---:|
| Text | 3.379 / 3.358 | 3.382 | 3.356 |
| Math | 1.194 / 1.171 | 1.117 | 1.162 |

The inherited exploration diagnostic also replicates: mixture, committed code and uniform prior all lie within 0.002 nats, and the codes' main effect is 0.3%.

### 6.3 The noise floor: C0's three seeds

- **Sorting decides every math gate.** C0's sort NLL spans 0.600–0.867 across seeds (0.27). Add/sub sits within ±0.04 across every arm and seed. Mul carries a small, consistent cost in the V11 arms (K1 +0.05 to +0.14 vs C0). Every arm's math deficit against C0 is mostly sorting, the module whose answers span about 2.5 eight-token groups (the same locus V10.3 reported).
- **The G2 module margin (+0.02) is about 13× smaller than the baseline's own seed-to-seed spread on that module.** Verdicts stand as registered. But this is a design finding for v11.md §7.2: a per-module margin needs either more seeds or a larger sort panel.
- **Newmath:** C0 spans 2.58–3.02, so no cross-arm newmath claim survives. (An earlier status message called newmath X8's best relative result. That is withdrawn: X8's 2.83 and 2.94 sit inside C0's range.)

### 6.4 What each objective does to the concept state

| Run | Concept off, math Δ | ridge input | ridge state | ridge forecast | Forecast var. ratio | Pooled PR text / math |
|---|---:|---:|---:|---:|---:|---:|
| X8_s1 | +0.093 | 0.609 | 0.800 | 0.699 | — | 7.8 / 10.1 |
| X8_s2 | +0.110 | 0.556 | 0.567 | 0.621 | — | 8.1 / 9.6 |
| NOAUX_s1 | **+0.004** | 0.771 | 0.755 | 0.958 | **0.001** | 18.5 / 9.4 |
| NOAUX_s2 | +0.003 | 0.632 | 0.842 | 0.955 | 0.009 | 19.7 / 7.3 |
| K1NB_s1 | **+0.125** | 0.621 | 0.630 | 0.725 | 0.087 | 21.1 / 11.9 |
| K1NB_s2 | +0.074 | 0.572 | 0.571 | 0.636 | 0.183 | 25.1 / 10.6 |
| K1_s1 | +0.049 | 0.493 | **0.460** | 0.529 | 0.155 | 5.6 / 8.1 |
| K1_s2 | +0.036 | 0.490 | **0.465** | 0.633 | 0.107 | 6.7 / 10.4 |
| K1SC_s1 | +0.045 | 0.569 | **0.492** | 0.735 | 0.147 | 6.2 / 7.9 |

FVUs are normalized by each family's own target variance (for X8, the inherited `fvu_v103` on its own targets). The training-log "MSE" divides by the global variance instead, so the two are not comparable. Absolute FVUs are not comparable across arms either, because each arm's pooled target is different. Within-arm orderings are.

Findings, each labelled post-hoc:

1. **Without an auxiliary loss the concept route dies** (NOAUX; NOAUXSC confirms it with concept-off +0.008 and variance ratio 0.008). Switching off the whole concept path costs 0.004 nats on math. The forecast's variance is 0.1–1% of the target's, and the detached probe stays flat at about 6 nats throughout training. The code-free core is only used when something trains it.
2. **The latent objective makes the decoder lean on the concept state the most of any arm** (K1NB, concept-off +0.07 to +0.12). It does not compress the pooled states (PR 21–25 on text, same as NOAUX), and its core state is no better a predictor than its input: state ≈ input in both K1NB seeds, state worse than input in K1NBSC (0.78 vs 0.64). **But the NLL gain it appears to bring over NOAUX was the artefact.** With both clips fixed, K1NBSC − NOAUXSC is about 0 (§7): the route comes alive and the decoder leans on it, but it substitutes for trunk computation rather than adding to it.
3. **The bridge does its proximate job, and the information dies at the head.** Only in the bridge arms (K1 both seeds, K1SC) is the core state a better linear predictor of the next pooled target than its input (by 0.025–0.08). Every non-bridge arm has state ≥ input. But:
   - the PQ head's forecast loses that information: ridge(forecast) ≫ ridge(state), and the forecast's variance is 11–16% of the target's, a shrunk, mean-regressed prediction;
   - the decoder uses the concept path less than without the bridge (concept-off +0.04–0.05 vs K1NB's +0.07–0.12);
   - it compresses the pooled states to PR 5.6–6.7 on text.
   K1SC shows the same signature with the clip artefact removed, so it is the bridge's own effect.
4. **Two things compress the pooled states; two don't.** In the training log (text batches, median per 1,000 updates), the bridge arms' PR falls from the start while the others' rises. The two groups have separated by updates 1–2k:

   | Run | 0–1k | 1–2k | 2–3k | 3–4k | 4–5k | 5–6k |
   |---|---:|---:|---:|---:|---:|---:|
   | K1_s1 | 9.4 | 8.7 | 7.5 | 7.1 | 6.8 | 6.6 |
   | K1SC_s1 | 9.7 | 8.1 | 8.7 | 7.9 | 7.3 | 7.0 |
   | K1NB_s1 | 14.1 | 17.2 | 18.1 | 21.4 | 20.5 | 21.1 |
   | K1NBSC_s1 | 14.4 | 17.1 | 19.1 | 22.5 | 21.8 | 22.8 |
   | NOAUXSC_s1 | 13.3 | 10.9 | 12.3 | 17.6 | 16.9 | 18.6 |

   Final dev-panel values: The bridge and V10.3's K = 8 code recipe compress them (text PR 5.6–8.1). The code-free topology and the latent objective do not (18.5–25.1). Stated for text and math only: every arm sits at 3–5 on Sudoku (fixed layout), K1 seed 2 apart (9).
5. **G3's forecast clause fails in every concept arm.** The forecast head is the weakest link whatever trains it. A shared-σ Gaussian NLL through a convex PQ-codeword combination regresses the forecast to the mean, and a ridge refit on the state recovers what the head discards (patch list, §10).
6. **Visit 2 is null.** NLL gain 0.001–0.002, FVU gain ≤ 0.004, `visit1_for_visit2` Δ ≈ 0 on text and math. That is Tier 2.5's answer at this scale.

### 6.5 Text

Text Δ vs C0 (dev panel, paired bootstrap, CIs about ±0.005). The two seed columns use the inherited clip; the right-hand column is the separate-clip chain (dev panel, seed 1).

| Arm | Seed 1 | Seed 2 | Clean chain, seed 1 |
|---|---:|---:|---:|
| NOAUX / NOAUXSC | −0.008 | −0.021 | +0.011 |
| K1NB / K1NBSC | +0.002 | +0.001 | +0.005 |
| X8 | +0.013 | +0.007 | (no artefact; +0.013 / +0.007) |
| K1 / K1SC | +0.056 | +0.051 | +0.019 |

- **NOAUX's apparent text advantage was the same artefact.** Its effective step weights were text 0.75 vs math 0.14, so the clip upweighted text.
- **Clean ordering:** C0 < K1NBSC ≈ NOAUXSC < X8 < K1SC.
  - The code-free topology costs about 0.005–0.011.
  - The latent objective costs about 0.
  - The bridge costs about 0.015 on its own, and about 0.05 with the clip artefact.
- The design's prediction that "any text gain comes through the bridge" is **falsified** both ways.

### 6.6 Alignment and layout controls (`offsets_extra.json`)

**Math: the filler-offset curve is distraction, not alignment.** In `offsets_extra.json`, r filler tokens are inserted after BOS (seed 1). r = 8 and r = 16 keep the 8-token group alignment of r = 0, so a residue effect would make NLL drop back at those offsets.

| Run | r = 0 | 1 | 4 | 7 | **8** | 9 | **16** |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 1.055 | 1.087 | 1.251 | 1.306 | 1.367 | 1.369 | 1.544 |
| ORD (no 8-token grouping anywhere) | 1.334 | 1.357 | 1.515 | 1.604 | 1.673 | 1.699 | 1.952 |
| X8 | 1.194 | 1.246 | 1.390 | 1.519 | 1.576 | 1.606 | 1.637 |
| NOAUX | 1.424 | 1.442 | 1.472 | 1.518 | 1.540 | 1.550 | 1.603 |
| K1NB | 1.107 | 1.179 | 1.286 | 1.371 | 1.396 | 1.469 | 1.529 |
| K1 | 1.290 | 1.361 | 1.431 | 1.531 | 1.544 | 1.528 | 1.594 |
| K1SC | 1.455 | 1.469 | 1.560 | 1.683 | 1.733 | 1.770 | 2.052 |

- No arm recovers at r = 8 or 16.
- ORD, which has no group structure, follows the same rising curve. It is the negative control: filler distracts a plain transformer identically, so the curve has no period-8 component in any arm.
- The pre-registered alignment gate (max−min over r = 0..7 vs C0) fails as registered and is **uninformative**:
  - it measures distraction;
  - the seed scatter is about 30× its 0.01 margin;
  - it is confounded with r = 0 accuracy (dead-path NOAUX scores "less sensitive" than C0);
  - it has no CI.
- Newmath curves are non-monotone and go in different directions per arm (X8 worsens by 0.55; NOAUX and K1SC improve with filler). They support no alignment reading, and C0's own 2.58–3.02 seed range rules out interpretation anyway.

**Sudoku: X8 carries a period-8 layout dependence; the V11 arms do not.** The table gives degradation in nats when the grid is shifted by a header of 4 or 8 tokens. Shift 8 restores the 8-alignment. Each panel has 64 documents; there is no bootstrap on the within-model shift difference.

| Arm | Seed 1, shift 4 → 8 | Seed 2, shift 4 → 8 | Seed 3, shift 4 → 8 |
|---|---|---|---|
| C0 | 0.148 → 0.191 | 0.044 → 0.150 | 0.116 → 0.125 |
| ORD | 0.078 → 0.072 | 0.081 → 0.093 | |
| ORDE | 0.492 → 0.611 | | |
| **X8** | 0.164 → **0.031** | 0.165 → **0.095** | |
| NOAUX | 0.162 → 0.152 | 0.029 → 0.074 | |
| K1NB | 0.140 → 0.158 | 0.091 → 0.104 | |
| K1 | 0.183 → 0.169 | 0.179 → 0.258 | |
| K1SC | 0.177 → **0.107** | | |
| NOAUXSC | 0.164 → 0.240 | | |
| K1NBSC | 0.221 → **0.090** | | |

(Sudoku easy shown; Sudoku hard is within ±0.02 of it in every cell.)

- **The ECD backbone has a header-shift cost of about 0.04–0.19 that is not periodic.** C0, which has no concept path, has it, and it is seed-variable.
- **The shift cost falls when 8-alignment is restored in three arms:**
  - X8, at both seeds (by 80% and 42%);
  - K1SC and K1NBSC, one seed each (by 40% and 59%).

  It does not fall in K1NB or K1 (two seeds each), NOAUX or NOAUXSC, C0, or ORD.
- So the period-8 component appears where a *trained, live* concept path reads 8-token groups: V10.3's codes, and the V11 concept path once the clip artefacts are removed. It is absent where that path is dead (NOAUX) or has no group structure at all (C0, ORD). The inherited-clip K1NB and K1 don't show it, which is unexplained at two seeds.
- X8's shift-0 Sudoku NLL equals C0's (0.898 vs 0.896). This is extra alignment-specific *sensitivity*, not an alignment-dependent *advantage*.
- This is the V9-style format dependence the data-bias constraint was written to catch. **It is not V10.3-specific, and v11.md's random-prefix training (§7.1) is needed for any V11 run.**

- The ordinary transformer is layout-sensitive only once it has the Engram (ORD 0.08, ORDE 0.49–0.61, growing with shift, not periodic). That is observed, not explained.
- **Limit:** no arm was trained with random prefixes (v11.md §7.1 pairs that with this evaluation; this run skipped it to keep a clean X8/C0 comparison). So these controls detect format dependence but cannot show whether training would remove it.

### 6.7 Plain-model control (seed 1)

| Pair | Text | Math | Sudoku easy |
|---|---:|---:|---:|
| ORDE − ORD (value of the Engram to a plain model) | **−0.347** | −0.100 | −0.283 |
| C0 − ORDE | −0.016 | **−0.180** | +0.113 |
| K1 − ORDE | +0.039 | +0.056 | +0.014 |
| K1NB − ORDE | −0.015 | −0.128 | +0.103 |
| X8 − ORDE | −0.003 | −0.040 | +0.116 |

ORD's 0.3-nat text deficit is the missing n-gram table. With it restored, the ordinary transformer is within 0.016 of C0 on text, and K1 is worse than ORDE on text and math. The ECD backbone (C0) beats an ordinary transformer of 1.5× its dense size on math by 0.18 nats, mostly sorting (−0.33).

### 6.8 Generation

Greedy exact-answer accuracy is at the floor for every arm: 0–2 of 96 math items, 0–1 of 64 newmath, 0 of 8 Sudoku of each kind. The limit is the answer's token count + 64 (the inherited trainer used 24). X8 generates without committed codes, which is exact because at evaluation the mixture reduces to the committed code. Truncations are counted separately and occur only on newmath (0–13 of 64). No arm separates on generation at this scale.
## 7. Results: the separate-clip chain (seed 1)

Chain C0 → NOAUXSC → K1NBSC → K1SC: each step is one change, and no arm has a training-only module inside its trunk clip. The effective step weights (last 1,000 updates) match C0's:

| Run | Text | Math | Sudoku easy |
|---|---:|---:|---:|
| C0 | 1.00 | 0.37 | 0.33 |
| NOAUXSC | 1.00 | 0.36 | 0.36 |
| K1NBSC | 1.00 | 0.37 | 0.52 |
| K1SC | 1.00 | 0.48 | 0.71 |

**Training stream, last 1,000 aligned updates** (stream identity asserted per step):

| Step in the chain | Text | Math | Sudoku easy | Sudoku hard |
|---|---:|---:|---:|---:|
| NOAUXSC − C0 (topology) | +0.010 | −0.012 | +0.019 | +0.019 |
| K1NBSC − NOAUXSC (latent objective) | −0.005 | +0.007 | +0.004 | +0.006 |
| K1SC − K1NBSC (bridge) | +0.015 | **+0.402** | +0.005 | −0.005 |
| NOAUX − NOAUXSC (probe-clip artefact) | −0.022 | **+0.378** | +0.056 | +0.054 |
| K1NB − K1NBSC (probe-clip artefact) | −0.003 | +0.066 | −0.036 | −0.035 |
| K1 − K1SC (bridge-clip artefact) | +0.030 | −0.143 | −0.136 | −0.134 |

**Dev panel:**

| Run | Text | Math | add/sub | mul | sort | Concept off (math) | ridge state / input / forecast | Text PR |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| C0_s1 | 3.365 | 1.055 | 1.350 | 1.854 | 0.617 | — | — | — |
| NOAUXSC_s1 | 3.376 | 1.031 | 1.333 | 1.782 | 0.615 | +0.008 | 0.944 / 0.777 / 0.860 | 18.5 |
| K1NBSC_s1 | 3.371 | 1.049 | 1.349 | 1.776 | 0.645 | **+0.109** | 0.781 / 0.641 / 0.767 | 21.7 |
| K1SC_s1 | 3.384 | 1.455 | 1.364 | 1.879 | 1.270 | +0.045 | 0.492 / 0.569 / 0.735 | 6.2 |

**Which branch each landed on:**

1. **NOAUXSC ≈ C0 → the topology is free when unused** (no detectable math difference at one seed). Against C0 on the dev panel:

   | Pair | Text | Math | Sort | G2 margins (text ≤ +0.01, modules ≤ +0.02) |
   |---|---|---|---|---|
   | NOAUXSC − C0 | +0.011 [+0.005, +0.016] | −0.024 | −0.002 | text exceeds |
   | K1NBSC − C0 | +0.005 [−0.000, +0.011] | −0.006 | **+0.028 [+0.006, +0.050]** | sort exceeds |

   Both clean arms fail the registered G2 margins by those amounts. "No detectable difference" and "passes G2" are different claims, and only the first holds. C0's own sort NLL moves 0.27 between seeds (§6.3).
2. **K1NBSC − NOAUXSC ≈ 0 → at λ_c = 0.1 the latent objective is a null on NLL.** It does change the model: the concept route comes alive (concept-off +0.109 vs +0.008), forecast variance rises (0.068 vs 0.008) and PR is unchanged. The decoder routes through the concept state and gains nothing net. The concept path substitutes for trunk computation; it does not add to it. (Post-hoc, one seed.)
3. **K1SC still worse than K1NBSC → the bridge's own gradient is the cost.** Text +0.015, math +0.40: mostly sorting (+0.63; 1.27 vs 0.645), plus multiplication +0.10 [+0.08, +0.13]. Mul is the one module that moves consistently in every bridge arm. K1SC's math deficit is a single-seed magnitude. Its direction holds against every reference (K1NBSC, K1NB, X8, C0).
4. **Both earlier math stories were the probe artefact, seen from either side:** "NOAUX pays for the topology" and "the latent objective rescues math" (§6.1, G2b). NOAUX − NOAUXSC is +0.378 on math.
5. **K1NBSC vs X8:** math −0.138 at seed 1, but that is **X8's loss against C0**, not V11's gain. X8 − C0 is +0.139 / +0.124 here and +0.029 (X8 on CPU) / +0.119 in V10.3's own tables. The correct statement is K1NBSC ≈ C0 and X8 < C0, never "the latent objective beats V10.3".

## 8. Throughput and the 10M benchmark

**Solo tiny-shape step time** (`runs/solo/`, 300 updates each, one process on an otherwise idle GPU, median after update 50):

| Arm | Seconds per update | vs C0 |
|---|---:|---:|
| C0 | 0.158 | 1.00× |
| ORD | 0.186 | 1.18× |
| NOAUX | 0.249 | 1.57× |
| X8 | 0.257 | 1.62× |
| K1NB | 0.284 | 1.79× |
| K1 | 0.315 | 1.99× |

**Equal wall time:** in K1's time budget C0 would get about 2× K1's updates. No arm beat C0 at equal updates, so an equal-time comparison can only widen every gap. It was not run.

**10M reference shapes on the Air's GPU** (`bench_10m.json`; width 128, vocab 8,192, encoder 4/4/2, core 4, decoder 4, PQ 8×32, Engram 4 × ~156k × 32 on the CPU, batch 8, context 384, S = 3, `v10_1/data` train split). Each arm: 20 warm-up + 200 timed updates, run solo after a 3-minute idle.

| Arm | Dense deployed | Step p50 | p95 / p50 | Input tokens/s | Peak driver memory | Tokens in 16 h | Tokens per deployed param |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 4,037,656 | 0.435 s | 1.01 | 6,121 | 3.9 GB | 353M | 87 |
| NOAUX | 5,409,436 | 0.608 s | 1.22 | 3,980 | 4.1 GB | 229M | 42 |
| K1NB | 5,409,436 | 0.631 s | 1.06 | 4,056 | 4.1 GB | 234M | 43 |
| K1 | 5,409,436 | 0.705 s | **1.28** | **3,298** | 4.1 GB | 190M | **35** |

- **The prior** is V10's 10M run: 5.73M dense, same batch and context, 3,855 tokens/s *sustained* over 13 h, candidate steps included. These numbers are 3-minute solo bursts, so sustained rates will be lower. K1's burst rate is already 14% below the sustained prior, which confirms v11.md's statement that 3,855 is an upper bound.
- **Where the time goes** (synchronized split, K1): forward + losses 0.33 s, backward 0.33 s, optimizer 0.04 s. The Engram hash costs about 0.1 ms per batch (done by the prefetch thread in training).
- **K1's tail latency** (p95/p50 1.28) is a bridge cost: the number of bridge rows varies by family.

**Gradient share** (`grad_share.json`): norms of each loss *at its training weight* (λ applied, ramp = 1), on 20 fixed batches, by module group, as a ratio to NTP's norm on the same group:

| Run | Loss | Encoder | Pooler | Core | Head |
|---|---|---:|---:|---:|---:|
| K1NBSC_s1 | `L_v1` (λ_c/3) | 0.01 | 0.04 | 0.03 | 0.05 |
| | `L_v2` (2λ_c/3) | — | — | 0.07 | 0.12 |
| K1SC_s1 | `L_v1` | 0.00 | 0.04 | 0.03 | 0.07 |
| | `L_v2` | — | — | 0.10 | 0.21 |
| | bridge SmoothL1 | 0.06 | 0.56 | 0.95 | — (bridge's own params 0.91) |
| | bridge KL | 0.19 | **1.77** | **2.62** | — (2.42) |
| K1_s1 | bridge KL | 0.19 | **2.35** | **2.93** | — (1.93) |

- The latent objective is inert on the encoder and small on the core.
- The bridge's KL dominates the pooler and core gradient in both K1 and K1SC.
- **Anchor proxy** (KL with the real concept anchor vs a mean anchor, K1SC): math 0.21 vs 1.04, text 1.43 vs 1.50. The bridge's target is concept-specific on math and nearly not at all on text, so it carries content on exactly the family where it costs most.

## 9. Limits

- **Two seeds.** Every math row whose sign flips between seeds is reported as unresolved: K1 − K1NB (+0.18 / −0.13) and K1NB − X8 (−0.09 / +0.16). C0's three seeds show that sorting alone moves 0.27 nats seed to seed.
- **The diagnostic arms (K1SC, NOAUXSC, K1NBSC, ORDE) are one seed each** and were added mid-run after the clip artefact was found. They carry no gate verdicts.
- **No random-prefix training** (deviation from v11.md §7.1, decided before the run).
- **Generation is at the floor for every arm**, so the exact-answer metrics carry no information at this scale.
- **Exact resume not tested.**
- **Wall-clock:** runs shared the GPU three at a time, so elapsed times are not comparable across arms. §8 uses solo smokes.
- **Total wall time** went beyond the planned 5–6 h, because two clip artefacts found mid-run (bridge, then probe) needed four diagnostic runs (K1SC, ORDE, NOAUXSC, K1NBSC) plus their evaluations.
- **Thermal state** was not instrumented beyond `pmset` (which recorded nothing). Drift figures are confounded with co-scheduling.

## 10. Patch list for v11.md

1. **Separate clip groups for training-only modules.** The bridge and the detached probe must not share the deployed trunk's gradient clip. Under AdamW a shared clip acts as a per-family reweighting of the data mixture; it was demonstrated twice (K1 via the bridge, NOAUX via the probe).
2. **The inherited clip at 1.0 already reweights the mixture in every arm, C0 included.** Math batches are clipped in about 99% of steps, so a math step counts about ⅓ of a text step. Either record this as part of the recipe or clip per family. Any arm that changes gradient norms changes the effective mixture.
3. **λ_c = 0.1 is inert on the trunk but not on the head.** The latent objective sends about 0.5% of NTP's gradient to the encoder, 3–10% to the core and 5–21% to the PQ head (per term, §8), and it trains the codebook outright. With the artefacts removed it is a null on NLL (K1NBSC ≈ NOAUXSC) while making the decoder rely on the concept route. A λ_c screen (0.3, 1.0) is the cheap next test.
4. **The forecast head shrinks.** A shared-σ Gaussian NLL through a convex combination of PQ codewords regresses the forecast to the mean: forecast variance is 7–18% of the target's, and ridge(forecast) > ridge(state) in every trained V11 arm. Candidates are a per-dimension σ, a non-convex head, or reading the state directly (ridge refits recover it).
5. **The bridge's purpose ("text gain through the bridge") is falsified at this scale,** both raw (K1) and with the artefact corrected (K1SC). It makes the core state more predictive of the next target, but compresses the pooled states (text PR about 6), reduces the decoder's use of the concept path, and costs text about 0.02 on its own. Next experiment: the λ_h/λ_KL screen of §4.6, or drop the bridge.
6. **Single global normalization** of the pooled target gives the families unequal weight in the latent loss. The per-family normalized target variance in the final checkpoints is text 0.46–0.73, math 1.07–1.17, newmath 0.57–0.67, Sudoku 0.50–0.97 (K1, K1NB, K1NBSC), a spread of up to 2.5×. (An earlier working note put the symbolic families near 0.07. The evaluation files do not support that, and it is withdrawn.) Per-family normalization would equalize the weights; otherwise report the imbalance as part of the objective.
7. **`L_v2` deviation:** visit 2 is recomputed from stopped inputs on S = 2 boundaries, not read from stopped KV. Visit 2 is null at this scale by every measure.
8. **K = 1 parameter counts:** removing the codes saves 2,040 parameters; deployed dense 472,324.
9. **"Parameter-matched ordinary transformer" must match the Engram too.** ORD without the n-gram table is 0.35 nats worse on text purely from the missing table. Use ORDE.
10. **Per-module gate margins** (G2's +0.02) are about 13× below the baseline's own seed spread on sorting. They need more seeds or a larger sort panel.
11. **Random-prefix training is required, not optional.** A period-8 Sudoku layout dependence appears wherever a trained concept path reads 8-token groups (X8, K1SC, K1NBSC; §6.6).
12. **The alignment gate measures distraction.** Replace max−min over r = 0..7 with the r = 0 vs r = 8/16 contrast (periodicity) plus the Sudoku shift-4 vs shift-8 contrast, both with a bootstrap.
