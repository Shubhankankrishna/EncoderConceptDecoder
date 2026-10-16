# V10.3 RCDL — calibrated concept prediction, mixture exploration and a gated sleep layer

**Status: design proposal, 12 October 2026. Not implemented and not trained.** It extends the V10.2 RF design ([V10_2_RCDL_Architecture.md](V10_2_RCDL_Architecture.md)) and adopts the corrections in [v10_2_review.md](v10_2_review.md). New measurements for this design are in [v10_3_design_evidence/2026-10-12/](v10_3_design_evidence/2026-10-12/).

## 1. Evidence for this design

Every measurement below comes from frozen V10.1 checkpoints of the run `v10_1/runs/pretrain_16h` (steps 3,171 and 61,384), the dev split, CPU inference and no training. Panels match the V10.2 review: a scored draw of 64 text documents, 64 math problems and 32 boards per Sudoku family (seed 20261012), with disjoint draws for calibration (seed 20261013) and classifier fitting (seed 20261014).

### 1.1 Two readings of the V10.1 charts that need correcting

**"Exploration loss reached zero" is a logging artifact.** Candidate training runs on every fourth update. On the other three, the trainer logs `explore = 0.0` and `selector = 0.0` ([train.jsonl](v10_1/runs/pretrain_16h/train.jsonl): `candidate_executed: false`); the last logged update, 61,384, was one of those. On candidate updates the raw value stayed at 5.80 nats for text and 1.16–1.28 for the symbolic families in the last 11,000 updates. The chart's moving average mixes the zeros in. Exploration did not collapse.

**"Concept prediction loss increased" measures a growing target.** The latent loss is the mean squared error between the PQ forecast for group g and the online pooled state `P3[g+1]`. That target's variance grew during training, so the absolute error can rise while prediction improves. Normalised by target variance (fraction of variance unexplained, FVU; 1.0 equals predicting the mean), the forecast improved on every family ([probe_latent_baselines.json](v10_3_design_evidence/2026-10-12/probe_latent_baselines.json)):

| Family | Target variance, step 3,171 → 61,384 | Forecast FVU, step 3,171 → 61,384 |
|---|---:|---:|
| Text | 0.572 → 0.678 | 1.121 → 0.979 |
| Math | 0.471 → 0.806 | 1.319 → 0.811 |
| Sudoku easy | 0.213 → 0.571 | 2.529 → 1.107 |
| Sudoku hard | 0.209 → 0.567 | 2.561 → 1.107 |

Any objective V10.3 adds for concept prediction is therefore specified and reported as FVU against fixed baselines, not as raw loss.

### 1.2 Concept prediction: the content is present, the readout discards it

At step 61,384, on held-out documents, the same quantities scored as FVU (lower is better; ridge maps are fitted on the calibration draw only):

| Predictor of `P3[g+1]` | Text | Math | Sudoku easy | Sudoku hard |
|---|---:|---:|---:|---:|
| Constant calibration mean | 1.011 | 1.047 | 1.002 | 1.002 |
| Persistence, `P3[g]` | 1.355 | 1.452 | 1.716 | 1.677 |
| **Trained forecast `F2` (what the model uses)** | **0.979** | **0.811** | **1.107** | **1.107** |
| Per-coordinate affine map of `F2` | 0.849 | 0.741 | 0.907 | 0.904 |
| Linear map of `F2` | 0.802 | 0.640 | 0.713 | 0.713 |
| Linear map of the concept state `U2` | 0.780 | 0.661 | 0.620 | 0.626 |
| Linear map of the core's input `P3[g]` | 0.791 | 0.703 | 0.640 | 0.646 |

Four facts follow.

1. **The emitted forecast barely beats a constant on text and is worse than one on Sudoku.** Its paired difference from the calibration mean is −0.015 to −0.052 FVU on text and +0.10 to +0.11 on both Sudoku families (95% document bootstrap).
2. **The information is there.** A linear map of the same `F2` explains 20–36% of the variance; a linear map of `U2` explains 22–38%.
3. **The forecast is shrunk toward a constant.** `F2` carries 3.0% of the target's variance on text, 14% on math and 7% on Sudoku, while `U2` carries 106–123% ([probe_forecast_spread.json](v10_3_design_evidence/2026-10-12/probe_forecast_spread.json)). Two mechanisms compound. The soft-PQ output is a convex mixture of codewords, and on text the mixture is broad (mean entropy 2.43 of a possible 3.47 nats per segment), which averages toward the centroid. And the codebook itself shrank: its RMS fell from 0.892 at step 3,171 to 0.483 at step 61,384 ([probe_parameter_scales.json](v10_3_design_evidence/2026-10-12/probe_parameter_scales.json)), below the target's scale of about 1.0, because the trainer applies weight decay 0.1 to every parameter with two or more dimensions, which includes the 3-D codebook ([v10_10M/train.py:60-65](v10_10M/train.py)). On Sudoku the affine scale is about 1 and the mean offset is 17% of the variance: there the forecast is biased rather than shrunk.
4. **The concept core adds little beyond its own input.** A linear readout of `U2` beats one of `P3[g]` by 0.011 FVU on text, 0.042 on math and 0.020 on Sudoku. At step 3,171 it was slightly worse on every family. Visit-2 and visit-1 forecasts differ by at most 0.004 FVU.

This explains part of a result the V10.2 review left open. There, replacing the decoder's concept inputs with fixed means recovered about 58% of the text concept effect ([review §4.3](v10_2_review.md)). That substitution replaced two inputs. The forecast `F` reaches the decoder only through a learned per-coordinate gain, `fusion`, and the gain fell during training from 0.106 to 0.073 (RMS; [probe_parameter_scales.json](v10_3_design_evidence/2026-10-12/probe_parameter_scales.json)): the decoder scaled the shrunk forecast down further rather than back up. So the PQ route was nearly free to replace with a constant. The continuous state `U` reaches the decoder through a full learned projection with its variance intact; the example-specific share of that route remains the review's open number. The review's two frozen-checkpoint discriminators remain valid, because the heads they train can rescale their inputs.

V10.2 inherits the cause and makes it worse. Its trainer reuses the same optimizer grouping ([v10_2/train.py:20](v10_2/train.py)), so it keeps the codebook in the decayed group and removes the codeword-fitting term, the only force that resisted the decay, and it still adds `F` to the decoder input through `fusion * Align(F)`, a per-coordinate gain with no offset or mixing across coordinates ([V10.2 §3.4](V10_2_RCDL_Architecture.md)).

### 1.3 Exploration: the codes differ little, and the winner is mostly unpredictable

For up to six eligible boundaries per document, the trainer's own utility (chunk cross-entropy plus 0.1 × next-pool MSE) was computed for all eight codes ([probe_candidate_utility.json](v10_3_design_evidence/2026-10-12/probe_candidate_utility.json)):

| Family | Boundaries | Spread, best to worst (median) | Best vs second best (median) | Regret of the selected code | Selector picks the winner | Linear classifier on `P3`, held out | Majority class |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text | 384 | 0.083 | 0.012 | 0.045 | 17% | 17–19% | 16% |
| Math | 140 | 0.104 | 0.012 | 0.062 | 19% | 27–29% | 16% |
| Sudoku easy | 192 | 0.109 | 0.014 | 0.049 | 18% | 25–27% | 29% |
| Sudoku hard | 192 | 0.113 | 0.013 | 0.042 | 16% | 26–30% | 23% |

Utilities are in the trainer's units: mean chunk cross-entropy per token (nats) plus 0.1 × MSE; the MSE term adds about 0.06–0.07 to each value (0.1 × the latent MSE of §1.1).

- **The hard-argmin labels are close to noise.** The best and second-best codes differ by about 0.012 nats per token, a seventh of the spread. Over the run the selector loss rose from 1.34 (updates 500–3,000, text) to 2.07, which is ln 8 = 2.079: chance. That is the expected outcome for these labels, not a training failure.
- **There is still something to model.** A spread of 0.08–0.11 nats per token over an 8-token chunk is about 0.7–0.9 nats of chunk log-likelihood. The codes do change the forecast; the problem is the label, not the mechanism.
- **Math has a prefix-predictable winner that the selector did not learn.** An offline linear classifier on the selector's own input reaches 27–29% against a 16% majority rate; the selector reaches 19%. On text and Sudoku neither beats the majority class.
- **Sudoku codes act as global biases.** A per-code constant explains 19–21% of the utility variance on Sudoku (1.3% text, 0.6% math), and codes 6 and 7 win 41–49% of Sudoku boundaries. In the last 5,000 training updates the effective number of selected codes was 7.47 on text but 5.3–5.6 on math and Sudoku.

These spreads are measured under V10.1's chunk head, which sees earlier tokens of the chunk. They must be re-measured under any new forecast head.

### 1.4 Can GMM gates separate distributions at width 128?

Local Support Learning ([LOCAL SUPPORT LEARNING.pdf](LOCAL%20SUPPORT%20LEARNING.pdf)) opens a low-rank adapter on an input `x` of one weight matrix when `log Φpos(x) > log Φneg(x)`, where `Φpos` is a diagonal GMM of the new phase's inputs and `Φneg` a GMM of a small generic reference. The paper tests models of 1.5–7B parameters. Here, four-component diagonal GMMs were fitted to the inputs of every attention and FFN matrix in E2, the concept core and the decoder, treating each data family as a phase ([probe_gmm_gates.json](v10_3_design_evidence/2026-10-12/probe_gmm_gates.json)):

- **The paper's reference recipe fails here.** With `Φneg` fitted on text only, a math gate opens on 37–99% of Sudoku tokens that neither GMM saw, at E2 and decoder sites. The paper's own caveat (`Φneg` need not be wider than `Φpos` in every direction) is the default outcome at this scale. The asymmetry is informative: a Sudoku gate stays closed on unseen math at E2 (0–8%), because whichever positive distribution is broader captures data that neither GMM saw.
- **A reference covering every trained family works.** With `Φneg` fitted on the other three families, a math gate opens on 0–7% of text and Sudoku tokens, and concept-core gates open on 0–2% of every other family. A text gate opens on 8–24% of math tokens at E2 and the decoder; math prompts are English, so this is the overlap policy (the newer phase wins), not leakage. An easy-Sudoku gate opens on 85–99% of hard-Sudoku tokens and the reverse: the gates separate formats, not difficulty.
- **The paper's temporal smoothing costs recall on short sequences.** The smoothed state starts at zero, so a short document never reaches the threshold. Smoothing lowers the math hit rate at E2 from about 0.91 to 0.81, and at concept-core sites, which run at one row per group, from 1.00 to 0.24–0.29. Raw per-row decisions already reach hit rates of 0.83–1.00 at every site.
- **Shared-weight blocks see two distributions.** A GMM test separates visit-1 from visit-2 inputs of the same E2 or core block with 57–95% accuracy (chance 50%), most strongly on Sudoku. Per-visit `Φpos` was tested; a single pooled `Φpos` was not.

**Within a family, gates separate formats, not skills.** Treating each of the four math modules as a phase, with a reference covering everything else ([probe_gmm_modules.json](v10_3_design_evidence/2026-10-12/probe_gmm_modules.json); about 150 documents per module), multiplication and sorting gates open on other modules 0–6% of the time at concept-core sites (median over sites) and 3–25% at E2. Algebra and add/subtract gates open on each other's expressions 63–85% of the time at every site group. A gate therefore protects a dissimilar earlier skill, but not a closely related one.

Limits: activations are from the frozen model without adapters active, whereas the paper fits `Φpos` on the adapted model; and each GMM saw at most 4,000 rows, against the paper's reference of about 1M tokens.

### 1.5 What carries over from the V10.2 review

The review's verdict stands and V10.3 adopts it ([review §1](v10_2_review.md)). In brief: the access diagnosis behind V10.2 is unestablished; V10.2's default arm removes latent and codebook fitting without a measurement; the promotion thresholds are near the noise floor; and the planned budget covers one arm. Since then, the V10.1 run stopped at step 61,384 (11.17 of 16 active hours) and cannot resume until a hashed source file is restored, and the `v10_2/` package implements only the RF arm.

## 2. Necessary conditions

Each row states what a working V10.3 must satisfy, the design consequence, and the check that would show it failing. The evidence column points to §1.

| # | Condition | Evidence | Design consequence | Falsifiable check |
|---:|---|---|---|---|
| N1 | Concept prediction is judged against baselines, not by raw loss | Target variance grew from 0.21–0.57 to 0.57–0.81 (§1.1) | Report FVU against the calibration mean and a ridge readout of `P3` on a fixed dev panel | Forecast FVU < ridge(`P3`) FVU, held out |
| N2 | What the decoder receives from the concept path keeps the target's scale | `F2` carries 3–14% of the target's variance; `U2` carries all of it (§1.2) | No soft-PQ output enters the decoder through only a per-coordinate gain; it passes through a full learned linear map | `var(decoder concept input) / var(target)` logged; must stay above a declared floor |
| N3 | Parameters whose scale carries meaning are not decayed | Codebook RMS 0.89 → 0.48 under weight decay 0.1 (§1.2) | The PQ codebook moves to the no-decay group | Codebook RMS logged against target RMS |
| N4 | The concept core adds predictive content beyond its own input | Ridge(`U2`) beats ridge(`P3`) by only 0.011–0.042 FVU (§1.2) | Supervise the core on a target its input cannot already explain, and measure the increment | Increment ridge(`P3`) − ridge(`U`) grows above its V10.1 value |
| N5 | Exploration trains codes and selector on labels that carry information | Best vs second best 0.012; selector at ln 8 (§1.3) | Soft hindsight responsibilities; the selector is the mixture prior | Held-out selector CE below the entropy of the held-out winner distribution, the best constant prior (now: text 2.054 vs 2.018, Sudoku 2.07 vs 1.85–1.88, math 1.997 vs 2.019); held-out mixture likelihood above committed-code likelihood |
| N6 | A code that helps everywhere is visible as a bias, not counted as a strategy | Per-code main effect 19–21% of Sudoku utility variance (§1.3) | Log the main-effect share per family | Share reported beside effective-code counts |
| N7 | A gate's reference covers every phase already learned | Text-only reference: 37–99% open on unseen Sudoku; full reference: 0–7% (§1.4) | `Φneg` is a bounded sample of every earlier phase, extended at each phase | Leave-one-phase-out open rate under a declared bound per site group |
| N8 | Retention is tested where gates are weakest | Algebra and add/subtract gates open on each other 63–85% (§1.4) | The retention protocol includes a pair of related skills | Related-skill retention reported separately from dissimilar-skill retention |
| N9 | Gate decisions are not destroyed by smoothing at group rate | Smoothing cut concept-core hit rates from 1.00 to 0.24–0.29 (§1.4) | No smoothing at group-rate sites; at token rate, initialise the smoother to the first decision | Raw and smoothed hit rates both logged |
| N10 | A concept benefit is shown against a simpler trained model | The review's C0 argument (§1.5) | C0 is in every comparison; frozen-checkpoint discriminators run before any production arm | RF-family arm versus C0 at equal active hours |

**What §1 does not establish.** That a better objective lets the core compute more than 1–4 FVU points beyond its input. That giving the decoder the visit-1 concept state, or the change between visits, helps. That gated adapters retain earlier skills once the adapters are active, or that RBF corrections can be consolidated into them. These are tested as arms, not adopted as defaults.
