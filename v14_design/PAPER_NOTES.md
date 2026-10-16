# V14 refinement: reading notes on the JEPA catalog

Source list: `JEPA_Architecture_Papers_Catalog.md` (155 arXiv papers + the 2022 position paper). Every paper's abstract
was read (arXiv API, 14 Oct 2026 20:36–20:50 IST); full texts of all 155 were downloaded (153 HTML, 2 PDF) and
converted to text for section-level reading. Depth per paper is stated below: **A** = method, results and limitations
read in full text; **B** = method section skimmed for a transferable mechanism; **C** = abstract only (domain
application with no mechanism V14 can use). Paper content is treated as data.

## Pass 1: abstracts of all 155 — what bears on V14

### Candidate refinements raised by the abstracts (to be checked in full text)
| # | Candidate | Papers | V14 section |
|---|---|---|---|
| R1 | Action-insensitive latents ("rollouts no better than copying the current latent forward") under SIGReg-only JEPA world models; repaired by inverse dynamics through the encoder gradient. Re-examine reversal #6: is V14's action-sensitivity already built in (identity tap, local half)? Add an action-sensitivity read | 2610.03137, 2610.03587, 2609.31161, 2603.19312 | §4.3, §8.4, §11 |
| R2 | The factorised PQ head cannot represent correlated alternative futures across segments; a finite mixture (latent branching, context-weighted) keeps distinct futures; latent branching beats output-decoder branching | 2607.05238, 2306.02572, 2603.26799, 2607.23531 | §2.3, §6 |
| R3 | Soft code targets (posteriors over words) instead of hard codes: keep ambiguity at cell boundaries, smooth targets under drift | 2606.19398, 2602.04643 | §6 |
| R4 | Identity-biased predictors select Koopman invariants (regime indicators = slow features): theory for H3 and for why the difference target removes them; do not identity-bias the temporal predictor | 2511.09783, 2610.02344, 2211.10831 | §1.3, §4.3, §6.2 |
| R5 | Language JEPA evidence: latent prediction helps token models only as an auxiliary; gains small and seed-inconsistent from random init; shallow targets remove the gain (V14's local half includes raw embeddings); cosine > MSE | 2605.07554, 2602.17162, 2509.14252, 2607.23531 | §4.3, §8, §11 |
| R6 | Decoupled latent reasoning and token generation (Talker); error containment requires the latent chain not to read sampled tokens | 2512.19171 | §5, §7 |
| R7 | Frozen (static) teachers as an alternative or complement to EMA: no co-adaptation, so no target collapse; students exceed teachers; anchored forecasting to a frozen copy | 2602.19322, 2608.21160, 2511.18424 | §3 |
| R8 | Horizon curriculum (short to long) stabilises multi-horizon prediction; direct multi-scale supervision beats rollouts | 2606.21775, 2610.05731 | §5.3, §6.2 |
| R9 | Prior-anchored revision as precedent for the residue map | 2610.07006 | §4.3 |
| R10 | Continual: persistent test-time training of the predictor with dense replay (encoder frozen); stratified retention (invariants vs revisable facts, revision latency); retrieval replay for language JEPA; continual SSL taxonomy (predictive distillation ≈ latent anchoring) | 2610.00722, 2610.03713, 2609.36952, 2607.09785, 2609.32592 | §9 |
| R11 | Scale evidence: 8B JEPA predictor with compute power laws (robotics); EMA teachers in language (data2vec) | 2610.10515, 2202.03555 | §10 |
| R12 | Auxiliary-task theorem: joint training with an anchoring task keeps non-equivalent states distinct (NTP as V14's anchor) | 2509.12249, 2501.14622 | §3.1, §8 |
| R13 | Gaussian-quantile / finite scalar quantisation of a SIGReg-whitened target: a fixed grid has no dead codes and no drift | 2512.07168 (FSQ) + LeJEPA | §6.1 |
| R14 | EMA alone does not prevent collapse (I-JEPA + VICReg); regulariser ablations | 2410.19560, 2602.03604 | §8 |

### Triage for full-text reading
- **A (full text):** 2211.10831, 2410.19560, 2509.12249, 2610.02344, 2511.09783, 2102.06810, 2605.17165, 2607.05238,
  2601.14354, 2603.20111, 2603.26799, 2606.19398, 2602.04643, 2607.23531, 2512.19171, 2605.07554, 2602.17162,
  2609.36952, 2603.19312, 2610.03137, 2610.03587, 2609.31161, 2512.24497, 2604.03208, 2606.21775, 2610.10515,
  2608.07409, 2610.00722, 2610.03713, 2607.09785, 2602.19322, 2608.21160, 2610.07006, 2202.03555, 2602.03604,
  2609.33563, 2512.07168, 2610.05731, 2608.29434, 2602.11389, 2512.10942, 2306.02572
- **B (method skim):** the remaining architecture, world-model, language, generative and continual papers
- **C (abstract only):** domain applications in §9–§14 and §17 of the catalog with no transferable mechanism

**Correction to the triage (21:25).** LeJEPA (2511.08544) and H-JEPA (2610.06805) were missing from the A list although
V14 rests on them; both are now read in full. 2512.07168 is a speech tokenizer (JEPA + density-adaptive attention +
FSQ), not an FSQ paper; R13 is re-filed under cluster (b). The first digester dropped method sections (keyword filter,
4,000-character cap per section); every A-tier paper is read with the fixed digester (`digest2.py`: every numbered
section after the introduction, ~20k characters, related work and appendices skipped) plus targeted reads of appendices.

## Cluster (a): collapse, training dynamics, regularisers (read 21:18–21:28)

| Paper | Depth | What it shows (numbers from the paper) | What V14 takes |
|---|---|---|---|
| 2211.10831 Sobal et al., JEPAs focus on slow features | A | Moving-dot world model; VICReg- and InfoNCE-JEPA fail when the background noise is fixed within a sequence (they encode the background, not the dot); reconstruction and inverse dynamics succeed; changing noise is harmless. Conclusion: a fix needs hierarchy or "an additional constraint that prevents the representations from being constant across time" | H3's original statement; V14's document-centred SIGReg is a constraint of that kind and the Δ target removes the constant part. Cite in §1.3 and §8.3 |
| 2511.09783 Koopman invariants | A | Thm 3.4: with a linear predictor and an EMA target that tracks the encoder, the idealised JEPA loss is minimised iff the encoder's components lie in the Koopman-invariant subspace (regime indicators) and the predictor is the identity on it | The formal version of H3 (copy-current is the optimum of level prediction); an invariant's change is zero a.e., so a difference target removes exactly that subspace. Cite in §1.3/§4.3 |
| 2410.19560 C-JEPA | A | I-JEPA + VICReg variance/covariance + invariance. Linear-probe gain is mostly early: 63.7 → 68.3 → 69.5 at 100 epochs, 72.9 → 73.5 → 73.7 at 600. EMA-only I-JEPA did not collapse; "EMA is prone to collapse" is argued, not shown | Corrects pass-1 R14: a variance–covariance term on top of EMA speeds early training; it is not evidence that EMA alone collapses. Supports the VICReg arm's plausibility only |
| 2610.02344 Drive vs Decay | A (+ App. J) | Linear coupled dynamics: a mode escapes collapse iff drive/decay μ > 1; the count of super-critical modes bounds post-escape effective rank (ρ = 0.91, 810 runs); μ ∝ 1/(predictor init scale); masking lowers drive; **momentum sets the speed of escape or collapse, not which modes escape (Prop. J.7)**; SIGReg and VICReg lower the decay side; ResidualPred (identity-biased predictor) + SIGReg compose (ImageNet-1k pilot 25.2 → 29.2 → 31.7 → 32.2); "near-zero MSE can accompany degenerate embeddings"; effective rank necessary, not sufficient | EMA justified as a timescale (§3.2 already does; add J.7 as the reason the twin is given no anti-collapse role). Effective rank of each target half as a read. The prediction loss is not a collapse diagnostic (V14 already reads usage, χ, per-half accuracy). Identity bias of a *level* predictor is what 2511.09783 says converges to invariants; V14 predicts codes of differences, so C's residual identity bias is not that failure |
| 2102.06810 Tian et al., DirectPred | A | Linear BYOL/SimSiam: stop-gradient essential; weight decay balances predictor and trunk; predictor eigenspaces align with the input correlation; DirectPred sets W_p = U diag(√s + ε·max s) Uᵀ from an EMA correlation (ρ = 0.3), beating the gradient-trained linear predictor (77.9 vs 74.5, STL-10, 100 epochs); hybrid with SGD steps between solves best (80.3) | Precedent for V14's residue map P as a statistic re-solved from EMA second moments with a ridge floor. Cite in §4.3 |
| 2509.12249 Auxiliary tasks for JEPA | A | Thm 1: if latent dynamics are consistent and an auxiliary regression target is fit, observations that are not bisimilar under (dynamics, auxiliary) cannot share an encoding (one-sided). Counting world: reward auxiliary → 9 count clusters; random auxiliary → no count structure; reward only → coarse; transition loss only → complete collapse | NTP is V14's auxiliary in this sense: the online encoder must predict the next token, so prefixes with different continuations cannot collapse, and the twin inherits that. This is why V14 needs no encoder-side anti-collapse term; say so in §8.1. The auxiliary chooses which distinctions survive: NTP keeps token-predictive ones, which is why b needs the Δ/residue construction to hold anything non-local |
| 2610.03137 SLIM / LeWM+I | A | SIGReg-only LeWM collapses to action-insensitive latents when little of the frame responds to actions (spread/motion 3.8×10⁻⁴; rollouts ≈ copying); encoder-side IDM carries ≈ 85% of the repair, predictor-side IDM alone is null, shuffled-action IDM is null (the gradient, not the head); action sensitivity spread/motion ≳ 0.05 is necessary, not sufficient (28 configurations) | Reversal #6 stands: in language the group's tokens are observed and NTP already puts them in the encoder. New read: target-side action sensitivity — spread/motion of the twin's b (and a) over K alternative next groups at a fixed prefix; < 0.05 means b does not depend on what the group did |
| 2603.19312 LeWM | B+ (method, ablations, limitations) | SIGReg + prediction, end-to-end, no EMA or stop-gradient; M = 1024 slices with negligible effect of M; λ the only effective knob; limitation: low data diversity weakens SIGReg in simple, low-dimensional settings | Supports M = 256 and ν_S as the one real knob. V14's batches are single-family: low within-family diversity is the case LeWM flags (watch SIGReg's value per family) |
| 2511.08544 LeJEPA | A (theory headings, §4.3, §5–§6.2) | Isotropic Gaussian minimises worst-case linear-probe risk; slices resampled every step beat a fixed set (16 resampled > thousands fixed); 17 knots on [−5, 5]; O(1/N) minibatch bias negligible down to batch 16; λ = 0.05 default; small projector dims best (64: 75.3–75.7 vs 1024: 73.7–74.8); teacher-student not needed against collapse, small gain for ViTs; the projector is trained by both the prediction loss and SIGReg | V14's SIGReg constants match. Difference: V14's T_loc/T_glob are trained by SIGReg alone (deliberately, so NCP cannot make the target easy) — see the T-projector issue below |
| 2610.06805 H-JEPA | A (§4.3, App. K, App. L) | Prop. 1: no regulariser on the marginal law excludes slow-feature collapse. Prop. 2: time on the sample axis adds λ(T−1)κ at exact collapse (N-scaled statistic, uncentred), which the paper calls "a real but small barrier". L.6: time repulsion (negative temporal similarity) keeps the predictor off the identity but is "a motion floor, not a selector"; none of its runs clears the zero-action floor. Prop. 3: IDM has a data-fixed floor at collapse. K.1: SIGReg keeps effective rank high, but without IDM planning stays near the floor (DROID). K.2: SIGReg and IDM weights must be scaled together | §8.3's attribution must be precise: Prop. 2's uncentred barrier is small; the fixed Ω(1) amount comes from V14's document-centring (a point mass at collapse), which is V14's own step. L.6 is the closest published evidence for what centred SIGReg on b is: a motion floor, not a selector; selection of content comes from NTP |
| 2605.17165 Factorised latent dynamics (video) | A | Single seed, small data. A head predicting the teacher's temporal *difference* (LD-JEPA) +5.0 SSv2, −2.1 ImageNet-100; aligning student and teacher differences (Delta-JEPA) −8.1 / −4.6; 10 of 14 auxiliary objectives lose > 5 points on appearance | Weak support for a difference target helping temporal reasoning at a cost to static content; V14 keeps the local half and the level in the input view. No change |
| 2602.03604 EB-JEPA library | B | SIGReg less hyperparameter-sensitive than VICReg at similar peak (CIFAR-10); action-conditioned world models in randomised scenes need temporal-similarity and IDM terms on top of VICReg; k ≈ 4 rollout steps best vs single-step for video-JEPA | Supports SIGReg as default and VICReg as the arm |
| 2609.31161 A-JEPA identifiability | A (theorems, discussion) | JEPA = transition predictiveness − λ·entropy of the next representation; component-wise identifiability needs sufficient action-induced variation in the transitions; LeWM is a constrained instance (SIGReg ≈ entropy under a Gaussian prior); contrastive negatives ≈ entropy maximisation (Prop. E.1); two levels of failure: information loss and representation mixing | V14's L is the same two principles on the target side (NCP's cross-entropy; SIGReg on τ; codebook usage as a discrete entropy read). Isotropic SIGReg is rotation-invariant, so PQ's segments are partitions of the space, not identified factors: §6/§9 should not read segment codes as factors |

### Issue found while reading (not in any one paper): what centred SIGReg on b can and cannot do
T_glob is a free linear map trained by SIGReg alone. Near slow-feature collapse, when G's residual change is small,
T_glob can meet SIGReg's unit within-document variance by growing its scale, so the variance part of the barrier is
absorbed by T_glob and does not reach G; only the *shape* of the within-document distribution sends gradient to G.
This matches H-JEPA L.6 (a motion floor is not a selector) and §3.1's own position (V14 does not want G to change
faster). §8.3's sentence that centred SIGReg pushes residual variation "outward" on G overstates it. Second, a linear
projector trained only to make its outputs Gaussian prefers the most Gaussian-looking directions of its input, the
reverse of projection pursuit, where non-Gaussian directions are the structured ones; LeJEPA avoids this because its
projector also gets the prediction loss.

### Cluster (a): proposed changes (before consultation 12)
- **a1 (text, §8.1):** name NTP as the encoder's anchor (2509.12249 Thm 1): the reason V14 has no encoder-side
  anti-collapse term; SIGReg's job is the target side (whitening for PQ, within-document spread).
- **a2 (text, §3.2):** "It does not prevent collapse": momentum sets speed, not which modes escape (2610.02344 J.7);
  LeJEPA needs no teacher against collapse; C-JEPA's EMA-only baseline did not collapse.
- **a3 (text + references, §1.3, §4.3):** Sobal et al. 2022 as H3's origin; Koopman Thm 3.4 as its formal statement
  (level prediction with an identity-like predictor is minimised by invariants); the Δ target removes that subspace.
- **a4 (text + reference, §4.3):** DirectPred as the precedent for P (a linear map from EMA second moments, re-solved
  periodically, with a ridge floor).
- **a5 (new reads, §6.3, §4.4, §11):** (i) target-side action sensitivity: on the held batch every 500 steps, K = 8
  alternative next groups sampled from the model at temperature 1 for 256 fixed prefixes; spread/motion of the
  twin's b (and a) = mean variance across the K alternatives over mean squared size; gate 0.05, read as necessary,
  not sufficient; §4.4 row: below 0.05 in ≥ 3 families after the λ ramp → same response as R² > 0.95. (ii) Effective
  rank of each half, online and twin.
- **a6 (correction + read, §8.3):** attribution: H-JEPA's Prop. 2 (uncentred) is "a real but small barrier"; the
  fixed amount (0.131 vs 0.0003 per direction) comes from V14's centring. Near collapse T_glob's scale can absorb the
  variance part, so on G the barrier acts through shape only; H3 at the target is defended by the Δ construction and
  the reads, consistent with §3.1. New read: T_glob's gain (Frobenius norm over its initial value); growth above 10×
  means b's variance is being made by amplification (the noise failure).
- **a7 (design question, §4.3):** T_loc and T_glob are trained by SIGReg alone and so may select the most
  Gaussian-looking directions (projection pursuit in reverse). Options: (A) keep, add reads (share of the residual's
  variance in T's row space against its top-64 principal subspace; K3 probe on the full residual against b); (B) make
  T_loc and T_glob statistics, whitening on the top-64 principal directions of EMA covariances (DirectPred-style,
  frozen with P), so SIGReg trains only G through b and is dropped on a; (C) (A) as default and (B) as an arm. A quick
  check is possible on V13-tiny's final checkpoint: excess kurtosis and the EP statistic of within-document local
  features along principal versus random directions (CPU, background QoS).

### Cluster (a), late addition (21:35): H-JEPA's Table 12
H-JEPA's appendix ran exactly the configuration V14's §8.3 called unpublished: a ladder of SIGReg weights with time on
the sample axis and no inverse dynamics (uncentred; end-to-end, no target encoder). Every run collapsed, later at
larger weights but to the same endpoint (χ 0.008–0.016); the healthy IDM-on model had χ = 0.12, action sensitivity
0.28. Their caveat: the barrier is measured against the null value, so "a collapsed encoder need not score above a
healthy one". Applied in row 12's edits (§8.3 now cites it and states the two ways V14 is outside that regime).

## Cluster (b): probabilistic targets, soft codes, the language paradox (read 21:36–21:43)

| Paper | Depth | What it shows | What V14 takes |
|---|---|---|---|
| 2607.23531 The JEPA paradox in language | A | Deterministic squared-error latent prediction learns the conditional mean (a centroid over alternatives); masked text is not conditionally concentrated, so a text JEPA (BERT-Large, C4 100K, EMA 0.996) shows MI proxy saturating below 0.35 nats, then validation effective rank collapsing 4.66 → 1.57 at step ~9,340 while training loss still falls; downstream at or near chance; five seeds reproduce the ordering. Conditional variance estimated with K = 16 completions from an MLM oracle passed through the target encoder | Strongest external support for V14's cross-entropy over codes (§2.4 already argues it from C2); the cosine latent arm (a point regression) is expected to lose. Its K-completion estimator of conditional target variance is the same estimator as V14's new action-sensitivity read |
| 2607.05238 Branch-JEPA | A | K = 6 weighted latent atoms with a context-only router, trained by a full-set energy score, beat specialization (−6.5% ES), soft-WTA, Sinkhorn and a tuned MDN (−12.8% ES); at matched parameters, branching in the latent transition beats branching only at the output (N_eff 5.80 vs 5.25, better ES and Brier, all five seeds); fixed uniform masses come within 1.25% | V14's head is distributional per segment but factorised across segments, so it cannot represent correlated alternatives (e.g. (A1, B1) or (A2, B2) but not the cross pairs). Candidate arm: K latent branches from u_g with a router, the shared horizon head on each, exact mixture NLL. Also: ĉ (§6.4) is a conditional mean over the head's alternatives, the object these papers warn about |
| 2606.19398 S-JEPA | A | HuBERT-style masked prediction with soft GMM posteriors as targets (KL), online GMM from an EMA encoder, adaptive layer by effective rank, EMA decay alternating 0.999/0.9999; best SUPERB WER below 90M; 36.6% of frames carry > 1 bit (two-way ties). No controlled soft-vs-hard ablation ("left to future work") | Candidate arm: soft code targets, KL to the codebook's own posterior (the codebook read as an isotropic GMM, σ² from its running quantisation error, no new knob). Evidence indirect; motivated by code stability (V5, C3) |
| 2602.04643 SC-JEPA | A (method, ablation, theory headings) | Soft cosine-softmax codebook bottleneck with KL between predicted and EMA-target code distributions; removing the bottleneck → near-random; KL beats MSE alone; reconstruction needed against collapse; dual entropy terms (sharp per sample, diverse per batch) | Support for distributional targets over codes with an EMA teacher; no change |
| 2601.14354 VJEPA | A (§4, §4.7, conclusions) | Probabilistic predictor NLL + β·KL(target posterior ‖ N(0, I)); deterministic JEPA as a special case; Thm 1: no collapsed global optimum under target diversity when the predictor is distributional; unimodal Gaussians average modes | H1 (the head predicting the marginal) is never a global optimum of V14's cross-entropy when targets differ across contexts: an optimisation failure, not an attractor |
| 2603.20111 Var-JEPA | A (§3.3, §3.7) | Coupled-VAE ELBO for JEPA; ELBO surgery: E KL(q ‖ N(0, I)) = KL(aggregate ‖ N(0, I)) + I(x; s) | SIGReg is the aggregate-posterior half of that KL without the information term: it shapes the target's distribution without limiting its information (one sentence in §8.3) |
| 2602.05639 Joint Embedding Variational Bayes | B+ (ablation) | Student-t directional likelihood + KL; without KL, posteriors collapse and seeds diverge (±24–31 points) | Distributional training needs its regulariser; no change |
| 2603.26799 Gaussian (mixture) joint embeddings | B+ | Single joint Gaussians are unimodal (predict the average of modes) and the primal NLL has a "Mahalanobis trace trap" (the data term cancels to d/2, the volume term shrinks everything); GMMs with prototypes fix unimodality | Same lesson as Branch-JEPA; no change |
| 2306.02572 Latent-variable EBMs (tutorial) | B (§2.4, §3.2) | Latent variables capture what x does not determine about y; inference by minimising or marginalising (free energy) over z | Frame for the branch-head arm: a discrete latent with capacity log₂K bits, marginalised exactly |
| 2512.07168 JEPA as a neural tokenizer | B (FSQ section) | JEPA encoder + finite scalar quantisation (fixed per-dimension grid, no codebook learning) | Candidate arm: a fixed quantiser on the whitened segments (sign of 5 fixed orthonormal directions = 32 codes): exactly uniform on an isotropic Gaussian, no k-means drift, no dead codes; keeps ≈ 10% of a segment's variance against ≤ 19.5% for any 32-word quantiser (rate-distortion bound) |
| 2609.33563 MA-JEPA | B | Not a categorical latent (pass-1 note wrong): action-conditioned Gaussian identification under E[h] = 0, Cov(h) = I (Thm 4.1) | No change |

### Cluster (b): proposed changes (before consultation 13)
- **b1 (text, §1.3 H1, §2.4):** 2607.23531 as external evidence for cross-entropy over codes in text; VJEPA Thm 1:
  with a distributional predictor H1 is never a global optimum when targets differ across contexts.
- **b2 (arm, §6.2):** branch head: K = 4 small residual MLPs map u_g to u_g^(z), a router π(u_g), the shared horizon
  head reads each; NCP is the exact mixture NLL, −log Σ_z π_z Π_s p_z(c_s); read N_eff = exp H(π) per family.
  Latent branching in Branch-JEPA's sense (multiplicity before the shared head), a discrete latent in LV-EBM's sense.
- **b3 (fix to existing arms, §6.4, §7, §2.3 item 4):** ĉ is a conditional mean over the head's alternatives; the
  ĉ-injection arm, drafting and the energy-minimisation arm take a committed tuple (argmax, or sampled at temperature
  1) and its codewords; ĉ stays as a read.
- **b4 (arm, §6.2):** soft code targets (KL to the codebook's isotropic-GMM posterior, σ² from the running
  quantisation error); evidence indirect; read target entropy and V5.
- **b5 (arm, §6.1):** fixed sign quantiser per segment; exact uniformity and no codebook drift, at about half the
  retained variance; read V5 and per-family usage.
- **b6 (text, §8.3):** SIGReg as the aggregate-posterior half of a VAE's KL (Var-JEPA §3.7).
- **b7 (text, §2.4, §8.1):** the cosine latent arm is a point regression and is expected to lose (2607.23531); it
  stays last in the queue.
- **Open from (a):** given H-JEPA's Table 12, should V14's default change (e.g. promote an anti-slow term), or does the
  regime argument (no gradient into the target encoder; NTP anchor; Δ target) justify keeping SIGReg as backstop with
  IDM as the fallback? My view: keep, because the failure Table 12 shows is end-to-end encoder collapse, which V14's
  NTP anchor and EMA twin rule out, and V14's slow target was a construction problem that Δ removes.

## Cluster (c): language, tokens and generation (read 21:50–22:10, 14 Oct)

| Paper | Depth | What it shows (numbers as reported) | Bearing on V14 |
|---|---|---|---|
| 2512.19171 JEPA-Reasoner | full (§3–§7, App. C/D titles) | Reasoner = embedding layer + transformer predicting the next latent (cosine ×4) against an EMA (0.98) of the *embedding layer only*; latents L2-normalised and looped back; a separate Talker decodes latents to tokens and cannot solve the task alone. Tree search: predicted latents lie between sibling embeddings, the correct sibling's coefficient larger 99.89% of the time; Talker exact match 99.87%. CFG: less degradation than a transformer under token noise; vs COCONUT 0.459 vs 0.374 exact match. GSM8K 0.9B: 20.7 → 39.4 (5-shot), 20.8 → 51.9 (8-shot) against a fine-tuned transformer that did not improve from 5 to 8 shots; one run. §7.1 "no error propagation" is the identity ∂r/∂x̂ = 0 of a chain that never reads sampled tokens; §7.2's bound is the sphere's diameter | Two points. (i) V14 deliberately does the opposite of decoupling: its concepts read the realised tokens through the encoder, and its horizons are predicted directly (§5.3), so there is no latent rollout to compound. Belongs in §2.4 with the evidence caveats. (ii) The mixed-latent result is the evidence *against* §6.4's argmax change: a decoder trained on centroids reads them (99.87%). And sg(ĉ) would close NTP's route into the head as well as the argmax does, so the gradient argument does not choose between them; only "commit vs carry the mixture" does |
| 2509.14252 LLM-JEPA | §3.1, §3.3, §4.3, §5.2, §6, A.3 | NTP does not implicitly minimise the JEPA loss (monitored, no gradient), and adding the term leaves NTP loss unchanged (§3.3). γ/λ sweep (A.3): JEPA only → empty outputs (0.00%); weight ratios 0.01–1 all beat NTP alone (66.8–70.4 vs 57.3). Loss form: cosine 71.5, MSE 70.6, ℓ2-norm 2.2, InfoNCE 34.4. Batch-level loss dropout α = 0.5–0.75 raises accuracy at matched compute; guideline λ(1 − α) ≈ constant. Best (λ, k) anywhere in the grid; adjacent cells similar | §10.2 item 2: V14's per-row dropout with per-carrying-row means keeps the effective weight at λ, which is what LLM-JEPA's guideline does by hand. §8.2: the useful weight range is broad (two decades), which supports a cap rule over a sweep. §8.1: NTP anchor needed (JEPA-only fails) |
| 2605.07554 ProteinJEPA | §3, §5.3–§5.7, §6, A.10 | MLM + λ·cosine to an EMA teacher's *half-depth* state (+ VISReg; EMA and VISReg never ablated separately). Continued pretraining: gains in every seed and size, concentrated on structure/homology; MLM loss 0.002–0.005 nats worse. From random init: no reliable gain (50/41/23, sign p = 0.20; not replicated across seeds). Gain grows ~2–3× per ~4× parameters. Ladder (half-depth + shallower targets): loses all 8 structure/homology comparisons (median +0.013 → +0.004). Cosine > MSE (16/5/11). JEPA-only: large losses everywhere. NTP variant (ProGen3-112m, MoE): predict the teacher's half-depth state **4 positions ahead**: 11/4/1 continued, 10/3/3 from scratch (p = 0.046); NTP loss not improved | The closest published precedent to V14's horizon-1 NCP over a 4-token group. Sets the expectation: from scratch at 7.7M is the regime with the least evidence; the tiny test should be judged on target health (V1–V5), and a null downstream effect there is not decisive. Ladder result: shallow content in a target can dilute the deep part; V14's local half is shallow by design but sits in its own segments, so the question for V14 is what share of NCP's signal to u_g comes from the local half |
| 2602.17162 JEPA-DNA | §4.4 (Tables 4, 5), §5 | Continual pretraining of DNABERT-2: MLM alone +2.06%; MLM + JEPA *without* VICReg +1.58% (worse than MLM alone); with VICReg +4.79%; + scheduled masking +6.23%. Attention predictor 4 layers best; MLP worst. SGDM needed (AdamW degraded). Continual only, never from scratch | In a token model the latent term helped only with its anti-collapse term on. Consistent with V14 (SIGReg on). Optimiser finding is setting-specific |
| 2512.10942 VL-JEPA | §4.5, §4.6, §4.8 | Controlled: same frozen vision encoder, embedding prediction (0.5B predictor) vs token prediction (1B LLM): CIDEr 14.8 vs 7.1, top-5 41.0 vs 27.2 at 15M samples. Y-encoder (text target encoder) trained by gradient at an LR multiplier: 0.05–0.10 best, faster and slower both worse. Dropping the caption pretraining stage: −21.7 classification. Selective decoding (decode where the embedding stream changes): 2.85× fewer decodes at equal CIDEr | Target-encoder speed has an interior optimum in their setting too (corroborates §3.4's lag argument, different mechanism). Not a language-modelling result |
| 2202.03555 data2vec | §3.3, §5.4 | Targets: average of the top-K blocks' outputs, each normalised (parameter-free LN for NLP) before averaging, "to prevent layers with high norm dominating"; K = all layers near-best in speech, vision, NLP; teacher EMA with τ ramped; feature encoder and positional encoder **shared** (not EMA'd): "more efficient and slightly more accurate" | Two checks on V14. (1) Per-source normalisation of the local half: measured, not needed at tiny scale (below). (2) Sharing input layers between teacher and student: the one ablation on it favours sharing, which supports §3.3's shared Engram table |
| 2609.36952 ER-JEPA | §2, §3.2–§3.3 | Replay of past examples in LLM-JEPA fine-tuning: 83.65 vs 68.75 at 240 PFLOPs; historical replay beats the same loss on the current batch by 2.2–3.8 points; token-matched control 62.3 | Continual cluster (e): replay evidence in a language JEPA, fine-tuning regime |
| 2609.34407 JELAR, 2601.00366 BERT-JEPA, 2504.10512 JEPA4Rec, 2609.40129 VR-JEPA, 2607.14537 MIDI-RAE-JEPA, 2609.33110 D-JEPA, 2405.08679 audio design choices | method skims | Latent CoT targets built by a non-causal expert and fed back at inference (JELAR); InfoNCE needed to fight CLS collapse, MSE/cosine did not (BERT-JEPA); recommendation; video-generation guidance; equivariance + chunked SIGReg per hierarchy level (MIDI); swappable physics decoders, no EMA; audio masking design | No change to V14 |

**Measured on V13-tiny's final checkpoint (CPU, fresh dev rows, 16–24 rows per family; `$SP/scale_probe.py`, `$SP/r2_probe.py`).**
- Raw scales of the target's pooled sources: total variance (trace of covariance) of MeanPool(x) vs MeanPool(L^enc)
  19.0 vs 37.2 (grid), 23.3 vs 22.1 (text); of their group-to-group changes 34.2 vs 38.3 and 46.6 vs 40.3; of Δ^G only
  4.0 (grid) and 2.2 (text). PCA-64 of the raw [x; L] pool takes 28% (grid) / 50% (text) of its subspace from the x block;
  after per-source RMS normalisation 31% / 62%. So neither source dominates; data2vec's per-block normalisation is not
  needed at this scale.
- The residue map's held-out R² (ridge Δ^ℓ → Δ^G, fit on even rows, scored on odd rows): **grid 0.69, prog 0.80, state
  0.38, plan 0.94, text 0.12**; RMS of Δ^G 0.12–0.21, of its residue 0.05–0.14. No family is above §4.4's 0.95; plan is
  at the edge. (Proxy only: V13's G was trained against the level target, and V14's twin is V14's own encoder.)

**Proposals (cluster c).**
- c1. §2.2/§12 and §11: the from-scratch expectation (ProteinJEPA §5.3, §6; JEPA-DNA, LLM-JEPA and VL-JEPA all on
  pretrained models; JEPA-Reasoner pretrains with NTP first). Judge the tiny test on V1–V5; a null effect on downstream
  probes at 7.7M from scratch is the expected outcome, not a refutation. ProteinJEPA's 4-ahead NTP variant as the
  nearest precedent. Optional: at the 1B tier, NCP introduced at mid-training of an NTP run as an arm (§10.4).
- c2. §6.3: a read of the local half's share of NCP's gradient at u_g (autograd.grad of each half's cross-entropy with
  respect to u_g on the λ-cap's held batch, every 500 steps; the head only, so cheap). Answers ProteinJEPA's dilution
  question for V14. Read only, no rule.
- c3. §4.3 noise bullet: the measured R² per family above, as context for the 0.95 rule.
- c4. §2.4: why V14 does not decouple (JEPA-Reasoner), with the evidence caveats.
- c5. §6.4: JEPA-Reasoner's mixed-latent result stated as the evidence on the other side; the ĉ-injection arm gets two
  variants, the argmax tuple and sg(ĉ), which separate "commit vs carry the mixture" from the gradient route.
  (Conflicts with row 13's b3 decision; raised with the advisor.)
- c6. §10.2 item 2 and §8.2: LLM-JEPA's λ(1 − α) guideline and its two-decade weight range.
- c7. §3.3: data2vec's shared feature encoder as the one ablation on sharing input layers (supports the shared Engram
  table); §3.4: VL-JEPA's interior optimum for the target encoder's speed.
- c8. §2.2 table: rows for ProteinJEPA (if not present), JEPA-DNA, VL-JEPA, data2vec; JEPA-Reasoner moved to §2.4.

**Recomputed under §4.3's own specification (one running P across families; row 14's request).** Same rows; P fitted
on the even rows of all families pooled, scored on each family's odd rows (`$SP/r2_probe2.py`–`r2_probe4.py`):

| Family | per-family P | pooled P, equal rows | pooled P, training mix (text 10%, each reasoning family 22.5%) | same, per-source RMS-normalised pools |
|---|---:|---:|---:|---:|
| grid | 0.691 | 0.489 | 0.503 | 0.544 |
| prog | 0.796 | 0.545 | 0.586 | 0.579 |
| state | 0.381 | 0.253 | 0.245 | 0.231 |
| plan | 0.938 | 0.821 | 0.829 | 0.816 |
| text | 0.122 | −0.257 | **−0.502** | −0.597 |

A single map transfers the reasoning families' local-to-global coupling to text, where the coupling is weak: on text,
Δ^G − Δ^ℓ·P has 1.5× the variance of Δ^G about its mean, so the pooled P *adds* local content to b (the "second bag of
tokens" failure that P exists to remove). Measured directly (`r2_probe5.py`, text's odd rows, training-mix P): the
map's local term Δ^ℓ·P has 0.95 of Δ^G's own variance and 0.63 of the residual's, so about two-thirds of b's variance in
text would be local content put there by the map (an earlier note here inferred a third; the measurement replaces it). Normalising the pooled sources does
not change it (the mismatch is in the coupling, not the scale). Raised with the advisor as row 15.

## Cluster (d): world models, horizons and hierarchy (read 22:28–22:42, 14 Oct)

Already used in (a)/(b): H-JEPA, LeWM+I, AVL-JEPA, 2609.31161, the Koopman paper, Drive vs Decay, Sobal et al., the
factorised-dynamics study, Branch-JEPA, VJEPA.

| Paper | Depth | What it shows | Bearing on V14 |
|---|---|---|---|
| 2606.21775 VLWM | §3.2–§3.3, §4.3–§4.4, App. C | One predictor trained on horizons k ∈ 1…5 directly (L2 to an EMA target) beats LeWM's one-step rollouts at long goal offsets: TwoRoom-100 36% → 68%, TwoRoom-75 34% → 70%, Cube-100 42% → 50%. Cumulative-uniform horizon curriculum over the first 10 epochs: +4 to +6 points on PushT/TwoRoom long offsets, −6 and −8 on Cube mid offsets (Table 2); one run per cell. Longer-range supervision improved a linear probe of a slow state variable (block angle r 0.882 → 0.904) | Corroborates §5.3 (direct horizons, no rollout) and §5.3's "pressure" argument. The curriculum's evidence is mixed and small: not adopted; V14's falling weights (0.4…0.1) already favour short horizons |
| 2604.03208 HWM | §2, §3.4, App. D.1–D.3, §6 | Two world models at two timescales in one latent space, learned macro-actions. Beyond 1.5 s, the high-level model's single prediction has lower error than the low-level model's rollouts. Hierarchy beats flat models of equal or larger size (Push-T d = 75: 61% vs 17% and 15%; Maze 83% vs 63% and 59%). Learned macro-actions beat concatenated actions (83% vs 37%) and hand-crafted summaries. High-level stride: 50% at 6, 78% at 12, 75% at 14. Limitation: one latent space for all levels | Corroborates §5.3 and the hierarchy itself (not capacity). The learned-summary result matches V13's learned input pools. §5.5's stride (four concepts) has an interior optimum in their setting; V14's third level already has its own target view (their stated limitation) |
| 2512.24497 JEPA-WM design study | §4, §5.2–§5.3 | Frozen encoders (DINOv2/v3, V-JEPA 2); 2-step rollout loss best in simulation, 6-step on DROID (accuracy vs robustness of rollouts); AdaLN conditioning slightly best on average, task-dependent; predictor depth 6 saturates in simulation, bigger helps on real data; data scaling helps everywhere; 3 seeds for final models | V14 does not roll out; AdaLN's small, task-dependent edge matches the per-horizon FiLM arm's status. No change |
| 2603.14482 V-JEPA 2.1 | §2.2–§2.3.2 | Supervising the unmasked context tokens (local) raised dense scores (ADE20K 22.2 → 33.9 mIoU) but cost global ones (SSv2 72.8 → 62.5, IN1K 82.2 → 72.6), with a λ warm-up needed for stability; supervising four encoder depths ("deep self-supervision") recovered the global scores (72.0, 80.8) and improved dense further (38.6) | A third data point on local content in a target (with data2vec and ProteinJEPA): local supervision trades against global unless balanced. Strengthens c2's read |
| 2403.00504 IWM | §1, §4.2, App. C | Predictor capacity decides whether the encoder becomes invariant (abstract; better linear probes) or equivariant (keeps more; better with predictor fine-tuning); 12-layer predictor reached equivariance in 1 of 5 seeds, 18-layer in 4 of 5 | Context for §5.2 (V13's concept module unchanged). Seed sensitivity of a predictor-capacity effect: one more reason the tiny test's single seed is weak |
| 2602.19322 US-JEPA (SALT), 2511.18424 CrossJEPA | targeted passages | Static-teacher training (SALT, Li et al. 2025): a frozen pretrained teacher replaces the EMA teacher; "a high-performing student can emerge by predicting the latent space of a static teacher"; CrossJEPA caches a frozen 2D teacher's targets once | An arm V14 could run cheaply: V13-tiny's final encoder, frozen, as the target encoder (no co-adaptation, the codebook fitted once). Tests whether co-adaptation is needed at all |
| 2610.07006 STOCK-JEPA | abstract + method passages | Future representation = an anchor from a simple prior (projected statistics) + a revision predicted from context; separate losses; optimal revision reduces the anchor's error by E‖Δ‖² | The same split as b (the change beyond what the local features explain linearly): a precedent in another domain |
| 2607.25337 TD-JEPA (temporal distance), 2601.00844 value-guided JEPA, 2608.06799 PSG-JEPA, 2608.29029 Flow-JEPA | method/conclusion skims | Directed temporal-distance costs for MPC (Spearman 0.95 vs 0.65 for latent L2); IQL value shaping; training-only grounding heads for state and multi-horizon state change; vector-field (flow) dynamics robust to visual perturbation | Planner- or robotics-specific. PSG's transition grounding is inverse dynamics in language terms (already an arm, §8.4). No change |

**Proposals (cluster d).**
- d1. §5.3: VLWM's and HWM's numbers as corroboration of direct horizons; HWM's capacity-controlled result as evidence
  that hierarchy itself, not size, carried the gain.
- d2. §5.5: HWM's stride optimum; V14's third level already has a level-specific target view (HWM's stated limitation).
- d3. §6.2: the horizon curriculum considered and not adopted (mixed, small, one run per cell), one sentence.
- d4. A static-teacher arm (SALT): V13-tiny's final encoder frozen as the target encoder, for the queue's later-tier
  list. Question for the advisor: arm, or a sentence only.
- d5. §2.2 rows: VLWM, HWM, V-JEPA 2.1, SALT/US-JEPA, STOCK-JEPA.
- d6. §6.3's local-half read: add V-JEPA 2.1 beside ProteinJEPA, with data2vec as the counterexample.

## Cluster (e): continual learning (read 22:45–22:55, 14 Oct), against §9.2's list

| Paper | Depth | What it shows | §9.2 item |
|---|---|---|---|
| 2610.00722 JEPA-TTT | §3.3–§3.4, §4.2, §4.4, §6 | Test-time training of the predictor only, encoder frozen so targets cannot drift; one AdamW step per replanning step; predictor, optimiser state and buffer persist across episodes. +153% best score, +113% mean AUC over the frozen model across 8 dynamics shifts. Persisting the adapted state: best score 0.289 → 0.729 at equal updates. Dense windows (every temporal offset) help; replay adds a little (AUC 0.548 → 0.571). Forgetting **not measured** (their §6) | (a)/(b): precedent for "freeze the target encoder, adapt the dynamics"; persistence of adapted state matters. No retention evidence |
| 2610.03713 Stratified retention | §3–§5, §7 (position paper, no experiments) | Strata by how long knowledge stays true (invariants, regularities, instance configuration, transient). Standard forgetting metrics cannot tell breaking from warranted revision; a never-updating model gets the best forgetting score. Proposes invariant regression tests after every adaptation round, and revision latency (observations until predictions reflect a changed fact; never-revised facts counted separately). Replay that stores an old fact keeps reasserting it after the fact changed | (g): the knowledge-update stream should be scored by revision latency, beside the stable panels re-run after every phase (V13's A[i, ·] already is the regression suite); with replay, superseded bindings must be evicted from the buffer |
| 2607.09785 Continual-SSL survey | §IV-A distillation (targeted) | CaSSLe and PFR: *projected* distillation, a learned projector maps the current representation back to the previous model's, "freeing the encoder to learn new features"; CLA (EMA as the old model, online) acts mainly as a regulariser and scales poorly | (c): precedent for latent anchoring. V14 anchors through the frozen head, a fixed projector: stricter, less plastic. CaSSLe's form (a trainable copy of the head as the anchor's projector) is the named fallback if anchoring costs acquisition |
| 2609.36952 ER-JEPA | (read in cluster c) | Replay in LLM-JEPA fine-tuning: historical replay beats the same loss on the current batch by 2.2–3.8 points; uniform selection 85.0%, content-based 83.7% | Replay as a method (§9.3): uniform replay is not worse than content-based selection in a language JEPA |
| 2609.32592 SPACE | abstract + section titles | Streaming video memory: evict the entries whose removal least harms future prediction (counterfactual predictive utility) | The bank resets per chunk and has not overflowed (CPU test: 0 overflow); no change |

**Proposals (cluster e).** e1 §9.2(b): JEPA-TTT as the precedent, with "forgetting not measured". e2 §9.2(c): CaSSLe/PFR
as the precedent; the trainable-projector variant as the fallback if acquisition drops. e3 §9.2(g): revision latency
and never-revised count; evict superseded bindings from replay when replay is combined (O2). e4 §9.3: ER-JEPA's
uniform-vs-content result beside the replay method.

## Full K1–K5 checks on V13-tiny (finished 22:43, `v14_design/checks/checks.json`, 48 rows per family, six checkpoints)

Rules at the final checkpoint (`rules()` as pre-registered 19:40):
- H3: persistence within 5 points of the head in grid, prog, state (3 of 4), but χ_G < 0.2 only in grid (0.032) and
  prog (0.190); state 0.378, plan 0.907 → **not confirmed** (the rule needs both in ≥ 3 families).
- G's change dead: no family (χ_dG 0.82–0.99) → global half stays. H1: no family. Local-only matches u_g: no family
  (lrow probe 0.29–0.79 against u 0.79–0.99).
- **H2b confirmed** by plan alone: mean code-usage perplexity 3.78 (< 4); others 7.6–10.7.
- Default run: target view [a; b], λ_max 0.1, **uncentred SIGReg on b added** (the H2b response).
Consequence to raise: with the centred and the uncentred test both on b, the between-document variance of b is pushed
toward zero, so V4's χ_b ≥ 0.5 is largely enforced by the loss; V4 then reads whether the barrier holds, and the
evidence about slow collapse moves to V1, the action-sensitivity read and the per-half accuracy.

## Cluster (f): scale and evaluation (read 22:53–22:57, 14 Oct)

Already cited in V14 before this pass: V-JEPA 2, DINOv3, LeJEPA's scale, LLM-JEPA's, ConceptLM's.

| Paper | Depth | What it shows | Bearing on V14 |
|---|---|---|---|
| 2610.10515 RoboJEPA | §2.1, abstract, targeted | 8B predictor (the largest JEPA predictor reported) over a **frozen** V-JEPA 2.1-G encoder, 12 embodiments; teacher-forcing plus rollout loss; imagination error follows a second-order power law in compute and predicts planning performance; QK-normalisation "of particular importance" for stability when scaling; V-JEPA 2.1 targets scaled predictably and V-JEPA 2's did not | §10.3's ceiling for a JEPA predictor over a frozen target; scale precedent for the static-teacher arm (§3.6); the target's quality decided whether scaling was predictable |
| 2609.31680 JEPA for time series | §5 | JEPA pretraining's effect depends on the backbone: dense global-attention backbones gain little or lose, structurally constrained ones gain; swings above 25 points across backbones | §12.2 risk 6: a result on V13's backbone is not a result for another backbone |
| 2608.23720 Platonic world models; 2502.11831 intuitive physics; 2506.09943 CausalVQA; 2506.09987 MVP; 2606.09646 physics probing; 2602.08968/2605.21800 stable-worldmodel; 2603.29966 surgical scaling; 2111.06377 MAE | abstracts (read in pass 1); conclusions for 2608.23720 | Predictor alignment across vision priors; video evaluation suites and a platform; MAE | Vision/video evaluation; no bearing on a language model's design |

**Proposals (cluster f).** f1 §10.3: a row for a JEPA predictor over a frozen encoder (8B, RoboJEPA). f2 §12.2 risk 6:
the backbone dependence (2609.31680). f3 §3.6: RoboJEPA as the largest precedent of a predictor trained over a frozen
encoder.

## A-tier completion (22:55–22:57): the five A-tier papers not yet in a cluster table

| Paper | Depth | What it shows | Bearing on V14 |
|---|---|---|---|
| 2608.21160 Human-JEPA | abstract + targeted passages | "Anchored forecasting": dense targets pinned to a frozen copy of the initialisation prevented a silent collapse of dense perception; a pure past-to-future split instead of block masks avoided a 5-point action cost and a 17-point re-identification collapse | §3.6: precedent for a frozen-copy teacher (there the student starts from the same checkpoint; V14's arm starts the student from scratch so it compares with the default run). §2.4: temporal views over masking |
| 2610.05731 T-JEPA (remote sensing) | abstract + targeted passages | Predicts future latents at variable time gaps directly rather than rolling out; masked pixel reconstruction alongside; time-gap-dependent predictability | Corroborates §5.3 (no curriculum found in the passages read; R8's "curriculum" was VLWM's) |
| 2608.29434 point-cloud world models | conclusion, limitations, targeted | Names the geometric and the slow-feature shortcuts; Delta-JEPA's action-reconstruction loss beat SIGReg (Point-LeWM) on OGB-Cube, while SIGReg was more robust on sparse scenes (80% vs 43% of clean success under 0.25% noise; 99% vs 64% under 50% dropout) | §8.4: mixed evidence for the inverse-dynamics arm against SIGReg, environment-dependent |
| 2608.07409 UniJEPA | conclusion, limitations | Photometric and temporal world models in one latent space under one provable anti-collapse (Gaussian) regulariser; notes the Gaussian regulariser "may limit representational capacity at extreme scales" | A stated (untested) caution for SIGReg at scale; no change |
| 2602.11389 Causal-JEPA | conclusion | Object-level masking as counterfactual queries; strong gains on counterfactual visual reasoning | Object-centric vision; no bearing |

## B-tier method skims (22:56–22:58): abstract + first method section, one line each

| Paper | One line | Bearing |
|---|---|---|
| 2307.12698 MC-JEPA | Motion (self-supervised flow, coarse-to-fine) and content (VICReg) features learned in one shared encoder by separate objectives | Loose precedent for one encoder carrying a change half and a content half; no change |
| 2507.00028 HiT-JEPA | Hierarchical trajectory embeddings (point, segment, trajectory) | No transferable mechanism |
| 2608.00491 HP-JEPA | Hierarchical graph partitions as multi-resolution targets | None |
| 2411.16695 R-JEPA | Recurrent JEPA trained by forward propagation (no BPTT) on video | None |
| 2511.17354 DSeq-JEPA | Regions predicted in an order of discriminativeness | None |
| 2603.00049 BiJEPA | Bidirectional, cycle-consistent prediction with stabilisation | A causal LM cannot predict backward at inference; none |
| 2312.04000 LiDAR | LDA-rank metric (within-sample vs between-sample covariance) predicts linear-probe quality better than effective-rank metrics | §6.3's effective rank is already "necessary, not sufficient"; none |
| 2607.16274 JEPA predictor as operator | The frozen predictor completes occluded features across encoder families | None |
| 2610.05043 CI-JEPA | Predicts the representation change ΔZ under a counterfactual image edit | A weak vision precedent for change targets (§4.3); not cited |
| 2502.14819 PLDM | Planning with latent dynamics generalises from suboptimal offline data better than RL | None |
| 2507.19468 DINO-world | Future prediction in frozen DINOv2 space at scale; fine-tunable on actions | Same evidence as DINO-WM (§3.6 cites that) |
| 2610.02922 TwinJEPA | Action-preference objective on TD-JEPA-style dual encoders | None |
| 2509.25449 TS-JEPA | JEPA for time series | None |
| 2610.03106 S2S-JEPA | At long lead times forecast only the slow components that stay predictable | The mirror image of §5.3's far-horizon trap: far ahead, what is predictable is what is slow; V14's baselines guard against exactly that. Not cited |
| 2410.03755 Denoising JEPA | JEPA as generalised next-token prediction with a diffusion loss for generation | None |
| 2507.15216 JEPA + diffusion noise | Diffusion noise added to JEPA views | None |
| 2604.13518 PRL survey | Taxonomy: alignment, reconstruction, predictive representation learning | None |
| 2603.16281 Laya | LeJEPA (SIGReg, no EMA) for EEG | None |
| 2409.05929 M3-JEPA | Multi-gate MoE predictor for cross-modal alignment | None |
| 2607.15295 AV-JEPA | LeJEPA for audio-visual with modality dropout as views | None |

## C-tier (abstract only, read in pass 1): 56 domain applications with no transferable mechanism

Audio and music (2311.15830, 2408.02514, 2507.02915, 2509.23238, 2607.22000, 2403.18843; 2405.08679 was skimmed in cluster c); vision-language
and multimodal (2503.06380, 2605.03245, 2606.25225, 2510.00974); robot policies and driving (2602.10098, 2609.37250,
2608.10780, 2609.20277, 2605.20811, 2609.34085, 2602.18639, 2605.13013, 2609.24749); 3D, graphs, molecules and physics
(2404.16432, 2409.15803, 2608.05774, 2309.16014, 2506.18194, 2502.03933, 2610.08400); time series, trajectories and
tabular (2405.10093, 2406.12913, 2410.05016, 2605.31580, 2607.01145); biomedical (2403.11772, 2409.19407, 2410.08559,
2410.13867, 2507.03633, 2509.15470, 2601.15891, 2602.02603); remote sensing and Earth (2311.15153, 2412.14123,
2504.03169, 2605.31068, 2605.14120, 2607.09205); video benchmarks and perception (2506.09849, 2512.06232, 2510.01502,
2601.09524, 2606.23256); networks (2602.07000, 2607.09798); masked-image details (2308.00566, 2407.10733); 2404.08471
(V-JEPA, the method V-JEPA 2 extends) and 2304.07193 (DINOv2, superseded in V14 by DINOv3).

## Whole-document review (from 23:30): papers and numbers added

| Paper | Depth | What it shows | Bearing on V14 |
|---|---|---|---|
| 2411.04983 DINO-WM | §3.1, §4.3–§4.4 (headings of the rest) | A world model trained over *frozen* DINOv2 patch features plans zero-shot: success 0.90 on PushT against 0.30 for DreamerV3 and 0.32 for IRIS, Reach 0.92 against 0.64; the frozen features matter: one global vector per frame (DINO CLS) 0.44 on PushT, R3M 0.42, ResNet 0.20 | §3.6: precedent for learning dynamics over a fixed encoder, with its numbers; which teacher's features are used matters (cited in §3.6 since cluster d, without a notes entry until now) |

Decisions on the A-tier papers not applied: **T-JEPA (2610.05731)**, abstract and targeted passages: predicts latents at
variable time gaps directly, but with no measured comparison against rollouts; VLWM and HWM already carry §5.3, so no
edit. **UniJEPA (2608.07409)**, conclusion and limitations: its untested caution about Gaussian regularisers at extreme
scale gets one sentence under §10.3, which also states that V14's SIGReg acts only on the two target projections.
**Causal-JEPA (2602.11389)**: object-centric vision, no bearing.

**Post-hoc b4 follow-up** (`v14_design/check_b4_var.py` → `checks/b4_var.json`; not pre-registered; 8 fresh rows per
family, final checkpoint): within each family the 32 words leave 5–20% of the normalised targets' per-coordinate
variance unexplained (grid 4.6–5.7%, prog 11–19%, state 8–11%, plan 11–20%, text 12–16%); within-family variance per
coordinate 0.40–0.46 in plan, 0.57–1.13 elsewhere. Any 32-word quantiser of an isotropic Gaussian leaves ≥ 80.5%.
V13's targets took few distinct values per family (H2b's description; like K5, it cannot separate few concepts from a
collapsed quantiser). In §6.2.

**Reading depth of the cited catalog papers** (for the References heading): of the 41 cited, 13 were read in full
(method, results, limitations) and 28 at the sections that bear on V14 (listed in each row's depth column).
(SC-JEPA, 2602.04643, counts among the 28: its theory section was read by its headings only.)
