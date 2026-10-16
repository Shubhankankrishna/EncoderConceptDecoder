# V14 — Architecture: V13 with a stable, informative next-concept target

Status: **design complete, not implemented** (14 Oct 2026, written 18:50–19:45 IST). Nothing here is implemented or
trained. Sections were written in order and each was reviewed by the advisor before the next started; the
consultations are logged in [`v14_design/ADVISOR_LOG.md`](v14_design/ADVISOR_LOG.md).

Inputs: [`v13_NOTES.md`](v13_NOTES.md) (the V13 design, §0–§14), [`v14_NOTES.md`](v14_NOTES.md) §1 (the NCP-collapse
observation and its two hypotheses), the V13-tiny-v2 run (`v13_tiny/`: `DEVIATIONS.md`, `REPORT.md`,
`runs/base/log.jsonl`), and the V1–V12 constraints C1–C9 that V13 inherited.

**Rule for this document.** V14 is V13 plus changes. Every V13 mechanism not named here is kept exactly as
`v13_NOTES.md` and `v13_tiny/DEVIATIONS.md` specify it. Every change states the evidence for it, what it costs, and the
check that would show it wrong.

## 0. Summary

**What V14 is.** V13 with its next-concept target redesigned, and nothing else changed that the evidence does not
require. In the only V13 ever trained (V13-tiny, 7.7M parameters, one seed), the next-concept head predicted the
same code tuple for every group of the one document inspected, and at an interim checkpoint (about step 2,400) a probe
on the concept module's *input* already predicted the next group's codes at 76–91%, so the module had nothing left to
compute (§1.2; the final-checkpoint reads are tonight's). The likeliest reading (§1.3, to be
confirmed by tonight's checks): the target was the online encoder's pooled summary of the whole prefix, which barely
changes from one group to the next, which the JEPA literature calls slow-feature collapse (H-JEPA). V14 treats the
concept module as a latent world model trained in the JEPA way (§2, §5) and rebuilds what it predicts:
- the target comes from an **EMA twin** of the encoder, through a **target view** kept separate from the concept
  module's input (§3, §4);
- the view has two typed halves: what the group's tokens were (**local**), and how the group changed the document's
  global summary beyond what its own tokens explain (**global**: a difference target whose linear local part is
  regressed out by a running least-squares map P) (§4.3);
- one shared head predicts the codes of that view **1–4 groups ahead** (§6);
- slow collapse is countered **by construction**: the global half is a change, so nothing that is constant within a
  document can be in it (§4.3). **Document-centred SIGReg** (time on the sample axis) whitens each half for the
  codebook and penalises exact within-document collapse by a fixed amount, but it is a backstop, not the defence: near
  collapse the projector's scale can meet its variance part, and in H-JEPA's runs SIGReg with time on the sample axis
  did not stop slow collapse on its own (§8.3). Inverse dynamics, the barrier the robotics papers use, is only an arm,
  because in language the "action" is part of the observation and NTP already puts it in the encoder (§4.3, §8.4; a
  reversal of earlier advice, logged as consultation 6).

The decoder, the online encoder, the concept module's structure, the reach bounds and V13's continual-learning
machinery are unchanged (§3.1, §5.2, §7, §9).

**The six fixes the user listed are all in.**
- The EMA target encoder is the default (§3).
- Predicting 2–4 groups ahead is the default: horizons 1–4 (§6.2).
- A stricter dead-code reset is the default (§6.1); the larger codebook (N = 64) is an arm.
- λ is scheduled with a gradient-share cap (default, §8.2) and rises to 0.2 if H1 is confirmed.
- The variance–covariance term is in its general form: SIGReg enforces it and every higher moment, and VICReg's own
  terms are an arm (§8.3). It is placed on the target view rather than on m, which is now only the input (§4.1): a
  deliberate change from the request as worded.
- Lowering β₁ is an arm, safe only now that input and target are decoupled (§4.2).

| Part | V14 default | Arms (tested, not default) | § |
|---|---|---|---|
| Encoder | V13-tiny-v2's, plus an EMA twin of everything from the ids to the target view; momentum set by lag (100 steps, cosine to 1); the twin, normaliser, P and codebook freeze together at 75% of training | the local twin, with the local view | §3 |
| Concept input | V13's (learned β mix of three pooled sources, p_g = W_P m_g) | β₁ lowered | §4.2 |
| Target view | τ = [a; b]: a = T_loc[mean pools of x and L^enc]; b = T_glob(ΔḠ − Δℓ·P); first groups of documents masked | the local view only; statistic projectors (the default if §4.3's check fails) | §4.3–§4.4 |
| Concept module | V13's, read as the world model's forward dynamics | a successor-code head trained by TD; a third level (scale) | §5 |
| Head | one head for horizons 1–4, w = (0.4, 0.3, 0.2, 0.1); PQ 4 × 32 over τ; stricter dead-code rule | equal weights; "V13 plus"; N = 64; per-horizon FiLM (triggered); the branch head and soft targets (later tiers, read-gated); the fixed-quantiser control | §6 |
| Decoder | V13's (D-confined) | ĉ injection, with the argmax tuple's codewords (§6.4) | §7 |
| Objective | L_NTP + λ·L_NCP + ν_S·L_SIG; λ_max = 0.1 with the gradient-share cap; ν_S = 0.05 | VICReg's terms; inverse dynamics; the uncentred test on b (triggered); λ_max ∈ {0.05, 0.2}; the cosine latent term; NextLat | §8 |
| Continual learning | chosen by tonight's V13 matrix (§9.4); a frozen era; the adapter-off base as the target encoder | latent anchoring; a tuple-level NCP softmax; code gates; the knowledge-update stream | §9 |

**Decided before V14 runs.**
- Tonight's checks K1–K5 on V13-tiny's checkpoints (`v14_design/checks.py`, rules pre-registered in §1.4) fix the
  default run's configuration: the local view if G's change carries no signal, λ_max = 0.2 if the head predicts the
  marginal (H1), and the uncentred test on b if codes collapse within a family (H2b).
- Tonight's continual matrix (sequential LoRA, replay, V13's method) fixes V14's continual default (§9.4).
- The projector check of §4.3 (a SIGReg-only map against the top-64 principal subspace, on V13-tiny's local features)
  decides whether T_loc and T_glob are trained or set as statistics.

**Cost.**
- About 25K new trained parameters at the tiny scale (two projectors and the horizon embeddings): 7.77M trained and
  deployed, under the 10M limit. The twin stores another 3.9M during training only (no gradient, discarded after
  training), so about 11.7M are stored while training. The limit is read as the model's own parameters, as it is for
  every EMA-target method.
- About +17% time per training step on the M2, for the twin's forward (§3.5).
- At the 2.8T class: +6.7% of training FLOPs, and an fp32 twin of 1.26 TB sharded with the optimiser state (§10;
  accounting, not measurement).
- At inference nothing new runs except the target view where a code is needed (§7).

**How it is tested.** On the M2 against V13-tiny and the user's baseline transformer, with V13-tiny's data,
evaluation and 11-hour budget reused byte for byte. The readings V1–V7 and the rule "one arm after the default run,
chosen by its readings" are pre-registered in §11. **Every result will be one seed.**

**What is untested anywhere.**
- The residue map is new.
- A difference target backed by centred SIGReg has never been the only defence against slow collapse; uncentred
  SIGReg with time on the sample axis failed at it in every run of H-JEPA's ladder (§8.3).
- EMA targets have never been quantised by EMA k-means in an autoregressive language model.
- No JEPA-style objective has been reported near 3T parameters (§10.3, §12.1).

**The top risk** (§12.2): G's change may carry nothing non-local on this data (the residue map's R² near 1). Then V14
is V13 with a fast but local target, V2 fails, and the redesign does not make the concept path necessary.

## 1. What V14 must satisfy: inherited constraints and the new evidence

### 1.1 The inherited constraints

V13's nine constraints (v13_NOTES §1) stand. Each one constrains V14 the same way unless the last column says
otherwise.

| # | Constraint (evidence in v13_NOTES §1) | Consequence for V14 |
|---|---|---|
| C1 | A decoder that can reach distant tokens without the concept path routes around it. | Unchanged: Option A's role split and the D-confined decoder stay. New in V14: the same failure can recur *inside the objective*. If the NCP target can be predicted without the concept path, NCP shapes nothing (§1.3). |
| C2 | Squared-error regression of the next concept predicts the conditional mean. | Unchanged: NCP stays a classification over product-quantised codes. The EMA target encoder (§3.2) changes where the target comes from, not the loss form. |
| C3 | NCP has a price (V12: +0.28 nats on text at λ = 0.1); 37–64% of V12's codes reassigned between steps 4k and 8k. | λ stays scheduled and gains V13's gradient-share cap as a default (§8). A stable target is now a design goal, not an arm (§3.4, §6). |
| C4 | Every loop measured in this repository was inert. | Unchanged. The V13-tiny interim read agrees: at inference, exiting E2 after loop 1 changed answer NLL by at most 0.04 on any task type (`runs/tmp_eval/diag`, two items per type, interim). |
| C5 | A recurrent workspace starts at full scale and reads its inputs read-only. | Unchanged (E2). |
| C6 | Continual-learning gates protect only what their reference covers, and only dissimilar skills. | Unchanged; §9 adds a code-gated option aimed at the related-skill case. |
| C7 | Mechanisms measured on an incompetent base are uninformative. | Unchanged: V14's reads are gated on base competence against the same baseline transformer (§11). |
| C8 | A widened residual stream specialises, but not by guarantee. | Unchanged (two branches per role). |
| C9 | Long-range attention at 1M tokens must be compressed. | Unchanged (§7, and v13_NOTES §7.2). |

### 1.2 What the V13-tiny run showed about next-concept prediction

The V13-tiny-v2 run (7,744,181 parameters, one seed, 11 h on the M2 Air) is the only V13 ever trained. It was
pre-registered and is not changed mid-run (v14_NOTES §1). Its final-checkpoint readings (`v13_tiny/REPORT.md` §5) are
still pending. What exists now:

| Evidence | Value | Source and caveat |
|---|---|---|
| NCP loss at step 1, summed over 4 segments | 13.9–14.0 in every family | `runs/base/log.jsonl`; uniform guessing gives 4 ln 32 = 13.86 |
| NCP loss and per-segment accuracy, last logged batch of each family (steps 10,200–10,500 of about 12,400) | grid 0.40 / 0.97; programs 2.65 / 0.76; state 1.89 / 0.88; planning 1.12 / 0.90; text 1.96 / 0.87 | Same log. One single-family batch per 50 steps; the family explains 55% of the logged variance |
| Code-usage perplexity per segment (of 32) | 24.6 (step 10,500); 24.9, 25.1, 25.4, 23.1 per segment (interim checkpoint) | `ppl` in the log; `runs/tmp_eval/diag/diagnostics.json`. An EMA over batches of *all* families, so it cannot show collapse inside one family |
| Probe test (§6.2's acceptance test), interim checkpoint (about step 2,400) | probe on p_g: grid 0.897, programs 0.822, state 0.911, planning 0.756. u_g minus p_g: −0.6, −0.7, −0.3, +2.9 points. Fails the 5-point bar in all four families | `diagnostics.json` (about 1,800–1,900 groups per family). Interim; the pre-registered read is at the final checkpoint |
| One document, step 6,898: "Lily found a red ball in the garden and gave it to her brother." (4 groups) | The head predicted codes [8, 22, 15, 30] for every group. The targets m₁ and m₂ quantised to that same tuple (target probability 0.69–0.94 per segment); m₃ quantised to [9, 26, 20, 12] (target probability ≤ 0.0016) | `v13_viz/data/journey.json`. One document is weak evidence |
| Learned source mix β = (x, L, G) | (0.107, 0.107, 0.787) at initialisation → (0.106, 0.078, 0.816) at step 10,500 | Log. The global source took share from the local tap |

The second and fourth rows matter most. The NCP head is right on 76–97% of segment codes, but a probe on the
*current* group's input p_g already predicts the *next* group's codes at 76–91%. Predicting the next concept is
therefore close to an easy task, and the concept module adds nothing to it (u_g minus p_g ≈ 0).

### 1.3 Three hypotheses, and two easy solutions any target must defeat

v14_NOTES §1 names two hypotheses. The evidence above adds a third, and it splits the second.

- **H1: the head predicts the marginal.** u_g carries too little information, so the head outputs the most frequent
  codes whatever the input. Signature: the entropy of the predicted codes over a dataset is much lower than that of
  the target codes, and the conditional entropy H(target | u_g) stays close to H(target). With a distributional head
  trained by cross-entropy, H1 is never a global optimum when targets differ across contexts (VJEPA, arXiv
  2601.14354, Thm 1): if it occurs, training has failed to reach the optimum (too little in u_g, or too little
  pressure); the marginal is not the optimum itself.
- **H2: the targets collapse.** m maps to few codes.
  - *H2a, globally:* few codes in use overall. The usage perplexity of about 25 of 32 per segment argues against it.
  - *H2b, within a family:* each family uses few codes. The log cannot show this, because the perplexity is an EMA
    across family-homogeneous batches. v13_NOTES §6.1 promised a per-family perplexity; V13-tiny does not log it.
- **H3: the target is slow.** The code names the *document* (its family, its topic, its state so far) rather than the
  *group*, so it rarely changes from one group to the next and "copy the current code" scores nearly as well as the
  head. The mechanism is first-principles: m is 82% the pooled global stream G^enc, which has unbounded reach and
  summarises the whole prefix. A summary of a long prefix changes little when four more tokens arrive. Signature: the
  persistence rate P(code_{g+1} = code_g) is close to the head's accuracy, and the within-document share of m's
  variance is small, smallest for the G source. Supporting evidence: the probe on p_g (76–91%) and the one-document read.

  **H3 has a name in the JEPA literature: slow-feature collapse.** H-JEPA (Zhang, Terver, Rabbat, LeCun and
  Balestriero, arXiv 2610.06805, Oct 2026) defines an encoder as slow-feature collapsed when its embeddings are
  constant within each episode and depend only on what is fixed across it, and measures the degree by χ, the
  within-episode share of latent variance (0 at exact collapse). Its analysis makes three points that carry over to
  V14 directly. (i) Such an encoder reaches *zero* prediction loss, so training selects it: in their runs every
  regulariser-only configuration collapsed. (ii) Variance-covariance terms, whitening, entropy penalties and SIGReg act
  only on the *marginal* distribution of embeddings, and their Proposition 1 shows that an encoder can satisfy any
  such constraint while collapsing. A barrier has to act on the joint law across time, which for SIGReg means
  computing it with time on the sample axis (their Proposition 2); they call that barrier real but small, and in their
  ladder of SIGReg weights with time on the sample axis and no other term, every run still collapsed (their Table 12;
  §8.3). (iii) An inverse-dynamics term, which must identify the action between two consecutive latents, is the one
  Ω(1) barrier in their analysis; a term that rewards within-episode motion (time repulsion) kept the predictor off the
  identity but did not select what moved (their App. L.6). MotionJEPA (arXiv 2609.23881)
  reports the same bias toward slow features in standard JEPA training. In a language model the "episode" is the
  document, and the "action" between concept g and concept g+1 is the four tokens of group g+1, which are always known.
  But in a language model the action is part of the observation, which changes what an inverse-dynamics term does
  there (§8.4): V14 does not use it as its barrier.

  Two earlier results explain why the slow solution wins. Sobal et al. (arXiv 2211.10831) first showed JEPA world
  models locking onto what is fixed within a sequence (a static background) and ignoring the moving agent, and asked
  for "an additional constraint that prevents the representations from being constant across time". A theorem for
  linear predictors (arXiv 2511.09783, Thm 3.4) says why: with an EMA target that tracks the encoder, the loss of
  predicting the next *level* is minimised exactly by features that are invariant within a regime, with the predictor
  the identity on them. With the document as the regime, copy-current is the optimum of level prediction, not an
  accident of training; a target that is a *change* removes that subspace by construction, because an invariant's
  change is zero.

Two easy solutions follow from these. A V14 target must defeat both, so both are logged during training (§11):
- **Copy-current.** Predict that group g+1 has group g's code. It wins when the target is slow (H3). Measured by the
  persistence rate per segment and family.
- **Local-only.** Predict group g+1's code from the last R_L tokens alone, with no concept state. It wins when the target
  is a function of local context, and then the concept path is optional for NCP: C1 reappearing at the level of the
  objective. Measured by an MLP probe on L^enc at the release row 4g+3, against the same probe on u_g.

**What each listed fix does under each hypothesis.** This table is the reason the fixes cannot simply all be switched
on and expected to help:

| Fix (v14_NOTES §1) | H1: head predicts the marginal | H2: target collapse | H3: slow target |
|---|---|---|---|
| Raise or schedule λ | helps: more pressure on u_g | no effect on the target | inert: the task is already nearly solved; pays C3's cost on NTP |
| Predict 2–4 groups ahead | helps, if the target varies | no effect on the target | inert while the target is constant inside a document: copy-current wins at every horizon |
| Lower β₁ (the embedding's share) | helps the input only if the input is starved | in V13, β also sets the target, so it changes the target too | **hurts in V13**: the identity tap is the *fastest* part of m (the bag of the next four tokens). Safe only once target and input are decoupled (§4) |
| Variance and covariance term on m (VICReg-style, over the batch) | no effect | helps H2a; helps H2b only if computed within a family | inert over the batch marginal (H-JEPA, Prop. 1): global variance is satisfied by differences between documents and families |
| Stricter dead-code reset, larger codebook | no effect | helps H2a | inert: codes can be diverse globally and still constant within a document |
| EMA target encoder | no direct effect | no direct effect | no direct effect |
| *Added from the literature:* SIGReg with time on the sample axis (each document's group latents centred on their own mean, then pooled) | no effect | helps H2a and H2b (an isotropic Gaussian target spreads every PQ segment's usage) | **a backstop**: a constant-within-document latent centres to 0, which SIGReg penalises by a fixed amount at exact collapse; near it the projector's scale can meet the variance part, and the uncentred version failed in H-JEPA's runs (§8.3) |
| *Added:* predict the *change* of the global summary (ΔḠ), not its level (MotionJEPA's difference targets) | no effect | no effect | **helps**: the level is what is slow; the change is what the group did |
| *Added:* remove from that change the part the group's local features explain linearly (a running least-squares map, §4.3) | no effect | no effect | **helps**: what remains is what G added beyond the group's own tokens and local context, so the global half cannot become a second bag of tokens |
| *Considered, kept as an arm:* contrastive inverse dynamics, the group's tokens as the action (AC-MTM's Action-NCE form) | indirect | indirect | an Ω(1) barrier in robotics, where the action is outside the observation and the repair comes from its gradient into the encoder (arXiv 2610.03137). In language the action *is* the observation, so the term rewards encoding the tokens themselves, which is the local half's content (§8.4) |

The EMA target encoder fixes none of the three directly. Its job is different: it makes the target a slowly moving
function of the weights, so the codes keep their meaning while the head learns them (C3's 37–64% reassignment), and
so a code-keyed memory stays valid (§9). V14 adopts it for that reason, as the user asked, and does not claim it
cures the collapse: momentum sets how fast modes escape or collapse, not which (§3.2).

### 1.4 The checks that decide between them

Pre-registered here, before any of them is run. They need only the V13-tiny checkpoints. **They run on the CPU, after
the base run ends, and never during the continual-budget probe (20:15–20:25 IST on 14 Oct).** Each check uses 48
fresh dev-time rows per reasoning family (generated at seed 99, as `diagnostics.py` does; each row packs many
documents, and the document count is logged) plus 48 rows of held-out TinyStories validation text, at `snap_010`,
`snap_030`, `snap_050`, `snap_070`, `snap_090` and the final checkpoint. The script is `v14_design/checks.py`.

| # | Check | Computed as |
|---|---|---|
| K1 | Entropies | H(predicted codes), H(target codes) and H(target \| u_g), per segment and family. H(target \| u_g) is the held-out cross-entropy of an MLP probe from u_g, in bits |
| K2 | Persistence | P(code_{g+1} = code_g) per segment and family, beside the head's accuracy on the same groups |
| K3 | Local-only | MLP probe from L^enc at row 4g+3 to group g+1's codes, beside the same probe from u_g and from p_g |
| K4 | Variance split (H-JEPA's χ) | Within-document share of the variance of each normalised source (Pool x, Pool L, Pool G), of the change of Pool G between consecutive groups, and of m, per family. H-JEPA's reference values: χ = 0.008–0.016 in its collapsed runs, 0.12 with its inverse-dynamics term on |
| K5 | Per-family usage | Code-usage perplexity per segment within each family |

Decision rules, fixed now:

| Outcome | Reading | V14 default it fixes |
|---|---|---|
| K2's persistence rate is within 5 points of the head's accuracy in at least 3 of the 4 reasoning families, *and* K4's within-document share of the G source is below 0.2 in those families | H3 confirmed | The target view's global part is the *change* in the pooled global summary, not the summary itself (§4.3) |
| K4's within-document share of the G source's change is below 0.05 in at least 3 of the 4 reasoning families | G carries no group-level signal at all | Fall back to the local view alone (§4.4) |
| K1: H(predicted) < ½ H(target) and H(target \| u_g) > 0.9 H(target), in at least 3 of the 4 reasoning families | H1 confirmed | λ_max 0.2 becomes the default instead of an arm (§8) |
| K5: code-usage perplexity (the mean over the four segments) below 4 in any family | H2b confirmed | The uncentred SIGReg test on the global half b is added to the default (§8.3; batches are single-family, so it spreads codes within a family) |
| K3: the local-only probe is within 2 points of the u_g probe in at least 3 of the 4 reasoning families | The target does not need the concept path | The target must include a non-local part; the local view alone is ruled out as a fallback (§4.4) |
| Any trend across `snap_010` … final in K1–K5 | Whether collapse worsens or recovers | Reported; changes no default by itself |

Quantifiers count the four reasoning families; text is reported beside them and not counted, because its documents
are a different regime. If the K4-change rule and the K3 rule both fire, the K3 rule wins and the default target view
stays. `checks.py`'s `rules()` applies exactly these rules and writes the resulting configuration of the default run
(§11 item 4). These checks decide defaults. Every alternative stays an arm with the same code path, so a wrong default
costs one run, not a redesign.

### 1.5 What V14 changes, in one table

| Part | V13 | V14 | § |
|---|---|---|---|
| Encoder | E1, Engram, E2 (R = 2), E3, exit reads | Unchanged, plus an EMA twin that computes targets without gradient | §3 |
| Concept input | m_g = learned-β mix of three pooled sources; p_g = W_P m_g | Unchanged; β₁ becomes an input-only lever | §4 |
| NCP target | sg[m_{g+1}] from the online encoder, the same vector as the input | A separate *target view* from the EMA twin, with fixed composition: local segments plus segments for the change in the global summary | §4 |
| Codebook | 4 × 32, EMA k-means, dead below 2% of uniform, random reseed | 4 × 32 kept for the default; stricter dead rule with reseeding from badly fitted targets; late freeze of the codebook; N = 64 as an arm | §6 |
| NCP head | one horizon, g+1 | horizons g+1 … g+4 | §6 |
| Anti-collapse | none | SIGReg on the online target view with time on the sample axis (document-centred); a global half that is non-local by construction; VICReg's variance and covariance terms and contrastive inverse dynamics as named arms | §4, §8 |
| λ | ramp 0 → 0.1 over the first 10% | same ramp, plus V13's gradient-share cap as a default | §8 |
| Logged during training | NCP loss and accuracy, global usage perplexity | also persistence, a local-only probe, per-family perplexity, within-document variance, the residue map's R² | §6.3, §11 |
| Continual learning | bank, gated LoRA, sleep | the same machinery, with the default chosen by tonight's V13 matrix (§9.4); a frozen era (codebook, normaliser, P, head); the adapter-off base as the target encoder; latent anchoring, a tuple-level NCP softmax and code gates as arms | §9 |

## 2. V14 as a JEPA and an energy-based model

The user asked that V14 align V13 with joint-embedding predictive architectures (JEPA) and energy-based models (EBM),
while staying an autoregressive, generative model that learns continually, and that it scale to the Kimi K3 class
(about 3T parameters). This section says what that alignment means precisely, what V13 already had, and what the
literature (surveyed on 14 Oct 2026; references at the end) contributes to each part.

### 2.1 What a JEPA is, and how much of one V13 already was

A JEPA (LeCun, *A Path Towards Autonomous Machine Intelligence*, 2022) has a context encoder s_x = Enc(x), a target
encoder s_y = Enc̄(y), and a predictor ŝ_y = Pred(s_x, z) conditioned on what is to be predicted (and, optionally, a
latent variable z). Its energy is a distance between the prediction and the target representation, E(x, y) =
D(ŝ_y, s_y), trained to be low on compatible (x, y) pairs. Unlike a generative model, it never reconstructs y itself,
so it can discard what is unpredictable. In deep linear models, latent prediction is biased toward high-influence
(predictive) features where reconstruction is not (Littwin et al., NeurIPS 2024). The trivial way to make every
energy low is to map everything to one point (collapse), so every JEPA carries a mechanism against it.

V13 was already a JEPA in all but two parts, and those two are the parts the V13 run shows failing:

| JEPA part | V13 | V14 | § |
|---|---|---|---|
| Context x | the prefix, tokens ≤ 4g+3 | same | |
| Target y | group g+1 (tokens 4g+4 … 4g+7) | groups g+1 … g+4 (the next 16 tokens) | §6 |
| Context encoder | the online encoder (E1, Engram, E2, E3) and the input pool → p_g | same | §3, §4 |
| Predictor | concept module C, then the NCP head | C, then one head conditioned on a horizon token (I-JEPA's target-position conditioning) | §5, §6 |
| Latent variable z | none: the factorised categorical over codes carries the uncertainty | same | §6 |
| **Target encoder** | **the online encoder itself, behind a stop-gradient** | **an EMA twin** | §3 |
| **Target representation s_y** | **m_{g+1}, the same vector the concept module reads as input** | **a separate target view: local segments and the change of the global summary** | §4 |
| Energy D | cross-entropy over the PQ codes of s_y | same, one per horizon | §6 |
| **Anti-collapse** | **the stop-gradient and the codebook's dead-code reset** | **EMA plus predictor asymmetry; SIGReg with time on the sample axis; a global half made non-local by construction; codebook hygiene** | §8 |
| Hierarchy | two levels, HiLP's: tokens (decoder, NTP) under concepts (k = 4) | same; a third level is a scale arm | §5 |

### 2.2 What V14 takes from the JEPA literature

| Work | Finding (as reported by its authors) | What V14 takes |
|---|---|---|
| I-JEPA (arXiv 2301.08243) | Predict target-encoder representations of several target blocks from one context block. Targets must be large enough to be semantic, and the context informative enough. EMA target encoder; the predictor is conditioned on the target's position | Targets are whole groups, up to four groups ahead; the head is conditioned on the horizon; the EMA twin |
| V-JEPA 2 (arXiv 2506.09985) | EMA teacher; encoder scaled from 300M to 1B with the predictor fixed; the action-conditioned variant plans by minimising a distance to a goal embedding | Scale evidence for EMA teachers; the planning energy in concept space (§2.3) |
| LLM-JEPA (arXiv 2509.14252, ICLR 2026) | NTP plus λ times a JEPA term (cosine between the predicted and target embeddings of two views), in LLMs of 1B–8B. About 2× training compute, reduced by dropping the JEPA loss on a fraction α of examples (cost 2 − α). With the generative loss's weight set to 0 the model produces only empty outputs | The JEPA term is auxiliary and never replaces NTP; JEPA-loss dropout is the scale lever (§10) |
| LeJEPA (arXiv 2511.08544) | SIGReg: an Epps–Pulley test of each of M random 1-D projections against N(0, 1) (M = 256–1,024, 17 integration knots), linear in batch size, all-reduced across devices, minibatch bias O(1/N); default weight 0.05; the isotropic Gaussian is the embedding distribution that minimises downstream risk; stable to a 1.8B ViT-g; no teacher or stop-gradient needed | SIGReg is V14's distribution regulariser, placed with time on the sample axis (§8) |
| H-JEPA (arXiv 2610.06805) | A stack of JEPAs, each level predicting further ahead in its own latent space; SIGReg at every level; slow-feature collapse, why marginal regularisers cannot stop it (Prop. 1), SIGReg with time on the sample axis (Prop. 2, "a real but small barrier"; every run of its SIGReg ladder without inverse dynamics collapsed, Table 12), inverse dynamics as the Ω(1) barrier. Three levels raised Visual AntMaze success from 18% to 73% | The diagnosis of H3 and the barrier design; a third level as a scale arm |
| AC-MTM, "No Gaussian Required" (arXiv 2608.17542) | Contrastive inverse dynamics (Action-NCE): each latent transition must identify its action among the batch's actions; a training-only head. 80.0 ± 2.0% success against 58.0 ± 2.0% for SIGReg on OGBench Visual Scene (SIGReg there placed time on the group axis, which H-JEPA's Prop. 2 says carries no barrier) | Considered for V14's barrier and kept as an arm: in language the action is part of the observation (§8.4) |
| MotionJEPA (arXiv 2609.23881) | Standard JEPA training is biased toward slow features; its regulariser predicts embeddings of *differences* | The ΔḠ half of the target view (§4) |
| SD-JEPA (arXiv 2605.31111) | Split the latent into a low-dimensional progression subspace and a content subspace with separate anti-collapse terms on disjoint coordinates, which "compose additively rather than competing" | Precedent for typed target segments with their own regularisers (§4, §8) |
| TD-JEPA (arXiv 2510.00739, ICLR 2026) | Multi-step latent prediction by temporal-difference learning (successor features in latent space) | An arm for horizons beyond four groups: bootstrap from the twin's own predictions instead of more heads |
| Discrete JEPA (arXiv 2506.14373) | JEPA with discrete semantic tokens for symbolic prediction | Precedent for discrete JEPA targets (method details not verified here) |
| ConceptLM, "Next Concept Prediction in Discrete Latent Space" (arXiv 2602.08984) | A VQ concept vocabulary, NCP plus NTP; 70M–1.5B from scratch on up to 300B tokens, and an 8B continual pretraining | V13's own lineage (NCP-Arch in v13_NOTES) |
| HiLP, "Hierarchical Latent Prediction for Language Models" (arXiv 2608.05806) | A higher-level abstract latent reduces compounding error in next-latent rollouts | V13 §9.1's mapping stands: pooling is HiLP's abstract latent, C plus the head its upper dynamics, the decoder its combined head |
| Latent JEPA (arXiv 2610.01947) | Autoregressive learning combined with joint-embedding prediction of future views (chemistry) | Precedent for one model that is both autoregressive and a JEPA |
| DINOv3 (arXiv 2508.10104) | EMA-teacher self-distillation trained stably at 7B parameters | Scale evidence for the twin (§10) |
| Sobal et al., "JEPAs focus on slow features" (arXiv 2211.10831) | JEPA world models on a moving dot over a background fixed within each sequence encode the background and lose the dot; reconstruction and inverse dynamics do not; a background that changes every frame is harmless | H3's first statement, and the case for a constraint against representations that are constant in time (§1.3, §8.3) |
| Koopman invariants in JEPAs (arXiv 2511.09783) | With a linear predictor and an EMA target that tracks the encoder, the loss is minimised exactly by regime-invariant features and an identity predictor (Thm 3.4) | Why level prediction converges to copy-current, and why a difference target removes that subspace (§1.3, §4.3) |
| Auxiliary tasks for JEPA (arXiv 2509.12249) | With consistent latent dynamics and a fitted auxiliary target, observations that differ under them cannot share an encoding (Thm 1); the transition loss alone collapsed completely | NTP as the encoder's anchor, and the projectors as the one part of the target it does not reach (§8.1) |
| Drive vs Decay (arXiv 2610.02344) | Linear coupled dynamics: a mode escapes collapse iff drive exceeds decay; momentum sets the speed, not which modes escape (Prop. J.7); the escaping count predicts effective rank (ρ = 0.91) | The twin is given no anti-collapse role (§3.2); effective rank as a read (§6.3) |
| LeWM+I, "Keeping JEPA world models plannable" (arXiv 2610.03137) | A SIGReg-only world model whose frames barely respond to actions learns latents that ignore the action (spread/motion 3.8 × 10⁻⁴); an inverse-dynamics loss into the encoder repairs it (about 85% of the gain); on the predictor alone, or with shuffled actions, it does nothing | Reversal #6 kept, since NTP supplies the encoder-side signal in language; the target-side action-sensitivity read (§6.3) and the shuffled control of the inverse-dynamics arm (§8.4) |
| DirectPred (arXiv 2102.06810) | BYOL's linear predictor set from an EMA of the feature correlation matches or beats the gradient-trained one (77.9% against 74.5%, STL-10) | Precedent for P as a statistic (§4.3) and for the statistic-projector arm |

### 2.3 The energy-based view: three explicit energies, one optional

1. **The concept head is an energy-based model with an exact partition function.** For horizon k, a code tuple
   c = (c₁, …, c_S) gets energy E_k(c | u_g) = −Σ_s ℓ_{k,s}(c_s), where ℓ are the head's logits. Because the energy is
   a sum over segments, Z = Π_s Σ_n exp ℓ_{k,s,n} is exact, so the cross-entropy is maximum-likelihood EBM training
   over all 32⁴ ≈ 1.05M concepts with no sampling and no contrastive approximation. V13 already trained this energy;
   V14 makes its target meaningful (§4) and gives it four horizons (§6). In the information-theoretic reading of a
   JEPA (arXiv 2609.31161: transition predictiveness minus λ times the entropy of the next representation, with LeWM
   as a constrained instance), this cross-entropy is the first term, and SIGReg on τ, with the codebook's usage
   perplexity as its discrete read, stands in for the second. That paper also separates losing information from
   mixing factors within coordinates. V14 can only promise the first: an isotropic Gaussian target is
   rotation-invariant, so PQ's segments are partitions of the target space, not identified factors (V14's own
   inference), and nothing in §6 or §9 reads a segment's code as a factor.
2. **The decoder is an energy-based model over sequences.** Blondel et al. (arXiv 2512.15605, ICML 2026) build a
   bijection between autoregressive models and EBMs in function space from the chain rule, tie it to a special case
   of the soft Bellman equation, and use it to explain how next-token predictors can plan ahead. So V14's generator
   is energy-based already; it is trained by NTP because LLM-JEPA's ablation shows the generative loss is what keeps a
   model generating.
3. **Planning is energy minimisation in concept space** (V-JEPA 2's action-conditioned form, H-JEPA's top-down form):
   E_plan(c_{g+1}, …, c_{g+H}) = Σ_k E_k(c_{g+k} | u) + D(c_{g+H}, goal). This is the interface for later search and
   control over concepts (V9's concept search, V12's metacontroller). It is defined here so the codes it would act on
   are stable (§3.4); nothing in V14 trains or tests it.
4. **Optional, scale-only: inference-time energy minimisation (Energy-Based Transformers, arXiv 2507.02092).** EBTs
   predict by gradient descent on a learned energy from a random start, trained through the unrolled optimisation with
   second-order gradients, a replay buffer, Langevin noise and randomised step sizes (removing the random step size
   turned their "thinking longer" gain from +7.19% to −1.47%). Reported to scale faster than Transformer++ and tested
   to 800M parameters. V14 records it as an arm on the soft-PQ forecast ĉ at scale (the one use that keeps ĉ continuous, since EBT descends
   on a continuous object; ĉ is a centroid over the head's alternatives, §6.4) and does not use it in the tiny test.

The continual-learning side of the energy view (Li, Du, van de Ven and Mordatch, CoLLAs 2022: training objectives
that do not push down the energies of classes absent from the current data interfere less with old knowledge) is in §9.

### 2.4 What V14 does not take from the JEPA literature, and why

- **Masked or paired views.** I-JEPA masks blocks; LLM-JEPA needs two natural views of the same content (text and
  code). A causal language model has one natural view pair for every position, past and future, and generation needs
  the causal order. V14's views are temporal: the prefix predicts the next four groups.
- **A continuous regression loss as the default.** I-JEPA and V-JEPA use L2 or L1, LLM-JEPA cosine, H-JEPA L2. In this
  repository a regressed forecast kept 3–14% of its target's variance and was scaled down further (C2), because
  regression predicts the conditional mean of a multimodal future. The same holds for text in general: a text JEPA
  trained by squared-error latent prediction learned centroids over alternative completions, then collapsed
  (validation effective rank 4.66 → 1.57) and transferred at chance, because masked text is not conditionally
  concentrated ("The JEPA Paradox in Language", arXiv 2607.23531); LLM-JEPA's cosine term works as an auxiliary on top
  of NTP. V14 keeps the cross-entropy over codes; a cosine term on the twin's continuous target view (LLM-JEPA's form)
  is an arm, expected to lose, and late in the queue (§11).
- **Dropping the generator.** A JEPA alone is not generative. NTP stays the primary loss and the decoder stays the
  output.
- **LeJEPA without a teacher.** LeJEPA needs no teacher, but V14 must keep codes meaningful over time for the
  correction bank and the continual stage (§9), which a teacher-free target does not provide. LeJEPA itself reports a
  small gain from a teacher on ViTs.

## 3. The encoder, and its EMA twin

### 3.1 The online encoder is V13-tiny-v2's, unchanged

| Part | V14-tiny (= V13-tiny-v2, `v13_tiny/DEVIATIONS.md` item 15) | Scale | v13_NOTES |
|---|---|---|---|
| Embedding | x = E[id], N(0, 1), tied with the head (logits / √d); L₁ = L₂ = G₁ = G₂ = x | untied | §3.1 |
| E1 | 2 × [S, S, S, F]; S: window 8, partial RoPE, reads L, writes L and G; F: full attention, NoPE with a learned temperature, reads L and G, writes G | one block, w_E = 64, F = MLA | §3.2–§3.4 |
| Engram | after E1's first S layer; 2- and 3-grams, 8 hash heads per order, 16 tables of 8,191 × 8; one gate per branch | ≈ 20% of the inactive budget | §3.5 |
| E2 | [M, M, M, F] looped R = 2 over a workspace W from s₀, per-loop embedding e_r; reads {L, G, W}, writes W; exit G += s·W | MoE FFNs, router sees the loop index | §3.6 |
| E3 | [S′, S′, S′, F], window 16, writes G only | | §3.7 |
| Exit | L^enc = GR read of (L₁, L₂); G^enc = GR read of (G₁, G₂) | | §3.8 |
| Width, heads | d = 128, 4 heads, SwiGLU 256 | d = 7,168 at 2.8T | |
| Reach | R_L = 51, R_D = 66 exactly (`tests_g0.py`) | R_L = 200, R_D = 327 | §3.8 |

The central invariant (v13_NOTES §2.2) is untouched: every coefficient that touches L is computed from L only, so
information moves from L to G and never back, and information from more than R_D tokens back reaches the next-token
logits only through the concept module.

**Why the encoder does not change.** The slow-target hypothesis (H3, §1.3) is about G^enc being a summary of the whole
prefix. That is G's job. A G that changed quickly would be a G that forgot the prefix, and the concept path would
lose the only stream with unbounded reach. The problem is in using G's *level* as the thing to predict, so the fix
belongs in the target (§4), not in G's architecture. G's *training* does change: the barrier term of §8 (SIGReg on
the online target view) sends gradient into G's layers through the online target view, by design. Its direct routes
into L are stopped; like NCP's in V13, it still reaches L's layers indirectly, through every G-writer that reads L (v13_NOTES §4.2: the split restricts information, not
gradients). The L-firewall arm is what removes that. E2's loop also stays at R = 2: the interim read found loop 2 inert at inference
(§1.1, C4), but changing R would confound V14 against V13-tiny, which is the comparison §11 is built on.

### 3.2 Why a target encoder at all

In V13 the NCP target is sg[m_{g+1}], computed by the same weights that are being trained. The stop-gradient stops the
loss from pulling the target toward the prediction within one step. It does not stop the target from *drifting*: every
update to the encoder, from NTP or from NCP through the input side, changes what the target is. V12 measured the
consequence on fixed inputs: 37–64% of codes reassigned between steps 4,000 and 8,000 (C3). The head then learns a
mapping whose labels keep moving, and anything keyed by codes (the correction bank's partitions, §9) loses its meaning.

The standard fix in latent-prediction methods is to compute targets with a slowly moving copy of the network, an
exponential moving average of the online weights. BYOL and I-JEPA take their targets from such a copy, with the
momentum annealed toward 1 over training; data2vec does the same. **Measured elsewhere** (vision, speech and text
representation learning), **untested here**, and untested with a discrete, EMA-k-means-quantised target.

What it does and does not do, from first principles:
- It **slows the target down in training time**: the target at step t is a weighted average over roughly the last
  1/(1−μ) steps of weights. The head sees a target that is nearly stationary over the time it needs to fit it.
- It **does not make the target informative**. A twin of an encoder whose G summary is slow produces a slow target
  too. H1–H3 are addressed elsewhere (§1.3's table, §4, §6, §8).
- It **does not prevent collapse**. In the linear analysis of coupled online and EMA dynamics, momentum changes how
  fast a mode escapes or collapses, never which modes escape (Drive vs Decay, arXiv 2610.02344, Prop. J.7). LeJEPA
  drops the teacher without collapse once SIGReg is on, and in C-JEPA's ablation (arXiv 2410.19560) the EMA-only
  I-JEPA did not collapse either; variance–covariance terms there mostly sped early training (linear probe 63.7 → 68.3
  at 100 epochs, 72.9 → 73.5 at 600). V14's protection is elsewhere: NTP for the encoder (§8.1), the difference target
  for H3 (§4.3).
- It **cannot leak into the forward pass**. The twin's output enters only the NCP loss, as a target, under no gradient.
  The invariant, the reach bounds and the gradient routes of every online module are unchanged.

### 3.3 What the twin copies

The twin is an EMA copy of **every module on the path from the token ids to the target view** (§4.3), and nothing
else. The target view's default needs both L^enc and G^enc, so the default twin is the whole encoder:

| Twin | Modules copied | V14-tiny parameters | Share of the model's forward FLOPs (accounting formula) | 2.8T class (stored / share of forward FLOPs) |
|---|---|---:|---:|---:|
| **Full (default)** | embedding, E1, Engram projections and gates, E2 with s₀ and e_r, E2's exit, E3, both exit reads; T_loc and T_glob | 3,890,106 for the encoder part (15.6 MB in fp32) plus 24,576 for T_loc and T_glob | 69% | 315B / 20% |
| Local (fallback view, §4.4) | embedding, E1, Engram projections and gates, the L exit read; T_loc | 1,992,808 for the encoder part (8.0 MB) plus 16,384 for T_loc | 25% | 105B / 4.5% |

Parameter counts are exact sums of V13-tiny-v2's PyTorch module counts (`v13_viz/data/tiny_v2.json`); the FLOP
shares come from `v13_design/accounting.py`'s tiers. The twin has no optimiser state and keeps no activations.

**The Engram table is shared, not copied.** At the 2.8T class the table holds 550B parameters in host memory and is
the only part whose copy would not fit the plan. The tiny model shares it too, so the tiny test measures the
configuration that would scale. The cost: the twin's local view moves with the online table, which trains at 5× the
base learning rate (v13_NOTES §3.5). The code-agreement log of §3.5 shows whether that makes the target drift.

**At scale the twin includes the MoE routers and their balancing biases**, copied by the same EMA. It needs no
balance of its own: it computes no gradients, and its routing only has to be a slowly moving version of the online
routing.

### 3.4 Momentum: set by lag, and a target era that closes

An EMA with momentum μ averages over about 1/(1−μ) steps. BYOL and I-JEPA start at 0.996 and anneal to 1, but they
train for hundreds of thousands of steps; a constant borrowed from them would mean a lag of 250 steps out of V14-tiny's
roughly 12,400. So μ is set from the lags it has to sit between:
- **Long enough** that the target stays put while the head learns it. The head's fitting time is hundreds of steps:
  NCP's loss fell from 13.9 to about 2–5 within the first 650–1,050 steps of the V13 run (§1.2's source).
- **Short enough** that the twin follows the online encoder while it is still learning what the concepts should
  carry.
- **Ordered with the other two EMAs that now stack on the target.** The twin feeds the normaliser (running mean and
  variance of the target, decay 0.99, a 100-step lag), which feeds the codebook (EMA k-means, decay 0.99, a 100-step
  lag). The codebook must be able to follow the normalised targets, so the twin must not move faster than they can:
  its lag should be at least theirs.

**Default schedule:** μ(t) = 1 − (1 − μ₀)·½(1 + cos(π t / t_freeze)) for t < t_freeze, with μ₀ = 0.99 (a 100-step
lag, equal to the codebook's) and t_freeze at 75% of training time; μ = 1 afterwards. The lag is 100 steps at the
start, about 150 at 30% of training, 200 at 37.5% (half-way to the freeze), 400 at 50%, 1,000 at 60% and 9,000 at 70%,
and it grows without bound as t approaches t_freeze: in effect the twin stops moving a little before the freeze. This is BYOL's cosine-to-1 form, with
its end moved to the freeze point. The twin is initialised as an exact copy of the online encoder at step 0.

**The target era.** At t_freeze the twin, the target normaliser and the codebook freeze *together* (V12's D3
proposal, a late freeze of the codebook, extended to everything that defines a code). The last quarter of training
then fits a fixed target, and the continual stage inherits exactly that target (§9). Freezing the codebook alone
while the twin kept moving would let codes drift under fixed words.

### 3.5 Cost on the M2, and what would show it wrong

- **Time.** One extra no-gradient forward of the twin per step. A training step costs about four forwards (forward,
  backward, and the recomputation that activation checkpointing adds), so the full twin adds about 0.69 / 4 ≈ 17% per
  step, and the local twin about 6%. These are estimates from FLOP shares; REPORT §4's eager profile agrees on the
  share (E1 + E2 + E3 took 1,055 of 1,546 ms of a forward). The 200-step smoke test replaces them with measurements
  (§11).
- **Memory.** 15.6 MB of weights; the forward runs under no gradient, so its activations are freed as they are used.
  It stays inside the 16 GB machine's margin (the V13 run peaked at 5.5 GB).
- **What would show it wrong.** Every 500 steps, on one fixed held batch per family: (a) the fraction of segment codes
  that differ between the twin's targets and the online encoder's targets. If it exceeds 0.5 before t_freeze, the lag
  is too long. (b) The reassignment rate of the twin's codes since the previous reading. V14's twin is meant to cut
  V12's 37–64% per 4,000 steps; a rate that is not lower than V13-tiny's own (measured afterwards on its snapshots)
  means the twin did not stabilise anything.

## 4. Concept pooling: the input view and the target view

**Job.** Turn each completed group of k = 4 tokens into two different things: an *input* that tells the concept module
everything it may use, and a *target* that names what the group did, for the concept module to predict before it
happens. V13 used one vector for both. V14 separates them, because the two jobs pull in opposite directions.

### 4.1 Why one vector cannot be both

| Requirement | Input p_g (what C reads) | Target (what C must predict) |
|---|---|---|
| Carry the document's state so far | yes: C needs it | no: a document-level summary barely changes between groups (H3), so predicting it is the copy-current solution |
| Carry the group's own tokens | yes: the identity tap (v13_NOTES §4.2) | yes, but of the *next* groups, as the bag of future tokens |
| Need the concept path to be predicted | irrelevant | yes: a target the local stream predicts on its own makes the concept path optional (§1.3, local-only) |
| Be spread evenly over the code space | irrelevant | yes: PQ's codes are used evenly only if each segment's distribution is spread and the segments are independent |
| Move slowly over training | irrelevant | yes: the codes must keep their meaning (§3.2) |
| Who sets its composition | learned β, trained by NTP and NCP through the input side | fixed by design, plus a projector trained only by the barrier terms, never by NCP |

The JEPA convention resolves this the same way: the target is the *target encoder's* representation of the future,
not the context encoder's representation of the input (I-JEPA, V-JEPA).

### 4.2 The input view: V13's, unchanged

For group g (tokens 4g … 4g+3), three pooled sources from the online encoder, each with its own per-channel softmax
pool initialised to the exact mean (v13_NOTES §4.1):

  m_g = β₁ RMSNorm(Pool_x(x)_g) + β₂ RMSNorm(Pool_L(sg[L^enc])_g) + β₃ RMSNorm(Pool_G(G^enc)_g),  p_g = W_P m_g,

with β = softmax(b) initialised to (0.107, 0.107, 0.787) and the release rule unchanged: group g is complete after
token 4g+3 and readable by decoder rows t ≥ 4g+3 (v13_NOTES §4.3). In V13 this m_g was also the NCP target; in V14 it
is only the input.

**β₁ is now an input-only lever.** The identity tap carries the current group's tokens into the concept input. With
the target decoupled, lowering it no longer changes the target (§1.3's table), so it can be tested for what the user
meant by "strengthen the concept module's inputs": whether a weaker copy route makes C compute more. **Arm:** β
initialised to (0.05, 0.107, 0.843), read by §6.2's probe test. Default: V13's initialisation, so V14 differs from
V13 only where it must.

### 4.3 The target view τ_g: computed by the EMA twin

For each group g, the twin (§3) computes a d-dimensional target in two typed halves, one per pair of PQ segments
(d = 128, S = 4 segments of 32: segments 1–2 local, 3–4 global):

  ℓ_g = [ MeanPool(x̄)_g ; MeanPool(L̄^enc)_g ]                     (the group's local features, 256 dimensions)
  a_g = T_loc ℓ_g                                                  (local half: what the group's tokens were)
  Δ^G_g = MeanPool(Ḡ^enc)_g − MeanPool(Ḡ^enc)_{g−1},   Δ^ℓ_g = ℓ_g − ℓ_{g−1}
  b_g = T_glob ( Δ^G_g − Δ^ℓ_g P )                                 (global half: how the group changed the summary,
                                                                     beyond what its local features explain linearly)
  τ_g = [ a_g ; b_g ],  normalised by running per-coordinate mean and variance (EMA 0.99), then quantised by the
  product codebook (§6).

P (256 × 128) is the ridge least-squares map from the local change to the global change, P = (A + ρI)⁻¹B, with
A = EMA(Δ^ℓᵀΔ^ℓ) and B = EMA(Δ^ℓᵀΔ^G) accumulated over every batch's valid groups on the twin's tensors (decay 0.99,
ρ = 10⁻³·tr(A)/256, all-reduced across ranks at scale). It is a statistic, not a parameter: no gradient passes through
it, the online view applies the same P, and it freezes at t_freeze with the normaliser and the codebook. A running map
rather than a per-batch one, because batches are single-family and a per-batch map would make the target depend on the
family. P is 0 for the first 200 steps, while the statistics are young and the encoder moves fastest, and is then
re-solved every 100 steps at every tier: its statistics lag by about 100 steps anyway (EMA 0.99), and at scale the
solve is expensive (§10.2). Setting a linear map from running second moments rather than training it has a precedent:
DirectPred (arXiv 2102.06810, §4) sets BYOL's linear predictor from an EMA of the feature correlation, with a floor on
small eigenvalues, re-solved every step or every few, and it matched or beat the gradient-trained predictor (77.9%
against 74.5% on STL-10).

Bars mark the twin's tensors. The parts, and why each is there:
- **Mean pools, not learned pools.** The target has no pooling weights to learn, so there is no learned route by which
  the pool could drift toward an easy target. (V13's input pools stay learned.)
- **The local half** is the bag of the group's tokens and their local context: the fast part of the old target, which
  carries token identity (Future Summary Prediction's handcrafted target, v13_NOTES §4.2). On its own it is
  predictable from local context, which is why it is only half.
- **The global half is a difference.** The level of Ḡ is the slow part (H3); its change from group g−1 to g is what
  group g did to the document's state: the new fact, the state update, the next step of the plan. Predicting the
  change at horizon k needs the concept path whenever the change depends on more than R_D tokens of context.
  MotionJEPA's difference targets are the precedent, and the Koopman result (§1.3) is the reason: level prediction
  converges to the within-document invariants, and an invariant's change is zero. (In a single-seed video study, arXiv
  2605.17165, a head predicting the teacher's temporal difference helped temporal reasoning, +5.0 points on SSv2, and
  cost static content, −2.1 on ImageNet-100: one reason V14 keeps a local half and keeps the level in the input view.)
  The first group of a document has no previous group, so its
  b is undefined and it is **masked**: it is never a target (a target is always some group g+k with k ≥ 1 after a
  group g of the same document), and it never enters the normaliser, the codebook update or the barrier terms.
- **Typed halves.** The two halves get separate codebook segments and are regularised separately (§8): SD-JEPA's split
  into subspaces whose anti-collapse terms act on disjoint coordinates.
- **The projectors T_loc (256 → 64) and T_glob (128 → 64)** are linear, 24,576 parameters together. Their online
  versions are trained *only* by the barrier term of §8 (SIGReg on each half),
  and the twin holds their EMA. NCP's gradient never reaches them, because NCP sees only the twin's output under no
  gradient. So nothing that is rewarded for predicting the target can also make the target easier to predict. (The
  JEPA literature's projector heads play the same role; here the projector exists to whiten the target for PQ.)
  Their training rule is the one place where NTP's anchor does not reach the target (§8.1). A linear map trained only
  to make its outputs Gaussian can do so by mixing many structured directions, which keeps their information, or by
  keeping the least structured directions and dropping the rest, which loses it (projection pursuit in reverse;
  LeJEPA's projector avoids the question because its prediction loss trains it too). Which one happens on real local
  features is measured before V14 runs: a 256 → 64 map fitted by document-centred SIGReg alone on V13-tiny's final
  checkpoint, against the top-64 principal subspace of the same features, on retained variance and a linear
  token-identity probe. If its probe is more than 5 points below the principal subspace's, the default becomes the
  **statistic-projector arm**: T_loc and T_glob set like P, by whitening on the top-64 principal directions of EMA
  covariances, frozen with P; SIGReg on a is then dropped (nothing is left for it to train) and SIGReg on b acts on G
  only. Otherwise that arm stays an arm. (Pending: `v14_design/PAPER_NOTES.md`, a7.)

**The online twin of the target view.** The barrier terms need a version of τ with gradient. The online encoder
computes the same two halves with its own weights: a_g from sg[x] and sg[L^enc] (so only T_loc trains from it; the
two direct copies of the token side into the barrier terms are stopped, as v13_NOTES §4.2 stops them for the concept
path), and b_g = T_glob(Δ^G_g − sg[Δ^ℓ_g]·P) from G^enc with gradient (so the barriers reach G's layers and T_glob, and,
indirectly through the G-writers' reads of L, L's layers too, exactly as NCP's gradient does in V13; the local change
is stopped, so no barrier reaches x or L directly through it). The online view is used by nothing
except the barrier terms.

**The global half's two failure modes, and how each is handled.** The change of a residual stream's summary can be
uninformative in two opposite ways.
- *A second bag of tokens.* G starts from the embedding and is a residual stream, so to first order its change from
  one group to the next contains the embedding mass of the new group's tokens minus the old group's, plus whatever the
  S layers wrote from local context. That residue is a small part of G's level (which is why the level was slow) but
  can be most of its change, and then b would be fast for a trivial reason and duplicate a. **Removed by
  construction** for its linear part: P subtracts the part of the change that the local features explain linearly.
  A nonlinear local residue can remain; **read by** the local-only probe (K3) per half.
- *Noise.* If G's summary barely moves beyond its local residue, b is a small remainder that the running normaliser
  amplifies to unit variance. **Read by** the residue map's R² per family (the share of Δ^G's variance that the local
  change explains; R² near 1 means nothing non-local is left) and by the head's per-half accuracy against chance
  (1/32 per segment). V13's K4 read of χ for Pool G's change is the closest existing proxy for this R².

**Why the barrier is not an inverse-dynamics term.** H-JEPA's and AC-MTM's inverse-dynamics terms make a latent
transition identify the *action* that caused it. In robotics the action is outside the observation, so the term forces
the latent to carry the action's *effect* on the world. In a language model the "action" between two concept states
is the group's tokens, which are part of the observation, so the same term forces the latent to carry the tokens
themselves: the local half's content. With P removing the linear token residue, the term would push G to encode the
tokens nonlinearly instead, which is still local. The robotics evidence says where the repair comes from: in LeWM+I
(arXiv 2610.03137) the inverse-dynamics gradient into the encoder carried about 85% of the gain, the same term on the
predictor alone did nothing, and shuffled actions did nothing. In a language model that encoder-side signal about the
group's tokens is what NTP already provides. So V14 relies on the difference target, with centred SIGReg as its
backstop (§8.3), and inverse dynamics is a
named arm (§8.4). It is placed, in that arm, on b and never on u_g or a, which contain the identity tap.

### 4.4 Fallbacks fixed by the checks

| Check outcome | Target view |
|---|---|
| Default | τ = [a ; b], as above |
| K4: the within-document share of Pool G's change is below 0.05 (G carries no group-level signal) | τ = a only (four local segments), with the local twin of §3.3 (25% of the forward FLOPs instead of 69%) |
| K3: the local-only probe is within 2 points of the u_g probe | the local-only fallback is ruled out; the global half stays even if weak |
| *In V14's own run:* the global half's local-only probe is within 2 points of its u_g probe | b duplicates a through a nonlinear local residue: the run is reported as having no non-local target; neither arm of §8.4 is expected to help (both reward local content) |
| *In V14's own run:* the residue map's R² is above 0.95 in at least 3 families after the λ ramp | b is noise: the local-view fallback above applies from the next run |
| *In V14's own run:* the target-side action sensitivity of b (§6.3) is below 0.05 and below a tenth of a's own in at least 3 families after the λ ramp | b does not follow what the group did, given the prefix: the same response as the R² row |
| *In V14's own run:* the head's accuracy on the global half is within 2 points of chance | b is noise: its two segments are reported, not interpreted, and from the next run their weight in the NCP loss halves |

### 4.5 Sizes at scale

At the 2.8T class (d = 7,168; S = 56 segments of 128), each half has 28 segments: T_loc maps 14,336 → 3,584 and T_glob
7,168 → 3,584, about 77M parameters together, plus their EMA copies. Mean pooling costs nothing. The release rule and
k = 4 are unchanged at every tier.

## 5. The concept module: a latent world model

**Job.** Keep a state of the document at concept rate, update it as each group of tokens arrives, and predict, before
the tokens exist, what the next groups will do. The user's framing is the right one: the concept module is a world
model over concepts, which is why pooling and next-concept prediction belong together, and why HiLP's long-horizon
objective is the natural one to train it with.

### 5.1 The world-model reading

| World-model part | In V14 | Trained by |
|---|---|---|
| State | u_g, the concept state after group g | everything below, and NTP through the decoder's concept read |
| Action | the four tokens of group g+1: what is written next | (data) |
| Forward dynamics, action-conditioned | u_{g+1} = C(u_{≤g}, p_{g+1}): C reads group g+1's input, which contains its tokens (identity tap), their local context and the global stream. Teacher-forced, as every causal sequence model is | NTP (through the decoder) and NCP (through the input side) |
| Prior over futures, action-free | p(code of τ_{g+k} \| u_g) for k = 1 … 4, before group g+1's tokens are known: HiLP's upper dynamics, with a cross-entropy over codes instead of HiLP's SmoothL1 (C2) | NCP (§6) |
| Inverse dynamics | identify the action from the transition: from the online global change b_{g+1}, which group's tokens produced it | an arm only (§8.4): the action is part of the observation, so as a barrier it would reward encoding the tokens themselves |
| Policy | the decoder: it writes the next group's tokens, conditioned on the state (the concept read) and on local context | NTP |
| Planning energy | E_plan over code sequences, §2.3 | not trained (an interface) |

The forward dynamics and the policy already existed in V13. The prior existed with one horizon and a target that
turned out to be nearly constant within a document (§1.2). Inverse dynamics is the one part a language world model
gets almost for free and cannot use as a barrier, because its action is observed (§4.3); it is kept as an arm, on the
target view and never on u_g.

### 5.2 Structure: V13-tiny-v2's concept module, unchanged

n_C = 4 layers, each [gated causal self-attention over concepts ≤ g, RoPE on the group index, QK RMSNorm with a
learned temperature] → [gated cross-attention from concept g to sg[L^enc] at tokens 4g−12 … 4g+3] → [SwiGLU FFN],
with Delta-AttnRes over every sublayer and p_g as source 0; no loop (C4) and no scratchpad; 1,052,064 parameters
(exact). At the 2.8T class: 16 layers, MoE FFNs, Delta-AttnRes in blocks of 2, and one 512-dimensional shared
key-value entry per concept in the self-attention (v13_NOTES §5.1, §11.3).

Why it does not change:
- The V13 run's evidence points at the objective, not at C's capacity. A probe on C's *input* already predicted the
  old target at 76–91% (§1.2); C had nothing left to compute for it. A target that needs computing is the test of C's
  capacity, so V14 changes the target first.
- The decoder reads u_g. Leaving C alone keeps the decoder's input, and therefore §7, unchanged, and keeps V14
  comparable with V13-tiny.
- Horizon conditioning lives in the head (§6.2), not in C. C produces one state per group, as before.

### 5.3 Horizons are predicted directly, not rolled out

The prior predicts horizons 1 … 4 directly from u_g; it never feeds its own predictions back as inputs. Rolling a
latent model forward compounds its errors, which is the problem HiLP's higher-level latent was introduced to reduce
(arXiv 2608.05806) and the one H-JEPA's hierarchy addresses.

**Horizon is not reach.** R_L = 51 and R_D = 66 bound how far *back* the local stream and the decoder see; a horizon is
how far *ahead* the target lies. They are different axes. Predicting further ahead does not by itself make the concept
path necessary: the local-only probe from L^enc at row 4g+3 keeps its 51-token look-back at every horizon, and it can
predict τ_{g+k} exactly when τ_{g+k} does not depend on context further back than that. Necessity comes from
*look-back* dependence, and K3, run per horizon and per half, is what measures it. What longer horizons do give is
pressure: copy-current scores worse the further ahead the target is (unless the target is slow, which is why the target
view is built to be fast), and a state that predicts four groups ahead has to carry more than one that predicts one.
There is also a trap: targets far ahead drift toward what stays the same, the document-level content, which is the
copy-current solution again. The per-horizon persistence baseline (§6.3) watches for it.

### 5.4 The long horizon: a successor-code head (an arm, fully specified)

To predict far ahead without more heads or rollouts, a second head predicts the discounted distribution over all
future codes, trained by temporal-difference bootstrapping (TD-JEPA, arXiv 2510.00739: multi-step
latent prediction through successor features). Per segment s:

  y_{g,s} = (1 − γ) · onehot(code_s(τ_{g+1})) + γ · sg[ q̄_s(· | ū_{g+1}) ],     L_SR = Σ_s CE( q_s(· | u_g), y_{g,s} ),

where q is the successor head on u_g, and q̄ and ū come from a **target network**: an EMA copy of C and the successor
head on the twin's momentum schedule (§3.4), run on sg[p_{≤g+1}] from the online input pool (the twin has no input
pools). The twin alone does not cover C, so this is about 1.1M more copied parameters in the tiny model. At the last group of a document the bootstrap term is dropped (y = the one-hot). The
fixed point is q(· | u_g) = (1 − γ) Σ_{k≥1} γ^{k−1} p(code of τ_{g+k}): where the document is going, discounted.

- **Settings.** γ = 0.9, an effective horizon of about 10 groups (40 tokens), and γ = 0.97, about 33 groups (130
  tokens).
- **Its own easy solution.** A discounted future histogram is slow by nature (§5.3's trap), so it can be predicted by
  the discounted histogram of the *past*. That is the baseline its cross-entropy must beat, logged beside it; beating
  it is the head's whole justification. Whether what it predicts needs the concept path is K3's question, as for every
  horizon.
- **Why it is an arm in the tiny test.** It needs a second target network; its mixed-horizon target cannot be read
  with K2 and K3 the way a single horizon can; TD-JEPA's evidence is from reinforcement learning, not language; and a
  single 12k-step seed is a weak test of a bootstrapped objective.
- **Why it matters.** It is the long-horizon objective the user is after, at a fixed cost per group whatever the
  horizon. If the tiny arm beats its past-histogram baseline at γ = 0.97, it is the default at the first scale tier.

### 5.5 A third level (scale arm)

H-JEPA found that a third level helped long-horizon planning (Visual AntMaze 18% → 73% with three levels, 39% with
two). V14's analogue pools four concepts into a super-concept (16 tokens), gives it its own target view (the same two
typed halves at the coarser rate), its own SIGReg, and a small causal predictor at a sixteenth of the token rate.
The decoder does not read it, so the invariant and §7 are unchanged; the concept module's prior may read it as
context. It is a scale-tier arm, untested in language.

### 5.6 The probe test, per half

v13_NOTES §6.2's acceptance test (an MLP probe on u_g beats the same probe on p_g by at least 5 points on the next
group's codes) is kept and run **per half** of the target view. The local half should pass easily: by the time u_g
is computed, p_g has seen the current group, and the bag of the next tokens is largely predictable from local context.
The global half is the real read. Its failure would mean C does not predict how the document's state changes, which
is the world-model claim.

## 6. The concept head: codes, energies and horizons

### 6.1 The codebook

- **Product quantisation of τ.** S = 4 segments of 32 dimensions, N = 32 words each (V13's sizes): segments 1–2
  quantise the local half a, segments 3–4 the global half b. The normaliser (running mean and variance, EMA 0.99) and
  the EMA k-means words (decay 0.99) are fitted to the twin's targets, never by gradient, and freeze with the twin at
  t_freeze (§3.4).
- **Why SIGReg helps the codebook as well as the target.** PQ treats the segments as independent and k-means spreads
  its words to match each segment's distribution. Under an isotropic Gaussian, the target SIGReg pushes toward (§8),
  segments are independent and every segment's distribution is the same round cloud, so the words are used about
  evenly. Across the two halves, independence is helped by construction: P removes from b the part that the local
  features a is built from explain linearly (§4.3); their cross-correlation is logged (§6.3). The dead-code rule then
  becomes a safety net rather than the main defence.
- **Stricter dead-code rule.** A word whose EMA usage falls below 10% of uniform (V13: 2%) is replaced. V12's fix is
  kept (the word's count is compared with the EMA of rows per step, never with one batch's count, so minority
  families' words are not reset by every batch of another family). The replacement is the target segment in the
  current batch with the *largest quantisation error* (V13: a random one), so a reset word lands where the
  codebook fits worst; when several words die in one step they take *distinct* targets, in order of error. The rule
  stops at t_freeze with the words and the normaliser, so no code changes under the frozen head.
- **Codes at inference.** The twin is a training-time object and is discarded after training: running it at
  inference would cost 69% of a forward. Wherever a code is needed after training (the correction bank's partition key
  and any code gate, §9), it is computed from the *online* target view (with the frozen residue map P), quantised with the frozen
  normaliser and codebook. The local half's inputs (the embedding, E1, Engram) are frozen in the continual stage, so a key taken from
  the local half is stable across every phase. The guard is the twin-versus-online disagreement log of §3.5, on **both
  halves**: at the end of training at least 80% of segment codes must agree in each half. The global half is the one
  at risk, because its inputs (E2, E3 and the G-writers) keep training for the last quarter after the twin froze.
  Below the guard in either half, the normaliser, P and the codebook are refitted on the final online model's targets
  over base-stream rows (warm-started from the frozen words), and then the horizon head alone is re-fitted on the same
  rows for 500 steps with everything else frozen, so the head and the codes agree again. That refit defines the era
  of everything after training, including the continual stage (§9).
- **N = 64** words per segment is an arm.
- **A fixed quantiser is a control arm** (finite scalar quantisation, as in arXiv 2512.07168): each normalised
  32-dimension segment is coded by the signs of its projections on 5 fixed orthonormal directions (2⁵ = 32 codes;
  the directions drawn once from a seeded generator). It has no words to learn, so there is no k-means drift and no
  dead code (the dead-code rule and the era refit are inert under it), and the segments are independent by
  construction. Usage is approximately uniform, exactly so only if the normalised, *uncentred* target is an isotropic
  Gaussian. It keeps less: about 10% of a segment's variance (2/π on each of 5 of 32 directions) against at most 19.5%
  for any 32-word quantiser of an isotropic Gaussian (the rate-distortion bound). Its use is as a control: under it,
  V5's code reassignment measures the twin's drift alone, which separates the twin's drift from the codebook's, as no
  other arm does.
- **One codebook for every family**, as NCP at scale requires; usage perplexity is logged **per family and per
  segment** (V13 logged one global number).
- **At scale:** S = 56 segments of 128 (28 local, 28 global), N = 128; code counts, code sums and normaliser
  statistics are all-reduced across data-parallel ranks before every EMA update (v13_NOTES §9.4).

### 6.2 One head, four horizons, one energy each

The head is V13's NCP head with a learned horizon embedding added to its input:

  ℓ_k = MLP( RMSNorm(u_g) + e_k ),  k = 1 … 4,  MLP: d → 2d (GELU) → S × N logits,

so one set of weights serves every horizon, the way I-JEPA's predictor serves every target position through its mask
tokens. For horizon k the energy of a code tuple is E_k(c | u_g) = −Σ_s ℓ_{k,s}(c_s), with an exact partition function
(§2.3). The loss is

  L_NCP = Σ_k w_k · mean over valid (g, g+k) of Σ_s CE( ℓ_{k,s}, code_s(τ_{g+k}) ),   w = (0.4, 0.3, 0.2, 0.1),

where a pair is valid when g and g+k are in the same document (the first group of a document is never a target,
§4.3). The weights sum to 1, so λ means what it meant in V13 (§8). Nearer horizons get more weight because their
targets are more predictable and bear directly on the group the decoder is writing; the farther ones supply the
pressure against copy-current. This gives horizon 1 only 0.4 λ, where V13 gave it λ. Two arms bracket the choice:
equal weights (0.25 each), and "V13 plus", w = (1, 0.3, 0.2, 0.1), which keeps V13's full horizon-1 pressure and adds
the far horizons on top (a total of 1.6 λ). "V13 plus" raises NCP's total pressure, so it is read together with the H1
rule of §1.4, which raises λ_max to 0.2. At scale the segment sum is rescaled by 4/S, as in V13.

**The head could ignore the horizon.** Adding e_k before the first layer gives each horizon only a shift of the
hidden layer's input, and a slow target would make "the same prediction at every k" the easy optimum. §6.3 logs the
fraction of groups whose argmax tuple is identical for k = 1 … 4. If it stays above 0.9 after the λ ramp, the
pre-registered response is a per-horizon FiLM on the hidden layer (a learned scale and shift per k, 4 × 2 × 256
parameters).

**Two arms on the head, pre-registered for later tiers.** (i) *Branch head* (Branch-JEPA, arXiv 2607.05238; a
discrete latent variable in the sense of latent-variable EBMs, arXiv 2306.02572). The head is distributional per
segment but factorised across segments, so it cannot represent correlated alternatives: if the next group is either
(A₁, B₁) or (A₂, B₂), it also gives mass to (A₁, B₂) and (A₂, B₁). The arm adds K = 4 small residual MLPs
u_g → u_g^(z) and a router π(u_g), runs the shared horizon head on each, and trains the exact mixture likelihood
−log Σ_z π_z Π_s p_z(c_s), which keeps the partition function exact (§2.3). Branch-JEPA found multiplicity placed
before the shared decoder beat the same multiplicity placed after it (effective modes 5.80 against 5.25, with a
better energy score and Brier score in all five seeds). The arm runs only if §6.3's cross-pair read fires;
N_eff = exp H(π) per family is its own read. (ii) *Soft code targets* (S-JEPA, arXiv 2606.19398; SC-JEPA, arXiv
2602.04643): cross-entropy against the codebook's posterior over words, softmax(−‖τ_s − e_{s,n}‖² / T_s), instead of
the one-hot code, so a target near a boundary between two words is not forced to one of them, and a small drift of the
target moves its label only a little (C3). It has a knob: T_s is set so that the mean target entropy is 1 bit on the
held batch (S-JEPA's two-way-tie regime), re-set every 500 steps; the temperature implied by reading the codebook as
an isotropic Gaussian mixture is computed on V13-tiny's final checkpoint and reported beside it. Neither paper has a
controlled soft-versus-hard comparison, so this is the most weakly evidenced arm.

Cost: 66,048 parameters (V13's head) plus 4 × 128 for the horizon embeddings; four head evaluations per group, which
is negligible beside the encoder. At the 2.8T class one shared head is about 0.2B parameters; four separate heads would
be four times that.

### 6.3 What the head's reads are, during training

Logged on every logged batch, per horizon and per half (cheap: everything is already computed):
- the head's accuracy, against the **copy-current baseline** (the code of τ_g predicted for τ_{g+k}) and against
  chance (1/32);
- per-family, per-segment code-usage perplexity;
- the fraction of groups whose argmax tuple is the same at all four horizons (§6.2).

Every 500 steps on one held batch per family:
- K3's local-only probe per half (an MLP from L^enc at row 4g+3), beside the probe from u_g;
- χ for each half of the online target view (H-JEPA's within-document variance share);
- the twin-versus-online code disagreement and the twin's code reassignment rate (§3.5);
- χ for each half of the *twin's normalised target*, the object the codebook quantises (§8.3's trigger);
- the residue map's R² per family, and the mean squared cross-correlation between the halves a and b;
- the **target-side action sensitivity** of each half (LeWM+I's probe, arXiv 2610.03137, moved to the target): for 64
  fixed prefixes per family, K = 8 alternative next groups sampled from the model at temperature 1, each run through
  the twin; spread/motion is the mean variance of the pre-normaliser b_{g+1} (and a_{g+1}) across the alternatives over
  its mean squared size, both on the same tensor. Below 0.05 (LeWM's gate, borrowed from pixels: necessary, not
  sufficient) and below a tenth of the local half's own value, b does not follow what the group did (§4.4). A control
  with K identical alternatives must give 0 to numerical precision, or the read is void. Its cost is measured in the
  smoke test; it moves to every 1,000 steps if it exceeds 2% of training time;
- the effective rank of each half, online and twin (the exponential of the entropy of its normalised singular values;
  Drive vs Decay's count of directions that escaped collapse, necessary and not sufficient);
- T_glob's gain, its Frobenius norm over its value at step 0 (context for the noise reads, §8.3);
- the **cross-pair rate**: the fraction of the head's horizon-1 argmax tuples (and of tuples sampled at temperature 1)
  whose exact four-code tuple has never occurred as a target in that family's training rows so far, beside the same
  fraction for the realised targets (a 2²⁰-bit table per family covers the whole tuple space, 32⁴ = 2²⁰). A
  factorised head that combines segments from different alternatives shows a rate above the targets' own; above twice
  it in at least 3 families after the λ ramp, the branch-head arm (§6.2) is warranted.

### 6.4 The soft-PQ forecast

V13's soft-PQ forecast ĉ_{g+1} is the codebook's expectation under the horizon-1 distribution, de-normalised
(v13_NOTES §6.3). That expectation is a conditional mean over the head's alternatives: when two continuations are
likely, ĉ lies between their codewords and need not be near either, the failure the probabilistic-JEPA papers
describe for point predictors (arXiv 2607.23531, 2607.05238, 2601.14354). **V14 changes V13's ĉ-injection arm**: the
decoder is given the codewords of the head's argmax tuple, in training and at inference (a sampled tuple is used only
where diversity is wanted, in speculative drafting). The argmax passes no gradient, so the arm no longer opens a route
from NTP into the horizon head (§8.5). ĉ itself stays as a read (its distance to the realised target's codewords) and
as the input of the inference-time energy-minimisation arm (§2.3, item 4), which needs a continuous starting point;
that arm inherits the centroid caveat. The decoder reads u_g by default.

## 7. The decoder: the autoregressive generator

Unchanged from V13-tiny-v2 (v13_NOTES §7–§8; `v13_tiny/DEVIATIONS.md` items 15, 19–21): six layers of [local attention
whose queries come from the decoder state and whose keys and values are L^enc in a 16-token window] → [cross-attention
to the released concepts u_j, j ≤ ⌊(t+1)/4⌋ − 1, with a null entry and a sink logit] → [SwiGLU FFN], Delta-AttnRes with
L^enc as source 0, a tied output head scaled by 1/√d; 1,579,656 parameters; R_D = 66 exactly. The decoder option
(D-confined, D-mixed or D-own-SWA, v13_NOTES §8.2) remains the user's decision; V14-tiny keeps D-confined, as V13-tiny ran.

In §5.1's terms the decoder is the **policy**: it writes the next tokens from the world model's state (the concept read)
and local context. In §2.3's terms it is the **energy-based model over sequences**, trained by NTP. It reads u_g only:
not the horizon predictions and not the target view. MTP at scale (depth 1, weight 0.3 falling to 0.1) and the absence
of any correction-bank read in the decoder are V13's; the ĉ-injection arm is V13's with one change, the argmax
tuple's codewords in place of the expectation (§6.4).

**Generation.** Autoregressive sampling, token by token. The encoder runs incrementally (sliding-window caches for S,
key-value caches for F, a recurrent state per loop for each GDN-2 layer). Every fourth token completes a group: the
input pool and one concept-module step run, and the new u_g becomes readable from the next decoder row. The twin and
the barrier term are training objects, and the horizon head is not needed to generate (it may drive a speculative
drafter, an arm, as NCP-Arch's concepts drove one). The target view runs at inference only where a code is needed (the
correction bank's partition key and any code gate, §6.1 and §9): mean pools, the frozen residue map and two small
projections on the online encoder's tensors, with no twin.

## 8. Objectives

### 8.1 The total objective

  L = L_NTP + λ(t)·L_NCP + ν_S(t)·L_SIG   [ + ν_S(t)·L_SIG,b,uncentred if triggered (§8.3) ]
      [ arms: VICReg's two terms instead of SIGReg; contrastive inverse dynamics (§8.4); the cosine latent term;
        L_SR (§5.4); ĉ injection; L_VQ; NextLat ]

| Term | Definition | Weight |
|---|---|---|
| L_NTP | V13-tiny's next-token loss and scoring (text every token; reasoning answer tokens, normalised per example) | 1 |
| L_NCP | §6.2: horizons 1–4, w = (0.4, 0.3, 0.2, 0.1), cross-entropy over the twin's target codes | λ(t): 0 → 0.1 over the first 10% of training time, then the gradient-share cap (§8.2) |
| L_SIG | SIGReg on the online target view, **document-centred**, each half separately (§8.3) | ν_S = 0.05, ramped with λ |

The default has three terms. The fourth job a JEPA needs, keeping the target off the trivial solutions, is done by
construction rather than by a loss: the twin (no gradient into the target), the typed halves, the difference target
and the residue map P (§4.3). For the encoder itself the anchor is NTP: an encoder that must predict the next token
cannot give one representation to two prefixes with different continuations, and with consistent latent dynamics the
same holds for prefixes with different futures (arXiv 2509.12249, Thm 1, with NTP as the auxiliary task; in their
counting world the transition loss alone collapsed completely and the auxiliary alone gave only coarse classes). The
twin copies that encoder, so the anchor reaches every module of the target except the two projectors, which only
SIGReg trains: their training rule is the one place where nothing guarantees the target's content (§4.3, and the
statistic-projector arm). ν_S = 0.05 is LeJEPA's default weight, taken here against NTP, the loss that mostly shapes G, the one module SIGReg
shares with the other objectives (T_loc and T_glob are trained by SIGReg alone, where its weight only sets their step
size). Against NCP alone it is ten times LeJEPA's ratio. It ramps with λ, so the barrier starts when NCP starts. The
balance it actually strikes on G is a read: on the held batch, ν_S‖∇_G L_SIG‖ beside ‖∇_G λL_NCP‖ and ‖∇_G L_NTP‖.
Which arms the tiny test runs, and in what order, is fixed in §11; "VICReg instead of SIGReg" (the user's stated
form of the request) and "inverse dynamics added" are the first candidates. LeJEPA's own analysis predicts that the VICReg arm loses (its §5.2: a test of the mean and standard deviation alone admits shortcut solutions), and SIGReg was the less weight-sensitive of the two in EB-JEPA's ablation (arXiv 2602.03604); the arm is run because it is the request as stated. H-JEPA found that its SIGReg and inverse-dynamics weights must be scaled together (its K.2), so
if the inverse-dynamics arm is carried to the first scale tier it goes with a 3 × 3 sweep of the two weights over
{0.02, 0.05, 0.1}.

### 8.2 λ and the gradient-share cap (V13's untested rule, now the default)

λ ramps from 0 to λ_max = 0.1 over the first 10% of training time, as in V13. Every 500 steps, on one held batch, the
ratio ‖∇_C (λ L_NCP)‖ / ‖∇_C L_NTP‖ over the concept module's parameters is measured (two backward passes, in a
separate uncompiled step, because the compiled training step has no hooks: `v13_tiny/DEVIATIONS.md` item 23). After two
consecutive readings above 1, λ halves, with a floor of λ_max/8. C3 is the reason: NCP's price on text was +0.28 nats
in V12. λ_max ∈ {0.05, 0.2} are arms; 0.2 becomes the default if K1 confirms H1 (§1.4).

### 8.3 SIGReg, with time on the sample axis

For each half h ∈ {a, b} of the online target view, over the valid groups of the batch (first groups of documents
masked; documents with at least four valid groups, because a two-group document contributes an exact ± pair, not
two samples):

  z̃_g = z_g − mean of z over the groups of g's document,
  L_SIG,h = (1/M) Σ_{m=1}^{M} EP( { ω_mᵀ z̃_g } ),   EP(s) = ∫ | φ̂_s(t) − e^{−t²/2} |² e^{−t²/σ²} dt,

where z_g is the projector's output (a_g or b_g of the online view) *before* the running normaliser, which belongs to
the codebook path, and φ̂_s is the empirical characteristic function of the projected samples. There are M = 256
random unit directions ω_m, resampled every step from a generator seeded by the global step (so every
data-parallel rank draws the same ones), the integral by the trapezoidal rule on 17 knots over [−5, 5], σ = 1, and the
empirical characteristic function φ̂ all-reduced across ranks at scale (LeJEPA's implementation). L_SIG = L_SIG,a +
L_SIG,b. Cost: linear in the number of groups.

- **Why centre on the document.** Centring removes each document's mean, so only within-document variation is left
  for SIGReg to shape. It extends H-JEPA's Proposition 2: an *uncentred* test with time on the sample axis adds
  λ(T−1)κ at exact collapse, T times its null value, "a real but small barrier" in their words, and nothing when time
  is on the group axis. Centring is V14's own step. The sample size is kept large by pooling all documents' centred
  latents into one test rather than one test per document; a per-document test on a seven-group math problem would be
  dominated by the O(1/N) bias.
- **What it does against H3, and what it does not.** At exact slow-feature collapse every centred latent is 0, and the
  Epps–Pulley statistic of a point mass against N(0, 1) is a fixed positive amount, whatever the encoder does
  elsewhere: 0.131 per direction with these knots and σ, against 0.0003 for exactly Gaussian samples at V14-tiny's
  2,000 groups per batch (the O(1/N) floor; both computed numerically), about 400 times the floor where the uncentred
  test gives T times. Three things limit what that buys. (i) The amount is measured against Gaussian samples, not
  against a healthy encoder: a healthy but non-Gaussian within-document distribution also scores above the floor, so
  collapse need not score above health (H-JEPA's own caveat). (ii) The term acts on the projector's output, and T_glob
  is a free linear map: near collapse, when G's residual change is small, T_glob can meet the unit variance by growing
  its scale, so G does not have to pay the variance part; what reaches G is pressure on the *shape* of its
  within-document distribution. That is consistent with §3.1, which does not want G to change faster. (iii) In
  H-JEPA's ladder of SIGReg weights with time on the sample axis and no other anti-collapse term (their Table 12,
  uncentred, no target encoder), every run collapsed, later at larger weights but to the same endpoint (χ between
  0.008 and 0.016); time repulsion kept the predictor off the identity without selecting what moved (their App. L.6).
  So V14's defence against H3 is the difference target (§4.3); this term is its backstop at exact collapse and the
  whitening the codebook needs; and whether b is informative is decided by the reads: the residue map's R², the
  per-half accuracy against chance, K3 per half and the target-side action sensitivity (§6.3). V14 differs
  from H-JEPA's setting in the ways that matter, in this order. Its online encoder is anchored by NTP (§8.1), so the
  encoder cannot collapse the way theirs did; and its target is a change, so the slow solution of level prediction is
  not available to it (§4.3). Its target encoder also takes no gradient, but that alone would not carry the argument:
  in H-JEPA's own grid, detaching the targets did not change the outcome (their App. K.1). And NCP's input-side
  gradient still rewards, through the shared weights, a target that is predictable from u_g; under the Δ target the
  degenerate form of that is a constant change per document, which is what centred SIGReg penalises at exact collapse
  and what R², the per-half accuracy and the action-sensitivity read detect. So V4 is load-bearing. Centred SIGReg behind a difference target is still **empirically untested** as
  the only regulariser against slow collapse. If the live reads fail, the inverse-dynamics arm is the pre-registered
  fallback. T_glob's gain is logged beside them as context (§6.3): it is the inverse of G's absolute within-document
  movement, which is not by itself a failure, so no rule hangs on it.
- **Why it is the user's variance-covariance request, generalised.** A distribution whose every one-dimensional
  projection is N(0, 1) is N(0, I) (Cramér–Wold), which has unit variance in every direction and zero covariance. So
  SIGReg enforces VICReg's two terms and every higher moment. VICReg's two terms (variance hinge at 1 and the
  off-diagonal covariance penalty, on the same centred latents) are the named arm, so the request is tested as stated.
  In the variational reading, SIGReg is the aggregate-posterior half of a VAE's KL to N(0, I), without its information
  term (Var-JEPA, arXiv 2603.20111, §3.7): it shapes the target's distribution without limiting how much it carries.
- **Centring uses future groups.** That is acceptable: L_SIG is a training-only loss on the target side and never feeds
  the forward pass.
- **What centring does not bound.** Centred SIGReg shapes deviations from each document's mean but says nothing about
  the document means. If the mean change per document varies strongly across documents, the *uncentred* target that
  the codebook quantises can still be dominated by document identity: χ small for b, codes naming documents, H3
  intact. Forcing the uncentred marginal to N(0, I) as well would push each document's net displacement of the summary
  toward zero, which is not obviously harmless for a stream that must summarise, so it is not done blind. Instead, χ
  of each half of the twin's normalised target is logged on the held batch (§6.3); if χ_b < 0.5 after the λ ramp, the
  uncentred test on b is added. K5's confirmation of H2b is the second trigger.
- **The local half needs within-document rank.** SIGReg on a trains only T_loc, which becomes a learned whitening of
  frozen local features. That needs the within-document covariance of the local features to have rank at least 64,
  which short documents may not give; SIGReg's own value on a is the read.

### 8.4 Arm: contrastive inverse dynamics

Why it is an arm and not the default: in a language model the action between two concept states is the group's tokens,
which are observed, so the term rewards encoding the tokens themselves (§4.3). It is run because the robotics evidence
for it is strong and because it is the pre-registered fallback if centred SIGReg's live χ reads fail. If it wins, the
pressure helps despite pulling b toward local content; K3 per half shows how much of b became local. If it loses, the
language case differs from the robotics case as argued. The spec, for each valid pair (g, g+1):

  t_{g+1} = h_I( b^on_{g+1} ),   α_{g+1} = h_A( sg[ MeanPool(x)_{g+1} ] ),
  L_IDM = − mean_i log [ exp(cos(t_i, α_i)/τ_I) / Σ_j exp(cos(t_i, α_j)/τ_I) ],  τ_I = 0.1,

where h_I is a two-layer MLP (128 → 128 → 64), h_A a linear map (128 → 64), and j runs over the batch's valid groups
(groups whose token multisets are identical to group i's are masked as negatives). Both heads exist only in training.
b^on_{g+1} is already a transition (the change of the global summary from g to g+1), so no second state is needed. Its
gradient reaches T_glob and the G side; the action side's input is stopped, so the task cannot be solved by moving
the embedding. Anchors: AC-MTM's Action-NCE (80.0% against a batch-marginal SIGReg's 58.0% on its hardest task) and
H-JEPA's inverse dynamics, the one Ω(1) barrier in its analysis, and LeWM+I (arXiv 2610.03137), where the same kind of
term repaired a world model whose latent ignored the action. Its in-batch accuracy is logged per family; near chance
means it is not biting. Weight ν_I = 0.05, ramped with λ. **Control:** if the arm moves the reads, it is rerun with α
permuted across the batch (LeWM+I's shuffled-action null): a gain that survives the permutation comes from the extra
gradient acting as a generic regulariser, not from the term identifying what the group did.

### 8.5 Gradient routes

v13_NOTES §9.3's table, extended. "Indirect" means through the G-writers' reads of L and the S layers' shared outputs.

| Objective → | E1 L-writes, Engram L gates | G side (E1/E3 G-writes, E2, F) | input pools, W_P | T_loc | T_glob | concept module | horizon head | IDM heads (arm) | codebook, normaliser, P | decoder | embedding | twin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NTP | ✓ (three routes) | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| NCP | ✓ indirect | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| SIGReg, local half | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| SIGReg, global half | ✓ indirect | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (G starts at x) | ✗ |
| Inverse dynamics (arm) | ✓ indirect | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ (G side only; the action side is stopped) | ✗ |
| EMA update (no gradient) | — | — | — | — | — | — | — | — | — | — | — | copies the online encoder and T |
| EMA k-means and the residue map's statistics (no gradient) | — | — | — | — | — | — | — | — | move the words, the normaliser and P | — | — | — |

The ✗ under T_loc and T_glob for NCP is the property §4.3 relies on: nothing rewarded for predicting the target can
make the target easier. The only module that every objective reaches is the encoder's G side, which is the stream the
concept path is built on. In the ĉ-injection arm the decoder reads the codewords of the head's argmax tuple, which
passes no gradient, so NTP still does not reach the horizon head; with V13's soft ĉ it would have, through the
expectation (§6.4).

### 8.6 Optimiser

V13-tiny's AdamW (0.9, 0.95), learning rate and schedule for every new parameter: T_loc, T_glob, the horizon
embeddings e_k and, in the arm, the inverse-dynamics heads; e_k and the FiLM parameters (if the §6.2 response is triggered) take no weight decay. The twin, the
successor head's target network, the normaliser, P and the codebook have no optimiser. At scale (v13_NOTES
§9.4's split): the target projectors (and the arm's inverse-dynamics heads) go on AdamW under the elongated-shape rule (T_loc is 4 : 1), the
horizon head's matrices on Muon.

## 9. Continual learning

V13's continual-learning method is being measured tonight (14–15 Oct) on V13-tiny: sequential LoRA, LoRA with replay,
and V13's own method (the correction bank, wake and sleep, gated LoRA) over Dataset_v13's phases S1–S6, read by the
pre-registered rule of `v13_tiny/REPORT.md` §5 item 5 and Dataset_v13 §10.3's thresholds. V14's continual design is
therefore written as a **decision table on those results** (§9.4): the mechanisms are fixed now, and which of them is the
default is decided by tonight's matrix, not by preference.

### 9.1 What is inherited (v13_NOTES §10; `v13_tiny/DEVIATIONS.md` items 25–33)

| Memory | Changes when | Stores | Where |
|---|---|---|---|
| Engram table | pretraining only | static local n-gram patterns | E1 |
| RCDL correction bank | at use time, weights frozen | key–value corrections to the concept state: key = q_g = RMSNorm(u_g^pre)/√d, value = −η ∂L/∂u_g, RBF kernel with σ = 0.3, η capped so the largest written value has RMS 0.1, partitioned by a code | after C's last layer |
| Gated LoRA adapters | in a consolidation ("sleep") phase | a phase's skill, active only on that phase's inputs | E2, C, the decoder (rank 4 tiny) |

V13-tiny's protocol is kept unchanged so that V14 and V13 are compared on the same continual stream: 100 batches of 8
rows per phase (or the budget rule's n), identical rows for every method, a reservoir of 1,000 rows with a 20% replay
share, Φ_neg covering the base stream, text and every earlier phase (C6), no smoothing at concept rate, the continual
matrix A[i, j] by teacher-forced exact match, and Dataset_v13 §10.2's metrics (acquisition, final average, backward
transfer, forgetting, retention before return, relearning, consolidation).

### 9.2 What the JEPA and world-model view changes, from first principles

**(a) A fixed coordinate system.** Forgetting is, in part, drift of the representation that old skills read. In V13
nothing in the concept space was fixed: the codebook chased the online encoder (C3), so a code-keyed bank partition
meant something different after every update. V14's target era (the twin, normaliser, residue map and codebook frozen
together at t_freeze, §3.4) is inherited by the continual stage. Codes are fixed for its whole duration, so the bank's
partitions and any code-keyed gate mean the same thing in phase S6 as in S1. If the end-of-training agreement guard of
§6.1 fails in either half, the refit (and the head's re-fit) happens once, before S1, and defines one era for the
whole stage. In the continual stage the horizon head, T_loc, T_glob, P, the normaliser and the codebook are all frozen:
a frozen head is what makes the anchoring term below an anchor rather than a second training signal.

**(b) In the continual stage the frozen base is the target encoder.** The base weights are frozen and only adapters
train. The base with adapters off is therefore a fixed network, exactly what the twin is for during pretraining. No
twin is carried into the continual stage: the target view is computed by the adapter-off base. E1, E3, Engram, the
embeddings and the pools are frozen and carry no adapters (DEVIATIONS item 25). E1 and Engram read only the embedding,
so their part of that pass is shared with the adapted forward. E2 carries adapters and changes G, so E2, E3 (frozen,
but fed a different G), the exits and the pools are recomputed with adapters off: about 45% of a forward at the tiny
scale (accounting FLOP shares). L is written only by E1 and Engram, so the local half of the target view is the same
with adapters on or off. V13-tiny's gates use LSL's live-input form with no adapter-off pass (DEVIATIONS
item 29), and V14-tiny keeps that form for comparability. Once the adapter-off pass exists for anchoring, V10.3's
adapter-off gates come for free; they are an arm.

**(c) Latent anchoring: keep predicting the frozen target space while learning.** On every incoming (and replayed)
batch, the adapted model's concept states must still predict the base's target codes:

  L_anchor = λ_CL Σ_k w_k · mean over valid (g, g+k) of Σ_s CE( ℓ_{k,s}(u_g^adapted), code_s(τ_{g+k}^base) ),

added to the phase's own loss (and, in V13's method, to the sleep distillation). The methods' own losses differ in
scale (sequential LoRA's is an NTP loss of a few nats; V13's sleep loss is a KL of order 0.1 nats), so λ_CL is set per
phase: at the phase's first batch, λ_CL · L_anchor equals 10% of that method's own loss, and λ_CL then stays fixed for
the phase. It costs the adapter-off pass of (b) and needs no stored data: it is a self-supervised consolidation signal computed on whatever
arrives. It is the latent-space analogue of distilling the old model's outputs: the world model may learn new
dynamics but is held to its old description of the world. Its risk is plasticity: a frozen code space names new
concepts (S2's primitives) with old codes. Its read is acquisition against the same method without it.

**(d) Energy-based continual learning: do not push down energies the data never mentions.** Li, Du, van de Ven and
Mordatch (CoLLAs 2022) found that changing an energy-based model's training objective so that it does not lower the
energies of classes absent from the current data reduces interference with earlier knowledge, without memory or
growth, and that it combines with other methods. V14's NCP head is an EBM over codes (§2.3), and its full softmax
lowers the energy of every code the current batch does not contain. Restricting each segment's softmax to the words present in
the batch would change nothing here: a batch has about 2,000 valid groups at four horizons and only 32 words per
segment, so every word is present in every batch. The idea carries over at the level of whole concepts instead.
During continual phases only, the arm normalises over 𝒱, the set of observed code *tuples* among the batch's targets
and the replay rows' targets (at most a few thousand):

  CE = E_k(c_target | u_g) + log Σ_{c′ ∈ 𝒱} exp(−E_k(c′ | u_g)),   E_k(c | u_g) = −Σ_s ℓ_{k,s}(c_s).

A single-family batch then never pushes down the energies of other families' concepts. This arm gives up the exact
factorised partition function (§2.3) for a sum over |𝒱| tuples, which is still exact over that set and costs
|𝒱| × S additions per group. Applied to L_anchor and to the NCP term of the sleep phase. Not applied to the decoder's vocabulary softmax, whose
calibration generation depends on.

**(e) Gating on the concept code: V13's open problem.** V10.3's input-density gates failed on closely related skills
(add and subtract gates opened on each other's inputs 63–85% of the time, V10.3 N8), and no tested method protected a
related skill (v13_NOTES §10.5). The candidate it named, gating on the concept code, becomes testable in V14 because the
code is now meant to say what a group *did* (the global half), not which document it is in. The gate must be causal at
inference. Rows of group g read the *actual* global-half code of the last completed group, g−1 (two segments, 32² =
1,024 joint values): its target view needs G^enc only up to row 4g−1, so it is available when group g starts, needs no
prediction, and means "what just happened". It is computed live, by the adapted model, consistent with item 29's
live-input gates. The world model's *prediction* for group g (the horizon-1 prior's argmax from u_{g−1}) is the arm; it
is only as good as the head's global-half accuracy, which §5.6 calls the open question. Per phase, Φ_pos^code and Φ_neg^code are
histograms of that code over the phase's incoming rows and over the reference rows (add-one smoothing), and the gate is
open where log Φ_pos^code(c) > log Φ_neg^code(c): a discrete version of LSL's density-ratio gate. Rows in a document's first two groups (no
completed global code yet) fall back to the input-density gate. Read on S3, where the related pairs are.

**(f) The bank's partition key.** V13 partitioned by the segment-1 code of m_g. V14 uses the segment-1 code of the
group's local half a_g (online view, frozen codebook, §6.1): its inputs are frozen in the continual stage, so the key
is the same before and after every adapter. The global-half code is an arm (it names what the group did, which may
partition corrections better, but it moves when E2's adapters change G).

**(g) The knowledge-update stream.** Dataset_v13 §8.2's secondary stream (bindings that are added or revised, queried
later without the update in the prompt) was skipped in V13-tiny (DEVIATIONS item 5). It is the bank's natural task:
one-shot writes, no gradient into weights. It is an arm, scored separately from the reasoning macro, against the
mandatory exact-dictionary baseline.

### 9.3 The method, assembled

For each phase S1 … S6 (V13's order, scheduled boundaries supplied to every method equally):
1. **Wake.** Frozen base plus earlier gated adapters; the bank is written with V10.2's rule on the phase's incoming
   chunks (§9.1), partitioned by §9.2(f)'s key; reset after each chunk.
2. **Sleep.** Distil the bank-corrected model into the phase's fresh adapter (KL on scored tokens, one epoch over the
   chunk, gate held open while training, LSL's Alg. 4), plus L_anchor (§9.2c) when it is on.
3. **Fit the gates** on the phase's first 12 batches (Φ_pos) and the extended reference (Φ_neg): input-density gates by
   default, code gates in the arm.
4. **Evaluate** A[i, ·] on every stable task panel, before the next phase.

Replay (20%, 1,000 rows) is a method of its own and the reference bar; in O2 below it is also combined with V14's method.

### 9.4 Decision table on tonight's V13 results

Thresholds are Dataset_v13 §10.3's and REPORT §5 item 5's: "helps" = forgetting below sequential LoRA's and S0
retention before the return phase (S5) above it, with paired 95% intervals excluding zero, and an acquisition gain at
least half of sequential LoRA's; "continual advantage" = within 3 points of replay's acquisition and at least 5 points better
retained old-task accuracy.

| Outcome of V13's matrix | Reading | V14's continual default | V14's arms |
|---|---|---|---|
| O1: V13's method meets the continual-advantage bar | the bank, gates and sleep work on this stream | V13's method, unchanged | latent anchoring; tuple-level NCP softmax; code gates on S3 |
| O2: V13's method helps against sequential LoRA but loses to replay on final average | protection without enough retention | V13's method plus 20% replay, with latent anchoring on (Li et al.: such objectives combine with other methods) | tuple-level NCP softmax; code gates |
| O3: V13's method does not help (forgetting within noise of sequential LoRA) | the gates or the bank do not protect | replay plus latent anchoring; the bank and gates become arms; gate open rates per phase are read first (C6) | V13's method; code gates |
| O4: V13's method acquires less than half of what sequential LoRA acquires | plasticity, not protection, fails: gates too closed or corrections too weak | V13's method with gates fitted on a larger Φ_pos (24 batches) and rank 8, before any V14 addition. This is a new pre-registration for V14, to be calibrated on V14's own calibration stream (Dataset_v13 §8.3); it is not an amendment applied to V13's result | latent anchoring (expected to cost acquisition, so read last) |
| O5: S3's related counterparts forget under every method (V10.3's result) | the open problem stands | whatever O1–O4 decide | code gates are the primary S3 test |
| O6: the continual stage does not finish by the deadline | no reading | V13's method as designed, arms as listed | all |

Whatever the outcome, the replay bar stays: if a plain transformer with replay beats every V13 and V14 method, that
is the result to report (Dataset_v13 §10.3: "these criteria permit the architecture to lose").

### 9.5 What would show the additions wrong

- **Latent anchoring** costs more than 3 points of acquisition on S2's new primitives at matched exposure, or does not
  lower forgetting: dropped.
- **The tuple-level softmax** does not lower the drop in the NCP head's accuracy on earlier phases' groups (the head's own
  forgetting, read on the frozen codes), or harms acquisition: dropped.
- **Code gates** do not beat input-density gates on S3's counterpart retention, or open on more than 30% of the
  counterpart's groups (V10.3's input gates opened on 63–85%): the code does not separate related skills.
- **The fixed era** shows more than 20% code disagreement between the start and the end of the continual stage on the
  local half (it should be near 0, since its inputs are frozen): something in the key computation is not frozen. The
  global half is expected to move (E2's adapters change G); its disagreement is reported, not used as a falsifier.

## 10. Scale and stability at the 3T class

V14 scales the way V13 does (v13_NOTES §11): the backbone is V13-K3, built on Kimi K3's 2.8T-parameter layout, with
DeepSeek-V4's compressed attention for long range. This section adds only what V14's new parts cost and need at that
scale. **Every number here is accounting from V13's formulas** (`v14_design/accounting_v14.py`, which extends
`v13_design/accounting.py` and writes `accounting_v14.json`), **not a measurement.**

### 10.1 What V14 adds, per tier

| | V14-tiny (7.79M, formula) | 7B-A1.4B (first scale tier) | V13-K3 (2.8T class) |
|---|---:|---:|---:|
| Base model: stored / active / forward FLOPs per token | 7.79M / 6.74M / 13.8M | 6.84B / 1.39B / 2.75B | 2.78T / 115B / 213B |
| EMA twin, full (default) | 3.94M (51% of stored) | 1.94B (28%) | 315B (11%) |
| — added training FLOPs (one forward, no backward; a step counted as 3 forwards)¹ | +23% | +14% | +6.7% |
| — weight memory, bf16 / fp32 master | 8 MB / 16 MB | 3.9 GB / 7.8 GB | 0.63 TB / 1.26 TB |
| EMA twin, local (fallback) | 1.99M, +8.4% | 0.70B, +3.2% | 105B, +1.5% |
| Target projectors T_loc, T_glob (plus their EMA copies) | 24.6K (49K) | 3.5M (7.1M) | 77M (154M) |
| Residue map statistics A, B (fp32); one Cholesky solve | 0.4 MB; 5.6M FLOPs | 57 MB; 9.7 GFLOPs | 1.23 GB; 0.98 TFLOPs |
| One shared horizon head (four separate heads) | 66K (262K) | 9.4M (38M) | 0.21B (0.82B) |
| SIGReg, share of training FLOPs | 5 × 10⁻⁴ | 2 × 10⁻⁵ | 1 × 10⁻⁶ |
| *Arm:* successor head's target network (EMA of C and a head) | 1.1M | 0.92B | 417B |
| *Arm:* inverse-dynamics heads | 33K | 4.7M | 0.10B |

¹ With activation checkpointing a step costs about 4 forwards, and the shares fall by a quarter: V13-tiny trains that
way (`v13_tiny/train.py`: 5.4 GB instead of 10.9 GB at B = 8), so the full twin costs about +17% per step on the M2
(§3.5) and the local twin about +6%.

The twin's share of training compute *falls* with scale (23% → 6.7%) because the decoder, the concept module and the
MoE experts it does not copy grow faster than the encoder. Its memory does not fall the same way, because a MoE twin
must copy every expert it may route to.

### 10.2 Stability and systems items for the new parts

1. **The twin needs an fp32 master copy.** bf16 keeps 8 bits of mantissa (relative step 2⁻⁸ ≈ 0.004). An EMA step
   moves a weight by (1 − μ)(θ − θ̄); at μ = 0.999 that is 10⁻³ of an already small difference, below half a unit in
   the last place of most bf16 weights, so a bf16 twin silently stops moving as μ approaches 1. The twin is kept in
   fp32, sharded with the optimiser state (1.26 TB across the cluster at the 2.8T class, a few percent beside the
   online model's own fp32 master and optimiser state), and cast to bf16 for its forward. Stochastic rounding into a
   bf16 twin is the alternative if memory forces it.
2. **JEPA-loss dropout is the cost lever.** LLM-JEPA's random JEPA-loss dropout lowers its per-epoch cost from about
   2 to 2 − α for dropout rate α. V14's equivalent computes the twin's targets, L_NCP and L_SIG on a random fraction 1 − α of
   rows; at α = 0.5 the twin adds 3.4% at the 2.8T class. NCP's gradient becomes noisier; λ is unchanged, because each
   loss is a mean over the rows that carry it.
3. **The twin is a MoE forward too.** Its tokens are routed by its own (EMA) router, so it adds expert-parallel
   all-to-all traffic of the same volume as a forward. Its routing needs no balancing (no gradients, no capacity
   pressure beyond the forward's), but its tokens occupy expert capacity; it runs as a separate micro-batch phase.
4. **Everything new composes across data-parallel ranks.** Packed documents never span ranks, so document-centring
   for SIGReg is local to a rank; the characteristic-function sums are all-reduced and the random directions are
   seeded by the global step (LeJEPA's implementation). The residue map's statistics A and B, the codebook's counts and
   sums and the normaliser's moments are all-reduced before every EMA update (v13_NOTES §9.4 for the codebook).
5. **The residue map is solved every 100 steps, not every step**, as at every tier (§4.3). At d = 7,168 the map comes from a 14,336 × 14,336
   system (1.23 GB of statistics, about 10¹² FLOPs per Cholesky solve). Its statistics already lag by about 100 steps
   (EMA 0.99), so re-solving P every 100 steps loses nothing; the solve is done once and broadcast.
6. **One shared horizon head:** 0.21B parameters at the 2.8T class, against 0.82B for four separate heads.
7. **The successor-head arm needs a cheaper target network at scale.** An EMA copy of the concept module would copy
   its MoE experts (417B). At scale the arm bootstraps from the *online* concept module under a stop-gradient
   (semi-gradient TD, the textbook form), with the EMA only on the small successor head. That is less stable than a full
   target network, which is one more reason it is a tested arm before it is a default.
8. **Every new gradient is bounded.** SIGReg's Epps–Pulley statistic has bounded gradient and curvature (|∂EP/∂z_i| ≤
   4σ²/N, LeJEPA); the horizon head's cross-entropy is V13's; the twin, P, the normaliser and the codebook take no
   gradient. V14 adds no unbounded path to V13's stability analysis (v13_NOTES §14.7: QK RMSNorm with temperatures in
   every softmax attention, gated attention, SiTU-GLU experts, GDN-2's state norm logged with the KDA tie as fallback,
   the Muon / AdamW split, MoE balancing, the NCP gradient-share cap).
9. **The reach budget is unchanged:** R_D = 327 at the scale tier, ρ = 327 / 4,096 = 0.08 ≤ 1/8 at the shortest
   training length. Nothing in V14 touches L's reach or the decoder.
10. **Long contexts.** The target view, SIGReg and NCP are per group, so linear in sequence length; at the 64K phase the
    decoder's concept read becomes DeepSeek-V4's compressed attention with C as the compressor, as in V13 (v13_NOTES
    §7.2).

### 10.3 The evidence ceiling

| Mechanism | Largest reported scale | Source |
|---|---|---|
| EMA-teacher self-distillation | 7B (vision) | DINOv3 |
| JEPA with an EMA teacher | 1B encoder (video) | V-JEPA 2 |
| JEPA loss inside a language model | 8B fine-tuning; a 1B model pretrained from random initialisation on one small dataset | LLM-JEPA |
| SIGReg | 1.8B ViT-g, stable | LeJEPA |
| Next-concept prediction over a VQ concept vocabulary | 1.5B from scratch (300B tokens); 8B continual pretraining | ConceptLM |
| Inference-time energy minimisation | 800M | Energy-Based Transformers |
| Gated DeltaNet-2 | 1.3B | v13_NOTES §3.6 |
| The backbone layout | 2.8T | Kimi K3 (v13_NOTES §11) |
| V13 as a whole | 7.7M, one seed (in progress) | this repository (`v13_tiny/`) |
| V14 as a whole | not yet run | — |

No JEPA-style objective has been reported near 3T parameters. At that scale V14's additions are designed to be cheap
and bounded, not known to work.

### 10.4 What each tier must establish before the next (extending v13_NOTES §11.5)

- **Tiny (§11):** whether the target view defeats copy-current and local-only, whether centred SIGReg prevents slow
  collapse, what the twin costs on the M2.
- **7B-A1.4B:** the twin's measured cost and memory; χ, K3 per half and the residue map's R² at a language-model data
  mix; the successor head at γ = 0.97; the (ν_S, ν_I) sweep if the inverse-dynamics arm survives the tiny test.
- **2.8T class:** JEPA-loss dropout's α from the 7B tier's cost curve; the third concept level (§5.5) as an arm.

## 11. The tiny test and what it decides

**What is reused, byte for byte:** V13-tiny's data generators, tokenizer, ladders, rows, curriculum, continual
stream, evaluation code and 11-hour training budget (`v13_tiny/DEVIATIONS.md` items 1–14, 24–34). The model differs
from V13-tiny only by §1.5's table. That is what makes three comparisons valid on one machine: V14 against V13-tiny,
V14 against the baseline transformer the user runs next, and V13-tiny against that baseline. Every result is one seed:
"observed, seed-unreplicated".

**In order of information per run** (one M2 Air, about 11 hours per run, the baseline transformer occupying the next
day):
1. **G0 unit tests** for the new parts, in float64: τ_g uses only groups ≤ g of its own document; the twin equals the
   online encoder at step 0 and moves at the scheduled lag; no gradient reaches the twin, P, the normaliser or the
   codebook; first groups of documents are masked everywhere; centring uses documents with at least four valid groups; every
   (g, g+k) pair stays inside one document; P is 0 for the first 200 steps and each later solve matches a float64 ridge
   least-squares fit on the same statistics; every V13 test (reach 51 / 66, document isolation, release) still passes.
2. **A 200-step smoke test:** seconds per step and peak MPS memory with the twin (the accounting predicts about +17%:
   V13-tiny trains with activation checkpointing, so a step costs about four forwards and the twin adds
   one, §3.5), SIGReg's cost, the action-sensitivity read's cost (§6.3), every logged read non-empty.
3. **The default run**, 11 hours (about 10,600 steps at +17%, against V13-tiny's about 12,400), then
   V13-tiny's post-training pipeline unchanged (probes, final evaluation, diagnostics, continual stage). Before it
   starts, `diagnostics.py` is extended with §6.3's reads (the residue map's R², χ per half, K3 per half, the
   cross-correlation between the halves, the action sensitivity and effective rank per half) and V1's two baselines, so every reading below is computed by code written
   before any V14 result exists.
4. **The default run's configuration is fixed by K1–K5 before it starts** (§1.4, §4.4, §8.2), from tonight's checks on
   V13-tiny's checkpoints: the local-view fallback replaces the global half if K4 shows G's change dead, and
   λ_max = 0.2 is the default if K1 confirms H1. These rules set the default; they are not arms.
5. **One arm after the default run, chosen by its own readings**, in this order of precedence: V4 fails → inverse
   dynamics added (§8.4); V1 fails and the residue map's R² is above 0.95 in at least 3 families → the local-view run
   (§4.4); V1 fails otherwise → the H1 ladder (λ_max = 0.2 if the default did not already use it, else β₁ lowered);
   V2 fails → no arm, the nonlinear residue is reported (§12.3); V1–V4 pass → "VICReg instead of SIGReg", the user's
   stated form of the anti-collapse request.
6. **The queue, in order:** inverse dynamics added (if not already run); the successor head at γ = 0.97; β₁ lowered;
   "V13 plus" horizon weights; N = 64; the cosine latent term; the third level (scale only).
   Pre-registered for later tiers, and run in the tiny test only if their named read fires first: the statistic
   projectors (§4.3's check; if it fails they are the default instead), the branch head (§6.2, the cross-pair read),
   soft code targets (§6.2) and the fixed-quantiser control (§6.1). Listing them is not a claim that they run.

**Pre-registered readings for V14** (written now; V13's readings of REPORT §5 apply unchanged beside them):

| # | Question | Pass |
|---|---|---|
| V1 | Does the head beat copy-current and the marginal? | On the global half at horizon 1, the head's accuracy beats both baselines by at least 10 points: persistence (the current group's code) and the family marginal (each segment's most frequent word in that family on training rows); in at least 3 of the 4 reasoning families, at the final checkpoint. Persistence alone is weak on a difference target, so the marginal is the baseline that tests H1 |
| V2 | Does the target need the concept path? | The u_g probe beats the local-only probe (L^enc at row 4g+3) by at least 2 points on the global half in at least 2 families |
| V3 | Does the concept module compute? | §5.6's per-half probe test on the global half: u_g beats p_g by at least 5 points; reported per family, pass or fail (REPORT §5 item 3's form), beside V13-tiny's result in the same family |
| V4 | No slow-feature collapse | χ of the twin's normalised global half at least 0.5 at the final checkpoint, in every family. Why 0.5: with centred SIGReg holding the within-document covariance of b at I and the normaliser holding each coordinate's total variance at 1, χ_b = 1/(1 + σ²_means), so χ_b ≥ 0.5 means the variance of the document means is at most the within-document variance. That is a different quantity from H-JEPA's χ of a *level* (0.008–0.016 collapsed, 0.12 with inverse dynamics), which is why the bar differs. If §8.3's uncentred test on b fires during the run, the run is reported as "default + uncentred b" and V4 is read on that run |
| V5 | Are codes stable? | Code reassignment on fixed inputs between snap_010 and snap_050 (matched fractions of training time; the twin's lag grows from 105 to 400 steps there) is lower in V14 than in V13-tiny. The twin's lag reaches about 9,000 steps by 70% and its codes freeze at 75%, so later windows would favour V14 by construction: snap_030–snap_070 is reported beside it, not compared |
| V6 | What does it cost? | Text NLL at matched training time within 0.02 nats of V13-tiny's; the Reasoning score reported against V13-tiny and the baseline |
| V7 | Continual learning | §9.4's table, with Dataset_v13 §10.3's thresholds |

A V14 that passes V1–V4 and fails V6 has a better concept path at a price. One that fails V1 or V4 has not fixed what
it was built to fix, and item 5 above names the next run.

## 12. Evidence ledger, risks, and what would change the design

### 12.1 The new mechanisms, by the evidence behind them

| Mechanism | Measured here | Measured elsewhere | Untested |
|---|---|---|---|
| NCP as cross-entropy over PQ codes | V12; V13-tiny (fails its probe test, interim) | ConceptLM (to 1.5B, 8B continual) | — |
| EMA target encoder | — | BYOL, I-JEPA, V-JEPA 2, DINOv3 (to 7B) | with EMA-k-means-quantised targets; in an autoregressive language model |
| Target view separate from the input | — | the JEPA convention (I-JEPA, V-JEPA) | in this composition |
| Difference target (ΔḠ) | — | MotionJEPA | in language |
| Residue map P | — | — | entirely new |
| Typed halves | — | SD-JEPA's subspaces | as PQ segment groups |
| Document-centred SIGReg | — | SIGReg (LeJEPA, to 1.8B); H-JEPA Prop. 2 (theory, uncentred: "real but small"). Against: every run of H-JEPA's uncentred SIGReg ladder collapsed (Table 12) | centred, as the backstop of a difference target against slow collapse |
| Horizon-conditioned head | — | I-JEPA's target-position conditioning; MTP | over concept codes |
| Successor head (arm) | — | TD-JEPA (reinforcement learning) | in language |
| Inverse dynamics (arm) | — | H-JEPA, AC-MTM, LeWM+I (robotics, exogenous actions; the encoder-side gradient carries the repair) | with observed actions |
| Latent anchoring, tuple-level softmax, code gates | — | Li et al. (energy-based continual classification) | all three in this composition |

### 12.2 Risks, most likely first

1. **G's change carries nothing non-local on this data** (the residue map's R² near 1). Then V14 is V13 with a faster,
   local target: the concept path's necessity rests on the decoder's confinement alone, as in V13, and V2 fails.
2. **The difference target with its SIGReg backstop is not enough** (untested together; uncentred SIGReg alone failed
   in H-JEPA's ladder). V4 fails; the inverse-dynamics arm is next.
3. **The twin's cost** (about +17% per step on the M2) means about 15% fewer steps at matched time; V6 may fail for that
   reason alone, which the matched-step comparison on V13-tiny's snapshots separates.
4. **The target becomes too hard**, NCP's gradient share exceeds NTP's, the cap halves λ, and the concept module gets
   less shaping: H1 by another route. Read from the cap's log.
5. **Codes disagree after training** (the global half after the twin froze): the era refit of §6.1 runs.
6. **One seed** throughout.

### 12.3 What each outcome changes

| Outcome | Change |
|---|---|
| V1–V4 pass | The target view, the twin and centred SIGReg go to the 7B tier as designed |
| V1 passes, V2 fails | The target is fast but local: the global half needs more than the residue map (a nonlinear residue, or a target from a model without local access) |
| V4 fails | The inverse-dynamics arm becomes the default barrier, with the 3 × 3 weight sweep at the next tier |
| The residue map's R² ≈ 1 in most families | The local-view fallback becomes the default, the twin shrinks to the local twin (about +6% per step on the M2 instead of +17%), and the concept path's case rests on confinement |
| V6 fails by more than 0.1 nats | JEPA-loss dropout (α = 0.5) halves the twin's cost in the next run |
| The successor head beats its past-histogram baseline at γ = 0.97 | It becomes the 7B tier's long-horizon default |
| §9.4's O1–O6 | As the table says |

## References

Fetched and checked during this design session (14 Oct 2026); the claims taken from each are the ones cited in the
text.

*Joint-embedding predictive architectures and their anti-collapse terms*
- I-JEPA: Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture",
  https://arxiv.org/abs/2301.08243
- V-JEPA 2: Assran et al., https://arxiv.org/abs/2506.09985
- LLM-JEPA: Huang, LeCun and Balestriero, https://arxiv.org/abs/2509.14252
- LeJEPA (SIGReg): Balestriero and LeCun, https://arxiv.org/abs/2511.08544
- H-JEPA (slow-feature collapse; Propositions 1–2): https://arxiv.org/abs/2610.06805
- AC-MTM, "No Gaussian Required" (Action-NCE): https://arxiv.org/abs/2608.17542
- MotionJEPA (difference targets): https://arxiv.org/abs/2609.23881
- SD-JEPA (subspace decomposition): https://arxiv.org/abs/2605.31111
- TD-JEPA (temporal-difference latent prediction): https://arxiv.org/abs/2510.00739
- Discrete JEPA: https://arxiv.org/abs/2506.14373
- Latent JEPA: https://arxiv.org/abs/2610.01947
- Littwin et al., the implicit bias of JEPA toward predictive features (NeurIPS 2024): https://arxiv.org/abs/2407.03475
- DINOv3 (EMA-teacher self-distillation at 7B): https://arxiv.org/pdf/2508.10104

*Concept-level language models*
- ConceptLM, "Next Concept Prediction in Discrete Latent Space" (NCP-Arch in v13_NOTES): https://arxiv.org/abs/2602.08984
- HiLP, "Hierarchical Latent Prediction for Language Models": https://arxiv.org/abs/2608.05806

*Energy-based models*
- Energy-Based Transformers: https://arxiv.org/abs/2507.02092
- Blondel et al., autoregressive models as energy-based models (ICML 2026): https://arxiv.org/abs/2512.15605
- Li, Du, van de Ven and Mordatch, energy-based models for continual learning (CoLLAs 2022):
  https://arxiv.org/abs/2011.12216

*From the user's JEPA catalog (`JEPA_Architecture_Papers_Catalog.md`), read in full text on 14 Oct 2026; per-paper
depth and notes in `v14_design/PAPER_NOTES.md`*
- Sobal et al., "Joint Embedding Predictive Architectures Focus on Slow Features": https://arxiv.org/abs/2211.10831
- "Koopman Invariants as Drivers of Emergent Time-Series Clustering in Joint-Embedding Predictive Architectures"
  (Thm 3.4): https://arxiv.org/abs/2511.09783
- C-JEPA, "Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning" (Tables 3–4):
  https://arxiv.org/abs/2410.19560
- "Drive vs. Decay: On the Training Dynamics of Joint-Embedding Predictive Architectures" (Prop. J.7, Table 5):
  https://arxiv.org/abs/2610.02344
- Tian, Chen and Ganguli, "Understanding Self-Supervised Learning Dynamics without Contrastive Pairs" (DirectPred, §4):
  https://arxiv.org/abs/2102.06810
- "Why and How Auxiliary Tasks Improve JEPA Representations" (Thm 1): https://arxiv.org/abs/2509.12249
- LeWM+I, "Keeping JEPA World Models Plannable When Little of the Frame Moves" (Table 2, the action-sensitivity probe):
  https://arxiv.org/abs/2610.03137
- LeWorldModel (LeWM): https://arxiv.org/abs/2603.19312
- "Factorized Latent Dynamics for Video JEPA: An Empirical Study of Auxiliary Objectives":
  https://arxiv.org/abs/2605.17165
- EB-JEPA, "A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures":
  https://arxiv.org/abs/2602.03604
- "I Act Therefore I Am: When Is JEPA's Action-Conditioning Enough to Learn Causal Mechanisms?":
  https://arxiv.org/abs/2609.31161
- "The JEPA Paradox in Language: The Geometry of Linguistic Alternatives": https://arxiv.org/abs/2607.23531
- Branch-JEPA, "Finite-Support Predictive Distributions for JEPA World Models": https://arxiv.org/abs/2607.05238
- VJEPA, "Variational Joint Embedding Predictive Architectures as Probabilistic World Models" (Thm 1):
  https://arxiv.org/abs/2601.14354
- Var-JEPA, "A Variational Formulation of the Joint-Embedding Predictive Architecture" (§3.7):
  https://arxiv.org/abs/2603.20111
- S-JEPA, "Soft Clustering Anchors for Self-Supervised Speech Representation Learning": https://arxiv.org/abs/2606.19398
- SC-JEPA, "Stabilizing Latent Predictive Learning for Time-Series Anomaly Prediction": https://arxiv.org/abs/2602.04643
- "Introduction to Latent Variable Energy-Based Models": https://arxiv.org/abs/2306.02572
- "JEPA as a Neural Tokenizer" (finite scalar quantisation): https://arxiv.org/abs/2512.07168
- H-JEPA's Table 12 (the SIGReg ladder), App. K.1 and App. L.5–L.7, and LeJEPA's §4.3, §5.1–§5.2 and Table 1 were read
  in full in this pass; both papers are listed above.

Standard references, cited for well-known results: BYOL (Grill et al., 2020, https://arxiv.org/abs/2006.07733) and
VICReg (Bardes, Ponce and LeCun, 2021, https://arxiv.org/abs/2105.04906), both in the catalog and read at abstract
level; the Epps–Pulley test (1983) and the Cramér–Wold theorem.

Inherited through `v13_NOTES.md`, which gives their sources and what was taken from each: Kimi K3, DeepSeek-V4,
Gated DeltaNet-2, Engram, LSL (gated LoRA), RCDL (the correction bank) and Irie et al.'s dual form, V10–V12 of this
repository, and multi-token prediction.
