# V12 design, step by step (working notes; each step reviewed by the advisor, see ADVISOR_LOG.md)

## Step 2 — Backbone and segmentation (general data, frontier-scale path)

Requirement (user, 09:20): no overfitting to one distribution; the design must scale to frontier MoE size
(reference points: Kimi K3, 2.8T total MoE, 16/896 experts, Kimi Delta Attention + Attention Residuals, active
count undisclosed; GLM-5, 744B total / ~40B active, 256 experts with 8 routed + 1 shared, 28.5T tokens).

Decision 2.1 — keep E→C→D exactly as NCP-Arch lays it out (16 enc / 8 concept / 16 dec at 8.9B). The unifying
fact: **E is the lower half of a standard trunk, D is the upper half, and C is inserted at the split. That split
is mid-depth — where, in the ETA paper, linear subgoal decodability has risen (it rises until layer 4 of 6 and
stays strong to the final embedding, Fig. 3) and internal control works best (Fig. 4; controller at L/2). So the concept module's position and the metacontroller's control point are the same place by
construction.** Cost accounting: tiny ECDe ≈ 4.5 token-layer equivalents + 2 cross-attentions vs TF7's 7 layers,
so the tiny comparison is parameter-matched, not compute-matched; at scale the comparison that counts is
compute-matched (NCP-Arch's 1.74× claim rests on C costing 1/k per token). Both are declared.
- E (token rate, causal, L_E layers) → segment pooling → C (segment rate, causal over *completed* segments,
  L_C layers, run at 1/k of the token cost; the NCP-Arch lever: cheap concept-rate capacity) → D (token rate,
  causal, L_D layers, reads released concept states by cross-attention).
- At scale every FFN is an MoE (C's experts are where cheap capacity lives: one C token stands for k tokens).
- Attention: whatever the trunk uses at scale (hybrid linear/full such as KDA, or DSA); the ECD contract only
  needs causal token mixing in E/D and causal segment mixing in C.

Decision 2.2 — segmentation. Fixed k-token groups are a tiny-test convenience only. Evidence: V10.3/V11 period-8
layout dependence on Sudoku; pinpad subgoal segments are 1–19 steps, not aligned to any k. At scale use learned
boundaries: an H-Net-style boundary score from the cosine dissimilarity of adjacent E states, thresholded with a
ratio loss to a target mean segment length k̄ (compute budget for C = T/k̄); segment state = attention pooling
over the segment. Release rule unchanged: a segment is visible to D from its last token on.
Tiny test keeps k = 4 fixed so the rest of the stack is testable; this is declared.
Two boundary mechanisms, two clocks: the *segmentation* boundary (pretraining, content change, mean length k̄,
defines C's clock) and the *subgoal* boundary β (post-training, spans several segments, defines the
metacontroller's clock). β is learned on the frozen base, so segments need not align with subgoals — that is the
point of learning β. Risk: a raw cosine-dissimilarity detector would fire on every pinpad token (action and
position tokens alternate), so the detector must be trained (H-Net ratio loss + smoothing), not a heuristic.
**The tiny test neither supports nor falsifies dynamic chunking; it is the largest untested item in the scaling
path.**

Decision 2.3 — what D reads: released concept states u (continuous) via cross-attention, plus nothing else from
C. The NCP head is an auxiliary readout of u (Step 3), not an input to D (V10.1/V10.3: a soft forecast fed to D
shrinks toward the codebook centroid and is ignored).

Usage metric (declared before results): Δ NLL with the concept route zeroed (`nll_concept_off`). C counts as
*used* on a family if zeroing it costs ≥ 0.02 nats/token (pinpad: per action) with a paired interval excluding 0.
C never receives D-side gradient: for target = 'dec' the decoder-side target is stopped, and the shortcut monitor
(`acc_vs_concept_off_target`, NCP accuracy against targets computed with the concept route zeroed) checks that
the prediction is not trivial through D's own concept read.

## Step 3 — Next-concept prediction (NCP) as classification over a discrete segment codebook

Problem statement from evidence: V10.1/V10.3/V11 all regressed the next concept (MSE or soft-PQ mixture). The
forecast shrank to 3–14% of target variance (V10.3 §1.2), was a worse predictor than its own input, and the
explorative mixture found no differing futures. First principle (XM §2.1): under squared error the loss minimiser
is the conditional mean; the next segment is multimodal given the prefix (which subgoal next, which clause next),
so a regressor must blur. A categorical over discrete codes is mode-preserving.

Decision 3.1 — target: tau_{g+1} = sg[EMA-normalised segment summary of g+1]; summary = mean E state ('enc',
NCP-faithful) or mean D state ('dec', HiLP-style upper level; user's proposal). Quantiser: product codebook,
S = 4 segments × N = 32 words (at scale S·log2 N ≈ 64–128 bits per segment, e.g. NCP-Arch's 32 × 128), updated by
EMA k-means with masked dead-code reset — never by gradient, so there is no codebook-collapse/commitment term
and no weight decay on the codebook (V10.3 N3 fix generalised).
Decision 3.2 — loss: L_NCP = Σ_s CE(head_s(u_g), idx_s(tau_{g+1})), λ = 0.1. The joint over segments is the
product of per-segment categoricals (factorised): mode-preserving within a segment, independent across segments.
(An autoregressive-over-segments head would model the joint; deferred — cost S sequential softmaxes.)
Decision 3.3 — gradient path: NCP trains the head and, through u_g, C and E (the context side); the target is
stopped (NCP-Arch §3.3); D receives no NCP gradient. For 'dec' the target is computed from D's states and stopped.
Decision 3.4 — what is reported: per-family code accuracy, CE, code usage perplexity, and the information test
that killed V10.3: does a linear readout of u predict the next segment's codes better than a readout of C's own
input p_g (the core must add information)?
Why not soft-PQ MSE (V10/V11) and why not XM over K regressors (V10.3/V11): both are point/mixture regressors
whose failure is measured; CE over codes gets multimodality from the categorical itself at no K× cost.
Decision 3.5 (advisor #6) — pre-registered checks, written before any result:
- gradient share at the 4k checkpoint, one batch per family: ‖∂(λ L_NCP)/∂θ_E‖ / ‖∂L_NTP/∂θ_E‖ and the same on
  θ_C; acceptance: NCP's share on E ≤ 1× NTP's (V11's bridge was 2.6–2.9×). Above 1× is a λ finding, not a
  mid-run change. The acceptance is on E only; the share on C is reported, not judged.
- collapse monitor: per-family code usage perplexity → 1 means that family's summaries became uninformative;
  the only counter-force is NTP through D's concept read (ties to the 2.3 usage metric).
- shared-codebook bookkeeping: the dead-code threshold compares two EMAs across steps and families
  (0.02 · EMA(rows)/N); two-family check: both families keep codes (30 vs 2 in segment 0 with 8% minority mass —
  k-means allocates codes in proportion to row mass, so per-family usage is reported, not assumed).
- horizon ablation (listed, not run): a g+2 head (HiLP's gain is at longer horizons).
- XM is not dismissed: it still applies to a continuous within-segment residual, or as K discrete latent codes at
  the metacontroller (XMDLM's form) — judged in Step 6.

## Step 4 — NextLat and HiLP: which latent predicts which, at which rate

What V11 did (and why it was wrong): a *concept-anchored* token-rate rollout — from u_g and the actual next tokens,
an MLP rolled out up to 8 decoder states without re-grounding, with SmoothL1 + KL. That is neither NextLat (whose
anchor is the same network's own previous state h_t, d = 1 suffices) nor HiLP (whose upper level is a separate,
coarser predictor that never takes token input). Measured cost: 2.6–2.9× NTP's gradient into the core, +0.41 nats
on math.

V12 assigns each method to the level it was designed for:
- **NextLat at token rate on D** (the deployed predictor's own state): p_ψ([d_t ; E x_{t+1}]) → d̂_{t+1}, 3-layer
  GELU MLP with residual, SmoothL1 on RMS-normalised states to sg[d_{t+1}], KL through a frozen view of the output
  head, d = 1, λ_h = λ_KL = 1 (NextLat Table 5 defaults). Training-only; dropped at inference. Its job is to make
  D's state a belief state (NextLat Thm 3.2) — compact, future-predictive — which is also what makes the residual
  stream controllable later.
- **HiLP's upper level at segment rate = NCP with the 'dec' target** (the user's proposal, implemented as HiLP
  specifies): the concept state u_g (no token input) predicts the *next segment's decoder latent* sg[mean D
  states of g+1], as discrete codes (Step 3). HiLP's stop-gradient separation is preserved: D never receives this
  loss's gradient; C and E learn to anticipate what D will represent. HiLP's combined head is not needed: in ECD,
  D already *is* the combined head (it reads u by cross-attention under one NTP loss), so there is no second CE.
- Gradient routes (which loss reaches what):

  | Loss | head / NCP head | D | C | E | codebook |
  |---|---|---|---|---|---|
  | NTP | output head | yes | via D's cross-attention read of u | via h→D and via h→pool→C→u→D | — |
  | NCP-enc / NCP-dec (λ 0.1) | NCP head | **no** (target stopped) | yes (u_g) | yes (context side) | EMA only |
  | NextLat L_h, L_KL (D's state) | frozen view (L_KL) | yes (through d_t) | via D's read | via D | — |

  This departs from HiLP deliberately: HiLP's upper module sees sg[lower latents]; here C is shaped by NTP through
  D's read *and* by NCP. The NTP route is the counter-force to collapse (3.5) and lets C's content follow what D
  needs. ECDd's target is in D-space while the predictor reads C-space (unlike HiLP's u_t → u_{t+k}); what carries
  it is grounding: the target is NTP-trained and cannot collapse while NTP holds.
- NextLat details declared: L_h on every real position, L_KL on scored targets only (NextLat §C); states
  RMS-normalised before SmoothL1 (inherited from V11; the head is scale-invariant); predictor hidden 2d (within the
  paper's 1–4× range).
- Hypothesis, not consequence: that NextLat on D makes the *control-point* residual more controllable. NextLat shapes
  D's final state; the control point (E/D split) is reached only indirectly. TFNL vs TF7 tests it: mid-depth probe
  accuracy, effective rank of the control-point residual (NextLat's own label-free compactness measure), GT-controller
  success.
- **Not used:** multi-step rollouts without re-grounding (V11 bridge), and any token-space concept loss that
  reaches the trunk (MTP-like; V10.3).

Tested arms: ECDe (NCP-enc), ECDd (NCP-dec = HiLP upper level), ECDeNL (NCP-enc + NextLat on D), TFNL (NextLat on
a plain transformer — separates "NextLat helps" from "ECD helps"). Discriminating metrics: probe accuracy of the
subgoal/colour latents by depth and in the concept stream (NextLat's own evidence is probe-based), GT-controller
success, metacontroller switch F1; NLL per family is a guard.

## Step 5 — Loop of 2 at one specific region (the concept module)

Evidence: visit 2 was inert in every prior test (V10.3 ≤ 0.011 nats; V10.1 10M visit-2 state cosine 0.985–0.993;
V11 null). First principle: a second pass with the same weights and the same input can only add depth; it earns
its compute only if it sees something visit 1 did not, or has a different job.
Decision 5.1 — the loop lives in C only (segment rate: 1/k̄ of token cost, so a second visit is cheap at scale),
shared weights, visit embedding added.
Decision 5.2 (revised, advisor #8) — what the loop uniquely buys is **post-control depth at 1/k̄ cost**: with the
controller between visit 1 and visit 2, two C blocks act on the steered state instead of one (ECDe, loop = 1, is
controlled mid-C, i.e. one block after U). The paper's Fig. 4 says controllability depends on depth after the
control point. Confound declared: ECDL2e vs ECDe differs in post-control depth *and* in looping, so "better under
control" would mean "more concept depth after the controller helps", not "looping helps", unless ECDe is also run
with its control point at the C output. **If ECDL2e is not run, the loop is carried as an untested hypothesis
with every prior measurement saying inert.** Original wording kept for the record: visit 2 is the controlled visit: the metacontroller reads visit-1 states (belief) and writes its
low-rank controller U_t onto the visit-2 input; visit 2 then computes the *steered* concept that D reads. In
pretraining (no controller) the loop is plain weight-tied depth, expected to be ~inert on NLL (consistent with all
prior evidence); its value, if any, appears only under control. Test: GT-controller and metacontroller results at
the concept point for ECDL2e vs ECDe (one-change pair). If ECDL2e is not better under control, drop the loop.
Not adopted: HRM/TRM-style deep recurrence with deep supervision (HRM-Text's gains are compute-per-token and its
largest lever was the objective — response-only + PrefixLM — which V12 adopts directly: math is answer-only scored).

## Step 6 — Explorative modelling: where multimodality actually lives in V12

XM's precondition: the target is multimodal *given the conditioning*. In V12 that holds at three places:
1. **The next concept** (which subgoal / clause next). Handled by the categorical over codes (Step 3): generative
   expressivity N per segment without K forward passes. The V10.3/V11 "no differing futures" measurement came from
   text/arithmetic/Sudoku, where no persistent latent existed; on pinpad one does, so XM-at-concept on pinpad is
   *untested* and listed as an ablation, not settled.
2. **The metacontroller's abstract action** z (which subgoal to pursue). The paper uses a VAE (acausal encoder +
   KL); XM argues VAEs struggle (posterior collapse) and that exploration recovers the coupling instead. V12 keeps
   the paper's VAE as the primary (it has the only positive internal-RL evidence). The labelled ablation is
   **XM codes on the VAE's switches**: train the paper's metacontroller, freeze its switch unit, then fit K = 8
   learned controller codes (XMDLM's discrete-latent form) with hindsight soft-min over codes per β-segment.
   (A KL-free XM metacontroller with β trained on NLL alone has no force for sparse switching: its optimum is
   β = 1 everywhere — the paper's failed forced-switch ablation and V5's saturated β.) The payoff to test: a
   discrete abstract-action space makes internal RL a categorical bandit over K per segment, lower-variance than
   Gaussian z (the paper's E.2 argument applies more strongly). Run only if Phase 2 time remains, token-rate (TF)
   bases only, and only for an α whose thresholded switches/episode ∈ [1, 4] with pair F1 ≥ 0.3 (`meta_xm.py`).
3. **Exploration at post-training**: internal RL samples z at switches — that *is* exploration in the
   abstract-action space, the place the ETA paper shows it pays (raw-token sampling ~1e-6). And sleep-time
   dreaming (Step 8) samples K dreams from the z prior and keeps the most useful (Sleep's gradient-based top-k ≈
   reverse XM over self-generated data).
Not used: hard best-of-K on the token head (AR LMs are XM's least-bottlenecked family, XM §7).

## Expected pattern at the 4k checkpoints (written 10:20, before either lands)
1. Probes (pinpad dev): TF7 mid-depth (e_3/e_4) — target cell clearly above chance (pilot: 0.27 at 3k), colour
   weakly above (≈0.2), pair ≈ chance. ECDe concept stream (token view of u) — if NCP does what Step 3 claims,
   colour and pair are *higher* in u than at TF7 mid-depth at the same step, and target cell lower (u summarises
   segments, not the current step). A null (u ≤ TF7 mid on colour/pair) says NCP-enc did not induce the abstraction
   by 4k.
2. Parity (first vs second of the pair) is the layout-independent rule; expected above chance in both if any
   abstraction formed.
3. Gradient share (ECDe): λ·NCP / NTP on E ≤ 1 on every family (acceptance); expected well below 1 at λ = 0.1.

## Step 7 — The metacontroller (the user's main goal)

Paper-faithful core (ETA D.2/E.1), implemented in meta.py: frozen base; at the control point the residual e_t
(pre-control) is read; history GRU (n_h 32, causal); non-causal sequence embedder s(e_{1:T}) (bi-GRU, n_s 32);
encoder → N(μ_t, Σ_t), n_z = 8; causal switch β_t = f(e_t, h_{t-1}, z_{t-1}); z_t = β_t z̃_t + (1−β_t) z_{t−1};
hypernetwork → rank-16 U_t; e ← e + U_t e; loss = action NLL + α Σ_t KL(N(μ_t,Σ_t) ‖ N(0,I)); α swept over
{0, .05, .1, .17, .3, .5, 1}; β thresholded at 0.5 for internal RL; policy π(z | e_{1:t}) (GRU) replaces the
acausal encoder; critic-free PPO on sparse reward. Engineering: history/embedder/encoder run in parallel over time,
only the β recursion is sequential; recurrence on CPU, base on GPU.

What V12 changes relative to the paper (each a testable claim, none assumed):
7.1 **Where to control.** ECD's concept stream *is* mid-depth (Step 2.1). Two points are compared on the same
    frozen ECD base: concept stream (segment rate: the controller's clock is already k̄× coarser, shortening the RL
    horizon further) and decoder input (token rate, paper-faithful). TF's mid-depth residual is the control.
    Cost of concept-rate control: horizon shortened k̄×, switch *resolution* coarsened k̄× (a switch can only land at
    a group release, every 2 env steps at k = 4). Switch F1 is therefore scored on a common env-step clock with
    tolerance ±2 steps (also ±1 reported); KL is normalised per action at both points, so rate–distortion curves
    and F1 are compared across points, never α values.
7.2 **Unit of time.** One environment step (decision row), not one token; token streams interleave actions and
    observations, and the paper's timestep is an (o, a) pair.
7.3 **Success criteria are the paper's, not NLL**: probes → GT-controller compositional success (Fig. 4) →
    rate–distortion with switch-aligned points and switch F1 vs pair and colour switches (Fig. 7) → internal vs raw
    RL on a sparse-reward novel composition (Fig. 8), gated as pre-registered in phase2.py.
7.4 **ECD-native posterior (listed ablation):** the acausal embedding reads the *future concept codes* (the NCP
    targets of later segments — discrete, compact, already trained to summarise segments) instead of raw residuals.
7.5 **At scale:** the metacontroller is a post-training module on a frozen LLM, trained on clean reasoning traces
    (paper C.4.4: cleaner demos discover better), with internal RL on verifiable sparse-reward tasks (math, code)
    against GRPO; β learns step boundaries on its own clock (Step 2.2). Never co-trained with the base (paper
    Fig. 7; V5/V11). **Central uncertainty of the scaling path:** the paper's precondition is data generated by a
    latent compositional subgoal process; whether noisy reasoning traces carry a persistent latent at all is open.
    Mapping: LLM reasoning has no observation stream — every token is an action and the verifier is the
    environment — so 7.2's (o, a) timestep becomes one token, and the only coarser clock is the segmentation
    boundary of 2.2. That is an argument *for* concept-rate control at scale. GT controllers (7.3) are a
    label-dependent diagnostic that exists only in the tiny test.
7.6 **Wake–sleep iteration** (paper discussion, Schmidhuber): internal-RL successes become new self-generated data
    that sleep consolidates into the base (Step 8); the next metacontroller is fit on the updated frozen base.

## Step 8 — Continual learning: fast correction bank → sleep consolidation into gated low-rank experts

Evidence (V10.3 §8, not re-run here): RBF bank keyed by the concept state, weights frozen: −1.55 nats on an unseen
math module (88% of LoRA's gain), text/Sudoku unchanged; consolidation (bank → fresh gated LoRA by distillation,
bank reset): 90% of LoRA's gain, zero forgetting on dissimilar families, multiplication +0.18 vs plain LoRA +2.24;
the related add/sub skill unprotected (+0.26–0.46 under every method). Calibration flaw found: token-weighted screen.
Design (Sleep paper §3.2–3.4 mapped onto ECD):
8.1 **Wake / fast memory:** the RBF correction bank writes exact-gradient corrections keyed by u (the concept state)
    and reads only through the concept routes, so interference is bounded by the concept path's reach. η, σ
    calibrated per family (not token-weighted), write-RMS cap explicit.
8.2 **Sleep / NREM — consolidation by knowledge seeding:** unlock a pre-allocated, masked low-rank expert in the
    slower module (at tiny scale gated LoRA in C/D; at scale a new MoE expert — Sleep's "mask, then activate");
    freeze everything else; distil the bank-corrected model (teacher) into it with on-policy distillation (student
    samples, teacher logits; Sleep §3.3 GKD form) on self-generated dreams; then reset the bank (pruning analogue).
8.3 **Gate:** Local-Support-Learning GMM gate on frozen-base features with a reference Φneg covering every earlier
    phase (V10.3 N7); no smoothing at segment rate (N9).
8.4 **Sleep / REM — dreaming:** dreams are sampled from the corrected model with diversity from the metacontroller
    prior z ~ N(0, I) (the analogue of Sleep's random-expert routing for novelty); keep top-k by gradient-norm
    importance plus random samples (Sleep §3.4).
8.5 **Related skills** (the open V10.3 failure): replay of old-phase dreams anchored to the frozen pre-sleep model
    during consolidation, applied where the gate opens on old-phase inputs.
Status: 8.1–8.3 are V10.3's pipeline, tested at ~0.5M with the numbers above. **New in V12 and untested:** the
on-policy GKD form on self-generated dreams (V10.3 distilled on the labelled newmath stream), z-prior-diversified
dreams (8.4), anchored replay for related skills (8.5). Ordering: dreams use the pre-sleep metacontroller; sleep
changes the base; the metacontroller is refit afterwards (7.6). Not adopted: Sleep's RL imitation reward (its
ablation shows it matters, but it needs a semantic reward model absent at this scale). Design-only in V12 tiny
unless Phase 3 ends by 18:00 (advisor #2).

## Step 9 — Scaling path (frontier MoE class)

Reference points (web, 13 Oct 2026; vendor figures, not verified by us): Kimi K3 — 2.8T total MoE, 16 of 896
experts active, Kimi Delta Attention + Attention Residuals, 1M context, active count undisclosed (third-party
estimates 32–104B disagree). GLM-5 — 744B total / ~40B active, 256 experts (8 routed + 1 shared), DeepSeek Sparse
Attention, 28.5T pretraining tokens. Neither uses an ECD hierarchy; both are flat decoder-only MoE trunks.

Three evidence regimes (every claim below inherits one): **0.5–3M** — our own measurements, mechanism-level;
**1–9B** — the papers, component-level, never combined; **frontier** — no evidence for any V12 component as a
pretraining or post-training change.

**Two separate bets, in this order.** (1) *The metacontroller bet* needs no ECD trunk: it needs a frozen model whose
mid-depth residual carries a decodable latent. It is testable now on an open 1–8B checkpoint without pretraining:
probe mid-depth on math traces labelled by solution step (the Fig. 3 precondition), fit the paper-faithful
metacontroller with the α sweep, read the rate–distortion gap and whether β aligns with step boundaries; if that
holds, internal RL vs GRPO on GSM8K-class verifiable tasks — a rented-GPU experiment, not a pretraining run. If it
fails there, an ECD trunk cannot rescue it. (2) *The ECD + NCP trunk bet* is separate (NCP-Arch's compute-efficiency
evidence); the tiny test informs its objective choices.

How V12 maps onto such a trunk without changing what makes those models work:
| V12 component | At frontier scale | Evidence for it working at scale | Tiny-test status |
|---|---|---|---|
| E / C / D split | E = lower ~40% of layers, C = ~20% at 1/k̄ rate, D = upper ~40%; every FFN MoE; C's experts carry the cheap concept-rate capacity | NCP-Arch 8.9B / 5.73T: 1.74× compute efficiency (gains mostly from C's cheap capacity); the full model only *approaches* the size-matched vanilla at 85% compute — the win is efficiency, not quality | parameter-matched comparison only |
| Learned segmentation | H-Net-style trained boundary detector, ratio loss to k̄ ≈ 4–8 | H-Net (bytes→chunks) at ~1B; not with NCP | untested |
| Learned segments × NCP | variable-length segments change the pooled target's statistics; the shared normaliser/codebook is tested only at fixed k | none | untested (part of why segmentation is the largest risk) |
| NCP-CE over codes | PQ 32×128 (NCP-Arch's codebook size), EMA k-means, per-domain usage logged | NCP-Arch uses MSE-to-codebook-softmax; CE over indices is our change | tested (this run) |
| NextLat on D | λ = 1, d = 1, MLP predictor ≈ 2% params, dropped at inference | NextLat 1.3B/100B, HiLP 1B/100B: small downstream gains, ~83% throughput | secondary arm; may not run in budget |
| HiLP upper level (NCP-dec) | same head, D-latent target | HiLP 1B: lower long-horizon latent error | arm after TF7 (seed 1) |
| Loop in C | second C visit = post-control depth, 1/k̄ cost | none beyond looped-LM literature; every own measurement inert | secondary |
| Metacontroller | post-training on frozen trunk at the C stream; n_z 8–32; internal RL on verifiable tasks. **Scale-blocking engineering:** meta.py materialises per-row A_t, B_t (2·d·r per token: ~230k values/token at d ≈ 7k, r = 16, ~1 TB for a 2M-token batch). Fix: z is piecewise constant between switches, so emit U only at switch events and cache per segment, or use a basis mixture U_t = Σ_j w_j(z_t) A_j B_j; the β recursion becomes a scan over T/k̄ units | ETA paper only at ~3M-param bases on grid/ant; **never run on an LLM** | Phase 2 |
| RBF bank + sleep | bank keyed by C state; consolidation into newly unlocked MoE experts (Sleep "mask then activate") | Sleep paper's evidence for unlocking experts is at 1.7–8B (Qwen3, Llama); our consolidation evidence is gated LoRA at 0.5M (V10.3) — two different claims | V10.3 evidence, not re-run |

Optimiser and stability: inherit the trunk's (Muon/MuonClip-class for Kimi, AdamW for others); auxiliary heads
in their own clip group (V11 lesson); codebook outside the optimiser (EMA only).
What scale cannot fix (from first principles): co-training the metacontroller (collapse), regressing a
multimodal next concept (blur), fixed-k pooling on general text (misalignment). What only scale can test:
whether the C stream carries a decodable, controllable latent on natural reasoning data.

### Read at 4k (11:58), against the pattern above
- **Pre-registered null holds:** ECDe concept stream u: colour 0.167 vs TF7 e_3 0.179 (chance 0.134); pair at chance in
  both (0.256 / 0.274, chance 0.259); parity 0.614 vs 0.635. NCP-enc did not induce the abstraction by 4k.
- TF7's latent is *late*, not mid: parity 0.65 (e_4) → 0.85 (e_5); colour 0.24 and last-visited 0.35 peak at e_5–e_7;
  pair never rises above 0.28. ECDe's best layers are below TF7's late layers everywhere (colour 0.168, parity 0.637).
- Base gap (train stream, steps 3.6–4k): ECDe trails TF7 by 0.17 nats/action on pinpad, 0.38 on text, 0.04 on math —
  the parameter-matched handicap of 2.1 (4 token-rate layers vs 7). Phase-2 cross-base comparisons are confounded
  by competence.
- Gradient share (Decision 3.5, E-only acceptance): λ·NCP/NTP on E = 0.09 text / 0.64 pinpad / 0.30 math — **pass**;
  on C 1.3–2.5× (reported, not judged; C is NCP-dominated); on D exactly 0 (routing verified).
- Concept usage (Decision 2.1, threshold 0.02): pinpad +0.055/action, math +0.034, newmath +0.345 — **used**;
  text +0.008 — **not used**. (Threshold only; the paired interval was not computed.) So the probe null is the
  sharper statement: D reads C, and NCP-enc still did not produce colour/pair abstraction in u.
- Information test (fair protocol: probes fit on train rows, scored on dev; MLP probe = head's shape is primary):
  pinpad u 0.496 vs p 0.502; math 0.657 vs 0.644; text 0.267 vs 0.257. Probe-class noise floor 2–4 points
  (pinpad p2 0.493 linear vs 0.469 MLP): **every u − p difference is inside it**. Copy (0.18–0.24) and majority
  (0.08–0.26) are far below every probe: the codes carry real predictable structure. On math the trained NCP head
  (0.558) is below a fresh probe on its own input (0.644): **the V10.3 failure criterion is met on math**
  (plausibly: math is 15% of the stream and uses ~8 of 32 codes). The first (ordered-halves, linear-only) run of
  the test was discarded: its split put different math modules in train and test.

### Pre-registered 8k reads (written 14:30, before ECD's ckpt_8000 and ECDe's GT controllers land)
Dev NLL so far (pinpad act / text / math): 4k — TF7 0.691 / 3.943 / 1.827; ECD (NTP-only) 0.642 / 3.951 / 1.646;
ECDe 0.878 / 4.319 / 1.953. 8k — TF7 0.538 / 3.555 / 0.817; ECDe 0.579 / 3.893 / 1.120.
1. **NCP cost.** ECDe − ECD dev NLL at 8k per family: ≤ 0.02 on pinpad = transient (the price of C becoming live);
   > 0.1 = persistent at this budget → §6 candidate: an NCP schedule (λ ramp or late anneal), untested, and the
   doc says the C-usage gain was bought with NTP.
2. **ECD's own 8k state.** Either usage stays ≈ 0 and u ≤ p (dead C persists without an objective on it), or usage
   rises without NCP (D discovers C late). Attribution of ECDe's +7.7-point info gain: ECD mlp_u − mlp_p ≥ ~5
   points with usage in ECDe's range → topology; otherwise → NCP.
3. **ECDe GT controllers** (concept = c1, one C block after; dec_in = D's input, two D blocks after): same 3-pair task,
   same 0.03 floor, compared with TF7 L5's 0.031. Expected: no point on either base composes closed-loop; §5.2 leads
   with the precondition (GT pretraining-task success ≈ 0.05 vs the paper's ≈ 0.8).
