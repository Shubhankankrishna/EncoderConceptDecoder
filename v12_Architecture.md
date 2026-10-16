# V12 — Encoder · Concept · Decoder with a frozen-base metacontroller

**Status: DRAFT (design sections written 10:45, 13 Oct 2026; results through 17:30; α sweep completing).** Package: [`v12_tiny/`](v12_tiny/).
Step-by-step design notes: [`v12_tiny/DESIGN_STEPS.md`](v12_tiny/DESIGN_STEPS.md). First-principles notes:
[`v12_tiny/FIRST_PRINCIPLES.md`](v12_tiny/FIRST_PRINCIPLES.md). Advisor consultations:
[`v12_tiny/ADVISOR_LOG.md`](v12_tiny/ADVISOR_LOG.md).

## 0. Summary

**Design (§2).** V12 keeps the encoder · concept · decoder split and makes it an ordinary trunk.
- E is the lower layers and D the upper layers. C sits between them at mid-depth and runs once per segment over
  completed segments only.
- **Next-concept prediction (NCP)** is cross-entropy over a discrete product codebook (4 × 32, EMA k-means). It is
  stopped from reaching D.
- **NextLat** runs on D's own state. **HiLP** applies only at the concept level. The **loop of 2** is confined to C.
- The **ETA metacontroller** (hypernetwork controller, switch β, Gaussian z) is trained *after* pretraining on the
  frozen base, at the control point, with internal RL on z.
- **Continual learning:** V10.3's RBF correction bank and gated-LoRA sleep, unchanged.
- §3 maps the design from 1M to 10T parameters (MoE FFNs in E and D). The metacontroller is 33,393 + 1,088·d parameters at the paper's rank
  and n_z: about 0.2% of a 1B model, under 0.01% from about 100B. The recommended order is to test the metacontroller first, on a frozen open model.

**Test (§4).** About 1.08M parameters on an M2 Air for 12 hours.
- Corpus: 60% pinpad (a gridworld with a ground-truth pair-of-colours subgoal), 25% text, 15% math.
- Three bases at matched step 8,000: TF7 (plain), ECD (NTP only) and ECDe (NTP + NCP at λ = 0.1).
- The paper's evidence chain: probes → ground-truth controllers → metacontroller rate–distortion and switch
  alignment → internal RL (gated).

**What was found (§5).**
1. **NCP is what makes C real.** Without it, D never reads C. With it, removing C costs +0.42 nats/action on
   pinpad and +0.18 on math, and C's state beats its own input by 7.7 points on next-segment codes.
2. **NCP costs.** +0.28 nats on text and +0.27 on math against the same topology. C's gradient is 1.3–2.5×
   NCP-dominated, and 37–64% of codes reassign between 4k and 8k. The NCP arm is also the least competent
   closed-loop (0.16 vs 0.23 and 0.28).
3. **The control chain is blocked by base competence.**
   - The bases complete 16–28% of their own tasks (the paper's base completes about 80%), and the subgoal pair is
     at chance in every probe.
   - Ground-truth controllers carry the label but do not compose. Mid-C was the weakest of four control points.
   - The unsupervised metacontroller learns one held, episode-specific code with no subgoal identity. Its switches
     land worse than random placement (pair F1 0.012 vs chance 0.265).
   - So the RL gate fails and internal RL was not run.

**Decisions (§6).**
- Keep NCP as cross-entropy over codes; schedule λ; slow or freeze the codebook late.
- Choose the control point by measurement.
- Test the metacontroller first on a competent frozen base, then repeat §5.2–5.3 unchanged.
- The loop, learned segmentation, NextLat and HiLP-on-D remain untested (§7).

## 1. What we were doing wrong

Read against the six papers (ETA metacontroller, NCP-ArchPreview, HiLP, NextLat, Explorative Modeling, "Language
Models Need Sleep") and our own V5–V11 results, the failures share four causes. Scale is only one of them.

1. **The test bed had no latent to discover.** Every mechanism in this line (concepts, exploration, a
   metacontroller) pays only where the data has a latent variable that persists over many tokens, cannot be
   copied from a recent token, is multimodal given the prefix, and composes. Text, arithmetic and Sudoku at 0.5M
   parameters expose no such variable to a model that cannot yet model text. V5's GraphNav had one, but it was
   copyable four tokens back, segments lasted about 4 hops, and plain sampling already succeeded 35–89%.
2. **The metacontroller was co-trained or built on fixed chunks.** The ETA paper's controller works only on a
   *frozen* pretrained base. Co-training collapses to one switch (their Fig. 7), and forcing a switch every step
   fails at RL (Fig. 8). V11 trained everything together from scratch, and its bridge pushed 2.6–2.9× NTP's
   gradient into the core. V5 used fixed hold-4 windows; its β saturated (precision 0.25) or under-recalled (0.29).
3. **Next-concept prediction regressed a multimodal target.** Under squared error the best prediction is the
   conditional mean (Explorative Modeling §2.1). V10.1 to V11 regressed the next pooled state: the forecast shrank
   to 3–14% of the target's variance and predicted worse than its own input. An explorative K-mixture found no
   differing futures, because the data had none to find.
4. **Success was measured by NLL.** The paper's payoff is exploration in abstract-action space on sparse-reward
   tasks, where raw-token sampling succeeds about once in a million. Its evidence chain is: probes, then
   ground-truth controllers, then the rate–distortion gap and switch alignment, then internal RL. Our gates asked
   for small NLL gains, which the paper never claims.

What is genuinely scale-dependent:
- NCP-Arch's main lever is cheap concept-rate capacity, which pays only when a model is capacity-limited.
- HRM-Text's gains come from compute per token plus the objective. Response-only scoring and a PrefixLM mask were
  its largest lever.
- Tokens per parameter were not the difference: about 52 for us against about 60 for HRM-Text.

## 2. Architecture

### 2.1 Backbone: the concept module sits at the control point

E → C → D is laid out exactly as in NCP-Arch (16 encoder / 8 concept / 16 decoder layers at 8.9B).
- **E is the lower half of a standard trunk, D is the upper half, and C is inserted at the split.** That split is
  mid-depth. In the ETA paper, linear subgoal decodability rises until mid-depth (layer 4 of 6) and stays strong
  up to the final embedding (Fig. 3), while internal control works best when inserted at mid-depth (Fig. 4; the
  controller is placed at L/2, Table A9). So the concept module's position and the metacontroller's control point
  coincide by construction.
- E is token-rate and causal. C runs at segment rate, causal over *completed* segments, costing 1/k̄ per token: this
  is NCP-Arch's cheap-capacity lever, and at scale its FFNs are MoE experts.
- D is token-rate and causal. It reads released concept states by cross-attention; a segment is released from its
  last token on.
- D reads only u from C. The NCP head is an auxiliary readout of u, not a D input. In V10.1 and V10.3 the soft
  forecast fed to D shrank toward the codebook centroid and was ignored.
- **Usage metric, declared before results:** C counts as *used* on a family if zeroing the concept route costs
  ≥ 0.02 nats/token (per action on pinpad), with a paired interval excluding 0.
- **Cost accounting:** the tiny ECDe is parameter-matched to its TF7 control, not compute-matched (about 4.5
  token-layer equivalents plus two cross-attentions, against 7 layers). At scale the comparison that counts is
  compute-matched.

### 2.2 Segmentation: two boundaries on two clocks

Fixed k-token groups are a tiny-test convenience only. V10.3 and V11 showed period-8 layout dependence on Sudoku,
and pinpad subgoal segments last 1–19 steps, aligned to no k. There are two boundary mechanisms:
- **Segmentation boundary.** Set in pretraining, it marks content change and defines C's clock (mean length k̄). At
  scale it should be an H-Net-style *trained* detector with a ratio loss to k̄ ≈ 4–8. A heuristic is not enough: a
  raw cosine detector would fire on every pinpad token, because action and position tokens alternate.
- **Subgoal boundary β.** Learned in post-training on the frozen base, it spans several segments and defines the
  metacontroller's clock.

Because β is learned, segments need not align with subgoals. **The tiny test (k = 4 fixed) neither supports nor
falsifies dynamic chunking; this is the largest untested item.**

### 2.3 Next-concept prediction: cross-entropy over a discrete segment codebook

- **Target.** τ_{g+1} = sg[EMA-normalised summary of segment g+1], quantised by a product codebook: S = 4 segments ×
  N = 32 words in the tiny test, 32 × 128 at NCP-Arch scale.
- **Codebook updates.** The codebook is updated by EMA k-means with masked dead-code reset, never by gradient. So
  there is no commitment term, no codebook collapse through the optimiser, and no weight decay on it (the V10.3 N3
  fix, generalised).
- **Dead-code threshold.** It compares two EMAs taken across steps and families. A per-batch threshold resets a
  minority family's codes; this was found and fixed in this run.
- **Loss.** L_NCP = Σ_s CE(head_s(u_g), idx_s(τ_{g+1})), with λ = 0.1. The joint over segments is the product of
  per-segment categoricals. This is mode-preserving within a segment without K forward passes, which is the
  principled reason for the discrete codebook the user is bullish on.
- **Gradient path.** The target is stopped. NCP reaches the NCP head, C and E (the context side), never D.
- **Pre-registered checks.**
  - NCP/NTP gradient share on E ≤ 1 (V11's bridge was 2.6–2.9×).
  - Per-family code-usage perplexity → 1 is the collapse monitor.
  - The information test that V10.3 failed: does a readout of u predict the next codes better than a readout of
    C's own input?

### 2.4 NextLat and HiLP: each at the level it was designed for

V11 implemented both wrongly. It used a concept-anchored, 8-step, never re-grounded token-rate rollout, which is
neither NextLat (anchored on the same network's own state, d = 1) nor HiLP (an upper predictor without token
input). V12 places them as follows:
- **NextLat, token rate, on D's own state.**
  - p_ψ([d_t ; E x_{t+1}]) → d̂_{t+1}, trained with SmoothL1 to sg[d_{t+1}] and KL through a frozen view of the
    head; d = 1, λ = 1. The predictor is dropped at inference.
  - L_h runs on every real position: belief states form during context, including prompts and the pinpad header.
    Only the KL term is restricted to scored targets (NextLat §C).
  - States are RMS-normalised before SmoothL1 (inherited from V11).
- **HiLP's upper level, segment rate, = NCP with the decoder-latent target (the user's proposal).** u_g, with no
  token input, predicts the *next segment's decoder latent* sg[mean D states of g+1] as discrete codes. D never
  receives this gradient. The prediction is non-trivial only if the shortcut monitor holds: accuracy against
  targets computed with the concept route zeroed.

  This departs from HiLP deliberately. C is also shaped by NTP through D's read, which is the counter-force
  against collapse. In ECD, D already *is* HiLP's combined head, so no second CE is needed.
- **Not used:** rollouts without re-grounding, and token-space concept losses that reach the trunk.

### 2.5 Loop of 2, in C only

Every prior measurement found visit 2 inert: ≤ 0.011 nats in V10.3, state cosine 0.985–0.993 at 10M, null in V11.
What the loop uniquely buys is **post-control depth at 1/k̄ cost**: the metacontroller writes U between visit 1
(belief) and visit 2, which computes the steered concept. The test is ECDL2e against ECDe under control. It is
confounded (control depth versus looping) unless ECDe is also controlled at C's output. **If the arm is not run,
the loop is carried as an untested hypothesis, with every prior measurement saying inert.**

### 2.6 Explorative modelling: where multimodality lives

- **The next concept.** Covered by the categorical over codes. XM over the concept stream on pinpad is untested,
  because the old "no differing futures" result came from data without a persistent latent.
- **The metacontroller's abstract action.** The VAE stays primary. The ablation is *XM codes on the VAE's
  switches*: freeze the switch unit and fit K = 8 discrete codes by hindsight soft-min per β-segment. A KL-free XM
  metacontroller has no force toward sparse switching and drifts to β = 1. The payoff of discrete codes would be
  internal RL as a categorical bandit per segment.
- **Post-training exploration.** Internal RL samples z at switches. Sleep's dream selection is reverse XM over
  self-generated data.

### 2.7 Metacontroller

**Paper-faithful core** (ETA D.2/E.1, implemented in `v12_tiny/meta.py`):
- Frozen base; the pre-control residual e_t is read at the control point.
- History GRU, 32. Non-causal bi-GRU sequence embedding s(e_{1:T}), 32. Encoder → N(μ_t, Σ_t), n_z = 8.
- Causal switch β_t = f(e_t, h_{t−1}, z_{t−1}). Integration z_t = β_t z̃_t + (1 − β_t) z_{t−1}.
- Hypernetwork → rank-16 U_t, applied as e ← e + U_t e.
- Loss: action NLL + α Σ_t KL(N(μ_t, Σ_t) ‖ N(0, I)), α ∈ {0, 0.05, 0.1, 0.17, 0.3, 0.5, 1}.
- Internal RL: a causal GRU policy over z, β thresholded at 0.5, critic-free PPO on sparse reward.

**V12 changes (each a claim to test):**
1. **Where to control.** The concept stream (segment rate: a k̄× shorter RL horizon, but switch resolution
   coarsened k̄×) is compared with the decoder input (token rate) on the same frozen base; TF's mid-depth is the
   control. Switch F1 is scored on a common environment-step clock (±2 steps; ±1 also reported), against both pair
   and colour switches. Curves are compared across points, never α values.
2. **Timestep.** One environment step (an (o, a) pair), not one token.
3. **Success criteria are the paper's**, gated before RL as pre-registered in `phase2.py`.
4. **ECD-native posterior (ablation).** The acausal embedding reads future *concept codes* instead of raw
   residuals.
5. **At scale.** A post-training module on a frozen LLM trained on clean reasoning traces, with internal RL against
   GRPO on verifiable tasks. LLMs have no observation stream: every token is an action. So the only clock coarser
   than tokens is the segmentation boundary, which is an argument *for* concept-rate control. **Central
   uncertainty:** whether reasoning traces carry a persistent latent at all.
6. **Wake–sleep iteration.** Internal-RL successes become dreams for sleep (§2.8); the metacontroller is refit on
   the updated base. It is never co-trained.

### 2.8 Continual learning

**Inherited and tested at ~0.5M (V10.3):**
- The **RBF correction bank** is keyed by the concept state, writes exact-gradient corrections, and reads only
  through concept routes. With frozen weights it gave −1.55 nats on an unseen math module, with text and Sudoku
  unchanged.
- **Consolidation.** The bank is distilled into fresh gated LoRA with an LSL GMM gate whose reference covers every
  earlier phase, then the bank is reset. This kept 90% of LoRA's gain with zero forgetting on dissimilar families.
- The related add/sub skill stayed unprotected: +0.26 to +0.46.

**New in V12 and untested:**
- Sleep's on-policy knowledge-seeding form, on self-generated dreams.
- Dreams diversified by sampling z from the metacontroller prior (the analogue of Sleep's random-expert routing).
- Replay anchored to the frozen pre-sleep model, for related skills.

**At scale:** a masked, pre-allocated MoE expert is unlocked per sleep (Sleep: "mask, then activate"). Not adopted:
Sleep's RL imitation reward, which needs a semantic reward model.

### 2.9 Gradient routes

| Loss | heads | D | C | E | codebook |
|---|---|---|---|---|---|
| NTP | output head | yes | via D's cross-attention read of u | via h→D and h→pool→C→u→D | — |
| NCP (enc or dec target), λ 0.1 | NCP head | **no** (target stopped) | yes | yes (context side) | EMA only |
| NextLat L_h, L_KL | frozen head view for KL | yes | via D's read | via D | — |
| Metacontroller (post-training) | — | frozen | frozen | frozen | frozen |

## 3. Scaling path to frontier MoE

**Reference points.** These are vendor figures from web sources, 13 Oct 2026, not verified by us. Neither model
uses an ECD hierarchy; both are flat decoder-only MoE trunks.
- **Kimi K3:** 2.8T-total MoE, 16 of 896 experts active, Kimi Delta Attention and Attention Residuals. The active
  count is undisclosed.
- **GLM-5:** 744B total / ~40B active, 256 experts (8 routed + 1 shared), DeepSeek Sparse Attention, 28.5T
  pretraining tokens.

**Three evidence regimes.** Every claim in this section inherits one:
- 0.5–3M: our own measurements, at mechanism level.
- 1–9B: the papers, at component level, never combined.
- Frontier: no evidence for any V12 component.

**Two separate bets, in this order:**
1. **The metacontroller does not need the ECD trunk.** It needs a frozen model whose mid-depth residual carries a
   decodable latent, and it is testable now on an open 1–8B checkpoint without pretraining:
   - probe mid-depth on step-labelled math traces;
   - fit the metacontroller with the α sweep and read the rate–distortion gap and β/step alignment;
   - if those hold, run internal RL against GRPO on GSM8K-class tasks.

   If it fails there, an ECD trunk cannot rescue it. The tiny test strengthens this ordering. On a base that
   completes only 16–28% of its own tasks, every controlled point and the metacontroller fail for the same reason:
   the residual has no subgoal boundary to find. A large pretrained model is where that precondition can hold.
2. **The ECD + NCP trunk** is a separate compute-efficiency bet. NCP-Arch's full model only *approaches* the
   size-matched vanilla, at 85% of the compute.

| Component | At frontier scale | Evidence | Tiny status |
|---|---|---|---|
| E/C/D split | E ≈ lower 40%, C ≈ 20% at 1/k̄, D ≈ upper 40%; MoE FFNs | NCP-Arch 8.9B: 1.74× compute efficiency; efficiency, not quality | parameter-matched only |
| Learned segmentation | trained H-Net-style detector, k̄ ≈ 4–8 | H-Net ~1B, never with NCP | untested |
| Learned segments × NCP | variable-length targets change the normaliser's and codebook's statistics | none | untested |
| NCP-CE | PQ 32 × 128, EMA k-means, per-domain usage | NCP-Arch used MSE to a codebook softmax; CE is our change | tested here |
| NextLat on D | λ 1, d 1, ~2% params, inference-free | NextLat 1.3B, HiLP 1B: small gains, ~83% throughput | secondary arm |
| HiLP upper (NCP-dec) | same head, D target | HiLP 1B | arm after TF7 |
| Loop in C | post-control depth | own measurements inert | secondary |
| Metacontroller | frozen trunk, C stream. **Scale-blocking:** per-row A_t, B_t is 2·d·r values/token (~1 TB per 2M-token batch at d ≈ 7k). Fix: emit U only at switches and cache it per segment, or use a basis mixture Σ_j w_j(z) A_j B_j; make the β recursion a scan | ETA only at ~3M-param bases; **never run on an LLM** | Phase 2 |
| Bank + sleep | bank keyed by C; consolidation into unlocked MoE experts | Sleep at 1.7–8B (expert unlocking) vs ours: gated LoRA at 0.5M — different claims | V10.3 evidence |

**Unaffected by scale:** co-training the metacontroller (it collapses), regressing a multimodal concept (it
blurs), fixed-k pooling on general text (it misaligns).

**Testable only at scale:** whether the C stream carries a decodable, controllable latent on natural reasoning
data.

**Optimisation:** inherit the trunk's optimiser. Auxiliary heads get their own clip group (the V11 lesson), and the
codebook stays outside the optimiser.

## 4. The tiny test (`v12_tiny/`, Apple M2 Air, 12 h)

### 4.1 Data: a mixed corpus with one family that has a ground-truth latent
- **Pinpad (60% of batches).** A token-form gridworld from the ETA paper's Appendix A.1: 7×7 grid, 8 colours, 4
  walls, T = 100, the 16 pretraining tasks of Table A1, and a BFS expert with ε = 0.05 for pretraining and ε = 0
  for the metacontroller split.
  - Stream: `BOS (C_i pos_i)×8 (WALL pos)×4 SEP p_0 a_1 p_1 … END`, scored after the header.
  - There is **no task token**: the subgoal is only inferable from behaviour.
  - Ground truth (pair = abstract action, target colour, target cell) is kept per token for probes and switch F1
    only, never as an input.
  - Subgoal segments last 1–19 steps (pairs average 11 steps, 22 tokens), aligned to no group size.
  - Header cells carry object tokens, matching the paper's observation, which tags each cell by object. Without
    them the pilot's probes stayed at chance, because RoPE models could not bind colour to header slot.
- **Text (25%).** Random 256-token crops of the V10.3-tiny text corpus, read-only, all tokens scored.
- **Math (15%).** The V10.3-tiny arithmetic modules, scored on answer tokens only (response-only: HRM-Text's
  largest lever). The `newmath` module (linear 1-D algebra) is held out for continual learning.
- **Sudoku is excluded.** Its fixed 171-token layout is the known period-alignment confound for fixed groups (the
  user's V9 rule).
- Joint vocabulary 1,090: 66 pinpad tokens plus the 1,024-token BPE. A sealed pinpad test split exists and is
  never opened; all evaluation uses dev splits.
- Why mixed: per the user's instruction (09:20), no single distribution. Pinpad is the only family where the
  mechanisms can be checked against ground truth; text and math test that the hierarchy costs nothing where no
  such latent exists.

### 4.2 Arms and protocol
| Arm | Model | Objective | Dense params |
|---|---|---|---:|
| TF7 | plain causal transformer, 7 layers, d = 128 (size-matched control) | NTP | 1,058,944 |
| ECDe | E 2 · C 2 (k = 4) · D 2 with cross-attention, d = 128 | NTP + 0.1·NCP-CE (encoder target) | 1,075,712 + 66,048 head |
| ECDd | as ECDe | NTP + 0.1·NCP-CE (decoder-latent target, HiLP upper level) — **not run** (budget after the 13:23 OOM); the claim is carried untested, with the shortcut monitor as its specified test | same |
| ECD | as ECDe | NTP only (topology control) | same |
| TFNL / ECDeNL / ECDL2e | NextLat on D; loop of 2 in C | secondary | — |

- **Steps: 12,000 scheduled; every arm is compared at step 8,000** (mid-cosine, equally for all). ECDe ran out of
  MPS memory at step 8,275 while a third GPU process (an 8k evaluation) ran beside two trainings; its last
  checkpoint is 8,000, and resuming was declined (an optimizer-state reset plus 1.3 GPU-hours). TF7 completed 12,000;
  its 12k checkpoint is reported as a single-arm "4k more steps" point. ECD (NTP-only) runs the same 12k schedule
  and is stopped after its 8,000 checkpoint. Rule adopted: never three GPU processes.
- **Optimisation:** family-per-batch on a shared seed, so every arm sees the identical stream (hash
  logged). AdamW (0.9, 0.95), lr 1e-3, warmup 200, cosine to 10%, wd 0.03 on matrices; the trunk is clipped at
  1.0 and auxiliary heads are clipped separately.
- **Batches:** pinpad 128 episodes, text 32 × 256, math 128.
- **Shapes and memory:** lengths are bucketed to multiples of 32, and the MPS memory watermark is capped. An
  uncapped first launch bloated to 16 GB per process and swapped.
- **Phase 2, on frozen bases:**
  - ground-truth controllers at each control point;
  - the metacontroller α sweep (seed 1, 2,000 steps, batch 64) at the primary point (TF: e_5, chosen by the 11:58 rule; ECD: mid-C),
    plus ECD's decoder input at two α values.
- **Metacontroller budget, declared:** 2,000 steps × batch 64 per α point; the paper's Table A9 uses 64,000 × 512
  (1/256 of it). A 6,000-step sensitivity run at α = 0.3 per base is queued last; if it moves pair F1 by > 0.1,
  the sweep is labelled budget-limited.
- **Order and cut (pre-registered 12:15):** all GT controllers first, then α points interleaved across bases in the
  order 0.1, 0.3, 0, 1, 0.05, 0.17, 0.5; a base with < 4 α points started by 17:30 keeps only {0, 0.1, 0.3, 1}.
  *As implemented*, the counter counted only points launched since the driver's 16:46 restart, which happened after
  the OOM reruns. TF had five finished points at 17:30 but only three launched after the restart, so TF α 0.05 and
  0.17 were cut. The rule as written would have kept them. This deviation is recorded rather than rerun: TF's
  curve is already bracketed by its floor (α 0.5 and α 1 both at 0.376 with zero switches) and its information
  regime (α 0 to 0.3).
- **TF control point (pre-registered 11:58):** GT controllers at e_3 (= L/2, the paper's Table A9 placement) and at
  e_5 (where TF7's decodability has risen; the same relative depth as the paper's layer 4 of 6). This replicates
  the paper's decode-vs-control split: e_3 ≥ e_5 on 3-pair GT success = the paper's result holds at this scale;
  e_5 > e_3 = it does not, and the control point should follow decodability. The α sweep runs at the better one
  (e_3 if neither beats the base).
- **Phase 3:** internal RL against raw-action RL on the 3-pair novel composition 0-1-2-3-4-5 (both transitions are
  absent from Table A1). It runs only under the pre-registered gate in `phase2.py`.

### 4.3 Pre-registered rules and the one override
- **Pilot rule (advisor #2):** shrink the world if mid-depth pair probes stayed ≤ 0.30. **Overridden (advisor
  #3).** The diagnostic showed the base was competent: it represents *where* the target is (target-cell probe 0.27
  vs 0.03 chance). The deficit was abstraction, which the paper also finds emerging late (Fig. A1: 16K → 256K
  steps at batch 1024).
- **Concept usage (§2.1), NCP gradient share ≤ 1 (§2.3), RL gate (`phase2.py`):** all written before the
  corresponding result.

### 4.4 Correctness tests (`v12_tiny/tests.py`, 26/26 pass)
- The sealed split raises; 200/200 expert episodes replay to success through the environment.
- Future-token perturbation leaves earlier logits exactly unchanged for all 8 arm configurations.
- ECD release mask: no concept read before the first release, and the token view maps to the latest released
  group.
- Zero-initialised GT controllers and a zero-output hypernetwork are the identity.
- Token-rate control touches only decision rows and the following rows, and decision rows predict actions.
- A ground-truth switcher scores F1 = 1, with 2 pair switches per episode.
- PQ assignment is the nearest codeword and the EMA normaliser is unit-variance.
- NCP gradient reaches E and the head but not D; NextLat's KL reaches the predictor but not the frozen head view.
- The sampler is deterministic, and group labels are read at release rows.
- Internal-RL and raw-RL paths smoke-test on the CPU: the first unit always switches, and no switch is recorded
  after an episode ends.

## 5. Results

All numbers are dev-split, seed 1, from `v12_tiny/runs/`. Tables are regenerated by `v12_tiny/report.py` into
`v12_tiny/RESULTS.md`. Chance levels: pair 0.259, colour 0.134, parity 0.506, last-visited 0.169, target cell 0.029.

### 5.1 Phase 1: the bases (matched step 8,000)

**Competence (train-stream NLL, 400-step windows; ECDe − TF7).** Pinpad action NLL +0.17 at 4k → +0.11 at 6k →
**+0.04 at 8k**. Text +0.38 → +0.33. Math is too noisy in these windows (3–6 batches) to read; dev numbers are
in §5.4. The parameter-matched handicap of §2.1 (4 token-rate layers against 7) shows on text and shrinks on
pinpad.

**Probes by depth: where the subgoal lives.** Values are the best layer; u is the concept stream in token view,
and c1 is ECDe's control point, between the two C blocks.

| latent | TF7 e_3 (declared control point) 4k / 8k | TF7 best layer 4k / 8k | ECDe u 4k / 8k | ECDe c1 8k |
|---|---|---|---|---|
| pair (abstract action) | 0.27 / 0.26 | 0.28 / 0.29 | 0.26 / 0.26 | 0.26 |
| target colour | 0.18 / 0.19 | 0.24 / 0.26 | 0.17 / **0.23** | 0.19 |
| parity in pair | 0.64 / 0.64 | 0.85 / 0.85 | 0.61 / **0.89** | 0.74 |
| last visited colour | 0.28 / 0.29 | 0.34 / 0.38 | 0.28 / 0.34 | 0.29 |
| target cell | 0.20 / 0.20 | 0.26 / 0.26 | 0.17 / **0.43** | 0.21 |

- **At 4k the pre-registered null held:** u ≤ TF7 e_3 on colour and pair.
- **By 8k it reversed against the pre-registered comparator:** u beats TF7 e_3 on colour (0.23 vs 0.19) and parity
  (0.89 vs 0.64). Against TF7's best layer, u is far ahead on *where* (target cell 0.43 vs 0.26, parity 0.89 vs
  0.85) and behind on *which colour* (0.23 vs 0.26, last visited 0.34 vs 0.38).
- **The pair is at chance everywhere,** in both models at every checkpoint. The identity ceiling (last visited ≤
  0.40) bounds it: the models do not yet bind colour to place well enough to name the pair.
- **TF7's decodable latent is late, not mid-depth:** parity jumps 0.65 → 0.85 between e_4 and e_5 of 7. This is
  the same relative depth as the paper's "rises until layer 4 of 6".
- **Inside C the same decode-vs-control split appears:** c1 (the control point) is the least decodable C state.
- TF7 at 12k (single-arm point) improves only slowly: pair 0.30, colour 0.27, last 0.40.

**Concept usage (Δ dev NLL when u is removed; declared metric = zero-ablation, threshold 0.02).**

| ECDe | pinpad (per action) | text | math | newmath (held out) |
|---|---|---|---|---|
| 4k zero / mean | +0.055 / +0.061 | +0.008 / +0.006 | +0.034 / +0.020 | +0.345 / +0.088 |
| 8k zero / mean / shuffle | **+0.42 / +0.46 / +0.65** | +0.029 / +0.013 / +0.023 | **+0.18 / +0.27 / +0.14** | +0.17 / +0.003 / −0.036 |

D reads C heavily on pinpad and math by 8k; every ablation agrees on both. Text sits at the threshold. The
held-out newmath "usage" is an artefact of zero-ablation, a distribution shift for D: mean- and shuffle-ablation
remove it.

**Information test (next-segment codes, MLP probe with the head's shape; linear in brackets).**

| ECDe | u_g | p_g (C's input) | [p_g−1; p_g] | NCP head | copy | majority |
|---|---|---|---|---|---|---|
| pinpad 4k | 0.496 (0.506) | 0.502 (0.497) | 0.469 (0.493) | 0.525 | 0.20 | 0.08 |
| pinpad 8k | **0.576 (0.590)** | 0.499 (0.492) | 0.459 (0.487) | **0.620** | 0.21 | 0.09 |
| math 4k | 0.657 (0.653) | 0.644 (0.599) | 0.647 (0.623) | 0.558 | 0.24 | 0.26 |
| math 8k | 0.645 (0.647) | 0.630 (0.601) | 0.628 (0.622) | 0.654 | 0.28 | 0.29 |
| text 8k | 0.292 (0.314) | 0.289 (0.309) | 0.243 (0.307) | 0.318 | 0.20 | 0.26 |

- Changing the probe class alone moves accuracy 2–4 points; that is the noise floor.
- **Pinpad at 8k is the one clear gain:** u beats its own input by +7.7 points, and the head reaches 0.62.
- Math and text gains stay inside the noise floor.
- **The V10.3 failure criterion** (head < probe on its own input) **was met on math at 4k (0.558 vs 0.644) and
  cleared by 8k (0.654 vs 0.630).**
- **Attribution: it is NCP, not topology.** The ECD control (same topology, NTP only) never uses C:
  - usage at 4k ranges from −0.003 to +0.009; at 8k pinpad it is +0.002 / +0.002 / +0.08 (zero / mean / shuffle);
  - its u predicts ECDe's next codes *worse* than its own input does (pinpad 8k: 0.384 against 0.453).

  Without an objective on C, D routes around it and C is dead weight. NCP is what makes C used and
  informative.
- **That gain is paid for in NTP, and the cost persists at this budget.** Dev NLL at 8k (pinpad action / text /
  math):

  | arm | pinpad action | text | math |
  |---|---|---|---|
  | ECD | 0.551 | 3.615 | 0.846 |
  | ECDe | 0.579 | 3.893 | 1.120 |
  | TF7 | 0.538 | 3.555 | 0.817 |

  ECDe minus ECD is +0.03 on pinpad, +0.28 on text and +0.27 on math. That passes the pre-registered "persistent"
  threshold (> 0.1). At 4k ECD led TF7 on pinpad and math.
- A λ schedule (ramp, or late anneal) is the §6 candidate fix. It is untested.

**Codebook health.** Usage perplexity stays at 29–30 of 32 on every family (no collapse). **Drift is large:**
for fixed dev vectors, the normaliser plus codewords reassign 37% (pinpad) to 64% (math, text) of codes between
4k and 8k. Only one interval was measured, because ECDe has no later checkpoint. The head is learning against a
moving target.

**Gradient share at 4k (λ·NCP / NTP):**
- on E: text 0.09, pinpad 0.64, math 0.30 — the pre-registered ≤ 1 passes;
- on C: 1.3–2.5×, reported rather than judged — C is NCP-dominated;
- on D: exactly 0, so the routing is verified.

### 5.2 Phase 2: ground-truth controllers (Fig. 4 analogue)

**The precondition first.** The paper's base completes its own pretraining tasks at about 0.8 closed-loop. Ours,
uncontrolled, completes some pretraining task in **0.28** of episodes at 8k for TF7 (0.26 for the 3k pilot),
**0.23** for the NTP-only ECD and **0.16** for ECDe. The NCP arm is the least competent closed-loop, although its
pinpad action NLL is within 0.03 nats of ECD's. In this seed, whatever NCP adds on the info test does not reach
behaviour at this competence (training-seed variance of competence is unmeasured). With
ground-truth controllers on the pretraining tasks it reaches only 0.05–0.06. The Fig. 4 question — at which depth
does control compose — therefore cannot be answered at this competence. What the controllers can show is whether
the label information is usable at all.

Bank keyed by pair (G = 4), the paper's subgoal unit for the grid world ("two consecutive colored locations").
Results on 256 rollouts each; the noise floor of 0.03 was set before e_5 was seen.

| base · control point | dev_meta action NLL, base → GT | 12-colour post task, GT / base | 3-pair novel 0-1-2-3-4-5, GT / base | pretraining tasks, GT |
|---|---|---|---|---|
| TF7 8k · e_3 (L/2, declared) | 0.412 → 0.363 | 0.000 / 0.000 | 0.012 / 0.000 | 0.049 |
| TF7 8k · e_5 | 0.412 → 0.359 | 0.000 / 0.000 | **0.031** / 0.000 | 0.057 |
| ECDe 8k · concept (c1, mid-C) | 0.456 → 0.440 | 0.000 / 0.000 | 0.008 / 0.008 | 0.014 |
| ECDe 8k · decoder input | 0.456 → 0.424 | 0.000 / 0.000 | 0.004 / 0.008 | 0.016 |

- The controllers carry the label: action NLL falls 12–13%, plateauing by step 300 of 1,500. But no point composes
  closed-loop. The pair is at chance in every probe, so there is no pair direction in the residual to push.
- By the pre-registered rule e_5 is chosen for the TF α sweep: 8/256 against 0/256 is exactly the floor. All three
  indicators lean the same way, weakly. **The paper's "control best at L/2" is therefore neither replicated nor
  reversed; it is inconclusive at this competence.**
- RL gate (a) passes at e_5 by 0.001, so RL now depends entirely on gate (b): switch F1 ≥ 0.8.

### 5.2b Phase 2: the unsupervised metacontroller (rate–distortion and switch alignment)

The paper-faithful metacontroller (§2.7) was trained on each frozen 8k base: TF at e_5, chosen by the
pre-registered rule, and ECDe at mid-C. Budget: 2,000 steps × 64 per α point, 1/256 of the paper's. Full table:
`v12_tiny/RESULTS.md` (Phase 2b).

Two references make every point readable:

- **Floor.** With the rate at zero, the hypernetwork learns a constant low-rank adapter: NLL ≈ 0.376 on TF
  (base 0.412) and ≈ 0.40 on ECDe (base 0.456; from a rate-0.002 point, pending α = 1). Distortion is read against this floor, not against the base.
- **Chance F1.** Each episode keeps its number of predicted switches, placed uniformly at random over its valid
  units (20 draws, `switch_chance.py`).

[full α table: copy from RESULTS.md at "phase2 done"]

**ECDe at mid-C, α = 0** (17:48): rate 15.7 nats/action, NLL 0.359 (floor ≈ 0.40), 0.24 switches per episode.
Pair F1 is 0.037 against a chance level of 0.081, so below chance. Colour F1 is 0.092 against 0.069, slightly
*above* chance with precision 0.17. This is the only point above chance on any measure, and it needs `z_check.py`
and a second seed before it is read. ECDe α = 1 and the two 6,000-step tails were still running at this writing.

**What the switches do.** At α = 0 on TF, the rate is 28 nats/action. This is the acausal bound: the embedder
sees the whole episode. Pair F1 is 0.012 against a chance level of 0.265, and colour F1 0.109 against 0.290,
both *worse than chance*.
- A locator over 256 episodes shows why. Every episode switches at unit 1 or 2, immediately after the forced
  first unit, then holds one z for the remaining ~30 units.
- The held z carries **no subgoal identity**. A linear probe from z at unit 3 predicts neither the episode's first
  pair (0.24 at α = 0, 0.26 at α = 0.1; chance 0.28) nor its second (0.23 / 0.24; chance 0.27). Cosine between the
  forced z_0 and the held z is 0.73 (α = 0) and 0.50 (α = 0.1) (`z_check.py`).
- So the metacontroller learned an **episode-specific route code**, read once from the acausal embedding. It is
  neither a plan of subgoals nor a sequence of subgoal switches.
- This is the paper's single-switch degenerate mode. Here it arises on a frozen base, because the base's
  residual has no pair boundary to align with (pair probe at chance, §5.1).
- At α = 0.1, 0.32 nats/action buys −0.11 nats below the floor with about one switch per episode. The information
  in z is real, but it is route detail, not subgoal content, and it does not arrive at subgoal times.

### 5.3 Phase 3: internal RL (gated)

Gate (a) passed only at TF7 e_5, and only by 0.001. Gate (b), switch F1 ≥ 0.8 against pair switches, is the
binding condition. On TF, every metacontroller point with switches is *below* the chance level of randomly placed
switches (pair F1 0.012 vs 0.265 at α = 0; 0.009 vs 0.216 at α = 0.3). At α ≥ 0.5 on TF, and at every α tried at mid-C
on ECDe at α ≥ 0.1, β collapses to zero switches. The held z carries no subgoal identity (§5.2b).
Unless a later α point passes, **internal RL is not run.** This is a result about controllability at
competence 0.16–0.28, not a missing experiment. The paper's chain needs a base that executes its own pretraining
tasks, and the residual must carry the subgoal it is asked to switch between. Here the pair is at chance in every
probe, so there is no subgoal boundary for β to find.

## 6. Decisions taken from the results

| # | Decision | Evidence (§5) | Status |
|---|---|---|---|
| D1 | **Keep NCP as cross-entropy over a discrete codebook.** Without an objective on C, D routes around it: the NTP-only ECD control never used C. With NCP, C is used (pinpad +0.42, math +0.18) and informative (pinpad +7.7 points over its input). | 5.1 usage, info test, ECD attribution | adopted |
| D2 | **Schedule λ.** NCP at λ = 0.1 costs +0.28 text and +0.27 math at 8k against the same topology, and C is NCP-dominated (1.3–2.5× NTP's gradient). Candidates: a λ ramp from zero, a late anneal, or a cap on NCP's share of C's gradient. | 5.1 cost, grad share | untested candidate |
| D3 | **Slow or freeze the codebook late in training.** For fixed vectors, 37–64% of codes reassign between 4k and 8k; the head chases a moving target. Candidates: decay the EMA rate on a schedule, or freeze after warmup. | 5.1 drift | untested candidate |
| D4 | **Choose the control point by measurement, not by placement.** Mid-C was the weakest of four controlled points (−3.5% action NLL vs −12% at TF e_5). The design argument "C at the split = control point" stands as a hypothesis; at this competence the best point was a decodable late layer. | 5.2 | open |
| D5 | **The metacontroller chain is blocked by base competence, not by the metacontroller.** Composition fails even with ground-truth controllers at every point, the pair is at chance in every probe, and the base completes 0.28 of its own tasks (paper: about 0.8). The next experiment raises competence first (longer pretraining, cleaner demonstrations), then repeats §5.2–5.3 unchanged. | 5.2, 5.3 | adopted (ordering) |
| D6 | **At scale, test the metacontroller first, on a frozen open model.** Nothing here contradicts the §3 ordering; D5 strengthens it, because a large pretrained model is where the competence precondition can hold. | 3, 5.2 | adopted (ordering) |
| D7 | **Carry the loop, learned segmentation, NextLat and HiLP-on-D as untested.** No arm ran. Earlier measurements say the loop is inert. | 7 | untested |
| D8 | **Never run three GPU processes on the 16 GB M2; pad every MPS batch to fixed shapes.** Two OOM crashes (ECDe at 8,275; 14 of 16 α points) came from a third process and from shape-varying graphs. | 4.2 | adopted (engineering) |

## 7. Untested items and risks (ranked by how much of the design rests on them)

| # | Item | Why it matters | Status | Cheapest decisive test |
|---|---|---|---|---|
| 1 | Do natural reasoning traces carry a persistent, decodable mid-depth latent? | The metacontroller's precondition (ETA Fig. 3) | untested anywhere for LLMs | probes on step-labelled math traces in an open 1–8B model (§3, bet 1) |
| 2 | Learned segmentation (H-Net-style) with NCP | C's clock at scale; fixed k misaligns (V10.3/V11 period-8) | untested; the tiny test uses fixed k = 4 | ~100M ECD with a trained boundary detector vs fixed k, same compute |
| 3 | Metacontroller on an ECD concept stream | The point where the user's two bets meet | **run, blocked by competence:** mid-C was the weakest controlled point; GT controllers do not compose; the metacontroller learns no subgoal switches on either base (§5.2, 5.2b) | repeat on a base that completes its own tasks (≥ 0.8) |
| 4 | Internal RL beats raw RL on a novel composition | The paper's payoff; never reproduced by us | **gated off** (switch F1 far below 0.8, below chance) | `internal_rl.py` vs `raw` on 0-1-2-3-4-5, after #3 passes |
| 5 | Hypernetwork scaling (per-row A_t, B_t) | ~1 TB per 2M-token batch at d ≈ 7k | **half tested:** with binarised β (inference, internal RL) emitting U only at switches is *exactly* equal to the per-row path (max diff 0.0, `switch_emit_check.py`) and stores 13.9× fewer controller floats on a 64-episode batch; soft-β *training* still materialises per row | basis mixture U_t = Σ_j w_j(z_t) A_j B_j trained against the per-row hypernetwork on the same α |
| 6 | NCP-CE over codes induces abstraction beyond context aggregation | The objective the user is bullish on | **partly supported:** on pinpad u beats its own input by 7.7 points (also above the two-step input) and C is used only with NCP; but the pair is still at chance and NCP costs +0.28/+0.27 nats | λ schedule (D2) and codebook freeze (D3) at matched steps |
| 6b | HiLP upper level (NCP with the decoder-latent target) | the user's proposal | **arm not run** | ECDd vs ECDe: probes, info test, shortcut monitor (`acc_vs_concept_off_target`) |
| 7 | NextLat on D makes the control point more controllable | NextLat's evidence is about D's final state, not mid-depth | secondary arm | TFNL vs TF7: mid-depth probes, effective rank, GT-controller success |
| 8 | Loop of 2 earns its compute as post-control depth | Every prior measurement inert | secondary arm (confounded) | ECDL2e vs ECDe controlled at C's output |
| 9 | Discrete XM codes vs Gaussian z for abstract actions | Turns internal RL into a categorical bandit | gated ablation | `meta_xm.py` on a good-α VAE segmentation |
| 10 | Sleep consolidation from self-generated dreams (GKD form; z-prior diversity; anchored replay) | Continual learning without labelled replay | V10.3 tested the labelled-stream version only | V10.3's newmath protocol with dreams instead of the labelled stream |
| 11 | Related-skill protection | V10.3: add/sub +0.26–0.46 under every method | open failure | anchored replay ablation in the protocol of #10 |

Risks that are not "untested" but known:
- **Base competence at tiny scale.** The paper's probe accuracy rises between 16K and 256K steps at batch 1024
  (16M–262M episodes). Our 12K steps with 60% pinpad batches of 128 are ≈ 0.92M pinpad episodes: 1/18 of the
  paper's *earliest* point and 1/285 of its full run. A null in Phase 2 is first of all a statement about this.
- **Metacontroller budget** is 1/256 of the paper's (2,000 × 64 vs 64,000 × 512); see the sensitivity run.
- **Parameter-matched, not compute-matched.** ECDe's NLL against TF7's carries no efficiency claim.
- **One seed** for most comparisons; differences under ~0.02 nats or ~3 accuracy points are not read.

## 8. Review

This section records what the review *changed* and what each check does and does not cover. It is not a summary.

### 8.1 Inconsistencies found and fixed
| Where | Was | Now | Why |
|---|---|---|---|
| §7 row 6 | u beats [p_g−1; p_g] by 11.7 points | 7.7 points (u vs its own input), as in §5.1 and §6 | the 11.7 used the two-step MLP probe, which scored below its own linear version: the weakest baseline, not the fair one |
| §4.2 | Phase-2 TF point "mid-depth" | e_5, chosen by the 11:58 rule | the sweep ran at e_5 |
| §5.2b | ECDe floor ≈ 0.40 | labelled "from a rate-0.002 point, pending α = 1" | a floor needs a rate-0 point |
| §5.3 | β collapses "at every α" on ECDe | "at α ≥ 0.1" | at α = 0 there is no KL penalty to collapse β |
| §0 | metacontroller "under 0.001% of the model" | 0.2% at 1B, < 0.01% from ~100B | from the exact fit 33,393 + 1,088·d |
| §5.2b | "task-level code" (inferred from switch times) | episode route code with no subgoal identity | measured by `z_check.py`: first- and second-pair probes from the held z are at chance |
| §5.2 | competence for TF7 only | all three bases, hedged as one seed | a three-base comparison needs three numbers |
| §5.2b, §5.3 | "F1 near zero" | F1 against a chance level of random switch placement | a near-zero F1 cannot be read without the chance level; it turned out to be below chance |

### 8.2 What each check covers, and what it does not
- **One training seed per base.** Differences under ~0.02 nats or ~3 accuracy points are not read. The competence
  ordering (0.28 / 0.23 / 0.16) and the mid-C-is-weakest finding are single-seed.
- **Parameter-matched, not compute-matched.** ECDe's NLL against TF7's carries no efficiency claim.
- **Chance F1 is defined only when switches > 0.** Zero-switch points show "—".
- **Rollout padding** to a multiple of 32 matches unpadded logits within 3.5e-5. `internal_rl.py` is unpadded
  and was not run.
- **Metacontroller budget** is 1/256 of the paper's. The 6,000-step tails test sensitivity to a 3× budget, not to
  the full budget.
- **Info test:** probes are fit on train rows and scored on dev rows. Linear and MLP probes are reported, and the
  stronger one is used for each comparison.
- **Usage ablations** (zero, mean, shuffle) agree on pinpad. On newmath they disagree, so newmath is not read.

### 8.3 Claims that rest on one number
- "NCP makes C real": the ECD control's usage of C (≤ 0.024 nats) against ECDe's (+0.42 pinpad). Two runs.
- "Mid-C is the weakest control point": −3.5% vs −12% action NLL, one seed, one α.
- "The metacontroller learned no subgoal": TF α ∈ {0, 0.1} z-probes, plus the switch-time locator at α = 0.

### 8.4 Process
Advisor consultations: [count] in `v12_tiny/ADVISOR_LOG.md` (main) and [count] in
`v12_tiny/viz/VIZ_ADVISOR_LOG.md` (explorer). The user asked for more than 40 and more than 20. The counts are
reported as they stand. Each entry records advice that changed something, and none was added to reach a number.
