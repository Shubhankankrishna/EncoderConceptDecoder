# Concept module v2 — five designs (filled after the reading)

## Scoring rubric (fixed 2026-09-28 before reading, so scores are not post-hoc)

Each criterion 0–10, weighted to 100. Baseline ConceptCoreV2 (1 shared LoopLayer, R ~ U{1..4}) is row 0.

| # | criterion | weight | what earns a high score |
|---|---|---|---|
| C1 | Paper evidence that extra passes add depth on more-steps / OOD tasks | 20 | a mechanism that at least one paper shows extrapolating in steps with numbers; FULL C1 ONLY IF the intermediate passes have a GROUNDED target (per-pass NCP, decodable TRM-y, or a task intermediate) — ungrounded latent passes drift/homogenise (Coconut collapse at scale per LOTUS, our P3 rule, our own 0.1° stagnation). Grounding forms available without gold intermediates (LoopWM uses sg(E(o_k))): the stop-grad target for pass k is either the NCP target code (discrete) or sg(final-pass plan) (ELT distillation form) — a design must name which |
| C2 | Training signal that makes R matter | 20 | pass k+1 is forced to differ from pass k (per-pass loss, step-conditioned targets, anti-collapse) instead of the current R-invariance |
| C3 | Serial-composition capacity per pass | 15 | distinct layers / effective depth for hop, bind, carry |
| C4 | Plan capacity and NCP/PQ fit at d=256 | 10 | u keeps > 1 answer token; codebook layout divides 256; NCP head unchanged in spirit |
| C5 | Language-path safety | 10 | decoder gets a stable, usable plan at any R; NTP not hurt |
| C6 | Stability at 25M on MPS | 10 | no exotic ops, bounded state norm, healthy gradients through the loop |
| C7 | Compute budget | 10 | MACs/token at ¼ rate and params within ~4 × 1.12M + small overhead. Loopie's compute-matched critique: every core MAC could instead buy encoder/decoder width; the ¼ rate gives a 4× better exchange rate than a whole-model loop, but each design states core MACs/token and its share of the ~18 M total — a share > ~25 % of total drops C7 to ≤ 4/10. Exchange rate (Schwethelm via SMELT/LoopCoder, dense, iso-depth): r recurrences ≈ r^0.46 unique-block equivalents — a 4-layer core cycled R times ≈ 4·R^0.46 unique layers in LM-loss terms (R=4 → 7.6, R=8 → 10.4) for 4R layer-passes; loops are poor LM-loss value per FLOP WHEN THEY DISPLACE UNIQUE PARAMS. When params are held fixed (our case: the core keeps ~4×1.12M and adds passes) SMELT finds a net win (6.8–18 % compute) — but concentrated on composition / ICL / long samples, which is where our synthetic tasks live. Same conclusion (justify the core by composition), without over-discounting C7 |
| C8 | Implementability | 5 | fits `ConceptCoreV2.forward(c_hat, cdoc, R) -> u, logits, c_pred` and the existing trainer |

Constraint stated once (AMENDED during the design phase, advisor #24): the core runs at d 256 but c_hat and the decoder stay at
384, so every design uses in/out projections (384 ↔ 256). The PQ codebook lives in c_hat space and may therefore STAY 48 × 128 × 8
(NCP head W_q 256 → 384); a design may still choose a lower-bitrate layout (32 × 8, 64 × 4, 16 × 16 in a 256-d code space) and must
justify it under C4. Head layout 4 × 64 or 8 × 32. No warm start of core weights from the current d 384 checkpoints.

## Baseline audit (defects every design inherits the fix for)

1. **Not identity-at-init (C6 #1).** `LoopLayer`: `GRWrite` s = 2σ(0) = 1 and `GatedAttention.o`, `SwiGLU.w2` use PyTorch default
   init ⇒ pass 1 is a full-strength random residual write into S (S0 = 0). Loop, Think & Generalize zero-inits the attention and FFN
   output projections (identity block at init) and shows default init gives seed-dependent, non-scaling, overthinking loops
   (App. E); LoopFormer's AdaLN-Zero is the same idea. Fix: zero-init `o` and `w2` in every concept layer (keep s = 1; s·0 = 0).
   Together with the R distribution this is a plausible mechanistic cause of the measured stagnation (step angle → 0.1°, R1..R8 equal).
2. **Train R too small (C2).** R ~ U{1..4} (mean 2.5, max 4). Loop-Think: inference-time scaling appears only when train R > 4 and
   the max train R sets the extrapolation range; dynamic Poisson R with Rmin 2 beats fixed R.
3. **Loss placement rewards R-invariance (C2).** Full NCP/NTP loss on whichever R was sampled, equal weight. LoopFormer puts the main
   loss on R_max and only a 0.1-weighted, distilled loss on shorter routes.
4. **Unbounded state norm (C6 only).** S grows ≈ linearly (state_norm 9.77 → 18.49 → 27.07, +8.5/pass) — the recurrent path has no
   norm. HRM-Text MagicNorm (exit norm on the recurrent path, PreNorm inside) is the cheap bounded-state fix. NOT a cause of the
   R-invariance: GRRead already normalises what the layer reads, so a state growing along a fixed direction looks the same every
   pass; ‖S‖ growth is a consequence of the same update repeating. The mechanism of R-invariance remains C2 (loss placement + R_max).

## Cross-paper synthesis (running; one line per paper, read directly by the design phase)

**C2 axis — what makes R matter:**
- LoopFormer: main loss on R_max; short routes (S ~ U{1..L−1}) 0.1-weighted + distilled toward sg(h_L) → no stagnation
  (low cross-step CKA), budget elasticity; not step extrapolation.
- Loop-Think-Generalize: loss at the final pass only; Poisson R (Rmin 2, Rmax 8–16); zero-init outputs → depth
  extrapolation, but only when train R > 4; max train R sets the range. Parametric hops — transfer to in-context hops is partial.
- GRT: r ~ U{1..R}, loss at the sampled depth (= our baseline recipe) → early-exit robustness, fixed point by step 4,
  slightly WORSE beyond train R (6 → 8/10/12: 2.69/2.72/2.74). This is exactly the baseline's measured behaviour.
- C2 MECHANISM #3 — DEEP SUPERVISION (TRM), defined strictly: (a) each supervised group starts from a DETACHED carried state,
  (b) full BPTT WITHIN the group (TRM: truncating inside a group did not help; 1-step gradient cost 31 pts), (c) optional
  T−1 no-grad warm groups before the graded one. Evidence it breaks R-invariance: PTRM — running 3× more steps than trained
  IMPROVES (+3 pp), where GRT's uniform-r recipe degrades beyond train R. Evidence that plain per-iteration loss WITHOUT the
  detach does not help in LM: GRT Table 8 — Ouro (per-iteration loss) 3.19 vs GRT 3.14. A design scores on C2-#3 only if it
  names the detach boundary and the within-group gradient horizon.
- HRM-Text is its own recipe: single final loss (no deep supervision) + distinct H/L modules + TBPTT horizon warm-up K 2 → 5.
- ELT (vision): fixed L_max with full GT loss there EVERY batch + prefix students L_int ~ U(L_min, L_max) with λ·GT + (1−λ)·KL to
  sg(teacher), λ 1 → 0 → smooth elasticity AND modest extrapolation (trained L_max 4, best at L 6). Without ILSD: collapse off L_max.
- KEY DISTINCTION: the failure mode is not intermediate supervision per se — it is intermediate supervision that REPLACES the
  max-depth loss on that batch (GRT/baseline uniform-r). LoopFormer and ELT keep the full loss at R_max every batch.
- C2 COST HIERARCHY: LoopFormer = separate shortcut trajectory (~1.5× training FLOPs); ELT = prefix student (zero extra core
  passes, one extra head eval). For our chunk-rate NCP head the ELT form is essentially free → default for any asymmetric-loss
  design unless it needs LoopFormer's Δt conditioning.
- FIXED-POINT CONVERGENCE under fixed-R final-loss training — three independent sightings, same failure, different diagnostics:
  GRT (CKA ≥ 0.907 across steps 2–6, committed by step 4), Ouro (step agreement ≈ 100 % beyond T = 4), LoopCoder-v2 (effective rank
  peaks at loop 2 then narrows, successive updates anti-aligned cos θ −0.72/−0.46, middle loop near-dead). Our baseline's 0.1°
  stagnation is the fourth. Strengthens the rule; adds no new mechanism.
- ⇒ C2 is decided by which R carries the primary loss and how large R_max is, not by the gate architecture.

**C2 TAXONOMY (six recipes side by side — the scoring rule for C1/C2):**

| recipe | loss placement | detach | R_max trained | beyond R_train |
|---|---|---|---|---|
| GRT / our baseline | uniform r, loss only at the sampled depth | no | 6 / 4 | degrades (2.68 → 2.74) / flat |
| Ouro | every step, weighted by learned q (deep-heavy) | no | 4 (8 unstable) | degrades (MMLU 67.5 → 64.5) |
| LoopFormer | full at R_max + 0.1·short route + 0.1·distill | no | 8–24 | untested (elastic only) |
| ELT | full at L_max + prefix λ·GT + (1−λ)·distill, λ 1 → 0 | no | 4 | modest gain (best at 6) |
| Loop-Think-Generalize | final pass only, Poisson R | no | 8–16 | extrapolates iff train R > 4 |
| TRM / PTRM | per-group loss, carried state DETACHED, full BPTT in group | yes | N_sup 16 | +3 pp at 3× depth |
| LOTUS | post-loop loss (+ λ_step 0.05 grounded step loss), fixed R | no | 6 | dip (70.0 → 69.3 at R = 7) |

RULE: additive per-step loss alone (Ouro) removes the collapse but does not buy extrapolation. Extrapolation appears with
(i) a LARGE trained R_max (Loop-Think), (ii) a DETACHED carry that forces an improvement operator (TRM/PTRM), or (iii)
DISTILLATION toward the deep pass (ELT, modest). A design earns C1/C2 points by naming which of (i)–(iii) it uses.

INTERMEDIATE-SUPERVISION WEIGHTING (revised after advisor #21): the discriminating variable is the CARRY, not the weight.
  • WITH detached-carry groups / no-grad warm passes: EQUAL-weight per-pass loss is evidenced and extrapolates (URM: equal CE on
    every graded loop t = N+1..M; Diffusion-Timestep: equal MSE on all 4 steps of each window, then detach → K 160 → 10,000; TRM/GRAM
    deep supervision).
  • WITHOUT a detached carry (one BPTT graph over all passes): keep intermediate supervision LIGHT or DECAYING — ELT λ 1 → 0,
    LoopFormer weight 0.1, LOTUS per-iteration supervision on the same head HURT (68.2 vs 70.0 post-loop).
  A design must say which regime it is in; per-pass loss is not under-weighted "by default".

SHALLOW-PLAN FLOOR (reconciles Ouro's "exploitation" and Loop-Think's final-only loss): use ELT's schedule — shallow GT loss
weight λ decays 1 → 0 but distillation-to-sg(deep pass) remains, so the R=1 plan stays usable for the decoder (C5) without being
an independent target that rewards R-invariance.
 A design whose
  only C2 argument is "a better gate" scores low on C2 regardless of C6.

**C3 axis — distinct layers vs shared block:**
- GRT: shared-block representations across steps have CKA ≥ 0.907 (steps 2–6) and all align with dense layer ≈ 11 of 36;
  head roles are fixed by weight ("the same program each step"). Recurrence of one shared block compresses the path but does
  not build the deep hierarchy ⇒ paper-backed support for 4 physically distinct layers inside each pass.
- Loop-Think: learnable hop depth grows with L×R, but only with enough training; L=4 distinct layers per block in all runs.
- SMELT mechanism (closest analogue to our in-context hop/bind tasks): visit 2 keeps Q/K ("where to look", cos 0.89–0.93) but reads
  new V, writes larger aligned updates, and collapses the attention sink onto demonstration answers (BOS 0.60 → 0.02, answers
  0.24 → 0.85) — C1 support for passes doing in-context retrieval work.
- ELT: 1N × 32L = FID 10.30 vs 8N × 4L = 3.16 at equal effective depth — block width (distinct layers per pass) matters more
  than loop count.
- MoR: "recursive capacity bottleneck" — MoR is WORSE than vanilla at 135M and only catches up ≥ 360M; at our size sharing costs
  capacity → distinct layers are the safe choice.
- MODULATION: per-recursion LoRA/DoRA/prefix from scratch gave no gain (MoR Table 16). Passes that should differ need state-
  conditioned modulation (GRT gate, LoopFormer AdaLN on (t, Δ)) or distinct layers — not per-pass low-rank adapters.
- URM (ARC, data-scarce): at equal FLOPs a 4-layer block looped 8× beats 32 unique layers (40.0 vs 23.75 pass@1); 4 distinct layers
  per loop beat 2 (40.0 vs 36.25) — "4 distinct layers AS A LOOPED BLOCK", caveat: LM regime values unique params more.
- Serial-horizon arithmetic (T²MLR + Merrill/Li): state tracking of N steps needs ~N serial steps, or ~log2 N layers if attention
  learns a parallel scan. state_machine OOD = 8–12 steps: baseline 1 shared layer × R ≤ 4 cannot reach it serially and has ~2
  layers of scan depth; 4 distinct layers × R = 8 = 32 effective layers ≥ log2 12 ≈ 4 — feasible in principle.

**C3/C4 axis — role split of the 4 layers (they should not be four copies of one role):**
- TRM: decodable plan y vs latent scratch z (single-z −15.5 pts); the NCP head reads y; x enters only the z-update.
- HRM-Text: distinct H (slow) / L (fast) modules; shared H/L (TRM-1B) unstable at scale → a 2+2 split with an H2L3-style schedule.

**C4 axis — plan capacity / width:**
- LOTUS: c = 1 latent slot per step → 49.7 vs c = 25 → 70.0 — per-step latent WIDTH matters as much as depth; mirrors our measured
  limit (u carries ≈ 1 answer token: 0.59 → 0.21 → 0.06). Multi-slot concept (m slots per chunk) is a design option, but LOTUS's
  slots are supervised CoT-step slots, not a plan vector — transfer is suggestive; a design claiming it must cost the decoder read
  of m slots and the m·256 attention positions.

**Settled architectural constraint — MIDDLE-SPAN, NOT FULL-STACK (third-time result):** Huginn prelude-recur-coda, Saunshi middle
third, SMELT 50 % span optimum (1.9257 vs full-stack 1.9322 vs none 1.9445), MoR Middle-Cycle, T²MLR middle block > full recurrence,
GRT prelude/coda. Our macro layout (unique encoder → looped concept core → unique decoder) already is this; no design loops into the
encoder or decoder.

**Schedule axis (decided once for all designs):**
- Evidence is horizon-dependent: MoR Sequence (layer-repeat) worst at 10B tokens; Loopie model-loop ahead until ~1.2T tokens, then
  layer-loop wins. At our ~0.2B-token horizon the DEFAULT = BLOCK CYCLE (4 distinct layers as one block, cycled R times).
  Heterogeneous schedules (HRM-Text H2L3-style, TRM z-then-y) are variants a design may claim explicitly; layer-loop (L1 L1 L2 L2)
  is out at this horizon.

**C6 axis — init / stability:**
- Loop-Think: zero-init attention + FFN output projections (identity block) is essential; default init → seed-dependent.
- LoopFormer: AdaLN-Zero (identity at init) + 4k warm-up for stability at depth.
- GRT: gate bias +4 (copy ≈ 0.98 at init) — also identity-like at init; state + gate noise σ 0.1 help (−0.018 nats each).
- BPTT horizon tension: TRM (7M, puzzles) needs full BPTT through its 7 evaluations; HRM-Text (1B, LM) sees 35× gradient spikes
  under full BPTT over 8 steps, K = 5 deepest stable. Our regime is nearer HRM-Text. Reconciliation both support: full BPTT inside a
  group of ~4 passes, detach between groups, warm the horizon up.
- MagicNorm exit norm on the recurrent path bounds state variance over many passes.
- ALTERNATIVE BOUNDED STATE (LoopWM citing Parcae; Parcae not in folder; no ablation seen): diagonal contractive retention
  S ← Ā⊙S + (1−Ā)⊙update, Ā = exp(Δ·diag(−e^a)) ∈ (0,1) learned per channel — bounds ‖S‖ for ANY R by construction, which matters for
  the eval R-sweep beyond R_train where an exit RMSNorm is silent about drift direction.
- C6 RULE — SCALE THE LOOPED BRANCH DOWN BY ~√N (DeepLoop + Loop-Think + Ouro): a looped block's update is the sum of R aligned
  visit gradients read back by R visits → any design with an exit norm on the recurrent path (audit #4 fix, MagicNorm-like, i.e. a
  post-norm on the state — exactly where DeepLoop's α applies) must also use the loop-aware exponent: skip scale α = (2N)^{1/2},
  branch init β = (8N)^{-1/2} (= 1/16 at N = 4 layers × R 8 = 32), or zero-init (the β → 0 limit, Loop-Think); and a LOWER LR for the
  looped core (Ouro: "recurrent architectures require smaller LR"). DeepLoop p-sweep: p ≤ 0.40 all seeds diverge at R = 3.
- Fifth datum: SMELT scales every sublayer's residual update inside the looped span by 1/r.
- Fourth "scale the branch down" datum: Loopie SmallInit std (2.5d)^{-1/2} on block matrices (with DeepLoop β, Loop-Think zero-init,
  GRT gate bias +4).
- SECONDARY POINTERS (not in the folder, UNVERIFIED — cited only as Loopie §7.3–7.4 summaries): Kuo et al. "stochastic stopping
  improves extrapolation across unseen depths" (supports dynamic R); Rauba et al. "benefit depends on where compute is placed rather
  than recursion alone"; Moosa et al. "learned loop allocation correlates with difficulty without extrapolating to longer inputs".

**C5 axis — language loss:**
- T²MLR Table 2 (135M/10B): 2× full loop 42.99 and 3× middle loop 42.68 do not beat the plain Transformer 42.83 on zero-shot
  NLP; a training-compute-matched Transformer wins outright (45.30). GRT isoFLOP gaps agree (dense ahead at medium/large).
  ⇒ depth loops at this scale buy reasoning, not language loss: no design's C5 score may assume a val-loss gain from the loop.
- Ouro (1.4B): Mano at L = 24 ops — loop 3⊗4 = 92.2 % vs iso-FLOP unshared 12⊗1 = 34.8 % (loops win decisively on COMPOSITION);
  scaling-law appendix — at equal params × steps the unshared model always wins on LM benchmarks, gap grows with steps.
- PLT (~1B): loop-2 +5.0 avg / −0.045 loss when compute is not matched — same pattern.
- SMELT (MoE, FLOPs/params/KV all matched): looping wins 6.8–18 % compute, but the gain concentrates on long samples (1.52× tilt),
  in-context learning (Dyck k = 32: 29.8 vs 26.4 %) and code — composition/retrieval, not generic LM loss.
- MoR iso-param (118M, 10B tokens): recursion 1 → 2 → 3 valid NLL 2.968 → 2.921 → 2.911 — loops help at iso-param, unshared wins
  at iso-FLOP (same pattern).

**Ingredients any design may claim:**
- EVAL-TIME EXACT CARRY (from T²MLR): train with R parallel passes (= Jacobi iterations), but for the OOD step-extrapolation
  evals run the concept core chunk-by-chunk with an exact sequential carry → unbounded horizon exactly where step
  extrapolation is measured, at zero training cost (eval is off the training path).
- Zero-init output projections / identity-at-init (Loop-Think, LoopFormer, GRT bias +4, T²MLR γ = 0).
- PER-SEQUENCE R SAMPLING (LoopWM: "eliminates most loss spikes", unverified) instead of one R per batch — free; Loop-Think and Ouro
  sample per batch, so none of them tested it.
- LOOPED-MoE FFN (Sparse-Layers-Looped, user-added): replace the core's dense SwiGLU by top-k MoE (E 8 × d_ff 512, k 2 → active = dense
  1024; or E 4 × 512 for 2× storage), LB loss + router z-loss. State-conditioned routing diverges across passes (25–53 % of tokens fully
  disjoint on pass 2) → per-pass computation without a pass-index embedding, defined for any R. LM-regime iso-FLOP: dense looped 36.2 ppl vs
  Base 34.8; Looped-MoE 4×4 35.4 (best shape) / 8×2 35.9. Caveats: part of the gain is stored capacity; no R-extrapolation test.
- SHARED EXPERT POOL (Intern-S2-Mobius, weak evidence): one MoE pool shared by all 4 core layers and all passes, attention per layer
  kept distinct (16 × 512, k 2 → 6.3M stored vs 12.6M for 4 × 8; active FFN 1024).
- PER-PASS READOUT IS FREE (Sparse-Layers-Looped): loop boundaries converge toward the final distribution by construction (same exit
  layers) → per-pass NCP/logit-lens is well-formed without extra training.
- DEPTH-KV ACROSS PASSES (MoDA): joint softmax over sequence K/V and the same chunk position's earlier-layer/earlier-pass K/V (reuse
  the sequence K/V: 0 params, tiny cost at 256 chunk positions). Counters dilution of S by repeated GR writes. Use a BOUNDED slot
  buffer (last S passes) so R > R_train keeps the key count fixed. Pairs with post-norm (MoDA: post-norm + depth KV beats pre-norm at
  depth; at our width 3.4062 → 3.3484 vs pre-norm 3.3800 → 3.3656). Non-looped evidence.
- SMT UNIFORMITY LOSS (λ ≈ 1e-3) on the looped state against slot/rank collapse — monitor with a rank metric, not loss.
- TIMESTEP-FREE DETACHED-GROUP RECIPE (Diffusion-Timestep): no pass index; carried state with its OWN norm (h = LN_h(z)) and a separately
  normed readout; per-pass loss; BPTT over groups of ~4 passes with detached carry (L = 1 fails, L = 16 vanishes, L = 4 best); random
  pass count per sample → trained K ≤ 160, improves monotonically to K = 10,000. CAUTION: on under-determined (multimodal) targets
  trajectories settle then wander (57.93 %) — our NCP target is multimodal, so R-extrapolation must be measured, not assumed.
- RANDOM DEPTH + RANDOM STEP SCALE (EBT): randomising the number of refinement steps AND the step size was the most important
  ingredient for gains that grow with more steps (removing random step size "nearly eliminates" them) → keep per-sample random R and
  add a random per-pass residual gain on the loop update. An explicit scalar energy/verifier head on S could give halting and
  best-of-R selection (EBT views looped models as implicit EBMs that cannot verify). Full EBT training too costly (≥ 3.33×/step).
- PASS-CONDITIONING RULE (fixed after advisor #23 — an evidence statement, not a prohibition): pass conditioning is DEFAULT-OFF for
  designs that claim more-steps extrapolation (D1, D3, D4, D5), because it is unevidenced-to-help there (Diffusion-Timestep: time
  conditioning had "no significant effect"; Survey: Recurrent-Depth "tried, dropped") and adds a parameter path whose inputs are not
  in-range for R > R_train; state-only per-pass variation is the evidenced route. D2 may use LoopFormer's normalised (t, Δ): its
  random shortcut schedules expose small Δ in training, so the conditioning stays IN RANGE; what is untested is TRAJECTORY LENGTH
  R > L ("more loops beyond L not tested"). D2 evaluates R > L with the uniform schedule Δ = 1/R so its OOD more-steps result is
  interpretable.
- DELTA-RULE WRITE (Survey: DeltaNet = one gradient step on ½‖Sk − v‖²): S ← S + β(v − read(S)) instead of additive gated write →
  bounded state + improvement-operator reading of each pass; candidate for the user's "delta attention residual" slot.
- X-SIDE K/V REUSE ACROSS PASSES (DeepSeek-V4.1 CSA2 logic): c_hat does not change across passes → compute K_X, V_X once for all R
  passes (exact); recompute Q and S-side K/V per pass. Cuts core read cost ~ (R−1)/R on the X side. C7 ingredient.
- mHC RESIDUAL (DeepSeek-V4.1): n residual streams mixed by a doubly-stochastic (Sinkhorn) matrix → spectral norm ≤ 1, a non-expansive
  carry for loops; n = 2 ↔ TRM (y, z). Residual-slot candidate next to Qwen gated residual / delta residual.
- STOCHASTIC GUIDANCE (GRAM): S_r = u_r + ε_r with prior ε ~ N(μ_θ(u), σ_θ²(u)) and, in training, a posterior q(ε | u, next-chunk
  PQ code n[:,1:]) + KL (β ~0.05–0.1, KL-balance 0.8, free bits). NOT cdoc (= document id, masking only). The posterior reads future
  tokens → Free-Transformer-style leak bounded only by the KL; train/infer mismatch is a C5 risk. Principled replacement for naive randomness (random z_0 / stochastic decoding do NOT help
  TRM; σ = 0 target-conditioned guidance → 0 %, a leak). Fits our multimodal next-chunk target (deterministic recursion collapses to
  one attractor on multi-solution tasks). Eval: prior mean — UNTESTED assumption (GRAM always samples). Noise on the slow state only. C3/C4 (coverage) add-on,
  not a C2 mechanism and not a fix for the R-invariance.
- TBPTL (URM, numbers): with 8 passes, the first 2 forward-only (no grad) and passes 3–8 graded with a per-pass loss is best
  (39.13 vs 36.25 full BPTT, 22.50 with only the last pass graded).
- CONVSWIGLU (URM): short depthwise conv (kernel 2) AFTER the FFN expansion gives the dominant gain (45.3 → 53.8 on ARC-1); inside
  the attention path (after Q/K/V) it is useless or harmful. Our core currently convolves q/k/v over chunk positions → a design may
  move the short conv into the FFN (causal over chunks). Caveats: ARC regime; AXIS TRANSFER (tokens → chunk positions) —
  conditional ingredient only.
- SHARED Q/K, PER-PASS V (from SMELT's observation that visit 2 keeps Q/K — cos 0.89–0.93 — and changes V): share the Q/K projections
  across passes and give each pass its own V (or a per-pass V delta); ~2/3 fewer per-pass attention-projection params than fully
  distinct attention. Suggestive only (observed in a trained shared-weight model, never tested as a constraint); differs from MoR's
  failed uniform per-pass LoRA on Q/V.
- EVAL ADDITIONS for any v2 run: R-sweep beyond train max; per-pass logit-lens KL to the final output (HRM-Text Fig. 5 —
  "depth is being used"); per-pass NCP accuracy; OOD more-steps families at R = 4/8/12/16; STEP-AGREEMENT MATRIX A[i,j]
  (Ouro: fraction of NCP top-1 / decoded answers unchanged between passes i and j — the baseline's R-invariance predicts ≈ 100 %);
  per-pass EFFECTIVE RANK and successive-update alignment cos θ (LoopCoder-v2: negative cos + falling rank = oscillating useless
  passes; our baseline's ≈ +1 alignment at 0.1° = stagnation — both are failure signatures); CHUNK-0 ATTENTION MASS PER PASS (SMELT's
  sink mechanism — our GatedAttention's sigmoid gate may already remove the sink; check offline on main's checkpoint when the GPU
  is free).

**Anchor / injection:**
- GRT: fixed anchor re-injected every pass worth 0.70 nats over a drifting anchor (zeros anchor catastrophic).
- Loop-Think: no injection; injection does not cure overthinking in implicit reasoning.

**Rejected / orthogonal axes (considered, not re-litigated later):**
- Temporal recurrence over chunk positions (RLT: decoder final state s_{t−1} fed into token t): strictly serial across the
  256 chunk positions in training (256 × 4 layers per sequence), breaks the parallel causal attention of the core, and the
  report has no measured results. Rejected for a throughput-bound 24 h MPS run.
- UPDATE after T²MLR: the temporal axis IS viable in parallel (Jacobi) form, and it coincides with depth passes of a causal
  core: pass r reads the previous pass's state at earlier chunks (our causal attention over [X, S_{r−1}] already does this
  implicitly). An explicit shifted carry (chunk j−1's last-pass state fused before layer 1, zero-init tanh gates) is a cheap
  add-on that makes each extra pass extend the serial horizon by one chunk. Evidence: S5-Retrieval state tracking solved by a
  4-layer T²MLR where a same-size Transformer and LSTM fail. Kept as a design ingredient, not rejected.
- SHIFTED-CARRY CAVEAT (LoopCoder-v2): PLT's CLP REPLACES the same-position recurrence with the neighbour's state; the offset cost
  is a fixed per-loop tax (not compounding) that swamps the shrinking gain beyond loop 2 (SWE-bench 43.0/64.4/27.6/22.4 at R = 1–4).
  T²MLR ADDS the shifted cache to the unchanged stream and improves. ⇒ a design may use a shifted carry only as a gated, zero-init
  EXTRA input with S_j (same chunk, previous pass) kept as the main path. Reason that survives compaction: Ω is a constant per-
  boundary tax, but a SHIFT-ONLY recurrent chain moves one position per loop — token i at loop r carries a chain rooted at token
  i − r + 1 (PLT compensates by re-injecting E_i and attending over context). At R = 8 in a chunk-rate core a shift-only carry would be
  rooted 7 chunks back, so the same-chunk state must stay the main path.
- LoopWM's outer loop (state carried across environment steps) is the same temporal axis as RLT: rejected for training, allowed
  only as the eval-time exact carry (T²MLR form).
- Per-chunk adaptive depth routing (MoR): efficiency lever only (depth capped at N_r, no extrapolation), the core already runs at
  ¼ rate, and expert-choice top-k over chunk positions is non-causal in training (leaks future chunks into the plan). Rejected.

**Conditional (not core):**
- DiscoLoop decoded-embedding channel between passes: local test underpowered on the paper's task (vanilla never converged);
  contradicts the paper's mechanism on our testbench (bridge decodability 35–62 % → 2–5 %, +750 steps); allowed only in a design
  with per-pass supervision (then soft-decode through the PQ codebook); not retested this cycle; do not cite P3 as a replication.

**Sources for the user's two named residual options (read from source at the start of the design phase):**
`scratchpad/extra/Qwen4_tech_report.c.txt` (Qwen3.8-Flash-Next Gated Residual) and `scratchpad/extra/DeltaAttentionResidual.c.txt`
(Delta Attention Residuals, 2605.18855); also `extra/Attention_Residuals.c.txt` (2603.15031). Use `grep -a` on extracted texts.

## Experiment-plan caveats (collected while reading)
- WARM START (untie the current LoopLayer into 4 copies from the cont checkpoint): Nanbeige finds a loop trained FROM SCRATCH beats an
  upcycled one "significantly" — the only direct evidence in the folder. The warm start still answers the cheap question "does depth
  alone move hop?" in a few hours, but warm-start-ability earns NO scoring plus in the design phase.
- PER-PASS ATTENTION STATES confirmed (MoR recursion-wise caching, PLT's G-SWA recovering the KV-share loss, Nanbeige shared-KV
  lower): each pass attends over its own states — the current core already does this; no change.
- STABILITY vs COMPUTE: Ouro/Nanbeige instability at R > 2 is whole-stack, ≥ 1.4B, full BPTT; Loopie/SMELT's R = 2 is compute
  matching. Our chunk-rate core with grouped/truncated BPTT is outside both regimes.


## RESIDUAL SLOT — synthesis of the user's named options (Qwen GR, Delta Attention Residual) + papers (written after X1–X3)
| Option | What it is in a LOOPED core | Identity at init? | Bounded carry? | Evidence | Cost at d 256, T/4 = 256 |
|---|---|---|---|---|---|
| Qwen GR (current) | widened branches, gated read, write s = 2σ(·) | NO (s = 1) | NO (accumulates s·y each pass) | strong, non-looped (X1) | current |
| GR-read + zero-init write | keep GR read; write logit bias −4 (s ≈ 0.04) | ≈ yes | no (unless normed carry) | X1 + GRT bias +4 / zero-init lore | ≈ current |
| Delta Attention Residual over PASSES | S̃ + Σ_{r′<r} α v_{r′}, v = per-pass delta, zero-init query | YES (exact, null source) | additive; bounded perturbation | X2 (220M–8B), X3 (AttnRes family) | ≤ R extra sources/position; tiny |
| Delta-rule write | S ← S + β(v − read(S)), β ∈ (0,1) | yes if β-bias ≪ 0 | YES (error-correcting) | Survey #37 duality, DeltaNet line; no looped-core test | ≈ one extra read |
| mHC (doubly stochastic) | n streams, Sinkhorn-mixed | yes (I init) | YES (‖B‖ ≤ 1) | DeepSeek #39, AttnRes ablation 1.747 | n× state |
| MoDA depth-KV | joint softmax over sequence + earlier-pass K/V | yes (at init weight on depth small) | n/a (attention read) | #27 (non-looped) | trivial at T/4 |
CAVEAT (advisor #21) on Delta AttnRes over PASSES: the paper's sharpness argument relies on DIVERSE sources (attention vs MLP
outputs in different subspaces). In a weight-tied loop, pass-deltas come from the same block and our baseline shows near-identical
passes (step angle → 0.1°) — exactly the low-contrast condition that collapses softmax routing to uniform. With the 4-distinct-layers
cycle, deltas WITHIN a cycle are heterogeneous but same-layer deltas ACROSS cycles may not be. Any design using it MUST pair it with a
per-pass diversity source (MoE state-conditioned routing, per-pass V, K-candidate seeds) and put ROUTING MAX-WEIGHT per pass on the
eval sheet (collapse ⇒ ≈ 1/#sources).
Working conclusion for the design phase: keep the GR READ (best-evidenced), replace the loop-carry WRITE by a bounded one (delta-rule or
normalised carry), and add the DELTA ATTENTION RESIDUAL over per-pass deltas as the cross-pass retrieval path (user's named option,
safe-init, evidenced 220M–8B) — to be stress-tested by the advisor before it is fixed.

## C7 DENOMINATOR + ROW 0 (computed 2026-09-28 13:0x from code/model.py configs + runs/cont/config.json, CPU only)
Current model (d 384, heads 6×64 GQA kv 2, d_ff 1056, encoder SSSFSSSF, decoder confined 4 layers w 8, V 8192, T 1024, chunk 4):
params (config.json): encoder non-emb 15.51M, Engram table 50.33M (lookup), token emb 3.15M, concept module 2.06M (incl. codebook
0.049M = PQ 48 seg × 128 codes × 8 dims → 48 × 7 = 336 bits/chunk), decoder 7.53M; compute params 25.05M, stored 78.58M.
MACs/token (model.macs_per_token): encoder 16.292M, decoder 7.327M, head 3.146M → NON-CORE = 26.765M.
ROW 0 core (ConceptCoreV2: ONE shared LoopLayer at d 384 — attn 0.541M + SwiGLU 1.217M weights + GR; NCP head W_q):
  1.258M MACs/token at train R̄ = 2.5 (U{1..4}) = 4.5 % of total; 1.991M at eval R = 4 = 6.9 %; 3.946M at R = 8 = 12.8 %.
C7 BUDGET: core share ≤ 25 % ⇔ core ≤ 8.92M MACs/token ⇔ ≤ 35.7M MACs per chunk position (per 4 tokens).
d 256 block (the user's spec): per layer ≈ 4·256² (q,k,v,o MHA 4×64) + 256² (output gate) + 3·256·1024 (SwiGLU) = 1.114M weights
+ causal scores 2·(M/2)·256 = 0.066M (M = 256 chunk positions) ≈ 1.18M MACs per chunk position; 4 layers = 4.72M per pass per chunk
= 1.18M per token per pass; + in/out projections 384↔256 (2 × 0.098M per chunk = 0.049M/token) + NCP head.
⇒ passes allowed under the 25 % cap ≈ 7.5 (R = 4 → ≈ 4.8M/token = 15 %; R = 8 → ≈ 9.5M/token = 26 % — just over; GQA kv 2 gives
26.3 % at R = 8 (core_budget.py); X-side K/V reuse helps only designs with separate X/S key sets). Params: 4 × 1.114M + projections 0.197M + GR/heads ≈ 4.8–5.0M (vs 2.06M today) → compute params
25.05M → ≈ 27.9M (+11 %). PQ: can STAY in c_hat space (384-d, 48 × 128 × 8) with the NCP head mapping 256 → 384; changing it is a
per-design C4 choice (lower bitrate per Unified Latents, #32).

## DESIGN-SPACE PROPOSAL v0 (before advisor #22) — five designs, each anchored to a DISTINCT primary C2 mechanism
SHARED CHASSIS (all five, per user spec + settled constraints): middle-span core between encoder and decoder (interface unchanged:
forward(c_hat, cdoc, R) → u, logits, c_pred); in-proj 384→256 on c_hat (X) and out-proj 256→384 on u; FOUR DISTINCT layers,
d 256, SwiGLU FFN 1024, FULL causal attention over the 256 chunk positions (doc-masked), heads 4 × 64 (MHA) with sigmoid output
gate (as today); one "pass" = one cycle through the 4 layers; per-sample random pass count; S-space retained (X re-read every
pass through the GR elementwise-gated READ, carry S written by a BOUNDED write); X-side K/V reuse ONLY in designs whose attention keeps X-keys and S-keys
as SEPARATE key sets (not in the GR-read grammar, where K/V come from x = GRRead([X̃, S]) and change every pass — advisor #24); identity-at-init (write logits bias −4 / zero-init output projections); PQ stays 48×128×8 in c_hat space (NCP head
256 → 384) unless a design argues otherwise; full per-pass eval sheet.
D1 "S-FIX" (minimal fix of today's loop; C2 = LOSS PLACEMENT + LARGER R_max, mechanism (i)+(iii)): the current S-space grammar with
   4 distinct layers; R ~ U{1..6} train, main NCP/NTP loss always at R_max = 6 (full BPTT, one graph), light decaying (λ 1 → 0) loss
   at the sampled r + consistency ‖sg(u_Rmax) − u_r‖² (ELT/LoopFormer); fixes audit 1–4. Residual: GR read + zero-init write + LN carry.
D2 "LOOPFORMER-S" (user's favourite; C2 = SHORTCUT-CONSISTENCY SELF-DISTILLATION, (iii)): D1 chassis + LoopFormer (t, Δ) AdaLN-Zero
   conditioning on each of the 4 layers (normalised time → elastic depth, schedules with big steps early), long trajectory L = 4
   passes + one random shortcut S ~ U{1..3}, L = L_L + 0.1 L_S + 0.1 ‖sg(u_L) − u_S‖²; drift (CKA across passes) on the eval sheet.
D3 "DEEP-SUP GROUPS" (C2 = DETACHED-CARRY IMPROVEMENT OPERATOR, (ii)): TRM/URM/Diffusion-Timestep recipe — pass count K ~ U{2..8};
   passes grouped in windows of 2 (BPTT inside), carry DETACHED between windows, first ~25 % passes no-grad (URM TBPTL), EQUAL-weight
   per-pass NCP loss on graded passes, carry h = LN_h(S) with a separately-normed readout (Diffusion-Timestep dual views), NO pass index.
   Optional TRM two-state split (y = plan, z = reasoning) via 2 residual streams (mHC-style non-expansive mixing).
D4 "LOOPED-MoE + DELTA-RES" (C2 = STATE-CONDITIONED PER-PASS COMPUTATION + large random R, (i)): FFN → shared expert pool (16 × 512,
   k 2, one pool for all 4 layers and all passes; LB + z-loss) so each pass routes differently without a pass index; Delta Attention
   Residual over per-pass deltas (user's named option; paired with MoE diversity per advisor #21 caveat; bounded source window);
   R ~ U{1..8} + random per-pass residual scale (EBT); ELT-style loss (main at R_max, decaying at r).
D5 "K-PLAN (leak-bounded multimodal plan)" (C2 = (ii) detached groups as D3; PRIMARY novelty = C4 plan capacity): K = 4 learned seed
   slots S_0^k; run the core K times (cheap: core ≤ 25 %); score each by NCP loss vs the true next-chunk code; backprop best only
   (Explorative Modeling) → ≤ log2 4 = 2 bits/chunk leak (= Free-Transformer safe budget); inference: uniform seed or energy/confidence
   head pick (EBT); delta-rule write S ← S + β(v − read(S)).
Open questions for advisor #22: (a) are D1 and D2 distinct enough (both (iii))? (b) D4 mixes (i) with MoE — is "state-conditioned
per-pass computation" a legitimate primary C2 mechanism or a C3 one? (c) D5's cost (K core forwards) vs C7; (d) should one design
test the TEMPORAL plan carry (AI-inside-AI switch gate) or is that out of scope ("recurrence" = depth)?

## DESIGN-SPACE v1 (after advisor #22) — supersedes v0 where they differ
C7 SCORING BASIS (fixed now): C7 scores INFERENCE MACs/token at the design's stated eval R (share of total = core / (26.765M + core));
training overhead is reported separately as a multiple of today's training MACs and penalised only if > 1.5× (LoopFormer's own cost).
CHASSIS change: attention q/k/v shortconvs are DROPPED in the new core (URM: convs in the attention path useless-or-harmful); a causal
depthwise conv (kernel 2, over chunk positions) is offered only inside the FFN (ConvSwiGLU) as an optional, conditional ingredient.
D1 "S-FIX" = PURE mechanism (i): audit fixes 1–4 (identity-at-init write, R_max 8 instead of 4, bounded carry, loss placement = the
   final pass of the SAMPLED R only — no equal-weight loss across a fixed set, no distillation, no consistency term), R ~ U{1..8}
   per sequence; the Loop-Think-Generalize route. Honest minimal fix.
D2 "LOOPFORMER-S" = mechanism (iii): D1 chassis + LoopFormer (t, Δ) AdaLN-Zero conditioning + long trajectory + one shortcut +
   L = L_L + 0.1 L_S + 0.1 ‖sg(u_L) − u_S‖². EXCEPTION to the no-pass-index rule carried explicitly (reconciled in #23). Stated up
   front: LoopFormer buys BUDGET ELASTICITY (any R ≤ L good), not more-steps extrapolation → expect strong "R matters ≤ L" and weak
   OOD more-steps. Training overhead ≈ 1.5× (second trajectory).
D3 "DEEP-SUP GROUPS" = mechanism (ii): K ~ U{2..12} passes; windows of 4 passes with BPTT inside, carry DETACHED between windows
   (Diffusion-Timestep L = 4 best; or a 2 → 4 window curriculum per HRM-Text); first window no-grad only when K ≥ 8 (URM ~25 %);
   EQUAL-weight per-pass NCP loss on graded passes; LN_h carry + separately-normed readout; no pass index. Optional add-on (1 line):
   AI-inside-AI chunk-level plan switch gate.
D4 "LOOPED-MoE + DELTA-RES" = mechanism (i) exactly as D1 for C2 credit; PRIMARY NOVELTY = state-conditioned per-pass computation
   (C3): shared expert pool + Delta Attention Residual over per-pass deltas (paired with MoE diversity; routing max-weight on eval).
D5 "K-PLAN" = mechanism (ii) as D3; PRIMARY NOVELTY = leak-bounded multimodal plan (C4): LATE-BRANCH fork — passes 1..R−1 shared,
   only the LAST pass runs for K = 4 learned seeds (R = 4 eval → 3 + 4 = 7 pass-equivalents ≈ 8.3M MACs/token ≈ 23.7 %); best-of-K by NCP
   loss → ≤ 2 bits/chunk leak; C5 line item: train/infer mismatch (decoder trained on best plan, inference needs a selector — train an
   energy/confidence head on the K candidates, EBT-style). Optional add-on: AI-inside-AI switch gate.

## PER-PASS EVAL SHEET (common to all designs; design-specific rows added per design)
E1 R-sweep R ∈ {1,2,4,6,8,12,16} beyond train max: NCP CE, val NTP, probes (recall1, carry3, multi4, hop, bind) · E2 OOD more-steps
tasks (hop/bind/carry with more hops than trained) at R = 4/8/12/16 · E3 per-pass logit-lens: NCP head applied to every pass's state →
per-pass NCP acc + KL(p_r ‖ p_R) · E4 step-agreement matrix (argmax code agreement between passes) · E5 per-pass update cos θ (step
angle) and effective rank of S · E6 per-pass state norm (bounded?) · E7 chunk-0 attention mass per pass (sink) · E8 u-content: how many
answer tokens the plan carries (today 0.59 → 0.21 → 0.06) · E9 prequential epiplexity of the NCP loss curve (area above final loss) ·
E10 halting: KL(p_r‖p_{r−1}) < 0.01 ∧ H(p_r) < H_th exit rate per R (Loop-Think rule).

## D1 — "S-FIX": minimal fix of today's S-space loop — v2 (revised after advisor #24)
PRIMARY C2 MECHANISM: (i) large random train R (Loop-Think-Generalize) + identity-at-init; loss placement deliberately UNCHANGED
(final pass of the sampled R only) so D1 isolates the hypothesis "audit items 1 + 2 (+4) caused the R-invariance".
ARCHITECTURE (per chunk position j; M = 256 positions; X = c_hat ∈ R^384):
  X̃ = W_in X (384 → 256, computed once); S_0 = X̃ (NON-ZERO start, advisor #24: with S_0 = 0 and zero-init W_o/W_2 the state would be
  exactly 0 on every pass, u = 0, zero NTP gradient into the core and a 1/√eps norm Jacobian at S = 0 — a dead start).
  pass r = 1..R, layer l = 1..4 (4 DISTINCT layers, the block is cycled R times):
    x = GRRead_l^a([X̃, S])            elementwise sigmoid gate over 2 branches, low-rank r = 32 (Qwen GR read, current grammar)
    S ← S + s_l^a · Attn_l(x)          FULL (not windowed) causal attention over all 256 chunk positions (doc-masked), 4 heads × 64,
                                       GQA kv 2 (still full attention — only K/V heads are shared), sigmoid
                                       output gate, NoPE (as today's F-layers), NO q/k/v shortconvs (URM); W_o ZERO-INIT
    x = GRRead_l^f([X̃, S]);  S ← S + s_l^f · SwiGLU_l(x)   (d_ff 1024; W_2 ZERO-INIT)
  after each pass: S ← S / max(rms(S)/ρ, 1) (norm CLAMP at radius ρ = rms(X̃)-scale, not strict normalisation — identity while the
  state is small, bounds it otherwise; audit 4).
  readout: u_256 = zcRMSNorm(S_R); NCP logits = distance head(W_q u_256) over the PQ codebook (W_q 256 → 384, codes stay in c_hat space);
  decoder plan u = W_out u_256 (256 → 384, SMALL non-zero init (std 0.02) — zero-init exactly ONE thing per path: here the block
  outputs W_o/W_2, not W_out), c_pred as today.
  write scalar s = 2σ(w) keeps today's GRWrite; with zero-init W_o / W_2 every layer is an exact identity at init (Loop-Think App. E).
HEADS 4 × 64 (GQA kv 2). PQ 48 × 128 × 8 in c_hat space (336 bits/chunk) — unchanged (minimal fix; no warm-start issue for the codebook).
R DISTRIBUTION: R ~ clip(Poisson(4), 2, 8) per sequence (P: 2 .24, 3 .20, 4 .20, 5 .16, 6 .10, 7 .06, 8 .05; E[R] = 4.08). Eval R = 4
  (decode default) + R-sweep to 16; optional KL+entropy halting.
LOSS: L = L_NTP + λ_NCP · CE_NCP(final pass of sampled R) + L_VQ (unchanged). No per-pass loss, no distillation, no consistency.
BPTT: full through the sampled R (≤ 8 passes × 4 layers = 32 layer applications at 256 positions ≈ the encoder's activation memory).
BUDGET (core_budget.py): params 4.81M (vs 2.06M) → compute params ≈ 27.8M (+11 %); INFERENCE R = 4: 4.83M MACs/token = 15.3 % of total
  (C7 ✓); R = 8: 9.55M = 26.3 % (diagnostic mode, over cap); training E[R] 4.08: 4.92M/token → training total ×1.13 of today.
WHAT IT FIXES vs row 0: audit 1 (identity-at-init: zero-init W_o/W_2 with a non-zero start S_0 = X̃), audit 2 (R_max 4 → 8, Poisson, R_min 2),
  audit 4 (norm clamp on the carry), plus capacity (4 distinct layers vs 1 shared, C3) and URM's attention-conv removal. KEEPS audit 3 by design.
RISKS (stated up front): GRT (#3) is the counter-evidence — r ~ U{1..R} with loss at the sampled depth converged to a FIXED POINT by
  step 4 and was slightly worse beyond train R; Loop-Think's positive result used zero-init + larger R + a parametric toy task with
  10^5–10^6 steps. So D1's C2 is the least certain of the five in our 24 h LM regime; its value is as the controlled row that tells us
  whether init + R alone were the cause. Weak C1 grounding (no per-pass target).
  MAIN C2 RISK (advisor #24): Loop-Think escaped invariance because its task REQUIRED depth (an 8-hop answer cannot exist at pass 2).
  On an LM mix most chunks are solvable at pass 2, so the sampled-R loss (24 % of sequences at R = 2) is MINIMISED by R-invariance —
  the audit-3 mechanism, kept by design. D1's C2 therefore depends on the depth-requiring synthetic tasks (hop/bind/carry > 2 hops)
  weighing enough in the mix that pass 5 must differ from pass 4 somewhere; the Poisson tail (5 % at R = 8) barely exercises the top
  of the range in a ~4k-step run.
  EXPECTED OUTCOMES: if init + R_max were the cause → E5 step angle stays > 1° beyond pass 4 on depth tasks and E4 agreement < 1 between
  passes 4 and 8, with hop/bind accuracy rising with R (E1/E2); if audit 3 is the cause → step angle again collapses to ≈ 0.1° by
  pass 4, E4 ≈ identity-like agreement, R-curve flat (today's result at larger R). Either outcome is informative.
DESIGN-SPECIFIC EVAL: none beyond E1–E10 (E4/E5 are the decisive rows: does the step angle stay > 1° past pass 4?).

## D2 — "LOOPFORMER-S": the user's favourite, faithfully — DRAFT v1 (pre-review; advisor was overloaded — review pending as #26)
PRIMARY C2 MECHANISM: (iii) shortcut-consistency SELF-DISTILLATION (LoopFormer #1): main loss on the long trajectory, a random
shortcut trained toward sg(long endpoint) → "concept drift" (representations keep evolving across passes, low cross-pass CKA)
instead of stagnation. Claim scope stated up front: BUDGET ELASTICITY for R ≤ L; NOT more-steps extrapolation (LoopFormer never tested
R > L); OOD more-steps rows (E2) expected weak.
ARCHITECTURE: D1 v2 chassis (S_0 = X̃; 4 distinct d-256 layers, full causal attention 4 × 64 GQA kv 2, SwiGLU 1024, GR read, norm
  clamp on the carry, no attention shortconvs) PLUS per-pass conditioning c = φ(t) + φ(Δ) (sinusoidal 256 → 2-layer SiLU MLP; t =
  normalised time of the pass start, Δ = its step size, ΣΔ = 1) feeding an AdaLN-Zero modulator per layer (SiLU → Linear 256 → 4·256:
  α_attn, α_ffn, γ_attn, γ_ffn):  S ← S + s·α(c) ⊙ Attn(x ⊙ (1 + γ(c))), same for the FFN.
  ONE zero-init per path (advisor #24 rule): α zero-init (AdaLN-Zero) and W_o/W_2 keep DEFAULT init (not also zeroed); W_out small
  non-zero. At init every layer is identity (α = 0), S stays X̃, u = zcRMSNorm(X̃) ≠ 0 — no dead start; α receives gradient at step 1.
  Pass conditioning is the documented EXCEPTION to the default-off rule (elasticity design; conditioning stays in range because random
  shortcut schedules expose small Δ; untested = trajectory LENGTH R > L; eval for R > L uses uniform Δ = 1/R).
TRAJECTORIES: long L = 6 passes (4 layers × 6 = 24 effective layers = LoopFormer's 3 ⊗ 8 = 24), schedule uniform Δ = 1/6; one shortcut
  per batch with S ~ U{1..5} passes and a random schedule (Dirichlet(1) step sizes, sorted big-early per LoopFormer's best schedule).
  Both start from the same S_0 = X̃.
LOSS: L = L_NTP(decoder on u_L) + λ_NCP·CE_NCP(u_L) + 0.1·λ_NCP·CE_NCP(u_S) + 0.1·‖sg(u_L) − u_S‖² + L_VQ. The decoder runs ONCE (long
  plan only) — the shortcut costs only core passes (LoopFormer ran the whole model twice; we keep the language path single).
  Full BPTT through both trajectories (≤ 6 and ≤ 5 passes).
HEADS 4 × 64 (GQA kv 2). PQ 48 × 128 × 8 in c_hat space (unchanged).
R AT INFERENCE: default R = L = 6 (elastic: any R ∈ {1..6} with uniform Δ = 1/R; cheap mode R = 2–4); R-sweep to 12 for E1/E2 only.
BUDGET: params 5.99M (D1 4.81M + AdaLN/time MLP 1.18M) → compute params ≈ 29.0M (+16 %); INFERENCE R = 6: 7.19M MACs/token = 21.2 % (C7 ✓);
  R = 4: 15.3 %; training (long 6 + shortcut E = 3 passes): 10.73M/token → training total ×1.34 of today (≤ 1.5× ✓).
WHAT IT FIXES: audit 1 (AdaLN-Zero identity-at-init), 2 (L = 6 > 4), 3 (main loss on L only; shorter routes 0.1-weighted and distilled —
  exactly the LoopFormer asymmetry the audit named), 4 (clamp); stagnation targeted directly (drift).
RISKS: elasticity ≠ extrapolation (E2 weak by construction); consistency pulls short → long but does not make pass 6 better than
  pass 5 by itself (#1 (e)); AdaLN-Zero warm-up at 25M/24 h on MPS (#1 C6 note); LoopFormer's evidence is 1B/25B tokens on Pile —
  C1 discount for our scale; the second trajectory adds ≈ 0.34× training cost.
DESIGN-SPECIFIC EVAL: E11 cross-pass CKA (drift; LoopFormer's signature — low CKA across passes, vs stagnation); E12 quality vs budget
  curve R = 1..6 (elasticity: monotone improvement, R = 2 already usable); E13 schedule sensitivity (uniform vs big-early at fixed R).

## D3 — "DEEP-SUP GROUPS": detached-carry improvement operator — DRAFT v1 (pre-review; review pending)
PRIMARY C2 MECHANISM: (ii) DETACHED carry between windows + EQUAL-weight per-pass loss inside graded windows — the recipe with the
strongest depth-extrapolation evidence in the set: Diffusion-Timestep (#28: K ≤ 160 in training, monotone to 10,000; L = 4 windows
best, L = 1 fails, L = 16 vanishes), URM (#22: first ~25 % passes no-grad, per-loop CE on the rest; 4-layer block looped), TRM/GRAM
(#23: deep supervision, 16 → 256 steps), SMT (#26: credit amortised across detached steps).
ARCHITECTURE: D1 v2 chassis (S_0 = X̃, zero-init W_o/W_2, W_out small, 4 distinct d-256 layers, full causal attention 4 × 64 GQA kv 2,
  SwiGLU 1024, GR read, no attention shortconvs), NO pass conditioning (default-off rule), with Diffusion-Timestep's DUAL NORMED VIEWS:
  carry h = clamp(S) (bounded, separate norm) and readout u_256 = zcRMSNorm(S) with its own gain → NCP head / W_out.
SCHEDULE: pass count K ~ U{4..12} per sequence (E[K] = 8); passes grouped in WINDOWS OF 4 (BPTT inside a window, carry DETACHED at every
  window boundary); when K ≥ 8 the FIRST window runs no-grad (URM TBPTL ≈ 25 %); a trailing partial window (K mod 4) is graded.
LOSS: EQUAL-weight per-pass NCP CE on every pass of every graded window (the grounded per-pass target: the true next-chunk code — C1
  grounding condition satisfied) + L_NTP through the decoder on the FINAL pass only (decoder runs once) + L_VQ.
  (Regime check vs the revised rule: detached-carry groups ⇒ equal weighting is the evidenced choice.)
HEADS 4 × 64 (GQA kv 2). PQ 48 × 128 × 8 in c_hat space (unchanged).
R AT INFERENCE: default R = 6 (21.2 %); halting option E10; R-sweep 1–16 (the extrapolation claim is tested by E1/E2 at R = 8–16).
BUDGET: params 4.81M (same as D1) → compute params ≈ 27.8M; INFERENCE R = 6: 7.19M MACs/token = 21.2 % (C7 ✓; R = 4: 15.3 %); training
  E[K] = 8 passes with per-pass head evaluations: 9.90M/token → training total ×1.31 (≤ 1.5× ✓; backward only through graded windows,
  so the real training cost is lower than the forward-MAC ratio suggests). Activation memory ≤ one window = 16 layer applications
  (HALF of D1's worst case).
WHAT IT FIXES: audit 1, 2 (K up to 12, beyond R = 8), 3 (the per-pass loss is paired with a DETACHED carry, so the model cannot satisfy
  every pass's loss with one fixed point reachable from any start: each window starts from the previous window's detached state and
  must IMPROVE it — the improvement-operator argument), 4 (clamp + separate readout norm).
RISKS: (1) multimodal target — Diffusion-Timestep's under-determined boards SETTLED THEN WANDERED (57.93 %): on the LM next-chunk target
  more passes need not converge; E3/E4 must show the per-pass NCP acc rising or plateauing, not oscillating. (2) Equal per-pass NCP loss
  still includes early passes: if they can already reach the loss floor on easy chunks, the incentive to improve is carried by hard
  chunks only (same data-dependence as D1, weaker because the detach removes the invariant shortcut). (3) No evidence at LM scale for
  detached windows over DEPTH passes (all positives are puzzles / pixels / Sudoku). (4) Per-pass loss weights the NCP head more than
  today (λ_NCP tuning; watch val NTP — C5).
OPTIONAL ADD-ONS (one line each): TRM two-state split (y = plan, z = reasoning) as 2 carried streams with mHC-style doubly-stochastic
  mixing; AI-inside-AI chunk-level plan switch gate z_j = β_j z̃_j + (1 − β_j) z_{j−1} (β zero-bias → today's behaviour).
DESIGN-SPECIFIC EVAL: E11 per-window improvement (NCP CE at the end of window w vs w−1 — must be ≤ 0 on average); E12 settle-vs-wander
  rate at R = 16 (Diffusion-Timestep Table 5 protocol on NCP argmax codes).

## D4 — "LOOPED-MoE + DELTA-RES": state-conditioned per-pass computation — DRAFT v1 (pre-review; review pending)
PRIMARY C2 MECHANISM: (i), scored identically to D1 (random large R, final-pass loss). PRIMARY NOVELTY (C3): each pass computes
DIFFERENTLY without any pass index — the looped dense FFN applies one fixed map every pass, "an invariance that compounds when looped"
(Sparse-Layers-Looped #25, user-added); an MoE's state-conditioned routing diverges across passes (25–53 % of tokens fully disjoint on
pass 2), and 4 × 4 was the best Looped-MoE shape at fixed compute. Plus the user's named DELTA ATTENTION RESIDUAL across passes (X2).
ARCHITECTURE: D1 v2 chassis (S_0 = X̃, zero-init W_o and expert W_2, W_out small, full causal attention 4 × 64 GQA kv 2, GR read, clamp,
  no attention shortconvs) with:
  • FFN → ONE shared expert pool for all 4 layers and all passes (Intern-Mobius #30 pattern): E = 16 SwiGLU experts × d_ff 512, top-2
    (active FFN width 1024 = the user's spec), per-layer router 256 → 16, load-balance loss + router z-loss (#25), no shared expert.
  • DELTA ATTENTION RESIDUAL over per-pass deltas (additive, safe init): before each sublayer of pass r, x̂ = x + Σ_i α_i v_i with
    sources {X̃, v_{r−1}, …, v_{r−4} (bounded window of the last 4 pass-deltas), null}, v_r = S_r − S_{r−1}, α = softmax(w_lᵀ RMSNorm(v_i)),
    w_l ZERO-INIT (uniform α; the null source gives an exact identity path). Bounded window keeps the softmax support fixed for
    R > R_train (MoDA/X2 guard). PAIRED with MoE diversity per the advisor-#21 caveat (same-block pass-deltas may be near-identical).
  • EBT random step scale in training: each pass's writes multiplied by ξ_r ~ U[0.5, 1.5] (ξ = 1 at eval) — EBT's single most important
    ingredient for gains that grow with more steps (#35).
R DISTRIBUTION: R ~ clip(Poisson(4), 2, 8) (as D1). Eval R = 4 (15.4 %), R = 6 = 21.3 %; sweep to 16.
LOSS: L_NTP + λ_NCP·CE_NCP(final pass of sampled R) + L_VQ + 0.01·L_LB + 0.001·L_z. (Same C2 regime and risk as D1.)
HEADS 4 × 64 (GQA kv 2). PQ 48 × 128 × 8 in c_hat space.
BUDGET: params 7.98M (expert pool 6.29M stored once, amortised over 4 layers) → compute params ≈ 31.0M (+24 %; stored, not active);
  INFERENCE R = 4: 4.87M MACs/token = 15.4 % (C7 ✓; active FFN = dense 1024); training E[R] 4.08: ×1.13 of today.
WHAT IT FIXES: audit 1, 2, 4 as D1; adds per-pass diversity (C3) and cross-pass retrieval (Delta-Res) so pass r can re-read pass-1
  features instead of an overwritten S.
RISKS: C2 as D1 (sampled-R loss may still be minimised by invariance on easy chunks); MoE at chunk rate: 512 chunk positions per
  micro-batch → expert load noisy (LB mandatory; E = 16 may be too many for 25M-scale data — E = 8 fallback); stored params +24 %
  (Sparse-Layers: part of MoE gains is stored capacity); Delta-Res routing may still collapse if same-layer deltas are alike (E14).
DESIGN-SPECIFIC EVAL: E11 routing divergence across passes (fraction of chunk positions with disjoint top-2 sets pass r vs r+1 —
  Sparse-Layers Fig. 5 protocol); E12 expert load entropy; E14 Delta-Res routing max-weight per pass (collapse ⇒ ≈ 1/#sources).

## D5 — "K-PLAN": leak-bounded multimodal plan — DRAFT v1 (pre-review; review pending, advisor #23 pre-draft items applied)
PRIMARY C2 MECHANISM: (ii) detached windows exactly as D3 (scored as D3). PRIMARY NOVELTY (C4): our plan u is ONE deterministic vector
while the next-chunk target is multimodal (#23 GRAM, #29 CALM, #28 wandering) → a blurred "mean plan". Explorative Modeling (#38):
best-of-K training commits candidates to modes WITHOUT a posterior; the selection leaks ≤ log2 K bits per chunk: K = 4 → ≤ 2 bits/chunk =
the Free Transformer's evidenced safe budget (#31: ½ bit/token × 4 tokens; ≥ 4 bits/token collapses).
ARCHITECTURE: D3 chassis (S_0 = X̃, dual normed views, 4 distinct d-256 layers, full causal attention 4 × 64 GQA kv 2, SwiGLU 1024, GR
  read, no pass index) with a DELTA-RULE (retention) WRITE on the carry: S ← (1 − β) ⊙ S + β ⊙ Ŝ_r (Ŝ_r = the pass's proposal, β = σ(b),
  b init −4 ⇒ ≈ identity at init; convex combination ⇒ bounded, contractive carry — the vector form of the delta-rule / Parcae retention).
  LATE-BRANCH FORK: passes 1..R−1 shared; before the LAST pass add one of K = 4 learned seed vectors σ_k (init N(0, 0.02·rms(X̃)),
  DISTINCT so candidates can diverge) → K last-pass candidates u^k. Training: NCP CE per candidate vs the true next-chunk code; per chunk
  position backprop ONLY the best candidate k* (hard min) into core AND decoder (decoder sees u^{k*}).
  SELECTOR (for inference, no target): a value head V(u^k) (256 → 64 → 1) regressing each candidate's NCP CE (GRAM's LPRM #23 / EBT
  energy #35); inference picks argmin V — the selector sees only candidates + context ⇒ no leak at inference.
SCHEDULE / LOSS: as D3 (K_pass ~ U{4..12}, windows of 4, detach, first window no-grad when K_pass ≥ 8, equal per-pass NCP CE on graded
  passes — the fork only at the final pass) + L_NTP on u^{k*} + 0.1·MSE(V(u^k), sg(CE_k)) + L_VQ.
HEADS 4 × 64 (GQA kv 2). PQ 48 × 128 × 8 in c_hat space.
BUDGET: params 4.83M → compute params ≈ 27.8M; INFERENCE R = 4 (3 shared + 4 forked pass-equivalents + 4 head evals + selector):
  8.53M MACs/token = 24.2 % (C7 ✓ but at the cap; R = 6 = 28.9 % ✗ — D5 is an R = 4 design); training ×1.44 (≤ 1.5× ✓).
C5 LINE ITEM (train/infer mismatch): the decoder is trained on the BEST plan but at inference sees the SELECTED plan → the gap is
  exactly the selector's error; SELECTOR TOP-1 ACCURACY on held-out (vs the oracle best) is on the eval sheet (E11).
C1 HONESTY: Explorative Modeling's own admission — AR language models are its HARDEST case with "modest" gains; D5's C1 evidence is the
  weakest of the five and is scored that way; the leak bound is an argument, not evidence of LM gains.
RISKS: selector error → decoder distribution shift (C5); K = 4 seeds may collapse to one mode (seed-diversity metric E12); cost at cap;
  stacking a new training trick on D3's already-unproven-at-LM-scale schedule.
DESIGN-SPECIFIC EVAL: E11 selector top-1 vs oracle; E12 candidate diversity (pairwise code disagreement among the K plans); E13 NTP with
  oracle-best vs selected vs random plan (quantifies the mismatch); E14 leak check: train NTP gain from u^{k*} vs val with selector.


## REVISIONS after advisor #25 (apply to the drafts above; these lines take precedence)
CHASSIS (all five): S_0 = W_s X_raw (a SEPARATE small projection 384 → 256, distinct from W_in, init std 0.02) instead of S_0 = X̃ —
  with S_0 = X̃ and zero-init block outputs both GR-read branches are identical on pass 1, so the elementwise read gate has nothing to
  discriminate (a slower dead start). With a distinct W_s the branches differ from step 0 and S_0 ≠ 0 (u ≠ 0, NTP gradient flows).
D1: expected C2 score range 4–5/10 written NOW (before scoring): D1 differs from GRT (#3: loss at sampled depth → fixed point by step 4,
  worse beyond train R) only in init, R distribution and 4 distinct layers; its C2 may not exceed row 0 by more than that delta.
D2: the shortcut trajectory starts from sg(S_0) (X̃ / W_s detached on the shortcut path) so ‖sg(u_L) − u_S‖² cannot push W_in / W_s
  to make the LONG endpoint easier to match; gradients into W_in/W_s come only from the long trajectory.
D3: NO no-grad first window: all windows graded, carry detached between windows (a no-grad first window would mean the model never
  learns to produce the state window 2 receives; URM's TBPTL relies on an outer ACT loop we do not have). Training MACs unchanged
  (forward), backward now through every window (still ≤ 16 layer applications of activations at once).
D4: E = 8 experts × d_ff 512 (top-2, active 1024) is the DEFAULT (E = 16 under-fed at chunk rate: ~1k positions per expert per
  micro-batch); E = 16 is the ablation. Pool params 3·256·512·8 = 3.15M → D4 params ≈ 4.83M (−3.15M vs the 16-expert draft). Delta-Res
  sources ALWAYS padded to 6 slots {X̃, null, 4 pass-delta slots zero-filled when absent} so the softmax support is identical at every R.
D5: leak bound restated: ≤ 2 bits per chunk position, but the confined decoder (window 8) at any token sees the plans of 2 chunks →
  ≤ 4 bits visible to a decoder position — below the Free Transformer's 4 bits/TOKEN collapse point, but the margin is smaller than
  claimed; K stays 4 (K = 8 would reach the threshold).

D2 v2 (after advisor #26): (1) GRWrite scalars DROPPED in D2 — AdaLN's α(c) IS the write gate (LoopFormer-faithful; avoids two gates in
  series). (2) The (1 + γ(c)) modulation sits AFTER the GR read (x = GRRead([X̃, S]) then x ⊙ (1 + γ)) — the one stated deviation from
  LoopFormer's plain pre-norm stack. (3) Eval at R = L may use the big-early schedule (LoopFormer's best; ~1.4 ppl schedule spread in
  their data); E13 compares uniform vs big-early. (4) DECISION (not a risk): α initialised SMALL NON-ZERO (0.05) with W_o/W_2 at default
  init — near-identity at init with live gradients, because LoopFormer's reported 4k-step warm-up would be our whole run; a C6 penalty is
  pre-committed for the residual warm-up risk. (5) The decoder ALSO runs on the shortcut plan every 4th batch at weight 0.1 (LoopFormer's
  L_S is an LM loss) so the decoder sees short plans in training; otherwise the elasticity claim would be "NCP-elastic only".
  Training cost now (26.765 + 10.73 + 7.327/4) / 28.02 = ×1.40 (≤ 1.5× ✓).

D3 v2 (after advisor #27): (1) remainder passes MERGED into the last window (window sizes 4–7; never a 1–3-pass window — Diffusion-Timestep
  L = 1 fails, L = 16 vanishes). (2) MAIN MECHANISTIC RISK stated: X̃ is a clean input every pass, so a window could recompute from X̃ and
  ignore sg(S_prev) (Diffusion-Timestep's h was the ONLY clean path); diagnostic E13 = GR-read gate mass on the S-branch vs X̃-branch
  per pass (S-branch mass → 0 in windows ≥ 2 ⇒ carry ignored); named ablation: X̃-dropout in windows ≥ 2. (3) Carry = LN_h(S) with its
  own gain (the evidenced Diffusion-Timestep form; valid now that S_0 = W_s X ≠ 0); the clamp stays only in D1/D4. (4) Per-pass NCP CE is
  the MEAN over graded passes (λ_NCP keeps today's meaning; no ~8× NCP gradient). (5) C8: per-window backward (forward window → NCP → backward
  → detach → next; the last window's backward waits for the decoder's NTP) — required for the ≤ 16-layer activation bound and for the MPS
  per-micro-batch peak at K = 12 (14.3M core MACs/token in that batch); interface change: forward returns per-pass logits (P×B×M×48×128)
  and losses() averages over P — a small trainer change shared with D5. D3 does NOT "fit the existing trainer" unqualified.

D4 v2 (after advisor #28): Delta-Res slots = {X̃, null, v_{r−1..r−4}}: ABSENT delta slots MASKED (−∞; "does not exist yet"), NULL = live
  source with value 0 ("route nothing" — a learned down-scaling of the routed sum, the exact identity path). Support is fixed by
  construction only for R ≥ 5 (where the extrapolation guard matters); at R = 2–4 fewer competitors ⇒ sharper softmax — an honest limit.
  Zero-init exemption stated: Delta-Res queries w_l zero-init give UNIFORM (not zero) routing, a bounded perturbation; the single zero on
  the path stays the block output (W_o / expert W_2). RISK added: one expert pool shared by 4 layers (fewer stored params) is not what
  Sparse-Layers tested (per-layer pools) — weaker evidence, less per-layer specialisation; E11 reported PER LAYER; new E15 = expert overlap
  across the 4 layers within one pass (same experts everywhere ⇒ pool collapsed to a dense FFN). ξ ~ U[0.5, 1.5] × clamp = train/eval
  mismatch in how bounded the carry is → E6 adds clamp activation rate per pass (train vs eval). PRE-COMMITTED: C2 4–5 (as D1); C3 earns
  its score only if E11 shows divergence AND E15 shows layer specialisation; C7 15.3 % at R = 4.

D5 v2 (after advisor #29): (1) SOFT-MIN training (XM's smooth variant) instead of hard-min: candidate weights w_k = softmax_k(−sg(CE_k)/τ),
  τ annealed 1.0 → 0.3 — every seed gets gradient ∝ its fit, preventing winner-take-all seed starvation (learned seeds, unlike XM's fresh
  noise, would otherwise collapse to one live seed); E12 still measures collapse. Leak bound unchanged in kind (the weighting is a
  function of the K CEs; ≤ log2 K bits of selection information per chunk as τ → 0). (2) SELECTOR trained by KL to the SAME soft target
  softmax(−sg(CE_k)/τ) (ties handled; ordering preserved; one τ for both uses). (3) EXPOSURE: on 25 % of batches the decoder receives
  sg(u^{k̂}) (selector's pick, DETACHED) — the decoder learns robustness to selector error; the core is trained only through the soft-min
  NCP path (no mean-plan pressure through NTP on non-best candidates). (4) CARRY = D3's LN_h (retention write dropped) so D5's C2 genuinely
  inherits D3's detached-window argument; single zero-init per path = block outputs. (5) STATED HONESTLY: the self-bounding argument covers
  the two extremes (near-identical candidates ⇒ small mismatch; very different CE ⇒ learnable selector) but NOT the multimodal middle
  (different plans, similar CE — several valid continuations): there the selector cannot discriminate and the decoder sees a different-but-
  equally-valid plan at inference; probably benign, settled by E13 (oracle-best vs selected vs random NTP).
