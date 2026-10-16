# v4 vs an LLM-style transformer stack — tiny CPU experiment

Owner's request (2026-10-01, ≈ 22:20): before choosing v4 for the next experiment, compare the proposed v4 (scaled down)
with an LLM-style transformer stack (3 : 1 sliding-window : full attention), with recurrence in the middle layers to
match v4's depth; same objective and everything else equal; on v3-style text and on gridworld-pinpad
(Kobayashi et al. 2025, *Emergent temporal abstractions in autoregressive models enable hierarchical RL*,
arXiv 2512.20605). CPU only, v3 keeps training.

Status: pre-registration written 22:45 (§3), before any training reading; all arms finished 05:09, analyses 05:15,
verdict written 05:25 from `runs/vs_compare.json`.

## Verdict (pre-registered rule, §3)

**Caution** (pinpad part labelled **budget-limited**: the LLM's curve was still rising at 6M — held-in b2 0.81 → 0.85 → 0.88 over seed 2's last three evals — so per §3 there is no further extension and the label is carried). Text is at parity (Δ_text = v4 − LLM = -0.007 ± 0.004 nats, paired, 2 seeds; bound +0.05), and both pinpad primaries
are worse for v4 by far more than 2 SE (3 seeds, 6M tokens each): P1 held-out optimal-action rate -0.133 ± 0.004 (action NTP
+0.301 ± 0.005), P2 pair integrity in closed loop -0.183 ± 0.015 (sampled) / -0.307 ± 0.019 (greedy). "Caution" is the strongest verdict the table could
give with text at parity — the "LLM preferred" row required a text loss > 0.15 — so, stated plainly: **on the
hierarchical task alone the result was decisive in favour of the LLM stack** (z ≈ −12 to −65), and on language the two
are indistinguishable. The pre-stated scale caveat applies: "a negative v4 result is a cost estimate, not a
refutation" (§3). The decision is the owner's.

In one paragraph: on text the two stacks are tied at 5M tokens and neither reaches retrieval onset. On gridworld-pinpad
the LLM stack learns the subgoal-pair structure at ≈ 1.5–3M tokens on every seed and v4 does not by 6M; v4's concept
route is used (core off sends its pair-second accuracy to chance) but does not acquire the structure. Why v4 lags is
not resolved: distance to the layout header is not the main cause (§5). v4 degrades more gracefully in likelihood past
the training length (12-colour task), while still choosing worse actions. v4's second core pass and encoder F loop
contribute nothing at this scale on either dataset; the LLM's recurrence pays on both.

---

## 1. The two architectures (tiny, d 64)

| | v4 (recommended defaults) | LLM stack |
|---|---|---|
| layout | encoder 8 (SSSFSSSF) → pooler (6 tokens → 1 concept) → core 4 layers × M 2 on T/6 → decoder 4 (confined, plan shifted one chunk) | prelude 8 (SSSFSSSF) → middle 4 (SSSF) × M 2 → coda 4 (SSSF); one residual stream over all T tokens |
| recurrence | encoder F layers × 2; core × 2 | prelude F layers × 2; middle block × 2 (same places, same counts) |
| unique layers / sequential depth | 16 / 22 | 16 / 22 |
| attention ratio | encoder 6 S : 2 F; core all F; decoder all S (window 8) | 12 S : 4 F = **3 : 1**, window 60 (T/8) |
| compute parameters | 1,212,032 | 1,206,864 (−0.43%; FFN 304) |
| other stored parameters | Engram table 392,928; codebooks 20,480 | none |
| objective | NTP + 0.05 CE_coarse + 0.1 CE_fine + VQ (concept terms) | NTP |

Same primitives in both: `GatedAttention` (GQA 4 × 16 / 2 KV, q/k zcRMSNorm, output gate, short conv on q/k/v;
S = window + partial RoPE, F = full causal + NoPE + per-head temperature — the Llama-4-style local-RoPE / global-NoPE
split), SwiGLU, zcRMSNorm, tied embedding with an rms-normalised input.

Held equal: the training rows (same stream and seed; rows padded per v4's k = 6 for both; verified byte-identical in the
real training loop), tokenizer, NTP target mask, AdamW (0.9 / 0.95, wd 0.1, v4's decay rule), lr 2e-3, warm-up 400,
cosine to 0.1 of peak over the budget, B 4, T 480, clip 1, token budget, threads.
Cannot be held equal: v4's concept terms (comparison on NTP only), the Engram table (the LLM has none), FLOPs per token
(reported beside every result; the estimate counts matmuls and attention only — it omits v4's gated reads and writes,
pooler, plan builder and Engram reads and both arms' short convs).
Design choice, declared: initialisation. The LLM uses LLM practice (N(0, 0.02), output projections scaled by
1/√(2 × 22)); v4 keeps its modules' own initialisation (PyTorch defaults plus its zero-initialised branches), as in
the first tiny experiment.

## 2. Data

- **Text:** v3's corpus as in the first tiny experiment, mixed 50/50 with recall1 / carry3 / multi4 probe rows (gaps
  80–200 tokens). **Filler flaw fixed:** training fillers from 60,000 training documents (133,137 sentences; was 3,000
  documents), validation fillers from the val store (16,750 sentences; 2.4% exact overlap, short common sentences).
- **Gridworld-pinpad** (paper A.1): 7 × 7, 8 colours, 4 walls, ≤ 100 steps; 16 pretraining tasks (Table A1); expert =
  exact shortest path (BFS, avoiding other colours), uniform over optimal actions, ε 0; fresh layout every episode.
  Tokens: header (colour + cell × 8, wall + cell × 4, start + cell, SEP) then [action, new cell] per step, END.
  The task is latent. Mean episode 91 tokens. Evaluation: held-in tasks on new layouts; the 20 held-out three-pair
  compositions; the 12-colour post-training task (length generalisation, up to 212 tokens).

## 3. Pre-registered decision rule (written before any reading)

Primary readings, at equal real tokens, mean over seeds, with the seed spread:

- **T1 · text language:** Δ_text = val NTP(v4) − val NTP(LLM).
- **T2 · text retrieval:** probes recall1 / carry3 / multi4 (teacher-forced exact match).
- **P1 · pinpad held-out compositions:** action NTP and optimal-action rate on the 20 held-out tasks, split by
  boundary flag (2 = first action toward a pair's second colour: determined by the subgoal structure).
- **P2 · pinpad closed loop:** rollouts on fresh layouts — share whose first six visits form a pretraining task;
  pair integrity (after colour 2c the next visit is 2c+1).
- **P3 · pinpad representation:** linear probes of the latent subgoal (4 classes, chance 25%) and target colour
  (8 classes) from causal features; for v4, coarse-code NMI with the subgoal.

Verdict for "choose v4 for the next experiment" (set before readings; the advisor review is recorded in the log):

| verdict | condition |
|---|---|
| **v4 preferred** | Δ_text ≤ +0.05 **and** v4 ≥ LLM − 2 SE on P1 and P2, with a v4 advantage > 2 SE on at least one of P1 / P2 / T2 |
| **parity — v4 acceptable** | Δ_text ≤ +0.05 and no pinpad primary worse by > 2 SE |
| **caution** | +0.05 < Δ_text ≤ +0.15, or one pinpad primary worse by > 2 SE |
| **LLM stack preferred** | Δ_text > +0.15 on every seed **and** LLM ≥ v4 on P1 and P2 |

**Standard errors (fixed now).** Both arms are scored on identical items, so every SE is a *paired* SE:
- text NTP: per-row mean NTP on the 64 val rows, d_r = NTP_v4(r) − NTP_LLM(r); SE = sd(d) / √64;
- text probes: per-item exact match, d_i = hit_v4(i) − hit_LLM(i) over the 192 items per family; SE = sd(d) / √n;
- pinpad P1: per-episode optimal-action rate on the 200 held-out-task episodes, paired by episode; SE = sd(d) / √200;
- pinpad P2: per-layout outcome (0/1) on the same 256 layouts, paired by layout; SE = sd(d) / √256;
- across seeds: the per-seed paired differences are averaged; SE = √(Σ SE_s²) / n_seeds. "> 2 SE" means
  |mean difference| > 2 × this SE. The seed spread is reported beside it.

**P2 protocol (fixed before the rollout code is written).** 256 fresh layouts (seeded, identical for both arms and
all seeds). The model receives the 27-token header; at every step only the 4 action tokens are offered (END is never
offered); the environment appends the new cell. Two policies: temperature 1 over the 4 action logits (primary: the
expert is stochastic; seeded) and greedy (secondary). Stop at 6 coloured-cell visits or 100 steps. Scores:
task rate = share whose 6 visits form one of the 16 pretraining tasks (< 6 visits scores 0); composition rate = share
forming any valid 3-pair composition (held-out ones included); pair integrity = among visits to an even colour 2c that
are followed by another visit, the share followed by 2c+1; completion = share reaching 6 visits.

**P3 protocol.** Features at every position whose next token is an action (SEP or a cell token); labels = the latent
subgoal (4) and target colour (8) of that action. LLM sites: the input embedding and the output of each of the 22
layer applications. v4 sites: input embedding, encoder token stream (L, the decoder's input), encoder global stream G,
plan at t (built from chunks ≤ j − 1: causal), decoder output h. v4's ĉ and u of the *current* chunk are excluded from
the token-level probes because they contain up to 5 future tokens (future actions); v4's coarse-code NMI with the
chunk's majority subgoal is reported separately and labelled non-causal. Probe = multinomial logistic regression on
standardised features, AdamW lr 1e-2, wd 1e-3, 500 full-batch steps; trained on 400 held-in-task episodes, tested on
200 held-in-task episodes (new layouts) and 200 held-out-task episodes.

**Expectation for T2, stated in advance (22:50).** The LLM's full-attention layers can attend directly to a fact
80–200 tokens back; v4 must carry it through the concept route (decoder reach 70), and the first tiny experiment saw
no retrieval onset in 5–7M tokens. T2 is therefore expected to favour the LLM at this scale. That is why T2 can only
add to a v4 verdict, never subtract: it measures the scale at which the concept route turns on, not whether v4 is
viable.

**Budgets (fixed 23:10, before any main-arm reading).** Text: 5M real tokens per arm, seeds 0 and 1, evals every
1M. Pinpad: seed-0 pair at 3M with evals every 0.5M; **if held-in act_opt_b2 − act_chance_b2 < 0.10 at 2M for
both arms, seed 0 is extended to 6M** and seeds 1–2 get the budget at which that curve flattens (stated before they
start). Pilots (0.25–0.3M tokens, inside the 400-step warm-up) are engineering checks only and appear in no result.

**Pinpad extension (rule fired 23:44; mechanics fixed 23:50, before any 6M reading).** At 2M, held-in
act_opt_b2 − act_chance_b2 = 0.451 − 0.353 = 0.098 in both arms (< 0.10; the periodic eval uses 96 episodes,
≈ 290 pair-second actions, SE ≈ 0.03). So seed 0 is rerun at 6M (`vs_pin_*_s0_6M`, evals every 0.5M). The 6M runs are
*the* seed-0 pinpad reading; the 3M finals (`vs_pin_*_s0`) are reported only in a separate budget-point table. If the
curve has not flattened by 6M, seeds 1–2 also run at 6M, the pinpad verdict is labelled budget-limited, and there is no
further extension. **LLM + Engram arm:** decided at the 6M seed-0 reading — if v4 leads by > 2 SE on any pinpad primary,
one LLM + Engram arm (seed 0, 6M) separates the hierarchy from the table; otherwise the eval-time `engram_off`
intervention is reported, with the caveat that removing a component the network trained with is asymmetric.

**Filler-fix criterion (fixed 23:10).** The first experiment's probe-row NTP was 1.31 on training-pool items vs 4.72
on validation-pool items (the training fillers were memorised). The fix counts as confirmed if the final
|train-pool − val-pool| probe-row NTP gap is < 0.3 nats in every text arm. Validation probes use val-store fillers, so
the probe readings are clean either way; this is a diagnostic.

Scale caveat stated in advance: at d 64 and a few million tokens, a negative v4 result is a cost estimate, not a
refutation (the first tiny experiment never reached retrieval onset); a positive one is stronger evidence.

---

## 4. Pinpad budget point: seed 0 at 3M (not the seed-0 reading; §3)

Run end-of-training evaluation (`final.json`), 192 held-in / 200 held-out / 64 long episodes, teacher-forced. Rates are optimal-action rates (argmax over the 4 actions inside the expert's optimal set); b0 = inside a segment, b1 = first action toward a pair's first colour (next pair latent), b2 = first action toward a pair's second colour (fixed by the subgoal structure); chance = mean |optimal set| / 4.

| set | arm | action NTP | optimal rate | b0 | b1 | b2 (chance) | next-cell acc. |
|---|---|---|---|---|---|---|---|
| heldin | v4 | 0.955 | 0.674 | 0.722 | 0.476 | 0.476 (0.358) | 1.000 |
| heldin | LLM | 0.926 | 0.681 | 0.728 | 0.498 | 0.465 (0.358) | 1.000 |
| heldout | v4 | 0.950 | 0.679 | 0.727 | 0.505 | 0.445 (0.351) | 1.000 |
| heldout | LLM | 0.930 | 0.685 | 0.733 | 0.505 | 0.453 (0.351) | 1.000 |
| long | v4 | 1.102 | 0.677 | 0.732 | 0.422 | 0.458 (0.364) | 1.000 |
| long | LLM | 1.048 | 0.677 | 0.730 | 0.451 | 0.453 (0.364) | 0.999 |

Switch-offs on the held-out set (same checkpoint, evaluation only):

| arm | switch-off | action NTP | optimal rate | b2 |
|---|---|---|---|---|
| v4 | none | 0.950 | 0.679 | 0.445 |
| v4 | core_off | 1.140 | 0.640 | 0.383 |
| v4 | plan_swap | 1.066 | 0.624 | 0.392 |
| v4 | M1 | 0.948 | 0.678 | 0.443 |
| v4 | loop_off | 0.950 | 0.680 | 0.442 |
| LLM | none | 0.930 | 0.685 | 0.453 |
| LLM | M1 | 0.971 | 0.665 | 0.397 |
| LLM | loop_off | 0.978 | 0.661 | 0.437 |

Observed (not pre-registered): at 3M the two stacks are close on every reading; the LLM's action NTP is lower by 0.02–0.05. v4's pair-second decisions go through the concept route: with the core off, b2 falls from 0.445 to 0.383 (chance 0.351) and action NTP rises by 0.19. M = 1 and the encoder F loop barely matter for v4 here; for the LLM both cost 0.04–0.05 nats. Next-cell prediction: v4 reached 1.000 by 0.5M (Engram n-gram lookup: the next cell follows from the previous cell and the action), the LLM 0.133 at 0.5M and 0.999 by 1M.

## 5. Pinpad seed 0 at 6M: onset in the LLM (observed during training; written 00:55, before the final)

Periodic evals (96 episodes per set): the LLM shows an onset between 2.5M and 3M tokens — held-in b2 0.444 (2.0M) →
0.628 (3.0M) → 0.705 (4.0M), held-out action NTP 1.001 → 0.729. v4 at the same points: b2 0.434 → 0.444 → 0.455,
held-out action NTP 1.002 → 0.941. Schedule sensitivity: the 3M-annealed LLM run (§4) ended at b2 0.465, while the 6M run
passed 0.6 by 3M (at lr ≈ 0.6e-3); the 3M budget point therefore understates the LLM's onset.

**Reach diagnostic (pre-specified 00:55, before it is computed).** Hypothesis: which cell holds which colour is stated
only in the 27-token header. The LLM's full-attention layers read it at any distance; v4's decoder sees the last 70
tokens, so once the header is out of reach the layout must arrive through the concept route (the long-range retrieval
that never turned on in the first tiny experiment). Test: bucket every action by its token index i in the episode —
*near* (i ≤ 70: the whole header is inside the decoder's reach), *partial* (70 < i < 97), *far* (i ≥ 97: none of it) —
and compare the arms **within** a bucket (buckets differ in task phase, so never across). Pair-second actions fall at
roughly i ≈ 37, 59, 81 on a 6-colour task, so two of three are *near*.
Signature written in advance: if reach is the main cause, v4 ≈ LLM in the near bucket and the gap sits in the far
bucket. If v4 trails by a similar margin in the near bucket too, the cause is elsewhere (plan staleness, the
bottleneck's capacity, optimisation, the 1.35× FLOPs) and the report says so. This is a diagnostic of *why*; it does not
modify the pre-registered verdict (§3), which is read from the paired SEs as written.


**Seed-0 6M finals (`final.json`; the paired SE verdict comes with seeds 1–2):**

| set | arm | action NTP | optimal rate | b0 | b1 | b2 (chance) |
|---|---|---|---|---|---|---|
| heldin | v4 | 0.910 | 0.689 | 0.731 | 0.514 | 0.507 (0.358) |
| heldin | LLM | 0.632 | 0.824 | 0.845 | 0.668 | 0.802 (0.358) |
| heldout | v4 | 0.906 | 0.698 | 0.735 | 0.540 | 0.538 (0.351) |
| heldout | LLM | 0.617 | 0.822 | 0.844 | 0.670 | 0.788 (0.351) |
| long | v4 | 1.050 | 0.689 | 0.737 | 0.466 | 0.500 (0.364) |
| long | LLM | 1.729 | 0.809 | 0.838 | 0.607 | 0.760 (0.364) |

Switch-offs, held-out set:

| arm | switch-off | action NTP | optimal rate | b2 |
|---|---|---|---|---|
| v4 | none | 0.906 | 0.698 | 0.538 |
| v4 | core_off | 1.669 | 0.612 | 0.373 |
| v4 | plan_swap | 1.169 | 0.618 | 0.378 |
| v4 | M1 | 0.896 | 0.699 | 0.528 |
| v4 | loop_off | 0.904 | 0.699 | 0.535 |
| LLM | none | 0.617 | 0.822 | 0.788 |
| LLM | M1 | 0.685 | 0.783 | 0.712 |
| LLM | loop_off | 0.798 | 0.773 | 0.602 |

**Reach diagnostic result (01:08, seed 0, 6M).** v4 − LLM, paired by episode:

| set | bucket | b2 rate v4 / LLM | Δ (SE) | optimal rate Δ (SE) | action NTP Δ (SE) |
|---|---|---|---|---|---|
| heldin | near (4170 actions) | 0.509 / 0.773 | -0.264 (0.031) | -0.132 (0.008) | +0.251 (0.009) |
| heldin | partial (1558 actions) | 0.471 / 0.837 | -0.366 (0.048) | -0.178 (0.017) | +0.379 (0.023) |
| heldin | far (270 actions) | 0.737 / 0.842 | -0.105 (0.130) | -0.145 (0.032) | +0.515 (0.062) |
| heldout | near (4373 actions) | 0.534 / 0.792 | -0.258 (0.027) | -0.122 (0.007) | +0.262 (0.009) |
| heldout | partial (1629 actions) | 0.567 / 0.787 | -0.220 (0.045) | -0.147 (0.015) | +0.371 (0.021) |
| heldout | far (319 actions) | 0.333 / 0.762 | -0.429 (0.111) | -0.150 (0.040) | +0.483 (0.058) |
| long | near (1408 actions) | 0.495 / 0.777 | -0.283 (0.050) | -0.127 (0.011) | +0.223 (0.020) |
| long | partial (832 actions) | 0.667 / 0.798 | -0.132 (0.059) | -0.100 (0.017) | -0.451 (0.099) |
| long | far (1814 actions) | 0.408 / 0.730 | -0.323 (0.041) | -0.131 (0.011) | -1.763 (0.098) |

**Read against the signature written at 00:55: the reach hypothesis is not supported.** v4 trails by a similar margin in the *near* bucket (held-out b2 −0.258 ± 0.027, optimal rate −0.122 ± 0.007) as in the others, so the header being outside the decoder's 70-token reach is not the main cause. Post-hoc remark, labelled as such: "inside the reach" for v4 still means *through stacked window-8 layers* (encoder S layers and the confined decoder are all local); v4 has no token-level full attention anywhere on its token path, while the LLM has a full-attention layer every fourth layer. The diagnostic separates distance, not access type; the remaining candidates (the bottleneck's capacity for layout facts, plan staleness up to 11 tokens, optimisation, the LLM's 1.35× FLOPs) are not separated by this experiment.

**Long task (12 colours, longer than any training episode), observed.** The LLM's action NTP is worse (1.729 vs 1.050) although its optimal rate is higher (0.809 vs 0.689). A check of the END hypothesis (the LLM expecting the episode to end after six visits) rejects it: mean P(END) at actions after the sixth visit is 0.180 (LLM) vs 0.212 (v4). The LLM's NTP after the sixth visit is 2.826 vs 0.616 before (v4: 1.207 vs 0.890): its errors past the training length are confident ones. v4 degrades more gracefully in likelihood, not in action choice.

**Reach diagnostic, all three seeds (paired by episode, v4 − LLM, pair-second rate):** held-out near -0.307 ± 0.017, far -0.460 ± 0.064;
held-in near -0.285 ± 0.017, far -0.070 ± 0.076 (≈ 20 far actions per seed); long near -0.310 ± 0.029, far -0.246 ± 0.028. The near-bucket gap is large on every set,
so the 00:55 conclusion stands with three seeds: distance to the header is **not the main cause**. In the held-out set
the far gap is larger than the near one, so distance contributes there.

## 6. Results, all seeds (paired; `runs/vs_compare.md` has every reading)

### 6.1 Text (5M real tokens, seeds 0 and 1)

| reading | v4 | LLM | v4 − LLM (paired SE) |
|---|---|---|---|
| **T1** val NTP (64 rows) | 2.5983 (2.600 / 2.596) | 2.6055 (2.609 / 2.602) | **-0.0070 ± 0.0040** |
| **T2** lr_recall1 (192 items) | 0.013 | 0.013 | +0.000 ± 0.004 |
| **T2** lr_carry3 (192 items) | 0.000 | 0.000 | +0.000 ± 0.000 |
| **T2** lr_multi4 (192 items) | 0.000 | 0.000 | +0.000 ± 0.000 |
| probe-row NTP gap, train − val fillers | 0.19 | 0.20 | criterion < 0.3: met in all four arms |

Switch-offs (val NTP, mean of 2 seeds; Δ vs intact):

| arm | core off | plan swap | M = 1 | F loop off |
|---|---|---|---|---|
| v4 | +0.124 | +0.145 | +0.002 | +0.001 |
| LLM | — | — | +0.050 | +0.022 |

T1 reading: parity (|Δ| < 2 SE; −0.007 favours v4 nominally). T2: both at chance, as pre-stated (22:50). The text curves (periodic evals) start with v4 ahead (−0.20 at 1M, −0.07 at 2M, −0.03 at 3M, −0.015 at 4M) and converge.

### 6.2 Gridworld-pinpad (6M real tokens, seeds 0, 1, 2)

| reading | v4 (mean of 3) | LLM (mean of 3) | v4 − LLM (paired SE) |
|---|---|---|---|
| **P1** held-out optimal-action rate | 0.694 | 0.827 | -0.133 ± 0.004 |
| **P1** held-out pair-second (b2) rate | 0.520 | 0.809 | -0.289 ± 0.013 |
| **P1** held-out action NTP | 0.911 | 0.610 | +0.301 ± 0.005 |
| held-in optimal-action rate | 0.685 | 0.827 | -0.142 ± 0.004 |
| held-in b2 rate | 0.518 | 0.816 | -0.298 ± 0.014 |
| long task (12 colours) optimal rate | 0.694 | 0.808 | -0.114 ± 0.004 |
| long task action NTP | 1.058 | 2.166 | -1.109 ± 0.030 |
| **P2** rollouts (temp 1): task rate | 0.000 | 0.001 | -0.001 ± 0.001 |
| **P2** rollouts (temp 1): pair integrity | 0.145 | 0.326 | -0.183 ± 0.015 |
| rollouts (temp 1): 6 visits reached | 0.999 | 0.992 | +0.007 ± 0.003 |
| rollouts (greedy): task rate | 0.000 | 0.008 | -0.008 ± 0.003 |
| rollouts (greedy): pair integrity | 0.141 | 0.439 | -0.307 ± 0.019 |
| rollouts (greedy): 6 visits reached | 0.568 | 0.953 | -0.385 ± 0.019 |

§6 values are per-episode means (the paired unit of §3); the §4–5 tables are per-action aggregates from `final.json`, so the same quantity can differ in the third decimal (seed-0 held-out optimal rate 0.698 per action, 0.693 per episode). Chance for b2 ≈ 0.35 (mean |optimal set| / 4). Per-seed held-out optimal-rate differences: -0.127, -0.119, -0.154; b2: -0.250, -0.267, -0.352.
P2: the task rate is at floor for both (no model completes a pretraining task in closed loop: v4 0.000, LLM ≤ 0.02); **pair integrity is the P2 component that discriminates**, and it is the one worse by > 2 SE. P1 alone already triggers caution.

**P3 (linear probes) and code NMI.** Subgoal (4-way) probes sit at the majority baseline at every site of both arms (best over 3 seeds and all sites: v4 0.287 at the plan, LLM 0.275 at the last layer; majority 0.254–0.267). Target-colour (8-way) probes rise modestly above the majority baseline of 0.13: best LLM 0.247 / 0.234 (held-in / held-out, last layer, seed 2), best v4 0.171 / 0.178 (plan, seed 1). The probe code was sanity-checked (previous-token identity decodes at 1.000); on seed 0, even on the probe's own training set, the LLM reaches only 0.34 / 0.26. At d 64 neither stack holds the subgoal as a linearly readable label; the target is weakly readable, more so in the LLM. P3 does not discriminate in the pre-registered sense (no SE was pre-registered for it). v4's coarse-code NMI with the chunk's subgoal: 0.015 / 0.018 / 0.016 (45–47 codes used): the codebook does not align with subgoals.

Switch-offs, held-out pinpad (mean of 3 seeds):

| arm | switch-off | b2 rate | action NTP |
|---|---|---|---|
| v4 | none | 0.520 | 0.911 |
| v4 | core_off | 0.386 | 1.459 |
| v4 | plan_swap | 0.383 | 1.164 |
| v4 | M1 | 0.517 | 0.909 |
| v4 | loop_off | 0.520 | 0.913 |
| v4 | engram_off | 0.509 | 0.920 |
| LLM | none | 0.809 | 0.610 |
| LLM | M1 | 0.741 | 0.683 |
| LLM | loop_off | 0.698 | 0.743 |

**Engram arm decision (closed).** The pre-registered condition for an LLM + Engram arm (v4 ahead by > 2 SE on a pinpad primary) was not met, so none was run. Eval-time `engram_off` on v4 (all three seeds): optimal rate unchanged (0.69), next-cell accuracy 1.00 → 0.38, action NTP +0.01. v4 uses the table for positions, not for decisions. Caveat as pre-registered: removing a component the network trained with is asymmetric.

## 7. Observed findings (not pre-registered)

1. **Onset.** The LLM's pair-second rate leaves chance between 1.5M and 3M tokens on all three seeds (held-in periodic b2 reaches 0.81–0.88 by 6M); v4's rises slowly, from ≈ 0.42 at 1–2M to ≈ 0.50 at 6M (final evaluation 0.50–0.54), with no onset. The LLM's onset is schedule-sensitive: the 3M-annealed run (§4) ended at held-in b2 0.465 (chance 0.358), below the 6M run's 0.628 at the same token count.
2. **v4's concept route is used but does not learn the structure.** Core off or plan swap sends v4's b2 to chance (0.37–0.40) and costs +0.25 to +0.76 nats of action NTP; on text the same switch-offs cost +0.12 / +0.14. The route carries information; it does not carry the subgoal structure at this scale.
3. **Length generalisation, in likelihood only.** On the 12-colour task v4's action NTP is lower by 1.11 ± 0.03 (LLM per seed 1.77 / 2.98 / 1.75; seed 1 is an outlier), while the LLM still chooses better actions (-0.114 ± 0.004). The END explanation was checked and rejected (§5): the LLM's errors past the training length are confident ones. This is the only reading where v4 is ahead by > 2 SE, and it is not a primary.
4. **Recurrence.** v4's second core pass and F loop contribute nothing measurable on either dataset (text M = 1 +0.002, F loop off +0.001 nats; pinpad b2 −0.003 / 0.000). The LLM's do (text M = 1 +0.050, loop off +0.022; pinpad M = 1 b2 −0.07, loop off −0.11). v4's core itself is used (item 2). This repeats the first tiny experiment's F-loop finding (+0.002 at k 6) and bears on the open draft_v4 §10.6 decision (core 4 × 2 vs 5 × 2).
5. **Position transient.** v4's Engram gave it next-cell accuracy 1.000 at 0.5M vs the LLM's 0.133; equal from 1M.
6. **Cost.** At equal tokens v4 trained 7–10% faster in wall-clock on pinpad (6M: 3,070–3,681 s vs 3,398–3,937 s) and 6–9% faster on text (5M: 3,639–3,645 s vs 3,857–4,019 s), with the caveat that every arm shared the CPU with v3 (training, then its final evaluation) and another arm; the LLM spends 1.35× v4's matmul + attention FLOPs per token (pinpad estimate).

## 8. Implications for choosing v4 (for the owner)

- **What the result says.** At d 64 and ≤ 6M tokens, a plain 3 : 1 SWA : full LLM stack with matched parameters and depth is as good as v4 on language and clearly better on a hierarchical task built to reward temporal abstraction. v4's architectural bet (pooled concepts, a confined local decoder, a concept core) did not show the advantage it was designed for, at this scale.
- **What it does not say.** Whether v4's route turns on at full scale (v3's did: recall1 1.000 at 48 h). The first tiny experiment never reached retrieval onset either; v3 did at 19M parameters. A negative tiny result is a cost estimate.
- **Specific design signals.** (a) The second core pass and the F loop are not earning their compute here (also seen in the first tiny experiment). (b) v4 has no token-level full attention on its token path; the LLM has one every fourth layer. Whether that is the cause is untested (§5 remark). (c) The coarse codebook does not align with subgoals (NMI ≈ 0.02).
- **Optional follow-ups, not launched (owner's call):** (1) an all-SWA LLM (no full layers) on pinpad — tests the access-type explanation from the other side, ≈ 1 h per seed on CPU; (2) a v4 variant with one full-attention layer in the decoder — same question from v4's side, ≈ 1.5 h incl. tests; (3) pinpad at 12M tokens for both stacks — does v4 have a late onset, ≈ 2.5 h per pair; (4) the LLM + Engram arm only if a later reading has v4 ahead.

## 9. Engineering record

- Tests: `code/test_vs.py`, 12 tests, pass on both attention paths (`runs/vs_tests_block60.log`,
  `runs/vs_tests_noblock.log`); `code/mutants_vs.py`, 11 deliberate defects, all killed by assertions
  (`runs/vs_mutants.log`).
- Runs (`runs/`): main — `vs_text_{v4,llm}_s{0,1}` (5M), `vs_pin_{v4,llm}_s{0,1,2}_6M` (6M); budget point —
  `vs_pin_{v4,llm}_s0` (3M); pilots — `vs_pilot_*` (engineering only, excluded from every result). Each run holds
  `log.jsonl`, `final.json`, `ckpt.pt`, `stdout.log` (with `/usr/bin/time -l`), and for main runs `analysis.json`,
  `per_unit.npz`, `analysis.log`; `vs_pin_v4_s0_6M/reach.json` holds the seed-0 reach diagnostic.
- Paired comparison: `runs/vs_compare.json` / `.md` (`code/analyze_vs.py compare`); reach: `code/reach_vs.py`.
- Code: `code/model_llm.py` (baseline), `code/pinpad.py` (environment, expert, tokens, rollout env), `code/run_vs.py`
  (runner for both stacks), `code/analyze_vs.py` (P1–P3, rollouts, probes, NMI, paired SEs), `code/reach_vs.py`,
  `code/test_vs.py`, `code/mutants_vs.py`; `model_v4.py` gained one backward-compatible key (`dec_reads`).
- Scheduling: `launch_vs.sh` (AC gate, nice 19, 2 threads), `queue_vs.sh` (pinpad 6M then text seed 0),
  `sched_vs.sh` (remaining arms and analyses; ≤ 2 experiment processes while v3 ran). Queue in `runs/queue.log`.
- Clock: tests 22:39–22:43; pilots 22:44–23:19; main arms 23:19–05:09; analyses 04:47–05:15.
- Cost to v3 (owner-accepted, never stopped): mean heartbeat 2,264 tok/s from 22:44 to v3's 48 h end at 02:13
  (33 heartbeats) vs 3,234 tok/s in the clean window 18:10–22:40 (63 heartbeats), ≈ −30% for 3.5 h ≈ 12M tokens
  (≈ 370 steps, ≈ 2.4% of the run); no guard event caused by the experiment (the 00:30 memory re-exec is v3's own
  24 GiB rule). Recorded as V3-019 in `run_v3_19M/DECISIONS.md`. v3's final evaluation shared the CPU with the last
  arms and analyses and ran ≈ 3 h (24 h evaluation: 1 h).
- Advisor consults: `VS_LLM_ADVISOR_LOG.md` (numbered rows plus own-check rows).

