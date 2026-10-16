# v5_tiny: NCP → GRAM/XM → internal RL → GradCuit at 2M parameters

**Status: in progress (draft written during Phase B).** Numbers are in `RESULT_TABLES.md`, which `code/build_tables.py`
regenerates. Methods and declared deviations are in `METHODS.md`; predictions, gates and every later change are in
`PRE_REGISTRATION.md`; the advisor consults are in `ADVISOR_LOG.md`.

## 1. What was tested

The proposal (`NCP_GRAM_Z_Five_Phase_Architecture.md`, as narrowed by `review_v5.md`) has five phases:
- A: learn an NCP foundation;
- B: train a stochastic recursive concept generator with an external control Z against the frozen NCP;
- C: transplant it;
- D: learn persistent latent actions by internal RL;
- E: optimise Z per problem with GradCuit.

Here every phase is built and run at d=128 and about 2M parameters on a MacBook Air M2, on TinyStories text plus
GraphNav, a symbolic task designed to have:
- multiple valid futures;
- a persistent hidden variable (the current colour) that switches at unpredictable points;
- an exact verifier that reads only the header.

## 2. Phase A: the NCP foundation works at 2M parameters, and its concept route carries switch information

- NCP (2,022,144 parameters, 221M tokens, 85 min): text NLL 1.813; GraphNav walk NLL 0.113; greedy free-generation
  success 0.893.
- The in-context two-token lookup emerged abruptly between steps 7k and 8k (node NLL 2.15 → 0.04).
- **Route authority:**
  - The pre-registered gate passes on text (`mean` +0.089 nats) and fails on graph-walk NLL (+0.033 < 0.05).
  - The revised prediction (< 0.02 on graph) was exceeded.
  - The effect concentrates where the revised prediction put it, on colour tokens at switch points:
    `mean` +0.43, `swap` +0.62 nats.
  - Most of it flows through the CRC port (crc_mean +0.30) rather than the PQ route (pq_mean +0.07). Per review
    §B7, this does not establish PQ-concept semantics.
- **History protocol:** NCP's own inference protocol (predicted concepts fed back) costs +0.217 nats on text against
  refresh. Deployment here uses refresh.
- Concept forecasts beat the copy, mean and linear baselines in both domains.

## 3. Phase B: the replacement core, and the central finding about Z

All four variants trained 6,000 steps on the same batch sequence against the frozen NCP encoder, decoder, PQ heads
and codebook. The pre-registered gate is prior-sampled NLL (4 draws) within 0.05 nats of the teacher.

| variant | what it is | gate (text Δ, graph Δ) | other paths |
|---|---|---|---|
| V2 | deterministic recursive core (R=2, n_L=2) | **pass** (−0.005, −0.004) | — |
| V2a | one non-recursive block | **pass** (+0.004, +0.004) | recursion (V2) is worth 0.008–0.009 nats on held-out data (capacity-confounded) |
| V3 | Z with posterior and prior, no ε, no XM | **fail** (+0.345, +0.339) | prior-mean +0.147/+0.228; posterior −0.180/−0.110; best-of-8 −0.001/−0.031 (corrected) |
| V6 | Z + ε + Forward XM (K=2) | **pass** (+0.025, +0.017) | prior-mean +0.007/+0.002; posterior −0.154/−0.110; mixture-of-8 +0.002/−0.000 (corrected, see PRE_REGISTRATION 15:33) |

**What this establishes, from first principles:**
- **Z has authority over the frozen decoder.** When Z comes from the posterior (which sees the future window), graph
  NLL drops 0.11 nats below the teacher and colour NLL 0.50 below. The decoder executes the plan, including the colour
  switches the teacher cannot predict.
- **A variationally trained prior cannot supply that authority usefully (V3).** Its samples commit confidently to
  wrong colours, and the decoder obeys them. At V3's prior mean, replacing its plans with a constant *improves* colour
  at switches by 2.33 nats. This is review §B7's "posterior excellent, prior poor", observed.
- **Adding exploration over prior samples (V6) makes the prior deployable:** per-sample NLL is within 0.025 nats of
  the teacher, and the mixture over 8 prior samples matches it (+0.002 text, −0.000 graph). An earlier figure of
  "−0.05" was a measurement bug, now corrected. The attribution between XM and ε awaits V4 and V5.
- **The pre-registered P-NTP prediction** (Z invisible in NLL, ±0.03) was **wrong** by an order of magnitude for V3.
  The decoder did not ignore Z; it obeyed it.

**Free generation and control tests** (GraphNav, greedy tokens, pre-drawn noise; 512 instances each):

| core | standard success (hold4 prior) | hard-goal success | Z-swap: first colour differs | ε-only differs | persistence hold4 − resample |
|---|---|---|---|---|---|
| teacher NCP | 0.893 (greedy) | 0.361 | 0.596 (teacher's own T=1 spread) | — | — |
| V2 (no Z) | 0.889 | 0.379 | — | — | — |
| V2a (no Z) | 0.873 | 0.357 | — | — | — |
| V3 | 0.799 | 0.283 | 0.658 (≈ 2/3 chance: a near-uniform prior) | — | −0.015 ± 0.006 |
| V6 | 0.854 (prior-mean greedy 0.918) | 0.338 | 0.516 | 0.018 | +0.000 ± 0.004 |

- **Authority and calibration are separate properties.** Both Z-cores' plans steer the decoder. V3's prior steers it
  wrongly (lower success everywhere). V6's prior commits to useful choices: its swap rate is below the teacher's own
  sampling spread, and its deterministic prior mean gives the best standard-set success measured (0.918;
  unpaired, about 1.4 SE above the teacher, not claimed significant).
- **Z is not a persistent action out of the box.** Holding Z over 4-chunk windows leaves colour persistence at the
  token path's natural rate (about 0.75 in the data) for both cores. The decoder's 4-back copy supplies persistence
  more cheaply, and Z acts at the uncertain positions (first hop, switches). The pre-registered P-control gate
  therefore failed as worded, and its authority half (Z-swap − ε-only = +0.498 ± 0.023 for V6) passed. This is the
  condition ETA's discovery stage targets, so Phase D's primary metric is switch timing.
- **Guidance noise ε moves behaviour in only 1.8% of instances.** Behavioural variation lives in Z, the opposite of
  spec B7's "internal noise dominates" failure mode.

## 4. Phase C: transplant, blend curve, and the XM rescue

- **Blend curve (B_v6):**
  - λ=0 reproduces the teacher exactly on the deployable paths (Δ = 0.000), as the spec requires.
  - λ = 0.1–0.3 is marginally better than the teacher (−0.002, an ensembling effect).
  - λ=0.6 gives +0.004 and λ=1 gives +0.025/+0.017 (text/graph).
- **Continued training at λ=1** (1,500 steps, spec C2 default objective XM + L_var, δ = 0) moved V6 little: gate
  +0.021/+0.017; hard-goal sampled 0.338 → 0.363. In this design the transplant is effectively done in Phase B, which
  already trains at λ=1 through the frozen decoder.
- **C_v6 against the pre-registered C4 criteria: 3 pass, 1 half, 1 fail.** NLL, standard success (0.857 vs 0.893)
  and hard-goal success (0.363 vs 0.361) pass. Norm stability fails narrowly as worded (‖h‖ 12.68 → 13.73, a change of
  1.05 > 1.0; a point estimate under post-norm). Controls are half: authority passes (+0.557 ± 0.022), persistence
  fails.
- **The XM rescue (C_v3).** V3, whose prior failed the gate by 0.34 nats, was continued with the same C objective,
  which adds XM; V3 has no guidance noise. It now passes: +0.022/+0.020; mixture +0.003/+0.002; swap rate
  0.658 → 0.525. KL_Z stayed at 0.475, so XM calibrated the prior **without** reducing the information the posterior
  puts into Z. A same-length continuation without XM (`C_v3_noxm`) is the pre-registered control (slack). If it fails,
  XM is the necessary ingredient, and review §5's suggested order ("establish control with a simpler variant before
  paying for XM") is reversed by the data: the variational path alone yields an unusable prior.
- **Controller ordering** (C_v6): the deterministic prior mean (standard 0.910, hard-goal 0.416) > resampling Z every
  chunk (0.891) > hold4 (0.857 / 0.363). More frequent and less noisy Z decisions do better. C_v6's prior mean vs V2
  on hard-goal (paired): +0.037 ± 0.020 (1.9 SE), suggestive and below the 2-SE bar.

## 5. Phase D

(pending)

## 6. Phase E

(pending)

## 7. First-principles assessment

(pending)
