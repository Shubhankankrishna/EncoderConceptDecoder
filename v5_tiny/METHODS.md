# v5_tiny methods

This file separates three kinds of choice: **paper mechanism** (from the cited paper), **spec adaptation** (from
`NCP_GRAM_Z_Five_Phase_Architecture.md`, as reviewed in `review_v5.md`), and **implementation choice** (ours,
declared here). Code lives in `code/`; runs in `runs/`.

## Hardware and throughput

- MacBook Air M2 (8 cores, 16 GB) on MPS, fp32, with `torch.compile` (inductor-MPS).
- Measured single-job training throughput:
  - NCP: 70k tokens/s; V0: 61k tokens/s compiled (28k/24k eager);
  - bf16 autocast was slower on MPS (25k), so fp32 is used.
- Compiled CPU training failed (OpenMP mutex error), so CPU eager runs at about 18–20k tokens/s.
- **Inductor-MPS bug (found by `test_v5.py`).** Inside compiled code, reductions over sliced tensors are wrong: a masked
  sum was off by 29,277.
  - Workaround: compiled regions return per-token tensors only, and every reduction runs eagerly.
  - The test asserts eager = compiled for the loss and gradients (relative gradient difference 2.8e-7) and for the
    GramCore outputs including KL.
- Width is d=128 throughout (owner's instruction), with a 2048-entry vocabulary chosen for throughput: at d=128 the LM
  head dominates FLOPs.

## Data

Generators and statistics are in `code/data_v5.py`, `code/stats_graphnav.py` and `data/graphnav_stats.json`.

**Text:**
- TinyStories, from the local HF cache.
- ByteLevel BPE with 2048 entries, trained on 200k stories. The individual-digit pre-tokeniser was dropped.
- Train: one shard, 133.6M tokens. Val: 5.35M tokens.

**GraphNav:**
- Instance: 10 nodes drawn from 24 node tokens, and 3 colours, each colour a fixed-point-free function.
- The header gives goal, start and all 30 edges as 4-token chunks.
- The walk follows a hidden rule sequence: 2–4 segments, each holding a colour for 2–4 hops. The goal is the endpoint
  (hindsight); walks that visit the goal early are rejected.
- Every hop is one k=4 chunk, `[colour, <to>, node, <sep>]`. The position that receives ĉ for a hop chunk is the one
  predicting that hop's colour, which is the only uncertain token per hop. `test_v5.py` asserts this alignment.
- **Implementation choice:** segment durations of 2–4 rather than 1–3. The change was made after measuring 1–3
  (switch rate 0.42, run 1.89), which is too weak a persistent variable; 2–4 gives switch rate 0.25 and run 2.86.
  The choice was made before any model training.

**Mix and reductions:**
- 32 text rows and 32 GraphNav rows per batch.
- NTP is the per-token mean within each domain, then 0.5/0.5. Concept-level terms use the same reduction over scored
  targets.
- Only GraphNav walk and done tokens are scored; the header is random.

## Phase A: NCP foundation

| Element | Status |
|---|---|
| Encoder / concept / decoder stacks, causal; k=4 mean pooling; PQ soft expectation; causal shift (b_t = ĉ_{(t−3)//4}); NTP + α NCP-MSE + β VQ; target detached; VQ codebook-only | paper mechanism [R1] |
| 3 / 2 / 3 blocks at d=128, 4 heads, SwiGLU 384, RoPE, RMSNorm | implementation (local scale) |
| PQ with S=8, V=32, 16 dims; affine bound 128 of 128 (review §4.2) | spec adaptation |
| Pooled concept RMS-normalised without affine; stops the MSE target scale drifting | implementation, declared |
| IRC omitted (at most 3 depth states per module); CRC reduced to one concept→decoder port A_j = W_A·RMS(u_j), added as γ⊙A with the same shift; the encoder→concept CRC route is also omitted (the concept module sees pooled C only, for teacher and replacement alike) | implementation, declared |
| NCP-MSE as a per-feature mean (the report uses a squared vector norm), so α is not the paper's α | implementation, declared |
| λ_q = 0: no code cross-entropy (review §5.4) | spec adaptation |
| Codebook: data initialisation; dead-code restart every 250 steps until 80% of training; its own optimiser group with 10× lr and no weight decay | implementation |
| AdamW (0.9, 0.95), lr 2e-3, cosine, 300 warmup steps, wd 0.1 on matrices, clip 1.0, batch 64 × 192 | implementation; the paper used Muon + AdamW |
| V0: 8-block transformer with 1,968,256 parameters vs NCP's 2,022,144 (2.7% fewer, not exactly matched); per token, NCP does 6 token-level blocks + 2 concept blocks at T/4 vs V0's 8 token-level blocks | implementation |

- `codes_used = 0` in the log at restart steps is a logging artefact: the restart zeroes `usage` on the step that is
  logged.
- **History protocol.** NCP trains on clean pooled history; its paper feeds predicted concepts back at inference.
  `eval_a.py` reports teacher-forced NLL under predicted-concept history. Our deployment protocol is refresh: the
  encoder runs on the generated tokens and the actual pooled history is used, which is causal.

## Phase B: replacement core (`GramCore`, `phase_b.py`)

**Paper mechanism (GRAM [R2]):**
- hierarchical recursion with low state l and high state h;
- n_L low refinements per high transition, R high transitions, shared fL and fH;
- stochastic guidance ε added only to the high state, h = u + ε;
- a state-dependent Gaussian prior p(ε|u) and a target-conditioned posterior q(ε|u,y);
- fixed learned initial states.

**Spec adaptations:**
- An external control Z (d_z=8) is held over windows of chunks. It has a posterior q(Z | window targets, context at
  the window start) and a causal prior p(Z | context at the window start).
- The core reads the clean pooled history C through `in_proj`.
- Readouts: u_new goes to the frozen PQ heads; A_new is a separate readout for the CRC port.
- Distillation to the teacher's u and A uses weight δ = 1 for the first 10% of steps, falling linearly to 0 by 30%.
- Forward XM [R3]: K prior candidates, a hard minimum per window of **summed** token NLL, and gradient only to the
  winner.

**Implementation choices:**
- The recursion runs in parallel over all concept boundaries, with causal attention along the concept axis (the
  causality test passes for all variants).
- Z enters additively: A_Z·Z into h_0 and W_Z·Z into every fL input. The spec's attention-prefix tokens P_Z are not
  used.
- The recursion re-adds h, e and Z on every low step, as in the HRM/GRAM residual injection. With pre-norm residual
  blocks, the raw ‖h‖ grew from 12.4 to 53.3 over 2 transitions (CPU smoke test). Post-norm (RMSNorm) was therefore
  added after every fL and fH call, as in HRM/TRM's post-norm layers, before any Phase B training. ‖h‖ per transition
  is still measured in `eval_b.py`.
- R=2 and n_L=2 (6 block calls per query), with full unroll and no deep-supervision truncation. GRAM's truncated
  surrogate is not used.
- Window lengths are uniform in {1,2,3,4} chunks, re-drawn per row and per step.
- KL is in per-token units: Σ KL per row ÷ that row's scored tokens, then the 0.5/0.5 domain mean.
- β_Z = β_ε = 0.1 makes this a weighted ELBO. GRAM used β = 0.04–0.5 with KL balancing; we use no balancing.
- XM uses K=2 in FLOP-efficient mode. The candidate score has η_c = η_q = 0 (token NLL only).
- Cross-window coupling through decoder attention is ignored in per-window winner selection (declared approximation).
- Variants and V2a capacity:
  - V2 `det`: no Z, no ε;
  - V2a `adapter`: one non-recursive block, no Z or ε. V2a has 1 block vs V2's 2 shared blocks (V2 does 6 calls).
    This is a capacity difference, so V2a is a check on recursion, not an exact match;
  - V3 `z`: Z without ε;
  - V6 `gram --xm 1`.
- Equal steps are not equal compute: a V6 step costs about 2.5–3× a V2 step (measured under contention: 1.7 s vs
  0.68 s).
- `xm_win_frac0` is a slot-win frequency and is not a mode diagnostic (review §5.5). At initialisation it is about 0.75
  from tie-breaking.

## Phase C

- The blend ĉ^λ = (1−λ)ĉ_old + λĉ_new is applied to every port. λ=0 recovers the teacher exactly, since `plans`
  returns the teacher path.
- Phase B already trains at λ=1 through the frozen decoder (spec B7), so the blend schedule is an evaluation curve, not
  a training curriculum. Continued training (`phase_b.py --init`) runs at λ=1 without distillation.

## Phase D (`phase_d.py`)

**Discovery:**
- metacontroller with a causal switch β = σ(MLP([e_m; Z_{m−1}]));
- a non-causal bidirectional GRU proposal posterior over [e_m; c_{m+1}];
- convex integration Z_m = (1−β)Z_{m−1} + βZ̃_m, forced β=1 at the first walk boundary;
- loss = NLL + λ_D·KL(q_D ‖ Phase B prior), ungated (review §7.3), with λ_D = 0.1;
- the frozen core runs its prior guidance.

**Metacontroller differences from [R5]:** [R5] writes a low-rank residual U_t through a hypernetwork. Here the control
acts through the core's already-learned Z pathway (spec D1), and the switch reads e_m rather than a residual-stream
activation.

**RL:**
- Gaussian policy = the Phase B prior mean and log-scale plus a zero-initialised residual MLP, so RL starts exactly at
  prior sampling.
- The switch is the frozen hardened β (κ=0.5), forced at the first boundary.
- REINFORCE with a learned value baseline, γ=1, reward = verifier success.
- Greedy tokens; guidance noise is environment noise.

## Phase E (`phase_e.py`)

- Corrected loop (review §8.3): one ΔZ per episode, Z held over the walk, Adam, U=10, regeneration after every
  update.
- Reward 0/−1; loss = Σ log p of the rejected walk.
- Rows that are accepted are frozen.
- Fixed guidance noise first, then a fresh-noise variant whose gradient replays the noise used for that attempt.
- Baselines at U+1 generations: prior-Z resampling, token sampling at T=1, and a norm-matched random ΔZ.

## Phase A findings recorded at the time (11:10)

- **GraphNav lookup emergence.** Node NLL ran 3.198 (3k) → 3.134 (4k) → 2.315 (5k) → 2.204 (6k) → 2.153 (7k) →
  0.040 (8k). The two-token conjunctive in-context lookup appeared as a phase transition: a 2.1-nat drop within
  1,000 steps.
- **NCP's own inference protocol** (predicted concepts fed back into the concept module) costs +0.217 nats on text
  and +0.048 on graph against refresh (actual pooled history of the observed or generated tokens). This is a result
  about NCP at this scale and justifies our refresh deployment protocol.
- **Phase A exit-gate item "concept forecasts beat a declared simple predictor": passed.** Per-feature MSE: text
  NCP 0.477 vs linear 0.508, mean 0.593, copy 1.057; graph 0.377 vs 0.430, 1.152, 1.386.
- **Null injection inflates route effects.** `zero` costs +0.137 nats on text against `mean`'s +0.089, confirming
  v41's warning.

## V3 KL trajectory (posterior dominance)

KL_Z (per token, before the β_Z = 0.1 weight) rose from 0.165 at step 800 to 0.47 at step 5900 as δ decayed to 0.
That information costs 0.047 nats/token in the weighted objective and buys the posterior path about 0.3 nats
(posterior graph NLL 0.0036 at the end of training).

## Z convention at header positions (unified 14:00, advisor consult 16)

The core attends over all 48 concept positions, so header-position Z affects walk states. Everywhere (eval_b, Phase D
discovery and rollouts, Phase E), header positions 0..29 hold one prior draw from position 0. Walk positions follow the
phase's control schedule. Phase E's ΔZ is added to walk positions (≥ 30) only.

## Inference cost per token (analytic multiply-accumulates; T=192; attention scores at average length T/2)

| system | blocks per token | MACs per token |
|---|---|---|
| V0 | 8 token blocks + LM head | ≈ 2.16M |
| NCP | 6 token blocks + 2 concept blocks at 1/4 rate + PQ heads + port + LM head | ≈ 1.82M (−16% vs V0) |
| transplanted V6 | 6 token blocks + 6 recursive block calls at 1/4 rate + prior MLPs + LM head | ≈ 2.03M (≈ V0) |

The GRAM-style recursion (R=2, n_L=2) buys back nearly the whole NCP saving. Measured training throughput is in
the Hardware section.

## Notes on eval_b quantities

- The posterior path in `nll_modes` always uses the new core directly (λ=1 quantity) and does not pass through the
  blend, so blend-curve files report the same posterior value at every λ. Oracle, mixture and prior paths respect λ.
- Oracle and mixture text Δs in files written before 15:33 are mean-corrected by +0.0511 (see PRE_REGISTRATION). Their
  SEs are the uncorrected paired SEs (approximate). B_v3 and B_v6 are re-evaluated (teacher-forced only) in slack to
  give exact SEs.
