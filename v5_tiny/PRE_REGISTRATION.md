# v5_tiny pre-registration

Written 2026-10-03 ~09:15 IST, before any Phase A training run. The program is the five-phase plan in
`NCP_GRAM_Z_Five_Phase_Architecture.md`, narrowed as review_v5 §9 recommends. Every phase runs (the owner asked for all
five). D and E results are labelled with whether their prerequisite gates passed.

## Scope and honesty statements

- This is a 2–3M-parameter mechanism experiment at d=128 on a MacBook Air M2. It does not reproduce source scale.
- ARC-AGI is not attempted. GRAM needed 5 days on 8 GPUs for it, and a 2M model in 24 h cannot learn it meaningfully.
- HC-DLM [R7] is not available locally and was not read. The diffusion core (spec C3) is deferred, as review_v5 §6.4
  recommends.
- Paper mechanisms, our adaptations and implementation choices are separated in METHODS.md.

## Data (fixed for every variant)

- **Tokenizer:** ByteLevel BPE with 2048 entries, trained on 200k TinyStories stories. 33 reserved specials cover
  pad, eos, the GraphNav markers, 24 node tokens and 3 colour tokens. The digit splitter of the earlier recipe was dropped
  (TinyStories has few digits).
- **Text:** one TinyStories training shard (133.6M tokens) and the TinyStories validation split (5.35M tokens). Training
  uses random 192-token windows. Evaluation uses 384 fixed non-overlapping validation windows.
- **GraphNav (symbolic):** each instance is a random colour-functional graph (10 nodes, 3 colours) in the header, with
  goal and start given. The walk follows a hidden rule sequence: 2–4 segments, each segment holding one colour for 2–4
  hops. Each hop is exactly k=4 tokens. Measured on 500 instances (`data/graphnav_stats.json`):
  - multimodality: a median of 74 of the 2,322 rule sequences reach the goal at first arrival; 0% of instances have
    exactly one;
  - valid walks: median 1.2M;
  - colour-switch rate per hop: 0.254; mean run 2.86 hops;
  - goal-blind floor (the generator's rule policy, ignoring the goal): 5.2% success.
- **Verifier:** reads the header only, so it is deployment-accessible. Hops must be valid edges; `<done>` must come at
  the goal; at most 16 hops.
- **Mix:** 50% text rows and 50% GraphNav rows (32 + 32 per batch of 64 × 192 tokens). By scored tokens that is about
  6.1k text and about 1.3k graph per step. Only walk and done tokens are scored on graph rows.
- **Loss reduction:** a per-token mean within each domain, then 0.5·text + 0.5·graph. This up-weights graph tokens
  per token about 4.7×, by design. The concept-level terms use the same 0.5/0.5 domain reduction over scored targets.

## Phase A: NCP foundation vs V0

**Models:**
- NCP: encoder 3, concept 2, decoder 3 blocks; PQ with S=8, V=32 (affine bound 128 of 128). 2,022,144 parameters.
- V0: 8-block transformer. 1,968,256 parameters (2.7% fewer; not exactly matched).

**Training:**
- AdamW with lr 2e-3, cosine schedule, 300 warmup steps, 18,000 steps (221M row-tokens), seed 0.
- α (NCP-MSE, per-feature) = 1 and β (VQ) = 1.
- λ_q = 0: there is no code cross-entropy, per review §5.4.
- The codebook trains with 10× lr and no decay.

**Measurements:**
- validation NLL for text and graph, and graph NLL split into colour, node and done tokens;
- free-generation GraphNav success, greedy and at T=1;
- concept-forecast MSE against three baselines: copy-last, train-mean and a linear predictor;
- codebook usage.

**Route-authority gate (pre-registered):**
- Interventions:
  - `mean`: both routes replaced by training-mean vectors;
  - `swap`: plans from another row of the same domain;
  - `pq_mean` and `crc_mean`: one route at a time.
- `zero` is reported only as the comparison v41 found misleading; it is not a gate.
- **Route used** if `mean` raises text NLL by ≥ 0.02 nats or graph-walk NLL by ≥ 0.05 nats, in either case with
  paired SE such that Δ > 2 SE.
- If the gate fails, B–E test a channel with little authority. That is reported as a finding, and the program continues.

**History protocol:**
- NCP trains on clean pooled history.
- The paper feeds predicted concepts back at inference. We evaluate teacher-forced NLL under both clean (refresh) and
  predicted-feedback history.
- Our primary deployment protocol is refresh: the encoder runs on generated tokens, so actual pooled history is
  causal and available.

## Phase B: replacement ladder (frozen encoder, decoder, PQ heads and codebook)

| Variant | What it adds |
|---|---|
| V2 | deterministic GRAM-style recursive core (R=2, n_L=2) |
| V2a | adapter-only: one non-recursive block |
| V3 | external Z (d_z=8) with posterior and prior, no guidance noise |
| V6 | Z + GRAM guidance noise + Forward XM (K=2, hard min per window) |

V4 (no XM) and V5 (XM only) are run only if time allows.

**Loss reduction:**
- Reconstruction is the per-token mean NLL in the same domain-balanced form.
- KL terms are converted to per-token units: KL_ε per boundary ÷ 4, KL_Z per window ÷ (4 × window length). β = 1
  would then be the exact ELBO per token for text rows.
- We use β_Z = β_ε = 0.1, a declared weighted ELBO (GRAM used 0.04–0.5).
- The XM candidate score is the **summed** token NLL over the window (review §5.3).

**Z windows:** random lengths in {1,2,3,4} chunks, so the durations used in D are in distribution.

**Gate for advancing a variant:**
- pure-replacement (λ=1) text NLL within 0.05 nats of NCP;
- graph-walk NLL within 0.05 nats of NCP.

## Pre-registered predictions (written before results)

1. **P-NTP (Z):** an autoregressive decoder already resolves multimodality token by token (XM F.1). Z's value should
   appear in controllability, not in prior-sampled NLL. Specifically:
   - branch consistency of the colour held under a fixed Z;
   - Z-swap changing the next colour;
   - D/E reward.

   Prior-sampled NLL of V3/V6 is predicted to be within ±0.03 of V2. "No NLL gain" is the expected outcome, not a failure.
2. **P-posterior:** posterior-path NLL < prior-path NLL. A large gap with a poor prior means target-informed training
   is not transferring.
3. **P-route:** given v41's finding (plans barely used when the decoder has a direct token path), route authority
   on text is predicted to be small (< 0.05 nats). On GraphNav colour tokens it may be larger, because that is where
   the next-chunk uncertainty lives.
4. **P-control:** in V3/V6, holding Z fixed over a window gives higher within-window colour consistency than
   resampling Z per chunk, when measured with fixed guidance noise and greedy tokens.

## Phase C

- Blend curve at λ ∈ {0, 0.1, 0.3, 0.6, 1}.
- λ=0 must reproduce NCP exactly; this is a test.
- Continued training at λ=1, then free generation.
- Exit gate: text NLL within 0.05 nats of NCP; GraphNav greedy success within 5 points of NCP; no growth of residual
  norms across chunks.

## Phase D (conditional on the V3/V6 control gate P-control)

**Discovery:**
- metacontroller over Z with a causal switch β and a non-causal demonstration posterior;
- KL to the Phase B prior; frozen core run with its prior guidance.
- Report the soft-vs-hardened gap (κ=0.5) before RL.
- Report switch alignment with the true hidden colour switches as precision and recall against GraphNav's segment
  boundaries.

**RL:**
- Gaussian policy π(Z|s_j) at hardened switch points, with REINFORCE plus a learned baseline and γ=1.
- Reward is the verifier's success.

**Comparisons at equal attempts:**
- prior-sampled Z with the same switch;
- always-switch;
- fixed duration (every 3 chunks);
- token-level sampling.

**Exit:** policy pass@1 beats prior-Z pass@1 by > 2 SE on 512 held-out instances.

## Phase E

- GradCuit on a single ΔZ offset per episode, using the corrected loop of review §8.3.
- Reward is 0/−1 from the verifier; U = 10 updates; Adam with lr ∈ {0.01, 0.1, 0.3} (source 1e-3 is for [B,N,d]
  states).
- Fixed noise stream and greedy tokens first, then fresh noise.

**Baselines at the same number of generations (U+1):**
- resampling Z from the prior;
- token sampling at T=1.

**Exit:** success-within-budget above both baselines by > 2 SE.

## Seeds and statistics

- Paired SE over items is used for all gates.
- A second training seed for NCP and V0 is run if time permits; each claim states its seed count.
- Many tokens from one item are not independent trials.

## Contingencies, fixed before the Phase A launch (advisor consult 4)

1. **Node-lookup emergence.** At the step-4000 eval of NCP s0, graph node-token NLL must be < 2.0, meaning lookup has
   started. If it is not, NCP and V0 are both extended to 30,000 steps. The extension is restarted from the
   checkpoint with a new cosine schedule over the remaining 12,000 steps, and the B budget shrinks by the same wall
   time. If node NLL is still ≥ 2.0 at 30,000 steps, GraphNav is reported as not learned at this scale, and B–E run on
   it anyway with that label. Text results stay valid either way.
2. **Concurrent throughput.** At step 500 of the concurrent NCP + V0 runs, if the projected finish time is later than
   11:45 IST, both runs are cut to the step count that finishes at 11:45 (same count for both, recorded in the log).
3. `codes_used = 0` at restart steps is a logging artefact: the restart zeroes `usage` on the step the log reads it.
4. `plan_stats` for the route gate: means of chat and A per domain, over 50 training batches (Mixer seed 777), from
   the trained checkpoint. Training data only, never validation.
5. The three concept-forecast baselines (copy-last, train-mean, linear predictor) are computed post hoc on the saved
   checkpoint.

## Additions before the Phase B launch (advisor consult 5)

- **Phase B/C gate mode.** The 0.05-nat gate uses the **deployable** path: prior-sampled Z and ε, averaged over 4 draws.
  Posterior NLL and prior-mean NLL are diagnostics only.
- **Uniform step count.** All four B variants run the same number of steps on the same data order (Mixer seed 2000)
  with the same window generator. Equal steps are not equal compute: V6 costs about 3× V2 per step, and METHODS records
  the ratio.
- **Raw recursion norm.** ‖h‖ after each high transition is measured on held-out batches in eval_b (the training log's
  `h_norm` is post-readout and cannot show growth).
- **Slot wins.** `xm_win_frac0` is a slot-win frequency (review §5.5), not a mode frequency, and is never reported as a
  diagnostic.

## Deviation from contingency 2 (09:49, advisor consult 7), recorded as a deviation

The measured concurrent rate was 0.6 s/step per job (step 500→1000), against 0.174 s/step for one job alone, which
projects both runs to about 12:35. Applied literally, contingency 2 would cut both to about 5,000 steps. Instead the
**arrangement** was changed and the step count kept at 18,000 for both:
- V0 (MPS) was killed at step about 1,000.
- NCP continues alone on MPS.
- V0 was restarted from scratch on CPU (eager, 4 threads, same seed).

Consequence: NCP and V0 no longer share batch order (Mixer RNG state is not checkpointed), so the NCP-vs-V0 comparison
is unpaired on batches. That comparison was never a gate. The step-4000 node-lookup decision is made on NCP alone; V0's
node NLL is reported when it arrives.

## Node-lookup diagnostic (10:00, advisor consult 8; written before its first eval)

At step 3000, NCP node NLL is 3.198, flat since step 1000 and above ln 10 = 2.30. Colour (0.65) and other
position-relative tokens are learned; nothing that needs node identity matched into the header is. The step-4000
contingency is expected to fire. Its action (extend to 30k) cannot start before step 18,000 anyway, so the
extend/rerun decision is **deferred to the 18k checkpoint** pending this diagnostic.

**Diagnostic arm:**
- TransformerLM, 4 layers, GraphNav rows only, B=32, lr 2e-3, warmup 100, 2000 steps, CPU, 2 threads.
- Hop format (ii): `[cur_node, colour, dst, <sep>]`, which mirrors the header entry `[v, c, u, <sep>]`, so predicting
  dst is classic induction.
- Node NLL logged every 250 steps.

**Decision rules:**
- (ii) node NLL < 2.3 by step 1000 and NCP node NLL still ≈ 3.2 at 5000–6000 → **the format is the cause**. Switch to
  (ii) and rerun A from scratch. Alignment note: the colour becomes the chunk's second token, still covered by ĉ_j at
  positions 4j+3..4j+6. The verifier adds the check `cur == previous node`.
- (ii) also flat (> 3.0) at 2000 → **plumbing, not emergence**. First, train a 1-layer model to predict a uniformly
  random header node from the header alone; it must reach about 2.30 within about 300 steps.
- (ii) and NCP both start falling → emergence is just slow; extend per the contingency.
- Confound noted: the diagnostic is graph-only while NCP is mixed. A graph-only arm in format (i) is added if CPU time
  allows.

## Revised diagnostic (10:18, advisor consult 9; written before format (iii)'s first eval)

- Format (ii) is withdrawn. Its node metric averages trivially copyable `cur` tokens with `dst`, and its colour NLL
  sat at chance (1.10 ≈ ln 3), probably because walk chunks were locally identical to header entries.
- Format (iii) is `[cur, colour, dst, <to>]`: header-mirroring offsets, plus a `<to>` terminator as a local walk cue.
  The verifier checks `cur == previous node` and terminator `<to>`.
- Metrics are now per chunk offset: 0 (cur or done), 1 (colour), 2 (dst), 3 (terminator).
- dst reference levels: 3.18 any node; 2.30 header set; 1.10 match on v only; 0 full lookup.
- Arm (i) keeps running as the clean control: its node metric is dst.

**Rule for (iii), at step 2000:**
- dst < 1.10 → the format is the cause; rerun A in (iii).
- dst in (1.10, 2.30) and still falling → extend the diagnostic to 4000 before deciding.
- dst ≥ 2.2 and flat → the conjunctive in-context lookup is not learnable at this scale and budget; use the fallback.

**Fallback (fixed now):** GraphNav with a fixed small pool of graphs (e.g. 16), selected by a header ID token. The
lookup becomes weight memorisation instead of in-context reading. Multimodality, the persistent hidden colour and the
header-only verifier are kept (the verifier gets the graph from the ID). This explicitly removes the in-context-read
demand; the Phase B/D/E questions (does Z control the colour branch; does RL over Z reach goals) do not depend on it.

Cost accepted: two CPU diagnostic arms slow NCP to about 0.30 s/step, finishing about 11:15 instead of about 10:58.

## Outcome of the node-lookup contingency (10:24)

The NCP s0 run, in format (i) and mixed data, logged node NLL of 3.198 at 3k, 3.134 at 4k, 2.315 at 5k, 2.204 at 6k,
2.153 at 7k and **0.040 at 8k**. Done NLL was 0.09 and colour 0.588 at 8k. The two-token conjunctive in-context lookup
emerged between 7k and 8k: a phase transition, not a format defect. This matches the pre-registered branch "NCP starts
falling → emergence is slow", and it finished within the original 18k budget, so no extension and no format change.
Format (i) stays for all phases. The diagnostic arms were stopped. Their partial logs are kept in
`runs/diag_lookup_*.jsonl` (format (i) graph-only: dst 2.80 at 1k; format (iii): dst 2.10 at 500 with colour at chance).
The data_v5 `V5_HOP` flag remains but is unused (default "to").

## Revised predictions (10:27, advisor consult 10; before any trained-model route evaluation)

The original P-route text above is kept; this revision supersedes it. Reason: at 8k the decoder's token path reads the
header directly (dst 0.040), and colour (0.588) is solved by positional copy plus switch timing. The bypass is now
demonstrated on GraphNav.

- **P-route (revised):** the `mean` and `swap` deltas on graph-walk NLL are predicted to be near zero (< 0.02 nats).
  Any effect should concentrate on **colour tokens at switch positions** (hops whose colour differs from the
  previous hop), a few percent of scored graph tokens. A graph delta ≥ 0.05 would be a result against prediction.
  Text: still predicted small (< 0.05).
- **P-NTP (strengthened):** colour at a switch is uniform over the 2 other colours given the prefix, and the
  autoregressive decoder resolves it token by token. Z cannot show a value in NLL here. Its value can only appear in
  branch consistency, Z-swap effects and D/E reward.
- eval_a reports colour NLL split by switch vs non-switch hop (from the generator's hidden `hop_cols`; for analysis
  only).
- The smoke-model route numbers (`runs/smoke_ncp`) came from a model without the lookup. They were a harness test and
  are not evidence.

## Corrections and slack plan (10:55, advisor consult 11)

- **Correction to the 09:49 deviation note.** `Mixer(seed=1000+seed)` is a deterministic NumPy RNG, so a from-scratch V0
  with seed 0 sees the **same batch sequence** as NCP, offset by one batch (NCP used one batch for codebook init). The
  pairing is therefore close; the note's claim of "no shared order" was too strong.
- **V0 placement.** The paused CPU V0 was killed. V0 (seed 0, 18k steps) runs on MPS, solo and compiled, immediately
  after the B ladder and before b_eval.
- **In-distribution control schedule.** B trains Z windows of 1–4 chunks. The P-control gate is judged on `hold4`
  (window starts at chunks 30, 34, 38, 42, 46) against `resample` (a new Z every chunk). The single-window `hold` over
  17 chunks is kept as a stress variant only. Phase D adds `prior_hold4` to its baselines. In Phase E, Z0 is the
  per-window (hold4) prior sample and ΔZ is one shared offset added to every window; its resample baseline redraws
  the per-window priors.
- **Slack spending order** (sequential, one MPS job at a time, each only after the main chain's stage has finished):
  1. V4 (gram without XM) and V5 (XM only) at the B step count;
  2. second seed of V3 and V6;
  3. D and E on V3.

## Phase A result-driven but outcome-blind additions (11:12, advisor consult 12; before any B/C/D/E number)

- **Ceiling.** NCP greedy success is 0.893 on the standard GraphNav val set, flat across walk length (0.79–1.0) and
  segment count (0.88–0.92). No generator-property split gives headroom.
- **Hard-goal set (primary arena for D and E).** Same graph and header format; goal = a uniformly random node at
  shortest-path distance ≥ 4 from start (any colours). 512 instances, seed 31; distances 4/5/6 = 478/32/2. NCP base
  success: greedy 0.361, T=1 0.334. This is the base rate measured for difficulty calibration; the set is defined by a
  generator property, not by per-item outcomes. D and E are reported on both sets, each with its base rate.
- **Phase D RL** trains on the hard-goal generator (seed 4000 + run seed), disjoint from the evaluation seed.
- **B step count:** S = 6,000 for all four variants (solo rates: V2 0.19, V2a 0.15, V3 0.22, V6 0.46 s/step), run
  sequentially.
- eval_b adds `pq_mean` and `crc_mean` interventions on the replacement. Phase A showed CRC dominance (crc_mean
  +0.30 vs pq_mean +0.07 nats on colour at switches), so Z-control is never described as "controlling concept codes"
  unless the PQ route carries it.

## V2 gate check and C selection rule (11:47, advisor consult 13; before the gate-check number)

- **Gate check on V2** (`--tag gatecheck`, no control tests): text Δ ≤ 0.05 and graph Δ ≤ 0.05 → chain unchanged. If
  either fails, V6 is not killed. Instead, b_eval moves ahead of V0, and the diagnosis checks whether V2's `rec`
  trends upward after δ reaches 0 (step 1800). If it does, the readouts drifted out of the teacher's regime and a δ
  floor is the fix.
- **C selection rule:** C continues V6 and V3 regardless of their B gate status, and the gate status is carried as a
  label on every C/D/E number. D and E run on C_v6 (the full hybrid); D/E on C_v3 are slack item 3.

## After V3's gate failure (12:12, advisor consult 14; before V6's result)

- **V3 gate: failed.** Prior-sampled Δ is text +0.345 and graph +0.339; prior-mean Δ is +0.147 and +0.228. The
  posterior path is −0.180 and −0.110; oracle best-of-8 is −0.052 and −0.031.
- **P-NTP: failed** by an order of magnitude. **P-posterior: passed.** Z has authority (the decoder executes
  posterior plans), but the prior cannot supply them. At prior-mean Z, replacing V3's plans by constants *improves*
  graph by 0.156 and colour-at-switch by 2.33 nats. This is review §B7 row 1, "posterior excellent, prior poor",
  observed.
- **Metric note, not a gate change.** Four-draw E[−log p] penalises a committing prior by construction. The gate
  stays as registered.
- **V6 reading (fixed now):**
  - V6 passes the gate → XM is what makes prior samples deployable;
  - V6 fails → posterior dominance at β_Z = 0.1 is the cause, regardless of XM.
- **Phase C objective for both cores** is the spec C2 default: L_C = XM + λ_var·L_var with δ = 0, K = 2. This is the
  spec default, chosen after seeing V3's gate. C_v3 therefore runs `--variant z --xm 1`; the code previously
  ignored `--xm 1` for variant z, which is fixed.
- **Mixture NLL** is added to eval_b as a **post-hoc diagnostic** (not a gate): per window,
  −log((1/8)·Σ_i exp(−ℓ_i)) over 8 prior samples' summed window NLL (XM's smooth form).
- **Slack order (revised):**
  1. V5 (XM only, no posterior);
  2. V3 with β_Z = 1 (exact-ELBO weighting; direct test of the mechanism);
  3. V4 (ε without XM);
  4. second seeds;
  5. D/E on V3.

## V6 gate result and C success criteria (13:08, advisor consult 15; before b_eval's V3/V6 control numbers)

- **V6 gate: passed.** Prior-sampled Δ text +0.025, graph +0.017; prior-mean +0.007 and +0.002; posterior −0.154 and
  −0.110; oracle best-of-8 −0.115 and −0.049; mixture-of-8 (post-hoc diagnostic) −0.050 and −0.000. Under the
  registered reading this attributes deployability to XM, **with the caveat** that V6 also adds ε guidance. The
  attribution rests on V4 (ε, no XM) and V5 (XM, no ε), which now head the slack order.
- **Mechanism signal for the V4/V5 pair:** KL_Z trajectories (V3 rises; V6 holds near 0.19). If V4's KL_Z also
  holds, ε provided the cheaper channel; if it climbs like V3's, XM held the prior calibrated.
- **Phase C transplant success (spec C4), all at λ=1:**
  1. prior-sampled NLL within 0.05 nats of the teacher on text and graph;
  2. valid free generation: GraphNav greedy success on the standard set within 5 points of the teacher's 0.893, with
     hold4 prior sampling;
  3. hard-goal hold4 success within 5 points of the teacher's 0.361;
  4. stable norms: raw ‖h‖ change between the two transitions < 1.0;
  5. usable controls: P-control persistence delta (hold4 − resample) > 0, and Z-swap first-colour difference rate
     > 2 SE above ε-only.
- **Correction (13:10).** The premise that V6's KL_Z "held near 0.19" was wrong; it came from a snapshot up to step
  1500. Full trajectories: V3 0.16 → 0.465; V6 0.185 → 0.43 (steps 600 → 5400). Both posteriors load Z about
  equally, yet only V6's prior is calibrated, so KL_Z cannot discriminate V4 from V5. The attribution rests on the
  **gate results** of V4 (ε, no XM) and V5 (XM, no ε): whichever passes identifies the component that makes prior
  samples deployable.

## Phase A comparison and the α=0 ablation (14:40, advisor consult 17; before the ablation runs)

- **V0 vs NCP (seed 0 each):**
  - V0 is 0.033 nats better on text (1.780 vs 1.813).
  - V0 is 0.42 nats worse on GraphNav (0.536 vs 0.113), with node NLL 1.85 vs 0.0007 and free-generation success
    0.0 vs 0.893.
  - Colour-at-switch is similar (1.36 vs 1.21); only the node-identity path differs.
  - Learning dynamics: V0's node NLL drifted steadily (2.06 → 1.85 over 10k → 18k), while NCP was flat at 3.2 and
    then dropped 2.1 nats between 7k and 8k.
  - Both are stated together. The symbolic task is where the two architectures separate.
- **Ablation A_ncp_a0_s0:** NCP with α = 0 (no concept-prediction loss), with β, the architecture and seed 0
  unchanged. **Readings (fixed now):**
  - emerges by about 8k like full NCP → the lookup comes from the encoder→decoder architecture, not the concept
    objective;
  - stays flat or drifts like V0 → the concept objective drives it (consistent with NCP's learning-efficiency claim);
  - emerges late (12k–16k) → both contribute.
- **Slack order (final, recorded):**
  1. α=0 ablation, which answers the owner's question about what the concept route does;
  2. V4;
  3. V5 (V4 and V5 attribute Phase B's deployability);
  4. V3 with β_Z = 1;
  5. second seeds of NCP and V0;
  6. D/E on V3.

## D/E reading against the no-Z reference and the token-search baseline (14:50, advisor consult 18)

- **Reference line.** V2 (no Z) reproduces the teacher in free generation: standard 0.889 (teacher 0.893); hard-goal
  0.379 ± 0.021 (teacher 0.361; within one SE, not an improvement).
- **Readings, fixed while D/E are outcome-blind:**
  - policy or GradCuit > V2's 0.379 on hard-goal by 2 SE → Z-control adds capability beyond a deterministic
    transplant;
  - policy > its own prior but ≤ V2 → RL or optimisation only recovers the cost of stochastic control;
  - V6 prior-hold4 on hard-goal < V2 → Z costs something at the prior, and D must pay that back first.
- **Equal-budget token search** (run after E): V2 with T=1 sampling, best-of-11, same header-only verifier, both sets
  (`code/v2_bestof.py`). If GradCuit on V6 beats V6's internal baselines but not V2 best-of-11, the statement is
  "latent optimisation does not beat token search at equal budget on this task".
- eval_b now reports the P-control persistence delta paired over instances with ≥ 2 hops (with SE), and Z-swap minus
  ε-only first-colour differences, paired with SE.

## Authority vs calibration (14:50, advisor consult 19; before V3's control numbers)

- V2a passes the gate (+0.004/+0.004): the replacement contract is easy at this scale. Recursion (V2) is worth
  0.008–0.009 nats on held-out data, confounded by capacity (1 block vs 2 shared blocks).
- **Reading fixed now:** a core can pass the P-control gate (persistence and Z-swap effects) while failing the NLL
  gate and showing low success. Controls can be **effective and miscalibrated at once**. Authority (does Z change
  behaviour?) and calibration (does the prior choose good Z?) are separate properties, and both are reported for every
  Z-core.

## V3 controls read (15:00, advisor consult 20; before V6's control numbers)

- **Z-swap chance reference.** Two independent draws over 3 colours differ with probability 1 − Σp² (0.667 if
  uniform). V3's 0.658 is at that level, so the swap rate measures authority × prior spread and here shows a
  near-uniform prior: the calibration failure, visible in the control test. Authority is established by the posterior
  path (−0.18/−0.11) and the prior-mean route split (+2.3 nats at switches when plans are removed), not by the swap
  rate.
- **Teacher-calibrated reference** (run when V6's eval lands): the NCP teacher at T=1 with two seeds on the 512
  standard instances, reporting first-colour and full-sequence differ rates.
- **V6 signature (fixed now):** swap rate below V3's and near the teacher reference, with the prior-mean route split
  still showing informative plans. A lower swap rate alone would be ambiguous.
- **P-control: failed for V3** against the original wording. Persistence paired delta (hold4 − resample) is
  −0.015 ± 0.006. Both conditions sit at the token path's natural rate (0.725/0.739 vs 0.746 in the data): the
  decoder's 4-back copy supplies persistence, and Z acts at uncertain positions (first hop, switches). "Z as a
  persistent action" does not exist out of the box, which is the condition ETA's discovery stage targets.
- **Phase D primary discovery metric:** switch precision and recall against the true colour switches (switch
  timing). Z persistence between switches is secondary.
- **Bookkeeping:**
  - hold-stress 0.631 is out of distribution (one Z over 17 chunks vs trained windows of 1–4);
  - V3 hard-goal 0.283 < V2's 0.379, the live branch "Z costs at the prior; D must pay it back";
  - V3's mixture-of-8 is +0.090/+0.075: the prior is miscalibrated enough that even the mixture does not recover.

## V6 controls read (15:10, advisor consult 21; before C's results)

- **V6 signature: matched.** Swap rate 0.516, below V3 (0.658, at chance) and the teacher's T=1 spread (0.596), while
  prior-mean plans are informative (+0.39 at switches when removed).
- **C criterion 5: half-passed.** Authority passed (Z-swap − ε-only = +0.498 ± 0.023). Persistence failed
  (+0.0004 ± 0.0036). D carries the label "P-control failed as worded; authority gate passed".
- **Prior-mean greedy on the standard set is 0.918** (teacher 0.893, V2 0.889; unpaired, about 1.4 SE, not claimed
  significant). D adds a `prior_mean` controller (deterministic Z = μ(e_m) at every boundary). **RL "adds capability"
  only if the policy beats prior-mean greedy by 2 SE on hard-goal.** Beating only the sampled prior is "recovers
  sampling cost".
- Hard-goal success is now measured in sample and mean modes, with per-instance flags for paired comparisons.
- **ε-only** moves the first colour in 1.8% of instances: guidance noise carries almost no behavioural variation. This
  is a first hint that ε is not where V6's deployability came from; it is behavioural, not the gate, and V4/V5 decide.
- hold-stress: V6 0.822 vs V3 0.631. It is out of distribution and reported as such.
- resample (0.867) ≥ hold4 (0.854): per-chunk Z redraws do not hurt. Consistent with Z acting at uncertain positions.
- The C λ=0 blend eval is an exact-reproduction test: Δ must be 0.000.

## Measurement correction (15:33)

The λ=0 blend test reproduced the teacher exactly on the deployable paths (prior-sampled and prior-mean Δ = 0.000).
It also exposed a bug: the oracle best-of-8 and mixture-of-8 window sums omitted token positions 0–2 (which read no
plan). On text rows those positions are scored, so every oracle and mixture **text** Δ was biased by exactly
−0.0511 (the teacher's NLL on those positions per row ÷ scored tokens, identical across modes; it matches the λ=0
oracle text value of −0.0511 to 5 digits). Graph values were unaffected (positions 0–2 are header there). The code is
fixed for all later evaluations, and `build_tables.py` adds the exact correction to earlier files.

**Corrected numbers:**

| core | measure | before | corrected |
|---|---|---|---|
| V6 | oracle best-of-8, text | −0.115 | −0.064 |
| V6 | mixture-of-8, text | −0.050 | **+0.002** |
| V3 | oracle best-of-8, text | −0.052 | −0.001 |
| V3 | mixture-of-8, text | +0.090 | +0.141 |

**The earlier statement "V6's mixture-of-8 beats the teacher on text by 0.05" is withdrawn.** V6's 8-sample mixture
matches the teacher on text (+0.002) and on graph (−0.000).

## C_v6 scored; D and E bars (15:46, advisor consult 23; before D/E results)

**C_v6 against the C4 criteria: 3 pass, 1 half, 1 fail.**

| criterion | result |
|---|---|
| 1. prior-sampled NLL | +0.021 text, +0.017 graph: pass |
| 2. standard success | hold4 0.857 (teacher 0.893): pass |
| 3. hard-goal success | hold4 0.363 (teacher 0.361): pass |
| 4. norm stability | 12.68 → 13.73 (change 1.05 > 1.0): fail |
| 5. controls | authority +0.557 ± 0.022 passes; persistence −0.010 ± 0.004 fails: half |

Notes on criterion 4: it is a point estimate (one 64-row batch, one seed), and post-norm bounds ‖h‖ by the learned
RMSNorm gain, so the criterion can now register only bounded variation. B_v6 measured 0.68. The criterion stays
failed as worded.

**What C's 1500 steps did, relative to B_v6:**

| measure | B_v6 | C_v6 |
|---|---|---|
| prior-sampled Δ (text / graph) | +0.025 / +0.017 | +0.021 / +0.017 |
| hard-goal, sampled | 0.338 | 0.363 (paired vs V2: −0.016 ± 0.021) |
| prior-mean greedy (standard) | 0.918 | 0.910 |
| swap rate | 0.516 | 0.561 |

Modest movement. The transplant was effectively done in B, which already trained at λ=1 through the frozen decoder.

**Ordering of controllers:** prior-mean (0.910 / 0.416 hard-goal) > resample every chunk (0.891) > hold4 (0.857 /
0.363). C_v6 prior-mean vs V2 on hard-goal (paired): +0.037 ± 0.020, 1.9 SE, suggestive and below the bar.

**D bar:** "RL adds capability" requires the policy to beat prior-mean greedy on hard-goal (0.416 ± 0.022) by 2 SE,
i.e. about ≥ 0.46. Beating only the sampled prior (0.363) means "recovers sampling cost". The live switch baseline is
`always` (redraw every chunk), given resample ≥ hold4.

**E bar:** success within U+1 generations is also compared with prior-mean greedy (one generation, about 0.416). Added
arms: `gc_from_mean` (Z0 = μ, middle lr 0.1), `token_T1_from_mean` (matching token search) and `base_from_mean`.

## C_v3 rescue and its control (15:52, advisor consult 24; before the control runs)

- **C_v3** (B_v3 continued 1,500 steps with XM, δ = 0, seed 1) **passes the gate**: prior-sampled Δ +0.022/+0.020
  (B_v3: +0.345/+0.339); prior-mean +0.014/+0.005; mixture-of-8 +0.003/+0.002; swap rate 0.525 (B_v3 0.658). KL_Z
  ended at 0.475, as loaded as B_v3's. XM calibrated the prior without reducing the information the posterior puts
  into Z.
- **Control `C_v3_noxm_s0`:** the same continuation (1,500 steps, δ = 0, seed 1) with `--xm 0`. Run first in slack.
  Readings:
  - passes → the extra training did it, and the XM attribution collapses;
  - fails like B_v3 → **XM is the cause**, with no ε anywhere in the comparison.
- With the control failing, the "XM or ε" caveat on V6 is resolved in favour of XM. V4 and V5 stay informative (does
  ε do anything alone; does XM work without a posterior) but are no longer load-bearing.
- **Slack order:** C_v3_noxm, V2 best-of-11, nllfix, α=0, V5, V4, β_Z=1.
- **Carried numbers:**
  - C_v3 hard-goal sampled 0.336 vs V2 0.379, paired −0.043 ± 0.020 (2.2 SE; "Z costs at the prior" is live);
    prior-mean 0.369 (−0.010 ± 0.021).
  - Persistence +0.001 ± 0.003: null for the third and fourth cores (V3, V6, C_v6, C_v3).

## Phase D discovery result and the gated-KL arm (16:02, advisor consult 25)

- **Discovery (ungated KL, as registered): failed on the primary metric.** β saturates near 0.99 at every boundary.
  Switch precision is 0.253 (the base rate), recall 1.0, and the fraction of intermediate β is 0. The soft-hard gap
  of 0 is trivial because β is saturated. Mechanism: with ungated KL every proposal is charged whether or not the
  switch fires, so refreshing Z every chunk is free and helps NLL; β → 1 is the optimum. This is the same finding as
  the P-control null: Z is not where persistence lives, so refreshing it costs nothing behaviourally.
- With β̄ = 1 everywhere, the `prior` and `always` rows of the D table are identical by construction (same seed).
- **Reading for the running RL (ETA's β = 1 regime, where their internal RL failed):** policy > prior-mean 0.416 by
  2 SE on hard-goal would contradict ETA in this setting, because Z here acts per chunk at uncertain positions rather
  than as an extended action. Otherwise the result is consistent with ETA.
- **Post-hoc design arm `D_v6_gated_s0`** (review §7.3: "rederive the objective if switching becomes discrete"): loss =
  NLL + λ_D·Σ_m β_m·KL_m with β **not detached**; everything else, including the seeds, unchanged; then RL with the
  hardened switch.
  - **Quantitative prediction:** per-proposal KL ≈ 0.22 × 4 ≈ 0.9 nats, so a switch costs ≈ 0.09 at λ_D = 0.1. The
    NLL gain at a true switch is about 0.5 nats on the colour token; within a segment it is ≈ 0. **Sparse switching at
    true colour changes is the predicted optimum.**
  - **Readings:**
    - precision well above 0.25 with high recall and bimodal β → temporal abstraction over Z exists once information
      is charged per switch; proceed to RL and the D4 comparisons;
    - β → 0 except at the forced first boundary → held Z does not transfer across chunks (proposals are
      chunk-specific);
    - β → 1 again → refreshing still wins.
