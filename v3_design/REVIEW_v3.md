# Deep review of draft_v3.md: first principles, step by step

**Scope:** challenge the *assumptions* of the v3 design, not its bookkeeping (the §0–§10 section passes, advisor calls #32–#53, already checked consistency). One step per module. Each step records my own first-principles findings *before* the advisor call (a fail-safe), then the advisor's verdict, then what changed in `draft_v3.md`.

**Status:** R1–R7 advisor-reviewed and applied (calls #54–#60). R8 (the overall verdict) is written; its advisor check failed on overload and will be retried.

---

## R1 · Encoder

**Question:** does the encoder, as specified, put the information the concept route needs into C, at d = 256 and 9.64M params?

My findings:
1. **Where the parameters go.** FFNs hold 8 × 786k = 6.29M of the encoder's 9.64M (65 %); attention holds 6 × 262k + 2 × 363k = 2.30M (24 %); Engram projections, pool and gates hold the rest. FFN ratio 4 therefore makes the encoder mostly per-position storage. That is consistent with "more capacity without changing reach", but the only *global* mixing is still 2 F layers of attention (0.73M). The facts the probes need enter G only there.
2. **Checked: L-reach path with read 2.** S1 (10) → Engram 1 (+3 conv = 13) → S2 (23) → S3 (33) → F1 (no L write) → S4 (43) → Engram 2 (no conv, 43) → S5 (53) → S6 (63) → F2 (G only). Reach 63 ✓. Read 2's gate context is 43 ✓.
3. **L gets no processing after S6.** F2 writes G only, so `tok` is L after S6, and the decoder then adds 4 confined layers. Same as v1; not an issue, but it means the *last* encoder layer (F2) serves only the concepts.
4. **Half the encoder's attention budget (F layers) is trained only through the concept route (§5.1).** Early in training, when the route is weak, F1/F2 and the G stream receive little NTP signal. v1 shows the route did train (recall onset at ≈ 1,009–1,217). v3's E5 (CRC G share) watches for the failure. No change proposed.
5. **The F-heads-6 argument is geometric (v1's 6 × 64), not evidential.** No run isolates heads. Acceptable as a judgement; it is labelled one.
6. **Possible gap: no positional signal in F layers or the core except causal masking (NoPE) and short convs.** v1 was the same and learned hop and carry partially, so this is inherited, not new.

My verdict: no design change needed in the encoder. The main encoder risk is the width itself (already gated by E1/E2/E5).

**Advisor (call #54):** the findings hold. Three additions:
1. **The structural risk behind finding 4:** nothing in the encoder's signal says "put the fact into G at the fact position". The retrievability pressure arrives only after the core learns to look back (a two-stage dependency, which is v1's recall lag).
2. Finding 2 verifies reach, and only *indirectly* the read-2 "gate reads L only" contract. Add a direct structural check.
3. The pre-note for R2: after the cap trim the shared FFN is 58 % of the core's per-layer FFN capacity (1,024 of 1,760). That split fell out of the trim; it was never decided.

Stale items: own-1024 in §6.6, 20:15 in §7.3, the seed footnote in §1.2.

**Changed in draft_v3.md:** §3 role line (the two-stage dependency); §3.2 and §3.6 (read-2 gate check); §6.6 own-992; §7.3 milestone time; §1.2 seed footnote.

---

## R2 · Concept core

**Question:** will the 4-layer × ≤ 3-pass core, as specified, do *iterative* work, and is its capacity allocated where retrieval needs it?

My findings:
1. **§4.2 presents one cause for v1's stagnation; there is a second, and v3 makes it more likely.** §4.2 blames *front-loading*: unconditioned random R rewarded finishing in pass 1. The alternative is that **stagnation was data-limited**: if the concept-level computation this corpus needs fits in one pass, extra passes idle whatever the conditioning. v1's pass was 1 layer; v3's pass is 4 distinct layers, so pass 1 alone is already deeper than v1's 4 passes. Under this hypothesis v3 trains fine, passes the angle gate (the table differentiates passes by construction), and still fails M1. M1 catches it, but §4.2 should name both hypotheses. **Diagnostic that separates them:** the per-pass relative update norm ‖ΔS_i‖ / ‖S_i‖ and the learned write gains (1+α) for passes 2–3. Idle passes show both shrinking toward 0; front-loading shows ΔS₁ ≫ ΔS₂, ΔS₃ even in the (0, ⅓) context.
2. **The shared/own FFN split (1,024 / 736) fell out of the cap trim; it was never decided.** Either trim keeps the parameter-matched arm at own-992 (own + shared/4 = 992 in both cases), so the choice is purely about the split. The Mobius paper's evidence (a shared store beats layered FFNs on data efficiency at 7B) favours keeping the shared store larger. The owner's hypothesis (own FFNs hold layer-specific abstractions) favours balance. I propose keeping 736 / 1,024 and recording it as a decision with that reason.
3. **The PQ target lost a third of its information (224 vs 336 bits), and this costs nothing to restore in the compute budget.** c_pred is the plan's look-ahead (v1: c_pred off costs 16 points of carry). The codebook is excluded from compute params. 32 × **256** × 8 gives 256 bits for +32k stored params and ≈ 16k MAC per token. The price: the per-segment NCP range changes (max ln 256 = 5.55 vs 4.85), which loses per-segment comparability with v1/v2. Trade-off: comparability vs target information. I propose keeping 128 by default (v1's geometry, comparability) and naming 32 × 256 as a zero-cost option if E1 lags.
4. **Retrieval capacity rose a lot:** 6 heads × 12 applications = 72 head-applications per concept per forward, against v1's 6 × ≈ 2.5 ≈ 15. If recall still lags, attention capacity in the core is not the bottleneck. That rules one cause out in advance. Worth a sentence in §9's E1 interpretation.
5. **The half-scale first read** (layer 1, pass 1, S = 0) was noted in §10.2; the (t = 0) γ_C entries can compensate. No change.

My verdict: add the alternative hypothesis with its diagnostic (1), record the split as a decision (2), name the PQ option (3), and add the capacity note to E1 (4).

**Advisor (call #55):** findings 2 and 5 hold. Corrections:
- **1 is incomplete.** v1's passes were *identical* (constant update norm, angle → 0), which neither front-loading nor data-limited depth predicts. The mechanism is a **fixed point in direction space**: the GR read normalises S, so once S's direction settles, every pass sees the same input and writes the same vector. It is architectural, the table breaks it, and it is now §4.2's lead argument. That gives three signatures (fixed point / idle / useful), now in §4.7. (1+α) is confounded by weight decay; use the relative update norm.
- **3 is mis-framed.** Bits per concept dimension are unchanged (7 bits per 8 dims). 32 × 256 would be finer quantisation than v1, not a restoration. Rejected, and recorded in §4.6.
- **4:** cite v2 (160 head-applications, slowest onset), not v3's 72. Capacity never predicted onset.
- **New:** (⅔, ⅓) is exact on pairs, not on states; path independence is enforced only at t = 1. Accepted, as in the paper.

**Changed in draft_v3.md:** §4.2 (fixed-point lead, data-limited case), §4.3 (pairs vs states), §4.4 (the split as a decision), §4.6 (32 × 256 rejected), §4.7 (three signatures), §9.2 (onset interpretation).

---

## R3 · Recurrence mathematics (Φ, the table, trajectories)

**Question:** given R2's fixed-point mechanism, does the (t, Δ) design actually get the core *out* of v1's regime, and do the training forces point the right way?

My findings:
1. **v3 starts inside v1's failure regime, by design.** With the table at 0 (identity at init), every pass of the main trajectory applies the same function to the same C, so v3 at step 0 *is* the fixed-point regime. It must learn its way out, and the only force doing that is the gradient on the table (and on the weights through it). LoopFormer does the same (AdaLN-Zero), and it worked there, so zero init is defensible.
2. **But the table is weight-decayed (§4.3, §8.2), i.e. regularised toward the failure mode.** AdamW decay shrinks each entry by lr · wd = 6e-5 per step at peak LR, about e^(−0.25) over the run's integrated LR (6,400–8,850 steps × a mean LR of ≈ 3.3e-4 × 0.1). It is not dominant, but it is a force pointed *toward* pass-invariance, the exact mechanism R2 identified. The original reason for decay (decision #10: "treat like zero-centred norm gains") was about scale hygiene, not about this. **Proposal: put the (t, Δ) table in the no-decay group.** It is 43k parameters on a finite domain of 6 inputs, with no overfitting risk worth regularising. In NanoGPT-style LoopFormer the modulator weight is decayed, but its *output* is a nonlinear function of a Fourier embedding. Here the table entries *are* the output.
3. **Gradient flow across passes is healthy by construction.** Every sublayer is `S + s·f(read(C, S))`, so ∂S₃/∂S₁ contains the identity; v1 measured a pass-1/pass-4 gradient ratio of 1.14. v3_ref T4 confirms nonzero gradient at S₁, S₂, S₃ and C. There is no vanishing risk at 12 applications.
4. **The consistency target moves.** L_scons pulls u_s toward sg(u₃) while u₃ itself changes. It is a self-distillation target (LoopFormer's design). With λ_c = 0.1 it cannot dominate, and L_NCP(u_s) anchors u_s to the VQ target. No change.
5. **The shortcut sees a different input distribution only through the table.** For (0, 1) the same 4 layers must do the whole job in one pass, distinguishable from (0, ⅓) only by 7 gain vectors per layer. That is exactly AdaLN's expressiveness (per-channel scale and gate), so v3 matches the paper's modulator capacity on this domain. No change.

My verdict: one change, the table goes to the no-decay group (2). The rest holds.

**Advisor (call #56):** findings 3–5 hold.
- **1:** reword. v3 starts with no pass-dependent signal, the regime that *converges* to the fixed point; it is not *at* it (first writes differ).
- **2 is sized:** with Adam, decay removes ≤ 10 % of per-step movement, a bias rather than a wall. Decide on principle. **Adopted: no decay on the table**, reversing decision #10. Propagated to §4.3, §8.2 and the `param_groups` pattern. The gains are now bounded only by the clip, with ‖α‖ logged as the check.
- **Missing sentence (now in §4.2):** the shortcut terms do not oppose idling. The idle solution satisfies them all trivially. *The design permits iteration; it does not require it; M1 measures it.*
- **5:** the table's cost is no transfer between pairs. The rare rows still get about 1,900 updates. A Fourier modulator is named as the option if M4 is poor.
- **3:** measure ∂L/∂S₁ : ∂L/∂S₃ at milestones (it is constructed, not yet measured).
- **Finding 6 (cross-reference):** (⅔, ⅓) is exact on pairs, not on states (§4.3, added in R2).

**Changed in draft_v3.md:** §4.2 (the permits-not-requires paragraph), §4.3 (no decay; no transfer), §4.7 (gradient-ratio telemetry), §8.2 and §5.5 (param groups).

---

## R4 · Decoder

**Question:** can the confined 4-layer decoder at d = 256 turn (local tokens + plan) into the next-token distribution without becoming the bottleneck?

My findings:
1. **The output head at d = 256 is a rank bottleneck the draft does not mention.** Logits = zcRMSNorm(D) Eᵀ have rank ≤ 256 over 8,192 tokens (v1: ≤ 384). This is the "softmax bottleneck": a width cost that falls on *language* loss, especially WikiText (larger effective vocabulary, more contexts to separate), and not on recall. It is part of the width confound and a candidate explanation for v2's WikiText gap (v2 4.00 vs v1 3.53 at the ends of comparable stretches). Typical small LMs run at similar or worse ratios (GPT-2 small: 768 / 50k), so it is not disqualifying. It should be named in §1.4 / §10.5b as one concrete mechanism by which d = 256 can cost language loss, and E2 should report TinyStories and WikiText separately (the WikiText gap diagnoses the head).
2. **Plan freshness: a lag of up to 3 tokens is inherent.** Position t = 4k+2 still reads chunk k−1; the current partial chunk reaches the decoder only through `tok`. The same holds in v1; it is inherent to chunk-level planning. No change.
3. **The plan enters once (D₀ = tok + P) and must survive 4 layers of residual writes.** The residual keeps it available; v1's bench (add arm, 2 of 2 seeds) shows it works. No change.
4. **Four positions share one plan and differ only by `offset[r]` plus local context.** The decoder must decode 4 different next tokens from the same plan. This is what the confined layout was tested for. No change.
5. **Decoder heads 4 (a reduction from v1's 6)** are for local copying over a window of 8. Low risk; stated as a judgement.
6. **The decoder's FFN (ratio 4) is where token-level knowledge lives** (the encoder's L stream carries reach-63 local context, the core carries concepts). Giving it ratio 4 is consistent with the head bottleneck: the FFN shapes D before the rank-256 projection.

My verdict: add the softmax-bottleneck mechanism (1) to the width discussion and have E2 report the languages separately. No structural change.

**Advisor (call #57):** findings 2–5 hold.
- **1a:** my v2 WikiText evidence compared mismatched steps. At matched loss, WikiText is v1 4.05 vs v2 4.00, so there is no WikiText gap. The head bottleneck is a *mechanism* to name, not an evidenced cause.
- **1b:** E2 now applies per language (TinyStories and WikiText separately), not to the average.
- **1c:** the binding limit is the zcRMSNorm radius times E's row norms, more than rank. Log E's row norms and output entropy hourly.
- **Addition:** log ‖P‖ / ‖tok‖ (how much of D₀ is plan versus local), report-only, the decoder-side analogue of E5. The null plan's magnitude is fixed, while the real plan's is not.
- **6:** don't claim the decoder FFN is "where token knowledge lives" (Engram also feeds `tok`). This finding is withdrawn.

**Changed in draft_v3.md:** §5.1 (head mechanism, telemetry, plan ratio), §5.2 (null-plan note), §9.2 E2 per language, §10.5b.

---

## R5 · Objective

**Question:** does the loss push the model toward what v3 is for (concept-level retrieval and iteration), and can any term be satisfied by a degenerate solution?

My findings:
1. **NCP is a self-prediction objective and has a collapse direction.** The target is the PQ code of the encoder's *own* next concept (detached), while the prediction side trains the encoder through C. Lowering NCP by making concepts *more predictable*, i.e. less informative, is a gradient-reachable shortcut, the classic self-prediction collapse (BYOL/JEPA-style). What prevents it: NTP through the plan needs informative concepts; the target is detached; VQ with dead-code restarts keeps codes spread; v1 and v2 stayed healthy at 0.1 (codes_min 0.98–0.99, VE ≈ 0.65). v3 raises the total self-prediction weight slightly (0.1 + 0.05 shortcut NCP). No change in coefficients, but **name the collapse direction and watch the telemetry already present** (codes_min, VE, per-segment code entropy), with a notify rule if VE falls below v1's 0.62 at a matched step.
2. **Early on, the concept route's signal is mostly NCP.** NTP reaches the core only through the plan, and the plan pays off only once u and c_pred carry something. Early core learning is therefore driven by NCP (0.1) and the shortcut terms. That is consistent with R1's two-stage dependency and with v1's history. No change.
3. **Degenerate solutions of the shortcut terms:** idling satisfies them (R3). A *constant* u (u_s = u₃ = const) would also zero L_scons, but it would break NTP through the plan and NCP, so it is not an optimum. No change.
4. **No logit-stability term (z-loss).** v1 and v2 never needed one (no NONFINITE events at 6e-4). With the output norm bounded by zcRMSNorm (R4), logits cannot run away except through ‖E‖ growth, which R4's telemetry now watches. No change.
5. **The full-batch denominator slightly changes the effective weight of synthetic vs language documents.** Synthetic documents are unsplit and padded, language documents are split; with global token-weighting, each document weighs by its real-token count, where v1 weighed micro-batches equally. The mix shift is small and in the direction of the eval convention. Already stated as a hygiene change; no change.

My verdict: add the collapse note and a VE notify rule (1). The rest holds.

**Advisor (call #58):** findings 2–4 hold.
- **1 is drift, not collapse.** The NCP target is a detached, non-differentiable argmin, so no gradient makes it more predictable. The VE rule is now matched-step (more than 0.03 below v1 at the same step), not a flat 0.62.
- **5: direction dropped.** PAD is ≈ 5.7 %, so the magnitude is bounded; the direction is not derivable.
- **New, the change R5 makes: L_scons on u leaked into the shared out_norm gain.** The least-squares optimum is (1+γ₀)·cos(a, b) < 1+γ₀, a fixed-sign shrink on the vector the decoder consumes. **Verified numerically:** 256 of 256 channels pushed down (mean gradient +0.0016). Fixed by computing L_scons on the gain-free `rms(S_s)` vs `sg(rms(S₃))`. v3_ref test T4c added; **10 of 10 pass**.
- **VQ is one-sided** (no commitment term), which is why VE plateaus near 0.65. Noted in §4.6.
- NCP's early share is quantified (≈ 0.1 of NTP's weighted signal by step 554).

**Changed:** draft §6.1 (L_scons definition), §4.6 (shared norm, one-sided VQ), §6.2 (PAD magnitude), §6.3 (drift, notify rule, early share), §7.1 (10 tests); `v3_ref.py` (T4c).

---

## R6 · Budget, throughput, memory

**Question:** are the numbers that justify the design (19M, ≈ v1 speed, ≈ 1 epoch, memory at or below v2) derived soundly, and which of them could be wrong in a way that changes a decision?

My findings:
1. **The parameter count is exact and the convention is consistent.** Compute params exclude lookups (the Engram table, the tied embedding, the codebook), as v1/v2 did, and tokens per parameter use the same compute count. Chinchilla counts non-embedding parameters, so the convention matches. ✓
2. **The throughput fit has 2 points and 2 unknowns, so it is exactly determined, with zero degrees of freedom and no error estimate.** The ±20 % band is a judgement, not a confidence interval. The b-term (per sequential application) was identified only by v2, whose 40 applications also carried MoE dispatch and per-window backward, so b is biased high for v3 (stated). **The decision this could flip:** if the real speed is below 1,200 tok/s, G4 pulls levers. That is already gated by measurement. ✓ No change beyond stating "exactly determined".
3. **Per-parameter compute is uneven, and this is intended.** The core's 4.88M parameters run 4.5 times per training step (12 applications at eval), while the encoder and decoder run once. By compute the core is ≈ 23–25 % of forward MACs, similar to its parameter share. ✓
4. **Engram's training signal per row:** at 211–290M tokens, each of the 8 tables sees the same number of lookups, about 1,000–1,400 per row on average (heavily skewed toward frequent n-grams). That is enough for the ≈ 335k frequent types (§3.2), but the long tail of rows gets a handful of updates and relies on the zero-init W_v gate to be ignored. ✓ No change.
5. **Memory:** 1.62 GB of weights and optimizer state, plus activations comparable to v2's. The unmeasured risk is the native leak (V-011), whose cause is unknown; v3 keeps GQA's `repeat_interleave` and the Engram gather. The footprint guard covers it. ✓
6. **Unstated: the M2's unified memory is shared with the OS and the other processes.** v2 ran with swap at 17–20 GB because of the leak. v3's G4 "flat swap" criterion should be *relative to a baseline taken with the trainer stopped*, or leftover swap from v2 will fail it spuriously. Proposal: record the swap baseline before G3/G4, and reboot or clear swap after v2 ends if it is still high.

My verdict: one addition (6, a swap baseline before G4) and one wording fix (2, "exactly determined, no error estimate").

**Advisor (call #59):** findings 1, 3, 5 and 6 hold.
- **3:** in training the core is ≈ 31 % of MACs (4.5 passes) against a 25.7 % parameter share, i.e. over-weighted, not similar. This is the cost of recurrence.
- **4:** my per-row count was a uniform average. Frequent types get ≈ 450–600 hits each, rare types ≈ 2–4, as noise on shared rows that the 4 hash heads average out.
- **Additions:**
  - 126 unpriced small gain kernels per step (the C-cache should count kernels);
  - the shortcut's graph is ≈ 125 MB (now in §7.3);
  - the 80M table is all of the optimizer-state growth, so sparse Adam is the last memory lever.
- **2:** the fit is exactly determined, with no error estimate.
- **6** moves into §8.4 G4's criterion, with G5 checked after v2's final eval lands.

**Changed:** draft §3.2, §7.2, §7.3, §8.4.

---

## R7 · Training recipe and gates

**Question:** do the gates distinguish real effects from measurement noise, and does the protocol say what happens when a gate fires while nobody is watching?

My findings:
1. **E1 and E2 margins are above measurement noise.** recall1 is measured on 128 items (resolution 1/128 ≈ 0.008). At p = 0.25 the binomial standard error is √(0.25 × 0.75 / 128) ≈ 0.038, so E1's threshold (0.25) against its fail line (0.05) is about 5 SE apart ✓. E2's 0.10-nat margin sits far above the val noise on 128 rows (≈ 0.01–0.02) ✓.
2. **M1's 0.01-nat margin is at or inside the noise.** M1 compares two *different runs* (the default and the fixed-M = 1 arm), so it faces seed variance (typically 0.01–0.03 nats between seeds at this scale) plus eval-sample noise, on one seed each. A 0.01-nat "pass" could be noise. **Proposal:** evaluate M1 on the milestone-size eval set (all val rows, not the 128-row quick eval), raise the val margin to **0.03 nats**, and keep the recall leg (≥ 5 points ≈ 1.3 SE at p ≈ 0.5, so also weak; use ≥ 10 points). Or state that M1 is indicative only with one seed.
3. **Nothing says what happens when a gate fires overnight.** The monitor "notifies", but the default action is undefined. **Proposal, pre-registered:** early gates E1/E2/E5 → notify and **continue** to the 24 h milestone (a checkpoint and a full eval are cheap there, and the owner decides); E3 (NONFINITE, guard) → the existing policy; E4 (tok/s < 1,200 twice) → notify and continue (the schedule is time-based, so a slow run just sees fewer tokens).
4. **The timeline for the recommended arms is long.** The default takes 52–53 h (done early on 10-03). Each arm is N ≈ 2,400 steps ≈ 15–18 h, so fixed M = 1 plus own-992 ≈ 1.5 days more, finishing ≈ 10-04/05. State it in §10.6 so the owner prices the arms.
5. **The LR replay is sound but has a subtlety:** the default's logged LR per step includes warm-up by step and the cosine by *time*, so the arm replays the default's step → LR mapping including its throughput history. That is the intent (matched LR per step). ✓
6. **G3 is a scratch run at warm-up 20, not 300.** It tests stability at full LR, not the real schedule's early dynamics. That is V-009's design, and fine; it cannot validate the real warm-up. ✓

My verdict: tighten M1 (2), pre-register the overnight default action (3), state the arms' timeline (4).

**Advisor (call #60):** findings 1, 5 and 6 hold; 3 and 4 are right.
- **2:** verify the counts first (done: the quick eval uses 128 of 256 val rows per language and 128 probe items; v1 evaluated at R = 4). The binding noise for M1's val leg is seed variance (the bench's ≈ 0.02-nat spread), not eval noise. Hence the val margin is 0.03, and **recall (≥ 10 points on the full 256-item eval) is M1's real test**, with val corroborating. ≥ 5 points was 1.1 SE, too weak.
- **E1's SE arithmetic** now uses one convention (≈ 10 SE by the smaller SE).
- **3:** the notifications carry matched v1 numbers; *no gate authorises a stop*.
- **4:** the arms' LR replay is step-indexed; the arms finish ≈ midday 10-04.
- **New:** v1's reference was at R = 4 and v3's at M = 3 (fair, given v1's depth-insensitivity). Check at steps 100/200/300 of the real run that the zero-initialised pieces opened.

**Changed:** draft §9.2 (measurement facts), §9.3b M1, §8.4 (overnight default, zero-init check), §10.6 (arms timeline).

---

## R8 · Overall verdict

**What the deep review changed in the design** (not only the wording):
1. **Bug fixed:** the shortcut consistency loss was leaking into the shared readout gain (a fixed-sign shrink, verified numerically). It is now computed on the gain-free `rms(S)`; test T4c added; 10 of 10 tests pass.
2. **Decision reversed:** the (t, Δ) table is no longer weight-decayed, because its zero is the failure mode (a small effect, decided on principle).
3. **Mechanism corrected:** v1's stagnation is explained as a *fixed point in direction space* (the normalised read plus a settled S direction gives an identical write every pass), which the table breaks. It is now §4.2's lead argument, with three telemetry signatures (fixed point / idle / useful).
4. **Gate fixed:** M1 now rests on recall (≥ 10 points on the full 256-item eval), with val (≥ 0.03) corroborating. Before, a 0.01-nat val difference, inside seed noise, could have decided the design's central question.
5. **Protocol completed:** when a gate fires overnight the pre-registered default is notify and continue, and no gate authorises a stop. Swap baselines are taken before G4, and a real-run check confirms the zero-initialised pieces opened.

**First-principles verdict:**
- **Against v2, v3 should be better per token.** Every mechanism v2 was hurt by is removed or fixed: detached windows, idle MoE parameters (15.7M active of 18.8M), NextLat opposing NTP on W_u, per-pass NCP, 40 sequential applications. The encoder and decoder FFNs are wider. This is a mechanism argument, not a measurement.
- **Against v1, it is genuinely uncertain.** Width 256 is the one large decision outside the design's control. It costs capacity everywhere and adds an output-head limit. The FFN widening and the 4-layer core compensate by an unknown amount. The early gates E1/E2 (matched data, matched steps) will answer by step ≈ 1,800.
- **The recurrence claim is the design's central bet, and the design only *permits* it.** Nothing requires passes 2–3 to do work (the idle solution satisfies every shortcut term). The table breaks v1's fixed-point mechanism, but data-limited depth could still leave passes idle. M1, against the fixed-M = 1 arm, is the only test, and it needs the recommended arm to run.

**Top risks, ranked:**
1. **Width 256** keeps v2's token inefficiency (E1/E2).
2. **Idle passes:** recurrence does not earn its 31 % of training compute (M1).
3. **The two-stage dependency** delays recall onset (E1, E5; the answer arm if it happens).
4. **Throughput** below 1,290 (G4, levers).
5. **The native memory leak** recurs (guard).

**Readiness:** the design is ready for implementation (§8.5, CPU work now). Launch is gated on G0–G5, and the GPU is not free before ≈ 01:00 on 10-01.

**Advisor:** _pending: the call failed on overload at 19:10; the verdict above is my own and awaits the advisor's check._

## R9 · r2 revision after the external deep review (2026-09-29 ≈ 21:20)
Source: `draft_v3_review.md` (reviewed r1, SHA `97409b62…`; 6/10 as an experimental design). Every finding was re-checked against code, logs and the review's probes before adoption:
- T4b false positive reproduced by reading `v3_ref.py` (`u` is not in the L_scons graph);
- the second gather in `check_engram2.Encoder2` confirmed;
- the throughput counter is `ids.numel()` (slots incl. PAD), and the LR is logged every 25 steps.

No finding was rejected. The architecture and the 18,986,871 count are unchanged. The draft's claims, gates and experiment design were revised:
- M1 relabelled "full recipe vs single pass";
- a controlled ladder added: A1 plain one-pass / A3 plain three-pass / T3 t-conditioned / Full, with paired statistics and per-example plus per-hour reporting;
- geometry (M2) and the gradient audit (M3) demoted to diagnostics, and the carry-renorm trigger redefined;
- E5 made report-only, with a new `F_off` intervention;
- "upper bound" replaced by "fixed-checkpoint intervention effect";
- the conditioning, collapse and dominance claims restated as hypotheses or measured quantities;
- harness gaps C1–C7 recorded, with repairs pending at §8.5 step 0;
- arithmetic corrected: slots vs tokens, head MACs, LR direction plus a per-step LR trace, memory, +50 % shortcut work, per-micro-batch exposure.

Open for the owner: pilot ladder before the 48 h run; the LR-trace source.
The webapp (`arch_v3d.json`, `v3d.js`, `v3dfigs.js`, the 3D tour text, the page leads) was regenerated to match r2.
