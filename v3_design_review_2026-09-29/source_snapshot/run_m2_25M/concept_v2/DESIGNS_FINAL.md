# Concept module v2 — five designs (consolidated v2 spec, after advisor reviews #22–#29)

Source of truth for scoring and the webapp page. Derivations, evidence notes and the revision history are in `DESIGNS.md`,
`LOOP_PAPERS_NOTES.md` (42 papers + 3 residual papers) and `ADVISOR_LOG.md`. Budgets: `core_budget.py` → `budgets.json`.

## Shared chassis (all five)
- Position in the model: the middle-span core between the encoder and the decoder; interface unchanged:
  `forward(c_hat, cdoc, R) -> u, logits, c_pred` (D3/D5 also return per-pass logits for the per-pass loss).
- Width: in-projection W_in 384 → 256 on c_hat (X̃), separate start projection W_s 384 → 256 (S_0 = W_s c_hat ≠ 0, so the two
  read branches differ from step 0), out-projection W_out 256 → 384 for the decoder plan (small non-zero init).
- Block: FOUR DISTINCT layers, d 256, SwiGLU FFN 1024, FULL causal attention over all 256 chunk positions (doc-masked; not
  windowed), 4 heads × 64 with GQA kv 2, sigmoid output gate, NoPE; the q/k/v shortconvs of today's layer are removed (URM).
  One pass = one cycle through the 4 layers.
- S-space: every sublayer reads [X̃, S] through the Qwen-GR elementwise gated read (low-rank 32) and writes the carry S.
- Identity at init: exactly one zero-init per path (the block output projections W_o / W_2), except where stated.
- PQ: stays 48 segments × 128 codes × 8 dims in c_hat space (336 bits/chunk); NCP head W_q 256 → 384 + distance head.
- No pass-index conditioning (default-off for extrapolation designs), except D2 (LoopFormer's normalised (t, Δ)).
- Budget denominators: non-core MACs 26.765M/token; row 0 core 1.99M/token at eval R 4 (6.9 %); cap 25 %.

## D1 — S-FIX (minimal fix of today's loop) · C2 mechanism (i) large random train R
- Carry: norm clamp S ← S / max(rms(S)/ρ, 1) after each pass. Write: today's GR write scalar.
- R ~ clip(Poisson(4), 2, 8) per sequence (E[R] 4.08); eval R 4; sweep to 16; optional KL+entropy halting.
- Loss: NTP + NCP at the final pass of the sampled R only (placement deliberately unchanged) + VQ.
- Fixes audit 1 (init), 2 (R_max 8), 4 (clamp); keeps audit 3 by design. Expected C2 4–5/10 (GRT counter-evidence).
- Budget: 4.91M params · 4.83M MACs/token at R 4 (15.3 %) · training ×1.13.

## D2 — LOOPFORMER-S (the user's favourite, faithfully) · C2 mechanism (iii) shortcut-consistency self-distillation
- AdaLN(t, Δ) per layer; α(c) is the write gate (GR write scalar dropped), α init 0.05, modulation (1 + γ(c)) after the GR read.
- Init EXCEPTION: in D2 W_o / W_2 keep DEFAULT init — α is the single near-zero factor on the path.
- Long trajectory L = 6 passes (4 × 6 = 24 effective layers = LoopFormer 3 ⊗ 8) + one shortcut S ~ U{1..5} (Dirichlet schedule,
  big-early); the shortcut path reads sg(X̃) and starts from sg(S_0) — W_in / W_s get gradient only from the long trajectory.
- Loss: NTP(u_L) + NCP(u_L) + 0.1·NCP(u_S) + 0.1·‖sg(u_L) − u_S‖² + decoder on the shortcut plan every 4th batch (weight 0.1) + VQ.
- Claim: budget elasticity (any R ≤ 6), NOT more-steps extrapolation. Eval R 6 (big-early or uniform schedule; E13 compares).
- Budget: 6.08M params · 7.19M MACs/token at R 6 (21.2 %) · training ×1.41 (2 NCP head evaluations).

## D3 — DEEP-SUP GROUPS · C2 mechanism (ii) detached-carry improvement operator
- Carry h = LN_h(S) with its own gain; separately normed readout (Diffusion-Timestep dual views).
- K ~ U{8..12} passes (E[K] 10) so EVERY sample has ≥ 2 windows ([4,4], [4,5], [4,6], [4,7], [4,4,4]) — with K < 8 a sample would be one
  BPTT graph with equal per-pass loss, the unevidenced regime; windows of 4, remainder merged into the last window; BPTT inside a
  window, carry DETACHED between windows; ALL windows graded; per-window backward. Eval R 6 lies inside window 2 for every K.
- Loss: MEAN per-pass NCP over all passes (grounded per-pass target) + NTP on the final pass + VQ.
- Main risk: X̃ is clean every pass → windows may ignore the carry (E13 gate mass; X̃-dropout ablation). Multimodal target may
  settle-then-wander (E12).
- Budget: 4.91M params · 7.19M MACs/token at R 6 (21.2 %) · training ×1.40.

## D4 — LOOPED-MoE + DELTA-RES · C2 mechanism (i) as D1; novelty = state-conditioned per-pass computation (C3)
- FFN → one expert pool shared by the 4 layers: 8 SwiGLU experts × 512, top-2 (active width 1024), per-layer routers, LB + z-loss
  (16 experts = ablation).
- Delta Attention Residual over per-pass deltas: 6 slots {X̃, null (value 0, live), last 4 pass-deltas (masked −∞ when absent)},
  zero-init queries (uniform routing), additive.
- EBT random step scale ξ ~ U[0.5, 1.5] on each pass's writes in training (ξ = 1 at eval). Carry clamp as D1. R as D1.
- Pre-committed: C2 4–5 (as D1); C3 only if routing diverges across passes (E11 per layer) and experts specialise per layer (E15).
- Budget: 4.92M params · 4.86M MACs/token at R 4 (15.4 %) · training ×1.13.

## D5 — K-PLAN (leak-bounded multimodal plan) · C2 mechanism (ii) as D3; novelty = plan capacity (C4)
- D3 chassis (LN_h carry) with K ~ U{8..10} (E[K] 9; ≥ 2 detached windows per sample, narrowed vs D3 to stay ≤ 1.5× training) + LATE-BRANCH fork: passes 1..R−1 shared; the last pass runs for K = 4 learned seeds.
- Soft-min training: weights softmax(−sg(CE_k)/τ), τ 1.0 → 0.3 (no seed starvation); leak ≤ 2 bits/chunk, ≤ 4 bits visible
  to a decoder position.
- Selector: value head (256 → 64 → 1) trained by KL to the same soft target; inference uses its pick; 25 % of batches feed the
  decoder sg(selected plan) (robustness to selector error, no gradient into the core).
- C1 weakest (Explorative Modeling: autoregressive LMs are its hardest case). R 4 design (R 6 exceeds the cap).
- Budget: 4.93M params · 8.53M MACs/token at R 4 (24.2 %) · training ×1.48.

## Row 0 — today's ConceptCoreV2
One shared d-384 LoopLayer, R ~ U{1..4}, S_0 = 0, not identity at init, unbounded carry, loss at the sampled R.
2.06M params · 1.99M MACs/token at R 4 (6.9 %). Measured: R1..R8 equal, step angle 6.7° → 0.1°, OOD more-steps 0.

## D6 — LOOPFORMER-MoE-DS (best of D2 + D3 + D4, user request) · C2 mechanism (ii) + LoopFormer consistency + state-routed experts
- Scope stated honestly: D3's training (detached windows, every pass graded on the true next-chunk code) + LoopFormer's consistency
  ‖sg(u_K) − u_r‖² (weight 0.1) applied only to the passes of the FINAL window + D4's experts in every layer and pass.
- Where LoopFormer lives: its training recipe (distil late passes toward the endpoint); its (t, Δ) conditioning is replaced by
  experts routed on the state (ablation A1 keeps it testable, with D2's wiring).
- FFN: one shared pool of 8 SwiGLU experts × 512, top-2 (active width 1024), per-layer routers; load-balance + z-loss accumulated per
  window and backpropagated with that window; fallback per-layer E = 8 if the pool collapses (E15).
- Chassis as D3: S_0 = W_s c_hat, LN_h carry, K ~ U{8..12}, windows of 4 (remainder merged), per-window backward; eval R 6.
- Decisive checks: E16 routing divergence across windows (does the carry get a job), E17 drift in the final window, A4 dense FFN
  (does looped MoE matter at this scale), A2 consistency off.
- Budget: 4.92M params · 7.20M MACs/token at R 6 (21.2 %) · training ×1.40.
