# HRM Thinker v2 — 19M · d 256 · Engram 80M · Concept module v2 (D6) · Decoder + NextLat

Status: **design document, not implemented, not trained.** Nothing here touches the running training (`runs/`), the GPU or
the chain scripts. Every number below is produced by code on the CPU:
`model_v2/v2_budget.py` (params and MACs; it builds the real encoder and decoder classes at d 256) → `model_v2/v2_budget.json`,
and `model_v2/make_arch_v2.py` → `model_v2/arch_v2.json` (the machine-readable architecture for the 2D/3D pages, §11).
Advisor calls #41 onward: `concept_v2/ADVISOR_LOG.md`.

## 0. The design at a glance

| | today (run m2, d 384) | **v2 (this document)** |
|---|---|---|
| width d / heads | 384 / 6 × 64, GQA kv 2 | **256 / 4 × 64, GQA kv 2** |
| encoder | SSSFSSSF, d_ff 1056, **15.51M** | same layout, d_ff 704, **7.39M** |
| Engram table | 8 × 131,071 × 48 = 50.3M | **8 × 208,333 × 48 = 80.0M** (208,333 is prime) |
| concept width d_c / PQ | 384 / 48 × 128 × 8 (336 bits) | **384 / 48 × 128 × 8 (unchanged)** |
| concept module | 1 shared d-384 layer × R ∈ {1..4}, **2.06M** | **D6 LOOPFORMER-MoE-DS**: 4 distinct d-256 layers, full attention, shared pool of 16 experts × 512 (top-2 → active 1024), K ∈ {8..12} passes in detached windows of 4, R_eval 6, **7.92M** |
| decoder | 4 confined layers, window 8, d_ff 1056, **7.53M** | same layout, d_ff 704, **3.51M** |
| NextLat | off (D-006) | **on**, on the decoder's final hidden state (d = 1, λ_h = λ_KL = 1), 0.66M, training only |
| losses | NTP + 0.1 NCP + VQ | NTP + 0.1 NCP(every pass) + 0.1 consistency + VQ + 0.01 LB + 0.001 z + 1.0 NextLat-h + 1.0 NextLat-KL |
| **compute params** | **25.05M** | **18.83M** |
| stored (with embedding, table, codebook) | 78.6M | 101.0M |
| forward MACs / token | 28.0M in training (R̄ 2.5) · ≈ 28.8M at eval R 4 | 25.8M in training (K̄ 10) · 20.6M at eval R 6 |
| training compute vs today | 1.00 | **0.99** (D6 trains 10 passes on average; NextLat adds ≈ 7 %) |
| reach without the core | encoder 63, decoder 70 | 63, 70 (unchanged) |

Proportions (the user's lesson "encoder a giant monster, concept an ant"):

| share of … | encoder | concept | decoder | head |
|---|---|---|---|---|
| compute params, today | 62 % | **8 %** | 30 % | — |
| compute params, v2 | 39 % | 42 % | 19 % | — |
| params active per token, v2 (2 of 16 experts) | 47 % | 30 % | 22 % | — |
| forward MACs per token, v2 (eval R 6) | 38 % | 35 % | 16 % | 10 % |

## 1. What the previous run taught us (facts, not opinions)

1. **The concept module was starved.** 2.06M of 25.05M compute params (8 %) and 6.9 % of the compute at R 4. The toy-scale probe
   (decoder spec §15.4) had already located the long-range bottleneck in the core's retrieval (multi4: core output at the
   query → value tokens at chance), not in the decoder.
2. **Its loop did nothing.** R 1..8 gave the same result, the step angle fell from 6.7° to 0.1°, extra passes on OOD tasks gave 0.
   The audit found four causes: not identity at init, R_max 4, loss placement that rewards R-invariance, unbounded carry
   (`concept_v2/DESIGNS.md`). D6 was chosen to fix all four (§6).
3. **The encoder was oversized for its job.** 15.5M dense params sit beside a 50.3M Engram table that already stores local
   n-gram knowledge; the Engram gate is open (`engram_alpha_mean` 0.77 at step 1460), so the table is used.
4. **The concept stream works when the core is given a job.** Latest evaluations: val 2.8798 (step 620) → 2.8298 (1051) →
   2.7760 (1460) → 2.7496 (1642), recall1 0.992 against plan swap, PQ codes_min 0.99. The PQ layer (48 × 128 × 8) is a measured,
   working component — v2 keeps it untouched (§3.2).
5. **The machine is the constraint.** Apple M2, 16 GB, micro-batch B = 2 × T 1024 (D-019: B = 4 swapped), 32,768 tokens per optimizer
   step, 1,650–1,815 tok/s measured, swap already 4.9 GB in the current run. Every v2 choice is priced in memory and tok/s.
6. **NextLat was tested once and was neutral.** Decoder toy tasks (no planning component, d 128): loss +0.014 / +0.019 nats,
   no speed-up; hence D-006 "NextLat off". v2 turns it on for a reason that did not exist in that test (§7.3) and with a
   keep/kill rule that measures that reason (§9).

## 2. Step 1 — Is d = 256 a good width? (first principles)

- **What width buys.** d is the size of every per-token state (L, G, D) and of the tied embedding / output head (8,192 × d).
  At V = 8,192, a 256-dim tied head is the norm for 15–30M models; the head's rank limit (256) is far above what a 19M model
  can use in ~150M training tokens.
- **What replaces the lost width.** Local, token-level knowledge moves to the Engram table (80M lookups at almost no compute),
  and long-range structure moves to the concept stream, which now gets depth (6 passes × 4 layers at eval) and sparse capacity
  (16 experts). Width is the one resource that costs compute everywhere; depth-by-recurrence and lookups do not.
- **Why not smaller.** Heads stay 64-dim (the attention kernels and partial RoPE 16/64 are unchanged); 4 heads is the minimum that
  keeps GQA 4:2. d 192 would give 3 heads and break the 2-kv-head grouping.
- **Data regime.** ~150M tokens per 24 h (§8) is 0.4 × Chinchilla for 19M params (≈ 380M tokens) versus 0.3 × for today's 25M:
  the smaller model is closer to the compute-optimal point for the budget we actually have.
- **Honest cost.** Per-token width falls by a third, so the decoder's local modelling (NTP on easy tokens) is expected to be
  somewhat worse at equal tokens; the bet is that the concept stream and the larger table recover it. This is testable
  (screen, §9) and is the main risk of the design.

**Verdict: yes**, with the risk stated. The alternative (d 320, 5 heads) breaks GQA 2 as well and moves the model back to ~25M.

## 3. Step 2 — The parameter budget and the proportions

### 3.1 Convention
"19M" uses the same convention as today's "25M" (`code/model.py:param_table`): **compute params = encoder non-embedding + concept
module without the codebook + decoder.** The token embedding (tied head), the Engram table and the PQ codebook are lookups and are
reported separately; NextLat's ψ is training-only and reported separately.

### 3.2 The forced decision on d_c (concept width)
The PQ vocabulary needs d_c divisible by 48 segments; 256 is not. Two options:
- keep **d_c = 384**: encoder pools 4 × 256 → 384 (the existing `EncoderConfig.d_concept` path, orthonormal init), the core reads
  it through W_in / W_s 384 → 256 exactly as D6 was budgeted, the decoder reads c_pred through W_c 384 → 256; PQ 48 × 128 × 8
  (336 bits per chunk) and its measured behaviour (codes_min 0.99, recall1 0.992) are untouched;
- switch to d_c = 256 with 32 × 128 × 8 (224 bits): changes the NCP targets, the D6 budget and a working component.
**Chosen: d_c = 384.** It leaves the measured component and the scored D6 unchanged; cost +0.23M params. A concept then
compresses 4 × 256 = 1,024 token dims to 384 (2.7 ×). The 32-segment layout is an ablation, not the default.

### 3.3 Where the 19M goes (v2_budget.py, all rows at d 256, d_c 384, Engram 80M)

| row | encoder d_ff | core pool E | decoder d_ff | compute | enc / core / dec (params) | MACs share of the core |
|---|---|---|---|---|---|---|
| A — D6 exactly as scored | 704 | 8 | 704 | 15.68M | 47 / 30 / 22 % | 34.9 % |
| B | 704 | 12 | 704 | 17.26M | 43 / 37 / 20 % | 34.9 % |
| **C — recommended** | 704 | **16** | 704 | **18.83M** | **39 / 42 / 19 %** | 35.0 % |
| D | 704 | 16 | 1024 | 19.82M | 37 / 40 / 23 % | 33.4 % |

Why row C:
1. **The user's three constraints.** "Decoder almost the same" → the decoder keeps today's FFN ratio (704 / 256 = 2.75 = 1056 / 384)
   and layout. "Around 19M" → 18.83M. "D6 confirmed" → the D6 mechanism is unchanged; only the size of its expert pool moves.
2. **First principles: where should spare parameters go in a looped model?** Into experts. Top-2 routing keeps the active
   width at 1,024 (the requested "(256, 1024)" core), so adding experts adds capacity at no compute. *Sparse Layers are
   Critical to Scaling Looped Language Models* (paper 42 in `concept_v2/LOOP_PAPERS_NOTES.md`) shows that looped models scale
   through sparse layers, and that later passes pick different experts. 16 experts give 120 top-2 pairs against 28 for 8, so
   the passes have more distinct computations to route into.
3. **Honesty about what changed.** E = 16 is the "16 experts" ablation of D4, not the scored D6 (E = 8). It helps E11 and E16
   (routing divergence across passes and windows) and makes E15 (pool collapse) more likely. Row A remains the fallback: if E15
   shows collapse at E = 16, drop to E = 8 (15.68M) rather than widen anything else.
4. **Row D** reaches 19.8M only by widening the decoder, which the user asked not to change.

### 3.4 C7 (the old "core ≤ 25 % of compute" cap) is redefined for v2
The cap was defined against the d-384 model. At d 256 the non-core shrinks and the core is 35 % of the forward MACs at R_eval 6:

| eval R | 1 | 2 | 4 | **6** | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| core share of forward MACs | 8.7 % | 15.5 % | 26.5 % | **35.0 %** | 41.7 % | 51.6 % | 58.7 % |

R 4 would satisfy the old cap, but R 4 is the boundary of the first detached window; D6 chose R 6 because it lies inside window 2
for every K ∈ {8..12}, i.e. at a supervised, carry-dependent depth. So the cap is redefined, not the depth: **v2 cap = core ≤ 40 %
of forward MACs at the default R.** A core at ~35 % of the compute and 30 % of the active parameters is the proportion the user
asked for. The R-sweep to 16 stays an evaluation (E12), not a default.

## 4. Step 3 — Engram 80M: is it good? (numbers)

| question | answer |
|---|---|
| layout | orders (2, 3) × 4 hash heads = 8 tables, dim 48 (unchanged), rows 131,071 → **208,333 (prime, coprime with 2¹³)**; 8 × 208,333 × 48 = 79,999,872 |
| why more rows, not wider rows | the only change versus today is the number of slots: fewer hash collisions per table at the same bits per slot, and a clean comparison with the current run. Wider slots (dim 64, 156k rows) would change two things at once |
| is a slot trained? | ≈ 150M tokens / 24 h ÷ 208,333 rows per table ≈ **700 hits per row per day** on average (Zipf-skewed: frequent n-grams far more, rare ones fewer). Enough to train a slot |
| is the memory used? | yes in the current run: `engram_alpha_mean` 0.77, Wv norm 40 at step 1460 |
| compute | lookup + W_K, W_V 384 → 256 + gated read: 0.22M params, ≈ 0.2M MACs / token; independent of the table size |
| memory | table fp32 320 MB; with gradient + AdamW m, v (dense, as today) 1.28 GB, i.e. **+0.48 GB** over the 50M table |
| expected gain | the Engram paper (recalled from memory — it is not in the folder) reports that growing the memory at fixed compute keeps helping, roughly log-linearly. 50M → 80M is 0.68 of a doubling, so the expected gain is modest. The case for 80M is that it is nearly free in compute, not that it is large |

**Verdict: good**, provided the +0.48 GB fits. It does, because the v2 run replaces the current run (it never runs alongside it).
The ratio of table to compute params rises from 2.0 × to 4.2 ×. This is deliberate: local knowledge goes to lookups and compute
goes to the concept stream.

## 5. Step 4 — Encoder v2 (same design, d 256)

Unchanged mechanisms: two role streams L (local, reach 63, the only stream the decoder reads) and G (global); layer pattern
SSSFSSSF; Engram after layer 0; CRC depth-mix; chunk pooling; zcRMSNorm token output. Only the sizes change.

| part | spec | params |
|---|---|---|
| token embedding E (tied with the head) | 8,192 × 256, e0 = rms(E[x]) | 2,097,152 (not counted) |
| S layer × 6 | reads {L}, writes {L, G}; SWA w 8, partial RoPE 16/64, q/k/v causal depthwise conv k 4 (doc-aware), gated attention 4 × 64 GQA kv 2, SwiGLU 256 → 704 → 256, gated-residual read (rank 32) + per-stream write scales | 839,812 each (attention 264,320 · FFN 540,672 · GR 34,820) |
| F layer × 2 (layers 3, 7) | reads {L, G}, writes {G} only; full causal attention within the document, NoPE, per-head τ | 873,094 each (GR 68,098) |
| Engram (after S0) | 8 tables × 208,333 × 48; hash of canonical-id 2- and 3-grams; α = σ(rms(h)·rms(k)/√d); SiLU(conv(α·v)); writes {L, G} | 215,042 compute + 79,999,872 table |
| CRC depth-mix | softmax(W rms(G)) over {e0, rms L, rms G}, init 0.107 / 0.107 / 0.787 | 771 |
| chunk pool | 4 × 256 = 1,024 → 384, orthonormal mean-pool init; c_hat = √384 · c / ‖c‖ | 393,216 |
| tok output | tok = zcRMSNorm(L) → decoder | 256 |
| **encoder total** | | **7,394,345** |

Reach: S layers add (w − 1) + (k − 1) = 10 tokens each; `EncoderConfig.reach()` = 63 (unchanged: width does not change reach).

## 6. Step 5 — Concept module v2 = D6 LOOPFORMER-MoE-DS (16-expert sizing)

D6 = D3's training (detached windows, every pass graded on the true next-chunk code) + LoopFormer's consistency toward the
endpoint in the final window + looped experts routed on the state (D4, Sparse-Layers). Full derivation:
`concept_v2/DESIGN_D6.md`, scores: `concept_v2/scores.json` (D6 70.0).

### 6.1 Structure (per chunk position, M = 256 positions per row)
| part | spec | params |
|---|---|---|
| input | X̃ = W_in c_hat (384 → 256), re-read by every sublayer of every pass; S_0 = W_s c_hat (384 → 256), a distinct projection, so the two read branches differ from step 0 | 196,608 |
| layer ℓ = 1..4 (distinct weights) | GR read x = mean_i σ(U_i V_i rms(R_i) + 2)·(1 + γ_i)·rms(R_i) over R ∈ {X̃, S} (rank 32) → gated full causal attention 4 × 64, GQA kv 2, NoPE, doc mask, sigmoid output gate, no q/k/v convs → S += s·attn; GR read again → router (256 → 16) → top-2 experts → S += s·Σ_k g_k·expert_k(x) | 334,466 each (attention 262,272 · GR 67,584 · router 4,096 · writes 514) |
| shared expert pool | 16 SwiGLU experts 256 → 512 → 256, shared by the 4 layers; top-2 → active width 1,024 | 6,291,456 (393,216 per expert) |
| carry and readout | carry h = LN_h(S) with its own gain; readout u_r = separately normed S after pass r | 512 |
| NCP head | q = W_q u_r (256 → 384), split into 48 × 8; logits = −‖q_s − sg(e_{s,n})‖² / τ_s; c_pred = Σ_n π_n sg(e_n) | 98,352 |
| PQ codebook | 48 × 128 × 8, trained by L_VQ only, k-means++ init, dead-code restarts | 49,152 (not counted) |
| **core total** | | **7,924,792** |

Identity at init: exactly one zero-init per path: W_o of each attention and W_2 of each expert. S_0 = W_s c_hat ≠ 0, so the start is
not dead. No pass-index conditioning. Carry bounded by LN_h.

### 6.2 Schedule and losses inside the core
- **Training:** K ~ U{8..12} passes per sequence (E[K] = 10); windows of 4 passes, remainder merged into the last window
  ([4,4], [4,5], [4,6], [4,7], [4,4,4]); backprop through time inside a window; **carry detached between windows**; every
  window is graded.
- **Per-pass grounding:** every pass r predicts the true next-chunk PQ code (NCP CE from u_r).
- **LoopFormer consistency, final window only:** 0.1 · mean over passes r < K of the final window of ‖sg(u_K) − u_r‖² / d.
  Earlier windows get no consistency term, so they are never given the final plan as a second, possibly wrong, target.
- **MoE auxiliaries:** load-balance (0.01) and router z-loss (0.001), accumulated per window and backpropagated with that window, so
  the gradient reaches the routers that made those decisions.
- **Inference:** R = 6 (inside window 2 for every K). Sweep R 1..16 as an evaluation.
- **Where the decoder-side losses enter:** NTP and NextLat reach the core only through u_K and c_pred of the final pass, i.e.
  only through the final window. Earlier windows learn from NCP + LB/z alone. This is D6 as designed.

### 6.3 Per-window backward: the encoder gradient (implementation, stated because the D6 spec left it implicit)
X̃ = W_in c_hat is re-read every pass, so window 1's NCP backward reaches the encoder. A plain `.backward()` would free the
encoder graph before the final window needs it. Mechanism: make `c_hat` and `tok` leaves, let their `.grad` accumulate across
windows, and push both into the encoder once at the end:
```python
tok, c_hat, _ = enc(ids, doc, eidx)
c_leaf, t_leaf = c_hat.detach().requires_grad_(), tok.detach().requires_grad_()
S = W_s(c_leaf); K = randint(8, 12); wins = windows(K, 4)            # e.g. [4, 4, 4] or [4, 6]
for w, passes in enumerate(wins):
    X = W_in(c_leaf); loss_w = 0
    for r in passes:
        S = one_pass(X, S)                                           # 4 layers; routers + experts; LB/z accumulated
        u_r = readout(S); loss_w += 0.1 * ncp(u_r, next_codes) / K + moe_aux
    if w < len(wins) - 1:
        loss_w.backward(); S = carry(S).detach()                    # frees this window's graph; grads -> c_leaf.grad
    else:
        loss_w += 0.1 * consistency(u_window, u_K)                   # final window only
        logits, h = dec(t_leaf, u_K, c_pred_K, doc, E)
        loss_w += ntp(logits) + vq(c_hat) + 1.0 * nl_h(h) + 1.0 * nl_kl(h, logits)
        loss_w.backward()
torch.autograd.backward([tok, c_hat], [t_leaf.grad, c_leaf.grad])    # one backward into the encoder
```
Activation memory of the core ≤ one window (≤ 7 passes × 4 layers × B·256 positions).

### 6.4 MoE on MPS
Capacity-based dispatch (Switch style): tokens are sorted by expert into an (E, C, 256) tensor with capacity factor 1.25 and
run as one batched matmul per projection. Dropped tokens keep their residual (S unchanged for that sublayer). In a looped core that is a silent failure mode (a position
can lose its FFN on several passes and no loss notices), so the **dropped-token fraction per layer per pass** is logged and is part
of the E15 check (target < 2 %). Per micro-batch
(B = 2) a layer-pass routes 2 × 256 × 2 = 1,024 assignments, i.e. 64 per expert on average. Masked dense evaluation of all 16
experts would cost 8 × the FFN MACs and is not used.

## 7. Step 6 — Decoder v2 + NextLat

### 7.1 Decoder (unchanged design, d 256)
| part | spec | params |
|---|---|---|
| plan | position t reads chunk j*(t) = ⌊(t+1)/4⌋ − 1 (the newest chunk ending at or before t); P_t = W_u u_{j*} + W_c c_pred_{j*} + o_{r(t)}, r(t) = (t+1) mod 4; learned null plan when j* < 0 or in another document; W_u 256 → 256 (u arrives at 256; D6's W_out is dropped), W_c 384 → 256 | 165,120 |
| confined layer × 4 | D0 = tok + P; x = GRRead(D); q = W_Q x; k, v = W_K tok, W_V tok (keys and values from tok ONLY, SWA w 8, partial RoPE 16/64); gated attention 4 × 64 GQA kv 2; D += s·attn; SwiGLU 256 → 704 → 256 on GRRead(D) | 837,250 each (attention 262,272 · FFN 540,672) |
| output | h = zcRMSNorm(D); logits = h Eᵀ (tied) | 256 |
| **decoder total** | | **3,514,376** |

Confinement is untouched: no position reads another position's D or plan, so reach = 63 + (w − 1) = 70 at any depth, and anything
older than 70 tokens can only arrive through the plan.

### 7.2 NextLat: what it is (Teoh et al., *Next-Latent Prediction Transformers Learn Compact World Models*, 2025/26)
A training-only latent dynamics model ψ learns to predict the model's next hidden state from the current hidden state and the
next token: ĥ_{t+1} = h_t + f_ψ(LN[h_t ; rms(E x_{t+1})]). Losses (paper eq. 3–5, horizon d = 1):
- L_next-h = SmoothL1(sg[h_{t+1}], ĥ_{t+1}), with the target stop-gradded (prevents collapse; the paper's ablation found this best);
- L_KL = KL(p_sg(· | sg h_{t+1}) ‖ p_sg(· | ĥ_{t+1})), with the output head frozen so only ψ and the inputs of ψ learn from it;
- pairs that cross a document boundary are masked; inference does not use ψ.
The paper's theorem: if next-token consistency and transition consistency both hold, h_t is a **belief state**, a sufficient
statistic of the whole history for predicting the future. Next-token prediction alone does not force this, because attention
can look past tokens up ad hoc.

### 7.3 Where it goes, and why it fits this model (first principles)
- **Placement: on the decoder's final hidden state h_t = out_norm(D_4)**, the pre-logit state, exactly where the paper places it
  ("the final-layer outputs"). It is already implemented as `hrm_decoder.NextLat` (LN over the concatenation, 3-layer GELU
  MLP with a zero-init last layer, residual delta, SmoothL1 + KL with E detached, doc-boundary mask).
- **Why it fits better here than in a normal transformer.** In a confined decoder, h_t is a function of only the last 70 tokens
  and its own plan. For h_t to be a belief state it must carry everything about the older history, and **the only channel for
  that is the plan**, i.e. the concept stream. So NextLat's pressure lands on exactly the component this architecture exists to
  train. The gradient of L_next-h flows through h_t (not detached) → D → P → u, c_pred → core → encoder.
- **The plan-boundary transitions.** Positions 4j+3..4j+6 read chunk j; position 4j+7 reads chunk j+1. So one transition in
  four (4j+6 → 4j+7) crosses a plan boundary: ψ must predict the state built on the next plan from (h_{4j+6}, x_{4j+7}).
  The target is stop-gradded, so this gradient enters u_j (the old plan), not u_{j+1}: **the plan is pushed to anticipate
  what the next plan will change.** It is a latent-space cousin of NCP. NCP predicts the next chunk's discrete code on the
  encoder side; NextLat predicts the next decoder state on the decoder side. The other three transitions in four shape the
  decoder's local dynamics.
- **Why it is not redundant with the D6 consistency term.** The D6 consistency acts across passes of the core (depth). NextLat
  acts across positions of the decoder (time). They constrain different axes.

### 7.4 Settings and their reasons
| setting | value | reason |
|---|---|---|
| horizon d | **1** | the theorem needs only d = 1; the existing code, the D-006 test and the cheapest cost are d = 1. d = 2 doubles ψ + KL and is a screen arm |
| λ_next-h | 1.0 | paper default (all LM settings) |
| λ_KL | **1.0, fallback 0** | the paper's LM setting uses both. Its own d = 1 ablation found SmoothL1 alone strongest on long-offset probes, so λ_KL = 0 is a principled fallback. KL costs memory, not much compute: it adds a second B × T × 8,192 logits tensor plus its log-softmax and gradient, ≈ 4 × 64 MiB ≈ 0.25 GiB per micro-batch at B = 2. If the throughput screen shows swap > 3 GB, set λ_KL = 0 |
| action embedding | rms(E.detach()[x_{t+1}]) | the paper does not specify; detaching keeps "NextLat never updates the head" true (asserted by the existing decoder test D5) |
| ψ hidden | 512 (2d) | the paper used 1–4 × d and reported that narrower bottlenecks helped planning tasks; 0.66M params, ≈ 10 MB of optimizer state |
| posterior logits | reuse the main `logits[:, 1:]`, detached | identical to sg(h_{t+1}) Eᵀ (tied head), so no second posterior matmul. The existing `NextLat.loss` recomputes it; the v2 trainer passes the main logits |
| telemetry | exempt nl_h from the "loss must fall over 300 steps" verdict | the paper (§E) observed SmoothL1 rising during LR cooldown under AdamW while acceptance kept improving; watch KL and the probes instead |

## 8. Step 7 — The total loss and where each term's gradient goes

L_total = L_NTP + 0.1 · L_NCP(mean over all K passes) + 0.1 · L_cons(final window) + L_VQ + 0.01 · L_LB + 0.001 · L_z
          + 1.0 · L_next-h + 1.0 · L_KL

"Sum of all" means every term is added into one scalar, each with its weight. The weights of the three existing terms are
today's (NTP 1, NCP 0.1, VQ 1; e.g. live.json at step 1600: 2.239 + 0.1 · 3.204 + 2.652 = 5.211). The MoE weights are the
standard Switch / ST-MoE values. The NextLat weights are the paper's.

| loss | weight | encoder | core (early windows) | core (final window) | experts / routers | decoder | ψ | codebook |
|---|---|---|---|---|---|---|---|---|
| NTP | 1 | ✓ (tok, and c_hat via the plan) | — | ✓ via u_K, c_pred | ✓ | ✓ | — | — |
| NCP per pass | 0.1 | ✓ via c_hat | ✓ | ✓ | ✓ | — | — | — (sg e) |
| consistency | 0.1 | ✓ via X̃ | — | ✓ | ✓ | — | — | — |
| VQ | 1 | — (sg c_hat) | — | — | — | — | — | ✓ |
| LB, z | 0.01, 0.001 | — | ✓ | ✓ | ✓ | — | — | — |
| NextLat next-h | 1 | ✓ | — | ✓ via u_K, c_pred | ✓ | ✓ | ✓ | — |
| NextLat KL | 1 | ✓ | — | ✓ via u_K, c_pred | ✓ | ✓ | ✓ | — (E frozen) |

## 9. Step 8 — Compute, memory, throughput, screens

| item | v2 (row C) | today |
|---|---|---|
| forward MACs / token at eval (R 6) | encoder 7.91M · core 7.21M · decoder 3.39M · head 2.10M = **20.61M** | 28.0M (R̄ 2.5) |
| core, training forward (K̄ 10, per-pass NCP head) | 12.39M / token | 1.26M |
| NextLat training (ψ fwd + bwd, KL head fwd + input-grad) | 6.16M MAC-equivalents / token | — |
| training compute (3 × forward + NextLat) | **83.5M / token = 0.99 × today** | 84.1M |
| tok/s | **≤ 1,660–1,830 (upper bound)** | 1,650–1,815 measured |
| tokens per 24 h | ≤ 143–158M | ≈ 140–155M |
| extra memory | Engram +0.48 GB, NextLat KL ≈ +0.25 GiB / micro-batch, core ≤ one window of activations | — |

Two corrections of the obvious expectation. First, a smaller model does not train faster here: D6's training runs 10 passes on
average, so v2 costs the same compute per token as today and sees about the same number of tokens. Second, the tok/s figure
scales MACs linearly and so is an **upper bound**. D6 runs up to 48 sequential layer applications at chunk rate (K ≤ 12 × 4), plus
MoE dispatch, and these small kernels are exactly what MPS handles worst.

**What sets the training cost, and the only cheap knob.** v2's training cost is structurally set by D6's schedule: the core's
training forward (12.4M of 25.8M MACs per token) comes from K̄ = 10 passes. D3's rule that every sample has ≥ 2 detached windows
forbids K < 8. Dropping the pool from E = 16 to E = 8 does **not** save compute: top-2 routing keeps the active width at 1,024
(rows A and C differ by 0.01M MACs per token). It saves only parameters and some dispatch overhead. **λ_KL = 0 is the only cheap
knob (≈ 5 % of training compute plus ≈ 0.25 GiB).** Anything beyond that is a change to the D6 schedule (e.g. K ~ U{8..10} as
in D5, ≈ −10 % core compute), i.e. a design change, not a fallback.

**Memory baseline.** Today: 8.38 GiB allocated at B = 2 (D-019). In v2 every token-rate activation shrinks by 256 / 384. The core
keeps at most one window of activations, 256 positions × ≤ 7 passes × 4 layers per row, and the budget adds +0.48 GB (table
state) and ≈ +0.25 GiB (KL). The total is expected at or below today's, but that is an estimate; the throughput screen measures it.

**Screens before any main run (the project's 265-step format, 8.7M tokens each):**
1. **Throughput and memory:** floor 1,200 tok/s (the overnight policy's flag) and swap < 3 GB. Below the floor: λ_KL = 0; if still
   below, the schedule change above (a design decision for the user, not an automatic fallback). Never widen or deepen anything.
2. **NextLat keep/kill** (twin runs, NextLat on vs off, same seed and data): **kill** if val NTP is worse by > 0.02 nats (the
   project's tie band). **Keep** only if NTP is within the band **and** a belief-state measure improves: recall1 / carry3 against
   plan swap, or the paper's future-token linear probe on frozen h at offsets 4–20. NTP alone would only repeat D-006.
3. **D6 decisive checks** (already on the D6 eval sheet): E16 routing divergence across windows, E15 pool collapse with expert-load entropy and the dropped-token fraction per layer per pass (at E = 16 this
   decides row C vs row A), A4 dense FFN 1,024 (does looped MoE matter at this scale), A2 consistency off.

## 10. Step 9 — Tensor walk-through (micro-batch B = 2, T = 1,024, M = 256)

| stage | tensor | shape |
|---|---|---|
| input | ids, doc | (2, 1024) |
| embedding | e0 = rms(E[ids]) | (2, 1024, 256) |
| Engram indices | 8 hashed row ids per token | (2, 1024, 8) |
| Engram lookup | rows → concat | (2, 1024, 8, 48) → (2, 1024, 384) → W_K, W_V → (2, 1024, 256) |
| encoder streams | L, G after each of 8 layers | (2, 1024, 256) each |
| S-layer attention | q / k, v | (2, 4, 1024, 64) / (2, 2, 1024, 64), window 8 |
| CRC mix | softmax over {e0, L, G} | (2, 1024, 3) |
| pooling | 4 tokens → one concept, sphere √384 | (2, 256, 1024) → c_hat (2, 256, 384) |
| tok | zcRMSNorm(L) | (2, 1024, 256) |
| core input | X̃ = W_in c_hat, S_0 = W_s c_hat | (2, 256, 256) each |
| core attention | q / k, v, causal doc mask | (2, 4, 256, 64) / (2, 2, 256, 64), mask (2, 1, 256, 256) |
| router | logits → top-2 | (2, 256, 16) → indices, gates (2, 256, 2) |
| experts | dispatch at capacity 1.25 | (16, C ≈ 80, 256) → SwiGLU 512 → (16, C, 256) |
| per-pass readout | u_r | (2, 256, 256) |
| NCP head | q, logits, c_pred | (2, 256, 48, 8), (2, 256, 48, 128), (2, 256, 384) |
| plan | gather per position | (2, 1024, 256) |
| decoder | D, q / k, v (from tok) | (2, 1024, 256), (2, 4, 1024, 64) / (2, 2, 1024, 64) |
| output | h, logits | (2, 1024, 256), (2, 1024, 8192) |
| NextLat | [h_t ; rms E x_{t+1}] → Δ → ĥ; prior logits | (2, 1023, 512) → (2, 1023, 256); (2, 1023, 8192) |

## 11. Step 10 — Visualisation spec (for 2D figures and the 3D page like `?v=70#arch3d`)

**Machine-readable source: `model_v2/arch_v2.json`.** It has the same top-level keys the current 3D page reads from
`webapp/data/arch.json` (`params`, `table`, `macs`, `ecfg`, `dcfg`, `reach_enc`, `reach_dec`, `S`, `N`), so the encoder and
decoder paths of `view3d.js` `buildModel()` stay valid. The concept zone is different: `buildModel()` hard-codes one core slab
and a 50.3M label, so the pass ring, the expert shelf and NextLat need the new geometry described below. The full JSON is
reproduced in Appendix A. It adds the blocks arch.json has no slot for:
- `core`: D6 settings (E, d_expert, top_k, K range, window, R_eval, d_c, init, schedule);
- `nextlat`: placement, ψ, λs, stop-grads, mask;
- `losses`: name, weight, formula, which modules each term reaches;
- `nodes` (32) and `edges` (40): every box with its params, stream reads/writes and output shape, and every arrow with its
  tensor. NextLat edges carry `training_only: true`;
- `layout3d`: the zone x positions and colours of the current page (encoder −10 blue #3b78c2, core 0 teal #2e9c8c, decoder +10
  orange #e07b39).

**3D scene (extends the current page):**
- *Encoder zone:* same stack as today (tokens → embedding → S0 → Engram → S1 … F7 → CRC → pool), with the 8 Engram tables drawn
  1.59 × taller (208k vs 131k rows) and the L / G glass columns unchanged.
- *Concept zone:* 4 distinct layer slabs (they were 1 shared slab) on a **pass ring**: K pass markers around the stack with
  window brackets of 4 and a **detach mark** (torus) between windows. A **final-window bracket** is labelled "consistency →
  u_K". An **expert shelf** of 16 small blocks beside the stack has 2 lit per layer. Per-pass NCP arrows run from every pass
  to the PQ head, the codebook sits as a 48 × 128 grid, and the output port is u_K and c_pred.
- *Decoder zone:* plan builder → 4 confined slabs whose keys and values come only from the tok column (draw the kv arrows
  from tok, not from D) → out_norm → tied head (dashed link back to the embedding). The NextLat ψ module hangs beside
  out_norm in a translucent "training only" style, with an arrow h_t → ψ → ĥ_{t+1} and a dashed sg arrow from h_{t+1}.
- *Animation (optional):* passes lighting up around the ring, with the lit experts changing from pass to pass (the
  Sparse-Layers effect).

**2D figures (one per mechanism):**
1. whole model (3 columns, the proportions donut: params / active / MACs);
2. encoder streams L / G and reach 63;
3. Engram lookup (8 tables, 208,333 rows, gate α);
4. CRC + pooling to c_hat (384, sphere);
5. D6 pass schedule: K passes, windows, detach, per-pass NCP, consistency in the final window, R_eval 6 marker;
6. looped MoE: router → top-2 of 16 per layer per pass, with a heatmap of pass × expert;
7. confined decoder and the plan index j*(t), including the 1-in-4 plan-boundary transition;
8. NextLat: h_t, x_{t+1} → ψ → ĥ_{t+1} vs sg h_{t+1}, with the KL through the frozen head;
9. loss graph: the §8 table as a Sankey from losses to modules;
10. budget: rows A–D and the R-sweep of the core share.

## 12. Code changes v2 needs (it is not drop-in reuse)

| where | change |
|---|---|
| `EncoderConfig` | d_model 256, n_heads 4, d_ff 704, `d_concept=384`, `engram_rows=208333` (the rest as today) |
| `hrm_decoder.PlanBuilder` | `Wu` is `Linear(dc, d)` today; it must become `Linear(d_u = 256, d)`. Add a `d_u` field to `DecoderConfig` (already in `arch_v2.json` `dcfg`) |
| concept core | **new module** (`concept_core_d6.py`): 4 distinct layers, the shared expert pool with capacity dispatch, per-layer routers, LN_h carry, per-pass readout, windowed schedule, consistency; it reuses `GRRead`, `GRWrite`, `SwiGLU`, `ZCRMSNorm`, `ConceptVocab` and the NCP head. `ConceptCoreV2` has no windows or experts |
| `hrm_decoder.NextLat.loss` | accept the main logits as the posterior (no recompute) |
| `param_groups` | classify the new parameters: experts and routers (decay), LN_h and readout gains, ψ's LayerNorm and biases (no decay) |
| trainer | new loop: per-window backward with the `c_hat` / `tok` leaves (§6.3), K sampling, NextLat terms, LB/z logging |
| telemetry / dashboard | exempt nl_h from the falling-loss verdict; add per-pass NCP, routing entropy, dropped-token fraction per layer per pass, E15 / E16 panels |
| `code/model.py:macs_per_token` | **defect found:** it uses the encoder's `d_ff` for the decoder FFN too. Harmless today (equal d_ff), wrong for any split; `v2_budget.py` patches it |

## 13. Review (first principles, step by step)

Each decision is reviewed as **claim → evidence → risk → the one check that decides it → confidence**. Evidence grades:
**measured** in this project > **paper** (a paper in the folder) > **recalled** (not in the folder) > **argued** (first principles
only). The score (0–10) is my confidence that the decision is right *before* any v2 run. It is not a projected result.

| step | decision | evidence | main risk | deciding check | score | advisor (#45) |
|---|---|---|---|---|---|---|
| R1 width (§2) | d 256, 4 × 64 heads | argued; common practice at V 8,192 | NTP at equal tokens worse than today's d 384 by more than the tie band | screen val NTP at 8.7M tokens vs today's run at the same tokens — the step-265 values of screens S1–S3 (same 8.7M-token format; check that the data order matches), not `runs/main` (hourly evals, none at step 265) | 7 | agree; comparator corrected to the screens |
| R2 budget (§3.3) | row C: spare params → experts (E 16) | paper (Sparse-Layers: looped models scale through sparse layers, different experts on later passes); arithmetic measured by script | pool collapse with 16 experts and only 64 assignments per expert per micro-batch | E15 + expert-load entropy + dropped-token fraction per layer per pass at the screen | 7 | agree; third E15 signal (dropped tokens) added |
| R3 d_c (§3.2) | keep d_c 384, PQ 48 × 128 × 8 | measured (codes_min 0.99, recall1 0.992) | a concept (384) wider than the token state (256): the pool expands, not compresses, the per-token width | codes_min ≥ 0.95 and VE at the screen | 9 | agree |
| R4 C7 (§3.4) | redefine the cap to ≤ 40 % at R 6 | argued (R 6 = inside window 2 for every K) | the core eats compute that the decoder needed | the R-sweep (E12) + row-D comparison if NTP lags | 7 | agree |
| R5 Engram (§4) | 80M = more rows, same dim | measured use (α 0.77); recalled scaling | +0.48 GB on a machine that already swaps; gain small | swap at the screen; eval with the Engram read off (a new intervention, and a clean one: at α = 0, y = SiLU(conv(0)) = 0, so L and G are exactly unchanged; the loss delta = what the table carries) | 7 | agree; noted as a new, exact intervention |
| R6 encoder (§5) | same design at d 256 | measured design (unit tests: reach 63, doc isolation, causality are width-independent) | G-stream quality at lower width → weaker concepts | linear probe "fact concept → key" (decoder spec §15.4 method) at the screen | 8 | agree |
| R7 core (§6) | D6 with E 16 | argued + papers (scored 70.0, tie with D3); nothing measured yet | windows ignore the carry (E13), MoE adds nothing (A4), per-window backward bugs | E16 (routing divergence across windows), A4 (dense FFN), E11-D3 (per-window NCP improvement), an R-sweep that is not flat | 6 | agree |
| R8 decoder (§7.1) | same design, d_ff 704 | measured design (confinement tests D1–D4) | the smallest module (16 % of MACs) caps NTP | if NTP lags with good concept metrics → row D (d_ff 1024, 19.82M) | 7 | agree |
| R9 NextLat (§7.2–7.4) | on, d 1, λ_h = λ_KL = 1, on out_norm(D) | paper (belief-state theorem; big gains on planning tasks, small on 1.3B LM perplexity); **measured neutral** here (D-006, toy tasks) | −0.01…−0.02 nats NTP again; KL memory | keep/kill rule (§9): NTP within 0.02 AND recall1 / carry3 / future-token probe better; **both twins use the D6 core** (D-006 used the old core) | 5 | agree; twins must share the D6 core |
| R10 losses (§8) | 8 terms, weights as listed | measured for NTP / NCP / VQ; standard for LB / z; paper for NextLat | term interplay; the consistency scale (/d) too weak or too strong; LB noisy on small micro-batches | log each term's share of the gradient norm at steps 50 and 265 (step 0 is degenerate: W_o, W_2 and ψ's last layer are zero-init) | 6 | agree; steps 50 / 265, not 0 |
| R11 compute (§9) | ≈ today's training cost; ≤ 1,660–1,830 tok/s | measured MACs (script); tok/s is an upper bound | MPS small-kernel overhead: 48 sequential layer applications + MoE dispatch could drop to ≈ 1,000–1,300 tok/s | throughput screen, floor 1,200 tok/s | 5 | agree |

**Mean confidence 6.7 / 10.** The measured components (PQ, encoder and decoder designs, the Engram's use) are the strongest.
The unmeasured parts (throughput on MPS, whether NextLat helps here, whether the looped experts diverge) are the weakest, and
each has a screen that decides it.

**Why v2 should beat the previous run (first principles):**
1. **Proportion.** The concept module goes from 8 % to 42 % of the parameters and from 7 % to 35 % of the compute, and it is
   the only long-range path. The measured bottleneck (core retrieval) gets the resources.
2. **The loop can matter.** All four audit defects are fixed by D6: identity at init, K up to 12, per-pass grounded loss in
   detached windows (no reward for R-invariance), and a bounded carry (LN_h). Today's loop is provably idle (0.1° step angle).
3. **Per-pass variation without a pass index.** Experts routed on the state give each pass a different computation, which the
   looped-MoE paper measured as the source of looped scaling.
4. **Belief-state pressure lands on the plan.** In a confined decoder, NextLat can only satisfy its objective by making the
   plan carry the history.
5. **Local knowledge moves to lookups.** The 80M table takes over what the 15.5M dense encoder was spending on.

Expected, honestly: better long-range metrics (recall1 margin over plan swap, carry3, multi4); NTP similar or slightly worse
at equal tokens. The first is the design's purpose; the second is the price of d 256.

**What would make this design wrong, and the fallback for each:**
| result at the screen | meaning | fallback |
|---|---|---|
| NextLat kill (NTP worse > 0.02, no belief-state gain) | the confined-decoder argument does not hold at this scale | λ_h = λ_KL = 0; everything else unchanged (18.83M) |
| E15 collapse at E = 16 | too many experts for the routed-token count | row A (E = 8, 15.68M). This gives up "≈ 19M", and that trade is stated |
| throughput < 1,200 tok/s | MPS cannot run the D6 schedule fast enough | λ_KL = 0; then a user decision on K ~ U{8..10} (a design change) |
| flat R-sweep, E16 ≈ 0, A4 ≈ D6 | the loop and the experts do nothing even with D6's training | D3 (dense FFN, 4.91M) is the simpler equal-score design; the budget returns to the decoder (row D logic) |
| NTP worse than today > 0.02 at equal tokens and long-range metrics not better | width 256 is too narrow for this data | the width decision is wrong; revisit at d 384 with the D6 core (a new document, not a knob) |

Advisor: #45 agreed with every step (verdicts in the table's last column) and asked for four corrections, all applied: the R1
comparator, the dropped-token metric, the Engram-off intervention, and the gradient-share steps. #46 = close-out.
Log: `concept_v2/ADVISOR_LOG.md`.

## 14. Files
- `model_v2/MODEL_V2_19M.md` (this document)
- `model_v2/v2_budget.py` → `model_v2/v2_budget.json`: rows A–F, shares, R-sweep
- `model_v2/make_arch_v2.py` → `model_v2/arch_v2.json`: the architecture for the 2D/3D pages
- sources: `code/hrm_encoder.py`, `code/hrm_decoder.py` (NextLat), `code/concept_core_v2.py` (PQ, NCP head), `concept_v2/DESIGN_D6.md`,
  `concept_v2/DESIGNS_FINAL.md`, `Next-Latent Prediction Transformer.pdf`, `Sparse Layers are Critical to Scaling Looped Language Models.pdf`


## Appendix A — `arch_v2.json` (full, generated by `model_v2/make_arch_v2.py`; the 2D/3D pages read this)

```json
{
 "version": "HRM Thinker v2 · 19M · d 256 · Engram 80M · D6 core (E 16) · decoder + NextLat",
 "convention": "compute = encoder non-embedding + concept module without codebook + decoder (code/model.py:param_table)",
 "params": {"enc_layer_S": 839812, "enc_layer_F": 873094, "enc_layer_S_attn": 264320, "enc_layer_S_ffn": 540672, "enc_layer_S_gr": 34820, "enc_layer_F_gr": 68098, "engram_table": 79999872, "engram_proj": 215042, "crc_router": 771, "pool": 393216, "tok_norm": 256, "embed": 2097152, "core_total": 7924792, "core_layer_dense": 334466, "core_expert": 393216, "core_pool": 6291456, "core_io": 196608, "ncp_head": 98352, "codebook": 49152, "dec_total": 3514376, "dec_plan": 165120, "dec_layer": 837250, "dec_attn": 262272, "dec_ffn": 540672, "nextlat_psi": 657664},
 "table": {"encoder_non_embedding": 7394345, "engram_table": 79999872, "token_embedding": 2097152, "concept_module": 7924792, "codebook": 49152, "decoder": 3514376, "compute_params": 18833513, "stored_total": 100979689, "nextlat_psi_training_only": 657664},
 "macs": {"encoder": 7912448.0, "core_eval": 7206912.0, "core_train": 12388352.0, "decoder": 3393536.0, "head": 2097152, "fwd_eval": 20610048.0, "fwd_train": 25791488.0, "core_share_eval": 0.34967953495304815, "nextlat_train_equiv": 6160384, "train_equiv": 83534848.0, "train_vs_today": 0.9936462667570686},
 "ecfg": {
  "vocab_size": 8192,
  "d_model": 256,
  "n_heads": 4,
  "n_kv_heads": 2,
  "head_dim": 64,
  "d_ff": 704,
  "chunk": 4,
  "d_concept": 384,
  "window": 8,
  "rope_dims": 16,
  "rope_base": 10000.0,
  "layer_pattern": "SSSFSSSF",
  "engram_after": 0,
  "engram_orders": [2, 3],
  "engram_heads": 4,
  "engram_rows": 208333,
  "engram_dim": 48,
  "conv_kernel": 4,
  "gr_rank": 32,
  "gr_read_bias": 2.0,
  "crc_bias": [-1.0, -1.0, 1.0],
  "eps": 1e-06
 },
 "dcfg": {
  "d_model": 256,
  "n_heads": 4,
  "n_kv_heads": 2,
  "head_dim": 64,
  "d_ff": 704,
  "n_layers": 4,
  "window": 8,
  "rope_dims": 16,
  "rope_base": 10000.0,
  "chunk": 4,
  "d_concept": 384,
  "d_u": 256,
  "gr_rank": 32,
  "gr_read_bias": 2.0,
  "layout": "confined",
  "plan_in": "add",
  "enc_reach": 63,
  "eps": 1e-06
 },
 "core": {
  "design": "D6 LOOPFORMER-MoE-DS (E = 16 sizing)",
  "d": 256,
  "n_layers": 4,
  "n_heads": 4,
  "n_kv_heads": 2,
  "head_dim": 64,
  "attention": "full causal over the 256 chunk positions, doc-masked, NoPE, sigmoid output gate, no q/k/v convs",
  "gr_read": "Qwen-style elementwise gated read over [X~, S], rank 32",
  "carry": "LN_h(S) with its own gain",
  "S0": "W_s c_hat (384 -> 256)",
  "X_in": "W_in c_hat (384 -> 256), re-read every pass",
  "ffn": {"kind": "shared expert pool", "E": 16, "d_expert": 512, "top_k": 2, "active_width": 1024, "routers": "one per layer (256 -> E)", "aux": {"load_balance": 0.01, "z_loss": 0.001, "scope": "per window"}},
  "schedule": {"K_train": "U{8..12}", "window": 4, "remainder": "merged into the last window", "carry_between_windows": "detached", "backward": "per window; c_hat is a leaf whose grad accumulates across windows"},
  "R_eval": 6,
  "R_sweep": [1, 2, 4, 6, 8, 12, 16],
  "d_c": 384,
  "pq": {"S": 48, "N": 128, "ds": 8, "bits_per_chunk": 336},
  "zero_init": "block output projections W_o, W_2 of every expert",
  "pass_index_conditioning": false
 },
 "nextlat": {
  "placement": "decoder final hidden h_t = out_norm(D_4) (pre-logit), as in the paper (Sec. 3.2)",
  "horizon_d": 1,
  "psi": "LN[h_t ; rms(E x_{t+1})] -> Linear 512->512 -> GELU -> Linear 512->512 -> GELU -> Linear 512->256 (zero-init) ; h_hat = h_t + f",
  "psi_hidden": 512,
  "psi_params": 657664,
  "lambda_h": 1.0,
  "lambda_KL": 1.0,
  "stop_grad": ["target h_{t+1}", "output head E in the KL"],
  "mask": "pairs crossing a document boundary",
  "posterior_logits": "reuse main logits[:, 1:] detached",
  "training_only": true
 },
 "losses": [
  {"name": "NTP", "weight": 1.0, "formula": "CE(logits_t, x_{t+1})", "reaches": ["decoder", "core (final window via u_K, c_pred)", "encoder"]},
  {"name": "NCP (per pass)", "weight": 0.1, "formula": "mean over all K passes of CE over the 48 PQ segments of chunk j+1 from u_r", "reaches": ["core (every window)", "encoder (via c_hat)"]},
  {"name": "consistency (LoopFormer, final window)", "weight": 0.1, "formula": "mean_{r in last window, r<K} ||sg(u_K) - u_r||^2 / d", "reaches": ["core (final window)"]},
  {"name": "VQ", "weight": 1.0, "formula": "||sg(c_hat) - e_n||^2 (codebook only)", "reaches": ["codebook"]},
  {"name": "MoE load balance", "weight": 0.01, "formula": "E * sum_e f_e * p_e, per layer, per window", "reaches": ["core routers + experts"]},
  {"name": "MoE router z-loss", "weight": 0.001, "formula": "mean logsumexp(router logits)^2", "reaches": ["core routers"]},
  {"name": "NextLat next-h", "weight": 1.0, "formula": "SmoothL1(sg h_{t+1}, h_t + f_psi(LN[h_t; rms E x_{t+1}]))", "reaches": ["psi", "decoder", "core (final window via u_K, c_pred)", "encoder"]},
  {"name": "NextLat KL", "weight": 1.0, "formula": "KL(softmax(sg h_{t+1} E^T) || softmax(h_hat E^T)), E frozen", "reaches": ["psi", "decoder", "core (final window via u_K, c_pred)", "encoder"]}
 ],
 "reach_enc": 63,
 "reach_dec": 70,
 "S": 48,
 "N": 128,
 "T": 1024,
 "M": 256,
 "nodes": [
  {"id": "tokens", "zone": "enc", "label": "tokens", "out": "B x 1024 (ids)"},
  {"id": "emb", "zone": "enc", "label": "Token embedding E (tied)", "shape": "8192 x 256", "params": 2097152, "counted": false},
  {"id": "engram_tables", "zone": "enc", "label": "Engram tables", "shape": "8 x 208333 x 48", "params": 79999872, "counted": false},
  {"id": "engram", "zone": "enc", "label": "Engram read (after S0)", "params": 215042, "reads": ["L"], "writes": ["L", "G"]},
  {"id": "enc0", "zone": "enc", "label": "S0", "kind": "S", "params": 839812, "reads": ["L"], "writes": ["L", "G"]},
  {"id": "enc1", "zone": "enc", "label": "S1", "kind": "S", "params": 839812, "reads": ["L"], "writes": ["L", "G"]},
  {"id": "enc2", "zone": "enc", "label": "S2", "kind": "S", "params": 839812, "reads": ["L"], "writes": ["L", "G"]},
  {"id": "enc3", "zone": "enc", "label": "F3", "kind": "F", "params": 873094, "reads": ["L", "G"], "writes": ["G"]},
  {"id": "enc4", "zone": "enc", "label": "S4", "kind": "S", "params": 839812, "reads": ["L"], "writes": ["L", "G"]},
  {"id": "enc5", "zone": "enc", "label": "S5", "kind": "S", "params": 839812, "reads": ["L"], "writes": ["L", "G"]},
  {"id": "enc6", "zone": "enc", "label": "S6", "kind": "S", "params": 839812, "reads": ["L"], "writes": ["L", "G"]},
  {"id": "enc7", "zone": "enc", "label": "F7", "kind": "F", "params": 873094, "reads": ["L", "G"], "writes": ["G"]},
  {"id": "crc", "zone": "enc", "label": "CRC depth-mix", "params": 771, "out": "B x 1024 x 256"},
  {"id": "pool", "zone": "enc", "label": "Chunk pool 4 x 256 -> 384, sphere sqrt(384)", "params": 393216, "out": "c_hat B x 256 x 384"},
  {"id": "tok_norm", "zone": "enc", "label": "tok = zcRMSNorm(L)", "params": 256, "out": "B x 1024 x 256"},
  {"id": "core_in", "zone": "core", "label": "W_in / W_s 384 -> 256", "params": 196608, "out": "X~, S_0: B x 256 x 256"},
  {"id": "core0", "zone": "core", "label": "Core layer 1", "params": 334466, "reads": ["X~", "S"], "writes": ["S"], "detail": "gated full attention 4x64 GQA kv2 + router -> top-2 of the shared pool"},
  {"id": "core1", "zone": "core", "label": "Core layer 2", "params": 334466, "reads": ["X~", "S"], "writes": ["S"], "detail": "gated full attention 4x64 GQA kv2 + router -> top-2 of the shared pool"},
  {"id": "core2", "zone": "core", "label": "Core layer 3", "params": 334466, "reads": ["X~", "S"], "writes": ["S"], "detail": "gated full attention 4x64 GQA kv2 + router -> top-2 of the shared pool"},
  {"id": "core3", "zone": "core", "label": "Core layer 4", "params": 334466, "reads": ["X~", "S"], "writes": ["S"], "detail": "gated full attention 4x64 GQA kv2 + router -> top-2 of the shared pool"},
  {"id": "expert_pool", "zone": "core", "label": "Shared expert pool 16 x SwiGLU 256-512-256", "params": 6291456, "per_expert": 393216},
  {"id": "carry", "zone": "core", "label": "carry LN_h(S), detached between windows"},
  {"id": "ncp_head", "zone": "core", "label": "NCP head W_q 256 -> 384 + PQ distance head", "params": 98352},
  {"id": "codebook", "zone": "core", "label": "PQ codebook 48 x 128 x 8", "params": 49152, "counted": false},
  {"id": "plan", "zone": "dec", "label": "Plan P_t = W_u u + W_c c_pred + o_r", "params": 165120},
  {"id": "dec0", "zone": "dec", "label": "Confined layer 1", "params": 837250, "detail": "q from private D; k, v from tok only (window 8); gated SwiGLU 704"},
  {"id": "dec1", "zone": "dec", "label": "Confined layer 2", "params": 837250, "detail": "q from private D; k, v from tok only (window 8); gated SwiGLU 704"},
  {"id": "dec2", "zone": "dec", "label": "Confined layer 3", "params": 837250, "detail": "q from private D; k, v from tok only (window 8); gated SwiGLU 704"},
  {"id": "dec3", "zone": "dec", "label": "Confined layer 4", "params": 837250, "detail": "q from private D; k, v from tok only (window 8); gated SwiGLU 704"},
  {"id": "out_norm", "zone": "dec", "label": "h = zcRMSNorm(D)", "params": 256, "out": "B x 1024 x 256"},
  {"id": "head", "zone": "dec", "label": "logits = h E^T (tied)", "out": "B x 1024 x 8192"},
  {"id": "nextlat", "zone": "dec", "label": "NextLat psi (training only)", "params": 657664, "counted": false, "training_only": true}
 ],
 "edges": [
  {"src": "tokens", "dst": "emb", "tensor": "ids", "training_only": false},
  {"src": "emb", "dst": "enc0", "tensor": "e0 = rms(E x)", "training_only": false},
  {"src": "engram_tables", "dst": "engram", "tensor": "8 rows / token", "training_only": false},
  {"src": "enc0", "dst": "engram", "tensor": "L", "training_only": false},
  {"src": "engram", "dst": "enc1", "tensor": "L, G", "training_only": false},
  {"src": "enc1", "dst": "enc2", "tensor": "L, G", "training_only": false},
  {"src": "enc2", "dst": "enc3", "tensor": "L, G", "training_only": false},
  {"src": "enc3", "dst": "enc4", "tensor": "L, G", "training_only": false},
  {"src": "enc4", "dst": "enc5", "tensor": "L, G", "training_only": false},
  {"src": "enc5", "dst": "enc6", "tensor": "L, G", "training_only": false},
  {"src": "enc6", "dst": "enc7", "tensor": "L, G", "training_only": false},
  {"src": "enc7", "dst": "crc", "tensor": "e0, L, G", "training_only": false},
  {"src": "crc", "dst": "pool", "tensor": "mix", "training_only": false},
  {"src": "pool", "dst": "core_in", "tensor": "c_hat", "training_only": false},
  {"src": "enc7", "dst": "tok_norm", "tensor": "L", "training_only": false},
  {"src": "core_in", "dst": "core0", "tensor": "X~, S_0", "training_only": false},
  {"src": "core0", "dst": "core1", "tensor": "S", "training_only": false},
  {"src": "core1", "dst": "core2", "tensor": "S", "training_only": false},
  {"src": "core2", "dst": "core3", "tensor": "S", "training_only": false},
  {"src": "core3", "dst": "carry", "tensor": "S", "training_only": false},
  {"src": "carry", "dst": "core0", "tensor": "S (next pass)", "training_only": false},
  {"src": "expert_pool", "dst": "core0", "tensor": "top-2", "training_only": false},
  {"src": "expert_pool", "dst": "core1", "tensor": "top-2", "training_only": false},
  {"src": "expert_pool", "dst": "core2", "tensor": "top-2", "training_only": false},
  {"src": "expert_pool", "dst": "core3", "tensor": "top-2", "training_only": false},
  {"src": "carry", "dst": "ncp_head", "tensor": "u_r (every pass)", "training_only": false},
  {"src": "codebook", "dst": "ncp_head", "tensor": "sg(e)", "training_only": false},
  {"src": "carry", "dst": "plan", "tensor": "u_K (final pass)", "training_only": false},
  {"src": "ncp_head", "dst": "plan", "tensor": "c_pred", "training_only": false},
  {"src": "tok_norm", "dst": "plan", "tensor": "tok", "training_only": false},
  {"src": "plan", "dst": "dec0", "tensor": "D0 = tok + P", "training_only": false},
  {"src": "tok_norm", "dst": "dec0", "tensor": "k, v", "training_only": false},
  {"src": "dec0", "dst": "dec1", "tensor": "D", "training_only": false},
  {"src": "dec1", "dst": "dec2", "tensor": "D", "training_only": false},
  {"src": "dec2", "dst": "dec3", "tensor": "D", "training_only": false},
  {"src": "dec3", "dst": "out_norm", "tensor": "D", "training_only": false},
  {"src": "out_norm", "dst": "head", "tensor": "h", "training_only": false},
  {"src": "emb", "dst": "head", "tensor": "E (tied)", "training_only": false},
  {"src": "out_norm", "dst": "nextlat", "tensor": "h_t, sg h_{t+1}", "training_only": true},
  {"src": "emb", "dst": "nextlat", "tensor": "sg E x_{t+1}", "training_only": true}
 ],
 "layout3d": {
  "zones": {"enc": {"x": -10, "color": "#3b78c2", "title": "Encoder", "sub": "7.39M + 80.0M memory"}, "core": {"x": 0, "color": "#2e9c8c", "title": "Concept module v2 (D6)", "sub": "7.92M · 16 experts · R 6"}, "dec": {"x": 10, "color": "#e07b39", "title": "Decoder (confined) + NextLat", "sub": "3.51M + NextLat 0.66M (train only)"}},
  "note": "same zone positions and colours as webapp/js/view3d.js buildModel(); the core zone gains a pass ring (K passes, windows of 4, detach marks) and an expert shelf"
 },
 "est_tok_s_upper_bound": [
  1661,
  1827
 ]
}
```
