# HRM Thinker v4.1: draft_v4 revised from first principles, using three papers and two tiny experiments

Status: **draft, written and reviewed section by section with the advisor (2026-10-02).** The advisor log is in §12.
`draft_v4.md` is not edited. This document is standalone where v4.1 differs. Where v4.1 inherits a part unchanged it says
so and cites the draft_v4 section.
Nothing here starts a run. The pre-launch tiny gate (§9.4) is written as a pre-registration for the owner to approve.

**Sources (every number is traced to a file or a paper figure).**
- **[NCP]** *NCP-ArchPreview Technical Report: Moving towards Latent Space Language Models through Next Concept
  Prediction* (Intern-NCP Team, arXiv 2609.10715). File: `NCP-ArchPreview Technical Report.pdf`.
- **[ETA]** *Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning*
  (Kobayashi et al., arXiv 2512.20605). File: `Emergence in Residual Stream/…pdf`.
- **[GC]** *GradCuit: Credit-Assigned Gradient Flow Enables Robust and Interpretable Test-Time Latent Reasoning*
  (Yu et al., arXiv 2608.02585). File: `Loop Papers/GradCuit.pdf`.
- `draft_v4.md` (revision 2): the design being revised. Cited as **v4 §x**.
- `v4_tiny/REPORT.md`, `v4_tiny/GO_NOGO.md`: the first tiny experiment (d 64, CPU, ≤ 7.4M tokens). Cited as **tiny-1**.
- `v4_tiny/VS_LLM_REPORT.md`, `v4_tiny/runs/vs_compare.md`: v4 vs a 3 : 1 SWA : full LLM stack, text and
  gridworld-pinpad. Cited as **tiny-2**.
- `run_v3_19M/runs/v3_48h/eval_final.json`: v3's final evaluation, test and OOD splits, step 15,588. Cited as **v3-48h**.
- `Decoder Design/HRM_THINKER_DECODER_v1_SPEC.md`: why the decoder was confined in the first place. Cited as **dec-v1**.

---

## 0. v4.1 at a glance

| | v3 (finished) | v4 (draft_v4) | **v4.1 (this draft)** | where |
|---|---|---|---|---|
| encoder | 8 layers SSSFSSSF, L/G streams, Engram reads at depth 0 and 4 | same, F layers 3 and 7 applied twice, shared SwiGLU 384 | **unchanged from v4** (the pooler is the next row); the F loop is kept only if the tiny gate shows it pays | §4 |
| tokens per concept, pooler | 4, learned linear pool | 6, 4-head attention + SwiGLU 384 | **6, 4-head attention, no FFN** (mean-pool init); k = 4 control, k = 8 arm | §4.3 |
| concept core | 4 × 3 passes, (t, Δ) table, shortcut | 4 × 2, terminal-only | **unchanged from v4** (4 × 2, shared FFN 1,024) | §5 |
| codebook, NCP | PQ 32 × 128 × 8 | coarse 1,024 + PQ 32 × 256 × 8 | **unchanged from v4** | §5 |
| decoder | 4 confined layers (window 8), plan private, keys from `tok` | same + 2 Engram reads | **3 confined window-8 layers + 1 retrieval layer R** (full causal over `tok`, plan still private); **no decoder Engram reads** | §6 |
| decoder reach without the core | 70 tokens | 70 tokens | **the whole document** (through R); plan confinement still exact | §6.3 |
| objective | NTP + NCP + shortcut + L_scons + VQ | NTP + 0.05 CE_coarse + 0.1 CE_fine + VQ | **unchanged from v4** | §7 |
| compute parameters | 18,986,871 | 19,898,011 | **19,175,580** (824,420 spare) | §8 |
| after pretraining | — | RL door: options A / B / C | **concept interface with three uses**: internal RL (ETA), test-time latent optimisation (GradCuit), VQ-only adaptation (NCP) | §10 |
| decided before launch by | — | arms after the main run | **a pre-registered tiny gate** on the existing v4-vs-LLM harness (§9.4) | §9.4 |

**In one paragraph.** v4.1 keeps v4's encoder, concept core, codebook and objective. It makes one structural change and
two removals.
- **The change.** One decoder layer becomes a retrieval layer R. Its keys and values span the whole document, from the encoder's token stream `tok`, and its queries come from the plan-aware private stream.
  - It targets the most likely cause of v4's tiny-2 deficit, which is a hypothesis consistent with the data and not yet tested: v4's token path has no content-addressed long-range read at all.
  - It keeps the property that motivated confinement: no decoder position reads another position's plan, so plans are composed only in the core.
- **The removals.** The decoder Engram reads and the pooler FFN are removed, as tiny-1 indicated.
- **Two v4 options are dropped.** The co-trained switching arm (v4 §4.7) contradicts [ETA]'s Fig. 7, and option B for the 5-layer core (v4 §4.1) contradicts v3-48h.
- **The gate.** Everything is gated by a tiny pre-launch test whose decision rule is written before any reading (§9.4).

---

## 1. Evidence since draft_v4

draft_v4 was written from v3's 24 h validation readings. Three new sources exist. Each item below gives the number and
its file.

### 1.1 v3's final evaluation (v3-48h: test and OOD, step 15,588)

| reading | value | what it changes |
|---|---|---|
| second pass: multi4 R1 / R2a / R2b / R3 | 0.764 / 0.822 / 0.818 / 0.830 → R2 − R1 ≈ +0.055 to +0.059, ≈ 2.2–2.4 unpaired SE | M = 2 is supported at full scale (v4 §1.4 holds) |
| shared FFN off | multi4 0.830 → 0.666 (−0.164); language +0.009 / +0.004 nats | keep the core's full shared FFN 1,024; rejects option B of v4 §4.1 (it halves it) |
| ĉ_pred off (`c_off`) | **+0.021 / +0.025 nats**; multi4 0.830 → 0.779 | **crosses the 0.02 line** that GO_NOGO said would force a v4 §5.4 revisit. Revisited in §10.2 |
| OOD split | multi4 0.041 at every R; carry3 and recall1 lower. The OOD items differ from test in three ways at once (`run_m2_25M/code/probes.py`): unseen name syllables, evidence 500–820 tokens back instead of 128–420, more facts (4–6 instead of 1–3) | the concept route's retrieval does not generalise to longer gaps with new names. v4.1 adds a reading that can tell these apart (§9.1) |
| V3-016 rows | language and items point in different directions on M and the shared FFN | judge on probes as well as language (v4 §2.2, kept) |

### 1.2 tiny-1: draft_v4 scaled down (d 64, ≤ 7.4M tokens, single seed per arm)

| finding | number | consequence for v4.1 |
|---|---|---|
| decoder Engram reads are a memorisation channel | on memorised training text +0.47 to +0.70 nats; on unseen probe text −0.14 to −0.17; on held-out text −0.006 / −0.023. The encoder reads on the same table generalise (+0.14 to +0.17) | removed (§3 C2) |
| pooler FFN | writes (gain 0.56–0.71) but Δ ≈ 0 | removed (§3 C3) |
| poolers tied on language | attention vs v3 linear: +0.017 ± 0.018 (paired) | — |
| v3's per-slot pool collapses the coarse code | coarse perplexity 6.8–17.8 vs 23–30 for attention | attention pooler kept (§3 C4) |
| k = 6 vs k = 4 | gap widens +0.015 → +0.035 nats from 1M to 5M; 1.07× faster | k = 6 kept with the k = 4 control mandatory (v4 §3.5.5) |
| F loop | +0.002 at k = 6, +0.022 at k = 4 | decided by the tiny gate (§3 C6) |
| retrieval onset | not reached at d 64 by 7.4M tokens (v3 needed ≈ 59M at d 256) | tiny probes cannot judge retrieval; pinpad is the tiny task that discriminates (tiny-2) |

### 1.3 tiny-2: v4 vs an LLM stack (d 64, matched parameters and depth, paired SEs)

| finding | number |
|---|---|
| text | v4 − LLM = −0.007 ± 0.004 nats (2 seeds): parity |
| pinpad, held-out optimal-action rate | 0.694 vs 0.827: **−0.133 ± 0.004** (3 seeds) |
| pinpad, pair-second (b2) rate | 0.520 vs 0.809: −0.289 ± 0.013 (chance ≈ 0.35) |
| closed-loop pair integrity | −0.183 ± 0.015 (sampled), −0.307 ± 0.019 (greedy) |
| onset | the LLM leaves chance at 1.5–3M tokens on every seed; v4 drifts 0.42 → 0.50 with no onset |
| v4's concept route | used: core off sends b2 to chance (0.386) and costs +0.55 nats of action NTP |
| reach diagnostic | the gap is as large when the whole header is inside v4's 70-token reach: near-bucket **b2** gap −0.307 ± 0.017 (held-out, 3 seeds); the optimal-rate gap in that bucket is −0.122 ± 0.007 (seed 0). **Distance is not the main cause** |
| recurrence | v4: M = 1 −0.003 b2, F loop off 0.000. LLM: M = 1 −0.07, loop off −0.11. Text: v4 +0.002 / +0.001; LLM +0.050 / +0.022 nats |
| length generalisation | 12-colour task: v4's action NTP lower by 1.11 ± 0.03 (likelihood only; the LLM still picks better actions) |
| codebook vs subgoals | coarse-code NMI with the subgoal 0.015–0.018 |
| confound, named | v4's S layers use window 8; the LLM's S layers use window 60 |

**The diagnosis that matters for the design** (tiny-2 §5 post-hoc remark, made precise here from the code).
- v4's token path has **no content-addressed long-range read anywhere**:
  - Encoder S layers attend within 8 positions and write L.
  - The decoder's confined layers take keys and values from `tok` (= L) within 8 positions (`hrm_decoder.py`, `ConfinedLayer`).
  - "Reach 70" is a receptive field through stacked window-8 layers, not a lookup. To use a header cell 40 tokens back, the information must be relayed position to position through six windows.
- **The only long-range read in v4 is the core's attention over pooled concepts.** The plan excludes the newest 0 … 5 tokens (mean 2.5 at k = 6, v4 §5.3). One plan vector serves all 6 positions of a chunk, so it is formed before the decoder knows which of those tokens it is writing. (tiny-2's post-hoc remark said "up to 11 tokens"; that double-counted the one-chunk shift, which is what brings the plan up to date, not extra lag.)
- The LLM has a full-attention layer every fourth layer on the stream that predicts tokens.
- The near-bucket result says the gap is **access type, not distance**. That is a hypothesis consistent with the data, not a tested cause. The follow-ups that would test it (an all-SWA LLM; v4 with one full decoder layer) were not run. §9.4 runs the second.

---

## 2. The three papers, read for what transfers

### 2.1 [NCP]: NCP-ArchPreview

**What it is.** OLMo-3-7B is split into a 16-layer Token Encoder and a 16-layer Token Decoder, with an 8-layer Concept Module between them at T/4. The model is 8.9B parameters, trained on 5.73T tokens.
- **Concepts.** Each concept is the mean pool of 4 encoder states. They are product-quantised (32 segments × 128 codes × 128 dims).
- **Prediction.** The Concept Module predicts the next concept as a softmax-weighted combination of codewords, trained by MSE to the detached next concept. The VQ loss is codebook-only, as in v3.
- **Feedback.** The prediction is shifted by k, repeated over k tokens, and added to the encoder output h. The Token Decoder is a standard decoder half: one residual stream, SWA with a full-attention layer every fourth layer. Concept representations are also injected "through normalized residual connections at every decoder layer" (§4.1).
- **Hierarchical residuals.** MUDDFormer-style intra-module (IRC) and cross-module (CRC) residuals, the latter with small-init diagonal scales.

**Results that carry weight.**
- Same loss as OLMo-3-7B with 51.3% of the tokens. Downstream +2.45 points at stage 1, +0.59 at stage 2.
- Parameter-aligned: approaches a 40-layer vanilla at 85% of its compute (Fig. 3). Ablation order: +CM improves, +residual improves, +NCP improves.
- NCP beats MTP at 3B and stacks with it (Table 8).
- IRC + CRC: −0.032 at 1B (Table 3).
- Muon with layer-wise QK norm is unstable; per-head QK norm fixes it ([NCP] §4.6).
- Updating only the 17M VQ + head parameters adapts domains with less forgetting than LoRA ([NCP] §5.1).
- Concept states raise a speculative drafter's accepted length by 4.17% ([NCP] §5.3).

| [NCP] fact | v4 | transfer to v4.1 |
|---|---|---|
| the concept module is an **additive side path** into a full decoder half; the decoder sees all encoder states with full attention every 4th layer | the concept route is the **only** long-range path; the decoder is 4 confined window-8 layers | **yes, in part**: v4.1 gives the decoder one full-range read (R) and keeps the plan private (§3 C1) |
| a concept vocabulary 7 bits per 128 dims is enough | v4 worried about VE capped at 0.65–0.70 (v4 §1.3b) | the codebook is a **prediction-space constraint**, not a reconstruction device: downgrade the VE concern, change nothing (§3 C13) |
| mean pooling, k = 4 | attention pooler, k = 6 | keep; attention starts as the exact mean (a superset) (§3 C4–C5) |
| NCP target constrained to the codebook | CE over coarse + fine codes | keep CE: it gives a distribution to sample from, which the RL door needs (§3 C13) |
| concepts injected at every decoder layer | plan enters at D₀ only | **defer**: dec-v1's related per-layer plan read ("read" arm) lost; [NCP]'s additive form is untested here (§3 C12) |
| IRC / CRC | L/G gated streams, a CRC router for the pool | **defer**: dec-v1 measured depth routers inert at 4 decoder layers (F2: +0.0005 to +0.008 nats) (§3 C12) |
| per-head QK norm | `GatedAttention` already has `ZCRMSNorm(dh)` per head (`hrm_encoder.py:169`) | already satisfied; v4 uses AdamW, not Muon |
| MTP, Muon | — | not adopted: different axes, with no v4-specific diagnosis |
| VQ-only adaptation | — | adopted as a post-training use of the concept interface (§10) |

**Limits.** Every [NCP] number is at 1B–8.9B parameters and 150B–5.7T tokens. v4.1 is 19M parameters at ≈ 0.6B tokens.
What transfers is the recipe and the architecture facts, not the size of the gains.

### 2.2 [ETA]: emergent temporal abstractions

**What it is.** Small transformers (6 layers, d 256) and SSMs are trained by next-action (and next-observation) prediction on expert trajectories. Gridworld-pinpad and ant-pinpad are used, with hidden subgoals.
- **Probes and control.** Linear probes decode the subgoal from mid-depth on (Fig. 3, A1). A per-subgoal low-rank linear controller e ← e + U e works best when inserted **near the middle** (Fig. 4).
- **The metacontroller.** It is a GRU encoder that is non-causal in training, with a causal switching gate β_t, z_t = β_t z̃_t + (1 − β_t) z_{t−1}, and a hypernetwork decoder to U_t. It is trained on a **frozen** base by NLL + α · KL(N(μ, Σ) ‖ N(0, I)).
- **What it learns.** β becomes quasi-binary and switches at subgoal boundaries (Fig. 6, A3).
- **Co-training collapses.** Training the metacontroller jointly with the base collapses to a single switch at the start (Fig. 7).
- **Internal RL.** RL in z-space (n_z = 8) with a thresholded β solves sparse-reward compositions where GRPO and CompILE fail. With β forced to 1 it also fails (Fig. 8).

| [ETA] fact | v4 | transfer to v4.1 |
|---|---|---|
| abstractions are found in a **frozen** pretrained model; co-training collapses (Fig. 7) | v4 §4.7 arm: co-trained κ switching | **remove the arm** from pretraining; switching belongs to the post-training metacontroller (§3 C9, §10) |
| control works best at mid-depth: "the mapping from abstract subgoals … to per-time-step low-level actions is implemented over multiple model layers" | the plan enters the decoder after 8 of 12 token-path layers (67%) | stated as a requirement and checked (§10.1); rebalancing is an arm, not a default, because the LLM with the same split won tiny-2 (§3 C14) |
| low-dimensional z, contracted horizon | coarse code, T/6 horizon | kept from v4 (§10) |
| next-observation prediction helps the representation (Fig. A2) | pinpad positions are predicted by NTP (λ = 1) | already true in the tiny pinpad format |
| fixed switching (β = 1) fails | fixed 6-token chunks | pretraining stays fixed-k (it works in [NCP] at scale); learned boundaries are post-training or v5 |

### 2.3 [GC]: GradCuit

**What it is.** N latent states are inserted at layer l of a frozen instruction-tuned LLM, between the prompt and the continuation.
- **Initialisation.** The latents start from the real hidden states of a fixed text prefix, z₀, plus an offset Δz = 0.
- **Optimisation.** At test time, Δz is optimised by Adam (lr 1e-3, ≤ 10 steps) on −R(x) Σ_t log π(x_t | x_<t, z). R is a binary self-verifier. Every continuation token reaches every latent through the remaining layers' attention.
- **Accuracy.** 64.5% average, +6.6 over CoT and +2.4 over the best baseline. Learning-rate sensitivity is lower than LatentSeek's (std 0.82 vs 1.53).
- **Random directions.** Random-direction updates (no reward gradient) still beat LatentSeek (Fig. 3) and beat a fixed prefix in 14 of 15 settings (Table 3).
- **Depth.** The best depth is 25–50%; 75% is worse (Fig. 5).
- **Where the gradient lands.** The gradient concentrates on reasoning-connector tokens, the transitions between steps (Fig. 4).

| [GC] fact | v4 | transfer to v4.1 |
|---|---|---|
| a latent site that **every later token reads** gives dense credit and robust optimisation | the core's *output* u_j reaches only the 6 tokens of chunk j + 1 (plans are private); the core's *input* ĉ_j reaches every later token through the core's causal attention | requirement met at ĉ_j, not at u_j: the site is chosen in §10.1 |
| initialise on-manifold (z₀ from real states) with a small offset | — | the v4.1 test-time variable is an offset on the real core input ĉ, initialised to 0 and re-projected to the sphere (§10.3) |
| best at 25–50% depth | the core sits beside the token path; the plan enters at 67% of it | as for [ETA]: stated and measured, not redesigned by prior (§3 C14) |
| random directions in the latent space already help | — | the controllability measurement includes a random-direction control (§10.3) |
| gradient concentrates on transition tokens | fixed chunks | supports learned boundaries at transitions (v5), not a v4.1 change |

### 2.4 What the three papers say together (first principles)

1. **A concept path should add to a capable token path, not replace its long-range access.**
   - [NCP] shows the additive form beating a matched LM at scale.
   - tiny-2 shows v4's exclusive form losing to a matched LM on the one task built to reward abstraction.
   - v1 confined the decoder to make the concept route *provable* (dec-v1 §7). v3 proved it: plan swap takes recall1 0.996 → 0.006. Having served as the proof, exclusivity has no remaining design reason.
   - **dec-v1's second reason does still hold**: a decoder that composes past core states leaves the core idle (its IHN F7 lesson). v4.1 keeps that part: plans stay private.
   - **v4.1 applies this principle to reads only, deliberately.**
     - R gives the token path long-range access to `tok` (C1).
     - A token-level exit for the encoder's G stream (C1b) would also let the decoder do global computation the core is meant to do, so it is an arm, not the default.
     - The [NCP]-style decoder that also reads past plans is the stack arm.
2. **The abstraction must be readable and steerable by the tokens that depend on it.**
   - Every later token must be downstream of the latent site ([GC]).
   - The site should sit with enough layers below it to unfold an abstract action into tokens ([ETA] Fig. 4, [GC] Fig. 5).
   - **v4.1 meets both halves for the *steering* site.** ĉ reaches every later token through the core, and sits at 45% of the longest path (§10.1).
   - **For the *read* point it is only tested.** The plan enters at 67% of the token path, and rebalancing is the C14 arm, because the LLM with the same split won tiny-2.
3. **Abstraction with learned duration is discovered after pretraining, on a frozen model** ([ETA] Fig. 7). Pretraining's job is to make the residual stream rich and controllable, with fixed-rate concepts ([NCP]). Learned switching comes later.
4. **The codebook constrains the prediction space.** It does not need to reconstruct the concept well ([NCP]'s 7 bits per 128 dims).
5. **After pretraining, the concept interface is a control surface with three uses.** It can be driven by RL ([ETA]), optimised at test time ([GC]), or adapted alone ([NCP] §5.1). The architecture should make all three possible without changing pretraining.

**Where each principle went:**

| principle | entries | not adopted, and why |
|---|---|---|
| 1 · additive, not exclusive | C1 (R), C2 (no output-side memorisation shortcut) | C1b and the stack: arms (they hand the core's job to the decoder) |
| 2 · readable and steerable | §10.1 site choice; §5's linear probes of u | C14 (read-point depth): arm |
| 3 · learned duration after pretraining | C9 (arm removed), §10.2 (switching gate in the metacontroller) | learned boundaries in pretraining: v5 |
| 4 · codebook as a constraint | C13 | the VE-raising PQ 64 × 128 × 4 arm stays an "if time" arm |
| 5 · three uses of the interface | C11, §10.2–§10.4 | none: no pretraining change is needed |

---

## 3. Change ledger: every addition, removal and kept-against-evidence item, checked three times

Each entry has three checks, then a verdict from {adopt, arm, defer, reject, keep}.
- **(i) first principle:** why it should matter, from what the part does.
- **(ii) evidence for and against:** with numbers.
- **(iii) cost, risk, alternative.**

"Arm" means it is tested in the tiny gate (§9.4) or after the main run, not used by default.

### C1 · Decoder retrieval layer R: one decoder layer reads `tok` over the whole document; plans stay private — **adopt (gated by §9.4)**

- **(i)** A token being written can only use facts it can look up.
  - In v4 every fact more than ≈ 8 tokens back must arrive through a 256-dimensional plan. The plan is built at concept rate, excludes the newest 0 … 5 tokens, and one plan vector serves all 6 positions of a chunk.
  - A full-range read whose query is formed by the position being predicted gives the token path a content-addressed lookup. In pinpad: "where is colour c?" against the header; in text: induction-style recall.
  - Keys and values come from `tok`, never from D, so no position reads another position's plan. Composition across plans stays in the core, which was dec-v1's reason for confinement.
- **(ii) For.**
  - tiny-2 pinpad: −0.133 ± 0.004 optimal rate, −0.289 ± 0.013 b2 on 3 seeds. The near-bucket gap rules out distance as the main cause and leaves access type standing.
  - [NCP]'s decoder has full attention every fourth layer and beats a parameter-matched vanilla.
  - The LLM, which has full layers on its token path, learns the pair structure by 3M tokens.
- **(ii) Against.**
  - The cause is not isolated: the window confound (8 vs 60) and the LLM's 1.35× FLOPs also differ.
  - dec-v1 measured stack = confined on the carry task: exact 0.969 / 0.984 for both. Where the core alone could carry the fact, a token path added nothing.
  - **It ends the exclusivity proof.** With R, a long-range probe can be answered by the token path. Probes then measure capability; the route's use is measured by core-off and plan-swap (§9.2).
- **(iii) Cost and risk.**
  - **Parameters:** 4 (per-head temperatures; NoPE, so no RoPE). The layer's other weights are those of the confined layer it replaces.
  - **Compute:** one layer of full causal attention over T = 1,020 keys, ≈ 260k MACs per token. That is ≈ +25% of one decoder layer, ≈ +1% of a step (estimate; the A/B decides).
  - **Risk:** the decoder bypasses the core, so the core idles. Kill line in §9.2.
  - **Alternatives:** the NCP-style stack (D non-private; plans leak), which is an arm in §9.4. Widening the windows instead (C16).
  - **Position:** layer index 2 of 4 (S S R S). Two layers form the query; one layer after composes the retrieved content with the local syntax. Chosen a priori, not tested.
- **Verdict: adopt**, conditional on §9.4. Arms: the NCP-style stack (D non-private) and R with G added to its keys (C1b).

### C1b · Let G into R's keys and values (encoder→decoder cross-module residual) — **arm**

- **(i)** G is the encoder's global stream, built by the F layers' full attention. Today it reaches prediction only through pooled concepts. tiny-2's LLM F loop paid (b2 −0.11 when off) where F writes the token stream; v4's did not (0.000), where F writes only G.
- **(ii)** For: the F-loop asymmetry above; [NCP] CRC encoder→decoder. Against: R with `tok` keys already gives retrieval; G at token level lets the decoder do global computation the core is meant to do.
- **(iii)** 256 parameters: kv source = `tok` + γ ⊙ rms(G), with γ = 0 at init. It is alive because ∂γ = rms(G) ⊙ ∂kv ≠ 0. A free switch-off (γ = 0).
- **Verdict: arm** (§9.4, if time). It also tells whether the F loop needs a token-level exit to pay.

### C2 · Remove the decoder Engram reads — **adopt (removal)**

- **(i)** A raw n-gram lookup at the output is a shortcut that rewards memorised continuations. The encoder already delivers Engram content to `tok` through read 1, after seven layers of processing.
- **(ii)** For: tiny-1 reads were +0.47 to +0.70 nats on memorised text, −0.14 to −0.17 on unseen probe text, −0.006 / −0.023 on held-out text. The encoder reads on the same table generalise (+0.14 to +0.17). Against: single seed at d 64; the owner's "spit out common tokens" hypothesis was not tested by frequency bin at full scale.
- **(iii)** Saves 427,522 parameters and ≈ 10% of decoder time. Risk: a frequent-token gain lost at full scale; v4's own kill criterion expected < 0.01 nats.
- **Verdict: remove**; re-test only if the owner asks (owner's brief: §11.4).

### C3 · Remove the pooler FFN — **adopt (removal)**

- **(i)** The FFN was to compose across heads' selections. A composition the core can do anyway (the core's first layer has an FFN) is redundant at the pool.
- **(ii)** For: tiny-1 Δ ≈ 0 with the FFN writing (gain 0.56–0.71). Against: one seed; composition may matter once retrieval turns on.
- **(iii)** Saves 294,913 parameters.
- **Verdict: remove** (the owner's post-tiny choice, already in GO_NOGO).

### C4 · Keep the attention pooler (4 heads, mean-pool init) against [NCP]'s plain mean — **keep**

- **(i)** A pool that can down-weight a token by its content is the precondition of k > 4 (v4 §3.5.2). At init it *is* the masked mean, so it can only add.
- **(ii)** For: tiny-1 attention avoids the coarse-code collapse of the per-slot pool (perplexity 23–30 vs 6.8–17.8); the owner chose it. Against: languages tied; [NCP] works with mean at scale.
- **(iii)** 198,424 parameters.
- **Verdict: keep.**

### C5 · Keep k = 6 against [NCP]'s k = 4 — **keep (k = 4 control mandatory, k = 8 arm)**

- **(i)** The owner's compression goal. With R, the plan no longer has to carry every recent detail, so the staleness cost of a larger k should fall.
- **(ii)** For: 1.07× speed; the full-scale gate at step 4,050 decides. Against: tiny-1 k6 − k4 widened to +0.035 by 5M; [NCP] uses 4.
- **(iii)** The v4 §3.5.5 gate is unchanged.
- **Verdict: keep**, as v4.

### C6 · The encoder F loop — **decided by §9.4 (default for the tiny: on)**

- **(i)** A second refinement of G helps only if G's refinement reaches prediction.
- **(ii)** For: the LLM's loop paid (b2 −0.11, text +0.022). Against: v4's loop was ≈ 0 three times (tiny-1 k6 +0.002, tiny-2 text +0.001, pinpad 0.000), at ≈ 13–20% of step time (v4 §3.3 estimate).
- **(iii)** ≈ 60–90M tokens over 48 h.
- **Verdict: keep only if the trained loop-off arm in §9.4 costs ≥ 2 SE on pinpad b2 or ≥ 0.01 nats on text.** Otherwise drop it and spend the time on tokens.

### C7 · Keep M = 2 — **keep**

- **(i)** Pass 2 reads S₁, so it can use what pass 1 retrieved (v4 §4.2).
- **(ii)** For: v3-48h R2 − R1 multi4 +0.055 to +0.059 (≈ 2.3 SE); the owner. Against: tiny-1 and tiny-2 saw nothing (M = 1 −0.003 b2, +0.002 nats).
- **(iii)** Unchanged.
- **Verdict: keep**, with v4's kill line (R2 − R1 multi4 ≥ 2 SE at 24 h).

### C8 · Core depth 4 × 2 vs 5 × 2 (v4 §10.6 (2)) — **keep 4 × 2; reject option B**

- **(i)** Option B buys a fifth layer by halving the shared FFN.
- **(ii)** v3-48h shared FFN off −0.164 multi4 (larger than the 24 h −0.129). Tiny: core depth and recurrence earned nothing.
- **(iii)** The freed 722,435 parameters (722,431 net of R's +4) would fit B, but B trades a measured contributor for untested depth.
- **Verdict: 4 × 2**, closing v4's open decision on the evidence.

### C9 · Remove the co-trained switching arm (v4 §4.7) — **adopt (removal)**

- **(i)** Joint NTP training has no pressure for sparse switching; the variational bottleneck on a frozen base supplies it.
- **(ii)** [ETA] Fig. 7: co-trained metacontrollers converge to "a single switch at the very beginning". v4 §4.7 already said the faithful use is post-training.
- **(iii)** Frees an arm slot (≈ 10 h). Its switching gate moves to §10.2.
- **Verdict: remove.**

### C10 · Keep ĉ_pred in the plan and revisit the RL door — **keep; framing revised (§10.2)**

- **(i)** If the text depends on the code, choosing codes steers the text.
- **(ii)** v3-48h c_off +0.021 / +0.025 nats, multi4 −0.051: above the 0.02 line. The decoder depends on the code more than draft_v4 §1.7 (2) assumed (0.013 at 24 h).
- **(iii)** None.
- **Verdict: keep.** The door is still A + C (§10.2); the stronger dependency makes C more likely to work.

### C11 · Post-training concept interface (internal RL, test-time optimisation, VQ-only adaptation) — **adopt (no pretraining change)**

- **(i)** [ETA], [GC] and [NCP] §5.1 each use the latent site after pretraining, in three different ways.
- **(ii)** Each is shown in its paper at its scale. None is shown at 19M.
- **(iii)** Zero pretraining parameters. One evaluation hook and one 48 h measurement (§10.3).
- **Verdict: adopt** as §10.

### C12 · [NCP]'s per-layer concept injection, IRC and CRC — **defer**

- **(i)** They give later layers direct access to earlier representations.
- **(ii)** Against: dec-v1 tested a related mechanism and it lost. The "read" arm replaced the D₀ add with a floored gated read at every layer, and learned the carry task in 1 of 2 seeds against the add's 2 of 2. [NCP]'s form, which adds at the input *and* re-injects at every layer, is untested here. dec-v1 also measured depth routers inert at 4 decoder layers (F2: +0.0005 to +0.008 nats). For: [NCP] Table 3, −0.032 at 1B.
- **(iii)** Small parameters, but memory traffic on MPS ([NCP] says the analytical FLOPs omit it).
- **Verdict: defer** both the additive per-layer re-injection and IRC / CRC (except C1b) to a deeper model. At 4 decoder layers with the plan already in D₀ there is little to re-inject.

### C13 · Keep the two-level codebook and CE-based NCP against [NCP]'s MSE — **keep**

- **(i)** CE gives a distribution over the next abstract action: a prior to sample from, which RL and test-time search need. [NCP]'s MSE to a soft codeword gives only a point.
- **(ii)** For: tiny-1 coarse code used (perplexity 23–30 of 64) and predictable; fine VE 0.72 of a 0.75 bound. Against: NMI with pinpad subgoals ≈ 0.016, so the code is not a subgoal label at d 64.
- **(iii)** None.
- **Verdict: keep.** v4 §1.3b's VE worry is downgraded: [NCP] works at 7 bits per 128 dims.

### C14 · Rebalance depth (shallower encoder, deeper decoder) toward [ETA]/[GC]'s mid-depth site — **arm (if time), not default**

- **(i)** The plan enters at 67% of the token path (8 of 12 layers). [GC] Fig. 5 finds 75% worse than 25–50%; [ETA] Fig. 4 finds the middle best.
- **(ii)** Against: tiny-2's LLM had the same 8 / 8 / 4 split (middle output at 16 of 22) and won. The owner considers the encoder v3's best part.
- **(iii)** Parameter-neutral (encoder 6 / decoder 6: −2 × 1,085,572 + 2 × 1,083,010).
- **Verdict: arm** in §9.4 if time.

### C15 · Keep the encoder shared FFN 384 — **keep**

tiny-1: used. v3's core shared FFN is the largest measured probe contributor. Unchanged.

### C16 · Widen S-layer windows 8 → 60 — **defer (arm if time)**

- **(i)** Window 8 existed to make reach(L) = 63 provable for the confined decoder. With R, that reason is gone.
- **(ii)** For: tiny-2's LLM used 60 (a named confound). Against: no isolated evidence. R already provides long-range access.
- **(iii)** Not free. The block-local kernel scores T × (block + window − 1): 67 → 119 keys per query at block 60 (`speed_v3.py`), ≈ 1.8× S-layer attention scores, still small next to the FFN.
- **Verdict: defer**; arm in §9.4 if time.

### Ledger summary

| | item | verdict | parameters |
|---|---|---|---|
| C1 | decoder retrieval layer R (plans private) | **adopt**, gated by §9.4 | +4 |
| C1b | G in R's keys | arm | (+256) |
| C2 | decoder Engram reads | **remove** | −427,522 |
| C3 | pooler FFN | **remove** | −294,913 |
| C4–C5 | attention pooler, k = 6 | keep | — |
| C6 | encoder F loop | decided by §9.4 | 0 |
| C7–C8 | M = 2; core 4 × 2 | keep; option B rejected | — |
| C9 | co-trained switching arm | **remove** (to §10) | — |
| C10 | ĉ_pred; RL door | keep; revisited | — |
| C11 | concept interface after pretraining | **adopt** (§10) | 0 |
| C12 | per-layer plan re-injection; IRC / CRC | defer | — |
| C13 | two-level codebook, CE | keep | — |
| C14 | depth rebalance | arm | 0 |
| C15 | encoder shared FFN | keep | — |
| C16 | wider windows | defer / arm | 0 |
| | **total** | v4 19,898,011 → **v4.1 19,175,580** | −722,431 |

---

## 4. Encoder: v4's encoder, with the pooler's FFN removed

### 4.1 Inherited unchanged (v4 §3.1–§3.4, §3.7)

- **Layers.** 8 layers SSSFSSSF, d 256, vocabulary 8,192, T 1,020.
  - S layers: window 8, 4 heads (2 KV), partial RoPE 16, q/k/v short convolutions. They read {L} and write {L, G}.
  - F layers: full causal attention, NoPE, per-head temperature, 6 heads (2 KV). They read {L, G} and write {G}.
- **Residual grammar.** Gated-residual reads and writes, zero-init.
- **Engram.** One gather, read 1 after layer 0, read 2 after layer 4.
- **Shared FFN.** SwiGLU 384 beside every layer's own FFN, with zero-init W₂ and a per-layer gain initialised to 1.
- **Outputs.** `tok` = zcRMSNorm(L), with reach(L) = 63. The CRC router mix m_t = mix₀ e₀ + mix₁ rms L + mix₂ rms G feeds
  the pooler.
- **What reach(L) = 63 still means.** It remains a unit-tested property of L. In v4.1 it no longer bounds what the
  decoder sees, because R reads `tok` over the whole document. It still bounds what each *key* of R summarises: every key
  is a local, 63-token view of its position.

### 4.2 The F loop (layers 3 and 7 applied twice): decided by the tiny gate

- **Mechanism.** As v4 §3.3. `enc_loop = 2` applies F layers 3 and 7 twice in place; `enc_loop = 1` is the bypass and the
  free switch-off.
- **Decision.** By the trained loop-off arm in §9.4 (rule in C6).
- **Fallback if the owner does not run the gate: loop off.**
  - Three readings in v4-type designs were ≈ 0: tiny-1 k = 6 +0.002 nats, tiny-2 text +0.001, tiny-2 pinpad b2 0.000.
  - It costs an estimated 13–20% of step time.
  - The LLM evidence for it applies only where G reaches prediction at token level, which in v4.1 is the C1b arm, not the
    default.
  - This is the owner's call.

### 4.3 The pooler: 4-head attention over the chunk, mean-pool init, no FFN

v4 §3.5.4 with the FFN branch removed (C3):
- **Keys and values.** Key k_i = W_K (rms m_i + p_{r(i)}); value v_i = W_V (m_i + p_{r(i)}). W_K, W_V ∈ ℝ^{256×256}.
- **Scores.** s_{h,i} = q_h · k_i^{(h)} / 8 + b_{h,r(i)}. Softmax over the chunk's real tokens; pads are masked; a chunk of
  pads only falls back to uniform weights.
- **Pooled.** c₀ = W_O [Σ_i α_{1,i} v_i^{(1)} ; … ; Σ_i α_{4,i} v_i^{(4)}], then ĉ = √256 · c₀ / ‖c₀‖.
- **Init.** q = b = p = 0, W_V = W_O = I, W_K random with std 0.02. ĉ at step 0 is the sphere projection of the masked
  mean of the chunk's real m_i: v3's initial pooling, with pads excluded.
- **Parameters.** 198,424 at k = 6 (instantiated); 198,944 at k = 8.
- **Telemetry, as tiny-1 learned.** Within-slot class ratios (top-100 tokens vs the rest, and content vs template in
  facts) and the position share of α's variance. Slot-averaged ratios mix position and content.

### 4.4 Interface to the core and the decoder

- **To the core:** ĉ (B, 170, 256) on the sphere; `cdoc = doc[:, ::6]`; the masks `nonempty`, `complete` and `pair_ok`
  at k = 6.
- **To the decoder:** `tok` (B, 1,020, 256). In the C1b arm, also G (B, 1,020, 256), the encoder's global stream after
  layer 7.
- **No Engram rows to the decoder.** The C2 removal drops v4 §3.7's second output.

### 4.5 Acceptance tests (v4 §3.8, adjusted)

- **Kept as v4 §3.8:**
  - 1: reach(L) = 63 with `enc_loop = 2`;
  - 3: the loop bypass;
  - 4: document isolation;
  - 6: chunk data at k = 6 (and k = 8 for the arm);
  - 7: pool masking;
  - 8: empty chunk;
  - 9: content-weighting is possible, with v3's linear pool as the mutant.
- **Changed:**
  - 2: bit-identity at step 0 now covers the pooler without its FFN. The pooler equals the masked mean before the sphere,
    and q and p_r receive nonzero gradients after one step.
  - 5: the Engram hand-off is replaced by a check that the decoder receives no `e` tensor.

---

## 5. Concept module: unchanged from v4, minus the co-trained switching arm

**Inherited from v4 §4, unchanged:**
- 4 distinct F-anatomy layers over {C, S} at concept rate, own FFN 736 plus the shared FFN 1,024.
- **M = 2 plain passes from S₀ = 0.** No (t, Δ) table, no sampled shortcut, terminal-only supervision on
  u = zcRMSNorm(S₂).
- **Two-level codebook.** Coarse E⁰, 1,024 × 256; fine PQ, 32 × 256 × 8 on the residual.
- **Head.** A distance head with coarse and fine logits. The fine NCP is conditioned on the true coarse centroid
  *inside the loss only* (v4 §6.2).
- **ĉ_pred** = c̄ + PQexp(q − c̄).
- Compute parameters 4,838,209.

**Evidence status of each inherited choice:**

| choice | v3-48h (full scale) | tiny-1 / tiny-2 (d 64) | v4.1 |
|---|---|---|---|
| M = 2 | R2 − R1 multi4 +0.055 to +0.059 (≈ 2.3 SE) | ≈ 0 | keep; v4's kill line at 24 h (C7) |
| shared FFN 1,024 | off: multi4 −0.164 | used (tiny-1) | keep at full size (C8) |
| 4 layers, not 5 | — | depth unearned | 4 × 2 (C8) |
| ĉ_pred in the plan | c_off +0.021 / +0.025 nats, multi4 −0.051 | — | keep (C10) |
| two-level codebook | — | coarse used and predictable; fine VE 0.72 of 0.75 | keep (C13) |

**Removed from v4:** the v4 §4.7 temporal-integration arm (C9). Its switching gate moves to the post-training metacontroller
(§10.2).

**Arms kept from v4:** pass-1 NCP (v4 §4.2) and the fine-only PQ 64 × 128 × 4 (v4 §4.6). Both are "if time", after the
main run.

**One addition to the 24 h report.** Linear probes of u for the probe families' answer entity, at both passes, beside
v4's bridge-by-pass probe. [ETA] reads abstraction from the residual stream with linear probes (Fig. 3). This is the same
measurement on v4.1's latent site.

---

## 6. Decoder: three confined layers and one retrieval layer R; plans stay private

### 6.1 Anatomy

| layer | kind | queries from | keys / values from | positions | attention range |
|---|---|---|---|---|---|
| 0 | confined | GRRead(D), the plan-aware private stream | `tok` | partial RoPE 16 | window 8 |
| 1 | confined | GRRead(D) | `tok` | partial RoPE 16 | window 8 |
| **2** | **R (retrieval)** | GRRead(D) | `tok` | **NoPE, per-head temperature τ_h** | **full causal, same document** |
| 3 | confined | GRRead(D) | `tok` | partial RoPE 16 | window 8 |

- **Shared by all four layers.** d 256, 4 heads (2 KV), head width 64, per-head q/k ZCRMSNorm, output gate, SwiGLU 1,024
  on D, gated-residual reads and writes.
- **Tied output head.** logits = zcRMSNorm(D) @ Eᵀ.
- **D₀ = tok + P**, with P from §6.4.
- **The one rule that has not changed.** Every layer, including R, takes keys and values from `tok`, never from D. D is
  private to its position.

### 6.2 R, exactly

- **Read.** x = GRRead([D]).
- **Projections.** q = q_norm(W_Q x) · τ_h, with τ_h = exp(log τ_h) and log τ_h initialised to 0 (as the encoder's F
  layers). k = k_norm(W_K tok); v = W_V tok.
- **Mask.** Causal, and same document. This is the `full` mask of `build_masks`, as the encoder's F layers use.
- **Update.** D ← D + GRWrite(x) · W_O(y ⊙ σ(W_g x)); then x′ = GRRead([D]); D ← D + GRWrite(x′) · FFN(x′).
- **Parameters.** Those of a confined layer (1,083,010) plus 4 temperatures, so 1,083,014.
- **Why NoPE.** It matches the encoder's F layers and [NCP]'s and the Llama-4-style split: local layers carry position;
  the global layer is a content lookup. Order within the window still enters through `tok`, whose L stream carries
  partial RoPE and convolutions.
- **Why position 2.** Layers 0–1 build the query from the plan and the local context. Layer 3 composes the retrieved
  content with the local syntax before the output. Chosen a priori; the gate does not vary it.

### 6.3 What each tensor may see

| tensor | sees |
|---|---|
| `tok` at position t | the last 63 tokens of its document (unchanged) |
| D_t, without the core (`core_off`) | **every earlier token of its document** through R (v4: 70 tokens) |
| D_t, plans | **only its own plan P_t**, built from the complete chunks before it (unchanged, exact) |
| u_j | concepts ≤ j of its own document (unchanged) |

- **What is lost.** A correct long-range probe answer no longer proves the concept route carried the fact. R could have
  read it.
- **What is kept.**
  - Plan confinement, exactly: perturbing u_j changes only positions 6j + 5 … 6j + 10.
  - So no decoder position composes plans. Composition across chunks is still the core's job, which is the reason
    dec-v1 §7 gave for confinement beyond provability.
- **What replaces the proof: measurement of use.**
  - `core_off` and `plan_swap` at every milestone.
  - The free switch-off "R local", which masks R to window 8 at evaluation.
  - The trained no-R arm (§9.4), the only reading of R's training-time value.

### 6.4 Plan builder (unchanged from v4 §5.3)

- **Formula.** P_t = W_u u_{j*} + W_c ĉ_pred,{j*} + o_{r(t)}, with:
  - j*(t) = ⌊(t + 1)/6⌋ − 1, the newest complete chunk;
  - r(t) = (t + 1) mod 6.
- **Null plan.** A learned null plan is used for the first 5 tokens of each document and across document boundaries.
- **Parameters.** 132,864.

### 6.5 Removed: the decoder Engram reads (C2)

- **Gone.** The two reads (427,522 parameters) and the hand-off of `e`.
- **What still reaches the decoder.** The encoder's two Engram reads still feed `tok`. R can now read those enriched
  token features from anywhere in the document.

### 6.6 Switch-offs and arms

| name | what it does | kind |
|---|---|---|
| R local | R's mask becomes window 8 (causal, same document) at evaluation | fixed-checkpoint switch-off (out of distribution) |
| core off / plan swap | as v4 | fixed-checkpoint |
| no-R (trained) | all four layers confined: exactly v4's decoder minus its Engram reads | trained arm. The existing tiny-2 v4 runs *are* this arm (§9.4) |
| stack (trained) | D non-private: the v4.1 decoder with keys and values taken from the same gated read x = GRRead([D]) that supplies the queries, instead of `tok`, in all four layers (layers 0, 1, 3 window 8, layer 2 full causal). No new norm is needed, because x is already normalised, as in dec-v1's `StackLayer`. Plans leak to later positions. This is dec-v1's "differs in exactly one place" contrast, with **identical parameters** (dec-v1's `StackLayer` adds q/k/v short convolutions, so it is not used) | trained arm: the [NCP]-style decoder |
| C1b (trained) | R's keys and values from `tok` + γ ⊙ rms(G), γ = 0 at init | trained arm; γ = 0 is its switch-off |

### 6.7 Acceptance tests and deliberate defects

| test | statement | v4 counterpart |
|---|---|---|
| D1a · reach through R | perturbing `tok` at p changes no logit at t < p, and the farthest changed position is the end of p's document (reach = the document) | D1 (was exactly p … p + 7) |
| D1b · reach with "R local" | with R masked to window 8, perturbing `tok` at p changes logits at exactly p … p + 7 | D1 |
| D1c · document isolation through R | perturbing `tok` in document A changes no logit in document B of the same row | new |
| D2 · plan confinement | perturbing u_j changes exactly positions 6j + 5 … 6j + 10, with R installed | D2, unchanged |
| D3 · null plan | null at exactly the first 5 tokens of every document; no plan crosses a document | D3 |
| D4 · no Engram | the decoder has no `e` input; the parameter table shows `dec_engram` = 0 | replaces v4's bypass test |

Deliberate defects, each of which must fail a named test:
1. R takes keys and values from GRRead([D]) instead of `tok`: D2 fails (the stack leak, dec-v1 §7).
2. R's mask omits the document term: D1c fails.
3. R's mask is not causal: D1a fails (a change at p reaches t < p).
4. The plan index is off by one (j* = ⌊t/6⌋ − 1): D2 fails.

---

## 7. Objective: unchanged from v4

**L = NTP + 0.05 · CE_coarse + 0.1 · CE_fine + VQ_coarse + VQ_fine.** The definitions, masks, full-batch denominators,
dead-code restarts and initialisation order are those of v4 §6.1–§6.3. Logging is unchanged: headline NTP, with the four
terms separately.

**Gradient-reach map, updated for R** (v4 §6.4):

| parameter group | NTP | NCP | VQ |
|---|---|---|---|
| decoder layers (including R), plan builder | ✓ | — | — |
| core, shared FFN, head | ✓ (through the plan) | ✓ | — |
| encoder L stream, S layers | ✓ through `tok`: **now also from every later position through R** | ✓ (through C) | — |
| encoder G stream, F layers, F loop, pooler | ✓ **only through the concept route** (unchanged; C1b would add a token-level path) | ✓ | — |
| Engram table | ✓ (encoder reads only) | ✓ (via C) | — |
| codebooks | — | — | ✓ |

- **Why this row matters.** With R reading `tok`, the L stream gains a long-range training signal. G does not.
- **Consequence for the F loop.** Its economics are unchanged by the default. Only C1b would let a refined G reach
  prediction directly. That is why C1b is the arm that tests whether the F loop needs a token-level exit to pay.
- **The 24 h G/L gradient-ratio check** (v4 §6.4) is kept, and is now expected to fall relative to v3.
  - The flag is a ratio below half of v3's, since L's share is expected to rise.
  - If it trips, the pass-1 NCP arm, or raising λ_c and λ_f, is the fallback.

---

## 8. Parameter and compute budget

### 8.1 Ledger (instantiated for the v4 base; the deltas below are counted from the parts)

| module | v4 (draft_v4) | v4.1 | change |
|---|---|---|---|
| encoder layers (8) | 8,948,776 | 8,948,776 | — |
| encoder shared FFN + gains | 294,920 | 294,920 | — |
| Engram reads 1 + 2 (non-table) | 429,060 | 429,060 | — |
| CRC router, pooler, tok norm | 494,364 | 199,451 | −294,913 (pooler FFN, C3) |
| **encoder** | **10,167,120** | **9,872,207** | −294,913 |
| **core** | **4,838,209** | **4,838,209** | — |
| decoder layers (4) + out norm | 4,332,296 | 4,332,300 | +4 (R's temperatures) |
| plan builder | 132,864 | 132,864 | — |
| decoder Engram reads | 427,522 | — | −427,522 (C2) |
| **decoder** | **4,892,682** | **4,465,164** | −427,518 |
| **compute total** | **19,898,011** | **19,175,580** | −722,431; **824,420 spare** |
| C1b arm | | +256 | |
| k = 8 arm | | +520 (pooler p_r and b) + 512 (offsets) | |

- **The base is instantiated.** `model_v4.build_model("full", k=6, pool="attn", dec_reads=())` gives 19,175,576 with
  `param_table`'s reconciliation assertion passing. The +4 is counted.
- **Before launch.** v4.1's own `param_table` must reproduce 19,175,580.
- **The spare 824,420 stays headroom.** No ledger entry earns it.

### 8.2 Compute per token (estimate; the interleaved A/B decides)

Against v4's micro-step estimate (v4 §7.2, ≈ 0.79–0.85 of v3's profile):

| change | effect |
|---|---|
| decoder Engram reads removed | the decoder's ×1.10 multiplier goes away: ≈ −1.7% of the step |
| pooler FFN removed | < −1% |
| R: one layer of full causal attention over 1,020 keys | ≈ 260k MACs per token, ≈ +25% of one decoder layer, ≈ +1% of the step |
| F loop off (if §9.4 or the fallback drops it) | ≈ −11 to −16% of the step (v4 §3.3's +13–20%, inverted) |

- **Net with the loop kept:** ≈ −1 to −2% against v4.
- **Net with the loop dropped:** ≈ −12 to −17%, ≈ 1.25–1.35× v3's speed.
- These are pre-compile estimates. `profile_step.py` and `ab_interleave.py` on the real v4.1 model replace them.
- **Memory.** R adds one T × T attention per micro-batch row. The encoder already runs two F layers twice. The net is
  within v4's footprint.
- **Inference cost.** v4's decoder had a per-token cost independent of context length (window 8 everywhere). R needs a
  KV cache over the document, the standard transformer cost. The encoder's F layers already need one, so this adds one
  more cached layer.

---

## 9. Evaluation, gates and the pre-launch tiny gate

### 9.1 Evaluation suite

v4 §9.1 applies, with these changes.
- **Removed:**
  - the decoder read switch-offs and frequency bins (C2);
  - "pooler FFN off" (C3);
  - "F loop off", if the loop is dropped.
- **Added:**
  - **R local** (§6.6).
  - Linear probes of u for the answer entity at both passes (§5).
  - **Single-factor OOD sets.** v3's OOD split changes three factors at once (names, distance 500–820, more facts), so its
    0.041 cannot say which one breaks retrieval.
    - In `run_m2_25M/code/probes.py` one flag (`self.ood = split == "ood"`, lines 36–37, 62, 103, 128) switches all three
      together.
    - **Engineering item before launch:** extend `ProbeFactory` to accept one OOD factor at a time. Then build three
      512-item sets per family (distance only, names only, fact count only), once, with fixed seeds, stored beside
      `test.pt`.
  - **The test-time controllability measurement** (§10.3), at 48 h only.

### 9.2 Gates at matched tokens against v3 (v4 §9.2, reread for v4.1)

- **G1–G5 are kept as capability gates:**
  - G1, nothing dead: the zero-init branches minus the removed ones, plus R's τ moved.
  - G2, language within v3 + 0.05.
  - G3, recall onset by step 1,792.
  - G4, probes alive at 4,050.
  - G5, the second pass works at 24 h.
- **They no longer attribute success to the concept route.** In v4.1 an early recall1 can come through R.
- **New, G6 · the concept route is used** (24 h, validation):
  - `core_off` costs ≥ **0.05 nats** of mean language loss (v3: +0.59; tiny-2 v4 text: +0.12), **or**
  - drops recall1, carry3 or multi4 by ≥ 2 SE.
  - Reported beside "R local".
  - **Kill line:** `core_off` < 0.02 nats *and* no probe moves by ≥ 2 SE. The owner then hears it plainly: "v4.1 is
    behaving as an LLM-like token path with an idle core, which is a different design from the one approved."
- **No gate stops the run** (v3 policy).

### 9.3 Reporting rules (v4 §9.3, plus three)

1. Probes are **capability** readings. Route attribution is read only from `core_off`, `plan_swap` and "R local", each
   as *use*, not exclusivity.
2. The v4 and v3 probe curves are compared at matched tokens as capability, with the note that v3's route was exclusive
   and v4.1's is not.
3. "R local" is out of distribution like every switch-off. A large cost proves R is used; a small cost suggests it is
   not, and only the trained no-R arm confirms.
4. **A pre-registered prediction of the division of labour (read at 24 h and 48 h).**
   - R is one content-addressed hop at token level. Multi-fact composition and pointer-following need the core.
   - So in v4.1, `core_off` should cost **hop and multi4 more than recall1**, and "R local" should cost **recall1 more
     than hop**.
   - If both hold, that is the attribution the probes can still give. If R local costs multi4 and hop as much as core
     off, R is doing the core's composition, and §11.5 lists that as a risk.

### 9.4 Pre-launch tiny gate (pre-registration; written before any v4.1 run exists)

**Purpose.** Test C1 (R) where v4 failed, before any full-scale run, using the existing tiny-2 harness and rule. Also
decide C6 (the F loop). **Nothing is launched until the owner approves this section.**

**Harness.** `v4_tiny/code/run_vs.py`, d 64. These additions are done-criteria; the pre-registration cannot be executed
without them:
- `--arch v41`: the tiny v4 config with decoder layer 2 replaced by R (§6.2).
- `--arch v41stack`: the same with keys and values from x = GRRead([D]) in all four decoder layers (§6.6). Parameters are identical to
  v41.
- `--arch v41` with `enc_loop = 1` in training: the loop-off arm.
- `interventions` for both: `core_off`, `plan_swap`, `M1`, `loop_off`, `R_local`.
- `analyze_vs.compare()` produces the paired differences v41 − no-R (the tiny-2 v4 runs), v41 − LLM, v41stack − v41 and
  loop-off − v41. It is keyed on (data, budget, arch, seed) as now.
- `test_vs.py` gains the §6.7 tests at d 64. `test_param_match` asserts v41 = 1,212,036 (no-R + 4) and v41stack = v41
  exactly. The LLM (1,206,864) stays within the existing −0.43% tolerance.

Held equal as in tiny-2 §1:
- the rows;
- AdamW 0.9 / 0.95, weight decay 0.1;
- lr 2e-3, warm-up 400, cosine to 0.1;
- B 4, T 480, clip 1, block 60;
- the budgets: text 5M real tokens, pinpad 6M.

**Two checks before any reuse** (each must pass, or the reused runs are re-run):
1. **The tiny-2 v4 runs are this gate's no-R arm.**
   - `vs_pin_v4_s{0,1,2}_6M` and `vs_text_v4_s{0,1}` were built with k = 6, pool = attn, `dec_reads = ()`, M = 2 and
     `enc_loop = 2`, at compute 1,212,032. This was checked in `run_vs.py` `build` and by instantiation.
   - v4.1-tiny must differ from them only in decoder layer 2. The parameter difference must be exactly +4.
2. **The tiny-2 LLM runs are reusable** only after `test_runner_rows` passes with `--arch v41`. That means byte-identical
   training rows, the same stream seed and the same k = 6 padding. `test_same_rows` must also pass, and the 11 mutants of
   `mutants_vs.py` must still be killed, plus the four R mutants of §6.7.

**Arms and budget:**

| priority | arm | pinpad 6M | text 5M |
|---|---|---|---|
| mandatory | **v4.1** (R at layer 2) | seeds 0, 1, 2 | seeds 0, 1 |
| mandatory | **stack** (NCP-style, D non-private, layer 2 full) | seeds 0, 1, 2 | — |
| mandatory | **v4.1 loop off** (trained, `enc_loop = 1`) | seeds 0, 1, 2 | — |
| reused | no-R (= tiny-2 v4); LLM (tiny-2) | done | done |
| if time | C1b (G in R's keys) | seeds 0, 1, 2 | — |
| if time | C16 (S windows 60 in the encoder and decoder, v4.1 otherwise) | seeds 0, 1, 2 | — |
| if time | C14: encoder 6 (SSFSSF: v4's layers 2 and 6 removed; Engram read 2 after the new layer 3; the shared FFN spans 6 layers; F loop on layers 2 and 5) / decoder 6 (S S R S S S) | seeds 0, 1, 2 | — |

- **Size.** 11 mandatory runs. On the 4 P-cores at 2 threads, two to three at a time, that is ≈ 4–5 h of wall-clock.
  This uses tiny-2's measured ≈ 55–65 min per 6M pinpad run with v3 sharing the CPU; v3 is now finished.

**Primaries** (unchanged from tiny-2 §3):
- P1: held-out optimal-action rate;
- P1b: held-out b2;
- P2: closed-loop pair integrity;
- T1: text validation NTP.

All are paired by episode or row, with SE across seeds = √ΣSE²/n.

**Outcomes and what each means.** These are written now.

| outcome | condition (P1, paired) | what it means | recommendation to the owner |
|---|---|---|---|
| **O1 · fix confirmed** | v4.1 − LLM ≥ −2 SE (and the text guard below holds) | the token path's missing long-range read was the main deficit | carry R to full scale (§6) |
| **O2 · partial fix** | v4.1 − no-R ≥ +2 SE, but v4.1 − LLM < −2 SE | R carries part of the gap | read the stack arm. If stack ≈ LLM, the rest is token-level plan composition: the owner chooses between v4.1 (the core composes) and the stack ([NCP] recipe; see the stack rule). If the stack is also short of the LLM, the rest is elsewhere (window 60, the LLM's 1.35× FLOPs, the bottleneck's capacity): carry v4.1, which is strictly better than v4, and say so |
| **O3 · no effect** | \|v4.1 − no-R\| < 2 SE | access type was not the cause | do not carry R. The remaining candidates are named (C16 window, FLOPs, capacity). tiny-2's verdict stands as the owner's input |
| **O4 · harm** | v4.1 − no-R ≤ −2 SE | R hurts | revert to v4's decoder |

**Conditions that apply in every outcome:**
- **Text guard (the one text bound).** R is carried only if text v4.1 − no-R ≤ +0.02 nats (paired, 2 seeds). Since no-R
  − LLM = −0.007 ± 0.004, this also keeps v4.1 within ≈ +0.013 of the LLM, so tiny-2's +0.05 bound against the LLM never
  binds and is not used.
  - **If P1 passes (O1 or O2) but the text guard fails, R is not carried.** The report states a language regression
    caused by R, with both numbers.
- **Route kill line on v4.1 itself.**
  - The route is used if `core_off` lowers pinpad held-out b2 by ≥ 2 SE, *or* raises text NTP by ≥ 0.02 nats. 0.02 is the
    same kill line as G6 at full scale (§9.2). The tiny gate asks only whether the route is alive, so it has no "clearly
    used" level (0.05 at full scale). Both sit far below the observed values: tiny-2 v4 +0.124, v3 +0.59.
  - **If R passes but the route is unused**, the report says: "at this scale v4.1 is an LLM-like token path with an
    idle core; that is a different design". The owner decides; it is not presented as a v4 success.
- **Stack rule** (dec-v1's kill rule carried over). If stack − v4.1 ≥ +2 SE on P1, the plan leak is doing work the core
  should do.
  - Read the stack's `core_off` and M = 1 before accepting it.
  - If its core is idle (`core_off` below the route kill line), the stack is an LLM with overhead: reject.
  - If its core is used, the owner decides between privacy of plans and the gain.
- **Loop rule (C6).** Keep the F loop at full scale only if loop-off − v4.1 is ≤ −2 SE on held-out b2, or the loop-off
  arm's text is worse by ≥ 0.01 nats. Text is read only if run. Otherwise drop the loop.
- **Budget-limited label** (tiny-2 §3). If the better arm's held-in b2 is still rising over its last three evaluations at
  6M, the reading is labelled "budget-limited". There is no extension beyond 6M.
- **Reporting.** Every reading comes with its paired SE and FLOPs per token. R adds ≈ +1% at d 64 by the same estimate as
  §8.2.

**If the owner does not run the gate.** The full-scale default is v4.1 with R and with the F loop off (§4.2), and the
24 h "R local" and G6 readings are the only attribution. This is weaker, and the report must say that R was untested.

---

## 10. The concept interface after pretraining: three uses, one site

v4 §5.4 treated "the RL door" as one question. The three papers show three distinct uses of a pretrained latent site:
- **internal RL**, with a metacontroller on the frozen model ([ETA]);
- **test-time optimisation** of an offset at the site ([GC]);
- **adaptation of the concept vocabulary alone** ([NCP] §5.1).

None of them changes pretraining. All three need the same site with the same properties, so the site is chosen first.

### 10.1 Choosing the site from the requirements

**Requirements, each from a paper:**
1. **Every later token is downstream of the site.** This gives dense credit assignment ([GC] eq. 7) and lets a control
   persist over the following abstract action ([ETA] Algorithm 1).
2. **Enough computation below the site to unfold the abstraction into tokens.** [ETA] Fig. 4: control near the middle is
   best. [GC] Fig. 5: 25–50% of depth is best, 75% is worse.
3. **On-manifold start.** The variable is an *offset* on real states, initialised to 0 ([GC] App. A.1). [ETA]'s
   controller is e + U e, so it is the identity when U = 0.
4. **Low-dimensional actions and a contracted horizon** for RL ([ETA]: n_z = 8, one decision per abstract action).

**The candidates, checked against the architecture:**

| site | 1 · every later token downstream? | 2 · position on the longest path (encoder 10 sequential layers with the loop, core 8, decoder 4 = 22) | verdict |
|---|---|---|---|
| u_j (the plan's input) | **no.** Plans are private (§6.3): an offset on u_j reaches exactly the 6 tokens of chunk j + 1, and later tokens only if the core rereads it, which it does not | after 18 of 22 (82%) | rejected |
| **ĉ_j, the core's input** | **yes.** An offset on ĉ_j changes S₁ and S₂ at every later concept of the document through the core's causal attention, hence every later u, plan and token | after 10 of 22 (45%) | **chosen** |
| S₁, between the passes | yes, as ĉ_j, through pass 2's attention | after 14 of 22 (64%) | the alternative, measured beside ĉ in §10.3 |
| D at decoder layer 2 (token rate) | only through R's keys, which read `tok`, not D: **no** | 20 of 22 | rejected |

- **Requirement 3:** met at every site by an offset initialised to 0.
- **Requirement 4:** met by the metacontroller of §10.2. Its code z is 8-dimensional and acts at concept rate (T/6), or
  less often when β holds.
- **The 67% figure in §2.2** is the position of the plan's entry on the *token path* (8 of 12 layers). It concerns where
  the abstraction is *read*, not where it is *steered*. The steering site ĉ sits at 45% of the longest path, inside
  [GC]'s best band. C14 (rebalancing) remains the arm that tests whether the read point matters.

### 10.2 Use 1: internal RL through a metacontroller on the frozen model (v4 option C, made concrete)

**Revisiting v4 §5.4 with v3-48h.**
- v4 §1.7 (2) argued that the decoder barely depends on the code (c_off 0.013 at 24 h). At 48 h on test, c_off is
  +0.021 / +0.025 nats and multi4 −0.051.
- The text depends on the code more than assumed, but the continuous state u still carries most of the plan (v3 u_off
  +0.50 at 24 h). So:
  - **A + C stays the recommendation.** Pretraining is unchanged; control is post-training at the ĉ site.
  - **Option B (posterior-code dropout during pretraining) stays not recommended.** It is [ETA]'s co-trained case, which
    collapsed (Fig. 7).
  - **New, a cheaper discrete variant C′.** A policy over the coarse code at concept rate replaces ĉ_pred's coarse part
    with a chosen centroid E⁰_k̂. The 0.02-nat c_off bounds its authority over the text, so it is a secondary action space
    beside C.

**The metacontroller** ([ETA] App. D.2, at concept rate):
- **Reads and writes the same site, as [ETA] does** (App. D.2: μ, Σ, β and the GRU all read e_{t,l} before control, and
  the write is ê = e + U e).
  - The read is ĉ_j **before** control. It comes from the encoder, so the control cannot change it.
  - The history is h_{j−1} = GRU([ĉ_{j−1}; S₂,j−1], h_{j−2}), n_h = 32. S₂ of strictly earlier concepts is legal (it
    already includes earlier controls). [ETA] D.3 is the reason for a recurrent read: a single site does not hold the
    whole history.
  - Reading S₂,j itself would be circular, because S₂,j is computed from the controlled ĉ_j.
- **Proposes** z̃_j ~ N(μ_j, Σ_j), with n_z = 8. In training μ and Σ come from a non-causal sequence embedding
  s(ĉ_1:M) (an SSM with n_s = 32), from ĉ_j and from h_{j−1}.
- **Switches** by z_j = β_j z̃_j + (1 − β_j) z_{j−1}, with a causal gate β_j = f(ĉ_j, h_{j−1}, z_{j−1}). This is the
  switching gate moved out of v4 §4.7 (C9). Its duration is now learned *on top of* fixed 6-token concepts, so the
  abstract action can last several concepts.
- **Writes** a low-rank (16) linear controller U_j = f_hyp(z_j) into the core's input, ĉ_j ← ĉ_j + U_j ĉ_j, before
  pass 1.
- **Trained** on the frozen v4.1 by NLL + α · KL(N(μ, Σ) ‖ N(0, I)), with α swept as [ETA] does (0 to 1). Training uses
  the clean data; [ETA] App. C.4.4 found clean demonstrations matter for the metacontroller.
- **Internal RL.** A causal policy π(z_j | ĉ_j, h_{j−1}) replaces the non-causal encoder. β is thresholded. RL runs in z-space
  with the critic-free relative advantage of [ETA] App. C.5.2.
- **The first test bed is gridworld-pinpad with the tiny v4.1,** because it has hidden subgoals and a sparse-reward
  post-training task (12 colours), as in [ETA].
- **Status:** a specification for after pretraining. Nothing here is built or run before the owner asks.

### 10.3 Use 2: test-time latent optimisation ([GC]), pre-registered as a controllability measurement at 48 h

**This is a measurement, not a gate.** It answers whether the pretrained v4.1's latent site can be steered toward a
correct answer, and how that compares with random steering and with the u site.

**What it isolates.** `tok` and R are computed before the pooler, so an offset on ĉ cannot reach the token path at all.
Its only route to the answer is core → plan → decoder.
- The measurement is therefore the **concept route's control authority**, which is exactly what the RL door (§10.2)
  needs to know.
- It does not depend on how well R solves the item.
- If R already answers most probe items, the initially-wrong pool is small. So N is reported per family, and the oracle
  objective (B) on all items is the reading with the most items.

**Items.**
- 128 items per family (recall1, carry3, multi4, hop) from the test split.
- Objective (A) stops at once on items already answered correctly, so the "does no harm" reading comes from (B) and the
  random controls, which run all items for a fixed K.

**Variable.**
- An offset Δ on ĉ_j for every chunk of the prompt span (the chunks before the answer's first token), initialised to 0.
- The model's weights are frozen.

**Objectives, both run:**
- **(A) [GC]-faithful.**
  - Greedy-generate the answer; R = 0 if it is correct, −1 if not.
  - Loss = −R · Σ log p(generated answer tokens) (App. A.1 eq. 8), so a wrong answer is pushed away.
  - Stop when R = 0 or after K = 10 updates.
- **(B) oracle upper bound.** Maximise log p(gold answer), K = 10. This measures how much the site *can* move the
  answer, not how a reward finds it.

**Optimiser.** Adam (0.9, 0.999).
- ĉ lies on the sphere of radius √256, per-dimension rms 1, so the learning rate is per dimension in those units.
- Three learning rates are pre-registered: {1e-3, 1e-2, 3e-2}. All are reported, because [GC] Fig. 2's point is
  sensitivity.
- **Sphere.** After each update ĉ_j + Δ_j is re-projected to radius √256.
  - [GC] applies no norm constraint. v4.1 differs on purpose: the core was trained only on sphere inputs, and §10.1's
    requirement 3 is to stay on the manifold the model was trained on.
  - The unconstrained variant is not run.

**Controls.**
- **No optimisation**: the baseline accuracy.
- **Random direction, as [GC] ran it.** Each step adds η · ε / ‖ε‖ with ε ~ N(0, I), the same η, no Adam and no norm
  matching (App. A.1, Fig. 3).
- **Random direction, norm-matched.** The same, scaled to the Adam run's mean step norm. Unlike [GC], this makes the
  comparison with the reward-guided run fair.
- **The same procedure at the u site.** It is expected to be weaker by §10.1's reach argument.
- **The same procedure at S₁.**

**Metrics.**
- The fraction of initially-wrong items fixed within K.
- The fraction of initially-right items broken.
- The mean gain in log p(gold).
- The mean number of updates used.
- All per family, with binomial SEs.

**Cost.** ≤ 10 forward + backward passes per item on the 19M model: minutes, off the training clock.

**Status.** It runs as part of the 48 h evaluation, and only if the owner launches v4.1. Nothing runs before that.

### 10.4 Use 3: concept-vocabulary-only adaptation ([NCP] §5.1)

- **Trainable set.**
  - The codebooks: E⁰ 262,144 and PQ 65,536.
  - The head: W_q, the temperatures and the out norm, 65,825.
  - Total 393,505, 2.0% of compute plus codebooks.
- **The route it uses.** The decoder reads codes only through ĉ_pred (W_c ĉ_pred in the plan). That is the same route
  v3-48h's c_off measured (+0.02 nats), so the authority of this interface is bounded by it at this scale.
- **Status:** a description. Nothing runs.

---

## 11. Whole-architecture review

### 11.1 Data flow and shapes (T = 1,020, k = 6, M = 170)

1. **ids, doc** (B, 1,020) → Engram indices → one gather → e (B, 1,020, 384), used by the encoder only.
2. **Encoder** (8 layers, F loop per §4.2) → L, G (B, 1,020, 256); `tok` = zcRMSNorm(L); router mix m.
3. **Pooler** (4-head attention, no FFN) → ĉ (B, 170, 256) on the sphere. **This is the steering site of §10.**
4. **Core** S₁ = Φ(ĉ, 0), S₂ = Φ(ĉ, S₁) → u = zcRMSNorm(S₂) → coarse logits (B, 170, 1,024), fine logits
   (B, 170, 32, 256) → ĉ_pred (B, 170, 256).
5. **Plan.** P_t = W_u u_{j*} + W_c ĉ_pred,{j*} + o_{r(t)}; null for the first 5 tokens of a document.
6. **Decoder.**
   - D₀ = `tok` + P.
   - Layers 0, 1 and 3 are confined (keys and values `tok`, window 8).
   - Layer 2 is R (queries GRRead(D); keys and values `tok`, full causal, same document; NoPE; τ_h).
   - Logits = zcRMSNorm(D) @ Eᵀ, shape (B, 1,020, 8,192).

**What each tensor may see:**
- `tok` sees ≤ 63 tokens back.
- R's output at t sees `tok` at every earlier position of the document.
- D_t sees only its own plan.
- u_j sees concepts ≤ j.
- **The token path's reach is the document. Plan composition happens only in the core.**

### 11.2 Step 0: what each new or changed part does

| part | init | at step 0 | alive because |
|---|---|---|---|
| R (replaces confined layer 2) | as a confined layer; log τ_h = 0 | **not an identity**: a full-range read from the first step; "R local" is its switch-off | its weights are those of a normal layer |
| pooler without FFN | q = b = p = 0, W_V = W_O = I | masked mean of the chunk | ∂q through the random W_K; ∂p through the values |
| everything else | as v4 §10.2 (encoder shared FFN, codebooks) | as v4 | as v4 |

### 11.3 Mistake ledger: lessons since draft_v4, and v4.1's countermeasure

| lesson (source) | v4.1 |
|---|---|
| An output-layer n-gram read becomes a memorisation channel: positive on seen text, negative on unseen (tiny-1) | decoder Engram reads removed (C2) |
| A branch that writes is not a branch that helps: the pooler FFN wrote with gain 0.6–0.7, Δ ≈ 0 (tiny-1) | removed (C3); every branch is judged by a switch-off, not by its write share |
| A per-slot pool collapses the coarse code (tiny-1) | attention pooler kept (C4) |
| Training fillers drawn from too few documents let probes be memorised (tiny-1, design flaw) | fillers from 60,000 documents, validation fillers from the val store, kept for the tiny gate (tiny-2 §2) |
| The F loop and the second pass showed ≈ 0 where their output reaches prediction only through concepts; the LLM's paid (tiny-2) | loop decided by a trained arm with a fallback of off (C6); C1b tests the token-level exit |
| v4's token path had no content-addressed long-range read; the gap was access type, not distance (tiny-2) | R (C1), gated by §9.4 |
| A post-hoc figure ("11 tokens stale") double-counted the shift (tiny-2 §5 remark) | corrected to 0 … 5 (§1.3); numbers in the design are rederived from the code |
| Language and items disagree on M and the shared FFN (v3-48h) | both kept on the item evidence (C7, C8) |
| Co-trained switching collapses ([ETA] Fig. 7) | the switching arm moved to post-training (C9, §10.2) |
| Pre-registered arms were not run (V3-016) | the tiny gate is small (11 runs, ≈ 4–5 h) and runs *before* the full-scale launch |

### 11.4 The owner's brief, point by point

| owner | v4.1 | agreement |
|---|---|---|
| keep the encoder: 8 layers, L/G streams, gated residual, Engram at several depths | unchanged | **agree** |
| encoder looping | F-layer loop kept only if the trained arm shows it pays; fallback off (three ≈ 0 readings) | **agree on the mechanism; its default now follows the evidence** |
| shared FFN in the core and in the encoder | both kept; core at full 1,024 (v3-48h: −0.164 multi4 without it) | **agree** |
| concept module 4–5 layers, recurrence 2 | 4 × 2; 5 × 2 rejected (it halves the shared FFN) | **agree on 2 passes; 4 layers on evidence** |
| bigger codebook | coarse 1,024 + PQ 32 × 256 × 8 | **agree** |
| 6 or 8 tokens per concept, attention (or FFN, or both) instead of averaging | k = 6, 4-head attention, mean init; the FFN removed (the owner's own post-tiny choice); k = 4 control, k = 8 arm | **agree** |
| **decoder as v3, with Engram, sharing the encoder's table** | **decoder Engram removed** (tiny-1: a memorisation channel, −0.14 to −0.17 on unseen probe text); **one decoder layer becomes R** (tiny-2's access-type gap; [NCP]'s full-attention decoder) | **disagree, on evidence**: both are stated with their numbers; R is gated by §9.4 and the owner decides |
| losses as v3 | v4's form (NTP + coarse and fine CE + VQ) | **agree** |
| < 20M, d 256 | 19,175,580, d 256 | **agree** |
| concept-level RL is more sample-efficient | true for a small action at a T/6 (or β-contracted) horizon: the metacontroller's z (§10.2); plus test-time optimisation and vocabulary adaptation | **agree, extended** |
| "take advice from the papers" | [NCP]: additive concept path, full-attention decoder half; [ETA]: frozen-base switching, mid-depth control; [GC]: a dense-credit latent site, measured at test time | §2, §3, §10 |

### 11.5 Risk register

| risk | sign | fallback |
|---|---|---|
| R bypasses the core (the route idles) | §9.4 route kill line; G6 at 24 h (`core_off` < 0.02 nats, no probe moves) | the owner hears "LLM-like token path with an idle core"; revert R or accept the different design |
| R does the core's composition | the §9.3 (4) prediction fails: "R local" costs hop and multi4 as much as `core_off` | the core's role shrinks to plans for local tokens; consider the stack arm's reading |
| R has no effect (§9.4 O3) | \|v4.1 − no-R\| < 2 SE | revert to v4's decoder; the window and FLOPs candidates remain |
| the loss of the exclusivity proof hides a dead route | probes look fine while `core_off` is small | G6 and the kill line read it directly |
| the F-loop decision is wrong at full scale | the 24 h "F loop off" switch-off, if the loop is kept | v5 revisits |
| the G/L gradient ratio falls too far (G trained only through the route) | ratio below half of v3's at 24 h | pass-1 NCP arm, or raise λ_c and λ_f (v4 §10.5) |
| OOD retrieval stays at 0.04 | the single-factor OOD sets show which factor | names → Engram / embedding; distance → R and the core; fact count → the core |
| inference KV cache over the document | — | standard transformer cost; noted in §8.2 |
| memory growth, data rebuild, coarse code unused | as v4 §10.5 | as v4 §10.5 |

### 11.6 Open decisions for the owner

1. **Approve and run the tiny gate (§9.4)**: 11 runs, ≈ 4–5 h on the CPU. Or launch without it under §9.4's fallback,
   which is weaker.
2. **Accept the loss of the exclusivity proof.** Probes become capability readings; the route is read from `core_off`,
   `plan_swap` and "R local" (§6.3, §9.3).
3. **Decoder Engram removed and R added**: the two points where v4.1 departs from the brief (§11.4).
4. **F loop fallback if the gate is not run**: off (recommended) or on.
5. **If §9.4 gives O2 with the stack ≈ LLM**: v4.1 (plans private, the core composes) or the stack ([NCP] recipe).
6. **Concept interface after pretraining**: A + C (recommended), with C′ and §10.3 as measurements. B is not recommended.
7. **v5**: learned concept boundaries. [ETA]'s switching on the frozen model (§10.2) is the first step, and [GC]'s
   gradient concentration on transition tokens says where boundaries should fall.

---

## 12. Advisor log

| # | step | verdict |
|---|---|---|
| 1 | approach, after reading the three papers, draft_v4, tiny-1, tiny-2 and v3-48h | Lead with a change ledger with three checks per entry; inherit unchanged sections by reference. Verify from code before pricing (decoder layer anatomy, kernel window cost, OOD definition, the tiny window confound). Expect the decoder to be the central change: state what is lost (the exclusivity proof), do not assert the cause, and pre-register a tiny gate on the existing harness. Depth rebalancing is an arm (the LLM with the same split won). Remove the co-trained κ arm (ETA Fig. 7). Let the gate decide the F loop. Keep M = 2. Use [NCP]'s coarse codebook only to downgrade the VE worry. Spare parameters are not a reason to add. |
| 2 | §0–§3 (evidence, papers, ledger) | Numbers trace. Fixes: staleness is 0 … 5 tokens, not 11 (tiny-2's remark double-counted the shift); the −0.307 is the b2 gap; [NCP]'s per-layer re-injection is not dec-v1's read arm, so C12 is downgraded to defer; §0's "diagnosed" hedged to match §1.3. For §6–§9: R needs a switch-off ("R local"), tests and a mutant (keys from D must fail D2), the gradient-reach map, probes as capability readings with a route kill line, and a full outcome table with reuse checks and a loop fallback. |
| 3 | §4–§9 (spec, budget, gate) | Sums check; the outcome table partitions P1. Fixes: `probes.py` has one OOD flag, so single-factor sets are an engineering item; D1a asserts no change before p and reach to the document end, not "every position"; one text bound (v4.1 − no-R ≤ +0.02) and the P1-pass / text-fail case named; the stack arm made parameter-identical (keys from D, not dec-v1's `StackLayer` with convolutions); harness changes listed as done-criteria; the two route thresholds reconciled; inference KV cost stated. Added: the division-of-labour prediction (§9.3 (4)). For §10: the steering site must be ĉ (or S₁), not u, because plans are private. |
| 4 | §10–§12 (interface, review, log) | The site argument is right (u_j is never reread by the core; ĉ_j and S₁ propagate), and the 10 / 14 / 18 of 22 depths check. Fixes: the metacontroller read S₂,j, which is computed from the ĉ_j it writes (circular), so it now reads and writes the pre-control ĉ_j with a GRU over earlier ĉ and S₂, as [ETA] does; the stack arm's keys and values come from GRRead([D]), as `StackLayer` did, so no new norm is needed; the random control is run as [GC] did and also norm-matched; "does no harm" comes from (B) and the random controls; the sphere re-projection is a stated choice. Added: an offset on ĉ cannot reach the token path, so §10.3 measures the concept route's control authority alone; report N per family. |
| 5 | review passes and final check | The three review passes (consistency by grep of every § / C reference and parameter figure; first principles, principle by principle and inversely; the owner's brief and standing rules) were reviewed together. They changed: the §2.4 principle-to-entry table, labels on external section references, the §10.3 status line, and, after a full read, three stale cells (§2.1's per-layer row now says defer, matching C12; §2.3's site row now says ĉ_j, not u_j, matching §10.1; the §0 encoder row points to the pooler row) plus C8's net figure. No design or gate change. Deliver with the owner message; launch nothing. |
