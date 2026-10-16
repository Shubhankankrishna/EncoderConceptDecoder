# HRM Thinker v4 — draft architecture (< 20M compute, d = 256), derived from v3

Status: **complete draft, reviewed** (2026-10-01). Each section was written, saved, reviewed with the advisor and revised.
The whole document was then cross-checked as one. **Revision 2 (2026-10-01):** §3.5 was rewritten after the owner's
correction on concept compression. It now covers k tokens → 1 concept with an attention + FFN pooler, k = 6 vs k = 8 measured, and v3's pooling as
it actually trained. Every dependent number was propagated. The advisor's verdicts are logged in §11. Nothing in this document changes the running v3 run
(`run_v3_19M`, step ≈ 11,400 of ≈ 16,250 at first writing; ≈ 11,800 at revision 2).

**Sources.** Every number below is traced to a file:
- `run_v3_19M/runs/v3_48h/eval_24h.json`: the v3 24 h milestone evaluation, **validation split**, step 8,133.
- `run_v3_19M/runs/v3_48h/log.jsonl`: the v3 hourly and step-matched evaluations, validation split.
- `run_m2_25M/runs/main/eval_final.json`: the v1 final evaluation, **test split**, step 4,050.
- `run_m2_25M/webapp/data/ref_v1v2.json`: the v1 and v2 validation evaluations.
- `run_v3_19M/DECISIONS.md`: V3-001 to V3-017, including the pre-registered v4 rules in V3-016.
- *Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning* (Kobayashi et al.,
  Google, arXiv 2512.20605), cited as **[ETA]**.
- *Loop the Loopies!* (IQuest, arXiv 2607.16051), cited as **[Loopie]**.

**Comparison rule.** Language is compared validation against validation (v1 `ref_v1v2.json`, v3 `log.jsonl` /
`eval_24h.json`). Probes are compared with the split named beside each number. The like-for-like test-split comparison
waits for v3's 48 h evaluation.

---

## 0. v4 at a glance

| | v3 (running) | v4 (this draft) | where |
|---|---|---|---|
| width, vocabulary, row | d 256 · 8,192 · T 1,024 | d 256 · 8,192 · **T 1,020** | §3.5 |
| compute parameters | 18,986,871 | **19,898,011** (< 20M, 101,989 spare) | §7.1 |
| encoder | 8 layers SSSFSSSF, L/G gated-residual streams, one Engram gather → reads at depth 0 and 4 | same, **F layers 3 and 7 applied twice** (layer-loop; reach(L) stays 63), **shared SwiGLU 384** | §3.3, §3.4 |
| tokens per concept | 4, learned linear pool over the concatenation (starts as the exact average, trained into a recency-weighted per-slot map) | **6 (k = 8 as an arm), attention + FFN pooler**: 4-head attention selects, SwiGLU 384 composes; mean-pool init, pads masked | §3.5 |
| concept core | 4 layers × 3 passes, (t, Δ) table, own FFN 736 + shared 1,024, sampled shortcut 1 / 2a / 2b | **4 layers × 2 passes**, no table, own 736 + shared 1,024, **terminal-only** | §4.1–§4.4 |
| codebook | PQ 32 × 128 × 8 | **coarse 1,024 × 256 + PQ 32 × 256 × 8 on the residual** | §4.6 |
| decoder | 4 confined layers, plan from (u, ĉ_pred) shifted one chunk and repeated over the next 4 tokens, no Engram | same, **two Engram reads (after layers 0 and 2) of the encoder's gathered rows**, plan repeated over 6 tokens | §5 |
| objective | NTP + 0.1 NCP + 0.05 NCP(shortcut) + 0.1 L_scons + VQ | **NTP + 0.05 CE_coarse + 0.1 CE_fine + VQ_coarse + VQ_fine** | §6 |
| est. throughput | ≈ 3,200 tok/s | ≈ 1.1–1.2× v3 (A/B decides) | §7.2 |
| RL door | — | coarse code = NCP target and abstraction measure; post-training controller at concept rate (owner decides A + C vs B) | §5.4, §10.6 |

## 1. What v1 and v3 measured (the evidence base)

### 1.1 The runs

| | v1 (run_m2_25M) | v3 (run_v3_19M) |
|---|---|---|
| width, compute parameters | d 384, 25.05M | d 256, 18.99M |
| core | 1 shared layer looped R ~ U{1..4} | 4 distinct layers, M = 3 passes + shortcut 1 / 2a / 2b |
| PQ codebook (segments × codes × dims) | 48 × 128 × 8 (`run_m2_25M/code/model.py`), 336 bits per concept | 32 × 128 × 8, 224 bits per concept |
| throughput (M2, MPS) | ≈ 1,700 tok/s | ≈ 3,200 tok/s (block-local attention + `torch.compile`, V3-006) |
| steps (32,768 tokens each) | 4,050 in ≈ 22 h | 8,133 at 24 h; ≈ 16,250 at 48 h |

Both runs use the same data, tokenizer and probe items, and the same order in epoch 0, so step k means the same tokens
(V3-006: this holds up to step ≈ 8,556). This holds for v1 and v3 only: v4 re-packs the rows (§8.2).

### 1.2 Language and probes over training (validation)

| step (hours of v3 training) | v3 val | v1 val at the same step | recall1 v3 | carry3 | multi4 | hop |
|---|---|---|---|---|---|---|
| 1,217 (3.7 h) | 3.319 | 3.227 | 0.148 | 0.000 | 0.000 | 0.000 |
| 2,376 (7.1 h) | 3.005 | 2.913 | 0.773 | 0.000 | 0.000 | 0.062 |
| 4,050 (12.2 h) | 2.813 | 2.754 (v1 final) | 0.992 | 0.617 | 0.086 | 0.102 |
| 8,133 (24 h, milestone) | 2.596 | — | 0.992 | 0.922 | 0.523 | 0.141 |
| 11,348 (33 h) | 2.528 | — | 1.000 | 0.945 | 0.688 | 0.164 |

- **At the same tokens v3 trailed v1 by ≈ +0.09 nats** on both languages from step 1,217 to ≈ 3,100. The gap then
  narrowed to +0.08, +0.07 and +0.059 at 4,050, while v1's learning rate annealed to its floor and v3's was still 5.2e-4.
  - The gap is constant only while both runs are mid-schedule.
  - v3 has 24% fewer compute parameters and a narrower width (256 vs 384).
- **Per hour, v3 is ahead.** It passed v1's final validation loss at 15.2 h (step 5,151) and is 0.23 nats below it at 33 h.
- **Long-range probes improved far past v1.** At v1's last step v1 had multi4 0.055 and carry3 0.594 (val). v3 reaches
  multi4 0.688 and carry3 0.945 at 33 h.
- **hop stays low** (0.10–0.19). By depth at 24 h it scores 0.256 / 0.125 / 0.081 for 2 / 3 / 4 links, which fits
  ≈ 0.5 success per link applied independently. Retrieval is solved (recall1 1.000); following a pointer is not.

### 1.3 What each component carries (24 h switch-off tests, validation, Δ next-token nats)

| switched off | TinyStories | WikiText | mean | v1 (test, 22 h) |
|---|---|---|---|---|
| concept core (null plan) | +0.666 | +0.514 | +0.590 | +0.578 / +0.455 |
| plan swapped (another document's) | +1.115 | +0.384 | +0.749 | +0.872 / +0.222 |
| core state u in the plan | +0.598 | +0.407 | +0.503 | +0.511 / +0.430 |
| **predicted concept ĉ_pred in the plan** | **+0.013** | **+0.013** | **+0.013** | +0.024 / +0.020 |
| Engram (both reads) | +0.476 | +0.719 | +0.598 | — |
| Engram read 2 | +0.015 | +0.042 | +0.028 | — |
| encoder F layers | +0.168 | +0.111 | +0.140 | — |
| (t, Δ) table | +0.007 | +0.009 | +0.008 | — |
| shared FFN | +0.004 | +0.003 | +0.004 | — |
| **1 pass instead of 3** | **+0.007** | **+0.004** | **+0.005** | R1 vs R4: +0.001 / +0.001 |

**Probes under the same switch-offs** (24 h, validation, 512 items per probe; one item is 0.2 points; the standard error
of a difference near 0.5 is ≈ 0.031):

| switched off | recall1 | carry3 | multi4 | hop | synthetic macro |
|---|---|---|---|---|---|
| intact (3 passes) | 0.996 | 0.889 | **0.547** | 0.154 | 0.649 |
| 1 pass | 0.996 | 0.881 | **0.457** (−0.090, ≈ 2.9 SE) | 0.141 | 0.644 |
| 2 passes · 2a / 2b | 0.996 | 0.893 / 0.891 | 0.537 / 0.537 | 0.146 / 0.141 | 0.651 / 0.651 |
| shared FFN off | 0.996 | 0.861 | **0.418** (−0.129) | 0.145 | 0.625 |
| ĉ_pred off | 0.996 | 0.873 | 0.496 (−0.051) | 0.141 | 0.626 |
| (t, Δ) table off | 0.996 | 0.895 | 0.553 | 0.158 | 0.641 |
| Engram read 2 off | 0.996 | 0.893 | 0.531 | 0.141 | 0.638 |
| Engram off | **0.492** | 0.070 | 0.000 | 0.000 | 0.279 |
| encoder F layers off | 0.934 | 0.240 | 0.000 | 0.037 | 0.232 |
| plan swapped | 0.006 | 0.000 | 0.000 | 0.000 | — |

The linear probe of the hop bridge entity (the intermediate node) reads 0.175 / 0.175 / 0.175 after passes 1 / 2 / 3. v1
had 0.126 / 0.126 / 0.117.

**Reading** (fixed-checkpoint switch-offs are out-of-distribution: a large cost proves use; a small cost only suggests
non-use, and a trained arm confirms):
1. The concept route carries long-range information, and its content is mainly the continuous state u.
2. **Language loss under-reports what the core does.** Every core part is ≤ 0.013 nats on language, yet three of them
   move multi4, the hardest probe that improves:
   - a second pass, worth 0.09; the third pass adds 0.01;
   - the shared FFN, worth 0.13;
   - ĉ_pred, worth 0.05.

   A next-token average over ordinary text is dominated by local prediction, and these parts work on the few positions
   that need multi-fact recall.
3. **Recurrence: the second pass does measurable work on multi4; the third does not.** 2a and 2b keep 0.537 of the 0.547.
   Passes still do not compose a hop: the bridge entity is equally decodable after every pass, and hop moves by ≤ 0.013.
4. The decoder barely uses ĉ_pred for language (0.013 nats in v3, 0.02 in v1), but it does use it for multi4.
5. **Engram is indispensable**, for language (+0.60 nats) and for retrieval: recall1 halves and multi4 goes to 0 without
   it. The encoder F layers matter almost as much for the probes.
6. The (t, Δ) table is the one v3 addition with no measurable use, on language or on any probe.

### 1.3b The codebook: saturated in use, and capped in fit by its segment shape

| | v1 (test, final) | v3 (val, 24 h → 33 h) |
|---|---|---|
| codes used, least-used segment | 100% | 100% |
| perplexity (of 128) | 108.5 | 117.7 |
| variance explained (VE) | **0.651** | **0.651 → 0.653** (flat since step ≈ 2,500: 0.646) |
| NCP cross-entropy vs code entropy (nats) | 2.70 vs 4.69 | 3.07 vs 4.77 |
| segments × codes × dims | 48 × 128 × 8 (336 bits) | 32 × 128 × 8 (224 bits) |

- **"Saturated" is right about use.** Every code is used, and perplexity is at 92% of the maximum.
- **Fit is a different matter.** The two runs have different widths and segment counts (336 vs 224 bits) and reached the
  **same VE, 0.651**. They share the per-segment shape: 8 dimensions quantised with 128 codes.
  - VE is set by bits per dimension within a segment, not by the number of segments or the total bits.
  - The rate-distortion bound for a roughly Gaussian segment is D/σ² ≈ 2^(−2·log₂N / ds).
  - At N 128 and ds 8 that is 2^(−1.75) ≈ 0.30, so VE ≤ ≈ 0.70. k-means reaches 0.65, close to the bound.
- **What changes VE** (§4.6 uses this):

  | change | bits per dimension | VE bound | likely VE | other effect |
  |---|---|---|---|---|
  | N 256, ds 8 | 1.0 | 0.75 | ≈ 0.70 | codebook-only parameters, excluded from the compute count |
  | N 128, ds 4 (S 64) | 1.75 | 0.91 | ≈ 0.85 | NCP must predict 64 segments |

- **Bigger is not automatically better for the RL door.** By §1.7 (1), an RL action should be *small*. A finer codebook
  is a better reconstruction target and a larger action space. §4.6 resolves this.

### 1.4 The pre-registered v4 rules (V3-016), applied to the 24 h readings

These rules were fixed on 2026-09-30 at 18:40, before the milestone.

| rule (V3-016) | 24 h reading | the rule says | owner's v4 decision |
|---|---|---|---|
| c_off < 0.02 → drop ĉ_pred from the plan | +0.013 | drop ĉ_pred | keep concept prediction; open the door to concept-level RL. The tension is stated in §1.7 (2), resolved as options in §5.4, and left to the owner in §10.6 |
| R1 − R3 < 0.02 → M = 1 | +0.005 | M = 1 | **M = 2** (owner: "I think recurrence will work") |
| shared_ffn_off < 0.02 → remove the shared FFN | +0.004 | remove | **keep it** ("small" allowed; §4.1 keeps the full 1,024 on the probe evidence), and add one to the encoder |
| table_off < 0.02 → remove the table, only together with M = 1 | +0.008 | remove only with M = 1 | open: decided in §4 |
| engram_read2_off < 0.02 → one Engram read | +0.028 (WikiText +0.042) | **keep read 2** | keep; extend Engram to the decoder |

**The V3-016 rules were written on language deltas only, and the probe table above changes two verdicts:**
- **M = 2 is supported by the evidence.** The 1-pass multi4 deficit of −0.090 is recovered by 2 passes (0.537), and a
  third pass adds 0.010. The owner's M = 2 keeps what recurrence demonstrably buys at two-thirds of v3's core compute.
  - *Qualifier.* These readings come from the M = 3-trained model. Its shortcut paths (1, 2a, 2b) were trained only
    through NCP and L_scons; the decoder never received a shortcut state in training (`micro_step` sends only `out_norm(S3)`
    to the decoder).
  - The 1-pass path was sampled with p = 0.5, so it is a trained path, which makes the −0.090 more credible.
  - Still, an M = 2-trained model is a different object, and the M = 2 arm confirms.
- **The shared FFN is used.** Removing it costs multi4 −0.129 and synthetic −0.024. Keeping it is supported; "small" is a
  sizing question for §4.
- **ĉ_pred:** language says drop it, multi4 says it carries 0.05. Keep it in the plan (§1.7 (2) adds the RL argument).
- **The (t, Δ) table** shows no use anywhere: drop it (§4).
- **Engram read 2** shows use: keep it.

Every owner decision still carries, in its section:
- the number that argues for and against it;
- a kill criterion, read at matched tokens against v3, or within v4 against its control arm.

### 1.5 Engineering facts carried into v4

- **Throughput.** Block-local window attention (+6.1%) and `torch.compile` of every layer (+54% total) give ≈ 3,100–3,400
  tok/s on the M2. The compiled `k_norm` gradient is wrong on MPS (8.5×), so `k_norm` runs eager (V3-006b).
- **Step time** (one micro-step, V3-006 profile, before compile): encoder 17.5%, core 15.2%, decoder 7.0%, backward 56.7%.
  Any looping of the encoder is priced against these shares (§3).
- **Memory.** The process footprint grows ≈ +0.9 GiB/h while MPS driver memory stays flat (V3-016). A planned re-exec at
  24 GiB handles it; v4 should find the leak.
- **Batch.** micro-batch 2 × 16 accumulation × 1,024 tokens. Micro-batch 4 thrashed swap (V3-006), so memory headroom does
  not convert into a larger micro-batch on this machine.
- **Data.** Documents and evaluation items are padded to multiples of 4 tokens (`data.py`, `pack_items`), and T = 1,024.
  Any chunk size that does not divide both needs a data rebuild (§3.5).

### 1.6 What [ETA] shows, and what maps onto this architecture

What the paper does:
- Autoregressive models trained only on next-action prediction come to represent the **hidden subgoal** linearly in the
  residual stream. Decodability rises with depth and peaks around mid-depth to the final layer.
- **Control is best at mid-depth.** A linear residual controller e ← e + U e inserted near the middle of the base model
  turns it into a goal-directed policy. Inserted earlier or later, it controls less well.
- **Metacontroller.** A GRU encoder, non-causal during training (it sees an embedding of the whole sequence), samples a
  code z̃_t ~ N(μ_t, Σ_t). A **switching gate β_t** integrates it as z_t = β_t z̃_t + (1 − β_t) z_{t−1}. A decoder maps z_t
  to the controller U_t. Training is next-action likelihood plus α · KL(N(μ, Σ) ‖ N(0, I)).
  - β becomes quasi-binary and switches exactly at subgoal boundaries, with no switching regulariser.
  - With **β forced to 1** (a new code every step) the internal-RL ablation fails.
- **The frozen base model matters.** Co-training the metacontroller with the base model collapses to one switch at the
  start of the sequence. The abstraction is discovered *in* a model pretrained by next-token prediction, then exploited.
- **Internal RL.** After the self-supervised stage, a causal policy π(z_t | e_{1:t}) replaces the non-causal encoder and is
  trained by RL in z-space, with a thresholded switch. This works because:
  - z is **low-dimensional** (n_z < n_e);
  - the horizon is **contracted** (one decision per abstract action, not per token).

What maps onto v3/v4:

| [ETA] | v3 / v4 | match |
|---|---|---|
| abstract goal decodable from the residual stream | u (the core state) carries recall / carry / multi4 (§1.3, u_off +0.50) | yes |
| controller acts at mid-depth | the plan enters the decoder at its input (D₀ = tok + P) | partial; the decoder is only 4 layers |
| discrete abstract action with a learned **variable** duration | fixed 4-token chunks: a code every 4 tokens | **no**: v3's chunking is the paper's ablated "fixed switching" case (§1.7 (3)) |
| low-dimensional action z | PQ code, 32 segments × 7 bits = 224 bits | **no**: far larger than a token (§1.7 (1)) |
| frozen base, then a metacontroller | joint training only | not yet: a post-training stage is the faithful use (§10) |
| prior N(0, I) with KL | NCP: a learned autoregressive prior over codes | analogous; NCP is the "prior over the next abstract action" |

### 1.7 Three corrections to the brief, stated before any design

1. **The concept action space is larger per step than the token space, not smaller.**
   - One v3 concept is a PQ code with 128³² = 2²²⁴ values. Four tokens are 8,192⁴ = 2⁵² strings.
   - So (N^S)^{T/4} = 2^{56·T} is much larger than V^T = 2^{13·T}. Raising N to 256 makes it 2^{64·T}.
   - [ETA]'s efficiency does not come from a smaller *set* of sequences. It comes from:
     - acting in a low-dimensional latent space that the pretrained model maps onto meaningful behaviour;
     - fewer decisions, so the credit-assignment horizon is T/k, not T.
   - The horizon part transfers: T/4 here. The set-size argument does not.
   - **Consequence (resolved in §4.6 by a 1,024-way coarse code):** concept-level RL must act on something small, not on
     the full PQ code. Candidates are one PQ segment,
     a learned low-dimensional z, or a residual offset on u. Concept-level RL is a post-training stage; v4 pretraining has
     to make it *possible*, not perform it.
2. **The decoder barely depends on the discrete code (ĉ_pred).**
   - c_off costs 0.013 nats in v3 and 0.02 in v1, and multi4 −0.05. The text is written from u, almost regardless of the
     code.
   - If the generated text does not depend on the code, choosing or reinforcing codes cannot steer it.
   - "Concept prediction exactly as in v3" and "open the door to concept-level RL" are therefore in conflict.
   - Making the decoder depend on a chosen code is possible. One route: train the decoder to read the code of the chunk it
     is writing, the VQ-VAE split between prior (NCP) and decoder. But it changes what next-token loss means (§5, §6).
   - This is an owner decision, listed in §10.6. The draft does not pick it silently.
3. **Fixed chunks are the case [ETA] ablated.**
   - The paper's abstract actions last a learned, variable number of steps (the switching gate β). With β forced to 1, a
     new action at every step, internal RL fails.
   - v3's concepts switch every 4 tokens, whatever the text does. Chunks of 6 or 8 are also fixed rates.
   - The paper's mechanism would be a **learned segment boundary** at concept level. That is a larger change, so it is the v5
     direction (§10.6).
   - This does not weaken the owner's compression goal; it is argued on its own grounds (§3.5). The content-aware pooler
     built for k = 6 is also the piece a learned boundary would reuse.

---

## 2. First principles for v4

### 2.1 Budget

- **Size.** Compute parameters < 20M at d 256, by the repository convention: encoder without embedding and Engram table,
  core without codebook, plus decoder. v3 is 18,986,871, so there is ≈ 1.0M of headroom.
  - Encoder looping costs no parameters.
  - Every new module has to be priced in §7 before it is accepted.
- **Chinchilla.** About 20 tokens per parameter gives ≈ 400M tokens for 20M parameters. At v3's ≈ 3,200 tok/s, 48 h is
  ≈ 550M token slots, about 1.4× Chinchilla. v4's estimate (§7.2) is ≈ 1.1–1.2× that speed, ≈ 600–660M.
  - v4 can afford *some* extra compute per token: looping, a second Engram path.
  - Every 10% of throughput lost costs ≈ 55M tokens, about 14% of the Chinchilla budget.
- **Wall clock.** The machine is the same M2 Air, 16 GB, fanless, so throughput claims are decided by interleaved A/B
  (`ab_interleave.py`), never by sequential runs (V3-006).

### 2.2 Principles (each traced to evidence)

1. **Keep what measurably carries load.** Encoder L/G streams with gated residual, the Engram table, the confined decoder,
   and u as the plan content (§1.3).
2. **Judge parts on the probes as well as on language.** Language deltas missed the second pass (multi4 +0.09) and the
   shared FFN (+0.13). v4's gates and switch-off reports always include multi4, carry3, hop and the synthetic macro beside
   next-token loss.
3. **Do not keep a mechanism that showed no use anywhere.** The (t, Δ) table: language +0.008, every probe within noise.
4. **Spend compute where switch-offs show use.**
   - Engram: +0.60 nats; recall1 halves without it.
   - Encoder F layers: +0.14 nats; multi4 goes to 0 without them.
   - These argue for more Engram access (the decoder) and for encoder capacity.
5. **Make the abstraction usable, not only present.** For the concept-RL door, the decoder must *depend* on the abstract
   action (§1.7 (2)), and the action space must be small (§1.7 (1)).
6. **One change, one switch-off.** Every v4 addition gets a zero-init or bypass path, so a fixed-checkpoint switch-off and a
   trained arm can price it, as v3 did.
7. **Same corpus and evaluation as v1 and v3.**
   - Comparisons are split-matched.
   - k = 6 re-packs the rows, so cross-version comparisons are at matched token count rather than step-indexed. The k = 4
     arm is the clean within-v4 bridge (§8.2).
   - The same holds for the k = 8 arm.

---

## 3. Encoder

The owner's view is that the v3 encoder is the best part of the design and should stay: 8 layers, L and G streams with gated
residual, and Engram read at more than one depth. The evidence agrees:
- **The encoder feeds the concepts.** Concepts are 82–86% G (router share). F layers off costs +0.14 nats and takes multi4
  to 0. Engram off costs +0.60 nats and halves recall1.

v4 therefore keeps the encoder intact and changes only three things, each with a bypass or a control arm so it can be
priced:
1. a second application of the F layers (layer-wise looping, §3.3);
2. a small shared FFN (§3.4);
3. 6 tokens per concept (8 as an arm) with an attention + FFN pooler instead of averaging, the owner's compression goal
   (§3.5).

The Engram rows are also handed on to the decoder (§3.2, §5).

### 3.1 Skeleton (unchanged from v3)

| piece | v4 | source |
|---|---|---|
| width | d 256, vocabulary 8,192, T 1,020 (k = 6, §3.5) | v3 (T changed) |
| layers | 8, pattern `SSSFSSSF` | v1 / v3 |
| S layer | window 8, 4 heads (2 KV), partial RoPE 16, q/k/v short convolutions, reads {L}, writes {L, G}, SwiGLU 1,024 | v3 |
| F layer | full causal attention, NoPE, per-head temperature, 6 heads (2 KV), reads {L, G}, writes {G}, SwiGLU 1,024 | v3 |
| residual grammar | gated-residual read (GRRead: per-stream sigmoid gate, rank 32) and write (GRWrite: 2σ(·) scale), zero-init → identity at step 0 | v1 |
| output | tok = zcRMSNorm(L) to the decoder; concepts from the CRC router mix of {e₀, L, G} | v1 |

**Why the stream split is the load-bearing idea.**
- L only ever sees a bounded window. Its reach is exactly 63 tokens, proven by unit test (`hrm_encoder.reach()`).
- The decoder reads L, never G, so the decoder alone cannot see past 63 + 7 = 70 tokens.
- Every long-range probe puts its evidence ≥ 128 tokens back. A correct answer therefore proves the concept route carried
  the fact, and the plan-swap control (recall1 0.996 → 0.006) confirms it.
- **Any v4 encoder change must keep reach(L) = 63.** This one rule decides §3.3.

### 3.2 Engram: one gather, read at encoder depths 0 and 4, rows handed to the decoder

- Unchanged in the encoder. Eight hash tables over 2-grams and 3-grams of canonical token ids, 208,333 rows × 48 dims each
  (80M stored, excluded from the compute count).
- **One gather per forward.** `e = table(eng_idx)` has shape (B, T, 384). It feeds:
  - read 1, after layer 0: gated, with a short convolution, writing {L, G};
  - read 2, after layer 4: own keys and values, zero-init values, reading L only.
- Read 2 is kept: +0.028 nats when off (WikiText +0.042).
- **New in v4:** the same `e` tensor is passed to the decoder, which adds its own reads (§5).
  - There is no second gather and no second table. The table is trained by all reads together, as read 1 and read 2
    already share it in v3.
  - Reach is safe. The hash keys are the last 2–3 tokens, which every decoder position already sees. The decoder's Engram
    adds *stored knowledge*, not reach.

### 3.3 Looping: second application of the F layers only (layer-wise, [Loopie]-style)

**Options, priced.**

Step-time shares are from the V3-006 profile. The encoder is 17.5% of the forward pass; with its backward in proportion,
that is ≈ 42% of a micro-step. The table prices the loop **alone, at v3's speed**. The whole v4 model is faster overall
(§7.2), because the core gets cheaper.

| option | what repeats | reach(L) | extra encoder compute | est. step time | est. tokens in 48 h |
|---|---|---|---|---|---|
| A · none (v3) | — | 63 | 0 | 1.00× | ≈ 550M |
| **B · F layers twice (layer-loop)** | layers 3 and 7, each applied twice in place: 0 1 2 **3 3** 4 5 6 **7 7** | **63** | ≈ +28–45%: an F layer has 1.12× an S layer's parameters, but runs full 1,024² attention (6 heads) where S runs 71 keys, so ≈ 1.4× on FLOPs | ≈ 1.13–1.20× | ≈ 460–490M |
| C · whole encoder twice (model-loop) | 0…7, 0…7 | **123**: breaks the probe floor | +100% | ≈ 1.42× | ≈ 390M |
| D · S layers twice (layer-loop) | each S layer twice | **123**: breaks the probe floor | ≈ +72% | ≈ 1.30× | ≈ 420M |

**Why only B.**
1. **Reach.**
   - S layers write L and each adds 10 tokens of reach (window 7 + convolution 3). Applying any S layer twice extends
     reach(L), and with it what the decoder sees without the core.
   - C and D raise reach(L) to 123 and the decoder's reach to 130, past the 128-token probe floor. That would silently
     void the long-range probe evidence.
   - F layers write only G, which the decoder never reads, so looping them leaves reach(L) at exactly 63.
2. **Function.**
   - The F layers build G, and concepts are 82–86% G. Removing them takes multi4 from 0.547 to 0 and carry3 from 0.889 to
     0.24.
   - A second refinement of G is aimed at what the concept module consumes.
   - *Analogy, not support:* [ETA] finds abstract goals decodable from mid-depth on, and residual *control* inserted at
     mid-depth works best. That is about inserting a controller, not repeating a layer, so it motivates the location only
     loosely.
3. **Cost.**
   - B adds no parameters. It costs ≈ 13–20% of step time, about 60–90M tokens over 48 h (15–22% of the Chinchilla budget,
     §2.1). The A/B measures the real price.
   - C costs more than twice that, and violates the reach rule.

**What [Loopie] does and does not tell us.**
- Layer-loop (each layer twice in place) beat model-loop in their runs only after ≈ 1.2T tokens. Before that, model-loop
  was slightly ahead.
- Their compute-matched gain came from activation memory: half the stored layers allowed a doubled micro-batch. That lever
  does not exist here: micro-batch 4 thrashed swap (V3-006).
- So at ≈ 0.5B tokens, [Loopie] gives the *pattern* (local, adjacent reuse, R = 2, the largest marginal gain at R = 2) but
  not a promise.
- Here B is chosen for reach and function. Its value is decided by measurement.

**Specification of B.**
- F layer ℓ ∈ {3, 7} is applied twice with the same weights:
  - (L, G) ← F_ℓ(L, G) gives G′;
  - then (L, G′) ← F_ℓ(L, G′).
- The gated-residual writes make the second application start as a refinement: the write gate scales the update.
- **Bypass:** `enc_loop = 1` gives the v4 encoder without the loop, so the fixed-checkpoint switch-off "F loop off" is
  free.
- No per-iteration parameters. The (t, Δ) table in v3's core, which played that role, showed no use (§1.3).

**Decision rule** (pre-registered, in the order run). A fixed-checkpoint "F loop off" switch-off on the main run can show
that the second application is *used*. Only a trained no-loop arm can show that training with it was *worth it* at matched
tokens and matched clock. The rule below therefore needs that arm (§8.3).

1. **Throughput.** Interleaved A/B (`ab_interleave.py`) of A vs B on the full v4 model gives the measured price; §9
   records it. No threshold is fixed in advance from an unmeasured estimate.
2. **Quality.**
   - **When to read it.** In v3, carry3 and multi4 were still 0.000 at step 2,376 and started around steps 2,559 and
     3,669, so early probe gates cannot fire.
   - **Early reading (≤ 2,376):** recall1 onset step, and val.
   - **Deciding reading:** the B arm and the A arm are both run to **step 4,050** (≈ 12 h of v3 time), where v3 had carry3
     0.617 and multi4 0.086.
   - B is kept only if:
     - at **matched tokens** (step 4,050) it beats A on at least one of:
       - val by ≥ 0.02 nats;
       - carry3 by ≥ 2 SE (≈ +0.06);
       - recall1 onset ≥ 300 steps earlier;
     - **and** at **matched wall-clock** (A's clock at its step 4,050) B is not worse on val or carry3. B costs tokens per
       hour, so winning per token while losing per hour buys nothing.
   - Otherwise B is dropped and its cost goes back to tokens.

### 3.4 Shared FFN in the encoder (owner)

- **Evidence.** The core's shared FFN costs only 0.004 nats when removed, yet multi4 falls 0.547 → 0.418 without it
  (§1.3). A central, layer-shared memory is used for multi-fact work.
- **Design.**
  - One SwiGLU (d 256 → f → 256), shared by all 8 encoder layers, as a parallel branch beside each layer's own FFN. It
    reads the same gated input x and writes through the same write gate: y = FFN_ℓ(x) + g_ℓ · FFN_shared(x).
  - g_ℓ is a learned scalar gain per layer, **initialised to 1**. The shared W₂ is zero-init, so the encoder is exactly
    v3's at step 0, and W₂ receives gradient from step 1.
  - Both factors at zero (g = 0 and W₂ = 0) would leave the branch dead forever: each factor's gradient is multiplied by
    the other.
  - In S layers it writes {L, G}, in F layers {G}, following each layer's write rule. It is position-wise, so reach is
    unchanged.
- **Size.** f = 384, so 3 · 256 · 384 = 294,912 parameters. That is 37.5% of the core's 786k shared FFN, sized to fit the
  ≈ 1.0M headroom (§7) together with the decoder's Engram reads.
- **Why not one FFN shared between the encoder and the core?** They run at different rates (tokens vs concepts) on
  different input distributions (the CRC mix vs the core state). One memory would serve two masters. Keep them separate.
- **Kill criterion.** At 24 h, if the encoder `shared_ffn_off` costs < 0.01 nats *and* moves no probe by ≥ 2 SE, remove it
  in v5.

### 3.5 Concept compression: k tokens → 1 concept with an attention + FFN pooler (owner's goal)

**The owner's hypothesis.**
- v3 takes the encoder's output, cuts it into chunks of 4 tokens, and pools each chunk into one concept.
- The concept module runs on these T/4 concepts and predicts the next concept (NCP).
- The decoder then shifts the prediction one chunk forward and repeats it over the k tokens it is writing (§5.3).
- The proposal:
  1. take **6 or 8 tokens per concept** instead of 4;
  2. replace averaging by **attention over the chunk's tokens** (or, more radically, an FFN, or attention followed by an
     FFN) that turns them into one concept;
  3. the concept module then runs on T/6 or T/8 concepts.
- Through training, the pooler learns what a concept does not need: stop words and other locally recoverable tokens.
  Those reach the decoder through its other channels:
  - the token stream `tok`, with reach 63 + 7 = 70 tokens;
  - the Engram rows (§5.2).
- The goal is **compression 4 → 6 (or 8) by design**: the pooler, not a fixed rule, decides what a concept keeps.

This section takes the hypothesis step by step:
- what v3 actually does (§3.5.1);
- why a content-aware pooler is the missing piece (§3.5.2);
- k = 6 vs k = 8 (§3.5.3);
- the pooler design (§3.5.4);
- the gates (§3.5.5).

#### 3.5.1 What v3's pooling is, and what it became

- **Code** (`hrm_encoder.py`, `pool = nn.Linear(4·256, 256)`):
  - per token, the router mixes three views: m_t = mix₀ e₀ + mix₁ rms L + mix₂ rms G;
  - the 4 mixed tokens of a chunk are concatenated: c = W_p [m₁; m₂; m₃; m₄];
  - c is projected onto the sphere of radius √256.
- **At initialisation it is exactly averaging.** W_p = [I/4, I/4, I/4, I/4] (`cat([P]*k)/k` with P = I, because the
  concept width equals d).
- **But W_p is trained, and it does not stay an average.** In the step-11,348 checkpoint (read-only), per chunk slot:

  | slot in chunk | 1 | 2 | 3 | 4 (newest) |
  |---|---|---|---|---|
  | ‖W_i‖_F (init 4.0) | 7.69 | 7.64 | 8.34 | **12.08** |
  | diagonal mean (init 0.25) | 0.248 | 0.285 | 0.330 | **0.431** |
  | cosine with slot 4 | 0.30 | 0.49 | 0.73 | 1 |

- **What that means.**
  - v3's pool learned a **recency-weighted, per-slot** map: the newest token of the chunk counts about 1.6× the oldest.
  - The **same weights apply to every chunk, whatever words are in it.** Slot 2 is weighted identically whether it
    holds "the" or "Paris".
  - The router mix does depend on content, but it only chooses *which view* of a token to use (e₀, L or G): a softmax over 3
    views per token, so it can never remove a token.
  - **v3 has no way to give a stop word less weight than a name.** That is the mechanism the owner's hypothesis
    targets.

#### 3.5.2 Why a content-aware pooler makes larger chunks plausible (first principles)

1. **A concept is not a summary of its tokens. It is a message to the future.**
   - The decoder already sees the last 70 tokens directly (`tok`), plus the Engram rows. Anything recoverable from them
     need not be in a concept: most function words, local syntax, the second half of a word whose first half is visible.
   - What *must* be in a concept is what the decoder cannot see: content more than 70 tokens back.
   - Example: in the probes the value tokens sit ≥ 128 tokens before the answer, so they can only arrive through
     concepts.
   - So the pooler must learn to keep values **where they are facts that will be needed later**, and to drop them where
     they are locally recoverable. That decision depends on content, not on word class.
2. **v3's concepts already select, despite the fixed pool.**
   - A linear read-out of a concept's own 4 tokens is only 54–70% (G-E1, 24 h evaluation).
   - Yet at the same checkpoint (step 8,133, 24 h evaluation) recall1 is 0.996, carry3 0.889 and multi4 0.547 through
     those concepts.
   - The selection happens upstream, in G and the router. The pool itself cannot select by content.
3. **The capacity argument for larger k.**
   - With averaging, every extra token in the chunk dilutes every other token: the k-mean of k vectors.
   - With content-weighted pooling, one important token can take most of a head's weight, whatever k is.
   - A larger chunk then costs capacity only when it holds *more facts that matter*, not more tokens.
4. **What an FFN adds after attention.**
   - Attention gives a convex combination of the chunk's tokens. It can select, but it cannot compose: "not" + "good",
     or "x" + "=" + "5", need a nonlinearity on the selected parts.
   - A small FFN on the pooled vector, after a multi-head pool, composes across the heads' selections.
   - Each head has selected a different token or part, so the FFN sees up to 4 selections side by side. Attention
     selects; the FFN composes.

#### 3.5.3 k = 6 or k = 8: measured, not assumed

The probe and padding numbers below are measured:
- probe items: `run_m2_25M/data/eval/val.pt`, `lr_*` families, 512 items each;
- padding and null-plan share: 3,000 random training rows (33,957 documents; median 38 tokens, mean 85.2).

| | k = 4 (v3) | **k = 6** | k = 8 |
|---|---|---|---|
| concepts per row (T) | 256 (1,024) | 170 (**1,020**) | 128 (**1,024**) |
| concepts in a median 38-token document | 9 | 6 | 4 |
| decisions per row, the [ETA] horizon | T/4 | T/6 | T/8 |
| real tokens after document padding to k | 98.1% | 96.9% | 96.1% |
| tokens predicted with the null plan (the first k − 1 of each document) | 3.5% | 5.9% | 8.2% |
| mean plan staleness (tokens the plan does not yet include) | 1.5 | 2.5 | 3.5 |
| recall1 / carry3 / multi4: the query name (in "What is the code of X?") inside the plan | **100%**, min margin 4 | **100%**, min margin 2 | **100%**, min margin 0 |
| hop: start name inside the plan / hop count inside the plan | 100% / 100% | 100% / 100% | 100% / **75%** |
| bind: the added constant c inside the plan | 50% | 35% | 22% |
| content per concept (256 dims, same codebook) | 1× | 1.5× | 2× |
| core cost vs v3's core: attention / position-wise (§4.8) | 1 | 0.21 / 0.31 | 0.12 / 0.23 |
| micro-step vs v3's profile (§7.2) | 0.946 (v3); ≈ 0.87–0.90 for the v4 k = 4 control | 0.79–0.85 | 0.76–0.82 |
| block-local attention | 64 | 60 | 64 |

**Reading the table.**
- **Both k = 6 and k = 8 keep the main probes valid.**
  - The question's first mention of the name is always inside a complete chunk the plan includes.
  - The second mention ("The code of X is"), right before the answer, is inside the plan for only 53% / 31% / 26% of
    items at k = 4 / 6 / 8. The core can rely on the first mention at every k.
- **At k = 8, a quarter of hop items have the hop count outside the plan.**
  - For those, the core must retrieve the chain without knowing the depth.
  - It can still answer by carrying the chain and letting the decoder pick the depth from `tok`. So this is a
    harder task, not a hard 0.75 ceiling.
  - Reported with this caveat at k = 8, as bind is at every k.
- **bind does not test the core at any k.** The constant sits after the last complete chunk in half the items even at
  k = 4. This fits its flat 0.05–0.09 in v1 and v3. It stays for comparability, with the caveat (§9.3).
- **Speed: a correction to the expectation that T/8 makes the core "much more efficient".**
  - After M = 2 passes and no sampled shortcut (§4), the core is only about 8–12% of a v4 step at k = 6.
  - Going to k = 8 saves another ≈ 3–4% of the step.
  - So **k = 8's case is the longer horizon and the compression itself, not speed.** Most of the speed comes from k = 6
    plus M = 2.
- **k = 8's costs are content and the document start.**
  - 2× the tokens into the same 256 dimensions and the same codebook. §1.3b showed the per-segment fit is already
    capped.
  - 8.2% of tokens predicted with no plan: a median document has only 4 concepts.
- **k = 8's engineering advantage.** T stays 1,024 and the block 64. Documents and probe items still need re-padding to
  multiples of 8.

**Recommendation: k = 6 is the v4 default, k = 8 is a priced arm, k = 4 is the control.** All three use the same
pooler (§3.5.4), so the comparison isolates the compression factor.
- **k = 6** reaches the owner's goal with a 2-token query margin, 5.9% null-plan tokens and full hop validity.
- **k = 8** is the bolder step. It is tested as an arm against the same gate (§3.5.5).
- If k = 8 passes, it replaces k = 6 for v5.

#### 3.5.4 The pooler: attention selects, a small FFN composes

**Three candidates compared.**

| candidate | what it is | parameters (k = 6) | verdict |
|---|---|---|---|
| v3 linear (concat) | c = W_p [m₁; …; m_k] | 393,216 | per-slot weights: the limitation of §3.5.1 |
| **concat-FFN** (the owner's radical option) | c = W₂ φ(W₁ [m₁; …; m_k]), 1,536 → 256 → 256 | 458,752 (affordable) | rejected on sharing, not cost |
| attention pool only | c = Σ α_i v_i, content-weighted | 198,424 | selects but cannot compose |
| **attention pool + FFN (v4)** | pooling by multi-head attention, then a SwiGLU on the pooled vector | **493,337** | selects and composes, with shared weights |

**Why not concat-FFN.**
- It costs 458,752 parameters at k = 6, which fits the budget.
- The reason against it is **parameter sharing**:
  - A concat-MLP has separate weights for every slot. What it learns about "the" in slot 2 does not transfer to "the" in
    slot 5. It has k× the weights and k× less data per slot.
  - It must also mix slots in its 256-wide hidden layer before it can gate any of them.
- In the attention pool, one W_K and one W_V serve every slot. A token is judged by what it is, wherever it sits, and
  position enters only through p_r.

**The v4 pooler.** One block per chunk.
- **Cost.** W_K and W_V are per-token projections over all T positions: ≈ 131k MACs per token, about 13% of one encoder
  layer. The softmax-pool, W_O and the FFN run at concept rate (170 positions per row).
- **Total.** < 1% of the step (§7.2).
- **Inputs** for token i of the chunk:
  - m_i: the router mix, exactly as v3;
  - r(i) ∈ {0 … k−1}: its position in the chunk.
- **Attention.** H = 4 heads, head width 64.
  - Key: k_i = W_K (rms m_i + p_{r(i)}). Value: v_i = W_V (m_i + p_{r(i)}). W_K, W_V ∈ ℝ^{256×256}.
  - Score for head h: s_{h,i} = q_h · k_i^{(h)} / 8 + b_{h,r(i)}. q_h ∈ ℝ⁶⁴ is a learned query; b is a learned
    position bias.
  - Weights: α_{h,·} = softmax over the chunk's **real** tokens. Pads are masked.
  - Pooled: c₀ = W_O [Σ_i α_{1,i} v_i^{(1)} ; … ; Σ_i α_{4,i} v_i^{(4)}], with W_O ∈ ℝ^{256×256}.
- **FFN** (pre-norm, residual, gated):
  - c₁ = c₀ + g · W₂ (SiLU(W₁ rms c₀) ⊙ W₃ rms c₀);
  - hidden 384, the same width as the encoder's shared FFN (§3.4); g is a scalar gain.
- **Sphere.** ĉ = √256 · c₁ / ‖c₁‖, as v3.
- **Why the parts.**
  - Multiple heads: one head can follow the entity, one the most recent token, one the predicate; the FFN composes them.
  - p_r in the values: without it Σ α v is order-blind ("dog bites man" = "man bites dog").
- **A trade, stated.** Earlier drafts let every head pool the full 256-dim value (W_O 256 × 1,024). The standard
  multi-head form gives each head a 64-dim slice after W_V. That saves 131,072 parameters, which pay for the FFN. In
  exchange, W_V must learn to route what each head needs into its slice.

**Initialisation: exactly v3's starting point (mean pooling), with every branch alive.**

| part | init | consequence |
|---|---|---|
| q_h, b, p_r | 0 | every α is uniform over the real tokens |
| W_V, W_O | I | c₀ = mean of the chunk's real m_i: v3's initial pooling, pads excluded |
| W_K | random, std 0.02 | q receives gradient through it; W_K receives gradient once q moves |
| FFN W₂ | 0; g = 1; W₁, W₃ random | FFN branch outputs 0, so c₁ = c₀; ∂W₂ = g · h ≠ 0, so the branch is alive |

p_r receives gradient through the values from step 0.

**Empty chunk.** A chunk with no real tokens (only pads) would give a softmax over nothing (NaN).
- Rule: the mask falls back to uniform over all k slots for that chunk.
- `nonempty` then removes it downstream, as in v3.
- Unit-tested (§3.8).

**Parameters.**

| part | k = 6 | k = 8 |
|---|---|---|
| W_K, W_V, W_O | 196,608 | 196,608 |
| q (4 × 64) | 256 | 256 |
| p_r (k × 256) | 1,536 | 2,048 |
| b (4 × k) | 24 | 32 |
| FFN (3 × 256 × 384) + g | 294,913 | 294,913 |
| **pooler total** | **493,337** | **493,857** |
| vs v3's linear pool (262,144) | **+231,193** | +231,713 |

At k = 8 the decoder's offsets also grow (o: 8 × 256, +512 over k = 6). So k = 8 costs 1,032 more parameters than k = 6
in total. Both stay < 20M (§7.1).

**Telemetry.** Per head, logged at each evaluation:
- the attention entropy;
- the weight on each token-frequency band (the 100 most frequent tokens vs the rest).

This is how the owner's claim is seen in training: the weight on the most frequent tokens (the stop words) should fall
below 1/k on the heads that carry facts.

#### 3.5.5 What decides it (pre-registered)

- **Arms.** The k = 4 control and the k = 8 arm use the same pooler as the default. Each runs to step 4,050 (§8.3).
- **k = 6 is kept** (vs k = 4) if all hold:
  - val is within **+0.03 nats**: the price of compression, which the cheaper core must pay for;
  - recall1 onset is no later than **+300 steps**;
  - carry3 is within **2 SE**;
  - tok/s is **≥ 1.03×**. The expected ratio is only ≈ 1.06–1.10×, because the k = 4 control already has M = 2 and no
    shortcut: its core multiplier is 2/5, so its micro-step is ≈ 0.87–0.90 against k = 6's 0.79–0.85. A 1.05× threshold
    would sit inside A/B noise.
- **k = 8 is promoted** (to v5's default) if it passes the same four thresholds against k = 4 (expected tok/s ≈ 1.10–1.16×),
  and is no worse than k = 6 on val by more than 0.02 nats.
  - Hop is read with the 75% caveat.
  - Coarse-code perplexity must stay > 25% of K, so 2× content per concept has not collapsed the code.
- **If k = 6 fails on quality but passes on speed:** k = 4 with the new pooler. This is still an improvement over v3's
  per-slot pool.
- **The pooler itself** is not gated against v3's linear pool. The k = 4 control already uses it, and v3's pool is known
  (§3.5.1).
  - Arm 5 (§8.3) may add k = 6 with attention only, no FFN, if the FFN's write share stays below 5%.

**Probe packing.**
- Probe items are re-packed with k-token padding (`pack_items(k)`), and the language rows at T = 1,020 (k = 6) or 1,024
  (k = 8).
- Every probe fact ("The code of X is a b c.") spans 8–12 tokens, so at k = 6 a fact covers 2–3 concepts.
- Evidence distances (≥ 128 tokens) are unchanged in tokens. They shrink in concepts (≥ 21 at k = 6, ≥ 16 at k = 8),
  which is easier for the core's attention.

**[ETA] and fixed vs learned length.**
- k = 6 or 8 is still a fixed switching rate.
- The pooler is the piece that can grow into [ETA]'s learned boundary: the same block pooling variable-length spans
  instead of fixed chunks. That is the v5 direction (§10.6).

### 3.6 Encoder summary

| | v3 | v4 |
|---|---|---|
| layers, width, heads, FFN | 8 · 256 · S4/F6 · 1,024 | same |
| F-layer applications | 1 | **2** (layer-loop, layers 3 and 7) |
| shared FFN | — | **SwiGLU 384, zero-init W₂, per-layer gain** |
| Engram | one gather → read 1 (depth 0), read 2 (depth 4) | same, **rows handed to the decoder** |
| concept compression | 4 tokens, learned linear pool over the concatenated chunk (mean init) | **6 tokens, attention + FFN pooler** (mean-pooling init), T = 1,020; k = 8 arm at T = 1,024 |
| reach(L) | 63 | **63** (unit test) |
| compute parameters | 9,641,007 | 9,641,007 + 294,912 (shared FFN) + 8 (gains) + 231,193 (pooler over v3's pool) = **10,167,120** (§7 instantiates) |
| step-time share | 1.00× | ≈ 1.13–1.20× from looping (A/B decides) |

### 3.7 Interface handed to the concept module and the decoder

- **To the core:**
  - ĉ, shape (B, M = T/k, 256), on the sphere of radius √256 (k from §3.5);
  - `cdoc = doc[:, ::k]`;
  - the three chunk masks (`nonempty`, `complete`, `pair_ok`), defined as in v3 with k tokens per chunk.
- **To the decoder:**
  - `tok` = zcRMSNorm(L), shape (B, T, 256), reach 63;
  - the gathered Engram rows `e`, shape (B, T, 384). These are a function of canonical token ids inside each position's
    n-gram window only, never of L or G (§5).

### 3.8 Encoder acceptance tests (inherited by §9)

1. **reach(L) = 63 with `enc_loop = 2`.** A perturbation at position p changes L at exactly p … p + 63 (the v1 unit test,
   re-run).
2. **Bit-identity at step 0.** With the shared W₂ = 0 and every per-layer gain g_ℓ = 1, the v4 encoder with
   `enc_loop = 1` equals v3's encoder up to the pooling (float64, small config). The pooler at init (W_V = W_O = I, q = b =
   p = 0, FFN W₂ = 0) equals the masked mean of the chunk's real m_i (v3's pooled quantity), before the sphere projection.
   A second check: after one optimizer step the gradients of the shared W₂, the pooler's FFN W₂, q and p_r are all
   nonzero (every branch is alive).
3. **Loop bypass.** `enc_loop = 1` on a v4 checkpoint runs and gives the "F loop off" switch-off.
4. **Document isolation** is unchanged (v1 test).
5. **Engram hand-off.** The decoder receives the same `e` tensor (data pointer equality); there is no second gather.
6. **Chunk data, per k.** k = 6: T = 1,020. k = 8 arm: T = 1,024. Every document and probe item starts on a k-token
   boundary. The first concept of each document contains only that document's tokens. The `nonempty`, `complete` and
   `pair_ok` masks are recomputed for k.
7. **Pool masking.** Changing a pad token's embedding changes no `nonempty` concept. Empty chunks are covered by test 8.
8. **Empty chunk.** A chunk of pads only gives finite values (uniform fallback), is `nonempty = 0`, and contributes no loss.
9. **Content-weighting is possible.** On a toy batch, the pooler can be trained to put > 0.9 of one head's weight on a
   marked token at any slot. v3's linear pool cannot, which is the deliberate-defect mutant.

## 4. Concept module

**Inherited.** The concept module stays v3's in anatomy:
- distinct F-anatomy layers over {C, S} at concept rate;
- full causal attention within a document, 6 heads;
- gated-residual reads and writes;
- S₀ = 0;
- a ZC-RMSNorm readout u;
- a distance head over a product-quantised vocabulary trained by VQ.

**What changes, and why:**
1. M = 2 passes (owner; supported by §1.4).
2. The (t, Δ) table is dropped (no use anywhere, §1.3).
3. The sampled shortcut trajectory is removed. Supervision is terminal-only, as draft_v3 §4.2 (4) intended (§4.2).
4. The codebook grows and gains a **coarse level**: a small, predictable abstraction label, and the action itself only
   under option B (§1.7 (1), §4.6, §5.4).
5. *Arm, not default:* [ETA]'s temporal integration of the abstract action (§4.7).

### 4.1 Layers: 4 distinct layers, own FFN 736, shared FFN 1,024 (kept full size)

**Budget after §3.** Encoder 10,167,120 + core 4,881,216 − table 43,008 + decoder 4,464,648 = 19,469,976. The decoder's
two Engram reads and the k = 6 offsets (≈ 428k, §5) leave ≈ 102k.

| option | layers × passes = effective depth | own FFN | shared FFN | core compute params | total compute | fits < 20M? |
|---|---|---|---|---|---|---|
| **A · 4 × 2 (default)** | 4 × 2 = 8 | 736 | **1,024** | 4,838,208 | ≈ 19.90M | yes, ≈ 102k spare |
| B · 5 × 2 | 5 × 2 = 10 | 640 | 512 | 5,072,840 | ≈ 20.13M | **no** with the pooler's FFN; ≈ 19.84M (≈ 162k spare) if the pooler drops its FFN |
| C · 5 × 2, own 736 | 5 × 2 = 10 | 736 | 1,024 | 5,834,696 | ≈ 20.89M | **no** |

**Why A.**
- **The shared FFN is the core part with the largest measured probe effect.** Removing it costs multi4 −0.129 (§1.3).
  - Option B buys a 5th layer by halving the shared FFN and trimming every own FFN.
  - That trades a measured contributor for untested depth.
- **Depth evidence is against more depth for its own sake.** The third pass added multi4 +0.010; the second added +0.090.
  The value is in the *second visit*, not in more layers per visit.
- **Compute.** At k = 6 and M = 2, with the sampled shortcut gone (§4.4), option A costs ≈ 21% of v3's core attention and
  ≈ 31% of its position-wise work. That saving pays for the encoder loop.
- The owner allowed "4 or 5"; A is the evidence-based choice, and B is the listed alternative if the shared FFN's share
  turns out to be small. The 24 h report logs the shared FFN's write share, which v3 did not.

### 4.2 Recurrence: M = 2, plain passes, terminal-only supervision

- **Trajectory.** S₀ = 0, S₁ = Φ(C, S₀), S₂ = Φ(C, S₁), then u = zcRMSNorm(S₂).
- **No conditioning on the pass index.** The (t, Δ) table is dropped: +0.008 nats, every probe within noise (§1.3).
- **Why plain passes can still differ.** The second pass reads S₁, which already holds what pass 1 retrieved for every
  concept. So its attention keys and values are built from a richer state.
  - That is exactly the mechanism a two-step retrieval needs: find the fact, then use it as the next query.
  - The multi4 gain of the second pass (+0.09) is direct evidence that it happens.
- **Terminal-only supervision (default); no shortcut term.**
  - At M = 3, v3 sampled one shortcut trajectory per micro-batch (1 / 2a / 2b). These were **separate trajectories**
    (different table rows, batched rows), not intermediate states of the main path. Each was supervised at its own
    endpoint (NCP + L_scons).
  - At M = 2 without the table, the only 1-pass state is S₁ of the main path. Supervising it would be *intermediate*
    supervision of the main trajectory, which draft_v3 §4.2 (4) chose not to do: "each trajectory is supervised at its
    endpoint only; intermediate states are never pulled toward the final answer".
  - v2's per-pass NCP being best at R2 was ambiguous evidence. It fits intermediate targets harming late states, and also
    late passes trading NCP for NTP.
  - **There is also a mechanical reason here.** L_scons(rms S₁, sg rms S₂) sends gradient only into S₁ and pulls it
    toward S₂. Its minimum is S₂ ≈ S₁, which is a second pass doing nothing. That would erase the one thing recurrence
    demonstrably buys (multi4 +0.09).
  - **v4 default:** NTP and NCP on u = zcRMSNorm(S₂) only. Full backprop through both passes.
- **Arm (not default):** λ_s = 0.05 · NCP(zcRMSNorm(S₁)), without L_scons.
  - It buys a usable early exit, the property that made v3's 1-pass path reach multi4 0.457, at no extra forward.
  - **Kill criterion** at step 4,050: pass 2's multi4 advantage over pass 1 (R2 − R1) must stay ≥ 2 SE. If S₂ ≈ S₁, the
    supervision has collapsed the second pass.
  - The v3 gradient audit found the shortcut gradients aligned with the main loss (cosine 0.07–0.61, never negative), but
    those were separate trajectories, so the audit does not cover this case.
- **The hop problem is not solved by M = 2, and the draft says so.**
  - hop is ≈ 0.5 per link with the bridge entity equally decodable after every pass (§1.2–§1.3).
  - V3-016's "pointer-following pass" asks pass 2's query to be built from what pass 1 retrieved. In this anatomy that is
    already true structurally (pass 2 reads S₁).
  - What failed is more likely the *retrieval* of the bridge at pass 1: 0.175 decodable, flat.
  - v4 does not add a mechanism for hop. It reports hop by depth and the bridge-by-pass probe at every milestone. A
    targeted mechanism waits until M = 2's own bridge-by-pass reading exists.

### 4.3 Shared FFN

Kept, parallel to each layer's own FFN, zero-init W₂, as in v3: y = FFN_own(x) + FFN_shared(x). Full size 1,024 (§4.1). The
24 h report adds its write share (‖FFN_shared(x)‖ / ‖y‖ per layer), which v3 did not log.

### 4.4 What is removed from v3's core

| removed | v3 evidence | saving |
|---|---|---|
| (t, Δ) table | +0.008 nats; probes within noise | 43,008 parameters; 7 gain multiplies per layer per pass |
| third pass | multi4 +0.010; language +0.001 | ⅓ of core compute |
| sampled shortcut trajectory (1 / 2a / 2b, batched rows) and its loss terms | 2a ≈ 2b ≈ 3 passes; terminal-only at M = 2 (§4.2) | 0.5 · 1 + 0.25 · 1 + 0.25 · 2 = 1.25 extra pass-rows per 3 main passes: **≈ +42% core compute on average** |

### 4.5 Readout and plan content

- u = zcRMSNorm(S₂) is the plan's main content. u_off costs +0.50 nats; it is what the decoder writes from (§1.3).
- ĉ_pred stays in the plan (multi4 −0.05 without it), with its role extended by §4.6 and decided in §5.

### 4.6 The codebook: what is the action? (two levels: a coarse code and the PQ codes)

**What the codebook is for.** It plays three roles, and they want different sizes:

| role | wants |
|---|---|
| NCP target (the prior over the next abstract action) | enough detail to make prediction informative |
| reconstruction of the concept (VE) | many bits per dimension (§1.3b) |
| **the action for concept-level RL** | **few choices per decision** (§1.7 (1)) |

A single PQ codebook cannot serve all three. 32 segments × 256 codes is 256 bits per concept, more than the 78 bits of the
6 tokens it summarises (8,192⁶ = 2⁷⁸).

**v4: two levels (residual quantisation).**
1. **Coarse code**, the abstraction label (the action itself only under option B, see below).
   - One code per concept from K = 1,024 vectors in the full 256 dimensions (10 bits).
   - Nearest neighbour to ĉ: k_j = argmin_k ‖ĉ_j − E⁰_k‖.
   - Trained by VQ (codebook-only gradient, k-means++ init, dead-code restarts), as v3.
2. **Fine codes**, the detail: PQ over the residual ĉ_j − E⁰_{k_j}, with **32 segments × 256 codes × 8 dims** (the owner's
   larger codebook).
   - Expected VE ≈ 0.70. This comes from the fine level's 1 bit per dimension (bound ≈ 0.75, §1.3b).
   - **The coarse level adds almost nothing to VE.** 10 bits over 256 dimensions is 0.04 bits per dimension, ≈ 5% of the
     variance at most. In 256 dimensions, 1,024 centroids are nearly equidistant from most points.
   - The coarse code is a **cluster / topic label**: the abstract action and an NCP target, not a reconstruction step.
   - **Two 24 h measurements decide whether it is an action at all:**
     - **Coarse-code perplexity:** is it used?
     - **Coarse NCP cross-entropy vs ln 1,024 = 6.93:** is it predictable? A near-uniform coarse prediction would be no
       action prior.

**When the coarse code is the action, and when it is not.**
- Under the recommended default (A pretraining + C post-training, §5.4), the decoder barely depends on codes (c_off
  0.013). The coarse code is then **the NCP target and the measurement of abstraction** (perplexity, CE vs 6.93).
- The action under C is the post-training controller's low-dimensional z at concept rate. The T/6 horizon holds; the
  K-sized set is replaced by the z-space.
- The coarse code becomes the action itself only under option B. This dependency is the owner decision in §10.6.

**Why the coarse code is still the intelligent version of the owner's sample-efficiency argument.**
- A policy choosing coarse codes makes one 10-bit decision per 6 tokens: K^{T/6} = 2^{1.7 T} sequences, against V^T = 2^{13 T}
  for tokens.
- Now the inequality the brief wanted is true, and the horizon is contracted 6×. In the k = 8 arm it is K^{T/8} = 2^{1.25 T}
  and 8×.
- The fine codes are the "how": produced by the model given the coarse choice, as [ETA]'s decoder produces a controller
  from a code.

**NCP over both levels.**
- **Head.** The same distance head: q = W_q u (v3).
  - The coarse logits are −‖q − E⁰_k‖² / τ₀ over K = 1,024.
  - The fine logits are the v3 PQ distance head over 32 × 256, on the query's residual. In the loss the residual is taken
    against the true next coarse centroid (§6.2).
- **Loss.** NCP = CE(coarse) + mean over segments of CE(fine), the v3 form extended by one term. Targets are the *next*
  concept's codes on `pair_ok` pairs.
- **ĉ_pred** is the full predicted next concept, which the plan reads. It uses no future information; its exact frame is
  pinned in §6.2.

**Parameters.**
- Codebooks: E⁰ 1,024 × 256 = 262,144; PQ 32 × 256 × 8 = 65,536. Both are excluded from the compute count, as v3's codebook
  was.
- Head: one more temperature (τ₀). The coarse level adds no compute parameters.

**Arm.** Fine PQ at 64 segments × 128 codes × 4 dims (1.75 bits per dimension, VE bound ≈ 0.91, §1.3b), with no coarse
level. It answers "is reconstruction or abstraction what helps?" at step 4,050.

### 4.7 [ETA]'s temporal integration of the abstract action (arm, not default)

**The idea, as an arm.** [ETA]'s key ingredient was not the GRU. It was the **switching gate**: z_t = β_t z̃_t + (1 − β_t) z_{t−1}.
- An abstract action persists over a variable number of steps and switches when the behaviour changes.
- With β forced to 1, internal RL failed.

**What v4 can do cheaply at concept level, without learned chunk boundaries.**
- Plan state:
  - z_j = (1 − κ_j) u_j + κ_j z_{j−1};
  - κ_j = clamp(s, 0, 1) · σ(w · u_j + b), so κ ∈ [0, 1) and z cannot diverge;
  - the scalar s is initialised to 0, so z = u exactly at step 0;
  - s is alive through ∂L/∂s = σ(·)(z_{j−1} − u_j) · ∂L/∂z (the clamp's gradient is taken as 1 at s = 0);
  - reset at document boundaries.
- The plan reads z instead of u, so the abstract action can **persist across several 6-token concepts** and switch when κ
  drops. This is variable duration at concept granularity.
- **Cost.** M = 170 per row (128 in the k = 8 arm). The recurrence is linear and κ is one scalar per concept, so it runs as one masked
  M × M weighting (B × 170 × 170), shared across channels.
  It is 258 parameters (w, b, s), and one bypass (s = 0).
- **Why an arm.**
  - [ETA] shows that switching sparsifies when the base model is **frozen** and the bottleneck is variational (KL). In
    their co-trained case it collapsed to one switch at the start.
  - v4 trains jointly with no KL, so the faithful use is post-training on a frozen v4 (§10.6).
  - The arm tests the cheap joint version: does κ learn sparse, structured switching at all? Read it from the κ histogram
    at 24 h. If κ stays ≈ 0 or ≈ 1 everywhere, it is dead.

### 4.8 Concept-module summary

| | v3 | v4 |
|---|---|---|
| layers × passes | 4 × 3 | **4 × 2** |
| own / shared FFN | 736 / 1,024 | 736 / 1,024 |
| pass conditioning | (t, Δ) table | **none** (removed) |
| shortcut | sampled 1 / 2a / 2b trajectory, batched (+42% core compute) | **none** (terminal-only); pass-1 NCP is an arm |
| concepts per row | 256 (k = 4) | **170** (k = 6); 128 in the k = 8 arm |
| codebook | PQ 32 × 128 × 8 | **coarse 1,024 × 256 + PQ 32 × 256 × 8 on the residual** |
| NCP | PQ cross-entropy | **coarse CE + PQ CE** |
| compute parameters | 4,881,216 | **4,838,208** (+1 temperature) |
| core compute vs v3 | 1.00 (including the shortcut's +42%) | ≈ 0.30 / 1.42 ≈ **0.21** (attention), ≈ 0.44 / 1.42 ≈ **0.31** (position-wise); k = 8 arm: ≈ 0.12 / ≈ 0.23 |

### 4.9 Concept-module acceptance tests

1. **Causality and document isolation at concept rate** (v3 tests, at k = 6 and at k = 8 for the arm).
2. **Terminal-only.** In the default, no loss term reads S₁. In the arm, S₁ is computed once and the same tensor feeds
   NCP(S₁).
3. **Residual VQ.** On fixed inputs, coarse assignment, then residual, then PQ assignment reproduces ĉ within the stated VE.
   Dead-code restarts reset the Adam moments of both levels.
4. **NCP targets** are the next concept's coarse and fine codes, on `pair_ok` pairs only.
5. **Arm 4.7 bypass.** With s = 0, z ≡ u, bit-exact.


## 5. Decoder

**The owner's design.** Keep v3's decoder and connect Engram to it at more than one depth, sharing the encoder's table.
Frequent tokens and stored n-gram knowledge can then come straight from the table, and the encoder and decoder spend their
weights on representation rather than on memorised facts.

### 5.1 Anatomy (unchanged from v3)

- **Layers.** 4 confined layers, d 256, SwiGLU 1,024, 4 heads (2 KV), window 8, partial RoPE 16, tied output head (the
  encoder's embedding).
- **The private stream D.**
  - Start: D₀ = tok + P_t.
  - Each layer reads D through a gated-residual read.
  - Queries come from that read; **keys and values come from `tok` only**, within the window.
  - Updates go to D. Nothing ever reads D at another position.
- **Reach without the core** is 63 + 7 = 70 tokens at any depth, proven by unit test.
- **Why it stays.** It is what makes "the concept route carried it" provable (§3.1). It also makes the §3.5 argument hold:
  the decoder reads the last 70 tokens directly, so function words and local syntax never had to be in a concept.

### 5.2 Engram in the decoder: two reads of the encoder's gathered rows

- **Where.** After decoder layers 0 and 2 (of 4): early, to feed the remaining layers; and late, close to the output, where
  "spit out the common token" matters most.
- **What each read is.** v3's Engram read 2 anatomy, applied to D:
  - h = GRRead([D]); k = W_K e_t; v = W_V e_t;
  - α = σ(⟨rms h, rms k⟩ / √256);
  - D ← D + GRWrite(h) · SiLU(α · v).
  - e_t is the encoder's gathered row for position t (§3.2), and W_V is zero-init.
  - There is no short convolution: a convolution would mix neighbouring positions' rows. That is harmless for reach, but
    not needed, since the n-gram key already spans t − 2 … t.
- **No second gather and no second table.** The decoder reuses `e` (B, T, 384) from the encoder's one gather. The table's
  rows are trained by all four reads (two encoder, two decoder).
- **Reach safety.** Two conditions together give it:
  1. e_t is a function of canonical token ids in position t's own n-gram window (t − 2 … t). It is never a function of L,
     of G, or of the concepts, and every decoder position already sees those ids.
  2. The read's gate and write read only D at position t.

  So reach stays 70, plan confinement stays exact, and §5.5 re-runs the D1 and D2 tests with both reads installed.
- **What it should buy, and what it might not.**
  - The decoder already receives Engram information *indirectly*: encoder read 1 writes into L, and `tok` = zcRMSNorm(L).
  - The new reads give direct access at decoder depth, with the decoder's own query. This is the owner's "spit out common
    tokens" path.
  - The measured value of Engram in the encoder (+0.60 nats, recall1 halved without it) argues it is worth trying. It does
    not prove the decoder needs a second route.
  - **What it really adds.** Encoder read 1 writes into L at layer 0, and seven S layers process it before `tok`. The
    direct reads add a *raw* n-gram feature at the output, effectively a trigram-model feature for the very next token.
    That is the owner's "spit out common tokens" hypothesis, and it is a claim about **frequent** tokens.
  - **24 h report:**
    - switch-offs per read: decoder read 0 off, read 2 off, both off. Depth 0 vs 2 is then answerable.
    - the both-off loss delta **binned by token frequency** (top-100, top-1k, the rest);
    - each read's mean gate α.
  - **Kill criterion** (read where the effect should live): both-off costs < 0.01 nats on the top-1k tokens *and* < 0.01
    overall *and* no probe moves by ≥ 2 SE → remove in v5.
- **Parameters.** Per read: W_K 98,304 + W_V 98,304 + GRRead 16,896 + GRWrite 257 = **213,761**. For two reads,
  **427,522**.

### 5.3 Plan builder: shift one chunk, repeat over k tokens (as v3, with k = 6)

**In words (the NCP mechanism, unchanged from v3 except k).**
- The core reads concepts 0 … j and, at concept j, outputs:
  - its state u_j;
  - the predicted next concept ĉ_pred,j, the concept of chunk j + 1.
- **Shift.** That plan is used one chunk later, while the decoder writes the tokens of chunk j + 1. Those are exactly the
  tokens the prediction is about.
- **Repeat.** The same plan vector is added at all k positions that write chunk j + 1. A learned offset o_r tells each
  position where it is inside the chunk (r = 0 … k − 1).
- **Start of a document.** Positions before the first complete chunk of a document have no plan yet. They get a learned
  null plan. That is the first k − 1 tokens: 5 at k = 6, 7 at k = 8, 3 in v3.

```
k = 6     tokens:  [ chunk j : 6j … 6j+5 ] [ chunk j+1 : 6j+6 … 6j+11 ]
          core:      u_j , ĉ_pred,j (= predicted concept of chunk j+1)
          decoder:   positions 6j+5 … 6j+10 (which predict tokens 6j+6 … 6j+11) all read
                     P = W_u u_j + W_c ĉ_pred,j + o_r ,  r = 0 … 5
```

**As formulas.**
- P_t = W_u u_{j*} + W_c ĉ_pred,{j*} + o_{r(t)}, with:
  - j*(t) = ⌊(t + 1) / k⌋ − 1, the newest complete chunk;
  - r(t) = (t + 1) mod k, the position of the predicted token within its chunk.
- **Null plan.** The learned null replaces W_u u + W_c ĉ when j* < 0 or the chunk is in another document.
- **Offsets.** o is k × 256: +512 parameters at k = 6 and +1,024 at k = 8, over v3's 4 × 256.
- **ĉ_pred is the full predicted next concept.** Its exact definition, and how the fine part is conditioned on the coarse
  code, are pinned in §6.2.
- **What the plan cannot include (§3.5.3).** The plan is built from complete chunks only, so it excludes the newest
  0 … k − 1 tokens. Those reach the decoder through `tok`. A query must sit in a complete chunk for the core to answer
  it. Measured: the recall-family question is inside the plan for 100% of items at k = 4, 6 and 8, with a minimum margin
  of 4, 2 and 0 tokens.

### 5.4 Making the text depend on the abstract action (the RL door): options for the owner

**The fact.** In v3 the decoder writes from u. ĉ_pred is worth 0.013 nats and multi4 0.05 (§1.3). If concept-level RL is
to steer generation, the text must *depend* on the chosen code.

| option | what changes | effect on next-token loss as a metric | what [ETA] says |
|---|---|---|---|
| **A · as v3, plus the coarse level in ĉ_pred (default)** | nothing beyond §4.6 | unchanged, comparable | the door is opened *after* pretraining (C) |
| B · posterior-code dropout | in training, with probability p (e.g. 0.5), the plan reads the **actual** next chunk's coarse embedding E⁰_{k_{j*+1}} instead of the predicted one | training NTP is then conditioned on ≤ 10 bits of the chunk being predicted (≤ 1.16 nats per token (0.87 at k = 8) at most), so training loss is no longer a likelihood. **Evaluation must use the predicted path only** (still comparable) | not tested in the paper; it is the VQ-VAE prior/decoder split |
| C · post-training controller on a frozen v4 | after pretraining, train a small controller (the [ETA] metacontroller: an encoder with a switching gate, KL to a prior) that writes a residual offset **at concept rate**: into u, or into S between the two passes. Then do RL in its low-dimensional code space. Only a concept-rate site gives the T/6 contracted horizon; D at decoder layer 2 (token rate) is the fallback if u-control is too indirect | none (pretraining unchanged) | **this is what worked**: co-training collapsed, frozen base plus controller discovered the abstractions |

- **Recommendation.** A for pretraining, C for the door.
  - [ETA]'s strongest result is that abstractions are found *in a frozen pretrained model*. Training the base to obey
    codes from the start (B) is closer to their co-trained case, which collapsed.
  - B is listed because it is the only option that makes the pretrained decoder controllable by codes directly. Its cost
    is the meaning of the training loss.
- **This is an owner decision (§10.6).**

### 5.5 Decoder summary and acceptance tests

| | v3 | v4 |
|---|---|---|
| layers, width, FFN, heads, window | 4 · 256 · 1,024 · 4 · 8 | same |
| Engram | — | **2 reads (after layers 0 and 2) of the encoder's gathered rows**, zero-init W_V |
| plan (shift one chunk, repeat over k) | j* = ⌊(t+1)/4⌋ − 1, offsets 4 × 256, null for 3 tokens | **j* = ⌊(t+1)/6⌋ − 1, offsets 6 × 256, null for 5 tokens** (k = 8 arm: 8 × 256, null for 7) |
| ĉ_pred | PQ expectation | **coarse + PQ expectation** |
| compute parameters | 4,464,648 | 4,464,648 + 427,522 + 512 = **4,892,682** |

**Tests:**
1. **D1 reach, with both Engram reads installed.** Perturbing `tok` at p changes logits only at p … p + 7.
   - **Engram path, exact bound.** Perturbing a token id at p changes e at exactly p, p + 1, p + 2 (the 3-gram window).
     D is private and the keys/values come from `tok`, so through the Engram path alone the logits change at exactly
     p … p + 2. The full-model reach from ids stays 70, via `tok`. The test isolates the Engram path by holding `tok`
     fixed.
2. **D2 plan confinement, k = 6.** Perturbing u of chunk j changes exactly its 6 owner positions (6j + 5 … 6j + 10). The
   k = 8 arm checks 8j + 7 … 8j + 14.
3. **D3 null plan** at exactly the first k − 1 tokens of every document (5 at k = 6, 7 at k = 8); no plan leaks across
   documents.
4. **Bypass.** With both W_V = 0, the decoder equals v3's decoder (k = 6 plan), bit-exact in float64.
5. **Shared gather.** The decoder's `e` is the encoder's tensor (data pointer equality).


## 6. Objective

**The owner's view is that v3's losses are right.** v4 keeps their form, removes the two shortcut terms (terminal-only,
§4.2), and extends NCP and VQ to the two-level codebook (§4.6).

### 6.1 Terms

L = NTP(u) + λ_c · CE_coarse + λ_f · CE_fine + VQ_coarse + VQ_fine, with u = zcRMSNorm(S₂).

| term | definition | weight | trains |
|---|---|---|---|
| NTP | cross-entropy of token t+1 at position t, over valid targets (same document, not a pad) | 1 | everything except the codebooks |
| CE_coarse | cross-entropy of the next concept's coarse code k_{j+1} under −‖q_j − E⁰_k‖² / τ₀, over `pair_ok` pairs | **λ_c = 0.05** | head (W_q, τ₀), core, encoder via C |
| CE_fine | mean over 32 segments of the cross-entropy of the next concept's fine codes, distances from the conditioned residual query (§6.2), over `pair_ok` pairs | **λ_f = 0.1** (v3's λ_ncp) | head, core, encoder via C |
| VQ_coarse | ‖sg ĉ_j − E⁰_{k_j}‖² / 32 over complete chunks (scaled to v3's per-segment-mean convention) | 1 | E⁰ only |
| VQ_fine | v3's PQ term on the residual r_j = sg ĉ_j − E⁰_{k_j}: mean over segments of the squared error, over complete chunks | 1 | PQ codebook only |

**Weights.**
- λ_f keeps v3's 0.1 on the fine term, which has the same form and scale as v3's NCP.
- λ_c = 0.05: the coarse cross-entropy can be up to ln 1,024 = 6.93 nats, about twice the fine per-segment scale.
  Halving its weight keeps the total NCP pull near v3's.
- Both are starting points. The 24 h gradient audit reports the share each term contributes to the core's gradient, as
  v3's §6.4 audit did.

**The printed loss will look larger: log the terms separately.**
- ‖ĉ‖² = 256. From v3's printed vq ≈ 2.6 at VE 0.65, the per-concept variance is ≈ 240.
- The coarse level removes ≈ 5% of it, so VQ_coarse ≈ 0.95 × 240 / 32 ≈ **7**, and VQ_fine ≈ 2.4.
- The headline loss moves from v3's ≈ 5.1 to ≈ 12 with no change in NTP. In v3 the headline loss was about half VQ; in v4
  it would be about two-thirds.
- **What to do:**
  - the dashboard and logs headline `ntp` (and loss − VQ);
  - VQ_coarse, VQ_fine, CE_coarse and CE_fine are logged separately;
  - the divergence guard already uses the NTP EMA (v3 `train.py`), so it is unaffected.

**No commitment term.**
- ĉ is detached in both VQ terms, so codes follow the concepts, never the reverse.
- v3 did the same, and its encoder developed concepts the decoder could use: u_off +0.50.

### 6.2 The fine NCP is conditioned on the true coarse code inside the loss; the plan never sees it

The fine targets are PQ codes of the residual r_{j+1} = ĉ_{j+1} − E⁰_{k_{j+1}}, i.e. relative to the **true** next coarse
centroid. So:
- **In the loss (teacher-forced, as residual-VQ priors do):**
  - the fine logits are distances from q_j − E⁰_{k_{j+1}}, the same frame as the targets;
  - this uses the target's coarse code only to build the fine logits *inside the loss*, so nothing reaches the plan or the
    decoder.
- **For the plan, in training and inference alike:**
  - ĉ_pred = c̄ + PQexp(q_j − c̄), with c̄ = Σ_k π_k E⁰_k;
  - π is the coarse softmax, and PQexp is v3's soft expectation over the fine codes.
  - It is differentiable and uses no future information. Option B (§5.4) is the only design that would put the true code
    into the plan.
- **For generation or RL:** sample (or take the argmax of) k̂ from π, then the fine codes from the distances to q_j − E⁰_k̂.
  The coarse code is the decision; the fine codes are its realisation.
- **Stated plainly: the plan's fine expectation lives in a frame no loss trains.**
  - The PQ head is trained in the q − E⁰_{k_true} frame. The plan evaluates it at q − c̄.
  - That is acceptable, because the plan is a learned input and the decoder learns to read whatever comes out.
  - **Arm, if ĉ_pred's share stays ≈ 0 at 24 h:** the plan's frame uses the hard k̂ = argmax π, with gradient through q
    only.

**Initialisation and restart order.**
- **τ₀ (coarse temperature):** log(2 · 256) = log 512, by v3's rule of log(2 · ds) per segment, so the first steps do not
  saturate.
- **Dead-code restarts:** coarse rows first; then recompute the residuals with the updated E⁰; then restart dead fine
  codes from those residuals. Otherwise fine restarts seed from stale residuals.

### 6.3 Normalisation and masks (carried from v3, recomputed for k)

- **Full-batch denominators.** All 16 micro-batches of an optimizer step are fetched first, and every term is a sum over
  the whole optimizer batch divided by its total count (v3, V3-004).
- **Masks.**
  - NTP: valid targets (same document, not a pad).
  - `nonempty` and `complete` (k real tokens), and `pair_ok` (current chunk nonempty, target chunk complete, same
    document), all at the run's k (6; 8 in the arm).
  - VQ terms over `complete`; NCP over `pair_ok`.
- **Dead-code restarts** every 100 optimizer steps, for both codebooks, from complete real chunks of the last micro-batch.
  Restarted rows get their Adam moments zeroed (v3, unit-tested).
  - E⁰ is restarted from ĉ.
  - The PQ codebook is restarted from the residuals of the current coarse assignment.
  - Codebook init: k-means++ on the first optimizer batch, coarse first, then fine on its residuals (≥ 4 × K candidates
    for the coarse level, so ≥ 4,096 complete chunks: one optimizer batch holds 32 rows × 170 = 5,440 concepts, and v3's
    accumulate-until fallback covers a short one). At k = 8 a batch holds 32 × 128 = 4,096 concepts, fewer once only
    complete chunks count, so the fallback always runs there.

### 6.4 Gradient-reach map (what trains what)

| parameter group | NTP | NCP (both levels) | VQ |
|---|---|---|---|
| decoder layers, plan builder, decoder Engram reads | ✓ | — | — |
| core layers, shared FFN, head (W_q, τ) | ✓ (through the plan: u and ĉ_pred) | ✓ | — |
| encoder L stream, S layers | ✓ (through `tok`) and ✓ (through C) | ✓ (through C) | — |
| encoder G stream, F layers, pooler | ✓ (**only through the concept route**: plan → u → core → C) | ✓ | — |
| Engram table | ✓ (all four reads) | ✓ (encoder reads, via C) | — |
| codebooks E⁰, PQ | — (detached in the head) | — | ✓ |

- **The G stream is trained only through the concept route.** The decoder reads only `tok` (L). So the F layers, the F
  loop and the pooler learn only from NTP via the plan and from NCP.
- The shortcut gradients that also reached them in v3 are gone (terminal-only). That puts more weight on λ_c and λ_f
  being large enough.
- **24 h check:** report ‖∇ G-stream parameters‖ / ‖∇ L-stream parameters‖ (F layers + pool vs S layers) and compare with
  the same ratio measured on v3's checkpoint. Flag if v4's ratio is lower.

### 6.5 Objectives of the arms

| arm | objective change |
|---|---|
| pass-1 NCP (§4.2) | + 0.05 · (CE_coarse + CE_fine)(zcRMSNorm(S₁)), no L_scons |
| posterior-code dropout (§5.4 B) | none in the loss. The plan uses E⁰_{k_{j+1}} with probability p in training; evaluation uses the §6.2 plan only |
| fine PQ 64 × 128 × 4, no coarse (§4.6) | v3's NCP and VQ with S 64, N 128 |
| k = 4 control (§3.5) | masks at k = 4, T = 1,024 |


## 7. Parameter and compute budget

### 7.1 Parameter ledger (repository convention: compute = encoder without embedding and Engram table + core without codebooks + decoder)

| module | v3 | v4 | change |
|---|---|---|---|
| encoder layers (8) | 8,948,776 | 8,948,776 | — (the loop reuses them) |
| encoder shared FFN (SwiGLU 384) + 8 gains | — | 294,920 | +294,920 |
| Engram read 1 + read 2 (non-table) | 429,060 | 429,060 | — |
| CRC router, pooler, tok norm | 263,171 | 494,364 | +231,193 (attention + FFN pooler, 493,337, replaces the linear pool, 262,144; §3.5.4) |
| **encoder** | **9,641,007** | **10,167,120** | +526,113 |
| core layers (4) | 3,985,952 | 3,985,952 | — |
| core shared FFN 1,024 | 786,432 | 786,432 | — |
| (t, Δ) table | 43,008 | — | −43,008 |
| head (W_q, τ) + out norm | 65,824 | 65,825 | +1 (τ₀) |
| **core** | **4,881,216** | **4,838,209** | −43,007 |
| decoder layers (4) + out norm | 4,332,296 | 4,332,296 | — |
| plan builder | 132,352 | 132,864 | +512 (6 offsets) |
| decoder Engram reads (2) | — | 427,522 | +427,522 |
| **decoder** | **4,464,648** | **4,892,682** | +428,034 |
| **compute total** | **18,986,871** | **19,898,011** | +911,140 (**< 20M**, 101,989 spare) |
| k = 8 arm | | 19,899,043 | pooler p and b +520, decoder offsets +512 (100,957 spare) |
| codebooks (excluded) | 32,768 | 327,680 (E⁰ 262,144 + PQ 65,536) | |
| embedding (excluded) | 2,097,152 | 2,097,152 | |
| Engram table (excluded) | 79,999,872 | 79,999,872 | |

These are estimates from v3's instantiated ledger plus the parts' formulas. §8.5 requires an instantiated count with
`param_table`'s reconciliation assertion before launch, as v3 did.

### 7.2 Compute per token (estimate; interleaved A/B decides)

- **Source profile.** `profile_step.py` (V3-006, before compile) profiled v3 with the **2b shortcut**: 3 main + 2 shortcut
  pass-rows. That is v3's most expensive case; the sampled average is 3 + 1.25.
- **Forward shares in that profile:** encoder 41.9%, core 36.4%, decoder 16.8%, losses 5%. The backward is taken to follow
  the same shares.
- **v3's average micro-step** is therefore ≈ 0.419 + 0.364 × 4.25/5 + 0.168 + 0.05 ≈ 0.946 of the profile.

| part | profile share | v4 multiplier vs profile | why |
|---|---|---|---|
| encoder | 0.419 | × 1.16–1.23 | F-layer loop (+13–20%) and the shared FFN (+≈ 3%) |
| core | 0.364 | × 0.18–0.27 | 2 pass-rows instead of 5, on 170 concepts instead of 256 (attention ∝ M², position-wise ∝ M); the pooler's W_K, W_V run per token (≈ 131k MACs, ≈ 1.5% of the encoder), its pool, W_O and FFN at concept rate: < 1% of the step |
| decoder | 0.168 | × ≈ 1.10 | two Engram reads (gate and projections; no gather) |
| losses (NTP, NCP, VQ) | 0.050 | × ≈ 1.0 | a 1,024-way coarse head on 170 concepts is small next to the 8,192-way NTP head on 1,020 tokens |
| **micro-step** | **1.00** (v3 average 0.946) | **≈ 0.79–0.85** | |

- **Expected throughput ≈ 1.1–1.2× v3's average**, about 3,500–3,800 tok/s, *despite* the encoder loop.
  - k = 6, M = 2 and dropping the shortcut free more compute in the core than the loop spends in the encoder.
  - **The k = 8 arm.** Core × 0.10–0.20 (2/5 × 0.25 attention, 2/5 × 0.5 position-wise), so the micro-step is
    ≈ 0.76–0.82. That is only ≈ 3–4% faster than k = 6: once M = 2 and the shortcut is gone, the core is ≈ 8–12% of the
    step.
  - This is a pre-compile estimate. `profile_step.py` and `ab_interleave.py` on the real v4 model replace it before launch.
- **Tokens in 48 h** at that rate: ≈ 600–660M, about 1.5–1.7× Chinchilla for 19.7M parameters.

### 7.3 Evaluation cost (off the 48 h clock)

- **v3's 24 h evaluation** took 61 min. Items were 2,257 s of 3,679 s, scaling with the number of interventions (13).
- **v4 adds** per-read decoder switch-offs (3), F loop off, coarse-off, pooler FFN off, and frequency bins: ≈ 19 interventions. Budget
  ≈ 1.4–1.5 h at 24 h.
- **At 48 h** it is more: two checkpoints, test + OOD.
- The hourly quick evaluation stays ≈ 80 s inside the clock, as in v3.

### 7.4 Memory

- **Activations.**
  - The core holds fewer activations: 170 concepts × 2 passes, against 256 × (3 + 1.25 shortcut). The pooler holds one
    block over 170 positions.
  - The encoder holds two more F-layer applications.
  - Net: roughly v3's footprint. v3's MPS driver memory was ≈ 5.8–7 GiB at micro-batch 2.
- **Process footprint.** v3's grows ≈ 0.9 GiB/h on the Python side (V3-016). That is an engineering item for v4 (§8.4),
  not an architecture item. Until it is fixed, v3's 24 GiB re-exec guard carries over.


## 8. Training recipe

### 8.1 Schedule, optimiser, batch (carried from v3)

- **Learning rate.**
  - One 48 h time-based cosine (warm-up 300 steps, peak 6e-4, floor 0.1×), with pauses and out-of-process evaluations
    excluded.
  - A milestone at 24 h: save, exit 5, launcher evaluation, resume (V3-004, V3-007).
- **Optimiser.** AdamW, β (0.9, 0.95), weight decay 0.1. No decay on biases, embeddings, the Engram table, temperatures,
  convolution kernels, codebooks, the null plan, the offsets, the pooler's position vectors and biases, and the pooler's FFN gain.
- **Batch.** micro-batch 2 × 16 accumulation × **1,020** tokens = **32,640 tokens per step**, 0.4% fewer than v3's 32,768.
  All comparisons with v1/v3 are made at **matched tokens**, not matched steps.
- **Speed paths.**
  - Block-local window attention with **block 60** (1,020 = 17 × 60); the k = 8 arm keeps v3's block 64.
  - `torch.compile` of every layer, with `k_norm` pinned eager (V3-006b; re-run `compile_check.py` with the new modules:
    the pooler, the decoder Engram reads, the F-layer loop).
  - Step memo off.

### 8.2 Data (rebuilt for k = 6)

- **Rebuilt** from the same tokenised corpus, same document order, same tokenizer, documents padded to multiples of 6
  with rows of 1,020 tokens:
  - `train/` and `val/` `{tok, doc}.npy`;
  - every `lang_eval/*_{tok, doc, ctx16}.npy`;
  - `maintenance/`.
- **Unchanged:** the evaluation and probe items (stored unpacked, packed at eval time with `pack_items(k = 6)`), the
  canonical-id table and Engram hashing, `token_bytes.npy`.
- **Check before launch:** every row's real-token fraction and document starts (all on 6-token boundaries); first-epoch
  order identical to v3's at the document level. Expected within-document real fraction: 96.9% (§3.5.3).
- **The k = 8 arm** needs its own copy: documents padded to multiples of 8, rows of 1,024, block 64 (96.1% real). Probe
  items are packed with `pack_items(k = 8)`.
- **What is lost.** Re-packing at T = 1,020 changes which documents share a row and the row order. v4 and v3 see the same
  corpus, but different sequences at any token count. "Step k = the same tokens" (§1.1) no longer holds across versions.
  - Comparisons with v1/v3 are at matched token *count*, same corpus, different packing.
  - The clean comparison is within v4: the k = 4 control arm.

### 8.3 Arms, in order

| order | arm | runs to | decides |
|---|---|---|---|
| 1 | **k = 4 control** (same v4 otherwise: attention + FFN pooler, F loop, decoder Engram) vs the v4 default k = 6 | step 4,050 each | the compression factor (§3.5.5 gate) |
| 2 | **k = 8** (same v4 otherwise; T 1,024) | 4,050 | whether compression 8 is affordable: promoted to v5's default if it passes §3.5.5 |
| 3 | **no F loop** (v4 default otherwise), with the main run's 24 h "F loop off" switch-off and the measured A/B price as supporting evidence | 4,050 | keep or drop the loop (§3.3): only the trained arm answers "worth it" |
| 4 | posterior-code dropout (§5.4 B), only if the owner chooses it | 4,050 | controllability vs the training-loss meaning |
| 5 | fine PQ 64 × 128 × 4 without coarse (§4.6); pass-1 NCP (§4.2); temporal integration (§4.7); pooler without FFN (§3.5.5) | 4,050 each | each named question |

Each arm is ≈ 10 h at the expected v4 speed. Arms run *after* the main 48 h run, on its step-indexed learning-rate trace,
as v3 planned (draft_v3 §9.3b).

**Realism.** v3 pre-registered arms and ran none (V3-016). So:
- **Mandatory:** arm 1, the k = 4 control. It is the only clean within-v4 comparison now that packing differs from v3
  (§8.2).
- **Strongly recommended:** arm 2 (k = 8, the owner's bolder compression) and arm 3 (no F loop).
- **If time:** arms 4–5.

### 8.4 Engineering items before launch

1. **Memory growth.** Find v3's Python-side footprint growth (compile caches, logging buffers, retained tensors). Run a
   2-hour footprint trace with `tracemalloc` snapshots on the v4 trainer.
2. **Dashboard file.** The trainer seeds its dashboard history from the jsonl logs at start, so a re-exec or relaunch no
   longer empties `v3_live.json`'s successor (V3-017).
3. **Headline metric.** Log NTP, CE_coarse, CE_fine, VQ_coarse and VQ_fine separately. The dashboard headline is NTP
   (§6.1).
4. **New telemetry.** Shared-FFN write share (encoder and core); decoder Engram gate means; pooler attention entropy per
   head, its weight on the 100 most frequent tokens vs the rest (§3.5.4), and the pooler FFN's write share; coarse-code perplexity; F-loop update ratio ‖ΔG second application‖ / ‖ΔG first‖.

### 8.5 Implementation order (each with a done-criterion)

1. Data rebuild at k = 6 (and k = 8 for arm 2) → §8.2 checks pass.
2. `model_v4.py` from v3's `model.py` → the instantiated `param_table` reconciles and is < 20M.
3. The acceptance harness `test_v4.py` → §3.8, §4.9 and §5.5 pass, each with a deliberate-defect mutant that fails.
4. `compile_check.py` and `mps_check.py` on v4 → gradients within tolerance.
5. `profile_step.py` and `ab_interleave.py` → the measured tok/s replaces §7.2's estimate.
6. Launch, with the v3 launcher, guards, watcher and monitor policy carried over.


## 9. Evaluation and acceptance gates

### 9.1 Evaluation suite (v3's, with v4 additions)

- **Hourly quick evaluation (v3).**
  - Validation language on 128 + 128 rows, with nats and bits per byte.
  - 5 probes × 128 items, plus plan swap on recall1.
  - Codebook health for both levels.
  - Loop geometry for the two passes.
- **24 h milestone (validation) and 48 h final (test + OOD): `final_eval.py` with the v4 interventions.**
  - core off, plan swap, u off, ĉ_pred off, shared FFN off (encoder and core separately), Engram off (all four reads),
    encoder read 2 off;
  - **decoder read 0 off, read 2 off, both off, with frequency bins**;
  - **F loop off**, F layers off, 1 pass vs 2;
  - **coarse code off in ĉ_pred**;
  - **pooler FFN off** (c₁ = c₀).
- **Additions:**
  - pooler weights: which positions and token types each head favours, the weight on the 100 most frequent tokens, and
    the pooler FFN's write share;
  - coarse-code perplexity and coarse CE vs ln 1,024;
  - the G/L gradient ratio vs v3;
  - hop by depth, and the bridge-by-pass probe for the two passes.

### 9.2 Early gates (read at matched tokens against v3)

v3 values are interpolated at the same token count; v4 step ≈ v3 step × 32,768 / 32,640.

| gate | at v3's step | criterion | note |
|---|---|---|---|
| G1 · nothing dead | 100 / 200 / 300 | every zero-init branch has moved: encoder shared W₂, both decoder W_V, pooler q, p_r and FFN W₂; coarse codebook perplexity > 25% of K (the §10.5 threshold) | v3's OPENED check, extended |
| G2 · language | 1,217 / 1,792 / 2,376 | val within v3 + 0.05 nats (k = 6 may cost up to 0.03, §3.5.5) | v3 at those steps: 3.319 / 3.121 / 3.005 |
| G3 · recall onset | 1,792 | recall1 ≥ 0.25 | v3 had 0.594 |
| G4 · probes alive | 4,050 | carry3 ≥ 0.40 and multi4 > 0 | v3 had carry3 0.617, multi4 0.086 |
| G5 · second pass works | 24 h | R2 − R1 multi4 ≥ 2 SE | the reason M = 2 was kept (§1.4) |

No gate stops the run (v3 policy); gates inform the owner.

### 9.3 Reporting rules

- **bind** is reported with its validity caveat (§3.5.3): it does not test the core at any k.
- **hop at k = 8** is reported with its caveat (§3.5.3): the hop count is outside the plan for 25% of items.
- **Switch-off deltas** are fixed-checkpoint effects, not training-time value (v3 rule).
- **v1 and v3 comparisons** name their split. Language is compared validation against validation, and test against test
  only after both 48 h evaluations exist.
- **Different packing.** v4 vs v3 is at matched token count over the same corpus with different packing (§8.2), so small
  differences (< 0.02 nats) are not attributable. The k = 4 arm is the within-v4 reference.


## 10. Whole-architecture review (step by step)

### 10.1 Data flow and shapes

1. **ids, doc:** (B, 1,020) → Engram indices (B, 1,020, 8) on CPU → one gather → e (B, 1,020, 384).
2. **Encoder** (8 layers, F layers applied twice):
   - L, G (B, 1,020, 256);
   - tok = zcRMSNorm(L), reach 63;
   - router mix m (B, 1,020, 256).
3. **Pooler** over each 6-token chunk (4-head attention, then SwiGLU 384) → ĉ (B, 170, 256) on the sphere; cdoc =
   doc[:, ::6]; masks at k = 6.
4. **Core:** S₁ = Φ(ĉ, 0), S₂ = Φ(ĉ, S₁) → u (B, 170, 256) → head: coarse logits (B, 170, 1,024), fine logits
   (B, 170, 32, 256) → ĉ_pred (B, 170, 256).
5. **Plan:** the plan of chunk j (u_j, ĉ_pred,j) is shifted one chunk and repeated over the 6 positions writing chunk
   j + 1 (j*(t) = ⌊(t+1)/6⌋ − 1), plus o_r; null for the first 5 tokens of each document.
6. **Decoder:** D₀ = tok + P; 4 confined layers, with Engram reads of e after layers 0 and 2 → logits (B, 1,020, 8,192).

**What each tensor may see:**
- tok sees ≤ 63 tokens back.
- D_t sees tok within 8 positions back, its own plan, and e_t.
- u_j sees concepts ≤ j of its own document.
- **Reach without the core is 70 tokens; with the core, the whole document**, through plans of complete chunks only.

### 10.2 Step 0: every new branch is identity and alive

| branch | init | output at step 0 | alive because |
|---|---|---|---|
| encoder F loop | same weights applied again | a second refinement (not identity: the loop is a real change from step 0, with `enc_loop = 1` as the bypass) | — |
| encoder shared FFN | W₂ = 0, g_ℓ = 1 | 0 | ∂W₂ = g · h ≠ 0 |
| pooler attention | q = 0, b = 0, p = 0, W_K random, W_V = W_O = I | masked mean of the chunk | ∂q through the random W_K; ∂p through the values |
| pooler FFN | W₂ = 0, g = 1 | 0 (c₁ = c₀) | ∂W₂ = g · h ≠ 0 |
| decoder Engram reads | W_V = 0, GRWrite = 1 | 0 | ∂W_V = α · write · ∂D ≠ 0 |
| coarse codebook | k-means++ on real concepts | — | VQ from step 1 |
| arm §4.7 | s = 0 | z = u | ∂s ≠ 0 (clamp gradient 1 at 0) |

### 10.3 Mistake ledger: each v1 / v2 / v3 lesson and v4's countermeasure

| lesson (source) | v4 |
|---|---|
| Recurrence looked idle on language (v1 R1 ≈ R4; v3 1 vs 3 = 0.005) | judged on probes too: the 2nd pass is worth multi4 +0.09, so M = 2 is kept and the 3rd pass dropped (§1.4) |
| The (t, Δ) table never earned its place (v3) | removed |
| Intermediate supervision may harm late states (v2) | terminal-only; pass-1 NCP only as an arm, without L_scons (§4.2) |
| The codebook fit is capped by the segment shape, not by N or segment count (v1 = v3 = 0.651) | 1 bit per dimension (N 256, ds 8) for fit; a coarse level for abstraction; the S 64 × ds 4 arm (§4.6) |
| The decoder barely uses the discrete code (c_off 0.013) | stated; the RL door goes through post-training control (C), or B by owner choice (§5.4) |
| Language deltas missed real contributors (shared FFN, 2nd pass) | every switch-off reports probes beside nats (§2.2) |
| Probes read too early cannot fire (carry3, multi4 = 0 at ≤ 2,376) | the deciding readings are at 4,050 and 24 h (§3.3, §9.2) |
| The pool, initialised as an average, trained into a per-slot recency map that cannot weight by content (v3 step-11,348 checkpoint) | attention + FFN pooler with shared weights across slots; per-head frequency-band telemetry shows whether it selects (§3.5) |
| Dashboard reset at re-exec (V3-017) | history seeded from logs (§8.4) |
| Memory growth needs a re-exec (V3-016) | find the leak before launch (§8.4) |
| Throughput claims from sequential runs are noise (V3-006) | interleaved A/B only (§2.1) |

### 10.4 The owner's brief, point by point

| owner | v4 | agreement |
|---|---|---|
| keep the encoder: 8 layers, L/G streams, gated residual, Engram at several depths | kept | **agree** |
| encoder looping (whole, subset, or layer-wise as in [Loopie]) | layer-wise loop of the two F layers only; S-layer or whole-encoder loops would break reach(L) = 63 and void the probes | **agree with a constraint** |
| shared FFN in the core (small) and in the encoder | core keeps the full 1,024 (multi4 −0.129 without it); encoder adds a SwiGLU 384 | **agree; core size kept on evidence** |
| concept module 4–5 layers, recurrence 2 | 4 × 2 (5 layers would halve the shared FFN to fit) | **agree on 2 passes; 4 layers chosen on evidence** |
| bigger codebook | PQ 32 × 256 × 8 plus a coarse 1,024-way level | **agree, extended** |
| 6 or 8 tokens per concept; attention (or FFN, or both) instead of averaging; the core runs on T/6 or T/8 | default k = 6 with an attention + FFN pooler; k = 8 as a priced arm; k = 4 control. v3's pool shown to be a trained per-slot map, which cannot down-weight stop words (§3.5.1). Speed correction: k = 8 saves only ≈ 3–4% over k = 6, because the core is already ≈ 8–12% of the step | **agree; k = 8 tested, not default** |
| decoder as v3 with Engram, sharing the encoder's table | two reads at decoder depths 0 and 2, reusing the encoder's gather | **agree** |
| losses as v3 | same form, shortcut terms removed, NCP and VQ over two levels | **agree, adapted** |
| < 20M, d 256 | 19.90M, d 256 | **agree** |
| concept-level RL is more sample-efficient: (N^S)^{T/4} vs V^T | false for the full PQ code (2²²⁴ per concept vs 2⁵² per 4 tokens); true for a small action with a T/6 horizon: the coarse code under option B, or the post-training controller's z under C | **corrected** (§1.7, §4.6, §5.4) |
| [ETA] is similar to v3 | similar in the residual abstraction and the predicted abstract action. Different in fixed vs learned duration, frozen base vs joint training, and low-dimensional z vs a large PQ code | **qualified** (§1.6) |

### 10.5 Risk register

| risk | sign | fallback |
|---|---|---|
| k = 6 loses too much (query margin ≈ 2 tokens; 1.5× content per concept) | the §3.5.5 gate at 4,050: val > +0.03 or recall onset > 300 steps late | k = 4 with the new pooler |
| the pooler stays near the mean (no content weighting learned) | head entropy near log k; frequent-token weight ≈ 1/k at 24 h | the k = 4 vs k = 6 gap then measures dilution, not selection; v5 revisits the pooler |
| the F loop buys nothing at ≈ 0.5B tokens ([Loopie]'s gains came after 1.2T) | 24 h "F loop off" < 0.01 nats and no probe change | drop the loop; spend the time on tokens |
| G stream under-trained without the shortcut gradients | G/L gradient ratio below v3's | pass-1 NCP arm, or raise λ_c / λ_f |
| coarse code unused or unpredictable | perplexity < 25% of K; CE ≈ 6.9 | drop the coarse level (v3-style PQ only) |
| decoder Engram redundant with tok | per-read off < 0.01 even on frequent tokens | remove in v5 |
| memory growth, swap | footprint slope as in v3 | v3's 24 GiB re-exec guard |
| data rebuild error (misaligned chunks) | §8.2 checks; D2/D3 tests at k = 6 (and k = 8 for the arm) | block the launch |

### 10.6 Open decisions for the owner

1. **The RL door (§5.4).** A + C (recommended: pretraining unchanged, a post-training controller at concept rate, as in
   [ETA]), or B (posterior-code dropout during pretraining, which makes the decoder obey codes but changes what training
   loss means).
2. **Core depth.** 4 layers × 2 with the full shared FFN (recommended), or 5 × 2 with own FFN 640 and shared 512. That now
   fits < 20M only if the pooler drops its FFN (§4.1).
3. **Compression factor.** Recommended: k = 6 default, k = 8 as arm 2, gates in §3.5.5 binding (k = 4 with the new pooler as
   the fallback). Alternative: make k = 8 the default and k = 6 the arm.
4. **v5 direction: learned concept boundaries.** Variable-length spans with a switching gate, [ETA]'s actual mechanism.
   The pooler is built to extend to it.


## 11. Advisor log

| # | step | verdict |
|---|---|---|
| 1 | approach, before writing (superseded on chunking by rows 4 and 10) | Skeleton first, mirroring v3's numbering. Apply V3-016 to the 24 h readings openly. Raise three corrections with numbers (action-space size, decoder ignores ĉ_pred, fixed chunks are [ETA]'s ablated case). Price everything against the 1.0M headroom. Chunk 4 default with attention pooling; chunk 6 as a priced arm. Encoder looping on a subset, decided by A/B. Decoder Engram reuses the encoder's gather. Codebook N 256 with the VE-ceiling caveat. |
| 2 | §0–§2 | Fix v1's codebook shape (it is 48 × 128 × 8, d 384), add the codebook evidence to §1, and correct "constant +0.09" (it narrowed to +0.059 as v1 annealed). Qualify the M = 2 reading: shortcut paths of an M = 3-trained model. → done; VE identical in v1 and v3 traced to the shared 8-dim × 128-code segment. |
| 3 | §3 encoder | Spine right: reach(L) = 63 makes F the only loopable layers. Fixes: shared-FFN gain init 1 (both factors at zero would be dead); the quality gate moved to step 4,050 (carry3 and multi4 are 0 at ≤ 2,376) plus a matched wall-clock condition; loop price widened to 1.13–1.20× (F attention is full 1,024²); [ETA] mid-depth marked as analogy; interface §3.7 added. For §5, re-run the D1 reach test with the decoder Engram installed. |
| 4 | §3.5 rewritten for k = 6 (owner) | "Message to the future" is the right argument. Fixes: the staleness row had it backwards (the plan excludes the most recent k−1 tokens, where the query sits) → query-in-complete-chunk constraint with per-probe margins (≈ 4 → ≈ 2 tokens), and bind flagged as not testing the core at either k. "Only gradient" softened. Values pool m_i (v3's quantity); W_K init stated. Data-rebuild inventory added. For §9: order the arms (k first), and judge the loop from the switch-off plus the A/B price unless that is ambiguous. |
| 5 | §4 concept module | Option A and the table removal are right; the two-level codebook makes the owner's inequality true. Fixes: pass-1 supervision reversed draft_v3's terminal-only rule → default is terminal-only, pass-1 NCP (without L_scons, which would collapse pass 2) is an arm with a kill criterion; the coarse code adds ≈ 5% VE at most, so it is a cluster/action label with perplexity and NCP-CE checks; κ bounded with clamp; the shortcut saving quantified (+42% core in v3). |
| 6 | §5 decoder | Arithmetic and plan index right; A + C over B is what [ETA] supports. Fixes: exact Engram-path reach bound (p … p + 2, a loose bound cannot catch a leak); the fine NCP conditioned on the true coarse centroid (pinned in §6.2); option C at concept rate (u or S between passes) for the T/6 horizon; the owner's "common tokens" hypothesis tested by frequency-binned deltas, per-read switch-offs and gate means. |
| 7 | §6 objective | Consistent with §4–§5. Fixes: §4.6 and §5.4 now say the coarse code is the action only under option B (under A + C it is the NCP target and abstraction measurement; C's action is the controller's z); the printed loss becomes ≈ 12 because VQ_coarse ≈ 7, so headline NTP and log the terms separately; the plan's fine frame is untrained, stated, with a hard-k̂ arm; τ₀ init log 512; coarse-then-fine restart order; the G/L gradient-ratio check at 24 h. |
| 8 | whole draft, cross-section | Ledger holds. Fixes: §3.5 compute figures made consistent with §4 (21% / 31%); §2.1 Chinchilla reconciled with §7.2; §1.4's ĉ_pred row points to §5.4 / §10.6; §3.3's gate requires the trained no-loop arm, so §8.3 now runs it; "same tokens at step k" is lost by re-packing (§8.2, §9.3), so the k = 4 arm is mandatory; §7.2 re-derived from the profile that actually ran the 2b shortcut (≈ 1.1–1.2×); §7.3 evaluation cost added; arms realism (mandatory / recommended / if time). Then a full read-through fixed the remaining drift: T in §3.1, the §4.2 heading, the coarse code's label, the loop bypass wording, the perplexity thresholds, the learned-boundary status, the comparison principle and the status line. |
| 9 | final check before delivery | Self-checks were the right ones (ledger, references, five numbers spot-checked against `eval_24h.json`, full read-through). Last fixes: §1.1's same-tokens sentence scoped to v1 and v3; the consults recorded in `run_v3_19M/ADVISOR_LOG.md`. Deliver a short map of the draft: follows / disagrees / corrections / owner decisions. |
| 10 | owner's correction on concept compression (k tokens → 1 concept) | The owner's objection is visible in the draft: only k = 6 was priced (they asked for 6 or 8), there was no FFN option, v3's pool was called "linear" without saying it starts as exact averaging, and §5.3 was formulas, not shift-and-repeat. Lead §3.5 with the checkpoint evidence (v3's pool became a recency-weighted per-slot map, slot norms 7.7 / 7.6 / 8.3 / 12.1). Keep only the `lr_*` rows of the visibility measurement (the synthetic-family heuristic did not match those templates). Recommend one k and price the other. |
| 11 | k and pooler chosen, counts done, before writing | Counts and the step-0 identity verified (19,898,011; 101,989 spare; +1,032 at k = 8). Price concat-FFN fairly (458,752, affordable) and reject it on parameter sharing, not cost. State the head-slice trade (−131,072 funds the FFN). Define the k = 8 promotion gate. Word hop at k = 8 as a harder task, not a 0.75 ceiling. State the speed finding as a correction (k = 8 ≈ 3–4% over k = 6). Pin the empty-chunk rule. |
| 12 | review 1: §0, §1.7, §3.5, §5.3 | Ledger verified (10,167,120 / 19,898,011). Fixes: second-mention visibility given as 53 / 31 / 26%, not "not at any k"; the pooler's cost split (W_K, W_V per token, the rest at concept rate); §3.8 test 7 scoped to `nonempty` (empty chunks use the uniform fallback); the tok/s gate lowered to 1.03× with the expected 1.06–1.10× stated, because the k = 4 control already has M = 2. The advisor's earlier k = 8 core figures (0.07 / 0.14) were wrong; §4.8's formula gives 0.12 / 0.23, kept. |
| 13 | review 2: §4–§6 | §5.3 arithmetic right (6j+5 … 6j+10; 8j+7 … 8j+14; null for k − 1). Fix: §3.5.2 mixed two checkpoints, so restored the step-8,133 values (recall1 0.996, carry3 0.889, multi4 0.547, checked against `eval_24h.json`). k = 8 notes added in §4.7 and §5.4. |
| 14 | review 3–4: §7–§10 | Sums re-added by hand (encoder 10,167,120; core 4,838,209; decoder 4,892,682; total 19,898,011). "Pooler FFN off" moved into the switch-off list. Option B of §4.1 no longer fits < 20M with the pooler's FFN (20.13M); stated in §4.1 and §10.6. A mistake-ledger row added for the v3 pool. |
| 15 | final check before delivery | Two prose inconsistencies fixed: §3.5.5 pointed to arm 4 (pooler without FFN is arm 5); §7.2 called the whole pooler concept-rate (W_K, W_V are per token, ≈ 1.5% of the encoder; still < 1% of the step). Status line dated. Consults recorded in `run_v3_19M/ADVISOR_LOG.md` (rows 28–33). |
