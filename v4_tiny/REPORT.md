# v4_tiny: a scaled-down draft_v4 on CPU, tested in depth

Owner's request (2026-10-01): build a small version of the proposed v4 architecture on the CPU, in a new folder, and
test it deeply enough to be sure the architecture works, without touching the running v3 training.

**Folder.** `v4_tiny/`
- `code/`: `model_v4.py`, `test_v4.py`, `mutants_v4.py`, `data_v4.py`, `run_v4tiny.py`, `analyze_v4tiny.py`,
  `decomp_v4tiny.py`, `recall_check.py`, `reads_check.py`, plus unmodified copies of v3's modules.
- `runs/analysis.md`: every eval and final table, generated from the logs.
- `runs/`: one folder per arm, plus test logs.
- `data/`: documents re-extracted read-only from v3's shards.
- `ADVISOR_LOG.md`: every advisor consult, numbered.
- `GO_NOGO.md`: the go / no-go reading for v4 after this experiment.

## Verdict (the owner asked: "so that we are sure the architecture will work")

1. **It is implemented exactly as drafted, and it is correct.** The full-size model instantiates to the draft's ledger
   (19,898,011), every sub-part equal. 15 acceptance tests pass at k = 4/6/8. 13 of 13 deliberate defects are caught.
   v4's encoder and decoder reduce bit-exactly to v3's when the new branches are switched off.
2. **Every new branch is alive. All are used except three at this scale:**
   - the **pooler FFN** writes but does not help (Δ ≈ 0);
   - the **F loop at k = 6** gives only +0.002 (+0.022 at k = 4, so it depends on k; single seed);
   - the **decoder Engram reads** are zero to negative on text (−0.006 / −0.023) and negative on unseen probe text
     (−0.14 to −0.17). They are strongly positive only on *memorised* training text (+0.47 to +0.70): **a memorisation
     channel.** The encoder's Engram reads on the same table generalise (+0.14 to +0.17 on text).
3. **The concept route works for language.** Removing it costs +0.12 to +0.25 nats. Plans are row-specific. The second
   pass, ĉ_pred and both shared FFNs are each used.
4. **The pooler selects by content,** unlike v3's per-slot pool (which also collapses the coarse code). At this budget it
   selects *structure*: frequent tokens and template words. The owner's stop-word claim is not testable until retrieval
   is learned.
5. **Retrieval onset was not reached** at 5–7M tokens with d 64 (v3 needed ≈ 59M at d 256). It was checked with both
   the training and the unseen filler pools after a design flaw was found (§ Design flaw). That is a budget statement,
   not a mechanism failure: test 1 shows the route carries a perturbation past the decoder's reach, and switching the
   core off costs +0.12–0.25 nats.
6. **Poolers are tied on language** at tiny scale (attention + FFN, v3 linear / plain conv, gated conv). k = 6 costs
   ≈ 0.03 nats vs k = 4 at 5M and is 1.07× faster.

**Follow-up not run in the window** (cost at today's rates, two arms at a time on the P-cores, v3 ≈ 30% slower meanwhile):
1. **Corrected rerun:** fillers sampled fresh per item from all 457,614 documents, evaluated on both pools; ≈ 1 h per
   5M-token arm.
2. **k = 8** vs k = 6: ≈ 1 h per arm.
3. **Second seeds** of k4 / k6 / the poolers: ≈ 1 h each.
4. **A long single run** (≈ 20–30M tokens, ≈ 5–6 h) to find retrieval onset at d 64, the one question this window could
   not answer.

**How to read this report: three tiers, by how much transfers to the full 19.9M model.**

| tier | question | transfers to full scale? |
|---|---|---|
| 1 · correctness | Does the code implement draft_v4 exactly, with no leak, no dead branch and the right parameter count? | **Yes, with certainty.** Deterministic tests on the same code; the full-size model is instantiated from the same file. |
| 2 · learnability | Does the concept route carry facts beyond the decoder's 70-token reach? Does the pooler learn to select by content? Are the codebooks healthy? | **As mechanism.** If it works at d 64, the mechanism exists; how strong it is at d 256 is not measured here. |
| 3 · arm ordering | k = 6 vs k = 4; attention + FFN vs v3's linear / plain conv vs gated conv (k = 8 not run) | **Indicative only.** Tiny scale, single seed, ≈ 5M tokens per arm. Read with the standard errors given. |

---

## Setup

**Tiny model.** Same topology as the full v4; widths divided by about 4.

| part | tiny | full v4 |
|---|---|---|
| width, vocabulary, row | d 64, V 8,192 (v3's tokenizer), T 480 (divisible by 4, 6, 8) | d 256, V 8,192, T 1,020 / 1,024 |
| encoder | 8 layers SSSFSSSF, head dim 16, S / F heads 4 / 6, FFN 256, F layers 3 and 7 applied twice, shared SwiGLU 96 with per-layer gains | same pattern, FFN 1,024, shared 384 |
| Engram | 8 tables × 4,093 rows × 12, one gather, reads at encoder depths 0 and 4 and decoder depths 0 and 2 | 8 × 208,333 × 48 |
| pooler | 4-head attention + SwiGLU 96 (31,193 parameters) | 4-head + SwiGLU 384 (493,337) |
| core | 4 layers × M = 2, own FFN 184 + shared 256, no table | own 736 + shared 1,024 |
| codebooks | coarse K = 64 + PQ 8 × 256 × 8 (1 bit per dimension, as the draft) | coarse 1,024 + PQ 32 × 256 × 8 |
| decoder | 4 confined layers, FFN 256, plan shifted one chunk and repeated over k | FFN 1,024 |
| compute parameters | **1,257,475** at k = 6 | 19,898,011 |

**Data.**
- **Text.** 457,614 documents (38.6M tokens) re-extracted from 40,000 random rows of v3's training shard. Validation: 5,766
  documents from v3's validation split, 64 rows.
- **Packing.** Re-packed per k: each document padded to a multiple of k, the same document order for every k. Real
  fraction ≈ 0.86 at T 480.
- **Probes.** v3's `ProbeFactory` families recall1, carry3 and multi4.
  - Evidence gaps are 80–200 tokens, beyond the decoder's 70-token reach, so **only the concept route can answer**.
  - Fillers are real sentences from 3,000 training documents (6,519 sentences; see Design flaw). Periodic and final
    evaluation items use a different filler sample.
  - Each item is its own document, starting on a k boundary.
- **Mix.** Training rows are 50% packed probe items and 50% text (70% probe items in the trio).

**Training.**
- Batch: B 4 rows × 480; AdamW (0.9, 0.95), weight decay 0.1, clip 1.
- Learning rate: 2e-3, warm-up 400 steps, cosine to 0.1× over the token budget.
- Loss: NTP + 0.05 CE_coarse + 0.1 CE_fine + VQ_coarse + VQ_fine (draft §6.1).
- Codebooks: k-means++ on complete chunks of 3 batches; dead-code restarts every 200 steps (coarse first).
- Engineering: block-local window attention (block 60) and the mask memo.

**Evaluation.**
- Every 1M real tokens: validation NTP, and 128 items per family (teacher-forced greedy exact match, v3's scorer rule).
- Final: 192 items per family, under intact, core off, plan swap, ĉ_pred off, 1 pass, F loop off, pooler FFN off and
  decoder reads off. Plus codebook health and pooler telemetry (weight vs uniform on frequent tokens and on fact content
  vs template words; slot profile; entropy).

---

## Tier 1 · Correctness (complete)

### 1.1 The full-size model is exactly the draft's ledger

`model_v4.py` builds the full configuration. The Engram table is instantiated at 11 rows for counting; the real table is
counted analytically, as in v3.

| | draft §7.1 | instantiated |
|---|---|---|
| compute parameters, k = 6 | 19,898,011 | **19,898,011** |
| compute parameters, k = 8 | 19,899,043 | **19,899,043** |
| encoder layers / shared FFN + gains / Engram reads | 8,948,776 / 294,920 / 429,060 | same |
| router + pooler + tok norm (pooler alone) | 494,364 (493,337) | same |
| core layers / core shared FFN / head + out norm | 3,985,952 / 786,432 / 65,825 | same |
| decoder layers + norm / plan / decoder Engram reads | 4,332,296 / 132,864 / 427,522 | same |

### 1.2 Fifteen acceptance tests

- **Setup.** float64, CPU. Every structural test runs on a randomized model: at init the new branches output exactly
  zero, so an un-randomized test would prove nothing about them.
- **Result.** All 15 pass with block-local attention (`runs/tests_block60.log`) and with full-mask attention
  (`runs/tests_noblock.log`), on the final code.

| # | test | draft test | what it proves |
|---|---|---|---|
| 1 | end-to-end causality, k = 4/6/8 | §4.9 (1), §8.5 | Perturbing token p never changes a logit at t < p. The change reaches beyond position p + 70, so the concept route is live. |
| 2 | document isolation, k = 4/6/8 | §3.8 (4) | Editing document 0 leaves every logit of document 1 bit-identical. |
| 3 | reach(L) = 63 with the F loop | §3.8 (1) | Applying F layers 3 and 7 twice leaves the decoder-visible stream's reach at exactly 63. |
| 4 | pooler init = masked mean | §3.8 (2) | At step 0 the attention + FFN pooler equals the mean of the chunk's real tokens (exact to 1e-12). v3's linear pool differs on partial chunks, because it averages pads in. |
| 5 | every new branch alive | §3.8 (2), §9.2 G1 | At step 1, gradients are nonzero for the shared W₂ (encoder and core), the pooler's q / b / p / FFN W₂, both decoder-read W_V and read-2 W_V. The pooler's W_K, its FFN gain and the encoder's shared gains are exactly 0 at step 1 by construction, and nonzero at step 2. |
| 6 | codebook gradient isolation | §6.1 trains column | NCP gives zero gradient to both codebooks. The VQ terms give gradient only to the codebooks. VQ_coarse never touches PQ. |
| 7 | §6.2 frames | §6.2 | The fine CE depends on E⁰ only through the *true* next coarse code. The plan's ĉ_pred depends on all of E⁰ via c̄. |
| 8 | decoder D1 / D2 / D3, k = 4/6/8, with both Engram reads | §5.5 (1)–(3) | `tok` reach is exactly 7. Engram rows act per position. A token id reaches the logits through the reads at exactly p … p + 2. The plan of chunk j changes exactly positions kj + k − 1 … kj + 2k − 2 (shift one chunk, repeat over k). The null plan covers exactly the first k − 1 tokens. No cross-document leak. |
| 9 | empty chunk and pad masking, k = 4/6/8 (rows packed as the data are) | §3.8 (7), (8) | A row-end all-PAD chunk gives finite loss and gradients and is excluded from VQ and NCP. Changing the PAD embedding changes no nonempty concept. |
| 10 | restart order | §6.2 restart order | Dead coarse rows are restarted first. All 56 dead fine codes are residuals against the *updated* E⁰. |
| 11 | bit-identity with v3's encoder | §3.8 (2), (3) | v4's encoder with `enc_loop = 1`, the linear pool and shared W₂ = 0 equals v3's encoder bit for bit. `enc_loop = 2` changes ĉ but leaves `tok` bit-identical. |
| 12 | 13 switch-offs | §9.1 switch-offs | Each runs, stays finite and changes the output. |
| 13 | the pooler can select by content (toy) | §3.8 (9) | After training only the pooler, the best head puts 0.975 of its weight on a marked token at *every* slot. Copy error is 0.001 vs 0.339 for v3's linear pool, which cannot select by content. The gated conv reaches a 0.38 share at the same budget and 0.87 given 1,000 steps (it selects, more slowly). |
| 14 | block-local attention parity | §8.4 speed path | Block 60 equals full-mask SDPA at k = 4/6/8 (max difference < 1e-10). |
| 15 | decoder bypass = v3's decoder, k = 4/6/8 | §5.5 (4) | With both read W_V = 0, the v4 decoder equals v3's decoder (k-chunk plan) bit for bit; any W_V ≠ 0 changes it. |

**Not covered.**
- §3.8 (5), the shared gather: true by construction (`e` is passed from the encoder to the decoder), not asserted.
- §4.9 (3), residual VQ reproduces ĉ within the stated VE: not tested as a reproduction check. VE is *measured* in tier 2.
- §4.9 (5), the bypass of the §4.7 arm: out of scope (that arm is not implemented in the tiny model).

**Footnote on reach.** The "S layers looped too" mutant moves reach(L) from 63 to the draft's analytic 123 (§3.3, options
C/D). The measured value is 119–123 depending on position, because the farthest path's influence is about 1e-14 in
float64 and sometimes rounds to exactly zero. 123 is reached at p = 90.

### 1.3 The tests have teeth: 13 of 13 deliberate defects caught

Each mutant rewrites one line of the model (`mutants_v4.py`, `runs/mutants.log`). Every one is killed by an assertion,
not by a crash.

| mutant | caught by |
|---|---|
| plan reads the chunk being written (index off by one) | causality (future leak), D2 |
| pooler ignores the pad mask | pooler init, pad masking |
| head trains the coarse codebook (no detach) | codebook isolation |
| fine NCP in the plan's frame q − c̄ | §6.2 frames |
| decoder Engram read mixes the previous position | D1 (rows no longer per position) |
| S layers looped too | reach(L) (63 → ≈ 123; see the footnote) |
| fine restart from residuals of the old E⁰ | restart order |
| core attention not causal | causality |
| VQ sends gradient into the encoder | codebook isolation |
| shared-FFN branch held dead | alive at step 1 |
| decoder reads a future Engram row | causality |
| pooler softmax across chunk boundaries | causality, pooler init |
| gated conv: gate held at 1 (a content-blind conv) | pooler selects by content (test 13) |

---

## Design flaw found at 17:50 (in this experiment, not in v4)

**What.** `filler_sentences` drew the probe fillers from only 3,000 training documents: 6,519 sentences. At p_syn
0.5–0.7 over 5–7M tokens, each was seen on the order of ten times, and the tiny model **memorised them**. Every periodic
and final recall evaluation used items built from a *different* pool (sentences from other documents).

**How it was found.** The trio decomposition gave next-token loss on probe rows:
- 4.7 nats on validation-pool rows, against 2.9–3.0 on validation text;
- 1.31 on rows built with the training pool, so the training loss of ≈ 2.0 on a 70%-probe stream was explained.

**What it does not touch** (measured on validation text, whose documents were seen less than once):
- all of tier 1;
- k6 vs k4;
- the pooler comparison;
- codebook health;
- core off and plan swap on text;
- the within-slot top-100 / rest ratios on text;
- the engineering record.

**What it touched, and the re-measurement (18:00–18:04).**
- **Recall:** re-measured with 192 fresh items built from the *training* pool. It is **at chance in both pools for all five
  arms** (recall1 intact 1–4, swap 0–4, core off 0–3 of 192; carry3 and multi4 0). "No onset" stands.
- **Decoder Engram reads:** re-measured; see §2.1. The result turned into a finding.
- **Facts content / template ratios:** carry a caveat. The template words were repeated in every training item, so part
  of that preference may be memorisation.

**Fix for a rerun.** Sample fillers fresh per item from all 457,614 documents (or from text the arm never trains on), and
evaluate on both pools.

## Tier 2 · Learnability

**Pre-registered plan for the second pair (written at 14:42, before the 3M evals).**
- **Onset** means recall1 > 0 with plan swap ≈ 0 and core off ≈ 0.
- **If either arm shows onset by 3M real tokens:**
  - the second pair is k6 with v3's linear pool and k8 with attention + FFN, at p_syn 0.5;
  - budget: onset + 1.5M tokens.
- **If neither shows onset by 3M:**
  - the second pair is k6 with attention + FFN and k6 with the linear pool, **both at p_syn 0.7**, so the pooler
    comparison is not confounded by the data mix;
  - k8 moves to follow-up.

**The owner's convolution hypothesis (added 14:55, pre-registered before the 3M readings).**
- **A plain convolution with kernel k and stride k over the chunk *is* v3's pool.** Its weight reshapes exactly to
  W_p [m₁; …; m_k]. So the "linear" arm already tests it; it is named "linear / plain conv (v3)" below.
- **A depthwise multi-channel kernel is a restriction of it.** It has per-channel, per-slot weights and is still blind to
  content.
- **The arm that tests something new: the gated conv** (`pool = "gconv"`, `PoolGConv`).
  - It keeps the depthwise multi-channel kernel w_r (k × d).
  - It adds a per-token content gate per channel group, the owner's "multi-head": g_i = 2σ(W_g rms(m_i) + b_{h,r(i)}).
  - c₀ = W_O Σ_i real_i · g_i ⊙ w_{r(i)} ⊙ m_i, then the same SwiGLU residual and the sphere.
  - Init: g = 1, w = 1/k, W_O = I. That gives the same ĉ as the attention pooler at step 0 (test 4).
  - Difference from attention: gates are sigmoids, so selection is absolute, with no competition across the chunk.
- **Full-size cost** (k = 6, H 4): **363,033** parameters vs 493,337 for attention + FFN, so 130k cheaper. Total compute
  is 19,767,707.
- **Tier 1 for it.**
  - Tests 4 (init), 5 (alive) and 13 (selection) are extended to it.
  - Mutant "gate held at 1" (a content-blind conv) is killed by test 13: 13 of 13 mutants are now killed.
  - **Toy finding:** at an equal budget (400 steps) the gated conv puts a 0.38 share on the marked token vs attention's
    0.975 (uniform is 0.167). Copy error: 0.164 vs 0.001 vs 0.341 for the plain conv. Given 1,000 steps at lr 1e-2 it
    reaches error 0.004 and a 0.87 share.
  - So **it can select, but learns selection about 3–5× more slowly than attention** in the toy.
- **The next trio replaces the second pair.**
  - If onset appears by 3M: k6-linear, k6-gconv, k8-attn+FFN, at p_syn 0.5, budget onset + 1.5M.
  - If not: k6-attn+FFN, k6-linear, k6-gconv, **all at p_syn 0.7**, for the rest of the window.

**Branch taken (15:05): no onset by 3M.**
- At 3M both arms are at chance: recall1 1/128, and k6's plan-swap reading of 3/128 is above intact, the guessing
  signature.
- So the trio is k6 attn+FFN, k6 linear / plain conv and k6 gated conv, all at p_syn 0.7, 8M real tokens each, with a
  stop file at the window's end. The first pair (p_syn 0.5) runs to its 5M final, for the pooler telemetry and k4 vs k6.

**Expectation, written before the trio's results (15:06).**
- v3 at d 256 needed ≈ 59M tokens (1,800 steps × 32k) to lift recall1 off zero. The trio sees ≈ 8M at d 64 (actual:
  5.00 / 5.00 / 7.42M).
- If the trio ends at chance, the correct statement is **"onset not reached at this scale and budget"**, not "the concept
  route fails". The causality test shows the route carries signal end to end; the codebooks are healthy; and the pooler
  telemetry is read on its own.

**Onset rule for the trio finals (pre-registered 16:59, after gconv's 5M eval of recall1 4/128 vs swap 1/128).**
- At 192 items per family, chance is ≈ 1.3 hits.
- **Onset:** intact ≥ 8/192 (Poisson tail ≈ 0.1%) with plan swap ≤ 2 and core off ≤ 2.
- **Above chance, onset not established:** 4–7 with swap and core off ≤ 2.
- **No onset:** anything else.
- gconv's 5M reading is "first reading above chance, not onset".

**How the pooler telemetry is read (pre-registered 15:36, before any final exists).**
- **Text: top-100 / uniform < 1 with rest / uniform > 1 on some head** → the pooler down-weights frequent tokens by
  content (the owner's stop-word hypothesis). Both ≈ 1.0 means no selection learned yet.
- **Probe facts: content / uniform > template / uniform** → names and values are favoured over " The code of / is / .".
  Caveat: recall is at chance, so the model has had no reason yet to prize fact tokens. A null here means "not yet".
- **Slot profile** (uniform = 1/k). v3's trained pool skewed toward the newest slot (≈ 1.6×).
  - A skewed profile with top-100 / rest ≈ 1 means the pooler re-learned v3's position weighting.
  - A flat profile with top-100 / rest ≠ 1 means content selection.
- **Entropy / ln k = 1** means uniform.
- **Switch-offs at the final** are tier-2 "alive and used" readings (Δ val NTP): F loop off, pooler FFN off, decoder reads
  off, 1 pass, ĉ_pred off.

### 2.1 The first pair at 5M real tokens (k4 and k6, attention + FFN, p_syn 0.5)

**Retrieval: no onset.** recall1, carry3 and multi4 are at chance in every eval of both arms: ≤ 2 of 192 at the final, and
2–3 on the training pool (§ Design flaw), intact and under every switch-off. Chance is ≈ 1/150 per answer token (≈ 150 value words), so 1–2 items. This is the case the 15:06 expectation describes: onset not reached at this scale and
budget.

**The concept route is used for language, even without retrieval.** Paired standard errors over 64 validation rows:

| switch-off (Δ val NTP, nats) | k4 | k6 | reading |
|---|---|---|---|
| core off | **+0.208 ± 0.023** | **+0.124 ± 0.021** | concepts carry information the decoder cannot get from its 70 tokens |
| plan swap (another row's plans) | **+0.250 ± 0.031** | **+0.137 ± 0.023** | the plan is row-specific, not a generic bias |
| ĉ_pred off | +0.024 ± 0.003 | +0.015 ± 0.002 | the predicted next concept is used (v3 full scale: 0.013) |
| 1 pass instead of 2 | +0.008 ± 0.002 | +0.014 ± 0.003 | the second pass is used |
| F loop off | +0.022 ± 0.006 | +0.002 ± 0.001 | used at k4; practically zero at k6 (below the draft's 0.01 kill line); single seed |
| pooler FFN off | −0.008 ± 0.003 | −0.002 ± 0.001 | **the pooler FFN writes (gain 0.71 / 0.56, ‖W₂‖ 4.1 / 4.0) but does not help**; it meets the 0.01 kill line at this scale |
| encoder shared FFN off | **+0.024 ± 0.005** | **+0.014 ± 0.002** | used (gains drifted from 1 to 0.43–0.74, ‖W₂‖ 3.7); above the §3.4 kill line |
| core shared FFN off | +0.008 ± 0.002 | +0.012 ± 0.002 | used |
| all four Engram reads off | **+0.288 ± 0.020** | **+0.296 ± 0.024** | Engram is the largest single contributor, even with the tiny table |
| decoder Engram reads off | −0.006 ± 0.007 | **−0.023 ± 0.009** | **on text the decoder reads do not help; at k6 they hurt**, while writing heavily (‖W_V‖ ≈ 17). The mechanism is in the table below. |

**The decoder Engram reads are a memorisation channel** (switch-off Δ, paired SE; positive = the reads were helping):

| distribution | k4 s0 | k6 s0 | k6 attention p0.7 | k6 linear p0.7 | k6 gconv p0.7 |
|---|---|---|---|---|---|
| probe rows, **training** filler pool (memorised text) | **+0.530 ± 0.009** | **+0.474 ± 0.008** | **+0.657 ± 0.012** | **+0.675 ± 0.012** | **+0.697 ± 0.013** |
| probe rows, unseen fillers | −0.137 ± 0.007 | −0.165 ± 0.006 | −0.173 ± 0.010 | −0.146 ± 0.009 | −0.156 ± 0.010 |
| validation text | −0.006 ± 0.007 | −0.023 ± 0.009 | −0.022 ± 0.009 | −0.009 ± 0.005 | −0.008 ± 0.006 |
| *encoder* reads off, validation text (same table) | +0.161 ± 0.011 | +0.165 ± 0.015 | +0.136 ± 0.012 | +0.147 ± 0.012 | +0.162 ± 0.018 |

- **On repeated text, the decoder reads carry a large part of the memorised continuation; on new text they inject the
  wrong one.** The encoder reads on the same table generalise.
- Table collisions are therefore excluded. Distribution shift is confirmed in this specific form: a raw n-gram feature
  read near the output, trained on text that repeats.
- **Transfer condition.** Whether this fires at full scale depends on how much v3's corpus repeats (≈ 2 epochs over 280M
  tokens, so not zero). The draft's §5.2 kill criterion is the right instrument; add a switch-off reading on *repeated vs
  novel* text.
- Pre-registered discriminator (a) (a larger text effect at p_syn 0.7) did not fire: −0.022 vs −0.023 at k6 attention.
- Discriminator (b) (the reads help on probe rows) fired, but only on memorised text: they help on training-pool rows
  (+0.47 to +0.70) and hurt on unseen ones (−0.14 to −0.17).

**The pooler selects by content, and at this stage it selects structure.**
- The decomposition (`decomp_v4tiny.py`) compares token classes *within the same slot* on full chunks, so position cannot
  produce the effect.
- **Text, top-100 frequent tokens vs the rest:**
  - k4: 1.17 / 1.18 / 1.17 / 1.12 per slot;
  - k6: 1.05 / 1.20 / 1.32 / 1.13 / 1.13 / 1.13.
  - Frequent tokens get *more* weight.
- **Probe facts, names and values vs template words (" The code of", " is", "."):**
  - k4: 0.50–0.60;
  - k6: 0.33–0.45.
  - The template words get 2.2–3.0× the weight of the content at k6, and 1.7–2.0× at k4.
  - Caveat (design flaw): the template words were repeated in every training item, so part of this preference may be
    memorisation. The text ratios above are unaffected.
- **Recency.** On top of this there is a strong recency skew: the newest slot gets 0.30–0.50 vs uniform 0.17 / 0.25. Slot
  position explains 60–71% of the variance of α.
- **Reading, with the draft's own argument.** A concept keeps what pays off downstream.
  - So far the only reward reaching the pooler is next-chunk prediction, for which frequent and template tokens are the
    most useful.
  - Names and values pay off only through the far query, which no arm has learned yet.
  - So the owner's stop-word hypothesis is **not yet testable** at this budget. The present direction (structure first)
    is what that argument predicts for a model that has not learned retrieval.
  - What *is* established: the pooler is not v3's per-slot map. It weights tokens by what they are.

### 2.2 The owner's brief at tiny scale (5M tokens, single seed, k4 / k6)

| requested part | measured contribution (Δ val NTP when removed) | verdict at this scale | transfers? |
|---|---|---|---|
| concept module (core) | +0.21 / +0.12, plans row-specific (swap +0.25 / +0.14) | **used** | mechanism: yes |
| recurrence, second pass | +0.008 / +0.014 | used, small | yes, as a sign |
| F-layer loop (encoder looping) | +0.022 / +0.002 | used at k4 only; below the kill line at k6 | weak; [Loopie] expects gains only much later |
| encoder shared FFN | +0.024 / +0.014 | used | yes, as a sign |
| core shared FFN | +0.008 / +0.012 | used | yes, as a sign |
| attention pooler, attention part | content-selective (within-slot ratios ≠ 1), not a per-slot map | works as designed; selects *structure* until retrieval is learned | mechanism: yes |
| pooler FFN | writes, Δ ≈ 0 | not useful yet | open |
| decoder Engram reads | −0.006 / −0.023 on text; −0.14 to −0.17 on unseen probe text; **+0.47 to +0.70 on memorised training text** | **a memorisation channel**; the encoder's Engram reads on the same table generalise (+0.14 to +0.17) | the mechanism, yes; its size depends on how much the corpus repeats |
| two-level codebook | fine-level VE on the residual 0.72 (§1.3b bound 0.75); coarse ppl 46–62% of K, coarse CE 1.1–1.2 (k4 / k6; all five arms: 36–62%, 1.07–1.53) | healthy | the fine-level figure, yes; the coarse share, no (K / d) |
| ĉ_pred in the plan | +0.024 / +0.015 | used | yes |
| 6 vs 4 tokens per concept | +0.035 nats at 5M, 1.07× faster | see tier 3 | indicative |

**Codebooks (final, 5M).**
- Total VE is 0.846 (k4) / 0.830 (k6). The coarse level alone gives 0.459 / 0.380.
- **The fine level on the residual** explains (VE − VE_c)/(1 − VE_c) = **0.715 / 0.726** of what remains. That is
  consistent with the 1-bit-per-dimension bound ≈ 0.75 that §1.3b derives.
- The large coarse share is an artifact of K / d (64 centroids in 64 dimensions vs 1,024 in 256) and does not transfer;
  the draft's "≤ 5%" estimate is for the full ratio.
- Coarse perplexity is 39.7 / 29.5 of 64 (62% / 46%, above the 25% threshold).
- CE_coarse is 1.10 / 1.22 vs ln K = 4.16: the next coarse code is very predictable, likely template structure.

## Tier 3 · Arm comparison (single seed, indicative)

**Comparison rule (written 16:28, before the trio's numbers exist).**
- Pooler comparisons are made **within the trio only** (k6 attention + FFN / linear / gated conv, all at p_syn 0.7, same
  seed, same data stream), at matched real tokens from the eval table.
- k6_attnffn_s0 (p_syn 0.5) vs k6_linear_p07 is a data-mix comparison, not a pooler one, and is not made.
- The informative cross-mix pair is **k6_attnffn_s0 vs k6_attnffn_p07**: same model, 0.5 vs 0.7 synthetic. It shows what
  the mix buys.
- Training `ntp` depends on the mix; only validation NTP is compared across arms.
- Finals are not schedule-matched: gconv was stopped at 7.42M (lr ≈ 0.11 of peak), the late pair at 5.00M (lr ≈ 0.38 of
  peak). Matched-token eval rows are the comparison.

**Matched-token validation NTP (same k, seed, rows and lr at each row; only the pooler differs).**

| real tokens | attention + FFN | linear / plain conv (v3) | gated conv (owner) | coarse ppl (attention / linear / gconv) |
|---|---|---|---|---|
| 1M | 3.268 | 3.244 | 3.249 | 22.9 / **6.8** / 17.6 |
| 2M | 3.122 | 3.124 | 3.097 | 27.7 / **17.8** / 25.0 |
| 3M | 3.137 | 3.156 | 3.051 | 29.8 / **15.5** / 27.0 |
| 4M | 3.059 | 3.027 | 3.050 | 25.6 / **16.1** / 26.6 |

- **Language: tied within checkpoint wobble**, per the pre-registered rule. gconv's 2M–3M lead did not hold at 4M.
- **The one consistent pooler difference is in the codebook.** v3's per-slot pool gives the lowest coarse perplexity at
  every row (6.8–17.8 vs 23–30): its concepts collapse onto fewer coarse codes. This is a second argument against it,
  besides content-blindness.
- **Retrieval:** at chance in all three at every row. gconv's 5M reading (4/128 vs swap 1/128) went back to 0 at 6M: a
  blip.
- **What the mix buys** (k6 attention, p_syn 0.5 vs 0.7, same model): text val is worse by 0.15 / 0.25 / 0.29 nats at
  1M / 2M / 3M. That is consistent with seeing fewer text tokens. It bought no recall.
- **k6 vs k4** (p_syn 0.5, matched rows): +0.015 / +0.012 / +0.020 / +0.028 / +0.035 nats at 1M–5M (k6 3.116 / 2.871 /
  2.847 / 2.780 / 2.812; k4 3.101 / 2.859 / 2.827 / 2.752 / 2.777).
  - The gap **widens steadily** through the schedule and reaches the draft's +0.03 gate at ≈ 4–5M tokens.
  - Speed: 1.07× (inside the draft's expected 1.06–1.10).
  - Read at this scale, compression 6 costs ≈ 0.03 nats and is not yet paid back. The draft's gate at step 4,050 decides at
    full scale. No paired SE exists for this comparison (different packing per k).

**Decision rules written before the late pair's 4M row (≈ 17:20).**
- **gconv vs the others: matched eval rows only, no SE** (its final is at a different token count).
  - gconv led at 2M (−0.025) and 3M (−0.09).
  - If it also leads at 4M: "gconv ahead at three consecutive matched rows (single seed)".
  - Otherwise: "tied within checkpoint wobble (≈ 0.03–0.1 nats)".
- **Attention vs linear at the finals:** both stop at 17:35 at 5.00M (step 3,500) with near-equal speed, so a paired SE over the same
  64 rows is valid.

**Finals (17:43; not token-matched for gconv).**

| arm | real tokens | lr at end | val NTP | recall1 / carry3 / multi4 (of 192), validation pool | coarse ppl / 64 | text: top-100 / rest by slot | facts: content / template by slot | position share of α |
|---|---|---|---|---|---|---|---|---|
| attention + FFN p0.7 | 5.00M | ≈ 0.38 of peak | 3.001 | 5 / 0 / 0 | 28.8 | 1.05–1.19 | 0.50–0.57 | 0.57 / 0.62 |
| linear / plain conv p0.7 | 5.00M | ≈ 0.38 of peak | 2.977 | 2 / 0 / 0 | 22.8 | — (no weights) | — | — |
| gated conv p0.7 | 7.42M | ≈ 0.11 of peak | 2.913 | 1 / 0 / 0 | 26.9 | 1.10–1.25 (gate share) | 0.45–0.52 (gate share) | 0.06 / 0.08 |

- **Attention vs linear at the matched finals, paired over the same 64 rows:** attention − linear = **+0.017 ± 0.018**
  nats, so **tied** (< 1 SE). This confirms the matched-row reading.
- **gconv's lower final val (2.913) comes from more tokens and a fully decayed schedule.** At matched rows it tied (4M:
  3.050 vs 3.059 / 3.027).
- **gconv's gate share has almost no position component** (6–8% of variance vs 57–62% for attention). Position lives in
  its slot weights w_r instead, so its telemetry is not comparable to attention's on position. Its content preference
  is the same as attention's: frequent and template tokens up.
- **Onset rule:** none of the three shows onset (§ Design flaw: checked on both filler pools).

---

## What this means for draft_v4 (recommendations for the owner; the draft is not edited here)

| draft section | finding here | recommendation |
|---|---|---|
| §5.2 decoder Engram reads | a memorisation channel: +0.47 to +0.70 on memorised text, −0.14 to −0.17 on unseen probe text, ≈ 0 to −0.02 on text; the encoder reads generalise | make the per-read switch-off and a no-reads arm **mandatory**, not "if time"; add a switch-off reading on repeated vs novel text to the §5.2 kill criterion |
| §3.5.4 pooler FFN | writes (gain 0.56–0.71) but Δ ≈ 0 | turn it into a priced arm with the 0.01-nat kill line at step 4,050, or drop it and return 294,913 parameters (then §4.1's option B, 5 × 2, fits under 20M again) |
| §3.5.4 pooler alternatives | the owner's gated conv ties with attention on language and is 130k cheaper (363,033); a plain conv is v3's pool | add the gated conv as a priced arm beside attention |
| §3.5.1 why replace v3's pool | the per-slot pool collapses the coarse code (lowest perplexity at every row) besides being content-blind | add it as a second argument |
| §8.4 / §9.1 pooler telemetry | slot-averaged ratios mix position and content; this experiment needed the within-slot split to read them | log within-slot class ratios and the position share of variance |
| §3.3 F loop | +0.022 at k4, +0.002 at k6 (single seed) | keep the no-loop arm (§8.3 arm 3); the effect may depend on k |
| §3.5.5 k = 6 gate | k6 − k4 widens to ≈ +0.03 by 4–5M at tiny scale; 1.07× speed | keep the gate as written; the step-4,050 reading is the decision |
| §6 / §1.3b codebook | fine-level VE on the residual 0.72 vs the 0.75 bound; coarse code used and predictable | no change |

---

## Engineering record (and what the experiment cost v3)

| item | result |
|---|---|
| doc-shift mask memo | It was off in `EncoderV4` (≈ 90 masks recomputed per forward). Now on; arithmetic identical (test 11 still bit-exact). |
| block-local window attention (v3's `speed_v3.py`, block 60) | +34% throughput (330 → 442 real tok/s on one thread); parity < 1e-10 (test 14). |
| `torch.compile` (inductor, CPU) | 1.02×, loss parity 4e-6. Dropped. |
| batch | B 4 × T 480 (B 16 vs 8 gave identical per-token speed; B 4 halves memory). |
| efficiency cores (`taskpolicy -b`) | v3 unaffected (3,000 / 3,042 tok/s vs a clean control of 2,907), but slow arms: 580 real tok/s (B 8, 3 threads, no block) and 557 (B 4, 1 thread, block 60). |
| P-cores, owner-approved (3–4 h, slower v3 accepted) | Two arms in parallel (2 threads each, ≈ 1,400 real tok/s per arm, RSS ≈ 1.35 GB each), then three (≈ 760–890 tok/s each). Total RSS 1.8–2.6 GB. Swap stayed flat at 5.9–6.1 GB. No new v3 guard event at any check. |
| v3 heartbeats (tok/s) | **Reference, no arms (E-core runs or idle), 13:52–14:23:** 2,907–3,229. **Two arms on P-cores, 14:30–15:01:** 2,474, 2,294, 2,333, 2,246, 2,249, 2,121 (≈ −27%). **Three arms, 15:06–16:58:** 2,472, 2,074, 2,088, 2,101, 2,075, 2,108, 2,077, 2,172, 1,954, 1,892, 1,922, 1,948, 1,935, 1,958, 1,963, 2,040, 2,160, 2,036, 1,871, 2,173 (to 17:32; ≈ −35%). v3's hourly eval took 101–113 s instead of 86 s. |
| v3 after the arms stopped | 17:41 1,920; 17:48 1,979 (finals running); **17:57 1,563** (five 1-thread analysis jobs on the P-cores, the worst reading of the window); 18:05 1,728 (re-measurement jobs). All experiment processes ended at 18:04. **First clean heartbeat, 18:10: 2,900 tok/s**, back to the reference (2,907). |
| what it cost v3 | v3's schedule is time-based and it ends at 48 h regardless. ≈ 1,050 tok/s lost over ≈ 3.65 h (14:25–18:04, including the analysis tail) ≈ **13–14M tokens ≈ 400–420 optimizer steps ≈ 2.5% of the run**, unrecoverable. v3's arithmetic is unaffected (deterministic batches; no guard event). Its quality kept improving through the window (hourly val 2.513 / 2.507 / 2.504 / 2.500 / 2.497, a new best each time). |
| correction | The first "10–13% slowdown" reading (advisor row 4) was against a stale baseline. Against the clean control it was ≈ 3–6%. |
