# V10.3 tiny — test-bed specification

**Written 12 October 2026, before any training result was read.** This is the executable specification of the V10.3 design ([V10_3_RCDL_Architecture.md](../V10_3_RCDL_Architecture.md) §1–§2) at about one-twentieth of the 10M budget. Its purpose is to test the V10.3 changes and the sleep layer cheaply before a sub-10M model is built. Tiny-model numbers do not transfer quantitatively to 10M; they can falsify mechanisms and catch bugs.

## Size

The dense limit is 600k parameters. As in every V10 version, the sparse Engram table is counted separately.

| Arm | Dense | Training-only dense | Deployed dense | Engram (sparse) |
|---|---:|---:|---:|---:|
| C0 | 311,704 | 0 | 311,704 | 1,050,080 |
| A | 484,782 | 41,720 | 443,062 | 1,050,080 |
| X8 | 503,428 | 29,064 | 474,364 | 1,050,080 |
| X4 | 502,912 | 29,064 | 473,848 | 1,050,080 |

Width 56, 4 heads of 14, vocabulary 1,024, context 256. E1 = 2 blocks with Engram after block 1, E2 = 1 two-stream block run twice, exit reader, E3 = 1 block (the only full-attention encoder block; the others use a 64-token window). Shared 8→2 pooler. Concept core = 2 blocks × 2 visits, code width 16, K ∈ {4, 8}. Soft PQ head 4 segments × 16 words × 14. Decoder = 2 blocks with encoder-depth CRC. Engram: orders 2 and 3, two hashes each, about 16k rows per hash, 16 dimensions.

## Arms

| Arm | What it is | Question |
|---|---|---|
| C0 | Same encoder, Engram and decoder blocks with E1/E3 CRC; no pooler, concept core, PQ or concept losses | Does the concept branch earn its cost? C0 is 35% smaller and faster; it is compared at equal updates, with its throughput reported separately |
| A | The unmodified V10.1 model class (`v10_1.model.V101`) and V10.1 losses: latent MSE, codeword fitting, chunk head, hard-argmin exploration with selector CE; V10.1 optimizer grouping (codebook decayed) | The baseline V10.3 must beat |
| X8 | V10.3: V10.2 RF model (forecast-only token-ID head, concept-history reader) with a full learned map in place of the per-coordinate forecast gain, codebook exempt from weight decay, and mixture-likelihood exploration with the selector as the prior; K = 8 | Do the V10.3 changes work? |
| X4 | X8 with K = 4 | Is K = 4 enough? |

Not in the tiny version: the visit-1 state or visit delta as a decoder input. The V10.3 design measured that they add at most 0.017 FVU beyond U2 (§1.2).

## Training protocol

All arms use the same seed and the same step-based schedule: progress = step / max_steps drives both the data-family mixture and the learning rate. Every arm with the same seed therefore sees the identical document stream. Batch 16, context 256, AdamW lr 1e-3 with betas (0.9, 0.95), warmup 200 updates, cosine to 10%, dense clip 1.0. CPU SparseAdam Engram at 5× LR with its own clip at 1.0. Candidate updates every fourth step for A and X. FP32 on Metal. Data: the clean V10.1 corpus re-tokenized to 1,024 tokens; the algebra module is held out of pretraining as the new phase for the sleep tests. Tiny-model NLLs are not comparable with V10.1's, because the vocabulary differs.

## Predeclared go/no-go for building the sub-10M V10.3

Panels: text 256 documents, 128 problems per pretraining math module, 64 boards per Sudoku family (dev split). Intervals are paired document bootstraps.

**G1 — V10.3 beats V10.1's recipe.** The better X arm has text dev NLL ≤ A's at equal updates, with the 95% interval of X − A excluding zero, and its math exact-match is not below A's.

**G2 — the concept branch is not a net cost.** The better X arm's text NLL is within 0.02 nats of C0's at equal updates, or better. C0's result at equal wall time is reported alongside.

**G3 — the mechanisms do what they claim.**
- N2: the forecast that reaches the decoder, after its learned map, has at least 25% of its target's variance (V10.1 at 10M: 3%).
- N3: X's codebook RMS ends within 20% of its initial value.
- N5: X's held-out selector CE is below the entropy of the held-out winner distribution, and the held-out mixture log-likelihood exceeds the committed-code log-likelihood.

**G4 — sleep layer.** After learning the held-out algebra module, gated LoRA keeps text NLL and the two dissimilar math modules (multiplication, sorting) at least as well as plain LoRA, by a margin of 0.01 nats per token or 2 accuracy points. The add/subtract module, the related skill, is reported separately. G4 does not block the 10M build. It decides whether the sleep layer enters the 10M specification or remains an arm.

Promotion to a sub-10M build requires G1, G2 and G3. If G1 fails, V10.3's changes are not supported at this scale. If G2 fails, the concept branch costs more than it returns. Both are reported as failures, not reinterpreted.

## Additional rules fixed at launch (15:46 IST, before any result)

- **Budget.** Each arm runs 6,000 updates (seed 1). If at step 6,000 the arms' dev curves are still falling steeply and overlap within their noise, the report says the budget was too short to differentiate the arms. It does not say the changes had no effect.
- **Device.** To fit the wall budget, X8 and A train on CPU (three threads each) while X4 and C0 train on Metal, concurrently. G1 (X8 against A) is a same-device comparison. Comparisons across devices (X4 against A, X8 against C0) are confounded by floating-point differences: CPU and Metal agree to 5×10⁻⁷ per gradient, but training trajectories diverge. The second-seed runs (C0 and the better X arm) train together on CPU, so G2 also gets one same-device pair.
- **N2 supplements.** The 25% threshold stays as declared. The report also gives `var(fusion_map(F)) / var(E3 state)` at the same rows, the stream the forecast is added to, and the trajectory of the map's scale, so that a miss can be explained either way.

## Data-bias audit (16:10 IST, before any result; `audit_data.py` → `data_audit.json`)

V9's GraphNav responses had a designed four-token step structure that matched V9's four-token concept chunks (review_v9.md:113). That entangled the architecture with the data. The audit checks whether the tiny corpus has structure aligned to V10's 8-token groups.

| Family | Answer start mod 8 | Periodic tokens (max share of one residue, header excluded; uniform = 0.125) | Answer length (tokens) | Verdict |
|---|---|---|---|---|
| Text | 97% of documents are cropped at a uniform random offset | 0.134–0.139 | — | No alignment |
| Math (3 modules) | Spread over all 8 residues: χ² 13.0 train / 16.5 dev, df 7 (p = 0.001 cutoff 24.3); 29–34 distinct starts | ≤ 0.18 | median 8–10, 37–51% ≤ 8 | No alignment |
| Newmath (algebra) | Max share 0.144; χ² 170 train (mild unevenness, no single residue) / 12.6 dev | ≤ 0.29 for content tokens | 3–5 | No alignment |
| Sudoku easy/hard | **Constant**: every record is 171 tokens, and the answer always starts at position 89 (residue 1) | Digits 0.131–0.136; the newline and "Solution" sit at fixed positions | 82 | **Fixed layout.** Groups of 8 do not match rows (9) or boxes (3×3), so this is not matched structure, but every board maps cells to groups identically |

The data were built for V10.1 from DeepMind Mathematics modules, public Sudoku sets and a text corpus. The only V10.3 change is re-tokenization. The format has no padding to multiples of 8: sequence length mod 8 is uniform for math and newmath.

Predeclared consequences:
- G1 and G2 rest on text NLL, where group boundaries fall at random content positions, and on math exact-match, where answer starts are spread across residues. Neither depends on Sudoku.
- No Sudoku result is cited as evidence that the concept path helps unless it survives the grid-shift control. `evaluate.py` re-scores each Sudoku board with four extra newline tokens after the header, moving every cell by half a group. Every arm, C0 included, sees the same out-of-distribution shift. If the concept arms degrade more than C0, their Sudoku advantage is partly alignment.
- V10.1 and V10.2 used the same fixed Sudoku layout, so their Sudoku-only claims carry the same caveat.

## Selection rules (16:15 IST, before any result)

- The sleep tests use **X8_s1**, the user's primary K, chosen now so they do not depend on any evaluation.
- The second-seed X arm is whichever of X8_s1 and X4_s1 has the lower text NLL in its step-6,000 `quick_eval.json` entry. The full evaluation is not used for this choice.
- If the clock runs out, work is dropped in this order: consolidation, then second seeds, then sleep. The G1/G2/G3 evaluations are never dropped. The report says what was dropped.

## Clarifications (16:25 IST; only C0_s1 has been evaluated, and no A or X result exists)

- **G1 math clause.** At this scale exact match is at the floor: C0_s1 solved 1 of 96 pretraining-module problems. "Math exact-match not below A's" is therefore assessed with a paired bootstrap over per-problem correctness on identical problems. The clause fails only if A exceeds X with a 95% interval that excludes zero. Raw counts are reported.
- **N5 scope.** "Held-out" means the dev split of the four pretraining families. Newmath was never trained, so its exploration numbers are reported but not scored.
- **G4 reading guards.** G4 is reported together with `plain − base` and `gated − base`.
  - If plain LoRA forgets less than 0.01 nats/token on all three retention sets, G4 is "not testable at this adaptation budget".
  - If gated LoRA gains less than half of plain's improvement on newmath, the gate is over-closed and the retention comparison is vacuous.
  - Newmath prompts use vocabulary ("Solve", "=", letters) that no pretraining family uses. That makes prompt-row gating the favourable case for LSL, so answer-row open rates on the related add/subtract module are reported specifically.
- **Grid-shift reading.** C0 has no groups, yet degrades 0.17 nats under the 4-token Sudoku shift: it learned the fixed layout from the BOS anchor. The test for the concept arms is the difference in degradation from C0, not the absolute value.

## Run-time rules (16:45 IST, before any A/X or sleep result)

- **Sleep test.** `sleep.json` is saved after every condition. If the time already used projects the RBF condition to finish after 18:10, the run is stopped before consolidation and the report says consolidation was not run. Consolidation is the first item dropped.
- **Report deadline.** At 18:20, if the second seeds are below step 5,000, the report is written from seed 1 and the seeds are listed as unfinished.
- **Seed choice.** If the best X by full evaluation (`compare.py`) differs from the quick-eval choice used for seed 2 (`seed2_choice.txt`), the report says so. Both rules were predeclared.
- **N2 reading.** X has no latent-MSE term, so for X the ratio `var(map(F)) / var(next pooled state)` compares the scales of two different spaces. The report keeps the declared 25% rule, but gives it with the `var(map(F)) / var(E3)` supplement, and uses the `aligned_off` intervention as the functional test of whether the forecast reaches the decoder. The raw-F FVU column is meaningful only for A, which is trained toward the pooled state.

## Dropped item (17:05 IST)

- **Equal-wall-time C0 was not run.** The machine had no spare memory (swap 4.8/6.1 GB) with the sleep test and both seeds running. At equal updates X8 already trails C0 on text. Solo, C0 takes 0.160 s/step against X8's 0.261, so at equal wall time C0 would get about 1.6× as many updates and the gap could only widen. The report gives the throughput ratio and C0_s1's own quick-eval slope instead.
