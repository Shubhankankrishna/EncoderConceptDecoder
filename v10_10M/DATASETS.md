> Audit update, 11 October 2026: the current math corpus contains upstream serialized byte-literal wrappers. The original format handling below did not remove them. See [audit/REPORT.md](audit/REPORT.md) and the separate corrected preparation entry point before using math loss as evidence of reasoning.

# Data selection and evaluation

This run mixes public educational text, school-level mathematics and 9×9 Sudoku. It contains no GraphNav task, four-token answer template designed around compression, or task generated to favor this model. The tokenizer is fitted only on the selected training partitions. Digits are isolated before BPE, so a Sudoku answer contains 81 digit tokens, regardless of the model's 8→2 pooling rule.

## Selected sources

| Source | Why selected | Training material in this run |
|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Real educational prose with broad vocabulary and longer dependencies; materially harder than TinyStories-only pretraining | 26,998 documents; 35,167,033 freshly tokenized tokens |
| [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset) | Independently defined arithmetic/algebra/comparison tasks with checkable answers | 239,904 sampled questions, about 10.01M tokens |
| [Sudoku-Extreme corpus](https://huggingface.co/datasets/sapientinc/sudoku-extreme) | Public native easy and hard puzzles; full-grid constraint checking, 81-cell answer | 50,000 puzzles: 12,802 native easy-source and 37,198 hard-source instances |

The mathematics subsets are multi-operation addition/subtraction, multiplication, one-variable linear equations and sorting. Samples are selected across complete train parquet files, not only the first easy block. Their median prompt+answer length is 40 tokens and median answer length is 8, with prompt/answer boundaries distributed across all eight possible offsets. Short numeric answers are legitimate tasks; they are never presented as long-horizon planning evidence.

Sudoku is serialized as a natural task marker, an 81-character board, a solution marker and the full 81-digit solution. Total length is 171 tokens with this tokenizer. No special padding is inserted to make it a pooling multiple. The easier subset comes from upstream `puzzles0_kaggle`, `puzzles1_unbiased`, and `puzzles2_17_clue`; “easy” is an upstream source label, not a guarantee that a small autoregressive model can solve each puzzle. Original ratings are retained. No solution-derived extra clues are inserted.

## Mixture and realistic difficulty

At each microbatch, sample 65% text, 20% mathematics and 15% Sudoku. These are **batch/loss sampling probabilities**, not token fractions. Within Sudoku, the native easy-source probability anneals from 90% to 60% during the 24-hour run; hard-source exposure rises from 10% to 40%. Actual supervised token counts are logged separately. All sources are present from the beginning.

Educational language and basic symbolic structure are plausible learning goals for a 5.47M dense backbone with sparse lexical memory. Robust long-form generation, general-purpose reasoning and high Sudoku-Extreme accuracy are not warranted expectations after one laptop day. Hard Sudoku is a stretch evaluation, not the sole training distribution or a promised result. Underperformance stays in the report; examples are not replaced with easier custom tasks after seeing results.

The [official TRM repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) identifies Sudoku-Extreme, Maze-Hard and ARC tasks and uses a different recurrent architecture, supervision and accelerator setup. Its reported accuracy and runtime are not forecasts for this autoregressive V10 implementation. [Generative Recursive Reasoning](https://arxiv.org/abs/2605.19376) provides further motivation for Sudoku and other constraint tasks, but its stochastic multi-trajectory method is not reproduced by merely adding a K=8 code table.

Maze-Hard/ARC are deferred: they add another serialization, long contexts and substantially different evaluation demands. Pure TinyStories was rejected as the main text corpus because the user specifically requested more demanding data. Raw web mixtures with little quality control and competition-math-only corpora were rejected as poor uses of the limited training budget. The selected mixture is a reasoned starting choice, not an empirically proven globally best dataset.

## Provenance, reuse and splitting

FineWeb-Edu is reused from the existing `v7_tiny/data/fineweb_edu.jsonl.gz` cache. It was originally the first 30,000 eligible documents from `sample-10BT`; this is a small, order-biased sample, not a representative sweep of the full corpus. Existing document-level train/dev/test assignments and lexical near-dedup metadata are preserved, and duplicate source hashes are removed. This avoids another multi-gigabyte download. V10 weights start from scratch; no V9 weights are imported. These are not newly independent external held-out documents relative to all previous repository experiments.

Mathematics parquet data is pinned to the original dataset repository revision `ee37b6c8f92cceb4e32abae57ed6fd35a0863845`. Each of four train files supplies up to 60,000 uniformly sampled rows. Each official test file supplies 2,000 rows; the selected test questions are partitioned deterministically by hash into development and sealed test. Exact normalized duplicate questions are removed, with selected held-out questions taking precedence over training examples. Algebraically equivalent or paraphrased problems may remain; no semantic deduplication is claimed.

Sudoku bytes are pinned to revision `58942f96baeb572ca3127e2a9e9c70f330783d6b`. The first 50,000 rows of the upstream shuffled training CSV and 2,048 official test rows are streamed without downloading the entire corpus. The upstream card describes equivalent-puzzle separation between official train/test. Hashing divides the selected official test sample into development and sealed test before training. Equivalence between those two derived held-out subsets is not independently established. There is no online puzzle augmentation that could cross partitions.

`data/manifest.json` records source URLs, revisions, payload hashes, vocabulary hash, counts and exclusions. `research/data_audit.json` verifies stored hashes, shapes, vocabulary bounds, zero cross-split identity overlaps, strict validity of every stored Sudoku solution and answer-boundary distributions. Sealed examples are processed for integrity and tokenizer encoding, but are not used for fitting or development evaluation.

## Metrics that cannot be satisfied by short-format shortcuts

* Text: held-out token NLL/perplexity, with an explicit no-concepts ablation. A lower concept MSE cannot replace this metric.
* Mathematics: answer-only teacher-forced NLL plus unconstrained greedy exact whole-answer matching with EOS. A correct prefix or an unterminated answer fails.
* Sudoku: answer-only teacher-forced NLL plus unconstrained greedy full-board validity: exactly 81 digits, original clues preserved, all nine rows/columns/boxes contain 1–9, and EOS. Copying the input clues or repeating a token fails. No solver repairs or constrained decoding are used.
* Concept diagnostics: target coordinate standard deviation, selector code counts and token NLL change when concept guidance is removed. An unused latent path remains a failure mode even if the encoder/decoder language path learns.

Periodic development generation uses only 8 math examples and 2 examples per Sudoku subset to bound overhead. These counts are explicitly reported and are too small for a strong accuracy claim. The separate evaluation CLI uses larger sets; sealed-test evaluation requires an explicit command after development choices are fixed. No sealed score is automatically used to select checkpoints. A proper final research report still requires a matched non-concept baseline, repeated seeds and enough complete-answer trials.

## What the v9 audit supports

`v9_tiny/frozen/code/data.py` uses four-token edge records, four-token lookup/composition/binding answers, and four-token walk/termination chunks. Some GraphNav tasks are long sequences; it would be inaccurate to describe the whole dataset as only four tokens long. The concern is structural alignment with four-token compression and templated outputs.

The existing `runs_q5_12h/ASSESSMENT.md` records a GRPO collapse to malformed repeated `DONE` chunks, zero success after its first fit, and no convincing MCTS superiority in the reduced comparison. That is evidence of collapse and a confounded comparison, not proof of every proposed reward-hacking mechanism. This implementation uses supervised pretraining and independent complete-answer metrics, without importing those reward functions or declaring success from an auxiliary loss.

## Licenses and attribution

FineWeb-Edu's card lists ODC-By and retains source-page attribution in metadata. The mathematics project is Apache-2.0; the source repository and paper remain credited here. The Sudoku card links its community/tdoku origins but does not provide a clear single dataset license in the inspected metadata; the local experiment preserves provenance and does not assert redistribution rights. Large raw/tokenized data is excluded by `.gitignore`.
