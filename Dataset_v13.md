# Dataset_v13: a capacity-calibrated reasoning and continual-learning specification

**Date:** 14 October 2026, India time. **Status:** researched specification only; no corpus has been generated and no new training has been run for this document. The difficulty ranges, mixture and acceptance thresholds below are proposed experimental choices, not measured results. The user explicitly requested specification only and an architecture-neutral dataset. **Reviewed and extended on 14 October 2026 (§14):** difficulty ladders replace admission bands (§7.4), with published small-model anchors (§3), inductive-bias tags (§5.7) and source checks. It remains a specification only.

## 1. Recommendation and scope

Build a small, procedurally generated, exactly verifiable suite with **four equally important reasoning families**: grid-rule induction, short program/algorithm execution, state tracking, and explicit-goal planning. Add a small simple-English language stream as a guard against destroying ordinary language modelling. Give each reasoning family an easy prerequisite tier, a substantial working tier, and separate compositional and length-generalization challenges. Then expose the trained models to an ordered stream of new skills, interfering skills, changed representations and returning skills.

**Do not certify difficulty from parameter count or from how human-friendly a puzzle looks.** Certify a tier using a plain full-attention transformer trained from scratch at the actual parameter and compute budget. A useful working tier is one on which that baseline has learned substantial held-out competence while retaining room to improve. An all-zero or all-perfect panel is a poor primary comparison, although both can be useful diagnostics. Since the review in §14, difficulty ladders replace tier admission (§7.4). The references set each ladder's range, and models are compared over the whole ladder and at the level where they cross 50%, so no single guess about difficulty decides the result.

Use a **sub-5M primary track** and a separately calibrated **sub-10M track**. Do not pool their scores. A 24-hour budget is defined explicitly in §11; it is not a promise that the entire multi-model study fits into one day on the current laptop.

This suite would answer: can a small model acquire, combine and retain controlled skills under a fixed resource budget? It would not establish general intelligence, frontier language ability, or an official ARC-AGI score. ARC-like induction belongs in the suite; unmodified ARC belongs in a separately reported external evaluation.

### The architecture-neutrality contract

1. Freeze generators, difficulty decisions, splits, metrics and primary weights **before reading v13 results**. Calibration uses the reference transformer, plus trivial controls and exact solvers. Record failed calibration attempts too.
2. No generation rule may depend on concept width, codebook size, pooling factor, decoder receptive field, recurrence count, bank kernel, or adapter gate. No examples are selected because v13 solves them or because a concept ablation fails on them.
3. Every model receives the same observable prompt, training targets, tokenizer, example order and permitted feedback. Latent programs, object masks, optimal plans, future events, subgoal boundaries and phase labels are evaluator metadata, except when the same explicitly labelled supervision track supplies them to all models.
4. Primary scoring gives equal weight to the four reasoning families and equal weight to their registered task types. Short/local tasks remain present. Long-range recall and concept interventions are secondary diagnostics, not a route to winning the main score by construction.
5. Architecture-specific memory is allowed as a method under test, with its bytes and runtime counted. Competitors receive the same storage allowance and access to past observations. An oracle task router or unlimited regeneration of old examples is not a fair continual-learning baseline.
6. Report both **equal observed data** and **equal wall-clock cost**. Shared weights are counted once as parameters, but all recurrent passes, retrieval, consolidation, auxiliary training and search count toward computation.
7. Freeze the test suite; if the baseline is too weak or too strong, revise a versioned calibration specification, not the test after seeing which architecture wins.

Architecture neutrality cannot mean that all problems favour every inductive bias equally. It means selecting a diverse suite independently of the proposed architecture, exposing the tradeoffs, and declining to declare a universal winner from one tailored task. A ranking also holds only for its training protocol. Amos et al. found that "random initialization leads to gross overestimation of the differences between architectures", and that with self-pretraining on the task data, vanilla transformers matched S4 on Long Range Arena ([paper](https://arxiv.org/abs/2310.02980)). A result on this suite is a result for training from scratch on this stream at this budget.

## 2. What the repository already tells us

The dataset concern is justified, but it does not make all earlier evaluations wrong. Several different limitations have been mixed together:

| Local observation | What it supports | What it does not support |
|---|---|---|
| V12's main models were about **1.06–1.08M** parameters, compared at 8,000 updates | They tested a much smaller capacity than the requested 5–10M range | Failure is not evidence that every sub-10M model fails |
| Recorded pinpad competence was TF7 0.281, ECD 0.230, ECDe 0.164 | The base policies were weak, making downstream control difficult to interpret | It was not requested-goal accuracy: the base judge accepted completion of **any** training task |
| ECDe lost to the same ECD topology on text/math prediction while its concept route became more useful | Objective interference or optimization is a real alternative explanation | Changing data alone is not known to remove NCP's cost |
| The longer TF7 controller run reached pair-switch F1 0.697, versus 0.358 random placement; the corresponding ECDe run remained at zero | The short controller sweep was budget-sensitive | A short null result does not prove the mechanism impossible |
| V10.3's RBF adaptation lowered new-math NLL substantially, but its reported exact match was **0/64** | Loss improvement and solution competence must be measured separately | Retaining a loss gain is not yet retaining a usable new skill |
| Related arithmetic suffered under earlier sleep methods; a token-weighted harm screen was dominated by text | Continual evaluation needs related-skill retention and balanced per-task accounting | Protection of dissimilar domains is not enough |

Sources: [V12 manifest](v12_tiny/runs/TF7_s1/manifest.json), [V12 results](v12_tiny/RESULTS.md), [rollout judge](v12_tiny/rollout.py), [long TF controller result](v12_tiny/runs/TF7_s1_8k/meta/L5_a0.3_s1_n6000.json), [long ECDe controller result](v12_tiny/runs/ECDe_s1_8k/meta/concept_a0.3_s1_n6000.json), [V10.3 report §8](V10_3_tiny_REPORT.md).

The current [v13 architecture notes](v13_NOTES.md), §12, propose largely retaining pinpad/text/math and adding recall, with an estimated multi-run duration exceeding a day. **This document proposes a replacement experimental dataset/protocol; it does not silently modify that architecture document.** Its published-model claims were not re-audited here. Before implementation, reconcile its data mix, tokenizer, compute schedule and gates with this specification.

## 3. Research: which existing datasets actually help?

The Google/DeepMind work most closely matching the request is **Mathematics Dataset, CLRS/CLRS-Text, Neural Networks and the Chomsky Hierarchy, and Procedurally Generated Matrices**. They address different parts of the problem; none establishes that its entire benchmark is an appropriate 24-hour curriculum for our small causal model.

| Source and direct evidence | What to borrow | Limitation for this experiment |
|---|---|---|
| **DeepMind Mathematics**, Saxton et al., 2019. Procedural question/answer generation, with easy/medium/hard training partitions and generalization tests. [Paper](https://arxiv.org/abs/1904.01557), [official generator](https://github.com/google-deepmind/mathematics_dataset) | Controlled operand size, expression structure, verifiable answers; a small arithmetic/algebra subset | “School-level” is not a model-capacity guarantee. Do not import the complete hard mixture or assume its labels match our model |
| **DeepMind CLRS**, Veličković et al., 2022. Thirty algorithms with input/output and optional intermediate hints; the canonical setup includes size-16 training and size-64 testing. [Paper](https://arxiv.org/abs/2205.15659), [official code](https://github.com/google-deepmind/clrs) | Exact solvers, controllable input sizes, separate interpolation and extrapolation | The graph-based baseline and hint supervision differ from a causal token model. Our adaptation must be named separately |
| **DeepMind CLRS-Text**, Markeeva et al., 2024. Textual algorithm trajectories and exact final-array evaluation; published training includes Gemma 2B. [Full paper](https://arxiv.org/html/2406.04229v1) | Text adapters and optional machine-checked traces; small arrays and graphs | A 2B experiment is not evidence of success below 10M. Start with a few algorithms and calibrate |
| **DeepMind Chomsky hierarchy study**, Delétang et al., 2023. Fifteen formal tasks; small networks, including five-block width-64 transformers; separate in-range learning from length generalization. [Full paper](https://arxiv.org/html/2207.02098v3), [code](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy) | Compact alphabets and independently controlled length; copy/reverse/count as prerequisite checks | Its generalization failures show why excellent training fit does not certify extrapolation. Do not treat its findings as a theorem about every transformer |
| **DeepMind PGM**, Barrett et al., 2018. Procedural visual relations with multiple generalization regimes. [Paper](https://arxiv.org/abs/1807.04225), [official data](https://github.com/google-deepmind/abstract-reasoning-matrices) | Separate rule, attribute and combination shifts | Native image perception adds a separate difficulty. A symbolic adaptation is not PGM. Official data is marked non-commercial research only |
| **Google CFQ / distribution-based compositional assessment**, Keysers et al., 2020. Splits distinguish familiar atoms from unfamiliar combinations. [Paper](https://arxiv.org/abs/1912.09713) | Hold out combinations while keeping primitive coverage | Borrow the split principle, not the full semantic-parsing task. Our simple split is not a replication of their optimized divergence construction |
| **TinyStories**, Eldan & Li, 2023, Microsoft Research. Reports coherent language from models below 10M with restricted story vocabulary. [Paper](https://arxiv.org/abs/2305.07759), [official dataset](https://huggingface.co/datasets/roneneldan/TinyStories) | A modest language-learning and retention stream appropriate to small models | Story fluency and external-judge ratings do not replace exact reasoning tests; it is not Google work |
| **ARC-AGI and RE-ARC**. ARC tests inference of a rule from demonstrations; RE-ARC provides generators/verifiers for the 400 original training tasks. [ARC](https://github.com/fchollet/ARC-AGI), [RE-ARC](https://github.com/michaelhodel/re-arc) | Demonstration/query episodes, transformation diversity, procedural training anchors | New grids from a familiar generator are not new rules. Split by task/generator as well as by instance; audit verifier correctness |
| **ConceptARC**, Moskvichev et al., 2023. Groups tasks by concepts and varies abstraction demands. [Paper](https://arxiv.org/abs/2305.07141) | Per-concept external reporting | Keep it external in this plan; do not use it for training and later call it unseen. TRM trains on its 160 tasks, so no TRM result is ConceptARC-unseen |
| **Tiny Recursive Models**, Jolicoeur-Martineau, 2025. Reports a 7M model with 44.6% ARC-AGI-1 and 7.8% ARC-AGI-2 public-evaluation accuracy under its two-attempt protocol. [Paper](https://arxiv.org/html/2510.04871v1) | Evidence that parameter count alone does not rule out substantial grid reasoning | Recursion, deep supervision, augmentation and evaluation compute matter; this is not evidence that our causal model reproduces the result in a day |
| **bAbI** and **BabyAI**. Controlled fact-based questions and grounded instructions, respectively. [bAbI paper](https://arxiv.org/abs/1502.05698), [BabyAI paper](https://arxiv.org/abs/1810.08272) | Independent controls for memory, binding, instructions and action sequences | Adapted tasks below are our own symbolic variants, not scores on these benchmarks |
| **DeepMind experience replay** and **GEM metrics**. Study retention, acquisition and transfer in sequential learning. [Replay paper](https://arxiv.org/abs/1811.11682), [GEM paper](https://arxiv.org/abs/1706.08840) | Strong replay comparator; an accuracy matrix over time rather than final loss alone | Their native domains/protocols differ; the memory and resource limits below are our choices |

Two additional engineering references are useful: [DeepMind bsuite](https://github.com/google-deepmind/bsuite) motivates small controllable capability tests, while [DeepMind tsuite](https://github.com/google-deepmind/tsuite) explicitly checks memory and cross-episode contamination. They are not substitutes for a reasoning dataset.

### What the ARC result means for a 24-hour target

The TRM implementation reports **about three days on four H100s** for its ARC runs—roughly 288 H100 device-hours. Its smaller Sudoku example is reported at about 18 hours on one 48GB L40S. This makes ARC-like reasoning a credible small-parameter research direction, but gives no basis to promise the published ARC score on an M2 Air in 24 hours. [Official implementation and run instructions](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)

**What the sub-10M ARC numbers measure (§14).** TRM's ARC build command includes the evaluation subset (`--subsets training evaluation concept --test-set-name evaluation`). Its paper gives each puzzle "a specific embedding", uses "1000 data augmentations" per example, adds "the 160 tasks from the closely related ConceptARC dataset", and answers with "the most common answer out of the 1000 data augmentations" ([paper](https://arxiv.org/html/2510.04871v1)). For HRM, ARC Prize found that training includes "the 400 tasks from the ARC-AGI-1 evaluation set" and called the method "fundamentally a zero-pretraining test-time training approach": "the model can only be applied on puzzles with `puzzle_id`s it has seen at training time". The same analysis found that the hierarchy "had minimal performance impact when compared to a similarly sized transformer", and measured 32% on ARC-AGI-1 and 2% on ARC-AGI-2 on the semi-private set ([analysis](https://arcprize.org/blog/hrm-analysis)). These are per-puzzle, transductive results with heavy augmentation and voting. This suite's grid family asks a causal model to induce an unseen rule from demonstrations in its context, with no puzzle identifier, so its scores are not comparable with those numbers.

Use small ARC-style rule families in the main curriculum, and retain full ARC as a stretch/external measure. Do not downsample hard ARC tasks until v13 wins and call that a neutral benchmark.

### Published small-model anchors: harness checks, never the headline (§14)

A run that scores near zero could mean a hard task or a broken harness. These published results, reproduced at this suite's budget, tell the two apart. They never enter the reasoning score: they are reported beside §10.4's diagnostics, and a harness that misses an anchor's pattern is debugged before any architecture comparison.

| Anchor | Published setting and result | Run here as | What it checks |
|---|---|---|---|
| Addition by small transformers ([Lee et al., 2023](https://arxiv.org/abs/2307.03381)) | NanoGPT with 6 layers, 6 heads and width 384 (about 10.6M), from scratch. With reversed output, a phase transition near 2,500 training samples, after which addition is learned perfectly; a detailed scratchpad reaches perfect addition with 1,000. Length generalization beyond the trained digit counts is difficult | the programs family's addition ladders, in both tracks (§5.3) | the harness learns what is known to be learnable at this size; length failure is expected, not a bug |
| Scratchpads ([Nye et al., 2021](https://arxiv.org/abs/2112.00114); MIT and Google Research) | models from 2M to 1B, trained on 1–8-digit addition and tested on 9–10 digits: without a scratchpad they "completely fail"; with one they improve consistently with size | the trace track's extension ladder | the trace track should move the length frontier; the answer-only track should not |
| Formal languages ([Delétang et al., 2023](https://arxiv.org/html/2207.02098v3); DeepMind) | a transformer of five blocks at d_model 64 and a single-layer LSTM of 256 units; train lengths 1–40, test 41–500; 1,000,000 steps at batch 128. Transformers fail Parity Check out of range but solve Even Pairs; LSTMs solve regular and counting tasks | parity, even pairs, cycle navigation and reverse string, at this suite's budget | the pattern only: in range against out of range, and the transformer–LSTM ordering. The published budget far exceeds 24 hours on the M2, so its numbers are not targets |
| Associative recall ([Arora et al., 2023](https://arxiv.org/abs/2312.04927)) | attention "outperforms gated-convolutions by a large margin" on associative recall, and recall accounts for 82% of the perplexity gap they study | a multi-query recall panel in the state family | the attention-favoured end of §5.7's tags |
| In-context learning from bursty data ([Chan et al., 2022](https://arxiv.org/abs/2205.05055); DeepMind) | a 12-layer transformer of width 64 (about 831K parameters) learns in context when the query's class appears 3 times in its context; in-context learning trades off against in-weights learning, and the two co-exist under a Zipfian class distribution | the grid family's new-parameter panel (§5.2) | in-context rule induction can emerge far below 5M when the data support it; if it does not, check the data distribution before the capacity |

## 4. First-principles design

Difficulty has at least six independent dimensions: **representation burden, number of state variables, computational depth, dependency distance, ambiguity of the goal/rule, and output horizon**. These must be recorded separately. A long prompt with one copied answer can be easier than a short prompt requiring variable binding and two transformations.

Small models should spend most of their capacity learning reusable operations rather than spelling a huge vocabulary or memorizing unrelated world facts. Use compact symbols, a small shared tokenizer, exact simulators and controlled language. This reduces irrelevant burden without giving away the solution.

The complete task must be identifiable from permitted information. If an episode can require two different answers for the same observable prompt, failure may reflect ambiguity rather than insufficient reasoning. Primary planning therefore states its goal. Primary grid induction supplies enough demonstrations to determine the query answer within the registered rule grammar. Hidden-goal inference is a separate experiment.

Long outputs compound mistakes. As an illustrative independence calculation, 98% correctness per decision gives only `0.98^40 ≈ 44.6%` probability of forty correct decisions; 95% over eighty gives about 1.65%. Real errors are correlated, but this explains why token loss, individual steps and whole-task success must all be reported. Begin with short complete solutions and extend the horizon in distinct panels.

An abstract rule must survive changes in irrelevant attributes. Change colours, entity names, object placement, instruction wording and record order while preserving semantics. Conversely, construct matched examples where changing one relevant relation changes the answer. This exposes copying, colour-frequency and recency shortcuts.

Continual learning requires a nonstationary stream and restricted access to the past. Shuffling all tasks forever is multitask training. Conversely, silently changing a mapping while demanding both its old and new answers for an identical prompt is ill-posed. Contextual changes and genuine factual updates need different evaluation contracts (§8).

Finally, **a predictable latent representation is not the same as task competence**. Do not select datasets because next-concept prediction is particularly easy, nor because their optimal solver naturally has exactly four-token steps. Probe such mechanisms only after reporting ordinary solution performance.

## 5. The proposed dataset

Working name: **Small Reasoning and Continual Learning Suite, V13**. This is an experimental suite name, not an existing public benchmark.

### 5.1 Families, exposure and scoring

The mixture below applies to **base stage S0**. Continual phases use the incoming distributions in §8, so old tasks are not automatically refreshed by this mixture.

| Family | Share of processed, non-padding training tokens | Primary observable skill | Main metric |
|---|---:|---|---|
| Grid-rule induction | 22.5% | Infer and apply a spatial rule from examples | Exact output grid, including dimensions |
| Programs and algorithms | 22.5% | Execute and compose short deterministic operations | Exact final answer/state |
| State tracking and binding | 22.5% | Update entities and combine facts | Exact query answer |
| Explicit-goal planning | 22.5% | Reach the stated goal under transition rules | Simulator-verified task success |
| Simple English | 10% | Preserve basic language modelling | Held-out NLL; separate from reasoning score |

These are **processed-token exposure shares**, not probabilities of drawing a batch, example counts or score weights. Use quota scheduling over non-padding prompt-plus-target tokens. Record supervised-token counts and example counts separately. Within each reasoning family, equalize registered task-type exposure before varying difficulty. The primary reasoning aggregate is the unweighted mean of the four family accuracies; language NLL is never numerically averaged with accuracy.

Primary loss on procedural examples is target-token CE including EOS, normalized per example; average within a batch. Inputs containing independently sampled random facts, operands or layouts are unscored. Text uses ordinary next-token CE. Use family-homogeneous, fixed-shape batches under the quota scheduler and a shared list of example IDs for matched-data runs. Log realized loss contributions: token quotas do not imply equal gradient contributions.

### 5.2 Grid-rule induction: ARC-like, with a bounded rule grammar

An episode contains **three input/output demonstration pairs and one query input**, followed by the query output as the supervised target. The operation name and rule program are not in the prompt. Use flat row-major colour tokens with height/width and row delimiters; all models use the same representation. A registered option also scores the second and third demonstration outputs wherever the preceding demonstrations already determine them within the grammar (checked by the enumeration below). That roughly triples the supervised tokens per episode, identically for every arm; the primary metric stays on the query.

Candidate generator grammar, with the S0/S2 availability restrictions in §8.1:

- Spatial transforms: horizontal/vertical reflection, quarter turns, bounded translation.
- Object operations: select the unique largest/smallest connected object, extract its tight bounding box, preserve/remove a specified class identifiable from demonstrations.
- Attribute operations: a consistent colour substitution, count objects, extend a short visible pattern.
- Two-operation compositions with legal type/shape transitions, such as select-largest then reflect. Later primitives are held out for the continual stream.

Each primitive needs a deterministic implementation and an independently checked evaluator. Use 4-neighbour connectivity, zero as background, and explicit rejection rules for ties, clipping and overlapping objects. Non-background colours can be consistently permuted. Translation uses zero fill with no wraparound; reject clipping in the first tiers. Recolouring choices must be inferable from the demonstrations. A crop returns exactly the selected object's bounding rectangle. Ambiguous pattern extension is excluded until its finite grammar and uniqueness test exist.

| Tier | Initial geometric envelope | Rule complexity | Intended role |
|---|---|---|---|
| L0 | 1×n rows, or 3–4 cells per side, 1 object, 2–3 colours | One simple primitive | Prerequisite/correctness checks |
| L1 | 4–6 per side, 1–3 objects, 2–4 colours | One primitive; independently vary irrelevant features | Working competence |
| L2 | 5–8 per side, 2–4 objects, up to 5 colours | Two primitives; familiar and held-out combinations scored separately | Main challenge |
| L3 | 9–12 per side or three primitives | Size or composition-depth extension, one axis at a time | Stress panel, not admission gate |

These dimensions are provisional; serialization must fit the context limit without truncating a grid or removing required demonstrations. Reduce dimensions, not the demonstration count secretly, if a tier does not fit. Reserve larger-context evaluation for separately trained compatible models.

For each episode, enumerate all legal programs within the declared grammar/depth that fit the demonstrations. Retain the episode only if they agree on the query output. This certifies **answer-identifiability within that grammar**, not a unique universal explanation. Store the candidate-program count. If enumeration is infeasible for an extended grammar, that grammar cannot enter the primary track under a claimed uniqueness guarantee.

Include copying and single-primitive lookup as explicit controls. Balance no-change examples so identity is legitimate but cannot dominate. Report performance by foreground density and compare with copy-input, all-background and nearest-training-example baselines. Cell accuracy alone rewards blank grids and is secondary.

**Reading the demonstrations, not recognizing a program (§14).** Under a bounded grammar a model can store a few programs in its weights and use the demonstrations only to pick one. That is task recognition; ARC asks for task learning. Three provisions keep the two apart:
- *Parametric rules.* Most primitives take a parameter drawn fresh for each episode: a colour map, a translation offset, a selection criterion, a fill colour. Many distinct rules then share each program, so the demonstrations must be read to tell them apart. DeepMind found that in-context learning emerges when training data are bursty, contain "large numbers of rarely occurring classes", and give items meanings that are "dynamic rather than fixed" (Chan et al., 2022). Per-episode parameters supply all three.
- *A new-parameter panel.* Parameter values are assigned to splits by hash, as instances are in §6.1. Test episodes use program families seen in training with parameter values that never were. Recognition alone cannot solve them.
- *A fixed checkpoint curve.* In-context learning "first emerges, then disappears and gives way to IWL [in-weights learning], all while the training loss decreases" ([Singh et al., 2023](https://arxiv.org/abs/2311.08360)). The new-parameter panel is evaluated after every 10% of base exposure, at the same points for every arm. The scored checkpoint is the final one; the curve is reported and never used to pick a checkpoint per arm. Weight decay is shared, so the paper's finding that L2 regularization can keep in-context learning is a note, not a per-arm setting.

**One-row rungs.** The grid size ladder starts with one-row grids in the style of 1D-ARC ([paper](https://arxiv.org/abs/2305.18354), [repository](https://github.com/khalil-research/1D-ARC); 18 task types such as moving, filling, hollowing, flipping, mirroring, denoising, pattern copying and recolouring by size; MIT licence). A 1×n grid costs n tokens rather than n², so rule induction can be calibrated cheaply before 2D grids. The official 1D-ARC tasks stay an external panel (§5.6). Their task count and array sizes were not verified.

Use RE-ARC as an optional external procedural anchor after auditing its selected generators. Keep its original task IDs grouped in splits; its own difficulty scores are not automatically our calibrated tiers. The bespoke grammar suite must be labelled **ARC-like**, never “ARC-AGI accuracy.”

### 5.3 Programs and algorithms

Use a small, typed language, not arbitrary Python. No hidden imports, external knowledge or floating-point rounding questions. Values are bounded signed integers and short lists. Initial task types are equal-weight subpanels:

1. Arithmetic: add/subtract and short expressions; later multiply, exact integer division and one-variable linear equations.
2. Lists: minimum/maximum, sorting, filtering and a short composition of two operations.
3. Register programs: assignment, addition/subtraction, one conditional, then a bounded repeat. Every variable is initialized.
4. Graphs: reachability, shortest distance on an unweighted graph, then a short valid path. Equal-length valid paths are accepted by the verifier.

| Tier | Operand/list/graph range | Computational bound |
|---|---|---|
| L0 | Single-digit integers; lists 2–3; graphs 3–4 nodes | One operation or one hop |
| L1 | Integers in −20…20; lists 3–6; graphs 4–6 | 2–4 executed operations; up to 2 hops |
| L2 | Integers in −99…99; lists 6–10; graphs 6–10 | 4–8 operations; branches or repeat count 2–3; up to 4 hops |
| L3 | Lists/graphs 12–16 or 9–12 operations | Extend one bound at a time |

Bound all intermediate values to −999…999 in the main track; reject overflow and division by zero. Sample positives/negatives, carry/borrow cases, duplicate list entries and branch outcomes explicitly. Do not let AST token count stand in for executed depth. Logs must include both.

Example, answer-only track:

```text
<program>
a = 7 ; b = 3 ; a = a + b ; b = a - b ; return b
<answer> 7 <eos>
```

Machine-checked execution traces can provide a **separate trace-supervised track**, given to every architecture. They are not privileged training hints for the concept model. In that track, inference must generate its own trace, and trace tokens count toward time and exposure. Compare trace and answer-only results separately.

**Formats set difficulty as much as size does (§14).** At about 10.6M parameters, reversing the output digits of addition "significantly enhances the performance", and a detailed scratchpad "achieves perfect addition with just 1000 samples" (Lee et al., 2023). Trained on 1–8-digit addition and tested on 9–10 digits, models without a scratchpad "completely fail" (Nye et al., 2021). The digit order and the track are therefore frozen before calibration, identically for every arm. Multi-digit numeric answers are written least-significant digit first by default. The answer-only and trace tracks have separate ladders and are never pooled. Graph tasks have a known frontier: a transformer can learn graph search "when given the right training distribution", but "as the input graph size increases, the transformer has greater difficulty", and more parameters do not resolve it ([Saparov et al., 2024](https://arxiv.org/abs/2412.04703)). Graph size is the axis on which to expect a frontier.

DeepMind mathematics and small CLRS instances can seed audited adapters. Do not reuse the previously corrupted serialized-byte wrappers from the repository, and do not compare losses across different tokenizers as if they were the same unit.

### 5.4 State tracking and binding

Generate small worlds with entities, locations, containment and transfer events. Query the current location, owner, count, or a two-relation consequence. Use both compact records and controlled-English renderings of the same semantic world, with the world grouped across splits.

| Tier | Entities / events | Required reasoning |
|---|---|---|
| L0 | 2–3 / 2–4 | One fact or most recent relevant update |
| L1 | 3–5 / 4–8 | Update one attribute; ignore distractors |
| L2 | 5–8 / 8–16 | Object transfer plus movement; two-link binding |
| L3 | 8–12 / 16–24 | Longer history or three links, separately |

Example:

```text
Mira is in the kitchen. Jo is in the hall.
Mira takes the key. Mira moves to the hall.
Mira gives the key to Jo. Jo moves to the shed.
Where is the key?
<answer> shed <eos>
```

Transfers require co-location; the example satisfies that condition when Mira moves to the hall. An object's location follows its current carrier until dropped or transferred. Reject illegal events. Do not leave transition semantics to common sense.

Balance queries across entities and values. Give multiple entities the same candidate values so an unordered bag of mentioned values cannot solve binding. Use identical suffixes with different earlier relevant facts, and matched histories with one altered event. Keep latest-mention and most-frequent-value controls.

Dependency-distance bins are fixed **in tokenizer tokens**, proposed as ≤32, 33–64, 65–128, 129–256 and 257–512. They are descriptive bins independent of any architecture's receptive field. Include short and long cases; do not change their weights to make the concept path necessary. Under §7.4 these bins are the rungs of an ID ladder whose ends the references set.

### 5.5 Explicit-goal planning

Use symbolic grids and a common simulator, inspired by grounded-instruction environments. Observation contains the full initial map, start, objects and **explicit goal**. Begin with navigation; introduce a key/door only after navigation competence. No visual encoder is required.

| Tier | Initial world envelope | Plan requirements |
|---|---|---|
| L0 | 4×4, no doors | 1–4 actions to one goal |
| L1 | 4–5 per side, simple obstacles | 4–8 actions; one target |
| L2 | 5–7 per side, limited obstacles | 8–16 actions; two ordered goals or one key/door |
| L3 | 7–9 per side | 17–32 actions or three goals, separately |

Use cardinal moves, plus explicit pickup/open operations only in the key/door subpanel. Reject unreachable problems with an exact shortest-path search over the full state, including key ownership and goal progress. Randomize action-token naming only in the registered representation-shift phase, with a legend supplied to all models.

Training can use one deterministic shortest solution. Evaluation accepts **any** valid solution reaching the stated goal within the fixed action limit; exact teacher-path match is not the success criterion. Report excess path length relative to optimum. An output that reaches a different familiar goal fails.

Primary evaluation asks the model to produce a complete plan, then executes it. An optional closed-loop panel returns simulator observations after each generated action to every model. Keep those protocols separate. Future simulator states and subgoal progress are not fed to one architecture alone.

### 5.6 Language and external anchors

Use a train-only, deduplicated TinyStories sample for the 10% language stream. Evaluate NLL on held-out whole stories. Fit the shared tokenizer on allowed training text only. Synthetic state-world questions provide objective language-understanding checks; a subjective story judge is not the main reasoning evaluator.

Keep small external panels separate: untouched eligible ARC tasks, ConceptARC, and an audited mathematics/CLRS subset. Document whether the model has seen any source task or derived generator. Retain all eligibility decisions before model evaluation; never select a “small-model ARC subset” by deleting its errors. Report custom and official protocols under different names.

### 5.7 Inductive-bias tags (§14)

Neutrality requires the suite's tilt to be visible. Every registered task type carries three tags, fixed in the manifest before any model run. They yield subscores and never change the primary weights.

| Tag | Values | Basis |
|---|---|---|
| Favoured by | attention; recurrence; neither known | *Attention:* in-context recall and copying; attention outperforms gated convolutions on associative recall by a large margin (Arora et al.). *Recurrence* (LSTM-like, or linear RNNs with negative eigenvalues): tracking state over lengths beyond training. Transformers fail parity out of range while LSTMs solve regular and counting tasks (Delétang et al.); linear RNNs whose transition matrices have only positive eigenvalues "cannot solve parity" ([Grazzi et al.](https://arxiv.org/abs/2411.12537)); SSMs "cannot solve simple state-tracking problems like permutation composition" ([Merrill et al.](https://arxiv.org/abs/2404.08819)). *Neither known:* arithmetic, sorting, graph search, planning |
| Dependency distance | §5.4's bins in tokenizer tokens | from each scored target token to the farthest input token it depends on |
| Copy versus compute | the share of target tokens that are verbatim copies of input tokens | identity-heavy grid transforms are mostly copy; arithmetic answers are mostly compute |

Where a family allows it, it contains task types at both ends of the first tag: binding queries and long update sequences in the state family, list lookup and long register programs in the programs family. The manifest records any family that cannot. Tags come from the literature and the task definitions, never from model results, and would be the same if V13 had no recurrent block, no concept path and no confined decoder. A design with a bounded direct reach and a compressed long-range path, V13 among them, should show its cost in the copy-heavy, long-distance cells; the tags make that visible without changing the score.

## 6. Splits, serialization and reproducible construction

### 6.1 Three different generalization questions

| Panel | May share with training | Must be held out | Interpretation |
|---|---|---|---|
| **ID** | Primitive and program/rule family, and parameter values from the training-assigned set | Semantic instances, worlds, layouts and their augmentations | Can it solve fresh examples of what it learned? |
| **New parameter** (§5.2) | Program family and primitive | Parameter values (colour maps, offsets, selection criteria), by hash | Does it read the demonstrations, or recognize a stored program? |
| **Composition** | Every primitive and its basic meanings | Registered ordered combinations / program skeletons | Can it recombine known skills? |
| **Length/size** | Rule/program family | One larger length, size or horizon range | Can it extend the same computation? |

An **unseen primitive** is a new-learning task in the continual stream, not an ordinary zero-shot failure in the primary composition score. Unknown primitives may enter an explicitly labelled few-shot induction panel only when demonstrations make their effect identifiable.

Split at the level required by the claim. ID can legitimately share a rule family; composition cannot. Do not falsely require every program skeleton to be disjoint for ID, and do not falsely call a new seed from the same skeleton “novel composition.”

Proposed deterministic allocation:

- Assign normalized semantic instances to train/calibration/dev/test by a stable hash, with 80/5/5/10 buckets. Use distinct root namespaces for independent replications. Duplicate hashes cannot cross splits.
- For composition, enumerate legal ordered primitive pairs. Reserve approximately 20% for dev and 20% for test, keeping each primitive represented in training. Freeze the exact list, not only its percentages. Exclude semantic equivalents across partitions where the grammar permits them.
- For external RE-ARC, group by original generator/task ID before making a rule-generalization split. A separate same-generator panel may measure instance generalization.
- Assign source worlds, raw story documents, base grids and base layouts before rendering or augmentation. Colour permutations, name substitutions, rotations, paraphrases and trace/answer-only versions inherit the source group.
- Canonicalize graph labels, entity names and grid colour/dihedral symmetries when they are irrelevant to the specific task. A transformation must also transform direction/goal labels where needed. For small graphs use exact canonicalization; otherwise document the limits of an approximate duplicate screen.

Semantic equivalence across arbitrary programs is not generally cheap to decide. Use a bounded DSL, algebraic normalization for registered identities, and exhaustive checks on small domains where feasible. A finite random probe signature is a screen, not proof. Report remaining leakage risks and keep a untouched external panel.

### 6.2 Proposed sizes and sample accounting

Training should be a reproducible procedural stream, not a small fixed file recycled without recording it. Create shards in advance or generate on a separate CPU path, with a bounded cache. Do not run a large solver in the accelerator training loop.

Initial construction budget: **100,000 accepted episodes per reasoning family**, expandable by the same frozen generator/seed rules if the measured token budget needs more. This is a starting supply, not an assertion that all 400,000 are necessary or sufficient. Grid episodes count all demonstrations plus the query as one episode. Record unique semantic instances, augmentations and effective repetitions separately.

Calibration: 128 independent instances per registered task type × candidate rung for rapid screening; confirm the ladder endpoints (§7.4) on a disjoint 512-instance calibration panel. Development/final evaluation: target **1,024 independent instances per ladder** (128 per rung over 8 rungs, §7.4) for each task type's primary ladders, plus the composition and new-parameter panels, balanced across task types. For cells with a smaller finite semantic universe, enumerate it or report the actual number; do not inflate the sample size with duplicate augmentations. A primary instance owns its entire augmentation group.

Continual evaluation: a fixed 512-instance development panel and 1,024-instance final panel per phase and family where applicable. Save checkpoints during the stream; use development panels for the announced intermediate diagnostics, then evaluate the frozen final-test matrix retrospectively once protocol selection is over. Do not adapt to test scores or let an evaluation query update the bank.

### 6.3 Representation contract

Use one shared tokenizer with **at most 2,048 entries**, reserved structural tokens, single-token digits/colours/actions, and byte fallback for text. Fit BPE only on permitted base-training text. Reserve syntax for future tasks in advance, but do not teach their withheld semantics. Do not train a tokenizer on future continual-task solutions.

Use sequence buckets 128, 256, 512 and 1,024, with identical bucket policies for all arms. Short records may be packed only with block-diagonal attention and correct state resets. Reset recurrent state, caches, Engram n-grams and concept-group positions at document boundaries. Whole multi-example grid episodes remain intact.

Padding is unscored and masked. No arbitrary truncation of a question, solution or required support example. Rejection because of length is logged per tier, to reveal whether the length cap accidentally selects easier instances. Keep 2,048-token extensions out of the primary track unless trained and budgeted for all arms.

Natural variable record lengths should vary position relative to chunk boundaries. Do not insert a concept-sized delimiter or align one reasoning operation with one compression group. An evaluation-only harmless-prefix test may vary offsets uniformly over 0…15 tokens, independent of any model's pooling factor; it is a robustness diagnostic and does not alter main scoring.

### 6.4 Record and manifest schema

Illustrative schema, to implement later:

```json
{
  "dataset_version": "v13.0-candidate",
  "example_id": "stable-content-id",
  "semantic_group_id": "source-instance-and-augmentations",
  "family": "grid",
  "task_type": "object_transform",
  "split": "train",
  "panel": "id",
  "tier": "L2",
  "prompt": "serialized observable demonstrations and query",
  "target": "serialized output with EOS",
  "generator": {"revision": "pinned-at-build", "seed": 31001},
  "metadata_private": {
    "program_signature": "canonical-DSL-signature",
    "primitive_ids": ["select_largest", "reflect_x"],
    "executed_depth": 2,
    "support_candidates": 1,
    "minimum_plan_length": null,
    "phase_id": 0,
    "verifier_version": "pinned-at-build"
  }
}
```

The training reader allowlists only observable fields; `metadata_private` must never become input accidentally. Difficulty is derived from the actual program/solver properties, not the illustrative values above. Manifests contain source revisions and URLs, license notices, generator and verifier hashes, exact split rules, excluded examples, source normalization, tokenizer hash, accepted/rejected counts, length distributions and semantic-group overlap checks. “Pinned-at-build” is a requirement, not an already fetched commit.

## 7. How to make it neither trivial nor impossible

### 7.1 Calibration model and capacity accounting

Use a plain causal full-attention transformer as the difficulty reference. A concrete starting template is bias-free attention, a conventional 4d two-matrix FFN, RoPE, normalization and tied token embeddings. Approximate counts, excluding small normalization terms:

`P ≈ 12 × layers × width² + vocabulary × width`

| Track | Starting template | Approximate count | Admission cap |
|---|---|---:|---:|
| Small | 10 layers, width 192, 2,048 vocabulary | 4.82M | Strictly below 5M |
| Larger | 9 layers, width 256, 2,048 vocabulary | 7.60M | Strictly below 10M |

These are reference templates, not mandatory replacements for the research architecture. An exact unique-parameter counter must include embeddings, Engram tables, adapters and learned auxiliary heads. Report peak training parameters and deployed parameters separately. Recurrent sharing does not multiply stored weights, but it does multiply computation. A large lookup table cannot be excluded to describe a model as sub-5M. Non-gradient memory is reported in bytes in addition to parameters (§9).

Match competitors within 5% of the declared deployed parameter budget where feasible, always within the track cap, and publish the counts. A training-only auxiliary head is an explicit extra resource. An exact same-total-cap companion run is needed if that extra capacity is material. Do not hide differences behind an “active parameter” number.

**A second range reference (§14).** A range set by a transformer alone ends where the transformer fails, and would cut the tasks on which other inductive biases succeed. Each track therefore also calibrates an LSTM at the track's cap (Delétang et al.'s recurrent reference was a single layer of 256 units). It is fixed now, before any v13 run, and is never v13 or a model sharing v13's novel parts (role-masked streams, the concept path, the GDN-2 loop, Engram). The two references set only the ends of each ladder (§7.4); neither admits nor drops a task. The panel is itself a choice: it spans attention and recurrence, not every inductive bias. It adds one pilot per track to the calibration budget (§11.1).

### 7.2 Calibration procedure

1. **Correctness first:** exact solvers must pass every generated item; independent unit/metamorphic checks cover edge cases. Tiny overfit tests check whether the model can fit 128 fixed examples of each task. Failure here triggers implementation/optimization diagnosis before reducing dataset difficulty.
2. **Baseline screen:** train the small reference on the candidate mixed stream for a capped two-hour pilot, on the intended hardware. Evaluate L0–L3 calibration cells at several checkpoints. This is a difficulty screen, not the definitive architecture comparison.
3. **Fix the ladders once (§7.4):** the references set each ladder's endpoints by §7.4's rule. No cell is admitted or dropped for its observed accuracy; a cell is removed only for a failed identifiability, serialization or teacher check, and every removal is recorded. Confirm the endpoints on the disjoint calibration panel; freeze the difficulty manifest before v13 runs.
4. **If a cell fails:** first check answer identifiability, sequence length, loss mask, serialization and teacher correctness. Then simplify one intrinsic variable, such as executed depth or number of objects. Do not shorten dependence specifically to a baseline or v13 receptive field.
5. **If a ladder is trivial for every reference:** retain a small prerequisite allocation and move the challenge upward in a registered order: more varied instances → a second operation → extra state variables → longer horizon. Do not raise every axis simultaneously.
6. **If two hours is inconclusive:** label the cell uncalibrated. Continue the reference pilot within a separately reported calibration budget or leave it in the stress panel. Slow optimization is not evidence of a fundamental capacity limit: generalization "can happen well past the point of overfitting", and "smaller datasets require increasing amounts of optimization" ([Power et al., 2022](https://arxiv.org/abs/2201.02177)).
7. **Larger track:** calibrate the same candidate ladder independently at 7.6M. A small-track failure/larger-track success is evidence about this budget, not a universal parameter threshold. Never select a different test subset for each architecture within a track.

Calibration checkpoints are disposable. Start scored models from fresh seeds; never reuse a pilot's learned weights, optimizer, vocabulary fitting on future material, bank, adapters or codebook statistics. This matters especially when the calibration stream includes primitives that must be new at S2 in the scored stream.

Descriptive bands. Since §14 they label rungs on a ladder, read at the scored budget on **fresh examples**, and no longer admit cells:

| Tier | Reference exact success | Role |
|---|---:|---|
| Prerequisite | ≥90%, and comfortably above relevant shortcuts | Confirms data/model interface; small training share |
| Working | About 50–85%; nontrivial controls clearly lower | Supplies learnable, unsaturated main comparisons |
| Challenge | About 15–50% where chance/shortcut floors permit | Measures improvement and composition; report independently |
| Stress | May be below 15% or uncalibrated | Visible ceiling test; excluded from headline score |

These numerical bands are pragmatic choices, not published laws. A balanced binary task with 50% chance cannot qualify as a working task at 50%; require a margin over its strongest registered shortcut and stratify by answer space. Report uncertainty rather than rounding a score across a threshold.

**Coverage gate:** for the full-suite claim, each of the four reasoning families needs at least one primary ladder whose range brackets a reference's 50% point at the pilot budget. If that cannot be achieved, publish a named reduced suite and its omitted families before comparing models. Never silently remove an architecture's weak family.

### 7.3 Curriculum and evaluation separation

During base training, use the same predeclared curriculum for every model:

- First 10% of the base exposure: 70% prerequisite, 30% working.
- Next 20%: 30% prerequisite, 60% working, 10% train-side challenge.
- Remaining 70%: 15% prerequisite, 65% working, 20% train-side challenge.

Percentages apply within each reasoning family. The language share stays at 10% of total processed tokens. “Train-side challenge” uses only permitted training combinations; dev/test compositions remain absent. On each ID ladder (§7.4) the three labels are fixed, contiguous ranges of rungs set when the ladder is frozen: prerequisite the lowest, working the middle, train-side challenge the highest trained rungs. The test-only rungs of extension ladders never appear in training.

Do not promote each model on its own validation success: that gives different training data to different architectures. A live adaptive curriculum is a separate method requiring its own control. Main tests use the frozen working/challenge panels, not the training mix weights. Always publish tier-wise results alongside the aggregate.

The definitive endpoint may reveal that a cell saturates or stays near floor despite its pilot. Report that fact and all fixed-panel scores. Do not recalculate the headline by deleting inconvenient cells. A later dataset revision requires fresh test instances and a new version.

### 7.4 Difficulty ladders: measure the frontier instead of guessing the band (§14)

Admitting cells whose reference accuracy falls in a band has three defects. The pilot runs for two hours while scored base training runs for up to twelve, so cells admitted at 50–85% are likely to saturate by the end: the ceiling this suite exists to avoid. Admitting cells by an observed accuracy also selects on the pilot's noise. And a range set by one transformer ends where that transformer fails. Ladders, with §7.1's two references, remove all three.

**Construction.** Each task type has one primary ID ladder and, where it has a size axis, one extension ladder. A ladder varies one axis alone (executed depth, number of objects or entities, dependency distance in tokens, list or grid size, plan or operand length) over 6–10 rungs with 128 items per rung, so an 8-rung ladder holds §6.2's 1,024 instances. Other axes are diagnostic ladders, evaluated on the development panel only. Rungs and items are fixed before any scored model run; the reference pilots (§7.1) choose only the endpoints. L0–L3 become coarse labels for contiguous ranges of rungs.

**Which rungs are trained.** The two kinds of ladder answer different questions, and their scores never share a name:
- *ID ladders* (depth, object or entity count, dependency distance): the frozen training manifest covers every rung. Their area measures competence on what was trained.
- *Extension ladders* (list length, grid size, plan length, operand length): training covers the lower rungs only; the upper rungs are test-only. They are §6.1's length/size panel. The boundary between trained and test-only rungs is fixed in the generator specification before the pilot. Where the references fail out of range at once, as Lee et al. and Delétang et al. both found, the endpoint rule puts the top two rungs past the first untrained rung, so an extension ladder typically has about three test-only rungs; that is the expected shape, not a misfire. The grid-size ladder's top is also bounded by the 1,024-token context, so its "beyond ladder" reads "beyond context".

**Endpoints, the only place the references touch the suite.** At the pilot budget, the bottom rung is the lowest level at which *every* reference (§7.1) reaches 90% exact success. The top rung lies two levels beyond the first level at which *no* reference exceeds the rung's floor by more than 5 points; the two-level margin absorbs the gap between the pilot budget and the scored budget. If a scored model's frontier reaches the top two rungs, the ladder is extended in a versioned revision with fresh items. Existing rungs are never dropped.

**Metrics.** Let the floor c be the larger of chance and the strongest registered shortcut on that rung.
- *Ladder area:* mean exact success over rungs, each rung weighted equally, reported raw and above floor, (p − c)/(1 − c).
- *Frontier L50:* the level at which the fit p = c + (1 − c)·σ(β·(L50 − level)) lies halfway between floor and ceiling. It is the analogue of METR's 50% time horizon, which fits p = σ((log h − log t)·β) to success against human task time ([METR](https://arxiv.org/abs/2503.14499)). Intervals come from a hierarchical bootstrap over task types, then instances, and over seeds where there are several. A fit that does not cross inside the ladder is reported as "beyond ladder" or "below ladder", never extrapolated.

Wherever a model's capability lies, some rungs are informative for it, so neither "too easy" nor "too hard" can hide a difference; this is the direct answer to the concern that motivated the suite. Fluid Benchmarking makes the same argument for language models, that "the relative value of benchmark items depends on an LM's capability level", and uses item response theory to place models on one ability scale ([paper](https://arxiv.org/abs/2509.11106)). With several scored systems, a two-parameter item response model over rungs is an optional later refinement; the logistic frontier is the frozen primary form.

## 8. Continual learning: a stream that can reveal acquisition and forgetting

### 8.1 Primary skill stream

Base stage S0 teaches the initial primitives and selected combinations from all four families, plus language. Reserve some primitives and combinations for later. Exact allowed primitive/skeleton lists must exist in the frozen manifest before generation.

The proposed first partition is concrete:

| Family | Available in S0 | Withheld until S2 |
|---|---|---|
| Grid | Reflections, quarter turns, translation, recolouring, largest/smallest selection with canvas size preserved | Tight bounding-box crop; pattern extension stays optional until identifiability is verified |
| Programs | Add/subtract, list extrema/sort, assignment/conditional/bounded repeat, graph reachability/distance | Multiplication and exact division; linear equations remain an optional later extension |
| State | Movement, pickup/drop, person-to-person transfer, current location/owner/count | Nested containers and transfer of a container with its contents |
| Planning | Navigation and ordered visits without locked doors | Pickup key and unlock matching door |

Selection on a preserved canvas and crop are different operations. Other generator features are not automatically S0 training content merely because §5 describes them. The base evaluation covers only S0's skills. S2 test examples use fresh instances of the newly taught primitives; permanently held-out combinations involving them stay in a separate composition panel.

Each phase has a fixed incoming exposure budget. Its phase ID is available to the evaluator for bookkeeping, **not** as a routing label to the model. Natural task syntax and explicit goals remain visible; concealing the meaning of the request is not the intended challenge. Scheduled consolidation boundaries are supplied equally to all methods. This is boundary-aware continual learning; a smooth, boundary-hidden stream is a later robustness extension. Continual-learning scenarios are classified by "whether at test time task identity is provided" and, if not, "whether it must be inferred" ([van de Ven et al.](https://arxiv.org/abs/1904.07734)). Here the task is identifiable from the observable request, which places the stream nearest the easiest scenario. Its results should not be compared with class-incremental benchmarks, in which identity must be inferred.

| Phase | Incoming data | What stays in retention evaluation | Specific question |
|---|---|---|---|
| S1: new combinations | New ordered combinations of already learned grid/program/world/planning primitives | S0 tasks on fresh instances | Does prior knowledge speed recombination? |
| S2: new primitives | A withheld operation in each applicable family: e.g. grid crop, bounded multiply, container transfer, key/door | S0 and S1, including similar operations | Can the model acquire new skills without losing old ones? |
| S3: close interference | Dense training on confusable but explicitly distinguishable tasks: largest versus smallest, add versus subtract, owner versus location, ordered goals A→B versus B→A | The counterpart of each pair, matched in difficulty | Do gates protect related skills, not merely different domains? |
| S4: representation shift | Held-out legal renderings, new name/colour assignments on fresh worlds, coordinate or action vocabulary with an explicit legend | Original renderings of the same skills and untouched semantic instances | Is knowledge reusable across surface changes? |
| S5: composition after acquisition | Previously withheld combinations involving S2's new primitives | S0–S4 | Does recently learned knowledge become composable? |
| S6: return | Fresh S0/S1 instances, with a preregistered mixture | Every acquired skill | How much was retained before return, and how fast is relearning? |

S1–S5 introduce stable meanings; they do not overwrite the answer to an identical observable request. New sample IDs from an old task do not count as a new skill. S6 is deliberately a return distribution, not evidence of novel transfer. Measure retention **before** its updates and relearning afterward.

For S4, reserve the actual rendering families/legends before S0. Merely permuting colours again after training on arbitrary colour permutations is ordinary augmentation, not evidence of a new domain. Such cases serve as controls, while genuinely withheld renderings carry the shift claim. S3's counterparts are explicitly named in the request or inferable from demonstrations; a hidden task switch cannot demand an unknowable answer.

The primary phase order respects prerequisites. A repeat uses fresh instances and a preregistered alternate valid order of the independent shifts, rather than moving a composition before its necessary primitive has ever been taught. Publish order sensitivity; six phases cannot establish performance over an indefinitely long lifetime.

An important split distinction: a composition reserved for a future training phase is **temporarily unseen**, while a final-test composition remains never trained. Separate their registries. After S1 teaches a composition, its fresh-instance score is post-acquisition ID, not zero-shot compositional generalization. Keep permanently held-out compositions for the final composition panel.

Each S1–S5 phase trains only its announced incoming tasks, plus any memory/replay the method is allowed to retain. Do not automatically mix in every earlier family: that would conceal forgetting. S6 is the explicit return exception. Text is evaluated as an old skill throughout, and may be revisited only through the same replay allowance or the announced S6 mixture.

### 8.2 Secondary knowledge-update stream

Skill learning and one-shot factual memory are different. Add a separately scored episodic stream of **64 initially known bindings plus 16 new bindings per phase**, using a small fixed vocabulary of entity names. After observing an update, evaluate later queries without repeating that update in the prompt. Reset prompt/cache state between write episodes and queries; only the declared persistent memory/weights carry information.

Two legal cases:

- **Contextual coexistence:** `world=North, gate=red` and `world=South, gate=red` have different codes. The semantic world label is present in updates and queries for every model. It is not an oracle phase ID.
- **Revision:** one entity's current code changes. Current-value queries must return the new code; retaining the superseded answer is an error. Historical queries are allowed only if they include a timestamp/version and the stream taught that contract.

The evaluator knows the correct database state. Models cannot access it. An exact dictionary over seen keys is a mandatory transparent baseline; high exact-key accuracy alone establishes storage, not abstract learning. Add paraphrased queries and two-link composition over learned bindings to test use of stored facts. A bank that only recognizes the original key should be described accordingly.

Keep this secondary stream out of the four-family reasoning macro. If time is limited, complete the primary skill stream before adding it. It is especially relevant to a fast memory bank but must not be given enough weight to favour that architecture by construction.

### 8.3 Learning eligibility and tuning

Measure every phase before learning, after matched incoming exposures, after consolidation if present, and at later stages. A skill never acquired cannot meaningfully be “retained.” For working-tier retention claims, mark a skill eligible when its immediate post-learning exact accuracy reaches a preregistered 60% and clears its shortcut baseline. Report all skills and failures too; eligibility must not improve the headline by hiding failed acquisition.

Calibrate learning rates, memory bandwidths, gate thresholds and replay ratio on a **separate continual calibration stream**, with different worlds/program groups. Freeze them before the scored stream. Positive/negative gate examples may use only already observed training data. Future phases, dev/test labels and evaluator metadata are forbidden. A gate must be screened on each old task type, not on a text-dominated pooled loss.

## 9. Comparators, memory and fairness

There are two different questions, and their experiments should not be conflated:

| Question | Minimum comparisons | Interpretation |
|---|---|---|
| Is the **base architecture** better? | Plain full-attention TF; v13 with its selected objective; same v13 topology with auxiliary objective off | Separates architecture from auxiliary-loss effects on the same data |
| Is the **continual method** better? | Frozen/no-update; sequential adaptation without replay; matched adaptation with reservoir replay; proposed bank/consolidation method | Separates retaining old behaviour by doing nothing from learning while retaining |

For method attribution, branch from the **same base checkpoint** and match which weights can update. Compare plain versus gated LoRA on the same base, for example. Compare a complete v13 system with a TF+replay system separately as a system comparison. A result across different bases, optimizers and memory mechanisms cannot isolate which component helped.

An offline joint-training reference, allowed to revisit all seen data, is useful as a reference ceiling when affordable, but it has privileged access and is not an online competitor. A compact recurrent model is a useful additional inductive-bias control when resources permit; its absence must not delay the essential TF baseline or be replaced with an untrained straw man.

**Default additional persistent-memory cap: 8 MiB per method.** This proposed cap covers replay examples, correction-bank keys/values, routing statistics, metadata and extra persistent learned adapters. Also report zero-memory and smaller-cap curves later. All learned parameters remain subject to the model-track cap. Report full model bytes and optimizer/checkpoint peak memory separately.

Do not equate “1,000 memories” across methods: a token record and two float vectors have different sizes. As an illustration, 1,024 pairs of width-192 float32 key/value vectors already occupy 1.5 MiB before metadata. Measure serialized and resident bytes, deduplicating shared tensors, rather than assuming this illustration is the final allocation.

The replay baseline uses a reservoir over **seen** examples, evicting to fit the byte cap. Proposed starting replay exposure is 20% of adaptation tokens, with its tuning cost logged and frozen on the calibration stream. Replay tokens, reference-model passes and consolidation count toward each method's budget. Additional memory cannot be filled with arbitrarily many seeds that regenerate unseen old data for free: the scored learner has access to observations it actually saw, not to the evaluator's generator. Regeneration of observed examples from stored seeds, if allowed at all, must be allowed to every method, byte-accounted and time-accounted, and reported as a distinct compression protocol.

A bank-only method can use gradients to compute a correction value without changing trunk weights; those backward passes still cost compute. Consolidation may use a frozen base plus its current bank as teacher; its extra forward passes count. A bank-reset test must actually remove the bank and its cached outputs before evaluation. Adapter proliferation is capped; do not store one unbounded adapter/model per phase.

Evaluation is read-only. Snapshot parameters, banks, routing state and statistics at each phase. Evaluate each held-out item from the prescribed reset state, with no label-dependent writes. For the secondary knowledge task, preserve the trained persistent store but reset ephemeral prompt state. Any predict-then-update online metric runs on a separate ordered stream and cannot alter the retention test model.

## 10. Evaluation and acceptance rules

### 10.1 Base reasoning

Report exact task success, valid-output rate, per-target NLL, and learning curves by family, task type and tier. Grid scores require all cells and dimensions correct. Programs require the correct final state. Planning uses the simulator. State tracking uses the canonical answer. Partial cell/token accuracy remains diagnostic.

The frozen primary score is:

`Reasoning = mean over 4 families [0.75 × ID ladder area + 0.25 × challenge area]`

Both terms are measured above floor (§7.4), with equal task-type weights within each family. The ID ladder area averages the primary ID ladders. The challenge area is the equal mean of the extension ladders' areas and the above-floor accuracies of the composition panel and the new-parameter panel (§5.2); where the grammar allows, composition is itself a ladder over composition depth. Each family's frontier L50 per ladder is reported beside the scalar, with its interval. Fix the ladder list before comparing architectures. If a ladder is uncalibrated, do not backfill it after seeing v13 results: revise and disclose the suite before the comparison. Language, external benchmarks, the anchors of §3 and mechanism probes are excluded from this scalar and reported alongside it.

Greedy single-output decoding is primary, with family/tier output caps fixed from the generator bounds, not the particular test answer length. Optional sampling/search gets a separate success-versus-time curve with the same allowance for all models. Do not compare pass@many with pass@1. When reporting official ARC, name the exact source version, split, permitted attempts and use of test-time adaptation; the suite's single-attempt grid score is not the published TRM protocol.

Use paired bootstrap intervals over independent semantic groups; average augmentations within a group before resampling. For external task suites, resample tasks. These intervals describe evaluation uncertainty, not variation across training seeds. Use at least two **paired** training seeds for a preliminary repeatability claim, three when practical. A single-seed result is exploratory even with thousands of test examples. Out-of-distribution panels need more. Across 100 fine-tuning runs of BERT, in-distribution accuracy stayed between 83.6% and 84.8%, while one out-of-distribution subset ranged from 0.00% to 66.2% ([McCoy et al., 2020](https://arxiv.org/abs/1911.02969)). The challenge component therefore supports a claim only with at least three training seeds per arm, and is labelled exploratory otherwise. Read on the base checkpoint, three seeds of two systems cost 3 × 12 h × 2 = 72 device-hours, 48 more than one seed each.

### 10.2 Continual learning

Let `A[i,j]` be exact accuracy on task panel j after learning phase i, and let `s(j)` be the stage when that task was first taught; base tasks have `s(j)=0`. Use task-balanced averages and publish the matrix, not just an aggregate. Keep a unique task registry: seeing the same skill again at S3 or S6 must not give it another vote in the final average. Phase-specific acquisition panels can still be shown separately.

| Measure | Definition / purpose |
|---|---|
| Acquisition | Immediate accuracy on new tasks, gain from their pre-phase score, and incoming examples/tokens/time to a fixed threshold |
| Final average accuracy | Mean `A[T,j]` over all learned stable task panels |
| Backward transfer | Mean `A[T,j] − A[s(j),j]` over eligible earlier panels; positive means improvement |
| Forgetting | Mean `max(A[k,j] for k=s(j)…T) − A[T,j]`; includes final time so it is nonnegative |
| Retention before return | S0/S1 accuracy just before S6, before replaying the returning distribution |
| Relearning | Exposure/time required in S6 to recover a fixed fraction of prior competence |
| Forward transfer | Pre-learning new-task accuracy and acquisition speed versus a matched control with no relevant prior skill exposure |
| Online adaptation | Separate predict-before-update performance after each change; labels revealed only after prediction |
| Consolidation | New-skill gain retained after bank removal, old-skill accuracy changes, memory and time |

For forward transfer, a fresh random model is a useful reference but confounds basic representation learning. A stronger control begins from the same base and receives equal intervening compute on unrelated permitted tasks. Report which control was actually run; do not invent a transfer advantage without it.

Two further readings (§14). *Plasticity:* over long task sequences, standard deep learning can lose the ability to learn. On ImageNet, Dohare et al. report accuracy falling "from 89% accuracy on an early task down to 77%, about the level of a linear network, on the 2000th task" ([paper](https://arxiv.org/abs/2306.13812)). Report each method's acquisition speed per phase. Six phases are far too few to show this effect, so a null here is not evidence against it. *Spurious forgetting:* in continual learning of language models, drops "often reflect a decline in task alignment rather than true knowledge loss" ([Zheng et al., 2025](https://arxiv.org/abs/2501.13453)). S6's relearning measure is the natural test: fast recovery from a small, fixed exposure points to lost alignment rather than lost knowledge. Report it beside forgetting rather than reading forgetting alone as loss.

Knowledge revisions from §8.2 are excluded from stable-task forgetting: the superseded current-value label is intentionally obsolete. Report update accuracy, stale-answer rate, unaffected-fact retention and context confusion separately.

### 10.3 Proposed success criteria

These are practical preregistration thresholds, to be locked after calibration and before architectural comparisons:

- **Base competence:** the architecture must acquire the working rungs of the ID ladders (§7.4). Publish failures; no interpretation of concept use can substitute for exact task success.
- **Continual advantage:** at matched incoming exposure and the same memory cap, the proposed method should be within 3 percentage points of the replay baseline's new-task acquisition and improve retained-old-task accuracy by at least 5 points, with paired uncertainty reported. Also report the full acquisition/retention tradeoff if this threshold is not met.
- **Related-skill protection:** show a benefit on S3's confusable counterparts, with no greater than a 3-point acquisition deficit. A win only on unrelated language versus grids is insufficient for the strong claim.
- **Consolidation:** retain at least 90% of the bank's new-task accuracy gain over the frozen base after resetting the bank, while old-task accuracy falls by no more than 3 points. The ratio is meaningful only if the bank improved accuracy by at least 5 points; otherwise report absolute changes and mark it inconclusive.
- **Efficiency:** show the same metrics at the actual 24-hour limit, including memory operations. A matched-example gain that disappears at matched time is a sample-efficiency result, not a compute-efficiency gain.
- **Replication:** positive outcomes need a second paired seed/order check before being treated as reliable. Multiple thresholds across many cells require disclosure; report all registered outcomes, not only successful ones.

These criteria permit the architecture to lose. If a plain transformer plus replay wins, that is the correct outcome of a neutral test.

### 10.4 Mechanism diagnostics, kept out of the winner score

Concept zero/mean/shuffle interventions, probes of intermediate representations, recurrence removal, bank retrieval attribution and switch alignment can explain results. They do not determine dataset membership or primary weights. Compare concept-output probes with capacity-matched input probes and ordinary transformer layers; predict held-out ground-truth quantities where feasible rather than only the model's own moving codes.

Randomly sampled prompts can make future token groups intrinsically unpredictable. If NCP is trained on them, record target entropy, copy/majority baselines and family-wise gradient norms. Do not manufacture a predictable four-token rhythm to make NCP look good. Restricting NCP to deterministic execution regions or changing its weight is a declared objective ablation, not an unannounced dataset repair.

## 11. A realistic 24-hour budget

### 11.1 Default: 24 hours per complete model/method trajectory

Assumption: the current M2 Air-class environment mentioned in the repository, using one accelerator training job at a time unless a throughput/memory measurement establishes a better schedule. This document does not launch jobs or assume their current availability.

Generator implementation, corpus verification and the shared calibration pilot occur **before** the scored 24-hour experiment and their costs are reported separately. With the LSTM range reference (§7.1), the pilot is two runs per track. This is a training/evaluation budget, not a claim that research, coding and all comparisons take a day. A complete two-system comparison therefore needs up to 48 device-hours plus calibration; extra seeds and ablations cost more.

| Per-trajectory budget | Allocation | Contract |
|---|---:|---|
| Base learning | Up to 12 h | Shared exposure fitted to the slower arm's capacity; frozen curriculum |
| Sequential learning | Up to 9 h | Six phase caps × 90 min, including any replay/bank writes/consolidation |
| Evaluation/checkpointing/overhead | 3 h | Intermediate checks plus final matrices; evaluation cost is measured |
| **Total** | **24 h** | Hard wall-clock envelope, not 24 h plus free sleep/search |

Within each 90-minute continual phase, reserve up to 75 minutes for incoming learning and 15 minutes for consolidation. For the primary equal-incoming-exposure stream, freeze each method's update/replay/consolidation schedule during calibration; faster completion is recorded as an efficiency benefit, not filled with unannounced extra updates. All methods may tune a valid schedule under the same overall cap. A separate equal-time stream can let methods without consolidation spend the remaining time on allowed adaptation/replay, with repetitions recorded.

Phase boundaries for matched-data scoring use the exact same ordered incoming examples; resource-limited methods must reach every phase for a full-stream comparison. Size the shared incoming exposure from the slowest method's measured **75-minute** capacity, leaving its consolidation reservation intact. Shrink the common phase exposure **before the scored runs** if needed. Do not let a slow arm silently skip later phases. Report both incoming tokens and the additional replay/teacher tokens: matching incoming experience does not match every internal computation.

Equal-data and equal-time continual trajectories cannot generally both be reconstructed from one run: extra updates change the starting state of the next phase. Choose **equal incoming exposure** as the primary stream, with a hard per-phase time cap; report its actual time. Run a separately initialized equal-time stream for a causal equal-time comparison when affordable. With only one stream, learning-curve checkpoints give descriptive resource tradeoffs, not a fully matched second experiment. Similarly, a same-data continual comparison must branch from base checkpoints trained on the same base exposure, not merely from each model's 12-hour endpoint.

Use throughput measured with the actual model, mix, sequence buckets, precision and optimizer. For base budget `h` hours and sustained **non-padding processed** rate `r` tokens/s:

`processed tokens = 3,600 × h × r`

| Illustrative measured rate, not a prediction | Tokens in 12 h of base learning |
|---:|---:|
| 1,000 tokens/s | 43.2M |
| 3,000 tokens/s | 129.6M |
| 6,000 tokens/s | 259.2M |

Supervised targets may be a small fraction of these counts. Publish both. Repeat/augmentation counts matter as much as nominal token totals. The old [V12 pilot](v12_tiny/pilot_bench.json) used different short sequences and models; its throughput is not a forecast for v13 at 1,024 tokens.

The 3-hour evaluation reservation is provisional. Time full-grid generation and simulator rollouts during the pilot. If the planned final matrix will not fit, pre-register fewer evaluation instances with wider intervals, or transfer evaluation to a separately accounted budget and call the claim “24 training hours plus evaluation.” Do not silently overrun or omit the hard panels.

### 11.2 If the entire available experiment is only 24 hours

Use a **screening study**, not the full evidence claim:

| Total shared allocation | Work |
|---|---|
| 2 h | Reference calibration (the transformer and the LSTM range reference, 1 h each) and actual-shape timing; stop if no ladder brackets a reference's 50% point |
| 10 h | Two bases, sequentially, 5 h each: plain TF and the chosen v13 arm |
| 8 h | Their matched continual streams, 4 h each over six shorter phases |
| 4 h | Evaluation, verification and report |

Data generation/verification still needs to be ready before this schedule. One seed, two systems and short adaptation cannot attribute every component or establish robust lifelong learning. Freeze/replay/bank ablations and repeats become follow-up experiments. Never present six 24-hour methods as a single 24-hour test.

### 11.3 Reduce scope before reducing validity

When compute is tight, prioritize a complete paired baseline/design comparison, the close-interference phase and exact competence metrics. Cut extreme stress sizes, additional α/loss sweeps, optional external test-time search, the secondary knowledge stream and extra diagnostics first. Preserve the same primary families and predeclared splits for both models. If even that is too much, register a reduced suite openly before evaluation.

## 12. Implementation handoff and required validation

This is the order for a later implementation task. No files named below are claimed to exist yet.

1. **Freeze the generator contract:** define the DSL operations, transition semantics, S0/S2 primitive registry, permanent and temporary composition holds, difficulty axes and renderer variants. Implement exact solvers independently of neural training.
2. **Create candidate data:** export manifests, small inspectable examples and diagnostic distributions before large shards. Verify every target. For imported sources, preserve attribution, source split identities and license conditions. TinyStories' dataset card lists CDLA-Sharing-1.0; RE-ARC's repository is MIT; imported PGM data has a separate non-commercial restriction. A code license does not automatically license every imported data asset.
3. **Audit splits:** test byte-identical, normalized and augmentation-group overlap. Inspect semantic equivalences for the bounded grammar. Keep source task IDs. Confirm no evaluator-private field reaches the model.
4. **Validate serialization and targets:** grid shape round-trip; graph relabelling; object and colour permutations; all legal planning solutions accepted; incorrect goals rejected; arithmetic overflow/carry cases; state updates; EOS and padding masks; no cutting of demonstrations.
5. **Validate state isolation:** change an earlier packed document and check that another document's predictions are unchanged. Confirm all recurrent memories/caches/codebook update masks obey their declared training/evaluation boundaries. Evaluate twice in reversed item order to catch hidden writes.
6. **Validate optimization:** unique parameter ownership across optimizers, accurate total-parameter count, meaningful tiny-set overfit, fair shared tokenizer and exposure logging. The prior review found shared-embedding optimizer overlap in unrun V12 NextLat arms; do not carry that issue into a new benchmark.
7. **Run the reference calibration:** publish per-rung accuracy, shortcuts and learning curves for both references (§7.1), and the ladder endpoints they set (§7.4). If either reference fails to learn, identify whether the generator, training or capacity is the cause before calling the task impossible.
8. **Lock the scored protocol:** primary family weights, test hashes, seed/order plan, memory cap, byte accounting, feedback rules, consolidation access and measured time allocations.
9. **Run paired comparisons:** first architecture/system performance, then same-base method attribution as resources allow. Include the simple replay method before spending the budget on many specialized variants.
10. **Release the report:** exact accuracy curves, full continual matrix, acquisition/retention tradeoffs, per-family loss, parameter/memory/compute counters, all failed gates, calibration costs and unrun experiments.

Suggested future artifact layout:

```text
datasets/v13/
  SPEC.md                  frozen copy of this specification
  sources.json             revisions, attribution, normalization and licenses
  primitives.json          legal operations and stage availability
  split_manifest.json      semantic groups and permanent/temporary holds
  difficulty_manifest.json baseline calibration decisions
  tokenizer/               train-only tokenizer and reserved-symbol contract
  calibration/             disposable pilot panels and acceptance records
  train/                   procedural shards or deterministic stream manifests
  dev/                     readable development panels
  test/                    sealed evaluation targets and semantic-group IDs
  continual/               ordered incoming streams and read-only evaluation panels
  audit/                   verifier, leakage, ambiguity and state-isolation results
```

The scored training process should not have access to test targets or the unlimited historical generator. Sealing is an access/protocol property, not merely naming a directory `test`.

## 13. Review of this specification

**Review type:** author self-review after drafting, with source checks, local-evidence checks, arithmetic checks and a consistency pass. This is not an independent peer review and not empirical validation of the proposed dataset.

| Review question | Finding and revision |
|---|---|
| Does it favour v13 by making concepts necessary? | Removed architecture-relative distance thresholds from task selection. Equal 22.5% training-token shares across reasoning families; equal family score weights. Concept/recall interventions remain diagnostic |
| Does “small model” guarantee the tasks are achievable? | No. Added disposable baseline calibration, explicit admission bands, overfit checks and an uncalibrated status. No achieved v13 score is claimed. *Admission bands superseded by §7.4's ladders (§14)* |
| Does a grid result become an ARC claim? | No. Bespoke grammar, same-generator RE-ARC, held-out rules and official ARC are different panels; exact output and attempt budgets are explicit |
| Are later “new” primitives already in base training? | The broad generator grammar initially risked this. Added an explicit S0/S2 availability table and separated canvas-preserving selection from crop |
| Does the stream test a real distribution shift? | Added a requirement that S4 rendering families be genuinely held out; another colour permutation already covered by augmentation is only a control |
| Can changed rules demand incompatible labels? | Stable-skill tasks have explicit request semantics. Contextual facts and current-value revisions have separate contracts and metrics |
| Can retention be won by never learning? | Acquisition, frozen baseline, eligible-retention labels and the complete accuracy matrix are required; NLL alone cannot pass |
| Is continual learning confused with in-context recall? | The primary stream changes learned skills; the secondary factual stream resets prompt state and measures persistent storage separately |
| Is the memory comparison fair? | Byte cap, optimizer/model accounting, seen-data-only replay, consolidation cost and no oracle phase router are explicit |
| Can one trajectory prove both equal-time and equal-data gains? | Not generally. The draft was tightened to require separate trajectories for causal comparison; checkpoints alone are descriptive |
| Is the 24-hour plan honest? | It is a per-trajectory cap with setup/calibration disclosed; a separate total-24-hour screening option states its weaker evidential scope |
| Are uncertainty and non-results visible? | Test uncertainty is separated from training-seed variance; stress results, failed acquisition and all registered outcomes remain reportable |

**Remaining empirical questions:** which candidate cells fall in the working band (superseded by §7.4: where each model's frontier lies); throughput at the actual sequence lengths; how many grid rules remain after ambiguity filtering; whether a 2,048-token vocabulary adequately supports the language guard; how much phase-order variance exists; whether any fast-memory advantage transfers from exact-key storage to new skill instances. These require the later implementation and pilot, not stronger wording in this document.

Document checks passed: local source links resolve, the illustrative JSON parses, table column counts and code fences are consistent, training mixture sums to 100%, both displayed time budgets sum to 24 hours, and the parameter/memory/token arithmetic was recomputed. The research section contains 23 distinct primary paper, repository or dataset links (it holds 28 after the §14 additions). These checks validate the specification's internal presentation, not a model's ability to solve the proposed data.

**Decision:** proceed with this as an architecture-neutral candidate specification. Treat the reference-model pilot as the gate for declaring a dataset capacity-appropriate (since §14, the pilot sets ladder endpoints rather than admitting cells). The first claim to earn is reliable exact competence on fresh working-tier tasks; the second is acquisition with retention under a fixed stream and resource budget.

## 14. Second review and extension (14 October 2026)

**Review type:** a second review by Claude (Anthropic), after the self-review in §13. Every source below was fetched during the review, and three advisor consultations were held: before the review, after the source checks and before writing, and before completion. This is still a specification: no corpus has been generated and nothing has been trained.

**Verdict on the original.** The neutrality contract, exact verifiers, split design, continual stream and byte-capped memory accounting are sound, and are kept. One defect bore directly on the question that motivated the suite: is it too easy or too hard for a sub-10M model? Difficulty was admitted by one transformer's accuracy band after a two-hour pilot, then scored after up to twelve hours of training. Cells admitted in the band would likely saturate by then, admission selected on the pilot's noise, and the range ended where that transformer failed. §7.4 replaces admission with ladders.

**What changed, and where.**

| Change | Where | Why |
|---|---|---|
| Difficulty ladders, with an above-floor area and a logistic 50% frontier, replace admission bands; the primary scalar uses ladder areas | §7.2–§7.4, §6.2, §10.1, §11.2 | wherever a model's capability lies, some rungs are informative for it |
| An LSTM range reference beside the transformer | §7.1, §11.1 | a range set by one architecture ends where it fails |
| Inductive-bias tags, reported as subscores | §5.7 | makes the suite's tilt visible without moving weights |
| Parametric rules, a new-parameter panel, a fixed checkpoint curve, one-row rungs | §5.2, §6.1 | separates reading the demonstrations from recognizing a stored program |
| Answer format and track frozen before calibration | §5.3 | format shifts difficulty as much as parameter count |
| Published small-model anchors as harness checks | §3 | tells a hard task from a broken harness |
| What the sub-10M ARC numbers measure | §3 | they are per-puzzle, transductive results, not comparable with this suite |
| A ranking holds for its training protocol | §1 | rankings can change with the protocol |
| At least three seeds for challenge claims, with the cost | §10.1 | out-of-distribution seed variance is large |
| Scenario, plasticity and spurious-forgetting readings | §8.1, §10.2 | states what the stream measures and what it cannot |

**Neutrality check on these additions.** Each was tested against one question: would it be chosen differently if v13 had no recurrent block, no concept path and no confined decoder? None would. The LSTM reference rests on the transformer's published failures on regular languages, not on v13's GDN-2 loop. The tags come from the literature and the task definitions. The copy-heavy, long-distance cells, where a bounded-reach design should show its cost, are kept and tagged.

**Sources checked in this review.**

| Claim | Source fetched | Status |
|---|---|---|
| TRM: 7M parameters; 44.6% on ARC-AGI-1 and 7.8% on ARC-AGI-2 with two tries; 1,000 augmentations; ConceptARC's 160 tasks; per-puzzle embedding; majority vote; about 3 days on 4 H100s | [paper](https://arxiv.org/html/2510.04871v1), [repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | verified |
| TRM's ARC build includes the evaluation subset | the repository's build command | verified as a flag; TRM's text does not say whether evaluation demonstrations enter training |
| HRM trains on the 400 ARC-AGI-1 evaluation tasks; 32% and 2% on the semi-private set | [ARC Prize analysis](https://arcprize.org/blog/hrm-analysis) | verified |
| Chomsky-hierarchy configuration and budget | [paper](https://arxiv.org/html/2207.02098v3) | verified |
| CLRS-Text trains Gemma 2B on all 30 tasks; models "barely extrapolate" | [paper](https://arxiv.org/html/2406.04229v1) | verified |
| Licences: TinyStories CDLA-Sharing-1.0; RE-ARC MIT; PGM non-commercial research only | dataset card and repositories | verified |
| Lee et al.; Nye et al. (MIT and Google Research); Chan et al. (DeepMind; about 831K parameters); Singh et al.; Amos et al.; METR; Fluid Benchmarking; McCoy et al.; Arora et al.; Saparov et al.; Power et al.; van de Ven et al.; Dohare et al.; Zheng et al.; Merrill et al.; Grazzi et al. | linked where cited | verified as quoted |
| 1D-ARC: 18 task types, MIT licence | [repository](https://github.com/khalil-research/1D-ARC) | verified; task count and array sizes **not verified** |
| SCAN's split accuracies, MQAR's state-size curves, a recovery-with-little-data result in Zheng et al. | — | **not verified**; none is used as a number |

Google and DeepMind work relevant to the original request: Mathematics, CLRS and CLRS-Text, the Chomsky-hierarchy study, PGM and CFQ (already in §3), plus Nye et al.'s scratchpads and Chan et al.'s bursty in-context learning, which this review adds. Reconciling this suite with `v13_NOTES.md` §12 (its data mix, parameter accounting and tokenizer) is left to the user, as §2 already says.
