# V7 tiny protocol, frozen before the long queue

## Question

Does a jointly trained encoder/concept/decoder with next-concept forecasting improve quality or cost over an ordinary Transformer of the same architectural size? Does the additional refit/prior-control recipe create useful, efficient concept-space repair beyond token retry? Does AB allocation improve on the same proposal kernels under simpler allocation?

Each question receives its own comparison. A search gain does not establish NCP benefit; passing algebraic checks does not establish model quality.

## Hardware, sizes and budget

One Apple M2-class GPU with 16GB shared memory, FP32, context 384, no dropout. V7 has 5,365,938 foundation parameters; the ordinary twelve-block Transformer has 5,366,208. Two independent control adapters add 41,472. The baseline FFN widths are eleven blocks of 464 and one of 448. All blocks own distinct parameters.

The queue ceiling is 24 hours from launch, reserving the final six hours for sealed testing/reporting. Warmed batch/compilation choice is locked from `runs/calibration/benchmark.json`. The common foundation target count is frozen **before pilot outcomes** as `floor_to_million(clip(min_measured_foundation_rate * 4500, 16M, 40M))`; the final number appears in `runs/plan.json`. Each arm uses that identical scored-target count. Completed targets and wall time are logged. Partial endpoints cannot establish a matched-completion comparison. Model comparisons are token-matched; actual compute differences remain visible.

Compiler acceptance requires eager/compiled gradient relative error below 0.003, plus cache agreement. Full MPS SDPA-backward compilation was rejected after wrong K-normalization gradients. Attention backward stays eager inside otherwise compiled training; static cached inference is separately checked.

## Locked partitions and outcomes

Fresh FineWeb-Edu documents are exact/lexically deduplicated before random document splitting, with training/dev/test hashes retained. Development and test contain 1,500 documents each. Primary text evaluation takes one initial block of at most 384 tokens from 1,024 test documents, selected by a fixed seed. NLL is token-weighted; uncertainty resamples documents, not tokens. This is short-context held-out likelihood, not a natural-language generation-quality evaluation.

Symbolic training is streamed, excluding canonical evaluation instances. Training mixes 60% GraphNav, 20% pointer following, 10% lookup and 10% two-step composition. Node counts advance from 10 to a 10/14/18 mixture during the first 15% of the budget. Training response masks exclude prompt likelihood; verifiers demand exact values or complete legal paths and terminators. Hard/long references are constructed with BFS, but inference receives only the public header and instruction.

Sealed evaluation contains 512 standard and 512 hard navigation cases, plus 256 each for long navigation, pointer, pointer-long, lookup, reversed layout and composition. Exact item independence and shuffled labels do not establish graph-isomorphism or generator-family independence.

## Foundation selection and confirmation

Development pilots use 3M scored targets each, separately trained from scratch with seed 970:

- NCP maximum alpha 0.1 with LR 0.0005/0.001/0.002.
- Ordinary Transformer with the same LR grid.
- NCP alpha 0.01 and 1 at the selected NCP LR; alpha-zero and plain-routing alpha-zero/0.1 pilots at that LR.

Select the lowest equal-domain average of fresh text NLL and standard symbolic token NLL. Numerical pilot rejection is recorded; programming failures stop the queue. No test results choose a setting. The full NCP, alpha-zero hierarchy and ordinary Transformer are then trained at the common target budget on seeds **171, 272, 373**. Paired data streams are shared across those three architecture arms within each seed. Independent foundation seeds, rather than extra search seeds, define the repeatability scope. The alpha-zero control uses the NCP-selected LR to isolate the loss; the ordinary Transformer receives its own equal-size LR tuning allowance.

The primary NCP comparisons are against both alpha-zero and the ordinary Transformer on text NLL and strict standard success. Require text preservation upper 95% CI <=0.02 nats/target and standard success lower difference CI >=-2 percentage points, plus evidence of a text or task improvement. Forecast skill alone is insufficient.

## Refit, exploration and transplant

For each NCP seed, copy and train only its concept core for 8M targets, with the surrounding encoder/decoder/head/vocabulary frozen. A port distillation term starts at one, then decays to zero by 30% of the refit budget. Accept the full refit only when paired text preservation meets 0.02, standard symbolic NLL degradation is <=0.02 and paired strict-success preservation meets -2 points. A rejected refit remains published; exploration anchors to the original foundation instead.

Train only prior-control adapters for 6M targets, two whole suffix schedules per example, uniformly sampled window lengths 1/2/4 and beta=0.25. Text rows contain a single document/suffix for coherent winner selection. Freeze and hash the deterministic model. Controls use projected Gaussian draws, RMS radius two, and a smooth write cap 0.1. No future-conditioned posterior is used.

Seed-171 controls receive the same 6M additional target allowance: direct adapters on the foundation, mean-only candidate training (beta=1), best-only candidate training (beta=0), deterministic core continuation and full-model continuation. An alpha-zero refit/exploration lineage is also attempted with identical stage budgets and the same preservation rule. These controls have different trainable parameter counts and compute; report both. All-zero parity, full transplant endpoints, cache replay, finite gradients, codebook/forecast statistics and deployable core merging are audited on the trained checkpoints.

## Repair calibration and sealed comparisons

Development prior coverage is measured at zero and after 1/2/4/8 controls for window lengths 1/2/4. Lock the best hard-task eight-draw coverage window, preferring the earliest grid entry on ties. Known-valid development trajectory optimization (16 cases, 32 updates) is a privileged reachability diagnosis, never credited as ordinary search.

All timed methods first attempt the same zero-control greedy answer and stop at verified success. Timing starts before control allocation/generation, after small CPU argument/RNG setup. Total per-search elapsed time and any deadline overrun are disclosed. Only complete answers verified before their deadline get success credit. Generation, prefill, verification, control gradients, branch construction and cache operations consume the allowance. Compiled warm-up is measured separately and disclosed.

Development retry grid: temperature/top-k (0.6/unrestricted), (0.8/128), (1.0/unrestricted). GradCuit grid: LR 0.01/0.03/0.1 with KL zero, plus LR 0.03 with KL 0.05. Sixteen hard cases select each setting at the middle budget; deterministic ties prefer smaller temperature/LR. All proposals are capped at 16 generations including the common first answer. The wall-time curve uses 4/8/16 times measured median warmed single-answer duration (at least 0.25 seconds per budget). Comparison on 32 hard development cases includes retry, latent sampling, combined sampling, selected GradCuit, AB-prior, random chain, AB-hybrid and its random/chain allocation controls. All test settings are frozen in `runs/inference_lock.json`.

Primary sealed search evaluates retry, selected GradCuit, AB-hybrid and its two matched-kernel allocation controls at the middle budget on all 512 hard cases for each available NCP lineage. The ordinary Transformer receives its own equal-size development retry calibration and sealed retry on the same items at that absolute wall-time budget; compare V7 search against it as well as against V7's own retries. Exploration's extra 41,472 parameters are disclosed in that system comparison. Seed 171 also receives a 128-case full cost curve and a 256-case standard preservation comparison. The alpha-zero lineage receives a 128-case diagnostic search comparison.

The 80% strict standard development competence gate controls interpretation and search allocation. If the seed-171 NCP foundation is below it, NCP search and Transformer retry are limited to 64 hard items per seed, with a 32-item cost curve and 64-item standard check, all labeled diagnostic. This predeclared reduction avoids spending most of the remaining budget on repairing an unestablished foundation. The full sealed foundation likelihood/task evaluations still run. Search does not become a successful primary repair benchmark because a diagnostic point estimate improves.

Require a >=3 percentage-point absolute gain and paired 95% difference interval above zero for a claimed search improvement, with standard preservation. AB must also beat the matched-kernel simple allocation controls. Per-model paired intervals are conditional on that model; the three independent trained seed outcomes and their spread are listed separately. An interval crossing zero is inconclusive. No retuning follows test exposure.

## Stop, recovery and omissions

The driver is serial and refuses source/specification/data changes after lock. Every stage stores its latest finite atomic recovery checkpoint, losses, target count, actual timing and failure traceback. A failed main stage stops the queue and publishes its checkpoint location. Training runs have internal time limits; later stages may be skipped to preserve final testing. Testing that reaches the deadline is labeled partial. The monitor observes progress and never changes source or jobs.

Long-context training, larger width/depth, distributed/BF16 validation, full V5/V6 bridge, paper routing reproduction, control-dimension sweeps, a matched token-control site, general variable-binding language tasks, retrained port-only ablations, fractional-feedback search, AB-MCTS-M and prefix/latent-dynamics MCTS remain outside this 24-hour experiment. Their absence limits claims and is not silently treated as a passed test.
