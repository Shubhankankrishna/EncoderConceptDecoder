# V6 tiny development experiment

Frozen before the long training launch on 4 October 2026. The precise implementation/design hashes and launch time are stored in PLAN.json and each stage configuration.

## Question

Can a competent deterministic core expose useful bounded Z alternatives, and can verifier-guided control updates beat token retries after their extra computation is charged?

Primary architecture: 3/2/3 NCP foundation, d=128, four heads, FFN=384, vocabulary 2,048, chunk size four, eight PQ segments with 32 codes each, R=2 and nL=2. Two context-conditioned adapters have intermediate width 16 and control dimension eight. Injections vanish exactly at zero, use a 10% input-RMS write cap and an RMS(Z) bound of two. No posterior, internal epsilon or learned switch is introduced.

## Data

- Text: 133,553,935 local TinyStories training tokens; 529,875 EOS-delimited training documents before duplicate accounting. Tokenizer is inherited unchanged. Evaluation documents are exact-token deduplicated against training and each other, then split into development and test documents with seed 60401. Uniform document sampling and chunk-aligned packing replace V5's unconstrained random token windows; this is a declared recipe difference.
- Symbolic: GraphNav with ten nodes, three colored transition functions, shuffled public headers and goal-conditioned walks. Strict public verification checks transitions and every token of the stopping chunk. A BFS solver validates reachability and supplies privileged diagnostic targets only.
- The 256 standard/512 hard instances in each development/test split are created before training. Their canonical graph/start/goal fingerprints are disjoint and excluded from every training stream. Graph training fingerprints are retained for audit. Text near-duplicate overlap remains unmeasured.
- Each update has 32 text and 32 graph rows, T=192, totaling 12,288 input slots. Record valid inputs, scored targets, unique text documents and symbolic instances separately. Foundation/replacement losses are token-balanced within each domain, then 50/50. Adapter suffix losses average eligible rows within domain, then 50/50.
- Complete single-document chunks only enter VQ initialization/loss/usage/restarts. NCP pairs require complete source/target chunks from the same document. Attention and plan shifts respect document boundaries. Codebook replacement resets the corresponding first/second optimizer moments.

## Training queue

| Run | Planned training | Intervention |
|---|---:|---|
| A_ncp_s0 | 18,000 updates | Fresh NTP + NCP + VQ foundation |
| B0_det_s0 | 6,000 | Frozen foundation; deterministic core; scheduled initial port distillation |
| B1_gated_b025_s0 | 6,000 | Frozen foundation/core; gated adapters; K=2 coherent prior schedules; beta=.25 |
| B1_direct_b025_s0 | 6,000 | Two direct bias-free Z lifts; otherwise matched protocol |
| B1_gated_b0_s0 | 6,000 | Gated adapters; winner-only beta=0 |
| B0_extra_compute_s0 | Calibrated equivalent B1 training time | Additional deterministic training; no adapters |
| A_no_forecast_prefix_s0 | 9,000 of an 18,000-step schedule | Alpha=0; compared with the saved 9,000-step NCP prefix |

Foundation/core AdamW peak LR=.002, 300/200 warmup updates, cosine to 10% of peak, matrix weight decay .1 and clipping at one. Foundation codebook LR multiplier is ten. Adapter peak LR=.001, 200 warmup updates, cosine, zero weight decay and clipping at one. The forecast-prefix control retains the full foundation schedule and restart horizon so its first 9,000 updates match the NCP prefix apart from alpha.

Two candidates share identical prompt boundaries and window assignments. Training window length is sampled per row from {1,2,4}. A whole suffix selects one complete schedule; no independent window-winner stitching. Zero control is logged as the reference and never allowed to suppress the adapter gradient by winning the pool. Teacher-forced training is not a guarantee of useful free-generation diversity.

The extra-core step count is selected from warmed throughput before outcomes are observed. Actual wall time and updates are reported; this is approximate compute matching, not an assumption of exact equality. All data/recipe changes apply to matching controls.

Before B1, screen B0 against its foundation on 128 development text/standard instances: both NLL gaps ≤.02 and standard success loss ≤2 points. These are mean screens, not confidence-bound confirmation. If it fails, allow one declared 1,500-step deterministic continuation with a fresh optimizer, no port distillation and the same training protocol, then screen again. Record all additional cost. If it still fails, skip dependent adapter/search experiments and report the failed prerequisite. This single recovery limit is fixed before launch.

## Budget and stability

Target approximately six hours, maximum approximately seven, with the final 90 minutes reserved for evaluation. Thermal drift is tracked in long-run logs. A stage reaching its allocated budget saves a finite checkpoint and records its shorter training; planned prefix completion is distinguished from an unexpected time limit. Evaluation can also be partial and must say so. No idle waiting is added to manufacture a duration.

FP32 is the executed precision. Benchmark the compiled trainer and fixed-size cache kernels against ordinary losses, gradients and generation. Vectorize both candidates into one retained computation graph when measured memory permits; the original-example batch and objective denominators remain unchanged. Score-then-replay is an equivalent lower-memory option. Encoder features are computed once per candidate set and once per rejection replay/KL pair. Model/candidate/optimizer state is released with each subprocess to prevent long-run cache accumulation.

Atomic checkpoints include weights, optimizer states, global step/configuration, codebook state, sampler state, Python/NumPy/Torch RNG and the control-generator RNG. Learning-rate state is a pure function of the saved step and immutable schedule. Resume must reproduce the next batch, controls and update. Invalid loss/gradients block the optimizer update and record failure; automatic invisible LR recovery is forbidden.

## Evaluation

Development: 64 standard and 128 hard instances for each control; main likelihood includes up to 192 text/standard rows. Held-out primary and full main comparisons: 256 standard and 512 hard instances, batch 16, paired identities across policies. A larger independent-corpus/three-foundation-seed confirmation is deferred.

Methods: zero-control greedy, temperature-one token retries, independent prior draws, shared-offset rejection repair, local-window rejection repair, local repair with KL=.05, random updates at a fixed development step scale, and random updates matched to the online gradient/Adam proposal norm. The matched-random arm pays for its gradient computation and logs actual post-projection norms. Repair has ten updates and eleven candidate slots; every updated candidate is regenerated and verified.

Each policy uses the same first zero-control greedy answer. Adam operates on current-problem controls with LR=.1, betas (.9,.999), epsilon 1e-8, no decay, per-problem gradient clipping and masked future/accepted moments. Repair minimizes rejected-sequence log probability plus optional forward KL from the detached zero-control distribution on the same rejected prefixes. It is an RL-style greedy rejection surrogate, not an unbiased stochastic policy-gradient estimator.

Primary method is fixed as gated beta=.25 with KL=.05. Choose the strongest B0/extra-core retry checkpoint on development hard success. Lock its mean development eleven-candidate retry time per batch before reading test results. At that same elapsed-time limit, compare that checkpoint's retries with the fixed V6 repair method; an attempt finishing after the limit cannot count as an accepted answer, and its incomplete/late cost still counts. Report the separate eleven-attempt comparison and complete success/cost traces.

Compilation warmup is fixed and measured separately for each model/execution mode. Within evaluation, method order rotates across batches. Per-item records include every attempt, output, verifier reason, token count, cumulative batch time, backward/replay counts and update norms. Shared batched timing describes this batch setting, not isolated per-user latency.

Extra diagnostics: transplant fractions {0,.25,.5,.75,1}; prior windows one/four; matched anchor/suffix NLL; distinct verified trajectories from four prior draws on up to 64 hard problems; two privileged starts ×25 updates on at most 16 development failures; rejection-gradient contributions by token type; bounded-prior interpolation; text continuations; post-training exact frozen/zero-anchor audits.

## Interpretation

G0 software contracts can pass in this run. G1–G5 quality means and paired item-bootstrap intervals are reported. This one foundation seed cannot satisfy the draft's independent-training-seed confirmation or its complete promotion gates. Do not equate oracle fitting, reduced rejection loss, increased colour persistence or a large Z-swap rate with deployment success. If repair loses to retries at equal time, report the loss directly. Width/context scaling, low precision and multi-device stability remain untested.
