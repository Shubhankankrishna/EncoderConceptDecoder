# Paper review and method translation

Reviewed sources: the root `NCP-ArchPreview Technical Report.pdf`, `Loop Papers/GradCuit.pdf`, `AB-MCTS.pdf`, the full V7 design, and the V5/V6 implementations and completed raw outcomes that motivated it. The main architecture/method/objective sections and relevant appendices were read; NCP architecture/objective/routing figures and the GradCuit and AB method pages were visually checked. The older paper library was screened during the repository review recorded in the V7 draft. This document distinguishes paper evidence from this experiment's adaptations.

## NCP-ArchPreview

The report separates token encoding, concept pooling/computation, product-quantized forecast and token decoding. Next-token loss retains fine detail; next-concept loss predicts a detached representation of the next complete token chunk; vocabulary fitting trains codebook rows against detached concept targets. Soft forecasting supplies a differentiable path through the concept head and vocabulary. IRC and CRC give additional routes across depth and modules.

The relevant distinction is gradient ownership: detaching the next concept target does not detach the current/source encoder representation. NCP can train the source encoder, concept core, forecast head and codebook through the prediction. The VQ fit alone trains the codebook. V7 tests these routes explicitly. Its NCP loss is a **feature mean** over valid, same-document neighboring chunks; its VQ fit also uses a feature mean. The paper presents both feature-sum and algorithmic mean expressions, so scalar coefficients cannot be assumed interchangeable. The schedule is an adaptation: zero NCP for the first 5% of targets, ramp to a development-selected maximum by 20%, then hold. Codebook initialization/restarts use training-only concepts and reset replaced rows' optimizer moments.

The report's full signed/depth-wide IRC and CRC are not reproduced. V7 uses fixed-size softmax banks and bounded writes. Per-head Q/K normalization is also a stability choice shared with the baseline; the report's principal results used layer-wise Q/K normalization. The paper's billion-parameter/trillion-token results are motivation, not validation of this tiny configuration. At depth 3/4/3, completed four-block IRC groups are not subsequently consumed, so this run principally tests initial/current routing and cross-module routes.

Historical evidence constrains the hypothesis. The positive V5 NCP control and the reversed V6 matched-prefix control used different interfaces, objectives and budgets. V6 removed parts of paper routing. The completed V6 GradCuit-versus-retry comparison did not establish a significant GradCuit win. V7 therefore retains full alpha-zero and ordinary Transformer controls, plus frozen-refit and direct-control comparisons. It does not assume that all of the prior recipe is beneficial.

## GradCuit

The paper optimizes an inference-time latent prefix at an intermediate attention layer while freezing model weights. Accepted/rejected feedback drives the local update. Its documented implementation uses greedy generation, sequence-summed log probabilities, an Adam learning rate of 0.001, and regeneration after each update including the last. Its inference equations and experimental implementation must be read together; a superficially similar latent update is not automatically the same estimator.

V7 changes the intervention to two bounded concept-stack controls, trained from an inference-available Gaussian prior. It uses response-mean log probability, per-problem clipping, an RMS control ball, a declared development LR grid and an optional reference-KL arm. On rejected generated answers it descends their mean log probability, then regenerates and verifies the actual answer. This is a **GradCuit-style local heuristic**, not a claim of unbiased policy-gradient estimation or exact paper reproduction. Successful items stop; model weights and accepted items' optimizer states remain fixed. Generated termination tokens are scored, prompt/padding likelihood is excluded, and actual generated tokens refresh every next concept.

The derivative at all-zero controls must remain nonzero; skipping the adapter arithmetic at zero would break this. Teacher-forced low loss is insufficient: free-generation prior coverage and a separately labeled privileged valid-target diagnostic check whether useful controls are reachable. Privileged labels never enter ordinary search results.

## AB-MCTS

AB-MCTS-A uses adaptive branching and Thompson sampling to choose broad generation or continued refinement. Its GEN, CONT and answer statistics collect different observations. For a new direct child, the parent's GEN receives that reward; the parent's answer receives it; strict ancestors receive answer and CONT updates. The immediate parent's CONT excludes this direct-child observation. Beta priors are initialized at 0.5/0.5 for binary outcomes. A fixture verifies these exact backup rules.

V7's tree is over **realized complete answer/control refinements**, not imagined latent transitions. Nodes own their schedule, actual generated tokens and separate cache state. Fresh projected Gaussian controls, local perturbations, suffix resampling and one-step gradient refinements are proposals. AB selection competes with random-node and chain allocation using the same kernels and depth cap. Standalone GradCuit retains its own sequential optimizer moments; hybrid children reset their local optimizer.

Primary reward is strict complete-answer acceptance. Every failed answer scores zero before the first success, and search stops at success. This means AB has no dense correctness signal among early failures; possible gains can arise from diversification/counts rather than learned semantic progress. The algorithm has no hidden shortest-path oracle. Prefix-concept MCTS, learned latent dynamics, AB-MCTS-M and learned value models are not implemented here.

## First-principles review

1. Detail access remains causal and available to the decoder; concept compression does not discard the only exact-fact route.
2. The next-concept target is correctly detached while source gradients remain live. Domain reductions, padding and document boundaries are explicit.
3. Freezing surrounding weights stabilizes the refit interface, but preservation and matched continuation controls decide whether refitting is useful.
4. Exploration comes from an inference-available prior, with one coherent winner per document suffix. Zero controls recover the accepted deterministic anchor exactly.
5. A transplant replaces both forecast and continuous ports. Lambda-zero recovers the original foundation; lambda-one/zero-control recovers the selected deterministic refit.
6. Repair success comes from actual token generation and a public verifier. Backward passes, retries and verifier work consume the same wall-time allowance.
7. A positive point estimate with an interval spanning zero is inconclusive. Failed gates, rejected pilots, skipped work and partial endpoints are published.
8. Distinct layer parameters and fixed-size source banks allow ordinary capacity growth. A 5M-model experiment cannot establish empirical neural scaling laws.

The implementation is therefore a testable subset of the V7 research programme. It preserves the central staged mechanism and tests its decisive alternatives, while recording the missing larger-scale and broader ablations in `COMPLIANCE.md`.
