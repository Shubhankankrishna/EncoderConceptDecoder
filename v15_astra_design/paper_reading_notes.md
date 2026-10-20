# V15 Astra — Literature reading and adaptation notes

Prepared 16–17 October 2026 for [V15 Astra](../V15_Astra.md). All nine supplied PDFs were consulted, with method/equation reading, experimental and appendix checks, and visual inspection of selected pages. They total **266 PDF pages**. This is a technical synthesis, not a reproduction of every derivation or an independent replication of the papers' experiments. Historical work was examined through the V1–V14 audit trail, with particular attention to V9–V14 reports/results and the two earlier V15 proposals.

The local PDFs, including their annotations and revisions, are the reviewed copies. Their SHA-256 fingerprints are in [source_manifest.json](source_manifest.json). Public links identify the papers; a later revision at an unversioned URL may differ from the local copy. Page numbers below are PDF page positions. Full-text extraction was used for navigation; figures/equations with difficult extraction were inspected as rendered pages. Bibliographies and every proof have not been independently verified.

## 1. Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation

**Primary source:** [arXiv:2610.14162](https://arxiv.org/abs/2610.14162). Local: `V15-Papers/Next-Concept Modeling.pdf`, 40 pages. Main focus: §§3–4, PDF pp. 5–12; objective/accounting and configuration appendices pp. 21, 23–24; tokenizer/level analyses pp. 29, 31, 35. Equation page 6 was visually inspected.

**Problem formulation.** NCM learns a hierarchy of discrete representations before learning its generator. The codec is a multiscale residual VQ-VAE. The encoder emits a full-length latent sequence. Each level downsamples the residual left by earlier levels, quantizes it, upsamples selected code vectors, and adds their contribution to a cumulative reconstruction. Later levels have more positions. The final level reaches token resolution. Its decoder produces token distributions from the cumulative reconstruction.

The generator is autoregressive over **levels**, while positions within a level are conditionally independent. The hierarchy can therefore express dependence through earlier coarse choices while predicting a fine level in parallel. This factorization is central: it is not ordinary token autoregression under a new name, and it is not a continuous next-state MSE target. The decoder remains part of generation after the codec is frozen.

**Training details that matter.** Nested level dropout teaches the decoder to use partial hierarchies. Commitment and codebook training keep quantization usable. The paper also studies corrupted earlier-level conditioning to improve robustness. Generator training uses a prefix/level masking scheme: observed prompt information is available, but the target level is not simply fed its own labels. A weighted sum of code losses can be an optimization choice; an exact probability claim requires the actual normalized product distribution.

The appendix distinguishes generator-only model scaling from the codec's parameters, and counts exposure in both stages. For example, LM1B uses about 6.80B codec and 6.55B generator token positions, totaling 13.36B. “Two stages” does not mean the data are seen only once. The configuration table includes **zero encoder Transformer blocks** for its listed tokenizer presets and substantial decoder/embedding capacity; adding a four-block encoder and Depth Memory is an Astra hypothesis, not a requirement copied from NCM.

**Results and limits.** The paper reports strong generation-speed/quality tradeoffs against diffusion baselines. Unconditional quality uses external-model generative perplexity plus sample entropy, not exact model likelihood. Conditional translation/summarization use task metrics, sometimes multiple candidates/MBR. Codec reconstruction is very accurate on tested domains, but rare strings and numbers still deserve scrutiny. Smaller numbers of function evaluations are not a hardware-independent speedup.

**Astra decision.** Retain the frozen decoder and the normalized hierarchy. Use a conservative high-rate 16-token codec first, rather than forcing exact text through a few dozen bits. Count codec training, storage and decoding. Replace only the generator with HRM, so the hypothesis is identifiable. A simple Transformer generator is the indispensable control.

## 2. Depth Memory

**Primary source:** [arXiv:2610.18809](https://arxiv.org/abs/2610.18809). Local: `V15-Papers/DEPTH MEMORY.pdf`, 34 pages. Main focus: §§3–4, PDF pp. 3–10; implementation and state accounting pp. 19–21, 31. Page 5's regression formulation was visually inspected.

**Problem formulation.** Ordinary residual addition accumulates sublayer contributions. Depth Memory instead treats earlier contributions as information that can be queried while retaining the ordinary residual stream. Its Linear Depth Memory carries a value-key associative matrix and an inverse covariance matrix for each position/group. The extra inverse-covariance stream distinguishes exact recursive regression from an uncorrected outer-product or simple delta-rule memory.

**Mathematical contract.** For weighted key/value observations, the RLS gain divides by a positive forgetting-plus-quadratic denominator. The memory update writes the residual prediction error; the covariance update changes what later keys count as novel. The forgetting factor scales both prior observations and the initial regularizer in the paper's regression objective. A formula with this recurrence cannot simultaneously claim to solve a different fixed-ridge problem without an additional correction.

Keys/queries, importance weights, read gates and residual writes have distinct roles. A read enriches a sublayer's input. Its newly computed delta is added to the ordinary residual and written for later depth. A final read can expose accumulated information to the output. State is reset per forward example: the depth index is not a text-time or lifetime-learning index.

**Evidence and systems limits.** The paper studies 400M and 1.5B language models with substantial token budgets and specialized hardware-aware implementations. Reported small parameter overhead does not mean small activation memory: value-key and covariance matrices exist per token/group, and training may retain multiple depth states. Optimized CUDA performance does not transfer automatically to a small local-device model. Competitive aggregate results do not establish a benefit in a four-block, 16-token codec.

**Astra decision.** Specify a small grouped RLS adaptation in E/D with exactly counted projections and state; compare with plain residual E/D. Use FP32 state and FP64 reference solves. Keep it out of the initial HRM until its benefit is isolated. A generic associative delta rule must be labeled an alternative, not described as the paper's RLS implementation.

## 3. HRM-Text: Efficient Pretraining Beyond Scaling

**Primary source:** [arXiv:2605.20613](https://arxiv.org/abs/2605.20613). Local: `V15-Papers/HRM_Text.pdf`, 26 pages. Main focus: PDF pp. 3–5, 7, 9–10; training/ablation appendices pp. 22, 24, 26. Architecture Figure 2 on page 3 was visually inspected.

**Mechanism.** A slow high-level state and a fast low-level state interact. The primary schedule has two high-level cycles, with three low-level updates then one high-level update per cycle. The same module weights are reused. A model with half its unique recurrent blocks in each module executes roughly four full-core equivalents under that schedule, rather than one.

MagicNorm uses sequential PreNorm residual blocks inside each module and a final module-exit normalization. The paper uses parameterless RMSNorm, SwiGLU, rotary positions and gated attention. Warmup credit assignment increases the differentiable recurrent horizon from the last two to the last five steps. The truncated gradient is a practical optimization choice, not the full gradient through all forward iterations.

**Data/objective matter.** The work trains from scratch on instruction/response pairs, uses a response-only task-completion objective and PrefixLM masking, and controls response styles through condition tags. It reports 40B unique tokens and 60B exposure, with stratified sampling and removal of explicit `<think>` spans. Comparisons to web-pretrained models therefore mix architecture, dataset and objective differences. A claim that recurrence alone caused the full gap is unsupported.

**Astra adaptation.** Put recurrence in the frozen-code generator. Use explicit query-state inputs, a separate observed-prefix context stem and level heads. Early z is reinjected as conditioning so a truncated horizon still has a live path to mode parameters. Use full recurrence gradients for tiny latent search. Compare recurrence to a simple prior at both equal parameter count and equal total compute. Autoguidance is deferred: its reused shallow state still needs head computation and a justified quality/cost comparison.

## 4. Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation

**Primary source:** [arXiv:2607.27372](https://arxiv.org/abs/2607.27372). Local: `V15-Papers/Explorative Modeling.pdf`, 36 pages. Main focus: PDF pp. 5–8; algorithm/probabilistic details pp. 28–29 and discrete-model/limitations discussions pp. 32–34. Page 7's forward/reverse exploration diagram was visually inspected.

**Mechanism.** A simple forward-XM objective generates K candidates and trains the best match to a data example. This changes the loss geometry relative to reconstructing all examples with one prediction. Reverse matching searches over data examples for a generated candidate; it introduces different coverage/collapse concerns. Hard winner selection and smooth mixture likelihood are related motivations but different objectives and gradient rules.

**Why the distinction matters.** Hard-min training can score candidates without gradients and backpropagate through only the winner. A soft log-sum-exp likelihood weights all branches by their posterior responsibilities. Dropping nonwinning gradients is not exact soft-mixture training. For discrete likelihoods, branch scores must be log probabilities of complete normalized events. Averaging or arbitrarily reweighting log probabilities inside the mixture generally breaks that interpretation.

The language experiments use a masked-diffusion model and learned latent embeddings associated with masked positions; a candidate may be a latent assignment over positions. That differs from Astra's one coherent mode per concept hierarchy. The paper's separation-based expressivity argument in squared-error settings does not establish a universal K-distinct-modes guarantee for a categorical code model. Identical branch distributions are a direct possible collapse.

**Astra decision.** Use an exactly enumerated finite uniform mixture first. This is an XM-inspired adaptation, not a claim to reproduce every XM objective or its end-to-end generation property. Astra still has a separately trained codec and teacher-forced level prediction. Compare hard XM as a separately named arm only if needed. Measure mode diversity, correctness, mode-selection regret and equal-budget sampling; do not infer value from embedding distance or mode count alone.

## 5. Generative Recursive Reasoning

**Primary source:** [arXiv:2605.19376](https://arxiv.org/abs/2605.19376). Local: `V15-Papers/Generative Recursive Reasoning.pdf`, 27 pages. Main focus: PDF pp. 3–5, 9–10, 15–17. Page 4's stochastic hierarchy and latent-variable objective were visually inspected.

**Mechanism.** GRAM introduces a stochastic trajectory into recursive reasoning. In its hierarchical instantiation, low-level refinement is deterministic, while a learned Gaussian perturbation is added to the high-level transition. The mean and variance depend on the current state. Generation uses the prior; training uses a target-conditioned variational posterior.

The theoretical trajectory ELBO includes a reconstruction/answer term and a sum of transition KL terms. Its practical deep-supervision/truncated scheme backpropagates through the final transition and uses a truncated surrogate. The paper explicitly distinguishes that from optimizing the full trajectory ELBO exactly. ACT and a latent process reward model add further training targets and inference decisions.

**Implication.** GRAM is stronger and more specific than “HRM plus random z.” A target-conditioned posterior can improve training but creates a prior/posterior gap that must be measured. Adding it to a discrete code mixture changes the model and its objective, not just its initialization. Stochastic width can be valuable without proving every additional recurrence helps.

**Astra decision.** Defer a full GRAM arm until the deterministic HRM prior works. Then compare learned stochastic trajectories with the finite-mode model using an explicit prior/posterior objective and counted inference-selection cost. Avoid carrying an untrained posterior-dependent behavior into deployment. This follows the earlier project's prior/posterior difficulties rather than assuming noise itself solves multimodality.

## 6. GradCuit: Credit-Assigned Gradient Flow Enables Robust and Interpretable Test-Time Latent Reasoning

**Primary source:** [arXiv:2608.02585](https://arxiv.org/abs/2608.02585). Local: `V15-Papers/GradCuit.pdf`, 19 pages. Main focus: PDF pp. 3–6 and implementation pp. 14–15. Page 4's factorization/gradient path was visually inspected.

**Mechanism.** GradCuit places optimizable prefix representations at an intermediate Transformer layer. Later continuation positions attend to those states, so the sum of continuation log probabilities has a differentiable path to each latent. Model weights stay fixed while a latent offset changes. The important idea is credit assignment through the model's internal interactions, not merely replacing an output embedding and hoping gradients cross a decoded prefix.

**Implementation details.** The supplied paper initializes from a fixed textual reasoning prefix, uses the model's middle layer as a standard configuration, and studies other insertion depths. It greedily generates an answer, obtains a self-verifier reward of zero for accepted answers and −1 otherwise, recomputes the detached continuation under teacher forcing, and updates only the offset. It uses Adam at 1e-3, no clipping/norm constraint in that recipe, and at most ten updates. The greedy estimator is a practical policy-gradient-style procedure, not automatically an unbiased on-policy expectation estimator.

Results cover several instruction-tuned models and reasoning panels. A self-verifier can be wrong; verifier calls and repeated generation are real costs. Its layer study does not establish a universally optimal midpoint, nor the performance of a 10M concept model.

**Astra decision.** Use the likelihood of the actual sampled **code actions** for gradients. The frozen decoder's argmax is part of reward evaluation; gradients need not and cannot ordinarily pass through that discrete operation. Fresh stochastic samples, a detached baseline and bounded δ define the reference estimator. This differs from the paper's heuristic details and must beat simple retry and random-offset search at equal time. Earlier V6 evidence makes that comparison essential.

## 7. Titans: Learning to Memorize at Test Time

**Primary source:** [arXiv:2501.00663](https://arxiv.org/abs/2501.00663). Local: `V15-Papers/Titans(1).pdf`, 27 pages. Main focus: PDF pp. 3–14. The MAC/MAG masks on page 9 were visually inspected.

**Mechanism.** Titans treats neural memory updates as an inner associative-learning process. Key/value projections form the association; memory weights adapt through surprise-related gradients, momentum and forgetting. Other network/projection parameters are learned through an outer training process. Queries read the current memory without changing it.

The paper separates contextual writable memory from persistent learned tokens and short-term attention. MAC inserts retrieved information as context, MAG combines attention and memory through gating, and MAL stacks memory and attention. These are different integration choices with different bottlenecks. The paper also studies deep memory rather than restricting every memory to a single matrix.

**Evidence limits.** Language experiments use hundreds of millions of parameters and billions of training tokens; long-context/recall results depend on the benchmark and integration. A claim that the memory state stays fixed-size does not imply unlimited exact recall. Nor does it imply bounded whole-system memory when every sequence/user carries separate fast weights and training keeps their update graphs.

**Astra decision.** Preserve the distinction between weights learned offline, writable episode state and fixed task knowledge. Begin with an explicitly specified, small associative initializer rather than injecting a newly initialized neural memory into every frozen module. Any learned read/write gates or projections in a later Titans arm require a real outer training objective. Do not call a free-floating matrix update a reproduction of MAC/MAG/MAL.

## 8. ATLAS: Learning to Optimally Memorize the Context at Test Time

**Primary source:** [arXiv:2505.23735](https://arxiv.org/abs/2505.23735). Local: `V15-Papers/Atlas.pdf`, 31 pages. Main focus: PDF pp. 5–8, 12–14, 18, 29–30. Table page 18 was visually inspected because extracted table columns were interleaved.

**Mechanism.** ATLAS broadens associative memory learning from the current example toward a recent-context objective. Polynomial feature expansion can provide a richer key space, while a window-based Omega rule and an internal optimizer improve memory updates. The paper examines neural/deep memory, Muon-style Newton–Schulz processing, and attention hybrids. Feature-space size and optimizer state contribute to real cost.

The polynomial argument is not a free capacity multiplier: explicit degree-two features in m key dimensions have a quadratic number of coordinates. Kernel claims depend on the actual feature coefficients. Replacing the representation with a smaller projection changes the function class. Update ordering matters too: putting a learning rate inside a scale-normalizing Newton–Schulz operation is not interchangeable with multiplying the resulting update by that rate.

**Reading the results carefully.** The visually checked table shows Transformers ahead of ATLAS on the reported average in-context recall panel, even though ATLAS narrows the gap relative to recurrent baselines and performs strongly on other tasks. The ablation table does not show a monotonic win for every component on every metric: for example, the no-Muon row has lower reported perplexity than the displayed base row, while the reasoning metric differs. Broad claims in prose should be read alongside those rows. The model table and experiments use substantial language-model scales and exposure, not a tiny concept-offset memory.

**Astra decision.** Choose an explicit 153-dimensional quadratic feature map for a 16-dimensional key and a 256-dimensional offset value. Use a recent-four-record MSE objective with ordinary momentum as a simple, auditable ablation. Name it window memory, disclose the departure from deep ATLAS/Muon, count all state and compare against bounded retrieval. Upgrade the optimizer or memory architecture only after this interface provides useful reuse.

## 9. Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories

**Primary source:** [arXiv:2606.03979](https://arxiv.org/abs/2606.03979). Local: `V15-Papers/LLM Need Sleep Google.pdf`, 26 pages. Main focus: PDF pp. 5–13 and implementation/results qualifications pp. 25–26. Page 8's knowledge-seeding and distillation equations were visually inspected.

**Mechanism.** The continuum memory system organizes components by update frequency. Sleep transfers knowledge from faster-changing modules toward slower, more durable parameters. Knowledge seeding can add lower-rank expert capacity, freeze existing parameters, and train the new parameters through generalized distillation. The method combines teacher-generated and student-generated contexts and an imitation/reward component. Resetting faster memory after transfer creates room for further adaptation.

Dreaming adds a learned curriculum/self-modification process. It proposes synthetic practice, tests adaptation on held-out feedback and updates the proposal policy, rather than merely sampling text and declaring it correct. The paper's RL/imitation and curriculum machinery have costs and data dependencies distinct from ordinary replay.

**What not to infer.** Freezing old weights reduces direct overwriting but does not prove unchanged behavior: new experts, routes or gates can still change predictions. Growing stored capacity is not free even if sparse active compute remains controlled. Self-generation supplies practice and recombination, not an external guarantee of new factual truth. Reported evaluation protocols matter: some mathematical results are average@16, and the ARC setup uses a filtered small train/held-out subset and multiple dreams; those results are not a blanket full-benchmark pass@1 claim.

**Astra decision.** Implement the useful functional distinction first: verified wake behavior should become available with zero offset and empty fast memory. Distill conditional code behavior into a bounded adapter; use replay and old-model anchors; validate and roll back transactionally. Student-generated histories are included as a clearly labeled conditional imitation objective. Two fixed slots limit capacity, so this is not the paper's entire growing continuum system. Checked synthetic variants may follow; a learned self-edit policy requires a separate experiment.

## 10. Synthesis and remaining uncertainty

The strongest combination is an interface contract, not a collection of paper names: frozen codec labels; an actual next-concept generator; early latent conditioning; normalized code probabilities; a valid discrete-action gradient; isolated fast state; and verified consolidation into bounded slow parameters.

The papers supply plausible mechanisms at different scales and settings. They do not establish that their benefits add together. The main unresolved interaction is the cost of HRM plus exploration relative to the shorter sequential schedule that makes NCM attractive. The second is whether small codec reconstruction and continuous-offset search are good enough for exact tasks. The third is whether sleep can acquire new capabilities without simply reducing adaptation to protect old ones. Astra's staged controls are designed to answer these questions before scaling.
