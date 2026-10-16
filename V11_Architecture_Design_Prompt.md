# V11 RCDL architecture design prompt

Prepared 12 October 2026. Intended recipients: **GPT-6 Astra Ultra, Opus 5.5, and Fable 5.1**, using the model names supplied by the project owner. Use the same prompt independently with each model; no model has been contacted by preparing this document.

Attach this file and the contents of `V11_Paper/`. That folder contains the five supplied papers and a compact V10.3 evidence packet. Source filenames are preserved, including their original spelling. Its `manifest.json` records hashes. The evidence summary below makes the request usable even without repository access; explicitly report which attachments you actually read.

**Everything between BEGIN PROMPT and END PROMPT is the reusable prompt.** The separate `V11_Prompt_Review.md` records the review of this brief.

---

## BEGIN PROMPT

You are designing **V11 RCDL**, the successor to my V10.3 Encoder–Concept–Decoder research architecture. Produce one coherent, implementable architecture proposal, an economical experiment plan, and a critical review of your proposal. Do not implement it or start training in this task.

Work from information availability, predictive sufficiency, uncertainty, optimization, computation, and measured evidence. Give equations, tensor shapes, explicit assumptions, concise design rationales, and falsifiable decisions. I need an auditable technical argument, not private internal deliberation or a collection of paper names. Distinguish **observed result**, **source claim**, **your inference**, and **untested proposal** throughout.

My preferred direction is ECD. Optimize within that research direction while making it possible to discover that an individual mechanism is ineffective. Neither dismiss ECD because a tiny experiment was weak nor promise that scale will fix it. An attractive diagram is insufficient: every inference path must be causal, every loss must have a gradient contract, and every added cost must have a testable purpose.

### 1. Refine the research thesis

Use this as the starting hypothesis, then improve its precision:

> The encoder builds a causal representation of the observed sequence. A recurrent concept module maintains a compressed predictive state, learns transitions and longer-horizon structure, and may eventually support temporally extended control. The decoder realizes a token distribution using both this predictive state and a direct route to precise encoder information. Hindsight training can teach the concept state what was useful to predict, but deployed predictions must use information available at that time. Local RBF corrections and gated adapter consolidation support later adaptation without rewriting the entire model for every new observation.

“Thinking,” “world model,” “translator,” and “meta-controller” express desired roles. Turn them into measurable properties; do not assume modules acquire them because they have those names. Text prediction can learn a predictive state of text without identifying causal dynamics of the external world. A small decoder still needs enough capacity for lexical choice, syntax, copying, and token-level dependencies. An encoder is trained toward useful representations; it does not automatically supply an optimal target.

A motivating example is a phone number: exact digits can remain available through Encoder→Decoder CRC while the concept module represents the surrounding intent or future task state. Specify how to test selective use of these routes with controlled counterfactuals. Do not hard-code “numbers bypass concepts” or treat routing weights alone as proof of semantic specialization.

### 2. Commitments and research variables

Preserve these in the primary V11 design:

1. **Encoder–Concept–Decoder**, with **E1, E2, E3** explicitly separated. E2 retains the two gated streams. Explain their distinct update equations and roles.
2. **Exactly two recurrent visits** through the shared E2 stack and exactly two through the shared concept stack. Count unique blocks separately from executed block applications. Auxiliary prediction horizon, number of candidate codes, and duration of a future controller action are different quantities from recurrence.
3. A causal compressed concept path, beginning from the existing **eight encoder token states → two concept slots** reference. Keep concept prediction, causal shifting/repetition, and decoder next-token prediction. Any alternative pooling or temporal resolution must be a clearly labeled ablation against this reference.
4. **Encoder→Concept CRC, Encoder→Decoder CRC, Concept→Decoder CRC.** Specify the sources, alignment, normalization, gates, and release times of each. Preserve the existing aligned forecast route and concept-history access in the reference; justify any simplification in the proposed successor.
5. **Next-concept prediction plus an additional NextLat- or HiLP-inspired latent objective attached to the concept system.** Choose a minimal primary objective after comparing both inspirations. Do not silently replace segment concept prediction with only token forecasting.
6. An **Explorative Modeling inspired candidate mechanism**, with a deployable prefix-conditioned policy. Keep a K=1 control. Learned K=4 or K=8 strategy codes are reference options, not a proof that four or eight meaningful strategies exist.
7. **RBF correction and gated LoRA sleep/consolidation**, with explicit retention tests and stable memory addressing. Interpret “RBL correction” in my description as the existing RBF correction bank.

Preserve V10.3's full learned forecast-to-decoder projection and codebook no-decay policy as initial controls. Revisit them only through an isolated, justified comparison. Preserve a separately accounted Engram option; do not assume its hash table is a reliable factual database or that it can replace reasoning.

The target representation, teacher update rule, loss weights, candidate objective, controller training stage, CRC source granularity, intra-module residual scheme, and encoder/concept/decoder capacity allocation are research variables. My **120 encoder / 60 concept / 20 decoder** example is a future allocation to analyze, not a required optimum.

Give a practical width-128 reference under **10 million trainable dense parameters**, including training-only dense heads, with approximately **20 million learned Engram table parameters counted separately**. Distinguish buffers, teacher copies, RBF storage, optimizer states, and deployed parameters. Start the capacity comparison from the earlier 4/4/2 encoder split, at least four concept blocks, and at least four decoder blocks, all with ordinary 4× FFNs; a smaller decoder is a labeled comparison requested here. Show whether the budget actually fits. Explain any proposed departure rather than hiding it in parameter accounting.

Use the earlier **three sliding-window encoder attention blocks to one full-history block** as the reference schedule, with causal masks on both. State how that schedule applies to unique E2 blocks and their repeated visits. Keep full completed-group causal attention in the concept reference. Specify the decoder's attention pattern explicitly; alternative attention families belong in measured comparisons.

### 3. What V10.3 actually established

Within `V11_Paper/`, read `V10_3_Evidence/V10_3_tiny_REPORT.md`, `V10_3_Evidence/v10_3_tiny/ARCHITECTURE.md`, and that same `v10_3_tiny/` directory's `comparison.json`, `REPORT_TABLES.md`, and `runs/sleep_X8_s1/sleep.json`. The root V10.3 architecture document contains earlier V10.1 checkpoint diagnostics; do not mistake those for measurements of trained V10.3. The tested implementation is `v10_3_tiny`, not a trained 10M V10.3 model. Selected implementation files and earlier design documents are in `V10_3_Evidence/source/`, preserving their project-relative paths; use them to resolve the inherited pooler, two-stream, GradCuit, loss, and memory contracts. That source subset is for inspection and is not a complete runnable project.

The tiny experiment used width 56, vocabulary 1,024, context 256, an E1/E2/E3 split of 2/1/1, two E2 visits, two concept blocks with two visits, and two decoder blocks. X8 had **503,428 dense training parameters**, of which **29,064 were training-only**, plus **1,050,080 Engram parameters**. C0 removed the concept path and had **311,704 dense parameters**. It was a useful simpler control, not a parameter-matched transformer.

Six runs completed 6,000 updates: C0, A, X8, X4 at seed 1, and C0/X8 at seed 2. A used the older V10.1 recipe. Same-seed runs used the same intended training stream and schedule; some seed-1 arms ran on different devices. Solo Metal smoke timings were about **0.160 s/step for C0** and **0.261 for X8**. There was **no completed equal-wall-time C0 comparison**, no auxiliary-loss-off X8 control, and no isolated Engram on/off study. Sampling with replacement does not imply every example was fresh.

NLL is in nats per scored token; lower is better:

| Metric | C0 seed 1 | X8 seed 1 | C0 seed 2 | X8 seed 2 |
|---|---:|---:|---:|---:|
| Text NLL | 3.3641 | 3.3820 | 3.3507 | 3.3556 |
| Pretraining math NLL | 1.0877 | 1.1166 | 1.0434 | 1.1621 |
| Held-out algebra NLL | 2.9495 | 3.0679 | 2.4744 | 3.2485 |
| Sudoku easy NLL | 0.9114 | 0.8473 | 0.9118 | 0.9337 |

X8 improved over A on seed-1 text and aggregate math, largely through sorting, but did not consistently beat C0. Strict generation remained near the floor: X8 solved 1/96 pretraining math examples, 0/64 held-out algebra examples, and 0/16 Sudoku boards. The old math evaluator's 24-token generation limit is a confound for longer sorting answers; repair it before interpreting generation differences.

Mechanistic evidence matters:

- Removing X8's entire concept path increased seed-1 NLL by approximately **0.004 text / 0.100 math / 0.131 Sudoku easy**. Removing the continuous-state reader harmed math more than removing its PQ route: approximately **0.053 versus 0.004**. This demonstrates use inside that trained model; it does not establish benefit over a separately trained simpler model.
- Reducing concept visits from two to one had very little effect: about **0.000 / 0.002 / 0.007** on those families. Preserve two visits as requested, but give the second visit a measurable refinement role and test its incremental contribution.
- Ridge probes predicting the next pooled group yielded X8 FVU for core state U versus its pooled input P3 of **0.954 vs 0.954 on text**, **0.655 vs 0.637 on math**, and **0.493 vs 0.512 on Sudoku easy**. Lower is better. There is no text increment, math gets worse, and Sudoku shows a modest increment. Do not repeat the source report's categorical claim that the core adds no information in every family.
- X8's mixture-likelihood objective differentiated all K branches correctly, yet mixture-versus-committed gains were around **0.002 nats/token or less** and little better than a uniform prior. Candidate redundancy was not explained by a missing-gradient bug. Winner classification accuracy is not the objective of a mixture likelihood.
- Codebook RMS remained near its initial scale after exemption from weight decay. This is a useful controlled change, not isolated proof that this change caused downstream improvement.
- Fixed-layout Sudoku put every answer at token 89 in a 171-token record. Prefix shifts and seed changes undermine a reasoning interpretation of its seed-1 advantage. Require randomized layouts and alignment controls for V11.
- The **26/26 correctness tests passed**, covering causality, release masks, mixture gradients, RBF derivatives, round trips, and CPU/Metal parity. These tests do not certify usefulness, sustained scalability, exact checkpoint resume, or multi-phase retention.

The sleep results are the strongest reason to retain RCDL. Starting from X8 seed 1:

| Adaptation | New algebra NLL | Text NLL | Add/sub NLL | Multiplication NLL | Sorting NLL |
|---|---:|---:|---:|---:|---:|
| Frozen base | 3.068 | 3.382 | 1.335 | 1.780 | 0.758 |
| Plain LoRA | 1.308 | 3.509 | 1.735 | 4.017 | 1.110 |
| Gated LoRA | 1.306 | 3.386 | 1.725 | 3.122 | 0.757 |
| RBF bank | 1.518 | 3.382 | 1.598 | 1.916 | 0.789 |
| Bank consolidated into gated adapters | 1.487 | 3.382 | 1.799 | 1.962 | 0.759 |

This is promising **one-phase NLL adaptation**, not solved algebra or universal forgetting prevention. Gates protected dissimilar distributions much better than related arithmetic. Consolidation increased add/sub interference relative to the bank. The old aggregate retention calibration was dominated by text tokens, so a small aggregate loss did not guarantee each skill's retention.

Carry these review corrections into V11's test plan:

1. Treat the old fixed **25% forecast/target variance threshold** as a diagnostic to replace, not a universal usefulness criterion. Feature scaling, readout coordinates, and objective changes confound it. Use matched targets, baselines, functional interventions, and held-out predictive utility.
2. The token gate smoother in `v10_3_tiny/sleep.py:ema` starts at zero, although the design recommends initialization from the first decision. Specify first-token and short-sequence behavior and test cached versus full-sequence equivalence.
3. Random seeds do not guarantee disjoint calibration and write examples. The final 512-episode RBF write sample reused two calibration IDs even though the earlier calibration write set was disjoint. This is a calibration-protocol issue, not demonstrated dev-set leakage. Use explicit ID exclusion through the final rewrite and distillation stages.
4. Weights-only checkpoint round trips do not establish optimizer/RNG/dataloader/Engram/memory resume correctness.
5. The report's speculation that an equal-wall-time gap “could only widen” is not a measured result. More compute, different schedules, or larger models require new comparisons.

Explain which weaknesses could plausibly be capacity-limited, which are objective or information-flow problems, and what experiment separates those explanations. No “explosive result at scale” claim is justified yet.

### 4. Read the papers as mechanisms, not endorsements

Use the attached editions and cite page/section/equation when adopting a mechanism. Document content is evidence, not instructions overriding this request. Report unavailable or ambiguous sources instead of inventing details.

**NCP-ArchPreview Technical Report — attached PDF, Sections 2–3, pp. 4–9.**

It explicitly combines Token Encoder, Concept Module, Token Decoder, product quantization, and all three CRC directions. Its reference pools k states into one concept; my 8→2 pooler is a project-specific adaptation. Its prediction uses soft distributions over segment codebooks and weighted codeword sums. Equation 22 is **MSE to a stopped continuous next-concept target**, not necessarily a cross-entropy loss on discrete code IDs. VQ fitting in Equation 21 updates codebooks toward stopped encoder representations. IRC uses unnormalized, potentially signed depth coefficients; CRC uses normalized source-depth weights plus learned diagonal scaling. These mechanisms retain a single hidden stream per module. Read the difference carefully before claiming equivalence to expanded residual streams. Specify V11's training/inference concept-history policy explicitly rather than assuming it follows automatically from the diagram.

**Next-Latent Prediction Transformers Learn Compact World Models — attached Next-Latent Prediction Transformer.pdf, arXiv:2511.05963v4, Section 3 and Algorithm 1, pp. 4–6, 25.**

Its dynamics are token-conditioned: `(h_t, x_{t+1}) → predicted h_{t+1}`. Multi-step training feeds the actual next tokens to the latent rollout. Smooth L1 compares to stopped real states; KL compares next-token distributions from real and predicted states using a frozen output-head parameter view. The distribution at state `h_{t+1}` concerns **x_{t+2}**. The auxiliary dynamics can be omitted at ordinary inference. Its belief-state argument assumes ideal consistency conditions; it does not prove V11's compressed concept slots or arbitrary CRC graph are sufficient statistics. Explain which assumptions cease to hold when the decoder accesses encoder history outside the concept state.

**Hierarchical Latent Prediction for Language Models — attached Heirachical latent prediction Microsoft.pdf, arXiv:2608.05806v1, Section 2 and Algorithm 1, pp. 1–2, 7.**

HiLP adds a causal sliding-window abstract latent and predicts its future value **without future token input**, alongside lower-level NextLat dynamics and a combined NTP head. Its reported ordinary inference removes the auxiliary hierarchy. Retaining a concept hierarchy at inference is our architectural extension. Its combined head reads the observed abstract latent, not automatically the predicted future abstract state. Specify detach choices independently: the text describes isolating lower representations, while the displayed combined-head expression has a direct `h_t` input. Detaching only the SWA input does not detach that direct path. Do not inherit an ambiguous gradient contract. The paper's substantial training overhead also needs accounting; “training-only” does not mean free.

**Explorative Modeling — attached PDF by Alexi Gladstone, Heng Ji, and Yilun Du, arXiv:2607.27372v1, Section 3, pp. 6–9, Appendix C.**

Forward exploration samples multiple predictions and trains through the best matching candidate; reverse exploration searches over targets. The paper studies continuous and discrete settings. Continuous future-state targets make mode averaging a relevant concern, but this does not make every NextLat transition highly multimodal after conditioning on the actual next token. V10.3's all-branch mixture objective differs from hard best-of-K. The paper also discusses soft mixture interpretations: compare the objectives under explicit likelihood/kernel assumptions instead of declaring them unrelated or identical. Never substitute a hindsight-selected candidate into a prefix-only inference path without a valid deployment policy.

**Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning — attached PDF, arXiv:2512.20605v2, pp. 5–7 and Appendix D.**

This work trains a metacontroller over a **pretrained frozen** action model, using a noncausal sequence encoder during controller discovery and a causal switching mechanism. A later causal abstract-action policy replaces privileged future conditioning for internal RL. Its co-trained comparison loses the useful switching structure in the reported setting. Treat this as an important staged-training alternative, not a universal theorem against joint ECD pretraining. Fixed chunks or K strategy embeddings alone are not temporally extended actions. A claim of meta-control requires interventions, useful duration/termination, and transfer or planning tests.

Use these additional primary references for targeted comparisons:

- **V-JEPA 2**, [paper](https://arxiv.org/html/2506.09985v1), Section 2.1 and Section 3: masked video feature prediction uses a stopped EMA-encoder target; V-JEPA 2-AC separately learns action-conditioned future-state prediction with a frozen video encoder. This motivates target design and staged learning. It does not establish equivalence between masked video completion and causal language prediction. Compare online stopped targets, EMA targets, and a staged frozen target on their stability, semantic usefulness, lag, and cost.
- **Anthropic global workspace**, [research summary](https://www.anthropic.com/research/global-workspace) and [technical article](https://transformer-circuits.pub/2026/workspace/index.html): J-space is an empirically identified representation subspace inside an existing language model, not an explicit ECD architecture. Use its causal intervention and broadcast perspective to motivate tests of shared concept access. Do not claim it proves ECD, recurrence, or a separate controller is necessary.
- **Kimi Attention Residuals**, [primary paper](https://arxiv.org/abs/2603.15031): compare content-dependent attention over depth and blockwise aggregation with NCP IRC. Distinguish depth mixing from token attention and cross-module routing.
- **DeepSeek mHC**, [primary paper](https://arxiv.org/abs/2512.24880): expanded residual streams and constrained mixing require explicit state dimensions and stability conditions. Multiple skip routes through differently pooled spaces do not by themselves implement mHC.
- **Qwen**, [Qwen3-Next primary description](https://qwen.ai/blog?id=qwen3-next) and [Qwen3.8 repository](https://github.com/QwenLM/Qwen3.8): the precise phrase “Qwen 3.8 Next gated residual attention” is not a verified mechanism in this brief. Resolve the intended version and equation from primary material before using it. Distinguish gated attention outputs or gated linear attention from depth-residual attention; mark unresolved naming explicitly.

Do not infer any particular Kimi K3 parameter count. My **1–3 trillion parameter** ambition is a hypothetical scaling target, not a verified product specification or performance prediction.

### 5. Solve the clocks and causal interfaces first

Define zero-based token positions: after observing `x_0…x_t`, decoder row `t` predicts `x_{t+1}`. For the reference pool, group `g` contains positions `8g…8g+7`, and both slots depend on that complete group. Thus both have release time `r_g = 8g+7`, regardless of whether slots are called local/global or first/second. A pooling slot may not be released at the midpoint merely because two slots exist.

Use distinct symbols for:

- `h_t`: causal encoder state at token time t;
- `p_g ∈ R^(2×d_c)`: observed pooled slots, available at `r_g`;
- `u_g^(1), u_g^(2)`: concept states after each of two visits;
- `f_{g+1|g}`: a forecast of future group slots using information available through `r_g`;
- `z_g`: the chosen strategy/candidate variable, distinct from a PQ segment index;
- `b_t`: the causally released concept signal delivered to decoder row t;
- `q`: the exact representation used to address RBF memory.

Derive the decoder alignment table for positions 0–17, including empty prefix, BOS convention, completed groups, incomplete tails, and padding. Show which slot is repeated to which decoder rows, how both predicted slots map to the next eight target positions, and why each source has release time ≤t. Audit the **raw-state Concept→Decoder CRC** separately: making the predicted-concept path causal does not make every other path causal. Full concept self-attention means full **completed-group causal** history in the deployed autoregressive path, not bidirectional access to future groups. Both slots of an already released group may attend to each other; distinguish this mask from a strict triangular mask over individual slots.

The core design problem is that NextLat operates at token rate while our concept core operates at group rate. Compare at least these two legitimate adaptations, then choose one:

1. A token-rate latent transition head attached to the concept system, using the latest released concept state and an explicitly defined causal token bridge. It predicts a specified encoder/decoder-derived next state after receiving the next token. Show that it trains the concept module rather than solving everything from the bridge alone.
2. A hierarchical group-rate objective: causal future-concept prediction without future tokens, optionally complemented by a chunk-conditioned hindsight transition after the chunk has been observed. Specify how this differs from original NextLat and from HiLP's overlapping SWA hierarchy.

For each, define what becomes available before versus after sampling `x_{t+1}`. A target-conditioned auxiliary branch may learn from the real `x_{t+1}` during training; its resulting state cannot help predict that same token. If used at inference, say whether it conditions on a generated token, branches over candidates with normalized probabilities, or is discarded. Prevent an algebraic loop in which the decoder needs a concept that already needs its own output token. A hindsight posterior can train a causal prior, but ordinary evaluation must use the prior.

Explain the difference between an observed state, a prospective forecast, and a hindsight reconstruction. Define whether generation re-encodes newly emitted tokens at group boundaries, recursively feeds predicted concepts, or uses both; train and test the selected policy. Define cache updates for both recurrent visits and prevent a later group or strategy decision from changing cached representations used for already generated logits.

### 6. Choose losses that make predictive information useful

Propose a small justified set of losses. At minimum specify decoder NTP, segment next-concept prediction, the chosen additional latent objective, exploration training, and the separate adaptation/consolidation objectives. If you keep, replace, or drop V10.3's forecast-only token head, explain the overlap with the new head and isolate the change.

For every loss give an equation, reduction denominator, valid-position mask, weight/schedule, units, and a table of which parameters receive gradients: E1/E2/E3, pooler, concept visits, strategy codes/prior, PQ codebook, predictor, decoder, vocabulary head, teacher, RBF values, and adapters. Preserve the inherited **temporary GradCuit offset at zero during pretraining**: the implementation names this code-space perturbation `xi` in `(e_k + xi) W_z`; my earlier description called it δ. It is not a scalar gradient-cut coefficient. Specify its shape, lifetime, and any later local-steering use separately from ordinary learned strategy embeddings, stop-gradients, temporal shifts, and RBF corrections.

Choose the prediction target: pooled encoder state, concept state, final decoder state, a projected target space, or a justified combination. Explain why it contains future-relevant information beyond what a cheap readout of the current pooled input already supplies. Avoid asking a compressed concept to reproduce unpredictable lexical detail if the decoder's direct route handles it. Conversely, do not define a target so easy or low-rank that constant prediction succeeds.

Specify normalization, target variance/rank monitoring, codebook learning, and how the target avoids collapse or drift. Stop-gradient alone is not a universal anti-collapse guarantee. An EMA teacher costs memory and introduces lag; a teacher reading the complete document may also include information beyond the declared forecast horizon. Define its masks and information boundary. Keep evaluation target spaces fixed where necessary for comparisons across checkpoints.

If using NextLat KL, freeze the vocabulary head's parameters for that term **without blocking the derivative through the predicted latent**. Teacher logits are stopped. State whether gradients flow through the starting state and into the encoder/concept trunk. For HiLP's combined head, define both the direct token-state branch and the abstract-state branch. A token-rate final decoder state and a group-rate concept vector are not interchangeable just because their widths match.

Specify the forecast's relationship to quantization. Product-quantized segment vocabularies and K exploration codes serve different purposes. Decide whether continuous prediction supervises a pre-quantized state, the codebook-constrained prediction, or both; explain how conflicting objectives and soft-codeword averaging are controlled. Include NCP-style MSE, discrete-label alternatives, and continuous targets in the comparison, but select one primary recipe rather than enabling every loss.

Measure module-wise gradient norms **before and after clipping**, clipping frequency, relative parameter update norms, and NTP-versus-auxiliary gradient alignment on sampled steps. A pre-clip gradient norm above 1 is not by itself evidence that CRC or a new head is needed. Loss weights should reflect signal, interference, and actual quality gains, not only equal numeric magnitudes.

### 7. Make exploration a learned predictive distribution

Locate the conditional uncertainty exploration is intended to model. Future chunks without their tokens may have many plausible continuations; a transition conditioned on the actual next token may have much less uncertainty. Explain where K candidates help and why increasing model size alone would or would not solve averaging there.

Compare hard forward best-of-K, a soft mixture likelihood, and a continuous stochastic candidate generator. Give the chosen objective and gradient estimator. If squared errors are converted to likelihoods, define the kernel variance/temperature and dimensional normalization; do not relabel arbitrary log-sum-exp scores as calibrated likelihoods. Specify how a trainable prior is learned and used at inference, how the code is committed, and how randomness is seeded and cached.

Evaluate held-out predictive gain, calibration, coverage, candidate-dependent future behavior, and committed/prior-sampled performance. Log the hindsight oracle only as an upper bound. Compare against K=1, uniform sampling, and a cheap larger predictor at matched cost. Neither high entropy nor large vector distance proves useful diversity; neither a noisy winner label nor low winner accuracy alone falsifies a mixture objective.

Account separately for candidate scoring, saved/recomputed activations, second visits, auxiliary vocabulary projections, and actual number of token/latent targets. Define scheduled candidate updates so skipped measurements are **null/not executed**, never logged as zero loss. Keep longer rollout horizons distinct from the required two core visits.

### 8. Define the residual topology and test selective access

Draw one computational graph showing the main pooled path, three CRCs, aligned forecast, concept-history read, both E2 streams, both recurrent visits, auxiliary heads, and RBF insertion point. Mark different token/group rates, stop-gradients, train-only branches, and caches.

Select a primary intra-module residual mechanism after comparing ordinary residuals, NCP IRC, Kimi/block depth attention, and mHC-style expanded streams. Do not stack all of them by default. Separate output gating of token attention from attention over depth. If you combine mechanisms, identify the distinct failure each solves and the added cost.

Answer explicitly: **Are the main pooled route and CRCs multiple information paths, multiple simultaneously maintained residual streams, or both in your equations?** Reconcile sequence lengths and widths before treating them as the same state space. An mHC analogy becomes an implementation only after specifying stream count, mixing operators, constraints, initialization, and composition across recurrence.

For 200 unique blocks, bound the number of retained depth sources, gradient storage, memory traffic, and cross-device transfers. Provide a block/source summary alternative to reading every prior layer at every destination. Derive attention costs from the chosen mechanism: source-depth mixing is not automatically token cross-attention of cost O(TC).

Train the model to use exact-detail and predictive routes without making the concept route either an irrelevant extra branch or an artificial bottleneck. Consider route dropout or capacity restrictions only as explicit ablations. Use train-from-scratch controls as well as inference interventions so a route-dependent model is not mistaken for a better architecture. Test concept swaps, matched mean substitutions, encoder-detail perturbations, family-specific route ablations, and second-visit removal with appropriate equal-magnitude controls.

### 9. Retain RCDL with a precise memory lifecycle

Keep a frozen-base RBF correction stage and a gated-adapter consolidation stage in the design. Identify exactly where the bank reads and adds its correction relative to concept recurrence, forecasting, and decoding. If it corrects the final concept state, a write gradient must account for every downstream route consuming that corrected state; do not silently update only the PQ branch while a continuous CRC remains dominant.

Specify kernel, key normalization, code/visit conditioning, correction dimensions, bandwidth calibration, trust-region or write cap, capacity/eviction policy, and write authorization from labeled training/adaptation episodes. Specify whether correction paths also affect auxiliary heads. Use an immutable memory-era identity covering relevant model weights, tokenizer, preprocessing, pooling, query transform, head definitions, and memory hyperparameters.

Explain what happens when the representation, EMA teacher, adapters, or readout changes. Can the old bank still be addressed and interpreted? If not, require an explicit reset, re-keying, or validated migration. A teacher-stable target does not automatically stabilize an online RBF query. Copying a correction vector into a new coordinate system is not consolidation.

Define the local-support gate's positive and negative reference distributions, per-visit handling, causal state initialization, OOD behavior, and overlap policy. References must cover previously learned families. Retention is measured per family and on related skills; surface-form separation alone is insufficient. Count any extra adapter-off forward pass used to obtain frozen gate queries.

For sleep, define the bank-augmented teacher, trainable adapter student, replay mixture, distillation targets, acceptance criteria, rollback state, and bank reset timing. Evaluate **several sequential adaptation phases**, including related arithmetic skills and changes of format, with forward transfer, backward transfer, worst-family forgetting, exact accuracy, and elapsed adaptation cost. Compare plain LoRA, gated LoRA, RBF only, replay-supported adapters, and consolidation in a staged budget. Explicitly exclude all calibration IDs from final bank writes and score on untouched examples.

### 10. Treat meta-control as a testable later capability

Explain what a concept would control: a future-state goal, a decoder modulation, an internal operator, or merely a prediction feature. If claiming temporally extended actions, specify initiation, persistence, causal termination, and an executable policy. Two recurrent refinement visits do not constitute a long-duration action.

Compare end-to-end predictive pretraining with a staged route: learn a useful base representation first, freeze selected components, discover controller structure with permitted hindsight supervision, then train or distill a causal controller policy. Reconcile the temporal-abstraction paper's frozen-backbone evidence with V11's desired joint training. Do not activate privileged future conditioning during ordinary language evaluation.

Make post-training optional in the initial implementation but architecturally compatible: supervised tasks or interactive environments, explicit actions/rewards where appropriate, and recomposition tests. Demonstrate that a persistent concept intervention changes a useful multi-token behavior across contexts before calling it an option or a world-model controller. Preserve ordinary language quality and exact copying as constraints.

### 11. Design a scaling path with real accounting

Separate three deliverables:

1. **Mechanism test bed:** roughly V10.3-tiny scale, with inexpensive controls and corrected evaluators.
2. **Practical V11 reference:** width 128, <10M trainable dense parameters plus ~20M Engram, aimed at the existing Apple MacBook Air with **16 GB shared/unified memory and an 8-core Metal GPU**. Use a 16-hour pilot budget, optionally 24 hours only after specified evidence; this is a plan, not authorization to launch here.
3. **Scaling hypothesis:** intermediate model sizes before hypothetical 1T and 3T totals, with dense versus MoE alternatives, total versus active parameters, tokens, training FLOPs, communication, and serving costs. Give symbolic formulas or explicitly assumption-based estimates where hardware is unspecified.

Compare encoder-heavy, concept-heavy, and balanced allocations at matched parameter and compute budgets. Include my 120/60/20 proposal as a stress case. Layer count alone is not parameter allocation, and compressed concept layers may have very different FLOPs from token-rate layers. Specify which of the 120 encoder blocks belong to recurrent E2; do not multiply every encoder block by two. Count shared recurrent weights once but their computation twice.

Quantify vocabulary/head cost, ordinary 4× FFNs, attention/KV storage, token length T, concept length approximately T/4, per-visit caches, CRC source storage, teacher copies, candidate work, adapters, RBF tables, Engram transfer/optimizer state, and activation checkpointing. Use the actual dimensions of the two-slot representation. Keep peak memory and bytes moved separate from parameter count. Full-concept attention still scales quadratically in concept length unless you change it explicitly.

For Metal, propose measured benchmarks for pooling, attention masks, candidate batching, auxiliary logits, CPU Engram lookup/transfer, synchronization, logging, evaluation, and checkpointing. Report ordinary-step, candidate-step, and amortized throughput, useful scored tokens/second, p50/p95 step time, thermal effects, and peak memory/swap. Do not improve a throughput number by silently dropping objectives or counting padded/candidate tokens as ordinary data. A candidate/evaluation schedule can explain a jagged chart without constituting a hardware problem.

### 12. Experiments, data, and telemetry must decide the design

Use established natural text and symbolic datasets with difficulty appropriate for each model size. Do not choose tasks whose solution format deliberately matches the concept group length. Inspect licensing, train/dev/test separation, duplicates, target leakage, tokenization, and answer-length distributions. Keep Sudoku as one stress test, not the principal evidence of language reasoning. Add layout/random-prefix controls to training as well as matched held-out evaluations, so an OOD header edit is not the only alignment test. Preserve the same tokenizer and scoring units for each comparison; raw NLL is not comparable across tokenizers.

Propose a **prioritized, affordable** matrix rather than a factorial sweep. It should identify:

- A concept-free control with the same shared backbone and a separate parameter- or compute-matched ordinary transformer control.
- V10.3 reproduction and a no-auxiliary-loss concept arm to separate topology from auxiliary-gradient interference.
- The chosen NextLat adaptation against the chosen HiLP adaptation, changing only the relevant objective/head where feasible; then the minimal combined recipe only if justified.
- Two-visit versus one-visit diagnostic controls, K=1 versus selected exploration, individual CRC ablations, and meaningful target/teacher alternatives.
- Decoder size/allocation and Engram on/off studies, scheduled after the main mechanism is credible.
- Multi-phase memory/consolidation evaluations with per-family retention constraints.

Use matching data streams and paired document metrics, at least two seeds for screening and a third seed for a promotion claim when feasible. Distinguish example-sampling uncertainty from run-to-run variation; a document bootstrap cannot replace seed replication. Show quality at equal updates/data, equal FLOPs, and equal active wall time where feasible; label any missing comparisons. Predeclare a practically meaningful effect or non-inferiority margin and the confidence-interval rule, with a compute/latency allowance. Do not pass a gate merely because two near-zero accuracies are statistically indistinguishable.

Measure decoder NLL and generation accuracy, latent forecast quality against mean/persistence/linear-input baselines, calibrated distributional scores for multimodal predictions, source-controlled concept interventions, visit-2 improvement, prior-versus-oracle candidate utility, long-horizon transfer, and retention. A fixed-target linear probe measures accessible information under that probe, not every possible form of information. Lower auxiliary loss without better downstream quality is an unresolved result.

Provide correctness tests for future-token perturbation, all group-boundary lengths, packed-document isolation, train/inference cache parity, teacher forcing versus free generation, stop-gradient contracts, candidate-gradient correctness, RBF write derivatives, first-token gate behavior, and exact checkpoint resume including optimizers, schedulers, RNGs, data position, Engram, memory, and adapter/gate state. Include adequate generation limits and report truncation separately from wrong answers.

Specify a versioned telemetry schema suitable for a later web UI. Include run/config/code/data hashes, step and active wall time, family, scored-token denominators, all raw and weighted losses, `candidate_executed`, rollout horizon, gradient/clipping/update metrics by module, target scale/rank/drift, PQ usage, CRC intervention summaries, candidate predictive metrics, recurrent-visit gains, RBF writes/reads/support/caps, gate-open rates, per-family evaluation confidence intervals, truncation counts, latency categories, memory/swap, and checkpoint provenance. Distinguish absent/not-executed metrics from zero. Bound telemetry overhead and keep asynchronous evaluation tied to an explicit checkpoint.

### 13. Required response and final review

Produce a document suitable to save as `V11_RCDL_Architecture.md`, organized as follows:

1. **Recommendation:** one primary architecture and the specific V10.3 failures it addresses. A short table separates retained components, changed components, optional experiments, and rejected alternatives.
2. **Source/evidence ledger:** what you actually read, exact supporting sections, material ambiguities, and which claims are hypotheses.
3. **Computational specification:** graph, tensor-shape table, token/group/visit clock, release/shift table, three CRCs, residual topology, and RBF location.
4. **Objective specification:** equations, target policy, gradient-flow table, exploration prior/posterior distinction, and train versus inference behavior.
5. **Pseudocode:** one ordinary training step, one candidate step, incremental generation across a group boundary, one RBF update, and one sleep/consolidation cycle.
6. **Configurations and cost:** explicit tiny and <10M configurations, verifiable parameter arithmetic, compute/memory estimates, and a conditional path to larger models. Give a smallest worthwhile experiment first.
7. **Evaluation and telemetry:** dataset plan, prioritized ablations, baseline budgets, correctness checks, success/failure thresholds, and schema.
8. **Adversarial review and revision:** find the strongest causal, gradient, collapse, bypass, candidate-selection, memory-drift, and scaling failures in your own proposal. Correct resolvable problems in the final specification and list unresolved risks. Ensure the final diagram, equations, and pseudocode agree.

Finish with three explicit answers: **Why should this concept state become more useful than V10.3's? What result would falsify that explanation? What is the cheapest experiment that should happen before a 16-hour run?** If evidence supports only a training auxiliary rather than a deployed world-model/controller role, state that clearly while preserving the requested ECD research design.

Do not fabricate experiments, claim trillion-scale validation, force all papers into one architecture, or use model size as an explanation without a discriminating test. Make a reasoned choice where several designs are possible; leave a short list of consequential uncertainties rather than an unspecified menu.

## END PROMPT
