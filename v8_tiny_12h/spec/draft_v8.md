# V8: a competent encoder–concept–decoder, controllable concepts, and local continual learning

Date: 5 October 2026. Revision: substantive second review from first principles. Status: **design only; no V8 implementation, training, or validation has been performed.**

This design combines the useful questions from V6 and V7 with Local Support Learning (LSL). It does not assume that their mechanisms already work together. It is based on the three preceding drafts, the V6/V7 implementation and saved evidence, and the supplied `LOCAL SUPPORT LEARNING.pdf`. The review at the end is a self-review, not an independent experiment.

## 1. Decision and scope

Build a roughly **2.45M-parameter dense encoder–concept–decoder** and give it enough task-specific training to establish competence. Preserve next-concept prediction as an explicit experimental arm, alongside an otherwise identical alpha-zero arm and an ordinary stacked Transformer of almost exactly the same size. Only a competent checkpoint may proceed to concept-core replacement, exploratory latent training, GradCuit-style refinement, or AB-MCTS.

After that, test **phase-specific, activation-gated LoRA** for continual learning. Freeze the accepted backbone and earlier adapters; learn a new adapter on each new distribution and fit a local-support gate for its activations. Revalidate retention, concept controllability, and search on every accepted model version.

The intended model remains:

```text
tokens → causal encoder → complete-chunk pooling → causal concept Transformer
   │                                            ├→ forecast head → soft PQ concept
   │                                            └→ continuous concept port
   └────────────────────────────→ causal shift/repeat → causal decoder → next token

After competence:
  compatible replacement core → prior-trained Z controls → GradCuit / AB-MCTS

After accepting an anchor:
  local-support-gated LoRA banks on E/C/D attention projections
  → new accepted model version → repeat retention and search compatibility checks
```

Two kinds of adapters have different jobs. **Z adapters** change a candidate answer through temporary inference controls; their weights are trained once per accepted anchor. **Continual LoRA adapters** store learned updates from successive datasets. Search changes Z, not LoRA weights. Continual learning changes the accepted model version, not an individual search candidate.

The main design choices are deliberately limited:

1. Start near the capacity and optimization regime that learned in V6.
2. Budget graph examples, useful decision targets, and text targets separately.
3. Test the learning chain before launching a large experiment matrix.
4. Keep a useful deterministic path and permit every proposed enhancement to lose.
5. Make cache state, gate state, model ownership, and information access explicit.
6. Scale ordinary dense training first. Search and continual adapter banks have separate costs.

This document specifies the complete research program. Passing a gate permits the next experiment; it does not require deploying every component. The final accepted model may have alpha zero, simple routing, no refit, ordinary retries, or fewer continual phases if the evidence favors those choices.

## 2. What went wrong, and what the evidence actually supports

### 2.1 V7 did not establish a competent foundation

| Observation | Saved evidence | Consequence for V8 |
|---|---|---|
| V6 alpha-zero foundation: 2,022,144 parameters, 9,000 updates, 288,032 graph instances including initialization sampling | `v6_tiny/runs/A_no_forecast_prefix_s0/train_summary.json` | Reproduce this setting before interpreting a new recipe. |
| Its development greedy success was 51/64 standard and 37/128 hard | `v6_tiny/runs/eval_no_forecast_prefix_dev/summary.json` | A small model can learn this particular graph protocol. This is not independent-seed or scaling evidence. |
| The matched V6 NCP prefix scored 0/64 standard and 0/128 hard | `v6_tiny/runs/eval_forecast_prefix_dev/summary.json` | The NCP coefficient, units, and gradient interaction need investigation. |
| V7 hierarchy foundations had 5,365,938 parameters; the flat baseline had 5,366,208. Each received about 3,825 updates and 36,700–36,900 graph examples | `v7_tiny/runs/A_*/train_summary.json` | More parameters did not replace missing task exposure. |
| V7 also changed data, context, task mixture, graph difficulty, normalization, initialization, routing, loss scaling, and schedule | V6/V7 model, data, runtime, and driver code | Do not attribute the failure to model size or NCP alone. |
| All nine principal V7 foundations had zero standard/hard navigation success on development evaluation; ordinary Transformers also failed | `v7_tiny/runs/A_*/dev.json` | Investigate shared learning and decoding before blaming the hierarchy. |
| NCP seed 272 failed at the first move on 113/128 standard examples; single-hop lookup was weak across foundations | Saved development trajectories and lookup evaluations | Explicitly measure edge retrieval and correct decision tokens. |
| Refit preservation could pass at zero success before and after | V7 refit development summaries and gate code | Preservation requires absolute competence as well as a difference bound. |
| Development search completed 864 method–budget–problem records with zero successes | `v7_tiny/runs/dev_search/summary.json` and candidate records | These are repeated comparisons on 32 problems, not 864 independent problems. Stop expensive downstream work when coverage is absent. |

V6's full NCP foundation saw about 576,000 graph training examples; V7 saw roughly sixteen times fewer. Relative to the successful V6 alpha-zero prefix, V7 saw roughly eight times fewer. V7 hard graphs also had 14 nodes rather than V6's 10. **Insufficient exposure is a leading explanation, not a demonstrated sole cause.** Extending the unchanged V7 recipe is not guaranteed to repair it.

The V7 verifier accepted valid alternate paths and independently checked reference answers were legal and within the response limit. There remains a trained-decoding validation gap: short text cache checks and untrained fixtures do not establish equivalence of optimized decoding on long, trained graph prompts. This is a gap to close, not a proven decoder bug.

V7 was stopped by user request on 5 October. `v7_tiny/runs/status.json` records `stopped_by_user`; the interrupted stage was `A_flat_s171_test`. Do not describe its partial work as a completed sealed-test study or infer failure from intentionally failing preflight fixtures.

### 2.2 Keep mechanisms as hypotheses, not inherited conclusions

- **NCP:** positive evidence in the NCP paper and historical V5 makes it worth testing. V6 supplies strong contrary evidence for one tiny recipe. Alpha zero removes the auxiliary gradient; it does not remove the encoder, concept stack, soft codebook, continuous port, or decoder.
- **Replacement and transplant:** compatible cores can preserve behavior, but preserving zero skill is not useful. A refit must preserve a competent anchor and justify its additional training or lower deployment cost.
- **Exploratory modeling:** V6 prior sampling was promising on hard examples at equal candidate count. This does not establish a robust compute-matched advantage over retries.
- **GradCuit:** there was evidence of useful gradient directions in earlier runs. V6's primary comparison did **not** establish superiority over retry: standard 20.31% versus 23.44%, hard 8.59% versus 8.98%; both paired intervals included zero.
- **AB-MCTS:** an allocation method needs candidate coverage and useful refinement. It cannot manufacture graph knowledge from a consistently incompetent proposal model.
- **LSL:** the supplied paper studies large pretrained models and reports useful retention, with explicit limitations. Its behavior in a 2–3M hierarchy, or under Z interventions, is untested.

The complete V6 replacement/search pipeline was built from its weaker NCP foundation. It was not established on the strongest alpha-zero checkpoint. V8 must test the pipeline on the checkpoint actually shown to be competent.

## 3. First-principles requirements

### 3.1 Information must arrive before a decision

A graph action requires retrieving an edge from the prompt before selecting a node. Therefore preserve a full token-resolution encoder-to-decoder route. Pooling is an additional abstraction, not the only storage for exact facts. A model that cannot retrieve one edge does not yet provide a meaningful test of multi-hop latent search.

### 3.2 An auxiliary target must help the task gradient

Predicting the next pooled concept may encourage useful abstraction. It may also compete with node selection, especially when its target representation moves during joint training. Measure loss units, gradient magnitudes and alignment, then compare held-out behavior. A low NCP error or broad code usage is not sufficient evidence of useful concepts.

### 3.3 Search needs authority, coverage, and an honest evaluator

Z must affect future decisions without corrupting the reference model. Prior sampling must reach useful alternatives. A verifier must evaluate actual generated answers using public task information. These are separate tests; nonzero latent gradients satisfy none of them by themselves.

### 3.4 Continual learning is a locality problem, with limits

An update that helps a new distribution should ideally remain inactive elsewhere. Gates can approximate that locality using layer activations. However, old and new examples may share activations, or require contradictory outputs for the same observable input. No support gate can retain two contradictory mappings without additional context or an explicit precedence policy.

A useful backbone also does not guarantee useful concept representations. The direct token path may carry almost all task information. Forecast accuracy, code usage, controlled interventions and retrained port ablations must distinguish a competent model from a competent **concept interface** before attributing gains to latent reasoning.

### 3.5 A scalable design needs bounded, accountable work

Use distinct dense Transformer layers for capacity, parallel teacher-forced training, and explicit memory/computation formulas. Do not put unbounded tree search inside the required foundation training step. Count every recurrent visit, candidate backward pass, gate density evaluation, adapter bank, and reference-model copy.

## 4. Backbone and exact tiny configuration

### 4.1 Three starting models

| Model | Purpose | Configuration | Learned parameters |
|---|---|---|---:|
| V8-C bridge | Reproduce V6 learning before changing the recipe | V6 E3/C2/D3, d128, FF384, 4 heads, vocabulary 2048 | 2,022,144 |
| **V8-D primary** | Dense hierarchy with a larger independent concept stack | E3/C4/D3, otherwise the same block dimensions | **2,448,640** |
| Ordinary Transformer | Primary architecture baseline | 10 full-token causal blocks, d128, 4 heads; FF400 in the first 9, FF384 in the last | **2,450,048** |

All use tied input/output embeddings, rotary positions, pre-RMSNorm, bias-free attention/FF projections, and gated FFNs. The hierarchy also contains its forecast head, codebook, continuous port, and decoder injection scale. The flat model contains none of those modules. The FF widths above are active capacity, not unused parameter padding; the size difference is 1,408 parameters, or about 0.058%.

For this configuration a hierarchy block has `13d² + 2d = 213,248` parameters. Adding two concept blocks to V6 gives 2,448,640. The ordinary block with FF width f has `4d² + 3df + 2d` parameters. These are analytic counts for the specified architecture; the implementation must assert actual instantiated counts and report active, resident, and trainable counts separately.

Start with FP32, context T=192, pooling factor k=4, eight PQ segments of width 16, and 32 code entries per segment. Keep the V6 tokenizer and special-token identities for the bridge. Keep the same vocabulary for the main tiny comparison. Increasing vocabulary or changing tokenization is a different experiment.

V8-D's first architectural change from the bridge is C2→C4. QK normalization, residual-scaled initialization, full routing, longer context, and harder graph mixtures are separately identified interventions. They must not all be introduced before verifying the bridge and the extra concept depth.

### 4.2 Causal computation

Let x_t be the input token at zero-based **document-local** position t; logits at t predict x_(t+1). Each attention operation is causal and document-isolated. For a document starting at packed offset b, add b to the token indices below; conceptual chunk boundaries belong to the document, not its accidental row offset.

```text
H_t = Encoder(embed(x))_t
C_j = RMS(mean(H_(4j), ..., H_(4j+3)))
U_j = ConceptTransformer(C_≤j)_j
p_(j,s) = softmax(W_head,s U_j + b_head,s)
qbar_j = concat_s Σ_c p_(j,s,c) Q_(s,c)
A_j = W_A RMS(U_j)
D_t = H_t + shift(qbar)_t + γ ⊙ shift(A)_t
logits_t = tied_head(Decoder(D)_t)
```

`γ` starts at 0.1 as in V6. `shift(v)` first exposes v_j at logit position `4j+3`, after its entire source chunk has arrived; it then holds that vector for four prediction positions. Thus the first predicted token receiving v_j is x_(4j+4). Initial positions receive zero concept injection. Partial chunks do not expose a future-pooled vector.

An incomplete final document chunk produces no usable concept. Document boundaries reset positions, attention history, pooling state, Z segment indexing, and later LSL smoothing state. No shifted vector crosses a document boundary. The primary fast packer follows V6: place every document start at a multiple of four using unscored padding. A document-local reference implementation must agree with that packed path. A fast path receiving an unaligned document start must repack correctly or reject it explicitly; simply discarding a mixed chunk and resuming a row-global grid changes that document's later pooling boundaries. Test packing equivalence as well as absence of cross-document leakage.

During generation, encode the actual generated token prefix and refresh completed chunks. Do not roll predicted concepts forward as if they were verified world states. Prefill may end at any of the four chunk offsets; the first answer token can be affected by a concept whose source chunk contains the final prompt tokens. Cache invalidation starts at that source computation, not merely at the first generated token.

`qbar` is a differentiable mixture of discrete codebook entries; A and Z are continuous. **The deployed model is not fully discrete.** MCTS can sample finite actions over this control space without enumerating all PQ tuples or asserting symbolic meaning for a code index.

### 4.3 NCP, VQ, and gradient ownership

Define the main objective explicitly:

```text
L_A = L_token + α(u) L_NCP + L_VQ
L_token = 0.5 mean_text(scored-token CE) + 0.5 mean_symbolic(scored-token CE)
L_NCP = balanced_domain_mean[mean_features((qbar_j - stopgrad(C_(j+1)))²)]
L_VQ = mean_valid_chunks mean_segments ||Q_(s,nearest(stopgrad(C_j,s))) - stopgrad(C_j,s)||²
```

Only valid same-document adjacent concept pairs enter NCP, and the target chunk must include a scored prediction target. Graph headers are context, not answer CE. The next-concept target is detached; the prediction branch can still update its encoder input through the concept stack. VQ fits Q to detached pooled features and does not directly pull the encoder toward a quantized target. Token CE may update the encoder, concept stack, head, Q, A port, and decoder.

The VQ expression deliberately preserves V6's **sum over each 16-dimensional segment**, followed by averaging over segments/chunks. A per-feature MSE would be sixteen times smaller; do not silently change it. At another width, record segment width and any unit conversion. Log token loss, NCP, VQ, and gradient norms separately; comparing raw total loss across different alpha values is misleading.

The primary NCP schedule proposal is alpha zero through update 6,000, linearly increasing to **0.1** over updates 6,000–8,000, then constant. This is a testable attempt to let token competence form first, not a known solution. Mandatory control: alpha identically zero, same architecture, data order, optimizer horizon, and evaluation. The historical bridge also repeats V6's alpha=1 and alpha=0 prefixes unchanged.

Before confirmation, use one engineering seed to compare final alpha {0.01, 0.1} and diagnose constant-alpha behavior. Select by task competence and text performance, not total loss. Freeze the choice before independent seeds. A delayed NCP branch must retain all pre-activation learning metrics; it cannot claim an NCP benefit from skill already acquired at alpha zero.

At fixed diagnostic batches, record gradient norms and cosine alignment between token CE and NCP on encoder/concept/head parameters. Inspect text, node-choice, and formatting targets separately. These diagnostics identify conflict; they do not authorize an unregistered dynamic loss-balancing algorithm. If NCP damages competence, the alpha-zero branch remains the valid anchor and the NCP hypothesis remains unresolved or negative for that recipe.

The pooled target is still a moving representation despite `stopgrad`: the encoder changes through other gradient paths. Moreover, for a fixed representation, squared-error prediction tends toward the conditional mean of possible future concepts. When several legal continuations exist, that mean need not be a useful realizable continuation. Neither fact proves NCP is harmful, but both prevent interpreting lower MSE as solved planning.

Track forecast error against a same-domain persistence predictor, target variance/effective rank, segment code usage, and free-generation effects of independently removing qbar or A. These removals are distribution-shift diagnostics. Before claiming the ports are beneficial architectural components, train the relevant port-disabled control from scratch with its own active parameter count and matched training exposure; an inference-only perturbation cannot establish that claim. Add no collapse penalty or new target encoder silently if these diagnostics fail.

### 4.4 Optional V7 routing branch

The main model initially keeps the simple V6 ports. A separately trained routing variant tests all three cross-module directions and bounded intra-module history without changing the data or objective simultaneously.

For layer l, maintain a bank containing the module input, current residual, and completed four-layer group summaries. Normalize bank entries and use learned softmax coefficients to construct a mixture R_l. Add an extra term `(0.25/sqrt(module_depth)) tanh(a_l) ⊙ safeRMS(x_l) R_l` to the ordinary block input; initialize a_l=0. Store at most four recent summaries plus the module input and current residual. This bounds the extra input's RMS relative to x_l; it does not bound the entire network's dynamics. At shallow depth this has no completed-group history available to a later layer; report that fact instead of claiming the full paper IRC was exercised.

For CRC, retain the mandatory direct H route and add separately scaled source-depth mixtures: pooled encoder exports into C, encoder exports into D, and causally shifted concept exports into D. Use two encoder exports and two concept exports, at middle/final depths. Normalize source exports, use softmax mixing, and cap each added route by `(0.25/sqrt(destination_depth)) tanh(a_route) × safeRMS(destination)`, with a_route initially zero. Sum and log the bounds when two routes enter the same layer. Source positions must satisfy the same complete-chunk timing as the main ports. Forecast qbar and continuous A remain separately observable.

This is a V8 bounded-routing proposal, not an exact NCP-paper implementation or a claim that V6 already had IRC/CRC. Count its projection and routing weights; give it a correspondingly matched flat baseline if it becomes the promoted architecture. Demonstrate that routing parameters affect a used path and receive gradients at the actual depth. Select or reject it before final confirmatory training.

## 5. Data and training exposure: repair the shared learning problem first

### 5.1 Provenance and partitions

Create immutable manifests for engineering train/dev, confirmation train/dev, continual calibration, and final sealed tests. Previously inspected V5–V7 evaluations are historical development data. Do not call them fresh tests. Use new graph seeds and canonical graph/query hashes, document-level text splits, exact token-content deduplication, and a recorded near-duplicate policy. The tokenizer and dataset revisions are part of the manifest.

A graph identity includes edge structure, query, node labels, and relevant generation constraints. Keep isomorphic/relabelled pairs grouped when testing structural generalization. Exclude held-out identities during generation and record actual sampled hashes. A test against an empty `seen_symbols` set is not evidence of exclusion: generate examples first, then verify the nonempty recorded exposure set.

Train and evaluate four distinguishable skills: direct edge lookup, short composition, pointer/binding, and full valid graph navigation. Keep graph headers, response grammar, public goal, and legal edge rule available to every baseline. Randomize irrelevant orderings to prevent a position shortcut, but use fixed shared streams for paired comparisons.

### 5.1.1 Separate deterministic retrieval from ambiguous demonstrations

In the V6 generator, hidden sampled colors and durations create a valid walk; the prompt exposes the graph, start and goal, not that entire hidden plan. Multiple legal answers may therefore correspond to the same public input. The model cannot be expected to reproduce an unobserved random choice exactly. The destination after a given current node and chosen color **is** deterministic, so conditional node retrieval remains an identifiable learning target.

Report (a) teacher-forced node accuracy given the reference color, (b) node accuracy given the model's chosen color and legal generated prefix, (c) first complete legal transition, and (d) strict goal-reaching success. Do not substitute exact demonstration match or color-token accuracy for legal planning. The ≥98% first-move gate below means a complete legal transition under the public graph, not equality to the demonstrated move.

Preserve the original demonstrations for the historical bridge. Before main confirmation, run one isolated engineering comparison of original random-walk demonstrations versus a deterministic canonical valid path for the **same sampled graph/start/goal distribution**. A fixed-tie-order BFS may construct training labels offline; all models receive the same labels, and data-generation cost is recorded. No solver answer is supplied to the inference/search policy. Report differences in scored targets and path lengths, and run a token-budget-matched sensitivity comparison if drawing an efficiency conclusion. Choose the supervision recipe before confirmation; if this comparison is not run, retain historical supervision and name the ambiguity as unresolved.

Keep graph size and goal-selection difficulty separate. The common protocol includes 10-node standard goals, **10-node hard goals at shortest distance ≥4** for the V6 bridge, 14-node hard goals, and 18-node long graphs. Increasing node count is not itself a definition of hardness. Record shortest-distance strata and graph rejection rates so changes in conditional sampling are visible.

### 5.2 A small learning ladder before long runs

| Check | Required observation | Action on failure |
|---|---|---|
| Reference/scorer fixtures | All generated reference paths accepted; malformed/illegal paths rejected; alternative legal paths accepted | Repair the harness, not model size. |
| Tiny memorization | Near-100% correct generation on 128 fixed lookup/short-path training examples for hierarchy and flat model | Inspect labels, gradients, masks, tokenizer, capacity and decoding. |
| Fresh one-hop retrieval | At least 95% point accuracy on a held-out 512-item engineering set | Do not start long navigation/search runs. |
| Fresh two-hop composition | At least 90% on 512 items after a bounded dedicated training check | Inspect exposure and conditioning on the generated first action. |
| V6 bridge | Reproduce useful V6 alpha-zero behavior with the same data recipe, LR horizon and eager reference decoding | Determine whether the discrepancy is data, implementation, or stochastic variation. |
| V8 depth bridge | C4 must learn the same simple tasks under the same protocol | Retain C2 as the working control; isolate why depth changed learning. |

These are proposed engineering gates. The dedicated learning checks train on their corresponding simple tasks; they do not demand untrained one-hop generalization from a navigation-only checkpoint. Set initial caps of 2,000 updates for the 128-example memorization check (require ≥99% strict training success) and 6,000 updates for the fresh one/two-hop ladder. Use checkpointed learning curves and the same caps for hierarchy and flat controls. A failed cap triggers a recorded diagnosis or separately declared extension, not automatic downstream training. These checks establish learnability and correctness, not final model quality. A memorization checkpoint never becomes the primary foundation.

### 5.3 Main foundation allocation

Start each confirmation arm from scratch with **24,000 optimizer updates**, effective batch 64, context 192. Per update allocate 32 text rows, 24 GraphNav rows, four lookup rows, two composition rows, and two pointer rows. Initially keep graph tasks at 10 nodes and within the context/response envelope.

| Exposure per foundation | Planned count, excluding initialization probes |
|---|---:|
| GraphNav examples | **576,000** |
| Lookup examples | 96,000 |
| Composition examples | 48,000 |
| Pointer examples | 48,000 |
| Text rows | 768,000 |
| Total input slots | 294,912,000 |

These are exposure counts, not automatically unique examples. Packing can include multiple text documents in a row. The trainer must additionally log unique documents/graph identities, nonpadding inputs, scored targets by task, node-choice targets, edge retrieval opportunities, and repetitions. Padding, prompt headers, and replayed candidate evaluations must not be counted as new supervised targets.

Use an initial short-path curriculum within the 10-node training family, calibrated on the engineering seed, then fix the same update-indexed schedule for all confirmation arms. Its graph/query distribution must converge to the historical V6-style distribution and retain short-path examples; the demonstration policy is locked separately under section 5.1.1. Do not advance different architectures through easier/harder data at different update numbers in the main matched comparison.

The main text mixture is a fixed 50/50 allocation of scored text targets from TinyStories and the selected FineWeb-Edu training subset, with documented sources and deduplication. The historical bridge retains V6's original text mixture. Changing from that bridge mixture is an explicitly recorded data intervention. No claim about general language quality follows from this small corpus and vocabulary.

Keep the 32 text rows fixed while a deterministic source scheduler tracks a cumulative scored-target deficit and chooses the next source toward 50/50. Exact equality in every batch is not required; log the realized share. Do not quietly double rows, truncate targets differently by architecture, or present 50/50 document draws as 50/50 target exposure. Auxiliary task quotas guarantee examples, not equal gradient mass: the main symbolic CE remains token-averaged, and per-task target counts/gradient contribution must be reported. A task-reweighted loss is a separately named engineering intervention.

Record metrics at updates 2k, 4k, 6k, 8k, 12k, 18k and 24k, with cheap intermediate health logs. Include first-hop accuracy, node-choice CE/top-1, format CE, strict complete-path success, legal prefix length, termination rate, and ordinary text NLL. Low formatting loss cannot substitute for correct nodes.

Twenty-four thousand updates is a proposed initial allocation, not a theorem about necessary data. If capability is still improving but below gate, a separately declared exposure extension may be studied symmetrically. If the engineering run fails, stop dependent work and diagnose it before replicating the failed recipe across seeds. Never quietly reduce each model's exposure to squeeze more arms into a fixed wall-clock window.

### 5.4 Harder and longer tasks come after 10-node learning

After standard competence, run a separate continuation stage at context 384, initially 8,000 updates with the same effective batch and per-task quotas. Of the 24 GraphNav rows, keep 12 standard 10-node examples, eight 14-node examples and four 18-node examples. This adds 192,000 GraphNav exposures, half rehearsing the original distribution. Keep answer allowance proportional to the generated valid reference envelope and validate it before training. Pin a maximum context and response limit per split; a response that exceeds the allowance is a failure/censored attempt, not permission to extend the model's window silently.

This is supervised curriculum continuation with rehearsal, **not** the later continual-learning experiment with reference fitting and no old-task gradient replay. The same continuation is available to every baseline. Report standard retention, 14-node and 18-node results separately. Unseen-length evaluation must use lengths not already included in training and must not be called validated long-context language modeling.

### 5.5 Optimization and throughput

For the V6 bridge, reproduce the original optimizer, initialization, 18k LR horizon and 9k prefix exactly. For the main 24k proposal start from AdamW with betas (0.9,0.95), epsilon 1e−8, peak LR 0.002, weight decay 0.1, 300-update warmup, cosine decay to 10% of peak, clipping at global norm 1. Keep norms/biases out of weight decay. Q has no decay and initially retains the V6 10× LR multiplier; Q initialization, early dead-code restart schedule and optimizer-moment resets must be pinned from the bridge implementation. A 1× Q-LR alternative is a separate engineering ablation.

Allow the flat model the same small LR-search budget, then freeze its selection before confirmation. State both exposure-matched and total-training-compute comparisons. Same parameter count does not imply same FLOPs: four hierarchy blocks run at quarter sequence length, whereas all flat blocks run at token resolution.

Use fixed shape buckets, pretokenized/memory-mapped text, vectorized symbolic generation, bounded prefetch, and effective-batch accumulation where needed. Benchmark sustained execution after compilation warmup while reporting compilation separately. Report input slots/s, nonpadding tokens/s, **scored targets/s by domain**, graph examples/s, memory, step time and end-to-end stage time. Never compare a slots/s number to a targets/s number.

Establish forward **and gradient** agreement on real batches before enabling compiler, fused, mixed-precision, or static-cache paths. Preserve a full eager reference. Prior MPS compilation issues make a matching loss alone insufficient. No optimization may change masks, candidate reduction, optimizer-step count, or data exposure silently.

### 5.6 Gradient and schedule contracts across stages

Freezing a weight is not equivalent to running its forward computation under `no_grad`. The frozen decoder must propagate derivatives to a refitted core or Z; frozen blocks after a new LoRA bank must propagate derivatives to that bank. Cache/detach a prefix only when it is independent of **every** currently optimized variable.

| Stage | Variables updated | Frozen computation that must retain input gradients | Safe detached reference work |
|---|---|---|---|
| Foundation | All declared backbone weights | None by design | Detached NCP/VQ targets only |
| Refit | New core blocks/norm and W_A | Frozen PQ head and decoder | Fixed encoder features and teacher outputs |
| Z training / GradCuit | Z-adapter weights / candidate Z respectively | Controlled core, head and decoder | Fixed-token encoder features and zero-reference probabilities |
| Continual LoRA | Only the new phase's A/B matrices | Every downstream layer, including frozen prior banks on the active path | Reference-density fitting and offline evaluation |

For gradient accumulation, sum loss numerators and use the effective batch's true domain/target denominators; averaging microbatch means with unequal token counts changes the objective. NCP and VQ require their own valid-pair/chunk denominators. Z selection uses mean CE per eligible example followed by domain balancing. Empty domains contribute no fake zero-loss observations; a single-domain continual phase uses ordinary mean CE over its scored targets. Count each underlying training example once, regardless of candidate/replay forwards.

Proposed non-foundation optimizer defaults, locked after engineering: harder-context continuation retains AdamW moments but starts a new 8k cosine schedule at peak 5e−4 with 200 warmup updates; refit starts a fresh optimizer at peak 1e−3, weight decay 0.1, warmup 200; Z training uses peak 1e−3, weight decay 0, warmup 200. These use the foundation betas/epsilon, global gradient clip 1, and decay to 10% of peak. Continual LoRA starts a fresh optimizer per bank, uses the selected 1e−4/3e−4 peak, zero weight decay, 100-update warmup and the same cosine floor. Retain the accepted alpha after foundation; do not restart its ramp in refit or curriculum continuation. Same-objective controls use the same optimizer reset policy, data order and horizon. These are V8 proposals, not inherited empirical optima.

## 6. Foundation gate and experimental ownership

Use an engineering development set for repeated diagnostics, then a separate locked 512-item gate set per symbolic split for confirmation. Proposed standard competence thresholds are: one-hop point accuracy ≥95% with 95% Wilson lower bound ≥90%; first navigation move ≥98%; complete standard greedy success ≥80% with Wilson lower bound ≥75%. These are engineering acceptance criteria chosen to rule out the V7 failure, not universal definitions of intelligence.

Evaluate text separately. An architecture-quality claim requires a predeclared noninferiority test on text NLL (initial margin 0.02 nats/token) and a useful symbolic or efficiency advantage over the ordinary Transformer. A competent hierarchy may enter a component study even if it does not beat the flat baseline, but the report must not call it a superior backbone.

The primary foundation comparison is V8-D delayed-NCP, V8-D alpha-zero, and the ordinary Transformer, on three independent training seeds after engineering selection. Pair data schedules where possible, while separating initialization/training randomness from search randomness. If only one foundation seed is affordable, label it a pilot and preserve its full exposure; search seeds do not repair that limitation.

Report every scheduled foundation seed and the fraction passing the gate. Downstream estimates are explicitly conditional on competent anchors; dropping failed seeds and averaging only survivors must not inflate the architecture's overall success rate. Fix how an anchor is selected on development evidence before final testing, and compare methods on the same selected anchor before comparing different architectures.

Only checkpoints passing the competence gate can become downstream anchors. Freeze and hash the encoder, decoder, head, Q, interface definition, tokenizer, and data/evaluation protocol at that point. Record whether the accepted anchor uses NCP. A failed NCP arm does not block an independently competent alpha-zero arm, and does not establish that NCP is intrinsically ineffective. Gate sets are development/acceptance data, not sealed tests; intervals used after repeated selection are operational screens. Final inferential claims require the separately untouched test sets and locked choices.

## 7. Compatible replacement core and complete transplant

### 7.1 Primary dense replacement

Freeze the accepted foundation's encoder, decoder, forecast head, codebook, and γ. Replace only the owner of `(U, A)`: the four concept blocks, concept final norm, and W_A. Initialize the primary replacement as an exact copy. Consequently the initial replacement is behaviorally identical to the accepted foundation; any later gain comes from additional training or changed constraints, not the act of copying.

Train the replacement through the frozen decoder on fresh training examples. Use token CE, the anchor's declared NCP setting where applicable, and an initial interface penalty:

```text
L_refit = L_token + α_ref L_NCP
        + λ(u) [mean||U_new-stopgrad(U_anchor)||²
                + mean||A_new-stopgrad(A_anchor)||²
                + mean||qbar_new-stopgrad(qbar_anchor)||²]
```

Use feature means, balanced domains, and valid chunks. Start λ=1 for 10% of the run and decay to zero by 30%. Freeze Q: no VQ updates or restarts during refit. Initial budget: 6,000 updates with foundation quotas. A same-shape fresh-initialization replacement is an optional harder transplant test and must not replace the copy control.

The copied dense replacement has the same deployed size as its source. During training the teacher core is resident; count that copy and its extra forward work. A same-budget joint continuation and a concept-only continuation without the teacher penalty are controls for any claimed benefit of refitting. Copying the same core and continuing with the same objective is otherwise the same algorithm as concept-only continuation; do not count that relabeling as a distinct architectural advance.

Show two comparisons rather than calling unlike budgets equal: the same-update/data comparison at 6,000 steps, and a continuation control receiving the same total charged training compute/time as the refit. The latter may receive a different number of examples because it has no teacher forward. Apply this distinction to Z's two-candidate training as well. For a claim that the **whole pipeline** is more efficient, charge foundation, refit, Z, gate fitting and inference together and give the ordinary Transformer a comparable total development/training budget; component-only inference wins cannot establish that claim.

### 7.2 V6 recurrent replacement arm

Also test the V6-style recurrent alternative: two learned shared Transformer blocks, two outer passes with two low-state updates and one high-state update each, full gradients through all six visits, learned input/output projections and separate U/A outputs. Match the frozen interface. Keep per-visit caches distinct even when weights are shared.

This tests compression and recurrent computation, not the same parameter/computation hypothesis as the four-block dense core. Report six visits, active size and wall/FLOPs explicitly. The historical V6 implementation deployed 2,055,424 parameters before Z; a new implementation must recount rather than inherit that number by name. Dense and recurrent replacements need not both be accepted.

### 7.3 Acceptance and interpolation

Require the new core to pass the absolute standard competence gate. Require the lower confidence bound for paired standard success difference to be at least −2 percentage points, and the upper confidence bound for text NLL increase to be at most 0.02. Apply a separately powered hard-task retention bound where the anchor is competent. Plan sample size from engineering discordance/variance **before** opening the confirmation set, as specified in section 13.2. If the locked sample is inconclusive, label the result unresolved; do not keep adding examples until the interval passes.

Zero-before/zero-after is not preservation of a useful skill. A bootstrap interval [0,0] in that case cannot establish population equality. If a core fails, retain the accepted foundation and mark the replacement branch rejected.

For transplant diagnostics, interpolate the **complete interface**:

```text
U_λ = (1−λ) U_anchor + λ U_new
A_λ = (1−λ) A_anchor + λ A_new
qbar_λ = frozen_PQ_head(U_λ)
```

At λ=0 reproduce the anchor; at λ=1 reproduce the replacement. Intermediate λ is a diagnostic, not permission to choose λ on sealed tests. Never transplant only U while accidentally retaining the old A path. Deployment contains one accepted core; a dual-core mixture must disclose both cores and its cost.

## 8. Exploratory modeling: train useful Z, preserve zero exactly

Freeze the accepted deterministic model. In the dense primary core add two control sites, before concept blocks 1 and 3, each conditioned on its current input v:

```text
r(v,z) = U_z [ (1 + 0.5 tanh(G RMS(v))) ⊙ (V_z z) ]
I(v,z) = 0.1 safeRMS(v) · r(v,z) / sqrt(1 + mean(r(v,z)²))
v_controlled = v + I(v,z)
```

Use bias-free G, V_z and U_z, rank 16, d_z=8. Initialize G=0; initialize U_z and V_z from a zero-mean normal with standard deviation 0.08, and measure initial output authority on training-only calibration examples. Both factors must be nonzero so the bilinear path can learn. This smooth capped variant is inspired by V6/V7, not an exact copy of V6's clipping rule. The cap bounds injected RMS, not the eventual change in logits or behavior.

Each site has `2dr + r*d_z` parameters; two sites add **8,448**. For every trained adapter state, I(v,0)=0. The primary dense model with Z has **2,457,088** learned parameters. In the recurrent arm, use the two explicitly declared recurrent injection sites and count their repeated evaluations.

Controls are schedules of bounded Gaussian samples, projected to RMS ≤2, held for L∈{1,2,4} complete concept positions. Reset per document. Control indexing and which prompt-completion chunk first becomes active must be identical in training, cached generation and replay. Train text continuation controls only on the designated scored suffix, not arbitrary prompt likelihood.

For each example draw K=2 complete schedules. Compute scored suffix CE per example and use:

```text
L_Z = balanced_domain_mean[(1−β) min_k CE(example, Z_k)
                           + β mean_k CE(example, Z_k)]
β = 0.25 initially
```

Selection is over complete candidate schedules, not independently chosen tokens or chunks from different candidates. Zero is not included as a candidate that can permanently win and starve the adapter. This is teacher-forced training with a prior-sampled latent; it must later establish free-generation coverage. Do not train random Z to match a deterministic NCP target that suppresses variation.

Initial budget: 6,000 updates with the same example quotas; count two concept/decoder candidates and the actual backward cost. A concatenated candidate batch and sequential/replayed candidates must produce equivalent gradients and reductions. Freeze audits include every anchor tensor, Q, and normalization parameter.

Required controls are direct linear Z at the same two sites, β=0 best-only, β=1 mean-only, matched extra deterministic training, and Z trained directly on the original competent foundation without refit. These distinguish selection, gating, extra compute and replacement. A smaller staged screening matrix may reject weak variants before seed confirmation, but none may be described as confirmed without being run.

### Coverage gate

First verify exact zero behavior, prior finite outputs, nonzero causal authority, and no frozen-weight changes. Then use free generation on a fixed 512-item development screen per intended search split. For a primary hard-split search study, retain the proposed absolute floors of 10% deterministic greedy success and 20% best-of-eight prior coverage, with the latter's 95% lower bound above 10%. These alone do **not** pass the gate: an inert prior that repeats the zero answer can satisfy them.

Let b_i indicate zero-greedy success and p_i indicate success among eight independently sampled nonzero schedules with greedy token decoding. Report:

```text
zero coverage       = mean(b_i)
prior-only coverage = mean(p_i)
union coverage      = mean(b_i OR p_i)
conditional rescue  = sum((1−b_i) p_i) / sum(1−b_i)
```

In addition to absolute floors, require at least 64 baseline failures in the fixed screen, at least five distinct rescued problems, conditional rescue ≥5%, and its 95% Wilson lower bound >1%. Count problems, not successful draws. Keep all 512 items in the reported overall rate; the failure-conditioned subset diagnoses extra opportunity and is not the primary test population. If too few failures remain, label the search study **ceiling-limited** rather than calling the model incompetent. Do not sample extra problems adaptively to fill the failure subset. These are proposed operational thresholds to lock before confirmation, not evidence of superiority to token retries.

Measure useful disagreement and success, not merely distinct strings, latent variance, or teacher-forced loss. A privileged optimization toward a known reference may diagnose representational reachability, but it is never a primary inference method. If reachability exists but the prior fails, fix proposal training before expanding search. If both fail, stop downstream search for that split after a bounded diagnostic (at most 32 examples and 15 minutes).

Log changed legal actions and repeated-output rates at each control hold length. Compare the same rescue statistics for token retries. A positive prior gate merely justifies the cost of comparing search methods; only the later paired, equal-cost comparison can establish a gain. Label a gate failure by cause—weak base, inert controls, absent prior reach, numerical failure, or ceiling limitation—because these imply different next experiments.

## 9. GradCuit and AB-MCTS on realizable concept interventions

### 9.1 Common environment and budgets

Freeze model and adapter weights. The state required to reproduce a candidate includes the prompt, actual token history, Z schedule, optional code patches, model/gate version, generation settings, RNG, concept positions and cache provenance. A graph node ID alone is not a sufficient neural state. Search depth, neural layer depth, recurrence visits and answer length are different quantities.

The primary verifier returns strict task success from public graph headers and the generated answer. It has no reference answer access. Strong shortest-path/BFS solutions, target-conditioned gradients, hidden labels and held-out answer strings cannot be used as unreported proposal or selection aids. If a programmatic solver is included, label it as a separate reference algorithm.

Match total elapsed inference budget per problem, including prefill, candidate generation, verification, backward passes, optimizer state, tree operations, gate evaluations and cache reconstruction. Also report candidate-count curves; they answer a different question. Tune ordinary token retries on development data with the same tuning allowance. Early-stop all methods on verified success and report censored/unfinished candidates consistently.

The proposed reference time is the median cost on a fixed development set of evaluating the zero answer plus 15 stochastic token retries **with early stopping disabled for this calibration only**. This avoids a stronger model receiving a smaller budget simply because it succeeds earlier. Lock the numeric time before testing. Within an anchor compare all methods at that time; for an across-architecture inference comparison use one common time derived from the selected ordinary-Transformer baseline, not a different favorable deadline for each architecture. Also show absolute seconds on the curves.

Report 1×/2×/4× time curves and a common cap of 64 candidates, refinement depth 16, and eight duplicate-resampling attempts per expansion. Batch throughput and per-request latency are separate endpoints: if timing a fixed batch, each item shares the synchronized batch clock; never divide batch duration by batch size and call that its deadline. Interleave method order and account for cold compilation/load separately from warm inference. A candidate earns success credit only when generation **and verification** finish before the deadline. Count timed-out gradient/replay work even if it produces no candidate. Stop future attempts after a timely verified success.

### 9.2 GradCuit-style repair

On a failed generated answer y, hold its tokens fixed for replay and optimize only active Z controls:

```text
L_repair(Z) = mean_decision_positions log p_Z(y_t | prompt, y_<t)
           + κ mean_positions KL(p_zero(.|same prefix) || p_Z(.|same prefix))
Z ← project_RMS≤2(masked_Adam_step(Z, grad L_repair))
```

Descending the first term reduces likelihood of the failed trace. For the symbolic primary method, decision positions are the color and destination slots plus an emitted stopping decision; fixed `<to>`/`<sep>` tokens are excluded from this first term. Grammar slots are determined from the attempted response layout, including malformed attempts, without asking which action would be correct. KL covers all scored response positions to discourage collateral damage. This is an intentional V8 change from suppressing every response token and must be compared on development data with the full-response variant. Text repair needs a separately defined verifiable task and mask; ordinary text NLL evaluation is not a text-search reward.

With binary-only feedback the method does not know which decision caused failure. Even a legal move may be suppressed; a local gradient can worsen the answer. A verifier-provided first illegal action may define a targeted suffix only in a separately labelled richer-feedback condition shared by all relevant methods. Never substitute the correct action from a hidden reference.

Use projected Adam with betas (0.9,0.99), epsilon 1e−8, no weight decay and active-gradient norm clipping at 1. Maintain per-candidate optimizer state and per-coordinate step counters; inactive coordinates and their moments do not advance. A new one-step tree edge starts fresh optimizer state from its parent's Z; sequential repair retains its own state across steps, and that distinction is logged. Test η∈{0.01,0.03,0.1}, κ∈{0,0.05} on development data with a fixed tuning budget. Regenerate after each control proposal; a better replay objective is not a verified success. Stop refinement at declared step/depth/time limits and keep a zero/reference candidate available. Compare with random proposals matched to the realized update norm and active coordinates, not just the nominal learning rate.

This is a GradCuit-inspired latent repair method. It is not an exact paper reproduction, unbiased policy gradient, or guaranteed improvement. Compare to equal-norm random directions, prior latent resampling and token retries, charging their actual costs.

### 9.3 AB-MCTS-A over complete answers

Start with complete-answer refinement, as in the earlier V6.1 design. The root generates fresh prior controls. An answer node can generate a child by locally perturbing its controls, resampling a suffix, or applying the declared GradCuit kernel. Each child is fully regenerated and verified. The tree stores attempts; it is not a learned latent world simulator.

The zero-control greedy answer is the identical, charged first candidate for every method. For AB-Z, token decoding is greedy so different attempts expose the effect of controls. Non-root local mutation chooses one actually used control window uniformly, adds `0.25 × Normal(0,I)` and projects to the RMS ball. Suffix mutation chooses a used window uniformly and draws fresh prior controls from there onward. The non-gradient bank mixes these two operators 50/50; the hybrid bank uses local/suffix/one-step gradient proposals with probabilities 1/3 each. All nonadaptive allocation controls use the same bank and settings. These are V8 proposal choices, not operators supplied by the paper. Development may select mutation scale from {0.1,0.25,0.5} and control hold length from {1,2,4} once, with cost and selection recorded.

Use the paper's AB-MCTS-A GEN/CONT allocation with Beta(0.5,0.5) initial statistics and Thompson samples. Binary success is the primary reward. A direct child reward updates the **producing GEN**; rewards generated deeper update ancestor **CONT** and relevant answer-subtree statistics, not every ancestor GEN. An unexpanded continuation arm with no eligible answer child cannot be selected. Depth limits and duplicate suppression must propagate eligibility.

At a node, sample the enabled GEN and CONT statistics and choose the larger draw. If CONT wins, Thompson-sample each eligible answer child's subtree statistics, descend to the winner and repeat until GEN wins. For each statistics object use `Beta(0.5+sum(reward), 0.5+sum(1−reward))`. A newly generated child's own reward enters its answer statistics once; its own GEN/CONT remain at the prior. Do not update the producing node's CONT for its direct child. Use explicit seeded tie-breaking; no UCT/PUCT term is added to this selector.

Exact deterministic duplicate configurations do not earn another reward observation. Equal output text under different controls does not make the search states identical. Derive RNG streams from problem, method, proposal/node identity and operation type so batch compaction cannot change another problem's trajectory. The first zero candidate is a declared departure from a purely prior-sampled root; matched allocation methods share it.

Use the numerical backup fixture already documented in `draft_v6_1.md`, section 18: root direct scores [.8,0,.2] yield GEN=(1.5,2.5); deeper scores [.8,1,.3] yield CONT=(2.6,1.4). Those fractional scores are an algorithm fixture; the main binary-reward experiment still uses 0/1 outcomes.

With binary rewards and stopping at the first success, **every reward available to selection before termination is zero**. Therefore this primary binary arm tests count-driven allocation and proposal structure; it cannot establish feedback-sensitive quality refinement. Tree growth or a win over retries does not repair that identifiability limit.

An additional graded-public-feedback arm is required for a claim about adapting to candidate quality. Before running it, lock a deterministic score computed solely from the attempted response and its public graph:

```text
if strict_success: r = 1
else:
    a = number of attempted hop chunks, including a malformed final partial hop
    h = number of consecutive legal prefix hops before the first invalid hop
    l = repeated directed edges within that legal prefix
    f = 1 if the full response obeys the hop/terminator grammar, else 0
    g = 1 if a nonempty wholly legal walk ends at the goal, else 0
    r = clip(0.4*h/max(1,a) + 0.2*f + 0.2*g - 0.1*l/max(1,h), 0, 0.9)
```

Exclude a recognized terminator from a; check termination grammar independently of whether the goal is correct. A malformed/truncated attempted hop is counted in a and cannot inflate h. A legally reached goal without a valid terminator may get partial credit but remains a strict failure. The scorer does no shortest-path planning. This shaping is a heuristic: it can still prefer a short legal walk to an initially awkward successful route and does not measure distance to the solution.

Give the same score/features to same-kernel flat, sequential, random-tree and adaptive methods, and still use **strict success** as the outcome. Test whether score actually predicts successful child refinements on engineering data. If it does not, report the guidance hypothesis as unconfirmed. Beta updates with fractional scores are the declared bounded-score heuristic, not a calibrated Bernoulli posterior or a confidence interval. Keep binary and graded results separate, with their additional comparisons accounted for statistically.

### 9.4 Comparisons that identify the source of a gain

| Method | What it tests |
|---|---|
| Zero greedy and tuned token retries | Deterministic competence and ordinary inference scaling |
| Independent latent sampling, plus a tuned latent/token combination | Value of the trained prior without a tree |
| Sequential GradCuit and random-direction repair | Value of gradient-informed proposals |
| Flat/sequential allocation with the same non-gradient mutation kernels | Whether mutation alone explains AB-MCTS results |
| AB-MCTS with prior/local mutation | Adaptive breadth/depth allocation |
| Fixed/random allocation of the same prior/local/gradient mixture | Control for adding GradCuit to the generator bank |
| AB-MCTS with that same hybrid bank | Incremental value of adaptive allocation with gradient proposals |

A hybrid method beating token retries does not by itself establish an MCTS benefit. Require comparisons against the same kernels, feedback and total compute. Log proposal identities and costs so results can be reconstructed from events.

The matched hybrid allocation controls also reset optimizer state for each one-step gradient edge, exactly as AB-MCTS does. A standalone multi-step GradCuit chain that retains Adam moments is a different proposal process and cannot be the sole control for attributing a gain to tree allocation. Publish actual kernel counts, reached depth, duplicate rates and time spent on allocation versus neural work.

Primary search is AB-Z. A later **AB-PQ** arm may patch a bounded number of selected concept-code segments, with separately trained/tested compatibility and decoder authority. Discrete indices are finite actions over a soft/continuous model, not a proof that the network is discrete. A true hard-code model or prefix-expansion MCTS requires its own training and state contract and is deferred.

## 10. Local Support Learning: gated LoRA for continual phases

### 10.1 What transfers from the supplied paper

The paper's unit of locality is the **input to a weight matrix**, not an entire task label. It trains a new adapter with its own gate open, then fits a positive activation-density model. A shared negative/reference density supplies an input-dependent threshold. Previous adapters and gates remain active and frozen. The optional smoothing state is causal.

The method uses generic reference data to fit the negative density; it does not require the original pretraining corpus. This distinction matters: it is not literally data-free. The paper also acknowledges gate train/inference mismatch, imperfect retention, dependence on base representations, phase-linear cost, and a gap between single-matrix theory and the full multilayer model. Its consistency arguments need assumptions about density/support estimation; a misspecified fixed GMM need not recover the true support even with more samples.

Our reduced projection, mixture sizes, adapter rank, target modules, tiny model, and Z/search interactions are **new V8 choices**, not published LSL results. See PDF pp. 4–6 for the mechanism, pp. 8–10 for evidence/limits, pp. 16–17 for cost, pp. 19–21 for theory, and pp. 21–22 for fitting and the open-gate training rule.

### 10.2 Adapter placement and ownership

On the accepted dense E3/C4/D3 model, add rank-4 LoRA to Q and V projections in all ten blocks. Q and V receive the same normalized block input, so they share one gate per block and phase. Use these matrix-local gates rather than one sequence router. The primary variant leaves embedding, K/O projections, FFNs, norms, codebook, forecast head and continuous port frozen.

The first encoder block is a special case: its Q/V input is the normalized current-token embedding **before** attention, so its gate has no prefix context and cannot distinguish two tasks using the same token there. Later E gates and the C/D gates have contextual inputs. Report per-layer overlap/opening rates and include an engineering ablation disabling the first encoder block's continual bank. Do not claim every gate is semantic or contextual, or hide the different learned/gating counts of that ablation.

For a target matrix W with input a:

```text
W_phase(a) = W_base a + Σ_(p=1..P) g_(p,layer,t) · (α_L/r) B_p A_p a
r=4; α_L=2r
```

Initialize A_p randomly and B_p=0; only the new A_p/B_p are trained in phase p. At initialization the new LoRA update is zero; B gets its first gradient before A can acquire a nonzero gradient. This is distinct from the Z adapter's bilinear initialization rule. Gates for overlapping supports can open simultaneously and their updates add. There is no hidden task-ID oracle or winner-take-all routing in the primary method.

A concept-only LoRA variant tests whether the concept module alone can hold the new skill. It is an ablation, not a presumption that text adaptation can always avoid the encoder/decoder. On a recurrent replacement, attach banks to its actual unique matrices, track gate states per execution visit, and report the resulting different parameter and compute counts.

Frozen old LoRA matrices must still be differentiated with respect to their input when a new upstream bank is learning. Gates use hard decisions and are not trained by the task gradient. Fit densities from detached observations; do not detach the entire downstream model. An unchanged parameter hash cannot detect accidentally missing gradients into the new bank, so test those gradients separately.

### 10.3 Gate equations and tiny starting configuration

For each shared-input gate group l, draw and freeze a projection R_l from dimension d to e=32. Use scaled Gaussian entries, keep its seed in the manifest, and project the actual normalized input a:

```text
h_l = R_l a
Φ_neg,l(h) = Σ_(k=1..8) π_neg,k Normal_diag(h; μ_neg,k, σ²_neg,k)
Φ_pos,p,l(h) = Σ_(k=1..4) π_pos,k Normal_diag(h; μ_pos,k, σ²_pos,k)
b_(p,l,t) = 1[log Φ_pos,p,l(h_t) − log Φ_neg,l(h_t) > τ], initially τ=0
s_(p,l,t) = ρ b_(p,l,t) + (1−ρ)s_(p,l,t−1), s_initial=0
g_(p,l,t) = 1[s_(p,l,t) > 0.5]
```

Use log-sum-exp, normalized mixture weights, and positive diagonal variances. Apply a documented variance floor, initially `max(1e−6, 1e−4 × reference_feature_variance)` per dimension, and audit sensitivity. Do not silently renormalize each projected token to unit norm: radial density information may be useful. Projection does not prove preserved support geometry or density ratios.

The primary tiny setting uses **ρ=1** because short symbolic answers cannot tolerate long opening delays. With ρ=0.1, seven consecutive positive decisions are needed to cross 0.5 from zero; on concept positions that can correspond to roughly 28 tokens. Test ρ=0.3 as a secondary smoothing condition with explicit switch-delay metrics. E/D smoothing advances per token, C smoothing per completed concept; padding does not advance either clock. Reset per document/request, never across unrelated requests.

When teacher-forced training evaluates prior frozen gates, implement the smoothing recurrence exactly, by a causal scan if parallelizing. Do not use a bidirectional convolution or later tokens to decide an earlier gate. Beam/tree branches inherit their own prefix state and then evolve independently.

### 10.4 Fitting protocol

1. Establish the competent, frozen, accepted model version P0, including its accepted core and Z adapter weights. At Z=0 collect reference activations from a separate, fixed generic mixture resembling pretraining inputs; fit Φ_neg once per group. Reserve 1M reference input tokens and record actual valid activation counts. The same reference budget is available to controls.
2. For continual phase p, freeze P0, earlier LoRA banks and all existing gates. Insert only the new rank-4 bank. Force **its** gate open during training; previous gates operate normally. Train on the new phase's training partition at Z=0 using scored token CE. Do not retrain NCP/codebooks or silently replay old supervised tasks.
3. After training, freeze the new bank. With the new gate still forced open and older gates active, collect new-phase activations at each actual matrix input. Fit Φ_pos,p per group on these activations.
4. Enable the fitted gate and evaluate the complete resulting model. Quantify loss and task accuracy relative to the same adapter forced open. This measures the paper's train/inference mismatch at tiny scale.
5. Evaluate all previous tasks and new held-out tasks. Accept only if the new skill is useful and retention gates pass. Otherwise retain the previous accepted version, archive the candidate, and do not claim continual learning succeeded.

Use reservoir samples capped initially at 65,536 valid activations per gate group, bounded EM iterations (50, tolerance 1e−4), three initialization restarts selected by training likelihood, and module-by-module fitting. Counts, sampling weights, priors and seeds are part of each gate artifact. These are bounded approximations to the paper's fitting procedure; retaining all activations is a separate cost/accuracy alternative. Fit no density on final test data.

Reference data and positive-density data must exclude padding and respect document boundaries. Log activation counts separately for prompt/header tokens and answer tokens, since long graph headers can dominate a token-uniform reservoir. Primary fitting uses the deployment-relevant full visible sequence at Z=0, with its sampling measure fixed before testing; an answer-only or domain-reweighted density is a separate variant. Evaluate gates on both teacher-forced prefixes and free-running continuations, because correct gating on reference prefixes need not persist after a generated error.

The initial positive/negative component counts are 4/8, reduced from the paper's larger-model setting. Allow a preregistered 8/16 engineering alternative if gate calibration fails. If tiny representations cannot separate supports, changing mixture size may not help; always-open LoRA and task accuracy must distinguish fitting failure from lack of learned skill.

Later upstream LoRA updates alter downstream activation distributions. Consequently even frozen older gates and a fixed negative reference density can become less reliable. Recheck old gate behavior under the **actual full new model**, not stored P0 activations alone. A refreshed negative-density method is a separately versioned ablation with its additional reference access and refitting cost; it is not the fixed-reference LSL primary protocol.

In this hierarchy there is a second possible drift: encoder LoRA changes pooled C while the codebook and forecast head remain frozen. Measure quantization error, forecast error relative to a persistence baseline, code usage, and causal port influence before and after every phase. These diagnose whether the concept interface remains useful; behavioral retention still decides acceptance. Do not quietly unfreeze Q, add NCP gradients, or refit the core to rescue a continual phase. Each would change the primary frozen-backbone LSL experiment and require a separately named comparison.

### 10.5 Continual tasks, controls and acceptance

Use three declared phases drawn from distributions withheld from foundation training: a new text subdomain, a new composition/binding family, and a graph serialization/node-label distribution shift within the fixed vocabulary. Publish each generator and split before confirmation. Make task information explicit in the observable prompt where semantics differ; do not construct identical inputs with contradictory desired outputs and promise both can be retained.

A starting per-phase allocation is 2,000 updates, batch 64, with all rows from that phase's new training distribution. Pilot a bounded {2k,4k,6k} learning-curve extension and LoRA LR {1e−4,3e−4} on engineering data; fix exposure and LR before confirmation, matched to always-open LoRA. Test at least the declared phase order and its reverse. Do not begin a phase whose always-open adapter fails to learn and then blame its support gate.

| Control | Required interpretation |
|---|---|
| Frozen P0, no updates | New-task difficulty and retained original performance |
| Same newly trained bank with its new gate forced open, previous gates unchanged | Isolate the new gate's installation cost from its adapter's learning capacity |
| Separate always-open sequential-LoRA trajectory, same rank/data/updates | Ordinary adapter learning and forgetting across phases |
| LSL-gated phase LoRA | Incremental retention/learning tradeoff from local support |
| Ordinary Transformer with the same Q/V rank-4 banks and gate budget | Whether any continual-learning advantage requires the concept architecture |
| Concept-only gated LoRA | Whether the concept module is sufficient for adaptation |
| Oracle phase-ID gate, diagnostic only | Privileged routing comparator; not a guaranteed performance upper bound |
| Explicit small replay-budget baseline | Optional practical alternative; disclose old-data access and equalize total training cost |

Keep R_(p,j), the task performance on task j after phase p, and text NLL by domain. Report new-task gain, backward change from the previous accepted version, forgetting from each task's best earlier checkpoint, gate false-open/false-close proxies on known held-out phase labels, simultaneous openings, switch latency, predictive KL and actual serving cost. Phase labels diagnose routing; they are not a substitute for actual behavior and do not define the true activation support.

Require the absolute foundation competence gate after every phase, paired retention margins as in refit, and a demonstrable new-task gain over the immediately preceding accepted version with no new update. Also report the comparison to P0. For text, use a positive held-out NLL reduction with its interval; for symbolic tasks, require at least a five-percentage-point gain with a positive lower confidence bound. The gated adapter should lose no more than two points of new symbolic accuracy, or 0.02 nats of new text NLL, relative to its always-open counterpart, with appropriate uncertainty. If evidence is insufficient, label acceptance unresolved. Adjusting margins after reading confirmation results invalidates the confirmatory claim.

Here the new-gate tolerance compares the **same new bank's weights**, with only its gate toggled and previous banks/gates unchanged. The separately trained always-open multi-phase baseline has a different learned trajectory after phase one; it answers the broader retention question and is not that isolated gate ablation. An oracle phase-ID route is not mathematically an upper bound: useful adapters can transfer across phases and task supports can overlap.

Check pre-phase headroom on engineering data. If a frozen version already solves a proposed new symbolic task above 95%, a fixed five-point learning gain is impossible or nearly so; choose a genuinely new task before locking the study, or label that phase retention-only. Do not choose a new task after observing confirmation failures. Include mixed-domain documents and order switches without providing an external phase ID; otherwise obvious dataset markers alone may explain apparent routing success.

Apply retention margins both against the preceding version and cumulatively against P0 on original tasks, or against the checkpoint where each later task was first accepted. Otherwise several permitted two-point losses could accumulate into substantial forgetting. Predeclare the interval/multiplicity method across retained tasks. Apply the same learning/retention evaluation to the flat model with matched LoRA/GMM capacity; an LSL benefit alone is not a hierarchy benefit.

Promotion and measurement are different decisions. Retention failure prevents promoting a candidate or launching search on it. Preserve and evaluate its failed results rather than deleting that arm. A preregistered bounded three-phase **research-only** control trajectory may continue after retention failure to measure forgetting, with no deployment/promotion claim; this exception does not bypass the initial foundation gate and must be budgeted before the run. If an arm stops, show its missing later phases and do not compare only surviving trajectories as though they were an unconditional continual-learning result.

Retention evaluation may read held-out old tasks; primary LoRA optimization cannot train on them. Generic reference fitting is disclosed data access, not old-task gradient replay. Search-generated test answers must never become continual training examples. Any later distillation of search traces requires a fresh training-only generation pipeline, independent verification, and separately counted compute.

The reference manifest must say whether its graph/text inputs overlap prior task distributions, whether it contains old response tokens, and whether it is sampled from prior training inputs or independently collected generic inputs. Either way, this primary protocol has **reference-data access**; it cannot be advertised as learning without any old-distribution information. Negative-density fitting, positive-density fitting, engineering selection and LoRA training must all exclude sealed test documents/prompts. Old evaluation data may decide operational promotion but may not fit density parameters or select new hyperparameters after the confirmation lock.

### 10.6 Parameter and storage budget

For Q and V in each of ten blocks, a rank-4 bank adds `10 × 4dr = 20,480` learned parameters per phase. These figures apply to the primary dense model, not an uncounted routing/recurrent variant.

| State | Learned parameters |
|---|---:|
| Dense foundation | 2,448,640 |
| Foundation + two Z sites | 2,457,088 |
| Above + three continual banks | **2,518,528** |
| Above + eight continual banks | **2,620,928** |

At e=32, negative K=8 and positive K=4, the projections and GMM means/variances/weights store `10 × [128×32 + 8×65 + P×4×65]` scalars: 53,960 at P=3, or 66,960 at P=8. Thus the P=3 deployment has about **2.572M learned-plus-gating scalars**, before small counters/state buffers, and the P=8 version about 2.688M. GMM statistics are fitted quantities but not neural gradient-trained parameters; report both rather than hiding them. Optimizer, reference reservoir, teacher copies and caches require additional memory.

Use a hard bank admission cap of eight phases for this tiny configuration, with three in the primary study. If admission would exceed the cap, stop accepting new phases or launch an explicitly authorized consolidation experiment. Dynamic gated updates cannot be exactly merged into one static base weight matrix. Distilling into a new base can forget skills and therefore requires complete retention tests, a new version and newly fitted gates.

## 11. Interactions between continual learning, Z, gradients and tree state

This integration is a new hypothesis beyond the LSL paper. The safest experiment order is:

```text
competent foundation → accepted replacement or foundation fallback → trained Z
→ search baseline on P0 → continual phase 1 → revalidation → phase 2 → phase 3
                       ↘ search on each accepted phase version when coverage passes
```

Do not run the full search matrix after every failed continual candidate. A phase must pass deterministic retention first.

The zero-control contract is version-specific: `F_p(x,Z=0)` equals the deterministic accepted **phase-p** model. It generally does not equal F_0 because continual learning intentionally changes the function. Disabling every continual bank recovers P0; this diagnostic does not prove automatic gates preserve P0 on old tasks. Likewise, closing a gate at the current token does not erase earlier adapter effects stored in the prefix/KV state.

Z changes concept activations; later concept and decoder LSL gates can therefore open or close differently. Primary search must evaluate the **actual gates under each candidate**. During gradient repair, hard comparisons are treated as discrete decisions: differentiate the selected smooth branch, with no gradient through the boolean threshold. At a boundary the function is discontinuous; the gradient is not a guarantee of a useful crossing. Record crossing frequency, check finite differences away from boundaries, and verify each proposed candidate by regeneration. The KL reference is the **current accepted phase version at Z=0 on the same fixed replay prefix**, not P0 and not cached activations from the failed nonzero-Z candidate.

Do not reuse a root candidate's gate decisions for a changed Z schedule, hold gates secretly open for favorable search results, or substitute a straight-through estimator without a separate declared arm. Recompute density decisions and the causal EMA from the earliest affected prefix, or replay from a verified snapshot before it. Search must include this cost. An all-Z robust gate fitted on augmented activations is a possible later experiment, but may widen support and worsen retention; it is not silently added to the primary LSL method.

Every cache key includes checkpoint hash, continual-bank/gate version, document/prompt tokens, source positions and control-prefix identity. Copy both KV and smoothing state when branching. Editing an earlier control invalidates all downstream concept/decoder state, including later gates. Changing the accepted model version invalidates old model caches; encoder LoRA means P0 encoder states cannot generally be reused under P1.

After a continual phase, rerun prior coverage and repair-direction checks. A Z adapter trained on P0 is not guaranteed useful on P1 even if Z=0 stays exact. If compatibility fails, mark search unavailable for that version. Retraining a phase-specific Z adapter around the new frozen anchor is a separate, costed study; retain old versions for reproducibility and do not count adapter replacements as free continual learning.

## 12. Scalability in the Transformer sense

The dense backbone has independent axes for width d, encoder/concept/decoder depth, vocabulary and training tokens. Keep head width and PQ segment width controlled when scaling, and increase distinct layers rather than treating more recurrent visits or a larger search tree as parameter scaling. A first research ladder is d∈{128,256,512}, with paired flat baselines and data/compute learning curves at every size. The larger points are future studies, not authorized runs or proven scaling.

Ignoring vocabulary and FF constants, full-token attention costs scale as `(L_E+L_D) T² d`; concept attention as `L_C (T/k)² d`. Dense projections/FFNs contribute approximately `(L_E+L_D) T d² + L_C (T/k) d²`. The token stacks and output head remain significant. Chunking does not make the whole model linear-time or establish long-context ability.

Use parallel teacher forcing and dense kernels for foundation/refit; batch independent candidates for exploration/search within memory caps. Causal per-group LSL density calculations are vectorizable. Smoothing can use an associative affine scan; ρ=1 requires no temporal aggregation. Dynamic adapter sparsity only improves throughput if the implementation actually avoids work; report measured execution, not the fraction of gates closed.

Store immutable token/control histories and statistics for tree nodes, but keep at most four live full KV trajectories per problem in the initial implementation, independent of the 64-candidate history cap. Use a fixed eviction policy with replay from a verified ancestor/prompt; charge recomputation. Shared immutable prefix storage is allowed only with verified copy-on-write semantics for both KV and gate state. A tree node does not automatically own a full mutable model cache. Larger serving configurations must specify live-trajectory and byte budgets before scaling request concurrency.

At one gate group, sharing h across phases gives density cost roughly `O(d e + (K_neg + P K_pos)e)` per evaluated activation, plus active LoRA work `O(P r(d_in+d_out))` in the worst case. Storage and inference still grow with P. Cap P, r, e, mixture size, live candidates and tree memory separately. A tiny bank under three million scalars is not evidence of constant-cost lifelong learning.

Larger deployments can investigate exact efficient attention, trained grouped-query attention, sharding and prefix sharing, but must retain reference-equivalence tests. These are implementation/scaling studies, not automatic properties established here. Recurrence caches are indexed by visit; search prefix sharing is keyed by actual model/control/gate state.

A scalability claim requires measured zero-search learning curves against ordinary Transformers at matched total training compute, followed by separately measured search and continual-learning tradeoffs. Do not transfer published LLM scaling exponents, the LSL paper's GPU timings, or its retention percentages to this tiny architecture.

## 13. Verification and statistical plan

### 13.1 Functional contracts before performance claims

1. **Causality/document isolation:** alter future tokens and other documents; earlier logits must remain unchanged. Check document-local chunk boundaries, aligned packed/unpacked equivalence and explicit repack/rejection of unsupported unaligned inputs. Check the first/last position of every chunk and prompt/answer boundary.
2. **Trained decoding parity:** for hierarchy, recurrent and flat models, compare full eager decoding, incremental decoding and each production fast cache on actual standard/hard graph prompts and text. Cover all prompt offsets, long continuations, terminators, and active Z/LSL states. Compare logits and decision margins, not just final strings. Establish numerical tolerances from FP32 reference behavior before inspecting test success.
3. **Reference/verifier independence:** validate held-out references, accept alternate legal paths, reject malformed paths and early false terminators. Verify every reference fits the declared length cap. Compare an independent simple checker on generated fixtures.
4. **Gradient contracts:** compare compiled/eager, microbatch-accumulated/full-batch and candidate-batched/sequential gradients; audit target stop-gradients, finite updates and all frozen tensor hashes. Verify nonzero expected gradients through frozen decoder/layer operations into the optimized core, Z or new LoRA bank. Check Z derivatives away from LSL thresholds and report threshold crossings separately.
5. **Refit/transplant:** exact copy initialization, complete λ endpoints, preserved frozen ports/Q, six distinct recurrent-visit caches, and competence plus uncertainty at acceptance.
6. **Zero/control contracts:** Z=0 matches the accepted version after adapter training; every used injection site has causal authority; inactive Z coordinates and their optimizer moments stay unchanged.
7. **Continual contracts:** only the new LoRA bank changes; forced-open training differs explicitly from installed-gate inference; all-banks-disabled matches P0; EMA resets and gate decisions match prefill/incremental/branch replay.
8. **Search contracts:** correct AB-MCTS backup fixture, eligible arms, deterministic reproduction, bounded depth/memory/time, same-kernel/optimizer-state controls, and reward/selection information audits. An inert prior that repeats zero-greedy outputs must fail the rescue gate; the graded-score fixtures must cover empty answers, repeated edges, truncated hops, reached-goal/missing-terminator and late verification.
9. **Data contracts:** nonempty sampled-identity checks against exclusions, no near-duplicate leakage hidden by relabelling, no scored prompt/padding targets, and correct document-aware NCP masks.
10. **Recovery/accounting:** atomic checkpoint and summary writes, optimizer/RNG/data-cursor resume, actual completed work counts, stage labels updated during inline evaluations, and unambiguous completed/partial/rejected/failed/stopped states.

Passing these contracts establishes implementation consistency, not a good model. Smoke tests and trained audits are reported separately.

### 13.2 Outcomes and intervals

Use at least 512 fresh items per main symbolic quality split. For narrow retention claims, choose a larger **fixed** sample in advance using engineering discordance/variance and the number of planned comparisons. An initial planning value is 2,048 paired items per retained symbolic task, subject to an engineering power calculation and available budget before the lock. With paired discordance q and true mean difference near zero, the approximate standard error is `sqrt(q/n)`; at q=0.1 this is about 1.4 percentage points for n=512 and 0.70 points for n=2,048. Thus a two-point retention margin may be unresolved at 512 even when point estimates match. This approximation is for planning, not the final interval method.

No ordinary-confidence-interval procedure may add samples or stop upon significance after inspecting confirmation outcomes. If sequential sampling is wanted, specify a valid sequential interval/alpha-spending method and a maximum budget before the first observation. Otherwise finish the fixed allocation and label an inconclusive result unresolved. Operational development-gate decisions are distinguished from final test inference.

Bootstrap text results by source document, clustering all crops/rows from that document, not by token or arbitrary packed row. For success rates report Wilson/binomial intervals. For paired differences preserve the full win/loss concordance table and use a paired interval procedure locked before evaluation. For sparse/zero-success cases a conservative default is the difference of simultaneous marginal exact binomial intervals obtained by Bonferroni allocation; it loses pairing efficiency but avoids a degenerate bootstrap being mistaken for certainty. Use one-sided intervals for declared noninferiority decisions and two-sided intervals for effect estimates, with their confidence levels stated.

Report each independent trained seed, then a hierarchical/paired summary with problems nested appropriately. Search RNG repeats are conditional repeats within a trained model; they are not new foundations. For search, plot verified success against both charged wall time and attempts, with paired problem identities. Report failures, incomplete candidates, abstentions and censored time explicitly.

Declare one primary contrast per hypothesis: delayed NCP versus alpha zero; hierarchy versus flat; replacement versus extra continuation; prior training versus direct controls/retry; GradCuit versus tuned retry and random repair; AB-MCTS versus the same kernels under nonadaptive allocation; LSL versus always-open LoRA on the new-skill/retention tradeoff. Label exploratory ablations and handle multiple comparisons rather than selecting the best of many results as a single preplanned success.

Organize confirmatory families before testing: backbone quality, replacement/exploration, search allocation, and continual learning/retention. Use a locked familywise procedure such as Holm across the named contrasts within each family, and simultaneous bounds for all old-task retention requirements of a proposed phase. Report unadjusted effects/intervals alongside adjusted decision criteria. Three training seeds give limited evidence about between-seed variance; do not let thousands of repeated problem evaluations create a falsely precise model-population claim.

Text evaluation includes held-out NLL, domains, exposure/generalization curves, and generation diagnostics. Graph success does not establish language reasoning; tiny text NLL does not establish LLM quality. Concept ablation must regenerate the affected answer, not score a fixed answer and infer causal use from that alone.

## 14. Queue policy and resource allocation

The implementation must make stage eligibility executable, not just prose:

```text
run correctness + learning ladder
if bridge or decoding fails: STOP dependent training; report actual blocker

run one engineering foundation at full proposed exposure
if no competent foundation: STOP refit, Z, search and continual learning

lock recipe, data, gates and comparisons; run independent foundation confirmation
for each competent anchor:
    retain A192; attempt the locked harder-context continuation when budgeted
    accept A384 for the declared hard-task lineage only after its retention/gate checks
    test replacements and continuation controls
    choose accepted core, or retain foundation with rejection recorded
    train Z; audit zero behavior and free-generation coverage
    if coverage passes: run locked development GradCuit/AB-MCTS comparisons
    else: bounded diagnostic only; mark search unconfirmed
    fit reference gates; attempt declared continual phases sequentially
    for each phase:
        train new LoRA → fit positive gates → evaluate learning and retention
        if acceptance fails: retain prior version; stop its promotion/search chain
                             preserve any predeclared research-only control trajectory
        if acceptance passes: version checkpoint; recheck Z/search compatibility
run untouched final tests on all completed locked arms/versions, including rejected ones
do not retrain, retune or change earlier decisions based on these final outcomes
write report with accepted/rejected/partial/skipped lineage
```

No primary promotion or search stage may bypass a failed absolute competence gate merely because weights are finite. The narrowly bounded research-only continual controls in section 10.5 are measurements of failure, not exceptions that authorize promotion or search. The continual branch may still test retention on a competent deterministic model if Z/search fails, but it cannot claim a successfully integrated search model. A failed refit may fall back to the foundation; that fallback must be named in every dependent artifact.

Treat the main 24k foundation, the harder-context continuation, and each continual version as distinct endpoints. Freeze the standard-competent A192 checkpoint before attempting A384 continuation. Compare all architectures at each same-data endpoint. For a downstream hard-search study choose A384 only if it passes standard retention and the intended hard-split gate; otherwise retain A192 and label hard search unavailable when its coverage fails. The queue must explicitly place this continuation **before** the dependent refit/Z lineage, rather than leaving its position implicit.

Prioritize the program in three independently reportable allocations: (1) bridge, learning checks and competent matched foundations; (2) accepted refit/Z and search comparisons; (3) bounded continual-learning sequences and search revalidation. Engineering-only routing, target-policy and gate-density alternatives are screened before each respective confirmation lock. Every idea remains testable, but a later allocation begins only when its prerequisites and budget are satisfied. This prevents a large secondary matrix from consuming the foundation's learning budget.

Reserve enough budget for full foundation exposure and final evaluation before allocating secondary arms. Measure stage throughput and estimate costs as `updates × measured seconds/update + evaluation + compilation + fitting`, with uncertainty and a final-testing reserve. Do not promise a 12/24-hour completion time from earlier slots/s numbers. A full three-seed, three-foundation, multi-phase study may exceed a day on this machine.

Reduce the breadth of an engineering screen or explicitly label partial work before reducing the required learning exposure. Do not launch all seeds of an already failed recipe. While below the competence/coverage gate, expensive search is capped at the bounded diagnostic, not an hours-long matrix of zeros. Stop or extend work only under the future run's explicit authorization and budget policy; this document itself starts nothing.

Every result should name foundation seed, accepted core hash, Z version, continual version, gate/reference manifest, optimizer/data exposure, and inference budget. Reports must distinguish architecture parameters from resident teachers, shared-base adapter storage from multiplied full checkpoints, successful completion from budget expiry, and numerical health from scientific success.

## 15. Source map and evidence boundaries

| Source | Use in this design |
|---|---|
| [V6 draft](v6_draft.md) | Causal hierarchy, separate concept ports, replacement/transplant, prior-trained Z and acceptance principles |
| [V6.1 draft](draft_v6_1.md) and [AB-MCTS paper](AB-MCTS.pdf) | Answer-refinement tree contract, GEN/CONT, backup fixture, compatible PQ intervention and cost controls |
| [V7 draft](draft_v7.md) | Dense concept scaling, explicit routing, NCP comparisons, frozen interfaces and experiment gates |
| [Local Support Learning](LOCAL%20SUPPORT%20LEARNING.pdf) | Per-matrix local gates, positive/reference GMMs, phase-specific adapters, smoothing, fitting and limitations |
| [V6 report](v6_tiny/REPORT.md) and [primary result](v6_tiny/runs/PRIMARY_RESULT.json) | Actual completed tiny outcomes and the limits of the GradCuit claim |
| [V6 model](v6_tiny/code/model.py), [training](v6_tiny/code/train.py), [runtime](v6_tiny/code/runtime.py) | Parameter accounting, soft PQ, shift convention, VQ units, candidate selection, optimizer and zero-control mechanisms |
| [V6 graph generator](v6_tiny/code/graphnav.py), [V6 data](v6_tiny/code/data.py), [V7 data](v7_tiny/code/data.py) | Hidden demonstration choices, deterministic edge lookup, graph-size/difficulty distinctions, masks and aligned document packing |
| [V7 driver](v7_tiny/code/drive.py), [audit](v7_tiny/code/audit.py), [checks](v7_tiny/code/checks.py), [status](v7_tiny/runs/status.json) | Actual gate ordering, trained-cache gaps, data-test weakness and stopped-run status |
| V6/V7 saved training, development, refit, exploration and search summaries | Exposure differences, first-hop failures, zero-success limitations and prior outcomes |
| [NCP-ArchPreview report](NCP-ArchPreview%20Technical%20Report.pdf) | Motivation for predictive concepts; large-scale evidence described in the preceding drafts is not V8 validation |
| [Explorative Modeling](Loop%20Papers/Explorative%20Modeling.pdf) and [GradCuit](Loop%20Papers/GradCuit.pdf) | Original motivation for candidate-selected controls and latent refinement; the V8 mechanisms are adaptations |

The new paper's substantive main text and appendices were read, with its gating and fitting pages visually checked. The design reuses the earlier drafts' documented paper analyses where noted; it does not claim a new page-by-page reread of every paper in the repository. Saved model binaries were not executed to create new evidence for this document.

## 16. First-principles self-review of this design

This review checks the proposed information flow, gradients, evidence, costs and failure handling. It is a design review and arithmetic check, not an implementation test.

The initial pass established bounds on routing writes, ordinary-Transformer LoRA controls, cumulative retention and candidate-specific gate/cache replay. The requested deeper review then found and corrected the following substantive issues. These are corrections to the **design**, not claims about newly measured V8 behavior.

| Issue found | First-principles reason | Revision |
|---|---|---|
| Pooling origin was ambiguous for unaligned packed documents | The same document can change function if row offsets change its chunk boundaries | Define document-local chunks; align the fast packer and require repack/rejection otherwise |
| Demonstration likelihood was too easily equated with planning correctness | Hidden random walk choices are not uniquely inferable from graph/start/goal | Separate conditional edge retrieval, legal generation and exact labels; isolate a canonical-label engineering comparison |
| A competent token path could hide an unused concept path | Token quality does not identify which port carries information | Add forecast/persistence, rank, port and retrained-ablation diagnostics |
| Absolute prior coverage could pass with identical repeated outputs | Repeating already-solved problems is not exploration | Require distinct rescues of zero-baseline failures and report conditional/full-population rates |
| Binary first-hit MCTS could be called quality-adaptive | All available rewards before stopping are zero | Limit the binary claim; define a matched graded-feedback arm for the quality-allocation hypothesis |
| Frozen weights could be implemented with detached downstream graphs | A trainable core/adapter still needs derivatives through fixed computations | Add stage-specific gradient ownership and accumulation contracts |
| GradCuit's mask, optimizer and state controls were underspecified | Different suppression targets or retained moments change the proposal algorithm | Define decision masks, projected masked Adam, and identical-state controls for allocation comparisons |
| Retention sample sizes could grow after seeing an inconclusive result | Repeatedly peeking at ordinary intervals invalidates their advertised coverage | Fix sample sizes using engineering estimates, or preregister a valid sequential procedure |
| Every LSL gate was called contextual | The first encoder Q/V input precedes attention | State its token-only limitation and test disabling that bank |
| A gate toggle was conflated with a separately trained always-open trajectory | Later banks are trained under different previous routing functions | Separate same-weights gate ablations from full sequential baselines |
| Oracle routing was described as an upper bound | Cross-phase transfer can make task-ID isolation worse | Call it a privileged comparator, not a mathematical bound |
| Hard continuation and failed-seed reporting were implicit | Endpoint selection and survivor filtering can change the experiment being reported | Define A192/A384 lineage, report every seed, and separate promotion from research-only control measurements |

Parameter/exposure arithmetic, planning standard errors and local source links were checked. Source code and saved JSON were read; no model inference, training, new experiment or repository test suite was launched for this review.

| Review question | Resolution | Remaining empirical question |
|---|---|---|
| Does smaller mean automatically better? | No. Restore useful exposure at 2.45M and include the V6 bridge and same-size flat model. | Whether the revised hierarchy uses its capacity better. |
| Can formatting loss conceal failed reasoning? | Evaluate node decisions, one-hop retrieval and free generation before navigation/search. | Which learning bottleneck remains after the bridge. |
| Must a valid answer reproduce a hidden random demonstration? | No. Separate deterministic edge correctness from multimodal path supervision. | Whether canonical labels improve learning at matched exposure/compute. |
| Is NCP being assumed beneficial? | Keep delayed-NCP and alpha-zero matched arms; explicit VQ units and gradient diagnostics. | Whether any selected NCP schedule improves task/text tradeoffs. |
| Can concepts leak future tokens? | Expose a chunk only at its last input position for next-token logits; reset document state. | Implementation must pass trained full/fast parity. |
| Is the whole network discrete? | No. Soft PQ, A and Z are continuous; sampled search actions are finite. | Whether optional PQ patches have useful authority. |
| Is replacement preserved only nominally? | Require absolute competence plus powered retention intervals and complete U/A endpoints. | Whether refitting improves quality or cost over continuation. |
| Can exploration destroy the deterministic anchor? | Bias-free zero-exact Z injection and frozen tensors; acceptance is version-specific. | Whether nonzero controls offer useful prior coverage. |
| Can search hide a poor proposal model? | Coverage/competence gates precede large matrices; bounded failed diagnostics. | Whether gradient/tree allocation beats strong retries. |
| Does prior coverage prove useful exploration? | No. Require additional strict successes on fixed baseline-failure problems. | Whether the rescue gain survives equal-cost comparison with retries. |
| Can binary first-hit rewards rank failure quality? | No. Test that hypothesis using a separately controlled graded-feedback arm. | Whether its heuristic score predicts useful refinements. |
| Are GradCuit and MCTS gains confounded? | Include equal-cost random directions, identical-kernel nonadaptive allocation and tuned retry. | Which component, if any, provides an advantage. |
| Does a local gate guarantee no forgetting? | No; overlap, density mismatch and upstream representation changes are explicit. | Tiny-scale retention and new-task learning. |
| Can small accepted losses accumulate across phases? | Require both adjacent-version and cumulative retention bounds on each task. | Sample size and bank capacity needed to retain all accepted tasks. |
| Could LSL explain a gain independently of the hierarchy? | Include the ordinary Transformer with the same adapter/gate budget. | Whether the concept structure improves the tradeoff. |
| Can a weak tiny model inherit LSL's published results? | No; require a competent base and always-open learning control. | Whether tiny activations separate phase supports. |
| Does every gate know the prefix/task? | No. The first encoder gate sees the current token embedding; deeper gates have context. | Whether its bank helps or introduces avoidable interference. |
| Can continual encoder updates invalidate concepts? | Keep Q/head fixed, measure interface drift and reject unacceptable behavior; rescue refits are separate arms. | Whether the frozen concept vocabulary remains useful on new distributions. |
| Does LoRA break the zero-Z promise? | Zero matches each accepted phase's deterministic model, not the original model. | Search compatibility after each accepted update. |
| Can cached gates survive altered latent prefixes? | Recompute gates/EMA and invalidate downstream caches per candidate/version. | Runtime cost and numerical equivalence. |
| Can one differentiate through a hard support gate? | Only piecewise branch derivatives are used; crossings are verified by regeneration. | Whether useful repair survives the discontinuities. |
| Is continual learning indefinitely constant-size? | No. Count all banks/GMMs, cap eight phases, require tested consolidation for expansion. | Capacity/retention under longer phase sequences. |
| Are all phases still under 3M parameters? | Primary dense + Z + three LoRA phases is 2,518,528 learned parameters, plus gate statistics. | Routing/recurrent extensions require their own counts. |
| Are thresholds proven optimum or statistically sufficient? | No. They are proposed locked criteria; unresolved intervals do not pass. | Actual sample sizes needed for narrow margins. |
| Can a narrow margin be earned by repeatedly adding test samples? | No. Use a fixed powered sample or a preregistered valid sequential design. | Whether the available budget supports the intended claim. |
| Does a tiny positive result establish LLM scalability? | No. Separate zero-search capacity/data scaling, search scaling and continual-phase scaling. | Larger-width, depth, context and corpus behavior. |

**Acceptance principle:** first demonstrate a capable predictor; then preserve it under replacement, create useful alternatives, allocate search efficiently, and learn new distributions without unacceptable forgetting. Each arrow is an experiment with a fallback and a stopping rule. The design is successful only to the extent that those experiments pass.
