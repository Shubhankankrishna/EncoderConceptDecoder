# V6.1 draft: adaptive tree search through the concept interface

Date: 4 October 2026  
Status: architecture and experiment specification; no implementation or new experiment launched.  
Placement: a new **Phase F, after the complete V6 GradCuit and final-testing queue**.  
Primary question: does adaptive allocation between fresh latent proposals and refinement of previous proposals improve verified success at the same elapsed-time budget?
Scalability requirement: **the neural architecture must support Transformer-style growth in width, depth, data and training compute while preserving parallel training and useful learning.** Sections 4.1-4.7 specify this architectural requirement. The serving requirements in sections 14.6-14.12 are supporting engineering, not evidence of neural scaling. The tiny experiment establishes a mechanism only.

## 1. Decision

**MCTS can be integrated with V6. Discreteness is not a prerequisite, and the present V6 concept module is not fully discrete.**

Use the lightweight **AB-MCTS-A (Beta)** selection and backup mechanism from the supplied paper. Put a search controller around the frozen V6 generator. Each answer node stores a concept-control configuration, a complete generated answer, and verification feedback. The controller chooses between generating another alternative and refining an existing alternative. It never treats a predicted concept as an observed future state.

Implement the following sequence when this draft is handed off:

1. **AB-Z:** search through the already trained, bounded Z interface. This is the primary frozen-checkpoint experiment and needs no new shared model parameters.
2. **PQ compatibility screen:** measure whether bounded, discrete code-index interventions at the soft-PQ readout preserve enough competence to be worth searching.
3. **AB-PQ:** if that screen passes, search over those discrete concept actions with Z fixed to zero. This isolates whether the codebook itself supplies useful alternatives.
4. **AB-Z-GradCuit:** separately test a gradient-assisted refinement operator. Attribute any gain to the combination and charge its replay/backward computation.

These are staged experiments, not an instruction to combine every mechanism into the first model. AB-Z tests search allocation on the deployed training distribution's interface; AB-PQ tests a new inference interface. Their conclusions must remain separate.

The recommended architecture preserves the original first answer exactly. Search starts only after that answer fails the existing strict verifier. Every reported success must have been generated and verified within budget; reporting does not require an uncharged extra regeneration.

### Scope of this deliverable

This file adds the design of Phase F. It does not append a process to the active queue, modify training, replace the current report, or establish that MCTS works. The future implementation should live in a separate `v6_1_mcts/` directory. The original V6 run must finish and its results must be preserved before Phase F begins.

## 2. What the AB-MCTS paper actually establishes

Source: Inoue, Misaki, Imajuku, Kuroki, Nakamura and Akiba, *Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search*, supplied [AB-MCTS.pdf](AB-MCTS.pdf), arXiv:2503.04412v5, 7 November 2025, 30 pages. References below use PDF page numbers, which match the printed page numbers.

### 2.1 The object being searched

The paper's ordinary answer node contains a **complete answer**. A root expansion produces an independent answer; a non-root expansion produces a revised answer conditioned on a previous answer and external feedback. A node is scored directly, so the paper does not need an additional game-style rollout after answer generation. [Paper sections 3.1-3.2, pp. 3-4; appendix A.1, pp. 14-15.]

Thus three kinds of depth must not be confused:

| Quantity | Meaning here |
|---|---|
| AB-MCTS tree depth | Number of parent-to-child answer refinements |
| Response position | Which generated chunk/token is being predicted |
| V6 recursive depth | The six applications of the shared low/high core blocks |

The proposed main experiment keeps response context and core recursion unchanged. Extra tree depth buys more candidate evaluations, not extra untrained core iterations.

### 2.2 Why adaptive branching matters

Every answer node has a `GEN` action representing a fresh child proposal. An already expanded answer may generate more children. Branching is chosen using observed scores rather than fixed exclusively by a node's visit count. This can allocate work to broad sampling on one problem and repeated refinement on another. [Paper algorithm 1 and figure 2, p. 4; appendix A.2, p. 15.]

The adaptation is useful only if the proposal mechanism can reach distinct useful answers and feedback helps identify productive refinement regions. The search controller does not create missing model competence.

### 2.3 Which variant to borrow

| Variant | Paper mechanism | V6.1 decision |
|---|---|---|
| AB-MCTS-M | A mixed-effects model shares statistical information between sibling subtrees; MCMC is used during selection | Defer; measure a cheap selector first on this small local model |
| AB-MCTS-A Gaussian | Independent node statistics with conjugate updates | Defer; bounded verifier scores permit the simpler Beta variant |
| AB-MCTS-A Beta | `GEN` versus aggregated `CONT`, independent Beta score models, Thompson selection | Default; exact proposed backup rules are specified in section 9 |
| Progressive widening | A visit-count rule determines when to add children | Useful comparator if adaptive search shows a gain; not interchangeable with the paper's algorithm |

The Beta default in the paper is alpha=beta=0.5. For fractional scores, its update is a practical score-modeling convention, not an exact Bernoulli likelihood for independent binary outcomes. Adaptive, correlated descendants also prevent interpreting its internal uncertainty as a calibrated confidence interval on final task success. [Paper section 3.4, pp. 6-7; appendix A.4.2, p. 17; B.2, p. 20.]

### 2.4 Results and limits that matter for this transfer

- The main experiments use complete Python solutions, frontier LLMs, and external tests or validation scores. They generally allocate **128 generation calls**, not eleven V6 candidates. Extended ARC experiments reach 512 calls. These budgets do not imply a benefit at a tiny budget. [Paper pp. 7-10.]
- AB-MCTS is competitive across tasks, but it does not win every setting. In table 1, repeated sampling is strongest on ARC-AGI at the main budget, and standard MCTS leads DeepSeek-V3 LiveCodeBench. Progressive widening can be competitive with suitable settings. [Paper p. 7; appendix C.4, p. 23.]
- The paper's own limitations identify evaluator quality and real computation costs beyond API-call counts. V6.1 must measure wall time, generation, verification, tree management, cache copies and gradients. [Paper p. 10.]
- Appendix D's multi-model extension is optional inspiration for multiple proposal operators. It is not evidence that mixing V6 mutation operators will work. Its reported over-30% coverage at 250 calls differs from the 19.2% selected-answer Pass@2 result. [Paper pp. 26-30.]
- AB-MCTS-A tends to grow wider trees than AB-MCTS-M. Deep refinement needs repeated `CONT` choices. Width/depth statistics therefore belong in the evaluation. [Paper appendix C.5 and C.7, pp. 24-25.]

**Transfer claim:** borrow adaptive allocation and score backup. Replacing language-based revision with concept-control mutation is our new hypothesis, not an exact reproduction of the paper's answer-refinement generator.

## 3. What the implemented V6 actually computes

The current implementation is authoritative for this extension. The root [v6_draft.md](v6_draft.md) predates the running implementation and still contains statements such as an uninstantiated parameter ledger. V6 tiny subsequently instantiated and tested its reference configuration.

### 3.1 Model and information paths

The tiny model has width 128, four attention heads, feed-forward width 384, vocabulary 2,048, context 192 and chunk size four. There are three encoder blocks, a replaced concept stack, and three decoder blocks. The replacement makes four low-block and two high-block applications with shared weights and separate states/caches.

For a single document, using zero-based positions:

~~~text
H_t = causal_encoder(x_0 ... x_t)
C_j = RMS_normalize(mean(H_[4j:4j+4]))
(u_j, A_j) = recursive_core(C_0 ... C_j, Z_0 ... Z_j)
p_[j,s,k] = softmax_k(heads(u_j))[s,k]
qbar_[j,s] = sum_k p_[j,s,k] Q_[s,k]
qbar_j = concatenate(qbar_[j,0], ..., qbar_[j,7])
decoder_input_t = H_t + qbar_j(t) + gamma * A_j(t)
j(t) = floor((t+1)/4) - 1
~~~

For negative `j(t)`, the plan is zero. The decoder at position t predicts token t+1. All routes respect document boundaries. The core depends on actual pooled features of observed/generated tokens; it is not an autonomous transition model over predicted code indices. [Implementation: `model.py`, `Foundation.encode/pool/pq/decode`, `Core.forward`, `System.plans`.]

### 3.2 Discrete vocabulary versus continuous execution

| Object | Actual type | Consequence for search |
|---|---|---|
| Q | Eight tables, each with 32 entries of dimension 16 | Finite code indices exist |
| Nearest-code index in VQ training | Discrete argmin for fitting Q to detached pooled concepts | Does not make the deployed plan an argmax code |
| Decoder-facing qbar | Probability-weighted codebook mixture | Continuous; hard code replacement is a distribution change |
| Core u and A | Continuous learned outputs | Code-index edits do not automatically update A coherently |
| Token features H | Continuous, full causal token route | The decoder can rely on information outside PQ |
| Z | Eight continuous coordinates per response window, bounded by RMS<=2 | Search can sample finite candidate controls without discretizing the model |

NCP trains a next-concept forecast loss against the next pooled concept; it does not supervise a sequence of discrete semantic action labels. The current path uses `softmax` and a weighted sum, not sampled code IDs. There is no deployed categorical concept transition model or trained value head.

The eight tables would permit `32^8 = 1,099,511,627,776` joint hard tuples per chunk. Enumerating those tuples is impractical. A code coordinate is not proven to represent a color, destination, subgoal, or option duration. A high code probability is a proposal preference, not a probability of solving the task.

### 3.3 Existing control interface and anchors

V6's gated adapters insert bounded functions of Z into the initial high state and each low update. The raw write contains a factor of Z, so it vanishes at zero. The foundation and deterministic core are frozen in B1. Therefore, absent a new PQ intervention:

~~~text
V6(Z=0) = deterministic B0 replacement
~~~

This remains the initial answer for every new search policy. For AB-PQ, an empty patch map must additionally reproduce this same answer. For a hybrid using both mechanisms, the baseline is `(Z=0, patches=empty)`.

Shared trained parameter counts:

| Configuration | Active parameters |
|---|---:|
| NCP foundation | 2,022,144 |
| Deterministic replacement | 2,055,424 |
| Replacement plus gated adapters | 2,063,904 |
| Replacement plus direct adapters | 2,057,472 |

The search algorithm and fixed PQ patch operator add **zero shared trained parameters**. Tree records, RNG streams, controls and caches are inference state. A PQ-only run at Z=0 does not establish a benefit from B1 training; its effective function can use B0 directly. Loaded teacher modules and inactive adapters still count toward resident memory.

### 3.4 Current evidence is a starting point, not a search result

The saved [development competence screen](v6_tiny/runs/ANCHOR_GATE.json) reports:

| Measure | Foundation | B0 replacement |
|---|---:|---:|
| Text NLL | 1.815398 | 1.811224 |
| Symbolic NLL | 0.273567 | 0.256880 |
| Standard greedy success, 128 problems | 7/128 = 5.47% | 17/128 = 13.28% |

B0 passed the declared mean-retention screen. Absolute symbolic success is still low. This motivates checking whether failure comes from proposal coverage, graph transitions, format or stopping before attributing a gain to more reasoning. These are single-foundation-seed development results. The running GradCuit evaluations and final policy decision were not complete when this proposal was researched.

Historical V5 success percentages use different experiment/verifier/data conventions. They are not matched baselines for this new phase.

## 4. First-principles derivation

### Step 1: define success before defining search

For GraphNav, success is a legal generated walk ending at the public goal with the complete `[DONE, SEP, SEP, SEP]` chunk. A changed code, lower loss, higher tree value or larger latent norm is not a successful answer.

### Step 2: identify what can be changed safely

The checkpoint already exposes a trained bounded Z interface and an exact zero anchor. Start there. PQ code indices offer a second intervention family, but their discrete use was not trained. Test that family's compatibility before allocating a large search budget.

### Step 3: keep observed facts separate from proposed plans

The encoder supplies facts from the real token prefix. A searched concept is a decoder plan. Decode the plan, observe the actual generated tokens, and re-encode those tokens. Never advance the environment by assigning a predicted qbar directly to the next observed C.

### Step 4: use a realizable candidate as the unit of evidence

One node corresponds to one complete control configuration and its actual generated answer. All future effects of an earlier intervention must be recomputed. Do not assemble a winning trajectory by combining independently scored windows from incompatible generations.

### Step 5: separate proposal quality from allocation quality

A wide sampler may find better candidates simply because its proposal operator is better. Compare adaptive search with flat sampling, sequential refinement, and a fixed allocation of the **same operators**. A stronger proposal alone is not evidence for adaptive branching.

### Step 6: make feedback limitations visible

Under binary acceptance, all unsuccessful candidates have score zero. Until a success occurs, there is no reward ordering of those failures. Thompson sampling can still allocate attempts, but it cannot identify closeness to a solution from an absent signal. Richer public feedback is a separate arm with an explicit score and matched comparisons.

### Step 7: bound the intervention, then measure its effect

Small latent/PQ writes protect numerical scale, not task correctness. Retain the baseline candidate, cap each intervention, audit causal effects, and verify each resulting answer. Keep format damage and codebook/continuous-port mismatch observable.

### Step 8: preserve candidate-local state

Branches need independent controls, proposal RNG and gradient optimizer state. Shared weights may be reused; mutable caches cannot be shared across divergent branches without dependency checks.

### Step 9: account for the complete cost

The useful endpoint is success versus elapsed time. Branch bookkeeping and proposal replay are paid work; a gradient-assisted child is usually more expensive than a random child. Report equal-candidate comparisons separately.

### Step 10: let the experiment reject the design

If flat retries win, use retries. If PQ edits mostly damage valid outputs, keep that arm diagnostic. If binary feedback cannot guide useful refinement, report that limitation rather than claiming internal concept reasoning from a tree visualization.

### 4.1 What “scalable like a Transformer” means for V6.1

The requirement is that increasing model capacity, useful training data and computation produces predictable improvements in the **underlying language model**, while training remains dominated by parallel tensor operations. It is not satisfied merely because a tree fits in memory or a model can be distributed over GPUs.

Empirical Transformer scaling results motivate measuring held-out loss against model/data/compute, as in [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361). The [compute-optimal training study](https://arxiv.org/abs/2203.15556) also shows why growing parameters while holding training data fixed can give a misleading comparison. Neither paper supplies a scaling law for V6's staged, shared-core architecture; that law has to be measured.

Preserve five structural properties:

1. **Parallel teacher-forced training:** process all token positions together with causal masks, and all completed concept positions together within each core visit. No Python-level decision tree or sequential per-token search belongs inside the mandatory training forward pass.
2. **Scalable capacity:** increase width and independently parameterized layer count. Repeating the same two blocks more often is extra compute, not equivalent to adding independent Transformer parameters.
3. **No fixed narrow mandatory information channel:** H and A retain full hidden width; the concept state grows with the backbone. Every token need not squeeze through eight categorical codes or eight Z coordinates.
4. **Continuous, dense training gradients:** use the existing soft-PQ path for the scale-first family. Discrete inference actions remain optional. A hard argmax bottleneck is a separately trained architecture, not a free scaling improvement.
5. **A strong ordinary forward pass:** train a capable zero-search model. MCTS spends extra inference compute when useful; it is not a prerequisite for computing every training target or producing every answer.

This is the architectural reason to retain V6's Transformer token path and put AB-MCTS outside the training graph. Search should exploit learned capacity; it should not replace scalable representation learning with an expensive inner tree loop.

### 4.2 A parameterized neural family, not just a wider tiny checkpoint

Define the scalable family by:

~~~text
d                 hidden width of encoder, concept state and decoder
L_e, L_d          independently parameterized encoder/decoder block counts
c                 distinct blocks in each low/high core stack
R=2, n_L=2        fixed trained outer/low recurrence counts initially
N_c=6*c           core block visits per concept position
k=4               initial token-to-concept chunk factor
FF=3*d            initial gated feed-forward intermediate width
V                 vocabulary size, with tied token embedding/output weights
S, Q              PQ segment count and codes per segment
d_z, r_z          control dimension and adapter intermediate rank
~~~

The tiny reference is `d=128, L_e=L_d=3, c=1, S=8, Q=32, d_z=8, r_z=16`. At c>1, replace fL/fH with **distinct c-block stacks**, reused across the same trained recurrence schedule. This increases learnable core capacity without increasing R. Give each block application its own state/cache slot. The matched foundation teacher has a 2c-block concept stack before replacement; it is not the unchanged two-block tiny teacher.

For every configuration, first train its own foundation and then its replacement. The zero-control anchor belongs to that model size and recipe. Never widen/deepen an already trained tiny checkpoint and claim that the existing anchor equality or learned competence transfers automatically.

Use the initial fixed-ratio family `L_e=L_d` for a clean depth study. Altering encoder/decoder allocation later is a compute-matched architectural ablation. Keep causal token mixing in both paths so a larger decoder can use both fine token information and compressed plans. Scale c when testing depth so the concept stack does not remain an arbitrarily tiny shared module beside an increasingly deep backbone.

### 4.3 Co-scale the concept interface and diagnose its bottlenecks

Keep the original fixed `S=8,d_z=8,r_z=16` width family as a control. In a separate **capacity-matched scaling family**, for widths divisible by 16 use:

~~~text
S = d/16; Q = 32        # keep PQ segment width at 16
d_z = d/16
r_z = d/8
~~~

This reproduces the tiny interface at d=128 and lets its capacity grow instead of freezing it while the backbone expands. It is a proposed retrained family, not an edit to Phase F's frozen checkpoint. Keep full-width u, A and H. Keep the normalized Z bound and relative write caps initially, while measuring resulting effects at each width.

For this family, the PQ probability projection costs `d*S*Q = 2*d^2` weights; the codebook costs `Q*d`; the gated adapters cost `4*d*r_z + 2*r_z*d_z + 2*r_z`. These remain polynomial in d and comparable to a bounded fraction of ordinary layer capacity. The joint hard tuple space grows exponentially in S, but **neither training nor search enumerates it**: training forms factorized soft mixtures; AB-PQ proposes sparse coordinate edits. A future sampler with uniform segment choice may become less efficient as S grows; measure that loss rather than silently using answer-informed segment selection.

Z controls exploration rather than carrying the entire meaning of the input. A fixed small Z can therefore remain useful, but this is empirical. Compare fixed versus growing Z/rank at matched training and inference cost. Conversely, retaining full-width H/A prevents a mandatory narrow code bottleneck but can allow the model to ignore PQ. Measure held-out loss and verified behavior under matched concept/A interventions; code entropy alone does not demonstrate meaningful use.

For larger contexts, keep the number of concept positions proportional to T/k. Do not compress an arbitrarily long document into a fixed number of concept vectors. The local pooling rule is parallel and length-extensible; whether it preserves the information needed for long-range tasks must still be tested.

### 4.4 Parameter and computation laws for the proposed family

For the current bias-free attention/gated-FF blocks, ignoring small normalization terms, a block has approximately `13*d^2` weights under MHA. With K/V-to-query head ratio g, it has approximately `(11+2*g)*d^2`; FF width remains 3d. The active replacement model has:

~~~text
P_active ~= V*d
          + (L_e + L_d + 2*c)*(11+2*g)*d^2
          + 3*d^2                    # core input, B_u and B_A projections
          + d*S*Q + Q*d + S*Q        # probability head and codebooks
          + 4*d*r_z + 2*r_z*d_z + 2*r_z
          + O((L_e+L_d+c)*d)         # norms, states and gains
~~~

Count tied embeddings once and remove inactive teacher modules from active-parameter totals. Count them separately in resident storage where still loaded. This formula is a design ledger; the future instantiated model must report an exact count.

During parallel training, dense matrix work is proportional to `d^2*T*(L_e+L_d+6*c/k)` and attention work to `d*T^2*(L_e+L_d+6*c/k^2)`, plus the vocabulary/PQ heads and training backward. With fixed k and bounded recurrence, these have conventional Transformer-like polynomial dependence on width, depth and sequence length. There is no multiplicative tree-size factor in ordinary pretraining.

However, shared parameters are executed repeatedly, and A/B0/B1 have different frozen and differentiated paths. A generic `6*parameter_count*tokens` estimate is insufficient for comparing the whole V6 recipe. Record executed forward/backward work and total staged training compute, plus separately priced inference search.

An illustrative capacity-matched family, using MHA, FF=3d and a **newly trained 32,768-token vocabulary** for the large rows, is:

| Width d | L_e / L_d | c per low/high stack | S / d_z / r_z | Approximate active parameters |
|---:|---:|---:|---:|---:|
| 128, V=2,048 | 3 / 3 | 1 | 8 / 8 / 16 | 2.064M, current exact reference |
| 512 | 4 / 4 | 1 | 32 / 32 / 64 | About 52M |
| 768 | 6 / 6 | 2 | 48 / 48 / 96 | About 151M |
| 1,024 | 8 / 8 | 2 | 64 / 64 / 128 | About 312M |
| 2,048 | 10 / 10 | 3 | 128 / 128 / 256 | About 1.51B |
| 4,096 | 12 / 12 | 4 | 256 / 256 / 512 | About 7.21B |

These rows show a concrete architectural route, not an optimal depth/width ratio or authorized training plan. Larger rows can use a trained GQA variant after its quality comparison, changing their parameter ledger. The controlled 7.633M/29.291M V6 width tests retain the old tokenizer/interface and remain a separate way to isolate width effects before the broader family. Context training, data quality and optimization are not supplied by parameter counts.

### 4.5 Preserve parallel learning and stable optimization

The core's recurrence is over a fixed number of **layer visits**, not a sequential loop over the T/k concept positions during teacher-forced training. Each visit uses causal attention over the concept sequence in parallel. Thus the critical path grows with layer depth/visits, as in a deep network; it does not acquire an extra autoregressive concept-position loop during training. Autoregressive generation remains sequential, as for an ordinary language model.

Keep MCTS out of A/B0/B1's required training objective. B1 may score several sampled Z schedules in a batch and replay selected/weighted candidates for gradients, as already specified. That bounded candidate axis is training overhead that must be charged; it is not a tree whose branching expands with model width or sequence length. Search-trace distillation is a later optional data-generation stage with separate accounting.

Use the existing pre-normalized attention/FF blocks and the declared core state normalization for the reference. For newly deeper models, test an explicitly recorded residual-initialization recipe: scale attention-output and FF-output projection initialization by `1/sqrt(N_path)` in addition to the width rule, with N_path=L_e for the encoder, L_d for the decoder and 6c for the core. This is a conservative initialization hypothesis to compare against the unscaled reference in small depth pilots; it is not a proved residual/gradient bound or a modification to existing weights.

Do not blindly transfer learning rates, NCP/VQ weights or codebook updates across widths. Monitor per-domain loss, per-visit activation and gradient RMS, relative parameter updates, codebook usage and actual control-write effects. Keep NCP averaged over feature dimensions and VQ averaged over segments; normalize VQ's within-segment distance when segment width changes. Increasing S must not multiply the loss merely by introducing more segments. When d_z and r_z grow, initialize/calibrate the Z-to-rank, rank-to-state and state-to-gate projections against their own fan-in and write RMS; a single width multiplier is insufficient for all three. Repeat the freeze/anchor and full-gradient-path checks after each architecture change.

An explicitly derived and tested [maximal-update parameterization / muTransfer](https://arxiv.org/abs/2203.03466) is a candidate for reducing width-tuning cost. Its use requires assigning scaling rules to V6's tied embedding/head, shared recurrent blocks, PQ projection/codebooks, gates and readouts, then checking width transfer. The simple initialization rule above is **not** a muP derivation. Until that work is done, use a small, equal-budget learning-rate/initialization pilot per width and report its cost.

### 4.6 The experiment that can establish neural scaling

Run the scaling study with **MCTS disabled first**. Its primary endpoint is held-out language-model loss and task competence of the ordinary zero-control forward pass.

1. Use at least three model sizes and multiple training-token budgets within the same tokenizer/data family. Hold evaluation data fixed and independent; record repeated versus unique training exposure.
2. Separate width-only, depth-only and joint-capacity comparisons. Do not attribute gains from a larger vocabulary, different dataset or GQA conversion to width alone.
3. At each size, compare an ordinary Transformer, the NCP foundation, deterministic replacement and B1-at-zero. Show both equal-data and equal-total-training-compute frontiers. Include all earlier stages in V6's cost.
4. Compare the fixed concept/control interface with the capacity-matched family. If larger H/A improves loss while concept interventions remain ineffective, the backbone scaled but the searchable concept mechanism did not.
5. Fit descriptive held-out loss curves versus parameters, data and executed compute, with uncertainty; check predictions against a withheld size/budget. Do not transplant published scaling exponents or treat three favorable tiny points as a billion-parameter forecast.
6. Only then add equal-inference-budget retries, GradCuit and AB-MCTS at each size. Report whether search uplift improves, saturates or disappears as the base model becomes stronger.

The promotion claim should be: **this architectural family continues learning better representations as capacity/data/compute grow, and the concept/search additions earn their additional cost**. If the ordinary model scales but search does not, keep those conclusions separate. If loss plateaus because the core/PQ interface cannot use more capacity, revise that bottleneck before a larger run.

### 4.7 Architecture review verdict

V6 has a plausible Transformer-like scaling route because its main learned computation is causal attention plus feed-forward blocks over dense, width-scalable states, with fixed-factor concept compression and bounded recurrence. The proposed changes make that route explicit: scalable independent depth, a co-scaling concept interface, parallel dense training and a capable search-free model.

**The architecture is designed to permit scaling; equivalent or better empirical scaling than Transformers is not yet demonstrated.** Serving efficiency in section 14 is necessary for practical execution, but the neural learning-curve experiment in section 4.6 is the evidence needed for the user's scalability requirement.

## 5. Architecture and execution boundary

~~~mermaid
flowchart LR
    P["Original task prompt"] --> G["Frozen V6 generator"]
    G --> C["Complete generated candidate"]
    C --> V["Strict public verifier"]
    V --> T["AB-MCTS-A tree statistics"]
    T --> S["Choose GEN or CONT"]
    S --> O["Selected proposal operator"]
    O --> Z["Bounded Z schedule"]
    O --> Q["Optional sparse PQ patch map"]
    Z --> G
    Q --> G
    V --> R["Return verified answer or budget-exhausted failure"]
~~~

The controller sits outside the shared neural weights; its actions enter the concept path. This is inference-time concept control. It does not add a learned actor, value network, recurrent world model, new encoder, larger vocabulary or more core iterations.

The **main tree is an answer-refinement tree**. Section 12 describes a separate future prefix-tree experiment if token-by-token commitment is desired. Mixing their backup semantics is an implementation error.

## 6. Candidate, tree and verifier contracts

### 6.1 Answer record

Each immutable answer node contains at least:

~~~text
problem_id, node_id, parent_answer_id, refinement_depth
checkpoint_sha256, implementation_sha256, proposal_config_hash
operator_id, proposal_seed, generation_seed
Z_bank or a reference to its immutable stored value
PQ_patch_map or empty map
generated_response, generated_length, output_hash
strict_success, format_ok, reason, legal_trajectory
selection_score, feedback_mode
proposal_seconds, generation_seconds, verification_seconds
replay_count, backward_count, generated_tokens, cumulative_problem/batch_cost
own answer-statistics; GEN statistics; CONT statistics; child IDs
~~~

An optional cache is a disposable acceleration artifact. A node must be reproducible from its prompt, controls, patches, checkpoint and seeds without that cache. A sampled posterior score is never stored as a verified reward.

### 6.2 Initial answer and termination

1. Generate the original zero-control, zero-patch greedy answer and verify it. This is candidate 1 and consumes real budget for every method.
2. Store it as an ordinary answer child of the virtual root and initialize the corresponding statistics as a root `GEN` expansion.
3. If successful, return immediately. Search may not turn that success into a failure.
4. Otherwise, run selection/proposal/generation/verification until success or the first binding budget limit.
5. Accept only completed answers whose verification finishes within the allocated budget. Retain late work in the cost record without crediting it as an on-time success.
6. Exhaustion returns `success=false`. A best-scoring unsuccessful answer may be included for inspection with its failure status.

GraphNav acceptance must remain identical to `data.verify`, including the requirement for at least one legal hop. Reaching the goal earlier in a walk is not sufficient if the answer subsequently leaves it or terminates incorrectly. A full valid terminator must be generated; it must not be filled in after generation.

### 6.3 Freeze and identity

Use evaluation mode and freeze every shared parameter. Hash weights before and after the run. FP32 remains the reference. Search state resets between tasks. No state or posterior transfers from test problem i to problem i+1.

The virtual root stores no fictional answer or reward. The original greedy answer is a normal observed candidate, not a free seed supplied by a previous experiment.

Forcing this common first answer is a declared adaptation of the paper: its zero-action proposal differs from subsequent root-prior draws. Count it once in root GEN and expose that initialization in every allocation comparator. Do not claim that these observations are identically distributed draws from one proposal law.

## 7. AB-Z: the first post-V6 experiment

This uses the trained B1 gated beta=.25 checkpoint, subject to the completed V6 trained-weight audit. The direct and beta=0 checkpoints are secondary interface comparisons, not alternatives selected after inspecting the new test outcomes.

### 7.1 Controls and timing

Keep the existing deployment window `L=1`, eight coordinates per window and RMS(Z)<=2. GraphNav has 124 prompt tokens and at most 68 response tokens: sixteen walk chunks plus a terminator chunk. Only **17 response-plan slots** can affect scored generation under T=192.

~~~text
response window w = 0,...,16
concept position j = 30 + w
plan j first affects decoder position t = 4j+3
that position predicts response token index 124 + 4w
~~~

The existing bank has 48 rows for implementation convenience. Unused rows are not 48 independent opportunities to affect this response. For a shorter candidate, only its actually used slots are eligible for local refinement. Prompt-internal controls before j=30 remain zero.

### 7.2 Proposal kernel

Fix these initial development defaults before testing:

| Expansion | Proposal |
|---|---|
| Root GEN after candidate 1 | Independent standard-Gaussian Z for all 17 response slots, projected per slot to RMS<=2; greedy token decoding |
| Non-root GEN | Copy the parent's Z; choose one actually used response window uniformly; add `sigma*epsilon`, epsilon~N(0,I8), sigma=.25; project that window to the same ball; regenerate from the original prompt |
| Other coordinates | Preserve the copied values exactly |

The local kernel is our adaptation of answer refinement. It is not language-conditioned self-correction from the paper. Parent selection is informed by the declared score. The binary-feedback primary kernel does not use hidden paths, distances, or the verifier's failure location.

An independent root draw may prescribe controls for positions not yet reached; these are ordinary pre-sampled controls, not information about future outputs. Local mutations operate only on positions used by the parent. Dormant copied controls remain fixed until a later candidate reaches them.

Start with one mutation size. If a development-only diagnostic sweep is needed, use the declared set `{.1,.25,.5}`, charge its cost, select once, and lock the chosen value before the fresh test set is evaluated. Do not continually retune based on test failures.

### 7.3 Required allocation controls

- **Flat prior-Z sampling:** only root GEN, same first greedy answer and prior.
- **Sequential local mutation:** always refine the newest unsuccessful node with the same local kernel.
- **Fixed 50/50 allocation:** alternate root draws and local mutations of the best currently scored node; use seeded random ties.
- **AB-Z:** use section 9's adaptive allocation with identical proposal kernels.

The fixed and sequential controls distinguish adaptive allocation from merely introducing local Z mutation. Report repeated greedy answers and distinct legal trajectories; a small numerical Z difference alone does not establish exploration.

## 8. Discrete concept actions: PQ compatibility and AB-PQ

### 8.1 An anchored intervention instead of global hardening

Directly replacing every soft mixture with an argmax code would change the decoder's input distribution everywhere. It would also leave the continuous A port computed from a state that was trained with the soft readout. That is a new architecture requiring its own training/compatibility study.

For the frozen-checkpoint extension, use a **discrete action index with a bounded continuous write**. At a selected response window, an action chooses one PQ segment and one code index. All other segments use their original mixture. This tests code-index authority while preserving an empty-patch baseline.

For a selected window w, segment s and code k, define on that candidate's actual current prefix:

~~~text
qbar = original soft-PQ output at j=30+w
v = zeros(D)
v[segment s] = Q[s,k] - qbar[segment s]
r(x) = sqrt(max(mean(x^2), 1e-12))
lambda_eff = min(rho, eta * r(qbar) / r(v))
q_search = qbar + lambda_eff * v
rho = .25; eta = .10
~~~

This is a convex interpolation toward one code in the selected segment, with a full-vector write cap RMS(q_search-qbar)<=.10*r(qbar). The cap is defined against qbar itself, not against the larger token feature H or the adapter input e. If v=0, the write is zero. All nonfinite inputs are rejected as numerical failures.

Keep A computed by the unmodified core on the same prefix. Therefore A is identical **for a fixed token prefix and Z**, but may change later because the generated prefix changes. Do not claim that A stays globally identical across two different answers.

The interpolation cap is a proposed guard, not evidence that the PQ/A pair stays on its training distribution. Measure replay KL, destination/format errors and generated success. An empty patch takes the original PQ computation path exactly, avoiding a changed arithmetic order at the baseline.

### 8.2 Discrete action and proposal space

Store a sparse map `w -> (s,k)`, with at most four edited windows and at most one edited segment per window in the initial arm. Strength rho and cap eta are fixed for the whole arm; they are not additional coordinates optimized per test item. All neural weights and Z remain frozen, with Z=0.

This makes a local action categorical without calling the entire core discrete. The complete action space is finite and large; the algorithm imposes no fixed child count per answer, but it is not literally infinite.

Use the following reproducible proposal rules:

1. Obtain p[j,s,:] from a causal replay of the parent answer, or save equivalent prefix-conditioned probabilities during generation. Charge any extra replay. Do not use the gold continuation.
2. Eligible windows are those that predicted at least one token in the parent answer. Choose a window uniformly and choose s uniformly from the eight segments.
3. Sample k with probability .9 from p restricted to its top four codes, renormalized; with probability .1 sample uniformly from all 32 entries. This avoids permanently deleting low-probability alternatives. Resolve probability ties by code ID.
4. A root GEN starts from the empty map and the zero-answer trace. Draw one through `min(4,eligible_windows)` distinct edited windows uniformly, then independently choose their segment/code actions.
5. A non-root GEN copies the parent's map. With probability .1, remove a uniformly selected existing patch if one exists. Otherwise add/replace an action at an eligible window. If already at four patches, choose an eligible existing patched window for replacement; if none is eligible, remove one dormant patch before adding the chosen eligible patch.
6. On regeneration, recompute p, qbar, lambda_eff and A from the **new actual prefix**. The stored (s,k) remains an action on that prefix; it never reuses a stale parent qbar or parent future C.

An action at an unreached window is dormant. Log it and retain it as part of the immutable plan; do not pretend it produced an intervention on that attempt. The cap of four patches is a pilot complexity restriction and must be reported.

An optional secondary proposal uses the public first-failure chunk to bias the window distribution: half uniform over eligible windows, half uniform over the failure window and its predecessor when eligible. It belongs only in the richer-feedback arm. A locally observed error can have an earlier cause, so earlier windows retain nonzero proposal probability.

### 8.3 Duplicate handling

Hash the problem identity, complete control/patch configuration, checkpoint, decoder policy and generation seed. For deterministic greedy generation, omit an otherwise unused generation seed from the identity; a different proposal RNG seed cannot make identical controls a novel candidate. An identical configuration is a deterministic duplicate: avoid recomputing it. Limit resampling to eight proposals per expansion request; if all are duplicates, mark that node's GEN as `proposal_stalled` for the remainder of this problem. Continue through its CONT if eligible descendants remain. Count all proposal time.

For a duplicate configuration, retrieve its actual earlier evaluation if needed for a diagnostic, but do not create a new independent reward observation. Do not inject a fake zero score for a proposal that was never generated. If no eligible expansion remains, terminate as `proposal_stalled`, separately from a time/candidate limit. Eight failed random draws do **not** prove that the action space is exhausted. This bounded stochastic truncation and duplicate suppression modify the proposal process; report their frequency rather than claiming an exact unbounded-paper reproduction. They apply identically to the matched allocation controls.

**Equal output tokens do not imply equal search states.** Two different control maps may generate the same current answer but allow different refinements. Keep their states distinct and log output duplication. Likewise, never merge states merely because their code indices or current graph node agree. Default to a tree with one parent per node; a transposition DAG needs different backup accounting and is deferred.

### 8.4 Compatibility gate before a full AB-PQ run

On development data only, compare the empty map, sparse proposals, and a separately labeled hard-code diagnostic. The sparse proposal screen uses the same kernel and intervention schedule as the proposed search, not a cherry-picked best sample.

Required measurements:

- Empty-map logits and generated tokens reproduce B0 exactly under the same backend.
- Mean and paired uncertainty for text controlled-suffix NLL and symbolic NLL degradation; full-text NLL is reported alongside suffix NLL.
- Single-proposal strict success, format/terminator validity and failure categories.
- Full-vector write RMS, cap-hit fraction, effective lambda, soft-PQ entropy and per-segment code usage.
- With token randomness fixed, the fraction of problems gaining an additional distinct legal trajectory and an additional accepted solution among eight proposals.

Proposed gate: upper paired 95% bound on NLL degradation <=.05 nats in each domain; upper bound on format-validity loss <=2 percentage points on each symbolic split; and useful behavioral authority in the fixed proposal set. For the initial 64 hard problems, require at least four problems with an additional distinct legal trajectory and at least one zero-answer failure converted to strict success among the eight draws. These small counts are screening thresholds, not proof of an average improvement. Keep all denominators visible. Bootstrap problem/document identities, keeping the repeated proposals from one identity together.

Use an initial development screen of 128 text rows and 128 standard/128 hard symbolic items, with the first 64 hard items also receiving eight fixed proposal draws. If the numerical/compatibility gate fails, skip full AB-PQ testing and retain the diagnostic as a negative result. AB-Z may still be evaluated independently.

If a confidence bound is inconclusive, permit one predeclared doubling to 256 text, 256 standard, 256 hard and 128 eight-draw hard items, using the next fixed manifest entries. Double the authority count thresholds as well. Record the first and enlarged estimates; make the compatibility decision only at the enlarged sample. If the enlarged bound remains inconclusive, mark F2a unresolved and skip F2b. This is a development decision with a fixed maximum sample, not a repeated-significance procedure on test.

### 8.5 Causal likelihood protocol for the compatibility screen

Freeze each patch map using only the prompt, its zero-answer **self-generated** trace and the proposal RNG. Never choose windows/codes by examining a held-out reference continuation. Then evaluate that fixed map on the reference using teacher forcing: recompute qbar and the bounded write from each reference prefix as it becomes available. Teacher forcing is an offline likelihood measurement, not a reward available to search.

- **Standard GraphNav:** use its original masked demonstration for NLL, retaining the generator's exact response mask. The demonstration can determine an evaluation target but cannot enter proposal selection. Use hard GraphNav only for free-generation/verifier measurements: its stored continuation is padding and its likelihood mask is zero. A BFS-reference NLL would be a separate, explicitly labeled diagnostic.
- **Text:** use one document per row, a fixed 192-token chunk-aligned crop, a 64-token visible prompt and the remaining 128 tokens as evaluation targets. Choose document/crop identities before evaluating outputs. Select documents long enough for this crop and disclose this length restriction. Generate the proposal trace from the visible prompt only, stopping at EOS or the context limit; sample patches on its eligible windows. No cross-document control transfer is permitted.
- For text, the generalized mapping is `j = P/4 - 1 + w` for a chunk-aligned prompt length P. At P=64, j starts at 15 and at most 32 response windows can predict in-context targets. The GraphNav-specific j=30 mapping must not be copied into the text implementation.
- Score full-row and controlled-suffix token-weighted NLL with the exact same target masks for the empty and patched paths. The suffix means all valid targets at positions >=P, including the indirect effect after a patched window; it does not mean only the four directly patched predictions. Never score a next token outside the row. Compute paired confidence intervals by resampling complete rows/documents and recomputing the weighted means.

This screen estimates the effect of the declared sampled intervention family, not the likelihood of an MCTS-selected answer. The hard-code diagnostic uses the same causal information restriction.

The hard-code diagnostic uses explicit argmax/sampled one-hot code tuples and is labeled out-of-distribution. It has a small development-only budget. It is not the baseline for the primary architecture and cannot silently replace the soft model if its result looks better.

## 9. Exact AB-MCTS-A selection and backup

This section follows the paper's node-aggregation variant. It is deliberately explicit because an ordinary all-ancestor MCTS backup gives the wrong `GEN`/`CONT` histories.

### 9.1 Tree structure

Every answer node N, and the virtual root, has:

- `GEN(N)`: propose a new direct answer child of N.
- `CONT(N)`: choose among N's already generated answer children and continue refining below one of them.

Answer children are stored under `CONT(N)`. They are generated by `GEN(N)` but are **not descendants of that GEN object**. Enable CONT only when an existing child has a reachable enabled GEN in its subtree. Enable GEN only if it is not stalled and another child would fit the refinement-depth cap. Apply these masks recursively; if neither choice is available, propagate ineligibility upward. If only one choice is enabled, force that choice. Resource/depth exclusions are checked before selection.

For each answer/GEN/CONT statistics object h:

~~~text
alpha_h = .5 + sum(r_i)
beta_h  = .5 + sum(1-r_i)
sample_h ~ Beta(alpha_h, beta_h)
~~~

At N, sample GEN and CONT and choose the larger value. If CONT wins, sample each eligible answer child using its answer statistics, choose one, and repeat at that child. Use an explicit seeded random tie rule. Stop descent when GEN is selected. There is no PUCT prior multiplier or UCT term in this primary selector.

### 9.2 Histories that each object represents

| Statistics object | Observations |
|---|---|
| Answer N | N's own evaluated score plus scores of all generated descendants |
| GEN(N) | Scores of direct answer children created by GEN(N) only |
| CONT(N) | Scores below those direct children, excluding the direct children's own scores |

When a new child X is generated from selected answer N:

1. Initialize X's answer statistics with X's score once. X's own GEN/CONT retain their prior.
2. Add that score once to GEN(N).
3. Add it once to N's answer statistics and to answer/CONT objects on the ancestor path above N.
4. Do **not** add it to CONT(N), and do **not** add it to any ancestor GEN.

For example, a root child score .8 updates GEN(root) but not CONT(root). A later refinement of that child scoring .6 updates the child's GEN and answer statistics, CONT(root), and root bookkeeping. It leaves GEN(root) unchanged. This distinction is the method's allocation signal. [Paper appendix A.5.2, p. 18.]

### 9.3 Abstract pseudocode

~~~text
generate and verify the identical zero-action answer
insert it as the root's first GEN child; charge its cost
if accepted: return it

while time, candidate and memory budgets permit:
    if virtual_root has no reachable enabled expansion: stop as proposal_stalled
    N = virtual_root
    while an eligible expansion exists below N:
        draw scores for enabled GEN(N) and CONT(N)
        if GEN selected: break
        N = Thompson-selected eligible answer child under CONT(N)
    proposal = propose_from(N, frozen_operator_configuration)
    if no novel feasible proposal:
        mark GEN(N) proposal_stalled; propagate eligibility upward
        continue, or terminate if no enabled expansion remains
    X = generate_complete_answer_from_original_prompt(proposal)
    feedback = strict_verify(X)
    record all cost, including late work and numerical failures
    if evaluation finished after deadline: stop without success credit
    r = predeclared_selection_score(feedback)
    insert X and apply the GEN/CONT backup exactly once
    if strict_success(X): return X

return budget_exhausted_failure with best observed diagnostic candidate
~~~

Set the unsuccessful-proposal-request bound to four times the candidate safety cap, so repeated duplicate proposals cannot create an unbounded CPU loop. Every loop iteration checks time. Count numerical-failure evaluations against the candidate cap as well; an invalid result must not create free attempts.

### 9.4 Numerical and stochastic safeguards

Use positive FP64 CPU Beta parameters and per-problem seeded random streams. Selection sampling is inexpensive bookkeeping; it does not require MCMC or GPU synchronization per individual draw. Log tree statistics using finite numbers.

A completed numerical-failure candidate contributes failure score zero and its full cost. Abort that problem if its zero-action baseline is nonfinite, or after three consecutive numerical failures. An infrastructure/process error is an experiment failure, not a fabricated poor model score.

Under binary rewards, the successful candidate terminates the problem; preceding rewards are all zero. Before success, the selector responds to failed-evaluation counts and tree structure, not measured differences in failure quality. The binary arm is consequently a conservative test of exploration allocation; any result cannot establish learned recognition of promising failures. Report whether the allocation collapses to mostly root sampling and whether it helps despite sparse feedback. The partial-score arm is the actual test of ranking unsuccessful refinement regions using richer feedback.

## 10. Feedback: primary strict acceptance and secondary partial scores

### 10.1 Primary feedback

`r = 1` if the strict verifier accepts, otherwise `r = 0`. This matches the information used by the current verifier-only repair comparison most directly. Proposal generation may use the task prompt, candidate tokens, causal model outputs and candidate length, but not hidden trajectory metadata or ground-truth continuations.

The measured endpoint is the fraction of problems solved before the resource limit. Because GraphNav has an exact public verifier, an accepted answer can be returned immediately. The paper's hidden-test answer-selection problem does not disappear for other tasks; it is simply absent under this exact GraphNav verifier.

### 10.2 Optional richer public feedback

To investigate the paper's more informative scoring regime, define a separate arm:

~~~text
if strict_success: r_partial = 1
otherwise:
    r_partial = .4 * reached_goal_on_legal_prefix
              + .3 * min(legal_prefix_hops,16)/16
              + .2 * complete_well_formed_terminator_present
~~~

This uses the public header and generated tokens only. Failed candidates score at most .9. The full response still goes through the unchanged strict verifier. Use its `reached` flag and legal `trajectory` length for the first two terms. The terminator flag is one only when the strict verifier reaches an actual complete DONE chunk and returns either `accepted` or `wrong_goal`; a terminator after an already invalid walk does not count. Never infer its tokens from padding.

This shaping can reward long legal loops or a valid-looking wrong answer. It is a hypothesis, not a correct distance-to-solution metric. Log loop counts and score components, include the binary ablation, and use strict success for all headline outcomes. An increase in partial score without success is a failure of this guidance signal.

Expose the same partial scorer to flat-best selection, sequential refinement, fixed allocation, beam comparison if run, and AB-MCTS. Compare adaptive allocation within the same feedback mode. Comparisons against original binary-only GradCuit must explicitly identify the extra feedback rather than attributing the complete gain to MCTS alone.

### 10.3 Information prohibited from ordinary search

Do not use generator `hop_cols`, `seg_id`, `seg_pos`, hidden demonstrations, gold answers, or shortest-path distance as proposal/value input. BFS is a separately labeled algorithmic reference using the public graph. It is also suitable for testing the verifier and reachability.

With ten nodes and three colors, classical graph search is an obvious strong reference. This experiment tests learned concept-control/search efficiency, not whether neural search is the best way to solve an explicitly supplied small graph.

## 11. Gradient-assisted children and an actual post-GradCuit cascade

### 11.1 AB-Z-GradCuit as a separate policy

Run this only after AB-Z's basic contract passes. Root proposals stay as in AB-Z. For a non-root proposal:

1. Copy the parent's controls and apply the same single-window random mutation.
2. Replay the parent's rejected tokens at those candidate controls; compute the existing rejection-plus-.05-KL loss against zero-control probabilities on those same prefixes.
3. Apply exactly one bounded MaskedAdam update, with fresh zero moments for this proposal, LR=.1, per-problem gradient clip one, and the existing active-window mask.
4. Regenerate from the original prompt and verify the resulting answer.

The random mutation gives repeated expansions of one parent distinct starting points. Fresh moments make each edge reproducible without an ambiguous inherited optimizer history. This differs from V6's sequential ten-update GradCuit, whose moments persist across updates. Keep that original method as its own baseline. The replay at a mutated control is a rejection surrogate on the parent's answer, not an on-policy unbiased RL gradient.

Charge mutation, both probability replays, backward, the update, regeneration and verification. Include the same proposal with its gradient step replaced by a norm-matched random step if claiming a useful gradient direction. Obtaining that norm online also costs a backward pass.

Do not simultaneously optimize PQ code IDs with this gradient. A future joint `(patches,Z)` arm must define which variables and bounds each operator can change and must be separately compared with both single-interface policies.

### 11.2 Experiment order versus per-problem cascade

"After GradCuit and final testing" primarily means **queue placement**: Phase F runs after the original experiment has finished. It does not make the previous run's answers, timings, failure selection or search traces free initial information for a new benchmark.

An optional deployment cascade is a different experiment:

~~~text
zero answer -> original bounded GradCuit -> AB-MCTS if still unsuccessful
~~~

Its timer begins before the zero answer. Include all earlier GradCuit attempts and gradients in the total budget, and transfer only their already generated candidate states with documented initialization of tree statistics. Do not count one recorded candidate more than once. Report both all-problem success and conditional success on the remaining failures, with the residual-set denominator. Compare with a retry policy receiving that same total budget.

The primary Phase F comparison starts every policy independently from the common zero answer. It does not select only favorable failures from the old test run.

## 12. Why a prefix-concept MCTS is a different later experiment

A conventional prefix tree could select a concept action, decode the next four tokens, then branch again at the next chunk. That is feasible, but it is not the paper's complete-answer refinement tree.

If implemented later, its state must include the original task, complete generated token prefix, full control/patch history, position and partial encoder chunk, plus equivalent causal caches. A code ID, a qbar vector, or the graph's current node alone is not a sufficient neural state. The decoder and core attend to earlier history.

Each edge applies one causally available concept action and generates an actual four-token chunk. Only then can the newly observed chunk be pooled into C. A leaf value comes from a real decoded continuation and public verification; it cannot come from comparing a predicted latent with an unseen future concept. Partial-chunk failure and complete stopping must be specified.

If token decoding is stochastic, distinguish a chosen latent action from its stochastic token outcomes; record outcome nodes or explicit rollout samples. Otherwise repeated outcomes under one action will be incorrectly treated as distinct controllable actions. A deterministic greedy pilot avoids that ambiguity but may have poor coverage.

A prefix tree also risks locking in an unrecoverable early error. It needs backtracking and its own budget-matched token-prefix/beam baseline. Do not reuse section 9's answer-refinement GEN/CONT formulas without stating what its rewards and refinement operators now mean. This is deferred until the simpler full-answer concept search establishes useful authority.

## 13. Phase F experiment plan

### 13.1 Entry conditions

Start after `v6_tiny/runs/status.json` is `completed` or `completed_with_limits` and the V6 driver/child processes have exited. Inspect `REPORT.md`, `PRIMARY_RESULT.json` when present, final summaries, skipped stages and trained-weight audits.

- Pin completed source/checkpoint/data hashes in the new experiment manifest.
- Require finite checkpoints and passed zero-anchor/frozen-weight audits for the selected B1 checkpoint.
- Record any missing final GradCuit comparison. A partially completed V6 queue does not magically provide a validated baseline; rerun the necessary baseline under Phase F's own declared budget and record that work.
- If the original run failed numerically or its correctness audit failed, mark Phase F blocked pending an explicit implementation repair and revalidation. A good-looking training loss is not permission to ignore a failed invariant.
- A weak but valid checkpoint may still be used for a labeled development search study. The current low symbolic success makes this distinction relevant.

The existing V6 driver hashes all Python files in `v6_tiny/code` before every child stage. Editing or adding files there while it runs can cause the next stage to reject the implementation. The new draft and later separate runner must respect that boundary.

### 13.2 Ordered subphases

| Order | Proposed output / arm | Decision it resolves |
|---|---|---|
| F0 | Frozen source/checkpoint snapshot, fresh split manifest, selector/causality/cache fixtures, cost calibration | Is the experiment mechanically valid and properly budgeted? |
| F1 | AB-Z versus flat-Z, sequential mutation, fixed allocation, strongest retry and original GradCuit | Does adaptive allocation improve the existing trained control interface? Primary experiment |
| F2a | PQ empty-path, perturbation and hardening diagnostics | Is the new discrete-action interface sufficiently compatible and useful? |
| F2b | AB-PQ versus flat sparse-PQ sampling, sequential/fixed PQ refinement and retry reference | Is adaptive search over code-index actions useful? Only if F2a passes |
| F3 | AB-Z-GradCuit versus AB-Z and sequential GradCuit | Do gradient-assisted children justify their extra cost? |
| F4, optional | Whole-budget GradCuit-to-MCTS cascade; prefix-tree study only as a separately specified extension | Does a deployment cascade or different tree representation help? |

F1 is preselected as the primary frozen-interface test. F2 and F3 are secondary hypotheses with separate gates; do not switch the headline primary method to whichever looks best on test. If a secondary arm is promising, confirm it on a new locked set.

### 13.3 Data and randomness

Use a new Phase F split because existing V6 development outcomes have already informed this proposal. Proposed allocation:

| Split | Standard GraphNav | Hard GraphNav | Use |
|---|---:|---:|---|
| Development | 256 | 512 | Compatibility, fixed tuning, baseline selection, timing; use declared subsets for small screens |
| Locked test | 256 | 512 | One frozen primary comparison, paired across methods |
| Text guard | 256 development and 128 test document rows | N/A | First 128 dev rows form the initial screen; remainder reserved for its one allowed expansion; no future-target search reward |

Reserve generator seeds `641100`/`641110` for new standard/hard development and `642100`/`642110` for test. Search seeds are `{0,1,2}` in a distinct RNG namespace. Verify that these seeds have not already been used for this role; if there is a collision, choose and record replacements before looking at outcomes.

Exclude all recorded training graph fingerprints, the V6 development/test fingerprints, and identifiable earlier audit/diagnostic graphs. Freeze a new manifest before tuning and record any exclusion coverage limitation. The existing `data.canonical` ignores header edge ordering but retains node labels; it does not deduplicate all graph isomorphisms. Node-relabeling robustness is a separate paired diagnostic.

Use tokenizer-identical text documents disjoint by exact token content from training. Development and test must be disjoint by document identity/content, not merely by crop; reserve uninspected eligible documents for the new locked test. Reconstruct prior document exposure where possible. Previously inspected V6/V5 evaluation material may support development/regression checks, but cannot be relabeled fresh confirmation. If an uninspected test pool cannot be verified, label the text guard a reused-corpus regression check and report that limitation rather than inventing a fresh split. Existing V5-origin validation text does not become an independent new corpus through another split. Record source and near-duplicate limitations.

Search RNGs are derived from `(phase seed, problem identity, method, node ID, operation type)`. Batch compaction or another problem finishing early must not change a remaining problem's proposals. Three search seeds measure search randomness on this checkpoint; they are not three independent foundation training seeds.

### 13.4 Fair comparison matrix

Mandatory references are the strongest deterministic retry checkpoint chosen using development only, plus the original V6 preselected GradCuit policy on the same main B1 checkpoint. Include original zero-control greedy success for every checkpoint.

Within an interface, keep proposal family, verifier, feedback mode, context, precision and token policy fixed while varying allocation. AB-Z/PQ use greedy token decoding to expose the effect of latent actions. Temperature-one token retries remain the strong alternative use of inference compute; the difference in token policy is part of that deliberately competing method.

If F1 shows a material gain, add a fixed-width answer tree and/or progressive-widening control using the same local operator. Predeclare a small width or widening configuration on development; do not call an untested weak baseline representative of all MCTS. A simple best-score-first refinement control is also useful in the richer-feedback arm.

For PQ-only comparisons, show B0 and the main B1 checkpoint at zero Z give the same function before crediting exploration-adapter training. For direct/gated/beta comparisons, use the same parent selection and matched perturbation/cost reporting; parameterization changes may change mutation size in output space.

### 13.5 Budgets and runtime

The original V6 seven-hour ceiling belongs to the original experiment. Phase F has its own budget and report. Do not extend or overwrite V6's deadline to conceal additional work.

Predeclare an initial **two-hour additional Phase F allocation** as a planning ceiling, covering correctness checks, calibration and the prioritized F1 evaluation. The implementation must benchmark first and list which secondary arms fit. If even F1's fixed sample allocation cannot finish, stop with partial evidence and record the unfinished comparison; do not shrink the test set after inspecting outcomes. Any extension is a separately recorded budget decision.

Default candidate safety cap: 128 complete evaluated candidates including the zero answer. Maximum refinement depth: 32 for the pilot, reported as a resource restriction. There is no fixed per-node branching width. Stop earlier on acceptance, elapsed time, numerical limits or exhausted novel proposals. Report any cap that binds.

Primary compute endpoint: hard-goal success at **T_ref**, the mean batch wall time of the selected deterministic eleven-candidate retry policy on Phase F development data, measured at batch 16 on the same device. Lock its numerical value before testing. Use identical paired batches for all policies. Also report standard success, and batch-one latency on a fixed small development subset; batch throughput is not individual-request latency.

Use one synchronized wall clock per fixed batch, beginning before its first prompt encoding and zero answer. Credit each round's results at the time the neural work and verification for that round finish; every item shares this cumulative batch timestamp. Do not divide a batch duration by 16 and call that each item's deadline. Count selection, transfers, cache work, gradients, inactive rows and verification. Report one-time load/compile/warm-up costs separately and include them in the overall Phase F allocation. Warm up with development fixtures before either method's timed test batches; do not warm a candidate on the test instance for free.

Interleave method order across paired batches with a fixed scheduling seed; record device/background-load conditions. This reduces systematic thermal/order bias, particularly on the current fanless machine. Use backend-appropriate synchronization: the current `runtime.sync` only synchronizes MPS and is not sufficient by itself for future CUDA timing.

Secondary curves: equal candidate limits `{1,2,4,8,11,32,64}` and elapsed-time limits `{T_ref,2*T_ref,4*T_ref}`, subject to the overall phase budget. Larger-budget retry and sequential-repair references must receive the same larger budgets; preserve the original eleven-candidate policy as a separately labeled reference.

No future completion time or MCTS speedup is assumed from the current generation benchmark. The observed local model is much cheaper than the paper's frontier-model calls, so selector and cache overhead may be material.

## 14. Throughput, caches and stable scaling

### 14.1 Start with a correct replay path

The initial implementation should regenerate the complete response from the original prompt for every novel candidate. Reuse the frozen encoder's unchanged prompt encoding as V6 already does. All dependent core/decoder state is rebuilt. Optimize branch suffix reuse only after full-replay agreement passes.

For a changed action at concept index j:

- **Z edit:** invalidates core outputs/cache entries from j onward and decoder entries from t=4j+3 onward.
- **PQ-only edit:** for an identical token prefix, encoder and core computations do not depend on the PQ patch. Decoder entries from t=4j+3 onward do. Once generated tokens diverge, subsequent encoder/core features and caches also diverge.
- The first response action at j=30 affects decoder position 123, which is inside the prompt. Reusing the entire decoder prompt cache after changing this action is wrong.
- Each of the six shared-block applications retains its own cache. Weight sharing never licenses sharing different visits' K/V tensors.
- The cache key includes token prefix, position, document, checkpoint, Z/patch dependency history, dtype and execution mode. A current graph node or code tuple is insufficient.

Static caches in `FastIncremental` are mutated in place. Branches must use independent writable storage or a verified copy-on-write implementation. Assigning the same list/tensors to two candidate nodes is not a safe clone.

A later suffix optimization may preserve the parent token prefix strictly before the earliest affected predicted token, recompute the affected decoder boundary, then regenerate the suffix. Verify this against full regeneration after earlier-window edits, shorter stopping, dormant actions, and changed branches. A no-grad inference cache cannot substitute for Z-dependent activations needed by a gradient replay.

### 14.2 Batch independent trees

At batch 16, select at most one pending child per unresolved problem per round. Batch their neural generation; update each tree only when its own candidate finishes. This avoids within-tree decisions made against stale simultaneous expansions. Freeze solved rows and preserve row-specific RNG streams.

Bucket compatible prompt/response lengths and cache shapes when beneficial. Record padding and inactive-row computation; do not report nominal slots as useful decoded tokens. Apply comparable prompt reuse and batching opportunities to retry and gradient baselines. Report scalar versus batched selector equivalence before optimizing scheduling.

### 14.3 Bounded memory

For the current fully transplanted multi-head model, an approximate full-history K/V ledger is:

~~~text
KV elements per candidate = 2*d*(6*T + 6*T/4) = 15*T*d
at d=128,T=192,FP32: 1,474,560 bytes = 1.40625 MiB
~~~

This excludes weights, temporary tensors, encoder features and gradient replay. Storing 128 full caches per problem would cost about 180 MiB; doing that for 16 problems would cost about 2.81 GiB before other memory. At larger width/context, this grows quickly.

Store lightweight tree records for every node, but at most eight recent candidate caches per active problem initially. Use a bounded LRU cache; evicting a cache does not delete a node's evidence. Keep a shared immutable prompt-encoder cache where valid. Candidate controls for the 17 active windows require only 544 bytes in FP32, excluding object overhead. Storing full PQ distributions for every node can be much more expensive; prefer the needed proposal summaries or a charged replay.

Preserve at least 20% measured headroom against the chosen process/device memory budget. If memory rises, reduce cache retention or neural microbatch size while preserving per-problem algorithmic semantics. Record the change. Do not silently prune promising tree nodes according to memory pressure.

### 14.4 Scaling claims

AB-MCTS-A removes MCMC fitting cost; it does not make answer generation or Z gradients free. Thompson selection over a wide node still scans child statistics. Charge that CPU work and cache transfer cost.

Width/context scaling inherits V6's full token-attention cost. MCTS does not remove its quadratic prefill attention. Test width, context, number of active problems and candidate budget as separate axes. A useful tiny result motivates, but does not establish, scaling to larger models.

### 14.5 Recovery and reproducibility

Write append-only candidate events plus atomic tree checkpoints containing all Beta counts, immutable candidate states, RNG streams, consumed budget and configuration hashes. Never reset elapsed/candidate cost on resume. A resumed candidate must either be reproducibly completed or recorded as interrupted work; do not count a half-written answer as success.

Store total wall time and active compute time separately when an external pause is involved. Exclude arbitrary pause duration only from a clearly labeled active-compute measurement. Keep per-node replay/generation/verification costs recoverable from the event log.

### 14.6 LLM-scale architecture contract

**The neural scaling contract is section 4; this section specifies how to execute it.** Efficient tree bookkeeping cannot establish that a larger model learns better, and a larger backbone does not automatically make tree search useful. Keep four independently measured axes: trained parameter count, supported context/output length, simultaneous requests, and search compute per request.

Separate three components with explicit interfaces:

1. **V6-compatible neural worker:** a trained encoder/core/decoder checkpoint; batched prefill, token steps and concept-boundary steps; validated precision; explicit prefix-cache handles. Frozen weights are shared across candidates and requests.
2. **Per-request search controller:** small CPU tree/statistics, immutable action deltas, reproducible RNG, feedback and a hard resource ledger. A controller never owns a copy of the model.
3. **Serving scheduler:** token/cache admission, continuous batching across independent requests, device-group routing, deadlines and bounded queues. It dispatches work to neural workers and returns completed candidate events to their owning controllers.

The initial Phase F implementation can run all three in one process. Their separation is a design requirement, not a request to build a distributed service for the tiny pilot. A custom recursive V6 worker is not automatically compatible with an existing decoder-only inference engine; adapters and cache semantics require implementation and validation. Nor can this core be inserted into an arbitrary pretrained LLM while claiming the tiny transplant guarantee. Each scaled backbone needs its own competent foundation, trained replacement and anchor checks.

### 14.7 Memory and compute must have explicit bounds

Let P be prompt length, U the generated-token limit, T=P+U, k the chunk size, L_e/L_d the encoder/decoder layer counts, and N_c the number of core block **visits** per concept. Let d be hidden width, V vocabulary size, b_KV cache bytes per scalar, and r_e/r_d/r_c the K/V-head to query-head ratios. With equal width across stacks, the approximate single-trajectory cache is:

~~~text
KV_bytes ~= 2 * b_KV * d * [(L_e*r_e + L_d*r_d)*T + N_c*r_c*(T/k)]

dense matrix work  ~ d^2 * [(L_e+L_d)*T + N_c*(T/k)]
attention work     ~ d   * [(L_e+L_d)*T^2 + N_c*(T/k)^2]
vocabulary work    ~ T*d*V
~~~

These are analytic leading-order ledgers, not measured FLOPs or latency. They omit constants, extra projections, buffers, sharding replication and communication. Per-layer widths/head layouts require summing per-layer terms. GQA changes K/V projections/storage; it does not justify multiplying all attention arithmetic by the K/V ratio.

At the same six token-layer/six core-visit layout, d=4,096, T=8,192 and two-byte MHA cache storage give **960 MiB per trajectory**, before weights and activations. A hypothetical trained GQA variant with r=1/4 throughout gives 240 MiB; 128 full copies would still be 30 GiB. These are examples exposing the cache problem, not a specified 7B architecture or evidence of quality.

Require these bounds:

- Store at most the configured number of **live writable trajectories** per request, initially one. Keep a separately bounded cache working set; its limit is a byte budget, not a constant eight full caches at every model size.
- Retain lightweight historical tree records up to the declared candidate limit, with Z schedules represented by a root seed plus immutable mutation deltas. Materialize controls only for the active candidate. Cap/checkpoint delta chains so reconstruction cannot grow without bound.
- Store generated-token histories once per candidate or through a persistent prefix structure. Their worst-case host memory remains O(candidate_limit*U); do not claim constant total tree memory. Bound host memory as well as accelerator memory.
- Maintain reference-counted prefix pages and copy on write after the earliest invalidated position. Release branch-only pages when a candidate finishes or a cached branch is evicted. Cache eviction changes replay cost, not recorded evidence.
- Store the top-four PQ proposal probabilities/IDs and needed scalars, or recompute them on demand with charged cost. Do not retain all layers' activations or a full `[nodes, positions, S, codes]` tensor on the accelerator.

Paged KV allocation and prefix sharing have established LLM-serving precedent in [PagedAttention](https://arxiv.org/abs/2309.06180). Their use here must additionally respect V6's separate encoder/core/decoder dependencies and recursive visit identities. The paper's reported serving speedups are not predictions for this architecture.

### 14.8 Long-context execution and architectural choices

Start scaled models with exact, memory-efficient causal attention kernels that support the required document masks and gradient paths. [FlashAttention-2](https://arxiv.org/abs/2307.08691) motivates this execution strategy; it reduces attention-memory overhead without turning full attention into linear-time attention. Preserve a small FP32 reference for loss/logit/gradient checks.

For newly trained larger backbones, **GQA is the preferred cache-efficient attention candidate**, validated against a matched MHA reference at the first scale transition. The [GQA paper](https://arxiv.org/abs/2305.13245) supports sharing K/V heads and studies uptraining; it does not make changing the current checkpoint's heads a lossless conversion. Set head counts explicitly, train or uptrain the new variant, and rerun transplant/anchor gates. Keep separate caches for every recursive visit even when K/V heads and weights are shared.

Keep k=4 and the trained recurrence initially. Larger chunks, increased recursion, sliding/local attention, recurrent memory and context extrapolation each change the learned information path. They need their own trained comparison. In particular, concept compression does **not** remove full token-path attention or the vocabulary head. The LLM-scale draft targets efficient conventional-transformer scaling first; no subquadratic long-context claim is made.

Adopt BF16 weights/activations/KV where supported and validated; keep sensitive reductions, clipping norms, codebook-distance calculations, loss/KL accumulation and optimizer state in FP32 as specified by the V6 stability contract. Quantized weights/KV are later, separately calibrated deployment variants. Recheck zero-action behavior within the same backend/precision; mathematical zero injection does not guarantee bitwise equality between different kernels.

### 14.9 Request scheduling and distributed execution

Use continuous batching **across problems** first, with at most one pending expansion per tree. This preserves sequential AB-MCTS selection while occupying the accelerator with independent work. Dispatch by available token/KV capacity rather than a fixed number of rows. Use length buckets or paged storage and include queue delay when reporting end-to-end service latency. The fixed-batch Phase F benchmark remains the reproducible algorithm comparison.

For multiple accelerators, a neural worker may be a tensor-parallel group when the checkpoint cannot fit efficiently on one device. Replicate whole worker groups for request parallelism; keep one controller owner for each tree and prefer routing that reuses its resident prefix pages. Candidate completions need unique IDs, idempotent backup and explicit cancellation/accounting. Communication, cache movement and replicated K/V overhead belong in measured cost.

Treat the encoder/core/decoder as one logical worker before introducing pipeline boundaries. Moving a small concept activation between devices on every recursive visit can make communication dominate. Pipeline/sharding choices must be profiled at the intended concurrency, including the repeated use of shared blocks. A controller cannot assume that distributing more tree nodes produces a linear speedup.

Parallel expansions within one request are optional later work: they change the sequential algorithm because multiple proposals use stale statistics. Define reservations, duplicate prevention, update order and a separate asynchronous comparison before enabling them. Do not quietly add virtual losses and still label the implementation the exact reference selector.

Separate a latency-sensitive forward-only search pool from any gradient-repair pool. AB-Z/AB-PQ require no per-request backward pass; AB-Z-GradCuit remains an explicitly priced higher-compute mode. Give both pools hard request deadlines and admission limits. If capacity is unavailable, defer or return the declared zero/search result status rather than silently reducing one policy's benchmark budget.

Search is optional at deployment. Offer a zero-search path and bounded search tiers with fixed candidate/token/time ceilings selected before evaluation. Invoke search on verifier failure when a reliable verifier exists. For open-ended language, define the evaluator and final-answer selection first; model confidence or a reward-model score alone is not exact correctness. Report errors and reward exploitation in that evaluator. An LLM-scale service cannot inherit GraphNav's perfect-verifier assumption.

### 14.10 Training the scaled backbone stably

MCTS adds no shared parameters in Phase F. To reach larger LLM sizes, retain the V6 training sequence—competent NCP foundation, replacement core, bounded exploration adapters—and train/evaluate each size independently. Include total A+B0+B1 compute and matched ordinary-transformer/deterministic controls.

At larger sizes, use sharded parameter/gradient/optimizer storage and activation checkpointing as needed; validate globally correct domain/token denominators under distributed and accumulated updates. The reused core must accumulate gradients from all visits while owning only one optimizer state per shared parameter. Preserve deterministic candidate-selection/replay semantics when reducing B1 activation memory. Frozen weights carrying gradients to Z still participate in backward; freezing them does not remove that activation cost.

Scale vocabulary and data deliberately. The 2,048-token TinyStories/GraphNav setup is a controlled pilot, not a general-language pretraining recipe. A broad-language tokenizer changes embedding/output cost and requires a new training configuration. Log unique text exposure, duplicates, domain mixture and total processed tokens; repeatedly training on the same small corpus is not data scaling. Lock size-specific training budgets from pilot learning curves rather than copying tiny step counts or claiming a universal tokens-per-parameter law.

Keep Z dimension, adapter rank, PQ segment count and code count fixed for the first controlled width tests, and compare them with section 4.3's explicitly retrained capacity-matched family. Measure whether useful control authority degrades as width grows. Give every capacity change parameter/compute counts. Never enumerate all joint PQ tuples; the sparse action API samples coordinates regardless of d. Larger width alone does not establish useful discrete semantics.

Require finite-loss/state/gradient pilots, update-to-weight and per-visit norm diagnostics, codebook-usage checks, exact resume tests and the zero-action anchor at every scale. Post-normalization, small writes and BF16 are engineering safeguards, not a proof of optimization stability.

### 14.11 Scale ladder and promotion evidence

The following is a roadmap, **not part of the current launch or its two-hour Phase F budget**:

| Stage | Model/context target | Required evidence before promotion |
|---|---|---|
| S0 | Current 2.064M active-parameter model, T=192 | Correct search semantics, verifier integrity, cost-matched mechanism test |
| S1 | V6's controlled d=256/d=512 family: about 7.633M/29.291M active parameters | Fresh trained anchors, independent seeds, same-data and same-training-compute comparisons |
| S2 | Supported contexts 512, 1,024, 2,048 at a fixed passing width | Actual longer tasks/documents, cache correctness, memory/time curves, preserved quality |
| S3 | Approximately 100M-350M parameter class; 2K-8K trained contexts; broad-language data | New audited width/depth/vocabulary/GQA ledger, distributed training/serving, a real language-model baseline |
| S4 | Approximately 1B, then 7B+ class if S3 justifies it | Multi-device fit, independent trained-seed evidence where feasible, stable long runs, quality-versus-total-compute advantage and service-level latency |

The later parameter classes are planning envelopes. They do not follow from simply inserting d into the tiny ledger: depth, vocabulary, heads and core capacity must be specified and counted before training. No numeric completion time or hardware-fit promise is made for them.

At each stage, measure zero-search quality separately from search uplift. Compare additional training/model size against additional inference search as alternative uses of compute. If a larger zero-search model outperforms a smaller searched model at the same useful-answer cost, that is part of the architecture decision.

Proposed systems gates, locked before each scale experiment:

- All causal/cache/backup/anchor fixtures pass under the chosen precision and sharding.
- Peak accelerator and host usage preserve at least 20% declared headroom, including compilation, replay, serialization and recovery peaks; no sustained per-request memory growth.
- At fixed candidate work, selector/bookkeeping overhead is <=5% of elapsed service compute. Above this threshold, optimize or record the scale gate as failed; do not conceal it inside generation time.
- Meet a predeclared p50/p95 end-to-end latency target at fixed offered load and report throughput, queue delay, accepted-answer cost and timeout rate. There is no universal useful LLM latency threshold.
- Show quality gains at matched wall time and accelerator-seconds, with uncertainty and all baseline costs. Added hardware can improve latency while increasing total compute; report both.

### 14.12 Scalable implementation acceptance additions

Before describing a future implementation as LLM-scale ready, extend section 18's checks to cover:

1. Persistent action-delta reconstruction equals materialized controls after branching, eviction and resume.
2. Paged/shared prefixes respect the decoder's first-response boundary and all six core-visit dependencies; copy on write isolates every branch.
3. Candidate IDs yield exactly one statistics update despite duplicate or out-of-order distributed completion events.
4. Admission, cancellation and timeout free request-specific pages, gradients and host tree records; interruption keeps consumed costs.
5. Distributed gradients/loss denominators and candidate-selection replay match the small single-device reference; no shared parameter receives duplicate optimizer steps.
6. Mixed precision, GQA, larger vocabulary and longer supported context each have their own anchor/quality/cost evidence.
7. Continuous-batch service runs pass sustained mixed-length/mixed-budget load tests with bounded memory and reported tail latency.

These requirements make scale an implementation and measurement contract. They cannot guarantee an LLM scaling law from a tiny symbolic experiment.

## 15. Text evaluation and the role of the verifier

TinyStories next-token likelihood is useful for checking the frozen model and PQ interventions. It is not a deployment reward that a search procedure may obtain from an unseen reference continuation.

For the initial Phase F:

1. Keep ordinary zero-action text likelihood/samples as a regression check.
2. Evaluate fixed random Z/PQ interventions on held-out text for full and controlled-suffix NLL. These targets are evaluation-only and never fed to the search proposal or selector.
3. Limit text-search success claims to explicitly constrained tasks with prompt-visible, independently checked conditions, such as a required word set or a specified structured output. Such scores measure those constraints, not narrative quality.
4. Treat fluency/coherence assessments as separate evaluations with a declared scorer and error analysis. Searching for maximum model likelihood can favor generic repetitive text and is not evidence of improved reasoning.

If a future scorer has only partial public tests, distinguish coverage (some generated answer passes hidden evaluation) from selected-answer success. Select the final answer using only available feedback. Do not use hidden outcomes to pick a node, as the paper's appendix-D coverage versus Pass@2 gap illustrates.

## 16. Metrics, statistics and decisions

### 16.1 Required reports

Report each method and feedback mode separately:

- Strict standard/hard solve rate; paired improvement over selected retry and GradCuit baselines.
- Success at the fixed candidate/time budgets; first-success time; failures and censored tasks.
- Generated and verified candidates, actual tokens, proposal replays, backward passes, verifier calls, CPU selector time, cache-copy time, peak memory and compilation time.
- Illegal transition, malformed walk, wrong goal, malformed/missing terminator, numerical failure, duplicate proposal, duplicate answer and resource exhaustion counts.
- Root GEN fraction, GEN/CONT choices by depth, average/max refinement depth, node degree distribution and unused budget after early success.
- Distinct legal trajectories, accepted alternatives, and improvement conditional on parent failure, always with the all-problem companion denominator.
- For PQ: chosen code mass/rank, effective lambda, patch count, dormant patches, code entropy, write caps and induced replay KL.
- For hybrid children: gradient norm, step norm, rejection loss and whether verified behavior changed. A lowered rejection loss alone earns no success credit.

Report partial evaluations by their actual per-method item counts and identity sets. Never silently divide an unfinished arm by the planned full sample size or pair arrays that contain different problems.

### 16.2 Uncertainty and selection

For each search seed, pair methods on identical problem identities. For a combined result, average the three search-seed outcomes within each problem and bootstrap problem identities (2,000 resamples) to obtain a 95% interval on the paired gain. Report per-seed results separately. This captures item/search variation conditional on one trained model; it does not measure foundation-seed uncertainty.

Preselect AB-Z with binary feedback as the primary candidate. On development, choose the strongest eligible comparator among the locked retry reference and original GradCuit policy at T_ref. Lock that comparator, proposal parameters, feedback mode and budgets before test. Report both comparisons regardless of which is selected. Do not treat auxiliary test winners as confirmatory discoveries.

### 16.3 Proposed acceptance ladder

| Gate | Pass condition / interpretation |
|---|---|
| F-G0 correctness | Empty-action anchor, causal boundary, branch independence, selection/backup and verifier fixtures pass |
| F-G1 usable proposals | The fixed proposal family yields verified behavioral alternatives; duplicates and validity damage are quantified |
| F-G2 allocation benefit | Adaptive allocation improves over flat/sequential/fixed allocation with the same operators and feedback at matched cost; paired interval supports the direction |
| F-G3 pilot deployment benefit | At T_ref, primary AB-Z improves hard success by at least 3 percentage points over its locked strongest comparator, with positive paired 95% lower bound; standard-success loss upper bound <=2 points |
| F-G4 discrete-interface compatibility | Section 8.4 passes before attributing PQ search results to a viable concept-action interface |
| F-G5 confirmation | Selected method reproduces on fresh test problems and at least three independently trained foundation/core seeds before a broad architecture claim |

The 3-point threshold is a proposed practical criterion, not a power calculation or prediction. A result may be useful mechanism evidence while failing the deployment gate. A confidence interval crossing zero is unresolved. Binary-feedback failure alongside a partial-score gain establishes dependence on richer task feedback.

## 17. Future implementation map

Only this Markdown design is authored now. The following files are a handoff specification, not files created by this task:

| Proposed path under `v6_1_mcts/` | Responsibility |
|---|---|
| `spec/draft_v6_1.md` | Immutable copy of the selected revision |
| `source_v6/` | Pinned copy of the completed V6 source, with imported provenance and no live source edits |
| `code/concept_actions.py` | Z mutation, sparse PQ map, exact empty-action path and intervention records |
| `code/generate_controlled.py` | Readout hook in full replay and both incremental paths; branch-safe generation |
| `code/ab_tree.py` | Answer/GEN/CONT objects, Thompson selection, exact histories and duplicate/exhaustion accounting |
| `code/feedback.py` | Strict verifier wrapper and separately named partial scorer |
| `code/evaluate_f.py` | Fresh paired splits, all baseline policies, fixed budgets, event logs and paired statistics |
| `code/check_f.py` | Contract fixtures listed below |
| `code/drive_f.py` | Standalone post-V6 dependency checks and Phase F budget; no automatic V6 relaunch |
| `PLAN.json`, `runs/`, `REPORT.md` | Frozen settings, candidate evidence, skips/failures and conclusions |

The PQ override must affect **both** the ordinary `System.plans` path and the cached `Incremental`/compiled `FastIncremental` readout. Editing only `Foundation.pq` or only the full-sequence forward is not sufficient unless all cached paths demonstrably call that same new contract. The original `repair_gradient` is not patch-aware; the joint PQ+gradient arm therefore stays deferred unless replay is explicitly extended and tested.

A proposed public interface is:

~~~text
generate_controlled(checkpoint, original_prompt,
                    z_bank, pq_patch_map, token_policy,
                    proposal_seed, generation_seed, budget_context)
    -> complete_candidate + verifier_result + causal_trace + cost_record

expand_answer(parent, operator_config, problem_rng)
    -> immutable proposal or explicit proposal_exhausted

backup_once(parent, new_child_score)
    -> update the documented answer/GEN/CONT histories
~~~

Prefer pure proposal/selection functions and explicit state over implicit mutation of a globally current model/cache. Sharing frozen weights is safe; sharing candidate-specific mutable controls is not.

## 18. Checks required before future execution

These are future implementation acceptance checks. This drafting task does not run them or claim that they already pass.

1. Empty PQ map and zero Z reproduce B0 logits and greedy generations on ordinary, malformed-boundary and short-response fixtures.
2. A future patch cannot change earlier logits. Editing j=30 can affect decoder position 123, but no earlier position.
3. A concept is computed only from a completed actual token chunk. No predicted qbar is substituted for observed C.
4. Full replay, eager incremental and compiled static-cache paths agree for fixed controls/patches, including edits to earlier windows and stopping chunks.
5. A nonzero PQ patch reaches the decoder through the intended segment; untouched segments and A agree for the same prefix/Z.
6. Convex-interpolation and RMS-cap bounds hold, including zero/tiny base vectors, identical code vectors and nonfinite input rejection.
7. A sibling mutation cannot alter its parent or another sibling's controls, moments, cached K/V, tokens or statistics.
8. Core caches have six distinct visit slots; decoder boundary invalidation includes the final prompt position when required.
9. Proposals condition on actual parent prefixes, never future labels; fixed seeds reproduce actions after replay/cache eviction.
10. Per-problem proposal and selector sequences remain equivalent under row permutation, solved-row compaction and microbatching.
11. GEN/CONT histories reproduce the paper's appendix A.5.2 example and the numerical fixture below.
12. Each evaluated candidate updates the intended histories once. Direct children do not update their parent's CONT; descendants do not update ancestor GEN.
13. Enabled-choice masks propagate correctly through a tree containing depth-limited leaves and stalled GEN nodes. With all expansion targets unavailable, search terminates without an infinite resampling loop and without claiming that random proposal stalls prove exhaustive enumeration.
14. Exact deterministic duplicate configurations receive no repeated independent evidence; equal answers from different controls remain distinct refinement states.
15. The candidate safety cap includes the first zero answer and every generated/verified candidate. The last allowed candidate is actually verified.
16. Time is checked before expensive work and after verification; late success is not credited, and its consumed work remains in the cost record.
17. Accepted answers and solved-row state stay fixed. Exhaustion, numerical failure, interruption and experimental failure have distinct statuses.
18. Strict stopping checks all four generated terminator tokens; valid loops, early DONE, wrong goals and incomplete output cannot falsely pass.
19. Partial feedback cannot exceed .9 on failures or read hidden generator fields; binary and partial modes produce separate evidence files.
20. A tiny synthetic selector problem where refinement is useful and one where only fresh draws help exercise both choices. They test behavior, not universal optimality.
21. Hybrid replay is finite at zero and a finite-difference fixture matches selected Z gradients; fresh-moment semantics and active-window masks are enforced.
22. A checkpoint/resume fixture reproduces next selection/proposal/backup and preserves already consumed budget.
23. Before/after shared-weight hashes match; all source and checkpoint hashes recorded in the final report match the executed artifacts.
24. Reconstruct all success/cost curves from candidate events and verify that paired statistics use matching identities and completed observations.
25. Measure sustained memory, cache eviction/rebuild equivalence and timing under the maximum declared batch, depth and candidate cap.

### Backup numerical fixture

Use the paper's example: root children have own scores .8, .0, .2; the first child has descendants .8 and 1.0, and the third has descendant .3. With Beta(.5,.5) priors:

~~~text
GEN(root) observations = [.8,.0,.2]       -> alpha=1.5, beta=2.5
CONT(root) observations = [.8,1.0,.3]     -> alpha=2.6, beta=1.4
answer child 1 observations = [.8,.8,1.0] -> alpha=3.1, beta=.9
GEN(child 1) observations = [.8,1.0]      -> alpha=2.3, beta=.7
CONT(child 1) has no deeper observations -> alpha=.5, beta=.5
~~~

Adding a new direct child of child 1 with score .5 gives GEN(child 1)=(2.8,1.2), answer child 1=(3.6,1.4), CONT(root)=(3.1,1.9). GEN(root) and CONT(child 1) remain unchanged. The new answer's own statistics are (1.0,1.0).

## 19. Deferred fully discrete architecture

A truly hard-code V6 variant would need more than inference-time argmax. It would replace qbar with concatenated selected Q entries and train the decoder/core to tolerate those discrete choices while preserving a competent reference route. Possible future training methods include an explicitly declared straight-through estimator or a separately trained proposal conditioned on context. Neither is part of the current validated V6 contract.

Before such a variant is promoted, require a hard-versus-soft competence comparison, measured code utilization and useful causal interventions, training/inference alignment, gradient-estimator bias discussion, and additional training-compute controls. Decide explicitly whether A is retained, conditioned jointly on discrete choices, or reduced; removing it is a separate architectural ablation.

If successful search traces are later distilled into a proposal policy, use fresh training/development partitions and verified traces only, keep test traces out of training, and count the additional weights/data/compute. That is a new learned model, unlike the frozen Phase F design.

The present recommendation is therefore **discrete actions over a continuous trained concept interface**, with a protected no-action path and an explicit compatibility gate. It does not claim a discrete latent world model or learned symbolic semantics.

## 20. Source and provenance map

The design review covered the complete supplied 30-page paper, including appendices A-D, with visual inspection of the tree/algorithm/backup pages and key result/limitation tables. It also covered the full root V6 draft and the current model, training, control, generation, verification, evaluation and queue paths listed below.

| Source | Design dependency |
|---|---|
| [AB-MCTS.pdf](AB-MCTS.pdf), pp. 3-6, 14-18 | Answer-refinement trees, GEN/CONT, Beta update and backup |
| Same paper, pp. 7-10, 19-25 | Baselines, generation budgets, timing/evaluator limits, breadth/depth behavior |
| Same paper, pp. 26-30 | Multiple generators and coverage versus answer-selection limits |
| [v6_draft.md](v6_draft.md), sections 3-6, 8-11 | Original design, causal timing, frozen anchors, controls, costs and gates |
| [model.py](v6_tiny/code/model.py) | Soft PQ, codebook fitting, continuous port, core recurrence and adapters |
| [runtime.py](v6_tiny/code/runtime.py) and [train.py](v6_tiny/code/train.py) | Whole-schedule selection, masks, checkpoints and immutable implementation hash |
| [generation.py](v6_tiny/code/generation.py) | Prompt boundary, incremental/static caches, full terminator, active controls and MaskedAdam |
| [data.py](v6_tiny/code/data.py) and [graphnav.py](v6_tiny/code/graphnav.py) | Public verifier, graph identity, exact format and available versus privileged information |
| [evaluate.py](v6_tiny/code/evaluate.py), [audit.py](v6_tiny/code/audit.py), [anchor_gate.py](v6_tiny/code/anchor_gate.py) | Original search policies, replay objective, trained-anchor checks and preliminary evidence |
| [drive.py](v6_tiny/code/drive.py), [PRE_REGISTRATION.md](v6_tiny/PRE_REGISTRATION.md), [SPEC_COMPLIANCE.md](v6_tiny/SPEC_COMPLIANCE.md) | Placement after the original queue, budget and evidence boundaries |
| [PagedAttention](https://arxiv.org/abs/2309.06180) | External primary-source basis for paged KV allocation and prefix sharing; V6-specific cache rules remain our design |
| [GQA](https://arxiv.org/abs/2305.13245) | External primary-source basis for grouped K/V heads and a separately trained/uptrained attention variant |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | External primary-source basis for memory-efficient exact attention execution, without a subquadratic-compute claim |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) and [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | Empirical model/data/compute scaling motivation; their fitted laws are not assigned to V6 |
| [Tensor Programs V / muTransfer](https://arxiv.org/abs/2203.03466) | Width-aware hyperparameter-transfer direction, subject to a separate V6 parameterization derivation |

For the six additional systems/scaling papers, the primary-source abstracts were checked to ground these limited claims. They were not reviewed page by page like the supplied AB-MCTS paper. The architecture family, scale ladder, initialization pilot, service budgets and V6 integration details are proposed design choices, not reported results from those papers.

Provenance at drafting:

~~~text
AB-MCTS.pdf SHA256:
084818ed5fa8aac24321a5af830f0789655d0387f8222673c4a11a5831ce9dbf

v6_draft.md SHA256:
7bef1ead33fbb49d4067fc714f4b25b04bae46b0805365af6f02027d0d4f3543

V6 code aggregate SHA256 using runtime.implementation_hash's algorithm:
2043e18416dd800952fbc38062ec4913537ebeb645c3f04c86ed5b93dc8c7811
~~~

The eventual Phase F manifest must additionally pin the actual completed checkpoint hashes and final V6 evidence. Those future artifacts are not replaced by the source hashes above.

## 21. Review of this draft from first principles

This second pass reviews the written design against the actual information flow, observable success condition and cost boundary. It is a design review, not a claim that an unimplemented system passed execution tests.

| Review question | Resolution in this revision | Remaining uncertainty |
|---|---|---|
| Does NCP make V6 a discrete search environment? | No. Separate the discrete codebook indices from the soft decoder mixture, continuous A route and token history. Start with the trained Z interface; gate discrete-index writes. | Whether any code-index intervention has useful semantic authority |
| Can MCTS work without a hard-discrete core? | Yes. Sample finite control proposals and use actual generated answers as nodes. Avoid enumerating the continuous control space or all joint PQ tuples. | Coverage and efficiency of these particular proposal kernels |
| Is the state sufficient to reproduce a candidate? | Preserve full prompt, controls, patches, generated prefix/history, model identity and RNG; never identify neural state solely by current graph node or code ID. | Future cache implementation must establish replay equivalence |
| Is extra search depth confused with neural recursion or response position? | Keep all three separate. The primary tree refines complete answers; shared core recurrence stays at its trained depth. | A later prefix tree requires a different explicit contract |
| Can feedback rank unsuccessful attempts? | Binary feedback cannot rank their quality. Label the primary experiment an exploration-allocation test and study partial feedback separately. | Shaped feedback may favor loops instead of solutions |
| Does backup implement the paper's mechanism? | Direct-child rewards update the producing GEN; deeper rewards update ancestor CONT, without updating ancestor GEN. Include the paper's numerical fixture. | Sol must run the fixture and event reconstruction on the implementation |
| Does a small PQ write guarantee compatibility? | No. Specify a causal, fixed-proposal likelihood/behavior screen and reject or leave unresolved the PQ arm when its gate fails. | Bounded vector magnitude may still cause large decoder behavior changes |
| Can the compatibility screen accidentally use answers to choose patches? | Separate self-generated proposal traces from teacher-forced measurement. Define text prompt boundaries and prohibit hard-GraphNav padding NLL. | Prior text exposure and near-duplicate coverage may be incomplete |
| Are duplicates and dead ends truly defined? | Ignore exact deterministic duplicate evaluations, preserve distinct control states with equal outputs, and mark bounded resampling stalls explicitly. Propagate eligibility through depth-limited nodes. | Proposal suppression changes the unbounded algorithm and must be measured |
| Is a gain fairly attributed to MCTS? | Compare flat, sequential, fixed and adaptive allocation using the same operators/feedback; keep GradCuit and token retries at matched elapsed cost. | A two-million-parameter model may spend too much time on controller overhead |
| Can a result corrupt or silently extend the running experiment? | Put Phase F after the original completed queue in a separate pinned source tree with its own manifest, deadline and report. | Actual completed checkpoints and final V6 conclusions must be checked at handoff |
| Can the neural architecture grow like a Transformer? | Define scalable width, independent token/core depth and a capacity-matched PQ/Z family; preserve dense gradients and parallel teacher-forced training. MCTS is absent from the required training forward pass. | The proposed depth/width/interface family must earn its capacity on held-out learning curves |
| What establishes architectural scaling rather than serving capacity? | Evaluate zero-search loss across model sizes and token budgets against ordinary Transformers at matched total training compute, then measure search uplift separately. | Published Transformer exponents and tiny success scores cannot establish V6's scaling law |
| Is the architecture designed for LLM-scale operation? | Specify bounded live trajectories, paged shared prefixes, sparse control histories, independent-request batching, worker groups, sharded training and an explicit size/context ladder. | Each later scale must pass memory, latency, quality and correctness gates; it is not implemented here |
| Does the concept bottleneck make long-context attention cheap? | No. Account for both full token stacks, the vocabulary head and every recursive visit. Consider trained GQA and exact efficient kernels without claiming linear attention. | A different subquadratic backbone would require a separate trained architecture study |
| Would a positive tiny run establish scalability or broad model quality? | No. Track memory/time scaling, preserve single-training-seed caveats, and require independent trained seeds before broader claims. | Larger-width/context behavior and unrestricted text quality remain untested |

**Design conclusion:** implement F0 and the AB-Z allocation comparison first. Advance to AB-PQ only after its declared compatibility/authority screen; add gradient-assisted children as a separately costed experiment. Independently, scale the underlying neural family through the search-free learning-curve study in section 4.6. A useful search layer and a scalable learned backbone are two claims requiring different evidence. If the simpler retry policy or ordinary Transformer wins at matched cost, preserve that result. The design retains a later path toward a genuinely hard-code model if the evidence warrants new training.
