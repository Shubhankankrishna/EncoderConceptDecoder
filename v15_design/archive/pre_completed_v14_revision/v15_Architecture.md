# V15 — Predictive plans with memory across concept depth

**Status:** proposed architecture, reviewed analytically; not implemented, trained, or demonstrated to outperform v14. Written 16 October 2026, IST. “Better” is the hypothesis to test, with explicit criteria below.

**Design in one sentence:** retain v14-tiny's causal local/global backbone, replace its moving residual targets with a small frozen hierarchy of future-span codes, expose *predicted* code alternatives to the token decoder, and replace the concept stack's explicit depth attention with a bounded-size associative memory that preserves its ordinary residual path.

The primary research candidate is **V15-Δ**: a delta-rule memory only inside the concept stack. **V15-RLS** substitutes the Depth Memory paper's exact regression update as an ablation. The large RLS module is not installed throughout the token backbone merely because its paper reports a win at a different scale.

This document specifies a new model and its evaluation. It changes no running v14 process. Companion evidence and analytical checks live in `v15_design/`; the final self-review is §16.

## 0. Evidence boundary and reading order

Read §§1–3 for the reasoning, §§4–10 for the implementation contract, §§11–14 for cost and experiments, and §§15–16 for the review.

| Source | What is used | What it does not establish |
|---|---|---|
| `v14_Architecture.md`, especially §§1, 4, 6–9, 11 | Causal role split; target-instability problem; local and copy baselines; preregistered tests | That every proposed v14 mechanism was implemented or succeeded |
| `v14_tiny/model.py`, `v14_tiny/runs/base/config.json`, `v14_tiny/DEVIATIONS.md` | The actual 6,807,755-parameter model: E2 R=1, four concept blocks, four decoder blocks, two E3 layers | A matched causal comparison against the larger V13 model |
| `v14_tiny/tests_g0.py` | Exact local look-back distances 51 and 66; group release rule | A proof for future modified code without rerunning tests |
| `Next-Concept Modeling.pdf`, pp. 5–7, 26–29, 32–34 | Frozen hierarchical codec; useful partial reconstruction; conditional coarse/fine prediction; error robustness | Human-semantic code identity; automatic benefit for v14; M2 speedups |
| `DEPTH MEMORY.pdf`, pp. 5–9, 22, 28–34 | Preserve residual stream, store layer deltas, content-dependent reads, delta/RLS alternatives, state accounting | Persistent conversational memory; a universal gain over existing depth attention |
| `Dataset_v13.md`, §§6, 9–10 | Semantic-group splits, exact task success, equal-time comparisons, continual evaluation | That code accuracy can substitute for task success |

At the initial evidence read on **16 October, 01:16 IST**, v14 base `DONE` existed and the post pipeline had started final evaluation at 01:09. The final base results and concept diagnostics files did not yet exist. The last logged training record was step 12,950, approximately 101.6M processed tokens; this is a log checkpoint, not an asserted final optimizer step. The report's old “not yet run” heading is stale. Plumbing outputs under `runs/plumbing/` are not final experiment results.

The final training logs show broad code use, approximately 31 effective entries out of 32 in the aggregate. That is evidence of occupancy, not semantic usefulness. No v14 pass/fail claim is inferred here before its registered evaluation completes. Later results should be appended as a dated design amendment, never silently substituted into this evidence snapshot.

The papers were read in full in the preceding reviews, saved in `paper_reviews/next_concept_modeling_first_principles.md` and `paper_reviews/depth_memory_first_principles.md`. This design rechecks their relevant methods and ablations. The NCM release differs from its PDF in some configurations; this proposal defines its own settings explicitly rather than importing those configurations.

## 1. Step 1 — Define the problem before choosing modules

The model must predict the next token from the available prefix. Let X be that prefix, Y its future, L the information available through the local path, and U the concept state. A useful concept computation must preserve information about Y that is hard to extract from L alone and make that information usable by the decoder.

Three distinct requirements follow:

1. **Content:** a target must retain something relevant about Y. A high-entropy target can be pure noise.
2. **Predictability:** the retained information must be inferable from X. Random future input cannot become predictable because a latent changes rapidly.
3. **Use:** exploiting the prediction must improve held-out token or task performance at an acceptable cost. A better auxiliary loss alone does not establish this.

V14 supplies useful diagnostics for these requirements, but its default decoder reads U and never reads the NCP prediction. Its target projectors are trained by distributional regularization; a well-spread projection need not retain task-relevant content. Subtracting a local linear prediction from a global change removes one shortcut but may also leave noise or remove useful predictable information. These are design risks already acknowledged in v14, not claims that its pending results failed.

**V15's response:** make target content reconstructively grounded and fixed; predict it conditionally; provide a measured path from that prediction to the decoder. Keep an independent path for observed history because uncertain forecasts cannot replace facts already read.

There is no information-theoretic free gain. A forecast Z computed from X satisfies I(Y;Z | X)=0. Its value is a different computational organization, inductive bias, and supervision. Parameter-matched and auxiliary-only controls are therefore mandatory.

## 2. Step 2 — Separate the three axes

| Axis | Role in v15 | Dependency |
|---|---|---|
| Token time | Ordinary autoregressive text generation | A row predicts the next token using only its prefix |
| Representation detail | One coarse code for a future 16-token span, four residual codes for its four groups | Refinement is conditioned on a coarse code |
| Network depth | Recall earlier sublayer contributions inside the concept module | A sublayer reads memory written by earlier sublayers at that concept position |

“Next concept” in NCM means next *detail level*. In v14 it means a future *position*. V15 uses both indices explicitly: a coarse/fine level and a future-group horizon h∈{1,2,3,4}. Depth-memory index l is neither of these.

A shared coarse variable can coordinate correlated alternatives. For example, a prior over `(singular, is)` and `(plural, are)` should not independently mix `(singular, are)`. Conditional code prediction addresses this at the coarse/fine interface. It does not prove that all four future groups become independent, so v15 replans from realized tokens instead of rolling out a four-group plan as if it were certain.

## 3. Step 3 — Decide what survives from v14

The reference is the **implemented shrunk v14-tiny**, not the larger unimplemented settings in the root specification.

| Component | V15 decision | Reason |
|---|---|---|
| Vocabulary, tokenizer, d=128, four token heads, FFN=256 | Keep | Isolate the new representation and depth mechanisms |
| E1, Engram, E2 R=1, E3 with two layers, exits | Keep | No evidence from these papers justifies replacing the whole encoder |
| Two local and two global branches; L→G allowed, G→L forbidden | Keep | Preserve the interpretation of long-range information flow |
| Group stride k=4; source pools; learned source mixture | Keep | Maintain the release schedule and a comparable input to C |
| Concept block sequence: self-attention, local read, SwiGLU; four blocks | Keep operators, replace depth aggregation | Test retrieval across depth without adding loops |
| C's Delta-AttnRes aggregation | Replace by residual + grouped delta memory in V15-Δ | Direct test of the Depth Memory hypothesis against an existing depth mixer |
| Decoder: four blocks, fixed local K/V, historical concept read | Keep | Retain ordinary generation and access to observed history |
| EMA backbone twin, family residue maps, online target projectors, SIGReg, moving PQ | Remove from the v15 target path | Replace their coupled moving coordinate system with a separately trained frozen codec |
| Factorized four-segment NCP head | Replace | Predict a coarse code and conditional refinements, with explicit uncertainty |
| Forecast-to-decoder route | Add one gated read after the existing decoder stack | Make predicted codes operational without adding a full decoder pass |
| Correction bank and per-phase density-gated adapter growth | Not in default continual method | Their completed comparative evidence is unavailable; use a bounded replay baseline (§12) |

The new model is a **joint redesign**, not a one-variable v14 experiment. Factorial controls in §13 separate its components. A historical v14 comparison cannot attribute an improvement to either paper individually.

## 4. Step 4 — Fix causality, shapes, and release before writing losses

Use zero-based document-relative token positions. Model row t consumes x_t and predicts x_{t+1}. Group g contains x_{4g:4g+4}; its release row is e_g=4g+3. At row t, the newest released group is

`g(t) = floor((t+1)/4) - 1`.

If g(t)<0, the historical read uses its existing null path and the new plan read returns zero. For t=e_g,…,e_g+3, the forecast issued at e_g can guide the next four predictions. At t=e_g+4 the next completed group issues a new forecast. Example: u_0 first becomes legal at row 3 and can help predict token 4; it cannot affect row 2's prediction of token 3.

For a released group g, the **training-only target window** is

`Y_g = x[e_g+1 : e_g+17]`, restricted to the same document.

The student input is only `x[:e_g+1]`. The target encoder may inspect all of Y_g because it supplies labels, not forward inputs. No target-derived code, vector, length, mask, or validity flag enters the plan bank consumed by the language model.

Packed documents retain v14's alignment contract: starts are aligned to groups of four and padding has a separate invalid identity. Attention, convolutions, GDN states, pools, historical reads and forecast selection are masked/reset by document identity. A group is released only when its four input slots are valid real tokens of that document. Partial terminal groups need not produce a subsequent forecast.

For targets near an end: EOS is a real token; right padding after EOS is masked. A root target is valid if Y_g has at least one real token. Fine target h is valid if its four-slot portion has at least one real token. Empty fine slots produce no label loss. The codec pools over valid positions only, including a masked mean for a partial group. It never pools another document. **Inference always predicts every horizon and never receives these future-validity masks.** Generation stops using emitted EOS, not a true future length.

At tiny scale, with microbatch B=2 and T=1,024:

| Tensor | Shape |
|---|---|
| L_enc, G_enc, decoder state | B×T×128 |
| p, U | B×G×128, G=T/4 |
| Target window | B×G×16, labels only |
| Codec group vectors | B×G×4×64 |
| Coarse logits | B×G×32 |
| Fine logits at one supplied root | B×G×4×32 |
| Horizon-1 joint table | B×G×32×32 |
| Selected plan vectors | B×G×4×128 |
| Concept depth memory | B×G×4×32×4 |

Depth-memory states are indexed by concept position and group of channels. They are never carried from concept g to g+1. Temporal information still travels through C's masked self-attention and the inherited encoder.

Invalid input groups have zeroed concept outputs and skip depth-memory writes. Attention implementations must provide a finite null path rather than softmax over an entirely masked row. Entirely invalid target windows are excluded from codec execution/loss; their label placeholders cannot introduce NaNs into a masked reduction.

## 5. Step 5 — Train a small, ordered, frozen target codec

### 5.1 What it must preserve

The target is the observed future span itself, in order. It is not a family-conditioned residual of a changing student. Preserving future token information can require distant context to predict even when the target encoder only sees the future span. Whether that happens on this dataset is a measured property, not something declared by calling a target “global.”

The codec receives **only Y**, with no student state, prefix, family ID, task label, or privileged simulator state. It has its own embedding table, independent of the student's evolving embedding. It is bidirectional inside this target window; such bidirectionality is legal only on the label side.

Configuration: vocabulary 2,048; width 64; two PreLN bidirectional encoder blocks; four heads; SwiGLU width 128; learned 16-position encoder embeddings. Pool each four-token portion after encoding, project with a bias-free 64×64 map, and apply learned RMSNorm to obtain z_h. Mask all invalid slots. Use a separate one-block bidirectional decoder, its own 16-position embeddings, final RMSNorm, and an output head tied to the codec token embedding. All attention/FFN matrices are bias-free; norm gains are learned.

### 5.2 Two levels of residual codes

There are **two codebooks**, E0 and E1, each 32×64. E1 is shared across the four fine positions; position is supplied to the codec decoder and predictor separately.

```
z_bar = masked_mean(z_1, ..., z_4)
c*    = nearest(E0, z_bar)
r*_h  = nearest(E1, z_h - E0[c*])
z_hat_h = E0[c*] + E1[r*_h]
```

The codec decoder repeats z_hat_h into the four token slots of group h, adds its position embeddings, and reconstructs the 16-token window. With coarse-only input it repeats E0[c*] at every valid slot. **There is no skip connection from encoder features or original token embeddings into the reconstruction decoder.** This is essential: otherwise reconstruction can succeed while codes carry nothing.

The full hierarchy has at most `5 log2(32) = 25` fixed-width code bits for 16 tokens. It cannot be a lossless codec for arbitrary spans. Exact reconstruction is not the goal. This bottleneck is a deliberate, unvalidated choice for useful predictable structure; a 64-entry capacity arm is separate and changes the registered model.

### 5.3 Codec objective and update order

On each training span choose coarse-only reconstruction with probability 0.25 and full reconstruction with probability 0.75. The loss is mean cross-entropy over its valid tokens plus commitment penalties with coefficient 0.25. Average across valid spans under the original family mixture. No Gaussian-shape or uniform-usage objective is used.

For commitment, penalize `||z_bar - sg(E0[c*])||²` at the coarse level; for full reconstructions also penalize the masked mean of `||z_h - sg(E0[c*]) - sg(E1[r*_h])||²`. MSE is a mean over coordinates. Root-only decoding uses a straight-through root (`z_bar + sg(E0[c*]-z_bar)`); full decoding uses `z_h + sg(z_hat_h-z_h)`. Codebook vectors are updated by EMA cluster means, not gradient. The encoder and reconstruction decoder receive reconstruction gradients.

Assignments and losses use the **pre-update** codebook. After the optimizer step, update codebook counts/sums with decay 0.99: E0 uses saved detached z_bar assigned to each root, while E1 uses saved detached residuals `z_h - E0_pre[c*]` assigned to each fine entry. Do not recompute residuals with the newly updated E0. Update both books from valid assignments on every codec batch, including coarse-only reconstruction batches. Initialize root centroids from distinct valid training z_bar vectors, then initialize fine centroids from distinct residual vectors under those roots; initialize counts at uniform expected usage. Check dead entries every 100 codec updates; an entry below 1% of uniform EMA usage on two consecutive checks is reseeded from a distinct high-error training vector in its own root or residual space. Every reset is logged. Freeze and hash encoder, decoder, embeddings, norms, codebooks and all preprocessing together at the end of codec training. No subsequent reset or normalization update is permitted.

This implementation contract deliberately resolves the NCM paper/release discrepancy about which levels receive VQ penalties: the formula above is the v15 rule. EMA update order and frozen state are checkpointed explicitly.

### 5.4 Readiness checks and limitations

Before the student uses the codec, compare held-out reconstruction with a phase-position-only unigram predictor fitted on training data. Proposed engineering gates: coarse-only CE improves by at least 0.05 nats/token in the equal-family mean; adding fine codes improves coarse-only CE by at least 0.05; both changes are positive in at least three of five families. These thresholds are design choices, not literature constants. Record loss and token accuracy on entities, numbers, delimiters and task answers separately.

Also report root/fine entropy, within-document diversity, exact replay of code assignments, and reconstruction under code swaps. Low utilization alone does not fail a genuinely low-entropy family. If the reconstruction gates fail within the budget, report the candidate as not ready; do not silently grow the codebook, extend training, or call a constant code a concept.

A successful codec still may emphasize unpredictable lexical detail. Predictability and decoder-use tests remain required. Its frozen state removes target drift during student training, but its content can still be poor.

## 6. Step 6 — Preserve computation while making earlier contributions retrievable

The four C blocks retain their three sublayers, twelve in total. Replace `Stack` aggregation **only for C**. Decoder depth aggregation and encoder role mixing remain unchanged in the primary candidate.

For each concept position independently, keep residual x∈R^128 and four memories M_j∈R^(32×4). Use G_m=4 groups, address dimension d_k=4, and hidden address width H_a=16. Start x=p_g and M=0; write p_g once as the initial value. All memory operations use FP32 on MPS.

At each sublayer l:

```
q, read_gate = ReadAddress_l(x)
r = concat_j[ gamma_lj * read_gate_j * M_j q_j ]
delta = F_l(PreNorm_l(x + r))
x_new = x + delta
k, beta = WriteAddress_l(x_new)
M_j = M_j + beta_j * (delta_j - M_j k_j) k_j^T
x = x_new
```

After the twelfth sublayer, perform one final read and return `u_g=x+r_out`. Do not permanently add each read r into the running residual before the sublayer update; the permanent update is delta. Store delta rather than the repeatedly accumulated x.

Each address network is learned RMSNorm(128) → affine 128→16 → SiLU, followed by separate affine outputs for four length-4 addresses and four gates. Normalize each address to L2 norm at most one using `v / max(||v||, 1e-6)`. Read gates are `2 sigmoid(a)` (initially 1); write beta is `sigmoid(b)` (initially 0.5). Read gain is `gamma=0.5 tanh(s)` per group, initialized to 0.1. Queries and keys are dynamic; all address sites have separate weights. The initial embedding write has its own write address; the final read has its own read address. There are thirteen read and thirteen write sites.

The homogeneous part of a write is `M(I-beta k k^T)`. For `||k||≤1` and `0≤beta≤1`, its spectral norm is at most one. This bounds amplification by that linear state transition. It **does not** bound the full neural network, cumulative writes, derivatives through address networks, or memory norms. Log those and clip global gradients at 1.0.

### 6.1 Why delta is the default

Depth Memory Table 8 reports losses 2.5987 (RLS), 2.6028 (delta), and 2.6554 (PreLN) in its 400M ablation. Preserving the residual path matters far more there than the solver-specific increment. V15 chooses the simpler state first, at one-quarter token rate, and measures against the existing AttnRes—not merely against PreLN. The proposed gate range and grouping are v15 settings, not a claim to reproduce the paper's delta variant.

At this shallow depth, a matrix memory can use **more** activation storage than the existing explicit depth cache. Fixed size across depth is not a promise of lower memory at twelve sublayers. No custom reversible backward is part of the initial MPS implementation; ordinary autograd is the reference.

### 6.2 RLS comparison, fully specified

Keep the same reads, stored deltas and address networks. Treat beta as an observation weight omega; set lambda=1 and alpha=1. For each group initialize P=I and use weighted key/value `kw=sqrt(omega) k`, `vw=sqrt(omega) delta`:

```
gain = P kw / (1 + kw^T P kw)
error = vw - M kw
M_new = M + error gain^T
P_new = P - gain kw^T P
```

P is a second per-position state, not a learned parameter or a cross-token cache. Accumulate in FP32; verify against a CPU FP64 ridge solve, test symmetry/positive definiteness and gradient parity. For this fixed-lambda comparison, RLS exactly solves a ridge regression over produced observations in exact arithmetic. A learned forgetting factor, windowed writes, or custom unwinding is a later independently validated optimization, not silently added to this experiment.

## 7. Step 7 — Predict coherent alternatives from the prefix

Let `v_g=RMSNorm(u_g)` with a learned norm. The coarse prior is an affine 128→128, GELU, affine 128→32 followed by softmax: q0(c|u_g).

The fine predictor is shared across horizons and roots:

```
a_h = W_u v_g + b_u + horizon[h]       # 64 dimensions
fine_logits(h,c) = W_f GELU(a_h + root_address[c]) + b_f
q_h(r | u_g,c) = softmax(fine_logits(h,c))
```

`root_address` is a learned 32×64 table owned by the prior, distinct from the frozen E0. Horizon embeddings are 4×64. This conditional head is small and avoids a full transformer pass for every root alternative.

The clean auxiliary likelihood is

`q(c,r_1:4 | u) = q0(c|u) product_h q_h(r_h|u,c)`.

The four refinements remain independent conditional on u,c. The model can represent shared coarse alternatives but not every possible joint dependency between future groups. It consumes only horizon 1 as an immediate forecast and observes actual tokens before issuing another, limiting reliance on that approximation.

### 7.1 Exact loss convention

For a valid anchor, use root CE and the weighted mean of valid fine CEs with w=(0.4,0.3,0.2,0.1), renormalized over its valid horizons. Then

`L_prior_clean = mean_valid_anchors[0.5 CE_root + 0.5 CE_fine_mean]`.

This is a balanced auxiliary objective, not the unweighted joint negative log-likelihood or the NCM ELBO. Its coefficient is not numerically comparable to v14's sum over four PQ segments. Report per-component unweighted CE as well.

### 7.2 Incorrect coarse predictions

There is a mismatch if the fine predictor only sees true roots during training and predicted roots during use. V15 adds a small robustness term without sending true roots to the decoder.

For each anchor draw one alternative root c' from `sg(q0(.|u))`. Recompute that root's residual labels using the frozen codec features:

`r_h(c') = nearest(E1, sg(z_h) - E0[c'])`.

Apply fine CE to these labels under c', weighted across valid horizons as above. Define `L_prior = L_prior_clean + 0.1 L_fine_alt`. Sampling and labels are detached. The coefficient 0.1 is proposed, not a measured optimum. A zero-weight control is mandatory in calibration.

This differs intentionally from NCM's corruption recipe, which keeps clean fine labels: here residual code meaning depends explicitly on the root, so changing a root while retaining a residual label would no longer describe the same target decomposition. Alternative-root residuals may lie outside the fine codebook's well-fitted region; log their distortion separately. Do not update the frozen codebook to make these labels easier. This new robustness term is a hypothesis and may be removed in a named ablation if it fails.

### 7.3 Baselines and uncertainty

Evaluate root CE, fine CE under true and predicted roots, and joint horizon-1 CE. Compare with training-fitted marginal distributions, a smoothed previous-realized-window predictor, a capacity-matched predictor on the recent local view, and one on p_g. Exact baseline definitions and data splits must be recorded before scoring.

The previous realized window is the **past** 16 tokens ending at e_g, encoded with the same frozen codec. Never use Y_(g-1)'s training label as a copy baseline: that future window overlaps Y_g and is not fully observed at e_g. This fixes a subtle lookahead shortcut introduced by overlapping future targets.

Target windows overlap in training, so probe splits and uncertainty estimates group all windows from one document/semantic instance. Do not randomly split adjacent windows. Code predictability on random newly sampled prompt content is intrinsically limited; report prompt, execution and answer regions separately without filtering the training distribution to manufacture success.

## 8. Step 8 — Let the decoder consult forecasts it can actually obtain

For horizon 1, evaluate q1 for all 32 roots, in parallel or root-chunked, and form the exact 32×32 joint table

`J_g[c,r] = q0(c|u_g) q1(r|u_g,c)`.

Select the four largest **joint pairs**, with lexicographic tie breaking, not independent top roots and top refinements. Let their unnormalized probabilities be pi_i, retained mass m=sum pi_i, and tail mass 1-m. Use frozen paired vectors

`b_i = concat(E0[c_i]/s0, E1[r_i]/s1)`,

where s0 and s1 are scalar RMS values over all entries/coordinates of each frozen codebook, floored at 1e-4 and frozen with it. Use a common scale per book, **not per-code normalization**, which could erase distinctions carried by codeword magnitude. Apply two learned 64-dimensional gains at the bridge input, one for each half. Keeping the halves separate avoids losing identity through cancellation in an early sum. Selected pairs, codebook values and probabilities are stopped with respect to gradients into the prior. The bridge still learns its projections and gates.

Append one learned null key/value with base mass `max(1-m,1e-6)`. The four candidate entries keep masses pi_i; do not renormalize away the tail. This is attention metadata, not an exact marginal text likelihood. The null is an unmodeled-alternatives channel; its value does not claim to be the expectation of all omitted plans.

At row t with a legal released forecast, take the output h_t of the existing four-layer decoder. A single four-head plan attention reads the four vectors plus null, with log base mass added to its attention scores. Queries use learned RMSNorm(h_t) and a learned embedding for offset `t-e_g ∈ {0,1,2,3}`; keys/queries are L2 normalized per head, with one learned positive temperature, initialized at 1. Q,K,V,O are bias-free 128×128 matrices. The null has one 128-dimensional key and value. No token position from a future target is an input.

```
a_t = PlanAttention(h_t, predicted_bank_g, log_masses, offset)
gate_t = sigmoid(w^T [RMSNorm(h_t); m; H(q0)/log(32); onehot(offset)] + b)
h'_t = h_t + ramp * gate_t * a_t
logits_t = final_norm(h'_t) E_student^T / sqrt(128)
```

Initialize gate weights to zero and bias to logit(0.1). The ramp is specified in §9. There is no teacher-code injection at any phase of decoder training. Candidate selection is a discrete forward operation; decoder loss does not backpropagate through it or change the code dictionary.

Historical U reads remain. A forecast is uncertain and cannot replace observed long-range facts. This means a bypass exists and plan use is not guaranteed. V15 earns the claim that plans help only by outperforming its auxiliary-only control and passing interventions with U held fixed (§13).

A bank of alternatives is still combined by soft attention; it can blend incompatible candidates. Keeping separate entries permits selection but does not guarantee logically consistent mode commitment. Exact latent marginalization or sampling one persistent block plan would require another model and cost analysis. They are not claimed here.

## 9. Step 9 — Separate representation learning, prediction, and use

The primary comparable tiny budget is **11 device-hours total per base trajectory**, including codec training, codec readiness evaluation, student training and blocking in-training checks. Final registered evaluation and continual learning have a separate declared budget. This does not authorize launching a run in the current task.

**Stage A, first 10% of total budget:** train and validate the codec, at most 1.1 hours. Use only the permitted training stream and a disjoint calibration split. Pick the lowest full reconstruction CE checkpoint among those passing the coarse/full readiness checks, without looking at test results. Freeze it. If none passes, stop this candidate and report the failure; a revised codec is a newly versioned design.

The codec uses the existing training-family proportions (text 0.10; each of the four reasoning families 0.225), and traverses the *training* curriculum from 0 to 1 within Stage A. It never reads evaluation grammars/instances. The student traverses that curriculum from 0 to 1 within Stage B. Report both streams' example IDs/token counts, including repeated exposures. This is additional permitted training exposure whose compute is charged, not pretrained external knowledge. Initialization buffers are bounded training buffers and count toward Stage A.

**Stage B, remaining budget:** train the student from scratch. Let s∈[0,1] be progress through its allotted 9.9 hours. For s<0.1, train only the inherited next-token/task objective, with the chosen C depth mechanism. Ramp lambda from 0 to 0.05 over s=0.1…0.2, then keep it at 0.05. Enable the plan read with a linear ramp from 0 to 1 over s=0.3…0.4. The schedule is fixed; no oracle plan curriculum is used.

`L_student = L_NTP + lambda(s) L_prior`.

L_NTP preserves v14's actual scoring weights: all text tokens; the registered reasoning answer-token normalization. The plan objectives use valid groups across the permitted stream. Codec reconstruction used to define targets is not an extra student loss. There is no student SIGReg, learned residualization map, target EMA or moving normalization statistic.

AdamW uses the implemented v14 recipe: learning rate 1e-3 with warmup/cosine schedule, betas=(0.9,0.95), weight decay 0.03 for applicable matrices, global clip 1.0, and the Engram table's existing treatment. New biases, norm gains, read gains and horizon embeddings receive no weight decay. The codec uses AdamW 3e-4, the same betas, matrix weight decay 0.01, and clip 1.0; codebooks use EMA, not an optimizer.

Use microbatch B=2, T=1,024 with four gradient accumulation steps to retain eight rows per update. Accumulate weighted NTP sums and their denominators before division; similarly weight auxiliary anchor sums by valid counts. Do not average four microbatch means with different numbers of valid targets. Any fixed-step secondary comparison uses token-based versions of the same stage fractions, locked before that comparison.

The global student gradient norm is logged in the ordinary backward pass. Expensive per-objective gradient audits run on saved checkpoints in a separate process after releasing the training device, and their time is charged if they block the budget. No repeated retained-graph eager diagnostic is inserted into the compiled training loop. V14's observed allocator growth around such reads motivates this systems choice; it does not prove every eager diagnostic leaks.

### 9.1 Gradient contract

| Signal | Online encoder/pools | C and its depth memory | Prior heads | Plan bridge/decoder | Frozen codec |
|---|---|---|---|---|---|
| NTP | Yes through existing legal routes | Yes through historical U | No through detached plans | Yes | No |
| Prior clean/alternative | Yes through p and U | Yes | Yes | No | No |
| Codec reconstruction, Stage A only | No student exists yet | No | No | Codec decoder only | Codec is trainable only in this stage |

“Frozen codec” includes its token embeddings, encoder, all position and norm parameters, decoder and codebooks. Shared weights with the student are forbidden. Dropout is disabled for label extraction, or zero throughout the codec configuration. Labels are deterministic for a fixed window and codec hash.

## 10. Step 10 — Specify actual generation

1. Process the current observed/generated token through the causal encoder and local decoder caches.
2. When a complete four-token group is available, compute p_g and the four-block C step, including its within-depth memory. Each new concept position starts from a fresh depth-memory state; previous positions remain available only through legal temporal caches.
3. Predict q0 and all root-conditioned horizon-1 refinements; keep the four joint pairs and their masses. Cache this small bank for the next four output predictions.
4. Decode the next token using local evidence, released historical concepts and the forecast bank. Use emitted tokens to advance the model. No sampled code is substituted for an observation in the encoder.
5. Reset temporal caches, forecast cache and document-relative counters at a document boundary. Stop at emitted EOS or the preregistered task cap.

The output distribution is an ordinary normalized token softmax. V15 does **not** inherit NCM's parallel text-generation algorithm or its 10–31× speedups. Its only new sequential forecast work is two small head stages per completed group; conditional refinements can be batched over roots. Whether removing the EMA target offsets new costs during training is an empirical timing question.

With both historical concepts and plan banks replaced by fixed nulls, the decoder's maximum non-concept look-back remains the inherited 66 tokens: its K/V are always L_enc, and no new layer mixes decoder states across time. With either route enabled, long-range information must originate through C. Depth-memory reads are position-wise and cannot independently widen L's receptive field. These statements require implementation reach and suffix-invariance tests before any run.

For this reach test, disabling the plan branch means returning exactly zero from the entire bridge, including its probability metadata and gate. Zeroing candidate vectors while leaving prefix-dependent probabilities or entropy in a learned null gate would leave a hidden concept route and is not a valid local-only test.

## 11. Step 11 — Account for parameters, memory, and time

### 11.1 Tiny parameter ledger

These are formula counts under the exact bias/norm settings above, not counts from an implemented v15 model. Verify with an instantiated model before acceptance.

| Change from implemented v14-tiny | Trainable scalar count |
|---|---:|
| Starting model | 6,807,755 |
| Remove NCP norm, horizon embeddings, old NCP MLP, two target projectors | −91,136 |
| Remove C's twelve depth-attention query vectors | −1,536 |
| Add thirteen read + thirteen write address sites | +65,884 |
| Add coarse/fine prior, including its norm | +33,408 |
| Add forecast bridge | +66,696 |
| **V15-Δ student** | **6,881,071** |

Address-site accounting: a write has `d + H_a(d+1) + (G_m d_k)(H_a+1) + G_m(H_a+1) = 2,532` scalars. A read adds four learned gains, giving 2,536. Thirteen of each gives 65,884. Existing twelve sublayer norms remain and are not counted twice.

Prior accounting: coarse head plus norm 20,768; fine context projection, horizon embeddings, root-address table and output layer 12,640. Bridge accounting: four 128×128 projections 65,536; query norm 128; two code-half gains 128; four offset embeddings 512; null key/value 256; temperature 1; scalar gate with 134 inputs and bias 135.

The standalone codec has **260,608 trainable scalars during Stage A**, plus **4,096 EMA codebook scalars**. It is frozen during student training. Student plus the full frozen codec stores approximately **7,145,775 model scalars**, excluding optimizer states and nonparameter buffers. Deployment needs the student and the two codebooks (approximately 6,885,167 scalars); it need not run or store the codec encoder/decoder unless retrospective encoding is explicitly requested. Preserve the complete codec artifact separately to reproduce labels and train later stages.

V15-RLS has the same learned parameter count as V15-Δ in the specified fixed-alpha/fixed-lambda arm. Its extra inverse covariances are activation state, not weights.

### 11.2 Working-state accounting

For B=2, T=1,024, G=256, d=128, G_m=4, d_k=4, FP32:

| Object | Element count | Size |
|---|---:|---:|
| One C memory M | B G d d_k = 262,144 | 1 MiB |
| One RLS inverse covariance P, optional | B G G_m d_k² = 32,768 | 0.125 MiB |
| Thirteen saved M states, naive reference | 3,407,872 | 13 MiB |
| Horizon-1 joint table | B G 32² = 524,288 | 2 MiB |
| Fine-head hidden vectors for all roots at h=1 | B G 32×64 = 1,048,576 | 4 MiB |
| Four plan vectors per anchor | B G 4×128 = 262,144 | 1 MiB |

These are component sizes, **not peak training-memory predictions**. Gradients, address activations, masks, attention, optimizer states, compiler caches, allocator fragmentation and temporary tensors are additional. The old explicit C source cache at this shape can be smaller than thirteen M matrices. Removing a full EMA encoder and its target machinery may offset costs, but must be timed and measured.

Target windows overlap: extracting one 16-token target per four-token anchor processes up to four times the original token positions through the small frozen codec. Include this cost. Cache fixed-row labels by `(codec hash, tokenizer hash, document ID, span start, actual token hash)`. The alternative-root term also needs frozen z_h; do not assume cached primary code IDs suffice. Bound cache size and report disk/RAM use. New synthetic examples still require encoding. No cache is keyed only by a sequence index whose underlying data can change.

### 11.3 M2 launch gates for a future implementation

Use a single heavy device job. Never run v15 smokes concurrently with v14 evaluation or continual learning. The first implementation must complete:

1. CPU correctness checks and a small deterministic backward pass.
2. MPS forward/backward checks beyond every scheduled event, including codec freeze and plan activation; fp64 conversions occur on CPU only.
3. A warm steady-state timing run and at least 1,500 student updates covering snapshot/telemetry events. Inspect live tensor memory and driver allocation separately.
4. A measured training peak target ≤8 GiB driver allocation with no sustained pressure-induced swap growth. Persistent >9 GiB or rising post-cleanup live allocation fails the launch gate. These are headroom policies for this 16 GB machine, not universal architectural constants.

If B=2 fails, test B=1 with accumulation eight under the same global loss normalization and report the throughput change. Do not quietly shorten the context or remove difficult examples. Checkpointing and bounded cache cleanup may change execution, but forward/gradient parity is required. No background evaluator should repeatedly load full checkpoints while a run is near its memory limit.

## 12. Step 12 — Keep continual learning a separate claim

The depth matrix is a temporary computational state. It is reset for each new concept position and cannot remember a previous session. Stable codec IDs also do not by themselves prevent forgetting: the mapping from inputs to those IDs and the token decoder can still change.

The default continual comparison uses **one bounded rank-4 LoRA adapter plus 20% uniform replay from a 1,000-row reservoir**, keeping the base weights frozen. Reuse Dataset_v13's exact phase stream, exposure, scoring and reference panels. Adapt the C self-attention/local-read/FFN matrices and the existing decoder projection/FFN matrices; optionally making other matrices trainable is a separately named arm. The forecast bridge, prior, depth-address networks, embeddings and codec stay frozen. Gradients can pass through the frozen prior into adapted C.

Use incoming/replay NTP plus the same 0.05 prior auxiliary on their fixed codec targets. Compare against exactly the same LoRA/replay method with that auxiliary off, and against sequential LoRA without replay. Freeze the entire coordinate-system artifact for all phases. Do not fit a new codec or reset codewords when a new family arrives.

This conservative choice may restrict plasticity. Record acquisition, retention and code distortion on new primitives. If the fixed target vocabulary or frozen head becomes a bottleneck, report the deficit and test a named trainable-prior or versioned-codec arm. A new codec version invalidates cached labels and code-indexed banks; it requires explicit re-encoding/migration, not reuse of old integer IDs.

The correction bank, wake/sleep consolidation, code gates and adapters added per phase remain separate experiments. Their cost and efficacy cannot be justified by the Depth Memory result. For persistent fact updates, keep Dataset_v13's exact-dictionary baseline and separate revised facts from stable-skill forgetting.

A continual advantage requires the existing dataset's criteria: within 3 points of replay's acquisition and at least 5 points better retained old-task accuracy, with paired uncertainty and related-skill checks. This architecture has not yet met them.

## 13. Step 13 — Design experiments that can reject v15

### 13.1 Required controls

| ID | Target/forecast | C depth aggregation | Purpose |
|---|---|---|---|
| R14 | Implemented v14 recipe | Existing AttnRes | Historical reference; rerun for a controlled comparison when possible |
| B | No old/new auxiliary, no forecast bridge | Existing AttnRes | Clean NTP shell with the same inherited backbone |
| T | Frozen codec prior auxiliary, bridge off | Existing AttnRes | Does fixed hierarchical prediction help by itself? |
| TP | Same auxiliary plus predicted-plan read | Existing AttnRes | Does operational use add benefit beyond T? |
| D | No codec auxiliary or forecast read | Proposed residual + delta memory | Does memory help independently? |
| TPD | Same targets and forecast as TP | Proposed residual + delta memory | **V15-Δ primary candidate** |
| TPR | Same targets and forecast as TP | Residual + RLS memory | Is the exact solver worth its cost? |

`B, TP, D, TPD` form the primary two-factor comparison. `T` separates prediction supervision from forecast use. TPR is a later solver comparison. Also include an ordinary parameter-matched causal transformer and, for the plan read, a generic prefix-MLP feature bridge with comparable added parameters and no codec objective. The latter controls for adding another nonlinear transformation of the same prefix.

Variants using the codec may share one trained codec for low-noise scientific comparisons, but publish both amortized and **full per-system cost**. Charge the entire codec preparation to each codec-based system in an 11-hour end-to-end claim. B and D may use their full allowance for student training. An equal-student-token comparison that ignores codec cost is a secondary sample-efficiency analysis.

Share the codec only within a paired-seed comparison. A second *full-pipeline* replication retrains the codec as well as the student with its paired seed; multiple students on one fixed codec measure student variance conditional on that codec and must be labeled accordingly.

A concrete optional ordinary-transformer size control is vocabulary 2,048, d=192, fourteen PreLN blocks, six heads, SwiGLU width 544, bias-free attention/FFN matrices, RoPE, learned RMSNorms and a tied token head: **6,849,984 parameters**, about 0.45% below V15-Δ. It uses full document-causal attention and the same data, scoring, sequence length and total budget. Use an already registered baseline unchanged if one is in progress; label this additional control separately. Matching parameter count does not match active FLOPs or time, so report all three and give each baseline an equal calibration opportunity.

Do not claim to have isolated both mechanisms after training only R14 and TPD. A complete factorial and replication require multiple runs; they cannot honestly fit in one 11-hour trajectory or an arbitrary single day.

### 13.2 Primary outcome and promotion rule

Use Dataset_v13's frozen Reasoning scalar and ladders, exact generated task success, valid-output rate, and text NLL. Generated greedy single-output evaluation is primary. Teacher-forced answer accuracy is reported separately and cannot be relabeled as free-running exact success. Preserve all generator caps, semantic splits and task weights.

Proposed tiny promotion thresholds, to lock before new comparison results:

- At equal total device time, TPD improves Reasoning by at least **0.02 absolute** over the strongest completed matched control, with a paired 95% evaluation interval above zero.
- Text NLL is no more than **0.02 nats/token worse**; no reasoning family drops by more than **0.03 absolute**. Report both point estimates and intervals rather than hiding tradeoffs behind one scalar.
- Pass the memory/causality gates and publish training throughput, total examples, codec time, prefill, decode and first-token latency.
- Repeat the primary pair with at least two paired training seeds before calling it repeatable; challenge/generalization claims require three, as Dataset_v13 specifies. Bootstrap intervals over evaluation items do not estimate training-seed variation.

These are proposed practical margins, not guarantees or optimized thresholds. If no control is competent on the registered working rungs, first report failed competence; mechanism interpretations remain inconclusive. If only a matched-token result wins, call it sample efficiency, not end-to-end efficiency.

### 13.3 Test each mechanism separately

**Codec content:** coarse/full reconstruction gates, order-sensitive controls, entity/number fidelity, code stability on fixed windows. Swap coarse codes across matched spans and quantify both intended and unintended changes. High utilization is insufficient.

**Predictability:** on fixed targets, the U predictor should reduce joint horizon-1 CE relative to marginal and the legally observed-window baseline. For a claim of non-local prediction, require at least 0.05 nats/anchor reduction against a capacity-matched local predictor in at least two reasoning families, with document-grouped intervals above zero. Report root/fine losses separately; the threshold is exploratory until preregistered. A global U-versus-local difference does not isolate C's computation: also compare with p_g.

**Operational use:** compare TP with T at equal budget. On the same trained TP/TPD checkpoint, hold historical U and local states fixed and replace only the forecast bank with (a) its null branch, (b) a same-family, same-position, similar-entropy donor bank, and (c) a matched marginal bank. Restore the original bank as a recovery control. A proposed supporting signal is ≥0.01 nats/scored-token degradation or ≥1 point exact-success loss under the donor swap on two reasoning families, with intervals above zero. This alone does not prove semantic abstraction: donor states can still be off distribution, and the bridge might use a prefix fingerprint. The trained TP-versus-T and generic-bridge comparisons remain essential.

**Oracle ceiling, diagnostic only:** on held-out examples substitute actual target codes, clearly labeling the read as privileged and excluding it from primary scores. If an oracle helps but predictions do not, prediction/exposure is the bottleneck. If even an oracle provides no benefit, either the codec/bridge is poor or the continuous route already supplies everything needed. Never train the primary decoder on this oracle input or use its score to select test hyperparameters.

**Depth memory:** compare TP/TPD/TPR; ablate read gates at inference, preserve the residual path, measure memory read magnitude and reconstruction of earlier deltas. Gate magnitude alone is not utility. A simpler PreLN C is an additional reference, since LDM's published PreLN advantage does not establish an advantage over v14's existing AttnRes.

**Uncertainty:** report joint NLL, Brier score, retained top-four mass, true-pair recall@4, calibration by confidence bins, null attention and task performance by mass bin. Do not interpret root entropy as a calibrated probability of correctness. If top-four mass is persistently small, the defined model appropriately exposes uncertainty; enlarging K is a new costed arm, not a post-hoc repair.

**Counterfactual robustness:** compare zero and 0.1 alternative-root loss. Report reconstruction distortion of alternative-root code pairs and fine CE under predicted roots. A gain must survive final task and equal-time checks; reducing that auxiliary loss alone is insufficient.

## 14. Step 14 — Make telemetry explain the mechanism

Retain v14's per-family training and system observability. Add a small fixed-schema event stream with model/codec/dataset hashes, phase and immutable configuration. Separate current minibatch observations from fixed held-out evaluation. Pending evaluation must show “pending,” never zero.

| View | Required measurements | Interpretation |
|---|---|---|
| Overall progress | Codec vs student stage, elapsed device time, tokens, effective batch, skipped updates | Whether a quality comparison has the same budget |
| Target quality | Coarse/full CE, distortion, code entropy by family, label hash/replay checks | What the fixed vocabulary preserves |
| Forecasts | Root/fine/joint CE, observed-window and marginal baselines, per-horizon masks/counts | Whether the future is being predicted rather than a shortcut |
| Forecast use | Top-four mass, null attention, bridge gate/read norm, fixed intervention results | Whether the decoder benefits from the prediction |
| Depth computation | Per-site read/write norms, gate distributions, M norms, RLS P diagnostics when used | Whether retrieval is stable and active |
| Real tasks | Generated exact success, text NLL, ID/challenge ladders, per-family uncertainty | Whether the architecture actually improves behavior |
| Cost | Codec/label-extraction/student/diagnostic time, live tensors, driver allocation, RSS, swap trend | Whether a gain survives the machine's constraints |
| Continual | Full A[i,j] matrix, acquisition/retention, adapter/replay size, codec-era ID | Whether stable coordinates translate into retention |

Log aggregate floats detached from autograd, not tensors or graphs. Do not expose large intermediate arrays through the web dashboard. Fixed diagnostic sets use document/semantic IDs; window-level counts are not treated as independent examples. The telemetry system may remain local/LAN as in v14; this design task does not alter its running service.

## 15. Step 15 — Implementation gates and scale discipline

The implementation order is deliberate:

1. Freeze the input manifests and baseline configuration; implement codec extraction and boundary tests first.
2. Train/validate the codec under Stage A's cap, then prove label immutability.
3. Implement prior-only T on the unchanged AttnRes shell. Check all losses and denominators on tiny hand-computed examples.
4. Add the predicted-plan bridge; verify that arbitrary changes to targets do not change logits when student prefix and frozen model state are fixed.
5. Implement C's residual + delta memory; verify its gradients and checkpointed/full execution equivalence before introducing compilation.
6. Run CPU and MPS integrity checks, then the bounded hardware gate. Run registered comparisons only after those pass.
7. Add RLS only as the named comparison. Validate its reference and any later optimized backward independently.
8. Complete free-running final evaluation before continual learning and preserve all checkpoints/results, including failed candidates.

Mandatory implementation assertions:

- Suffix changes after t cannot affect row t logits; changing another packed document cannot affect this document.
- g is released at e_g, never e_g−1; target labels and future-validity masks have no forward dependency into the student.
- Turning off both historical and plan reads restores the inherited exact local reach; finite-difference tests check the boundary and one token beyond it.
- Full-sequence teacher-forced execution equals incremental cached execution on the same realized prefix, including incomplete groups and document resets.
- Frozen codec assignments are bitwise repeatable on the same device and unchanged by student optimizer steps. Cross-device ties use deterministic index tie-breaking and are reported if numerical boundaries differ.
- The joint root/fine table sums to one; the top-four plus tail masses sum to one before attention flooring; independent marginal pairing is not used.
- NTP gives no gradients to the detached prior/codec branch; prior loss gives no gradients to the bridge or token decoder; existing indirect encoder routes remain legal.
- Delta/RLS state is separate for every concept position; no hidden state survives a new position or document through this mechanism.
- RLS agrees with a direct ridge solve on small FP64 cases; its denominator is positive; checkpoint/unwind implementations, if later introduced, pass forward and gradient parity.
- Resume restores data order, optimizer/RNG state, stage counters, codec hash, training budget, and codebook freeze status. No checkpoint loads a v14 moving-code artifact as a v15 codec.

The first scale tier is a controlled larger dense model, after a tiny benefit repeats. Keep the two mechanisms separately ablatable. Revisit code capacity, forecast span and memory address dimension with measured quality-versus-cost curves rather than scaling them blindly with width. Existing full token/global and concept attention still has quadratic components; **this proposal makes no million-token or trillion-parameter scalability claim**. Compressed temporal attention, MoE, parallel decoding and speculative verification require separate specifications. The papers do not establish these properties for the proposed composition.

## 16. First-principles self-review, step by step

This is the author's adversarial self-review, not an independent external review. Each row records a potential flaw examined while drafting and the resulting specification decision.

| Review question | Finding and correction | Remaining falsifier |
|---|---|---|
| R1. Is “concept” being defined by high entropy? | No. Replace distribution-shaping-only targets with reconstructively grounded fixed codes; retain content/predictability/use as separate tests (§§1,5,13). | Codec learns syntax, frequency or noise without useful downstream structure. |
| R2. Does a fixed codec solve relevance? | It solves coordinate drift only. No semantic or anti-collapse theorem is claimed. Reconstruction and predictive usefulness both gate interpretation. | Strong reconstruction but weak conditional predictability or no task gain. |
| R3. Is future information leaking through plans? | Forecast bank is built from q(U) only in training and inference. Ground-truth codes and target masks are label-only; first-group behavior is explicit (§§4,8). | Suffix or mask perturbation changes an earlier prediction. |
| R4. Is the copy baseline legal with overlapping targets? | Initially tempting previous-target copying would reveal future tokens. Replaced by encoding the last *observed* 16-token window (§7.3). | Baseline implementation touches tokens after e_g. |
| R5. Are coarse and fine alternatives independently mixed? | Use an exact conditional joint table and joint top-four selection. This removes independent-pair selection, while acknowledging remaining cross-horizon independence and soft-attention blending (§§7,8). | Low true-pair coverage, incompatible plan blending, poor calibration. |
| R6. Does corruption preserve residual meaning? | Fine labels under a changed root are re-encoded against that root, using the frozen residual dictionary. This is a new, explicit v15 hypothesis rather than an asserted NCM reproduction (§7.2). | Alternative-root distortion or the extra loss harms final quality. |
| R7. Can the decoder ignore plans? | Yes. Preserve the bypass for observed facts and require TP-versus-T, generic-bridge, and held-U interventions. Do not force use with hidden oracle information (§13.3). | No equal-time gain from the bridge. |
| R8. Does memory already exist in v14? | Yes: C and D already use explicit depth attention. Compare against it, replace only C aggregation, and retain residual additions (§§3,6). | TPD fails to beat TP; memory is dropped from the next named candidate. |
| R9. Is RLS justified as the default? | Its paper's incremental gain over delta is small. Default to bounded delta updates; preserve an exactly specified RLS comparison, including extra state (§6). | Delta fails and RLS wins sufficiently at equal time; revise after evidence. |
| R10. Does a nonexpansive write make the network stable? | Only the homogeneous linear M transition has that bound. Inputs, gates, address derivatives and backbone still require monitoring (§6). | M/read/gradient norms or nonfinite updates fail the gate. |
| R11. Is there free compute from the codec? | No. Include Stage A, fourfold overlapping span exposure, label extraction, cache storage and all student overhead (§§9,11). | An apparent matched-token win disappears at equal total time. |
| R12. Is the codec capacity sufficient? | Twenty-five bits cannot preserve arbitrary 16-token spans exactly. Define a useful bottleneck and gate it; no near-lossless claim (§5). | Root/fine reconstruction gains fail or important details disappear. |
| R13. Is depth memory persistent knowledge? | No. Separate transient depth state from continual adapters/replay and preserve versioned target artifacts (§12). | Continual acquisition/retention is worse than replay. |
| R14. Are v14 conclusions being invented? | Final evaluation was pending at the captured time. Treat target/noise concerns as hypotheses, and distinguish old spec from actual shrunk model (§0). | Completed v14 results remove the motivation or favor a simpler change. |
| R15. Are causal claims identifiable? | Use the factorial, auxiliary-only, generic-feature, and parameter/time-matched controls. One combined v15-versus-v14 run is insufficient (§13). | No isolated contribution or only a parameter-count effect. |
| R16. Is the document a validated implementation? | No. Analytical accounting and toy checks are separate from required model, gradient, cached-generation and MPS tests (§15). | Any required implementation gate fails. |
| R17. Can normalization or a null ablation erase/hide information? | Replaced per-code RMS normalization with a single frozen scale per codebook, preserving magnitude differences. A local-only test bypasses all plan metadata, not just candidate values (§§8,10). | Collinear-code information is lost, or a supposedly disabled branch still depends on U. |

**Completed analytical verification:** all nine checks in `v15_design/review_checks.json` pass: parameter accounting, component-state memory, group-release timing and the legal copy baseline, partial-target counts, a correlated-alternatives example, joint top-four/tail probabilities, the bounded homogeneous delta transition, RLS agreement with a direct ridge solve, and specification-presence checks. The ridge comparison differs by at most 3.34e-16 in its FP64 toy case. These results validate formulas and bookkeeping only; they do not establish model causality, gradient correctness, peak MPS memory, training stability, or improved task performance. Those remain the implementation and experiment gates above.

**Review disposition:** coherent enough to implement as a falsifiable research candidate. The unresolved empirical risks are codec relevance/capacity, prediction uncertainty, ignored or blended plans, a shallow-depth memory that may lose to AttnRes, and cost on MPS. No claim that these have been solved is made by naming the version v15.

**Selection rule after experiments:** if TP wins and TPD does not, retain predicted plans with existing depth attention as a separately named V15-P. If TPD wins but TP does not, inspect the D control before crediting concepts. If only D wins, adopt the memory-only result and reject the forecast hypothesis. If no variant beats the matched baselines, retain the best baseline and report that v15 did not improve it. If a frozen codec fails on new continual tasks, do not conceal the failure by changing its IDs mid-evaluation.

## Sources and reproducibility artifacts

- [V14 architecture](v14_Architecture.md), [implemented deviations](v14_tiny/DEVIATIONS.md), [actual configuration](v14_tiny/runs/base/config.json), [model](v14_tiny/model.py), [boundary tests](v14_tiny/tests_g0.py), [dataset contract](Dataset_v13.md).
- User-provided [Next-Concept Modeling paper](</Users/shubhankankrishna/Desktop/RL/Papers Direct/Next-Concept Modeling.pdf>), especially §§3–4 and Appendices A,F,H; [detailed review](paper_reviews/next_concept_modeling_first_principles.md).
- User-provided [Depth Memory paper](</Users/shubhankankrishna/Desktop/RL/Papers Direct/DEPTH MEMORY.pdf>), especially §§3–4 and Appendices C–F; [detailed review](paper_reviews/depth_memory_first_principles.md).
- NCM source audit pinned to `c70274273f489abc73832b2b98584a8483b6b203`; Depth Memory source audit pinned to `9a6865feeb43a153a4501cc9a6457e047ffef3c3`. The design follows the explicit rules in this document where they differ from either release.
- [Source manifest and evidence snapshot](v15_design/evidence_manifest.json), [analytical check results](v15_design/review_checks.json), [check implementation](v15_design/check_design.py). These checks support bookkeeping and the toy mathematical claims; they are not tests of an implemented v15 model.
