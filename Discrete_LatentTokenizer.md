# Discrete latent tokenization for Astra

**Design review: 20 October 2026. Status: proposed research design, not an implemented or validated tokenizer.**

Companion: [V15 Astra architecture design](V15_Astra_RePR_InfiLoop_Architecture.md).

## 1. Decision

Yes: the central idea in **Generative Representation Learning for Scalable Latent Tokenization** can sensibly inform a discrete text tokenizer. The defensible adaptation is:

1. Estimate how much an additional text view helps predict a withheld token.
2. Teach position-resolved representations to preserve that predictive information.
3. Quantize those representations through an explicit, limited-capacity bottleneck.
4. Train and test a decoder using only the information that will actually be available at deployment.
5. Train a separate conditional prior over the frozen discrete codes.
6. Accept the system only if free text generation improves at a measured total compute budget.

This is **RePR-inspired discrete tokenization**, not a reproduction of the paper's image results. Quantization introduces a new rate–distortion problem that the paper does not solve for us.

An important correction to the premise: the paper's mathematical objective is **not restricted to continuous data**. Section 4.1 explicitly discusses discrete data and categorical text likelihoods. Its experiments and practical diffusion estimator operate on images and continuous feature spaces. The genuinely new work here is making the *representation* a compact sequence of discrete symbols while retaining details and making those symbols predictable.

The recommended first comparison is **reconstruction-only product quantization versus the identical tokenizer with RePR supervision**. Do not combine a new tokenizer, a larger model, a new loop rule, and search in the first experiment.

## 2. What the source establishes, and what it does not

Primary source: the supplied [Scalable Latent Tokenization PDF](</Users/shubhankankrishna/Desktop/RL/Papers Direct/ SCALABLE LATENT TOKENIZATION.pdf>), arXiv [2610.20533v1](https://arxiv.org/abs/2610.20533), by Baade et al.

| Source location | Useful contribution | Boundary of the evidence |
|---|---|---|
| Sections 3–4.1, pp. 3–4 | Representations can be supervised by conditional versus unconditional log-likelihood differences. Categorical models are explicitly applicable. | A learned teacher produces an estimate of the data's dependence, not exact mutual information. |
| Section 4.3, p. 5, Eqs. 13–14 | Local likelihood-ratio increments telescope under consistent probability models. | Arbitrary independently trained masked conditionals need not define one coherent joint distribution. |
| Section 4.4, p. 5, Eq. 15 | Regress a representation difference against a stopped generative information target. | Adding quantization changes the approximation class and optimization. |
| Sections 5.2–5.3, pp. 6–7 | Directly supervise spatial features and reuse small masked views efficiently. | Image translation equivariance is not permission to discard text order or entity binding. |
| Ablations, p. 9 | Practical weighting matters; the unweighted diffusion-style estimator can fail numerically. | The diffusion weighting and image hyperparameters are not text defaults. |
| Appendix A.3, p. 17 | The implementation deliberately asymmetrizes the factorization; PMI need not be positive semidefinite. | A single symmetric embedding dot product is not a universally adequate factorization. |
| Appendix A.4, pp. 17–18 | Idealized reconstruction arguments connect complete information factors to input recovery. | Finite codes, imperfect teachers, insufficient masks and finite rank do not inherit exact reconstruction. |

The paper reports image representation/generation results. Its “7×/8×” tokenizer-compute comparisons include extrapolation; they are not measured speedups for text or an M2 laptop. Its latent-forcing process eventually generates pixels conditioned on features: that is different from proving that a finite latent code alone is lossless.

As checked on 20 October 2026, the [official RePR repository](https://github.com/AlanBaade/repr) announces full code release by 21 October; the available README/demo are not a complete released training implementation. This design is based on the supplied paper and its appendices, not an invented audit of unavailable code.

## 3. Start with the information that must survive

Let:

- \(x=(x_1,\ldots,x_L)\): a target block of at most 32 lexical tokens.
- \(u\): up to 256 preceding lexical tokens, available at generation.
- \(E_\psi(x)\): a context-independent block encoder.
- \(c=Q(E_\psi(x))\): the emitted discrete code sequence.
- \(D_\omega\): a text decoder.
- \(G_\theta(c\mid u)\): a future code predictor.

The initial codec is \(x\rightarrow c\rightarrow \hat x\). It does not receive an unquantized encoder skip connection. Its decoder does not receive the target text except as the normal, causally shifted training prefix.

These are separate requirements:

| Requirement | Operational measurement |
|---|---|
| Retain semantic dependencies | Held-out information-target error and context-sensitive downstream tasks |
| Retain literal details | Exact names, numbers, route IDs, negation, punctuation and ordered bindings |
| Make codes learnable | Conditional code loss and free generation with a fixed predictor budget |
| Reduce computation | End-to-end latency/throughput including decoding and preparation |
| Remain a valid tokenizer | Deterministic encoding, versioned code identity, no hidden continuous channel |

A semantic representation can map “move to node 3” and “move to node 8” nearby and still be a bad action tokenizer. Conversely, an excellent reconstruction code can be unnecessarily difficult to predict from context. The design must measure both fidelity and predictability.

An independent random identifier has no cross-view predictive information unless the views expose information about it. RePR alone therefore cannot be relied on to preserve every nonce, digit or spelling. Reconstruction supervision and, where needed, an explicitly accounted exact-detail mechanism are essential.

### Capacity is a hard constraint

With \(M\) latent slots, \(G\) product-code groups and \(K\) choices per group, the nominal payload is:

\[
B_{\rm payload}=MG\log_2K.
\]

A 32-token block from a 4,096-item lexical vocabulary has a **maximum fixed-width** representation of \(32\times12=384\) bits. Mapping every possible such block to 256 bits cannot be injective. A generative decoder cannot recover an arbitrary discarded choice with certainty.

Natural text may occupy a much smaller, highly nonuniform subset. That makes useful lossy or distribution-dependent compression plausible, not guaranteed. For example, eight independent uniform node IDs from 24 possibilities already require \(8\log_2 24=36.68\) bits for their identities alone; order, syntax and other content add requirements.

Report nominal payload, learned entropy estimates, actual packed storage, reconstruction distortion and decoding cost separately. A lower code count is not automatically a compression ratio or a speedup.

For a frozen deterministic encoder/quantizer, information can only be discarded:

\[
I(Y;C)\le I(Y;E(X)),\qquad I(X;C)\le H(C)\le MG\log_2K
\]

when the entropy units are bits. Even an ideal decoder's minimum expected reconstruction log loss is \(H(X\mid C)\), in the corresponding units. Training a better decoder cannot reduce uncertainty that the code has irreversibly introduced.

This also explains why evaluating a continuous RePR representation before quantization is insufficient. Its predictive information must survive in C. Conversely, minimizing code entropy without a fidelity constraint can produce an easily predicted but uninformative constant code. The actual research target is a favorable **rate–distortion–predictability tradeoff**, not any one loss in isolation.

## 4. Derive the text information target

### 4.1 Sample views without leaking the answer

Within a training block, sample sets of visible positions \(A\) and \(B\), and a withheld position \(j\). The token \(z=x_j\) must be absent from both views. \(A\) and \(B\) may overlap.

Preserve original position IDs, visibility masks and special-token semantics. Do not renumber sparse visible tokens as if gaps did not exist. Sample masks independently of token values; split documents before producing their views.

Use a categorical teacher with a well-defined query:

\[
p_\phi(z\mid U,j),\qquad U\in\{B,A\cup B\}.
\]

The union is a canonical set of observed position–token pairs: a token visible in both views appears once. The teacher must not learn artificial “A versus B” tags that change an otherwise identical conditioning event.

For the first implementation, use one target token per sample. This avoids treating a collection of masked-token marginals as an exact joint likelihood.

### 4.2 Estimate the incremental information

Define:

\[
\delta_\phi(A,B,z)
=\log p_\phi(z\mid A\cup B,j)-\log p_\phi(z\mid B,j).
\]

This asks: *how much does observing A change the probability assigned to the actual withheld token, beyond B?*

Use FP32 `log_softmax` and gather the observed token's log probability. Stop gradients through the target. This calculation needs two conditional evaluations, not the paper's Monte Carlo integration over diffusion noise.

For a multi-token increment \(z_{1:k}\), the legitimate extension is:

\[
\delta_\phi
=\sum_{r=1}^{k}
\left[
\log p_\phi(z_r\mid A\cup B,z_{<r})
-\log p_\phi(z_r\mid B,z_{<r})
\right].
\]

The token positions and causal target-prefix mask are part of each condition. A bidirectional masked model's independent logits are not automatically this chain-rule joint.

### 4.3 Interpret the target correctly

For exact conditionals from the true distribution, the expression is pointwise conditional mutual information. Individual samples can have **negative** values: seeing A can make the observed z less likely. Only the appropriately averaged quantity has the usual nonnegative mutual-information interpretation.

Do not clamp negative targets to zero. Do not call a positive mean estimated with a weak teacher proof of useful semantics.

The chain-rule identity gives:

\[
\sum_r\log\frac{p(z_r\mid A,B,z_{<r})}{p(z_r\mid B,z_{<r})}
=\log\frac{p(z_{1:k}\mid A,B)}{p(z_{1:k}\mid B)}.
\]

With separately learned masked conditionals, this can remain a useful distillation objective without establishing that every possible masking/order choice agrees with one joint data model. Record that approximation.

### 4.4 Validate the teacher before distillation

Required checks:

1. **Subset invariant:** when \(A\subseteq B\), the canonical conditions are identical, so \(\delta_\phi=0\) within numeric tolerance.
2. **Target exclusion:** changing the withheld gold token must not change either input tensor or pre-gather logits.
3. **Known-dependence examples:** expose or hide a relevant digit/name and inspect the resulting probability changes.
4. **Unrelated-context control:** replace A with an unrelated view and measure the change across a held-out panel; do not assume every individual ratio must decrease.
5. **Held-out predictive competence:** report teacher CE on names, digits, binding and ordinary text, alongside overall CE.
6. **Exact toy distribution:** verify telescoping against an enumerable joint distribution before relying on neural estimates.

A teacher unable to recognize the relevant dependence cannot reliably teach it to a smaller discrete representation.

## 5. Factorize the target without an invalid symmetry assumption

The tempting design is one embedding \(r(v)\) with score \(r(A)^Tr(B)\). Its Gram matrix is symmetric positive semidefinite. An arbitrary PMI matrix need not be.

For example, a binary joint distribution

\[
P=\begin{bmatrix}.4&.1\\.1&.4\end{bmatrix}
\]

has uniform marginals and PMI matrix

\[
\begin{bmatrix}\log1.6&\log.4\\\log.4&\log1.6\end{bmatrix}.
\]

Its eigenvalues are approximately **−0.4463 and 1.3863**. A same-space Euclidean Gram matrix cannot reproduce it exactly.

Use a shared position-aware encoder with two distinct, small linear role heads:

\[
R_A(v)=W_A F(v),\qquad R_B(v)=W_B F(v).
\]

Here \(F(v)\in\mathbb R^{M\times d_e}\) always has the same ordered output slots, including for masked views. At the discrete stage, \(F\) is the **dequantized emitted code representation**, not a separate full-precision bypass.

Regress:

\[
\widehat\delta
=\frac{1}{\sqrt{Md_e}}
\sum_{m=1}^M
\left\langle R_A(A)_m,\,
R_B(B\cup\{z\})_m-R_B(B)_m\right\rangle,
\]

\[
\mathcal L_{\rm repr}
=\mathbb E\left[
\frac{(\widehat\delta-\operatorname{sg}(\delta_\phi))^2}
{\sigma_{\rm train}^{2}+\epsilon}
\right].
\]

\(\sigma_{\rm train}\) is a fixed target RMS estimated on training data. Scale without subtracting a global mean, so that the zero-information target and difference structure remain intact. The normalization \(1/\sqrt{Md_e}\) is fixed for a given configuration; it is not a claim about the natural scale of PMI.

Use the same \(R_B\) function for B and B+z. Changing its role definition between increments can destroy the intended telescoping interpretation. Distinct A/B heads provide the required asymmetry without that problem.

This is an adaptation of the paper's factorization, not its exact architecture. Finite rank and quantization may still leave irreducible approximation error. A small representation-loss error is an auxiliary result; it does not establish good text generation.

## 6. Concrete first codec

### 6.1 Input and encoder

- Train a byte-fallback lexical tokenizer on the training split only.
- Initial vocabulary: **4,096 total entries including special tokens**.
- Block length: **32 lexical tokens**, not 32 words. Report the actual words-per-block distribution.
- Independent block encoding permits reusable caches and avoids changing code identity when the prompt changes.
- Initial encoder: width 192, two pre-normalized attention/SwiGLU blocks, positional information retained.
- At stride two, concatenate each ordered pair of encoder outputs and project 384→192. Produce \(M=16\) latent slots.
- For masked-view training, keep all 32 position placeholders; the visibility mask controls their observed content.

This is a pilot configuration, not a demonstrated optimum. The encoder may attend across the block; the ordered pooling establishes local slot anchors without pretending each slot contains only two independent tokens.

### 6.2 Product quantization

Split each 192-dimensional slot into two 96-dimensional groups. Each group has a separate 256-entry codebook:

\[
c_{m,g}=\arg\min_{k<256}\|e_{m,g}-V_{g,k}\|_2^2,
\qquad
\tilde e_m=[V_{1,c_{m,1}};V_{2,c_{m,2}}].
\]

Emit integer code IDs. Use deterministic nearest-neighbor tie breaking and a documented numeric dtype. The encoder, preprocessing and codebooks jointly define the tokenizer.

Why start here:

- Small categorical heads: two 256-way choices, not one enormous global code.
- Each group labels the encoder feature directly; its target is not redefined by a previous generated residual.
- Position-resolved slots preserve an opportunity to bind content to order.
- The capacity can be changed explicitly by stride, groups or codebook size.

Product quantization does **not** make groups statistically independent. The prior must model dependence or declare that approximation.

| Candidate | Slots | Groups × choices | Scalar code IDs | Nominal bits / 32 text tokens | Fixed-width comparison |
|---|---:|---:|---:|---:|---:|
| Fidelity pilot, stride 2 | 16 | 2 × 256 | 32 | 256 | 1.5× smaller than 384 bits |
| Compression candidate, stride 4 | 8 | 2 × 256 | 16 | 128 | 3× smaller than 384 bits |
| Uncompressed lexical baseline | — | — | 32 | 384 | Reference only |

These exclude headers; an explicit length in 0…32 takes up to six more bits. For fixed full blocks, length is implicit. For final partial blocks, EOS/length handling must be deterministic and charged to the format. At generation, never provide the gold output length.

The stride-two pilot emits **32 scalar IDs for 32 lexical tokens**. Smaller heads or a better representation might help, but there is no scalar-sequence-length reduction yet. Even stride four is useful only if its distortion and decoder costs are acceptable.

### 6.3 Why not start with residual VQ or scalar quantization?

| Option | Strength | Relevant concern | Decision |
|---|---|---|---|
| Product VQ | Simple finite alphabet; stable labels independent of generated prefixes | Codebook collapse; dependencies between groups | Primary pilot |
| Finite scalar quantization | No learned nearest-neighbor codebook | Bin saturation, useful geometry and joint alphabet still need testing | Small fallback ablation if VQ training is unstable |
| Residual VQ | Progressive rate and reconstruction refinement | Later residual targets depend on the actual previous reconstruction; repeats a known Astra difficulty | Defer until the simple codec passes |
| One code per 32-token block | Very short sequence | Enormous required alphabet or severe detail loss | Reject as the initial design |

If residual VQ is later used, changing earlier codes requires recomputing later residual targets. The completed v4 follow-up tested this correction and still failed free route generation; residual consistency is necessary for that objective, not sufficient for success.

### 6.4 Hard bottleneck and gradient estimator

Initialize codebooks from training encoder features. Use EMA centroid updates and a straight-through estimator:

\[
e_{\rm ST}=e+\operatorname{sg}(\tilde e-e).
\]

The forward pass sees the actual quantized vector; backward uses the surrogate gradient through e. This is biased gradient estimation, not differentiation through the discrete argmin.

With EMA codebooks, use a commitment term:

\[
\mathcal L_{\rm commit}=\|e-\operatorname{sg}(\tilde e)\|_2^2.
\]

Do not simultaneously claim codebooks are EMA-only and update them through an optimizer codebook loss. Log assignments, active entries, entropy, nearest-code margins, feature norms and held-out distortion. Replace dead entries only during codec training; any later replacement changes tokenizer identity.

Uniform code usage is not the objective. Forcing all entries to be equally common can damage a naturally nonuniform text distribution.

## 7. Decoder and exact-detail contract

Start with a small autoregressive decoder conditioned on the quantized slots:

\[
p_\omega(x\mid c)=\prod_i p_\omega(x_i\mid x_{<i},c).
\]

Train with causal teacher forcing; **evaluate with free decoding from codes**, without a gold text prefix or gold length. During the initial codec evaluation, do not also give the decoder the raw source text, encoder features or external route.

The initial loss is:

\[
\mathcal L_{\rm codec}
=\mathcal L_{\rm tokenCE}
+\lambda_{\rm repr}\mathcal L_{\rm repr}
+\lambda_{\rm commit}\mathcal L_{\rm commit}.
\]

Token CE is averaged over real target tokens, including the applicable EOS convention. Use explicit masks, not padding frequency, to set the objective. Warm up reconstruction before adding the representation term; compare \(\lambda_{\rm repr}=0\) with a small predeclared grid. Monitor gradient contributions rather than copying the paper's image loss weight.

Three ways this can fail must be distinguished:

1. **Encoder loss:** different important inputs collapse to the same code.
2. **Decoder failure:** the code contains usable detail but free decoding cannot recover it.
3. **Prior failure:** gold-code decoding works, but predicted codes do not.

A strong autoregressive decoder can obtain low teacher-forced CE by exploiting the gold prefix while barely using the codes. Therefore compare ordinary, zeroed, shuffled and unrelated codes under free decoding. Require useful code dependence; do not infer it from low CE alone.

### Optional exact-detail mechanisms

For verified structured plans, an ordered node-ID channel and deterministic lexicalizer can guarantee the displayed sequence. This is a valid application design and an important baseline. It is **not evidence that the latent tokenizer preserved those IDs**.

For arbitrary text compression, a raw-token escape stream can transmit details that the latent channel loses. Charge the escape flag, payload and computation; label this a **hybrid codec**. At generation time the model does not know the future gold text, so an oracle escape stream is unavailable. Copying is legitimate only from an observed prompt or verified external state, with an actual predicted pointer/selection.

A parallel token decoder is a later speed experiment. Independent output logits impose a factorization that may damage agreement and exact sequence fidelity. Shared latent conditioning helps but does not by itself prove that this factorization is adequate.

## 8. Train in dependency order

### Stage A — data and teacher

Prepare normalized raw documents, document-level splits, token offsets and masks in batches. Keep ordinary text and synthetic exact-detail probes separately scored. Deduplicate near copies and route templates across splits where appropriate.

Record the actual corpus and its license/provenance. If the available corpus is FineWeb, label it FineWeb; do not describe that evaluation as LM1B or a reproduction of the NCM checkpoint's training distribution. Fit the new lexical tokenizer only after fixing the document split.

Train or select a modest categorical teacher. A larger frozen teacher is optional, but all teacher preparation/inference cost belongs in the experiment ledger. Distillation cannot be advertised as cheap by omitting the teacher.

Initial views: sample varied visible fractions, including small and large A/B, with one withheld target. Do not import the image paper's 75%/93.75% mask thresholds or 32 diffusion samples as defaults.

Freeze the accepted teacher before the first controlled codec comparison. The paper shares generative and representation components during training; separating them here is a deliberate simplification to avoid moving targets and to make ablations interpretable.

### Stage B — continuous warm-up

Train the encoder/decoder reconstruction path and optional representation heads before hard quantization. Keep this stage short and equal across comparison arms. Its reconstruction result is diagnostic only: it is not a discrete-tokenizer result.

### Stage C — quantization-aware training

Initialize the codebooks, enable the hard forward bottleneck, and train reconstruction plus commitment. Compare the matched RePR arm with identical data order, code capacity and training budget.

RePR supervision now consumes dequantized emitted features on all views. A full-precision representation branch that is discarded at deployment would weaken the claim that the discrete codes learned the information objective.

Use the same decoder size in both arms. Otherwise a reconstruction improvement could come from extra decoder capacity.

### Stage D — freeze and cache

Freeze **encoder, pooling, codebooks, decoder, lexical tokenizer and all preprocessing**. Save hashes and a codec version. Cache gold codes in bounded offline batches, checking roundtrip samples from every shard.

Moving the encoder changes code IDs even if the codebook tensors remain unchanged. Moving the decoder changes the meaning of a “good” predicted code. Either change invalidates the relevant cached targets/comparisons and requires explicit versioning.

### Stage E — train the prior

For a normalized baseline, flatten codes in a fixed slot/group order:

\[
p_\theta(c\mid u)
=\prod_{m,g}p_\theta(c_{m,g}\mid u,c_{<(m,g)}).
\]

A pair of 256-way heads is not equivalent to a joint 65,536-way distribution if the heads operate independently. Conditional group prediction preserves that distinction.

Start with teacher-forced prior training, then evaluate its own prefixes. If sampled-prefix training is added, report it as a separate intervention. Product-VQ labels remain the code of the gold feature; unlike residual VQ they do not need residual re-encoding after a prefix error, although off-manifold sampled prefixes still create a learning problem.

The prior's generation input must never contain target-block codes, target lexical tokens, future encoder features or teacher scores derived from the answer.

### Stage F — evaluate free text and optional parallel generation

Generate codes from the available context, then decode them freely. A parallel masked-code or iterative refinement model is an optional later arm. Its training loss is not automatically the normalized autoregressive code NLL above.

Only compress to stride four or add parallel token decoding after the simpler system passes its fidelity gate. Change one property per comparison.

## 9. Correct likelihood accounting

For the initial codec-only decoder, the deployed latent language model is:

\[
p(x\mid u)=\sum_c p_\theta(c\mid u)p_\omega(x\mid c).
\]

Code cross-entropy is not text perplexity. With deterministic encoder \(c^*=E_Q(x)\), a valid single-path upper bound on text NLL is:

\[
-\log p(x\mid u)
\le -\log p_\theta(c^*\mid u)-\log p_\omega(x\mid c^*).
\]

Report it as a bound, with both terms and token normalization, not as an exact marginal likelihood. If the decoder also conditions on u, replace its conditional accordingly and account for the changed architecture.

During training, adding the frozen prior's code cost to the encoder objective could encourage predictable codes. It could also discard inconvenient detail. That is a later rate–distortion experiment with an explicit reconstruction constraint, not part of the first causal test of RePR.

## 10. Required experiment matrix and gates

All numbers below are **proposed engineering acceptance thresholds**, not paper results or statistical guarantees. Register them before evaluating fresh held-out data.

| Gate | Test | Decision |
|---|---|---|
| Probability/masking | Subset-zero, no target leakage, exact toy telescoping, deterministic view construction | Must pass before representation training |
| Tiny memorization | Freely reconstruct all 16 fixed synthetic blocks from hard codes; repeat with three seeds | Failure stops scaling and triggers codec/decoder diagnosis |
| Exact details | Fresh panel of at least 256 routes, IDs, numbers, negation and binding examples | Require ≥99% exact critical fields and ≥95% exact normalized blocks before planner integration |
| Ordinary text | At least 1,024 unseen blocks; token accuracy, exact blocks, edit rate, rare-token accuracy, code-use ablation | Report all; do not substitute token accuracy for exact sequence accuracy |
| Compression promotion | Stride four versus stride two on the same held-out panel | At most one percentage-point critical-field drop and a measurable end-to-end benefit |
| Prior competence | Matched direct-token baseline; predicted-code versus gold-code results | A large gap blocks claims that the tokenizer solves generation |
| RePR value | Same architecture/capacity/budget, with and without RePR; repeat promising result over three seeds | Promote only a reproducible Pareto improvement in fidelity, predictability or measured compute |

The threshold panel is intentionally stricter than a semantic similarity score, because Astra's observed failure is incorrect literal action content. It is not a guarantee of universal losslessness.

For open-ended continuation, exact agreement with one reference is not a complete quality metric. Also measure relevant-context advantage, entity consistency, contradiction/negation errors, repetition and continuity across 32-token boundaries. Deterministic synthetic tasks provide exact answers; ordinary text provides broader realism.

Use fresh final tests: the existing v4 route test has already been examined and can now serve only as a development regression set. Bootstrap by source document or graph, not by many overlapping blocks from the same source.

## 11. M2 / 16 GB execution design

- Run teacher preparation, codec training, code caching and prior training sequentially.
- Begin with a small pilot; benchmark actual CPU/MPS paths before assigning a wall-time budget.
- Start model microbatch at one and grow only after warm-up peak-memory and sustained-throughput measurement. Use accumulation when helpful.
- Cache compact integer codes in shards. Separate training data from memory-mapped validation data; avoid loading the full corpus into unified memory.
- Keep optimizer states only for currently trained modules. Do not keep the existing ~208M frozen NCM stack resident alongside a new teacher and codec.
- Use FP32 for log probabilities, reductions, EMA statistics and the loop's normalization state. Adopt reduced-precision dense computation only after numeric/backward checks.
- Profile the complete 32-token generation path. The fidelity-first autoregressive code prior plus autoregressive text decoder may be **slower** than direct lexical generation.
- Preserve source/config/tokenizer/data identities and RNG/optimizer state. A resume must not silently adopt changed codebooks or a new lexical tokenizer.

Do not promise that a complete new teacher, tokenizer and language model will become competitive in a 24-hour laptop run. The first local budget should buy evidence about the bottleneck; later budgets depend on those measurements.

## 12. Minimal pseudocode

```text
# Offline teacher, accepted and frozen first
for training block x:
    A, B, j = sample_views_without_target_leakage(x)
    logp_plus = teacher.logprob(x[j], visible=union(A, B), query_position=j)
    logp_base = teacher.logprob(x[j], visible=B, query_position=j)
    target = stop_gradient(logp_plus - logp_base)

    f_A  = dequantize(quantize(encoder(masked_view(x, A))))
    f_B  = dequantize(quantize(encoder(masked_view(x, B))))
    f_Bz = dequantize(quantize(encoder(masked_view(x, B union {j}))))
    info_prediction = aligned_dot(head_A(f_A), head_B(f_Bz) - head_B(f_B))

    full_codes = quantize(encoder(x))
    token_loss = decoder.causal_teacher_forced_CE(x, full_codes)
    optimize(token_loss + lambda_repr * scaled_squared_error(info_prediction, target)
             + lambda_commit * commitment_loss)
    update_codebooks_by_EMA()  # not also optimizer codebook updates

freeze_and_hash_complete_codec()
cache_codes_by_document_split()
train_conditional_code_prior_without_future_inputs()
evaluate(predicted_codes_then_free_text_decode)
```

The straight-through expression in Section 6.4 defines gradients for the pseudocode's quantization path. All learned components, data masks and loss normalizations must be explicit in an implementation.

## 13. Evidence and remaining uncertainty

The strongest reason to try this is that RePR gives a direct way to supervise predictive information across the **features we actually quantize**. Its strongest risk is that useful dependence, exact reconstruction and easy conditional code prediction can compete for a small channel.

The decisive experiment is not whether a RePR loss decreases. It is whether adding that loss to an otherwise identical discrete codec improves a held-out fidelity/compute tradeoff, and whether a small prior can generate those codes.

Local arithmetic checks were run without loading a model:

- Likelihood-ratio telescoping on a finite joint distribution: maximum error \(2.22\times10^{-16}\).
- A symmetric PMI counterexample has a negative eigenvalue, supporting the asymmetric-head design.
- Code capacities and the companion architecture's major parameter counts are explicitly calculated.

See [check script](design_reviews/20261020/check_design_math.py) and [results](design_reviews/20261020/design_math_results.json). These checks validate algebra and accounting only. They do not establish learning, GPU correctness or speed.

### Source identity

Supplied tokenizer PDF SHA-256:
`99176e8d7d2dac40928bfaf30b53c933a2e0ad353f4df13fc2e17134d1b13a01`.

Relevant local evidence:
[v4 reviewed contract](v15_astra_ncm_v4/REVIEWED_SPEC.md),
[renderer diagnosis and follow-up design](v15_astra_ncm_v4/followup/README.md),
[completed follow-up results](v15_astra_ncm_v4/runs/20261019_renderer_prefix_followup/final_results.json).
