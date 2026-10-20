# V15-XHRM-Atlas — Detailed Architecture Review

**Reviewed:** 16 October 2026, IST  
**Document:** [V15-XHRM-Atlas.md](V15-XHRM-Atlas.md), all 268 logical lines  
**Overall score: 4.6 / 10 — promising research direction; substantial specification revision required before implementation.**

This score evaluates the document as the generalized, implementable architecture specification it claims to be. It does not estimate the accuracy of a future trained model. No XHRM-Atlas implementation was executed, and no training or existing experiment was rerun for this review.

The original proposal was preserved. Its reviewed SHA-256 is:

~~~text
fdbb34056197767d9a58ecea4d39e5f89eebc6afd9c21f0a09c740b8fc8fdc07
~~~

## 1. Assessment and scoring

The strongest part of this proposal is its diagnosis: unstable targets, an unused prediction head, an insufficiently demonstrated contribution from the Concept Core, and an expensive continual-learning system deserve direct experiments. Freezing the entire target codec, making predictions available to the decoder, retaining explicit causal boundaries, and requiring matched-budget comparisons are sensible responses.

The weakest part is the integration. Several equations describe different models from the surrounding prose. The weighted auxiliary loss is presented as an exact likelihood; a detached prediction bridge is presented as a gradient route; a two-level, one-subspace decoder interface is presented as generalized to three levels and two subspaces; and the fast-memory equations do not specify compatible input/output dimensions. These are consequential issues because two competent implementations could follow this document and train different systems.

The proposal also moves too quickly from historical failures to named remedies. A failed probe does not establish that HRM recurrence is the solution. A poor growing-bank result does not establish that a polynomial fast-weight memory will win. These should be hypotheses with independent rejection criteria.

### Weighted rubric

| Criterion | Weight | Score / 10 | Reason |
| --- | ---: | ---: | --- |
| Motivation and use of project evidence | 20% | 8 | Strong connection to real failures, with several overstatements requiring correction. |
| Mathematical consistency | 20% | 3 | Likelihood claim, hierarchy generalization, mixture conditioning, and memory dimensions need repair. |
| Implementation completeness and causal contracts | 20% | 4 | Good release/locality foundations; recurrent state, gradients, and memory lifecycle remain underspecified. |
| Compute, memory, and scaling credibility | 15% | 3 | Unsupported exact totals, incompatible codec estimates, and unmeasured mode overhead. |
| Experimental design and falsifiability | 15% | 5 | Useful gates and controls, but a nested ladder is called factorial and the final arm bundles several mechanisms. |
| Clarity, traceability, and reproducibility | 10% | 4 | Readable component map, but 478 unresolved citation tokens and missing operational definitions. |
| **Weighted total** | **100%** | **4.6** | **Revise before committing substantial training compute.** |

This is an editorial score, not a statistically calibrated measurement. The major deductions concern correctness and reproducibility, not the ambition of the architecture.

### Priority findings

“Blocking” means the specification needs correction before a faithful implementation can be judged. “High” means the issue could invalidate the intended experiment or interpretation.

| ID | Priority | Proposal location | Finding |
| --- | --- | --- | --- |
| F01 | Blocking | §4.3, lines 166–173 | Weighted CE plus counterfactual loss does not yield the claimed exact mixture likelihood. |
| F02 | Blocking for larger tiers | §4.3–§5, lines 161–187 | Subspace indices and complete ancestor-conditioned retargeting disappear; plan vectors have the wrong declared width outside Tiny. |
| F03 | Blocking | §5–§6, lines 187–215 | Detaching the entire plan package removes the claimed gradient path through that package. |
| F04 | Blocking | §6.1, lines 204–218 | Polynomial-feature memory output, residual target, and MAG addition lack a consistent dimension contract. |
| F05 | High | §4.2, lines 143–157 | Module composition, recurrent clocks, carry states, and mode-specific caches are incomplete. |
| F06 | High | §6.2–§6.3 | Wake verification, write timing, sleep target construction, and memory address drift are unresolved. |
| F07 | High | §7, lines 233–245 | Scaling estimates are not reproducible; standard dense codec arithmetic exceeds the stated Mid and Frontier counts. |
| F08 | High | §8, lines 251–268 | The experiment ladder cannot isolate all advertised causes; Stage 3 cost and a promised G4 gate are absent. |
| F09 | High | §0, lines 14–17 | Historical metrics are partly overgeneralized and acquisition comparisons need an explicit baseline. |
| F10 | Medium | Throughout | Missing definitions, broken references, and inherited-but-unpinned details obstruct reproduction. |

## 2. Evidence audit: what V13/V14 actually establish

I checked the final saved result files, rather than treating report prose or smoke outputs as proof of completed experiments:

- [V14 base evaluation](v14_tiny/runs/eval/base/results.json).
- [V14 continual-learning evaluation](v14_tiny/runs/eval/cl/cl_results.json).
- [V14 diagnostics](v14_tiny/runs/eval/diag/diagnostics_v14.json).
- [V14 generated report tables](v14_tiny/REPORT_TABLES.md).
- [Earlier V15 architecture and evidence synthesis](v15_Architecture.md).
- [Earlier V15 self-review](v15_design/review_from_scratch.md).

### 2.1 Target drift: valid motivation, narrower interpretation

The V14 diagnostic records global-code reassignment of **96.36% between the 10% and 50% snapshots**. The corresponding pooled local/global figure is **86.22%**, and global reassignment between the 30% and 70% snapshots is **90.37%**. Final teacher/online agreement is **40.96% globally** and **86.48% locally**.

The proposal should identify the snapshot interval and global/local distinction. Reassignment measures how often fixed examples receive different labels; it is not directly a measurement that 96.36% of dictionary entries changed their semantic meaning. Code permutation, geometry changes, and movement of examples across boundaries can all contribute.

Freezing the complete deterministic codec fixes target reassignment for the same input and preprocessing. It does **not** guarantee useful semantics, predictable targets, stable student representations, or successful continual adaptation.

### 2.2 Diagnostic V3: the incremental global probe criterion failed

The final V14 global-code probe margins for Concept Core output versus pooled input are:

| Family | Output-minus-input probe accuracy |
| --- | ---: |
| Grid | +3.32 percentage points |
| Program | −1.08 percentage points |
| State | −0.88 percentage points |
| Planning | +0.25 percentage points |

All miss the specified five-point criterion. However, V14 passes its copy/marginal and local-view diagnostics. Local-code probe margins also differ from the global-code margins. The justified statement is that **incremental global-code information at the output was not demonstrated by the registered probe threshold**.

This does not prove that the Concept Core performs no useful computation. It motivates trained bypass controls and stronger probes alongside downstream answer quality. Preserve the earlier V15 requirement to split probes by document or semantic instance; adjacent overlapping windows must not be independently divided between probe training and testing.

### 2.3 “Held-out rungs 8/9 scored 0%” is too broad

The saved V14 ladder results are:

| Task | Rung 8 | Rung 9 |
| --- | ---: | ---: |
| Grid 1 | 0.00% | 0.00% |
| Grid 2 | 0.00% | 0.00% |
| Arithmetic | 7.03% | 3.91% |
| List | 43.75% | 44.53% |
| Register | 18.75% | 19.53% |
| Graph | 34.38% | 32.81% |
| State | 61.72% | 57.81% |
| Planning | 0.00% | 0.00% |

Rewrite the executive claim to identify **grid and planning**. Stronger recurrence may help these tasks, but the observed failure does not identify depth, recurrence, routing, training exposure, or objective design as its cause.

### 2.4 Continual learning: replay won, but the acquisition comparison needs precision

From the saved S1–S6 gains:

| Method | Final average | Mean acquisition gain | Forgetting | S0 retention before return |
| --- | ---: | ---: | ---: | ---: |
| Sequential | 0.3239 | 0.14583 | 0.0384 | 0.3229 |
| Replay | 0.3341 | 0.11363 | 0.0287 | 0.4000 |
| V14 specialized method | 0.3033 | 0.05945 | 0.0174 | 0.4105 |

The specialized method acquires **52.32% as much as replay**, or **40.77% as much as sequential training**, by this mean-gain calculation. “Less than half” is correct against sequential training, not against the replay baseline immediately named in the proposal.

The saved paired difference against replay is −0.0307 for final average, with interval [−0.0386, −0.0236]. The S0 retention difference is +0.0105 with interval [−0.0091, +0.0287]. These use the saved paired estimator and may differ slightly from subtracting rounded table entries.

Therefore lower forgetting did not establish superior continual learning: reduced acquisition matters, and retention superiority over replay was not established by that interval.

The approximately **17.01 GiB versus 4.34 GiB** driver-allocation comparison supports a memory and runtime investigation. Driver allocation is not identical to live tensor bytes or physical resident RAM. The historical record does not isolate one component as the sole cause. A fixed-size replacement must still prove bounded live state, bounded update workspaces, and competitive acquisition.

### 2.5 Preserve the positive baseline evidence

The earlier evidence synthesis records V14 Reasoning **0.3725 versus V13 0.2829**, and text NLL **2.6861 versus 2.8004**. Reasoning is a registered composite, not universal answer accuracy. These are historical recipe comparisons with differing exposure, not a controlled attribution to one architectural change.

The proposal should preserve this context. It is improving an imperfect but useful system, and should retain strong existing controls.

## 3. Research-source audit

The six named research directions have identifiable primary sources. The concern is the transfer of their conclusions into this new combination, not that the names are fabricated. The proposal currently contains **478 unresolved span-reference tokens** and no usable bibliography; replace these with stable paper versions and section references.

| Primary source | Supported connection | Boundary of the transfer |
| --- | --- | --- |
| [Next-Concept Modeling, 2610.14162v1](https://arxiv.org/html/2610.14162v1) | Two-stage learning with a discrete multiscale codec supports the frozen-target direction. | Its coarse-to-fine generation does not establish this causal PlanAttention hybrid's quality, speed, or text likelihood. |
| [HRM-Text, 2605.20613v1](https://arxiv.org/html/2605.20613v1) | Dual-timescale recurrence, module-exit normalization, and warmup credit assignment are grounded in the source. | The paper also changes pretraining to instruction/response completion with PrefixLM masking; its gains cannot be assigned solely to the recurrent module. |
| [Explorative Modeling, 2607.27372v1](https://arxiv.org/html/2607.27372v1) | Its mixture interpretation motivates testing multiple latent-conditioned predictions. | Appendix F explicitly limits the separation-based expressivity result to squared-error settings; it does not transfer that theorem to discrete heads. |
| [GradCuit, 2608.02585v1](https://arxiv.org/html/2608.02585v1) | Optimizing intermediate latent states against continuation feedback motivates latent-search experiments. | Inserted prefix states in instruction-tuned Transformers differ from additive offsets in this concept core. The layer study finds task-dependent early/middle placements, not a universal optimal 50% boundary. |
| [Titans, 2501.00663](https://arxiv.org/abs/2501.00663) | Persistent memory and neural test-time memory motivate architectural components. | Their presence does not establish useful task adaptation or make this exact residual-addition branch an evaluated Titans configuration. |
| [ATLAS, 2505.23735v1](https://arxiv.org/html/2505.23735v1) | Polynomial features and context-window memory optimization support studying richer fixed-size memories. | They do not establish the proposed 99.8K-parameter shape, sub-MiB implementation, or transfer from successful search residuals. |

Keep literature-derived components separate from project-specific inventions: the shared document codec stem, final-cycle mode branching, mixture-to-plan interface, search-residual memory targets, and sleep loss are new integration choices here.

## 4. Stage 1: frozen hierarchical residual codec

### 4.1 What is technically promising

The proposal correctly separates target construction from student learning, uses a discrete bottleneck with no token skip path, specifies pre-update residuals for EMA updates, and includes a readiness test. Nested level dropout gives the coarse level a direct reconstruction responsibility.

The hierarchy arithmetic is internally consistent for the declared presets:

| Tier | Code count × bits per code | Nominal bits per window |
| --- | --- | ---: |
| Tiny | (1 + 4) × 1 × log₂32 | 25 |
| Mid | (1 + 2 + 4) × 1 × log₂1024 | 70 |
| Frontier | (1 + 2 + 8) × 2 × log₂8192 | 286 |

These are fixed-length capacity counts before coding overhead, not measured entropy, semantic content, or guaranteed compression quality. Tiny's 25 bits cannot losslessly identify arbitrary 16-token strings from a 2,048-token vocabulary, which require up to 176 bits. Losslessness is unnecessary for a useful latent target, but exact entities, numbers, and algorithmic state need direct fidelity tests.

### 4.2 The shared causal stem changes what a “future-span code” means

**Location: §3.2, lines 102–105.**

At Tiny scale the encoder sees embeddings from the selected future window. At larger scales the stem sees the document prefix before producing each future token's feature. Two identical future windows following different prefixes can therefore produce different labels.

This can be a defensible contextual target. It is not a context-independent tokenizer of the future span. It is also **not automatically a future leak into the student**, provided the codec is used only to generate supervision.

Choose and document one contract:

1. A window-only codec, whose labels depend only on the selected window and validity mask.
2. A contextual codec, whose labels also depend on a declared preceding context and boundary policy.

For the second choice, store the context length, chunking/cache rules, absolute/document-relative position handling, and reset behavior in the codec identity. A streaming extraction pass and a full-document extraction pass must agree under the chosen contract. Tiny and larger tiers then test related but different target definitions.

### 4.3 The straight-through estimator is incomplete

**Location: §3.3–§3.4, lines 115 and 123–124.**

The document defines z, averaged residuals, and reconstructed quantized states, but does not define the continuous surrogate $\bar z_m^{(l_{\text{cut}})}$ used in the straight-through expression. Different choices propagate different gradients to individual groups.

For example, using each group's z as the surrogate and using a coarse partition average have identical quantized forward outputs but different backward credit assignment. The coarse-only objective makes this distinction consequential.

Define the surrogate, whether earlier quantized levels are detached in later commitment losses, and how gradients pass through pooling. Define $\nu_{g,m}$, including whether partial groups receive token-count weights or binary validity. Empty spans must neither contribute to reconstruction/commitment losses nor update EMA statistics or dead-code resets.

### 4.4 EMA and dropout need one precise algorithm

State whether dropped levels still collect residual statistics and update codebooks, or update only when active. Specify accumulation across microbatches, EMA initialization/smoothing, tie-breaking, and how several dead codes are reseeded without all receiving the same residual. Hash the embedding, stem, window encoder, norms, pooler, positional tensors, codebooks, preprocessing configuration, and relevant buffers.

Freezing weights alone is insufficient if training-mode dropout, EMA buffers, reseeding, or preprocessing can still change the target function.

### 4.5 Readiness is a necessary screen, not semantic validation

The 0.05-nat thresholds are reasonable proposed gates, but they are not established universal cutoffs. The document's §3.4 requires every finer level to improve and specifies family coverage; G1 later reduces this to full-versus-coarse improvement. Use one definition.

Add per-family code usage, coarse/fine distortion, valid-length distributions, number/entity fidelity, and comparison against shuffled or constant codes. A shallow decoder can still exploit positional or distributional regularities. Better reconstruction alone does not show that the prefix can predict the codes or that the token decoder benefits from them.

Fit the codec on training data only. Use development data for readiness and checkpoint choice, and keep final evaluation sealed. Overlapping windows from one semantic instance belong to the same split.

## 5. Stage 2: causality, recurrence, mixtures, and plans

### 5.1 The release rule and local-reach arithmetic are useful

For stride four, the completed-group rule

$$
g(t)=\lfloor(t+1)/4\rfloor-1
$$

correctly releases group zero at row three. The stated Tiny calculation gives $R_L=51$ and $R_D=66$; the larger-window calculation gives 197 and 324. Fixed encoder keys/values prevent local token reach from growing merely because decoder depth increases, assuming every other decoder dependency respects that same contract.

Use the reach convention consistently: these values describe maximum backward offsets. Explicitly identify whether an implementation's window length includes the current row.

The gate and role-mask restrictions are valuable. However, “all long-range information goes through C” means a **graph routing constraint**, not proof that C performs useful deliberation. It can forward information already present in the global encoder state. Keep a trained $U=p$ bypass and a simple projected-p control.

Future labels, future validity masks, true future lengths, and target-conditioned beam candidates must remain absent from student forward inputs. Also preserve the earlier V15 warning: the previous overlapping future-window label is not a legal copy baseline. Use a fully observed past window.

### 5.2 F05: write the recurrence as executable state transitions

The displayed fast update sums blocks that all receive the same previous state. Read literally, those are parallel transforms, not a sequential stack of PreNorm blocks. If “Block” returns a full residual state instead of a delta, the outer addition can also double-count residuals.

A clear module definition would instead specify an injected input a, then:

$$
h_0=a,\qquad
h_\ell=h_{\ell-1}+f_\ell(\operatorname{Norm}(h_{\ell-1}),\text{allowed context}),
\qquad F(a)=\operatorname{Norm}(h_n).
$$

This is illustrative notation, not a completed architecture. The exact slow/fast input injection must still be declared. It makes block composition and residual ownership explicit. Module-internal PreNorm composition followed by an exit norm is the relevant [HRM-Text construction](https://arxiv.org/html/2605.20613v1).

The specification needs separate indices for:

- Document/concept position g.
- Outer cycle c.
- Fast update s.
- Internal block index.
- Exploration mode k.

Define the fast state entering cycle two, the slow update in cycle one, and which states carry across concept positions. State whether mode k is a shared identity across the document or a newly selected identity at each anchor.

For incremental decoding, identify every cache by layer, recurrence step, and mode where necessary. Sharing final-cycle keys/values between modes can silently evaluate a different function from the batched training pass. Persistent prefix tokens need their own visibility and reset rules; they must not acquire suffix information through an unintended bidirectional path.

**Acceptance test:** full-sequence causal evaluation and incremental evaluation agree for multiple documents, partial groups, K=1 and K>1, with explicit cache reset tests.

### 5.3 The advertised “50% midpoint” uses the wrong clock

With two cycles and three fast updates per cycle, the unbranched sequence is:

~~~text
L L L H | L L L H
~~~

The proposed branching/offset point immediately before the final H occurs after **seven of eight module calls**, or 87.5% of this unrolled call count. Equal numbers of unique L and H block parameters do not place that operation halfway through the computation. FLOP-weighted position will differ again because L has a local cross-read.

An additive offset there may still work. Describe it as a **final-H input offset** and compare a small number of preregistered placements. Do not cite the GradCuit layer experiment as proof that this interface is optimal.

The BPTT schedule also needs rounding rules for the integer horizon and a definition of whether K parallel final-H evaluations form one recurrent time step. Retaining a graph for latent optimization must be distinguished from keeping model weights trainable.

### 5.4 F01: the XM loss is not the claimed normalized likelihood

**Location: §4.3, lines 166–173.**

Weighted auxiliary cross-entropies can be useful objectives. The error is calling their exponentiated sum a normalized probability of a future span.

For a minimal example matching Tiny's factor structure, consider one binary root and four binary residuals. Let every categorical head assign probability 1/2 to either outcome and use the proposal's weights:

$$
\ell_{\mathrm{clean}}
=\tfrac12\log 2+\tfrac12(0.4+0.3+0.2+0.1)\log 2
=\log 2.
$$

There are $2^5=32$ tuples, each assigned $\exp(-\ell)=1/2$. Their total “probability” is **16**, not one. This failure already exists with K=1 and with the counterfactual term disabled. Mixing K identical such heads does not repair it.

The counterfactual loss adds another difficulty: it scores alternative decompositions under sampled roots, not simply the observed tuple's probability. Putting it inside the mixture exponent changes mode responsibilities based on an auxiliary robustness criterion.

#### A mathematically consistent repair

Let C denote the complete code tuple, h the observed prefix state, and $a_k(h)$ normalized mode weights. Define a normalized autoregressive tree factorization with all required indices:

$$
q_k(C\mid h)=
\prod_{l,i,j}
q_{k,l,i,j}\!\left(c_{l,i,j}\mid h,C_{\operatorname{parents}(l,i,j)}\right).
$$

The parents must form an acyclic, declared conditioning order. If subspaces or siblings are conditionally independent, say so; otherwise their earlier codes must enter the condition.

Then use:

$$
\mathcal L_{\mathrm{code}}
=-\log\sum_k a_k(h)q_k(C^*\mid h),
\qquad
r_k=\frac{a_k(h)q_k(C^*\mid h)}
{\sum_j a_j(h)q_j(C^*\mid h)}.
$$

Compute this with log-sum-exp. Keep horizon balancing and counterfactual retargeting as separately named auxiliary losses, with coefficients and ablations. If retaining the original weighted objective, label it a **mixture of auxiliary scores** and withdraw the exact-likelihood claim.

Even the repaired expression is an exact likelihood of **code tuples**, not raw future text. A deterministic many-to-one codec does not establish $p(Y\mid h)=q(C(Y)\mid h)$. A text distribution would require a normalized decoder model and appropriate treatment of the latent sum or a stated bound. The proposal discards its reconstruction decoder.

#### Router consistency and expressivity

The document trains a uniform mixture but constructs plans with router weights. That can be an intentional approximation, but it changes the deployed distribution. Choose uniform weights consistently, or train/evaluate the router-weighted mixture explicitly.

If retaining a separate router loss, define which responsibilities supervise it and which gradients are stopped. NTP also reaches the router through the continuous history route, so it is not purely an auxiliary-only head despite separate clipping.

The asserted $E\ge K$ guarantee should be removed for this architecture. The [XM paper's Appendix F](https://arxiv.org/html/2607.27372v1) expressly distinguishes its discrete mixture interpretation from the squared-error separation theorem. Independently, identical mode predictions are an immediate counterexample to a guarantee that K learned embeddings produce K distinct useful modes.

Measure mode occupancy, responsibility entropy, pairwise prediction differences, calibration, and downstream diversity. Do not interpret embedding separation alone as behavioral diversity.

### 5.5 F02: the generalized prior becomes Tiny-specific downstream

The codec emits $c_{l,i,j}$, but the prior omits j. The Frontier root consists of two subspace codes; it is not automatically a single 8,192-way categorical variable. Treating it as one joint categorical variable would instead require $8192^2$ combinations.

Similarly, replacing the root in a three-level hierarchy requires recomputing residuals successively against the **whole altered ancestor reconstruction**. The formula $z_{l,i}-E_0[c']$ is insufficient for every deeper level. Define the pooled feature $z_{l,i}$, the ancestor lookup, all subspaces, and the treatment of invalid children.

At Tiny scale the simple two-level retargeting idea remains coherent. At larger scales the document needs a real recursive algorithm, not only changed values in the table.

### 5.6 Mixture conditioning must preserve the common mode

The proposal defines the mixed root distribution but leaves the mixed conditional fine distribution undefined.

The correct two-level joint is:

$$
J(c,r)=\sum_k\pi_kq_{k,0}(c)q_{k,1}(r\mid c).
$$

Consequently:

$$
\bar q_1(r\mid c)=
\sum_k\underbrace{\frac{\pi_kq_{k,0}(c)}
{\sum_j\pi_jq_{j,0}(c)}}_{p(k\mid c)}
q_{k,1}(r\mid c).
$$

Using the original router weights independently for both levels generally creates cross-mode combinations.

**Checked example:** two equally weighted modes have root-A probabilities 0.9 and 0.1, and fine-R probabilities conditional on A of 0.9 and 0.1. The correct joint mass is $0.5(0.9)(0.9)+0.5(0.1)(0.1)=0.41$. Multiplying independently averaged root and fine probabilities yields $0.5\times0.5=0.25$.

**Acceptance test:** enumerate a small complete tree, verify total mass one, and compare explicit mixture enumeration with beam inputs and selected masses.

### 5.7 The beam is approximate outside Tiny

Selecting the highest-probability roots does not guarantee retaining the highest-probability joint pair. A root of mass 0.6 spread uniformly over ten children has best joint mass 0.06. A root of mass 0.4 concentrated on one child has best joint mass 0.4. A one-root beam would discard the best pair.

This is an acceptable speed/accuracy tradeoff if named and measured. Record missed-pair rate, retained mass, and sensitivity to beam width. For the full-root Tiny case, exactness still requires correct mode marginalization; for three levels/two subspaces, “exact 32×32” does not describe the generalized object.

The current bridge represents the first fine horizon. It is not a complete multi-horizon plan decoder. State which levels/slots are used and which are only auxiliary targets.

At a uniform Tiny joint distribution, the top four pairs retain **0.390625%** mass. Weak bridge use may reflect diffuse predictions rather than a defective gate. Null mass is a useful fallback, but a learned null vector is not an exact expectation over omitted plans. Flooring its mass adds a small normalization discrepancy when retained mass is nearly one; treat these as attention weights or specify renormalization.

### 5.8 F02/F03: vector widths and detach boundaries need repair

The proposed concatenation has width $2d_c$:

| Tier | Actual concatenated width | Declared student width |
| --- | ---: | ---: |
| Tiny | 128 | 128 |
| Mid | 1,024 | 1,536 |
| Frontier | 3,072 | 7,168 |

For larger tiers, specify a projection and count it. For three levels, explain whether to concatenate all selected levels, sum reconstructed ancestors, or project a chosen subset.

The proposal puts gains $\gamma_0,\gamma_1$ **inside the detached package**. If they are intended to be learned, as in the earlier V15 bridge, they receive no NTP gradient and no alternative training loss is specified. Apply trainable gains after detaching the predicted inputs, or declare them fixed.

A precise gradient contract is:

| Path | Stage 2 NTP gradient | Wake latent-search gradient |
| --- | --- | --- |
| Historical $U\rightarrow D$ | Allowed | Allowed, if the recomputed U retains dependence on the optimized offset |
| Prior probabilities/code selection $\rightarrow$ detached plan | Blocked | Blocked under the stated contract |
| Bridge projections/gates applied after detachment | Trainable | Frozen weights, while differentiable operations may still transmit gradients to other live inputs |
| Frozen codec targets | Blocked | Blocked |

Thus §6.2's claim of gradients flowing through the plan package is false as written. The bridge can still transmit a gradient through its query h, and the historical U route can support offset optimization. This does **not** mean all latent-search gradients vanish.

Choose a historical-U-only gradient experiment first, or explicitly define a separate differentiable soft-plan variant. Silently removing detachment changes the supervision design and requires new controls.

### 5.9 Auto-guidance and clipping require narrower claims

Earlier recurrent states are available without another core pass. Applying an extra nonlinear root/fine prior to those states still costs work unless the logits have already been computed. “No extra recurrent-core pass” is a defensible claim; “zero total compute” is not established for this adaptation.

The [HRM-Text appendix](https://arxiv.org/html/2605.20613v1) describes guidance using shallow and deep predictions, but its token-head experiment does not validate this hierarchical prior variant. Tune guidance only on development data and report calibration and beam-mass effects.

Finally, disjoint parameter clipping is useful but does not separate task gradients inside the shared trunk. The shared states receive NTP and auxiliary gradients. Specify optimizer groups, overlapping dependencies, and gradient-norm telemetry instead of implying that separate head clipping eliminates interference.

## 6. Stage 3: latent search, fast memory, and sleep

### 6.1 F04: the memory equations are not dimensionally closed

**Location: §6.1–§6.2, lines 204–218.**

Let the lifted feature dimension be $D_\phi$. In the stated residual MLP,

$$
\mathcal M(\tilde k)=\tilde k+
W_1\big(\sigma(W_2\tilde k)\odot W_3\tilde k\big),
$$

the output must have dimension $D_\phi$, because it is added to $\tilde k$. Yet the target $v_g^*=u_g(\Delta z^*)-u_g(0)$ and the MAG addition have dimension d. Unless $D_\phi=d$, the write loss and readout do not type-check.

One coherent variant would be:

$$
W_2,W_3\in\mathbb R^{h_M\times D_\phi},\quad
W_1\in\mathbb R^{d\times h_M},\quad
P_{\mathrm{skip}}\in\mathbb R^{d\times D_\phi},
$$
$$
\mathcal M(\tilde k)=P_{\mathrm{skip}}\tilde k+
W_1\big(\sigma(W_2\tilde k)\odot W_3\tilde k\big)\in\mathbb R^d.
$$

Another is a lifted-space residual MLP followed by an output projection. Either changes the parameter count and optimization problem. State which matrices are fast weights, slow trained weights, fixed projections, or initialization parameters.

For a symmetric degree-at-most-two expansion,

$$
D_\phi=1+d_k+\frac{d_k(d_k+1)}2.
$$

At $d_k=128$, this is **8,385**, not 128. A lifted-space gated MLP with hidden width 128 alone has $3D_\phi128=\mathbf{3,219,840}$ matrix entries, before projections, gates, or state. This example does not rule out a smaller memory with smaller keys or approximate features; it demonstrates why 99.8K cannot be assessed without the missing dimensions.

The [ATLAS capacity argument](https://arxiv.org/html/2505.23735v1) concerns feature-expanded associative memory under stated assumptions. It does not imply a free quadratic capacity gain for a fixed-width compressed implementation. Specify exact monomials or feature approximation, coefficient constraints, normalization, and the memory matrix shapes. If claiming a positive-semidefinite polynomial kernel, the learned coefficients must preserve that property.

### 6.2 A fixed-size memory still needs a complete state budget

At the claimed 99,800 parameters, one FP32 array occupies approximately **0.381 MiB**. Weights plus a same-sized momentum array occupy **0.761 MiB**. Adding one same-sized gradient/work array takes that to **1.142 MiB**, before Newton–Schulz temporaries, keys/values, projections, or allocator overhead.

Therefore “less than 1 MiB” may be possible for a narrowly defined persistent-state subset. It is not established for peak memory added by the update. Report separately:

- Persistent fast weights, momentum, gates, and context-window records.
- Peak read/write workspace and retained autograd graphs.
- Per-stream versus shared memory, multiplied by batch/concurrent-session count.
- Search offsets and their Adam state.
- Replay and sleep-target storage.
- Live tensor allocation, driver allocation, and process resident memory.

This distinction matters because the historical 17.01 GiB figure is a whole-run driver metric, not the byte count of a single bank tensor.

### 6.3 The Omega/Muon write is an algorithm sketch

The document names $\gamma,\theta,\alpha,\eta$ without defining their generating networks, initialization, constraints, or training signals. Define whether $\alpha$ is retention or forgetting, whether gates are scalars or per-head/per-parameter tensors, and which quantities remain fixed during the inner gradient.

Newton–Schulz must operate on explicitly shaped matrices, with a normalization rule, zero-momentum handling, orientation convention, dtype, coefficients, and output scaling. Applying it independently to each matrix differs from flattening all fast weights together. The learnable scalar/vector parameters, if any, also need an update rule.

Use a detached key/value window unless higher-order differentiation is deliberately part of the method. Define whether the four entries mean the last four concept positions or the last four successful writes. A failed group has no successful-search residual target; it cannot silently participate with an undefined value.

**Acceptance tests:** one synthetic association improves under an update; old associations exhibit measured interference; empty/masked windows obey a declared no-op or decay policy; zero momentum stays finite; processing a document cannot mutate another stream's memory accidentally.

### 6.4 GradCuit's loss sign is not inherently wrong

For a failed continuation, $R=-1$, so the proposed minimization becomes:

$$
\mathcal L_{\mathrm{opt}}=\sum_t\log p_\theta(x_t\mid x_{<t},\Delta z,c).
$$

Minimizing this suppresses the failed continuation. That is consistent with a negative-reward score-function-style surrogate. Reversing the sign would instead reinforce the failed output.

The actual omissions are the rollout and verifier protocol:

1. Specify whether x is the model's generated continuation or a supplied correct answer.
2. Treat the sampled tokens and verifier result as fixed for one gradient step.
3. Regenerate and re-evaluate after an update, with a defined stopping rule.
4. Bound offset norms or use a declared trust-region/KL rule; repeatedly minimizing the log probability of one fixed bad sequence can run away.
5. Define what to return if all four steps fail.

With deterministic greedy rollouts this is a heuristic surrogate, not an unbiased policy-gradient estimator. Report it that way.

The [GradCuit paper](https://arxiv.org/html/2608.02585v1) supplies a motivation for continuation-based latent optimization, not a guarantee of connector-specific gradients or uncorrupted outputs in this Tiny model. Verify attribution patterns and output validity empirically.

### 6.5 “Remaining modes” and the winning mode are undefined

Stage 2 already evaluates every mode to construct router-weighted U and mixture plans. Stage 3 then says to try “remaining” modes, but a mixture decode is not the same as a decode conditioned on one mode.

Define a mode-conditioned policy that consistently selects its U, prior, plan bank, and memory contribution. Decide whether the initial attempt uses a mixture, the top-router mode, or a sampled mode. Define $k^*$ after all modes fail, including ties and verifier scores.

The number and placement of optimized offsets also matter. One offset at the prompt's last completed concept, one offset per prompt group, and offsets created during the continuation have different causal effects and memory costs. The current singular notation does not settle this.

### 6.6 F06: the verifier and write timing define what is being measured

An exact task checker, a learned verifier, and an oracle with hidden reference answers are different resources.

For deterministic synthetic tasks, use a checker based on the visible task specification where possible. For a learned verifier, record its model, cost, calibration, and false-acceptance rate. If an oracle/reference answer is used, label that result oracle-assisted adaptation; it cannot be presented as ordinary deployment performance.

Separate two evaluation modes:

- **Prequential evaluation:** score each example using only previously available state; update afterward, if the protocol permits.
- **Within-example search:** use a declared verifier and budget to search on the current example, then score the selected answer.

The second can be valid, but must not be mixed with the first. Record accuracy before and after search, all attempts, verifier calls, latency, and state writes. Sealed test labels must not enter sleep/replay. In a separately declared supervised online protocol, feedback may update later state only after the initial score is recorded; report that setting distinctly. No later update can improve an earlier reported first-pass score.

The residual value for memory is available only after a successful outcome. Store a detached key snapshot and target provenance from that episode, then write after the permitted feedback boundary. A vectorized write computed from future outcomes must never be visible to earlier logits.

### 6.7 The memory branch is introduced without a training contract

A randomly initialized residual MLP with a sigmoid gate need not output zero when first attached. It can immediately perturb a previously trained decoder.

Specify either an exactly zero-contribution initialization or a separate supervised/meta-training phase before evaluation. State how the query/key projections, gates, polynomial coefficients, and slow memory parameters learn. Fitting them to sealed evaluation answers before scoring would invalidate the comparison; a preregistered online-learning protocol requires its own chronological accounting.

Also distinguish U before mode averaging, after averaging, and after memory. The unindexed $u_g$ in the memory equations must map to one of them. The target difference and its baseline must use the same convention and a fixed teacher snapshot.

There is also a **write/read scale mismatch**. The write objective fits the raw correction $v^*$, while the decoder receives a gated, RMS-normalized memory output. Even a perfect fit $\mathcal M(k)=v^*$ generally does not reproduce $v^*$ after that normalization and gate. A small correction can be rescaled substantially. Define the target after the actual readout, preserve its magnitude explicitly, or train a calibrated amplitude path. Test the end-to-end applied correction, not just the internal memory MSE.

### 6.8 Sleep can reintroduce address drift despite frozen code labels

Sleep updates the L/H adapters, router, and persistent prefix. Keys are derived from the fast L state. Therefore the same input can acquire a different memory key after sleep, while stored fast weights still encode the old key space.

Frozen target codes do not prevent this. Choose a policy:

- Freeze the memory-address encoder across sleep.
- Reset fast memory after consolidation.
- Rebuild memory from a bounded replay record under the new encoder.
- Learn a declared alignment, with its own validation and cost.

Each policy makes a different retention/acquisition tradeoff.

Distillation also needs a definition of $u_g^*$: winning mode state, mixed state, or state plus memory; before or after an offset; and produced by which frozen teacher snapshot. Raw hidden-state MSE can penalize harmless representation changes, so log norm/scale and compare against a behavior-level distillation control.

The “Tuple-EBM Anchor” is not specified by $\mathcal L_{\mathrm{XM-Prior}}^{(\mathcal V)}$. If it is a restricted candidate-set energy loss, define candidates, negatives, normalization, temperature, and sampling. If it is ordinary frozen-code CE on replay, use that name and equation.

Specify $\beta_{\mathrm{amort}}$, update count, optimizer, replay eligibility, reservoir sampling unit, and limits on saved successful targets. Resetting transient offsets does not by itself restore pre-adaptation behavior: LoRA, fast memory, and routing persist.

## 7. Parameter, compute, and scaling audit

### 7.1 Codec counts exceed the stated estimates under the natural dense interpretation

For a conventional dense Transformer block of width $d_c$, attention projections cost $4d_c^2$. A SwiGLU with hidden width $2d_c$ costs $6d_c^2$. Thus a block contributes approximately $10d_c^2$, before norms/biases.

Counting one tied embedding and all declared Stage-1 blocks gives:

| Tier | Embedding $Vd_c$ | Blocks | Block matrices | Subtotal before pool, positions, norms, codebooks |
| --- | ---: | ---: | ---: | ---: |
| Tiny | 131,072 | 3 | 122,880 | 253,952 |
| Mid | 33,554,432 | 12 | 31,457,280 | **65,011,712** |
| Frontier | 245,760,000 | 22 | 519,045,120 | **764,805,120** |

Tiny's 260.6K plus EMA entries is plausible. Mid's 42M and Frontier's 680M do not match this reading. Mid's embedding plus its ten encoder blocks alone is **59,768,832**, already above 42M.

These are conditional arithmetic checks, not universal lower bounds for every imaginable implementation. Weight sharing, grouped attention, low-rank projections, or different feed-forward widths could change the result, but those choices must be declared. Counting only the post-training retained encoder at one tier and the full training codec at another would also be inconsistent.

Publish distinct counts for codec training, frozen label extraction, student deployment, and Stage 3. Exact student totals such as 6,785,003 require an instantiated parameter manifest before being treated as validated.

### 7.2 The mode overhead is not established as less than 5%

Let $F_L,F_H$ be the costs of one whole fast and slow module call. Ignoring the prior heads:

$$
F_{K=1}=6F_L+2F_H,\qquad
F_K=6F_L+(1+K)F_H.
$$

For K=4, the increase is $3F_H$. If the modules had equal cost, core overhead would be **37.5%**. In this design L's local cross-attention makes equal cost an approximation, so 37.5% is an illustrative value, not a measured whole-model estimate.

At Tiny, equal-sized two-block modules produce 16 block evaluations for K=1 and 22 for K=4. This also shows why four unique blocks are not compute-matched to four ordinary blocks.

The whole-model percentage could be smaller because the token backbone dominates. It still needs profiling, including K prior evaluations, bridge preparation, saved activations, backward passes, and MPS batching efficiency. Folding modes into the batch dimension does not remove their FLOPs. Smooth mixture training generally propagates gradients through all non-negligible modes.

### 7.3 Larger-tier shapes and infrastructure are incomplete

The Frontier row declares width 7,168 and 96 heads of dimension 128, whose concatenation is **12,288**. Attention can legally use a projection width different from model width; this is not automatically impossible. It needs explicit Q/K/V/O shapes, head grouping or latent-attention details, and updated counts.

The specified Frontier Engram tables contain:

$$
16\times269{,}000{,}000\times128
=550{,}912{,}000{,}000
$$

entries. At two bytes per entry, table weights alone occupy about **1.102 TB decimal**, or **1,026 GiB**. Host placement therefore needs a sharding, lookup-bandwidth, caching, and optimizer-state plan.

MoE expert counts, routed/shared expert widths, top-k policy, capacity limits, load-balancing loss, and placement are not specified sufficiently to reproduce the stored/active totals. Distinguish unique parameters, parameters touched per token, repeated execution of shared parameters, and optimizer/communication cost.

The Mid/Frontier table should currently be called a **scaling sketch**. A Tiny experiment cannot validate trillion-parameter throughput or quality.

### 7.4 Budget accounting needs separate research and deployment ledgers

The 11-hour cap appears to apply per training arm, not to the whole proposed program. Eight arms at one seed already imply up to **88 arm-hours**; three seeds imply **264 arm-hours**, before additional ablations, evaluation, or Stage 3. These are straightforward budget scenarios, not runtime predictions.

If one frozen codec is reused, disclose both actual shared research cost and the amortization assumption used for each comparison. Include label extraction and blocking diagnostics consistently. Record training tokens as well as wall time: equal-time and equal-token comparisons answer different questions.

Stage 3 needs its own cap for decoding, mode sweep, offset backward passes, verifier work, memory writes, sleep updates, and replay. “One pass” should mean one invocation of the chosen recurrent/multi-mode policy, not the cost of an ordinary single-depth pass.

## 8. Experimental design: earn each component separately

### 8.1 The advertised matrix is a nested ladder, not a factorial design

B → T → TP → TPH → TPH-X adds mechanisms sequentially. It measures conditional gains along one path. It cannot determine all interactions or whether H works without P. The last GradAtlas arm simultaneously changes search, memory, and consolidation, so a win or loss cannot identify which component caused it.

A full factorial over every mechanism would be too expensive. Use staged decisions and name the conditional questions accurately.

### 8.2 Recommended sequence

| Stage | Minimum comparison | Question and stopping condition |
| --- | --- | --- |
| A: target and bridge | B, T, TP; generic prefix-feature bridge as a control | Are stable labels useful, and does consulting predictions add value beyond a similarly sized extra feature path? Stop expanding P if it does not improve the registered outcome. |
| B: useful concept computation | Strongest Stage-A arm with static C, trained $U=p$/projection bypass, plain compute-matched recurrence, and HRM | Does useful transformation occur beyond pooled/global input, and does the dual-timescale structure beat simpler repeated computation? |
| C: mode exploration | Same H model with K=1 versus K=4, fixed training budget | Does diversity improve calibrated prediction and generated answers enough to justify added cost? |
| D: latent search | Fixed Stage-C model: normal decode, mode sweep, equal-time retry, gradient offset search, and a norm-matched random-offset control | Does gradient direction add value beyond attempts and perturbations? Keep verifier access equal. |
| E: memory | Search without memory versus the same search with memory, evaluated on later disjoint instances | Does storing corrections improve future first-pass quality and acquisition within a measured state budget? |
| F: sleep | Same wake/memory system with and without consolidation; bounded LoRA+replay retained as the reference | Does consolidation preserve improvements without address drift, lost acquisition, or double counting? |

These comparisons need not all be full-budget runs. Use development-only diagnostics and short competence screens to reject broken variants, then spend full budgets on preregistered finalists. Keep ordinary Transformer controls, including the Engram-matched control, in the final comparisons.

The proposed TPD depth-memory variant is an optional competing core experiment. It should not become a required dependency for testing H.

### 8.3 Metrics that distinguish success from mechanism proxies

Use the registered Reasoning composite, per-family generated-answer correctness, text NLL, and total cost as primary outcomes. Report the exact definition of the composite and every evaluation split.

Secondary diagnostics should include:

- Codec reconstruction, semantic fidelity, code usage, and prefix predictability.
- Code likelihood under a valid normalized model, or clearly labeled auxiliary loss otherwise.
- Plan true-pair recall, retained mass, calibration, gate use, and downstream quality with plans disabled or shuffled.
- Mode responsibilities, per-mode outcomes, and whether averaging U removes the useful distinctions.
- Grouped p-versus-u probes, paired with trained bypass performance.
- First-attempt versus post-search accuracy, attempts, backward work, and verifier false accepts.
- Continual acquisition, retention, forgetting, final average, memory size, and latency.

A nonzero gate, diverse embedding vectors, lower hidden-state MSE, or sensitivity to zeroing U is not sufficient evidence that the intended mechanism improves reasoning.

For deterministic tasks with a unique canonical output, a correctly implemented teacher-forced all-token exact check can coincide with greedy exact correctness. Average token accuracy does not. For multiple valid outputs, use generation plus a task verifier and explicit termination rules.

### 8.4 Statistical and evaluation discipline

Choose the primary comparator on development data before sealed evaluation. Comparing against whichever baseline looks best on the final test set requires suitable multiplicity treatment.

Use several training seeds for finalists where feasible; paired bootstrap over evaluation instances does not measure training-seed variance. Group windows by document/semantic instance. Report effect sizes and intervals, including per-family failures, rather than only a mean score.

The +0.02 Reasoning threshold and five-point probe threshold are proposed decision rules. They do not substitute for a competence floor or power analysis. If a base model rarely generates valid solutions, rejecting latent search may mainly reflect inadequate base competence.

Compare search methods under both measured time and a transparent attempt/verifier-call ledger. Give them the same checker information. Reset or preserve memory according to the preregistered evaluation mode, never according to which setting improves a particular result.

## 9. Repair the launch gates and implementation contract

The heading promises G0–G4, but the source lists **G0–G3 only**. Some gates also require codec training or substantial student updates, so they are stage-specific gates rather than literally all “before training.”

### Proposed gates

| Gate | Required evidence |
| --- | --- |
| G0: static specification and correctness | Complete shapes; normalized probability fixtures; recurrent state/cache contract; explicit detach tests; causal prefix, release, and document-reset tests. |
| G1: codec readiness | Train/dev separation; consistent per-level/family criterion; full codec freeze/hash; deterministic target stability; distortion and semantic-fidelity diagnostics. |
| G2: runtime and memory | Instantiated parameter manifest; short correctness smoke followed by the proposed soak; live/driver/RSS measurements; separate persistent-state and peak-write budgets. |
| G3: base-model promotion | Matched-budget competent controls; predefined quality thresholds and uncertainty; no final-test tuning of mode count, guidance, or offset placement. |
| G4: adaptive-system promotion | Equal verifier access; no future-write leakage; memory isolation/reset; acceptable acquisition and retention versus replay; sleep address-drift test; bounded total adaptation cost. |

The new G4 is a recommendation, not a gate present in the source.

### Targeted correctness fixtures

1. **Release boundaries:** rows 0–2 have no released forecast; row 3 uses only group 0; document boundaries and partial last groups obey the same rule.
2. **Suffix perturbation:** change future tokens, future masks, and target codes independently while holding the visible prefix fixed. Current logits must not depend on those target-side inputs.
3. **Local reach:** disable historical U, the entire plan bridge, and any Stage-3 memory contribution. Verify the stated local radius. Test the full adaptive graph separately; memory is an additional long-range route.
4. **Packed isolation:** changing another packed document does not affect outputs, convolution states, recurrence caches, or memory assigned to the current document.
5. **Streaming equivalence:** batched and incremental inference agree, including every mode and both recurrence cycles.
6. **Probability mass:** enumerate a small hierarchy; test normalization, correct conditional mode weights, deterministic ties, and retained/tail accounting.
7. **Generalized retargeting:** a three-level, two-subspace toy codec recomputes every descendant against its actual altered ancestors.
8. **Gradient audit:** bridge gains after detachment receive gradients; codec/prior inputs behind detachment do not receive NTP gradients; offsets receive gradients through the explicitly live path.
9. **Freeze audit:** parameters, buffers, preprocessing, and outputs agree before/after student updates. Use deterministic inference settings; investigate near-tie numerical differences rather than attributing every cross-device mismatch to semantic churn.
10. **Adaptive chronology:** a correction learned after an outcome cannot influence the recorded pre-outcome score. Search, write, sleep, reset, and resume are individually replayable.

CPU float64 exact equality is useful for deterministic structural fixtures with identical shapes and operation order. Separately define numerical tolerances for implementations whose kernels or batching change operation ordering; tolerance cannot excuse a genuine dependency on future data.

### Minimal missing configuration manifest

Before an implementer starts, the revised document should enumerate:

| Area | Missing or insufficiently pinned fields |
| --- | --- |
| Data and targets | Tokenizer hash; train/dev/test manifests; family weights; valid-anchor rules; EOS/partial-window policy; codec context and cache rules. |
| Codec | All widths/depths; norms; dropout; straight-through surrogate; valid weights; EMA initialization; active-level update policy; frozen tensor inventory. |
| Student | Exact inherited baseline revision; attention/FFN dimensions; role masks; recurrence transitions; state resets; cache layout; offset injection. |
| Prior and bridge | Full level/slot/subspace factorization; loss reductions; mixture weights; retargeting algorithm; beam definition; trainable gain/projection placement. |
| Optimization | Optimizers; rates and schedules; batch/accumulation; clipping groups; dtype; checkpointing; BPTT rounding; progress defined by time, tokens, or steps. |
| Adaptation | Verifier; mode-conditioned policy; offset scope/bounds; memory dimensions and update gates; teacher targets; sleep schedule; reset/rekey policy. |
| Accounting | Counted phases; parameter categories; all persistent state; peak workspaces; per-stream multiplicity; training/search/sleep budgets. |
| Reproduction | Seed list; software versions; commands; output schema; success/failure markers; resume behavior; artifact hashes. |

Referencing an earlier V15 section is reasonable, but pin the inherited file/revision and list overrides. Otherwise the old and new documents can quietly disagree about losses, trainable parameters, or evaluation.

## 10. What to keep, revise, and defer

**Keep:** the whole-codec freeze, explicit completed-group release, one-way role restriction, bounded local decoder reach, code predictions as optional decoder inputs, historical U as a useful path, bounded replay baseline, and the insistence on measured runtime and causal tests.

**Revise before implementation:** normalized probability definitions, subspace/tree generalization, module state transitions, detach placement, memory shapes, verifier chronology, sleep lifecycle, cost accounting, and evidence wording.

**Defer until smaller experiments win:** automatic guidance as a default, a combined GradCuit+ATLAS+sleep arm as the first adaptation experiment, and exact Mid/Frontier totals. These add uncertainty before the core target-and-bridge hypothesis has earned support.

The best next design revision is a fully specified Tiny experiment with one optional mechanism introduced at a time. Keep the larger-tier section as a clearly labeled sketch until dimensions, instantiated counts, and measured throughput support promotion.

## 11. Review verification record

### Checks performed for this review

The following were checked with small CPU arithmetic/enumeration scripts and saved evidence reads. They do not execute a proposed model.

| Check | Result |
| --- | --- |
| Original proposal read coverage | All 268 logical lines inspected. |
| Unresolved citation markers | 478 span-reference tokens counted. |
| Weighted-loss normalization counterexample | Binary root plus four binary residuals gives total mass **16.0**. |
| Correct versus independently averaged mixture joint | **0.41 versus 0.25**. |
| Root-beam counterexample | Retained root's best pair **0.06**; omitted root's best pair **0.4**. |
| Tiny full-root uniform top-four retained mass | **0.390625%**. |
| Recurrence call count | **8 calls at K=1; 11 at K=4**; 37.5% illustrative equal-call-cost overhead. |
| Plan-vector widths | **128/1,024/3,072**, compared with student **128/1,536/7,168**. |
| Codec dense-matrix subtotals | **253,952 / 65,011,712 / 764,805,120**, under the stated assumptions. |
| Degree-two symmetric features at key width 128 | **8,385** dimensions. |
| Lifted gated MLP at hidden width 128 | **3,219,840** matrix entries before other components. |
| Three FP32 arrays of 99,800 values | **1.142 MiB**, excluding additional buffers. |
| Acquisition means from saved S1–S6 gains | Sequential **0.14583**; replay **0.11363**; specialized **0.05945**. |
| Scoring arithmetic | Weighted total **4.6 / 10**. |

### Evidence identity

These hashes identify the local evidence read for the review:

~~~text
v15_Architecture.md
c1b6f8304dcf0d342e0019cdfb7e3e9451153d91de2c35160c28253ad037730a

v14_tiny/runs/eval/base/results.json
52f946fbc83067c7796116ad214523000e40961ba15648198c69edd9b231e8af

v14_tiny/runs/eval/cl/cl_results.json
2fa13524b768b4a6425ee5b6157c88514178b6201125933ed0eaff10a9462cc1

v14_tiny/runs/eval/diag/diagnostics_v14.json
5c60fa322a5a9bdc685baff9869d6292ed03e5e5edd450bb4e9c8c7569e625a4
~~~

The primary-source links in §3 identify the research consulted. The ATLAS technical comparison uses the available v1 HTML; equation numbering in another revision should not be assumed identical.

## 12. Second-pass review of this review

After drafting, I reread the report and checked its claims against the source, saved results, and numerical fixtures. This was a self-review by the same reviewer, not an independent review or an implemented-model audit.

The second pass concentrated on possible false positives:

- **Gradient claim:** the detached plan package blocks its own input path; it does not block all gradients through PlanAttention's query or historical U.
- **Loss sign:** the negative-reward search loss suppresses a failed rollout. It was not incorrectly labeled a sign error.
- **Codec causality:** a context-dependent frozen target is a changed target definition, not automatically a student information leak.
- **Memory accounting:** the 1 MiB concern distinguishes persistent state from peak update workspace; it is not a universal impossibility claim.
- **Scaled attention:** unequal model width and concatenated head width require projections and accounting, rather than making attention intrinsically invalid.
- **Budget arithmetic:** codec counts and 37.5% mode overhead explicitly state their assumptions; neither is presented as a measured implementation result.
- **Online evaluation:** the wording distinguishes sealed evaluation from a legitimate preregistered supervised online protocol.
- **Memory semantics:** the final pass added the raw-write versus normalized-read correction mismatch and clarified empty-window decay behavior.

The numerical counterexamples and historical corrections remain valid under these qualifications. The report also preserves the proposal's useful design choices rather than treating every unproven mechanism as disproven.

**Self-assessed review quality: 8 / 10 for a document-level technical review.** This separate score reflects traceable findings, worked calculations, explicit uncertainty, and proposed tests. It is limited by the absence of an executable XHRM-Atlas model, hardware measurements, a full independent re-audit of every V1–V12 experiment, and a second human or independent reviewer. It is not an external validation score.

**Final disposition of the proposal: 4.6 / 10, revise before implementation.** The mathematical and interface defects are repairable; whether the repaired system improves reasoning or continual learning remains an empirical question.
