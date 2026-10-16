**Depth Memory — first-principles review and comparison with NCM**

Reviewed 15 October 2026. Source: the user-provided 34-page PDF, *Depth Memory*, arXiv:2610.18809v1. Page numbers below refer to that PDF. I examined its derivations, main results, ablations, efficiency appendix, and selected released source. I also checked its core recurrence against a direct small numerical solve. I did not reproduce model training or benchmark its GPU kernels.

The paper gives a coherent way to retrieve earlier computations across neural-network layers. Its most convincing empirical result is improved language-model perplexity over a standard residual transformer and the selected depth-mixing baselines. Downstream task improvements over the strongest baseline are much smaller. The proposed memory is an additional computational state with an associated hardware cost.

**Which kind of memory is this?** A transformer mixes information across token positions, across feature channels, and across depth. This paper addresses the third axis. For each token, the model keeps a memory of representations computed at earlier layers in the current forward pass. Attention layers still mix information across tokens; the memory itself does not introduce a new cross-token memory mechanism.

It does not directly provide persistent memory across conversations, a longer context window, retrieval from an external database, or continual learning across training tasks. The word “memory” refers to internal activation state. The learned weights still determine how that state is created and queried.

**Start with the residual stream.** In a standard pre-normalized transformer,

\[
x_l=x_{l-1}+\delta_l,\qquad
\delta_l=F_l(\operatorname{Norm}(x_{l-1})).
\]

Thus, the state is the embedding plus the sum of previous layer updates. This preserves an identity path that helps information and gradients propagate. But a later layer sees the accumulated mixture rather than a separately addressable record of each update.

Addition alone does not retain the provenance of arbitrary summands: many collections of vectors have the same sum. Learned networks can still preserve useful information in feature subspaces, rescale it, and cancel it; the standard residual stream is not incapable of selective computation. The proposed improvement is to provide a more explicit retrieval mechanism rather than requiring all retrieval structure to be implicit in the accumulated vector.

Attention Residuals keeps earlier representations and attends over them. This gives direct access but increases stored state with depth in the full variant. Hyper-Connections maintains several interacting streams. The paper's Depth Memory framework describes these alternatives using a collection of states and Read/Write operations. That taxonomy is useful; it is not itself a performance theorem.

**Build an associative memory.** Suppose a layer produces a value vector v and a key k describing where it should be stored. The simplest matrix memory performs

\[
M\leftarrow M+vk^\top,\qquad r=Mq.
\]

Expanding the read gives

\[
r=\sum_i v_i(k_i^\top q).
\]

Each earlier value contributes according to how its key matches the query. This is unnormalized linear attention over depth. It can use signed coefficients, unlike a softmax convex combination.

If keys are orthogonal and the query equals a stored key, retrieval is straightforward. Correlated keys interfere: querying one address also retrieves contributions from other addresses. This motivates fitting a map that accounts for key correlations rather than merely adding outer products.

**Derive the regression update.** The unweighted regression objective is

\[
\min_M\sum_{i\le l}\|v_i-Mk_i\|^2.
\]

The actual method adds positive observation weights, forgetting of earlier observations, and a ridge term that is discounted along with the history. For one memory group, define weighted observations

\[
\bar k_l=\sqrt{\omega_l}k_l,\qquad
\bar v_l=\sqrt{\omega_l}v_l.
\]

The sufficient statistics obey

\[
A_l=\lambda_l A_{l-1}+\bar k_l\bar k_l^\top,
\qquad B_l=\lambda_l B_{l-1}+\bar v_l\bar k_l^\top,
\]

initialized with A[-1]=alpha^(-1) I and B[-1]=0. The minimizer is M_l=B_l A_l^(-1). Positive ridge initialization and positive weights make A positive definite in exact arithmetic.

Maintain P=A^(-1). The Sherman–Morrison rank-one identity gives

\[
\kappa_l=\frac{P_{l-1}\bar k_l}
{\lambda_l+\bar k_l^\top P_{l-1}\bar k_l},\qquad
u_l=\bar v_l-M_{l-1}\bar k_l,
\]

\[
M_l=M_{l-1}+u_l\kappa_l^\top,
\qquad
P_l=\frac{P_{l-1}-\kappa_l\bar k_l^\top P_{l-1}}{\lambda_l}.
\]

The error u asks what the current memory fails to reconstruct. The gain kappa determines how to correct the memory while accounting for the geometry of previously observed keys. P represents inverse weighted key covariance, not a calibrated uncertainty estimate about the model's answer.

The exactness claim is specific: this recurrence solves the stated quadratic objective for the currently produced key–value observations in exact arithmetic. It does not prove optimal language prediction, perfect recovery of every earlier layer, or optimal selection of which information is useful. Keys and values are learned representations, and the whole network remains a nonconvex optimization problem.

The discounted ridge penalty is important. Holding the ridge penalty constant while forgetting observations would add a full-rank term to A's update. The paper explicitly notes that the same rank-one inverse update would then no longer suffice.

**Why this solver fits the depth axis.** Recursive least squares is sequential. Across a long token sequence, that can be inconvenient for parallel training. Transformer layers are already evaluated sequentially, so a recurrence between layers aligns with an existing dependency. This is the main architectural insight: placing a familiar solver on an axis where its sequential nature is less costly.

The model keeps three streams per token: the ordinary residual vector x, matrix memory M, and inverse covariance P. A query derived from the current residual state reads the memory; gated readout enriches the next layer input. The new layer update is added to the ordinary residual stream and written as a value into memory. Write addresses are derived from the updated residual state. The embedding is written initially, and a final read enriches the language-model head input.

The readout is used to construct the layer input; it is not indiscriminately accumulated into x as a second permanent residual update at every read. Keeping this distinction prevents accidental double-counting in an implementation.

Separate groups split the value dimensions, each with its own small address space. With key dimension eight, a group's memory readout lies in a subspace of rank at most eight at a given token and depth. This is a compressed retrievable summary, not a lossless archive of arbitrarily many independent layer outputs. The residual path and subsequent nonlinear computation remain essential.

**What the measured gains amount to.** Table 2 reports these main comparisons:

| Scale and measure | PreLN | Block AttnRes | mHC | LDM |
|---|---:|---:|---:|---:|
| 400M, validation perplexity | 12.73 | 12.12 | 12.23 | 12.08 |
| 400M, average zero-shot accuracy | 49.05% | 50.61% | 49.55% | 50.46% |
| 1.5B, validation perplexity | 9.48 | 9.26 | 9.25 | 9.15 |
| 1.5B, average zero-shot accuracy | 58.41% | 59.42% | 59.68% | 59.71% |

Against PreLN, validation perplexity improves by approximately 5.1% at 400M and 3.5% at 1.5B. Against the strongest alternative on validation perplexity, the improvement is about 0.3% at 400M and 1.1% at 1.5B. The 1.5B average accuracy advantage over mHC is only 0.03 percentage points. Without repeated-seed uncertainty estimates, that tiny accuracy gap is not persuasive evidence of a meaningful capability advantage.

LDM also has the best reported WikiText and LAMBADA perplexities at both sizes. This broader consistency strengthens the language-modeling result. It does not turn the small aggregate benchmark lead into a large reasoning result.

The comparisons have several good controls: matched backbones and data order, learning-rate sweeps for each method, and disclosed baseline implementation choices. LDM adds roughly 1% parameters, so comparisons are close in size but not exactly parameter matched. Training budgets are 50B and approximately 100B tokens. The learning-rate sweeps do not substitute for independent training seeds.

Aspect-ratio experiments extend evidence across several width/depth choices, but do not establish unlimited benefits from depth. At the extreme 400M configurations, Hyper-Connections does better, and LDM's perplexity worsens at the deepest tested shape. The quality experiments reach 1.6B; the larger 14B models in the paper are efficiency benchmarks, not evidence of 14B quality improvements.

**The ablations identify the main contribution more precisely.** At 400M trained on 15B tokens, Table 8 gives:

| Variant | Held-out cross-entropy, lower is better |
|---|---:|
| Standard PreLN | 2.6554 |
| LDM with RLS | 2.5987 |
| LDM with delta-rule updates | 2.6028 |
| LDM with outer-product updates | 2.6037 |
| LDM storing the cumulative residual vector | 2.6227 |
| Memory-only layer input | 2.7296 |
| No post-layer residual addition | 2.6611 |

The RLS improvement over the delta-rule version is 0.0041 nats, compared with a 0.0567-nat improvement over PreLN. In this setting, roughly 93% of the observed LDM-versus-PreLN loss reduction remains with the simpler delta-rule recurrence. This is descriptive arithmetic, not a causal decomposition, because architectural effects can interact.

The strongest message is to preserve the ordinary residual stream and add useful retrieval. An exact solver gives an additional smaller benefit in this ablation. Storing new layer contributions works better than repeatedly storing cumulative states. Dynamic addressing helps. Removing learned write weighting or forgetting has almost no effect here. Table 8 even gives a slightly lower loss for eight groups than the default sixteen; the evidence does not support a universal “more groups is better” rule.

The plausible explanation for the cumulative-state penalty is repeated storage of the same earlier information. That explanation is consistent with the results, but the ablation does not prove it is the only cause.

**The systems contribution is necessary.** A naive implementation stores M at every layer for backpropagation. Since M has D times D_k entries per token, this is expensive. The authors instead save each rank-one update and reconstruct the preceding state during the backward pass:

\[
M_{l-1}=M_l-u_l\kappa_l^\top.
\]

They also defer materializing updates within a short window. Reads use the window's base matrix plus a sum of recent rank-one corrections. This trades full-matrix memory traffic for smaller vector operations. It does not skip transformer layers or change the intended recurrence. Their main configuration uses windows of four updates.

Only the memory-state history is unwound. Ordinary transformer activations and compact per-layer context are still required; inverse covariance information is saved. “Reversible memory” should not be interpreted as constant-memory training of the entire transformer. Finite-precision subtraction also needs numerical validation beyond the exact algebra.

For one token, ignoring saved backward intermediates, state storage scales as

\[
D + D D_k + G D_k^2
\]

numbers for residual, memory, and inverse covariance, respectively. Fixed size means fixed with respect to depth. It still scales with batch, token count, width, groups, and key dimension.

The measured memory costs are explicit in Table 6. At the 1.7B benchmark configuration, training peak per GPU increases from 80.2 GiB for PreLN to 94.0 GiB for LDM, approximately 17.2%. Prefill at batch 64 and 2K context increases from 7.8 to 14.7 GiB. Decode increases from 19.4 to 21.2 GiB. These are particular benchmark settings, not predictions for v14.

Throughput is competitive with richer depth mixers and generally below the standard PreLN baseline in the reported plots. Improving token efficiency can still improve time to a target quality, but the relevant comparison must combine quality curves with actual wall time. A training-token speedup alone is insufficient.

**Source and numerical checks.** I inspected the official release at commit `9a6865feeb43a153a4501cc9a6457e047ffef3c3`. Its reference implementation matches the core RLS equations, stores per-position/group state, initializes memory for a forward segment, and preserves the residual path. It uses FP32 recurrence state for lower-precision model execution. The repository includes separate reference, flexible, fast-training, and fast-inference backends. [Pinned reference implementation](https://github.com/davidgonmar/depth-memory/blob/9a6865feeb43a153a4501cc9a6457e047ffef3c3/pretrain/OLMo-core/src/olmo_core/nn/ldm/backends/reference.py).

The optimized release targets NVIDIA Hopper and Blackwell. The reference path supports CPU/CUDA; no equivalent optimized Apple MPS path is advertised. Therefore the paper's throughput should not be assumed for this Mac. [Pinned hardware and reproduction notes](https://github.com/davidgonmar/depth-memory/blob/9a6865feeb43a153a4501cc9a6457e047ffef3c3/README.md).

An independent 24-step FP64 toy calculation with varying positive weights and forgetting compared the recurrence against directly solving the weighted ridge system after every update. Maximum absolute discrepancy was 5.55e-16; undoing a saved rank-one update had maximum discrepancy 1.11e-16. This confirms the algebra for that small numerical example. It does not validate the released custom backward kernels, BF16 training stability, or reported benchmark quality. Numerical results are saved in `2026-10-15_numerical_checks.json` beside this review.

**What would strengthen the evidence.** Repeated training seeds would help interpret small differences among strong baselines. Equal-wall-time or equal-FLOP quality comparisons would clarify the practical benefit. A parameter-matched ordinary transformer would control the extra address-network capacity. Retrieval diagnostics could measure how well earlier layer updates remain recoverable and whether the retrieved information causally improves predictions. Tracking covariance conditioning and unwind error would help characterize numerical robustness at greater depths. These are proposed checks, not claims that the authors' experiments failed.

**How the two papers fit together.** NCM changes the order in which an output is generated. Depth Memory changes the information available while a network computes a representation. V14's next-concept objective changes what its representation is trained to predict.

| Dimension | NCM | Depth Memory / LDM | Current v14 |
|---|---|---|---|
| Progression | Coarse-to-fine output levels | Earlier-to-later layers | Future token-group horizons |
| Stored or predicted object | Discrete residual codes | Continuous layer-update associations | Quantized future representation targets |
| Main purpose | Fewer sequential generation rounds | Better reuse of previous computation | Useful predictive concept representations |
| Directly demonstrated | Generation quality–speed tradeoff | Perplexity gains with extra state | Determined by v14's own registered evaluation |
| Principal unresolved issue | Semantic meaning and factual fidelity | Benefit per hardware cost and solver complexity | Predictive and causal usefulness beyond shortcuts |

They are architecturally compatible in principle: an NCM generator could use LDM internally. Compatibility does not establish additive benefits. Both introduce information bottlenecks, learned routing, and extra computation whose interactions need measurement.

For v14, first finish the current registered evaluation. If the concept targets are not predictably useful, adding depth memory may improve a decoder while leaving the concept hypothesis unresolved. If a stable concept path is useful but information becomes difficult to access across substantial depth, a small depth-retrieval experiment becomes more motivated. V14's shallow tiny modules also differ substantially from the 24–28-block main models here.

A controlled follow-up could compare the existing residual path, a small gated memory with a simple update, and RLS memory. Keep data order, causal boundaries, optimizer budget, and evaluation fixed; include a parameter-matched control and report both equal-token and equal-time results. A simpler memory is justified as a comparison by the paper's own small solver-specific gain, not as a guaranteed substitute. Test one architectural change at a time before combining it with a new concept hierarchy.

Neither paper justifies treating active code utilization, low intermediate loss, or a learned memory matrix as sufficient evidence of understanding. The decisive question is whether the information preserved by the mechanism produces measurable, reliable improvements on held-out tasks at an acceptable computational cost.
