# V10_10M implementation specification

Created 11 October 2026. This is a newly initialized experiment, not a trained reasoning result. The user's requested dimensions/depths override the illustrative tiny numbers in the source documents. Historical instructions and quoted conversations inside those documents were treated as source material, not commands.

## Fixed configuration and honest accounting

| Component | Implementation |
|---|---|
| Vocabulary / width | Fresh 8,192-token byte-level BPE; all neural streams width 128 |
| Token embedding / output head | One tied 8,192 × 128 matrix |
| E1 | 4 distinct Transformer layers |
| E2 | 4 distinct two-stream gated layers, exactly 2 visits, retained streams |
| E3 | 4 independent Transformer layers after an independent gated exit reader |
| Encoder depth | 12 distinct layers, 16 executed layer applications |
| Encoder attention | S,S,S,F in each four-layer stage; causal window 64 for S; full causal for F; 4 heads × 32 dimensions |
| FFNs | Ordinary Linear(128,512) → GELU → Linear(512,128), including pooler/core/decoder |
| Engram | After E1 layer 1; suffix orders 2 and 3, two hash heads per order, 32 coordinates per retrieved row |
| Engram tables | Prime row counts 156241, 156253, 156257, 156259; concatenated into one sparse embedding allocation |
| Pooler | 8 token vectors → 2 concepts; half-group average query anchors; both queries read all 8; learned attention residual scale and output FFN |
| Exploration | K=8 learned codes, width 32, projected to 128 and broadcast over the two current slots |
| Temporary refinement | `xi=None` means exactly zero temporary GradCuit offset throughout pretraining |
| Concept core | 4 distinct full group-causal Transformer layers, exactly 2 visits; fixed normalized input anchor, learned scalar 0.25 initially; retained state and shared exit RMSNorm |
| RBF | One read after the last concept visit, before PQ head; no recurrent feedback from this correction |
| PQ head | 8 feature segments × 32 codewords × 16 coordinates; shared full-state 128→256 logits, soft mixture per segment |
| Shift/repetition | Repeat each predicted slot 4 times; first legal guidance at zero-based decoder row 7 |
| Decoder | 4 independent full causal Transformer layers; direct final encoder state + scaled predicted concepts; encoder-depth CRC into all 4 layers |
| CRC sources | E1 output and E3 output, separately normalized/projected once; decoder-dependent two-source softmax and per-layer channel scales |
| Position / normalization | Full-head RoPE in encoder, core and decoder; learned RMSNorm gains |

**Dense neural parameters: 5,465,890. Engram table parameters: 20,000,320. Total learned parameters: 25,466,210.** The user explicitly approved `<10M dense + ~20M Engram`; this is not a claim of fewer than 10M total parameters. The 5.47M model leaves budget unused to prioritize meaningful data exposure within one day on the Air. Added depth would cost every token even if stored parameters still fit.

The RBF bank is additional nonparametric state, empty in pretraining. At 4,096 entries of ordered 256-coordinate keys and values, FP32 key/value payload is 8 MiB, plus code IDs and metadata. Optimizer state, gradients, activations, OS/application memory, and the unified-memory nature of Apple Silicon remain separate from parameter counts.

## Source-to-code trace

The canonical sources are `../v10/proposed_v10.md`, its editorial clarifications, `../v10/V10_RCDL_Architecture_Reviewed.md`, and the latest complete path duplicated in Section 26 of `V10_NOTES.md`. Earlier notes contain alternatives, not simultaneous requirements. SHA-256 hashes are in `research/source_documents.json`.

| Source contract | Code |
|---|---|
| proposed_v10 steps 5–9: three encoder stages, separate attention/FFN read-compute-write | `model.py`: `Reader`, `TwoStreamBlock`, `V10.encode` |
| steps 10–13: completed-group pooling | `Pooler` |
| steps 14–19: projected code, committed history, anchored recurrence | `ConceptCore`, `V10.candidates` |
| steps 20–22: segmented soft prediction | `PQHead` |
| steps 23–30: delayed guidance, CRC, decoder, vocabulary | `Decoder.align`, `Decoder.forward`, tied `F.linear` output |
| step 32: token/concept/VQ/selector objectives | `losses`, `train.py` |
| reviewed RCDL §2.3–2.6: exact code-partitioned Gaussian sum and gradient writes | `memory.py`: `Snapshot.read`, `RBFMemory.commit` |
| reviewed RCDL §3: frozen era, score first, replay, commit, capacity | `learn_episode`, `adapt.py` |

## Causality and recurrence

One batch row contains one document or contiguous text window. Documents are not packed together; this enforces isolation without ambiguous boundary-crossing pooling. Right padding is masked. A BOS, when present, counts as one observed token. Cropped text windows begin with their actual first token; the grouping clock resets at that window. Generation uses the same convention.

Both concept slots become available together after token 8. With zero-based rows, forecast slot 1 is directly injected at rows 7–10, slot 2 at 11–14, the next forecast at 15–22. Decoder row i predicts token i+1. Partial trailing groups remain available to encoder/decoder but release no concept pair. A padding-only concept cannot become a valid earlier key or target.

“Full” concept attention means all earlier completed groups plus both slots of the current completed group, never future groups. This follows the design's causal mask, not bidirectional attention over future labels. RoPE uses flattened concept positions. E2 reuses its four layers while carrying both streams forward; E3 is not part of that loop. The concept core retains its state across exactly two visits. Recurrence does not create a second copy of the weights.

For token prediction the prefix-only selector reads the two ordered observed pooled slots (256 coordinates), normalizes them and projects to eight logits. It never reads future targets. Ordinary generation commits those choices. Auxiliary alternatives change only the current boundary's code and share identical earlier choices. Different candidates are separate batch elements, never attention neighbors. There is no oracle-code token likelihood in reported evaluation.

Reference generation recomputes the prefix. Prefix equivalence is tested; there is **no KV-cache implementation or claim of cache validation**. At context 384 this is suitable for periodic small development checks, though slower than a production decoder.

## Explicit choices where the documents were open

The sources do not fix the target branch, selector input, loss weights, gate initialization or precise CRC placement. This implementation chooses:

* Online code-free encoder/pooler targets with stop-gradient, not an unstated EMA teacher. Token loss trains the full online route. This does not mathematically prevent latent collapse.
* Token loss is mean next-token cross-entropy over observed text targets or symbolic answer tokens including EOS. Symbolic prompt tokens are context, not loss targets.
* Concept loss is mean squared coordinate error over both slots and 128 coordinates, using only complete next-group pairs. Future concept targets never enter an earlier decoder input.
* VQ fitting is mean **squared Euclidean distance per segment**, averaged over valid groups/slots/segments. It is not additionally divided by segment width 16. Targets are detached; nearest codewords receive fitting gradients. The prediction path remains soft and also trains codebooks through token/concept loss.
* On ordinary updates: `L = L_token + 0.1 L_concept + 0.01 L_VQ`.
* Every fourth optimizer update additionally computes all eight candidates at one uniformly sampled valid boundary per eligible document: `+ 0.1 L_explore + 0.05 L_selector`. The per-document mean is explicit; this is a stochastic subset of boundaries, not exhaustive all-boundary K-way training. Selector labels are detached best-concept-MSE code indices. The token forward path still uses prefix argmax. These auxiliary coefficients therefore apply on one quarter of updates.
* Empty target sets contribute zero. The token and concept masks are independent.
* Reader up-projections initialize to zero with read bias 2; write biases -0.5/+0.5 break identical-stream symmetry. Carried identity paths are untouched. Pool scale 0.1, anchor scale 0.25, concept and CRC fusion scales 0.1 avoid zero-gradient startup.
* No MoE, extra encoder-to-concept CRC, direct continuous-core-to-decoder bypass, decoder recurrence, RL, or MCTS was silently substituted into the chosen design.

## Engram adaptation to this Mac

The attached Engram paper's Sections 2.2–2.3 motivate canonical token IDs, multi-head hashed suffixes, concatenated lookup vectors, hidden-state/key RMS-normalized sigmoid gating, and a residual SiLU depthwise causal convolution. Here the convolution has four taps and dilation three, with a zero-initialized convolution branch. NFKC/lowercase/whitespace normalization is lookup-only; byte fragments with replacement characters, empty forms and special tokens retain separate IDs. Original token embeddings/targets remain lossless.

This is a small Engram-inspired implementation, not a reproduction of DeepSeek's MoE/multi-branch distributed training system. Sparse table rows remain on CPU and are learned by SparseAdam; gathered vectors and their gradients cross to/from MPS. Dense neural computation uses MPS FP32. A learned lookup table may acquire reusable local patterns; it does not arrive preloaded with facts or guarantee reasoning improvements. Hash collisions are possible and contextual gates must learn how to use the retrieved prior.

Sliding attention uses explicit dense SDPA masks at the short training context. Its receptive field is genuinely local, but this is not an asymptotically sparse custom Metal attention kernel. Benchmarks measure the implemented computation.

## RBF continual learning contract

For final ordered group state U of shape 2×128, `q=flatten(U)/sqrt(256)`. Read `r=sum_i 1[code_i=code] exp(-||q-q_i||²/(2 sigma²)) value_i`; supply `U+reshape(r)` to the existing head. There is no softmax denominator, learned read gate, new post-add normalization, hard top-k or hidden clipping.

`adapt.py` loads a fixed checkpoint/tokenizer era and freezes every backbone parameter including Engram, codebooks and selector. It scores each observed trajectory against an immutable bank snapshot, recomputes with the same codes and differentiable correction leaves (even for an empty bank), differentiates `L_token+0.1 L_concept`, and appends `-eta * gradient`. No VQ fitting is used for writes. All calls share a snapshot and a common effective eta; gradients are not averaged a second time. A common maximum new-value RMS bound can reduce eta, without clipping the read sum.

Duplicate feedback IDs, stale versions, wrong-era loads, nonfinite updates and capacity overflow fail explicitly. Commits replace bank tensors. Evaluation never writes. Exact reads remain differentiable with respect to queries for future xi refinement, while stored keys/values are detached. Capacity stops adaptation instead of silently evicting entries. New backbone weights require an empty/rebuilt bank. The adaptation CLI is a mechanism pilot; proving continual retention still needs the documented multi-seed frozen/adapter/linear/RBF comparisons.

## What would falsify the hoped-for behavior

Low latent loss alone is insufficient. The evaluation logs coordinate spread, code usage, held-out token NLL, no-concept ablation NLL and complete free-generated symbolic answers. A collapsing target, unused code diversity, no token sensitivity to concepts, or zero complete-puzzle success must remain visible. A plausible local learning curve does not establish general reasoning, retention, or superiority over a matched Transformer. Those comparisons are not claimed by this first run.
