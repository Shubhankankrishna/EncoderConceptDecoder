"""Write the v3 proposal and evidence ledgers. Reads only saved audit evidence."""
from pathlib import Path
import json,statistics,math,re
A=Path(__file__).resolve().parent;ROOT=A.parent;S=A/'source_snapshot'
snap=json.loads((A/'snapshot.json').read_text())
depth=json.loads((A/'checkpoint_001703_diagnostics.json').read_text())
inv=json.loads((A/'inventory.json').read_text())
def render(x):
    x=x.replace('{{ROOT}}',str(ROOT)).replace('{{A}}',str(A)).replace('{{S}}',str(S))
    return re.sub(r'\]\((/[^)\n]+)\)',lambda m: '](<'+m[1]+'>)' if ' ' in m[1] else m[0],x)

proposal=r'''# Proposed v3: a small, fully trained recurrent scratchpad

**Proposal date:** 29 September 2026. **Evidence frozen:** 10:20:29 IST, with v2 official evaluation through step 1703 and training summaries through step 1750. **Status:** design and mechanical checks completed; v3 has not been trained. The active v2 run was not changed.

## 1. The design I recommend

**Build v3 around v1's working 384-wide encoder, zero-initialized dense scratchpad, and confined decoder. Keep recurrence short, give the final task loss an uninterrupted gradient through every pass, and improve the supervision of actual answers.** Do not carry the D6 expert pool, detached windows, NextLat, or final-window consistency into the default recipe.

This is deliberately a v1-compatible architecture with a revised training and evaluation design. The evidence does not justify another wholesale replacement of the components that already learned useful retrieval. The main proposed improvement over v1 is better allocation of training signal and recurrent depth, not more architectural machinery.

The recommended model has **25,052,781 compute parameters and 78,578,925 stored parameters**, under the repository's convention. It has one shared dense concept block, not four blocks repeated 8–12 times. Ordinary training microbatches use two passes 75% of the time and four passes 25% of the time. Microbatches containing designated multistep tasks use four passes. All passes use full backpropagation. Inference exposes a normal two-pass mode and an explicitly requested four-pass mode; both must be evaluated.

Your interpretation of `S` is right: it is a writable scratchpad, while `C` is reread as evidence. **I retain `S₀ = 0`.** However, two claims must be separated: v1 learned a useful scratchpad representation, but its completed-model results did not show a substantial benefit from extra recurrence. v3 should preserve the first achievement and test the second, rather than assume it.

The proposal has three non-negotiable success conditions:

1. Recover v1-level language and retrieval learning under a comparable budget.
2. Improve complete-answer performance without hiding a language regression behind an aggregate score.
3. Pay for additional passes only when they measurably improve quality.

There is no defensible guarantee that an untrained design will outperform v1 or v2. The architecture below is the candidate I would build first; the acceptance gates determine whether it deserves to be called a better model.

## 2. What was actually reviewed

The current inventory covers **382 non-excluded workspace files**: v1 and v2 code/configurations/logs, retained checkpoint metadata, shared data, concept and decoder experiments, proposed D1–D6 designs, model budgets, launch/profiling/test records, dashboard consumers, design documents, and literature files. Project text was captured and Python/JSON/JSONL parsed; model, data, training, evaluation, and experiment-result paths received semantic review. All current v1/v2 train and evaluation records were read into a frozen snapshot.

This is not a claim that every dependency, binary tensor element, or paper was reread line by line. Virtual environments, vendor/cache folders, and assistant metadata were excluded from semantic review. The earlier full-data audit and paper-review notes were reused for unchanged material; the coverage ledger states that explicitly. Old design scores are treated as opinions, not experimental results.

The new work includes a CPU-only depth/branch sweep of v2 checkpoint 1703 on the same eight sampled validation rows per language used in the earlier audit. I also checked the proposed retained architecture's parameter count, full-gradient path, causality, document isolation, batch-order invariance, plan indexing, padding-mask examples, and answer-loss accumulation on a small float64 model. These checks establish mechanical consistency, not learning quality.

Evidence navigation:

- [Latest v1/v2 snapshot and matched windows]({{A}}/snapshot.json)
- [Latest checkpoint depth/branch measurements]({{A}}/checkpoint_001703_diagnostics.json)
- [Results and interpretation ledger]({{A}}/RESULTS.md)
- [Every-file inventory and coverage]({{A}}/FILE_COVERAGE.md)
- [Mechanical design checks]({{A}}/design_checks.json)
- [Previous underperformance investigation]({{ROOT}}/v2_underperformance_review_2026-09-28/REPORT.md)

## 3. What v1 and v2 now tell us

### 3.1 The v2 deficit persists beyond warm-up

| Comparison | v1 | v2 | Interpretation |
|---|---:|---:|---|
| Validation around 2.1 training hours | 4.1221 at step 365 | 5.1363 at step 267 | Early gap was not just a misleading mature-model comparison |
| Validation around 13.55 training hours | 2.8816 at step 2559 | 3.3952 at step 1703 | v2 remains 0.5136 nats worse at almost equal time |
| Training NTP, steps 1701–1750 | 2.3246 | 2.5563 | v2 is also worse on the same epoch-0 data window |
| Time to step 1750 | 9.133 hours | 13.949 hours | About 52.7% more time for v2 |
| Recall at roughly 13.55 hours | 94.53% | 0.78% | The retrieval-learning delay is now substantial |

The official step-1703 v2 language loss is 2.5744 on TinyStories and 4.2161 on WikiText, averaging 3.3952. It lies above both neighboring v1 evaluations at steps 1593 and 1792, which score 3.0823 and 3.0316. Thus even a simple same-step bracket establishes a gap of approximately 0.31–0.36 nats without pretending an interpolated value is a measured checkpoint.

The late comparison is not a controlled single-variable experiment. At step 1750, v2's LR is approximately `4.95e-4`, versus v1's `3.87e-4`; the full schedules now differ materially. Width, recurrence, experts, losses, and runtime differ too. The comparison demonstrates underperformance, not its unique cause.

### 3.2 v1's useful concept route is established

The completed v1 test evaluation gives:

| Test | Intact v1 | Core off | Continuous `u` off | Predicted concept `c_pred` off |
|---|---:|---:|---:|---:|
| TinyStories NTP | 1.9768 | 2.5548 | 2.4881 | 2.0008 |
| WikiText NTP | 3.6290 | 4.0843 | 4.0586 | 3.6490 |
| Long-range recall exact | 97.07% | 0.39% | 2.54% | 95.90% |
| Three-token carry exact | 58.01% | 0% | 0% | 42.19% |
| Four-query multi-value exact | 8.20% | 0% | 0% | 1.37% |

The continuous plan is important. The PQ prediction branch has a smaller language effect but materially helps some complete-answer tasks. Therefore v3 retains both branches initially. Removing PQ because it looked unhelpful in one early v2 language sample would discard contrary mature-model evidence.

The v1 continuation's best validation is 2.7287, but that checkpoint includes additional training beyond the main run. Its score is not a fair scratch-training baseline for early v2 or an untrained v3.

### 3.3 Useful scratchpad does not imply useful extra depth

At v1's final checkpoint, TinyStories NTP is 1.9780 at R1 and 1.9768 at R4; WikiText is 3.6295 versus 3.6290. Recall is almost unchanged. Carry exact is slightly better at R1 than R4 in the stored test. The v1 core is useful, while repeated applications mostly reproduce a similar answer.

The state evidence agrees: v1's average state norm grows from about 9.77 to 69.41 over eight passes, while the angle between successive update vectors shrinks from 6.70° to 0.11°. Growing state magnitude is not the same as accumulating new reasoning. Importantly, the measured mature-model pass-1/pass-4 gradient ratio is only 1.14. That does **not** justify diagnosing a live gradient catastrophe merely because zero-state RMS normalization has a potentially large derivative.

The isolated concept experiments are encouraging but limited. The zero-state L1S model generalizes beyond its trained hop depths on a small synthetic task, but six-hop R8 accuracy varies markedly between seeds: approximately 77.1% versus 21.6% in the saved review. Learned nonzero initialization helps some isolated runs, so zero initialization is not proven universally optimal. In the end-to-end tests, extra passes were largely flat even when the concept-only model benefited. v3 must pass end-to-end tests, not inherit the best isolated result as a promise.

Sources: [v1 final test and loop dynamics]({{S}}/run_m2_25M/runs/main/eval_final.json), [isolated initialization review]({{S}}/Concept Design/RESULTS_review.txt), [end-to-end concept results]({{S}}/Concept Design/RESULTS_E2E.txt).

### 3.4 New v2 depth measurements argue against removing recurrence categorically

Fixed checkpoint 1703, eight validation rows per language; language losses pool valid token sums/counts within each language. These are diagnostic samples, not the full official 128-row evaluation.

| Passes | TinyStories NTP | WikiText NTP | Mean sampled NCP |
|---:|---:|---:|---:|
| 1 | 2.8226 | 4.3785 | 3.6060 |
| 2 | 2.7713 | 4.3517 | 3.5859 |
| 4 | 2.7405 | 4.3427 | 3.6000 |
| 6 | 2.7321 | 4.3409 | 3.6220 |
| 8 | 2.7298 | 4.3403 | 3.6353 |
| 12 | 2.7293 | 4.3400 | 3.6443 |

R6 improves every one of the 16 sampled rows over R1. Mean language improvement is **0.0640 nats**; R6→R12 adds only **0.00188 nats**. NCP is best at R2 and then gets worse. Thus the claim “v2's loops destroy everything” is false on this measurement. The problem is that the overall training system spends a great deal to obtain a small incremental language benefit and has not learned useful retrieval at v1's rate.

The latest branch ablation also updates yesterday's evidence: turning `c_pred` off now **worsens** sampled language NTP on both datasets. Its contribution has become useful with training. This reinforces retaining the branch rather than designing around an early transient.

![Updated evidence for the v3 design]({{A}}/evidence.png)

### 3.5 Which earlier concerns remain relevant?

- **Objective competition:** four checkpoint-267 microbatches showed opposing NextLat/NTP gradients on several shared core/decoder matrices. Those gradients were not remeasured at 1703, so they are historical mechanistic evidence, not a claim about today's exact gradient field. Separate decoder experiments also found NextLat neutral to slightly negative on the available tasks. Default-off is justified; permanent rejection is not.
- **Routing:** prefix-ordered expert capacity caused batch-row bias in the earlier diagnostic. The dispatch code is unchanged. Aggregate assignment drop has improved to about 6.2% at step 1750, so it is wrong to repeat the early 18–29% real-row measurements as today's measured rates. v3 avoids the mechanism with a dense FFN.
- **Credit assignment:** D6 still detaches after early windows. Its unique initial-state projection cannot receive final NTP gradients. Shared later loop applications do receive them. v3 removes the detach boundaries and does not need that separate initial projection.
- **Codebook:** v2 now has 99.32% mean code usage, 96.88% minimum-segment usage, and explained variance 0.6473. This is not a collapsed codebook. Healthy compression has not ensured useful answer retrieval.
- **Memory/runtime:** v2's step-1750 throughput is about 1,217 slots/s for its last 25-step interval; the matched 50-step throughput is about 1,141. System swap is above 13 GB in the recent logs. Swap is system-wide; it does not establish which process caused it. v3 should retain a measured memory margin rather than spend it on a larger lookup table by default.

## 4. First-principles requirements

### Requirement A: retain information before trying to reason over it

The encoder must preserve identities, value sequences, order, and query context. A larger recurrent core cannot recover details that its input representation has discarded. v2 cut the combined encoder/decoder non-embedding capacity by 52.7% and tried to compensate with more expert/loop computation. That tradeoff is not supported by the current outcome.

**Decision:** restore width 384 and v1's encoder/decoder dimensions before inventing a more complex middle module. Keep the 50.33M Engram table that already fit and trained; the 80M table is not a prerequisite for a good scratchpad.

### Requirement B: distinguish evidence from working state

Let `C` be the fixed encoder concepts for the current input, and `S` the writable state. `C` is reread every pass and is differentiable back to the encoder. `S₀ = 0` says no computation has yet been written into the scratchpad; it does not mean the model lacks input. The first pass reads `C` and writes a nonzero state.

**Decision:** preserve the two-stream read and identity residual write from v1. No learned seed, latent branches, or synthetic “thought token” is required for the first candidate.

### Requirement C: useful operations must receive credit from the actual task

If the last token loss depends on a sequence of useful state updates, its gradient should reach that sequence. When depth is only two to four passes, the memory saving from cutting the graph is a weak reason to deprive earlier operations of that credit.

**Decision:** full backpropagation through all executed passes, including the input concepts. Only the codebook target/lookup detach rules remain. No detached recurrent windows.

### Requirement D: the loss must value answering, not only easy surrounding text

Most token positions in a long retrieval example are evidence, filler, and prompt. A three-token answer in a 400-token document contributes roughly 0.75% of that document's unweighted token mean. Successful filler prediction can coexist with a wrong complete answer. This arithmetic is illustrative, not a measured corpus-wide fraction.

**Decision:** retain the ordinary language objective and add a small, explicitly normalized loss over annotated answer spans. The term uses correct output labels, not a second evolving latent teacher. Its benefit still needs an ablation and a language-regression gate.

### Requirement E: additional compute must earn its budget

Reusing parameters saves stored weights; it does not make their execution free. Neither changing expert IDs nor changing latent angles proves that the task improves.

**Decision:** train a short loop, report quality versus actual time and tokens, and compare with a separately trained one-pass control. A four-pass mode is not promoted just because it exists.

## 5. Exact architecture specification

### 5.1 Chassis and dimensions

| Component | v3 default |
|---|---|
| Vocabulary / tokenizer | Existing 8,192-token tokenizer, unchanged canonical IDs |
| Context / chunk | 1,024 tokens / four tokens per concept |
| Token and concept width | 384 / 384 |
| Encoder | Eight layers, `SSSFSSSF`, v1 two-stream local/global wiring |
| Attention | Six query heads, two KV heads, head dimension 64 |
| Dense FFN | SwiGLU width 1,056 |
| Engram | Eight tables × 131,071 rows × 48 dimensions = 50,331,264 parameters |
| Concept core | One shared dense full-causal `LoopLayer`, reread `C`, write `S` |
| Initial state | Exact zeros, shape `[B, T/4, 384]` |
| Concept attention | v1 gated attention, NoPE, learned per-head temperature, document-aware short convolutions |
| PQ codebook | 48 segments × 128 codes × eight dimensions = 49,152 parameters |
| Concept prediction | Distance logits and soft code expectation, evaluated at terminal depth |
| Decoder | Four confined layers, width 384, local window eight, additive plan injection |
| Token head | Tied embedding projection |
| MoE / NextLat / consistency / latent branching | Absent from default v3 |

This matches v1's parameter shapes. It permits an exact architecture control and a strictly checked warm start. Architecture compatibility does not mean the changed training objective is equivalent.

```mermaid
flowchart LR
    T[Observed tokens] --> E[384-wide encoder]
    E --> L[Local token features]
    E --> C[Read-only concepts C]
    Z[Zero S] --> R[Shared dense recurrent block]
    C --> R
    R --> S[Updated scratchpad S]
    S --> R
    S --> U[Terminal readout u]
    U --> Q[Predicted PQ concept]
    U --> P[Target-aligned plan]
    Q --> P
    L --> D[Confined decoder]
    P --> D
    D --> O[Next-token distribution]
```

### 5.2 Scratchpad update, precisely

For a stream vector `z`, define:

`n(z) = z / sqrt(mean(z²) + 1e-6)`

`g_i(z) = sigmoid(U_i V_i n(z) + b_i)`

`Read(C,S) = ½ [g_C(C) ⊙ (1+γ_C) ⊙ n(C) + g_S(S) ⊙ (1+γ_S) ⊙ n(S)]`.

There are separate read parameters for attention and FFN. The low-rank factor has rank 32. The write scalar is:

`Write(x) = 2 sigmoid(wᵀx + b)`, initialized to 1.

One pass is:

```text
x = Read_attention(C, S)
S = S + Write_attention(x) * FullCausalGatedAttention(x, document_mask)
x = Read_ffn(C, S)
S = S + Write_ffn(x) * SwiGLU(x)
```

The same block parameters are reused at every pass. `C` is not overwritten or detached. Readout is `u = zcRMSNorm(S_K)` after the chosen terminal pass. There is **no carry normalization, clipping, overwrite gate, or forced contraction between passes** in the default. Those would change the already-tested v1 update rule.

Why keep this seemingly simple update? It has an identity path for the current state, state-dependent content retrieval, and a fresh read of the evidence. It can implement repeated work. It can also learn to repeat almost the same work; the training/evaluation design must distinguish those outcomes. No claim is made that one pass equals one logical hop.

### 5.3 Initialization and the zero-state objection

Keep v1's actual initialization: nonzero standard attention/FFN projection weights, read-gate biases +2, zero-centered norm gains initialized to zero, write gates initialized to 1, and residual short-convolution kernels initialized to zero. Zero convolution kernels mean identity convolution, not zero attention.

With `S₀ = 0`, the first read is approximately `0.5 × sigmoid(2) × n(C)`, which is nonzero. Therefore the first attention/FFN pass can populate the scratchpad. **Do not combine zero scratchpad with zeroing all write-path output projections and a closed write gate.** That would introduce a different startup behavior and potentially delay useful gradients.

The half-scale first read is retained as part of the v1 control. A “fix” that doubles it, adds learned initialization, or floors state normalization changes the experiment. The isolated learned-initialization results are a reason to test such an arm later, not to combine it silently with the main proposal.

### 5.4 Preserve the decoder contract and causal boundary

At token position `t`, the model predicts token `t+1`. The latest allowed concept index is:

`j*(t) = floor((t+1)/4) - 1`.

The plan is valid only when that concept belongs to the same document. Its latest source token is `4j*+3 ≤ t`. The first three predictor positions of each aligned document use the null plan. This prevents pooling future tokens from leaking the answer into its own prediction.

The plan is:

`P_t = W_u u[j*(t)] + W_c c_pred[j*(t)] + offset[(t+1) mod 4]`.

The confined decoder starts `D_t = tok_t + P_t`. Its local attention keys and values come from the plan-free local token features; queries come from the private decoder state. Keep this confinement. The older separate-plan-read arm failed in one of two seeds, and the stacked alternative did not clearly beat the confined decoder in the available screen.

All concept attention and short convolutions respect document boundaries. A future-token edit must leave earlier logits unchanged, and changing a different packed document must leave the current document unchanged. Training must also be invariant to batch-row order: dense FFNs avoid v2's capacity-based exception.

Source of the retained mechanism: [v1 concept core]({{S}}/run_m2_25M/code/concept_core_v2.py:15), [encoder]({{S}}/run_m2_25M/code/hrm_encoder.py:298), [plan indexing and decoder]({{S}}/run_m2_25M/code/hrm_decoder.py:74).

## 6. Recurrent depth and gradient flow

### Default training depth

- For ordinary microbatches, sample `K=2` with probability 0.75 and `K=4` with probability 0.25.
- If any document in the microbatch is tagged `lr_hop`, `pointer_chase`, or `graph_walk`, use `K=4` for the entire microbatch.
- Sample depths with a separately checkpointed RNG. Log both microbatch counts and token-weighted depth counts.
- Compute NTP, NCP, and the answer term at the terminal depth only. Do not ask every early state to predict the same final answer, and do not add a penalty requiring intermediate states to match the terminal state.
- Backpropagate once through the complete two- or four-pass graph. Gradient accumulation is across independent microbatches, not across detached windows of the same example.

The ordinary mean is 2.5 passes. If a fraction `p` of microbatches is forced to four passes, the mean is `2.5 + 1.5p`, assuming the ordinary draw is independent. The actual token-weighted mean must be reported; it cannot be inferred from task token percentages because packing changes how often a microbatch contains a designated task.

The task tag affects compute, not model input. It comes from the known training generator; it must not encode the answer. At inference there is no oracle task label or gold hop count: evaluate all cases in fixed R2 and fixed R4 modes. A user-selected “more computation” mode is allowed; choosing the winning R after seeing a test answer is not.

Why not train the hard cases equally at R1? Doing so rewards compressing the entire answer into a one-pass route even when the intended mechanism is iterative. This is a hypothesis about incentives, not proof that the new schedule will create an algorithm. The encoder already has global processing, so one-pass shortcuts may remain possible.

For `S_r = Fθ(C,S_{r−1})`, the terminal task gradient reaches earlier state via products of the intervening Jacobians. The shared parameter gradient also sums contributions from each application of `Fθ`. Full backpropagation preserves both paths. D6's state detach removes the across-window state path even though later shared-weight uses remain trainable.

The small-model mechanical check found nonzero final-NTP gradients at all four retained states: approximately `[0.0587, 0.0336, 0.0273, 0.0261]` in that test. These are not predicted full-model gradient magnitudes. They confirm that the specified graph contains the intended paths.

No claim of guaranteed contraction, bounded state norm, or monotonic improvement is made. R8 is an extrapolation diagnostic only. Do not advertise unlimited test-time reasoning before it is measured.

## 7. Training objective: direct task credit with a small concept auxiliary

The proposed objective is:

`L = L_token + λ_answer(s) L_answer + 0.1 L_NCP(final) + L_VQ`.

`λ_answer(s) = 0.1 × min(1, s/300)` for scratch training, where `s` is completed optimizer updates. This ramp is a proposed conservative setting, not an experimentally optimized value.

### 7.1 Token loss

Let `m_t = same_document(t,t+1) AND target_token(t+1) != PAD`. EOS is a valid target.

`L_token = sum(m_t × CE_t) / max(1, sum(m_t))`.

The sum and denominator cover the full accumulated optimizer batch. Precompute the valid-target counts across its microbatches, then weight each microbatch's numerator by the common denominator. This avoids giving a heavily padded microbatch the same weight as a full one. Every comparison arm must use the same corrected normalization; it is a separate change from the architecture.

### 7.2 Answer loss

For synthetic document `i` with `Q_i` annotated answer spans, define:

`a_i = (1/Q_i) × sum_q [ (1/|span_iq|) × sum_(t+1 in span_iq) CE_t ]`.

Then:

`L_answer = mean_i(a_i)` over synthetic documents with valid, nonempty answer spans in the full optimizer batch. If there are none, the term is exactly zero. Language documents contribute to `L_token`, not to this extra term. Documents without reliable annotations must not receive guessed labels.

This gives a multi-token answer a coherent training signal without letting its length alone determine its weight. Averaging queries inside a document also prevents a four-query example from automatically receiving four times the document weight. The ordinary token term still trains the full surrounding text. Answer tokens are intentionally counted in both terms; the extra weight is the point.

At `λ_answer=0.1`, a document with few answer tokens can get much more answer emphasis than under NTP alone. That can help retrieval or hurt language. It is a deliberate tradeoff to be measured. Start with 0.1, compare with zero, and only investigate 0.03 or 0.3 if the initial controlled comparison warrants it. Do not introduce a large adaptive loss system before this basic comparison.

The existing shards contain token/document arrays but do not preserve the necessary training answer-span sidecars. **That metadata is a prerequisite.** A loader cannot simply infer all answer spans by looking for `=` or a question mark. Section 8 specifies how to create it without contaminating inputs or tests.

### 7.3 NCP and VQ

Keep the PQ vocabulary and final-pass coefficient 0.1 because the mature model uses the predicted-concept branch on several tasks. Do not apply the NCP target to every pass; the v2 curves show that intermediate depths and final token decoding prefer different states.

Define a nonempty chunk as containing at least one non-PAD token, and a complete real chunk as containing four non-PAD tokens. Use NCP only when the current chunk is nonempty, the target chunk is complete, and both belong to the same document. The target remains the next chunk's detached PQ assignment. Averaging is over valid pairs and then over the 48 code segments.

For codebook initialization, hit counts, VQ fitting, and dead-code replacement, use complete real chunks only. This deliberately excludes partial end-of-document chunks from the auxiliary codebook objective. Those tokens still participate in NTP, and the model still encodes every chunk required by the decoder. The valid-pair count and excluded fraction must be logged.

For each valid chunk, `L_VQ` sums squared error over the eight dimensions of each selected code vector, then averages over the 48 segments and valid chunks. Use common full-optimizer-batch denominators for both NCP and VQ, as for the token loss. This preserves v1's per-code error scale while correcting padding and unequal microbatch weighting.

VQ sees detached encoder concepts; the NCP distance head sees detached code vectors. Thus VQ updates the codebook, while NCP and token losses train the encoder/core/decoder through their permitted paths. Keep the separate gradient-norm diagnostic for the codebook. Initialize scratch-training codebooks with v1's k-means++ procedure on complete real chunks from the first optimizer batch; require at least 128 candidates per segment and use training data only. Preserve the source codebook for warm starts.

Retain the 100-optimizer-update dead-code check interval. After the optimizer update, restart codes unused during that interval using complete real concepts from the accumulated batch, and reset the restarted rows' first/second optimizer moments so old momentum does not describe a newly sampled vector. If no complete real candidates are available, defer the check and preserve hit counts. Apply this policy equally to the corrected control and v3.

No-valid-pair batches return a finite zero auxiliary loss with zero gradients, not NaNs. The mechanics check includes both a valid real pair and excluded PAD/cross-document pairs.

### 7.4 What is deliberately absent

No NextLat hidden-state loss, NextLat KL, MoE balance/z losses, state consistency penalty, forced angular-change penalty, entropy-maximizing routing objective, or intermediate-state identity target. These are not all inherently bad; none has earned a place in the default local recipe. The goal is to make final answer and language objectives interpretable before adding another competing target.

## 8. Data and curriculum policy

### Preserve the comparison corpus first

Keep the existing tokenizer and initial 45% language, 22% symbolic, 13% retrieval, 10% code, 10% long-range-probe token mixture. Keep the original language sources and task distributions for the first controlled experiment. Do not simultaneously introduce a new corpus and claim an architectural win.

Re-emit generator metadata alongside the same synthetic token documents: family, generator seed, tokenized answer spans, query count, hop count when defined, source identity, and document hash. Pass these through the packer's existing source-offset tracking. Emit immutable sidecars indexed by packed row and document offset. Language splitting must preserve source-document identity; synthetic documents remain unsplit.

The regenerated token and document arrays must match the existing shards byte for byte before calling this “the same dataset.” If deterministic regeneration cannot reproduce them, label the result a new dataset version and rerun both the corrected v1 control and v3 on it. Never overwrite the existing data folder or use an approximate answer parser to hide this difference.

The data loader should return `ids`, `doc`, Engram indices, answer-span metadata, and the hard-task flag. Only `ids`, `doc`, and indices enter the forward model. Masks/tags are trainer metadata. Unit tests must verify the answer-token index `i` maps to logits at `i−1` and never supplies the answer as an earlier model input.

### Do not confuse padding cleanup with a new language advantage

The earlier full audit measured 3.817% all-PAD training NCP target pairs, versus about 9% in validation/test shards. Fixing this makes the auxiliary target more meaningful, but v1 inherited the same issue and still learned retrieval. Padding cleanup is not claimed as a complete explanation of v2's deficit. Compare all new arms with identical masks and report legacy metrics separately when needed for continuity.

### Curriculum is an optional second experiment

The default proposal does **not** replace the current data with an unvalidated easy-to-hard schedule. The saved concept-only tests show that a curriculum can help isolated hop composition, while end-to-end gains were much weaker. First test the answer term and short full-gradient loop on unchanged examples.

If annotated answer loss falls but compositional tasks still fail, use a versioned second-stage curriculum confined to the existing long-range allocation: single-value recall, then multi-token carry/multiple keys, then mixed two-to-four-hop tasks. Keep previously learned tasks in the mixture. Promote stages using a dedicated development set, not test/OOD success. Any increase in the number of supervised synthetic answers must also be given to the control.

## 9. Optimization, schedule, and resource budget

### Scratch-training recipe

| Setting | Proposed value |
|---|---|
| Optimizer | AdamW, betas `(0.9, 0.95)`, epsilon `1e-8`; decay 0.1 with v1's parameter-group exemptions |
| Peak LR | `6e-4` |
| Warm-up | 300 optimizer updates |
| End LR | `6e-5` |
| Schedule | Linear warm-up, then one cosine over a predeclared 4,050-update budget |
| Batch | Microbatch 2 × accumulation 16 × length 1,024 |
| Budget | 132,710,400 processed token slots; also report actual non-PAD/loss-bearing tokens |
| Precision | FP32 for the initial M2 comparison |
| Gradient clipping | Global norm 1.0, with separate codebook/non-codebook telemetry |
| Epoch order | Deterministic reshuffle using seed + epoch; preserve cursor and RNG across resume |
| Checkpoints | Atomic saves, last + best validation + periodic recoverable checkpoints |
| Evaluation | Fixed-update checkpoints and fixed-time measurements; save at least at 250, 500, 1,000, 2,000, and 4,050 |

Keep the v1 no-decay group for token embeddings, Engram tables, codebooks, temperatures, convolution kernels, biases, null plans, and chunk offsets; zero-centered norm gains remain decayed. Use a step-based schedule for the scientific comparison so a faster model does not silently receive a different LR at the same data budget. At a hard wall-time stop, save the partial run and state how much of the schedule completed. Do not extend the budget and re-warm a trained model mid-comparison. For a later production run with a wall-time budget, calibrate once and declare a continuous schedule before launch.

The old v1 schedule changed during training, and its continuation restarted at a high LR and temporarily worsened validation. Therefore the old logs are reference evidence; a clean new control is needed for strict causal claims.

### Parameter count, checked from the actual architecture

| Part | Parameters |
|---|---:|
| Encoder excluding embedding/Engram | 15,510,573 |
| Core excluding codebook | 2,009,912 |
| Confined decoder | 7,532,296 |
| **Compute parameters** | **25,052,781** |
| Engram table | 50,331,264 |
| Tied token embedding | 3,145,728 |
| PQ codebook | 49,152 |
| **Total stored parameters** | **78,578,925** |
| Extra training-only head | 0 |

This proposal assumes the same M2/16-GB environment and prioritizes the demonstrated v1 width. No current user requirement fixes v3 to 19M compute parameters. A 19M cap would require a different, explicitly controlled compromise; silently keeping that cap would preserve one of v2's largest confounds.

The repository's analytical MAC proxy gives about 27.78M MAC/token at R2 and 28.76M at R4, versus 28.02M at mean R2.5. The R4 proxy is only 2.6% above that v1 mean-depth proxy because the core runs at quarter token rate and has one block. **These are analytical proxies, not MPS performance predictions.** They omit some operations and do not model dense masked-kernel execution, memory traffic, autograd storage, allocator behavior, or thermal throttling.

The parameter/gradient/two-Adam-moment tensors alone are approximately 1.26 GB in FP32 under dense-state accounting; activations, attention intermediates, logits, caches, allocator reserves, and the OS come on top. Historical v1 fit at B2, but that does not guarantee a new implementation's peak. Profile the exact trainer for at least 100 updates under stable machine conditions. A practical gate is at least 90% of the contemporaneous corrected-v1 throughput with no steadily increasing memory/swap trend. Do not extrapolate from one warm microbatch.

## 10. Why this is a better-founded candidate than another D6 variation

| Evidence/problem | Default v3 decision | Expected mechanism | Still unproved |
|---|---|---|---|
| v2 weakened token modules | Restore width 384 and v1 FFNs | Recover established representation/decoding capacity | How much of v2's gap was caused by width |
| v1 continuous plan is useful | Keep dense `C,S` scratchpad and `u` route | Preserve learned long-range information route | Additional recurrence benefit |
| v2 has capacity-order drops | Dense FFN, no dispatch capacity | Remove that failure mode entirely | Whether a well-trained MoE could eventually win |
| D6 cuts early state credit | Full two/four-pass graph | Final token/answer loss reaches earlier state updates | Whether uninterrupted credit alone improves learning |
| NextLat has mixed local evidence | Default off | Reduce competing objectives and extra projection cost | Whether a smaller/reintroduced version helps later |
| Easy surrounding tokens dominate counts | Small document/query-balanced answer term | Increase task-relevant supervision using true labels | Language tradeoff and optimal coefficient |
| Repeating final NCP favors a different depth | NCP at terminal depth only | Avoid imposing that target on every intermediate state | Whether a better intermediate target would help |
| v1 extra passes mostly flat | Short loop, fixed-depth controls, explicit R2/R4 results | Limit wasted computation and test depth benefit | Whether v3 learns true iterative composition |

This is a set of falsifiable design choices. It is not a causal decomposition of the entire v2 loss gap.

## 11. Controlled experiment sequence

### Step 1: mechanics and a data-sidecar audit

Before any expensive run, require strict checkpoint-shape compatibility, causal/document/padding tests, final-loss gradient checks for every pass, exact accumulation equivalence, deterministic resume, and hashes proving whether the corpus is unchanged. The current small CPU checks cover part of this list; the new loader/trainer and resume path do not yet exist and are not claimed tested.

### Step 2: separate hygiene, depth, and answer supervision

Use a short ordered comparison rather than a large combination sweep:

| Arm | Architecture | Masks/normalization | Depth | Answer term |
|---|---|---|---|---|
| A: historical-style reproduction | v1 | Legacy | Uniform R1–4 | Off |
| B: corrected v1 control | Same v1 | New explicit masks, accumulation denominators, codebook restart policy | Uniform R1–4 | Off |
| C: short-depth candidate | Same v1 | Same as B | Proposed R2/R4 policy | Off |
| D: proposed v3 | Same v1 | Same as B | Proposed R2/R4 policy | Ramp to 0.1 |

Use the same tokenized data, update count, LR sequence, task annotations, and seeds for B/C/D. The change from A to B is data/loss hygiene, B to C is depth policy, and C to D is answer emphasis. If resources are tight, run B and D first; only run C to attribute a detected difference. Do not run four full 48-hour jobs merely to complete a table.

The first 250–500 updates are a numerical/language sanity screen, not a verdict on eventual hop reasoning. Continue viable candidates to at least 1,000–2,000 updates before a retrieval-learning comparison. Use two seeds for the decisive compact task experiments and, if feasible, the final B/D comparison. If only one full seed is affordable, label seed sensitivity unresolved.

### Step 3: test whether recurrence itself earns its place

Compare the winning recipe with a separately trained fixed-R1 model of identical parameter count and a fixed-R2 model. A fixed-weight truncation is informative, but cannot replace these training controls. For the narrow question of weight sharing, an untied shallow stack at matched execution cost is an additional small-scale control; disclose its larger parameter count rather than call it both compute- and parameter-matched.

Measure carry reset and carry swap after the first pass at fixed inputs/weights. Measure state norm, readout change, update angle, and task performance together. Reset sensitivity shows dependence on state, not necessarily a correct algorithm. Increased angle without better answers is not success. Similarly, forcing states to change would only manufacture an attractive telemetry curve.

### Step 4: full budget and locked evaluation

Only a viable candidate proceeds to the declared 4,050-update budget. Select checkpoints on development validation. Evaluate test once after selecting the checkpoint, loss weights, and operating depth. Keep the historical tests as continuity diagnostics, but do not repeatedly tune against them and then call them untouched holdouts.

## 12. Acceptance and rejection gates

These are proposed engineering thresholds, not forecasts of attainable scores.

1. **Language non-regression:** at equal data budget, neither TinyStories nor WikiText validation NTP should be more than 0.03 nats worse than corrected control B. Also report the paired mean difference and uncertainty. A gain in one language must not conceal a large loss in the other.
2. **Useful skill gain:** target at least a five-percentage-point absolute gain in a predeclared aggregate of carry/multi-value/hop/binding exact-match scores, with no more than a two-point recall loss. Publish every family separately; confirm the selected gain on a locked split. Report paired confidence intervals with source-document clustering where provenance is available. Small test sets do not justify precise improvement claims.
3. **Runtime:** at least 90% of B's measured steady trainer throughput, with peak memory and end-to-end evaluation/checkpoint costs reported. Better sample efficiency can justify lower throughput only if equal-time quality also improves.
4. **Depth value:** promote R4 over R2 only if it delivers a predeclared useful gain—initially 0.01 nats language loss or five percentage points on the designated hard-task aggregate—without a material regression elsewhere. If it does not, ship R2. If trained R1 ties R2 at lower cost, use R1 and report that extra recurrence failed to earn its cost.
5. **No bookkeeping victory:** improvements must survive correct PAD masks, identical answer generation/scoring, batch permutation, fixed-depth inference, and train/eval separation.

The exact thresholds can be adjusted before experiments for a different product priority, but they must not be moved after seeing which arm wins. A v3 candidate that only returns to v1 quality is a recovery from v2, not evidence of a new reasoning breakthrough.

## 13. Evaluation must distinguish retrieval, composition, and generation

Keep the existing full language and task suites, but repair known interpretation problems and add orthogonal tests:

- Report teacher-forced answer token NLL, teacher-forced full-answer exact match, and free-running exact generation separately. Gold previous answer tokens make teacher-forced exact match easier than autonomous generation.
- Split OOD axes: new identifiers at seen distance/depth; longer distances with seen identifiers/depth; extra hops with seen identifiers/distance; and the combined shift. A failure on all three shifts at once does not identify a recurrence failure.
- Include long-range value sequences and several simultaneous keys. Publish per-token offsets as diagnostics, while judging the task by the whole answer. Multiplying offset accuracies is not a substitute for measuring their joint correctness.
- For multi-hop tasks, shuffle fact order, add distractors, and hold out compositions/graphs, not just document strings. Verify that the answer cannot be found through a label or formatting shortcut.
- Remove train/development/test duplicate documents and overlapping generated task instances. The old audit found a few cross-evaluation overlaps; preserve old results as historical data and create a versioned clean benchmark for new selection.
- Context truncation must preserve enough left context for the scored token's entire local path. The previous 72-token truncation placed some predictors inside the 70-token local receptive-field boundary; use a verified longer window such as the checked 88-token construction for a local-path baseline.
- Keep NTP in nats as the primary language comparison; if reporting bits per byte, use a correct byte-level denominator. The old token-byte table has small discrepancies, so do not turn them into an apparent model gain.
- Log R1/R2/R4/R8 sweeps on a fixed development sample, but only claim beyond-training-depth generalization from a locked test protocol. More passes should never be selected per test item using its gold answer.

The existing task suite is mostly algorithmic/synthetic and two language sources. Even a successful v3 on this suite is not thereby a general-purpose assistant or a broad reasoning model.

## 14. Optional practical route: improve a good v1 checkpoint first

Because the architecture is shape-compatible, the fastest practical candidate is a **separate warm-start experiment from the retained v1 continuation best checkpoint**. That model already has useful recall and language ability; restarting from random weights is unnecessary if the immediate goal is a better usable local checkpoint.

This route and the scratch experiment answer different questions. The source checkpoint inherits approximately 5,876 optimizer updates across main plus continuation, or 192.54 million processed slots; some continuation data are replayed. A warm-started v3 must not be compared against scratch v2 as though it had no pretraining cost.

For the warm-start trial:

1. Record and preserve the exact parent checkpoint and all available optimizer/cursor/RNG state. Load model weights strictly; every expected key/shape must match. Use a new run directory and provenance record.
2. First verify that the unmodified loaded model reproduces its source evaluation before changing training settings.
3. Use a small learning rate, proposed `6e-5 → 2e-5` over a predeclared 1,000-update trial, with the answer coefficient ramped over the first 100 updates. Do not re-warm to `6e-4`.
4. Fork a matched low-LR control with the answer term off. Keep cursor/data, optimizer-moment policy, masks, LR, and depth policy identical. If optimizer state is retained, retain it in both arms; if it must be reset, reset it in both and disclose that.
5. Treat new sidecars or rebuilt data as versioned inputs. The old resume CLI restores old arguments; a new objective needs an explicit fork/init path, not an accidental resume that silently reinstates the old recipe.

This route may improve task utility quickly, but cannot establish superior from-scratch sample efficiency. No warm-start training was launched as part of this proposal.

## 15. Implementation work, ordered by dependency

1. **Freeze inputs:** parent source hashes, tokenizer/canonical IDs, data manifest, checkpoint provenance, and a clean development/test protocol.
2. **Add metadata:** deterministic answer spans and hard-task flags, packer offsets, validated sidecars, and exact token-array identity checks. If identity fails, stop calling the dataset unchanged.
3. **Create a separate v3 module:** reuse v1 encoder/core/decoder shapes; add optional state-return instrumentation. Preserve parameter names for strict checkpoint loading. Optimize only mathematically equivalent calculations after a reference check.
4. **Implement the trainer:** terminal-depth losses, full recurrent graph, global accumulation denominators, answer ramp, codebook real-chunk masks and restart-state handling, and separate depth RNG.
5. **Validate execution:** exact small-model gradient/accumulation tests, PAD/empty-batch handling, causal/document isolation, step/resume equivalence, and evaluation forward parity. A resume must restore the data cursor, depth RNG, LR position, optimizer moments, and metadata version.
6. **Profile the complete M2 workload:** B2/accum16, loss construction and backward included, warm cache and steady thermal state, memory and swap recorded. Do not run a competing MPS workload while another model is training.
7. **Run the limited comparison:** B/D first if resources require it, then the attribution and recurrence controls justified by the result.

For initial generation, use the existing full-context evaluation path. A later incremental implementation needs pass-specific caches because a recurrent layer's keys/values depend on its pass state; reusing a single cache across depths is not generally equivalent. Test any cache against full recomputation before claiming generation speedups.

## 16. First-principles review of this proposal

This review was performed after drafting the architecture and recipe. It deliberately asks how the proposal could be wrong.

| Design step | Challenge | Review resolution |
|---|---|---|
| Restore v1 width | Does width alone explain v2's failure? | No. It restores an evidenced working allocation; a narrowed control would be needed to quantify the width effect. |
| Keep `S₀=0` | Did learned initialization win isolated tests? | Sometimes, yes. Zero state is chosen for end-to-end evidence, simplicity, and compatibility, not because alternatives are disproven. |
| Keep v1 normalization | Does zero RMS imply a broken model? | Potential derivative amplification is not proof of an actual failure; mature v1 gradients were healthy. A new normalization was removed from the default proposal. |
| Keep recurrence | Did v1 show useful extra passes? | Barely. Retain a short scratchpad route and require fixed-R training controls before claiming iterative gains. |
| Remove D6/NextLat | Does that prove experts/NextLat are bad? | No. The local complexity/benefit tradeoff is weak. Default-off is a resource decision, and future reintroduction needs a controlled win. |
| Answer weighting | Is this a new competing objective? | It changes the weighting of correct task labels and can still harm language. Keep coefficient small, use a ramp, and test C versus D. |
| Answer metadata | Is it already present in training shards? | No. Sidecar generation and identity checks are mandatory, not an assumed implementation detail. |
| Hard-task depth | Is a training-only task flag an inference oracle? | It must not become one. Publish fixed R2/R4 results on every test item; no gold-depth selection at inference. |
| Full backpropagation | Could memory exceed the M2 budget? | K≤4 fits the historical depth range, but only the real trainer profile can confirm current peak memory. |
| Terminal NCP only | Could early states become ungrounded? | Final NTP/answer gradients reach them; that permits useful intermediates but does not guarantee them. Test state interventions and task gains. |
| More answer accuracy | Could it be teacher-forcing or test leakage? | Separate free-running generation; use clean, locked splits and annotation alignment tests. |
| Parameter/MAC budget | Are analytical FLOPs actual speed? | No. Exact parameter counts are verified; runtime remains an empirical gate. |
| Warm start | Would it make v3 appear unfairly strong? | Count inherited training and compare only with a matched warm-start control. Keep scratch claims separate. |
| Minimal architecture | Could v3 simply reproduce v1's ceiling? | Yes. That is the main remaining risk. Do not claim a breakthrough if answer/depth controls fail. |

### What I rejected during the design review

I considered a new normalization floor for the scratchpad, a forced bounded/overwrite update, larger expert pools, learned initial state, intermediate-answer/bridge heads, and a new memory cross-attention architecture. Each has a plausible argument; none has stronger local end-to-end evidence than the retained core. Adding them together would recreate v2's attribution problem. They are deferred hypotheses, not hidden pieces of v3.

I also rejected the stronger claim that v2's loop is simply harmful: the fresh checkpoint sweep directly contradicts it for token prediction. Finally, I rejected using old proposal scores or impressive isolated best-checkpoint numbers to assign a predicted capability score to v3.

### Checks completed and checks still required

Completed here: exact parameter count from the architecture, a small full-gradient/causality/document/batch-permutation check, target-aligned plan verification, representative real/PAD auxiliary masks including a zero-loss all-PAD case, and document-balanced answer-loss decomposition across microbatches. The outputs are in [design_checks.json]({{A}}/design_checks.json). These are mechanics checks, not model-training tests.

Still required before launch: sidecar construction, full trainer implementation, full optimizer-batch loss equivalence, restart-row optimizer-state tests, deterministic pause/resume, MPS stability/profile, and the controlled learning comparisons. They are listed as future work because this request is for a proposal, not authorization to replace the active run.

The final document's key numerical claims, links, source hashes, and consistency checks are recorded in [proposal_review_checks.json]({{A}}/proposal_review_checks.json). This is a self-review, not an independent second reviewer.

The final automated review passed all 83 checks. It verified 408 local references, all 268 captured text-file hashes, and that the 66 inventoried active v1/v2 code files still match the snapshot. The review corrected one throughput transcription and made optimizer, auxiliary-loss normalization, and codebook-restart settings explicit. These checks establish consistency of the proposal and its evidence; they cannot establish that an untrained v3 will learn better.

## 17. Relation to the literature

The literature supports recurrence as a legitimate mechanism, not a guarantee for this particular small language model. [Universal Transformers](https://arxiv.org/abs/1807.03819) introduced shared recurrent self-attention across depth. [Loop, Think, & Generalize](https://arxiv.org/abs/2604.07822) reports compositional benefits under controlled training, while also identifying degradation from excessive recurrence. Those observations motivate measuring depth rather than assuming more is always better.

[Pretraining Recurrent Networks without Recurrence](https://arxiv.org/abs/2606.06479) uses supervised memory-transition labels to address recurrent credit assignment. That is a different setup from giving every D6 pass the same next-PQ target. Its result does not validate the current detached-window recipe by analogy. No new teacher-memory system is included in v3.

The repository's broad paper notes were used as background. The three primary records above were checked directly for these limited contextual claims; this proposal does not present itself as a new full literature survey. The recommendation is driven mainly by local code and measurements.

## 18. Final specification and decision

**v3 default:** v1-compatible 384-wide encoder and confined decoder; 50.33M Engram; one dense shared `C,S` core; exact zero initial state; short R2/R4 training; full backpropagation; final-pass NCP and codebook loss with explicit real-chunk masks; a small annotated-answer term; no NextLat, MoE, detached windows, or consistency loss.

The fastest practical path is a separately recorded low-LR warm-start trial from good v1 weights. The strongest scientific test is a matched scratch comparison with the corrected v1 control. These are complementary experiments, not interchangeable evidence.

**The improvement thesis is: preserve the information route that already works, give it clearer task credit, and stop paying for unearned recurrent computation.** If that thesis fails its controlled tests, the next change should target the measured remaining bottleneck—concept identity/retrieval, answer supervision, or recurrent transition—not another untested bundle of architectural ideas.
'''
(ROOT/'proposed_v3.md').write_text(render(proposal))

# Preserve every official evaluation and equal-window comparison in a compact appendix.
ls=['# v3 proposal: result ledger','',f"Frozen at {snap['time']}. This appendix accompanies [proposed_v3.md]({ROOT}/proposed_v3.md).",'',
    '## Every official v1/v2 quick evaluation','',
    '| Run | Step | Hours | Language mean | Recall | Carry | Multi | Hop | Bind | PQ explained variance |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for name in ['v1','v1_cont','v2']:
    for e in snap['runs'][name]['eval']:
        ps=e['probes'];vals=[name,str(e['step']),f"{e['wall_h']:.4f}",f"{e['val_lang_mean']:.6f}"]
        vals += [f"{100*ps[k]['exact']:.2f}%" for k in ['lr_recall1','lr_carry3','lr_multi4','lr_hop','lr_bind']]
        vals += [f"{e['codebook']['var_explained']:.6f}"]
        ls.append('| '+' | '.join(vals)+' |')
ls+=['','## All matched 50-update training windows','',
     'v1 logs 50-update means and v2 logs 25-update means. Two v2 means are combined. The same epoch-0 data order and slot count apply; later learning-rate schedules differ.','',
     '| End step | v1 NTP | v2 NTP | Gap | v1 LR | v2 LR | v1 hours | v2 hours | v1 slots/s | v2 slots/s |',
     '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for e in snap['matched_train']:
    ls.append(f"| {e['step']} | {e['v1_ntp']:.6f} | {e['v2_ntp']:.6f} | {e['v2_ntp']-e['v1_ntp']:.6f} | {e['v1_lr']:.8f} | {e['v2_lr']:.8f} | {e['v1_h']:.4f} | {e['v2_h']:.4f} | {e['v1_tps']:.1f} | {e['v2_tps']:.1f} |")
ls+=['','## Current v2 state: every training summary','',
     '| Step | NTP | NCP | NextLat h | NextLat KL | Drop % | Consistency | Gradient norm | Slots/s | Swap MB |',
     '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for e in snap['runs']['v2']['train']:
    if 'ntp' not in e:continue
    ls.append('| '+' | '.join([str(e['step'])]+[f"{e[k]:.5f}" for k in ['ntp','ncp','nl_h','nl_kl']]+[f"{100*e['drop']:.3f}",f"{e['cons']:.6f}",f"{e['gn_preclip']:.4f}",f"{e['tok_s']:.1f}",f"{e['swap_mb']:.1f}"])+' |')
ls+=['','## Fresh checkpoint 1703: fixed weights, changing depth','',
     'Eight fixed validation rows per language, same sample seed as the preceding review. This is a small diagnostic, not the official full evaluation.','',
     '| R | TinyStories | WikiText | NCP (batch mean) | State norm (batch mean) | Update norm (batch mean) |',
     '|---:|---:|---:|---:|---:|---:|']
for rr in [1,2,4,6,8,12]:
    ls.append(f"| {rr} | {depth['language_summary']['tinystories'][str(rr)]:.6f} | {depth['language_summary']['wikitext'][str(rr)]:.6f} | {statistics.mean(b['ncp'][rr-1] for b in depth['language_batches']):.6f} | {statistics.mean(b['state'][rr-1]['norm'] for b in depth['language_batches']):.6f} | {statistics.mean(b['state'][rr-1]['update_norm'] for b in depth['language_batches']):.6f} |")
ls+=['','## Historical experiments and what transfers','',
     f"- [v1 full final evaluation]({S}/run_m2_25M/runs/main/eval_final.json): useful core/continuous plan, modest PQ contribution on language but meaningful complete-answer contribution, little marginal depth benefit.",
     f"- [Decoder results]({S}/Decoder Design/RESULTS_DECODER.txt): confined additive plan is viable; separate-plan read has seed instability; NextLat has no reliable win; multi-value retrieval is a bottleneck upstream of the decoder.",
     f"- [Isolated concept results]({S}/Concept Design/RESULTS_v2.txt): depth can help controlled composition; small scale and isolated interface do not establish an end-to-end gain.",
     f"- [Initialization review]({S}/Concept Design/RESULTS_review.txt): zero state and learned starts each have supporting runs; large seed sensitivity forbids declaring a universal winner.",
     f"- [End-to-end concept results]({S}/Concept Design/RESULTS_E2E.txt): plan is necessary but depth gains largely disappear; evidence against transferring isolated wins automatically.",
     f"- [Gradient steering/branching]({S}/Concept Design/RESULTS_H1_H2.txt): verifier-assisted or isolated latent-branch results do not justify their extra cost in the default language model.",
     f"- [Earlier complete dataset and result audit]({ROOT}/audit_v2_2026-09-28/RESULTS_APPENDIX.md): shared corpus, PAD auxiliary targets, eval overlaps, all v1 family results and generations.",
     f"- [Earlier v2 gradient/routing diagnostics]({ROOT}/v2_underperformance_review_2026-09-28/REPORT.md): direct evidence at steps 138/267; not silently presented as remeasured at step 1703.",
     '', 'All raw train/eval fields are preserved in snapshot.json. Metrics not promoted to causal explanations remain available, including codebook entropy, temperatures, gate means, memory, losses, and per-pass NCP vectors.']
(A/'RESULTS.md').write_text(render('\n'.join(ls)+'\n'))

def coverage(p):
    suffix=Path(p).suffix
    if suffix=='.pdf':return 'Literature artifact inventoried; prior paper notes/extracts reused. Three primary abstract records rechecked online; no claim of fresh full-PDF review.'
    if suffix in ['.rtf','.docx','.zip']:return 'Design/archive material inventoried; canonical Markdown/code/result versions used for current claims.'
    if p.startswith('run_v2_19M/code/'):
        return 'Captured, parsed and compared with prior audit; model/trainer/data/evaluation paths traced; unchanged code findings reused with updated logs.'
    if p.startswith('run_m2_25M/code/'):
        return 'Captured/parsed; model, core, encoder, decoder, build/data, task/probe and evaluation paths traced for v3; operational helpers treated as infrastructure.'
    if p.startswith('run_m2_25M/data/'):
        return 'Shared data artifact inventoried; earlier full shard/hash/overlap audit reused, current source and validation slices read. Not a fresh full-corpus scan.'
    if '/runs/' in p:
        if suffix=='.pt':return 'Checkpoint metadata inventoried. v2 step1703 weights read on CPU; v1 final/continuation evidence read from saved results. Aliases are not independent models.'
        return 'Current text/structured result captured and parsed; all main/continuation/v2 logs and key screen/calibration/test evidence reviewed. Operational records do not establish model quality.'
    if '/webapp/' in p:return 'Presentation code/data captured and parsed/indexed; metrics traced to source logs. No independent learning evidence inferred from a chart or design score.'
    if p.startswith('Concept Design/') or p.startswith('Decoder Design/') or p.startswith('Encoder Design/'):
        return 'Canonical specification/code/results captured; experiment mechanisms and outcomes compared with deployed implementation. Old toy results retain scale/seed caveats.'
    if '/concept_v2/' in p or '/model_v2/' in p:return 'Prior D1–D6/model proposal, budget, review and literature notes captured; intent distinguished from measured v2 behavior.'
    if p.endswith('DECISIONS.md') or p.endswith('STATUS.md'):return 'Read operational/architecture provenance; actual code and logs take priority over intended or stale status.'
    return 'Inventoried; canonical task/design context retained. No unsupported empirical claim based on this file.'
ls=['# File coverage for the v3 proposal','',
    f"Inventory captured {len(inv)} files after excluding dependency/cache/vendor/assistant metadata and prior audit copies. The [inventory]({A}/inventory.json) stores paths, sizes, inodes, modification times, text hashes and parse information. The [snapshot]({A}/snapshot.json) records excluded directories. This is a transparent coverage ledger, not a claim that every binary value or paper page was individually interpreted.",'',
    'The current pass updates the earlier deep v1/v2 audit: all current logs were reread, source was traced for the proposal, fresh checkpoint depth diagnostics were run, and retained architecture mechanics were checked. Unchanged full-corpus checks and historical paper reading are explicitly inherited.', '',
    '| File | Bytes | Review/use |','|---|---:|---|']
for e in inv:
    p=e['path'];f=S/p;target=f if f.exists() else ROOT/p
    ls.append(f"| [{p}](<{target}>) | {e['bytes']:,} | {coverage(p)} |")
ls+=['','## Current review artifacts','',
     '- `snapshot.py`, `snapshot.json`, `inventory.json`, `source_snapshot/`: immutable-in-intent evidence capture at the declared cutoff; rerunning snapshot.py would deliberately update that cutoff.',
     '- `latest_depth_check.py`, its log, and `checkpoint_001703_diagnostics.json`: single-thread CPU fixed-weight diagnostics, no optimizer operations.',
     '- `check_design.py`, `design_checks.json`, and log: exact parameter count and small-model mechanics; not a trained v3.',
     '- `build_proposal.py`, `RESULTS.md`, this ledger, and root `proposed_v3.md`: reproducible proposal/evidence writeup.',
     '- `plot_evidence.py`, `evidence.png`: figure derived from saved results.',
     '- `review_proposal.py`, `proposal_review_checks.json`: final arithmetic, source, link and consistency review.',
     '', 'No model implementation, training configuration, current dataset, or running process was changed. Checkpoint rotation may later remove an original binary referenced by the ledger; the measured results and source snapshots remain.']
(A/'FILE_COVERAGE.md').write_text('\n'.join(ls)+'\n')
print('Wrote',ROOT/'proposed_v3.md')
