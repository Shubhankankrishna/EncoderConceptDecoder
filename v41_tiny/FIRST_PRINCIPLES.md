# v4.1 from first principles

This explanation describes the implemented tiny model, rather than assuming
that its intended division of labour has been learned. Experimental evidence
and the final recommendation are in
[REPORT.md](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/REPORT.md).

## 1. Begin with the prediction problem

An autoregressive model must estimate

\[
p(x_{t+1}\mid x_0,\ldots,x_t).
\]

It needs three things: access to relevant facts, a way to combine those facts,
and a way to convert its result into the next token. Architecture specifies
which computations are possible. Training decides which ones the model uses.
An available route is not evidence that a useful representation travels on it.

```mermaid
flowchart LR
    X["Known token prefix"] --> E["Encoder: local L, global G"]
    E --> TOK["tok = norm L"]
    E --> POOL["Attention pool: 6 tokens → C"]
    POOL --> CORE["Core: 4 layers × 2 passes"]
    CORE --> U["u and predicted next concept"]
    U --> P["Newest completed plan at each token"]
    TOK --> D["Private D = tok + plan"]
    P --> D
    D --> SS["Local layers S, S"]
    SS --> R["Global retrieval R"]
    TOK -->|"Read-only token K/V"| R
    R --> S["Local layer S"]
    S --> Y["Next-token logits"]
```

All four decoder layers read token K/V; R's separate arrow highlights the new
global route. The private D arrows connect computations at the same token.

The comparison transformer carries one residual stream through 16 distinct
layers, with 22 layer applications because some weights are reused. Local
attention uses window 60, and every fourth layer uses full causal attention.
Every layer can read earlier residual states. It predicts the next token
directly and trains on next-token loss.

Code: [transformer schedule and residual updates](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_llm.py:31).

v4.1 divides this work into a token encoder, a compressed concept core, and a
token decoder. The intended advantage is to do some expensive global work at
T/6 positions while retaining precise token-level access to facts.

In the pinpad test, the selected task is hidden from the model. It sees the
layout and previous actions, not an instruction naming all six targets. At a
new pair's first colour there can be several plausible intended targets. After
visiting an even colour, its odd partner is determined: 0→1, 2→3, 4→5, 6→7.
This is why the first action toward that second colour, b2, is a particularly
useful test of composition and fact access. Overall expert-action accuracy is
not a fully specified task-solving score with an expected ceiling of 100%.
The model must combine current position, visited-colour history, the learned
pair relation and the target location; teacher-forced histories provide valid
previous actions, whereas generation must produce its own.

Code: [latent tasks and known pair structure](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/pinpad.py:32)
and [first/second-colour decision labels](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/pinpad.py:140).

## 2. The encoder makes two different streams

The eight encoder layers follow `S S S F S S S F`. S layers update a local
stream L. F layers update a global stream G with full causal attention. Each F
layer is applied twice in the default. The same parameters are reused; this
adds computation rather than another independent set of weights.

The residual stream is width 64. Local attention has four width-16 query heads
and two K/V heads; global encoder/core attention has six width-16 query heads
and two K/V heads. Its attention projection is therefore width 96 before the
output projection returns to width 64. The head count need not divide the
residual width in this implementation.

Two encoder Engram reads use deterministic token-derived indices into a learned
table. They enrich L and G. They are learned lookup features, not the concept
codebook and not an external database. The decoder has no separate Engram read.

The decoder receives `tok = norm(L)`. Its local receptive field has maximum lag
63, including short-convolution effects. G is globally contextualised, but is
not passed directly to the decoder. Instead, a learned router mixes the input
embedding, L and G before pooling. Thus global encoder computation reaches
ordinary token predictions through the concept route.

Code: [two-stream encoder and pooling router](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v4.py:214).

## 3. Six tokens become one continuous concept

For each six-token chunk, a four-head attention pool selects among its valid
token features. It starts as a mean pool. There is no pooler FFN in this model.
The result is normalised to approximately radius sqrt(d), which is 8 at d=64.
Call the continuous result C_j.

Compression reduces sequence length, not the number of facts the task requires.
It introduces a capacity and timing tradeoff: a chunk must combine several
token features in one vector, and a token can only use a chunk once all of that
chunk is available. Normalising a vector fixes its norm. It does not ensure
that different examples get different vectors or that the coordinates encode
human-interpretable concepts.

For B=4 and T=480, C has shape `[4,80,64]`; token features have shape
`[4,480,64]`. Documents are padded to multiples of six so a pooled chunk never
crosses a document boundary.

Code: [attention pool](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v4.py:96)
and [spherical normalisation](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v4.py:238).

## 4. The concept core has recurrent computation, not guaranteed reasoning

Four core layers are applied twice. Each layer reads C and the evolving state
S, attends causally to concepts in its document, and updates S. A shared FFN
provides additional transformations. The second pass reuses the first pass's
weights and carries its state forward.

The final normalised state is u_j. The prediction head also produces c_pred,j,
a soft prediction of the next concept. Reusing a block can refine a state, but
does not establish that the second pass performs another reasoning step. A
one-pass core still has four attention layers; its inputs also include global
encoder features. Therefore `M=1` is not a one-hop reasoning baseline.

The experiments distinguish a fixed-checkpoint `M1` intervention from training
a different architecture. Removing a pass at evaluation measures reliance on
that pass in that checkpoint. It cannot alone establish that the extra pass
was unnecessary throughout learning.

Code: [core layers and repeated state updates](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v4.py:243).

## 5. The codebook is a prediction vocabulary

At tiny scale the coarse book has 64 vectors of width 64. Fine residual product
quantisation has eight segments, 256 entries per segment, width eight. Their
total stored parameter count is 20,480.

C is assigned a nearest coarse centroid and nearest residual entries. The core
predicts the next chunk's coarse and fine assignments. At inference c_pred is
a soft weighted sum of coarse vectors plus soft residual vectors. The core
itself receives continuous C, and u also reaches the decoder directly.

Consequently discrete codes are not a mandatory information bottleneck through
which all reasoning must pass. A model can predict useful tokens while its
coarse assignments have little relationship to task subgoals. Code usage,
entropy and reconstruction quality cannot establish semantic abstraction;
labelled probes, a permutation null and causal control checks are needed.

The objective is

\[
L=L_{NTP}+.05L_{coarse}+.10L_{fine}+L_{VQ,coarse}+L_{VQ,fine}.
\]

NTP uses valid within-document next-token targets. NCP uses valid pairs of
completed chunks. The fine NCP loss conditions its residual frame on the next
chunk's assigned coarse code during training; that future coarse code never
enters the inference plan. VQ uses detached encoder concepts and trains only
the codebooks. Codebooks are detached in the prediction head, so NTP cannot
move the codebook directly. These gradient distinctions are tested explicitly.

Code: [soft code prediction and detached codebooks](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v4.py:296),
[NCP target frame](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v4.py:338)
and [training objective](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v41.py:132).

There is a useful first-principles counterexample to assuming this objective
guarantees abstraction. Suppose every chunk maps to the same sphere point C*.
One coarse code and residual pattern can represent it, and predicting the next
assignment becomes easy. The VQ and NCP objectives alone do not forbid that
solution. The next-token objective must make example-specific concepts useful
enough to avoid it. R supplies another route for facts, weakening that necessity.
This is a possible degeneracy of the objective, not a claim that the measured
models literally have constant C: their plan statistics show substantial
variation, and their continuous states have some readable task information.

## 6. A token gets the newest completed plan

Position t predicts token t+1 and uses

\[
j^*(t)=\lfloor(t+1)/6\rfloor-1,
\qquad
P_t=W_u u_{j^*}+W_c c_{pred,j^*}+o_{(t+1)\bmod6}.
\]

The first five positions of every document use a learned null plan. The
six slot offsets tell the decoder which chunk position it is predicting.

This alignment matters. Concept j includes tokens 6j through 6j+5. It may first
be used at position 6j+5, after all those tokens are known. Its direct output
plan reaches positions 6j+5 through 6j+10, precisely six next-token predictions.
The full graph is causal even though the encoder pools whole chunks in parallel.

The plan can exclude the newest zero through five tokens. On this pinpad
format, the 27-token header and alternating action/position tokens mean that
action queries occupy plan lags 1, 3 and 5. They never occupy lag zero. The
current cell is therefore often more recent than the latest complete concept.
That is a concrete architectural timing constraint, not yet a demonstrated
cause of any measured deficit. Natural phase comparisons also vary in layout
and path length, so they are observational.

Six tokens correspond to about three environment steps in the action/position
body. Reaching a target can take a variable number of steps, so a six-token
chunk is not automatically an entire subgoal. It can end mid-route or straddle
a subgoal change. The post-training switching controller described in the
design would have to learn meaningful durations; the pretraining model here
uses fixed boundaries.

Code: [legal plan indices, nulls and offsets](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/hrm_decoder.py:74).

## 7. Retrieval fixes fact access while keeping plans private

The decoder starts with `D_t = tok_t + P_t` and has four layers:

| Layer | Query | Keys and values | Range |
|---|---|---|---|
| 0, S | private D_t | tok | causal window 8 |
| 1, S | private D_t | tok | causal window 8 |
| 2, R | private D_t | tok | full causal, same document |
| 3, S | private D_t | tok | causal window 8 |

R uses per-head query/key normalisation, four learned temperatures, and no
rotary positions or short convolutions. Local S layers retain partial rotary
positions. Each layer also has output gates, residual reads/writes and an FFN.
The tied output head maps the final normalised D to vocabulary logits.

To derive R, think of finding a fact in a set of stored features. For each head,
make a query from the current private state, compare it with each stored key,
normalise the scores into a probability distribution, then read an average of
the corresponding values:

\[
q_t^h=\operatorname{qnorm}(W_q^h\operatorname{GRRead}(D_t)),\quad
k_s^h=\operatorname{knorm}(W_k^h tok_s),\quad v_s^h=W_v^h tok_s,
\]
\[
a_{ts}^h=\operatorname{softmax}_{s\leq t,\,doc(s)=doc(t)}
\left(\frac{e^{\ell_h}q_t^h\cdot k_s^h}{\sqrt{16}}\right),\qquad
r_t^h=\sum_s a_{ts}^h v_s^h.
\]

The learned log temperature ℓ starts at zero, so its positive multiplier starts
at one. Increasing it makes score differences sharper. The four heads can read
different features, and an output projection combines their reads into D.
GRRead is a learned, gated normalisation/mixing of its input streams; GRWrite
gates the residual update. Neither name denotes an explicit symbolic operation.
The FFN then transforms the updated state pointwise. All of these are learned
numerical operations; usefulness must be measured after training.

NoPE means R adds no rotary position transform to those query/key vectors.
It does not make R's memory position-free: tok already carries local order and
context from the encoder. Nor does whole-document access guarantee that the
softmax selects the correct fact.

With v4's confined decoder, a distant fact had to arrive through concepts or be
relayed through local encoder windows. R allows a direct content-addressed read
of any earlier token feature in the document. Whole-document access is an exact
structural property; successful retrieval is a learned capability.

Every layer takes K/V from tok rather than another position's D. Therefore
another token's plan cannot enter D_t through decoder attention. Perturbing an
output u_j has direct effects only at its six legal plan positions. The
parameter-identical stack control switches K/V to `GRRead(D)` in all four
layers, allowing past plans and past decoder computations to be combined.

Privacy does not imply that only the core can combine facts. A query built by
earlier decoder layers can change the next attention read, even when memory
is fixed. The encoder also transforms facts before R sees them. “One global
retrieval layer” does not make the whole token path a single-hop machine.

Pinpad gives a concrete example. The header lists colour/cell pairs in colour
order. The odd colour's cell feature can already include the nearby even
colour and its cell through local encoder attention. R can therefore read a
feature that has combined part of the pair relation before retrieval. This is
an available computation, not a claim that a particular head learned exactly
that algorithm. A layer count by itself cannot establish that multi-fact
composition has to occur in the concept core.

Code: [ordered colour/cell header](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/pinpad.py:119).

Code: [R query, token-memory K/V and stack control](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v41.py:16)
and [S S R S execution](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/model_v41.py:72).

## 8. The right latent site depends on the required horizon

An offset on output u_j is read directly by only six positions. An offset on
input C_j changes the core's future states, so it can influence later plans
throughout the document. An offset on the first-pass state S1 has another full
core pass beneath it. These are different control surfaces.

Counting layer applications along the encoder→core→decoder route gives the
following locations under the registered two-loop/two-pass configuration.
Concept applications operate at T/6, so this is a depth description, not a
compute-equivalence claim.

| Offset site | Applications above it | Applications below it | Direct fixed-input horizon |
|---|---:|---:|---|
| C, core input | 10 encoder | 8 core + 4 decoder | Can change later core plans in its document |
| S1, first-pass state | 10 encoder + 4 core | 4 core + 4 decoder | Can change later second-pass plans |
| u, final output | 10 encoder + 8 core | 4 decoder | Its six legal decoder positions |

More transformations below a site can make a high-level offset useful, but
depth alone does not prove why an oracle offset succeeds. The sites also have
different baseline representations and sensitivity.

R's K/V are computed from encoder tok and cannot be changed by a core-input
offset. R's query is plan-aware, however, so that offset can change which fixed
token memory R retrieves. It is incorrect to say that all of R is computed
before the pooler: its token memory is; its plan-conditioned query is not.

The frozen oracle experiment tests whether legal offsets can repair decisions.
It uses known optimal actions, full-width prompt offsets and a fixed ten-step
budget. Success would establish local control authority. It would not establish
that an eight-dimensional metacontroller can learn those offsets, infer when to
switch, or discover a useful sparse-reward policy.

Plan privacy is also a fixed-input graph property. During autoregressive
generation, an altered token becomes part of the next input prefix and can
affect much later outputs through ordinary feedback.

Code: [frozen latent-offset experiment](/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v41_tiny/code/control_v41.py:1).

## 9. Why the baseline is useful but imperfect

Compute-parameter counts are close: v4 1,212,032; v4.1 and its stack control
1,212,036; transformer 1,206,864. v4.1 adds exactly four temperatures and keeps
all inherited v4 weights identical at initialisation. Its stack control has
exactly the same parameters.

The transformer has a different initialisation, a larger local window, no
Engram table and no concept auxiliary objective. Its estimated forward matrix
and attention FLOPs are about 32% higher on pinpad. Equal parameters, training
rows and real-token budget isolate a useful practical comparison, but they do
not isolate every architectural choice or equalise total computation.

“Compute parameters” is the repository's ledger convention. It excludes the
embedding, codebooks and Engram lookup table. On text, v4.1 stores 2,149,732
parameters in total, versus 1,731,152 for the transformer. The vocabulary head
still performs computation even though its tied embedding is excluded from
the compute-parameter ledger. Full-size v4.1 has 19,175,580 compute parameters
and 101,600,284 stored parameters; the full-size model was instantiated for
counting, not trained in this experiment.

## 10. What evidence would justify scaling

The strongest result would combine improved held-out decisions, preserved text
loss, better closed-loop task structure, example-specific plan dependence,
useful recurrent computation, and calibrated latent control. Any one of these
can hold without the others.

The decisive distinction is between an information route that exists, a
checkpoint that reacts when that route is removed, and a representation that
helps on new facts and compositions. Constant plans, trained no-plan models,
counterfactual facts and multiple seeds make that distinction more measurable.
The experiment report evaluates those claims separately.
