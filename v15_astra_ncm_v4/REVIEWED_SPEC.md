# Astra v4 reviewed engineering contract

Reviewed against the complete 1,096-line proposed specification, the pinned NCM
source, official LM1B Small checkpoint, and Astra v3 quality findings on 19 October
2026. This file is the implementation contract; the supplied v1 design remains a
research proposal. Implementation and empirical acceptance are separate statuses.

## 1. Decision and scope

Proceed with a staged, falsifiable pilot. Reusing a trained generator is a sensible
response to v3's weak coarse-to-fine modeling. It does not establish that NCM can
solve graphs, nor repair open-ended language continuation automatically. The new
experiment learns graph decisions and a separately conditioned plan renderer.
The official LM1B model is unconditional. It is not an instruction model.

The released 128-token codec and 641-code hierarchy are retained exactly. The
previous adapted native32 codec and its checkpoints are incompatible and are not
loaded. This explicitly supersedes the earlier 32-token design goal for this pilot.
Neither a 256-token prompt interface nor 32-token predictions are claimed here.

## 2. First-principles review, in dependency order

| Question | Finding in the proposal | Correction and test |
|---|---|---|
| What is being learned? | Language modeling, dynamics, planning, and explanation are interleaved. | Report graph decision success and rendered-language fidelity independently. Never credit an exact simulator or oracle path as learned reasoning. |
| What information is sufficient? | Complete graph plus current node and goal suffices. Lossy text does not. | Exact adjacency and node attributes are authoritative; no oracle distance is a model input. |
| Is successor prediction meaningful? | An action specifies its destination, so next-node classification can copy it. | Retain as an interface diagnostic with weight .1; main targets are optimal-action distribution, discounted value and action return. Include non-goal/unreachable states and all legal counterfactual actions. |
| Is a value of one informative? | Undiscounted optimal success is one on every reachable graph. | V*=gamma^d, gamma=.95; unreachable value zero; goal value one. Search backs up gamma^edge_cost times successor value. Call the metric discounted-value MSE, not success-probability calibration/Brier score. |
| Can two global MLP updates propagate graph information? | A pooled vector can lose the topology before recurrence begins. | Keep node-resolved representations. Two shared message updates propagate over outgoing and incoming edges. Action features combine destination/current/goal/global state. Test node permutation and padding invariance. |
| Must reasoning width equal generator width? | 768 is inherited from NCM rather than a task requirement. | Use width128, recurrent hidden192; a bridge maps to NCM's768. The generator remains97M-class, unchanged. This reduces optimizer/activation cost. |
| Does recurrence add physical parameters? | No; but its value is assumed too readily. | R0/R1/R2 and parameter- and arithmetic-matched untied updates, two seeds, matched optimizer steps/batches/data order. Report true counts, runtime and optional R4 inference. No equivalent-parameter claim. |
| What should z represent? | A posterior conditioned on a deterministic successor may identify nothing beyond the action. | First use explicit two-valued preferences: shortest path or risk-averse path. Risk is an observed node attribute. Both targets come from exact positive-cost shortest paths. No training posterior, latent KL or unsupervised Gaussian-strategy claim. |
| Are two strategies necessarily different? | No. | Count distinct successful paths and cost/regret under each preference. K2 samples may duplicate. Hold the preference fixed through the episode/tree. Do not call diversity proof of improved success. |
| Where is NCM pretrained knowledge used? | Proposed reasoning module can ignore it entirely. | Say so: structured stages have no NCM computation. NCM is used for verified-plan language rendering in a separately measured adapter stage. No claim of pretrained transfer to planning without an additional controlled experiment. |
| Does the published sampler use query level 0? | Config supports it, but released eval defaults to the checkpoint's empirical level 0 distribution. | Audit both query-based generation and published unigram-seeded decoding. Report decoder penalties separately. Conditioned adaptation supervises level 0 explicitly via its query. |
| Is the conditioning bridge identical at deployment? | Separate custom training/generation paths can drift. | One level_logits implementation for sparse training and sequential generation; compare to full block-diagonal forward on all 12 levels, zero adapters and nonzero FiLM. |
| Can sparse levels bias the loss? | Uniformly sampling levels then using mean CE overweights coarse levels. | For pure position-average CE, sample j with probability ell_j/641. The pilot explicitly uses the mixture objective below to retain coarse-level supervision. Log per-level validation, including level 0. |
| Can raw text preserve LM1B behavior? | Existing local text is FineWeb, not LM1B. | Label retention as a FineWeb distribution-drift probe. Use frozen-base KL on identical inputs with adapters disabled; do not call it LM1B validation or a paper reproduction. |
| Does a correct codec imply good generation? | No. | Measure normalized-token reconstruction, exact blocks, symbolic roundtrip, free generated text, and plan fidelity separately. |
| What does MCTS verify? | Transition accuracy is not sufficient to establish good values/search. | Exact-env PUCT only, cycle/depth caps, discounted backups, legal action masks, accurate root visits and environment-call counters. Evaluate N=8/16, greedy, random, and deterministic BFS oracle. No default search promotion without a measured benefit. |
| Can a tiny graph solve prove general reasoning? | No. | Deduplicate topology by directed graph isomorphism; split whole graphs. Reserve test/OOD for final reporting. OOD has16–20 nodes versus5–12 training. Report uncertainty and failures. |

## 3. Grounded data and objectives

Generate 2,000 topology-distinct in-distribution graphs: 1,600 train, 200 validation,
200 final test; 200 additional larger OOD graphs. Graphs include cycles, dead ends,
multiple routes and unreachable current states. All states and legal actions are
available for supervised sampling. Sample graphs uniformly, then current nodes;
this does not let dense graphs dominate the batch. Seed every generator.
Evaluation starts have balanced quotas for 2 / 3 / 4+ required hops; larger OOD starts
have 2–3 / 4–5 / 6+ hops. Short random graphs otherwise overrepresent easy two-hop routes.

Two explicit preference settings use destination-entry cost
`c_z(s,a)=1+2*z*risk(a)`, z in{0,1}. Dijkstra supplies d_z. The optimal action target
is uniform over all minimizing legal actions. For unreachable current states all
actions tie and V*=Q*=0. Goal is absorbing and has no policy target. For all
nonterminal actions, `Q*=gamma**(c_z+d_z(next))`. Labels never enter observations.

The initial objective is policy CE + .5 value MSE + .5 Q MSE + .1 successor CE.
Goal/dead-end rows are excluded from policy and successor loss. Legality is100%
by construction, a software invariant rather than a learned achievement.

The shared cell is pre-normalized and exactly residual, with its final projection
initialized to zero. Do not initialize both projections to zero. Step-one gradients
reach the final projection; after it changes they reach the earlier projection.
The untied control has two half-hidden-width cells; report measured parameter and
operation differences rather than asserting exact FLOP equality.
An additional full-hidden-width untied two-cell control matches the recurrent
arithmetic at a larger physical parameter count. Evaluate both: one untied network
cannot match both weight-sharing parameter count and repeated dense-layer arithmetic
at the same time. Recurrence's pilot benefit gate compares against the strongest
R0, R1, parameter-matched untied and compute-matched untied control in both seeds.

## 4. Official assets and adapters

Pinned source: `c70274273f489abc73832b2b98584a8483b6b203`.
Pinned HF release: `d342ce8aaea58830eef3b8ad473664673e41f692`.
Generator SHA256: `b0789d5a2aad9114d40976d26bd1af09d8cea0b0bec7d385b6a44739f77256fe`.
Codec SHA256: `d88aa17b66d730f2cb1fb31c5dbe63888a10da47ed8bea29dcb1f89423c3059c`.
Use strict state-dict loading with weights_only=True. Download into this project's
assets; do not add files to the prior codec directory, whose file set is hashed.

LoRA rank 4, alpha 4, dropout 0 on actual fused QKV and attention output projections
in all 12 blocks:221,184 trainable adapter weights. Frozen originals and codec never
enter the optimizer. Original dropout stays disabled for controlled parity tests.
FiLM has a 128-wide bottleneck and level embedding; its final layer starts at zero.
Original positional encoding, attention masks, prefix token and all levels remain.
The codec runs on CPU; generator/adapters use MPS only after numerical/backward
tests. Cumulative position-resolved codec features are cached offline, never replaced
by one mean vector per hierarchy level.

The renderer's condition is an explicit selected verified path, encoded as a bounded
sequence of node IDs plus preference. This information is available after planning
at inference, so it is not future-state leakage. Training on oracle-selected paths
is renderer supervision, not evidence that the learned policy found those paths.
Canonical displayed answers remain exact structured paths. Free text may be wrong.

For the pilot renderer, use a documented mixture objective: .5 code-position-average
CE + .5 level-average CE. Uniformly sample a hierarchy level and multiply its mean
CE by `.5 + .5*12*ell_j/641`. This unbiased estimator of the mixture deliberately
gives coarse levels more supervision than position-average NLL alone. Keep ordinary
position-average NLL as an evaluation metric. This is a deliberate modification to
the sparse-level default in the review table; level 0 would otherwise receive very
few useful local updates. After 200 updates, 20% of training microbatches corrupt 5%
of preceding code IDs; rebuild their cumulative conditioning through the frozen
codec. Report teacher-conditioned and free-generation outcomes separately.

## 5. Search semantics

Single-agent maximizing PUCT. A node stores exact current node, graph, fixed
preference, visits and legal edges. Expansion uses policy priors/value only; no
NCM call inside search. Root visits equal total simulations. Backup multiplies by
gamma**cost at every traversed edge, starting from terminal value1 for goal or0
for dead end/cycle/depth cap. A path-dependent cycle cutoff must not be reused as
a globally proven losing value in a transposition table.

Search is evaluated after structured training. It is not trained by differentiating
through tree decisions. All outputs are verified by the exact environment. Report
simulations, actual transitions, network calls and wall time, not just nominal N.
Root mixtures of different *reward preferences* are not comparable values; keep
separate fixed-preference trees and compare verified paths under a specified cost.

## 6. Execution and acceptance

One exclusive heavy job across v3/v4; clean SIGTERM save; old runs preserved.
New 24-hour maximum includes setup performed for the replacement and all stages.
Source/config/data/asset identities, optimizer, step and all sampled RNG state are
saved. Deadline never extends on resume. Resource stop: available RAM<1.5GiB,
disk<3GiB, or unplugged battery<20%. Stop on nonfinite loss/gradient. Retain latest
and best validation checkpoints separately. Final test is never a selection input.

Required before launch: oracle and graph invariants; nontrivial solve examples;
permutation/batch/padding invariance; illegal-action rejection; correct discounted
search backup; official checkpoint/shape/logit parity; frozen-weight hash after an
adapter step; finite CPU/MPS forward/backward; complete checkpoint resume; short
memorization control. Passing these verifies implementation, not research merit.

Staged pilot: matched structured controls with two seeds, explicit-preference
training, then optional verified-plan NCM LoRA adaptation if its preparation and
retention checks pass. MCTS is evaluation-only. Gates produce explicit pass/fail/
inconclusive records. Failure of a quality gate must not be labeled a successful
new architecture. No automatic scale-up or fabricated certainty about results.

## Primary references

- [Official NCM paper](https://arxiv.org/html/2610.14162v1)
- [Pinned NCM implementation](https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203)
- [Official checkpoint release](https://huggingface.co/hazyresearch/ncm-checkpoints/tree/d342ce8aaea58830eef3b8ad473664673e41f692/lm1b)
- [LoRA paper](https://arxiv.org/abs/2106.09685)

The original proposal's Sections1–7 are feasible after the conditioning, spatial
state and sampling corrections above. Sections8–10 need the corrected supervision
and staged claims. Sections11–14 need discounted/cycle-safe search and exact resume
tests. Sections15–17 are retained as research discipline; their full matrix is a
follow-up program, not a claim that every variant fits one local24-hour pilot.
