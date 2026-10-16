# v4.1 tiny experiment — registered before training results

User authorisation: 2026-10-02, requested deep v4.1 testing against the previous
stacked-transformer baseline, including the available GPU and approximately
3–4 hours of work. This authorises the draft's tiny gate; no full-scale run is
part of this experiment.

## Architecture and controlled comparison

Reuse the previous tiny configuration: d=64, k=6, four-head attention-only pool,
no decoder Engram, encoder F loop=2, core four layers with two passes. Replace
decoder layer 2 with full causal same-document token-memory retrieval. It uses
private-D queries, tok K/V, no RoPE or short convolutions, four log temperatures.
Expected compute parameters: v4 1,212,032; v4.1 1,212,036; stack control 1,212,036;
LLM 1,206,864. The full-size v4.1 count must be 19,175,580.

Primary comparisons: v4.1 minus no-R v4, v4.1 minus LLM, stack minus v4.1,
and trained loop-off minus v4.1. Shared initial v4 backbone weights are preserved
for a fixed seed. All original v4 files/results remain untouched.

Keep B=4, T=480, AdamW (0.9,0.95), peak LR=.002, warmup=400, cosine floor=.1,
weight decay=.1, clip=1, real-token budgets and document padding identical.
Text budget=5M per seed, seeds 0,1; pinpad budget=6M per seed, seeds 0,1,2.
Run new v4.1 arms and fresh baseline checks. Reuse existing no-R/LLM results
only with explicit provenance and successful row, state, backend-parity checks.
Final report separates reused CPU baselines from fresh runs; backend training
differences are not treated as exact checkpoint equivalence.

## Correctness before training

CPU float64 checks: parameter ledger; inherited-weight identity; R manual
attention oracle; causality; document isolation; global and R-local token reach;
private-plan reach and legal null plans; no decoder Engram; full-model causality;
gradient reach and codebook isolation; stack plan leakage as a positive control;
block/full parity; padding and chunk-boundary cases; same actual runner rows
across all arms. Deliberate defects must be caught by assertions: noncausal R,
cross-document R, D K/V leakage, off-by-one plan, local-only R.

CPU/MPS float32 forward and every-parameter gradient parity are required before
MPS training. Test finite gradients and loss reduction on both text and pinpad.
Benchmark CPU and MPS with equal B/T and explicit synchronisation; use the faster
verified path. Compilation, if needed, needs its own gradient parity test.

## Measurements and decision rules

Text: held-out NTP, per-row paired differences, retrieval exact-match families,
train/validation filler gap, component switch-offs. R text guard: v4.1 minus
no-R <= +.02 nats. Original LLM parity guard: +.05 nats.

Pinpad: teacher-forced optimal actions, action NTP, first action toward a pair's
second colour (b2), position accuracy; held-in, held-out compositions, and the
12-visit length-generalisation task. Closed-loop sampled and greedy policies
on the same 256 layouts; completion, task/composition rate and pair integrity.
Linear probes use only causally legal token sites; code/subgoal NMI is labelled
non-causal. Core pass-1/pass-2 probes use only the newest completed chunk.

A component switch-off measures checkpoint reliance; trained arms measure
training-time value. Include core_off, plan_swap, M1, loop_off, R_local,
core_off+R_local, u_off, c_off, shared FFNs off, and encoder Engram off.
Core route alive: b2 drop >=2 paired SE or text NTP increase >=.02 nats.
Report an idle core plainly even if headline capability improves.

R outcomes: closes LLM gap / partial improvement / no detectable effect / harm.
Use draft v4.1 section 9.4 definitions and text guard. F loop retained only if
trained loop-off worsens b2 >=2 SE or text >=.01 nats. Also report seed differences
and a seed-level uncertainty estimate; item-level SE alone is not independent
evidence over training seeds. Curves still rising at budget are labelled
budget-limited. No extension or cherry-picked checkpoint based on held-out scores.

## Execution and reporting

Benchmark first, then prioritise a complete seed-0 v4.1/LLM/no-R comparison and
v4.1 multi-seed replication, followed by stack and loop-off arms. Exact completed
and outstanding arms, throughput, wall time, devices, hashes, token counts,
checkpoints, all failures and limits are recorded. If the 3–4 hour allowance is
insufficient, finish primary comparisons and report incomplete optional arms
without representing pilots or partial budgets as final evidence.

The final report must distinguish implementation correctness, learning outcomes,
generalisation, and latent controllability. No internal RL claims are earned by
the imitation-learning experiment. No source paper's result is attributed to
this model without a corresponding local measurement.

## Exploratory diagnostics added during the first seed

Specified after early periodic readings and before any final checkpoint result;
these are secondary, not pre-registered primary evidence. Fixed 128 seed-37
held-out prefixes at the first pair-second action; swap the target location with
an unvisited colour, require disjoint optimal actions, replay history to prove
it remains legal. Report fact-following under intact/core-off/R-local/plan-swap.

On the same original prefixes, freeze weights and optimise offsets at C, S1 and
u for ten steps, learning rates .001/.01/.03, initial offset zero. Oracle objective
is optimal-action-set probability. Run all items, report initially-wrong fixes
and initially-right damage. Compare random projected steps, scaled to oracle
projected step norms; report achieved norms. This is oracle controllability,
not a sparse-reward controller or reasoning proof. Preserve each input site's
baseline radius. All offsets are restricted to complete chunks of the prompt.

2026-10-02 13:12 IST, after seed-0 final switch-offs: core_off has a large effect,
plan_swap a small one. Add explicitly post-hoc constant-plan and zero-content
controls. Constant u/c_pred are estimated from 16 independent training-stream
batches (seed 997), then held fixed for all evaluation examples. Zero-content
keeps learned legal nulls and slot offsets but removes both u and c_pred.
Measure plan variation and causal fact dependence; this tests whether the
core_off effect is mainly the unusual learned-null replacement. An additional
trained no-plan arm retains all NCP/VQ terms and parameters but disables u/c_pred
in the decoder throughout training. It tests the value of learned plan content,
not the value of all auxiliary training. It is exploratory, not a primary arm.
