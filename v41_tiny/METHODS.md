# Methods, checks and reproducibility

## Scope and architecture matching

The experiment implements `draft_v4_1.md` §9.4's tiny gate. All new code and
results live under `v41_tiny`. It uses the existing `run_m2_25M/.venv` and reads
the existing `v4_tiny/data`. Original v4 source and result files are preserved.

| Setting | Tiny v4.1 | Full-size ledger, not trained |
|---|---:|---:|
| Width | 64 | 256 |
| Chunk length | 6 | 6 |
| Local heads / KV heads | 4 / 2 | 4 / 2 |
| Global encoder/core heads | 6 | 6 |
| Head width | 16 | 64 |
| Encoder / decoder FFN | 256 / 256 | 1,024 / 1,024 |
| Core layer FFN / shared FFN | 184 / 256 | 736 / 1,024 |
| Encoder shared FFN | 96 | 384 |
| Core layers / passes | 4 / 2 | 4 / 2 |
| Coarse entries | 64 | 1,024 |
| Residual PQ | 8 × 256 × 8 | 32 × 256 × 8 |
| Pooler | 4-head attention, no FFN | same |
| Decoder | S S R S; private plans | same |
| Decoder Engram reads | 0 | 0 |

Tiny v4.1 adds four zero-initialised temperatures to the exact no-R v4
backbone. Replacing the decoder does not consume additional RNG values, and
all inherited initial tensors are bit-identical for the same seed. The stack
control changes decoder K/V to its gated private-state read in all four layers.
It introduces no new parameters or short convolutions.

The transformer baseline has 16 distinct layers and 22 layer applications,
matching the nominal v4.1 counts. It uses a single token stream, window 60,
full attention every fourth layer and FFN width 304. Its initialisation and
objective differ from v4.1, as in the original comparison. The concept core
works at T/6 rather than T, so equal layer counts do not mean equal FLOPs or
equal depth on every path.

The trained R/no-R contrast jointly changes layer 2's attention range,
positional encoding (local RoPE to NoPE), and four temperatures. It identifies
the value of the R package, not a pure attention-mask effect. `R_local` retains
NoPE and temperatures, but is a fixed-checkpoint intervention. No trained
NoPE-local layer or equal-FLOP transformer arm was added.

## Data and budget

**Text:** 5M real input tokens per seed, seeds 0 and 1. Training mixes natural
text and the existing recall1/carry3/multi4 synthetic families 50/50. Text uses
the existing 8,192-token tokenizer. Validation language loss uses 64 fixed
held-out text rows. Each final retrieval family has 192 fixed items.

**Pinpad:** 6M real input tokens per seed, seeds 0, 1 and 2. Fresh 7×7 layouts
have eight colours, four walls and a start cell. Expert demonstrations use
exact shortest paths, uniformly sampling optimal actions. A 27-token layout
header is followed by alternating action and resulting-cell tokens, then END.
Tasks are latent, with 16 training tasks and 20 held-out three-pair
compositions. Final evaluation uses 192 held-in, 200 held-out and 64 longer
12-visit episodes.

Padding, tokenizer, data seed and stream consumption are shared across arms.
Codebook initialisation consumes two extra stream batches, and the transformer
explicitly consumes the same batches. The actual-runner row test captures the
first four training batches in all four architectures and checks every token
and document ID for equality. Both data streams use private seeded RNGs,
independent of model initialisation and model computations.

Real-token budgets exclude padding but count valid input tokens. Valid NTP
targets additionally exclude cross-document transitions. A run may exceed its
budget by its last batch. It is evaluated at that final checkpoint, not at a
checkpoint chosen for its held-out score.

## Optimisation and execution

All registered arms use B=4, T=480, AdamW β=(0.9,0.95), peak LR .002, 400-step
warmup, token-progress cosine decay to .1 of peak, weight decay .1 and gradient
clip 1. The inherited decay grouping is retained. Codebooks are initialised
with seeded CPU k-means++ and dead entries restarted every 200 steps, including
Adam moment resets. Concept auxiliary weights remain .05 coarse and .10 fine.

Periodic evaluation is every .5M pinpad or 1M text tokens. Atomic CPU-state
checkpoints are written every 500 steps and at completion. All training is
float32 eager execution; `torch.compile` is not used. All registered training
uses CPU. CPU2 and CPU4 variants occur as documented in the scheduler amendment;
each run stores its actual thread count.

Hardware is Apple M2, 16GiB unified memory and four performance CPU cores.
PyTorch is 2.10.0, Python 3.12.8, macOS 27.0.1 arm64. MPS is available; CUDA is
not. Synchronised benchmarks found the tiny CPU path faster than MPS for these
settings. MPS is used for the full-budget no-plan and matched-normal controls,
rollouts and frozen oracle control. Background concurrency makes end-to-end
throughput different from an isolated short benchmark.

The inherited S layers use the verified block-60 local-attention implementation.
R uses full masked SDPA. New stack S layers currently use dense masked SDPA
for their local mask; their mathematical range is still eight. This kernel
difference affects throughput and is not used to claim architectural speed
superiority. The FLOP estimate counts ideal local matrix/attention work, rather
than every operation a dense masked kernel executes.

## Correctness and deliberate-defect testing

The 15 new v4.1 checks cover parameter counts, inherited-state/RNG identity,
R's manual masked-softmax oracle, global and local reach, causality, document
and row isolation, private-plan indices, legal nulls, no decoder Engram,
whole-model causality, block/full parity, future reach from C, gradient routes,
padding/partial chunks and actual runner rows. All pass in float64 with
block=60; they also pass with block=0.

The inherited v4 suite has 15 passing checks and the baseline suite 12. That is
42 distinct acceptance checks and 57 executions including the repeated new
suite. Five new deliberate defects are detected: noncausal R, cross-document
R, K/V from D, an off-by-one plan and local-only R. All 11 inherited baseline
mutants and 13 inherited v4 mutants are also detected: 29 mutants total.

Notable measured tolerances:

- Manual R oracle: maximum float64 error 4.44e-16.
- Full versus block normal-model logits: maximum float64 error 8.88e-16.
- CPU/MPS float32 logits: maximum error 3.58e-7.
- All 489 parameter gradients compared on CPU/MPS; worst absolute error 7.26e-8.
- Six actual CPU2/CPU4 training steps: largest state difference 7.45e-9.
  Exact bit identity was first asserted and failed; that failure is preserved.
  This check supports numerical agreement, not full-training identity.
- The new transformer seed-0 checkpoint has all 246 state tensors bit-identical
  to the previous CPU reference, and every saved teacher-forced and closed-loop
  metric matches. This is stronger baseline-reuse evidence than short parity.
- An MPS engineering pilot trained for 350,186 real tokens / 205 steps, with
  finite losses, decreasing NTP, working codebook initialisation/restarts,
  moment resets, device round-trips and final CPU evaluation.

The first MPS rollout failed because float64 conversion occurred before the
CPU transfer. Moving `.cpu()` before `.double()` fixed evaluation; model
inference remains float32 MPS and seeded sampling uses float64 CPU logits.
Training was unaffected. The failed log and fix are preserved.

## Statistics and interpretation

Teacher-forced pinpad accuracy means the argmax among four action logits falls
in the expert's optimal set. The b2 subset is the first action toward a pair's
second colour, where the pairing structure specifies which target follows.
For comparison statistics, each episode has equal weight. Final runner JSON
also contains action-weighted metrics; those differ slightly and are labelled
separately when used. Text comparison uses equal-weight row losses, whereas
the runner also reports a token-weighted validation loss.

For each matched seed, differences are paired on identical episodes, rows,
probe items or rollout layouts: SE=sd(d)/sqrt(n). The registered aggregate item
SE is sqrt(sum(SE_seed²))/number_of_seeds. This is conditional item uncertainty,
not additional independent training replication. We separately report the
standard error and range of training-seed differences. Three pinpad seeds and
two text seeds remain small samples; the registered “2 SE” lines are decision
rules, not proof of universal architecture superiority.

Closed-loop evaluation uses the same 256 layouts for every model. It offers
only four actions, the environment supplies the resulting cell, and stops at
six colour visits or 100 steps. Temperature-1 sampling is seeded by model seed;
greedy is secondary. Completion is distinct from correct pair order or a valid
task: a policy can reach six visits while repeating the wrong colours.
Pair integrity is conditional on observing at least one transition from an even
colour; layouts with none are NaN. Its paired comparisons omit a layout if
either arm lacks an eligible pair, so their effective item counts can be below
256. Per-seed coverage and a secondary zero-for-no-pair all-layout convention
are shown in the tables. Completion and valid-composition rates use all layouts.

Representation probes train on 400 held-in episodes and test on 200 new held-in
and 200 held-out episodes. Completed-chunk core states are selected with the
same legal plan index; current unfinished chunks are excluded. Majority-label
scores accompany the probes. The reported majority score is the best constant
label on that evaluation set, an upper bound for a training-fitted constant;
it is not a separately learned predictor. Coarse-code/subgoal NMI is explicitly
non-causal. Its 256-permutation null independently relabels subgoal identities
inside each episode while preserving temporal structure.

Fixed-checkpoint switch-offs measure reliance and can be out of distribution.
Trained no-R, stack and loop-off arms measure learning outcomes. `core_off`
substitutes the learned header null at every token; it does not simply remove
plan information. The secondary controls replace valid u/c_pred with an
independent training-bank mean, or zero their content while retaining ordinary
nulls and offsets. Their paired raw arrays are saved.

The trained no-plan arm supplies a structural negative control for the route
gate. Its logits do not depend on core outputs, but `core_off` still lowers b2
from .7967 to .4783 because it inserts the header null into normally zero-content
body plans. Therefore the standard switch-off can falsely label a concept route
as used. We preserve the original registered decision and explicitly reject
that inference; calibrated content replacement and trained controls provide
the relevant additional evidence.

Counterfactuals use 128 fixed held-out prefixes (seed 37). At the first
pair-second decision, the target's location is exchanged with an unvisited
colour. Both histories are replayed to verify identical valid actions and
positions; optimal next-action sets must be disjoint. Unconditional changed-
fact accuracy is paired between models. “Changed correct given original
correct” has a different conditioning subset for each model, so it is
descriptive rather than an unqualified paired comparison.

## Additional GPU training and frozen control

The exploratory no-plan model keeps allocated parameters, the core, NCP/VQ
objectives, learned first-five-token nulls and slot offsets, but disables both
u and c_pred in the decoder throughout training. Wu/Wc have 8,192 allocated
weights but no token-loss use in this arm. Auxiliary losses can still alter
encoder training. It tests learned plan content's value, not whether every
piece of concept training can be deleted for free.

Its matched normal-plan model uses the same seed, MPS backend and 6M token
budget. These runs are secondary and were added after seeing the first primary
results. Their paired item differences therefore establish an exploratory
one-seed finding; they are not relabelled as pre-registered multi-seed evidence.

The oracle freezes all weights and optimises legal C, S1 or u offsets on the
same 128 original prefixes. It starts at zero offset, preserves each vector's
baseline radius and uses ten steps at learning rates .001/.01/.03. The
objective is probability assigned to known optimal actions. All items receive
the same budget. Initially wrong fixes and initially right damage are both
reported. Random projected proposals are scaled to oracle projected step
norms; achieved norms are reported because re-projection changes the norm.

This is a controllability upper bound. It uses oracle labels and a much larger
offset space than an eight-dimensional learned controller, and has no sparse
reward, learned switching or multi-step control policy.
Preserving a sphere radius is not a learned-manifold constraint; an offset can
leave the distribution of normal concept vectors despite preserving its norm.
No matched transformer-residual oracle control was run. Consequently this
experiment does not isolate an advantage specific to the concept interface.

## Reproduce and inspect

From the repository root, use the existing environment:

```sh
run_m2_25M/.venv/bin/python v41_tiny/code/test_v41.py --block 60
run_m2_25M/.venv/bin/python v41_tiny/code/mutants_v41.py
run_m2_25M/.venv/bin/python v41_tiny/code/run_vs.py --name my_v41_pin_s0 --arch v41 --data pinpad --seed 0 --real_tokens 6000000 --device cpu --threads 2
run_m2_25M/.venv/bin/python v41_tiny/code/run_vs.py --name my_v41_stack_s0 --arch v41stack --data pinpad --seed 0 --real_tokens 6000000 --device cpu --threads 2
run_m2_25M/.venv/bin/python v41_tiny/code/run_vs.py --name my_v41_loopoff_s0 --arch v41 --data pinpad --seed 0 --real_tokens 6000000 --device cpu --threads 2 --over enc_loop=1
V41_ROLLOUT_MPS=1 run_m2_25M/.venv/bin/python v41_tiny/code/analyze_vs.py arm my_v41_pin_s0
run_m2_25M/.venv/bin/python v41_tiny/code/report_tables_v41.py
run_m2_25M/.venv/bin/python v41_tiny/code/plot_v41.py
```

Use new run names. The archived experimental scheduler contains a one-session
handoff and should not be replayed on another machine; the individual runner
commands are the reproducible interface. `source_manifest.json` preserves the
original copied code hashes, `baseline_manifest.json` tracks 50 reference
artifacts, and execution/final manifests record the new code. Every complete
arm contains a final checkpoint, final metrics, analysis and per-unit arrays.

The two matched GPU controls can be reproduced with new names and the existing
environment. Normal training uses `--device mps --threads 2`; the no-plan arm
adds `--over train_no_plan=True` to the same v4.1 pinpad command. Secondary
diagnostics are run with `plan_diagnostics.py NAME`,
`counterfactual_v41.py NAME`, `phase_audit.py NAME` and
`control_v41.py NAME`. The last script requires MPS and is intended only for
normal plan-trained checkpoints. The no-plan arm's dormant projections must not
be enabled by bypassing its normal forward function during oracle control.
