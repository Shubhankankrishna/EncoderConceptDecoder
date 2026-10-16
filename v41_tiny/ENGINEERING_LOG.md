# Engineering log

2026-10-02 13:05 IST: first GPU rollout attempt rejected `.double()` on MPS.
Moved `.cpu()` before `.double()` so seeded sampling retains CPU float64,
while model inference stays MPS float32. Training was unaffected. Original
error preserved in v41_pin_s0_6M/analysis_attempt1.log.
# Scheduler and floating-point audit

The first throughput concurrency benchmark overlapped changing background
evaluation loads. It is preserved but not used to choose the scheduler. The
controlled repeat paused the current CPU training and CPU evaluation workers,
kept MPS training active, and bracketed two concurrent two-thread workers with
single four-thread measurements. Aggregate throughput improved about 30%.
The paused processes were resumed in a `finally` block; their batches and RNG
states were unaffected. Those measurement pauses are included in run wall time.

A six-step actual-runner CPU2/CPU4 checkpoint identity assertion was too strict:
491 state tensors were compared and some differed in floating-point roundoff.
The failed bit-identity result is preserved in `runs/thread_parity.json`.
The subsequent measurement, `thread_parity_numeric.json`, found a maximum
absolute difference of 7.45e-9; all tensors agreed at atol=2e-6, rtol=1e-5.
This is a numerical-characterisation result, not a claim that different thread
counts must produce bit-identical full training runs. Thread counts are retained
in each run's arguments. The primary fresh LLM *did* reproduce all 246 saved
state tensors bit-for-bit and reproduced every rollout metric.

The secondary constant-plan audit was generalised to a temporary hook at the
plan-construction boundary so the same procedure could evaluate the original
v4 checkpoints. Re-running all seed-0 diagnostic units with this hook reproduced
the previous optional-forward override exactly, including NaNs (maximum
difference 0). The result is preserved in `runs/plan_hook_parity.json`.
This evaluation-only change does not alter primary training.

At 16:09 IST, the saved-unit audit found that the seed-0 Engram-off summary
existed without its per-episode arrays. Regenerating that secondary evaluation
from the final checkpoint reproduced every existing summary value exactly.
Thirty arrays were added, and all existing arrays stayed bit-identical.
The previous JSON/NPZ and `engram_units_repair.json` are kept in the run folder.
The checker then continues over all checkpoints, aggregates and counterfactuals;
it does not skip a missing unit or treat file existence alone as validation.

The fresh LLM reproduction retained the original reference's M1 and loop-off
summaries, which also lacked raw intervention arrays. Both were evaluated from
the fresh checkpoint and their old summary values matched exactly. Sixty arrays
were added; all original units, including every rollout, stayed bit-identical.
The original summaries/units and `intervention_units_completion.json` are kept.
