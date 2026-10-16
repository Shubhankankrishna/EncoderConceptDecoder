# Secondary checks and amendments

These checks were added after the seed-0 results. They are exploratory and are
kept separate from the registered outcome table.

- Constant-plan and zero-content evaluations preserve normal first-five-token
  nulls and slot offsets. Their bank uses 16 independent training-stream batches,
  seed 997. Both pinpad and text are measured.
- A no-plan model was trained on MPS for the full 6M real-token budget, seed 0.
  It keeps the auxiliary objectives and allocated parameters. Its decoder never
  receives the learned u/c_pred contents. The learned null and slot offsets stay.
- At 14:07 IST, after no-plan training finished, a normal v4.1 seed-0 MPS run
  was launched for the same 6M budget. This is a matched-backend control: CPU/MPS
  step parity does not ensure equal trained checkpoints over thousands of steps.
- Counterfactual layouts use fixed legal histories and disjoint optimal next
  actions. Every paired prefix receives the same length and therefore plan phase.
- Natural plan-lag strata are observational: episode layout and path length
  affect which positions occupy a phase. They do not identify a causal effect of
  plan staleness. No padding is inserted to create an artificial phase shift.
- Frozen oracle offsets test control authority, with all learning rates and the
  random control reported. This is neither a learned metacontroller nor RL.

Evaluation additions do not alter primary training. Optional `constant_plan`
and `train_no_plan` are inactive in every registered arm. The MPS control is an
additional run; the registered token budget is unchanged.

At 14:22 IST the scheduler was changed for the remaining registered arms to
run two CPU workers, two threads each. A controlled benchmark, with other CPU
work paused and MPS training held active, measured 5,944 aggregate real tokens/s
versus 4,376–4,645 for one four-thread worker. The current seed-1 run remains
unchanged. Its parent scheduler is held until its child finishes, then an idle
scheduler handoff is recorded. This changes concurrency and thread counts,
not rows, budgets, seeds, architecture or optimisation settings. The original
queue log and both benchmark attempts are preserved. A six-step actual-runner
CPU thread-count checkpoint identity check accompanies the amendment.

At 15:16 IST, after examining the long-episode likelihood/action-accuracy
mismatch, add an explicit NLL decomposition on the existing 64 long episodes:
all-vocabulary action NLL = action-conditioned NLL + penalty for probability
mass outside the four action tokens. Report END probability and before/after
six completed visits separately. This is post-hoc; it does not alter primary
metrics or establish that visit count alone causes a distribution shift.

At 15:46 IST, extend the calibrated plan and fixed counterfactual audits to all
three original v4/no-R reference checkpoints. This checks whether the lower
plan-content reliance is specific to the R model. The same independent training
bank, legal nulls/slot offsets, evaluation episodes and counterfactual fixtures
are used. Constants are supplied at the actual plan-construction boundary; a
temporary evaluation hook supports both old and new models. Only new diagnostic
files are written into copied reference directories; the five hashed baseline
artifacts and original source/checkpoints remain unchanged. Report the paired
difference of intervention effects as a secondary interaction, without changing
the primary R outcome.

At 15:58 IST, audit the saved rollout metric's eligibility denominator. Pair
integrity excludes layouts with no transition from an even colour; its paired
SE uses layouts eligible in both arms. Report those per-seed counts and a
secondary all-layout score assigning zero when no pair was observed. Keep the
registered conditional metric and unconditional valid-composition/completion
rates unchanged. This requires no new inference or training.
