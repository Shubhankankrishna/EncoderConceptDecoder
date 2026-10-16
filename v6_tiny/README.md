# V6 tiny

An isolated experiment implementing revision 2 of `../v6_draft.md` on this machine.

The active gated model has **2,063,904 parameters**. The deterministic transplant has 2,055,424, and the direct-control comparison has 2,057,472. Original teacher modules remain loaded for training/blend diagnostics and are counted separately in resident-memory ledgers.

The run combines TinyStories text with GraphNav symbolic reasoning. Its target duration is approximately six hours, with a seven-hour ceiling and a reserved evaluation window. Short warmed timings are used to plan the queue; thermal drift can change the finish time. `PLAN.json` records the actual start and estimated finish once launched.

Open **REPORT.md** for the evolving experiment report. `runs/status.json` identifies the current stage. Every stage has a log, recoverable checkpoint, configuration and completion/failure record. The driver runs stages serially to avoid Apple GPU contention.

The experiment trains a fresh NCP foundation, a deterministic replacement, and direct/gated Z adapters. It compares whole-schedule candidate selection at beta=0 and beta=.25, extra deterministic training, zero-control behavior, transplant fractions, prior quality/diversity, privileged reachability, and verifier-only GradCuit versus retries and random controls. Attempt-count and elapsed-time comparisons are separate.

Retries and repair attempts reuse the frozen encoder's unchanged prompt features. Core/decoder caches are rebuilt after controls change, including the final prompt boundary that predicts the first response token.

Execution uses FP32, compiled model computations, shared frozen encoder features across candidates and replays, both candidates vectorized into one retained computation graph, and compiled fixed-size incremental caches. The optimizer still uses 64 original examples; vectorizing candidates does not change its loss denominator or effective data batch. Every one of the six recursive block visits has its own cache. Optimizations must pass ordinary-versus-compiled loss/gradient and cached-versus-replayed output checks.

This time budget covers one complete foundation/core seed and controlled variations. Three independent foundation seeds, large-width/context scaling, distributed training and low precision are later confirmation work. Passing software checks does not prove that GradCuit beats retries or that the architecture scales.

The text data/tokenizer in `../v5_tiny/data` are read-only. V6 owns its document split manifests, chunk/document masks, symbolic split files and all results. Evaluation documents are deduplicated against training by exact token content; near-duplicate text remains unmeasured. Evaluation graph fingerprints are excluded from training. Held-out text still originates in the previously inspected V5 validation corpus, so this is a development experiment.

## Files

- `PRE_REGISTRATION.md`: recipe, budget, comparisons and interpretation rules.
- `SPEC_COMPLIANCE.md`: mapping from the draft to implemented/executed checks.
- `PLAN.json`: calibrated queue and runtime provenance.
- `REPORT.md`: readable progress/final report.
- `code/`: independent model, data, trainer, cache implementation, checks, benchmark, evaluator and driver.
- `tests/`: saved correctness outcomes.
- `runs/`: training stages, per-item evaluation, audits and queue status.
- `spec/v6_draft.md`: exact design snapshot used for the run.

The experiment can be launched with the existing repository Python environment:

~~~sh
../run_m2_25M/.venv/bin/python code/drive.py --hours 6 --max-hours 7
~~~

The launch already uses a detached process and prevents idle sleep while that process runs. Do not launch a second copy while `runs/status.json` reports an active driver PID. To resume an interrupted individual stage, use its recorded arguments with `--resume`; source/configuration compatibility is checked before continuing.
