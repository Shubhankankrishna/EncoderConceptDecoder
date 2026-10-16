# Continuation after evaluation JSON export failure

The user authorized resuming the experiment after the queue failed at 12:36:38 IST on 6 October 2026. Both 2,448,640-parameter hierarchical foundations had completed 24,000 updates and 576,000 GraphNav examples each. The NCP checkpoint's trained decoding audit passed; failure occurred while saving its development evaluation, with `TypeError: Object of type bool_ is not JSON serializable`.

The numerical criteria in `evaluate.py` are unchanged. Competence and prior-coverage flags now return native Python booleans. The first error appeared when the final NumPy comparison was reached in an otherwise passing competence expression. Regression checks exercise the actual failed evaluation's metrics, stdout serialization, and both passing and failing prior-coverage outputs. Coverage fixtures are engineering tests, not model-quality measurements.

The continuation manifest now identifies both completed foundations by checkpoint digest, training-summary digest and original training implementation. Resume validates each arm's settings, data manifest and unchanged training files before reuse. The ordinary Transformer remains a fresh training run. The completed alpha-zero development evaluation is reused with its original source provenance; the failed NCP evaluation is recomputed and exported normally. The scientific queue sequence and diagnostic-only score policy are unchanged.

Checkpoint SHA-256 values:

- Alpha-zero: `720bb818dab7039d81943e9c40f88686afe73204ed7d59d8b3ebdc0981292b9c`.
- Delayed NCP: `d22a4ff4a00b7b3dc82b07aa3f96a1659b2153acaaa75ca0bb548654066ee0ce`.

The model, dataset, training objective, optimizer implementation and eager generation implementation are unchanged. Prior code, lock, continuation manifest, test records, failure, report and unfinalized NCP result are preserved in `spec/recovery_json_2026-10-06`. The current revision reran its execution preflights; `tests/json_recovery_contracts.json` and `tests/recovery_resume_contracts.json` document the new regression and reuse checks.

The original start remains 08:33:48 IST, the final-testing reserve begins at 19:03:48, and the ceiling remains 20:33:48 today. Recovery time consumes that same window. Budget-dependent incomplete work must remain explicit. This continuation adds no training seed and does not constitute architecture or scaling confirmation.
