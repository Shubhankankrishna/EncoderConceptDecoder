# Reduced Q5 development comparison

Status: completed. Created: 2026-10-10T21:26:47.543135+00:00.

Exploratory single-seed comparison with unequal training budgets. No equal-compute MCTS superiority claim.

Seed 921 only. The existing MCTS round was allowed to finish. GRPO is the only baseline. Sealed tests were not used.

| Arm | Standard | Hard10 |
| --- | ---: | ---: |
| reference | 83.20% | 33.98% |
| EI | 85.16% | 33.01% |
| grpo | 0.00% | 0.00% |

GRPO exposure and completed rounds: {"problems_per_round": 512, "rounds": 2, "measured_calibration_seconds": 25.271631240844727, "conservative_fit_benchmark": {"seconds": 2.0734801292419434, "examples": 128, "synthetic_labels": true, "discarded": true}, "available_training_seconds": 31804.960880041122, "projection_factor": 1.25, "chosen_utc": "2026-10-10T21:10:05.207735+00:00", "calibration_is_not_scientific_result": true, "projected_training_seconds": 1750.0871276855469}; [{"status": "completed", "round": 0, "problems": 512, "verifier_calls": 4096, "accepted": 2120, "fit": {"examples": 2656, "loss": -156.2886975668814, "completed": true}, "active_seconds": 755.5236158370972, "adapter_sha256": "162d209204403b3d22811cd74d50d507f954d22db4b69ad60af87b3cbfea045b"}, {"status": "completed", "round": 1, "problems": 512, "verifier_calls": 4096, "accepted": 0, "fit": {"examples": 0, "loss": null, "completed": true}, "active_seconds": 43.68242597579956, "adapter_sha256": "c937ad5938bac5a0244ac9d5ff7da1e75a088256547ffbf41c091b984f9dba3c"}]

Paired EI minus GRPO: {"standard": {"delta_percentage_points": 85.15625, "a_only": 436, "b_only": 0, "n": 512}, "hard10": {"delta_percentage_points": 33.0078125, "a_only": 169, "b_only": 0, "n": 512}}

Limitation/status reason: none.

Original Q6/QF remain held. This artifact does not mark the original full Q5 completed.
