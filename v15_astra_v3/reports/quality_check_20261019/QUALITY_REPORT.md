# Quality check after the user's concern

The model is not yet producing useful coherent, context-sensitive language. The fresh saved checkpoints confirm the concern; healthy gradients and passing engineering tests do not establish language quality.

There was also a monitoring error: the 30M dashboard still displayed step-0 samples after training reached step 2,706. Initial validation finished roughly 19 active seconds into training, its next evaluation required another 1,800 active seconds, and the first training slice ended at 1,800 seconds. Thus no post-initial evaluation occurred during that first slice. This observation does not excuse the fresh trained model's poor text.

The supervisor was briefly stopped cleanly, both latest checkpoints were evaluated sequentially under the exclusive job lock, and the same experiment was resumed with the same source/configuration/data identities and absolute deadline. Optimizer state, model weights and training RNG were not changed by the evaluation. The diagnostic snapshot copies are in snapshots/. No final-test examples were scored.

| Snapshot | Arm | Step | Code NLL | Shuffled minus correct context NLL | Greedy samples containing “cup” |
|---|---|---:|---:|---:|---:|
| latest | 30m | 2706 | 7.2110 | -0.000082 | 16/16 |
| latest | 10m | 2321 | 7.2058 | -0.000072 | 15/16 |
| equal_step | 30m | 1000 | 7.2872 | +0.000349 | 16/16 |
| equal_step | 10m | 1000 | 7.3148 | -0.000219 | 9/16 |

All comparisons use the same 16 validation examples. Lower code NLL is better; it is not text perplexity. Latest snapshots have unequal update counts and active training time. Only the step-1000 pair is matched in updates, sampled training examples and learning rate. At that point the 30M advantage is 0.02757 nats/code (exploratory paired-bootstrap interval 0.00852 to 0.04974 on this small fixed panel). This small-panel likelihood advantage does not establish better generated language.

The training-fitted, add-one per-level frequency baseline scores 7.3209 nats/code on this panel. The latest 30M score is only about 0.110 nats/code better. Output position and gold earlier code levels can help likelihood without successful use of preceding document context. The near-zero average shuffled-context difference shows no aggregate predictive benefit from correct context here; it does not prove that every prediction is invariant to context.

Fresh sampled text is word salad. Greedy samples still frequently collapse to “cup.” The old dashboard emphasized greedy output, whereas the new primary continuation samples categorical codes at temperature 0.8 over four chunks. Lower repeated-word counts under this different decoding protocol must not be called improved coherence. The 98.76% codec reconstruction score applies when encoding real text and decoding its codes; it is not a generation score for arbitrary model-produced codes.

The 30M checkpoint has received 21,648 sampled target-window examples, about 0.42 training-window equivalents over 51,885 windows. This is early training and cannot establish that the architecture can never learn. Equally, there is no basis to promise that simply waiting or enlarging it again will solve the problem.

Next investigation should prioritize a tiny-set memorization control, context-ablation/gradient checks, and teacher-conditioned versus free coarse-prefix decoding. The custom generator summarizes earlier target levels into one vector per level rather than consuming the NCM paper's position-resolved cumulative reconstruction. That remains a candidate representational bottleneck. Its causal role has not been demonstrated by this check, and changing it requires a separately versioned control rather than editing active model sources.

Telemetry now exposes the evaluation step, evaluation age and training-step lag, warns when samples are still from step zero, and presents low repetition without a green quality endorsement. Twenty-four telemetry backend tests pass, including the regression for stale initial samples; JavaScript syntax validation passes. These telemetry changes leave the registered training code unchanged.
