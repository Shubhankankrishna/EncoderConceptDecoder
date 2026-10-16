# V7 tiny experiment report

Generated UTC: 2026-10-04 20:02:23. Observed queue status: failed.

This is a bounded tiny-model study. Numerical/caching smoke checks establish implementation contracts; they do not establish model quality or scaling.

V7 foundation: **5,365,938** active parameters. Ordinary Transformer: **5,366,208** (270 more, 0.0050%). Exploration adds **41,472** parameters; deployment has **5,407,410**. A resident frozen teacher is excluded from active deployment counts.

Common foundation budget: **4,000** scored targets per arm and seed.

| Run | Completed scored targets | Train status | Fresh text NLL | Standard strict | Hard strict |
|---|---:|---|---:|---:|---:|
| A_alpha0_s171 (dev) | 1,654 | pilot_prefix_completed | 7.652 | 0.0% | 0.0% |
| A_flat_s171 (dev) | 1,654 | pilot_prefix_completed | 7.649 | 0.0% | 0.0% |
| A_ncp_s171 (dev) | 1,654 | pilot_prefix_completed | 7.652 | 0.0% | 0.0% |
| B0_ncp_s171 (dev) | 1,654 | pilot_prefix_completed | pending | pending% | pending% |
| pilot_flat_lr0.001 (dev) | 1,664 | pilot_prefix_completed | 7.639 | 0.0% | pending% |
| pilot_ncp_lr0.001 (dev) | 1,664 | pilot_prefix_completed | 7.637 | 0.0% | pending% |

Matched foundation differences use whole held-out document resampling for token-weighted NLL, and paired complete problems for task success. Positive task differences favor the first arm; negative NLL differences favor it.

## Refit, exploration and deployment


## GradCuit and AB-MCTS

Development lock: {}


## Limits and interpretation

NCP uses feature-mean forecast/VQ losses and bounded routing adaptations. GradCuit uses bounded concept controls and response-mean rejected log-probability descent, rather than the paper’s raw hidden prefix/sequence-sum objective. AB-MCTS-A searches realized complete answers; latent-only dynamics and prefix-tree MCTS are outside this run.

Three separately trained seeds are listed individually; problem-level bootstrap intervals are conditional on each trained model. Larger widths, long-context behavior, BF16, distributed training and LLM-scale gains are not established. Semantic deduplication and graph-isomorphism independence are not claimed.

Token budgets are matched; parameter counts are almost exact. Additional compute for the V7 recipe, tuning and inference is recorded rather than treated as free. A partial endpoint cannot support a matched-completion comparison. Inference port interventions are distribution-shift diagnostics, not retrained ablations.

**This report is a smoke pipeline test only. No quality conclusion is permitted.**

Failure: {"error": "RuntimeError('Source changed after experiment lock; refusing mixed implementations')", "traceback": "Traceback (most recent call last):\n  File \"/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py\", line 183, in main\n    try:run(q)\n        ^^^^^^\n  File \"/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py\", line 105, in run\n    d=q.eval(f'B0_ncp_s{seed}',b0);q.audit(f'B0_ncp_s{seed}',b0);base=devs[('ncp',seed)]\n      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py\", line 68, in eval\n    self.check_source();self.stage=name+('_test' if test else '_dev');self.status();out=Path(path).parent/('test.json' if test else 'dev.json')\n    ^^^^^^^^^^^^^^^^^^^\n  File \"/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v7_tiny/code/drive.py\", line 39, in check_source\n    if implementation_hash()!=self.sha:raise RuntimeError('Source changed after experiment lock; refusing mixed implementations')\n                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nRuntimeError: Source changed after experiment lock; refusing mixed implementations\n", "stage": "B0_ncp_s171"}
