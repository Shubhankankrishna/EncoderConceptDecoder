# Native32 paired GRAM startup report

The predecessor stopped cleanly at step **17,085**. Its final resumable and inference checkpoints are preserved. The new 30M GRAM arm is active; the matched 10M baseline is queued for the next exclusive time slot. Both start fresh.

| Setting | 30M arm | 10M baseline |
|---|---:|---:|
| Unique trainable parameters | 29,944,333 | 9,963,469 |
| Prediction target | 32 subword IDs | 32 subword IDs |
| Observed context | 256 subword IDs | 256 subword IDs |
| Target hierarchy | 161 codes / 12 levels | 161 codes / 12 levels |
| Effective batch | 4 × 2 | 4 × 2 |
| Fast / slow applications | 4 / 2 | 4 / 2 |
| Short MPS preflight seconds/update | 0.639 | 0.434 |

Both use FP32, hybrid position-level context plus summaries, the same data sampling sequence, and the same learning rate at the same update index. Their per-arm budget is approximately 11.61 hours each, alternating in 30-minute slots. Equal-step inference snapshots supplement the wall-time comparison. The hard deadline is **2026-10-20T01:06:12.696623+05:30**; one shared 24-hour clock includes preparation, switching and final evaluation. Short preflight timings are not sustained-throughput guarantees.

The native32 codec is an explicit adaptation of the pinned official NCM checkpoint, not an official pretrained short checkpoint. Its encoder/codebooks/quantizer remain frozen; its decoder was adapted on training chunks. CPU reload reconstruction on 512 validation chunks reached **98.76% non-PAD token accuracy**, **97.56% first/last-token accuracy**, and **71.48% exact non-PAD block accuracy**. The frozen adapted codec has 83,056,256 parameters. No codec reconstruction score is a claim about GRAM-generated language quality.

The cache contains **90,743 unique native blocks** and **64,544 causal windows**: 51,885 train, 6,361 validation and 6,298 final-test windows. Family isolation and cross-split exact-block filtering are checked. One test family touched by the initial compatibility probe was excluded in full from final scoring (five windows). See probe_scope.json for the exposure audit. This is a small reused public corpus, not an independent external benchmark.

Validation saves prior code likelihood, shuffled-context sensitivity, four rolling generated chunks, repetition, adjacent boundary text and end-to-end content-token throughput. These are diagnostics; semantic relevance and boundary coherence still require reviewing generated text. Changed-codec likelihood is not directly comparable with the old run.

**Verification:** 89 Python tests and four frontend tests passed. Both production-shape MPS models passed objective agreement, finite gradients, fixed-batch fitting, exact optimizer/RNG checkpoint recovery (zero measured weight difference), inference-export equivalence and rolling codec-feedback generation. The actual trainer resumed a saved time slice and reproduced its data-sampling sequence. Encoding interruption/resume was tested. Production sources and snapshots, configurations and cache file hashes match their registered identities. Only one heavy job is active.

Dashboard: <http://127.0.0.1:8766/?arm=30m#overview>; choose 10M in the model selector for the baseline. Full preflight evidence is in preflight.json; live process checks are in the run's launch_verification.json. The Mac was on battery during launch: connect AC power for sustained training. Guards stop below 20% battery unplugged, 1.5 GiB available RAM or 3 GiB disk headroom.
