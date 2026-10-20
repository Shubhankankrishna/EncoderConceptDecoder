# Astra work: designs, experiments and outcomes

[Project home](../README.md) · [Chronology](../EXPERIMENTS.md) · [Every saved execution](RUNS.md) · [Publication record](PUBLICATION_2026-10-20.md)

This guide covers work added since the original GitHub publication, organized on **20 October 2026, IST**. Existing folders and source snapshots retain their paths and bytes. Import commits use the current date; they do not reconstruct the work's original edit history. The six Astra catalog families separate proposals, mathematical checks, implementations and experimental outcomes.

## Read in sequence

| Step | Material | What it establishes |
| --- | --- | --- |
| 1 | [XHRM-Atlas proposal](../V15-XHRM-Atlas.md) and [review](../V15-XHRM-Atlas_Review.md) | Original synthesis and identified mathematical/engineering limitations |
| 2 | [Astra specification](../V15_Astra.md), [review](../V15_Astra_Review.md), [paper notes](../v15_astra_design/paper_reading_notes.md) | A frozen-codec design, normalized concept mixture, search/sleep contracts and analytical parameter examples |
| 3 | [Reviewed training blueprint](../V15_Astra_Full_Training_Blueprint_Reviewed/V15_Astra_Engineer_Training_Spec.md) and [engineering review](../V15_Astra_Full_Training_Blueprint_Reviewed/V15_Astra_Engineer_Review.md) | More detailed implementation proposal; mathematical checks do not demonstrate learning |
| 4 | [Original multi-strategy draft](../V15_Astra_MultiStrategy_Dataset_GRAM_ActionEncoder_Z_Spec.md) and [revised draft](../V15_Astra_MultiStrategy_Dataset_GRAM_ActionEncoder_Z_Spec%20%281%29.md) | Two distinct preserved texts; the `(1)` file is not silently deduplicated |
| 5 | [M2 implementation](../v15_astra_m2/README.md) and [18 October review](../v15_astra_review_20261018/FIRST_PRINCIPLES_REVIEW.md) | First GRAM-first pilot, codec audit and subsequent review |
| 6 | [V3 implementation](../v15_astra_v3/README.md), [specification](../v15_astra_v3/ARCHITECTURE_SPEC.md), [traceability](../v15_astra_v3/IMPLEMENTATION_STATUS.md) | Corrected infrastructure, raw-text and native-32 runs; stopped outcomes remain visible |
| 7 | [Pretrained-NCM engineering proposal](../V15_Astra_LM1B_NCM_LoRA_Recurrent_GRAM_Z_Action_MCTS_Engineering_Spec.md), [V4 implementation](../v15_astra_ncm_v4/README.md) and [reviewed implementation scope](../v15_astra_ncm_v4/REVIEWED_SPEC.md) | Structured graph-planner controls plus separately adapted pretrained language renderer |
| 8 | [Renderer follow-up](../v15_astra_ncm_v4/followup/README.md) and [final comparison](../v15_astra_ncm_v4/runs/20261019_renderer_prefix_followup/final_results.json) | Residual-label/roll-in correction, selected checkpoints and failed exact-route outcome |
| 9 | [Discrete tokenizer proposal](../Discrete_LatentTokenizer.md), [RePR/InfiLoop proposal](../V15_Astra_RePR_InfiLoop_Architecture.md), [20 October math review](../design_reviews/20261020/design_math_results.json) | Further design work; no training result is inferred |

## Actual run status

| Saved run | Status and evidence | Limit |
| --- | --- | --- |
| M2 `20261018_48h` | [GRAM stopped at 75,878](../v15_astra_m2/runs/20261018_48h/gram/status.json); [supervisor stopped](../v15_astra_m2/runs/20261018_48h/status.json) | Planned full HRM stage was cancelled; smoke HRM is not that comparison |
| V3 `20261018_24h_raw_gram` | [GRAM stop requested at 17,085](../v15_astra_v3/runs/20261018_24h_raw_gram/gram/status.json); [supervisor stopped](../v15_astra_v3/runs/20261018_24h_raw_gram/status.json) | Historical README deadlines describe the launch, not final completion |
| V3 native-32 10M | [Stop requested at 43,181](../v15_astra_v3/runs/20261019_native32_10m_30m/10m/gram/status.json) | Overall comparison stopped |
| V3 native-32 30M | [Slice completed at 27,866](../v15_astra_v3/runs/20261019_native32_10m_30m/30m/gram/status.json) | A completed slice is not a completed scientific program |
| V4 grounded pilot | [Complete status](../v15_astra_ncm_v4/runs/20261019_lm1b_grounded/status.json), [final results](../v15_astra_ncm_v4/runs/20261019_lm1b_grounded/final_results.json) | Twelve structured controls at seeds 42/43, then renderer; completion does not mean all quality gates passed |
| V4 renderer continuation | [Complete status](../v15_astra_ncm_v4/runs/20261019_renderer_prefix_followup/status.json), [final results](../v15_astra_ncm_v4/runs/20261019_renderer_prefix_followup/final_results.json) | One continuation seed; same update ceilings, different elapsed compute |

The V4 graph planners learned the synthetic task, but the initial joint probes did not faithfully render the selected routes. The continuation's untouched test panel contains 72 renderer examples. Its selected models give:

| Model | Selected additional step | Exact routes /72 | Mean route similarity | Code NLL |
| --- | ---: | ---: | ---: | ---: |
| Parent after 12,000 original updates | 0 | 0 | 0.14236 | 0.20997 |
| Existing-training control | 2,500 | 0 | 0.14306 | 0.18388 |
| Generated-prefix/residual-consistent arm | 3,000 | 0 | 0.12940 | 0.24227 |

The selected steps differ from the final 4,000-update statuses because checkpoint selection uses validation. Neither selected continuation produced an exact route. The lower control code loss is not a language-fidelity success. Read the [saved comparison](../v15_astra_ncm_v4/runs/20261019_renderer_prefix_followup/final_results.json) and [diagnosis](../v15_astra_ncm_v4/followup/README.md) for the complete protocol and generated outputs. This organization pass does not rerun or reinterpret the results as stronger evidence.

## Parameter budgets and reproducibility

The design's 7–10M examples count their proposed small codec. The M2/V3 pilot instead reuses an approximately 111.4M frozen official codec in addition to the trainable prior. V4 also reuses the approximately 97.1M official generator: its README reports about 209.35M physical parameters and 875,524 new trainable-module parameters. These are different systems and must not share a single model-size label.

Commands, dependencies, source revisions, effective plans, dataset identities, logs and validation records remain version-specific. Follow the [reproducibility guide](REPRODUCIBILITY.md) and each version README. Do not rerun preprocessing into an existing experiment's output directory. Existing training/dashboard services are outside this publication task.

## What GitHub contains

Source, frozen source snapshots, architecture drafts, reviews, configurations, tokenizers, manifests, manageable result evidence, meaningful logs, plots and dashboard source are published. Checkpoints, tensor caches, raw dataset bodies, downloaded PDFs, temporary upstream source checkout and live telemetry state remain local and have inventory records. Large result files receive structural digests where applicable. No files over 10 MiB are published.

The M2 nested ignore file previously hid all `runs/` and `artifacts/`. That rule was narrowed to preserve useful metadata and logs. The root policy and generated exact-path exclusions still keep heavy/bulk files local. The original experiment/model source is preserved.
