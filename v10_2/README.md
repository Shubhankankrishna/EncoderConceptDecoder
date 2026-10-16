# V10.2 RCDL

Executable RF architecture from [the root specification](../V10_2_RCDL_Architecture.md). Fresh weights, width 128, 12 distinct encoder layers (16 applications), four concept layers with two visits, four decoder layers, K=8, 8→2 pooling, and 20,000,320 CPU Engram parameters. Dense parameter count: **5,944,228**, including the 149,376-parameter training-only forecast readout.

The forecast head receives only PQ features and offset embeddings. Its eight weighted future-token losses supervise both visits, with weights 1/3 and 2/3. The decoder retains encoder CRC and aligned PQ guidance, and adds a separate causal reader over both slots of every completed concept group. The first group is released at prediction row 7. RBF correction remains after the final concept visit, and memory writes are disabled during pretraining.

## Run

From the project root, using the installed PyTorch environment:

```sh
/opt/anaconda3/bin/python -B -m v10_2.launch --run-dir /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/runs/pretrain_16h --hours 16 --batch-size 8 --microbatch-size 8 --foreach
```

The launcher refuses to overwrite an existing checkpoint or start a duplicate trainer. The detached supervisor holds a run lock and uses `caffeinate` to prevent idle sleep. Checkpoints include both optimizers, corpus/Python/NumPy/Torch/Metal RNG states, source dependencies, data identity, cumulative exposure, and elapsed time. Resume requires identical code/data and training settings, plus `--resume`. Do not edit imported Python files during a run. The deadline includes evaluation and checkpoint time, and excludes stopped downtime. No automatic restart or extension is scheduled.

The cleaned corpus and 8,192-token vocabulary are reused read-only from `v10_1/data`. Every file is hash-verified before training. Mixture, context 384, candidate cadence (updates 1,5,9,…), two visits, FP32, and independent dense/sparse clipping at 1 are retained. Dense AdamW uses LR 3e-4, 200-update warmup and a 16-hour cosine schedule; CPU SparseAdam uses five times that LR. Each effective batch contains eight documents.

## Telemetry

New dashboard: <http://127.0.0.1:8768/#live>. V10.1 history remains at port 8767. The new collector is read-only and stores its own SQLite database in `v10_2/telemetry_data`.

```sh
/opt/anaconda3/bin/python -B -m v10_2.telemetry --run-dir /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/runs/pretrain_16h --data-dir /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_2/telemetry_data --port 8768
```

API prefix `/api/v1`: `health`, `runs`, `overview`, `catalog`, `series`, `events`, `evaluations`, `artifacts`, `issues`, `download`, `schema`. Pagination uses event IDs, not step numbers. The event stream preserves complete raw payloads, source identity and timestamps; metric series flatten numeric fields. Training losses are `token`, `future`, `explore`, `selector`, and `total`; legacy latent/VQ/chunk losses are absent.

Every update logs raw objectives, weighted contributions, reduction counts, sampled boundaries, dense/sparse gradients, exposures, timings, code counts, forecast losses by visit/offset, and scheduled/executed candidate state. Candidate utility gaps/regret/agreement are unavailable on unscheduled updates. Component gradients are sampled every 20 updates. Evaluation adds individual route ablations, PQ segment entropy, forecast controls, and a fixed text probe of weighted objective gradient norms/cosines and each reader's output/gradient magnitude.

Routine evaluation is planned every 30 minutes, with eight dev documents per family, up to eight generations per math module and Sudoku difficulty, and a 300-second cap checked between work items. Completed and planned counts are recorded. Initial evaluation has no generation trials. These remain small development diagnostics, not a sealed-test benchmark. Each evaluation records the exact weights SHA256, source/data hashes, NLL panel indices, and generation identity-set hashes.

## Verification

See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) and `validation/`. CPU/Metal contracts exercise full width/depth with a reduced vocabulary/table. Separate full-size Metal smoke/resume and throughput checks use the real vocabulary, Engram and corpus. Testing artifacts are isolated from the fresh production run.

This implementation executes the combined RF arm requested by the user. The proposed A/R/F/RF research comparison, multiple-seed validation, full large-panel evaluation and continual-learning retention experiment have not been performed. Correctness tests and successful startup do not establish superiority over V10.1.
