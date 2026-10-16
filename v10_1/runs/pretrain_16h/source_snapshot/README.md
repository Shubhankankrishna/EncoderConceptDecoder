# V10.1 RCDL

Experimental implementation of [the root architecture](../V10_1_RCDL_Architecture.md). The candidate is tested for functional correctness; better trained reasoning and continual-learning retention remain unproven. See [validation results](VALIDATION_REPORT.md).

## Model and data

The full D+J+H+S candidate has **5,731,496 dense + 20,000,320 Engram parameters**. Width 128, 4× FFNs, encoder 4/4/4 distinct layers with two middle visits, four concept layers with two visits, and four decoder layers are retained. Both CRC paths, normalized visit supervision, the chunk readout, and utility-based exploration are active. Engram uses CPU sparse gradients; the dense model uses Metal FP32. GradCuit is zero and the RBF bank stays empty throughout pretraining.

The new `data/` corpus retains the original train/dev identities and fixes serialized math byte wrappers. Its 8,192-token tokenizer is fitted on training records only. Training starts with fresh weights. V10 checkpoints cannot be loaded with this tokenizer. The original corpus and reserved test partition remain unchanged; no test arrays are loaded by this trainer. Final test re-encoding is a later, separate step.

## Run and extension

Use `/opt/anaconda3/bin/python` from `/Users/shubhankankrishna/Desktop/EncoderConceptDecoder`.

```sh
/opt/anaconda3/bin/python -B -m v10_1.launch --microbatch-size 8 --hours 16 --memory-fraction .45
```

The intended run directory is `v10_1/runs/pretrain_16h`. The launcher refuses an existing checkpoint without `--resume`, holds a process lock, takes a source snapshot, and uses `caffeinate` to prevent idle sleep. It does not restart after reboot or override closed-lid sleep behavior. Keep the Mac plugged in and awake.

The run uses effective batch 8, context 384, dense LR 3e-4 with 200-update warmup and cosine decay to 10%, and Engram LR five times the dense LR. Candidates run every fourth update, beginning with update 1. Batch-family probabilities remain 65% text, 20% math, 15% Sudoku; actual supervised-token exposure is logged. These are batch proportions, not token proportions. The inherited document-uniform text sampler remains a limitation.

Budget is **16 hours of active process wall time**, including evaluation and checkpoint I/O, excluding stopped downtime. To extend a stopped/completed run to 24 total active hours:

```sh
/opt/anaconda3/bin/python -B -m v10_1.launch --microbatch-size 8 --hours 24 --memory-fraction .45 --resume
```

This resumes weights, both optimizers, data and model RNG states, and exposure counters. It keeps the original 16-hour cosine schedule; LR stays at its 10% floor thereafter. It does not grant an extra 24 hours or restart warmup. Source/data hashes and batch/optimizer settings must match. Do not edit Python sources during the experiment; a changed implementation requires an explicit new run. Post-training is a separate phase and is not automatically started.

`SIGTERM` to the verified trainer PID requests a graceful stop with `latest.pt` saved. `status.json` and `launch.json` identify that PID. Do not send a signal to an unverified or reused PID.

## Telemetry

The V10.1 dashboard/API is intended at **http://127.0.0.1:8767/#live** and **http://127.0.0.1:8767/api/v1**. V10 history remains on port 8766. The V10.1 collector has its own SQLite database in `v10_1/telemetry_data/` and never rewrites V10 events. It can also run directly:

```sh
/opt/anaconda3/bin/python -B -m v10_1.telemetry --run-dir /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/runs/pretrain_16h --data-dir /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v10_1/telemetry_data --port 8767
```

The trainer appends every update to `train.jsonl`: seven loss components, objective numerators/denominators, dense and coalesced sparse Engram norms, clip factors and implied post-clip norms, LR, candidate execution, selector regret, code counts, both visit losses, example indices, family exposure, source/data hashes, wall time, and memory. Component norms are measured every 20 updates and at update 1. Regret is a training-label oracle diagnostic, not deployable accuracy. Absent candidate measurements have zero denominators and null regret.

Status/console refresh every 20 updates; checkpoints every 10 minutes; host samples every 30 seconds; collector polling every five seconds. Per-update throughput excludes evaluation/checkpoint time; elapsed budget includes them. RSS and Metal counters overlap and must not be summed. GPU utilization, temperature, power and gradient histograms are not measured.

Routine development evaluation runs every 30 minutes: text and symbolic NLL, separate PQ/continuous/both-route interventions, one-versus-two visits, latent errors, code counts, 4 generated answers per math module, and 2 boards per Sudoku family. Evaluation has a 240-second budget and records actual and planned counts if incomplete. Math uses exact rational/decimal/list comparison with required EOS and separately retains raw exact-match counts. Sudoku requires a complete valid board and EOS. Initial evaluation records losses only. Small fixed panels are diagnostics, not a final benchmark.

`/events`, `/series`, `/catalog`, `/evaluations`, `/overview`, and `/download` use the existing schema-1 API. New metrics include `train.loss.chunk`, `train.loss.latent`, `train.gradient.engram_grad_norm`, `train.component_grad_norms.*`, `train.latent_visits.*`, and `train.objective_denominators.*`. Training event times use the recorded trainer clock. Evaluation payloads include example identities and selection hashes. Use event-ID pagination and preserve the distinction between live training, routine dev evaluation, and old offline audits.

## Reproduce checks

```sh
/opt/anaconda3/bin/python -B -m v10_1.tests --device cpu --output v10_1/validation/contracts_cpu.json
/opt/anaconda3/bin/python -B -m v10_1.tests --device mps --output v10_1/validation/contracts_mps.json
/opt/anaconda3/bin/python -B -m v10_1.pipeline_tests
```

Avoid competing Metal jobs during the real run. `validate_training` runs short A/B/C/D/S integration tests with the old tokenizer and in-memory cleaned examples; it is explicitly not the new production-data experiment. `parity` tests CPU/Metal agreement and full parameter counts. Stored validation artifacts identify their source hashes, data scope, and device.

## Interfaces and limitations

`Architecture.arm()` exposes A/B/C/D/J/H/S. Ordinary `V101()` selects the full candidate. `ablate=True` disables both decoder concept routes; `ablate_pq` and `ablate_continuous` isolate each. Candidate scoring uses detached K-way selection then a gradient-bearing winner recomputation, tested against retaining every graph. This equivalence depends on deterministic blocks.

`backward_effective_batch` supports smaller microbatches with objective-specific normalization. H-arm candidate accumulation is explicitly unsupported until independent chunk/candidate boundaries are preselected; direct-batch H evaluation remains available. Full candidate S is supported and tested.

For RBF adaptation, freeze the entire model, derive `model_era` using tokenizer/preprocessing/bandwidth identity, bind it with `bind_memory_era`, and construct a matching `RBFMemory`. `learn_episode` differentiates through both corrected routes and consumes feedback IDs exactly once, including zero-gradient events. Capacity is bounded; no automatic eviction is implemented. These mechanics pass tests, but useful adaptation and retention must be evaluated after pretraining.
