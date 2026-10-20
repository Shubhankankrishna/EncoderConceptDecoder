# GRAM P1 launch record — 18 October 2026

Status: launched. Read `status.json` and `gram/status.json` for current state.

- Start: **2026-10-18T22:31:01.399632+05:30**.
- Absolute experiment deadline: **2026-10-19T22:31:01.399632+05:30**.
- Training deadline: 2026-10-19T22:21:01.399632+05:30; remaining ten minutes are reserved for final validation and one sealed-test panel. Validation overhead is inside the 24-hour budget; preprocessing and acceptance tests were outside it.
- GRAM trainable parameters: **10,000,333**; deployment parameters: 9,762,828. The separate frozen official NCM codec has 111,404,672 parameters and is not included in the GRAM count.
- Fresh initialization, seed 151803; no old action checkpoint or preflight fitting weights reused.
- MPS FP32, batch 4, accumulation 2, one CPU thread. The actual smoke update took approximately 0.60–0.66 seconds after initialization. Sustained fanless throughput may differ.
- Checkpoint after the first update, every five minutes, and at a clean stop. Initial validation and then every 30 minutes use the same 64-example validation panel. The best validation checkpoint and latest resumable checkpoint are both retained.
- Fixed deadline survives bounded crash retries. Memory/disk/battery guards stop cleanly; a run-local STOP marker requests a checkpointed shutdown. An exclusive lock prevents concurrent v3 heavy jobs.
- A process-tied caffeinate assertion prevents idle sleep while the supervisor runs. Keep AC power connected and the lid open for the overnight run.

## Dataset and representation

The curated cache contains 3,275 complete FineWeb-Edu documents from the existing local source, with source attribution in `../../artifacts/fineweb_p0_20261018/documents.manifest.json`. Earlier held-out document identities and websites are excluded. This is reused public data, not a new independent corpus.

Frozen CPU encoding completed all **23,955 unique 128-ID blocks** in batches of eight. The final dataset has **13,974 train / 1,722 validation / 1,707 test windows**. Each window observes two prior blocks and predicts the next. Hostname-family and duplicate grouping precede the split; 46 windows touching two exact blocks shared across splits were removed. The accepted windows have zero exact text-block overlap across splits. Near-duplicate sketching is approximate, not a proof against all semantic overlap.

Official LM1B NCM codec: 12 levels, 641 codes per block, **8,192 choices per level**. The frozen codec state stayed unchanged. On 512 seeded blocks, token reconstruction agreement was **96.6994%**, while exact block recovery was **1/512**. This codec is lossy and uncased. Code likelihood is not text perplexity, and the project does not yet demonstrate a compute saving against a matched text model.

Dataset SHA256: `5dd9ebd6621d02b194f7030aba8fcb64b80cdc245354aa64b4d939af2039d1b5`.

## Acceptance evidence

- `../../reports/tests_launch_20261018.log`: **57 CPU tests passed**, including data preparation interruption/resume and exact training crash/resume.
- `../../reports/preflight_raw_20261018/report.json`: production-size MPS passed; CPU/MPS loss difference 0; fixed-batch loss 9.0144 → 8.1173 in 24 updates; resumed next-update parameter difference 0; posterior removal preserved prior generation.
- `../../reports/launch_smoke_20261018/acceptance.json`: actual production CLI completed four MPS updates, integrated validation/decoding, and atomic checkpoints.
- `../../reports/raw_cache_acceptance_20261018.json`: checksums, split integrity and codec audit.

## Scope and prior experiment

This is **P1 raw-text next-concept pretraining**, the first empirical stage in the reviewed architecture. It does not train the action encoder, candidate critic, global strategy z, search, or continual-learning system. The 11-problem multi-route fixture is not a production training corpus. No automatic HRM run follows. A single seed and a small reusable corpus cannot establish GRAM superiority or full architecture success; compare later controls on the same budget before making that claim.

The original 48-hour supervisor and trainer stopped cleanly at step **75,878**. Its pending HRM phase is cancelled. Recoverable checkpoints and shared official codec assets are retained; original source/data/plan checksums still match the protected snapshot. Details: `../../reports/retired_previous_run_20261018.json`.
