# V15 Astra v3 — isolated review implementation

This revision implements the actionable corrections from `FIRST_PRINCIPLES_REVIEW.md` and the compatible interfaces from `V15_Astra_MultiStrategy_Dataset_GRAM_ActionEncoder_Z_Spec (1).md`.

**18 October launch amendment:** The user explicitly requested replacement of the older run with v3 GRAM first for 24 hours. The old supervisor and trainer stopped cleanly at step **75,878**; the pending HRM stage is cancelled. Best/latest checkpoints, data and shared official NCM assets are retained. Source/data/plan hashes remain unchanged; only the authorized STOP marker and normal final checkpoint/status writes were added.

The replacement is **P1 raw-text next-concept GRAM pretraining**, with the action encoder, critic, global strategy, search and continual-learning stages disabled. **Started 2026-10-18T22:31:01.399632+05:30; ends 2026-10-19T22:31:01.399632+05:30.** See [the new run](runs/20261018_24h_raw_gram/) and its `status.json` for live state; `plan.json` records the exact start and absolute deadline once preparation and MPS preflight pass. Preprocessing precedes the 24-hour window, which includes periodic validation and a ten-minute final-evaluation reserve. No HRM stage is automatically appended.

New heavy-job commands still refuse to start while the old trainer or supervisor is running. A lock prevents concurrent v3 encoding/training/evaluation jobs.

Read [ARCHITECTURE_SPEC.md](ARCHITECTURE_SPEC.md) for the corrected contract and [MULTISTRATEGY_SPEC_REVIEW.md](MULTISTRATEGY_SPEC_REVIEW.md) for the step-by-step assessment of the new draft. [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) maps findings to code and tests.

Implemented:

* Stable near-equality Gaussian KL; terminal versus auxiliary objectives; cached level generation.
* Separately trained direct-action and nonrecurrent controls, same-core HRM/GRAM and parameter-matched HRM.
* Complete observation serialization, immutable hard-task fixtures, exact replay audits, paired metrics and matching bootstrap estimands.
* Raw document provenance/dedup/splitting, causal next-block caches, a budgeted P1 runner and free-generation/retention evaluator.
* Verified alternative-route panels, actor/critic input separation, candidate outcome contracts, persistent strategy state and episode-mixture reduction.
* Horizon-conditioned values, labeled conditional distillation, leave-one-out score gradients, two-slot adapters and bounded versioned memory.

Grounded fixture data: **72 new mazes / 991 transitions**, eight episodes per split and each of 5–9 / 10–19 / 20+ required moves. The independent audit passes, and 53 initial states defeat the fixed next-action distance heuristic. A stricter alternative-route fixture retains **11 problems, only one for training**; it is too small for strategy learning. These fixtures are not used as the new P1 training corpus.

The new raw corpus uses 3,275 complete documents from the existing provenance-preserved [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) cache, restricted to previous training identities and excluding previous dev/test websites. Source hostnames and normalized/lexical duplicate handling define the family split before block windows. The codec processes 23,955 unique 128-ID blocks in CPU batches of eight; 512 seeded blocks receive a reconstruction audit. Any window touching an exact shared block across splits is removed. Corpus attribution and limitations are in `artifacts/fineweb_p0_20261018/documents.manifest.json`; final usable window counts are in the encoded dataset. This is a new model on reused public data, not a newly collected independent benchmark.

Validation artifacts are in `reports/`: test log, parameter ledger, exact-data audit, protected-file snapshots, and final verification. Tests cover causal masks, cached/reference generation, FP64 KL values/gradients, posterior-free inference, multi-phase objectives, exact CPU crash/resume, measurement definitions, route replay, actor leakage, strategy lifecycle, and later-stage storage bounds. Launch validation now passes **57 CPU tests**, production-size MPS fitting, exact stochastic checkpoint recovery, CPU/MPS objective parity, and integrated validation/decoding. The raw codec audit matched 96.70% of non-padding tokens on 512 blocks (only 1/512 blocks exactly recovered). Symbolic observation fidelity for later grounded stages is not established by this raw-text audit.

## Commands from the repository root

The regression suite uses disposable CPU fixtures. Run it when the v3 heavy-job lock is free; its integration tests deliberately acquire the same lock:

```bash
nice -n 19 env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 v15_astra_m2/.venv/bin/python -m unittest discover -s v15_astra_v3/tests -t . -v

nice -n 19 env PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false v15_astra_m2/.venv/bin/python -m v15_astra_v3.audit v15_astra_v3/artifacts/hard_grid_v3 --out v15_astra_v3/reports/dataset_audit.json
```

For a future raw-text experiment, first supply a curated JSONL of documents following `configs/raw_document_schema.json`. Use a new output directory. The following commands are **prepared instructions, not scheduled work**, and will fail closed while the original experiment is active:

```bash
# Replace /absolute/path/to/curated_documents.jsonl with an actual licensed corpus.
v15_astra_m2/.venv/bin/python -m v15_astra_v3.raw /absolute/path/to/curated_documents.jsonl --out v15_astra_v3/artifacts/raw_p0 --history 2 --batch 4

# A short bounded pilot, after the raw cache and reconstruction audit exist.
v15_astra_m2/.venv/bin/python -m v15_astra_v3.train --task raw --arm code_gram --data v15_astra_v3/artifacts/raw_p0 --run v15_astra_v3/runs/raw_gram_seed151803 --config v15_astra_v3/configs/gram_shared.json --device mps --steps 1000 --seconds 1800 --batch 4 --accum 2

v15_astra_m2/.venv/bin/python -m v15_astra_v3.evaluate_raw v15_astra_v3/runs/raw_gram_seed151803/latest_inference.pt --data v15_astra_v3/artifacts/raw_p0 --out v15_astra_v3/reports/raw_gram_validation.json
```

Rerun the exact same training command to resume; changing data, source, seed, configuration, optimizer schedule or wall limit rejects resume. A run-local `STOP` file requests a checkpointed stop. Use a new run directory for a changed experiment.

Grounded scratch controls are separately available through `--task grounded`. `state_action` needs only the verified raw dataset; code arms first need `python -m v15_astra_v3.dataset DATASET`. Code-only symbolic arms additionally require exact parsed-observation round-trip acceptance. Do not mistake the small prepared fixture for a sufficient training corpus.

The original experiment's schedule has been cancelled by the user. The replacement uses `run24.py`, an exclusive supervisor, absolute deadlines, periodic prior-only validation, atomic checkpoints, bounded failure retries, and a sleep assertion tied to the supervisor PID. The later research stages in `configs/next_experiment_plan.json` remain gated. See `reports/retired_previous_run_20261018.json` for preservation details.
