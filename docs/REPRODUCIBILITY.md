# Setup and reproducibility

[Project home](../README.md) · [Chronology](../EXPERIMENTS.md) · [Working guide](WORKFLOW.md)

This is a collection of version-specific research systems, not one interchangeable package. A fresh clone preserves source and evidence, but does not contain local checkpoints, bulk datasets, or all prepared evaluation panels. There is no claim that every historical run can be reproduced from code alone.

## Lightweight repository validation

Use Python 3.12 and run `python3 tools/catalog.py check` and `python3 tools/catalog.py scan` from the repository root. No scientific libraries are required. CI uses the same checks. Syntax checks do not establish runtime correctness.

## Find a version's execution context

1. Find the family in the chronology and the run in the execution index or machine-readable catalog.
2. Read its effective configuration, data/tokenizer manifests, source snapshots, deviations, and evaluation protocol together.
3. Recover required local artifacts by relative path and checksum from the artifact inventory. Availability is local; the inventory is not a download service.
4. Follow that version's own README, protocol, or launch script, using a fresh output directory. Read scripts before running them: some historical launchers start long training, wait for other jobs, or refer to this Mac's original paths.

| Versions | Setup and execution evidence |
| --- | --- |
| Original components and V1–V3 | Each component folder's specification and harness; each `run_*` folder's `code/`, run configuration, status, and decisions. The old local environment was Python 3.12.8 with system packages; the environment folder is not published. |
| V4–V9 | Version-specific README/preregistration, run configurations, drivers, and recovery notes. Shared code can be imported from earlier version folders, so keep the whole repository tree available. |
| V10–V10.2 | The V10-10M requirements file records PyTorch `>=2.10,<2.11`, NumPy `>=1.26`, tokenizers `>=0.20`, PyArrow `>=17`, requests `>=2.32`, and psutil `>=5.9`. Treat these as recorded constraints, not a verified universal lockfile. Read each version's README and data audit. |
| V10.3–V12 | Their model/tests/train/evaluate modules, run manifests, paired comparisons, and source identities. Do not substitute current configuration for a saved historical run. |
| V13–V14 | Version-specific base/post scripts, effective configurations, manifests, deviations, diagnostics, and final result JSON. Saved V14 configuration records PyTorch 2.10.0 and seed 1301. |
| Original V15 / Astra designs | Analytical specifications and review checks. The later implementations differ from these proposals; see the Astra guide. |
| Astra M2 / V3 | Their READMEs, recorded environment, codec/data manifests, run plans and source hashes. The local venv is omitted. The official frozen codec and upstream checkout must be restored separately. |
| Astra NCM V4 / follow-up | Separate graph-planner and language-renderer stages; retain the official 128-token hierarchy and pinned generator/codec assets. Read the follow-up protocol before running; later-stage completion is not equivalent to faithful free generation. |

## Restore the Astra external dependencies

See [Astra progress](ASTRA_PROGRESS.md) for the actual run statuses. The local `v15_astra_m2/references/ncm` checkout is intentionally omitted. Restore [HazyResearch/ncm](https://github.com/HazyResearch/ncm) at commit `c70274273f489abc73832b2b98584a8483b6b203` to that path when reproducing these versions. The retained [NCM license](../third_party/ncm/LICENSE) applies to upstream code/configuration, separately from original project code.

The recorded [codec manifest](../v15_astra_m2/artifacts/codec/manifest.json) pins `hazyresearch/ncm-checkpoints` at revision `d342ce8aaea58830eef3b8ad473664673e41f692`; it lists exact asset paths and identities. The [BERT tokenizer provenance](../v15_astra_m2/artifacts/codec/lm1b/bert-tokenizer/provenance.json) pins `google-bert/bert-base-uncased` at `86b5e0934494bd15c9632b12f734a8a67f723594`. Small tokenizer/configuration files are preserved; weight tensors and binary tokenizer caches remain local. Restore artifacts by their manifests rather than silently substituting a newer checkpoint.

The recorded environment was an isolated venv using an Anaconda Python 3.12 / PyTorch 2.10 runtime plus OmegaConf 2.3.0 and Hydra 1.3.2. Consult [the environment record](../v15_astra_m2/reports/environment.txt) for the actual package list. This publication does not create or validate a new training environment.

## Machine and data limits

Many earlier experiments targeted Apple MPS; CPU, MPS, and compiled paths can behave differently. Existing reports record throughput and parity caveats. Some source/doc links contain the original machine's absolute paths; those remain preserved historical context. New navigation uses relative links. In particular, four existing Python files under V4/V4.1 contain the original absolute workspace root and may need an explicitly documented portability change before running elsewhere.

Downloaded third-party papers and upstream checkout material remain local. Published notes contain citations; dataset provenance and licensing remain version-specific. A code license does not grant dataset redistribution rights. Dataset bodies and sealed evaluation examples are omitted from newly generated catalogs.

Live V9 watchdog state/status/log/control files, V13 telemetry HTML/status/log files, and V14 observatory runtime files are generated by existing services and stay local. Watchdog, telemetry, and observatory source remain published. The running V13 generator rewrote its dashboard HTML during the import audit; this generated file is explicitly excluded from the source-preservation ledger and recorded in `experiments/preservation_exceptions.json`. No historical model/trainer source was rewritten by the import. Earlier archival imports can retain fixed watchdog snapshots; they do not represent current service state.

## Evidence priorities

Saved code/configuration establishes what was implemented; final result JSON and explicit completion markers establish what finished. Generated tables assist interpretation. A preregistration describes intended tests. A stale header, smoke checkpoint, or analytical design check does not prove final training completed or a hypothesis passed.
