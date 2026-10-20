# Astra publication — 20 October 2026

[Project home](../README.md) · [Astra chronology and outcomes](ASTRA_PROGRESS.md) · [Import commit index](../experiments/imports.json)

This publication preserves the work remaining after the original GitHub setup: XHRM-Atlas, Astra architecture and paper reviews, engineering blueprints, the M2 pilot, V3 raw/native-32 experiments, NCM V4 with its renderer continuation, and the later mathematical design reviews. The seven research import groups are recorded below. All are current-date snapshot imports, not reconstructed historical development commits or runnable releases.

Original model/training source and saved evidence retain their bytes and paths. Repository infrastructure, navigation and the M2 ignore rules are updated deliberately to expose manageable run metadata while keeping bulk payloads local.

## Publication policy

- Publish source, configuration, tokenizers, manifests, reviews, plots, meaningful logs and result files at most 10 MiB.
- Keep checkpoints, tensor caches, dataset bodies, downloaded papers, temporary upstream checkouts and live telemetry state local; catalog them by path, reason, size and stable SHA-256 when possible.
- Mark live telemetry as live without claiming a consistent checksum. Keep its generators and dashboard source in Git.
- Preserve the NCM and BERT notices separately from original project MIT licensing.
- Distinguish stopped runs, completed slices, smoke tests, completed evaluations and failed quality gates. Saved `complete` statuses are recognized alongside older `completed` statuses; plans contribute actual seeds/configurations to the catalog.

## Verification scope

Repository checks cover Python syntax, complete tracked/local inventory coverage, source preservation, navigation, file-size policy, import tags and credential patterns. The publication-rule regression tests use disposable fixtures and do not run experiment code. A separate fresh clone validates the published tree and full tag history. Training and model evaluations are not rerun for publication.

Machine-specific absolute paths remain in preserved historical records. Existing training/dashboard services are not started, stopped or reconfigured. The artifact inventory is identity metadata, not a remote backup of local payloads.

## Grouped commits and archival tags

| Group | Commit | Tag | Research files |
| --- | --- | --- | ---: |
| archive(v15-xhrm): preserve XHRM-Atlas proposal and critical review | [bc63ba7b](https://github.com/Shubhankankrishna/EncoderConceptDecoder/commit/bc63ba7bc3e7e50c0250bc11cba1ff7fd51b4b30) | `archive/v15-xhrm-atlas` | 2 |
| archive(v15-astra): preserve architecture, paper study, and analytical checks | [7dd50482](https://github.com/Shubhankankrishna/EncoderConceptDecoder/commit/7dd5048275ee1c5d64af8b39171dd0fa84422e0a) | `archive/v15-astra-design` | 7 |
| archive(v15-astra): preserve engineering blueprints and mathematical reviews | [d88b61f1](https://github.com/Shubhankankrishna/EncoderConceptDecoder/commit/d88b61f1c90f2d09f3a9b66180764036a7b0fe48) | `archive/v15-astra-blueprints` | 13 |
| archive(v15-astra-m2): import GRAM pilot, audits, and stopped-run evidence | [4500a3a3](https://github.com/Shubhankankrishna/EncoderConceptDecoder/commit/4500a3a356b677ff730ddc78f4db4089668bdff6) | `archive/v15-astra-m2` | 142 |
| archive(v15-astra-v3): import corrected raw-text and native-32 experiments | [97f7e90c](https://github.com/Shubhankankrishna/EncoderConceptDecoder/commit/97f7e90c9c4facd7ab2a88a8f5b14da9de7d3b9a) | `archive/v15-astra-v3` | 290 |
| archive(v15-astra-ncm-v4): import grounded planner and pretrained renderer pilot | [67b6ae98](https://github.com/Shubhankankrishna/EncoderConceptDecoder/commit/67b6ae98a2c8ef4dca268fae2577eedf69973efb) | `archive/v15-astra-ncm-v4` | 360 |
| archive(v15-astra-ncm-v4): record residual-consistent renderer continuation | [96cdcda7](https://github.com/Shubhankankrishna/EncoderConceptDecoder/commit/96cdcda796eda94fbd1ea7b8b9c397fd153db035) | `archive/v15-astra-renderer-followup` | 129 |

These imports add **943 research files**. The following organization commit adds the guides, notices, registry updates, preservation ledger and publication tests. The import index is extended consecutively; existing public history and archival tags are preserved.

## Local pre-commit validation

| Check | Result |
| --- | --- |
| Staged publication files | 5,980, including the machine-readable publication record |
| Python syntax | 1,201 files parsed |
| Experiment families / saved execution groups | 29 / 676 |
| Preserved source/document hashes | 1,794, including 360 added Astra source/document entries |
| Local artifact records | 1,534; about 34.7 GB decimal of payload identity metadata |
| Credential-pattern scan | 5,953 text/document files; zero findings |
| Publication/status regression tests | Three passed, including complete/stopped/slice statuses and actual plan seeds |
| New infrastructure whitespace check | Passed |
| Archival source whitespace | Retained; Markdown hard breaks and upstream whitespace are not rewritten |

The [machine-readable publication record](../experiments/publication_2026-10-20.json) contains commit identities, counts and limitations. Repository/catalog checks are rerun after the organization metadata is staged. A fresh clone of the final commit is checked before pushing; the final handoff reports that result and confirms the remote branch/tags. GitHub's repository workflow independently checks the pushed tree.
