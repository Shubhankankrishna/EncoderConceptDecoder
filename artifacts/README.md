# Local artifact inventory

[Project home](../README.md) · [Chronology](../EXPERIMENTS.md) · [Artifact records](manifest.jsonl) · [Large-result digests](large_results.json)

The Git repository contains code, configurations, tokenizers, reports, plots, and manageable experiment evidence. These files remain **local-only**: checkpoints/tensors, bulk datasets, databases, binary caches, original ZIP bundles, downloaded PDFs, temporary upstream material, and any file larger than 10 MiB. Meaningful text logs are retained when they fit the policy.

`manifest.jsonl` records a relative path, experiment family, byte count, exclusion reason, modification time, and SHA-256 checksum for each inventoried excluded file. Stable hashes are reused on refresh only when size and modification time agree. A file that changes during hashing is marked `changing`, with a null checksum. Known active telemetry/runtime output is marked `live` and has no checksum because a consistent snapshot is not taken while its service runs. Symlink hashes describe link text; regular-file hashes describe file bytes. Disposable environment and compiler-cache directories are intentionally skipped.

These paths and hashes establish file identity. **They are not a remote backup, public artifact download, or guarantee of reproducibility.** Keep the original workspace or a separately managed backup. This setup does not upload large files through Git LFS or create a storage bill.

`large_results.json` contains bounded structural summaries/scalar readings for oversized result evidence. Array lengths and first/last log records describe structure; they are not statistical summaries or substitutes for complete results. Raw dataset/example text is excluded from those summaries.

Refresh with `python3 tools/catalog.py refresh` from the project root. Validate restored heavy artifacts with `python3 tools/catalog.py verify-local-hashes`. The manifest does not load pickled models or execute artifact contents.
