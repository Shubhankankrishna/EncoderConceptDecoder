# Working on the next experiment

[Project home](../README.md) · [Chronology](../EXPERIMENTS.md) · [Setup and reproducibility](REPRODUCIBILITY.md)

## Existing history

The initial commit sequence is an ordered import of the existing workspace. Dates are import dates. A version's sources, reviews, controls, dashboards, and manageable evidence are grouped together. Original source snapshots retain their bytes and checksums. The later organization commit adds navigation and inventories.

Use `main` for integrated work. Create future branches as `experiment/vNN/short-description`, `analysis/vNN/short-description`, or `maintenance/short-description`. Keep one scientific question per experiment branch. Use normal fast-forward pushes; preserve published history.

## Record an experiment in order

1. Start from current `main`; create the experiment branch.
2. Before training, commit the hypothesis, baseline, dataset/tokenizer identity, seed, budget, effective configuration, evaluation protocol, success criteria, and intended output directory. Keep version-specific protocols in the existing version folder.
3. Run with a unique output directory. Do not overwrite earlier checkpoints, change frozen code during a run, or rebuild data used by another run.
4. Commit manageable evidence: effective configuration, source/data identities, summaries, evaluation results, relevant logs, and deviations. Record failed, stopped, and partial outcomes explicitly. Heavy outputs stay local and enter the artifact inventory.
5. Explain what the result supports, its limits, and the next decision. Keep post-hoc changes distinct from preregistered choices.
6. Refresh the catalog, stage the intended files, and run the checks below. Review and merge the branch into `main`; push when GitHub is configured.

Suggested commit subjects: `experiment(v16): register codec comparison`, `results(v16): record seed 1601 evaluation`, and `docs(v16): interpret comparison and limitations`.

## Update the inventories

Run from the project root after the output files have settled:

```sh
python3 tools/catalog.py refresh
git add .
python3 tools/catalog.py check-local
python3 tools/catalog.py scan
```

`refresh` updates the chronology, execution index, file coverage, compact large-result digests, exact-path ignores, and local artifact manifest. It hashes new or changed excluded files without loading model tensors. It never starts a trainer. Files changing during hashing are marked `changing` with no checksum. Environment folders and disposable caches are outside the research inventory.

The registry is `experiments/registry.json`. Add a new family there before a new version, with its actual paths, reading guide, evidence-based status, and limitations. For ongoing runs within an existing family, refresh discovers execution evidence automatically. Null seeds mean unknown; seeds inferred from directory names remain labeled.

`check` works in a fresh clone. `check-local` also verifies local coverage and artifact size/mtime consistency. `verify-local-hashes` reads all stable artifact payloads and verifies checksums; use it when restoring or transferring local artifacts.

## Keep evidence and infrastructure separate

Repository checks use only Python's standard library. Version-specific model contracts, dataset builders, and evaluations require their recorded environment and local artifacts. Run those deliberately for a scientific change; the repository workflow does not launch them automatically.

Preserved historical source files are listed in `experiments/source_hashes.jsonl`. If a later change deliberately revises one, document why and remove only its old preservation entry in that same reviewed commit. Keep frozen audit/source snapshots immutable. Organization tooling must not silently rewrite the preservation ledger.
