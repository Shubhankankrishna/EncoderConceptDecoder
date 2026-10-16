# EncoderConceptDecoder

Research into language models that encode observed tokens into shorter concept sequences, compute over those concepts, and decode tokens. The workspace follows the original component designs through V1–V14 experiments and the V15 proposal, including unsuccessful and interrupted work.

**Start with the [experiment chronology](EXPERIMENTS.md).** It links each stage's design, implementation, saved evidence, and interpretation limits. The [execution index](docs/RUNS.md) lists individual saved execution groups; the [JSON catalog](experiments/catalog.json) provides configurations, seeds where known, results, and artifact paths.

## Current research state

- **V14:** completed base training, diagnostics, and continual pipeline. Use its [generated tables](v14_tiny/REPORT_TABLES.md) and [final evaluation](v14_tiny/runs/eval/base/results.json). Its original preregistration report still contains a stale “not yet run” header.
- **V15:** a [proposed architecture](v15_Architecture.md), revised after V14 results. Analytical reviews exist; implementation and training have not been demonstrated.
- **Earlier versions:** preserve their own data assumptions, controls, budgets, failures, and reviews. Different versions are not directly comparable as a single leaderboard.

## Navigate the project

| Material | Guide |
| --- | --- |
| Chronological designs and experiments | [Experiment chronology](EXPERIMENTS.md) |
| Configurations, evaluations, engineering checks, and interrupted runs | [Execution index](docs/RUNS.md) |
| What to install and where to run each version | [Reproducibility guide](docs/REPRODUCIBILITY.md) |
| How to record the next experiment | [Working guide](docs/WORKFLOW.md) |
| Local checkpoints, datasets, papers, and large result payloads | [Artifact policy](artifacts/README.md) |
| Deep historical project review, dated 13 October | [Repository review](repo_review_2026-10-13/README.md) |
| Third-party source and paper provenance | [Third-party notices](THIRD_PARTY_NOTICES.md) |
| Clean-clone checks and import limitations | [Validation record](docs/VALIDATION.md) |

## Repository checks

From the project root, with Python 3.12 (standard library only):

```sh
python3 tools/catalog.py check
python3 tools/catalog.py scan
```

These validate repository organization, publication policy, historical source hashes, archival tags, and Python syntax. Use a full clone with tags; after a shallow checkout, run `git fetch --unshallow --tags` first. The checks do not run training or certify scientific results. The GitHub workflow performs the same lightweight checks.

The folders retain their original paths and source bytes. A fresh clone contains code and manageable evidence; checkpoints, bulk datasets, local databases, downloaded papers, and files over 10 MiB require separate local access. The artifact inventory records identity and location; it is not a remote backup.

## History and license

Git was introduced on 16 October 2026. The ordered `archive(...)` commits import the snapshots present on that date; they are not backdated development history. `archive/` tags identify import boundaries and are not runnable-release promises. Cross-version imports may only become usable once the full `main` tree is present.

Original project code is available under the [MIT license](LICENSE). Third-party source retains its own notices. Dataset and paper rights are separate; the project license does not grant rights to third-party datasets or publications.
