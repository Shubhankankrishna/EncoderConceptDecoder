# GitHub setup

[Project home](../README.md) · [Experiment chronology](../EXPERIMENTS.md) · [Working guide](WORKFLOW.md)

Repository: [Shubhankankrishna/EncoderConceptDecoder](https://github.com/Shubhankankrishna/EncoderConceptDecoder).

- Public repository, MIT license for original code, default branch `main`.
- `origin` uses HTTPS. GitHub CLI manages authentication; no access tokens are stored in project files.
- Repository-local author: Shubhankan Krishna, `76508110+Shubhankankrishna@users.noreply.github.com`.
- The initial 25 local commits and 22 annotated archival tags were linked to this identity before their first publication. Only author/committer/tagger metadata changed; original file trees and dates were preserved. The import index was updated to the corresponding public commit IDs.
- The local pre-publication identity backup is outside tracked files and is not pushed.

The no-reply format follows [GitHub's email reference](https://docs.github.com/en/account-and-profile/reference/email-addresses-reference). The account was created after July 2017, so the ID-based address applies.

## Daily use

Clone the whole repository, including history and tags:

```sh
git clone https://github.com/Shubhankankrishna/EncoderConceptDecoder.git
cd EncoderConceptDecoder
python3 tools/catalog.py check
```

For a new experiment, follow the working guide: branch, register the protocol and configuration, run in a unique output directory, record results and limitations, refresh the catalog, validate, and merge. Use issues with the Experiment template to track scientific questions. Pull requests use the repository's review template.

The [Repository checks workflow](../.github/workflows/repository-checks.yml) runs on pushes and pull requests. It validates catalogs, relative links, preserved source hashes, import/tag consistency, Python syntax, and credential patterns. It uses read-only repository permissions and starts no training.

## Local artifacts and live services

Checkpoints, bulk datasets, downloaded papers, databases, and oversized result payloads stay local and appear in the artifact inventory. GitHub contains their identities and manageable evidence, not a complete artifact backup.

V9 watchdog runtime outputs, V13 live telemetry output, and V14 observatory runtime files remain local. Their generators and frozen historical experiment evidence are retained. Updating a live service should not create recurring Git changes.

The earlier local validation record remains an audit of the tested scientific file trees. It is not a claim that the repository was already public during the local setup phase.
