# Local import validation — 16 October 2026

[Project home](../README.md) · [Chronology](../EXPERIMENTS.md) · [Machine-readable validation](../experiments/validation.json)

The organized repository was checked in a separate fresh clone of organization commit `506ff8ae4cbf4b5dc9f2931a7e5f6445a10ad8c7`. Tests altered only disposable clone fixtures; no experiment or training process was started.

| Check | Result |
| --- | --- |
| Experiment inventory | 23 families; 647 saved execution groups |
| Tracked files in the tested organization snapshot | 5,031 |
| Python syntax | 885 tracked Python files parsed |
| Historical source/document hashes | 1,434 match the preserved source ledger |
| Import history | 23 import/setup commits indexed; 22 annotated archival tags resolve to their recorded commits |
| New guide and catalog links | Passed |
| Credential-pattern scan, including Word document text | 5,008 text/document files; no findings |
| Local artifact checksums | 1,308 stable files verified; no mismatches |
| Live runtime files | 12 explicitly marked live, without a consistency/hash claim |
| Local-only payloads present in the fresh clone | Zero |
| Git object integrity | `git fsck --full` passed |
| Altered historical source fixture | Rejected |
| Oversized tracked file fixture | Rejected |
| Outside-project directory shortcut fixture | Rejected |
| Synthetic GitHub credential fixture | Detected |

The validation record and refreshed inventories are added after the tested organization snapshot. Repository checks are rerun for these metadata additions. The import boundaries are present-day archival snapshots, not reconstructed development dates or runnable releases.

The running V13 telemetry generator rewrote its generated dashboard HTML during setup. That output stays local, its exception is documented, and the generator source is preserved. A separate comparison against the initial full source/document baseline found no other changes. The nested V10-10M ignore file was deliberately adjusted to retain small run evidence; no model/trainer source was edited.

The initial local phase deferred GitHub publication. GitHub CLI is now authenticated, and all unpublished author, committer, and tagger identities use the account's GitHub no-reply address. File trees and original commit dates were preserved; archival tags and the import index were remapped together. Current publishing configuration is documented in the [GitHub guide](GITHUB.md). Heavy artifacts remain local; this repository is not their remote backup.

Pattern scanning and syntax checks are bounded validation. They do not certify arbitrary private text, scientific correctness, runtime behavior, or reproducibility without the separately stored artifacts. No training was rerun.

During GitHub setup, the V9 watchdog runtime outputs were also moved outside current Git tracking. Their earlier imported snapshots remain in the archival history; the running service and its files were not modified.

## Subsequent Astra publication

The [20 October publication record](PUBLICATION_2026-10-20.md) extends this original import audit with the Astra designs, implementations and outcomes. The historical counts above remain the counts of the 16 October snapshot.
