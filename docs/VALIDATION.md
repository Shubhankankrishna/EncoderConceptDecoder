# Local import validation — 16 October 2026

[Project home](../README.md) · [Chronology](../EXPERIMENTS.md) · [Machine-readable validation](../experiments/validation.json)

The organized repository was checked in a separate fresh clone of organization commit `4e9405ddb6c52956f6160e1066bed6075ccba33b`. Tests altered only disposable clone fixtures; no experiment or training process was started.

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

GitHub authentication and publication are deferred at the user's request. No remote repository or remote backup has been created. The repository currently uses Git's detected local author identity. Before the first public push, associate it with the account's verified no-reply identity; if updating the unpublished import metadata, update archival tags and the import index together and rerun checks.

Pattern scanning and syntax checks are bounded validation. They do not certify arbitrary private text, scientific correctness, runtime behavior, or reproducibility without the separately stored artifacts. No training was rerun.
