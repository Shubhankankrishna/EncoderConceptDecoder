# V10 telemetry contract — schema 1

The independent collector is running at **http://127.0.0.1:8766/api/v1**. It reads the active `pretrain_24h` run every five seconds and takes external host samples every 30 seconds. It neither imports PyTorch nor reads checkpoint tensors. The trainer and its frozen source fingerprint are unchanged.

## Storage and launch

Run these commands from `/Users/shubhankankrishna/Desktop/EncoderConceptDecoder` using `/opt/anaconda3/bin/python`:

```sh
/opt/anaconda3/bin/python -m v10_10M.telemetry.launch
/opt/anaconda3/bin/python -m v10_10M.telemetry.tests
/opt/anaconda3/bin/python -m v10_10M.telemetry.export
```

The launcher detaches the collector and verifies readiness. A filesystem lock prevents a second collector for the same data directory. A duplicate launch fails without interrupting the existing process. `--port`, `--run-dir`, and `--data-dir` are configurable. The service persists after this chat finishes, but does not automatically restart after a Mac reboot. Its local API stays available after training completes.

- `data/telemetry.sqlite3`: durable SQLite/WAL event and numeric metric store; use the API or an SQLite backup to copy a live database, not the `.sqlite3` file alone.
- `data/server.json`: collector PID, URL, start time and run path.
- `data/collector.log`: startup and collector/API errors.
- `data/overview.json`, `data/metrics_catalog.json`: atomically refreshed snapshots.
- `data/export/`: offline `events.jsonl`, `metrics.jsonl`, overview, catalog and an export watermark. Re-run export to refresh it. Exported events and metrics share one consistent transaction cutoff.

No logs are uploaded. HTTP is bound to loopback. GET endpoints are read-only. CORS permits local development origins such as `http://localhost:3000`. The read-only dashboard is served at **http://127.0.0.1:8766/#live**. It has no training-control endpoint. Only its five allowlisted HTML/CSS/JavaScript files are served; the database and Python sources are not web assets.

## API for dashboards and analysis

Data responses are JSON; the `/download` route also serves CSV attachments. Successful data responses carry `schema_version: 1`. The default run is `pretrain_24h`; pass `run_id` explicitly in a multi-run client.

| GET endpoint | Contents |
|---|---|
| `/health` | Collector PID, health, last successful poll and last error |
| `/runs` | Run manifests, source/data fingerprints, model configuration, parameter counts, status and process health |
| `/overview?run_id=pretrain_24h` | Current status, latest trainer evaluation, operational/research/data alerts, checkpoint metadata and measurement limitations |
| `/catalog?run_id=pretrain_24h` | Every numeric metric name, sample count and step coverage |
| `/series?run_id=pretrain_24h&name=train.loss.token&after_id=0&limit=1000` | Numeric time series: `points[{event_id,step,wall_time,value}]` |
| `/events?run_id=pretrain_24h&kind=training&after_id=0&limit=1000` | Full unmodified parsed training records with provenance |
| `/evaluations?run_id=pretrain_24h` | Full trainer evaluation records, including generated responses and rejection reasons |
| `/events?kind=probe` | Offline checkpoint gradients by objective/component/layer, ablations and paired math examples |
| `/events?kind=probe_evaluation` | Separate balanced development evaluation with per-module counts and responses |
| `/events?kind=audit` | Machine-readable audit findings |
| `/events?kind=host` | External process/system resource samples |
| `/artifacts` | Trainer evaluation files and checkpoint sizes/timestamps; no checkpoint download |
| `/issues` | Malformed or truncated source warnings |
| `/download?format=snapshot&run_id=pretrain_24h` | Full JSON event history through a consistent database watermark; overview/catalog reflect export completion |
| `/download?format=csv&name=train.loss.token&run_id=pretrain_24h` | Complete raw metric series as a CSV attachment; no render downsampling |
| `/schema` | Measurement semantics, available/unavailable metrics and routes |

`events`, `evaluations` and `series` return `next_after_id` and `has_more`. Paginate with **event ID**, never step: resumed training can repeat steps, and an older evaluation can be ingested after newer training. Limits must be 1–5000; cursors must be nonnegative. Unknown runs return 404, invalid queries 400. An unknown metric returns an empty series. Preserve the cursor even when no points arrive. Retain training ingestion order and break lines at repeated/rolled-back steps. Backfilled evaluation files may be plotted in checkpoint order. Retain event IDs rather than discarding duplicate steps.

Minimal frontend request:

```js
const base = 'http://127.0.0.1:8766/api/v1';
const overview = await fetch(`${base}/overview?run_id=pretrain_24h`).then(r => r.json());
const losses = await fetch(`${base}/series?name=train.loss.token&after_id=0&limit=1000`).then(r => r.json());
// Save losses.next_after_id; fetch again after five seconds.
```

## What each measurement means

- `train.loss.{total,token,concept,explore,vq,selector}`: interval means logged every 20 updates (and step 1). These average normalized batch losses, not a corpus-wide token-weighted loss. Exploration/selector are zero on updates that do not execute their auxiliary objective. The configured total is `token + 0.1*(concept + explore) + 0.01*vq + 0.05*selector`.
- `train.gradient.dense_preclip_norm`: actual dense L2 gradient norm before clipping, **only the last update in each logging interval**. It excludes the sparse Engram table. Historical per-step/per-layer norms cannot be reconstructed.
- `derived.dense_clip_scale` and `derived.engram_learning_rate`: inferred from documented clip=1 and Engram LR=5×dense LR. They are explicitly separate from measured metrics.
- `train.tokens_per_second`: input-token throughput over the logging window. Evaluation/checkpoint time can affect subsequent windows. `train.supervised_tokens.*` shows actual per-family supervision, distinct from the configured 65/20/15 percent **batch** mixture.
- `train.memory.*`: logged GiB (not decimal GB). RSS, MPS allocations and Metal driver counters overlap under unified memory; never sum them.
- `host.*`: external CPU, process memory, system memory/swap, disk and collector overhead. CPU percent can exceed 100% because it uses one logical CPU as 100%. The first CPU percentage is omitted because no sampling interval exists yet. These are not GPU utilization or temperature readings.
- `eval.*`: trainer development results. Full examples remain in event payloads. The existing fixed math generation slice covers only one module. Display this sampling limitation and the audit alerts.
- `probe.*`: newly measured offline checkpoint diagnostics on CPU, tagged with that checkpoint's step. Gradients are separated into token-only, concept-only and total-objective probes. They are **not** samples of the live optimizer's gradients.
- `probe_eval.*`: balanced offline development generation. Never join these into the trainer's evaluation curve without a visible series distinction.
- Accuracy exists only when `n > 0`. Always display `correct/n`, alongside the Wilson 95% interval if useful. A zero-trial evaluation is missing accuracy, not 0% accuracy. Intervals do not remove sampling or dataset biases.
- Code counts, entropy and effective code count describe observed selector usage. Small concept-ablation changes are heuristic research flags, not statistical significance tests.
- Trainer timestamps are reconstructed from run start plus logged elapsed time. Evaluation timestamps are completion file mtimes. Probe times are audit artifact mtimes; host times use the collector clock. All are Unix seconds UTC, each event has `time_quality`.

Unavailable live data: Engram gradient norms, per-layer gradients, gradient histograms, reliable GPU utilization/temperature. The RBF bank is intentionally empty during pretraining. An explicit checkpoint probe supplies some gradient diagnostics but does not fill historical gaps.

Complete JSONL records are transactionally ingested with source identity and byte offsets. Incomplete tails are retried; malformed complete records are quarantined as issues. Duplicate polls/restarts are idempotent. Observed truncation/rotation preserves old history. An overwrite entirely between polls that preserves inode, first line and file size cannot be reliably detected: the trainer must retain its append-only/atomic-write contract.

## Verification

Six telemetry contract tests cover partial writes, restart idempotency, duplicate training steps, truncation/rotation, malformed/nonfinite records, corrected evaluation files, zero-trial handling, audit metric separation, consistent exports, process health, API pagination/errors and local CORS. Live source record counts and API values were cross-checked after collector restart. Model source fingerprint remains the active run's original hash.

## Live dashboard

The UI follows the preferred `run_m2_25M/webapp/#v3train` design: native typography, light/dark panels, run hero, machine strip, six sparklines, sticky section chips, health cards and two-column chart cards. See [web/DESIGN.md](web/DESIGN.md) for the source review and UI test record. No remote fonts, chart libraries, build process, or model imports are required.

- **Live training:** six raw losses, EMA overlays, step/hour axes, recorded-history windows, raw-value inspection with pointer/keyboard, point comparison, code usage, concept ablations, gradient/throughput/LR and memory/exposure curves. Charts render only when visible and new data or controls require it.
- **Evaluation:** routine versus balanced audit sources, checkpoint selection, strict correct/count and 95% intervals, per-module balanced scores, response search/filtering and pagination. Routine history always shows the full recorded run.
- **Diagnostics:** audit findings, component/layer gradients by objective and family, a searchable metric catalog, CSV export, run configuration and complete JSON snapshot export. Checkpoint probes are explicitly labeled as offline measurements.

Refresh polls every five seconds while visible. Pause stops dashboard polling only; the collector and training continue. Connection failures preserve the last successful snapshot with its fetch time. Stale collector state, empty runs, zero-trial accuracy and unavailable measurements have explicit states.

Frontend data-rule tests:

```sh
/opt/anaconda3/bin/node --test v10_10M/telemetry/web/core.test.mjs
```

The optional isolated UI fixture runs with `python -m v10_10M.telemetry.web.qa_server` on port 8792. `data/qa_scenario.txt` accepts `zero`, `empty`, `offline` or `stale`; this switch exists only on the test server. It never controls or modifies a real run. The fixture server is stopped after QA.
