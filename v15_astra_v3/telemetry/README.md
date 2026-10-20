# Astra observatory

A local telemetry engine and dashboard for the 24-hour v3 GRAM experiment.

**Live dashboard: http://127.0.0.1:8766/**

The service runs separately from training. It reads the existing run files, keeps
an incremental history in SQLite, and serves a responsive dashboard without a
frontend build or an internet connection. It imports no PyTorch, allocates no MPS
memory, and never opens model checkpoints for deserialization. It only stats
checkpoint files. It does not edit the trainer's pinned source or configuration.

## What you can track

- Training step, examples seen, loss, prior-only validation code NLL, learning
  rate, KL, sigma, gradient norm, throughput, and the fixed experiment deadline.
- Full-run and recent-range loss charts, with raw training readings, a rolling
  mean over seven **logged** readings, and saved validation points.
- Generated text, preceding context, reference text, per-example code NLL, and
  dominant-word repetition for every saved evaluation. Final test results appear
  after the experiment produces them; they are excluded from validation selection.
- Live host CPU, available RAM, swap, power source, disk space, and the latest
  MPS driver memory reading reported by the trainer. MPS uses unified memory;
  driver memory and host memory must not be added together.
- Checkpoint sizes and freshness; best-validation and resumable latest snapshots.
- Saved evaluation events and ingestion errors, plus active alerts for stale
  metrics, dead/recovering processes, resource pressure, repetitive output,
  near-zero KL, missing checkpoints, and non-finite loss.
- CSV export of the actual logged training rows.

**Pause updates pauses the dashboard's polling only.** It does not pause training
or the independent collector. A disconnected dashboard preserves its last readings,
shows a warning, and retries. The dashboard offers no training control endpoints.

Code likelihood is not text perplexity or accuracy. Repetition is a descriptive
diagnostic: an output is flagged when one word accounts for more than half of its
word tokens, ignoring punctuation and case. This is not a semantic quality score.
Near-zero KL does not establish that a model has failed, or that stochastic
reasoning is useful. Inspect these signals together.

## Run locally

From `/Users/shubhankankrishna/Desktop/EncoderConceptDecoder`:

```bash
env PYTHONDONTWRITEBYTECODE=1 v15_astra_m2/.venv/bin/python -m v15_astra_v3.telemetry.server --port 8766
```

The service is already running detached for the current experiment. Do not launch
a second copy on its active port. Its PID and URL are recorded in
`telemetry/runtime/service.json`; its log is `telemetry/runtime/service.log`.
Stopping this telemetry process has no effect on the training supervisor or trainer.

For another run, supply an existing `--run /absolute/run/folder` and a new
`--database /absolute/telemetry.sqlite3`. A database is bound to one run and cannot
silently combine different experiments. Database storage inside the training run
is rejected. `--port 0` chooses a free local port; the actual URL is printed and
recorded in `service.json` beside that database. `--interval` accepts 2–60 seconds.

Only localhost is exposed. No CDN, analytics, remote fonts, external server,
login, download of model weights, or new GPU job is required. Live system counters
use `psutil`, already available in the training virtual environment. If it is
unavailable, log-based telemetry remains available and host counters are marked
unavailable.

## Collection and persistence

The collector polls every five seconds at background CPU priority. The trainer
logs its first three steps and then every 50 steps, so a five-second UI refresh
does not imply a new loss measurement every five seconds. Validation is saved
after the first step and approximately every 30 minutes. Checkpoints normally
arrive every five minutes. Freshness labels reflect the actual file timestamps.

Metrics are consumed using a persisted inode and byte offset. Incomplete final
JSONL lines wait for completion, malformed complete lines produce an event, and
file rotation/truncation triggers a rescan. Exact records are deduplicated by
content hash. Cursor updates and inserted rows commit in one SQLite transaction.
Saved evaluation files are ingested only when their size/mtime changes, and their
own timestamps define the activity journal. Non-finite numbers serialize as null.

SQLite uses WAL mode. Host observations retain three days of history; the run's
logged metrics and evaluation records remain in the database. Chart responses are
bounded and uniformly sampled over the requested range, retaining the first and
latest rows. CSV export contains all ingested training rows. At this run's logging
cadence, database and collector memory are modest; the measured values are in
`runtime/overhead_test.json`.

The HTTP API is versioned through `schema_version: 1` in the snapshot:

| Route | Result |
|---|---|
| `/api/health` | Collector freshness and collection errors; HTTP 503 when unhealthy |
| `/api/snapshot` | Run, model, latest metrics, validation, alerts, resources, checkpoints, activity |
| `/api/history?limit=1200&after_step=-1` | Bounded chart history; limit 2–5,000 |
| `/api/evaluations` | Saved evaluation summaries, including final test when available |
| `/api/samples?evaluation=eval_00003468.json&offset=0&limit=1` | Saved text and repetition diagnostic; no inference |
| `/api/events?limit=80` | Persistent ingestion/evaluation events |
| `/api/resources?limit=180` | Recent sampled host counters |
| `/api/export.csv` | All logged training rows as a CSV download |

Process health verifies command identity as well as PID. Stale training rows after
three minutes are flagged, with an explicit note that a longer evaluation can
delay logging. Completion and other terminal states take precedence over stale
training rows. The browser safely renders sample text using `textContent`.
Only the four allowlisted web assets are served; run files, SQLite, checkpoints,
and arbitrary filesystem paths are not exposed. Mutation methods return HTTP 405.

## Test

These tests use temporary CPU-only fixtures and never acquire the training lock:

```bash
env PYTHONDONTWRITEBYTECODE=1 v15_astra_m2/.venv/bin/python -m unittest v15_astra_v3.telemetry.tests.test_engine -v
node --test v15_astra_v3/telemetry/tests/view.test.mjs
```

Backend coverage includes interruption/restart cursors, partial lines, malformed
and non-finite records, rotation/truncation, read-only inputs, stale/dead/terminal
states, validation selection, final test isolation, host warnings, storage
ownership, collection recovery, API exports, path traversal, mutation rejection,
and an independent-process check that importing telemetry does not import torch.
Frontend unit tests cover formatting, flat/missing chart data, one-point charts,
validation rendering, and explicit terminal states.

Browser checks cover the live page, pagination, selecting earlier evaluations,
chart ranges, refresh pause/resume, context disclosure, and compact layout without
horizontal overflow. Live service restart and resource measurements are recorded
under `runtime/`; validation evidence and screenshots are collected in
`tests/verification.json` and `tests/dashboard-*.png`.
