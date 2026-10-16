# V14 Observatory

A read-only, mobile-friendly dashboard for this v14 experiment. No external
packages, CDNs, internet connection, model loading or GPU work are required.

## Open

- On the Mac: http://localhost:8714
- On the same Wi-Fi: use the address shown under **System & network**.

The server listens on `0.0.0.0:8714`. It serves only its UI and explicitly allowed
v14 logs/results; there is no training-control or arbitrary-file endpoint. Anyone
able to reach this port on the local network can read these experiment outputs.
No router forwarding or public hosting is configured.

## Service controls

From this directory:

```sh
python3 service.py start
python3 service.py status
python3 service.py stop
```

Start detaches the server from the terminal. It stays running when the terminal
or chat closes. Run start again after a Mac restart; it does not install a login
item. Keep the Mac awake for training and mobile access. If access from a phone
fails, confirm the current Wi-Fi address, the Mac firewall's Python allowance,
and that the network permits device-to-device traffic (guest networks may not).

For another port or loopback-only operation, stop the service first and run
`python3 server.py --host 127.0.0.1 --port 8714` in a terminal.

## What the readings mean

- **Overview**: current stage, logged progress, training-only finish estimate,
  actual stage statuses and operational alerts. The 24-hour budget is measured
  from `config.json`'s current base-run start, interpreted in Asia/Kolkata.
- **Training**: per-family losses, throughput, optimizer readings, persisted
  family token counters and checkpoint milestones. Counter saves can lag logs.
- **Concept health**: four horizons, head/copy accuracies, code diversity,
  regularizers and target freeze schedule. Live head/copy denominators differ;
  these are not V1 diagnostic comparisons. Freeze status is schedule-derived.
- **Evaluation**: V1–V5 are read from the diagnostic output, not re-inferred.
  V3 has per-family decisions only. V6 requires matched-time comparison, so
  the dashboard never invents its pass/fail. V7 shows saved continual metrics.
  Partial artifacts remain explicitly labeled. Failed stages remain failures
  even if the pipeline subsequently writes `ALL_DONE`.
- **System & network**: power, memory pressure, GPU, swap, disk and confirmed
  v14 processes. GPU/memory counters are host-wide; process CPU/RSS exclude
  workers. Host history begins when this monitor starts.
- **Logs & evidence**: bounded log tails and allowlisted artifact downloads.

UI and collector refresh every 10 seconds; OS/process sampling every 30 seconds.
Training emits metrics on its own cadence (usually 50 steps). Polling does not
make a new training observation. A disconnected or failed collector retains the
last good snapshot and labels it stale. The monitor never modifies experiment
files or restarts training. Its own SQLite history is retained for seven days
under `runtime/`; up to the latest 720 host samples appear in the UI.

The JSONL reader only consumes complete records, follows append updates, handles
truncation/replacement, and removes superseded steps after checkpoint rollback.
JSON artifacts retain the last complete version during in-place writes. The UI
samples long curves; the CSV exports all retained training observations.

## Files and checks

`server.py` is the engine and HTTP service; `static/` contains the UI;
`service.py` manages just this server; `runtime/server.log` records errors.

```sh
python3 -m unittest -v test_server.py
node --check static/app.js
```

Tests use temporary fixtures and never touch the live experiment.
