# Running and observing the paired experiment

Dashboard: <http://127.0.0.1:8766/?arm=30m#overview>. The model selector switches
between the 30M and 10M arms. Metrics and histories are kept in separate SQLite
databases under `telemetry/runtime/native32`; reading the UI never runs a model.

The run directory is `runs/20261019_native32_10m_30m`. Its `plan.json` records
the absolute deadline. `status.json` describes the active preparation/training
stage. `training_plan.json`, created after preflight, fixes both configurations,
source hashes, data identity, common learning-rate schedule and arm-time budgets.
The deadline never extends when a process resumes.

Preparation proceeds through codec acceptance, batch encoding and production-
shape MPS preflight. Training cannot bypass a failed acceptance or integrity
check. Encoding checkpoints record a completed prefix plus its digest; resuming
verifies that prefix before appending.

The 30M model starts first. The supervisor then gives the model with less
accumulated arm time the next 30-minute slot. One heavy-job lock covers codec,
training and evaluation; models never train simultaneously. Each training arm
saves optimizer and RNG state in `gram/latest.pt`, with inference-only exports
beside it. `equal_step_*_inference.pt` files support matched-data comparisons.

Stop the complete new experiment by creating its `STOP` file. The supervisor
requests a graceful child stop and saves the current training checkpoint or
encoding prefix. Do not send an uncatchable kill signal to a model writer unless
the graceful stop has demonstrably failed.

If a restart is needed after investigating a failure, first verify that the old
supervisor and child are gone. Run the same `v15_astra_v3.shortblock.run` module
using the pinned `v15_astra_m2/.venv/bin/python`, from the workspace root. A
remaining `STOP` file deliberately prevents continuation. Never edit registered
training sources, configuration or cache files to force a resume; make a new
versioned experiment if its scientific contract changes.

Training stops cleanly below 1.5 GiB available RAM, below 3 GiB disk headroom, or
below 20% battery when unplugged. Keep the laptop connected to power. A resource
stop is reported by the supervisor; the current implementation does not silently
restart a resource-stopped arm.

Validation chooses checkpoints. Final test scoring occurs after training and
does not select a checkpoint. The ten-minute final reserve is a bounded attempt,
not a promise that evaluation will fit under every thermal or system condition.
If the hard deadline interrupts evaluation, the supervisor records incomplete
evaluation and leaves all successfully written checkpoints and reports intact.

The previous `runs/20261018_24h_raw_gram` experiment remains stopped at 17,085.
Its STOP files prevent accidental continuation; all checkpoints are preserved.

Testing from the workspace root:

```sh
PYTHONDONTWRITEBYTECODE=1 v15_astra_m2/.venv/bin/python -m v15_astra_v3.shortblock.test_shortblock
PYTHONDONTWRITEBYTECODE=1 v15_astra_m2/.venv/bin/python -m unittest discover -s v15_astra_v3/tests -t .
PYTHONDONTWRITEBYTECODE=1 v15_astra_m2/.venv/bin/python -m unittest discover -s v15_astra_v3/telemetry/tests -t .
node --test v15_astra_v3/telemetry/tests/view.test.mjs
```

Run the full legacy integration suite only while the heavy-job lock is idle:
some tests intentionally exercise guarded CPU trainer entry points. Pure small
CPU unit tests and read-only telemetry do not launch competing MPS work.
