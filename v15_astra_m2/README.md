# V15 Astra — 10M GRAM first / frozen NCM / M2 Air experiment

Read `ARCHITECTURE_REVIEW.md` for the implementation verdict and explicit changes
to the reviewed blueprint. The first arm trains **10,000,333 GRAM parameters** using
the official frozen **LM1B NCM tokenizer** (approximately 111.4M additional frozen
parameters). This includes the 237,505 training-only posterior parameters. The
separate deployment checkpoint physically excludes the posterior and target encoder.
The second arm is a **10,001,292-parameter deterministic HRM** comparison. Both use
exact modular-planning and grid-maze environments. GRAM uses the short-unroll,
full-trajectory ELBO adaptation; MCTS, strategy modes and continual learning are
not implemented. GRAM runs first as explicitly requested by the user.

All commands below run from the repository root.

```bash
v15_astra_m2/.venv/bin/python -m v15_astra_m2.status
```

The active run directory contains `plan.json`, `status.json`, `gram.log`,
`hrm.log`, each arm's `metrics.jsonl`, `status.json`, `latest.pt`,
`best_nll.pt`, `best_policy.pt`, and eventually `FINAL_REPORT.md`.
Checkpoints are written atomically approximately every ten minutes and at normal
shutdown. Validation runs every thirty minutes; final ID/OOD tests run after both
arms. The approximately parameter-matched HRM control receives the same allocated
wall time. One seed is a feasibility pilot, not a definitive research result.

## Stop and resume

Create `STOP` in the active **run directory** to stop the entire supervisor after
the current bounded operation. Use `STOP` inside one arm directory only if you
intend to end that arm and allow the control arm to continue.

```bash
touch v15_astra_m2/runs/20261018_48h/STOP
```

To resume after removing that stop file, invoke the same supervisor with the same
run directory. The stored deadline is retained; it does not grant another 48 hours.
Check for an existing supervisor first; a file lock rejects concurrent launches.

```bash
v15_astra_m2/.venv/bin/python -u -m v15_astra_m2.supervise \
  --run v15_astra_m2/runs/20261018_48h --hours 48
```

Keep the Mac plugged in and the lid open. `caffeinate` prevents idle sleep while
the supervisor is alive; it cannot promise continued execution after lid closure,
shutdown or a reboot. Low battery, low available memory and low disk space stop
work safely. Restart attempts are bounded; errors are retained in logs.

## Validation artifacts

* `reports/codec_audit.json`: 2,000 development-block fidelity audit for both
  official tokenizers, action round trips, source parity, frozen state, partial
  reconstructions and MPS differences.
* `reports/unit_tests.log`: causal-mask, gradient, probability, checkpoint,
  parameter-count, pooling, parser and environment checks.
* `reports/blueprint_math_recheck.json`: nine original mathematical checks.
* `reports/mps_benchmark.json`: measured batch throughput and device allocation.
* `reports/preflight.json`: real-code CPU/MPS gradient parity and small-set overfit.
* `reports/gram_unit_tests.log`: Gaussian KL, objective units, full/truncated
  gradients, posterior removal, prior independence, parameter accounting.
* `reports/gram_preflight.json`: real-tokenizer GRAM prior-only overfit, CPU/MPS
  parity including Gaussian gradients, and three-seed sampling checks.
* `reports/data_audit.json`: independent verification of every transition, label,
  current-state input, dataset index, and split group.
* `reports/gram_mps_verification.json`: MPS stochastic optimizer/RNG save-resume
  check and production-shape memory/throughput measurement.
* `artifacts/data/manifest.json`: immutable code and source hashes, group splits,
  exact codec version and device, plus all counts.

## Reproduction

The isolated venv inherits the existing Anaconda Python 3.12 / PyTorch 2.10 runtime.
Only OmegaConf 2.3.0 and Hydra 1.3.2 were added to the venv. Full environment versions
are recorded in `reports/environment.txt`. Source and asset revisions are pinned
in the architecture review and codec manifests. The upstream repository and its
MIT license are retained in `references/ncm`.

```bash
v15_astra_m2/.venv/bin/python -m v15_astra_m2.tests
v15_astra_m2/.venv/bin/python -m v15_astra_m2.tests_gram
v15_astra_m2/.venv/bin/python -m v15_astra_m2.audit_codec
v15_astra_m2/.venv/bin/python -m v15_astra_m2.data --generate-only
v15_astra_m2/.venv/bin/python -m v15_astra_m2.preflight
v15_astra_m2/.venv/bin/python -m v15_astra_m2.preflight_gram
v15_astra_m2/.venv/bin/python -m v15_astra_m2.data --device cpu --batch 32
```

Do not rerun preparation while a training job is active. A completed dataset is
immutable; a changed codec or task generator requires a new cache directory/version.
Large weights, caches, venv and run checkpoints are excluded from Git.
