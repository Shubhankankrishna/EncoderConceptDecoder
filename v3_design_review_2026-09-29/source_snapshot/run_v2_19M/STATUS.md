# STATUS — HRM Thinker v2 (19M), 48 h run

## Layout
- `code/` — model (`model.py`, `concept_core_d6.py`, encoder/decoder from v1), `train.py`, `evaluate.py`, `final_eval.py`,
  `test_v2.py` (exact CPU tests), `profile_step.py` (MPS timing), scripts below. Data: `../run_m2_25M/data` (shared).
- `runs/v2_48h/` — the run: stdout.log, train_log.jsonl, log.jsonl (hourly evals), live.json, ckpt_last/best,
  eval_24h.json (milestone), eval_final.json / eval_best.json (48 h). `runs/chain.log` — launcher events.

## Commands (from run_v2_19M/)
- Start (first time): `bash code/start.sh` → launcher, 7 % battery guard, v2 dashboard on :8766.
- Pause: `bash code/pause.sh` (wait for "SAFE TO CLOSE LID"). Resume: `bash code/resume.sh`.
- Snapshot: `bash code/v2_check.sh`. Monitor policy: `code/V2_MONITOR_POLICY.md`.
- Dashboards: http://127.0.0.1:8766/#train (v1-style live page for v2), http://127.0.0.1:8765/#v2train (v2 telemetry:
  per-pass NCP, NextLat, consistency, MoE load/drops/routing divergence).

## Timeline
- 48 h of TRAINING time (hourly evals included). At 24 h: exit 5 → final_eval (~1–1.5 h, not counted) → resume.
  At 48 h: final eval of last + best (~2–3 h).
- The battery guard pauses at ≤ 7 % on battery and never resumes by itself (resume.sh after plugging in).
