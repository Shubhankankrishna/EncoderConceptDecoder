# v2 monitor policy (read before every check)

Role: OBSERVER of the v2 48 h run (`runs/v2_48h`). Every check: `bash code/v2_check.sh`, append ONE line to
`runs/monitor.log`, reply briefly.

## Never
- Never kill/signal the trainer, `run_job.sh`, `final_eval.py` or `battery_guard.sh`; never edit `run_job.sh` or
  `battery_guard.sh` while they run (bash reads them lazily); never start another GPU job; never write under runs/v2_48h.
- Never put the literal strings `train.py`, `final_eval.py`, `run_dir ../runs/` in a backgrounded command (pgrep patterns).

## Normal things that look alarming
- Trainer absent for ~1–1.5 h around 24 h of training time: exit 5 (milestone) → `final_eval.py` writes eval_24h.json →
  training resumes by itself. `MILESTONE_EVAL_PENDING=yes` during that time is normal.
- First ~300 steps: warm-up; loss falls fast. `drop` (MoE dropped assignments) can start near 0.10 and should fall.
- `nl_h` (NextLat SmoothL1) may rise late in training (paper §E); not a fault.

## Allowed actions (only on the exact trigger)
a. Last STOP is `STOP (pause)` AND power is AC → `bash code/resume.sh`.
b. `v2 trainer exit N (crash k/6)` lines are handled by the launcher; only if `v2 failed` is logged: read the stdout tail,
   and if it looks transient run ONE `bash code/resume.sh`. Never twice; otherwise notify.
c. Dashboard not 200 → restart only that server. Both instances are `webapp/server.p[y]`: find the pids with
   `pgrep -f "webapp/server.p[y]"` and tell them apart with `lsof -p <pid> | grep cwd` (run_m2_25M = :8765, run_v2_19M = :8766).
   8765: `cd ../run_m2_25M && nohup nice -n 19 /opt/anaconda3/bin/python3 webapp/server.py > webapp/server.log 2>&1 &` (via bash -c).
   8766 (the v2 live page): `cd run_v2_19M && HRM_MAIN=v2_48h PORT=8766 nohup nice -n 19 /opt/anaconda3/bin/python3 webapp/server.py > runs/dashboard.log 2>&1 &` (via bash -c).

## Flag (notify, do not act)
No new loss line ≥ 45 min while the trainer runs and no eval runs; tok/s < 1,000 twice; gn > 1; GUARD / NONFINITE /
Traceback; swap > 5 GB; disk < 5 GB; battery ≤ 20 % on battery; `drop` > 0.05 after step 300; "v2 failed".
