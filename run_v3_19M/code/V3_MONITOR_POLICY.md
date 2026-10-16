# v3 monitor policy (read before every check)

Role: OBSERVER of the v3 48 h run (`runs/v3_48h`). Every check: `bash code/v3_check.sh`, append ONE line to
`runs/monitor.log` (time · state · step · tok/s · last eval · recall1 · geometry · battery · verdict), reply briefly.

## Never
- Never kill/signal the trainer, `run_job.sh`, `final_eval.py` or `battery_guard.sh`; never edit `run_job.sh` or
  `battery_guard.sh` while they run (bash reads them lazily); never start another GPU job; never write under runs/v3_48h.
- Never put the literal strings `train.py`, `final_eval.py`, `run_dir ../runs/` in a backgrounded command (pgrep patterns).
- No gate authorises a stop. Only the owner does (draft_v3.md r2 §8.4).

## Normal things that look alarming
- Trainer absent for ~2 h around 24 h of training time: exit 5 (milestone) → `final_eval.py --split val` (10 interventions,
  M sweep, probes, health, audit, generation) writes eval_24h.json → training resumes by itself (≈ 2 min of torch.compile
  warm-up before the first step). `MILESTONE_EVAL_PENDING=yes` during that time is normal. After 48 h the final evaluation
  (test + OOD, then the best checkpoint) takes ≈ 3–4 h; `v3 done` ends the run (≈ 54 h after launch).
- First ~300 steps: warm-up; loss falls fast. Language improves long before recall (v1: recall appeared at ≈ 1,009–1,217).
- `scons` near 0 and `ang23` small early: diagnostics only (geometry is neither necessary nor sufficient; draft §4.7).
- Step-triggered evals at v1's eval steps (185, 265, 365, 554, …) on top of hourly evals: planned.
- Memory restarts (V-011 guard, carried): `GUARD {'kind': 'memory_restart', ...}` → `RE-EXEC after checkpoint` →
  `resumed from ...` within ~3 min is planned maintenance (torch.compile re-pays ≈ 2 min of warm-up on every start). Flag only if
  restarts come more often than every ~3 h, or no loss line follows within 6 min.

## Allowed actions (only on the exact trigger)
a. Last STOP is `STOP (pause)` AND power is AC → `bash code/resume.sh`.
b. `v3 trainer exit N (crash k/6)` lines are handled by the launcher; only if `v3 failed` is logged: read the stdout tail,
   and if it looks transient run ONE `bash code/resume.sh`. Never twice; otherwise notify.
c. Dashboard not 200 → restart only that server (8765: `cd ../run_m2_25M && nohup nice -n 19 /opt/anaconda3/bin/python3
   webapp/server.py > webapp/server.log 2>&1 &` via bash -c).

## Flag (notify, do not act)
No new loss line ≥ 45 min while the trainer runs and no eval runs; tok/s < 1,200 twice (draft floor); gn > 1 after step
300; GUARD (other than memory_restart) / NONFINITE / Traceback; swap > launch baseline + 2 GB (the run inherits ≈ 6 GB of swap from v2's
evaluation and a B4 benchmark; baseline recorded in DECISIONS V3-006); disk < 5 GB; battery ≤ 20 % on battery;
"v3 failed". Draft §8.4 notify items (continue the run): recall1 < 0.25 at step 1,792 (E1 warn; < 0.05 = E1 fail; v1
had 0.820); val > v1 + 0.20 in either language at two consecutive gate steps (v1: 3.227 at 1,217, 3.032 at 1,792,
2.913 at 2,376); any zero-init piece still exactly 0 in the `OPENED` line at step 300.
