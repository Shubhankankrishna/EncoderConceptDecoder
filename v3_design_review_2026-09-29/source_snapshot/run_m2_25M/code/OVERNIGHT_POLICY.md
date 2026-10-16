# Overnight monitor policy (D-043) — read this before every check

Role: OBSERVER. The user is asleep and said: do not stop the training, no radical changes, act only in an emergency when
no other option is left. Every check: run `bash code/opus_check.sh`, append ONE line to `runs/opus_monitor.log`
(time · state · step · tok/s · last eval · battery · verdict), reply briefly. Push a notification only for: a pause,
a failure, a stuck run, the continuation starting, or an action you took.

## Never
- Never kill, signal or restart the trainer (python … train.py), chain3, chain4, run_job2, or the battery guards.
- Never edit chain3.sh, chain4.sh, run_job2.sh, battery_guard*.sh (bash reads them while they run). Never run pause.sh.
- Never `kill $(pgrep …)` / `kill $(ps … | grep …)` from a command whose own text contains the pattern (it can match the
  tool's shell wrapper, or worse): print the pid in one command, check it with `ps -p <pid> -o command`, kill it in the next.
- Never start any GPU job (no final_eval, no tests, no benchmarks). Never write under runs/main or runs/cont.
- Never put the literal strings `train.py`, `chain4.sh`, `run_job2.sh`, `final_eval.py` or `run_dir ../runs/` in a
  long-running or backgrounded command line: chain4's start gate is `pgrep -f "train.py"` and pause.sh / guards /
  server pgrep for the others. Use `bash code/opus_check.sh` / `bash code/opus_watch.sh`; if grepping inline write `train.p[y]`.

## Normal things that look alarming
- ~05:50–07:10 Mon: NO trainer process. Main hit step 4,050 (STOP (max_steps)); chain3 runs final_eval.py (~1 h),
  P5, plots. final_eval runs on the GPU. Then "chain done" → within 60 s chain4 logs "cont start".
- First continuation loss line ≈ 17 min after "cont start"; first continuation eval ≈ 1 h after. `init_from_main.pt`
  appears at the start (the dashboard switches to runs/cont then).
- First ~1,600 continuation steps (until ≈ 17:00 Mon): loss / validation are EXPECTED to rise, peak ≈ +0.10–0.15
  (val ≈ 2.85–2.90 around 10:00–11:00), and only drop back below 2.754 after ≈ 1,450–1,560 steps (schedule-aware fit,
  D-041 addendum 3). recall1 / carry3 may dip at high LR and come back. Not a fault; the dashboard keeps it amber.
- Throughput dips while the user used the browser; ~1,700–1,950 tok/s is normal.

## Allowed actions (only on the exact trigger)
a. Last STOP line of the active run is `STOP (pause)` AND `pmset -g batt` shows AC Power → `bash code/resume.sh`
   (picks main or cont itself). Then re-add the keep-awake for the running chain (see e).
b. Paused on BATTERY (discharging) → `kill` the caffeinate that has `-w <chain4 pid>` (only that one; find it with
   `pmset -g assertions`) so the Mac can sleep instead of draining to 0. Re-add later per (e).
c. "chain done" logged, chain4 pid dead, runs/cont/init_from_main.pt absent, no NO_CONT → re-arm:
   `bash -c 'nohup bash code/chain4.sh > runs/chain4.stdout 2>&1 & disown'`, then (e).
d. chain.log says "cont not started: main final evaluation missing" → `mkdir -p runs/cont && ln runs/main/ckpt_last.pt
   runs/cont/init_from_main.pt`, then `RESUME=1 bash -c 'nohup bash code/chain4.sh > runs/chain4.stdout 2>&1 & disown'`;
   log that main's final evaluation must be redone by hand later. (Other "not started" reasons: do NOT force; notify.)
e. Keep-awake: after any (a)/(c)/(d) → `bash -c 'nohup caffeinate -ims -w $(cat runs/chain4.pid) >/dev/null 2>&1 & disown'`
   if chain4 is alive and no caffeinate asserts on its behalf.
f. "cont failed" in chain.log → read `runs/cont/stdout.log` + `launcher.log` tails; if it looks like a transient crash,
   ONE `bash code/resume.sh`. Never twice. Otherwise notify and stop acting.
g. Dashboard not 200 → restart only the server: find its pid with `ps -Ao pid,command | grep "webapp/server.p[y]"`,
   `kill <pid>`, then `bash -c 'nohup nice -n 19 /opt/anaconda3/bin/python3 <abs>/webapp/server.py > <abs>/webapp/server.log 2>&1 & disown'`.

## Flag (notify, do not act)
- No new loss line for ≥ 60 min while a trainer runs and no eval/final_eval in progress (possible hang) — do NOT kill.
- tok/s < 1,200 for two checks; gn > 1; any GUARD / NONFINITE / Traceback; swap > 5 GB; disk < 5 GB; battery ≤ 25 %
  on battery; "main failed" / "cont failed" / "not started".
