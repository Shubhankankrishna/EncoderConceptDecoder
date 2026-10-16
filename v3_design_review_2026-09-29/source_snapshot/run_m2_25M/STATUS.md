# STATUS (resume guide for a fresh session)

Updated 2026-09-27 22:20 IST (continuation armed, D-041).

## Done
- Exact tests T1–T9, D1–D5, P1–P4, MPS smoke (via `code/compat.py`, D-001), P5: all PASS (`runs/tests/`).
- Config check: 25.05M compute / 78.58M stored, reach 63/70 (`runs/tests/config_check.txt`).
- Data: `data/` (264.4M train tokens, MANIFEST.json, checks in `runs/data_checks.json`, samples `runs/data_samples.md`).
- Calibration: B = 2 × accum 16 (`runs/calibration.json`); real speed ≈ 2,000–2,170 tok/s after the non_blocking fix (D-024).

## Screens done (D-030..D-032)
S1 4.341 / S2 stack 4.352 / S3 R=1 4.336 (val nats, step 265). Screen gate PASS.

## Main run
Started 2026-09-27 07:49:33 via chain.sh (3,600 steps); extended to 4,050 steps at step 554 (D-033) and relaunched via
code/chain2.sh (same args, total_steps 4050, 24 h budget incl. elapsed; ends ~06:30–07:50 on 09-28). chain2.sh then runs
final_eval (last + best), P5 on trained weights, plots, report_tables.md. If the chain dies, restart the remaining lines of
code/chain3.sh by hand (run_job2.sh adds --resume itself).

## Running / next (original plan)
1. Screen S1 (`runs/screens/S1`, 280 steps). Launched 02:40 via `code/run_job.sh`.
2. After reviewing S1: `nohup caffeinate -dimsu code/chain.sh &` runs S2 (stack) → S3 (R = 1) → main (24 h, steps from
   `code/main_steps.py` on S1) → `final_eval.py` (last, best) → P5 on trained weights → plots.
3. Then write REPORT.md from `runs/**` (every number must come from a file).

## Continuation: +24 h of training after the main run (D-041) — ARMED
- Timeline (estimates): main STOP ~06:40 Mon 09-28 → chain3 final eval / P5 / plots → "chain done" ~08:00 → code/chain4.sh
  starts runs/cont → ends ~08:00–09:00 Tue 09-29 → its final eval (~1 h) → "cont done" in runs/chain.log.
- What it trains: from `runs/cont/init_from_main.pt` (hard link to main's final ckpt_last) with weights, AdamW state, data
  cursor and RNG carried; step/clock/best restart at 0; LR 6e-5 → 6e-4 over 300 steps (user's choice 07:50), cosine back to 6e-5 over 4,050
  steps; same mix, only unseen rows (≈ 0.47 → 0.94 epoch); micro_bs 2 × accum 16; 24 h budget; hourly evals. Logs carry
  `gstep` (= step + 4050) and `gtokens`.
- chain4 is waiting now (pid in runs/chain4.pid, holds no caffeinate). Alive check: `ps -p $(cat runs/chain4.pid)`.
- Cancel before it starts: `touch runs/NO_CONT` (from run_m2_25M; checked every minute).
- Re-arm after a reboot/logout (safe; refuses if one is already running; starts at once if "chain done" is already logged):
  `bash -c 'nohup bash code/chain4.sh > runs/chain4.stdout 2>&1 & disown'` — launch it via bash: zsh's `&` would run it,
  and tomorrow's trainer, at nice 5.
- pause.sh / resume.sh now pick the active run by themselves (main, or cont once runs/cont/init_from_main.pt exists and
  "cont done" is not logged). battery_guard2.sh watches runs/cont (started by chain4).
- DO NOT run pause.sh between main's STOP (~06:40) and "chain done" (~08:00): the final evaluation cannot pause.
  pause.sh would kill chain3, but its final_eval.py child keeps running. If it happened: wait until
  `pgrep -f final_eval.py` is empty, then `bash code/resume.sh` (it redoes main's final steps; chain4 then proceeds).
- Never edit chain3.sh, chain4.sh, run_job2.sh, battery_guard*.sh while they run (bash reads them by offset);
  pause.sh / resume.sh / train.py are safe to edit (train.py only with defaults unchanged, D-041).
- Dashboard: extended for the continuation and RESTARTED 22:3x at the user's request (pid 19607, nice 19; dashboard
  only). Live training + Overview show a two-stage plan (1 main run → final evaluation, 2 continuation armed/running/
  paused/evaluating/done, with its expected start). Once runs/cont/init_from_main.pt exists the charts append the
  continuation at global steps 4,050 → 8,100 (marked "continuation"), and status/ETA/heartbeat/checkpoints come from
  runs/cont. For the first ~600 continuation steps loss verdicts are capped at amber (LR re-warm bump is expected).
  Restart command for Terminal: `kill $(pgrep -f "python3 .*webapp/server.py"); nohup nice -n 19 /opt/anaconda3/bin/python3 webapp/server.py > webapp/server.log 2>&1 &`
- The continuation runs the current train.py in a fresh process, so the per-step heartbeat (D-037) goes live with it.

## Pausing (D-034)
- `bash code/pause.sh` → wait for "SAFE TO CLOSE LID". `bash code/resume.sh` to continue (starts code/chain3.sh).
- Battery guard (D-035): code/battery_guard.sh (main, 15%) / code/battery_guard2.sh (any run, 7% since 2026-09-28 10:00) auto-pause on battery; the continuation's guard2 is pid 4697.
- From the ~12:00 hourly save on 2026-09-27 the main run is driven by code/chain3.sh + run_job2.sh (swap_at_ckpt.sh).

## How to resume after a crash
- A training job resumes automatically inside `run_job.sh` (up to 6 attempts) from `<run>/ckpt_last.pt`.
- If the chain itself died: check `runs/chain.log`, then start the remaining steps by hand, e.g.
  `code/run_job.sh ../runs/main ...same args...` (it adds `--resume` when `ckpt_last.pt` exists).
- Never run two MPS jobs at once.

## Dashboard (D-036)
- `nohup nice -n 19 /opt/anaconda3/bin/python3 webapp/server.py > webapp/server.log 2>&1 &` → http://127.0.0.1:8765

## Heartbeat (D-037)
- train.py writes runs/main/live.json every step, but the trainer running since 12:47 loaded the old file: per-step data exists only after the next restart (pause.sh + resume.sh).

## Watch list
- 2026-09-27 13:50, step 1009: first non-chance probe — lr_recall1 intact 0.117 vs plan-swap 0.008 (chance ≈ 0.008);
  UNCONFIRMED (one hourly eval, subset). Check it holds (intact ≥ 0.08, swap ≈ chance) at the ~step-1200 eval.
- Speed ≈ 2,000–2,100 tok/s since step 650 → 4,050 steps ≈ 19–22 h elapsed; do not splice the schedule again (slack goes to
  final_eval, P5, optional S4).
- 2026-09-27 14:52, step 1217: lr_recall1 intact 0.336 vs plan-swap 0.008 → CONFIRMED (two consecutive evals, rising:
  0.008 @766, 0.117 @1009, 0.336 @1217; swap stays at chance). REPORT must track the intact-vs-swap gap per eval.
- 2026-09-27 22:07, step 2559: hop 0.109 and carry3 0.180 above chance for
  two evaluations running (hop 0.062 → 0.109, carry3 0.070 → 0.125 → 0.180); recall1 0.945 vs swap 0.016.
- Throughput dips (20:22 1,644; 21:32 1,435) coincided with heavy browser/3D testing on this fan-less MacBook Air
  (Chrome GPU helper 48% CPU, WindowServer 25–40%, kernel_task 34%); recovered to 1,852 at 22:03. No config change is
  safe for more speed (D-019: B = 4 swaps; B = 1 changes per-micro-batch codebook statistics).
