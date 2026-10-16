#!/bin/bash
# Resume the v3 run after pause.sh (or after a reboot).
cd "$(dirname "$0")/.."
RUN=runs/v3_48h
if pgrep -f "train.p[y]" >/dev/null; then echo "a trainer is already running; not starting another"; exit 1; fi
rm -f $RUN/PAUSE $RUN/ckpt.tmp
echo "$(date '+%F %T') v3 resume requested" >> runs/chain.log
bash -c 'nohup bash code/run_job.sh > runs/run_job.stdout 2>&1 & disown'
pgrep -f "battery_guard.s[h]" >/dev/null || bash -c 'nohup bash code/battery_guard.sh > runs/battery_guard.stdout 2>&1 & disown'
echo "launched (training resumes after any pending 24 h evaluation). Log: $RUN/stdout.log"
