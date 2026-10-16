#!/bin/bash
# One-shot (D-034): at the next hourly save, restart the main run on the pause-aware train.py (costs ~1 min).
cd "$(dirname "$0")/.."
M=runs/main
i0=$(stat -f %i $M/ckpt_last.pt)
while [ "$(stat -f %i $M/ckpt_last.pt)" == "$i0" ] || [ -e $M/ckpt.tmp ]; do sleep 5; done
sleep 15                                          # let hard-linking and pruning finish
pkill -f chain2.sh; pkill -f run_job.sh; sleep 1; pkill -f "run_dir ../runs/main"
while pgrep -f "run_dir ../runs/main" >/dev/null; do sleep 2; done
if [ -f $M/PAUSE ]; then echo "$(date '+%F %T') swap: stopped after hourly save, PAUSE present, not relaunching" >> runs/chain.log; exit 0; fi
nohup caffeinate -dimsu code/chain3.sh > runs/chain3.stdout 2>&1 &
echo "$(date '+%F %T') swap: pause-aware trainer launched (pause.sh now live)" >> runs/chain.log
