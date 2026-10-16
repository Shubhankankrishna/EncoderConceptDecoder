#!/bin/bash
# Start the v3 48 h run (first launch). Usage: bash code/start.sh [extra train.py args]
# Launches code/run_job.sh detached (via bash, nice 0; zsh's & would give nice 5), starts the 7 % battery guard,
# and the v3 dashboard instance (port 8766, reads runs/v3_48h). Refuses if a trainer already runs.
cd "$(dirname "$0")/.."
if pgrep -f "train.p[y]" >/dev/null; then echo "a trainer is already running; not starting"; pgrep -fl "train.p[y]"; exit 1; fi
if [ -f runs/v3_48h/ckpt_last.pt ]; then echo "runs/v3_48h already has a checkpoint: use code/resume.sh"; exit 1; fi
mkdir -p runs/v3_48h
echo "$(date '+%F %T') v3 start requested: $*" >> runs/chain.log
bash -c "nohup bash code/run_job.sh $* > runs/run_job.stdout 2>&1 & disown"
pgrep -f "battery_guard.s[h]" >/dev/null || bash -c 'nohup bash code/battery_guard.sh > runs/battery_guard.stdout 2>&1 & disown'
if ! lsof -i :8766 >/dev/null 2>&1; then                      # isolated v3 dashboard (its own webapp copy and heartbeat file)
  bash -c 'HRM_MAIN=v3_48h PORT=8766 nohup nice -n 19 /opt/anaconda3/bin/python3 webapp/server.py > runs/dashboard.log 2>&1 & disown'
fi
sleep 5
echo "started. Logs: runs/v3_48h/stdout.log · runs/chain.log · dashboard http://127.0.0.1:8766/#train (v1-style live page for v3)"
echo "v3 live telemetry file: run_m2_25M/webapp/data/v3_live.json"
