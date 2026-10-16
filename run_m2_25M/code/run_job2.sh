#!/bin/bash
# Resilient launcher: runs train.py; if the process crashes (non-zero exit other than a clean stop), resumes from
# ckpt_last.pt, up to 5 times. Usage: run_job.sh RUN_DIR [train.py args...]   (wrap in caffeinate -dimsu)
RUN_DIR="$1"; shift
cd "$(dirname "$0")"
source ../.venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$RUN_DIR"
for attempt in 1 2 3 4 5 6; do
  if [ -f "$RUN_DIR/ckpt_last.pt" ]; then RES="--resume $RUN_DIR/ckpt_last.pt"; else RES=""; fi
  echo "=== $(date '+%F %T') launch attempt $attempt $RES" >> "$RUN_DIR/launcher.log"
  python train.py --run_dir "$RUN_DIR" "$@" $RES >> "$RUN_DIR/stdout.log" 2>&1
  code=$?
  echo "=== $(date '+%F %T') exit code $code" >> "$RUN_DIR/launcher.log"
  if [ $code -eq 0 ]; then exit 0; fi
  if [ $code -eq 4 ]; then exit 4; fi              # paused by pause.sh (D-034): do not auto-resume
  sleep 20
done
exit 1
