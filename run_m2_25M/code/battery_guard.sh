#!/bin/bash
# Battery guard (D-035): if on battery and charge <= THRESH% while the main trainer runs, do a safe pause (code/pause.sh).
# Keeps watching afterwards (protects later resumes too). Never resumes by itself.
# Env for testing: THRESH (default 15), BATT_CMD (default "pmset -g batt"), DRY=1 (log instead of pausing), EVERY (s, 60).
cd "$(dirname "$0")/.."
THRESH=${THRESH:-15}; EVERY=${EVERY:-60}; BATT_CMD=${BATT_CMD:-"pmset -g batt"}
LOG=runs/chain.log
echo "$(date '+%F %T') battery guard started (pause at <= ${THRESH}% on battery)" >> $LOG
while true; do
  b=$($BATT_CMD)
  pct=$(echo "$b" | grep -Eo '[0-9]+%' | head -1 | tr -d '%')
  if echo "$b" | grep -q "discharging" && [ -n "$pct" ] && [ "$pct" -le "$THRESH" ] \
     && pgrep -f "run_dir ../runs/main" >/dev/null && [ ! -f runs/main/PAUSE ]; then
    echo "$(date '+%F %T') battery guard: ${pct}% on battery -> safe pause" >> $LOG
    if [ "$DRY" == "1" ]; then echo "DRY: would run pause.sh"; exit 0; fi
    bash code/pause.sh >> runs/battery_guard_pause.log 2>&1
    osascript -e "display notification \"Training paused safely at ${pct}% battery. Plug in, then run code/resume.sh\" with title \"HRM training\"" 2>/dev/null
  fi
  sleep $EVERY
done
