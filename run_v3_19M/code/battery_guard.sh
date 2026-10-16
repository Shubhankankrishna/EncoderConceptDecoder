#!/bin/bash
# v3 battery brake: on battery at <= THRESH % (default 7) while the v3 trainer or its evaluation runs -> code/pause.sh.
# Never resumes by itself. Env for testing: THRESH, BATT_CMD, DRY=1, EVERY (s).
cd "$(dirname "$0")/.."
THRESH=${THRESH:-7}; EVERY=${EVERY:-60}; BATT_CMD=${BATT_CMD:-"pmset -g batt"}
LOG=runs/chain.log
echo "$(date '+%F %T') v3 battery guard started (pause at <= ${THRESH}% on battery)" >> $LOG
while true; do
  b=$($BATT_CMD)
  pct=$(echo "$b" | grep -Eo '[0-9]+%' | head -1 | tr -d '%')
  if echo "$b" | grep -q "discharging" && [ -n "$pct" ] && [ "$pct" -le "$THRESH" ] && [ ! -f runs/v3_48h/PAUSE ] \
     && { pgrep -f "run_dir ../runs/v3_48h" >/dev/null || pgrep -f "final_eval.p[y].*v3_48h" >/dev/null; }; then
    echo "$(date '+%F %T') v3 battery guard: ${pct}% on battery -> safe pause" >> $LOG
    if [ "$DRY" == "1" ]; then echo "DRY: would pause at ${pct}%"; exit 0; fi
    bash code/pause.sh >> runs/battery_guard_pause.log 2>&1
    osascript -e "display notification \"v3 training paused at ${pct}% battery. Plug in, then run code/resume.sh\" with title \"HRM v3\"" 2>/dev/null
  fi
  sleep $EVERY
done
