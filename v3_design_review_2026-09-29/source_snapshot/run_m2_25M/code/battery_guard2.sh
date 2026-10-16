#!/bin/bash
# Battery guard for any run (D-041; same logic as battery_guard.sh / D-035, which only watches runs/main).
# Usage: bash code/battery_guard2.sh RUN   (RUN = runs/<name>, ../runs/<name> or <name>)
# If on battery and charge <= THRESH% while that run's trainer is running, do a safe pause (code/pause.sh). Never resumes.
# Env for testing: THRESH (default 15), BATT_CMD (default "pmset -g batt"), DRY=1 (log instead of pausing), EVERY (s, 60).
RUN=$(basename "${1:?usage: battery_guard2.sh RUN}")
cd "$(dirname "$0")/.."
THRESH=${THRESH:-7}; EVERY=${EVERY:-60}; BATT_CMD=${BATT_CMD:-"pmset -g batt"}   # 7 % since 2026-09-28 (D-035 addendum)
LOG=runs/chain.log
echo "$(date '+%F %T') battery guard 2 started for runs/$RUN (pause at <= ${THRESH}% on battery)" >> $LOG
while true; do
  b=$($BATT_CMD)
  pct=$(echo "$b" | grep -Eo '[0-9]+%' | head -1 | tr -d '%')
  if echo "$b" | grep -q "discharging" && [ -n "$pct" ] && [ "$pct" -le "$THRESH" ] \
     && pgrep -f "run_dir ../runs/$RUN" >/dev/null && [ ! -f runs/$RUN/PAUSE ]; then
    echo "$(date '+%F %T') battery guard 2: ${pct}% on battery -> safe pause of runs/$RUN" >> $LOG
    if [ "$DRY" == "1" ]; then echo "DRY: would run pause.sh"; exit 0; fi
    bash code/pause.sh >> runs/battery_guard_pause.log 2>&1
    osascript -e "display notification \"Training paused safely at ${pct}% battery. Plug in, then run code/resume.sh\" with title \"HRM training\"" 2>/dev/null
  fi
  sleep $EVERY
done
