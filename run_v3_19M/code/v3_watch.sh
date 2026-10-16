#!/bin/bash
# Exits (printing the line) on the first notable v3 event after it starts: STOP, GUARD, NONFINITE, Traceback, launcher
# events (paused / failed / done / milestone / evaluation / crash), or a new hourly evaluation. Read-only.
cd "$(dirname "$0")/.."
R=runs/v3_48h
n1=$(wc -l < $R/stdout.log 2>/dev/null || echo 0); n2=$(wc -l < runs/chain.log 2>/dev/null || echo 0)
while true; do
  sleep 30
  a=$(tail -n +$((n1 + 1)) $R/stdout.log 2>/dev/null | grep -E "STOP|GUARD|NONFINITE|Traceback|Error|EVAL\[" | head -1)
  b=$(tail -n +$((n2 + 1)) runs/chain.log 2>/dev/null | grep -E "v3 (paused|failed|done|milestone|trainer exit|24 h evaluation|battery)" | head -1)
  if [ -n "$a$b" ]; then echo "EVENT: $a $b"; exit 0; fi
done
