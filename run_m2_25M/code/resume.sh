#!/bin/bash
# Resume after pause.sh (D-034). Usage: bash code/resume.sh
# Resumes the continuation (runs/cont, D-041) once it has a checkpoint and is not done; otherwise the main run.
cd "$(dirname "$0")/.."
M=runs/main
if [ -e runs/cont/init_from_main.pt ] && ! grep -q "cont done" runs/chain.log; then M=runs/cont; fi   # cont started (even before its 1st ckpt)
if pgrep -f "train.py" >/dev/null; then echo "a trainer is already running (one MPS job at a time); not starting another"; pgrep -fl train.py; exit 1; fi
if [ -e $M/ckpt.tmp ]; then echo "removing stale $M/ckpt.tmp"; rm -f $M/ckpt.tmp; fi
rm -f $M/PAUSE                                   # must go first, or the trainer pauses again at its first step
cnt() { local n=$(grep -c "resumed from\|continuation from" $M/stdout.log 2>/dev/null); echo ${n:-0}; }
n0=$(cnt)
echo "$(date '+%F %T') resume requested ($M)" >> runs/chain.log
if [ "$M" == "runs/cont" ]; then
  RESUME=1 nohup bash code/chain4.sh > runs/chain4.stdout 2>&1 &              # starts battery_guard2 for runs/cont itself
else
  nohup caffeinate -dimsu code/chain3.sh > runs/chain3.stdout 2>&1 &
  pgrep -f battery_guard.sh >/dev/null || nohup code/battery_guard.sh > runs/battery_guard.stdout 2>&1 &   # D-035
fi
echo "launched; waiting for the resume line (~1 min)..."
for i in $(seq 1 60); do
  if [ $(cnt) -gt $n0 ]; then sleep 2; tail -2 $M/stdout.log; echo; echo "RESUMED ($M). Keep the lid open and on power."; exit 0; fi
  sleep 5
done
echo "no resume line after 5 min; check $M/stdout.log and $M/launcher.log"; tail -5 $M/stdout.log
