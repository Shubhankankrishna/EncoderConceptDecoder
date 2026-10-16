#!/bin/bash
# Event watcher for the overnight monitor (D-043). Read-only. Exits (printing the event) on the first new line in
# runs/chain.log, a new STOP / GUARD / NONFINITE / Traceback line in either run's stdout, a launcher exit code, or the
# appearance of the continuation's stdout. Started as `bash code/opus_watch.sh` (keeps sensitive strings out of cmdlines).
cd "$(dirname "$0")/.."
sig() { for f in runs/chain.log runs/main/stdout.log runs/cont/stdout.log runs/main/launcher.log runs/cont/launcher.log; do
  [ -f $f ] && echo "$f $(grep -cE 'STOP \(|GUARD|NONFINITE|Traceback|exit code' $f) $(wc -l < $f)"; done; }
c0=$(wc -l < runs/chain.log); s0=$(sig | grep -v chain.log | awk '{print $1, $2}'); cs0=$([ -f runs/cont/stdout.log ] && echo 1 || echo 0)
while true; do
  sleep 30
  c=$(wc -l < runs/chain.log)
  if [ "$c" -gt "$c0" ]; then echo "EVENT chain.log:"; tail -n $((c - c0)) runs/chain.log; exit 0; fi
  s=$(sig | grep -v chain.log | awk '{print $1, $2}')
  if [ "$s" != "$s0" ]; then echo "EVENT log signal:"; for f in runs/main/stdout.log runs/cont/stdout.log runs/main/launcher.log runs/cont/launcher.log; do
      [ -f $f ] && grep -E 'STOP \(|GUARD|NONFINITE|Traceback|exit code' $f | tail -1 | sed "s#^#$f: #"; done; exit 0; fi
  if [ "$cs0" = 0 ] && [ -f runs/cont/stdout.log ]; then echo "EVENT continuation stdout appeared:"; head -3 runs/cont/stdout.log; exit 0; fi
done
