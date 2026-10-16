#!/bin/bash
# Read-only status snapshot for the overnight monitor (D-043). Never writes under runs/ except nothing; never signals any process.
cd "$(dirname "$0")/.."
A=main; [ -e runs/cont/init_from_main.pt ] && A=cont
now=$(date +%s)
echo "time $(date '+%F %T')  active_run=$A"
echo "power $(pmset -g batt | tail -1 | sed 's/.*)\t//')"
echo "trainer $(pgrep -f 'train.py --run_dir ../runs/' | tr '\n' ' ')  $(pgrep -fl 'train.py --run_dir' | grep -o 'run_dir [^ ]*' | tr '\n' ' ')"
echo "chain3 $(pgrep -f 'code/chain3.sh' | wc -l | tr -d ' ')  chain4 $( [ -f runs/chain4.pid ] && kill -0 $(cat runs/chain4.pid) 2>/dev/null && echo alive:$(cat runs/chain4.pid) || echo none)  run_job2 $(pgrep -f run_job2.sh | wc -l | tr -d ' ')  final_eval $(pgrep -f final_eval.py | wc -l | tr -d ' ')  guard $(pgrep -f 'battery_guard2?.sh' | wc -l | tr -d ' ')"
echo "keepawake $(pmset -g assertions | grep -c 'caffeinate asserting')  pause_flags: main=$([ -f runs/main/PAUSE ] && echo YES || echo no) cont=$([ -f runs/cont/PAUSE ] && echo YES || echo no)  NO_CONT=$([ -f runs/NO_CONT ] && echo YES || echo no)"
for R in main cont; do
  f=runs/$R/stdout.log; [ -f $f ] || continue
  last=$(grep -E " step [0-9]+ loss " $f | tail -1)
  t=$(echo "$last" | cut -c1-19); age=""
  [ -n "$t" ] && age=$(( (now - $(date -j -f '%Y-%m-%d %H:%M:%S' "$t" +%s)) / 60 ))
  echo "[$R] last loss line (${age} min ago): $(echo "$last" | cut -c21-140)"
  echo "[$R] last eval: $(grep 'EVAL\[' $f | tail -1 | cut -c1-230)"
  s=$(grep -E 'STOP \(|GUARD|NONFINITE|Traceback|Error|resumed from|continuation from' $f | tail -2 | cut -c1-160); [ -n "$s" ] && echo "[$R] events: $s"
  [ -s runs/$R/GUARDS.jsonl ] && echo "[$R] GUARDS: $(wc -l < runs/$R/GUARDS.jsonl) lines, last: $(tail -1 runs/$R/GUARDS.jsonl | cut -c1-150)"
  echo "[$R] launcher: $(tail -1 runs/$R/launcher.log 2>/dev/null)"
done
echo "chain.log tail:"; tail -4 runs/chain.log | sed 's/^/  /'
echo "eval_final: main=$([ -f runs/main/eval_final.json ] && echo yes || echo no) cont=$([ -f runs/cont/eval_final.json ] && echo yes || echo no)"
echo "disk $(df -h . | tail -1 | awk '{print $4}') free  swap $(sysctl -n vm.swapusage | grep -o 'used = [0-9.]*M')  dashboard $(curl -s -o /dev/null -m 5 -w '%{http_code}' http://127.0.0.1:8765/)"
