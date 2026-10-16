#!/bin/bash
# Read-only snapshot of the v3 run for monitoring (never writes under runs/v3_48h, never signals anything).
cd "$(dirname "$0")/.."
R=runs/v3_48h
echo "time $(date '+%F %T')"
echo "power $(pmset -g batt | grep -Eo '[0-9]+%;[^;]*' | head -1)"
echo "trainer $(pgrep -f 'run_dir ../runs/v3_48h' | tr '\n' ' ') launcher $(pgrep -f 'code/run_job.s[h]' | tr '\n' ' ') guard $(pgrep -f 'battery_guard.s[h]' | tr '\n' ' ') eval $(pgrep -f 'final_eval.p[y]' | tr '\n' ' ')"
TP=$(pgrep -f 'run_dir ../runs/v3_48h' | head -1); [ -n "$TP" ] && echo "trainer memory: $(top -l 1 -pid $TP -stats mem,cmprs 2>/dev/null | tail -1) (footprint, compressed)"
echo "flags: PAUSE=$([ -f $R/PAUSE ] && echo yes || echo no) MILESTONE_EVAL_PENDING=$([ -f $R/MILESTONE_EVAL_PENDING ] && echo yes || echo no)"
[ -f $R/live.json ] && echo "live $(cat $R/live.json | cut -c1-300)"
grep -E " step [0-9]+ loss" $R/stdout.log 2>/dev/null | tail -1 | cut -c1-330
grep -E "EVAL\[" $R/stdout.log 2>/dev/null | tail -1 | cut -c1-330
grep -E "STOP|GUARD|NONFINITE|Traceback|Error" $R/stdout.log 2>/dev/null | tail -3
echo "chain.log:"; tail -4 runs/chain.log 2>/dev/null | sed 's/^/  /'
echo "evals: 24h=$([ -f $R/eval_24h.json ] && echo yes || echo no) final=$([ -f $R/eval_final.json ] && echo yes || echo no)"
echo "disk $(df -h . | tail -1 | awk '{print $4}') free  swap $(sysctl -n vm.swapusage | grep -Eo 'used = [0-9.]+M')  dashboards 8765=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8765/) 8766=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8766/)"
