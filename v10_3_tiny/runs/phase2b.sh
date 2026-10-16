#!/bin/sh
# Phase 2 (relaunched 16:45): each arm's evaluation starts as soon as its final.pt exists.
cd /Users/shubhankankrishna/Desktop/EncoderConceptDecoder
P=/opt/anaconda3/bin/python; R=v10_3_tiny/runs
wait_for() { while [ ! -f "$1" ]; do sleep 15; done; }
run_eval() { wait_for $R/$1_s1/final.pt; sleep 5; date > $R/phase2_$1_eval_started; $P -B -m v10_3_tiny.evaluate --run-dir $R/$1_s1 --device mps --threads 1 > $R/$1_s1_eval.log 2>&1; }
( run_eval X4 ) &
( run_eval A ) &
( wait_for $R/X8_s1/final.pt; sleep 5; date > $R/phase2_sleep_started
  $P -B -m v10_3_tiny.sleep --run-dir $R/X8_s1 --out-dir $R/sleep_X8_s1 --device mps --threads 1 --consolidate > $R/sleep_X8_s1.log 2>&1 ) &
( run_eval X8 ) &
wait_for $R/X8_s1/final.pt; wait_for $R/A_s1/final.pt; wait_for $R/X4_s1/final.pt; sleep 5
# Predeclared: second-seed X = lower text NLL at step 6000 in quick_eval.json.
BEST=$($P -c "
import json
q={a:[e for e in json.load(open('$R/%s_s1/quick_eval.json'%a)) if e['step']==6000][0]['nll']['text'] for a in ('X8','X4')}
print(min(q,key=q.get))")
echo "$BEST" > $R/seed2_choice.txt
{ date; vm_stat | head -4; sysctl vm.swapusage; } > $R/launch_s2.env
$P -B -m v10_3_tiny.train --arm C0 --run-dir $R/C0_s2 --max-steps 6000 --device cpu --threads 3 --seed 2 > $R/C0_s2.log 2>&1 &
$P -B -m v10_3_tiny.train --arm $BEST --run-dir $R/${BEST}_s2 --max-steps 6000 --device cpu --threads 3 --seed 2 > $R/${BEST}_s2.log 2>&1 &
date > $R/phase2_launched
wait
date > $R/phase2_done
