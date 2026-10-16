#!/bin/bash
# v4-vs-LLM scheduler for the remaining jobs. Concurrency: at most 2 experiment processes while v3 runs (its chain log
# has no "v3 done" line), at most 4 afterwards. Counts every run_vs / analyze_vs process (including queue_vs.sh's).
cd /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny
R=runs; CHAIN=/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/runs/chain.log
PY=/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/.venv/bin/python
limit() { grep -q "v3 done" $CHAIN && echo 4 || echo 2; }
running() { ps -Ao command | grep -E "python (run_vs|analyze_vs)\.py" | grep -v -E "grep|time -l" | wc -l | tr -d ' '; }
slot() { until [ "$(running)" -lt "$(limit)" ]; do sleep 30; done; }
train() { slot; echo "$(date '+%F %T') sched: launch $1" >> $R/queue.log; (QOS="" THREADS=2 bash launch_vs.sh "$@" &); sleep 20; }
analyse() { until [ -f $R/$1/final.json ]; do sleep 30; done; [ -f $R/$1/analysis.json ] && return; slot
  echo "$(date '+%F %T') sched: analyse $1" >> $R/queue.log
  (cd code && OMP_NUM_THREADS=2 nice -n 19 $PY analyze_vs.py arm $1 > ../$R/$1/analysis.log 2>&1; echo "$(date '+%F %T') sched: analysed $1 exit $?" >> ../$R/queue.log) &
  sleep 20; }
# wait until the queue_vs.sh phases have started both of their pairs (so the jobs below never jump ahead of them)
until grep -q "queue: phase 2" $R/queue.log; do sleep 30; done
train vs_pin_v4_s1_6M --arch v4 --data pinpad --seed 1 --real_tokens 6e6 --eval_every_tokens 5e5
train vs_pin_llm_s1_6M --arch llm --data pinpad --seed 1 --real_tokens 6e6 --eval_every_tokens 5e5
train vs_text_v4_s1 --arch v4 --data text --seed 1 --real_tokens 5e6 --eval_every_tokens 1e6
train vs_text_llm_s1 --arch llm --data text --seed 1 --real_tokens 5e6 --eval_every_tokens 1e6
train vs_pin_v4_s2_6M --arch v4 --data pinpad --seed 2 --real_tokens 6e6 --eval_every_tokens 5e5
train vs_pin_llm_s2_6M --arch llm --data pinpad --seed 2 --real_tokens 6e6 --eval_every_tokens 5e5
for n in vs_pin_v4_s0_6M vs_pin_llm_s0_6M vs_text_v4_s0 vs_text_llm_s0 vs_pin_v4_s1_6M vs_pin_llm_s1_6M \
         vs_text_v4_s1 vs_text_llm_s1 vs_pin_v4_s2_6M vs_pin_llm_s2_6M; do analyse $n; done
wait
echo "$(date '+%F %T') sched: all jobs launched/analysed" >> $R/queue.log
