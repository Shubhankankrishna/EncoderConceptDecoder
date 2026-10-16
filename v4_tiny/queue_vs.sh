#!/bin/bash
# v4-vs-LLM queue while v3 trains: two training processes at a time (owner-accepted v3 slowdown).
# phase 1: pinpad seed-0 pair at 6M (after the 3M budget-point pair ends); phase 2: text seed-0 pair at 5M.
cd /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny
R=runs
waitpair() { until [ -f $R/$1/final.json ] && [ -f $R/$2/final.json ]; do sleep 30; done; }
waitpair vs_pin_v4_s0 vs_pin_llm_s0
echo "$(date '+%F %T') queue: phase 1 (pinpad s0 6M)" >> $R/queue.log
QOS="" THREADS=2 bash launch_vs.sh vs_pin_v4_s0_6M --arch v4 --data pinpad --seed 0 --real_tokens 6e6 --eval_every_tokens 5e5 &
QOS="" THREADS=2 bash launch_vs.sh vs_pin_llm_s0_6M --arch llm --data pinpad --seed 0 --real_tokens 6e6 --eval_every_tokens 5e5 &
wait
echo "$(date '+%F %T') queue: phase 2 (text s0 5M)" >> $R/queue.log
QOS="" THREADS=2 bash launch_vs.sh vs_text_v4_s0 --arch v4 --data text --seed 0 --real_tokens 5e6 --eval_every_tokens 1e6 &
QOS="" THREADS=2 bash launch_vs.sh vs_text_llm_s0 --arch llm --data text --seed 0 --real_tokens 5e6 --eval_every_tokens 1e6 &
wait
echo "$(date '+%F %T') queue: phases 1-2 done" >> $R/queue.log
