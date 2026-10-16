#!/bin/bash
# block until the v3 trainer prints N new heartbeat lines after the current one (read-only)
N=${1:-1}
F=/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/runs/v3_48h/stdout.log
c0=$(grep -c "step [0-9]* loss" $F)
while [ $(grep -c "step [0-9]* loss" $F) -lt $((c0 + N)) ]; do sleep 20; done
grep "step [0-9]* loss" $F | tail -$N | awk '{print $1,$2,$4,$(NF-1)" tok/s"}'
