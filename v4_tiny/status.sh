#!/bin/bash
# one-shot status of the v4_tiny arms and of the v3 trainer (read-only)
cd /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny
date '+%T'; sysctl -n vm.swapusage
for d in runs/*/; do a=$(basename $d); [ -f $d/log.jsonl ] || continue; [ -f $d/final.json ] && { echo "$a FINAL done"; continue; }
  grep '"train"' $d/log.jsonl | tail -1 | python3 -c "import json,sys
t=sys.stdin.read().strip()
if t: r=json.loads(t); print('$a', r['step'], round(r['real_tokens']/1e6,2),'M', round(r['tok_s']), 'tok/s ntp', round(r['ntp'],3))"; done
grep "step [0-9]* loss" /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/runs/v3_48h/stdout.log | tail -3 | awk '{print "v3",$2,$4,$(NF-1)" tok/s"}'
echo "v3 GUARDS lines: $(wc -l < /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/runs/v3_48h/GUARDS.jsonl)"
ps -Ao rss,command | grep "run_v4tiny" | grep -v "grep\|time -l" | awk '{s+=$1; n++} END {printf "arms %d, RSS %.2f GB\n", n, s/1e6}'
