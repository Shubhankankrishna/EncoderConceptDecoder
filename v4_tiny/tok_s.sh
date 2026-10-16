#!/bin/bash
# mean trainer tok/s over the last N heartbeat lines of the v3 stdout log (read-only)
N=${1:-10}
grep -o "step [0-9]* loss.*tok/s" /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_v3_19M/runs/v3_48h/stdout.log | tail -$N | awk '{s+=$(NF-1); n++; last=$2} END {printf "trainer tok/s mean of last %d = %.0f (last step %s)\n", n, s/n, last}'
