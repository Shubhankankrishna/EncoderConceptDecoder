#!/bin/bash
# Run one v4-vs-LLM arm on CPU at lowest priority, only on AC power (waits otherwise). Never touches v3 or the GPU.
# usage: [THREADS=2] [QOS=""] bash launch_vs.sh <name> --arch v4|llm --data text|pinpad [runner args...]
NAME=$1; shift
cd /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code
mkdir -p ../runs/$NAME
until pmset -g batt | grep -q "AC Power"; do echo "$(date '+%F %T') $NAME waiting for AC power" >> ../runs/queue.log; sleep 120; done
export OMP_NUM_THREADS=${THREADS:-2} CUDA_VISIBLE_DEVICES="" PYTORCH_ENABLE_MPS_FALLBACK=0
echo "$(date '+%F %T') start $NAME $*" >> ../runs/queue.log
${QOS-taskpolicy -b} nice -n 19 /usr/bin/time -l /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/.venv/bin/python \
  run_vs.py --name $NAME --threads ${THREADS:-2} "$@" > ../runs/$NAME/stdout.log 2>&1
echo "$(date '+%F %T') end $NAME exit $?" >> ../runs/queue.log
