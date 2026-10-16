#!/bin/bash
# Run one tiny v4 arm on the efficiency cores (background QoS), lowest priority, CPU only.
# usage: bash launch_arm.sh <name> [runner args...]
NAME=$1; shift
cd /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v4_tiny/code
mkdir -p ../runs/$NAME
export OMP_NUM_THREADS=${THREADS:-3} CUDA_VISIBLE_DEVICES="" PYTORCH_ENABLE_MPS_FALLBACK=0
echo "$(date '+%F %T') start $NAME $*" >> ../runs/queue.log
${QOS-taskpolicy -b} nice -n 19 /usr/bin/time -l /Users/shubhankankrishna/Desktop/EncoderConceptDecoder/run_m2_25M/.venv/bin/python \
  run_v4tiny.py --name $NAME --threads ${THREADS:-3} "$@" > ../runs/$NAME/stdout.log 2>&1
echo "$(date '+%F %T') end $NAME exit $?" >> ../runs/queue.log
