#!/bin/sh
cd /Users/shubhankankrishna/Desktop/EncoderConceptDecoder
for arm in C0 A X8 X4; do
  /opt/anaconda3/bin/python -B -m v10_3_tiny.train --arm $arm --run-dir v10_3_tiny/runs/smoke_mps_$arm --max-steps 200 --eval-seconds 1e9 --device mps > v10_3_tiny/runs/smoke_mps_$arm.log 2>&1
done
for arm in C0 X8; do
  /opt/anaconda3/bin/python -B -m v10_3_tiny.train --arm $arm --run-dir v10_3_tiny/runs/smoke_cpu_$arm --max-steps 50 --eval-seconds 1e9 --device cpu --threads 4 > v10_3_tiny/runs/smoke_cpu_$arm.log 2>&1
done
echo SMOKE_DONE > v10_3_tiny/runs/smoke.done
