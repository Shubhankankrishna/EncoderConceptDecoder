#!/bin/sh
cd /Users/shubhankankrishna/Desktop/EncoderConceptDecoder
P=/opt/anaconda3/bin/python; R=v10_3_tiny/runs
{ date; vm_stat | head -4; } > $R/launch_s1.env
$P -B -m v10_3_tiny.train --arm X8 --run-dir $R/X8_s1 --max-steps 6000 --device cpu --threads 3 --seed 1 > $R/X8_s1.log 2>&1 &
$P -B -m v10_3_tiny.train --arm A  --run-dir $R/A_s1  --max-steps 6000 --device cpu --threads 3 --seed 1 > $R/A_s1.log 2>&1 &
$P -B -m v10_3_tiny.train --arm X4 --run-dir $R/X4_s1 --max-steps 6000 --device mps --threads 1 --seed 1 > $R/X4_s1.log 2>&1 &
$P -B -m v10_3_tiny.train --arm C0 --run-dir $R/C0_s1 --max-steps 6000 --device mps --threads 1 --seed 1 > $R/C0_s1.log 2>&1 &
wait
date > $R/launch_s1.done
