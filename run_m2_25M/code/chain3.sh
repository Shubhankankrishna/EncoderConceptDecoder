#!/bin/bash
# Tail of chain.sh from the main run onward, pause-aware (D-033, D-034). Started by resume.sh / swap_at_ckpt.sh.
cd "$(dirname "$0")"
source ../.venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
LOG=../runs/chain.log
N=265
STEPS=4050
echo "$(date '+%F %T') main start, total_steps $STEPS" >> $LOG
./run_job2.sh ../runs/main --micro_bs 2 --accum 16 --total_steps $STEPS --warmup 300 --hours 24 --eval_every_min 60 --extra_eval_steps $N --keep 3
rc=$?
if [ $rc -eq 4 ] || [ -f ../runs/main/PAUSE ]; then echo "$(date '+%F %T') main paused (rc $rc); run code/resume.sh to continue" >> $LOG; exit 0; fi
if [ $rc -ne 0 ]; then echo "$(date '+%F %T') main failed (rc $rc); chain stops" >> $LOG; exit 1; fi
echo "$(date '+%F %T') main done; final evals" >> $LOG
python final_eval.py --ckpt ../runs/main/ckpt_last.pt --out ../runs/main/eval_final.json > ../runs/main/eval_final.log 2>&1
if [ "$(stat -f %i ../runs/main/ckpt_best.pt)" != "$(stat -f %i ../runs/main/ckpt_last.pt)" ]; then
  python final_eval.py --ckpt ../runs/main/ckpt_best.pt --out ../runs/main/eval_best.json --parts lang,trunc,items,health --probe_only_interv > ../runs/main/eval_best.log 2>&1
else echo "best == last" > ../runs/main/eval_best.log; fi
for s in S1 S2 S3; do python test_p5.py --ckpt ../runs/screens/$s/ckpt_last.pt > ../runs/screens/$s/p5_trained.txt 2>&1; done
python test_p5.py --ckpt ../runs/main/ckpt_last.pt > ../runs/main/p5_trained.txt 2>&1
python plots.py ../runs/main ../runs/screens/S1 ../runs/screens/S2 ../runs/screens/S3 >> $LOG 2>&1
python report_tables.py > ../runs/report_tables.md 2>> $LOG
echo "$(date '+%F %T') chain done" >> $LOG
