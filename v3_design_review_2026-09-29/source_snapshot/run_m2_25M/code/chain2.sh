#!/bin/bash
# Tail of chain.sh from the main run onward (used after the step-count extension, D-033).
cd "$(dirname "$0")"
source ../.venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
LOG=../runs/chain.log
N=265
STEPS=4050
echo "$(date '+%F %T') main start, total_steps $STEPS" >> $LOG
./run_job.sh ../runs/main --micro_bs 2 --accum 16 --total_steps $STEPS --warmup 300 --hours 24 --eval_every_min 60 --extra_eval_steps $N --keep 3
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
