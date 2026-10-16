#!/bin/bash
# G3 stability screen (draft r2 §8.4): scratch run, warm-up 20, 60 steps at peak LR 6e-4; per-term gradient audit at
# steps 0, 30 and 60 (final_eval part "audit" on the checkpoint, val batches). Writes runs/g3/*.  Usage: bash code/g3_screen.sh [train flags]
cd "$(dirname "$0")"
source ../../run_m2_25M/.venv/bin/activate
D=${G3_DIR:-../runs/g3}
rm -rf $D; mkdir -p $D
COMMON=(--run_dir $D --warmup 20 --log_every 5 --eval_at_v1_steps 0 --eval_every_min 1e9 --no_final_eval --dash_file= --milestone_hours 0 "$@")
python train.py "${COMMON[@]}" --init_only > $D/stdout.log 2>&1 || exit 1
python final_eval.py --ckpt $D/ckpt_last.pt --out $D/audit_step0.json --split val --parts audit $FE_EXTRA > $D/audit0.log 2>&1 || exit 1
python train.py "${COMMON[@]}" --max_steps 30 --resume $D/ckpt_last.pt >> $D/stdout.log 2>&1 || exit 1
python final_eval.py --ckpt $D/ckpt_last.pt --out $D/audit_step30.json --split val --parts audit $FE_EXTRA > $D/audit30.log 2>&1 || exit 1
python train.py "${COMMON[@]}" --max_steps 60 --resume $D/ckpt_last.pt >> $D/stdout.log 2>&1 || exit 1
python final_eval.py --ckpt $D/ckpt_last.pt --out $D/audit_step60.json --split val --parts audit $FE_EXTRA > $D/audit60.log 2>&1 || exit 1
grep -E " step [0-9]+ loss|NONFINITE|GUARD|STOP" $D/stdout.log
grep -h "audit summary" $D/audit*.log
echo G3 DONE
