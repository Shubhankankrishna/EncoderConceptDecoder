#!/bin/bash
# v2 launcher (48 h, one schedule). Usage: bash code/run_job.sh [train.py args...]   (from anywhere; run it via nohup bash)
#   exit 0 of train.py  -> final_eval (48 h) on ckpt_last + ckpt_best -> "v2 done"
#   exit 5 (milestone)  -> final_eval (24 h) on ckpt_last -> resume training (the schedule does NOT restart)
#   exit 4 (paused)     -> stop (code/resume.sh continues; a pending 24 h evaluation is redone first)
#   anything else       -> resume from ckpt_last.pt, at most 6 crashes, then "v2 failed"
cd "$(dirname "$0")"
source ../../run_m2_25M/.venv/bin/activate
RUN=../runs/v2_48h
LOG=../runs/chain.log
mkdir -p $RUN
if pgrep -f "train.p[y]" >/dev/null; then echo "$(date '+%F %T') v2 not started: a trainer is already running" >> $LOG; exit 1; fi
echo $$ > ../runs/run_job.pid
caffeinate -ims -w $$ &                                         # keep the Mac awake (idle/disk/system) while this runs
crashes=0
milestone_eval() {
  touch $RUN/MILESTONE_EVAL_PENDING
  echo "$(date '+%F %T') v2 24 h evaluation start" >> $LOG
  python final_eval.py --ckpt $RUN/ckpt_last.pt --out $RUN/eval_24h.json >> $RUN/final_eval_24h.log 2>&1
  local c=$?
  if [ $c -eq 0 ]; then rm -f $RUN/MILESTONE_EVAL_PENDING; echo "$(date '+%F %T') v2 24 h evaluation done" >> $LOG
  else echo "$(date '+%F %T') v2 24 h evaluation exited $c (pending marker kept)" >> $LOG; fi
  return $c
}
while true; do
  if [ -f $RUN/PAUSE ]; then echo "$(date '+%F %T') v2 launcher: PAUSE present, stopping" >> $LOG; exit 4; fi
  if [ -f $RUN/MILESTONE_EVAL_PENDING ]; then milestone_eval || exit 4; fi
  if [ -f $RUN/ckpt_last.pt ]; then RES="--resume $RUN/ckpt_last.pt"; else RES=""; fi
  echo "=== $(date '+%F %T') launch $RES" >> $RUN/launcher.log
  python train.py --run_dir $RUN "$@" $RES >> $RUN/stdout.log 2>&1
  code=$?
  echo "=== $(date '+%F %T') exit code $code" >> $RUN/launcher.log
  case $code in
    0) echo "$(date '+%F %T') v2 training done; final evaluation (48 h)" >> $LOG
       python final_eval.py --ckpt $RUN/ckpt_last.pt --out $RUN/eval_final.json >> $RUN/final_eval.log 2>&1
       python final_eval.py --ckpt $RUN/ckpt_best.pt --out $RUN/eval_best.json --probe_only_interv >> $RUN/final_eval.log 2>&1
       echo "$(date '+%F %T') v2 done" >> $LOG; exit 0 ;;
    4) echo "$(date '+%F %T') v2 paused" >> $LOG; exit 4 ;;
    5) echo "$(date '+%F %T') v2 milestone (24 h) reached" >> $LOG
       milestone_eval || exit 4 ;;
    *) crashes=$((crashes + 1)); echo "$(date '+%F %T') v2 trainer exit $code (crash $crashes/6)" >> $LOG
       if [ $crashes -ge 6 ]; then echo "$(date '+%F %T') v2 failed" >> $LOG; exit 1; fi
       sleep 20 ;;
  esac
done
