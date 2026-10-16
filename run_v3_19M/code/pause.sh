#!/bin/bash
# Pause the v3 run safely. The trainer saves at the end of the current step (<= ~30 s; longer mid-eval) and exits 4.
# A running 24 h evaluation is stopped and redone on resume (MILESTONE_EVAL_PENDING stays).
cd "$(dirname "$0")/.."
RUN=runs/v3_48h
touch $RUN/PAUSE
echo "$(date '+%F %T') v3 pause requested" | tee -a runs/chain.log
if pgrep -f "final_eval.p[y].*v3_48h" >/dev/null; then pkill -f "final_eval.p[y].*v3_48h"; echo "stopped the running evaluation (redone on resume)"; fi
echo "waiting for the trainer to save and exit..."
while pgrep -f "run_dir ../runs/v3_48h" >/dev/null; do sleep 3; done
sleep 2
if [ -f runs/run_job.pid ]; then kill $(cat runs/run_job.pid) 2>/dev/null; fi
ls -l $RUN/ckpt_last.pt 2>/dev/null
grep -E "STOP" $RUN/stdout.log | tail -1
echo; echo "SAFE TO CLOSE LID. Resume later with: bash code/resume.sh"
