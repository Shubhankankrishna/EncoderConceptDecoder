#!/bin/bash
# Continuation of the main run for another 24 h of training time (D-041).
# Waits for chain3 to log "chain done" (main run + its final evaluation), then trains runs/cont from the final main
# checkpoint with a fresh warm-up (6e-5 -> 6e-4 over 300 steps) and cosine back to 6e-5 over 4,050 steps, then runs the
# final evaluation on it. Never holds caffeinate while waiting (the lid may be closed after a pause).
#   start (waits):   nohup bash code/chain4.sh > runs/chain4.stdout 2>&1 &
#   resume:          RESUME=1 bash code/chain4.sh   (code/resume.sh does this when runs/cont has a checkpoint)
#   cancel:          touch runs/NO_CONT   (checked every minute while waiting)
# Test overrides: RUNS (runs dir), CHAIN_LOG, DRY=1 (print the training command instead of running anything), POLL (s).
cd "$(dirname "$0")"
RUNS=${RUNS:-../runs}; LOG=${CHAIN_LOG:-$RUNS/chain.log}; POLL=${POLL:-60}
C=$RUNS/cont; M=$RUNS/main
say() { echo "$(date '+%F %T') $*" >> "$LOG"; echo "$(date '+%F %T') $*"; }
LOCK=$RUNS/chain4.pid                                          # one chain4 at a time (re-arming after a reboot is fine)
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then say "chain4 already running (pid $(cat "$LOCK")); not starting another"; exit 1; fi
echo $$ > "$LOCK"
[ "$DRY" == "1" ] || source ../.venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [ "$RESUME" != "1" ]; then
  last() { grep -n "$1" "$LOG" | tail -1 | cut -d: -f1; }     # line number of the latest match (empty if none)
  done_after_start() { local d=$(last "chain done") s=$(last "main start"); [ -n "$d" ] && [ "${d:-0}" -gt "${s:-0}" ]; }
  say "cont armed: waits for the main chain to finish (kill switch: touch $RUNS/NO_CONT)"
  while ! done_after_start; do                                 # idempotent: re-arming after "chain done" starts at once
    if [ -f "$RUNS/NO_CONT" ]; then say "cont cancelled (NO_CONT)"; exit 0; fi
    f=$(last "main failed"); st=$(last "main start")
    if [ -n "$f" ] && [ "$f" -gt "${st:-0}" ]; then say "cont not started: main failed"; exit 1; fi
    sleep "$POLL"
  done
  if [ -f "$RUNS/NO_CONT" ]; then say "cont cancelled (NO_CONT)"; exit 0; fi
  if pgrep -f "train.py" >/dev/null && [ "$DRY" != "1" ]; then say "cont not started: a trainer is still running"; exit 1; fi
  if ! grep "STOP (" "$M/stdout.log" | tail -1 | grep -Eq "STOP \((max_steps|time_budget)\)"; then say "cont not started: main did not finish normally"; exit 1; fi
  if [ ! -f "$M/eval_final.json" ]; then say "cont not started: main final evaluation missing"; exit 1; fi
  mkdir -p "$C"
  [ -e "$C/init_from_main.pt" ] || ln "$M/ckpt_last.pt" "$C/init_from_main.pt"     # hard link: the exact base, survives pruning
fi

ARGS="--init_from $C/init_from_main.pt --micro_bs 2 --accum 16 --total_steps 4050 --warmup 300 --lr 6e-4 \
--warmup_start_frac 0.1 --lr_min_frac 0.1 --hours 24 --eval_every_min 60 --keep 3"     # user's choice 07:50 (D-041 addendum)
if [ "$DRY" == "1" ]; then echo "DRY: ./run_job2.sh $C $ARGS"; exit 0; fi

pgrep -f "battery_guard2.sh" >/dev/null || nohup bash battery_guard2.sh "$C" > "$RUNS/battery_guard2.stdout" 2>&1 &
say "cont start (RESUME=${RESUME:-0}): run_job2.sh $C $ARGS"
caffeinate -dimsu ./run_job2.sh "$C" $ARGS
rc=$?
if [ $rc -eq 4 ] || [ -f "$C/PAUSE" ]; then say "cont paused (rc $rc); run code/resume.sh to continue"; exit 0; fi
if [ $rc -ne 0 ]; then say "cont failed (rc $rc); chain stops"; exit 1; fi
say "cont training done; final evals"
caffeinate -dimsu python final_eval.py --ckpt "$C/ckpt_last.pt" --out "$C/eval_final.json" > "$C/eval_final.log" 2>&1
if [ "$(stat -f %i "$C/ckpt_best.pt")" != "$(stat -f %i "$C/ckpt_last.pt")" ]; then
  caffeinate -dimsu python final_eval.py --ckpt "$C/ckpt_best.pt" --out "$C/eval_best.json" --parts lang,trunc,items,health --probe_only_interv > "$C/eval_best.log" 2>&1
else echo "best == last" > "$C/eval_best.log"; fi
caffeinate -dimsu python test_p5.py --ckpt "$C/ckpt_last.pt" > "$C/p5_trained.txt" 2>&1
say "cont done"
