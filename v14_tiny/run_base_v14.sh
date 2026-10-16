#!/bin/zsh
# V14-tiny base run (v14_Architecture.md §11; DEVIATIONS.md). Waits until V13-tiny's post pipeline has finished, so
# two training jobs never share the GPU; then two 60-step smoke tests decide activation checkpointing (choose_ckpt.py),
# then 11 h of training with resume-on-crash (at most 6 restarts).
# Launch:  nohup caffeinate -is ./run_base_v14.sh > runs/run_base_v14.out 2>&1 &
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
PY=/opt/anaconda3/bin/python3
mkdir -p runs/base
echo "$(date) waiting for ../v13_tiny/runs/post/ALL_DONE" >> runs/base/launch.log
until [ -f ../v13_tiny/runs/post/ALL_DONE ]; do
  sleep 60
  if ! pgrep -f "[.]/post_base[.]sh" > /dev/null && [ ! -f ../v13_tiny/runs/post/ALL_DONE ]; then
    echo "$(date) V13's post_base.sh is gone without ALL_DONE: starting anyway" >> runs/base/launch.log; break
  fi
done
echo "$(date) V13 finished; smoke tests" >> runs/base/launch.log
if [ ! -f runs/base/latest.pt ]; then
  $PY train.py --run runs/smoke_off --smoke 60 --ckpt off > runs/smoke_off.out 2>&1
  echo "$(date) smoke_off exit $?" >> runs/base/launch.log
  $PY train.py --run runs/smoke_on --smoke 60 --ckpt on > runs/smoke_on.out 2>&1
  echo "$(date) smoke_on exit $?" >> runs/base/launch.log
fi
CK=$($PY choose_ckpt.py 2>> runs/base/launch.log | tail -1)
case "$CK" in (on|off) ;; (*) CK=on;; esac
echo "$(date) activation checkpointing: $CK" >> runs/base/launch.log
for i in 1 2 3 4 5 6; do
  $PY train.py --run runs/base --hours 11 --ckpt $CK >> runs/base/train.out 2>&1 && exit 0
  echo "$(date) train.py exited with $?; restart $i in 30 s" >> runs/base/restarts.log
  sleep 30
done
