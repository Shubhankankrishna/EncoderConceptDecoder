#!/bin/zsh
# After base training: an MPS probe of cl.py at B = 8 (plumbing and timing on the idle machine), the final base
# evaluation and diagnostics, the continual budget (cl_budget.py, DEVIATIONS item 32), the three continual methods
# (same base, same incoming rows; v13 before replay so a failure surfaces early), the continual matrix, the tables.
# Each step logs to runs/post/<step>.out; a failing step is recorded and the next one runs.
# Launch:  nohup caffeinate -is ./post_base.sh > runs/post/post.out 2>&1 &      (CL_STEPS=n overrides the budget)
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
PY=/opt/anaconda3/bin/python3
mkdir -p runs/post
echo "$(date) waiting for runs/base/DONE" >> runs/post/status.log
# Wait for the base run; if its launcher is gone without DONE (restarts exhausted), fall back to latest.pt.
until [ -f runs/base/DONE ]; do
  sleep 60
  if ! pgrep -f "run_base.sh" > /dev/null && ! pgrep -f "train.py --run runs/base" > /dev/null && [ ! -f runs/base/DONE ]; then
    echo "$(date) base launcher gone without DONE: using latest.pt as the base" >> runs/post/status.log
    $PY -c "import torch; z = torch.load('runs/base/latest.pt', map_location='cpu', weights_only=False); torch.save(dict(model=z['model'], codebook=z['codebook'], step=z['state']['step'], elapsed=z['state']['elapsed'], tokens=z['state']['tokens']), 'runs/base/final.pt')" >> runs/post/status.log 2>&1
    touch runs/base/DONE_FALLBACK; break
  fi
done
while pgrep -f "train.py --run runs/base" > /dev/null; do sleep 10; done
step() { local name=$1; shift; echo "$(date) start $name" >> runs/post/status.log
         "$@" > runs/post/$name.out 2>&1; local rc=$?; echo "$(date) end $name exit $rc" >> runs/post/status.log; }
step probe_v13   $PY cl.py --method v13 --base runs/base/final.pt --out runs/post/probe --steps 4 --B 8 --phases S1,S2
step probe_seq   $PY cl.py --method seq --base runs/base/final.pt --out runs/post/probe --steps 5 --B 8 --phases S1
step final_eval  $PY final_eval.py --ckpt runs/base/final.pt --out runs/eval/base
step diagnostics $PY diagnostics.py --ckpt runs/base/final.pt --out runs/eval/diag
if [ -n "$CL_STEPS" ]; then N=$CL_STEPS; else N=$($PY cl_budget.py --probe runs/post/probe 2>> runs/post/status.log | tail -1); fi
case "$N" in (''|*[!0-9]*) echo "$(date) budget failed ('$N'): using 100" >> runs/post/status.log; N=100;; esac
echo "$(date) continual steps per phase: $N" >> runs/post/status.log
step cl_seq      $PY cl.py --method seq    --base runs/base/final.pt --out runs/cl --steps $N
step cl_v13      $PY cl.py --method v13    --base runs/base/final.pt --out runs/cl --steps $N
step cl_replay   $PY cl.py --method replay --base runs/base/final.pt --out runs/cl --steps $N
step cl_eval     $PY cl_eval.py --cl runs/cl --out runs/eval/cl
step report      $PY make_report.py
echo "$(date) ALL DONE" >> runs/post/status.log
touch runs/post/ALL_DONE
