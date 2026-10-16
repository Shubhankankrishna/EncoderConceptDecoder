#!/bin/zsh
# After V14-tiny's base training: the final base evaluation, V14's diagnostics (§6.3's reads on the snapshots, V1-V5),
# §6.1's end-of-training guard and, if it fails, the era refit (era_v14.py),
# the continual stage on V13-tiny's cached incoming rows (sequential LoRA, V14's continual default chosen by §9.4 from
# V13-tiny's matrix, replay), the continual matrix and the report. Each step logs to runs/post/<step>.out; a failing
# step is recorded and the next one runs. Python files are read when each step starts.
# Launch:  nohup caffeinate -is ./post_base_v14.sh > runs/post/post.out 2>&1 &
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
PY=/opt/anaconda3/bin/python3
mkdir -p runs/post
echo "$(date) waiting for runs/base/DONE" >> runs/post/status.log
until [ -f runs/base/DONE ]; do
  sleep 60
  if [ -f runs/base/latest.pt ] && ! pgrep -f "run_base_v14[.]sh" > /dev/null && ! pgrep -f "train.py --run runs/base" > /dev/null \
     && [ ! -f runs/base/DONE ]; then
    echo "$(date) base launcher gone without DONE: using latest.pt as the base" >> runs/post/status.log
    $PY -c "import torch; z = torch.load('runs/base/latest.pt', map_location='cpu', weights_only=False); torch.save(dict(model=z['model'], twin=z['twin'], codebook=z['codebook'], maps=z['maps'], step=z['state']['step'], elapsed=z['state']['elapsed'], tokens=z['state']['tokens'], lam_scale=z['state']['lam_scale']), 'runs/base/final.pt')" >> runs/post/status.log 2>&1
    touch runs/base/DONE_FALLBACK; break
  fi
done
while pgrep -f "train.py --run runs/base" > /dev/null; do sleep 10; done
step() { local name=$1; shift; echo "$(date) start $name" >> runs/post/status.log
         "$@" > runs/post/$name.out 2>&1; local rc=$?; echo "$(date) end $name exit $rc" >> runs/post/status.log; }
step final_eval  $PY final_eval.py --ckpt runs/base/final.pt --out runs/eval/base
step diagnostics $PY diagnostics_v14.py --run runs/base --out runs/eval/diag
step era         $PY era_v14.py --run runs/base --diag runs/eval/diag/diagnostics_v14.json
step cl_choice   $PY cl_choice.py --v13 ../v13_tiny/runs/eval/cl/cl_results.json --out runs/cl/choice.json
step cl_seq      $PY cl.py --method seq    --base runs/base/final.pt --out runs/cl --steps 100
step cl_v14      $PY cl.py --method v14    --base runs/base/final.pt --out runs/cl --steps 100
step cl_replay   $PY cl.py --method replay --base runs/base/final.pt --out runs/cl --steps 100
step cl_eval     $PY cl_eval.py --cl runs/cl --out runs/eval/cl --methods seq,replay,v14
step report      $PY make_report.py
echo "$(date) ALL DONE" >> runs/post/status.log
touch runs/post/ALL_DONE
