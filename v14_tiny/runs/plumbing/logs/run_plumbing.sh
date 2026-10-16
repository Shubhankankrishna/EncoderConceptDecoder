#!/bin/zsh
# CPU plumbing test of V14-tiny's post pipeline on the real snap_010 (V14 DEVIATIONS item 18; ADVISOR_LOG row 4).
# Every step on CPU at background QoS, one at a time, outputs only under this directory. Nothing touches the GPU.
PT=/private/tmp/claude-501/-Users-shubhankankrishna-Desktop-EncoderConceptDecoder/fccbea0c-feb8-4eee-b665-eba7f03fbda1/scratchpad/pt
V14=/Users/shubhankankrishna/Desktop/EncoderConceptDecoder/v14_tiny
PY=/opt/anaconda3/bin/python3
TR=$PT/runs
export TOKENIZERS_PARALLELISM=false
cd $V14
echo "$(date) waiting for snap_010.pt and latest.pt" >> $PT/status.log
until [ -f runs/base/snap_010.pt ] && [ -f runs/base/latest.pt ]; do sleep 30; done
mkdir -p $TR/base $TR/eval $TR/cl/rows $TR/cl_noera
for n in snap_010 snap_030 snap_050 snap_070 snap_090 final; do ln -sf $V14/runs/base/snap_010.pt $TR/base/$n.pt; done
ln -sf $V14/runs/base/config.json $TR/base/config.json; ln -sf $V14/runs/base/log.jsonl $TR/base/log.jsonl
ln -sfn $TR/cl/rows $TR/cl_noera/rows
step() { local name=$1; shift; echo "$(date) start $name" >> $PT/status.log
         taskpolicy -b "$@" > $PT/$name.out 2>&1; local rc=$?; echo "$(date) end $name exit $rc" >> $PT/status.log; }
step final_eval  $PY final_eval.py --ckpt $TR/base/final.pt --out $TR/eval/base --device cpu --quick 1 --no-curve
step diagnostics $PY diagnostics_v14.py --run $TR/base --out $TR/eval/diag --device cpu --quick
step era         $PY era_v14.py --run $TR/base --diag $TR/eval/diag/diagnostics_v14.json --device cpu --quick --force
step cl_choice   $PY cl_choice.py --v13 ../v13_tiny/runs/eval/cl/cl_results.json --out $TR/cl/choice.json
CL=(--base $TR/base/final.pt --steps 2 --B 2 --device cpu --workers 2 --res-rows 64 --choice $TR/cl/choice.json)
step cl_seq      $PY cl.py --method seq    --out $TR/cl $CL --era $TR/base/era.pt
step cl_v14      $PY cl.py --method v14    --out $TR/cl $CL --era $TR/base/era.pt
step cl_replay   $PY cl.py --method replay --out $TR/cl $CL --era $TR/base/era.pt
step cl_v14_noera $PY cl.py --method v14   --out $TR/cl_noera $CL --era $TR/base/no_era.pt
step cl_eval     $PY cl_eval.py --cl $TR/cl --base $TR/base/final.pt --out $TR/eval/cl --device cpu --methods seq,replay,v14 --quick 1
step report      env V14_RUNS=$TR V14_TABLES=$PT/tables.md $PY make_report.py
echo "$(date) ALL DONE" >> $PT/status.log
