#!/bin/zsh
# The rest of the CPU plumbing test (V14 DEVIATIONS item 19): cl_eval and make_report on the test's continual outputs
# (runs/plumbing/runs, trained from snap_010), started once base training has finished so it costs no training
# throughput. CPU only (nice 10); writes only under runs/plumbing.
# Launch:  nohup runs/plumbing/run_cl_eval_test.sh > runs/plumbing/logs/run_cl_eval_test.out 2>&1 &
cd "$(dirname "$0")/../.."
export TOKENIZERS_PARALLELISM=false
PY=/opt/anaconda3/bin/python3
T=$PWD/runs/plumbing/runs; L=$PWD/runs/plumbing/logs
echo "$(date) waiting for runs/base/DONE" >> $L/status.log
until [ -f runs/base/DONE ] || [ -f runs/base/DONE_FALLBACK ]; do sleep 60; done
echo "$(date) start cl_eval (rerun)" >> $L/status.log
nice -n 10 $PY cl_eval.py --cl $T/cl --base $PWD/runs/base/snap_010.pt --out $T/eval/cl --device cpu \
    --methods seq,replay,v14 --quick 1 > $L/cl_eval_rerun.out 2>&1
echo "$(date) end cl_eval (rerun) exit $?" >> $L/status.log
echo "$(date) start report" >> $L/status.log
V14_RUNS=$T V14_TABLES=$PWD/runs/plumbing/tables.md nice -n 10 $PY make_report.py > $L/report.out 2>&1
echo "$(date) end report exit $?" >> $L/status.log
echo "$(date) ALL DONE" >> $L/status.log
