#!/bin/bash
# Throughput A/B on the real trainer (MPS, full model, B micro x 16 accum = 32,768 tokens/step), scratch runs only.
# Usage: bash code/bench_ab.sh NAME [train.py flags...]   -> appends "NAME tok/s(step 8) power" to runs/bench/results.txt
# Each run: model build + k-means init, 8 optimizer steps; tok/s is the trainer's own window over steps 5-8 (steps 1-4
# include warm-up / compilation). No evaluation, no milestone, no dashboard file. Run configurations back to back, twice.
cd "$(dirname "$0")"
source ../../run_m2_25M/.venv/bin/activate
NAME=$1; shift
D=../runs/bench/$NAME.$(date +%H%M%S)
mkdir -p ../runs/bench
python train.py --run_dir $D --max_steps 8 --log_every 4 --eval_at_v1_steps 0 --eval_every_min 1e9 --no_final_eval \
  --dash_file= --milestone_hours 0 "$@" > $D.log 2>&1
code=$?
T=$(grep -E " step 8 loss" $D.log | grep -Eo "[0-9]+ tok/s" | head -1)
T4=$(grep -E " step 4 loss" $D.log | grep -Eo "[0-9]+ tok/s" | head -1)
MEM=$(python - "$D/train_log.jsonl" <<'EOF'
import json, sys
try:
    r = [json.loads(l) for l in open(sys.argv[1]) if '"tok_s"' in l][-1]
    print(f"alloc {r.get('mps_alloc_gib', 0):.2f} GiB driver {r.get('mps_driver_gib', 0):.2f} GiB swap {r.get('swap_mb', 0):.0f} MB")
except Exception as e:
    print("n/a", e)
EOF
)
P=$(pmset -g batt | grep -Eo '[0-9]+%;[^;]*' | head -1)
echo "$(date '+%F %T') $NAME exit=$code steps5-8: $T (steps1-4: $T4) $MEM power: $P flags: $*" | tee -a ../runs/bench/results.txt
rm -rf $D                                                       # scratch checkpoints (~1.2 GB each)
