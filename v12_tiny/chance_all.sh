#!/bin/zsh
# Chance baselines for every finished α point (CPU), until phase 2 ends.
cd "$(dirname "$0")"
while true; do
  for f in runs/*_8k/meta/*.json(N); do
    case $f in *xm_*) continue;; esac
    grep -q '"chance"' $f && continue
    r=${f:h:h:t}; t=${f:t:r}; [ -f ${f:r}.pt ] || continue
    OMP_NUM_THREADS=1 python3 -B switch_chance.py $r $t 2>&1 | grep -v -i "userwarn\|detach" | tail -1
  done
  pgrep -f "phase2.py" >/dev/null || { echo done; break; }
  sleep 60
done
