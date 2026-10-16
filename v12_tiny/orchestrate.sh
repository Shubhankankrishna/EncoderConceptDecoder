#!/bin/zsh
# Phase-1 → Phase-2 hand-off (advisor #12 slot plan). Light evals under tight MPS caps; trainings/phase2 at their own caps.
cd "$(dirname "$0")"
LIGHT=(env PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25 OMP_NUM_THREADS=2)
log() { echo "$(date +%H:%M) $*"; }
until [ -f runs/TF7_s1/ckpt_8000.pt ]; do sleep 30; done
log "TF7 ckpt_8000 probes"; $LIGHT python3 -B probe_diag.py TF7_s1 ckpt_8000 > runs/phase2_logs/probe_TF7_8000.log 2>&1
until [ -f runs/TF7_s1/final.pt ]; do sleep 30; done
log "TF7 done → launch ECD s1 (NTP-only control) 12k"
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 python3 -B train.py ECD 1 12000 > runs/ECD_s1.log 2>&1 &
until [ -f runs/ECDe_s1/ckpt_8000.pt ]; do sleep 30; done
log "ECDe ckpt_8000 probes/usage/info"
$LIGHT python3 -B probe_diag.py ECDe_s1 ckpt_8000 > runs/phase2_logs/probe_ECDe_8000.log 2>&1
$LIGHT python3 -B usage.py ECDe_s1 ckpt_8000 > runs/phase2_logs/usage_ECDe_8000.log 2>&1
$LIGHT python3 -B info_test.py ECDe_s1 ckpt_8000 > runs/phase2_logs/info_ECDe_8000.log 2>&1
until [ -f runs/ECDe_s1/final.pt ]; do sleep 30; done
log "ECDe done → phase2 TF7_s1 then ECDe_s1 (parallel=1 while ECD trains)"
python3 -B phase2.py --parallel=1 TF7_s1 ECDe_s1
log "orchestrate done"
