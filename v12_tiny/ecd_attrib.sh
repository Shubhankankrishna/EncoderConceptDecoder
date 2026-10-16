#!/bin/zsh
# ECD (NTP-only) attribution evals on CPU (rule after the 13:23 OOM: no GPU eval while two GPU jobs run).
# Stops ECD training after ckpt_8000 (12k cosine schedule kept for comparability) and derives runs/ECD_s1_8k.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=3
cpu() { python3 -B -c "import torch; torch.set_num_threads(3); $1" 2>&1 | grep -v -i "userwarn\|detach" | tail -3; }
log() { echo "$(date +%H:%M) $*"; }
log "ECDe 8k usage rerun (shuffle column)"; cpu "import usage; usage.main('ECDe_s1','ckpt_8000',device='cpu')"
until [ -f runs/ECD_s1/ckpt_4000.pt ]; do sleep 30; done; sleep 10
log "ECD 4k"; cpu "import probe_diag, json; r=probe_diag.main('ECD_s1',device='cpu',ckpt='ckpt_4000'); open('runs/ECD_s1/probe_diag_ckpt_4000.json','w').write(json.dumps(r,indent=1))"
cpu "import usage; usage.main('ECD_s1','ckpt_4000',device='cpu')"
cpu "import info_test; info_test.main('ECD_s1','ckpt_4000',device='cpu',cb_from='ECDe_s1:ckpt_4000')"
until [ -f runs/ECD_s1/ckpt_8000.pt ]; do sleep 30; done; sleep 10
pkill -f '^[^ ]*[Pp]ython[0-9.]* -B train.py ECD 1 12000' && log "ECD stopped after ckpt_8000"
python3 - <<'PY'
import json, os
from pathlib import Path
R = Path('runs'); d = R / 'ECD_s1_8k'; d.mkdir(exist_ok=True)
m = json.loads((R / 'ECD_s1' / 'manifest.json').read_text())
m.update(derived_from='ECD_s1', ckpt='ckpt_8000', note='ECD (NTP-only control) on the 12k schedule, stopped after ckpt_8000 for the matched-8k comparison.')
(d / 'manifest.json').write_text(json.dumps(m, indent=1))
if not (d / 'final.pt').exists(): os.link(R / 'ECD_s1' / 'ckpt_8000.pt', d / 'final.pt')
PY
log "ECD 8k"; cpu "import probe_diag, json; r=probe_diag.main('ECD_s1',device='cpu',ckpt='ckpt_8000'); open('runs/ECD_s1/probe_diag_ckpt_8000.json','w').write(json.dumps(r,indent=1))"
cpu "import usage; usage.main('ECD_s1','ckpt_8000',device='cpu')"
cpu "import info_test; info_test.main('ECD_s1','ckpt_8000',device='cpu',cb_from='ECDe_s1:ckpt_8000')"
cpu "import devnll; devnll.main('TF7_s1',['ckpt_4000','ckpt_8000','final'],device='cpu')"
log "ecd_attrib done"
