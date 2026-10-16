"""Launch once, hold an OS process lock, prevent idle sleep and retain logs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .prepare_data import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default=str(ROOT/'runs/pretrain_24h'))
    p.add_argument('--resume',action='store_true');p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--accumulate',type=int,default=1);a=p.parse_args()
    run=Path(a.run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    # Existing output requires explicit resume, never accidental overwrite.
    if (run/'latest.pt').exists() and not a.resume:raise SystemExit('Run exists; choose another directory or --resume')
    status_path=run/'status.json'
    if status_path.exists():
        state=json.loads(status_path.read_text())
        if state.get('status')=='running':
            try:os.kill(state['pid'],0)
            except ProcessLookupError:pass
            else:raise SystemExit(f'Run already active: PID {state["pid"]}')
    # The supervisor uses flock, including during startup before status.json exists.
    cmd=[sys.executable,'-u','-m','v10_10M.supervise','--run-dir',str(run),
         '--batch-size',str(a.batch_size),'--accumulate',str(a.accumulate)]
    if a.resume:cmd.append('--resume')
    with open(run/'process.log','a') as log:
        child=subprocess.Popen(cmd,cwd=ROOT.parent,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    sleepguard=subprocess.Popen(['/usr/bin/caffeinate','-is','-w',str(child.pid)],stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    (run/'launch.json').write_text(json.dumps(dict(supervisor_pid=child.pid,caffeinate_pid=sleepguard.pid,
                                                launched_unix=time.time(),command=cmd),indent=2))
    print(json.dumps(dict(supervisor_pid=child.pid,caffeinate_pid=sleepguard.pid,run_dir=str(run))))


if __name__=='__main__':main()
