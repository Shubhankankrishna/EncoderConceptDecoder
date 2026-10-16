"""Detach the locked trainer, prevent idle sleep, and verify startup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .train import parser,ROOT


def main():
    args=parser().parse_args();run=Path(args.run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    if (run/'latest.pt').exists() and not args.resume:raise SystemExit('Existing run; use --resume')
    if (run/'status.json').exists():
        state=json.loads((run/'status.json').read_text())
        if state.get('status')=='running':
            try:os.kill(state['pid'],0)
            except ProcessLookupError:pass
            else:raise SystemExit('Trainer already active')
    cmd=[sys.executable,'-B','-u','-m','v10_1.supervise',*sys.argv[1:]]
    env=os.environ.copy();env['TOKENIZERS_PARALLELISM']='false'
    with (run/'process.log').open('a') as log:
        child=subprocess.Popen(cmd,cwd=ROOT.parent,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    guard=subprocess.Popen(['/usr/bin/caffeinate','-is','-w',str(child.pid)],stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    launch=dict(supervisor_pid=child.pid,caffeinate_pid=guard.pid,launched_unix=time.time(),command=cmd,run_dir=str(run))
    (run/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    for _ in range(120):
        if child.poll() is not None:raise SystemExit('Trainer exited; inspect '+str(run/'process.log'))
        if (run/'status.json').exists():
            state=json.loads((run/'status.json').read_text())
            if state.get('pid')==child.pid and state.get('status')=='running':print(json.dumps(launch));return
        time.sleep(.25)
    raise SystemExit('Startup not confirmed within 30 seconds; inspect '+str(run/'process.log'))


if __name__=='__main__':main()
