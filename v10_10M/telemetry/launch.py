"""Start the independent telemetry collector without touching the trainer."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
from .server import ROOT, DEFAULT_DATA


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',default=str(ROOT/'runs/pretrain_24h'))
    p.add_argument('--data-dir',default=str(DEFAULT_DATA));p.add_argument('--port',type=int,default=8766);a=p.parse_args()
    data=Path(a.data_dir).resolve();data.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u','-m','v10_10M.telemetry.server','--run-dir',str(Path(a.run_dir).resolve()),
         '--data-dir',str(data),'--port',str(a.port)]
    with (data/'collector.log').open('a') as log:
        child=subprocess.Popen(cmd,cwd=ROOT.parent,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(40):
        if child.poll() is not None:raise SystemExit(f'Collector exited: inspect {data}/collector.log')
        try:
            with urlopen(f'http://127.0.0.1:{a.port}/api/v1/health',timeout=1) as response:health=json.load(response)
            if health['pid']==child.pid and health['status']=='ok':print(json.dumps(health,indent=2));return
        except (OSError,ValueError):pass
        time.sleep(.25)
    raise SystemExit(f'Collector PID {child.pid} did not become ready in 10s; inspect its log')


if __name__=='__main__':main()
