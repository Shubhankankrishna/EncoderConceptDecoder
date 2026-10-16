"""Analyse completed registered runs while the training queue continues."""
import json, os, pathlib, subprocess, sys, time
from experiment_queue import JOBS

HERE=pathlib.Path(__file__).resolve().parent
RUNS=HERE.parent/"runs"

if __name__=="__main__":
    pending=[j[0] for j in JOBS]
    env=dict(os.environ,V41_ROLLOUT_MPS="1",OMP_NUM_THREADS="1",PYTORCH_ENABLE_MPS_FALLBACK="0")
    while pending:
        ready=[n for n in pending if (RUNS/n/"final.json").exists()]
        if not ready:
            time.sleep(15);continue
        for name in ready:
            if not (RUNS/name/"analysis.json").exists():
                with (RUNS/name/"analysis_stdout.log").open("w") as log:
                    code=subprocess.call([sys.executable,str(HERE/"analyze_vs.py"),"arm",name],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
                print(json.dumps(dict(name=name,phase="analysis",exit_code=code)),flush=True)
                if code:sys.exit(code)
            pending.remove(name)
            subprocess.call([sys.executable,str(HERE/"compare_v41.py")],cwd=HERE,env=env,stdout=subprocess.DEVNULL)
    print("all registered analyses complete",flush=True)
