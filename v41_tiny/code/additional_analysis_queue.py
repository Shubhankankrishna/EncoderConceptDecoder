"""Finish the matched-MPS control's analysis after its training completes."""
import json,os,pathlib,subprocess,sys,time
HERE=pathlib.Path(__file__).resolve().parent
RUNS=HERE.parent/"runs"
if __name__=="__main__":
    name="exploratory_mps_plan_pin_s0_6M"
    while not (RUNS/name/"final.json").exists():time.sleep(15)
    env=dict(os.environ,OMP_NUM_THREADS="1",V41_ROLLOUT_MPS="1",PYTORCH_ENABLE_MPS_FALLBACK="0")
    with (RUNS/name/"analysis_stdout.log").open("w") as log:
        code=subprocess.call([sys.executable,str(HERE/"analyze_vs.py"),"arm",name],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
    (RUNS/name/"analysis_exit.json").write_text(json.dumps(dict(exit_code=code))+"\n")
    print(json.dumps(dict(name=name,phase="analysis",exit_code=code)),flush=True)
    sys.exit(code)
