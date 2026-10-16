"""Read-only checkpoint diagnostics, explicitly secondary to the main matrix."""
import json, os, pathlib, subprocess, sys, time
HERE=pathlib.Path(__file__).resolve().parent
RUNS=HERE.parent/"runs"

jobs=[]
for seed in range(3):
    for arch in ("v4","llm"):
        jobs.append((f"ref_vs_pin_{arch}_s{seed}_6M","phase_audit.py","phase.json"))
        if arch=="llm":jobs.append((f"ref_vs_pin_{arch}_s{seed}_6M","counterfactual_v41.py","counterfactual_units.npz"))
    for arm in ("v41_pin","v41_stack_pin","v41_loopoff_pin"):
        name=f"{arm}_s{seed}_6M"
        jobs.extend((name,script,target) for script,target in (
            ("plan_diagnostics.py","plan_diagnostics.json"),
            ("counterfactual_v41.py","counterfactual_units.npz"),
            ("phase_audit.py","phase.json")))
for seed in range(2):
    jobs.append((f"v41_text_s{seed}_5M","plan_diagnostics.py","plan_diagnostics.json"))
for name in ("exploratory_noplan_pin_s0_6M","exploratory_mps_plan_pin_s0_6M"):
    jobs.extend((name,script,target) for script,target in (
        ("counterfactual_v41.py","counterfactual_units.npz"),
        ("phase_audit.py","phase.json")))

if __name__=="__main__":
    env=dict(os.environ,OMP_NUM_THREADS="1",PYTORCH_ENABLE_MPS_FALLBACK="0")
    pending=list(jobs)
    while pending:
        ready=[j for j in pending if (RUNS/j[0]/"final.json").exists()]
        if not ready:time.sleep(15);continue
        for name,script,target in ready:
            if not (RUNS/name/target).exists():
                with (RUNS/name/(script.removesuffix(".py")+"_secondary.log")).open("w") as log:
                    code=subprocess.call([sys.executable,str(HERE/script),name],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
                print(json.dumps(dict(name=name,diagnostic=script,exit_code=code)),flush=True)
                if code:sys.exit(code)
            pending.remove((name,script,target))
    print("all secondary diagnostics complete",flush=True)
