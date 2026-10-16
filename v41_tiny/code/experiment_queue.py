"""Run the registered arms without touching previous experiments."""
import datetime, hashlib, json, os, pathlib, subprocess, sys, time

HERE=pathlib.Path(__file__).resolve().parent
RUNS=HERE.parent/"runs"
JOBS=[
    ("v41_pin_s0_6M","v41","pinpad",0,6e6,""),
    ("fresh_pin_llm_s0_6M","llm","pinpad",0,6e6,""),
    ("v41_text_s0_5M","v41","text",0,5e6,""),
    ("v41_pin_s1_6M","v41","pinpad",1,6e6,""),
    ("v41_stack_pin_s0_6M","v41stack","pinpad",0,6e6,""),
    ("v41_text_s1_5M","v41","text",1,5e6,""),
    ("v41_pin_s2_6M","v41","pinpad",2,6e6,""),
    ("v41_loopoff_pin_s0_6M","v41","pinpad",0,6e6,"enc_loop=1"),
    ("v41_stack_pin_s1_6M","v41stack","pinpad",1,6e6,""),
    ("v41_loopoff_pin_s1_6M","v41","pinpad",1,6e6,"enc_loop=1"),
    ("v41_stack_pin_s2_6M","v41stack","pinpad",2,6e6,""),
    ("v41_loopoff_pin_s2_6M","v41","pinpad",2,6e6,"enc_loop=1"),
]

def event(**kw):
    row=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
    print(json.dumps(row),flush=True)
    with (RUNS/"queue.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")

if __name__=="__main__":
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")}
    (HERE.parent/"execution_source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    event(kind="queue_start",pid=os.getpid(),jobs=JOBS,device="cpu",threads=4)
    for name,arch,data,seed,budget,over in JOBS:
        out=RUNS/name;out.mkdir(exist_ok=True)
        if (out/"final.json").exists():
            event(kind="skip_complete",name=name);continue
        if (out/"log.jsonl").exists():
            event(kind="refuse_overwrite",name=name);sys.exit(2)
        cmd=[sys.executable,str(HERE/"run_vs.py"),"--name",name,"--arch",arch,"--data",data,"--seed",str(seed),
             "--real_tokens",str(budget),"--device","cpu","--threads","4","--eval_every_tokens",str(.5e6 if data=="pinpad" else 1e6)]
        if over:cmd += ["--over",over]
        event(kind="start",name=name,command=cmd)
        with (out/"stdout.log").open("w") as log:
            p=subprocess.Popen(cmd,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK="0"))
            event(kind="running",name=name,pid=p.pid)
            code=p.wait()
        event(kind="end",name=name,exit_code=code)
        if code:sys.exit(code)
    event(kind="queue_complete")
