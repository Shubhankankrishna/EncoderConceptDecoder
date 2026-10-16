"""Two CPU workers for the remaining registered arms; same budgets and rows."""
import datetime,json,os,pathlib,signal,subprocess,sys,time
from experiment_queue import JOBS
HERE=pathlib.Path(__file__).resolve().parent
RUNS=HERE.parent/"runs"

def event(**kw):
    row=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
    print(json.dumps(row),flush=True)
    with (RUNS/"parallel_queue.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")

if __name__=="__main__":
    # The original parent is stopped to prevent it launching another child.
    # Its current training child continues unchanged and is adopted here.
    parent,current=3259,3845
    os.kill(parent,signal.SIGSTOP)
    event(kind="handoff_wait",original_parent=parent,current_child=current)
    while True:
        state=subprocess.run(["ps","-o","stat=","-p",str(current)],capture_output=True,text=True).stdout.strip()
        if not state or state.startswith("Z"):break
        time.sleep(5)
    out=RUNS/"v41_pin_s1_6M"
    if not ((out/"final.json").exists() and (out/"ckpt.pt").exists()):
        os.kill(parent,signal.SIGCONT)
        event(kind="handoff_failed");sys.exit(2)
    os.kill(parent,signal.SIGKILL)  # idle scheduler, after its training child exited
    event(kind="adopted_end",name="v41_pin_s1_6M",exit_code=0)
    pending=[j for j in JOBS if not (RUNS/j[0]/"final.json").exists()]
    active=[]
    event(kind="parallel_queue_start",device="cpu",workers=2,threads_per_worker=2,jobs=pending)
    while pending or active:
        for entry in list(active):
            p,name,log=entry
            code=p.poll()
            if code is None:continue
            log.close();active.remove(entry);event(kind="end",name=name,exit_code=code)
            if code:
                # Let the other already-started run finish before returning failure.
                for other,oname,olog in active:
                    oc=other.wait();olog.close();event(kind="end",name=oname,exit_code=oc)
                sys.exit(code)
        while pending and len(active)<2:
            name,arch,data,seed,budget,over=pending.pop(0)
            out=RUNS/name;out.mkdir(exist_ok=True)
            if (out/"final.json").exists():event(kind="skip_complete",name=name);continue
            if (out/"log.jsonl").exists():event(kind="refuse_overwrite",name=name);sys.exit(2)
            cmd=[sys.executable,str(HERE/"run_vs.py"),"--name",name,"--arch",arch,"--data",data,"--seed",str(seed),
                 "--real_tokens",str(budget),"--device","cpu","--threads","2","--eval_every_tokens",str(.5e6 if data=="pinpad" else 1e6)]
            if over:cmd += ["--over",over]
            log=(out/"stdout.log").open("w")
            p=subprocess.Popen(cmd,cwd=HERE,env=dict(os.environ,PYTORCH_ENABLE_MPS_FALLBACK="0"),stdout=log,stderr=subprocess.STDOUT)
            active.append((p,name,log));event(kind="start",name=name,pid=p.pid,command=cmd)
        if active:time.sleep(5)
    event(kind="parallel_queue_complete")
