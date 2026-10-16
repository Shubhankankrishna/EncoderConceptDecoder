"""Explicit paired comparisons; never collapse distinct trained configurations."""
import json, math
from pathlib import Path
import numpy as np

RUNS=Path(__file__).resolve().parent.parent/"runs"

def record(name):
    p=RUNS/name
    if not (p/"per_unit.npz").exists():return None
    return json.loads((p/"analysis.json").read_text()),np.load(p/"per_unit.npz")

def paired(a,b):
    assert a.shape==b.shape,(a.shape,b.shape)
    ok=np.isfinite(a)&np.isfinite(b);x=a[ok]-b[ok]
    return dict(mean=float(x.mean()),se=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None,n=int(len(x))) if len(x) else None

def comparison(pairs):
    results={};names=[]
    for left,right in pairs:
        A,B=record(left),record(right)
        if A is None or B is None:continue
        ja,a=A;jb,b=B
        assert ja["seed"]==jb["seed"] and ja["data"]==jb["data"] and ja["budget"]==jb["budget"],(left,right)
        names.append(dict(left=left,right=right,seed=ja["seed"]))
        for key in a.files:
            if key not in b.files or a[key].dtype.kind not in "fb":continue
            if key=="item_hit":
                assert np.array_equal(a["item_family"],b["item_family"])
                keys=[("probe/"+fam,a[key][a["item_family"]==fam],b[key][b["item_family"]==fam]) for fam in np.unique(a["item_family"])]
            else:keys=[(key,a[key],b[key])]
            for label,x,y in keys:
                z=paired(x,y)
                if z is not None:results.setdefault(label,[]).append(dict(seed=ja["seed"],**z))
    summary={}
    for key,per in results.items():
        vals=np.array([v["mean"] for v in per]);ses=np.array([v["se"] or 0. for v in per])
        summary[key]=dict(mean=float(vals.mean()),paired_item_se=float(np.sqrt((ses**2).sum())/len(ses)),
                          seed_se=float(vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else None,
                          seed_min=float(vals.min()),seed_max=float(vals.max()),n_seeds=len(vals),per_seed=per)
    return dict(pairs=names,metrics=summary)

def main():
    pairs={
      "pin_v41_minus_noR":[(f"v41_pin_s{s}_6M",f"ref_vs_pin_v4_s{s}_6M") for s in range(3)],
      "pin_v41_minus_LLM":[(f"v41_pin_s{s}_6M",f"ref_vs_pin_llm_s{s}_6M") for s in range(3)],
      "pin_stack_minus_v41":[(f"v41_stack_pin_s{s}_6M",f"v41_pin_s{s}_6M") for s in range(3)],
      "pin_loopoff_minus_v41":[(f"v41_loopoff_pin_s{s}_6M",f"v41_pin_s{s}_6M") for s in range(3)],
      "text_v41_minus_noR":[(f"v41_text_s{s}_5M",f"ref_vs_text_v4_s{s}") for s in range(2)],
      "text_v41_minus_LLM":[(f"v41_text_s{s}_5M",f"ref_vs_text_llm_s{s}") for s in range(2)],
      "fresh_LLM_minus_reference":[("fresh_pin_llm_s0_6M","ref_vs_pin_llm_s0_6M")],
      "pin_v41_minus_fresh_LLM":[("v41_pin_s0_6M","fresh_pin_llm_s0_6M")],
    }
    out={name:comparison(ps) for name,ps in pairs.items()}
    (RUNS/"v41_compare.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    lines=["# v4.1 registered comparisons", "", "Differences are left minus right. Item-level paired SE and training-seed SE are separate.",""]
    for name,res in out.items():
        lines += ["## "+name,"", "Pairs: "+", ".join(f"{p['left']} / {p['right']}" for p in res["pairs"]),"",
                  "| Metric | Difference | Paired item SE | Seed SE | Seeds |", "|---|---:|---:|---:|---:|"]
        for key,v in res["metrics"].items():
            if "n_" in key or "steps" in key:continue
            if key.endswith("ntp_action") or key.endswith("act_opt") or key.endswith("act_opt_b2") or key in ("row_ntp",) or key.startswith("probe/") or "roll_" in key:
                seed=f"{v['seed_se']:.4f}" if v["seed_se"] is not None else "unavailable"
                lines.append(f"| {key} | {v['mean']:+.4f} | {v['paired_item_se']:.4f} | {seed} | {v['n_seeds']} |")
        lines.append("")
    (RUNS/"v41_compare.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
