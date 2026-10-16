"""Standalone scientific plots from saved evaluations, including seed spread."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
RUNS=ROOT/"runs"
COLORS={"v4":"#8894a3","v41":"#2068b0","llm":"#dc872d","stack":"#2d9374","loopoff":"#9558a5"}

def curve(name,field):
    p=RUNS/name/"log.jsonl"
    if not p.exists():return None
    x,y=[],[]
    for line in p.read_text().splitlines():
        r=json.loads(line)
        if r.get("kind")!="eval":continue
        value=r
        for k in field.split("/"):
            if k not in value:break
            value=value[k]
        else:x.append(r["real_tokens"]/1e6);y.append(value)
    return np.array(x),np.array(y)

def draw(ax,names,field,label,color):
    cc=[curve(n,field) for n in names];cc=[c for c in cc if c is not None and len(c[0])]
    if not cc:return
    end=min(c[0][-1] for c in cc)
    start=max(c[0][0] for c in cc)
    grid=np.linspace(start,end,100)
    yy=np.stack([np.interp(grid,x,y) for x,y in cc])
    for x,y in cc:ax.plot(x,y,color=color,alpha=.22,lw=.9)
    ax.plot(grid,yy.mean(0),color=color,lw=2,label=f"{label} (n={len(cc)})")
    if len(cc)>1:ax.fill_between(grid,yy.min(0),yy.max(0),color=color,alpha=.12)

def main():
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    groups={"v4":[f"ref_vs_pin_v4_s{s}_6M" for s in range(3)],
            "v41":[f"v41_pin_s{s}_6M" for s in range(3)],
            "llm":[f"ref_vs_pin_llm_s{s}_6M" for s in range(3)],
            "stack":[f"v41_stack_pin_s{s}_6M" for s in range(3)],
            "loopoff":[f"v41_loopoff_pin_s{s}_6M" for s in range(3)]}
    labels=dict(v4="v4 / no R",v41="v4.1",llm="Stacked LLM",stack="v4.1 stack",loopoff="v4.1 F loop off")
    for key,names in groups.items():
        draw(axs[0,0],names,"heldout/act_opt_b2",labels[key],COLORS[key])
        draw(axs[0,1],names,"heldout/ntp_action",labels[key],COLORS[key])
    axs[0,0].set(title="Held-out pair-second action accuracy",ylabel="Optimal-action rate",ylim=(.3,1.))
    axs[0,1].set(title="Held-out action prediction loss",ylabel="Nats / action token",ylim=(.4,1.5))
    for key in ("v4","v41","llm"):
        names=[f"v41_text_s{s}_5M" if key=="v41" else f"ref_vs_text_{key}_s{s}" for s in range(2)]
        draw(axs[1,0],names,"val_ntp",labels[key],COLORS[key])
    axs[1,0].set(title="Text validation loss",ylabel="Nats / token")
    kinds=[("v4.1",1.212036,.392928,.020480), ("LLM",1.206864,0,0)]
    x=np.arange(2);base=np.zeros(2)
    for label,idx,color in (("Headline compute parameters",1,"#2068b0"),("Engram table",2,"#70a3c8"),("Codebooks",3,"#dc872d")):
        vals=np.array([r[idx] for r in kinds]);axs[1,1].bar(x,vals,bottom=base,label=label,color=color);base+=vals
    axs[1,1].set(xticks=x,xticklabels=[r[0] for r in kinds],ylabel="Millions of parameters",title="Tiny stored parameters, excluding equal embeddings")
    axs[1,1].legend(fontsize=8)
    for ax in axs.flat:
        ax.grid(alpha=.18,axis="y")
    for ax in axs.flat[:3]:ax.set_xlabel("Millions of real training tokens");ax.legend(fontsize=8)
    fig.suptitle("v4.1 tiny experiment — means and observed training-seed ranges\nPrior v4/LLM references shown; fresh LLM check reported separately",fontsize=13)
    fig.savefig(ROOT/"learning_curves.png",dpi=170)
    print(ROOT/"learning_curves.png")

if __name__=="__main__":main()
