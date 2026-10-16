"""Exploratory oracle controllability, not RL or deployable latent reasoning.

Frozen weights, real legal prefixes, zero offsets, 10 steps, three registered
learning rates. Compare C input, intermediate S1 and output u. Optimise the
probability of the expert optimal-action set; compare random projected offsets
with the oracle run's per-example step norms. All prompts run the full budget,
so both corrections and damage to initially correct predictions are measured.
"""
import json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import analyze_vs as A
import data_v4 as D
import model_v4 as M4
import pinpad as PP
import run_vs as R
from counterfactual_v41 import fixtures

def project(value,base):
    return value*base.norm(dim=-1,keepdim=True)/value.norm(dim=-1,keepdim=True).clamp_min(1e-8)

def batch(m,fx):
    T=-(-max(len(f["original"]) for f in fx)//6)*6
    ids=np.zeros((len(fx),T),np.int64);doc=np.zeros_like(ids)
    target=torch.zeros(len(fx),4,device="mps",dtype=torch.bool)
    for b,f in enumerate(fx):
        ids[b,:len(f["original"])]=f["original"]
        target[b,f["original_opt"]]=True
    x=tuple(v.to("mps") for v in D.to_batch(ids,doc,PP.Ident(),m.ecfg))
    with torch.no_grad():
        tok,C,_,_=m.enc(*x);cdoc=x[1][:,::6]
        S,states=m.core.run(C,cdoc);u=m.core.out_norm(S)
        mask=M4.chunk_masks(x[0],x[1],6)[1][...,None]
    pos=torch.tensor([len(f["original"])-1 for f in fx],device="mps")
    return dict(tok=tok,C=C,S1=states[0],u=u,cdoc=cdoc,doc=x[1],mask=mask,pos=pos,target=target)

def predict(m,b,site,value):
    if site=="C":
        S,_=m.core.run(value,b["cdoc"]);u=m.core.out_norm(S)
    elif site=="S1":
        _,mask=M4.build_masks(b["cdoc"],1);S=value
        for layer in m.core.layers:S=layer(b["C"],S,b["cdoc"],mask,m.core.shared)
        u=m.core.out_norm(S)
    else:u=value
    _,_,cp=m.core.head(u)
    lg,_=m.dec(b["tok"],u,cp,b["doc"],m.enc.embed.weight)
    return lg[torch.arange(len(b["pos"]),device="mps"),b["pos"],PP.ACT0:PP.ACT0+4]

def values(logits,target):
    p=logits.softmax(-1);a=logits.argmax(-1)
    return target.gather(1,a[:,None]).squeeze(1).detach().cpu().numpy(),(p*target).sum(-1).detach().cpu().numpy()

def run_one(m,b,site,lr,K=10):
    base=b[site].detach();delta=torch.zeros_like(base,requires_grad=True)
    opt=torch.optim.Adam([delta],lr=lr,betas=(.9,.999))
    with torch.no_grad():initial=values(predict(m,b,site,base),b["target"])
    step_norms=[];previous=base.clone()
    for _ in range(K):
        controlled=project(base+delta*b["mask"],base)
        lg=predict(m,b,site,controlled)
        loss=(torch.logsumexp(lg,-1)-torch.logsumexp(lg.masked_fill(~b["target"],float("-inf")),-1)).mean()
        opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            current=project(base+delta*b["mask"],base)
            step_norms.append((current-previous).flatten(1).norm(dim=-1).cpu())
            previous=current.clone()
    with torch.no_grad():oracle=values(predict(m,b,site,previous),b["target"])
    # Fixed per-batch/site seed, independent of model success.
    gen=torch.Generator().manual_seed(934+len(base)+{"C":0,"S1":1,"u":2}[site])
    current=base.clone();actual=[]
    with torch.no_grad():
        for norms in step_norms:
            noise=torch.randn(base.shape,generator=gen).to("mps")*b["mask"]
            noise=noise/noise.flatten(1).norm(dim=-1).clamp_min(1e-8)[:,None,None]
            new=project(current+noise*norms.to("mps")[:,None,None],base)
            actual.append((new-current).flatten(1).norm(dim=-1).cpu());current=new
        random_result=values(predict(m,b,site,current),b["target"])
    return dict(initial=initial,oracle=oracle,random=random_result,oracle_norms=torch.stack(step_norms).numpy(),random_norms=torch.stack(actual).numpy())

def metrics(initial,final,prob0,prob):
    wrong=~initial;right=initial
    return dict(accuracy=float(final.mean()),n_initial_wrong=int(wrong.sum()),
                fraction_wrong_fixed=float(final[wrong].mean()) if wrong.any() else None,
                n_initial_right=int(right.sum()),fraction_right_broken=float((~final[right]).mean()) if right.any() else None,
                optimal_probability_gain=float((prob-prob0).mean()))

def run(name):
    torch.set_num_threads(1)
    import speed_v3;speed_v3.enable_block_attention(60)
    m,args=A.load(name);m.to("mps").eval()
    for p in m.parameters():p.requires_grad_(False)
    fx=fixtures(128);result={"status":"oracle diagnostic, not RL", "n":len(fx),"steps":10,
                             "device":"mps","normalisation":"preserve per-concept baseline radius",
                             "random":"proposal scaled to oracle projected step norm, then projected; achieved norm reported","sites":{}}
    for site in ("C","S1","u"):
        for lr in (.001,.01,.03):
            outputs=[]
            for a in range(0,len(fx),16):outputs.append(run_one(m,batch(m,fx[a:a+16]),site,lr))
            collect=lambda mode,k:np.concatenate([r[mode][k] for r in outputs])
            initial=collect("initial",0);prob0=collect("initial",1)
            rec=dict(baseline_accuracy=float(initial.mean()))
            for mode in ("oracle","random"):rec[mode]=metrics(initial,collect(mode,0),prob0,collect(mode,1))
            rec["oracle_step_norm"]=float(np.mean([r["oracle_norms"].mean() for r in outputs]))
            rec["random_achieved_step_norm"]=float(np.mean([r["random_norms"].mean() for r in outputs]))
            result["sites"][f"{site}@{lr}"]=rec;print(site,lr,json.dumps(rec),flush=True)
    (Path(R.RUNS)/name/"control.json").write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__":run(sys.argv[1])
