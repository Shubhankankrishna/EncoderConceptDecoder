"""Change a relevant layout fact while keeping the entire action history valid.

At the first pair-second decision, swap the target odd colour's location with
another unvisited colour. Require disjoint optimal next-action sets. This tests
whether the prediction follows the changed fact, rather than merely the prefix.
No task/subgoal label is fed to the model.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import data_v4 as D
import pinpad as PP
import run_vs as R
import analyze_vs as A

def fixtures(n=128):
    result=[]
    for ep in PP.eval_episodes("heldout",1000,37):
        flags=PP.boundary_flags(ep)
        i=next(j for j,f in enumerate(flags) if f==2)
        target=ep["tgt"][i];cell=ep["tokens"][i-1]-PP.POS0
        layout=ep["layout"];old=list(layout["colors"])
        original=set(ep["opt"][i])
        for other in range(8):
            if other in (target,target-1):continue
            changed=old.copy();changed[target],changed[other]=changed[other],changed[target]
            opt,_=PP.optimal_actions(cell,changed[target],set(layout["walls"]),changed)
            if not opt or original.intersection(opt):continue
            seq=ep["tokens"][:i]
            new=PP.header_tokens(changed,set(layout["walls"]),layout["start"])+seq[27:]
            # Replay both histories; the first visited colour must remain the even partner.
            for colors,toks in ((old,seq),(changed,new)):
                c=layout["start"];visits=[]
                for j in range(27,len(toks),2):
                    nxt=PP.step(c,toks[j]-PP.ACT0,set(layout["walls"]))
                    assert toks[j+1]==PP.pos_tok(nxt)
                    if nxt!=c and nxt in colors:visits.append(colors.index(nxt))
                    c=nxt
                assert visits==[target-1] and c==cell,(visits,target)
            assert len(seq)==len(new) and len(seq)%2==1
            result.append(dict(original=seq,changed=new,original_opt=sorted(original),changed_opt=opt,
                               target=target,swapped_with=other,prompt_length=len(seq)))
            break
        if len(result)==n:break
    assert len(result)==n,(len(result),n)
    return result

@torch.no_grad()
def predictions(m,seqs,chunk=16,**kw):
    device=next(m.parameters()).device
    result=[]
    for a in range(0,len(seqs),chunk):
        batch=seqs[a:a+chunk];T=-(-max(map(len,batch))//6)*6
        ids=np.zeros((len(batch),T),np.int64);doc=np.zeros_like(ids)
        for b,s in enumerate(batch):ids[b,:len(s)]=s
        x=D.to_batch(ids,doc,PP.Ident(),m.ecfg)
        logits=R.fwd(m,*(v.to(device) for v in x),**kw)[0]
        for b,s in enumerate(batch):
            result.append(logits[b,len(s)-1,PP.ACT0:PP.ACT0+4].argmax().item())
    return np.array(result)

def run(name):
    torch.set_num_threads(1)
    import speed_v3;speed_v3.enable_block_attention(60)
    m,args=A.load(name);m.to("mps")
    fx=fixtures();records={};units={}
    sets={"intact":{}}
    if R.is_v4(m):sets.update(core_off=dict(core_off=True),plan_swap=dict(plan_swap=True))
    if R.is_v4(m) and hasattr(m.dec,"retrieval_index"):sets["R_local"]=dict(R_local=True)
    for mode,kw in sets.items():
        p=predictions(m,[f["original"] for f in fx],**kw)
        q=predictions(m,[f["changed"] for f in fx],**kw)
        good=np.array([x in f["original_opt"] for x,f in zip(p,fx)])
        changed_good=np.array([x in f["changed_opt"] for x,f in zip(q,fx)])
        units[mode+"/original_correct"]=good
        units[mode+"/changed_correct"]=changed_good
        units[mode+"/original_action"]=p
        units[mode+"/changed_action"]=q
        records[mode]=dict(n=len(fx),original_accuracy=float(good.mean()),changed_accuracy=float(changed_good.mean()),
                           prediction_changes=float((p!=q).mean()),n_original_correct=int(good.sum()),
                           changed_accuracy_given_original_correct=float(changed_good[good].mean()) if good.any() else None)
    out=Path(R.RUNS)/name
    units["plan_lag"] = np.array([len(f["original"]) % 6 for f in fx])
    np.savez(out/"counterfactual_units.npz", **units)
    (out/"counterfactual.json").write_text(json.dumps(records,indent=2)+"\n")
    print(json.dumps(records,indent=2))

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="fixtures":
        fx=fixtures();(Path(R.RUNS)/"counterfactual_fixtures.json").write_text(json.dumps(fx,indent=2)+"\n")
        print(f"validated {len(fx)} paired counterfactuals, disjoint optimal actions and legal identical history")
    else:run(sys.argv[1])
