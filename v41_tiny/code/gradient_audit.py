"""Separate action NTP, ordinary NTP and weighted concept-loss gradients."""
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

def run(name):
    torch.set_num_threads(1)
    import speed_v3;speed_v3.enable_block_attention(60)
    m,args=A.load(name);m.to("mps")
    eps=PP.eval_episodes("heldout",24,53)
    t,d,labels,_=R.pin_rows(eps,480);t,d=t[:4],d[:4]
    b=tuple(v.to("mps") for v in D.to_batch(t,d,PP.Ident(),m.ecfg))
    lg,aux=R.fwd(m,*b)
    tm=M4.target_mask(b[0],b[1]);targets=b[0][:,1:].masked_fill(~tm,0)
    nll=F.cross_entropy(lg[:,:-1].transpose(1,2),targets,reduction="none")
    act=tm&(torch.from_numpy(labels["kind"][:4,1:]).to("mps")==1)
    _,complete,pair=M4.chunk_masks(b[0],b[1],6)
    n0,n,_=m.core.assign(aux["c_hat"])
    coarse,fine=m.core.ncp_terms(aux["q"],aux["logits_c"],n0,n,pair)
    objectives=dict(ntp=(nll*tm).sum()/tm.sum(),action_ntp=(nll*act).sum()/act.sum(),
                    weighted_coarse=.05*coarse/pair.sum(),weighted_fine=.1*fine/pair.sum())
    groups={"encoder_S":[],"encoder_F":[],"encoder_shared":[],"pooler":[],"core":[],"R":[],"plan":[]}
    for name_,p in m.named_parameters():
        group=None
        if name_.startswith("enc.layers."):
            li=int(name_.split('.')[2]);group="encoder_S" if m.enc.layers[li].kind=="S" else "encoder_F"
        elif name_.startswith("enc.shared"):group="encoder_shared"
        elif name_.startswith("enc.pooler"):group="pooler"
        elif name_.startswith("core.") and name_ not in ("core.E0","core.e"):group="core"
        elif name_.startswith("dec.layers.2."):group="R"
        elif name_.startswith("dec.plan."):group="plan"
        if group:groups[group].append(p)
    params=[p for ps in groups.values() for p in ps]
    grads={}
    for obj,loss in objectives.items():
        g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        offset=0;grads[obj]={}
        for group,ps in groups.items():
            gg=g[offset:offset+len(ps)];offset+=len(ps)
            grads[obj][group]=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten().cpu() for p,v in zip(ps,gg)])
    report={"losses":{k:float(v) for k,v in objectives.items()},"groups":{}}
    for group,ps in groups.items():
        np_=sum(p.numel() for p in ps)
        norm={obj:float(grads[obj][group].norm()) for obj in objectives}
        a=grads["action_ntp"][group];c=grads["weighted_coarse"][group]+grads["weighted_fine"][group]
        report["groups"][group]=dict(parameters=np_,gradient_norms=norm,
                    action_gradient_rms=norm["action_ntp"]/math.sqrt(np_),
                    concept_to_action_norm=float(c.norm()/a.norm().clamp_min(1e-12)),
                    action_concept_cosine=float(torch.dot(a,c)/(a.norm()*c.norm()).clamp_min(1e-12)))
    (Path(R.RUNS)/name/"gradient_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":run(sys.argv[1])
