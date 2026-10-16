"""Replicate the auxiliary-gradient diagnostic across four independent B2 training batches, CPU only."""
from pathlib import Path
import datetime,json,math,sys,gc
import numpy as np
import torch
import torch.nn.functional as F
sys.dont_write_bytecode=True
torch.set_num_threads(1);torch.set_num_interop_threads(1)
A=Path(__file__).resolve().parent;ROOT=A.parent
sys.path.insert(0,str(A/'source_snapshot/code'))
from model import build_model,nextlat_losses
from data import Canon,make_batch,target_mask
P=ROOT/'run_v2_19M/runs/v2_48h/ckpt_step000267.pt'
ck=torch.load(P,map_location='cpu',mmap=True,weights_only=False)
cursor=ck['cursor'];step=ck['step']
with torch.device('meta'):m,ecfg,_=build_model()
m.load_state_dict(ck['model'],assign=True);del ck
for p in m.parameters():p.requires_grad_(False)
m.train()
DATA=ROOT/'run_m2_25M/data';canon=Canon(DATA)
toks=np.load(DATA/'train/tok.npy',mmap_mode='r');docs=np.load(DATA/'train/doc.npy',mmap_mode='r')
order=np.random.RandomState(0).permutation(len(toks))
out=dict(step=step,time=datetime.datetime.now().astimezone().isoformat(),device='cpu',K=10,batches=[])

def one(ids,doc,ei):
    core=m.core
    with torch.no_grad():tok,c,_=m.enc(ids,doc,ei)
    cl=c.detach().requires_grad_();tl=tok.detach().requires_grad_()
    selected={'concept_features':cl,'local_features':tl,'core_W_in':core.W_in.weight,'core_W_s':core.W_s.weight,'expert_w2':core.pool.w2,
              'decoder_Wu':m.dec.plan.Wu.weight,'carry_gain':core.carry_norm.gamma,'router_layer2':core.layers[1].router.weight}
    for p in selected.values():p.requires_grad_(True)
    targets=core.vocab.assign(c);cdoc=doc[:,::4];mask=core.mask(cdoc)
    X,S=core.init_state(cl);ncp=0.;lbz=0.;us=[]
    for r in range(1,11):
        if r==5:S=S.detach();X=core.W_in(cl)
        S,aux=core.one_pass(X,S,mask,first=r==1)
        u=core.readout(S);lc,cp=core.head(u,pred=r==10)
        ncp=ncp+.01*core.ncp_ce(lc,targets,cdoc)
        lbz=lbz+(.01*torch.stack([x['lb'] for x in aux]).mean()+.001*torch.stack([x['z'] for x in aux]).mean())/10
        if r>=5:us.append(u)
    logits,da=m.dec(tl,us[-1],cp,doc,m.enc.embed.weight)
    target=ids[:,1:].masked_fill(~target_mask(ids,doc),-100)
    ntp=F.cross_entropy(logits[:,:-1].reshape(-1,8192),target.reshape(-1),ignore_index=-100)
    lh,lkl=nextlat_losses(m,da['h'],logits,ids,doc)
    cons=.1*torch.stack([(us[-1].detach()-v).pow(2).mean() for v in us[:-1]]).mean()
    losses={'NTP':ntp,'NCP_weighted':ncp,'NextLat_h':lh,'NextLat_KL':lkl,'consistency_weighted':cons,'MoE_aux_weighted':lbz}
    grads={};r={'losses':{k:v.item() for k,v in losses.items()},'per_term':{},'aggregate':{}}
    for k,loss in losses.items():
        gs=torch.autograd.grad(loss,list(selected.values()),allow_unused=True,retain_graph=True)
        grads[k]={n:g.detach() if g is not None else None for n,g in zip(selected,gs)}
    r['initial_projection_gradients']={k: None if gs['core_W_s'] is None else gs['core_W_s'].norm().item() for k,gs in grads.items()}
    for k,gs in grads.items():
        r['per_term'][k]={}
        for n,g in gs.items():
            g0=grads['NTP'][n]
            if g is None or g0 is None:r['per_term'][k][n]=None;continue
            norm=g.norm();norm0=g0.norm();dot=(g*g0).sum()
            r['per_term'][k][n]=dict(norm=norm.item(),norm_ratio=(norm/norm0.clamp_min(1e-30)).item(),
                                    cosine=(dot/(norm*norm0).clamp_min(1e-30)).item(),ntp_projection_ratio=(dot/norm0.pow(2).clamp_min(1e-30)).item())
    for n,g0 in grads['NTP'].items():
        if g0 is None:continue
        for group,keys in [('all_aux',[k for k in grads if k!='NTP']),('nextlat',['NextLat_h','NextLat_KL'])]:
            ga=sum((grads[k][n] for k in keys if grads[k][n] is not None),torch.zeros_like(g0))
            z=ga.norm();z0=g0.norm();dot=(ga*g0).sum()
            r['aggregate'].setdefault(group,{})[n]=dict(norm_ratio=(z/z0).item(),cosine=(dot/(z*z0).clamp_min(1e-30)).item(),
                                                       ntp_projection_ratio=(dot/z0.pow(2).clamp_min(1e-30)).item())
    for p in m.parameters():p.requires_grad_(False)
    return r

for off in [0,32,64,96]:
    ix=order[cursor+off:cursor+off+2]
    ids,doc,ei=make_batch(np.asarray(toks[ix]),np.asarray(docs[ix]),canon,ecfg)
    r=one(ids,doc,ei);r.update(offset=off,rows=ix.tolist());out['batches'].append(r)
    (A/'gradient_replication.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    print('batch',off,'NextLat cosine/core_W_in',r['aggregate']['nextlat']['core_W_in']['cosine'],
          'projection',r['aggregate']['nextlat']['core_W_in']['ntp_projection_ratio'],flush=True)
    gc.collect()

# Analytical counterexample to treating Switch LB near 1 as a hard-routing balance certificate.
lg=torch.zeros(512,16);lg[:,0]=1e-6;lg[:,1]=.5e-6
pr=lg.softmax(-1);idx=pr.topk(2,dim=-1).indices
counts=F.one_hot(idx,16).sum((0,1));frac=counts.float()/counts.sum()
lb=16*(frac*pr.mean(0)).sum();C=80
out['lb_counterexample']=dict(load_balance_loss=lb.item(),assignment_counts=counts.tolist(),capacity=C,
                            fraction_dropped=(counts-C).clamp_min(0).sum().item()/counts.sum().item(),
                            note='Synthetic near-uniform probabilities; hard top-2 selects the same two experts. Not an observed live batch.')
(A/'gradient_replication.json').write_text(json.dumps(out,indent=2,allow_nan=False))
print('complete',flush=True)
