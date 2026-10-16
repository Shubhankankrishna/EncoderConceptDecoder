"""Separate real and padding chunk overflow on four CPU batches; fixed checkpoint, no training."""
from pathlib import Path
import sys,json,math
import numpy as np
import torch
import torch.nn.functional as F
sys.dont_write_bytecode=True
torch.set_num_threads(1);torch.set_num_interop_threads(1)
A=Path(__file__).resolve().parent;ROOT=A.parent;sys.path.insert(0,str(A/'source_snapshot/code'))
from model import build_model
from data import Canon,make_batch
import concept_core_d6 as D6
ck=torch.load(ROOT/'run_v2_19M/runs/v2_48h/ckpt_step000267.pt',map_location='cpu',mmap=True,weights_only=False)
cursor=ck['cursor']
with torch.device('meta'):m,e,_=build_model()
m.load_state_dict(ck['model'],assign=True);del ck
m.eval()
for l in m.core.layers:l.train();l.cf=1.25
DATA=ROOT/'run_m2_25M/data';canon=Canon(DATA)
t=np.load(DATA/'train/tok.npy',mmap_mode='r');d=np.load(DATA/'train/doc.npy',mmap_mode='r')
order=np.random.RandomState(0).permutation(len(t))
orig=D6.moe_dispatch
out=[];records=[];real=None
def wrap(x,logits,pool,top_k=2,capacity_factor=1.25,dense=False):
    y,a=orig(x,logits,pool,top_k,capacity_factor,dense)
    ix=a['idx'];flat=ix.reshape(-1);oh=F.one_hot(flat,pool.E);pos=(oh.cumsum(0)*oh).sum(-1)-1
    C=math.ceil(capacity_factor*x.shape[0]*top_k/pool.E)
    dropped=(pos>=C).view(2,256,2)
    rows=[]
    for b in range(2):
        rr=real[b];rows.append(dict(real_chunks=int(rr.sum()),padding_chunks=int((~rr).sum()),
            all_drop=dropped[b].float().mean().item(),real_drop=dropped[b][rr].float().mean().item(),
            padding_drop=dropped[b][~rr].float().mean().item() if (~rr).any() else None,
            real_losing_both=dropped[b][rr].all(-1).float().mean().item()))
    records.append(rows)
    return y,a
D6.moe_dispatch=wrap
with torch.no_grad():
    for off in [0,32,64,96]:
        ix=order[cursor+off:cursor+off+2];ids,doc,ei=make_batch(np.asarray(t[ix]),np.asarray(d[ix]),canon,e)
        real=(ids.reshape(2,256,4)!=0).any(-1);records=[]
        _,c,_=m.enc(ids,doc,ei);m.core(c,doc[:,::4],10)
        out.append(dict(offset=off,rows=ix.tolist(),dispatch_records=records))
D6.moe_dispatch=orig
(A/'routing_replication.json').write_text(json.dumps(dict(step=267,K=10,capacity_factor=1.25,batches=out),indent=2))
for b in out:
    print(b['offset'],[{k:float(np.mean([r[i][k] for r in b['dispatch_records']])) for k in ['all_drop','real_drop','real_losing_both']} for i in [0,1]],flush=True)
