"""Read-only CPU checkpoint diagnostics. No MPS, optimizer, or writes in either run.

Uses one CPU thread and memory-mapped checkpoint tensors assigned into a meta model.
Diagnostics are interventions on fixed weights, not controlled retraining ablations.
"""
from pathlib import Path
import argparse,datetime,gc,json,math,os,sys,time
import numpy as np
import torch
import torch.nn.functional as F
sys.dont_write_bytecode=True
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
A=Path(__file__).resolve().parent;ROOT=A.parent
sys.path.insert(0,str(A/'source_snapshot/run_v2_19M/code'))
from model import build_model,nextlat_losses
from data import Canon,make_batch,target_mask
import concept_core_d6 as D6

ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--rows',type=int,default=8)
ap.add_argument('--skip-grad',action='store_true'); args=ap.parse_args()
p=Path(args.checkpoint).resolve();stat=p.stat()
ck=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
step=ck['step']
O=A/f'checkpoint_{step:06d}_diagnostics.json'
res=dict(time=datetime.datetime.now().astimezone().isoformat(),checkpoint=str(p),checkpoint_inode=stat.st_ino,
         step=step,elapsed_h=ck['elapsed']/3600,cursor=ck['cursor'],device='cpu',threads=1,torch_version=torch.__version__,
         checkpoint_args=ck['args'],sample_seed=210928,rows_per_language=args.rows)
with torch.device('meta'):m,ecfg,dcfg=build_model()
m.load_state_dict(ck['model'],assign=True);m.eval()
for q in m.parameters():q.requires_grad_(False)
assert all(layer.pool[0] is m.core.pool for layer in m.core.layers)
assert all(q.device.type=='cpu' for q in m.parameters())
del ck;gc.collect()
DATA=ROOT/'run_m2_25M/data';canon=Canon(DATA)
def save():O.write_text(json.dumps(res,indent=2,allow_nan=False))
def note(s):print(datetime.datetime.now().strftime('%H:%M:%S'),s,flush=True)
note(f'loaded step {step} via mmap; CPU only')
save()

def masked_ce(logits,ids,doc):
    mask=target_mask(ids,doc)
    nll=F.cross_entropy(logits[:,:-1].reshape(-1,8192),ids[:,1:].reshape(-1),reduction='none').view_as(mask)
    return (nll*mask).sum(1),(mask.sum(1)),nll

def decode(tok,u,cp,doc,ids,**kw):
    logits,_=m.dec(tok,u,cp,doc,m.enc.embed.weight,**kw)
    sums,counts,_=masked_ce(logits,ids,doc)
    return dict(sums=sums.tolist(),counts=counts.tolist(),nats=(sums/counts).tolist())

@torch.no_grad()
def trace(ids,doc,ei,maxR=12,depths=(1,2,4,6,8,12),cf=None,interventions=False):
    core=m.core
    m.eval()
    if cf is not None:
        for l in core.layers:l.train();l.cf=cf
    tok,c,eaux=m.enc(ids,doc,ei);cdoc=doc[:,::4];mask=core.mask(cdoc)
    targets=core.vocab.assign(c)
    X,S=core.init_state(c);cache={};readouts=[];rout=[]
    out=dict(depths={},state=[],ncp=[],drops=[],drop_by_layer=[],routing_changes=[],concept={})
    real=(ids.reshape(len(ids),-1,4)!=0).any(-1)
    cr=c[real];mu=cr.mean(0)
    q=core.vocab.e[torch.arange(core.vocab.S),core.vocab.assign(cr)].flatten(-2)
    variance=(cr-mu).pow(2).sum();sse=(cr-q).pow(2).sum()
    out['concept']=dict(mean_norm=mu.norm().item(),mean_norm_fraction_sq=mu.pow(2).sum().item()/384,
                        var_per_concept=(variance/len(cr)).item(),vq_variance_explained=(1-sse/variance).item(),
                        crc_mix=eaux['mix'].mean((0,1)).tolist(),n_real=len(cr))
    for r in range(1,maxR+1):
        prev=S
        S,aux=core.one_pass(X,S,mask,first=r==1,cache=cache)
        u=core.readout(S);lc,cp=core.head(u,pred=r in depths)
        out['ncp'].append(core.ncp_ce(lc,targets,cdoc).item())
        out['drops'].append(torch.stack([a['dropped'] for a in aux]).mean().item())
        out['drop_by_layer'].append([a['dropped'].item() for a in aux])
        idx=torch.stack([a['idx'][:,0] for a in aux]);rout.append(idx)
        if len(rout)>1:out['routing_changes'].append((rout[-1]!=rout[-2]).float().mean().item())
        out['state'].append(dict(norm=S.norm(dim=-1).mean().item(),update_norm=(S-prev).norm(dim=-1).mean().item(),
                                 cosine_to_first=F.cosine_similarity(u,readouts[0] if readouts else u,dim=-1).mean().item()))
        readouts.append(u)
        if r in depths:
            out['depths'][str(r)]=decode(tok,u,cp,doc,ids)
            if interventions and r==6:
                for name,kw in [('core_off',{'core_off':True}),('u_off',{'use_u':False}),('c_off',{'use_c':False})]:
                    out['depths'][name]=decode(tok,u,cp,doc,ids,**kw)
                out['depths']['plan_swap']=decode(tok,u.flip(0),cp.flip(0),doc,ids)
    m.eval()
    return out

res['language_batches']=[]
for name in ['tinystories','wikitext']:
    toks=np.load(DATA/'lang_eval'/f'val_{name}_tok.npy',mmap_mode='r')
    docs=np.load(DATA/'lang_eval'/f'val_{name}_doc.npy',mmap_mode='r')
    ix=np.sort(np.random.RandomState(210928).choice(128,args.rows,replace=False))
    for j in range(0,len(ix),2):
        idx=ix[j:j+2];ids,doc,ei=make_batch(np.asarray(toks[idx]),np.asarray(docs[idx]),canon,ecfg)
        result=trace(ids,doc,ei,interventions=True)
        result.update(dataset=name,rows=idx.tolist())
        res['language_batches'].append(result);save()
        note(f"{name} rows {idx.tolist()} R6 {np.mean(result['depths']['6']['nats']):.4f}; R1 {np.mean(result['depths']['1']['nats']):.4f}; R12 {np.mean(result['depths']['12']['nats']):.4f}")

res['language_summary']={}
for name in ['tinystories','wikitext']:
    batches=[x for x in res['language_batches'] if x['dataset']==name]
    res['language_summary'][name]={}
    for dep in batches[0]['depths']:
        ss=sum(sum(x['depths'][dep]['sums']) for x in batches)
        nn=sum(sum(x['depths'][dep]['counts']) for x in batches)
        res['language_summary'][name][dep]=ss/nn
save()


print('COMPLETE', json.dumps(res['language_summary']), flush=True)
