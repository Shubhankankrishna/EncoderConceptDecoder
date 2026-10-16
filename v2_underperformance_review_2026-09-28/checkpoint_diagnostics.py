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
sys.path.insert(0,str(A/'source_snapshot/code'))
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
def trace(ids,doc,ei,maxR=16,depths=(1,2,4,6,8,10,12,16),cf=None,interventions=False):
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

# Use two next unseen full-length training rows at this checkpoint's cursor for routing diagnostics.
toks=np.load(DATA/'train/tok.npy',mmap_mode='r');docs=np.load(DATA/'train/doc.npy',mmap_mode='r')
order=np.random.RandomState(0).permutation(len(toks));indices=order[res['cursor']:res['cursor']+2]
ids,doc,ei=make_batch(np.asarray(toks[indices]),np.asarray(docs[indices]),canon,ecfg)
res['train_sample_rows']=indices.tolist()
res['capacity_sweeps']={}
for name,cf in [('no_drop',None),('capacity_1.25',1.25),('capacity_1.5',1.5),('capacity_2',2.0)]:
    rr=trace(ids,doc,ei,maxR=12,depths=(1,2,4,6,8,10,12),cf=cf)
    res['capacity_sweeps'][name]=rr;save()
    note(f"capacity {name}: R6={np.mean(rr['depths']['6']['nats']):.4f}, R12={np.mean(rr['depths']['12']['nats']):.4f}, drops={np.mean(rr['drops']):.3f}")

# Instrument actual dispatch, without changing its output, for row/position drop distribution.
dispatch_orig=D6.moe_dispatch
dispatch_records=[]
def inspect_dispatch(x,logits,pool,top_k=2,capacity_factor=1.25,dense=False):
    y,aux=dispatch_orig(x,logits,pool,top_k,capacity_factor,dense)
    if capacity_factor is not None:
        N=x.shape[0];E=pool.E;idx=aux['idx'];flat=idx.reshape(-1)
        oh=F.one_hot(flat,E);pos=(oh.cumsum(0)*oh).sum(-1)-1
        C=max(1,min(N,int(math.ceil(capacity_factor*N*top_k/E))))
        keep=(pos<C).view(2,N//2,top_k)
        drop=(~keep).float()
        quarter=drop.reshape(2,4,N//8,top_k).mean((2,3))
        probs=logits.softmax(-1);g=probs.gather(1,idx);g=g/g.sum(-1,keepdim=True)
        surv=(g.view_as(keep)*keep).sum(-1)
        re=(ids.reshape(2,-1,4)!=0).any(-1)
        dispatch_records.append(dict(capacity=C,expert_counts=oh.sum(0).tolist(),
             row_drop=drop.mean((1,2)).tolist(),quarter_drop=quarter.tolist(),
             full_expert_loss_fraction=(~keep.any(-1)).float().mean(1).tolist(),
             survivor_gate_mass=surv.mean(1).tolist(),real_drop=drop[re].mean().item(),padding_drop=drop[~re].mean().item() if (~re).any() else None))
    return y,aux
D6.moe_dispatch=inspect_dispatch
try:
    trace(ids,doc,ei,maxR=10,depths=(10,),cf=1.25)
finally:D6.moe_dispatch=dispatch_orig
res['dispatch_by_pass_layer']=dispatch_records
save()

# Same two rows, reversed batch order: no-drop eval is invariant; capacity dispatch need not be.
@torch.no_grad()
def batch_logits(ids_,doc_,ei_,cf):
    m.eval()
    if cf is not None:
        for l in m.core.layers:l.train();l.cf=cf
    tok,c,_=m.enc(ids_,doc_,ei_)
    u,_,cp=m.core(c,doc_[:,::4],10)
    logits,_=m.dec(tok,u,cp,doc_,m.enc.embed.weight)
    return logits
res['batch_order_test']={}
for name,cf in [('no_drop',None),('capacity_1.25',1.25)]:
    l1=batch_logits(ids,doc,ei,cf);l2=batch_logits(ids.flip(0),doc.flip(0),ei.flip(0),cf).flip(0)
    ss,nn,nll1=masked_ce(l1,ids,doc);ss2,_,nll2=masked_ce(l2,ids,doc)
    res['batch_order_test'][name]=dict(max_logit_difference=(l1-l2).abs().max().item(),mean_logit_difference=(l1-l2).abs().mean().item(),
                                    original_row_nats=(ss/nn).tolist(),reversed_row_nats=(ss2/nn).tolist())
    del l1,l2,nll1,nll2
save();gc.collect()
note('routing distribution and batch-order checks complete')

# Carry ablations and sublayer interventions: fixed checkpoint, fixed next-unseen rows, no drops.
@torch.no_grad()
def intervention(mode):
    m.eval();core=m.core
    tok,c,_=m.enc(ids,doc,ei);X,S=core.init_state(c);S0=S.clone();mask=core.mask(doc[:,::4]);cache={}
    handles=[]
    if mode=='attention_off':
        handles=[l.attn.register_forward_hook(lambda mod,inp,out:torch.zeros_like(out)) for l in core.layers]
    if mode=='experts_off':
        D6.moe_dispatch=lambda *a,**kw: (lambda ya:(torch.zeros_like(ya[0]),ya[1]))(dispatch_orig(*a,**kw))
    try:
        for r in range(1,11):
            if r==5 and mode=='reset_after4':S=S0.clone()
            if r==5 and mode=='shuffle_after4':S=S.flip(0)
            S,_=core.one_pass(X,S,mask,first=r==1,cache=cache)
        u=core.readout(S);_,cp=core.head(u)
        return decode(tok,u,cp,doc,ids)
    finally:
        for h in handles:h.remove()
        D6.moe_dispatch=dispatch_orig
res['carry_sublayer_interventions']={}
for mode in ['intact','reset_after4','shuffle_after4','attention_off','experts_off']:
    res['carry_sublayer_interventions'][mode]=intervention(mode);save()
note('carry/sublayer interventions complete')

if not args.skip_grad:
    # Gradient comparison at shared feature leaves and selected core/decoder weights.
    # The graph mirrors the actual detach boundaries; encoder parameters remain frozen to avoid full table gradients.
    m.eval()
    for l in m.core.layers:l.train();l.cf=1.25
    core=m.core
    with torch.no_grad():tok,c,_=m.enc(ids,doc,ei)
    c_leaf=c.detach().requires_grad_();t_leaf=tok.detach().requires_grad_()
    selected={'concept_features':c_leaf,'local_features':t_leaf,'core_W_in':core.W_in.weight,'expert_w2':core.pool.w2,
              'decoder_Wu':m.dec.plan.Wu.weight,'carry_gain':core.carry_norm.gamma}
    for q in selected.values():q.requires_grad_(True)
    targets=core.vocab.assign(c.detach());cdoc=doc[:,::4];mask=core.mask(cdoc)
    X,S=core.init_state(c_leaf);ncp=0.;moe=0.;us=[];K=10
    for r in range(1,K+1):
        if r==5:S=S.detach();X=core.W_in(c_leaf)
        S,aux=core.one_pass(X,S,mask,first=r==1)
        u=core.readout(S);lc,cp=core.head(u,pred=r==K)
        ncp=ncp+.1*core.ncp_ce(lc,targets,cdoc)/K
        moe=moe+(.01*torch.stack([x['lb'] for x in aux]).mean()+.001*torch.stack([x['z'] for x in aux]).mean())/K
        if r>=5:us.append(u)
    logits,daux=m.dec(t_leaf,us[-1],cp,doc,m.enc.embed.weight)
    sums,counts,_=masked_ce(logits,ids,doc);ntp=sums.sum()/counts.sum()
    lh,lkl=nextlat_losses(m,daux['h'],logits,ids,doc)
    cons=.1*torch.stack([(us[-1].detach()-u_).pow(2).mean() for u_ in us[:-1]]).mean()
    terms={'NTP':ntp,'NCP_weighted':ncp,'NextLat_h':lh,'NextLat_KL':lkl,'consistency_weighted':cons,'MoE_aux_weighted':moe}
    grads={};names=list(selected);params=list(selected.values())
    res['gradient_terms']={}
    for name,loss in terms.items():
        gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        grads[name]={k:g.detach().clone() if g is not None else None for k,g in zip(names,gs)}
        res['gradient_terms'][name]=dict(loss=loss.item(),norms={k:g.norm().item() if g is not None else 0.0 for k,g in zip(names,gs)})
    res['gradient_cosine_vs_ntp']={}
    for name,gg in grads.items():
        if name=='NTP':continue
        row={}
        for k,g in gg.items():
            g0=grads['NTP'][k]
            row[k]=float((g*g0).sum()/(g.norm()*g0.norm()).clamp_min(1e-30)) if g is not None and g0 is not None else None
        res['gradient_cosine_vs_ntp'][name]=row
    # Report alignment for the sum of auxiliaries too, rather than extrapolating from one term.
    agg={}
    for k,g0 in grads['NTP'].items():
        if g0 is None:continue
        ga=sum((g[k] for name,g in grads.items() if name!='NTP' and g[k] is not None),torch.zeros_like(g0))
        agg[k]=dict(ntp_norm=g0.norm().item(),aux_norm=ga.norm().item(),cosine=float((ga*g0).sum()/(ga.norm()*g0.norm()).clamp_min(1e-30)))
    res['aggregate_aux_gradients']=agg
    pair_same=doc[:,1:]==doc[:,:-1]
    res['nextlat_padding_pair_fraction']=float((pair_same & ((ids[:,:-1]==0)|(ids[:,1:]==0))).sum()/pair_same.sum())
    save();note('gradient conflict check complete')
note(f'finished {O}')
