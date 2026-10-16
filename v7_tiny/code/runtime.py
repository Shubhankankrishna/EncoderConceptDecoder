"""Explicit reductions, coherent schedules, ownership, and atomic recovery."""
import hashlib,json,math,os,random,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.nn import functional as F
from model import Config,System,Stack,Adapters,chunks,ball,freeze
from data import ROOT,DATA,T,K

DEVICE=os.environ.get('V7_DEVICE') or ('mps' if torch.backends.mps.is_available() else 'cpu')
# Metal admits only 31 buffer bindings. Reserve room for compiler temporaries.
if DEVICE=='mps':
    import torch._inductor.config as inductor_config
    inductor_config.max_fusion_unique_io_buffers=20
    inductor_config.max_fusion_size=16
    inductor_config.combo_kernels=False
    import torch._functorch.config as functorch_config
    functorch_config.donated_buffer=False
def sync(device=DEVICE):
    if str(device).startswith('mps'):torch.mps.synchronize()
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    with open(temp,'w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def atomic_save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    with open(temp,'wb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
class Log:
    def __init__(self,path):self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
    def __call__(self,**obj):
        obj['utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        with open(self.path,'a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush()
def cpu_tree(obj):
    if isinstance(obj,torch.Tensor):return obj.detach().cpu()
    if isinstance(obj,dict):return {k:cpu_tree(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return type(obj)(cpu_tree(v) for v in obj)
    return obj
def finite(obj):
    if isinstance(obj,torch.Tensor):return not obj.is_floating_point() or bool(torch.isfinite(obj).all())
    if isinstance(obj,dict):return all(finite(v) for v in obj.values())
    if isinstance(obj,(list,tuple)):return all(finite(v) for v in obj)
    return True
def implementation_hash():
    h=hashlib.sha256()
    for p in sorted((ROOT/'code').glob('*.py')):h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()
def tensor_hash(m):
    h=hashlib.sha256()
    for name,t in sorted(m.state_dict().items()):h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def save_checkpoint(path,s,opt,step,mix,g,config,extra=None):
    sync(next(s.parameters()).device)
    j=cpu_tree(dict(cfg=asdict(s.cfg),foundation=s.foundation.state_dict(),core=s.core.state_dict() if s.core else None,
        adapters=s.adapters.state_dict() if s.adapters else None,optimizer=opt.state_dict() if opt else None,step=step,
        mixer=mix.state() if mix else None,generator=g.get_state() if g else None,torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(),python_rng=random.getstate(),config=config,extra=extra or {}))
    if not all(finite(j[k]) for k in ('foundation','core','adapters','optimizer')):raise FloatingPointError('Nonfinite checkpoint rejected; previous checkpoint retained')
    atomic_save(path,j)
def load_system(path,device=DEVICE):
    ck=torch.load(path,map_location='cpu',weights_only=False);cfg=Config(**ck['cfg']);s=System(cfg)
    s.foundation.load_state_dict(ck['foundation'])
    if ck['core'] is not None:s.core=Stack(cfg,cfg.lc,True);s.core.load_state_dict(ck['core'])
    if ck['adapters'] is not None:s.adapters=Adapters(cfg);s.adapters.load_state_dict(ck['adapters'])
    return s.to(device),ck
def optimizer(s,lr):
    groups={0:[],.1:[]}
    for name,p in s.named_parameters():
        if not p.requires_grad:continue
        decay=p.ndim>=2 and not any(x in name for x in ('emb.','router.','.Q','adapters.'))
        groups[.1 if decay else 0].append(p)
    return torch.optim.AdamW([dict(params=v,weight_decay=k) for k,v in groups.items() if v],lr=lr,betas=(.9,.95))
def lr_at(progress,peak,warm=.01):
    if progress<warm:return peak*max(progress,1e-6)/warm
    v=min(1.,(progress-warm)/(1-warm));return peak*(.1+.9*.5*(1+math.cos(math.pi*v)))
def alpha_at(progress,maximum=.1,ramp=True):return maximum*(max(0.,min(1.,(progress-.05)/.15)) if ramp else 1.)
def nll(logits,x,m):
    t=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),x[:,1:].reshape(-1),reduction='none').view(len(x),-1)
    return t*m[:,:-1]
def balanced(values,weights,domain,den=None):
    nums=torch.stack([(values*(domain==i).reshape(-1,*([1]*(values.ndim-1)))).sum() for i in (0,1)])
    if den is None:den=torch.stack([(weights*(domain==i).reshape(-1,*([1]*(weights.ndim-1)))).sum() for i in (0,1)])
    means=nums/den.clamp_min(1);return means.mean(),means,den
def ncp_weights(batch):
    cd,valid=chunks(batch['docs']);cm=(valid[:,:-1]&valid[:,1:]&(cd[:,:-1]==cd[:,1:])).float()
    scored=batch['m'][:,K-1:K-1+K*(cd.shape[1]-1)].view(len(cd),-1,K).amax(-1)>0
    return cm*scored
def controls(batch,g,window=None):
    b,t=batch['x'].shape;m=t//K;device=batch['x'].device;dz=batch.get('dz',48)
    lengths=torch.tensor([1,2,4])[torch.randint(3,(b,),generator=g)] if window is None else torch.full((b,),window)
    valid_tokens=(batch['docs'].cpu()>=0).sum(1);prompt=(valid_tokens//2).clamp_min(1)
    for i,meta in enumerate(batch['metas']):prompt[b//2+i]=meta['prompt_len']
    boundary=(prompt//K-1).clamp_min(0);j=torch.arange(m)[None].expand(b,m)
    cd,valid=chunks(batch['docs']);enabled=(j>=boundary[:,None])&valid.cpu()
    index=((j-boundary[:,None]).clamp_min(0)//lengths[:,None]).clamp_max(m-1).to(device)
    bank=ball(torch.randn(b,m,dz,generator=g)).to(device)
    z=torch.gather(bank,1,index[:,:,None].expand(b,m,dz))*enabled.to(device)[:,:,None]
    start=(K*boundary+K-1).to(device);mask=batch['m']*(torch.arange(t,device=device)[None]>=start[:,None])
    return z,mask,index,enabled.to(device),lengths
def repeated(context,k):
    def rep(x):
        if isinstance(x,list):return [rep(v) for v in x]
        return x.unsqueeze(0).expand(k,*x.shape).reshape(k*len(x),*x.shape[1:])
    return tuple(rep(v) for v in context)
def candidate_scores(s,batch,context,zs,mask,fn):
    k=len(zs);b=len(batch['x']);ctx=repeated(context,k)
    docs=batch['docs'].repeat(k,1);x=batch['x'].repeat(k,1);m=mask.repeat(k,1)
    logits=fn(ctx,docs,torch.cat(zs,0))[0]
    return nll(logits,x,m).sum(1).view(k,b)
def selection(scores,mask,domain,beta):
    counts=mask[:,:-1].sum(1);eligible=(counts>0).float()
    rows=((1-beta)*scores.min(0).values+beta*scores.mean(0))/counts.clamp_min(1)
    return balanced(rows*eligible,eligible,domain)[0]
def candidate_backward(s,batch,zs,mask,beta,fn,replay=False):
    with torch.no_grad():ctx=s.context(batch['x'],batch['docs'])
    if not replay:
        value=selection(candidate_scores(s,batch,ctx,zs,mask,fn),mask,batch['domain'],beta);value.backward();return float(value.detach())
    with torch.no_grad():scores=candidate_scores(s,batch,ctx,zs,mask,fn);winner=scores.argmin(0)
    counts=mask[:,:-1].sum(1);eligible=counts>0
    den=torch.stack([((batch['domain']==i)&eligible).sum() for i in (0,1)]).clamp_min(1)
    weights=.5*eligible/den[batch['domain']]/counts.clamp_min(1)
    for i,z in enumerate(zs):
        row=candidate_scores(s,batch,ctx,[z],mask,fn)[0]
        (row*((1-beta)*(winner==i).float()+beta/len(zs))*weights).sum().backward()
    return float(selection(scores,mask,batch['domain'],beta))
