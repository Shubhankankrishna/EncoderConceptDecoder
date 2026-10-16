"""Atomic versioned checkpoints and loss denominators; V6-compatible math."""
import hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import *
ROOT=Path(__file__).resolve().parents[1]
DEVICE=os.environ.get('V8_DEVICE') or ('mps' if torch.backends.mps.is_available() else 'cpu')
def sync(device=DEVICE):
    if str(device)=='mps':torch.mps.synchronize()
def check_deadline(margin=0):
    deadline=float(os.environ.get('V8_DEADLINE_EPOCH','0'))
    if deadline and time.time()+margin>=deadline:raise TimeoutError('Declared experiment wall-time budget reached')
def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with open(tmp,'w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def cpu_tree(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return type(x)(cpu_tree(v) for v in x)
    return x
def finite(x):
    if isinstance(x,torch.Tensor):return not x.is_floating_point() or bool(torch.isfinite(x).all())
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
    return True
def atomic_save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');obj=cpu_tree(obj)
    if not finite(obj):raise FloatingPointError('Nonfinite checkpoint rejected')
    with open(tmp,'wb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
class Log:
    def __init__(self,path):self.path=Path(path)
    def __call__(self,**obj):
        obj['utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        with self.path.open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush()
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def implementation_hash():
    h=hashlib.sha256()
    for p in sorted((ROOT/'code').glob('*.py')):h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()
def tensor_hash(module):
    h=hashlib.sha256()
    for k,v in sorted(module.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def save_checkpoint(path,s,opt,step,mix,g,config,extra=None):
    sync(next(s.parameters()).device);atomic_save(path,dict(metadata=s.metadata(),model=s.state_dict(),optimizer=opt.state_dict() if opt else None,step=step,mixer=mix.state() if mix else None,generator=g.get_state() if g else None,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),config=config,extra=extra or {}))
def load_system(path,device=DEVICE):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if 'metadata' not in ck:
        s=System(Foundation(depth=2));s.foundation.load_state_dict(ck['foundation'],strict=True)
        if ck.get('core') is not None:s.core=Core();s.core.load_state_dict(ck['core'])
        if ck.get('adapters') is not None:raise ValueError('Historical V6 adapters use different math: bridge loads only deterministic anchors')
        return s.to(device),ck
    m=ck['metadata'];s=System(Flat() if m['flat'] else Foundation(m['depth']))
    if m['core']:s.core=Core() if m['core']=='recurrent' else DenseCore(s.foundation)
    if m['adapter']:s.adapters=Adapters(m['adapter'])
    for i in range(m['phases']):s.add_phase(m['rho'][i],m['forces'][i])
    if m.get('gate_buffers') and not m['phases']:
        s.add_phase();
        for block in s.active_blocks():block.banks=nn.ModuleList()
    s.load_state_dict(ck['model'],strict=True);return s.to(device),ck

def optimizer(s,lr=.002,stage='A'):
    groups=[];ps=[(n,p) for n,p in s.named_parameters() if p.requires_grad]
    wd=0 if stage in ('Z','CL') else .1
    for dim in (True,False):
        group=[p for n,p in ps if (p.dim()>=2)==dim and not (stage in ('A','HARD','MEM','SIMPLE') and n=='foundation.Q')]
        if group:groups.append(dict(params=group,weight_decay=wd if dim else 0,lr_mult=1.))
    if stage in ('A','HARD','MEM','SIMPLE') and not s.is_flat:groups.append(dict(params=[s.foundation.Q],weight_decay=0.,lr_mult=10.))
    return torch.optim.AdamW(groups,lr=lr,betas=(.9,.95),eps=1e-8)
def lr_at(step,total,peak,warm):
    warm=min(warm,max(1,total//10))
    if step<warm:return peak*(step+1)/warm
    return peak*(.1+.9*.5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm))))
def nll(logits,x,m):return F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),x[:,1:].reshape(-1),reduction='none').reshape(x.shape[0],-1)*m[:,:-1]
def balanced(values,weights,domain):
    nums=torch.stack([(values*(domain==i).reshape(-1,*([1]*(values.ndim-1)))).sum() for i in (0,1)])
    dens=torch.stack([(weights*(domain==i).reshape(-1,*([1]*(weights.ndim-1)))).sum() for i in (0,1)])
    means=nums/dens.clamp_min(1);present=dens>0
    return (means*present).sum()/present.sum().clamp_min(1),means,dens
def ncp_weights(batch):
    cd,valid=chunk_info(batch['docs']);cm=(valid[:,:-1]&valid[:,1:]&(cd[:,:-1]==cd[:,1:])).float()
    start=K-1;scored=F.pad(batch['m'],(0,1))[:,start:start+K*(cd.shape[1]-1)].reshape(len(cd),-1,K).amax(-1)>0
    return cm*scored

def controls(batch,generator):
    b,t=batch['x'].shape;m=t//K;device=batch['x'].device;lengths=torch.tensor([1,2,4])[torch.randint(3,(b,),generator=generator)];cd=chunk_info(batch['docs'])[0].cpu();j=torch.arange(m)[None].expand(b,m)
    starts=torch.where(torch.cat([torch.ones(b,1,dtype=torch.bool),cd[:,1:]!=cd[:,:-1]],1),j,torch.zeros_like(j)).cummax(1).values
    boundary=torch.randint(1,max(2,m//2),(b,),generator=generator)
    for i,meta in enumerate(batch['metas']):
        if meta['task']!='text':boundary[i]=max(0,meta['prompt_len']//K-1)
    begin=torch.maximum(starts,boundary[:,None]);index=(starts+(j-begin).clamp_min(0)//lengths[:,None]).clamp_max(m-1)
    enabled=(j>=boundary[:,None])&(cd>=0);bank=ball(torch.randn(b,m,DZ,generator=generator)).to(device)
    z=torch.gather(bank,1,index.to(device)[...,None].expand(b,m,DZ))*enabled.to(device)[...,None]
    mask=batch['m']*(torch.arange(t,device=device)[None]>= (K*boundary.to(device)+K-1)[:,None])
    return z,mask,index.to(device),enabled.to(device),lengths
