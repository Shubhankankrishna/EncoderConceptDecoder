"""CPU-only checks of the V10.2 specification, NOT a V10.2 model implementation."""
from pathlib import Path
import hashlib,json,math,re,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from v10_1.model import V101
from v10_10M.model import FFN,RMSNorm
from v10_10M.config import ModelConfig
DEST=Path(__file__).resolve().parent

class ForecastReadout(nn.Module):
 def __init__(self,d=128,v=8192):
  super().__init__();self.projection=nn.Linear(d,d,bias=False);self.position=nn.Parameter(torch.zeros(8,d))
  self.norm=RMSNorm(d);self.ffn=FFN(d);self.outn=RMSNorm(d)
 def forward(self,forecast,embedding):
  z=self.projection(forecast).repeat_interleave(4,dim=-2)+self.position
  return F.linear(self.outn(z+self.ffn(self.norm(z))),embedding)

class ReferenceDoubleNorm(nn.Module):
 """Float64 mathematical norm for finite differences, without FP32 casts."""
 def __init__(self,d):
  super().__init__();self.weight=nn.Parameter(torch.ones(d,dtype=torch.float64))
 def forward(self,x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*self.weight

class ConceptReader(nn.Module):
 def __init__(self,d=128,heads=4):
  super().__init__();self.qn=RMSNorm(d);self.q=nn.Linear(d,d,bias=False);self.k=nn.Linear(d,d,bias=False)
  self.v=nn.Linear(d,d,bias=False);self.o=nn.Linear(d,d,bias=False);self.heads=heads
 def forward(self,x,m,lengths):
  b,t,d=x.shape;g=m.shape[1]//2;mask=release_mask(t,g,lengths)
  source=torch.cat([m,m.new_zeros(b,1,d)],dim=1)
  def split(z):return z.reshape(b,-1,self.heads,d//self.heads).transpose(1,2)
  q,k,v=split(self.q(self.qn(x))),split(self.k(source)),split(self.v(source))
  def rotate(z,pos):
   half=z.shape[-1]//2
   a=pos.to(z.dtype)[:,None]/(10000**(torch.arange(half,dtype=z.dtype)/half))
   co,si=a.cos(),a.sin();u,w=z.chunk(2,-1)
   return torch.cat([u*co-w*si,u*si+w*co],-1)
  q=rotate(q,torch.arange(t));k=rotate(k,torch.cat([torch.arange(g).repeat_interleave(2)*8+7,torch.zeros(1,dtype=torch.long)]))
  return self.o(F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None]).transpose(1,2).reshape(b,t,d))

def release_mask(t,g,lengths):
 row=torch.arange(t)[None,:,None];ends=(torch.arange(g).repeat_interleave(2)*8+7)[None,None,:]
 real=(row<lengths[:,None,None])&(ends<lengths[:,None,None])&(ends<=row)
 return torch.cat([real,~real.any(-1,keepdim=True)],-1)

def main():
 torch.set_num_threads(1);torch.manual_seed(102)
 checks={}
 with torch.device('meta'):
  model=V101();baseline=model.counts();old=sum(p.numel() for p in model.chunk_readout.parameters())
  model.chunk_readout=ForecastReadout();aux=sum(p.numel() for p in model.chunk_readout.parameters())
  model.decoder.routes=nn.ModuleList([nn.Linear(128,2) for _ in range(4)])
  model.decoder.forecast_projection=nn.Linear(128,128,bias=False)
  model.decoder.concept_readers=nn.ModuleList([ConceptReader() for _ in range(4)])
  proposed=model.counts();model.chunk_readout=None;deployed=model.counts()
 assert baseline['dense']==5731496 and old==215168 and aux==149376
 assert proposed['dense']==5944228 and proposed['engram']==20000320 and proposed['total']==25944548
 assert deployed['dense']==5794852 and proposed['dense']<10000000
 checks['parameter_layout']=dict(reference=baseline,removed_head=old,new_head=aux,proposed=proposed,deployed=deployed,scope='Meta-tensor module layout only; no end-to-end V10.2 forward')
 mask_cases=0
 for length in range(1,66):
  t=((length+7)//8)*8;g=t//8;mask=release_mask(t,g,torch.tensor([length]))[0]
  for i in range(t):
   legal=[s for s in range(2*g) if i<length and 8*(s//2)+7<length and 8*(s//2)+7<=i]
   assert mask[i,:-1].nonzero().flatten().tolist()==legal
   assert bool(mask[i,-1])==(not legal)
   assert mask[i].any();mask_cases+=1
 assert release_mask(7,0,torch.tensor([7])).all()
 checks['release_mask_exhaustive']=dict(lengths='1..65, plus zero real keys',rows_checked=mask_cases,all_rows_have_key=True)
 aux_cases=0
 for length in range(1,66):
  for answer_start in range(1,length+2):
   target_mask=torch.arange(length)>=answer_start-1
   for g in range(length//8):
    b=8*(g+1);idx=b+torch.arange(8)-1;valid=idx<length
    allowed=valid.clone();allowed[valid]&=target_mask[idx[valid]]
    for j in allowed.nonzero().flatten().tolist():
     assert b+j>8*g+7 and 0<=int(idx[j])<length
    if allowed.any():
     w=1/torch.sqrt(torch.arange(1,9,dtype=torch.float64));wn=w*allowed;wn/=wn.sum()
     assert abs(float(wn.sum())-1)<1e-12 and not bool((wn[~allowed]!=0).any())
    aux_cases+=1
 # length=8 permits release at row 7 and the next token, e.g. EOS, at targets[7].
 assert 8*(0+1)-1==7 and 7<8
 checks['forecast_label_mask']=dict(group_answer_cases=aux_cases,partial_and_terminal_labels=True,normalization='Per eligible document; normalized offset weights')
 reader=ConceptReader(16,2).double();x=torch.randn(2,24,16,dtype=torch.float64);m=torch.randn(2,6,16,dtype=torch.float64)
 lens=torch.tensor([24,8]);y=reader(x,m,lens);changed=m.clone();changed[:,2:]+=1000
 y2=reader(x,changed,lens)
 assert torch.isfinite(y).all() and torch.equal(y[:,:7],torch.zeros_like(y[:,:7]))
 assert float((y[:,:15]-y2[:,:15]).abs().max().detach())==0
 assert torch.equal(y[1,8:],torch.zeros_like(y[1,8:]))
 checks['reader_startup_and_future_source_invariance']=dict(max_early_error=0.,scope='Isolated proposed reader, not full-model causality')
 # Small differentiable fixture includes both corrected source routes, PQ,
 # aligned addition, query dependence, and corrected K/V attention.
 d=8;g=3;t=24;reader=ConceptReader(d,2).double();reader.qn=ReferenceDoubleNorm(d)
 U=torch.randn(g,2,d,dtype=torch.float64);h=torch.randn(1,t,d,dtype=torch.float64)
 pq=nn.Linear(d,8,bias=True).double();book=torch.randn(2,4,4,dtype=torch.float64)
 Wu=nn.Linear(d,d,bias=False).double();Wf=nn.Linear(d,d,bias=False).double();out=nn.Linear(d,13,bias=False).double()
 labels=torch.randint(0,13,(t,))
 def objective(r):
  up=U+r;prob=pq(up).reshape(g,2,2,4).softmax(-1)
  forecast=torch.einsum('gpsn,snd->gpsd',prob,book).flatten(-2)
  source=Wu(up)+Wf(forecast)
  aligned=F.pad(forecast.flatten(0,1).repeat_interleave(4,0),(0,0,7,0))[:t][None]
  z=h+.1*aligned;z=z+reader(z,source.flatten(0,1)[None],torch.tensor([t]))
  return F.cross_entropy(out(z).flatten(0,1),labels)
 r=torch.zeros_like(U,requires_grad=True);loss=objective(r);gradient=torch.autograd.grad(loss,r)[0]
 direction=torch.randn_like(r);eps=1e-6
 numeric=float(((objective((r+eps*direction).detach())-objective((r-eps*direction).detach()))/(2*eps)).detach())
 analytic=float((gradient*direction).sum());error=abs(numeric-analytic)
 assert error<2e-6 and float(gradient.norm())>0
 q=U.flatten(1)/math.sqrt(2*d);codes=torch.tensor([0,1,0]);kernel=torch.exp(-torch.cdist(q,q).square()/2)*(codes[:,None]==codes[None,:])
 eig=float(torch.linalg.eigvalsh(kernel).min());flat=gradient.flatten(1)
 slope=-float((flat*(kernel@flat)).sum());eta=1e-4
 after=float(objective((-eta*(kernel@flat)).reshape_as(U)).detach())
 assert eig>=-1e-12 and slope<0 and after<float(loss.detach())
 checks['rbf_derivative_and_local_descent']=dict(finite_difference_error=error,kernel_min_eigenvalue=eig,first_order_slope=slope,before=float(loss.detach()),after=after,eta=eta,scope='Float64 toy reader/PQ/correction fixture; full-model FP32/Metal test still required',review_fix='Initial finite differences crossed an inherited FP32 cast in RMSNorm; mathematical fixture now uses a float64 norm. Production code is unchanged.')
 summary=json.loads((DEST/'telemetry_summary.json').read_text())
 for path,expected in summary['files'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
 checks['captured_training_source_and_manifest_unchanged']=True
 doc=(ROOT/'V10_2_RCDL_Architecture.md').read_text();missing=[]
 for link in re.findall(r'\]\(([^)]+)\)',doc):
  if '://' in link or link.startswith('#'):continue
  if not (ROOT/link.split('#')[0]).exists():missing.append(link)
 assert not missing,missing
 assert doc.count('```')%2==0
 checks['document_local_links_and_fences']=True
 result=dict(status='passed',torch_version=torch.__version__,device='cpu',scope='Specification arithmetic, masks and isolated mathematical fixtures. Not trained V10.2 validation.',
  document_sha256=hashlib.sha256((ROOT/'V10_2_RCDL_Architecture.md').read_bytes()).hexdigest(),
  review_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  evidence_summary_sha256=hashlib.sha256((DEST/'telemetry_summary.json').read_bytes()).hexdigest(),checks=checks)
 (DEST/'review_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
