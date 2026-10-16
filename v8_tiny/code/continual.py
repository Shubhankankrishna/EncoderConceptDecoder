"""Frozen-reference local-support fitting, positive support, routing diagnostics."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from model import *
from data import Mixer,DATA
from runtime import *

class Reservoir:
    def __init__(self,cap=65536,seed=811):self.cap=cap;self.rng=np.random.RandomState(seed);self.keys=np.empty(0);self.values=np.empty((0,32),np.float32);self.count=0
    def add(self,v):
        self.count+=len(v);keys=self.rng.uniform(size=len(v));keys=np.r_[self.keys,keys];values=np.concatenate([self.values,v]);take=np.argpartition(keys,-self.cap)[-self.cap:] if len(keys)>self.cap else np.arange(len(keys));self.keys=keys[take];self.values=values[take]


def fit_gmm(values,k,floor,seed):
    """Diagonal EM with random-sample initial means; avoids conflicting macOS OpenMP runtimes.
    Three independent starts, full-reservoir likelihood, fixed50-iteration/tolerance budget.
    """
    x=values.astype(np.float64);xx=x*x;n,d=x.shape;best=None
    for restart in range(3):
        rng=np.random.RandomState(seed+restart);mu=x[rng.choice(n,k,replace=False)].copy();var=np.tile(np.maximum(x.var(0),floor),(k,1));weights=np.full(k,1/k);previous=-np.inf;converged=False
        for iteration in range(1,51):
            precision=1/var;logp=-.5*(xx@precision.T-2*x@(mu*precision).T+np.sum(mu*mu*precision+np.log(var)+np.log(2*np.pi),1)[None])+np.log(weights)[None];normalizer=logsumexp(logp,axis=1);score=float(normalizer.mean());responsibility=np.exp(logp-normalizer[:,None]);nk=responsibility.sum(0).clip(1e-8);mu=responsibility.T@x/nk[:,None];var=np.maximum(responsibility.T@xx/nk[:,None]-mu*mu,floor);weights=nk/nk.sum()
            if abs(score-previous)<1e-4:converged=True;break
            previous=score
        precision=1/var;logp=-.5*(xx@precision.T-2*x@(mu*precision).T+np.sum(mu*mu*precision+np.log(var)+np.log(2*np.pi),1)[None])+np.log(weights)[None];score=float(logsumexp(logp,axis=1).mean())
        candidate=dict(means=mu,variances=var,weights=weights,converged=converged,iterations=iteration,score=score,initialization='random reservoir examples')
        if best is None or score>best['score']:best=candidate
    if not all(np.isfinite(best[key]).all() for key in ('means','variances','weights')):raise FloatingPointError('Nonfinite GMM parameters')
    return best

def fit(source,out,positive=False,mode='foundation',tokens=1000000):
    torch.set_num_threads(4);s,ck=load_system(source);freeze(s);blocks=s.active_blocks()
    if not positive and not hasattr(blocks[0],'projection'):
        s.add_phase()
        for b in blocks:b.banks=nn.ModuleList()
    if positive:
        for b in blocks:b.banks[-1].force=True
    samples=[Reservoir(seed=89900+i) for i in range(len(blocks))];inputslots=0;nonpad=0;symbolicprompt=0;symbolicanswer=0;mix=Mixer(89712 if not positive else 89713,64,mode);start=time.monotonic()
    for i,b in enumerate(blocks):
        def hook(a,docs,i=i,b=b):
            valid=docs>=0;h=F.linear(a,b.projection)[valid].detach().cpu().numpy();samples[i].add(h)
        b.capture=hook
    with torch.no_grad():
        while inputslots<tokens:
            batch=mix.batch(DEVICE);s(batch['x'],batch['docs']);inputslots+=batch['x'].numel();nonpad+=int((batch['docs']>=0).sum())
            for i,meta in enumerate(batch['metas']):
                if meta['task']!='text':symbolicprompt+=meta['prompt_len'];symbolicanswer+=int((batch['docs'][i]>=0).sum())-meta['prompt_len']
    for b in blocks:b.capture=None
    summaries=[]
    for i,(b,res) in enumerate(zip(blocks,samples)):
        components=4 if positive else 8;reference_var=b.reference_var.detach().cpu().numpy() if positive else np.var(res.values,axis=0)
        if not positive:b.reference_var.copy_(torch.tensor(reference_var,device=DEVICE))
        floor=np.maximum(1e-6,1e-4*reference_var)
        # sklearn uses scalar regularization; then apply the specified per-dimension floor.
        with threadpool_limits(limits=4):gmm=fit_gmm(res.values,components,floor,89900+i)
        owner=b.banks[-1] if positive else b;prefix='pos_' if positive else 'neg_'
        var=np.maximum(gmm['variances'],floor)
        for key,value in [('mu',gmm['means']),('var',var),('logw',np.log(gmm['weights']))]:getattr(owner,prefix+key).copy_(torch.tensor(value,dtype=torch.float32,device=DEVICE))
        summaries.append(dict(layer=i,activations_seen=res.count,reservoir=len(res.values),components=components,converged=bool(gmm['converged']),iterations=int(gmm['iterations']),training_log_likelihood=float(gmm['score']),variance_floor=floor.tolist()))
    if positive:
        for b in blocks:b.banks[-1].force=None
    config=dict(stage='positive_support' if positive else 'negative_support',source=str(source),implementation_sha256=implementation_hash(),reference_access=not positive,old_response_tokens_present=not positive,mode=mode,metadata=s.metadata())
    out=Path(out);out.mkdir(parents=True,exist_ok=True);save_checkpoint(out/'checkpoint.pt',s,None,ck.get('step',0),None,None,config);summary=dict(**config,input_slots=inputslots,nonpadding_input_tokens=nonpad,symbolic_header_tokens=symbolicprompt,symbolic_answer_tokens=symbolicanswer,groups=summaries,wall_s=time.monotonic()-start);atomic_json(out/'summary.json',summary);print(json.dumps(summary));return s

def gate_rates(s,batch):
    rates=[];handles=[]
    for i,b in enumerate(s.active_blocks()):
        def capture(a,docs,i=i,b=b):
            h=F.linear(a,b.projection);valid=docs>=0
            for p,bank in enumerate(b.banks):
                hit=(log_density(h,bank.pos_mu,bank.pos_var,bank.pos_logw)>log_density(h,b.neg_mu,b.neg_var,b.neg_logw))&valid
                rates.append(dict(layer=i,phase=p,positive_fraction=float(hit.sum()/valid.sum().clamp_min(1)),valid_positions=int(valid.sum())))
        b.capture=capture
    with torch.no_grad():s(batch['x'],batch['docs'])
    for b in s.active_blocks():b.capture=None
    return rates
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);ap.add_argument('--positive',action='store_true');ap.add_argument('--mode',default='foundation');ap.add_argument('--tokens',type=int,default=1000000);a=ap.parse_args();fit(a.source,a.out,a.positive,a.mode,a.tokens)
