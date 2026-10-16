"""Fixed development/test likelihood, strict generation, and charged search experiments."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
from model import freeze,ball
from data import ROOT,DATA,T,K,Text,canonical
from runtime import DEVICE,load_system,atomic_json,nll,sync,tensor_hash
from generation import generate
from search import search_one

SUITES=('standard','hard','long','pointer','pointer_long','lookup','layout','compose')

def paired(a,b,seed=97001,draws=4000):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if len(a)!=len(b) or not len(a):raise ValueError('Paired units must match')
    d=a-b;rng=np.random.RandomState(seed);means=np.empty(draws)
    for j in range(0,draws,100):
        means[j:j+100]=d[rng.randint(len(d),size=(min(100,draws-j),len(d)))].mean(1)
    return dict(difference=float(d.mean()),ci95=np.percentile(means,[2.5,97.5]).tolist(),units=len(d),method='paired unit bootstrap',draws=draws)

def weighted_nll_pair(a,b,seed=97002):
    # Resample whole documents; preserve token weighting within each draw.
    if a['document_ids']!=b['document_ids'] or a['counts']!=b['counts']:raise ValueError('Text units do not match')
    d=np.asarray(a['sums'])-np.asarray(b['sums']);c=np.asarray(a['counts']);rng=np.random.RandomState(seed);vals=[]
    for _ in range(4000):
        ix=rng.randint(len(c),size=len(c));vals.append(float(d[ix].sum()/c[ix].sum()))
    return dict(difference=float(d.sum()/c.sum()),ci95=np.percentile(vals,[2.5,97.5]).tolist(),units=len(c),method='paired document bootstrap, token-weighted ratio')

def load_eval(path,fast=False):
    s,ck=load_system(path);s.eval();freeze(s);s.fast_generation=fast
    return s,ck

@torch.no_grad()
def text_metrics(s,split='dev',n=256,batch=16):
    source=Text(split);rng=np.random.RandomState(97100);ids=rng.permutation(len(source.off)-1)[:n];rows=[];counts=[]
    for i in ids:
        a,b=map(int,source.off[i:i+2]);seq=source.edu[a:min(b,a+T)];x=np.zeros(T,np.int64);d=np.full(T,-1,np.int64);m=np.zeros(T,np.float32)
        x[:len(seq)]=seq;d[:len(seq)]=0;m[:max(0,len(seq)-1)]=1;rows.append((x,d,m));counts.append(int(m.sum()))
    sums=[]
    for j in range(0,len(rows),batch):
        x,d,m=[torch.as_tensor(np.stack(v),device=DEVICE) for v in zip(*rows[j:j+batch])]
        sums.extend(nll(s(x,d),x,m).sum(1).cpu().tolist())
    mean=sum(sums)/sum(counts)
    return dict(nll=mean,perplexity=math.exp(min(mean,50)),document_ids=ids.tolist(),sums=sums,counts=counts,
        scope='one initial <=384-token block per fresh held-out document; whole-document resampling')

@torch.no_grad()
def symbolic_metrics(s,split='dev',n=128,batch=16,suites=SUITES,latent_k=0,length=1,seed=0):
    out={}
    for suite in suites:
        ds=np.load(DATA/f'{split}_{suite}.npz');count=min(n,len(ds['x']));xs=ds['x'][:count];docs=ds['docs'][:count];ms=ds['m'][:count];ps=ds['prompt_len'][:count]
        sums=[];loss_counts=[];success=[False]*count;reasons=['']*count;responses=[[] for _ in range(count)];elapsed=0.;tokens=0;curves=np.zeros((latent_k+1,count),dtype=bool);costs=np.zeros(latent_k+1)
        for j in range(0,count,batch):
            x=torch.as_tensor(xs[j:j+batch],device=DEVICE);d=torch.as_tensor(docs[j:j+batch],device=DEVICE);m=torch.as_tensor(ms[j:j+batch],device=DEVICE)
            sums.extend(nll(s(x,d),x,m).sum(1).cpu().tolist());loss_counts.extend(m[:,:-1].sum(1).cpu().tolist())
        for p in sorted(set(ps.tolist())):
            indexes=np.flatnonzero(ps==p)
            for j in range(0,len(indexes),batch):
                ix=indexes[j:j+batch];r=generate(s,xs[ix,:p]);elapsed+=r['seconds'];tokens+=r['generated_tokens']
                costs[0]+=r['seconds']
                for row,i in enumerate(ix):
                    success[i]=bool(r['verification'][row]['success']);reasons[i]=r['verification'][row]['reason'];responses[i]=r['x'][row,p:int(r['lengths'][row])].tolist()
                    curves[0,i]=success[i]
                if latent_k and s.adapters is not None:
                    g=torch.Generator().manual_seed(seed+1000*j+int(p))
                    for candidate in range(latent_k):
                        bank=ball(torch.randn(len(ix),T//K,s.cfg.dz,generator=g)).to(DEVICE);rr=generate(s,xs[ix,:p],bank,length,encoding=r['encoding']);elapsed+=rr['seconds'];tokens+=rr['generated_tokens'];costs[candidate+1]+=rr['seconds']
                        for row,i in enumerate(ix):success[i]|=bool(rr['verification'][row]['success'])
                        curves[candidate+1,ix]=np.asarray(success)[ix]
        hist={r:reasons.count(r) for r in sorted(set(reasons))}
        out[suite]=dict(n=count,success_rate=float(np.mean(success)),success=success,nll=sum(sums)/sum(loss_counts),sums=sums,counts=loss_counts,
            reasons=hist,greedy_responses=responses,canonical=[canonical(xs[i,:ps[i]]) for i in range(count)],generation_seconds=elapsed,generated_tokens=tokens,
            latent_candidates=latent_k,window_length=length,candidate_coverage={str(k):float(curves[k].mean()) for k in range(latent_k+1)},candidate_cost_seconds=np.cumsum(costs).tolist())
    return out

def evaluate(path,out,split='dev',n=128,text_n=256,batch=16,fast=False,suites=SUITES,latent_k=0):
    s,ck=load_eval(path,fast);start=time.monotonic();before=tensor_hash(s)
    result=dict(checkpoint=str(path),split=split,active_parameters=s.active_count(),checkpoint_targets=ck['extra'].get('targets'),text=text_metrics(s,split,text_n,batch),
        symbolic=symbolic_metrics(s,split,n,batch,suites,latent_k),status='completed')
    if tensor_hash(s)!=before:raise RuntimeError('Evaluation mutated weights')
    result['seconds']=time.monotonic()-start;atomic_json(out,result);return result

def search_eval(path,out,split,methods,budgets,n=128,suite='hard',params=None,fast=False,seed=0,deadline=None):
    s,ck=load_eval(path,fast);params=params or {};ds=np.load(DATA/f'{split}_{suite}.npz');n=min(n,len(ds['x']));start=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Compilation and warm-up are disclosed separately, before any timed attempt.
    warm=time.monotonic();p=int(ds['prompt_len'][0]);generate(s,ds['x'][:1,:p],max_new=8)
    if s.adapters is not None:
        from generation import repair_gradient
        z=torch.zeros(1,T//K,s.cfg.dz,device=DEVICE,requires_grad=True);r=generate(s,ds['x'][:1,:p],max_new=8);repair_gradient(s,r,z,kl=0.)
    warm_s=time.monotonic()-warm;before=tensor_hash(s);rows=[];log=out/'attempts.jsonl'
    for i in range(n):
        p=int(ds['prompt_len'][i]);prompt=ds['x'][i,:p]
        # Rotate order per item to limit thermal/time-order confounding.
        arms=[(method,float(budget)) for budget in budgets for method in methods];arms=arms[i%len(arms):]+arms[:i%len(arms)]
        for method,budget in arms:
            if deadline is not None and time.time()>=deadline:break
            r=search_one(s,prompt,method,budget_s=budget,seed=seed+110000+i,**params);row=dict(item=i,canonical=canonical(prompt),method=method,budget_s=budget,**r)
            rows.append(row)
            with open(log,'a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
        if deadline is not None and time.time()>=deadline:break
        if i%8==7:atomic_json(out/'live.json',dict(items=i+1,total=n,seconds=time.monotonic()-start))
    if tensor_hash(s)!=before:raise RuntimeError('Search mutated weights')
    summary=dict(checkpoint=str(path),split=split,suite=suite,requested_items=n,parameters=params,warmup_seconds=warm_s,seconds=time.monotonic()-start,
        status='completed' if len(rows)==n*len(methods)*len(budgets) else 'partial_budget',arms={},comparisons={},scope='paired problem-level uncertainty on this trained seed; search RNG is not an independent training seed')
    for budget in budgets:
        for method in methods:
            rr=[r for r in rows if r['method']==method and r['budget_s']==budget];key=f'{method}@{budget:g}'
            summary['arms'][key]=dict(n=len(rr),success_rate=float(np.mean([r['success'] for r in rr])) if rr else None,items=[r['item'] for r in rr],success=[r['success'] for r in rr],
                total_seconds=sum(r['seconds'] for r in rr),gradient_seconds=sum(r['gradient_seconds'] for r in rr),generations=sum(len(r['records']) for r in rr),deadline_overrun_seconds=sum(r.get('deadline_overrun_s',0) for r in rr))
        retry=summary['arms'].get(f'retry@{budget:g}')
        if retry:
            ref=dict(zip(retry['items'],retry['success']))
            for method in methods:
                if method=='retry':continue
                arm=summary['arms'][f'{method}@{budget:g}'];a=dict(zip(arm['items'],arm['success']));ix=sorted(a.keys()&ref.keys())
                if ix:summary['comparisons'][f'{method}-retry@{budget:g}']=paired([a[i] for i in ix],[ref[i] for i in ix])
    atomic_json(out/'summary.json',summary);return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True);ap.add_argument('--split',choices=['dev','test'],default='dev');ap.add_argument('--n',type=int,default=128);ap.add_argument('--text-n',type=int,default=256);ap.add_argument('--batch',type=int,default=16);ap.add_argument('--fast',action='store_true');ap.add_argument('--suites',default=','.join(SUITES));ap.add_argument('--latent-k',type=int,default=0)
    args=ap.parse_args();torch.set_num_threads(4);evaluate(args.checkpoint,args.out,args.split,args.n,args.text_n,args.batch,args.fast,tuple(args.suites.split(',')),args.latent_k)

if __name__=='__main__':main()
