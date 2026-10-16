"""Public-header generation scoring, document-cluster uncertainty and absolute gates."""
import argparse,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.stats import beta
from model import *
from data import *
from runtime import *
from generation import generate

def wilson(k,n,alpha=.05):
    from scipy.stats import norm
    if n==0:return [0.,1.]
    z=float(norm.ppf(1-alpha/2));p=k/n;den=1+z*z/n;centre=(p+z*z/(2*n))/den;width=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,centre-width),min(1.,centre+width)]
def exact(k,n,alpha=.05):return [float(beta.ppf(alpha/2,k,n-k+1)) if k else 0.,float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.]
def paired(a,b,alpha=.05):
    a=np.asarray(a,dtype=bool);b=np.asarray(b,dtype=bool);n=len(a)
    if len(b)!=n:raise ValueError('Paired sample length mismatch')
    win=int((a&~b).sum());lose=int((~a&b).sum());wl,wh=exact(win,n,alpha/2);ll,lh=exact(lose,n,alpha/2)
    return dict(delta=float(a.mean()-b.mean()),interval=[wl-lh,wh-ll],gains=win,losses=lose,n=n,method='Bonferroni exact binomial bounds for gain/loss probabilities; valid at zero discordance')
def cluster_ci(numerators,denominators,ids,seed=811):
    totals=defaultdict(lambda:[0.,0.])
    for v,w,key in zip(numerators,denominators,ids):totals[str(key)][0]+=float(v);totals[str(key)][1]+=float(w)
    values=np.array(list(totals.values()));rng=np.random.RandomState(seed);draw=rng.randint(len(values),size=(4000,len(values)));samples=values[draw].sum(1);est=samples[:,0]/np.maximum(samples[:,1],1)
    return [float(np.quantile(est,.025)),float(np.quantile(est,.975))]

def text_eval(s,split='dev',family='base',source='edu',n=256,batch=32):
    device=next(s.parameters()).device;b,ids=fixed_text(split,n,family=family,source=source);nums=[];dens=[]
    with torch.no_grad():
        for i in range(0,n,batch):
            x=b['x'][i:i+batch].to(device);d=b['docs'][i:i+batch].to(device);m=b['m'][i:i+batch].to(device);loss=nll(s(x,d),x,m);nums.extend(loss.sum(1).cpu().tolist());dens.extend(m[:,:-1].sum(1).cpu().tolist())
    return dict(nll=sum(nums)/sum(dens),interval=cluster_ci(nums,dens,ids),numerators=nums,denominators=dens,document_ids=ids,unique_documents=len(set(map(str,ids))),targets=int(sum(dens)))

def symbolic_eval(s,name,split='dev',n=None,batch=32,latent_draws=0):
    a=np.load(DATA/f'{split}_{name}.npz');n=min(n or len(a['x']),len(a['x']));ps=a['prompt_len'][:n];success=[];first=[];records=[];prior=[];nodeacc=[];losses=[];device=next(s.parameters()).device;start=time.monotonic()
    for i in range(0,n,batch):
        x=a['x'][i:i+batch];m=a['m'][i:i+batch];docs=a['docs'][i:i+batch];p=int(ps[i]);assert np.all(ps[i:i+batch]==p)
        limit=min(T-p,100 if x.shape[1]==192 else 164)
        r=generate(s,x[:,:p],max_new=limit)
        with torch.no_grad():
            xt=torch.tensor(x,device=device);mt=torch.tensor(m,device=device);logits=s(xt,torch.tensor(docs,device=device));loss=nll(logits,xt,mt);losses.extend((loss.sum(1)/mt[:,:-1].sum(1).clamp_min(1)).cpu().tolist())
            if name not in ('lookup','compose','bind'):nodeacc.extend((logits[:,p+1].argmax(-1)==xt[:,p+2]).cpu().tolist())
        union=np.array([v['success'] for v in r['verification']],bool);prior_only=np.zeros(len(x),bool)
        for draw in range(latent_draws):
            g=torch.Generator().manual_seed(89000+draw*10000+i);bank=ball(torch.randn(len(x),T//K,DZ,generator=g)).to(device);rr=generate(s,x[:,:p],bank,max_new=limit);hits=np.array([v['success'] for v in rr['verification']],bool);prior_only|=hits;union|=hits
        for j,v in enumerate(r['verification']):
            response=r['x'][j,p:int(r['lengths'][j])];task,edges,st,goal,col=parse_prompt(x[j,:p]);legal=len(response)>=4 and COL0<=response[0]<COL0+N_COL and response[1]==TO and response[3]==SEP and edges.get((st,int(response[0])-COL0))==int(response[2])-NODE0
            success.append(bool(v['success']));first.append(bool(legal));prior.append(bool(prior_only[j]));records.append(dict(index=i+j,success=bool(v['success']),first_legal=bool(legal),reason=v['reason'],response=response.tolist(),prior_only=bool(prior_only[j]),union=bool(union[j])))
    k=sum(success);out=dict(name=name,split=split,n=n,successes=k,rate=k/n,interval=wilson(k,n),first_legal_rate=sum(first)/n,success_vector=success,first_vector=first,reasons=dict(Counter(v['reason'] for v in records)),conditional_reference_node_accuracy=float(np.mean(nodeacc)) if nodeacc else None,teacher_forced_nll=float(np.mean(losses)),seconds=time.monotonic()-start,records=records)
    if latent_draws:
        fails=~np.array(success);rescues=int((fails&np.array(prior)).sum());nf=int(fails.sum());out['coverage']=dict(draws=latent_draws,prior_only_rate=sum(prior)/n,prior_only_interval=wilson(sum(prior),n),union_rate=sum(v['union'] for v in records)/n,baseline_failures=nf,rescued_problems=rescues,conditional_rescue=rescues/max(nf,1),conditional_interval=wilson(rescues,nf))
        c=out['coverage'];out['coverage_gate']=out['rate']>=.1 and c['prior_only_rate']>=.2 and c['prior_only_interval'][0]>.1 and nf>=64 and rescues>=5 and c['conditional_rescue']>=.05 and c['conditional_interval'][0]>.01
    return out


def interface_audit(s,split='dev'):
    if s.is_flat:return dict(applicable=False)
    device=next(s.parameters()).device;text,ids=fixed_text(split,16);graph=np.load(DATA/f'{split}_standard.npz');x=torch.cat([text['x'],torch.tensor(graph['x'][:16])]).to(device);docs=torch.cat([text['docs'],torch.tensor(graph['docs'][:16])]).to(device);mask=torch.cat([text['m'],torch.tensor(graph['m'][:16])]).to(device);domain=torch.tensor([0]*16+[1]*16,device=device)
    with torch.no_grad():
        h,c,cd=s.context(x,docs);q,a,u=s.plans(c,cd);valid=chunk_info(docs)[1];cm=ncp_weights(dict(docs=docs,m=mask));target=c[:,1:];forecast=((q[:,:-1]-target).square().mean(-1)*cm).sum()/cm.sum().clamp_min(1);persist=((c[:,:-1]-target).square().mean(-1)*cm).sum()/cm.sum().clamp_min(1);vq,codes=s.foundation.vq(c,valid);used=[]
        for segment in range(S):used.append(int(codes[...,segment][valid].unique().numel()))
        full=s.foundation.decode(h,q,a,docs);none=s.foundation.decode(h,torch.zeros_like(q),torch.zeros_like(a),docs);token=balanced(nll(full,x,mask),mask[:,:-1],domain)[0];disabled=balanced(nll(none,x,mask),mask[:,:-1],domain)[0]
    return dict(applicable=True,forecast_mse=float(forecast),persistence_mse=float(persist),vq_segment_sse=float(vq),codes_used_per_segment=used,token_nll=float(token),ports_disabled_nll=float(disabled),port_ablation_delta=float(disabled-token),scope='32 fixed development rows, diagnostic only; no gate fitting')

def competence(e):
    lookup=e['symbolic']['lookup'];std=e['symbolic']['standard'];return lookup['rate']>=.95 and lookup['interval'][0]>=.90 and std['first_legal_rate']>=.98 and std['rate']>=.80 and std['interval'][0]>=.75

def evaluate(s,split='dev',names=('standard','lookup','compose','hard10'),n=512,fast=False):
    s.eval();s.fast_generation=fast;out=dict(split=split,active_parameters=active_count(s),metadata=s.metadata(),symbolic={})
    for name in names:out['symbolic'][name]=symbolic_eval(s,name,split,n)
    out['text']={source:text_eval(s,split,source=source) for source in ('edu','ts')}
    out['concept_interface']=interface_audit(s,split)
    if 'standard' in out['symbolic'] and 'lookup' in out['symbolic']:out['competence_pass']=competence(out)
    return out

def retention(candidate,anchor,alpha=.05):
    out=dict(symbolic={},text={});ok=True
    for name in set(candidate['symbolic'])&set(anchor['symbolic']):
        if name not in ('standard','lookup'):continue
        v=paired(candidate['symbolic'][name]['success_vector'],anchor['symbolic'][name]['success_vector'],alpha/4);out['symbolic'][name]=v;ok&=v['interval'][0]>=-.02
    for source in ('edu','ts'):
        a=candidate['text'][source];b=anchor['text'][source];nums=np.array(a['numerators'])-np.array(b['numerators']);interval=cluster_ci(nums,a['denominators'],a['document_ids']);out['text'][source]=dict(delta=a['nll']-b['nll'],interval=interval);ok&=interval[1]<=.02
    out['pass']=bool(ok and candidate.get('competence_pass',False));return out
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);ap.add_argument('--split',default='dev');ap.add_argument('--names',default='standard,lookup,compose,hard10');ap.add_argument('--n',type=int,default=512);ap.add_argument('--fast',action='store_true');a=ap.parse_args();torch.set_num_threads(4);s,ck=load_system(a.source);out=evaluate(s,a.split,a.names.split(','),a.n,a.fast);out['source']=a.source;out['implementation_sha256']=implementation_hash();atomic_json(a.out,out);print(json.dumps({k:v for k,v in out.items() if k not in ('symbolic','text')}))
