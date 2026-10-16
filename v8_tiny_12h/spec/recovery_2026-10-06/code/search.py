"""AB-MCTS-A over realized answer/control refinements, with charged wall time."""
import hashlib,time
from dataclasses import dataclass,field
import numpy as np
import torch
from model import ball
from data import T,K,COL0,N_COL,NODE0,TO,SEP,DONE,parse_prompt
from generation import generate,repair_gradient,active_windows,MaskedAdam
from runtime import sync

@dataclass
class Stats:
    rewards:list=field(default_factory=list)
    def add(self,r):
        if not 0<=r<=1:raise ValueError('Reward outside [0,1]')
        self.rewards.append(float(r))
    def draw(self,rng):return rng.beta(.5+sum(self.rewards),.5+len(self.rewards)-sum(self.rewards))
@dataclass
class Node:
    parent:object=None
    bank:object=None
    result:object=None
    depth:int=0
    children:list=field(default_factory=list)
    answer:Stats=field(default_factory=Stats)
    gen:Stats=field(default_factory=Stats)
    cont:Stats=field(default_factory=Stats)
    identifier:int=-1

def attach(parent,node,reward):
    parent.children.append(node);node.parent=parent;node.answer.add(reward);parent.gen.add(reward);parent.answer.add(reward)
    ancestor=parent.parent
    while ancestor is not None:ancestor.answer.add(reward);ancestor.cont.add(reward);ancestor=ancestor.parent

def eligible(n,max_depth):return n.depth<max_depth
def select(root,rng,max_depth=16):
    node=root
    while True:
        can_gen=eligible(node,max_depth);kids=[c for c in node.children if eligible(c,max_depth)]
        if not can_gen and not kids:return None
        if not kids:return node if can_gen else None
        if can_gen and node.gen.draw(rng)>=node.cont.draw(rng):return node
        node=max(kids,key=lambda c:c.answer.draw(rng))

def control_hash(bank):return hashlib.sha256(bank.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def search_one(s,prompt,method='retry',budget_s=2.,candidate_limit=32,length=1,lr=.03,kl=.05,temperature=.8,top_k=128,seed=0,feedback='binary'):
    device=next(s.parameters()).device;prompt=np.asarray(prompt)[None];p=prompt.shape[1]
    rng=np.random.RandomState(seed);g=torch.Generator().manual_seed(seed);root=Node();nodes=[];records=[];requests=0;duplicate_requests=0;consecutive_duplicates=0
    start=time.perf_counter();deadline=start+budget_s;grad_seconds=0.;encoding=None
    def reward(result):
        v=result['verification'][0]
        return float(v['success']) if feedback=='binary' else (1. if v['success'] else public_grade(prompt[0],result['x'][0,p:int(result['lengths'][0])]))
    def evaluate(bank,parent,operator,temp=0):
        nonlocal encoding
        before=time.perf_counter();result=generate(s,prompt,bank,length,temp,top_k,seed+len(records),encoding=encoding,max_new=min(164,T-p,68 if p==124 else 164))
        encoding=result['encoding'];finished=time.perf_counter();credited=finished<=deadline and bool(result['verification'][0]['success'])
        node=Node(bank=bank.detach().clone(),result=result,depth=parent.depth+1,identifier=len(nodes));attach(parent,node,reward(result));nodes.append(node)
        records.append(dict(attempt=len(records),node=node.identifier,parent=parent.identifier,operator=operator,depth=node.depth,seconds=finished-before,
            elapsed_s=finished-start,verified_before_deadline=finished<=deadline,success=bool(result['verification'][0]['success']),credited_success=credited,
            reason=result['verification'][0]['reason'],response=result['x'][0,p:int(result['lengths'][0])].tolist(),generated_tokens=result['generated_tokens'],
            forward_calls=result['forward_calls'],control_sha256=control_hash(bank),gradient_seconds=grad_seconds))
        return node,credited
    zero=torch.zeros(1,T//K,8,device=device);first,success=evaluate(zero,root,'common_zero')
    if method=='greedy' or success:return dict(success=success,records=records,seconds=time.perf_counter()-start,gradient_seconds=0.,duplicate_requests=0,tree_nodes=len(nodes))
    visited={control_hash(zero)};current=first;chain_opt=None
    while len(records)<candidate_limit and time.perf_counter()<deadline:
        requests+=1
        if requests>8*candidate_limit:break
        if method in ('retry','latent','combined'):
            parent=root;operator=method;bank=zero if method=='retry' else ball(torch.randn(1,T//K,8,generator=g)).to(device)
            temp=temperature if method in ('retry','combined') else 0
        else:
            hybrid=method in ('ab_hybrid','hybrid_random','hybrid_chain','ab_hybrid_randomdir');tree=method in ('ab_prior','ab_hybrid','ab_hybrid_randomdir')
            if tree:parent=select(root,rng)
            elif method in ('gradcuit','gradcuit_kl'):parent=current
            elif method in ('random_chain','hybrid_chain'):parent=current if eligible(current,16) else root
            else:
                candidates=[root]+[n for n in nodes if eligible(n,16)]
                parent=candidates[int(rng.randint(len(candidates)))]
            if parent is None:break
            if parent is root:bank=ball(torch.randn(1,T//K,8,generator=g)).to(device);operator='fresh_prior';temp=0
            else:
                active=active_windows(parent.result['lengths'],p,length,device);indices=active[0].nonzero().flatten().cpu().tolist()
                if not indices:break
                kernel='gradient' if method in ('gradcuit','gradcuit_kl') else str(rng.choice(['local','suffix','gradient'],p=[1/3,1/3,1/3])) if hybrid else str(rng.choice(['local','suffix'],p=[.5,.5]))
                bank=parent.bank.detach().clone();operator=kernel;temp=0
                if kernel=='gradient':
                    bank.requires_grad_(True);begin=time.perf_counter()
                    grad,meta=repair_gradient(s,parent.result,bank,length,kl if method!='gradcuit' else 0.)
                    if method in ('gradcuit','gradcuit_kl'):
                        if chain_opt is None:chain_opt=MaskedAdam(bank,lr)
                        else:chain_opt.z=bank
                        opt=chain_opt
                    else:opt=MaskedAdam(bank,lr)
                    oldbank=bank.detach().clone();meta_update=opt.update(grad,active,torch.ones(1,device=device,dtype=torch.bool))
                    if method in ('ab_hybrid_randomdir',):
                        direction=torch.randn(bank.shape,generator=g).to(device)*active[...,None];direction/=direction.flatten(1).norm(dim=1).clamp_min(1e-12)[:,None,None]
                        with torch.no_grad():
                            target=meta_update['step_norm'];lo=torch.zeros_like(target);hi=torch.ones_like(target)
                            for _ in range(8):
                                norm=(ball(oldbank+hi[:,None,None]*direction)-oldbank).flatten(1).norm(dim=1);hi=torch.where(norm<target,hi*2,hi)
                            for _ in range(16):
                                mid=(lo+hi)/2;norm=(ball(oldbank+mid[:,None,None]*direction)-oldbank).flatten(1).norm(dim=1);lo=torch.where(norm<target,mid,lo);hi=torch.where(norm>=target,mid,hi)
                            bank.copy_(ball(oldbank+((lo+hi)/2)[:,None,None]*direction))
                    sync(device);grad_seconds+=time.perf_counter()-begin
                    bank=bank.detach()
                    if time.perf_counter()>=deadline:break
                elif kernel=='local':
                    i=int(rng.choice(indices));bank[:,i]+=torch.randn(1,8,generator=g).to(device)*.25;bank=ball(bank)
                else:
                    i=int(rng.choice(indices));bank[:,i:]=ball(torch.randn(1,T//K-i,8,generator=g)).to(device)
        key=control_hash(bank)
        # Token retry samples differ by their token RNG; only deterministic same-control requests reuse a node.
        if temp==0 and key in visited:
            duplicate_requests+=1;consecutive_duplicates+=1
            if consecutive_duplicates>=8:break
            continue
        consecutive_duplicates=0
        visited.add(key)
        if time.perf_counter()>=deadline:break
        current,success=evaluate(bank,parent,operator,temp)
        if success:break
    return dict(success=success,records=records,seconds=time.perf_counter()-start,gradient_seconds=grad_seconds,duplicate_requests=duplicate_requests,
        tree_nodes=len(nodes),deadline_overrun_s=max(0.,time.perf_counter()-deadline),generations=len(records))


def public_grade(prompt,response):
    task,edges,start,goal,col=parse_prompt(prompt)
    if task!='graph':return 0.
    rows=[list(map(int,response[j:j+4])) for j in range(0,len(response),4)]
    terminator=bool(rows and rows[-1]==[DONE,SEP,SEP,SEP]);walkrows=rows[:-1] if terminator else rows
    attempts=len(walkrows);grammar=terminator and all(len(row)==4 and COL0<=row[0]<COL0+N_COL and row[1]==TO and NODE0<=row[2]<NODE0+24 and row[3]==SEP for row in walkrows)
    cur=start;h=0;repeats=0;seen=set()
    for row in walkrows:
        if len(row)!=4:break
        c,to,dst,sep=row
        if not(COL0<=c<COL0+N_COL and to==TO and sep==SEP):break
        edge=(cur,c-COL0,dst-NODE0)
        if edges.get(edge[:2])!=edge[2]:break
        repeats+=edge in seen;seen.add(edge);cur=edge[2];h+=1
    g=h>0 and cur==goal
    return float(np.clip(.4*h/max(1,attempts)+.2*grammar+.2*g-.1*repeats/max(1,h),0,.9))
