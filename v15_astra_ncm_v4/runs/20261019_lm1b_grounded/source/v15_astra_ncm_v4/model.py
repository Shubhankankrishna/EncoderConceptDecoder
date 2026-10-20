"""Permutation-equivariant state/action learner with shared spatial refinement."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class AgentConfig:
    width: int = 128
    hidden: int = 192
    steps: int = 2
    mode: str = 'shared'
    strategies: bool = False


def neighbors(h, adj):
    out = adj @ h / adj.sum(-1,keepdim=True).clamp_min(1)
    incoming = adj.transpose(1,2) @ h / adj.sum(-2).unsqueeze(-1).clamp_min(1)
    return out, incoming


class Refinement(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(3*width,hidden)
        self.fc2 = nn.Linear(hidden,width)
        self.gate = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self,h,adj,z):
        x = self.norm(h+z[:,None,:])
        out,inc = neighbors(x,adj)
        return h + self.gate.sigmoid()*self.fc2(F.gelu(self.fc1(torch.cat([x,out,inc],-1))))


class Agent(nn.Module):
    def __init__(self,config=AgentConfig()):
        super().__init__()
        self.config = config
        w = config.width
        self.input = nn.Linear(5,w)
        # Same complete observation encoder in every ablation.
        self.context = nn.ModuleList([nn.Linear(3*w,w) for _ in range(2)])
        self.strategy = nn.Embedding(2,w) if config.strategies else None
        if config.mode in ('untied','untied_compute'):
            # Separate controls match parameters OR per-update arithmetic. Weight
            # sharing prevents an untied control matching both at the same width.
            hidden=config.hidden if config.mode=='untied_compute' else config.hidden//2
            self.cells = nn.ModuleList([Refinement(w,hidden) for _ in range(2)])
        else:
            self.cells = nn.ModuleList([Refinement(w,config.hidden)]) if config.steps else nn.ModuleList()
        self.action = nn.Sequential(nn.Linear(4*w,w),nn.GELU(),nn.LayerNorm(w))
        self.policy = nn.Linear(w,1)
        self.q = nn.Linear(w,1)
        self.value = nn.Sequential(nn.Linear(3*w,w),nn.GELU(),nn.Linear(w,1))
        self.successor_query = nn.Linear(w,w,bias=False)
        self.successor_key = nn.Linear(w,w,bias=False)

    def sample_strategies(self,batch_size,count=2,generator=None):
        """Explicit uniform categorical prior; independent draws can coincide.

        No posterior or future observation is accepted by this API. The learned
        embedding maps each preference to its supervised behavior.
        """
        if not self.config.strategies:
            return torch.zeros(batch_size,count,dtype=torch.long)
        return torch.randint(2,(batch_size,count),generator=generator)

    def forward(self,b,steps=None):
        h = self.input(b['features'])
        mask = b['node_mask'].unsqueeze(-1)
        for layer in self.context:
            out,inc = neighbors(h,b['adj'])
            h = (h+F.gelu(layer(torch.cat([h,out,inc],-1))))*mask
        z = self.strategy(b['strategy']) if self.strategy is not None else torch.zeros_like(h[:,0])
        history = [h]
        r = self.config.steps if steps is None else steps
        if r and not self.cells:
            raise ValueError('No recurrent cell in R0 control')
        for i in range(r):
            cell = self.cells[i] if self.config.mode in ('untied','untied_compute') else self.cells[0]
            h = cell(h,b['adj'],z)*mask
            history.append(h)
        rows = torch.arange(len(h),device=h.device)
        current,goal = h[rows,b['current']],h[rows,b['goal']]
        pool = h.sum(1)/mask.sum(1).clamp_min(1)
        context = torch.cat([current,goal,pool],-1)
        action = self.action(torch.cat([h,context[:,None,:].expand(-1,h.shape[1],-1)],-1))
        logits = self.policy(action).squeeze(-1)
        logits = logits.masked_fill(~b['legal'],-1e9)
        # Exact dynamics remain authoritative. This learned pointer is only an interface diagnostic.
        transition = (self.successor_query(action) @ self.successor_key(h).transpose(1,2))/(h.shape[-1]**.5)
        transition = transition.masked_fill(~b['node_mask'][:,None,:],-1e9)
        return {'logits':logits,'value':self.value(context).squeeze(-1).sigmoid(),
                'q':self.q(action).squeeze(-1).sigmoid(),'transition':transition,
                'action_hidden':action,'state_hidden':pool,
                'residual_norms':[float((v-u).detach().norm(dim=-1)[b['node_mask']].mean().cpu())
                                  for u,v in zip(history,history[1:])]}


def objective(out,b):
    active = b['legal'].any(-1)
    policy = -(b['policy_target']*F.log_softmax(out['logits'],-1)).sum(-1)
    policy = policy[active].mean() if active.any() else out['logits'].sum()*0
    value = F.mse_loss(out['value'],b['value_target'])
    q = F.mse_loss(out['q'][b['legal']],b['q_target'][b['legal']]) if b['legal'].any() else value*0
    target = torch.arange(b['legal'].shape[1],device=out['logits'].device)[None,:].expand_as(b['legal'])
    transition = F.cross_entropy(out['transition'][b['legal']],target[b['legal']]) if b['legal'].any() else value*0
    loss = policy + .5*value + .5*q + .1*transition
    return loss, {'loss':float(loss.detach()),'policy_ce':float(policy.detach()),
                  'value_mse':float(value.detach()),'q_mse':float(q.detach()),
                  'successor_ce':float(transition.detach())}


class PlanConditioner(nn.Module):
    """Language renderer receives an explicitly selected, verified plan, available at inference."""
    def __init__(self,width=128,max_nodes=24):
        super().__init__()
        self.width,self.max_nodes = width,max_nodes
        self.encode = nn.Sequential(nn.Linear(max_nodes*max_nodes+2,width),nn.GELU(),nn.Linear(width,width))

    def forward(self,paths,strategies,device):
        features = torch.zeros(len(paths),self.max_nodes,self.max_nodes,device=device)
        for b,path in enumerate(paths):
            if len(path)>self.max_nodes or any(i>=self.max_nodes or i<0 for i in path):
                raise ValueError('Plan exceeds renderer contract')
            features[b,torch.arange(len(path),device=device),torch.tensor(path,device=device)] = 1
        z = F.one_hot(torch.as_tensor(strategies,device=device),2).float()
        return self.encode(torch.cat([features.flatten(1),z],-1))
