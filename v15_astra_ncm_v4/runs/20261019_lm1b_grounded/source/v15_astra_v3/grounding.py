"""Optional grounded branch. No training or causal strategy benefit is implied.

The base code generator is retained for raw-text retention checks. Candidate
actions enter only the critic; the actor accepts observed action history.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from .gram import AstraGRAM
from .value import ValueContract


class ActionEncoder(nn.Module):
    def __init__(self,actions=7,width=64):
        super().__init__();self.actions=actions;self.table=nn.Embedding(actions+1,width,padding_idx=actions)

    def forward(self,ids):
        if ids.dtype!=torch.long or ((ids<0)|(ids>self.actions)).any():raise ValueError('Invalid typed action ID')
        return self.table(ids)


@dataclass(frozen=True)
class EpisodeStrategy:
    episode_ids: tuple
    policy_version: str
    mode: torch.Tensor
    tokens: torch.Tensor

    def validate(self,episode_ids,policy_version):
        if tuple(episode_ids)!=self.episode_ids or policy_version!=self.policy_version:
            raise ValueError('Strategy belongs to another episode or policy')

    def branch(self):
        return EpisodeStrategy(self.episode_ids,self.policy_version,self.mode.clone(),self.tokens.clone())


class StrategyRouter(nn.Module):
    def __init__(self,width,modes=2,tokens=1):
        super().__init__();self.width=width;self.modes=modes;self.tokens=tokens
        if min(modes,tokens)<1:raise ValueError('Positive strategy dimensions required')
        self.router=nn.Linear(width,modes);self.root_projection=nn.Linear(width,width*tokens)
        self.bank=nn.Parameter(torch.randn(modes,tokens,width)*.02)

    def begin(self,root_memory,episode_ids,policy_version,mode=None):
        if len(episode_ids)!=len(root_memory) or not policy_version:raise ValueError('Episode identity required')
        root=root_memory.mean(1);logits=self.router(root)
        if mode is None:mode=torch.multinomial(logits.softmax(-1),1).squeeze(1)
        mode=torch.as_tensor(mode,dtype=torch.long,device=root.device)
        if mode.shape!=(len(root),) or ((mode<0)|(mode>=self.modes)).any():raise ValueError('Wrong mode')
        z=self.root_projection(root).view(len(root),self.tokens,self.width)+self.bank[mode]
        return EpisodeStrategy(tuple(episode_ids),policy_version,mode,z),logits


class GroundedActor(nn.Module):
    """Track E head on a preserved HRM/GRAM generator; no candidate input API.

    History is deliberately pooled with observed positional weights so PAD does
    not alter the normalization. No padding key enters attention as a fake move.
    Full masked action-history attention is a separately benchmarked alternative.
    """
    def __init__(self,backbone,with_history=False,action_width=64):
        super().__init__();self.backbone=backbone;d=backbone.config.width
        self.action_encoder=ActionEncoder(width=action_width) if with_history else None
        self.history_projection=nn.Linear(action_width,d,bias=False) if with_history else None
        self.head=nn.Linear(2*d,7)

    def memory(self,x,state,history=None,*,strategy=None,episode_ids=(),policy_version=''):
        mem=self.backbone.encode_context(x,state)
        if self.action_encoder is not None:
            if history is None:raise ValueError('Observed history tensor required; use [B,0] at episode root')
            e=self.action_encoder(history);valid=history!=self.action_encoder.actions
            weights=torch.arange(1,history.shape[1]+1,device=mem.device,dtype=mem.dtype)[None]*valid
            pooled=(e*weights[:,:,None]).sum(1)/weights.sum(1,keepdim=True).clamp_min(1)
            mem=mem+self.history_projection(pooled)[:,None]
        elif history is not None and history.numel():raise ValueError('History ablation is disabled')
        if strategy is not None:
            strategy.validate(episode_ids,policy_version);mem=torch.cat([mem,strategy.tokens],1)
        return mem

    def features(self,x,state,history=None,*,strategy=None,episode_ids=(),policy_version='',noise=None,mean=False):
        mem=self.memory(x,state,history,strategy=strategy,episode_ids=episode_ids,policy_version=policy_version)
        if isinstance(self.backbone,AstraGRAM):h,_=self.backbone._trajectory(mem,noise=noise,mean=mean)
        else:
            if noise is not None:raise ValueError('Deterministic arm does not accept Gaussian noise')
            h,_=self.backbone.reason(mem)
        return torch.cat([h.mean(1),mem.mean(1)],-1)

    def forward(self,x,state,history=None,**observed_controls):
        return self.head(self.features(x,state,history,**observed_controls))

    def supervised_loss(self,x,state,target_action,*,target_codes=None,history=None,noise=None,**controls):
        mem=self.memory(x,state,history,**controls)
        if isinstance(self.backbone,AstraGRAM):
            if target_codes is None:raise ValueError('Training-only canonical target codes required by transferred posterior')
            h,d=self.backbone.full_bptt(mem,target_codes,noise=noise);kl=d['kl'].mean()
        else:h,_=self.backbone.reason(mem);kl=mem.new_zeros(())
        nll=F.cross_entropy(self.head(torch.cat([h.mean(1),mem.mean(1)],-1)),target_action)
        # One categorical action likelihood, not 641 code terms. No division of
        # KL by 641 in Track E. Target codes are deterministic teacher features.
        return nll+kl,dict(action_nll=nll.detach(),kl_sum=kl.detach(),objective='single_step_action_terminal_ELBO')


class CandidateCritic(nn.Module):
    """Frozen-policy candidate success; horizon includes the candidate action."""
    def __init__(self,contract:ValueContract,action_width=64,hidden=64):
        super().__init__();self.contract=contract;self.action_encoder=ActionEncoder(width=action_width)
        self.network=nn.Sequential(nn.Linear(contract.feature_dim+action_width+1,hidden),nn.SiLU(),nn.Linear(hidden,1))

    def logits(self,features,candidates,horizon_left,*,contract_id):
        if contract_id!=self.contract.identity:raise ValueError('Critic policy/representation mismatch')
        if features.ndim!=2 or features.shape[1]!=self.contract.feature_dim:raise ValueError('Wrong features')
        if candidates.ndim!=2 or len(candidates)!=len(features) or (candidates>=7).any():raise ValueError('Candidate IDs must be real actions')
        h=torch.as_tensor(horizon_left,device=features.device)
        if h.shape!=(len(features),) or ((h<1)|(h>self.contract.max_horizon)|(h!=h.floor())).any():
            raise ValueError('Candidate requires at least one remaining action')
        a=self.action_encoder(candidates);f=features[:,None].expand(-1,candidates.shape[1],-1)
        hf=(h.to(features.dtype)/self.contract.max_horizon)[:,None,None].expand(-1,candidates.shape[1],1)
        return self.network(torch.cat([f,a,hf],-1)).squeeze(-1)

    def forward(self,*args,**kwargs):return self.logits(*args,**kwargs).sigmoid()

    def loss(self,features,candidates,horizon_left,outcomes,*,contract_id):
        logits=self.logits(features.detach(),candidates,horizon_left,contract_id=contract_id)
        if outcomes.shape!=logits.shape or not ((outcomes==0)|(outcomes==1)).all():raise ValueError('Verified binary outcomes required')
        return F.binary_cross_entropy_with_logits(logits,outcomes.to(logits.dtype))


def candidate_outcome(env,state,goal,candidate,horizon,continuation,*,contract_id,episode_group,split):
    """Exact candidate step followed by a pinned callable policy, not an oracle.

    The caller owns the policy/version binding. Every simulator call is counted;
    no BFS distances or teacher solutions enter the continuation callable.
    """
    from .tasks import step
    if horizon<1 or split not in ('train','valid','test'):raise ValueError('Invalid rollout record')
    s=step(env,state,candidate);calls=1;success=s==goal;legal=s is not None
    if legal and not success:
        for left in range(horizon-1,0,-1):
            a=continuation(env,s,goal,left);s=step(env,s,a) if a is not None else None;calls+=1
            if s is None:break
            if s==goal:success=True;break
    return dict(candidate=candidate,horizon_left=horizon,terminal_success=int(success),legal=legal,
                transition_calls=calls,contract_id=contract_id,episode_group=episode_group,split=split)
