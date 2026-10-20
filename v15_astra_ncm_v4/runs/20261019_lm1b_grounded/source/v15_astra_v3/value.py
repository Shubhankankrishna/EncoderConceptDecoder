"""Horizon- and rollout-policy-specific value contract for a later search stage."""
from dataclasses import dataclass,asdict
import hashlib
import json
import torch
from torch import nn


@dataclass(frozen=True)
class ValueContract:
    policy_version: str
    representation_version: str
    rollout_policy: str
    max_horizon: int
    feature_dim: int
    strategy_semantics: str = 'prior_mean_context'

    def __post_init__(self):
        if not all((self.policy_version,self.representation_version,self.rollout_policy,self.strategy_semantics)):
            raise ValueError('Policy, representation and rollout semantics must be pinned')
        if self.max_horizon<1 or self.feature_dim<1:raise ValueError('Positive dimensions required')

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()


class HorizonValue(nn.Module):
    def __init__(self,contract,hidden=64):
        super().__init__();self.contract=contract
        self.network=nn.Sequential(nn.Linear(contract.feature_dim+1,hidden),nn.SiLU(),nn.Linear(hidden,1))

    def forward(self,features,horizon_left,*,contract_id):
        if contract_id!=self.contract.identity:raise ValueError('Value/deployment policy or representation mismatch')
        if features.ndim!=2 or features.shape[-1]!=self.contract.feature_dim:raise ValueError('Wrong features')
        h=torch.as_tensor(horizon_left,device=features.device)
        if h.shape!=(len(features),) or (h<0).any() or (h>self.contract.max_horizon).any() or (h!=h.floor()).any():
            raise ValueError('Explicit integer remaining horizon required for every state')
        x=torch.cat([features,(h.to(features.dtype)/self.contract.max_horizon)[:,None]],-1)
        # Nonterminal states at exhausted horizon have zero future success.
        return torch.where(h>0,self.network(x).squeeze(-1).sigmoid(),torch.zeros_like(h,dtype=features.dtype))

    def validate_label(self,record):
        if record.get('contract_id')!=self.contract.identity:raise ValueError('Mixed rollout policy labels')
        h=record.get('horizon_left');outcome=record.get('terminal_success')
        if not isinstance(h,int) or not 0<=h<=self.contract.max_horizon:raise ValueError('Invalid label horizon')
        if outcome not in (0,1) or (h==0 and outcome!=0):raise ValueError('Invalid nonterminal value outcome')
        if record.get('terminal',False):raise ValueError('Terminal leaf values use the reward backup contract, not this head')
        return True


def search_state_key(*,env_version,state,horizon_left,policy_version,history_context='',strategy_context=''):
    if horizon_left<0:raise ValueError('Negative remaining horizon')
    # Include policy context even when environment dynamics are Markov: a
    # history-conditioned policy need not give the same proposal/value at s.
    payload=dict(env=env_version,state=state,horizon=horizon_left,policy=policy_version,
                 history=history_context,strategy=strategy_context)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
