"""Bounded, reversible lifecycle primitives; not a trained continual learner."""
import copy
import json
from dataclasses import dataclass
from collections import OrderedDict
import torch
from torch import nn


class GatedLoRA(nn.Module):
    def __init__(self,input_dim,output_dim,rank=4):
        super().__init__()
        if rank<1:raise ValueError('Positive rank required')
        self.a=nn.Parameter(torch.randn(rank,input_dim)*.02)
        self.b=nn.Parameter(torch.zeros(output_dim,rank))

    def forward(self,x,gate):
        if ((gate<0)|(gate>1)).any():raise ValueError('Gate must be in [0,1]')
        return (x@self.a.T@self.b.T)*gate


@dataclass(frozen=True)
class PromotionMetrics:
    new_success: float
    old_mean: float
    old_worst: float
    ood_success: float


@dataclass(frozen=True)
class PromotionThresholds:
    minimum_acquisition_gain: float = .01
    maximum_old_mean_drop: float = .01
    maximum_old_worst_drop: float = .02
    maximum_ood_drop: float = .01

    def accepts(self,before,after):
        values=list(vars(before).values())+list(vars(after).values())
        if not all(0<=v<=1 for v in values):raise ValueError('Finite success fractions required')
        return (after.new_success>=before.new_success+self.minimum_acquisition_gain and
                after.old_mean>=before.old_mean-self.maximum_old_mean_drop and
                after.old_worst>=before.old_worst-self.maximum_old_worst_drop and
                after.ood_success>=before.ood_success-self.maximum_ood_drop)


class BoundedVerifiedMemory:
    """Bound JSON bytes and entry count; clear feature-dependent values on promotion."""
    def __init__(self,max_entries=1000,max_bytes=1024*1024,policy_version='base'):
        if min(max_entries,max_bytes)<1:raise ValueError('Positive memory budgets required')
        self.max_entries=max_entries;self.max_bytes=max_bytes;self.policy_version=policy_version
        self.entries=OrderedDict()

    @property
    def bytes(self):return sum(len(v) for v in self.entries.values())

    def write(self,key,record):
        if record.get('split')!='train' or record.get('verified') is not True:
            raise ValueError('Only verified training experiences may be stored')
        if record.get('policy_version')!=self.policy_version:raise ValueError('Stale memory feature/value version')
        blob=json.dumps(dict(key=key,record=record),sort_keys=True,allow_nan=False).encode()
        if len(blob)>self.max_bytes:raise ValueError('Record exceeds byte budget')
        self.entries.pop(key,None);self.entries[key]=blob
        while len(self.entries)>self.max_entries or self.bytes>self.max_bytes:self.entries.popitem(last=False)

    def reset(self,policy_version):
        self.entries.clear();self.policy_version=policy_version


class AdapterSlots(nn.Module):
    """Exactly two stored bundles: deployed and candidate/previous rollback.

    GMMs/gates belong inside each bundle, so their buffers also count. Beginning
    the next candidate consumes the previous rollback slot. No adapter append.
    """
    def __init__(self,adapter_bundle,max_stored_bytes,policy_version='base'):
        super().__init__();self.slots=nn.ModuleList([copy.deepcopy(adapter_bundle),copy.deepcopy(adapter_bundle)])
        self.active=0;self.max_stored_bytes=max_stored_bytes;self.policy_version=policy_version
        self.candidate_open=False;self.rollback_version=None
        self.slots.requires_grad_(False)
        if self.stored_bytes()>max_stored_bytes:raise ValueError('Adapter and gate bundles exceed capacity')

    def stored_bytes(self):
        return sum(t.numel()*t.element_size() for slot in self.slots for t in slot.state_dict().values())

    def begin_candidate(self):
        if self.candidate_open:raise RuntimeError('Candidate already open')
        other=1-self.active
        self.slots[other].load_state_dict(self.slots[self.active].state_dict(),strict=True)
        self.slots[other].requires_grad_(True);self.candidate_open=True;self.rollback_version=None
        return self.slots[other]

    def decide(self,before,after,*,new_version,thresholds,memory=None):
        if not self.candidate_open:raise RuntimeError('No candidate')
        other=1-self.active;candidate=self.slots[other]
        accept=thresholds.accepts(before,after)
        if self.stored_bytes()>self.max_stored_bytes:accept=False
        if not all(torch.isfinite(t).all() for t in candidate.state_dict().values()):accept=False
        if accept:
            if new_version==self.policy_version:raise ValueError('Promotion must advance policy identity')
            self.rollback_version=self.policy_version;self.active=other;self.policy_version=new_version
            if memory is not None:memory.reset(new_version)
        else:
            candidate.load_state_dict(self.slots[self.active].state_dict(),strict=True)
        self.slots.requires_grad_(False);self.candidate_open=False
        return accept

    def rollback(self,memory=None):
        if self.candidate_open or self.rollback_version is None:raise RuntimeError('Rollback slot unavailable')
        self.active=1-self.active;self.policy_version=self.rollback_version;self.rollback_version=None
        if memory is not None:memory.reset(self.policy_version)

    def forward(self,*args,**kwargs):return self.slots[self.active](*args,**kwargs)

    def get_extra_state(self):
        return dict(active=self.active,policy_version=self.policy_version,candidate_open=self.candidate_open,
                    rollback_version=self.rollback_version,max_stored_bytes=self.max_stored_bytes)

    def set_extra_state(self,state):
        if state['max_stored_bytes']!=self.max_stored_bytes:raise ValueError('Checkpoint capacity mismatch')
        for key,value in state.items():setattr(self,key,value)
        self.slots.requires_grad_(False)
        if self.candidate_open:self.slots[1-self.active].requires_grad_(True)
