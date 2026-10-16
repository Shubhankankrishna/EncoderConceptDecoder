"""Frozen-era exact RBF updates for both V10.2 downstream routes."""
import hashlib
import json
import math
import threading
from pathlib import Path
from dataclasses import asdict
import torch
from v10_10M.memory import Snapshot
from .model import token_loss


def model_era(model,tokenizer_sha256,preprocessing_sha256,sigma=1.):
    """Fingerprint actual weights, architecture and query/read semantics once per era."""
    h=hashlib.sha256(json.dumps(dict(config=model.c.to_dict(),architecture=asdict(model.architecture),
        tokenizer=tokenizer_sha256,preprocessing=preprocessing_sha256,sigma=sigma,
        query='ordered_final_pre_memory_divide_16',write_loss='token_only_three_downstream_paths',
        implementation='v10_2_reference_v1',torch_version=torch.__version__),sort_keys=True).encode())
    root=Path(__file__).resolve().parents[1]
    for relative in ['v10_2/model.py','v10_2/memory.py','v10_10M/model.py','v10_10M/config.py']:
        h.update(relative.encode());h.update((root/relative).read_bytes())
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class RBFMemory:
    def __init__(self,era,width=256,capacity=4096,sigma=1.,codes=8):
        if not era or not math.isfinite(sigma) or sigma<=0 or width<=0 or capacity<=0 or codes<=0:
            raise ValueError('A fixed era and positive finite memory configuration are required')
        self.era,self.capacity,self.sigma,self.code_count=era,capacity,sigma,codes
        self.keys=torch.empty(0,width);self.values=torch.empty(0,width);self.codes=torch.empty(0,dtype=torch.long)
        self.version=0;self.events=set();self.lock=threading.Lock()

    def snapshot(self):
        with self.lock:return Snapshot(self.keys,self.values,self.codes,self.version,self.era,self.sigma)

    def commit(self,snapshot,keys,codes,gradients,event_id,eta=.1,max_value_rms=.1):
        if not event_id or not isinstance(event_id,str) or not math.isfinite(eta) or eta<=0 or not math.isfinite(max_value_rms) or max_value_rms<=0:
            raise ValueError('Positive finite shared step/bound and unique event ID required')
        keys,gradients=keys.detach().cpu(),gradients.detach().cpu();codes=codes.detach().cpu()
        if keys.ndim!=2 or keys.shape!=gradients.shape or keys.shape[1]!=self.keys.shape[1] or codes.shape!=(len(keys),) or codes.dtype!=torch.long:
            raise ValueError('Invalid key/value/code shapes or code dtype')
        if not torch.isfinite(keys).all() or not torch.isfinite(gradients).all():raise FloatingPointError('Nonfinite update')
        if bool(((codes<0)|(codes>=self.code_count)).any()):raise ValueError('Unknown code partition')
        nonzero=gradients.abs().sum(-1)>0;keys,codes,gradients=keys[nonzero],codes[nonzero],gradients[nonzero]
        with self.lock:
            if event_id in self.events:raise ValueError('Feedback already committed')
            if snapshot.version!=self.version or snapshot.era!=self.era or snapshot.sigma!=self.sigma:
                raise ValueError('Stale or incompatible snapshot')
            if len(self.keys)+len(keys)>self.capacity:raise OverflowError('Exact capacity reached; no eviction')
            eta_eff=eta
            if len(keys):
                maxnorm=float(gradients.norm(dim=-1).max())
                eta_eff=min(eta,max_value_rms*math.sqrt(keys.shape[1])/(maxnorm+1e-12))
                self.keys=torch.cat((self.keys,keys.to(self.keys.dtype)))
                self.values=torch.cat((self.values,(-eta_eff*gradients).to(self.values.dtype)))
                self.codes=torch.cat((self.codes,codes))
            # Even a zero-gradient feedback event is consumed and advances the version.
            self.events.add(event_id);self.version+=1
            return dict(writes=len(keys),version=self.version,eta=eta_eff,
                        bytes=(self.keys.numel()+self.values.numel())*4+self.codes.numel()*8)

    def state_dict(self):
        with self.lock:
            return dict(era=self.era,capacity=self.capacity,sigma=self.sigma,code_count=self.code_count,
                keys=self.keys.clone(),values=self.values.clone(),codes=self.codes.clone(),version=self.version,events=sorted(self.events))

    @classmethod
    def from_state_dict(cls,state,expected_era):
        if state['era']!=expected_era:raise ValueError('Wrong representation era')
        bank=cls(state['era'],state['keys'].shape[-1],state['capacity'],state['sigma'],state['code_count'])
        keys,values,codes=state['keys'],state['values'],state['codes']
        if keys.shape!=values.shape or codes.shape!=(len(keys),) or len(keys)>bank.capacity:
            raise ValueError('Invalid bank payload')
        if not torch.isfinite(keys).all() or not torch.isfinite(values).all():raise ValueError('Nonfinite bank payload')
        if codes.dtype!=torch.long or bool(((codes<0)|(codes>=bank.code_count)).any()):raise ValueError('Invalid bank code')
        if state['version']!=len(set(state['events'])) or len(state['events'])!=len(set(state['events'])):
            raise ValueError('Invalid event/version state')
        bank.keys=keys.detach().cpu().clone();bank.values=values.detach().cpu().clone();bank.codes=codes.cpu().clone()
        bank.events=set(state['events']);bank.version=state['version']
        return bank


def write_objective(output,targets,target_mask):
    return token_loss(output,targets,target_mask)


def learn_episode(model,bank,ids,lengths,targets,target_mask,event_id,expected_era,eta=.1):
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Freeze model and use eval()')
    if bank.era!=expected_era:raise ValueError('Wrong memory era')
    snapshot=bank.snapshot()
    with torch.no_grad():
        before=model(ids,lengths,bank=snapshot)
        score=float(write_objective(before,targets,target_mask))
    replay=model(ids,lengths,codes=before['codes'],bank=snapshot,write_pass=True)
    objective=write_objective(replay,targets,target_mask)
    gradient=torch.autograd.grad(objective,replay['correction'],allow_unused=True)[0] if objective.requires_grad else None
    if gradient is None:gradient=torch.zeros_like(replay['queries'])
    valid=replay['valid_groups']
    report=bank.commit(snapshot,replay['queries'][valid],replay['codes'][valid],gradient[valid],event_id,eta)
    report['prewrite_loss']=score
    return report
