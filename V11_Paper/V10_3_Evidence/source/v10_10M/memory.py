"""Exact, code-partitioned RBF correction; no writes during pretraining/evaluation."""
from dataclasses import dataclass
import math
import threading
import torch
from .model import losses


@dataclass(frozen=True)
class Snapshot:
    keys: torch.Tensor
    values: torch.Tensor
    codes: torch.Tensor
    version: int
    era: str
    sigma: float

    def read(self, queries, codes, chunk=512):
        shape = queries.shape
        q = queries.reshape(-1, shape[-1])
        k = codes.reshape(-1)
        result = torch.zeros_like(q)
        for start in range(0, len(self.keys), chunk):
            keys = self.keys[start:start+chunk].to(q.device)
            values = self.values[start:start+chunk].to(q.device)
            same = k[:, None] == self.codes[start:start+chunk].to(q.device)[None, :]
            dist = (q.square().sum(-1, keepdim=True) + keys.square().sum(-1)[None, :] - 2*q@keys.T).clamp_min(0)
            weights = torch.exp(-dist/(2*self.sigma**2))*same
            result = result + weights@values
        return result.reshape(shape)


class RBFMemory:
    def __init__(self, era, width=256, capacity=4096, sigma=1.):
        if not era or sigma <= 0:
            raise ValueError('A fixed representation era and positive bandwidth are required')
        self.era, self.capacity, self.sigma = era, capacity, sigma
        self.keys = torch.empty(0, width)
        self.values = torch.empty(0, width)
        self.codes = torch.empty(0, dtype=torch.long)
        self.version, self.events = 0, set()
        self.lock = threading.Lock()

    def snapshot(self):
        # Commits replace tensors, never modify existing snapshot tensors.
        with self.lock:
            return Snapshot(self.keys, self.values, self.codes, self.version, self.era, self.sigma)

    def commit(self, snapshot, keys, codes, gradients, event_id, eta=.1, max_value_rms=.1):
        if eta <= 0 or not event_id:
            raise ValueError('Positive shared step and unique feedback event ID required')
        keys, codes, gradients = keys.detach().cpu(), codes.detach().cpu(), gradients.detach().cpu()
        if not torch.isfinite(keys).all() or not torch.isfinite(gradients).all():
            raise FloatingPointError('Nonfinite memory update')
        nonzero = gradients.abs().sum(-1) > 0
        keys, codes, gradients = keys[nonzero], codes[nonzero], gradients[nonzero]
        with self.lock:
            if event_id in self.events:
                raise ValueError('Feedback already committed')
            if snapshot.version != self.version or snapshot.era != self.era:
                raise ValueError('Stale snapshot / incompatible representation era')
            if len(self.keys)+len(keys) > self.capacity:
                raise OverflowError('Exact RBF capacity reached; no implicit eviction')
            if not len(keys):
                return dict(writes=0, version=self.version)
            maxnorm = gradients.norm(dim=-1).max().item()
            eta_eff = min(eta, max_value_rms*math.sqrt(keys.shape[-1])/(maxnorm+1e-12))
            self.keys = torch.cat((self.keys, keys))
            self.values = torch.cat((self.values, -eta_eff*gradients))
            self.codes = torch.cat((self.codes, codes))
            self.events.add(event_id)
            self.version += 1
            return dict(writes=len(keys), version=self.version, eta=eta_eff,
                        bytes=self.keys.numel()*8 + self.codes.numel()*8)

    def state_dict(self):
        return {k: getattr(self, k) for k in ('era', 'capacity', 'sigma', 'keys', 'values', 'codes', 'version', 'events')}

    @classmethod
    def from_state_dict(cls, state, expected_era):
        if state['era'] != expected_era:
            raise ValueError('Memory belongs to another backbone/tokenizer era')
        bank = cls(state['era'], state['keys'].shape[-1], state['capacity'], state['sigma'])
        for k, v in state.items():
            setattr(bank, k, v)
        return bank


def learn_episode(model, bank, ids, lengths, targets, target_mask, event_id, eta=.1,
                  concept_weight=.1, indices=None):
    """Prequential teacher-forced scoring, then frozen trajectory replay and commit.

    This is observed next-token adaptation, not a claim of free-generation success.
    No VQ loss, selector updates, imagined feedback or backbone optimizer involved.
    """
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Freeze every backbone parameter and call eval() before adaptation')
    snap = bank.snapshot()
    with torch.no_grad():
        before = model(ids, lengths, indices, bank=snap)
        pre = losses(model, before, targets, target_mask, concept_weight=concept_weight, vq_weight=0, selector_weight=0)
    if not bool(target_mask.any()) and not bool(before['valid_groups'][:, 1:].any()):
        return dict(writes=0, version=snap.version, reason='no valid targets')
    replay = model(ids, lengths, indices, codes=before['codes'], bank=snap, write_pass=True)
    loss = losses(model, replay, targets, target_mask, concept_weight=concept_weight, vq_weight=0, selector_weight=0)
    grad = torch.autograd.grad(loss['total'], replay['correction'], allow_unused=True)[0]
    if grad is None:
        return dict(writes=0, version=snap.version, reason='unused correction')
    valid = replay['valid_groups']
    result = bank.commit(snap, replay['queries'][valid], replay['codes'][valid], grad[valid], event_id, eta)
    result['prewrite_token_loss'] = float(pre['token'])
    result['prewrite_concept_loss'] = float(pre['concept'])
    return result
