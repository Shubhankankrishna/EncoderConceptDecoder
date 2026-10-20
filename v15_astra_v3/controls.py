"""Separately trained controls; matching budget is distinct from matching core."""
from dataclasses import asdict,dataclass,replace
import torch
from torch import nn
import torch.nn.functional as F
from .model import AstraHRM,Config,Block
from .gram import AstraGRAM,GramConfig


class NonrecurrentCodePolicy(AstraHRM):
    def __init__(self,c=None):
        super().__init__(c)
        for name in ('fast','slow','h_init','l_init','init_proj'):delattr(self,name)

    def reason(self,mem,retain_states=False):
        return mem.new_empty(len(mem),0,self.config.width),[]


class StateActionPolicy(nn.Module):
    """Small seven-class control; no code tables/codec, no inactive parameters."""
    def __init__(self,c=None):
        super().__init__();self.config=replace(c or Config(ff=432),context_source='state');c=self.config
        self.state_embed=nn.Embedding(128,c.width)
        self.position=nn.Parameter(torch.randn(c.state_length,c.width)*.02)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.ctx_layers)])
        self.norm=nn.LayerNorm(c.width);self.head=nn.Linear(c.width,7)

    def logits(self,state):
        if state.shape[1]!=self.config.state_length:raise ValueError('Wrong state size')
        x=self.state_embed(state)+self.position
        for block in self.blocks:x=block(x)
        return self.head(self.norm(x.mean(1)))

    def forward(self,state,action,optimal_mask=None):
        logits=self.logits(state)
        # Reference imitation stays separate from optional set-valued supervision.
        if optimal_mask is None:return F.cross_entropy(logits,action)
        if optimal_mask.shape!=logits.shape or not optimal_mask.any(-1).all():
            raise ValueError('Every target must contain at least one optimal action')
        return -torch.logsumexp(logits.log_softmax(-1).masked_fill(~optimal_mask,-torch.inf),-1).mean()

    @torch.inference_mode()
    def generate_actions(self,state,legal_mask=None):
        logits=self.logits(state)
        if legal_mask is not None:
            if legal_mask.shape!=logits.shape or not legal_mask.any(-1).all():raise ValueError('Empty legal set')
            logits=logits.masked_fill(~legal_mask,-torch.inf)
        return logits.argmax(-1)

    def parameter_count(self):return sum(p.numel() for p in self.parameters())


ARMS=('state_action','code_nonrecurrent','code_hrm','code_gram','code_hrm_parameter_matched')


def make_model(arm,config=None,inference=False):
    if arm not in ARMS:raise ValueError(f'Unknown arm: {arm}')
    values=dict(config or {})
    values.setdefault('ff',544 if arm=='code_hrm_parameter_matched' else 432)
    if arm=='code_gram':return AstraGRAM(GramConfig(**values),include_posterior=not inference)
    c=Config(**values)
    return {'state_action':StateActionPolicy,'code_nonrecurrent':NonrecurrentCodePolicy,
            'code_hrm':AstraHRM,'code_hrm_parameter_matched':AstraHRM}[arm](c)


def copy_common_initialization(reference,other):
    """Pair equal-shaped shared modules; never silently partially load a checkpoint."""
    src=reference.state_dict();dst=other.state_dict();names=[]
    for name,value in dst.items():
        if name in src and src[name].shape==value.shape:
            value.copy_(src[name]);names.append(name)
    if not names:raise ValueError('No shared backbone parameters')
    other.load_state_dict(dst,strict=True)
    return names


def parameter_ledger(model):
    params=dict(model.named_parameters());total=sum(p.numel() for p in params.values())
    training_only=sum(p.numel() for k,p in params.items() if k.startswith(('posterior.','target_encoder.')))
    tables=sum(p.numel() for k,p in params.items() if k.startswith('code.'))
    return dict(trainable=total,training_only=training_only,deployment=total-training_only,
                code_tables=tables,frozen_codec_included=False,
                requires_codec=not isinstance(model,StateActionPolicy))
