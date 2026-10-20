"""Pinned official NCM with sparse-level differentiable LoRA/FiLM conditioning.

The codec stays CPU canonical. Full cumulative reconstruction is retained at every
position. One implementation of level logits serves both training and generation.
"""
import contextlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .common import ROOT, WORKSPACE, sha

GENERATOR = ROOT/'assets/lm1b/small/generator.pt'
GENERATOR_SHA = 'b0789d5a2aad9114d40976d26bd1af09d8cea0b0bec7d385b6a44739f77256fe'


class LoRALinear(nn.Module):
    def __init__(self,base,rank=4):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank,base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.a,a=5**.5)
        self.enabled = True

    def forward(self,x):
        return self.base(x) + F.linear(F.linear(x,self.a),self.b) if self.enabled else self.base(x)


class ConditionedNCM(nn.Module):
    def __init__(self,base,codec,width=128,rank=4):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.codec = (codec,)
        self.level_embedding = nn.Embedding(12,16)
        self.film = nn.Sequential(nn.Linear(width+16,128),nn.SiLU(),nn.Linear(128,1536))
        nn.init.zeros_(self.film[-1].weight);nn.init.zeros_(self.film[-1].bias)
        self.adapter_names = []
        for i,block in enumerate(base.transformer.blocks):
            for name in ('c_attn','c_proj'):
                setattr(block.attn,name,LoRALinear(getattr(block.attn,name),rank))
                self.adapter_names.append(f'transformer.blocks.{i}.attn.{name}')
        self.enabled = True
        self.levels = tuple(codec.levels)
        self.offsets = tuple(codec.offsets)
        with torch.no_grad():
            prefix = codec.model.encoder.wte(torch.tensor([[codec.eos]]))
        self.register_buffer('raw_prefix',prefix.detach().clone(),persistent=False)
        self.train(False)

    def train(self,mode=True):
        super().train(mode)
        # Original dropout stays off; stochasticity is explicit in data/sampling.
        self.base.eval()
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    def level_logits(self,raw,level,condition=None):
        base = self.base
        raw = raw.to(self.device,dtype=torch.float32)
        x = base._apply_post_upsample_conv_single(raw,level)
        x = base.transformer.wte(x)
        levels = torch.full((raw.shape[1],),level,dtype=torch.long,device=self.device)
        x = x + base.transformer.wle(levels)[None,:,:]
        if condition is not None and self.enabled:
            index = torch.full((len(x),),level,dtype=torch.long,device=self.device)
            gamma,beta = self.film(torch.cat([condition,self.level_embedding(index)],-1)).chunk(2,-1)
            x = x*(1+gamma[:,None,:])+beta[:,None,:]
        prefix = base.transformer.wte(self.raw_prefix.expand(len(x),-1,-1))
        x = torch.cat([prefix,x],1)
        mask = torch.zeros(1,1,x.shape[1],x.shape[1],device=self.device)
        mask[:,:,:1,1:] = -torch.inf
        rope = (getattr(base,f'rope_cos_gen_{level}'),getattr(base,f'rope_sin_gen_{level}'))
        for block in base.transformer.blocks:
            x = block(x,attn_mask=mask,rope_cos_sin=rope)
        return base.lm_heads[level](base.transformer.ln_f(x)[:,1:])

    @torch.no_grad()
    def teacher_inputs(self,codes):
        codec = self.codec[0]
        codes = codes.cpu().long()
        inputs = codec.model.vq.idx_to_ncm_input(codes)
        query = self.base.build_l0_query_input(len(codes)).detach().cpu()
        return torch.cat([query,inputs],1)

    @torch.no_grad()
    def generate_codes(self,condition=None,temperature=1.,level0=None,batch_size=1):
        b = len(condition) if condition is not None else batch_size
        codec = self.codec[0]; q = codec.model.vq
        zhat = torch.zeros(b,128,384)
        raw = self.base.build_l0_query_input(b).detach().cpu()
        codes = []
        for level in range(12):
            if level == 0 and level0 is not None:
                idx = level0.cpu()
            else:
                logits = self.level_logits(raw,level,condition)
                if temperature == 0:
                    idx = logits.argmax(-1).cpu()
                else:
                    probs = (logits/temperature).softmax(-1).cpu()
                    idx = torch.multinomial(probs.reshape(-1,8192),1).reshape(b,self.levels[level])
            codes.append(idx)
            if level<11:
                zhat,raw = q.get_next_autoregressive_input(level,zhat,q.embeddings[level](idx))
        return torch.cat(codes,1)

    def adapter_state(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    def load_adapters(self,state):
        params = {n:p for n,p in self.named_parameters() if p.requires_grad}
        if params.keys()!=state.keys():
            raise ValueError('Adapter parameter identity mismatch')
        with torch.no_grad():
            for n,p in params.items(): p.copy_(state[n])

    @contextlib.contextmanager
    def frozen_reference(self):
        previous=self.enabled
        layers=[m for m in self.modules() if isinstance(m,LoRALinear)]
        states=[m.enabled for m in layers]
        self.enabled=False
        for m in layers:m.enabled=False
        try:
            yield
        finally:
            self.enabled=previous
            for m,state in zip(layers,states):m.enabled=state


def load_ncm(device='cpu',rank=4):
    from v15_astra_v3.codec import FrozenNCMCodec
    if sha(GENERATOR)!=GENERATOR_SHA:
        raise ValueError('Official generator checksum mismatch')
    codec = FrozenNCMCodec('cpu',family='lm1b')
    from models.next_concept import NextConceptTransformer
    base = NextConceptTransformer(vqvae_local=codec.model,config=codec.config.model)
    checkpoint = torch.load(GENERATOR,map_location='cpu',weights_only=True)
    base.load_state_dict(checkpoint['model'],strict=True)
    inference = checkpoint.get('inference',{})
    del checkpoint
    base.eval().requires_grad_(False)
    wrapper = ConditionedNCM(base,codec,rank=rank).to(device)
    return wrapper,codec,inference


def decoded_text(codec,codes,inference,official_penalties=False,normalize_boundary=False):
    logits = codec.decode_logits(codes)
    if not official_penalties:
        return codec.text_from_ids(logits.argmax(-1))
    # Published LM1B sampler uses decoder frequency/adjacency penalties, separately reported.
    lg = logits.double(); b,t,v = lg.shape
    allow = inference['token_unigram'][:v].double()[None,:]*t
    counts = torch.zeros(b,v,dtype=torch.float64)
    history = []
    for pos in range(t):
        step = lg[:,pos,:].clone() - 11.5*(counts-allow).clamp_min(0)
        step += 2.*((t-pos)/t)*(allow-counts).clamp_min(0)
        for j,prev in enumerate(reversed(history[-4:])):
            step.scatter_add_(1,prev[:,None],torch.full((b,1),-10.*.5**j,dtype=torch.float64))
        ids = step.argmax(-1);history.append(ids)
        counts.scatter_add_(1,ids[:,None],torch.ones(b,1,dtype=torch.float64))
    rows = torch.stack(history,1).tolist()
    if normalize_boundary:
        # Explicit deviation from released helper: initial CLS is a beginning marker.
        rows = [row[1:] if row and row[0]==codec.eos else row for row in rows]
        sep=codec.bpe.encode('')[-1]
        rows=[row[:next((i for i,t in enumerate(row) if t in (codec.eos,codec.pad,sep)),len(row))] for row in rows]
    else:
        rows = [row[:row.index(codec.eos)] if codec.eos in row else row for row in rows]
    return [codec.bpe.decode([t for t in row if t not in (codec.eos,codec.pad)],skip_special_tokens=True) for row in rows]
