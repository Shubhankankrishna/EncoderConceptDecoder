"""Explicit length adaptation of the pinned NCM codec, never an official checkpoint.

Pretrained codebooks and decoder blocks are retained. The root downsampler's
kernel is summed in groups of four, its upsampler averaged, and positions are
interpolated. Reconstruction must pass the separately recorded acceptance gate.
"""
import hashlib
import json
from pathlib import Path
import types

import torch
from torch import nn
import torch.nn.functional as F

from ..codec import FrozenNCMCodec, state_hash
from ..isolation import sha256

LEVELS = (1, 2, 4, 6, 8, 10, 13, 16, 19, 23, 27, 32)
RECIPE = 'lm1b-native32-root-kernel-integration-position-interpolation-v1'


def resize_root(weight, *, down):
    if weight.shape[-1] != 128:
        raise ValueError('Expected original 128-position root kernel')
    result = weight.reshape(*weight.shape[:-1], 32, 4).mean(-1)
    return result * 4 if down else result


class ShortNCMCodec(FrozenNCMCodec):
    def __init__(self, device='cpu', adaptation=None):
        if device != 'cpu':
            raise ValueError('Canonical short codec is CPU-only')
        super().__init__('cpu', family='lm1b')
        origin = self.identity
        q = self.model.vq
        if q.learned_all_sampling or q.learned_l1_cascade:
            raise ValueError('Length conversion only covers the pinned simple root sampler')
        dim = q.embed_dim
        down = nn.Conv1d(dim, dim, 32, stride=32)
        up = nn.ConvTranspose1d(dim, dim, 32, stride=32)
        with torch.no_grad():
            down.weight.copy_(resize_root(q.learned_down_l1.weight, down=True))
            down.bias.copy_(q.learned_down_l1.bias)
            up.weight.copy_(resize_root(q.learned_up_l1.weight, down=False))
            up.bias.copy_(q.learned_up_l1.bias)
            position = self.model.decoder.wpe.weight
            short_position = F.interpolate(position.T[None], size=32, mode='linear', align_corners=False)[0].T
        q.learned_down_l1, q.learned_up_l1 = down, up
        self.model.decoder.wpe = nn.Embedding.from_pretrained(short_position, freeze=True)
        q.levels = list(LEVELS)
        # A stride-32 convolution over exactly 32 positions is a single
        # matrix product. The CPU convolution fallback is disproportionately
        # slow at larger batches on this M2. Keep the same registered weights.
        original_down, original_up = q._down_for_quant, q._up_to_full
        def down_root(quant, z, level_idx, target_len):
            if level_idx != 0:
                return original_down(z, level_idx, target_len)
            if z.shape[1] != 32 or target_len != 1:
                raise ValueError('Native root downsample shape changed')
            layer=quant.learned_down_l1
            return F.linear(z.transpose(1,2).reshape(len(z),-1),layer.weight.reshape(dim,-1),layer.bias)[:,None]
        def up_root(quant, z, level_idx, target_len):
            if level_idx != 0:
                return original_up(z, level_idx, target_len)
            if z.shape[1] != 1 or target_len != 32:
                raise ValueError('Native root upsample shape changed')
            layer=quant.learned_up_l1
            result=F.linear(z[:,0],layer.weight.reshape(dim,-1).T).reshape(len(z),dim,32)
            return (result+layer.bias[None,:,None]).transpose(1,2).contiguous()
        q._down_for_quant=types.MethodType(down_root,q)
        q._up_to_full=types.MethodType(up_root,q)
        self.model.context_length = self.block_size = 32
        self.levels = LEVELS
        self.offsets = [0]
        for length in LEVELS:
            self.offsets.append(self.offsets[-1] + length)
        self.model.eval().requires_grad_(False)
        adaptation_sha = None
        if adaptation:
            adaptation = Path(adaptation)
            payload = torch.load(adaptation, map_location='cpu', weights_only=True)
            if payload['recipe'] != RECIPE:
                raise ValueError('Adaptation recipe changed')
            self.model.decoder.load_state_dict(payload['decoder'], strict=True)
            adaptation_sha = sha256(adaptation)
        self.identity = dict(origin, kind='adapted_native32_ncm', official_short_checkpoint=False,
            parent_codec_id=origin['codec_id'], recipe=RECIPE,
            adapter_source_sha=sha256(Path(__file__)), adaptation_sha=adaptation_sha,
            adaptation_path=str(adaptation.resolve()) if adaptation else None,
            levels=list(LEVELS), offsets=self.offsets, block_size=32,
            frozen_parameters=sum(p.numel() for p in self.model.parameters()))
        self.identity.pop('codec_id')
        self.identity['codec_id'] = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()


def load_codec(manifest):
    if manifest['kind'] != 'adapted_native32_ncm':
        raise ValueError('This experiment requires the audited native32 codec')
    codec = ShortNCMCodec(adaptation=manifest.get('adaptation_path'))
    codec.assert_matches(manifest)
    return codec
