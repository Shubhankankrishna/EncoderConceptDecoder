"""Thin frozen wrapper around pinned HazyResearch NCM source, with explicit MPS area pooling.

No codec weights or quantizer buffers are trained. The MPS area implementation is
the adaptive-average definition (floor/ceil bins), tested against PyTorch CPU.
"""
import hashlib
import json
import math
import sys
import subprocess
import types
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'references/ncm'
ASSETS = ROOT / 'artifacts/codec/tinystories'
SOURCE_REV = 'c70274273f489abc73832b2b98584a8483b6b203'
WEIGHTS_REV = 'd342ce8aaea58830eef3b8ad473664673e41f692'
WEIGHT_SHA = {'lm1b':'d88aa17b66d730f2cb1fb31c5dbe63888a10da47ed8bea29dcb1f89423c3059c',
              'tinystories':'76e26003719db32dc3d542518951c286be07d6a5881f29ad5152d299764b0f81'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def state_hash(module):
    h = hashlib.sha256()
    for name, t in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def area_matrix(n, m, device='cpu'):
    w = torch.zeros(m, n, device=device)
    for i in range(m):
        lo, hi = math.floor(i * n / m), math.ceil((i + 1) * n / m)
        w[i, lo:hi] = 1. / (hi - lo)
    return w


class FrozenNCMCodec:
    def __init__(self, device='cpu', mps_area=True, family='tinystories'):
        sys.path.insert(0, str(SOURCE)) if str(SOURCE) not in sys.path else None
        from tokenization.bpe_tokenizer import BPETokenizer
        from models.vqvae import VQVAE
        self.family = family
        self.assets = ROOT / 'artifacts/codec' / family
        actual_revision=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
        assert actual_revision==SOURCE_REV,'Unexpected upstream NCM revision'
        assert sha256(self.assets/'tokenizer.pt')==WEIGHT_SHA[family],'Checkpoint checksum differs from pinned HF LFS object'
        self.config = OmegaConf.load(self.assets / ('small/config.yaml' if family == 'lm1b' else 'config.yaml'))
        cfg = self.config.tokenizer
        cfg.pre_tokenizer.vocab_file_path = str(ASSETS / 'bpe_vocab_8192.pickle')
        cfg.pre_tokenizer.merges_file_path = str(ASSETS / 'bpe_merges_8192.pickle')
        # Assets are downloaded only from the pinned official publisher.
        if family == 'tinystories':
            self.bpe = BPETokenizer.from_config(cfg.pre_tokenizer)
        else:
            from tokenization.hf_tokenizer import HFTokenizer
            bert=self.assets/'bert-tokenizer'
            self.bpe = HFTokenizer(str(bert) if bert.exists() else cfg.pre_tokenizer.model_id,
                                   **dict(cfg.pre_tokenizer.special_tokens))
        self.model = VQVAE(**cfg.vqvae)
        ckpt = torch.load(self.assets / 'tokenizer.pt', map_location='cpu', weights_only=True)
        self.model.load_state_dict(ckpt['model'], strict=True)
        del ckpt
        self.model.eval().requires_grad_(False)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.levels = tuple(int(n) for n in self.model.vq.levels)
        self.offsets = [0]
        for n in self.levels:
            self.offsets.append(self.offsets[-1] + n)
        self.vocab_sizes = tuple(self.model.vq.vqvae_vocab_sizes)
        self.block_size = self.model.context_length
        self.pad, self.eos = self.bpe.pad_token_id, self.bpe.eos_token_id
        assert self.offsets[-1] == 641
        assert len(set(self.vocab_sizes)) == 1
        self.mps_area = device == 'mps' and mps_area
        if self.mps_area:
            quant = self.model.vq
            matrices = {n: area_matrix(self.block_size, n, self.device) for n in self.levels}
            original = quant._down_for_quant

            def down(q, z, level_idx, target_len):
                if level_idx == 0 or q.learned_all_sampling:
                    return original(z, level_idx, target_len)
                return torch.matmul(matrices[int(target_len)], z)

            quant._down_for_quant = types.MethodType(down, quant)
        self.identity = self.manifest()

    def manifest(self):
        files = {str(p.relative_to(self.assets)): sha256(p) for p in sorted(self.assets.rglob('*'))
                 if p.is_file() and '.cache' not in p.parts}
        manifest = dict(source_url='https://github.com/HazyResearch/ncm',
                        source_commit=SOURCE_REV, weights_repo='hazyresearch/ncm-checkpoints',
                        weights_revision=WEIGHTS_REV, files=files, levels=self.levels,
                        offsets=self.offsets, vocab_sizes=self.vocab_sizes, block_size=self.block_size,
                        pad_id=self.pad, eos_id=self.eos, dtype='float32',
                        frozen_parameters=sum(p.numel() for p in self.model.parameters()),
                        frozen_buffers=sum(p.numel() for p in self.model.buffers()),
                        decoder_mode='greedy', mps_area=self.mps_area, family=self.family)
        manifest['source_files']={str(p.relative_to(SOURCE)):sha256(p)
                                  for folder in ('models','tokenization','utils') for p in sorted((SOURCE/folder).glob('*.py'))}
        manifest['codec_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        return manifest

    def tokenize(self, texts):
        ids, lengths = [], []
        for text in texts:
            row = self.bpe.encode(text)
            if self.family == 'tinystories':
                row += [self.eos]
            if len(row) > self.block_size:
                raise ValueError(f'Block exceeds {self.block_size} BPE IDs: {len(row)}')
            lengths.append(len(row))
            ids.append(row + [self.pad] * (self.block_size - len(row)))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(lengths)

    @torch.inference_mode()
    def encode_ids(self, ids):
        assert ids.ndim == 2 and ids.shape[1] == self.block_size
        # Official z_to_idx computes unused area-resized intermediates unsupported
        # on some MPS versions. Reproduce its exact code path without those extras.
        q = self.model.vq
        rest = self.model._encode_pre_quant(ids.to(self.device)).float().clone()
        out = []
        for i, n in enumerate(self.levels):
            z = q._down_for_quant(rest, i, n) if i < len(self.levels)-1 else rest
            code = q._distances(z.reshape(-1, q.embed_dim), q.embeddings[i].weight).argmin(1).view(len(ids), n)
            out.append(code)
            delta = q.embeddings[i](code)
            if i < len(self.levels)-1:
                delta = q._up_to_full(delta, i, self.block_size)
            rest -= q._apply_quant_resi(delta, i)
        return torch.cat(out, dim=1)

    @torch.inference_mode()
    def decode_logits(self, codes, max_level=None):
        codes = codes.to(self.device)
        if max_level is None or max_level == len(self.levels):
            return self.model.decode(codes)
        # Avoid upstream idx_to_zhats' aliased in-place z_hat list. Each partial
        # reconstruction uses the actual cumulative residual path at full length.
        q = self.model.vq
        z = torch.zeros(len(codes), self.block_size, q.embed_dim, device=self.device)
        for i in range(max_level):
            delta = q.embeddings[i](codes[:, self.offsets[i]:self.offsets[i+1]])
            if i < len(self.levels)-1:
                delta = q._up_to_full(delta, i, self.block_size)
            z = z + q._apply_quant_resi(delta, i)
        return self.model.decoder(z)

    def text_from_ids(self, rows):
        result = []
        for row in rows.cpu().tolist():
            if self.family == 'tinystories' and self.eos in row:
                row = row[:row.index(self.eos)]
            result.append(self.bpe.decode(row, skip_special_tokens=True))
        return result

    @torch.inference_mode()
    def decode_text(self, codes):
        return self.text_from_ids(self.decode_logits(codes).argmax(-1))

    def assert_matches(self, manifest):
        assert self.identity['files']==manifest['files'],'Codec asset identity changed'
        assert self.identity['source_commit']==manifest['source_commit']
        if 'source_files' in manifest:assert self.identity['source_files']==manifest['source_files']
