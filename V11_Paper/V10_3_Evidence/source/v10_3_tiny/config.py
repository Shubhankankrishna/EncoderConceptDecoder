"""Tiny test-bed configuration for V10.3 RCDL.

The V10 module classes read their sizes from a config object and call its
validate(); v10_10M.config.ModelConfig hard-asserts width 128, so the tiny
version supplies its own config with the same attribute names. The budget
convention matches V10: the dense-parameter limit (here 600k) excludes the
sparse Engram table, which is counted and reported separately.
"""
from dataclasses import dataclass, asdict


@dataclass
class TinyConfig:
    vocab_size: int = 1024
    width: int = 56
    heads: int = 4
    e1_layers: int = 2
    e2_layers: int = 1
    e3_layers: int = 1
    encoder_visits: int = 2
    concept_layers: int = 2
    concept_visits: int = 2
    decoder_layers: int = 2
    ffn_expansion: int = 4
    window: int = 64
    group: int = 8
    slots: int = 2
    codes: int = 8
    code_width: int = 16
    gate_rank: int = 16
    pq_segments: int = 4
    pq_words: int = 16
    engram_rows: tuple = (16381, 16411, 16417, 16421)
    engram_dim: int = 16
    engram_cpu: bool = True
    context: int = 256

    def validate(self):
        assert self.ffn_expansion == 4, 'v10_10M.FFN hard-codes 4x expansion'
        assert self.encoder_visits == self.concept_visits == 2
        assert self.group == 8 and self.slots == 2 and self.codes in (4, 8)
        assert self.width % self.heads == 0 and self.width % self.pq_segments == 0
        assert (self.width // self.heads) % 2 == 0, 'RoPE needs an even head width'
        assert min(self.e1_layers, self.e2_layers, self.e3_layers, self.concept_layers, self.decoder_layers) >= 1

    def to_dict(self):
        return asdict(self)
