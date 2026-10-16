from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    width: int = 128
    heads: int = 4
    e1_layers: int = 4
    e2_layers: int = 4
    e3_layers: int = 4
    encoder_visits: int = 2
    concept_layers: int = 4
    concept_visits: int = 2
    decoder_layers: int = 4
    ffn_expansion: int = 4
    window: int = 64
    group: int = 8
    slots: int = 2
    codes: int = 8
    code_width: int = 32
    gate_rank: int = 32
    pq_segments: int = 8
    pq_words: int = 32
    engram_rows: tuple = (156241, 156253, 156257, 156259)
    engram_dim: int = 32
    engram_cpu: bool = True

    def validate(self):
        assert self.width == 128 and self.ffn_expansion == 4
        assert self.e1_layers >= 4 and self.e2_layers >= 4 and self.e3_layers >= 2
        assert (self.e1_layers + self.e2_layers + self.e3_layers) >= 10
        assert self.encoder_visits == self.concept_visits == 2
        assert self.concept_layers >= 4 and self.decoder_layers >= 4
        assert self.group == 8 and self.slots == 2 and self.codes in (4, 8)
        assert self.width % self.heads == self.width % self.pq_segments == 0

    def to_dict(self):
        return asdict(self)
