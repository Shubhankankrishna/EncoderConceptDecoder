"""Tiny V10.3 RCDL and its comparison arms.

Arms (see ARCHITECTURE.md):
  C0      encoder + Engram + decoder with encoder-depth CRC; no concept branch
  A       the unmodified V10.1 model class (v10_1.model.V101) under TinyConfig
  X8, X4  V10.3: the V10.2 RF model with (i) a full learned map in place of the
          per-coordinate forecast gain and (ii) mixture-likelihood exploration
          (losses.py); the PQ codebook is excluded from weight decay (train.py)

The V10.1/V10.2 classes are reused unchanged; only the forecast map and the
tiny-config-safe constructor are new here.
"""
import torch
from torch import nn
from v10_10M.model import V10, Decoder
from v10_1.model import V101, Architecture as ArchitectureA
from v10_2.model import V102, Decoder102, ForecastReadout, history_geometry
from v10_1.model import EncoderConceptCRC
from .config import TinyConfig


class C0(V10):
    """Concept-free control: same encoder, Engram and decoder blocks with E1/E3 CRC."""

    def __init__(self, c):
        super().__init__(c)
        del self.pooler, self.core, self.head
        self.decoder.fusion = None                      # no forecast to fuse

    def forward(self, ids, lengths=None, indices=None, **_):
        if lengths is None:
            lengths = torch.full((len(ids),), ids.shape[1], device=ids.device, dtype=torch.long)
        early, h, mask, rope = self.encode(ids, lengths, indices)
        d = self.decoder; x = h
        sources = torch.stack([p(n(z)) for p, n, z in zip(d.src_proj, d.src_norm, (early, h))], -2)
        routes = []
        for i, layer in enumerate(d.layers):
            w = d.routes[i](d.route_norm[i](x)).softmax(-1); routes.append(w.detach())
            x = x + d.scales[i] * (sources * w.unsqueeze(-1)).sum(-2)
            x = layer(x, mask, rope)
        return dict(logits=torch.nn.functional.linear(d.outn(x), self.embedding.weight), decoder_routes=routes)

    def place(self, device):
        super().place(device); self.engram.canonical = self.engram.canonical.cpu(); return self


class Decoder103(Decoder102):
    """V10.2 decoder with the aligned forecast entering through a full learned map.

    V10.1/V10.2 add `fusion * Align(F)` with a per-coordinate gain. §1.2 of the
    V10.3 design measured that gain shrinking while F itself carried 3% of its
    target's variance; a full map can rescale and rotate. Initialised to
    0.1 * identity, which equals the old gain's initial value.
    """

    def __init__(self, original, c):
        super().__init__(original, c)
        self.fusion = None
        self.fusion_map = nn.Linear(c.width, c.width, bias=False)
        with torch.no_grad():
            self.fusion_map.weight.copy_(torch.eye(c.width) * .1)

    def forward(self, early, final, forecast, mask, rope, u, lengths, ablate_pq=False,
                ablate_continuous=False, zero_continuous=False, detach_pq=False, detach_continuous=False,
                ablate_aligned=False, ablate_cross_pq=False, ablate_reader=False):
        t = final.shape[1]; f = forecast.detach() if detach_pq else forecast
        state = u.detach() if detach_continuous else u
        guidance = self.align(self.fusion_map(f), t)
        if ablate_pq or ablate_aligned: guidance = torch.zeros_like(guidance)
        x = final + guidance
        cu = self.concept_projection(state); cf = self.forecast_projection(f)
        if ablate_continuous or zero_continuous: cu = torch.zeros_like(cu)
        if ablate_pq or ablate_cross_pq: cf = torch.zeros_like(cf)
        memory = (cu + cf).flatten(1, 2)
        geom = history_geometry(t, u.shape[1], lengths, self.c.width // self.c.heads, x.device)
        enc = torch.stack([p(n(h)) for p, n, h in zip(self.src_proj, self.src_norm, (early, final))], -2)
        routes = []
        for i, layer in enumerate(self.layers):
            w = self.routes[i](self.route_norm[i](x)).softmax(-1); routes.append(w.detach())
            x = x + self.scales[i] * (enc * w.unsqueeze(-1)).sum(-2)
            x = x + layer.attn(layer.an(x), mask, rope)
            if not ablate_reader: x = x + self.readers[i](x, memory, geom)
            x = x + layer.ffn(layer.fn(x))
        return self.outn(x), routes


class V103Tiny(V102):
    """V10.2 RF forward with Decoder103; V102.__init__ is bypassed because it
    rejects any architecture other than its own default."""

    def __init__(self, c):
        V10.__init__(self, c)
        from v10_2.model import Architecture as ArchitectureRF
        self.architecture = ArchitectureRF()
        self.encoder_concept_crc = EncoderConceptCRC(self.c)
        self.decoder = Decoder103(self.decoder, self.c)
        self.forecast_readout = ForecastReadout(self.c)
        self.memory_era = None


def build(arm, c=None):
    c = c or TinyConfig()
    if arm in ('X8', 'X4'):
        c = TinyConfig(**{**c.to_dict(), 'codes': 8 if arm == 'X8' else 4})
        return V103Tiny(c)
    if arm == 'A':
        return V101(c, ArchitectureA())
    if arm == 'C0':
        return C0(c)
    raise ValueError('Unknown arm ' + arm)


def counts(model):
    table = model.engram.table.weight.numel()
    total = sum(p.numel() for p in model.parameters() if p is not None)
    training_only = sum(p.numel() for n, p in model.named_parameters()
                        if n.startswith(('forecast_readout.', 'chunk_readout.')))
    return dict(dense=total - table, engram=table, dense_training_only=training_only,
                dense_deployed=total - table - training_only)
