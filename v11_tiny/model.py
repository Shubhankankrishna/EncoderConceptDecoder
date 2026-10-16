"""V11 tiny test bed: the arms of v11.md §6.6.

Arms
  C0     v10_3_tiny.model.C0 (concept-free control, same backbone)       -- inherited, unchanged
  X8     v10_3_tiny.model.V103Tiny with K = 8 (V10.3 recipe)              -- inherited, unchanged
  ORD    ordinary pre-norm transformer, 11 full-causal blocks, tied head, no Engram (parameter-matched)
  NOAUX  V11 topology (code-free two-visit core), NTP only (+ detached probe)
  K1NB   V11 topology + latent next-concept objective (L_v1, L_v2 at K = 1, L_vq), no bridge
  K1     K1NB + concept-anchored NextLat bridge (training-only)

The V11 topology reuses every inherited module (V10 encoder/Engram/E2/E3, Pooler,
EncoderConceptCRC, ConceptCore blocks, PQHead, Decoder103, ForecastReadout).
What is new: the code-free core (no strategy code, no prior; K = 1), the decoder's
pre-`outn` state as an output, normalization buffers for the target, the bridge,
and the two Gaussian log-variance scalars (v11.md §3.3, §4.2, §4.4).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from v10_10M.model import V10, Block, RMSNorm, Engram, geometry
from v10_1.model import EncoderConceptCRC
from v10_2.model import ForecastReadout, history_geometry
from v10_3_tiny.config import TinyConfig
from v10_3_tiny.model import C0, Decoder103, build as build_v103

V11_ARMS = ('NOAUX', 'K1NB', 'K1')
ARMS = ('C0', 'ORD', 'ORDE', 'X8') + V11_ARMS
TRAINING_ONLY_PREFIXES = ('probe.', 'bridge.', 'log_sigma1', 'log_sigma2')


class Decoder11(Decoder103):
    """Decoder103 unchanged, but forward also returns the pre-`outn` residual d_t (the bridge target)."""

    def __init__(self, original):
        nn.Module.__init__(self)
        self.c = original.c
        for name in ('fusion', 'src_norm', 'src_proj', 'route_norm', 'routes', 'scales', 'layers', 'outn',
                     'concept_projection', 'forecast_projection', 'readers', 'fusion_map'):
            setattr(self, name, getattr(original, name))

    def forward(self, early, final, forecast, mask, rope, u, lengths, ablate_aligned=False,
                ablate_reader=False, zero_state=False, zero_forecast_memory=False):
        t = final.shape[1]
        guidance = self.align(self.fusion_map(forecast), t)
        if ablate_aligned: guidance = torch.zeros_like(guidance)
        x = final + guidance
        cu = self.concept_projection(u); cf = self.forecast_projection(forecast)
        if zero_state: cu = torch.zeros_like(cu)
        if zero_forecast_memory: cf = torch.zeros_like(cf)
        memory = (cu + cf).flatten(1, 2)
        geom = history_geometry(t, u.shape[1], lengths, self.c.width // self.c.heads, x.device)
        enc = torch.stack([p(n(h)) for p, n, h in zip(self.src_proj, self.src_norm, (early, final))], -2)
        for i, layer in enumerate(self.layers):
            w = self.routes[i](self.route_norm[i](x)).softmax(-1)
            x = x + self.scales[i] * (enc * w.unsqueeze(-1)).sum(-2)
            x = x + layer.attn(layer.an(x), mask, rope)
            if not ablate_reader: x = x + self.readers[i](x, memory, geom)
            x = x + layer.ffn(layer.fn(x))
        return self.outn(x), x


class Bridge(nn.Module):
    """beta_{g,0} = Init(flatten u1_g); beta_j = beta_{j-1} + MLP(RMSNorm((W_h beta) * sigmoid(W_e sg[E x]))) (v11.md §3.3)."""

    def __init__(self, d):
        super().__init__()
        self.init = nn.Linear(2 * d, d)
        self.wh, self.we = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.norm = RMSNorm(d)
        self.up, self.down = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def step(self, beta, token):
        c = self.wh(beta) * torch.sigmoid(self.we(token))
        return beta + self.down(F.gelu(self.up(self.norm(c))))


class V11Tiny(V10):
    def __init__(self, arm, c=None):
        if arm not in V11_ARMS: raise ValueError(arm)
        super().__init__(c or TinyConfig())
        c = self.c; self.arm = arm
        # Code-free core (K = 1): no strategy codes, no code projection, no prior.
        del self.core.codes, self.core.code_projection, self.core.selector, self.core.selector_norm
        self.encoder_concept_crc = EncoderConceptCRC(c)
        self.decoder = Decoder11(Decoder103(self.decoder, c))
        self.probe = ForecastReadout(c)                     # detached diagnostic (v11.md §4.5)
        self.bridge = Bridge(c.width) if arm == 'K1' else None
        if arm != 'NOAUX':
            self.log_sigma1 = nn.Parameter(torch.zeros(()))   # visit-1 Gaussian kernel width
            self.log_sigma2 = nn.Parameter(torch.zeros(()))   # visit-2 (K = 1 mixture) kernel width
        # Normalization buffers for the pooled target (v11.md §3.3): mean, variance, initialised flag.
        self.register_buffer('target_mu', torch.zeros(2, c.width))
        self.register_buffer('target_var', torch.ones(2, c.width))
        self.register_buffer('target_ready', torch.zeros((), dtype=torch.bool))
        self.memory_era = None

    def place(self, device):
        super().place(device); self.engram.canonical = self.engram.canonical.cpu(); return self

    # ---- concept core -------------------------------------------------------------------------
    def concept(self, pooled, lengths, sources, visits=2):
        """Visit v: s <- s + alpha*a; s <- ECCRC(s); s <- blocks(s); s <- exit_norm(s). a is code-free."""
        b, g, p, d = pooled.shape
        x = pooled.reshape(b, g * p, d)
        a = self.core.anchor_norm(x)
        mask, rope = geometry(g * p, lengths // 8 * p, x.device, d // self.c.heads, group_slots=p)
        z = sources.reshape(b, g * p, 2, d)
        s, states = a, []
        for _ in range(visits):
            s = self.visit(s, a, z, mask, rope)
            states.append(s.reshape(b, g, p, d))
        return states, a.reshape(b, g, p, d), (mask, rope)

    def visit(self, s, a, z, mask, rope):
        s = s + self.core.anchor_scale * a
        s = self.encoder_concept_crc(s, z)
        for layer in self.core.layers: s = layer(s, mask, rope)
        return self.core.exit_norm(s)

    def normalized_target(self, pooled):
        return ((pooled - self.target_mu) * torch.rsqrt(self.target_var.clamp_min(.01))).detach()

    @torch.no_grad()
    def update_target_buffers(self, pooled, valid, momentum=.99):
        """EMA over complete groups of the committed pass; training mode only; called after backward."""
        if not self.training: return
        x = pooled.detach()[valid]                           # [n, 2, d]
        if not len(x): return
        m, v = x.mean(0), x.var(0, unbiased=False)
        if not bool(self.target_ready):
            self.target_mu.copy_(m); self.target_var.copy_(v.clamp_min(.01)); self.target_ready.fill_(True); return
        self.target_mu.mul_(momentum).add_((1 - momentum) * m)
        self.target_var.mul_(momentum).add_((1 - momentum) * ((x - self.target_mu) ** 2).mean(0))

    # ---- forward ------------------------------------------------------------------------------
    def forward(self, ids, lengths=None, indices=None, visits=2, ablate=False, ablate_aligned=False,
                ablate_reader=False, zero_state=False, zero_forecast_memory=False, use_visit1=False,
                mean_substitution=None):
        if lengths is None:
            lengths = torch.full((len(ids),), ids.shape[1], device=ids.device, dtype=torch.long)
        early, h, mask, rope = self.encode(ids, lengths, indices)
        pooled = self.pooler(h)
        sources = self.encoder_concept_crc.sources(self.pooler(early), pooled)
        states, anchor, core_geom = self.concept(pooled, lengths, sources, visits)
        u1, u = states[0], states[-1]
        if use_visit1: u = u1
        forecast, probs = self.head(u)
        f_dec, u_dec = forecast, u
        if mean_substitution is not None:
            f_dec, u_dec = mean_substitution[0].expand_as(forecast), mean_substitution[1].expand_as(u)
        hidden, pre = self.decoder(early, h, f_dec, mask, rope, u_dec, lengths,
                                   ablate_aligned=ablate or ablate_aligned, ablate_reader=ablate or ablate_reader,
                                   zero_state=zero_state, zero_forecast_memory=zero_forecast_memory)
        valid = torch.arange(pooled.shape[1], device=ids.device)[None, :] < (lengths // 8)[:, None]
        return dict(logits=F.linear(hidden, self.embedding.weight), pre_outn=pre, pooled=pooled, sources=sources,
                    anchor=anchor, u1=u1, u=u, states=states, forecast=forecast, probs=probs, valid_groups=valid,
                    core_geometry=core_geom, codes=None, correction=None)

    def visit2_from_stopped(self, out):
        """K = 1 visit-2 recompute from stopped committed inputs (u1, anchor, E->C sources).

        L_v2 must not reach the encoder, pooler or visit 1 (v11.md §4.2). Deviation from §5.2, logged in
        the report: every row is recomputed (not only the sampled ones against a stopped committed KV), so
        the shared core weights also receive gradient through earlier rows' keys/values.
        """
        b, g, p, d = out['u1'].shape
        mask, rope = out['core_geometry']
        s = self.visit(out['u1'].detach().reshape(b, g * p, d), out['anchor'].detach().reshape(b, g * p, d),
                       out['sources'].detach().reshape(b, g * p, 2, d), mask, rope)
        return s.reshape(b, g, p, d)


class Ordinary(nn.Module):
    """Plain pre-norm transformer: tied embedding, `layers` full-causal Blocks with RoPE, final RMSNorm."""

    def __init__(self, c=None, layers=11):
        super().__init__()
        self.c = c or TinyConfig()
        self.embedding = nn.Embedding(self.c.vocab_size, self.c.width)
        nn.init.normal_(self.embedding.weight, std=.02)
        self.layers = nn.ModuleList([Block(self.c) for _ in range(layers)])
        self.outn = RMSNorm(self.c.width)

    def place(self, device): return self.to(device)

    def forward(self, ids, lengths=None, indices=None):
        if lengths is None:
            lengths = torch.full((len(ids),), ids.shape[1], device=ids.device, dtype=torch.long)
        mask, rope = geometry(ids.shape[1], lengths, ids.device, self.c.width // self.c.heads)
        x = self.embedding(ids)
        for layer in self.layers: x = layer(x, mask, rope)
        return dict(logits=F.linear(self.outn(x), self.embedding.weight))


class OrdinaryEngram(Ordinary):
    """Ordinary + the inherited Engram after block 1 (same position as in E1): the fair plain-model control,
    since every ECD arm also carries the 1.05M-row n-gram table outside its dense count."""

    def __init__(self, c=None, layers=11):
        super().__init__(c, layers); self.engram = Engram(self.c)

    def place(self, device):
        self.to(device); self.engram.table.cpu(); self.engram.canonical = self.engram.canonical.cpu(); return self

    def forward(self, ids, lengths=None, indices=None):
        if lengths is None:
            lengths = torch.full((len(ids),), ids.shape[1], device=ids.device, dtype=torch.long)
        mask, rope = geometry(ids.shape[1], lengths, ids.device, self.c.width // self.c.heads)
        x = self.embedding(ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, mask, rope)
            if i == 0: x = self.engram(x, ids, indices)
        return dict(logits=F.linear(self.outn(x), self.embedding.weight))


def build(arm, c=None):
    if arm in ('C0', 'X8'): return build_v103(arm, c)
    if arm == 'ORD': return Ordinary(c)
    if arm == 'ORDE': return OrdinaryEngram(c)
    return V11Tiny(arm, c)


def counts(model):
    table = model.engram.table.weight.numel() if hasattr(model, 'engram') else 0
    total = sum(p.numel() for p in model.parameters())
    training_only = sum(p.numel() for n, p in model.named_parameters()
                        if n.startswith(TRAINING_ONLY_PREFIXES + ('forecast_readout.', 'chunk_readout.')))
    return dict(dense=total - table, engram=table, dense_training_only=training_only,
                dense_deployed=total - table - training_only)
