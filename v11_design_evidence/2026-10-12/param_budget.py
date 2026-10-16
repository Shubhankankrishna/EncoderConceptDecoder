"""Verifiable parameter arithmetic for the V11 configurations.

  python -B v11_design_evidence/2026-10-12/param_budget.py   (from the repo root)

Inherited modules are counted by instantiating the existing classes (V10 /
V10.2 / V10.3-tiny) with a config object; nothing is trained. Modules that are
new in V11 (bridge, visit-1 head is shared so free, mixture log-variance, LoRA)
are built here as plain nn.Modules with the shapes given in v11.md, purely to
count them. Writes param_budget.json next to this script.
"""
import json, sys
from dataclasses import replace
from pathlib import Path
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v10_10M.config import ModelConfig
from v10_3_tiny.config import TinyConfig
from v10_3_tiny.model import V103Tiny, C0
from v10_10M.model import RMSNorm, Block

OUT = Path(__file__).with_name('param_budget.json')


class Bridge(nn.Module):
    """Training-only concept-anchored latent bridge (v11.md §3.3, "Bridge"): Init, glu_cross, residual MLP."""
    def __init__(self, d, mult=4):
        super().__init__()
        self.init = nn.Linear(2 * d, d)
        self.wh, self.we = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.norm = RMSNorm(d)
        self.up, self.down = nn.Linear(d, mult * d), nn.Linear(mult * d, d)


def n(m):
    return sum(p.numel() for p in m.parameters())


def lora_count(c, rank=4):
    # Rank-r LoRA on every attention (qkv, out) and FFN (up, down) matrix of E2, concept core and decoder blocks.
    d = c.width
    per_block = rank * ((d + 3 * d) + (d + d) + (d + 4 * d) + (4 * d + d))
    return per_block * (c.e2_layers + c.concept_layers + c.decoder_layers)


def summarize(name, c, ordinary_layers=None):
    m = V103Tiny(c)
    table = m.engram.table.weight.numel()
    total = n(m)
    probe = n(m.forecast_readout)           # kept only as a detached probe in V11 (training-only)
    bridge = n(Bridge(c.width))             # training-only
    log_sigma = 2                           # log sigma^2 (L_mix) and log sigma_1^2 (L_v1); used only by losses
    modules = {k: n(getattr(m, k)) for k in ('embedding', 'engram', 'e1', 'e2', 'exit_reader', 'e3', 'pooler',
                                            'encoder_concept_crc', 'core', 'head', 'decoder', 'forecast_readout')}
    modules['engram_dense_part'] = modules['engram'] - table
    modules['engram_table'] = table
    decoder_inherited = n(m.decoder.layers) + n(m.decoder.outn) + n(m.decoder.src_norm) + n(m.decoder.src_proj) + n(m.decoder.route_norm) + n(m.decoder.routes) + m.decoder.scales.numel()
    modules['decoder_concept_paths'] = modules['decoder'] - decoder_inherited
    dense = total - table
    deployed = dense - probe
    v11_dense = dense + bridge + log_sigma
    v11_deployed = deployed                  # bridge, probe and both log-sigmas are training-only
    c0 = C0(c); c0_dense = n(c0) - c0.engram.table.weight.numel()
    out = dict(config=c.to_dict(),
               modules=modules,
               v10_3_shape=dict(dense=dense, training_only=probe, deployed=deployed),
               v11=dict(dense_including_training_only=v11_dense, training_only=dict(forecast_probe=probe, bridge=bridge, log_sigmas=log_sigma),
                        deployed=v11_deployed, engram_table=table, under_10M=v11_dense < 10_000_000),
               c0_same_backbone=dict(dense=c0_dense, engram_table=c0.engram.table.weight.numel()),
               lora_rank4_per_phase=lora_count(c),
               executed_blocks=dict(encoder=c.e1_layers + 2 * c.e2_layers + c.e3_layers, encoder_unique=c.e1_layers + c.e2_layers + c.e3_layers,
                                    concept=2 * c.concept_layers, concept_unique=c.concept_layers, decoder=c.decoder_layers))
    if ordinary_layers:
        d, v = c.width, c.vocab_size
        blk = n(Block(c))
        out['ordinary_transformer'] = {str(L): dict(dense=v * d + L * blk + d, layers=L) for L in ordinary_layers}
        out['ordinary_block_params'] = blk
    print(name, json.dumps({k: out[k] for k in ('v10_3_shape', 'v11', 'c0_same_backbone', 'lora_rank4_per_phase', 'executed_blocks')}, indent=1))
    print(' modules', modules)
    return out


def main():
    torch.manual_seed(0)
    ref = ModelConfig(e3_layers=2)                                  # width 128, 4/4/2, core 4, decoder 4
    ref.validate()
    tiny = TinyConfig()                                             # V10.3-tiny shape: width 56, 2/1/1, core 2, decoder 2
    tiny_ref = TinyConfig(e1_layers=4, e2_layers=4, e3_layers=2, concept_layers=4, decoder_layers=4)  # reference depths at width 56
    wide = replace(ModelConfig(e3_layers=2), decoder_layers=8)       # decoder-heavier stress case (labeled comparison)
    result = dict(reference_10M=summarize('reference_10M', ref, ordinary_layers=[18, 20, 22, 24, 26, 28]),
                  tiny_v10_3_shape=summarize('tiny_v10_3_shape', tiny, ordinary_layers=[6, 8, 10]),
                  tiny_reference_depths=summarize('tiny_reference_depths', tiny_ref),
                  decoder8_stress=summarize('decoder8_stress', wide))
    OUT.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# Analytic multiply-accumulates (MACs) per token. FLOPs ~= 2 x MACs forward,
# ~= 6 x MACs forward+backward for trained weights. Attention score/value terms
# use window w for sliding blocks and the mean causal context T/2 for full
# blocks. Group-rate work is divided by 8 (one group per 8 tokens).
# ---------------------------------------------------------------------------
def macs_per_token(c, T=512, S=4, K=8, ordinary=None):
    d, V, w, r = c.width, c.vocab_size, c.window, c.gate_rank
    blk = 12 * d * d
    def attn(full):
        return 2 * (T / 2 if full else min(w, T)) * d
    schedule = []
    for i in range(c.e1_layers): schedule.append(i % 4 == 3)
    e2 = [((c.e1_layers + i) % 4 == 3) for i in range(c.e2_layers)]
    e3 = [((c.e1_layers + c.e2_layers + i) % 4 == 3) for i in range(c.e3_layers)]
    enc = sum(blk + attn(f) for f in schedule) + 2 * sum(blk + attn(f) + 2 * (4 * d * r + 2 * d * 2) for f in e2) \
        + (4 * d * r) + sum(blk + attn(f) for f in e3)
    engram = 2 * (4 * c.engram_dim) * d
    pooler = 2 * (36 * d * d) / 8                       # early and final pooled sources
    eccrc = (2 * 2 * d * d) / 8
    G = T / 8
    core = 2 * c.concept_layers * (2 * (blk + 2 * (G) * d)) / 8 + (2 * c.code_width * d) / 8
    head = 2 * (d * c.pq_segments * c.pq_words + c.pq_segments * c.pq_words * d // c.pq_segments) / 8
    dec = c.decoder_layers * (blk + attn(True)) + 2 * d * d \
        + c.decoder_layers * (2 * d * d + 2 * (2 * G / 2) * d) + (2 * 2 * d * d) / 8 + c.decoder_layers * (2 * 2 * 2 * d * d) / 8 + (2 * d * d) / 8
    vocab = d * V
    deployed = enc + engram + pooler + eccrc + core + head + dec + vocab
    c0 = enc + engram + c.decoder_layers * (blk + attn(True)) + 2 * d * d + vocab
    bridge = (2 * d * d) / 8 + 2 * d * d + 8 * d * d
    kl_head = d * V                                     # student logits; teacher logits reuse the main head output
    probe = (S * 8 / T) * (d * d + 8 * d * d + d * V)   # detached probe on S sampled boundaries per document
    variants = (K * S * 2 / T) * (c.concept_layers * (blk + 2 * G * d) + d * c.pq_segments * c.pq_words)
    train_only = bridge + kl_head + probe + variants
    out = dict(T=T, S=S, K=K, deployed=deployed, c0=c0, training_only=dict(bridge=bridge, kl_head=kl_head, probe=probe, mixture_variants=variants, total=train_only),
               training_total=deployed + train_only, train_overhead_vs_deployed=train_only / deployed)
    if ordinary:
        out['ordinary'] = {str(L): L * (blk + attn(True)) + vocab for L in ordinary}
        target = deployed
        out['compute_matched_ordinary_layers'] = min(ordinary, key=lambda L: abs(L * (blk + attn(True)) + vocab - target))
    return out


def gate_params(c):
    d = c.width
    sites = [d, d, d, 4 * d]                            # inputs of qkv, attn-out, ffn-up, ffn-down
    per_block = sum(4 * (2 * x + 1) for x in sites)     # one 4-component diagonal GMM per site
    visits = 2 * c.e2_layers + 2 * c.concept_layers + c.decoder_layers
    return dict(per_gmm_block=per_block, gmms_per_phase_pos=visits * len(sites), floats_per_phase_pos_and_neg=2 * visits * per_block)


if __name__ == '__main__':
    ref = ModelConfig(e3_layers=2)
    tiny = TinyConfig()
    extra = dict(reference_10M=dict(macs=macs_per_token(ref, T=384, S=3, ordinary=list(range(16, 30))), gates=gate_params(ref)),
                 tiny_v10_3_shape=dict(macs=macs_per_token(tiny, T=256, S=2, ordinary=list(range(4, 14))), gates=gate_params(tiny)))
    data = json.loads(OUT.read_text()); data['compute'] = extra; OUT.write_text(json.dumps(data, indent=1) + '\n')
    print(json.dumps(extra, indent=1))
