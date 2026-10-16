"""V13 parameter, reach and long-context memory accounting (v13_NOTES.md §11).

    python3 v13_design/accounting.py            # prints the tables used in §11 and writes accounting.json

Counts are exact for the formulas below, which follow the module definitions in v13_NOTES.md §3–§9. They count
weight matrices and ignore norms, biases and scalar gates (all O(d) per sublayer). At scale every global or concept
attention is counted in MLA form (DeepSeek-V2 shapes); DeepSeek-V4's shared-KV attention with grouped output projection,
which §11.3 uses for the concept attentions, has a different count, so those rows are approximate. "Stored" is every parameter the model
keeps; "active" is the parameters one token's forward pass multiplies by (concept-rate layers are counted in full,
because every token's group passes through them once); "FLOPs/token" is 2 x (matmul parameters executed per token),
with concept-rate layers divided by k and looped layers multiplied by the loop count. Embedding lookups, the Engram
table lookup, attention-score FLOPs and the linear-attention recurrence are excluded from FLOPs (stated, not hidden).
"""
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class Tier:
    name: str
    d: int
    vocab: int
    heads: int
    head_dim: int
    # stage layout (stored layers)
    e1_blocks: int          # each block = [S, S, S, F]
    e2_layers_M: int        # GDN-2 layers in the E2 block
    e2_layers_F: int        # global layers in the E2 block
    e2_loops: int
    e3_blocks: int          # each block = [S', S', S', F]
    n_c: int                # concept-module layers
    n_d: int                # decoder layers
    k: int = 4              # tokens per concept
    w_e: int = 16           # encoder local window
    w_d: int = 16           # decoder local window
    w_c: int = 16           # concept -> L cross-attention window (tokens)
    t_train_min: int = 1024
    n_branch: int = 2       # branches per role in the encoder
    # FFN
    ffn: str = 'dense'      # 'dense' or 'moe'
    ffn_hidden: int = 256   # dense SwiGLU hidden
    moe_experts: int = 0
    moe_active: int = 0
    moe_shared: int = 0
    moe_latent: int = 0     # LatentMoE routed width (0 = full width)
    moe_expert_hidden: int = 0
    # attention at scale
    global_kind: str = 'full'   # 'full' or 'mla'
    mla_q_rank: int = 1536
    mla_kv_rank: int = 512
    mla_rope: int = 64
    global_nope: bool = True    # K3: NoPE on all MLA layers, so no decoupled RoPE dims are cached
    entry_dim: int = 512        # DeepSeek-V4 compressed-attention entry (shared K = V), RoPE on its last 64 dims
    indexer_dim: int = 128      # DeepSeek-V4 lightning-indexer key per entry
    concept_kv_groups: int = 8  # decoder layers sharing one concept entry: one per Block-AttnRes block
    # PQ
    pq_segments: int = 4
    pq_words: int = 32
    # Engram
    engram_slots: int = 0       # total table rows (all orders and heads)
    engram_dim: int = 0         # d_mem per row
    tied: bool = True
    mtp: bool = False
    notes: list = field(default_factory=list)


def attn_params(t: Tier, kind: str) -> int:
    """One attention sublayer incl. the G1 output gate. kind: 'swa' | 'full' | 'cross' | 'concept_read'.
    'concept_read' is the decoder's concept cross-attention at scale: its key/value down-projection is computed once
    per concept and block of layers (counted separately), so the per-layer count omits it."""
    d, h = t.d, t.heads * t.head_dim
    if kind == 'swa' or t.global_kind == 'full':
        return 4 * d * h + d * h                        # q, k, v, o + elementwise gate
    # at scale both the global layers and the concept cross-attention use the low-rank latent form
    # (MLA; for the cross-attention this is DeepSeek-V4's shared-KV MQA over compressed entries)
    # MLA (DeepSeek-V2 form) with a full-rank output gate (Kimi K3)
    q = d * t.mla_q_rank + t.mla_q_rank * t.heads * (t.head_dim + t.mla_rope)
    kv_down = 0 if kind == 'concept_read' else d * (t.mla_kv_rank + t.mla_rope)
    kv = kv_down + t.mla_kv_rank * t.heads * 2 * t.head_dim
    o = h * d
    gate = d * h
    return q + kv + o + gate


def gdn2_params(t: Tier) -> int:
    """Gated DeltaNet-2 sublayer: q, k, v, erase gate (d_k), write gate (d_v), decay (d_k), output gate, out proj."""
    d, h = t.d, t.heads * t.head_dim
    return 3 * d * h + d * h + d * h + d * h + d * h + h * d


def ffn_params(t: Tier):
    """(stored, active) for one FFN sublayer."""
    d = t.d
    if t.ffn == 'dense':
        p = 3 * d * t.ffn_hidden
        return p, p
    lat = t.moe_latent or d
    e = 3 * lat * t.moe_expert_hidden                  # one routed expert (GLU: gate, up, down)
    shared = t.moe_shared * 3 * d * t.moe_expert_hidden
    proj = 2 * d * lat if t.moe_latent else 0          # W_down, W_up of LatentMoE
    router = d * t.moe_experts
    stored = t.moe_experts * e + shared + proj + router
    active = t.moe_active * e + shared + proj + router
    return stored, active


def gr_params(t: Tier, n_read: int, n_write: int) -> int:
    """Gated-residual read (rank d/8 bottleneck over the read set) + scalar write predictor."""
    d, r = t.d, t.d // 8
    return n_read * d * r + r * n_read * d + n_write * n_read * d


def count(t: Tier) -> dict:
    d, nb = t.d, t.n_branch
    fs, fa = ffn_params(t)
    rows = {}

    def add(name, stored, active, flops_mult):
        rows[name] = dict(stored=stored, active=active, flops=2 * active * flops_mult)

    # encoder sublayer read sets: S reads L (nb), F reads L+G (2nb), M reads L+G+W (2nb+1), S' reads L+G (2nb)
    sw, fu = attn_params(t, 'swa'), attn_params(t, 'full')
    s_layer = sw + gr_params(t, nb, 2 * nb) + gr_params(t, nb, 2 * nb)      # attn + FFN residual ops
    f_layer = fu + gr_params(t, 2 * nb, nb) * 2
    sinkhorn = 2 * (nb * d * nb * nb)                                        # per layer, two roles
    e1_dense = t.e1_blocks * (3 * s_layer + f_layer + 4 * sinkhorn)
    add('E1 attention + residual ops', e1_dense, e1_dense, 1)
    add('E1 FFN', t.e1_blocks * 4 * fs, t.e1_blocks * 4 * fa, 1)

    m_layer = gdn2_params(t) + gr_params(t, 2 * nb + 1, 1) * 2
    e2f_layer = fu + gr_params(t, 2 * nb + 1, 1) * 2
    e2_dense = t.e2_layers_M * m_layer + t.e2_layers_F * e2f_layer + 2 * d * d   # + s0, loop embeddings ~O(d); exit write
    n_e2 = t.e2_layers_M + t.e2_layers_F
    add('E2 attention + residual ops (looped)', e2_dense, e2_dense, t.e2_loops)
    add('E2 FFN (looped)', n_e2 * fs, n_e2 * fa, t.e2_loops)

    sp_layer = sw + gr_params(t, 2 * nb, nb) * 2
    e3_dense = t.e3_blocks * (3 * sp_layer + f_layer + 4 * sinkhorn)
    add('E3 attention + residual ops', e3_dense, e3_dense, 1)
    add('E3 FFN', t.e3_blocks * 4 * fs, t.e3_blocks * 4 * fa, 1)
    exit_reads = 2 * gr_params(t, nb, 1)
    add('encoder exit reads', exit_reads, exit_reads, 1)

    # Engram: 16 tables (orders 2-3 x 8 heads) concatenated to d_mem; shared W_V, one W_K per branch (2*nb), conv
    if t.engram_slots:
        d_mem = 16 * t.engram_dim
        eng_proj = d_mem * d + 2 * nb * d_mem * d + 4 * d
        add('Engram projections', eng_proj, eng_proj, 1)
        rows['Engram table'] = dict(stored=t.engram_slots * t.engram_dim, active=0, flops=0)

    pool = 3 * (2 * d * d) + d * d                    # three sources x (W_Z, W_C) + W_P
    add('pooling', pool, pool, 1.0 / t.k)
    c_layer = attn_params(t, 'full') + attn_params(t, 'cross')
    add('concept module attention', t.n_c * c_layer, t.n_c * c_layer, 1.0 / t.k)
    add('concept module FFN', t.n_c * fs, t.n_c * fa, 1.0 / t.k)
    ncp_head = d * 2 * d + 2 * d * t.pq_segments * t.pq_words
    add('NCP head', ncp_head, ncp_head, 1.0 / t.k)

    d_layer = attn_params(t, 'swa') + attn_params(t, 'concept_read' if t.global_kind == 'mla' else 'cross')
    add('decoder attention', t.n_d * d_layer, t.n_d * d_layer, 1)
    add('decoder FFN', t.n_d * fs, t.n_d * fa, 1)
    if t.mtp:
        mtp = 2 * d * d + d_layer + fs
        add('MTP module', mtp, 2 * d * d + d_layer + fa, 1)

    emb = t.vocab * d
    rows['embedding'] = dict(stored=emb, active=0, flops=0)
    rows['output head'] = dict(stored=0 if t.tied else emb, active=emb, flops=2 * emb)

    if t.global_kind == 'mla':                       # decoder concept-entry projections, one set per group of layers
        cpr = t.concept_kv_groups * d * (t.entry_dim + t.indexer_dim)
        add('decoder concept entries (per release)', cpr, cpr, 1.0 / t.k)
    tot = {key: sum(r[key] for r in rows.values()) for key in ('stored', 'active', 'flops')}
    table = rows.get('Engram table', {}).get('stored', 0)
    dense_like = tot['stored'] - table - emb - (0 if t.tied else emb)
    codebook_buffers = t.pq_segments * t.pq_words * (d // t.pq_segments)

    # reach (v13_NOTES.md §3.8, §8.2); Engram span: (n_max - 1) + (kappa - 1) * delta = 2 + 9
    r_l = 3 * t.e1_blocks * (t.w_e - 1) + 2 + 9
    r_d_conf = r_l + t.w_d - 1
    r_d_own = r_l + t.n_d * (t.w_d - 1)
    inactive = tot['stored'] - tot['active']
    return dict(tier=asdict(t), rows=rows, total=tot, engram_table=table, embedding_and_head=emb * (1 if t.tied else 2),
                inactive=inactive, engram_share_of_inactive=table / inactive if inactive else 0.0,
                non_embedding_non_table=dense_like, codebook_buffers=codebook_buffers,
                reach=dict(R_L=r_l, R_D_confined=r_d_conf, R_D_own_swa=r_d_own,
                           rho_confined=r_d_conf / t.t_train_min, rho_own_swa=r_d_own / t.t_train_min))


def long_context_memory(t: Tier, T: int, bytes_per=2) -> dict:
    """Per-sequence cache at context T (values x bytes).

    - local layers (encoder S/S', decoder local attention) keep a window of K and V per layer;
    - encoder F layers keep an MLA latent per token (512 under NoPE; + 64 with decoupled RoPE); E2's F layer keeps one
      cache per loop; the tiny form keeps full K and V;
    - GDN-2 keeps a fixed d_k x d_v state per head and loop;
    - the concept module's self-attention keeps one 512-d entry per concept per layer (V4's shared-KV form);
    - the decoder's concept read keeps one 512-d entry plus one 128-d indexer key per concept per group of decoder
      layers (concept_kv_groups). Every decoder layer reads the same u_g, so a per-layer entry would only be another
      linear projection of the same vector; entries are computed once when the concept is released."""
    h = t.heads * t.head_dim
    if t.global_kind == 'mla':
        per_tok_global = t.mla_kv_rank + (0 if t.global_nope else t.mla_rope)
        per_concept_self = t.entry_dim
        per_concept_read = t.concept_kv_groups * (t.entry_dim + t.indexer_dim)
    else:
        per_tok_global = 2 * h
        per_concept_self = 2 * h
        per_concept_read = t.n_d * 2 * h                      # tiny: full K and V per decoder layer
    n_f_cache = t.e1_blocks + t.e2_layers_F * t.e2_loops + t.e3_blocks
    n_local = 3 * t.e1_blocks + 3 * t.e3_blocks
    local = n_local * t.w_e * 2 * h + t.n_d * t.w_d * 2 * h   # decoder confined: window of projected L^enc per layer
    glob = n_f_cache * T * per_tok_global
    gdn = t.e2_layers_M * t.e2_loops * t.heads * t.head_dim * t.head_dim
    concept_self = t.n_c * (T // t.k) * per_concept_self
    concept_read = (T // t.k) * per_concept_read
    vals = dict(local_windows=local, encoder_global=glob, gdn2_state=gdn, concept_self_attention=concept_self,
                decoder_concept_entries=concept_read)
    vals = {k: v * bytes_per for k, v in vals.items()}
    vals['total'] = sum(vals.values())
    return vals


TIERS = [
    # Engram everywhere: orders {2, 3} x K = 8 hash heads = 16 tables; engram_dim is one table's row width, so the
    # concatenated memory vector is d_mem = 16 * engram_dim (tiny 128, 7B 1,024, 2.8T 2,048; Engram-27B used 1,280).
    Tier('tiny (M2, §12)', d=128, vocab=1090, heads=4, head_dim=32, e1_blocks=1, e2_layers_M=3, e2_layers_F=1,
         e2_loops=2, e3_blocks=1, n_c=2, n_d=3, w_e=16, w_d=16, ffn_hidden=256, t_train_min=1024,
         engram_slots=16 * 8191, engram_dim=8, tied=True),
    Tier('7B-A1.4B (first scale tier)', d=1536, vocab=65536, heads=12, head_dim=128, e1_blocks=1, e2_layers_M=3,
         e2_layers_F=1, e2_loops=2, e3_blocks=1, n_c=6, n_d=16, w_e=64, w_d=128, t_train_min=4096, ffn='moe',
         moe_experts=56, moe_active=6, moe_shared=1, moe_expert_hidden=512, global_kind='mla', mla_q_rank=768,
         mla_kv_rank=256, entry_dim=256, engram_slots=19_100_000, engram_dim=64, tied=False, mtp=True, pq_segments=12,
         pq_words=128),
    Tier('V13-K3 (2.8T class)', d=7168, vocab=160000, heads=96, head_dim=128, e1_blocks=1, e2_layers_M=3,
         e2_layers_F=1, e2_loops=2, e3_blocks=1, n_c=16, n_d=56, w_e=64, w_d=128, t_train_min=4096, ffn='moe',
         moe_experts=768, moe_active=16, moe_shared=2, moe_latent=3584, moe_expert_hidden=3072, global_kind='mla',
         engram_slots=4_300_000_000, engram_dim=128, tied=False, mtp=True, pq_segments=56, pq_words=128,
         notes=['routed experts cut from K3\'s 896 to 768 so that Engram holds ~20% of the inactive budget (§3.5)']),
]

K3_REFERENCE = dict(total=2.78e12, active=104.2e9, layers=93, d=7168, experts=896, active_experts=16, shared=2,
                    latent=3584, expert_hidden=3072, heads=96, vocab=160000, attention='69 KDA + 24 MLA', mtp_layers=1)


def fmt(x):
    for unit, s in ((1e12, 'T'), (1e9, 'B'), (1e6, 'M'), (1e3, 'K')):
        if abs(x) >= unit:
            return f'{x / unit:.2f}{s}'
    return f'{x:.0f}'


def main():
    out = []
    for t in TIERS:
        c = count(t)
        c['memory_1M'] = long_context_memory(t, 1_000_000) if t.d >= 1024 else None
        c['memory_trainctx'] = long_context_memory(t, t.t_train_min)
        out.append(c)
        print(f'\n=== {t.name} ===')
        for name, r in c['rows'].items():
            print(f'  {name:40s} stored {fmt(r["stored"]):>9s}  active {fmt(r["active"]):>9s}  FLOPs/tok {fmt(r["flops"]):>9s}')
        T = c['total']
        print(f'  TOTAL stored {fmt(T["stored"])}  active {fmt(T["active"])}  FLOPs/token {fmt(T["flops"])}')
        print(f'  Engram table {fmt(c["engram_table"])}  embedding+head {fmt(c["embedding_and_head"])}  '
              f'non-embedding non-table {fmt(c["non_embedding_non_table"])}  codebook buffers {fmt(c["codebook_buffers"])}')
        print(f'  inactive {fmt(c["inactive"])}  Engram share of inactive {c["engram_share_of_inactive"]:.3f}')
        print('  reach', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c['reach'].items()})
        if c['memory_1M']:
            print('  per-sequence cache at 1M tokens (bytes, bf16):', {k: fmt(v) for k, v in c['memory_1M'].items()})
    Path(__file__).with_name('accounting.json').write_text(json.dumps(dict(tiers=out, k3_reference=K3_REFERENCE), indent=1))


if __name__ == '__main__':
    main()
