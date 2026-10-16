"""V14's additions to V13's accounting (v13_design/accounting.py), for v14_Architecture.md §10.

    python3 v14_design/accounting_v14.py          # prints the table and writes v14_design/accounting_v14.json

Everything here is accounting from V13's own formulas, not measurement. Per tier it adds:
  the EMA twin (full: embedding, E1, Engram projections, E2, E3, exits; local: embedding, E1, Engram projections,
  half of the exit reads), with its weight memory in bf16 and in an fp32 master copy;
  the target projectors T_loc (2d -> d/2) and T_glob (d -> d/2) and their EMA copies;
  the residue map's statistics A (2d x 2d) and B (2d x d) in fp32, and one Cholesky solve for P;
  the horizon embeddings (4 x d) and the one shared head (V13's NCP head, d -> 2d -> S*N);
  SIGReg's per-group cost (M = 256 slices x d multiply-adds, plus 17 knots x M characteristic-function terms);
  the arms: the successor head's target network (an EMA copy of the concept module and a head) and the
  inverse-dynamics heads.
Training FLOPs per token are taken as 3x the forward FLOPs (forward + backward); the twin adds one forward of its
modules per token (no backward).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'v13_design'))
import accounting as A                                          # noqa: E402

TIERS = A.TIERS + [A.Tier('tiny-v2 shape (formula)', d=128, vocab=2048, heads=4, head_dim=32, e1_blocks=2,
                          e2_layers_M=3, e2_layers_F=1, e2_loops=2, e3_blocks=1, n_c=4, n_d=6, w_e=8, w_d=16,
                          ffn_hidden=256, t_train_min=1024, engram_slots=16 * 8191, engram_dim=8, tied=True)]


def v14(t):
    c = A.count(t)
    rows = c['rows']
    tot = {k: sum(r[k] for r in rows.values()) for k in ('stored', 'active', 'flops')}
    emb = [k for k in rows if k == 'embedding']
    full = [k for k in rows if k.startswith(('E1', 'E2', 'E3'))] + ['encoder exit reads', 'Engram projections'] + emb
    local = [k for k in rows if k.startswith('E1')] + ['Engram projections'] + emb
    s = lambda keys, f: sum(rows[k][f] for k in keys)
    ex = rows['encoder exit reads']
    twin_full = dict(stored=s(full, 'stored'), flops=s(full, 'flops'))
    twin_local = dict(stored=s(local, 'stored') + ex['stored'] // 2, flops=s(local, 'flops') + ex['flops'] // 2)
    d, S, N = t.d, t.pq_segments, t.pq_words
    proj = 2 * d * (d // 2) + d * (d // 2)                       # T_loc (2d -> d/2) and T_glob (d -> d/2)
    head = rows['NCP head']['stored']
    out = dict(name=t.name, d=d, S=S, N=N, base_stored=tot['stored'], base_active=tot['active'],
               base_fwd_flops=tot['flops'],
               twin_full_params=twin_full['stored'], twin_full_share_stored=twin_full['stored'] / tot['stored'],
               twin_full_fwd_flops=twin_full['flops'], twin_full_share_train_flops=twin_full['flops'] / (3 * tot['flops']),
               twin_local_params=twin_local['stored'],
               twin_local_share_train_flops=twin_local['flops'] / (3 * tot['flops']),
               twin_full_bf16_GB=2 * twin_full['stored'] / 1e9, twin_full_fp32_master_GB=4 * twin_full['stored'] / 1e9,
               projectors=proj, projectors_with_ema=2 * proj,
               residue_stats_fp32_GB=4 * ((2 * d) ** 2 + 2 * d * d) / 1e9,
               cholesky_flops=(2 * d) ** 3 / 3,
               horizon_embeddings=4 * d, shared_head=head, four_separate_heads=4 * head,
               sigreg_flops_per_group=2 * 256 * d + 2 * 17 * 256 * 2,
               sigreg_share_per_token=(2 * 256 * d + 2 * 17 * 256 * 2) / t.k / (3 * tot['flops']),
               arm_successor_target_net=rows['concept module attention']['stored'] + rows['concept module FFN']['stored'] + head,
               arm_idm_heads=(d * d + d * (d // 2)) + d * (d // 2))
    return out


def main():
    res = [v14(t) for t in TIERS]
    (HERE / 'accounting_v14.json').write_text(json.dumps(res, indent=1))
    for r in res:
        print(f"{r['name']}: base {A.fmt(r['base_stored'])} stored, {A.fmt(r['base_active'])} active, "
              f"{A.fmt(r['base_fwd_flops'])} fwd FLOPs/token")
        print(f"  twin full {A.fmt(r['twin_full_params'])} ({r['twin_full_share_stored']:.1%} of stored), "
              f"+{r['twin_full_share_train_flops']:.1%} training FLOPs; bf16 {r['twin_full_bf16_GB']:.3g} GB, "
              f"fp32 master {r['twin_full_fp32_master_GB']:.3g} GB")
        print(f"  twin local {A.fmt(r['twin_local_params'])}, +{r['twin_local_share_train_flops']:.1%} training FLOPs")
        print(f"  projectors {A.fmt(r['projectors'])} (+EMA {A.fmt(r['projectors_with_ema'])}); residue stats "
              f"{r['residue_stats_fp32_GB']:.3g} GB fp32; Cholesky {r['cholesky_flops']:.3g} FLOPs")
        print(f"  shared head {A.fmt(r['shared_head'])} vs four heads {A.fmt(r['four_separate_heads'])}; "
              f"SIGReg {r['sigreg_share_per_token']:.2e} of training FLOPs per token")
        print(f"  arms: successor target network {A.fmt(r['arm_successor_target_net'])}, "
              f"inverse-dynamics heads {A.fmt(r['arm_idm_heads'])}")


if __name__ == '__main__':
    main()
