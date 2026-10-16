"""Generate the data files of the V13 Anatomy app from their sources (nothing is typed by hand).

    python3 v13_viz/extract.py                 # CPU only, a few seconds; never loads a checkpoint
    taskpolicy -b python3 v13_viz/extract.py --journey CKPT
                                               # also journey.json: one CPU forward of a copy of a training checkpoint
                                               # on the 16-token document (consultation 5: CPU-only at background QoS
                                               # costs seconds and no GPU memory, so it no longer conflicts with training)

Writes v13_viz/data/:
  tiny_v2.json   config and exact parameter counts of v13_tiny/model.py (PyTorch numel, every module and sublayer)
  tiers.json     v13_design/accounting.py run on its own tiers plus a tiny-v2-shaped tier (the formula-vs-exact gap)
  tokens.json    one short text document tokenised by v13_tiny/tok.py (<text> + 15 story tokens = 4 concept groups)
  journey.json   (--journey) measured values on tokens.json's document: pool weights, NCP codes vs targets, the
                 concept-read attention of the last row, next-token top-5
  snapshot.json  learned values from the newest line of v13_tiny/runs/base/log.jsonl (beta, tau), with its step and time
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'v13_tiny')); sys.path.insert(0, str(ROOT / 'v13_design'))
os.chdir(ROOT / 'v13_tiny')                                   # tok.py resolves its tokenizer path from here

import model as M                                             # noqa: E402
import accounting as A                                        # noqa: E402
from tok import Tok                                           # noqa: E402

OUT = HERE / 'data'; OUT.mkdir(exist_ok=True)


def n(mod):
    return sum(p.numel() for p in mod.parameters())


def tree(mod, depth):
    """Exact parameter counts: {'total', 'own', 'children': {name: tree}} down to `depth` levels."""
    own = sum(p.numel() for p in mod.parameters(recurse=False))
    node = dict(total=n(mod), own=own)
    if depth > 0:
        node['children'] = {k: tree(v, depth - 1) for k, v in mod.named_children()}
    return node


def tiny_v2():
    m = M.V13(); c = m.c
    cfg = {k: getattr(c, k) for k in dir(c) if not k.startswith('_') and not callable(getattr(c, k))}
    cfg['eng_orders'] = list(cfg['eng_orders'])
    tot, groups = M.count_params(m)
    mods = {k: tree(v, 3) for k, v in m.named_children()}
    for k in ('s0', 'e_r', 'src_norm', 'beta'):
        mods[k] = dict(total=getattr(m, k).numel(), own=getattr(m, k).numel())
    kinds = dict(e1=[l.kind for l in m.e1], e2=[l.kind for l in m.e2], e3=[l.kind for l in m.e3],
                 cm=list(m.c_kinds), dec=list(m.d_kinds))
    return dict(source='v13_tiny/model.py, V13() parameters counted by PyTorch (numel)', total=tot, groups=groups,
                modules=mods, kinds=kinds, cfg=cfg,
                reach=dict(R_L=51, R_D=66, bound_R_L=53, bound_R_D=68, T_row=1024,
                           source='DEVIATIONS.md item 16 (exact, tests_g0.py); bound from v13_NOTES.md §3.8'))


def tiers():
    out = []
    t2 = A.Tier('tiny-v2 shape (formula)', d=128, vocab=2048, heads=4, head_dim=32, e1_blocks=2, e2_layers_M=3,
                e2_layers_F=1, e2_loops=2, e3_blocks=1, n_c=4, n_d=6, w_e=8, w_d=16, ffn_hidden=256, t_train_min=1024,
                engram_slots=16 * 8191, engram_dim=8, tied=True)
    for t in A.TIERS + [t2]:
        c = A.count(t)
        c['memory_1M'] = A.long_context_memory(t, 1_000_000) if t.d >= 1024 else None
        c['memory_trainctx'] = A.long_context_memory(t, t.t_train_min)
        out.append(c)
    return dict(source='v13_design/accounting.py count() and long_context_memory(), run by extract.py',
                tiers=out, k3_reference=A.K3_REFERENCE)


def tokens():
    tk = Tok()
    cands = ['Lily found a red ball in the garden and gave it to her brother.',
             'Tom saw a big dog in the park and he ran to his mom.',
             'The little cat sat on the mat and looked at the bird.',
             'One day, a boy named Sam found a shiny key under a tree.']
    tried = []
    for s in cands:
        ids = tk.text(s); tried.append((s, len(ids)))
        if len(ids) == 15:
            doc = [tk.sid['<text>']] + ids
            nxt = tk.text(s + ' He')[15:16]                   # a plausible next token after the sentence, shown as the target
            pieces = [tk.decode([i]) for i in doc]
            return dict(source='v13_tiny/tok.py (the training tokenizer); document = <text> + story tokens',
                        sentence=s, ids=doc, pieces=pieces, vocab=tk.vocab,
                        next_id=nxt[0] if nxt else tk.eos, next_piece=tk.decode(nxt) if nxt else '<eos>')
    raise SystemExit(f'no candidate has 15 tokens: {tried}')


def snapshot():
    log = ROOT / 'v13_tiny/runs/base/log.jsonl'
    rows = [json.loads(l) for l in open(log) if l.strip()]
    r = rows[-1]
    return dict(source='v13_tiny/runs/base/log.jsonl, newest line', step=r['step'], hours=r['h'], frac=r['frac'],
                beta=r['beta'], beta_names=['x (identity tap)', 'sg L_enc (local tap)', 'G_enc (global)'],
                tau=r['tau'], tau_names=['E1 F (block 1)', 'E1 F (block 2)', 'E2 F', 'E3 F'],
                beta_init=[0.107, 0.107, 0.787], log_time=time.strftime('%Y-%m-%d %H:%M', time.localtime(log.stat().st_mtime)))


def journey(ckpt_path):
    """One forward of the checkpoint on tokens.json's document (CPU, eval, no grad). Every number is labelled with the step."""
    import math
    import torch
    import ncp as NC
    tk = json.loads((OUT / 'tokens.json').read_text())
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    step = int(ck['state']['step'])
    model = M.V13(M.Cfg(V=tk['vocab'])); model.load_state_dict(ck['model'], strict=True); model.eval()
    cb = NC.PQCodebook(model.c.d, model.c.pq_S, model.c.pq_N); cb.load_state_dict(ck['codebook'])
    stash = {}
    orig = M.ConceptRead.forward
    def read_fwd(s, x, ctx):                                  # copy of ConceptRead.forward that keeps the attention p
        B, T, _ = x.shape; H, dh = s.H, s.dh
        q0 = s.qk.q(s.wq(x).view(B, T, H, dh))
        ct, st = ctx['rope_tok']; q = M.rope(q0, ct[:, None], st[:, None], dh)
        un = ctx['u_norm']; G = un.shape[1]
        k, v = s.wkv(un).view(B, G, 2, H, dh).unbind(2); k = s.qk.k(k)
        cr, sr = ctx['rope_rel']; k = M.rope(k, cr[:, None], sr[:, None], dh)
        sc = torch.einsum('bthd,bghd->bhtg', q, k) / math.sqrt(dh)
        sc = sc.masked_fill(~ctx['read_mask'][:, None], float('-inf'))
        sn = torch.einsum('bthd,hd->bht', q0, s.qk.k(s.k_null)) / math.sqrt(dh)
        ss = s.sink[None, :, None].expand(B, H, T)
        p = torch.cat([sc, sn[..., None], ss[..., None]], -1).softmax(-1)
        stash.setdefault('read', []).append(p[0].mean(0))     # (T, G + 2), mean over heads
        y = torch.einsum('bhtg,bghd->bthd', p[..., :G], v) + p[..., G].transpose(1, 2)[..., None] * s.v_null
        return s.wo(y.reshape(B, T, -1) * torch.sigmoid(s.wg(x)))
    M.ConceptRead.forward = read_fwd
    try:
        ids = torch.tensor([tk['ids']]); T = ids.shape[1]; doc = torch.zeros(1, T, dtype=torch.long)
        with torch.no_grad():
            ref = None
            M.ConceptRead.forward = orig
            ref = model(ids, doc, return_all=True)['logits']
            M.ConceptRead.forward = read_fwd
            out = model(ids, doc, return_all=True)
            assert torch.allclose(out['logits'], ref, atol=1e-5), 'the attention-keeping copy changed the output'
            pools = {}
            for nm, mod, H in (('x', model.pool_x, out['x'] if 'x' in out else None), ('L', model.pool_L, out['L_enc']), ('G', model.pool_G, out['G_enc'])):
                if H is None: H = model.emb(ids)
                z = mod.wz(H).view(1, T // 4, 4, -1) + mod.B
                pools[nm] = z.softmax(2).mean(-1)[0].tolist()            # (groups, 4): mean over channels
            pred = out['ncp'][0].argmax(-1)                                # (G, S)
            tgt = cb.assign(cb.normalise(out['m'][0]))                     # (G, S): codes of each group's own m
            probs = out['ncp'][0].softmax(-1)
            ncp = [dict(group=g, pred=pred[g].tolist(), target=tgt[g + 1].tolist(),
                        p_target=[round(float(probs[g, s, tgt[g + 1, s]]), 4) for s in range(model.c.pq_S)])
                   for g in range(T // 4 - 1)]
            lp = out['logits'][0, -1].log_softmax(-1); top = lp.topk(5)
            tok = Tok()
            nxt = dict(top=[dict(id=int(i), piece=tok.decode([int(i)]), p=round(float(v.exp()), 4)) for v, i in zip(top.values, top.indices)],
                       rank_of_next=int((lp > lp[tk['next_id']]).sum()) + 1, p_next=round(float(lp[tk['next_id']].exp()), 4))
            reads = torch.stack(stash['read'][-model.c.n_d:])              # last forward: (n_d, T, G + 2)
            row = reads[:, -1].tolist()                                    # last row, per decoder layer: concepts..., null, sink
            beta = model.beta.softmax(0).tolist()
    finally:
        M.ConceptRead.forward = orig
    return dict(source=f'{Path(ckpt_path).name} copied from v13_tiny/runs/base/latest.pt; one CPU forward, eval mode',
                step=step, hours=round(float(ck['state']['elapsed']) / 3600, 2), run_hours=11, tokens=tk['pieces'], beta=beta, pool=pools, ncp=ncp,
                next=nxt, read_last_row=row, read_cols=[f'u_{g}' for g in range(T // 4)] + ['null', 'sink'],
                note='NCP: group g predicts the codes of m_{g+1}; target = codebook.assign(normalise(m_{g+1})). '
                     'pool: softmax weights over the 4 tokens of each group, mean over the 128 channels.')


if __name__ == '__main__':
    if '--journey' in sys.argv:
        ck = sys.argv[sys.argv.index('--journey') + 1]
        d = journey(ck); (OUT / 'journey.json').write_text(json.dumps(d, indent=1)); print('wrote journey, step', d['step']); sys.exit(0)
    for name, fn in (('tiny_v2', tiny_v2), ('tiers', tiers), ('tokens', tokens), ('snapshot', snapshot)):
        d = fn(); (OUT / f'{name}.json').write_text(json.dumps(d, indent=1)); print('wrote', name)
