"""§6.1's end-of-training guard and era refit, before the continual stage (v14_Architecture.md §6.1, §9.2(a)).

    python3 era_v14.py --run runs/base --diag runs/eval/diag/diagnostics_v14.json [--device mps] [--quick]

The guard (twin-versus-online agreement of segment codes at the final checkpoint, at least 0.8 in each half) is read
from diagnostics_v14.py's output. If it holds, nothing is refitted: era.json says so and the continual stage uses the
final checkpoint's maps, normaliser, codebook and head. If it fails in either half (or was not computed), the era is
refitted on the final online model's targets over base-stream rows (training seed, curriculum fraction 1.0, unseen row
indices; training text rows): the residue maps P_f from scratch (EMA statistics of the online dl and dG, solved once),
the normaliser exactly (mean and variance of the valid online targets), the codebook's words by EMA k-means from the
frozen words (two passes, the training update rule), and then the horizon head alone (its norm, e_h and MLP) for 500
steps with everything else frozen (AdamW, lr 1e-3, clip 1.0, the training horizon loss), on concept states cached once.
The result, runs/base/era.pt, is what cl.py loads (DEVIATIONS.md item 14).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch

import data as D
import model as M
import tok as TK
import v14 as V

FAMS = V.FAMILIES
FAM_IDX = {f: i for i, f in enumerate(FAMS)}
HEAD = ('ncp_norm', 'e_h', 'ncp')


def head_params(m):
    return [p for n, p in m.named_parameters() if n.split('.')[0] in HEAD]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='runs/base'); ap.add_argument('--diag', default='runs/eval/diag/diagnostics_v14.json')
    ap.add_argument('--device', default='mps'); ap.add_argument('--batches-per-family', type=int, default=24)
    ap.add_argument('--head-steps', type=int, default=500); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--force', action='store_true', help='refit even if the guard holds (plumbing test only)')
    a = ap.parse_args()
    dev = torch.device(a.device); run = HERE / a.run; t0 = time.time()
    try:
        guard = json.loads((HERE / a.diag).read_text()).get('guard', {})
    except Exception as e:
        guard = dict(error=repr(e))
    rec = dict(guard=guard, started=time.strftime('%Y-%m-%d %H:%M:%S'))
    if guard.get('passes') is True and not a.force:
        rec.update(refit=False, reason='guard holds in both halves')
        (run / 'era.json').write_text(json.dumps(rec, indent=1)); print(json.dumps(rec), flush=True)
        if (run / 'era.pt').exists(): (run / 'era.pt').unlink()
        return
    rec.update(refit=True, reason='guard fails in a half' if 'passes' in guard else 'guard not computed')
    nb = 2 if a.quick else a.batches_per_family; steps = 5 if a.quick else a.head_steps
    tk = TK.Tok()
    z = torch.load(run / 'final.pt', map_location='cpu', weights_only=False)
    m = M.V14(M.Cfg(V=tk.vocab)); m.load_state_dict(z['model']); m = m.to(dev).eval()
    for p in m.parameters(): p.requires_grad_(False)
    cb = V.Codebook(m.c.d, m.c.pq_S, m.c.pq_N); cb.load_state_dict(z['codebook']); cb = cb.to(dev)
    maps = V.ResidueMaps(len(FAMS), 2 * m.c.d, m.c.d); maps.load_state_dict(z['maps']); maps = maps.to(dev)
    k = m.c.k
    # ---------------------------------------------------------------------------------------------- base-stream rows
    import multiprocessing as mp
    pool = mp.get_context('spawn').Pool(5, initializer=D._init_worker)
    batches = []
    for f in FAMS:
        if f == 'text':
            ts = D.TextStream(tk); rows = [ts.row() for _ in range(8 * nb)]
        else:
            rows = pool.starmap(D.make_row, [(f, 10 ** 6 + i, 1.0, 1301) for i in range(8 * nb)], chunksize=4)
        batches += [(f, D.collate(rows[i:i + 8])) for i in range(0, len(rows), 8)]
    pool.terminate()
    order = np.random.default_rng(2468).permutation(len(batches))
    batches = [batches[i] for i in order]
    feats = []
    with torch.no_grad():
        for f, b in batches:                                                             # online features, once
            ids = torch.from_numpy(b['ids']).long().to(dev); doc = torch.from_numpy(b['doc']).long().to(dev)
            o = m(ids, doc, return_all=True)
            ell, gbar = V.group_feats(o['x'], o['L_enc'], o['G_enc'], k)
            feats.append((f, ell, gbar, doc[:, ::k], o['u']))
        # ------------------------------------------------------------------------------------------ maps, from scratch
        maps.A.zero_(); maps.Bm.zero_(); maps.cnt.zero_(); maps.frozen.fill_(False); maps._frozen_py = False
        for f, ell, gbar, gdoc, _ in feats:
            dl = ell - torch.nn.functional.pad(ell, (0, 0, 1, 0))[:, :-1]
            dG = gbar - torch.nn.functional.pad(gbar, (0, 0, 1, 0))[:, :-1]
            maps.update(FAM_IDX[f], dl, dG, V.has_prev(gdoc))
        rec['maps_ok'] = maps.solve(10 ** 6); maps.freeze()
        # ------------------------------------------------------------------------------------------ targets, normaliser
        tgts = []
        for f, ell, gbar, gdoc, u in feats:
            a_, b_, valid, _, _ = V.target_view(m, ell, gbar, gdoc, maps.P[FAM_IDX[f]].expand(len(ell), -1, -1))
            tgts.append((torch.cat([a_, b_], -1), valid))
        X = torch.cat([t[v] for t, v in tgts])
        old_codes = [cb.assign(cb.normalise(t)) for t, _ in tgts]
        cb.mu.copy_(X.mean(0)); cb.var.copy_(X.var(0, unbiased=False) + 1e-4)
        cb.frozen.fill_(False); cb._frozen_py = False; cb._init_py = True
        r0 = float(cb.resets)
        for _ in range(2):                                                               # EMA k-means, warm start
            for t, v in tgts:
                cb.update(t.reshape(-1, t.shape[-1]), v.reshape(-1).to(t.dtype))
        cb.freeze()
        codes = [cb.assign(cb.normalise(t)) for t, _ in tgts]
        ch = torch.cat([(c0 != c1)[v] for c0, c1, (_, v) in zip(old_codes, codes, tgts)]).float()
        rec['codebook'] = dict(resets_during_refit=float(cb.resets) - r0, codes_changed_a=float(ch[:, :2].mean()),
                               codes_changed_b=float(ch[:, 2:].mean()),
                               perplexity=[round(x, 2) for x in cb.usage_perplexity().cpu().tolist()])

    # ---------------------------------------------------------------------------------------------- the head alone
    def head_acc(i):
        with torch.no_grad():
            f, _, _, gdoc, u = feats[i]
            _, st = V.horizon_loss(m.horizon_logits(u), codes[i], gdoc, tgts[i][1])
        return st[0, 1:3].tolist()                                                       # horizon 1, halves a / b

    held = list(range(len(feats)))[-max(1, len(feats) // 10):]                          # last 10%: held out
    train_i = [i for i in range(len(feats)) if i not in held] or held
    rec['head_before'] = np.mean([head_acc(i) for i in held], 0).round(4).tolist()
    hp = head_params(m)
    for p in hp: p.requires_grad_(True)
    m.train()
    opt = torch.optim.AdamW(hp, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
    losses = []
    for s in range(steps):
        i = train_i[s % len(train_i)]
        f, _, _, gdoc, u = feats[i]
        loss, _ = V.horizon_loss(m.horizon_logits(u.detach()), codes[i], gdoc, tgts[i][1])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(hp, 1.0); opt.step(); losses.append(float(loss.detach()))
    m.eval()
    for p in hp: p.requires_grad_(False)
    rec['head_after'] = np.mean([head_acc(i) for i in held], 0).round(4).tolist()
    rec['head_loss_first_last'] = [round(float(np.mean(losses[:10])), 4), round(float(np.mean(losses[-10:])), 4)]
    head_sd = {n: p.detach().cpu() for n, p in m.state_dict().items() if n.split('.')[0] in HEAD}
    torch.save(dict(head=head_sd, codebook=cb.state_dict(), maps=maps.state_dict(), rec=rec), run / 'era.pt')
    rec['seconds'] = round(time.time() - t0, 1); rec['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (run / 'era.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == '__main__':
    main()
