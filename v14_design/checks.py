"""K1–K5 of v14_Architecture.md §1.4: which hypothesis explains V13-tiny's next-concept result (CPU only).

    taskpolicy -b python3 v14_design/checks.py                       # all default checkpoints
    taskpolicy -b python3 v14_design/checks.py --ckpts final --rows 8 --fams grid   # plumbing test

Run at background QoS after the V13 base run has ended, and never during the continual-budget probe
(20:15–20:25 IST on 14 Oct 2026). Rows are fresh dev-time rows (seed 99, as diagnostics.py uses) for the four
reasoning families and held-out TinyStories validation stories for text.

Per checkpoint and family, over valid (g, g+1) pairs inside one document:
  K1  H(predicted codes), H(target codes) (plug-in, bits per segment) and H(target | u_g), H(target | p_g),
      H(target | L^enc at row 4g+3) as held-out probe cross-entropies (bits per segment)
  K2  persistence P(code_g == code_{g+1}) against the NCP head's accuracy on the same pairs
  K3  probe accuracy from u_g, from p_g and from L^enc at the release row 4g+3 (local-only)
  K4  chi, the within-document share of variance (H-JEPA's measure), for the normalised sources Pool x, Pool L,
      Pool G, for the change of Pool G between consecutive groups, and for m
  K5  code-usage perplexity per segment within the family
The decision rules of §1.4 are evaluated on the last checkpoint and written beside the numbers. (The per-half reads
that §4 adds apply to V14's target view; V13 has one target, so K3 is reported per segment here.)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V13 = ROOT / 'v13_tiny'
sys.path.insert(0, str(V13))
os.chdir(V13)                                                   # tok.py and data.py resolve their paths from here
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402
import torch.nn.functional as F                                 # noqa: E402

import data as D                                                # noqa: E402
import final_eval as FE                                         # noqa: E402
import tok as TK                                                # noqa: E402
from model import rms                                           # noqa: E402

FAMS = ('grid', 'prog', 'state', 'plan', 'text')
REASONING = ('grid', 'prog', 'state', 'plan')
DEFAULT_CKPTS = 'snap_010,snap_030,snap_050,snap_070,snap_090,final'


def rows_for(fam, n_rows, tk):
    if fam != 'text':
        return [D.make_row(fam, 10 ** 7 + k, 1.0, 99) for k in range(n_rows)]
    out, rb = [], D.RowBuilder(tk, D.T_ROW)
    for ids in D.val_stories(tk, n=4000):
        if rb.add(ids, None, text=True): continue
        o = rb.out(); o['stats'] = {}; out.append(o)
        if len(out) >= n_rows: break
        rb = D.RowBuilder(tk, D.T_ROW); rb.add(ids, None, text=True)
    return out


@torch.no_grad()
def collect(m, cb, rows):
    k, S = m.c.k, m.c.pq_S
    P = dict(u=[], p=[], lrow=[], tgt=[], cur=[], pred=[])
    groups = dict(key=[], sx=[], sL=[], sG=[], m=[])
    pairs = dict(key=[], dG=[])
    for i in range(0, len(rows), 8):
        b = D.collate(rows[i:i + 8])
        ids = torch.from_numpy(b['ids']).long(); doc = torch.from_numpy(b['doc']).long()
        o = m(ids, doc, return_all=True)
        x = m.emb(ids); Lenc, Genc, mm, u = o['L_enc'], o['G_enc'], o['m'], o['u']
        B, T = ids.shape; G = T // k
        gdoc = doc[:, ::k]
        sx, sL, sG = rms(m.pool_x(x)), rms(m.pool_L(Lenc)), rms(m.pool_G(Genc))
        codes = cb.assign(cb.normalise(mm))                                   # (B, G, S): code of each group's own m
        pred = o['ncp'].argmax(-1)                                            # made at g, for group g+1
        valid = (gdoc[:, :-1] == gdoc[:, 1:]) & (gdoc[:, :-1] > 0)
        rel = k * torch.arange(G) + k - 1
        lrow = Lenc[:, rel]                                                   # L^enc at the release row 4g+3
        P['u'].append(u[:, :-1][valid]); P['p'].append(m.wp(mm)[:, :-1][valid]); P['lrow'].append(lrow[:, :-1][valid])
        P['tgt'].append(codes[:, 1:][valid]); P['cur'].append(codes[:, :-1][valid]); P['pred'].append(pred[:, :-1][valid])
        rowid = (i + torch.arange(B))[:, None].expand(B, G)
        key = rowid * 100000 + gdoc                                           # one key per (row, document)
        real = gdoc > 0
        groups['key'].append(key[real]); groups['sx'].append(sx[real]); groups['sL'].append(sL[real])
        groups['sG'].append(sG[real]); groups['m'].append(mm[real])
        pairs['key'].append(key[:, 1:][valid]); pairs['dG'].append((sG[:, 1:] - sG[:, :-1])[valid])
    cat = lambda d: {a: torch.cat(v) for a, v in d.items()}
    return cat(P), cat(groups), cat(pairs)


def chi(key, z):
    """Within-document share of variance: sum ||z - mean_doc||^2 / sum ||z - mean_all||^2, documents with >= 2 rows."""
    key = key.numpy(); z = z.double().numpy()
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= 2
    z, inv = z[keep], inv[keep]
    means = np.zeros((len(uniq), z.shape[1])); np.add.at(means, inv, z)
    means /= np.maximum(np.bincount(inv, minlength=len(uniq)), 1)[:, None]
    within = ((z - means[inv]) ** 2).sum(); total = ((z - z.mean(0)) ** 2).sum()
    return float(within / max(total, 1e-12))


def entropy_bits(codes, Nw):
    """Plug-in entropy per segment, in bits (codes: (n, S))."""
    out = []
    for s in range(codes.shape[1]):
        p = torch.bincount(codes[:, s], minlength=Nw).double(); p = p / p.sum()
        out.append(float(-(p[p > 0] * p[p > 0].log2()).sum()))
    return out


def probe(X, Y, S, Nw, epochs=8, seed=0):
    """diagnostics.py's MLP probe, also returning the held-out cross-entropy in bits per segment."""
    torch.manual_seed(seed)
    X = X.float(); n = len(X); perm = torch.randperm(n); tr, te = perm[:int(0.8 * n)], perm[int(0.8 * n):]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-5; Xn = (X - mu) / sd
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, S * Nw))
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=0.01)
    for _ in range(epochs):
        for i in range(0, len(tr), 512):
            bi = tr[i:i + 512]
            loss = F.cross_entropy(net(Xn[bi]).view(-1, Nw), Y[bi].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        lg = net(Xn[te]).view(-1, S, Nw)
        acc = float((lg.argmax(-1) == Y[te]).float().mean())
        ce = float(F.cross_entropy(lg.reshape(-1, Nw), Y[te].reshape(-1)) / np.log(2))
    return acc, ce


def check_one(m, cb, rows):
    S, Nw = m.c.pq_S, m.c.pq_N
    P, Gr, Pa = collect(m, cb, rows)
    tgt, cur, pred = P['tgt'], P['cur'], P['pred']
    r = dict(pairs=len(tgt), docs=int(len(torch.unique(Gr['key']))))
    r['K1'] = dict(H_pred=entropy_bits(pred, Nw), H_target=entropy_bits(tgt, Nw))
    r['K2'] = dict(persistence=[float((cur[:, s] == tgt[:, s]).float().mean()) for s in range(S)],
                   head_acc=[float((pred[:, s] == tgt[:, s]).float().mean()) for s in range(S)])
    pr = {}
    for name in ('u', 'p', 'lrow'):
        acc, ce = probe(P[name], tgt, S, Nw); pr[name] = dict(acc=round(acc, 4), ce_bits=round(ce, 4))
    r['K1']['H_target_given'] = {a: pr[a]['ce_bits'] for a in pr}
    r['K3'] = {a: pr[a]['acc'] for a in pr}
    r['K4'] = dict(chi_x=chi(Gr['key'], Gr['sx']), chi_L=chi(Gr['key'], Gr['sL']), chi_G=chi(Gr['key'], Gr['sG']),
                   chi_dG=chi(Pa['key'], Pa['dG']), chi_m=chi(Gr['key'], Gr['m']))
    r['K5'] = dict(perplexity=[round(2 ** h, 3) for h in r['K1']['H_target']])
    return r


def rules(final):
    """§1.4's decision rules on the last checkpoint. Quantifiers pre-registered 14 Oct 19:40, before any check ran:
    H3, H1, G's change dead and local-only need at least 3 of the 4 reasoning families (text is reported, not counted);
    H2b needs any family (mean perplexity over the four segments below 4). If both 'G_change_dead' and
    'local_only_matches_u' are confirmed, the local-only rule wins and the default target view stays (§1.4)."""
    mean = lambda v: float(np.mean(v))
    R = [f for f in REASONING if f in final]
    out = dict(reasoning_families=R)
    near = [f for f in R if mean(final[f]['K2']['head_acc']) - mean(final[f]['K2']['persistence']) <= 0.05]
    chiG = {f: final[f]['K4']['chi_G'] for f in final}
    out['H3'] = dict(families_with_persistence_within_5pt=near, chi_G=chiG,
                     confirmed=len(near) >= 3 and all(chiG[f] < 0.2 for f in near))
    per = dict(
        G_change_dead={f: bool(final[f]['K4']['chi_dG'] < 0.05) for f in final},
        H1={f: bool(mean(final[f]['K1']['H_pred']) < 0.5 * mean(final[f]['K1']['H_target']) and
                    final[f]['K1']['H_target_given']['u'] > 0.9 * mean(final[f]['K1']['H_target'])) for f in final},
        local_only_matches_u={f: bool(final[f]['K3']['lrow'] >= final[f]['K3']['u'] - 0.02) for f in final})
    for k, v in per.items():
        out[k] = dict(per_family=v, count_reasoning=sum(v[f] for f in R), confirmed=sum(v[f] for f in R) >= 3)
    h2b = {f: bool(mean(final[f]['K5']['perplexity']) < 4) for f in final}
    out['H2b'] = dict(per_family=h2b, confirmed=any(h2b.values()))
    lv = out['G_change_dead']['confirmed'] and not out['local_only_matches_u']['confirmed']
    out['default_run'] = dict(target_view='local only (a)' if lv else 'default [a; b]',
                              lambda_max=0.2 if out['H1']['confirmed'] else 0.1,
                              uncentred_sigreg_on_b=out['H2b']['confirmed'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', default=DEFAULT_CKPTS); ap.add_argument('--rows', type=int, default=48)
    ap.add_argument('--fams', default=','.join(FAMS)); ap.add_argument('--out', default=str(HERE / 'checks'))
    a = ap.parse_args()
    torch.set_num_threads(2)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tk = TK.Tok(); fams = a.fams.split(',')
    rows = {f: rows_for(f, a.rows, tk) for f in fams}
    res = {}
    for name in a.ckpts.split(','):
        path = V13 / 'runs' / 'base' / (f'{name}.pt' if name != 'final' else 'final.pt')
        if not path.exists(): print('missing', path, flush=True); continue
        t0 = time.time()
        m, cb, z = FE.load(path, torch.device('cpu'), tk)
        res[name] = dict(step=int(z.get('step', z.get('state', {}).get('step', -1))))
        for f in fams:
            res[name][f] = check_one(m, cb, rows[f])
            print(name, f, json.dumps(res[name][f]['K2']), json.dumps(res[name][f]['K4']), flush=True)
        (out / 'checks.json').write_text(json.dumps(res, indent=1))
        print(f'{name} done in {time.time() - t0:.0f}s', flush=True)
    last = [n for n in a.ckpts.split(',') if n in res]
    if last:
        res['rules_on'] = last[-1]
        res['rules'] = rules({f: res[last[-1]][f] for f in fams if f in res[last[-1]]})
    (out / 'checks.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res.get('rules', {}), indent=1), flush=True)


if __name__ == '__main__':
    main()
