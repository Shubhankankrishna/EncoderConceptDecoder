"""V14-tiny diagnostics: REPORT.md §2's V1-V5 and v14_Architecture.md §6.3's reads, on the base run's checkpoints.

    python3 diagnostics_v14.py --run runs/base --out runs/eval/diag              # post_base_v14.sh
    python3 diagnostics_v14.py --run <dir> --out <dir> --device cpu --quick        # plumbing test

Written on 15 Oct 2026 before any V14-tiny training step; DEVIATIONS.md item 13 records the choices below.

Rows. Fresh dev-time rows (seed 99, curriculum fraction 1.0, as V13-tiny's diagnostics) for the four reasoning
families and held-out TinyStories validation stories for text. "Training rows" (the family marginal and the cross-pair
table) are rows of the training seed (1301) at curriculum fraction 1.0 and the first rows of the training text stream.

Targets. The codes the head was trained on: the checkpoint's twin with its residue maps (the family's own P_f),
normaliser and codebook. Pairs (g, g+h) lie in one document and need tau_{g+h} (b needs a previous group);
copy-current needs tau_g as well. Segments 1-2 code the local half a, segments 3-4 the global half b (§6.1).

Checkpoints. Every read on snap_010, snap_030, snap_050, snap_070, snap_090 and final, except the action sensitivity,
the cross-pair rate and the local half's share of the head's signal, which run on snap_050 and final only. V5's code
reassignment: snap_010 against snap_050 (compared), snap_030 against snap_070 (reported), for V14-tiny and V13-tiny on
the same rows (V13-tiny's codes: its PQ codebook on m_g, every real group; computed by v5_v13.py in its own process).

Decisions (REPORT §2), at the final checkpoint:
  V1  global half, horizon 1, pairs with tau_g defined: head accuracy minus max(copy-current, family marginal) of at
      least 0.10, in at least 3 of the 4 reasoning families;
  V2  probe from u_g minus probe from L_enc at row 4g+3, global half, at least 0.02 in at least 2 reasoning families;
  V3  probe from u_g minus probe from p_g, global half, at least 0.05: pass or fail per reasoning family, beside
      V13-tiny's u - p in that family (V13's probe predicts the four segments of its single target);
  V4  chi of the twin's normalised global half at least 0.5 in every family (all five; reasoning families beside);
  V5  code reassignment (the fraction of segment codes that change on the same rows and groups) between snap_010 and
      snap_050 lower for V14-tiny than for V13-tiny, all five families pooled (per family reported).
The probes are V13-tiny's (an MLP 256 wide, 8 epochs, 80/20 split, seed 0), predicting all four segments of the next
group's code; accuracy is reported per half. §6.1's end-of-training guard (twin-versus-online agreement at least 0.8
in each half, final checkpoint, all valid groups pooled) is read here and decides the era refit (era_v14.py).
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data as D
import model as M
import tok as TK
import v14 as V

FAMS = V.FAMILIES
REASONING = ('grid', 'prog', 'state', 'plan')
FAM_IDX = {f: i for i, f in enumerate(FAMS)}
CKPTS = ('snap_010', 'snap_030', 'snap_050', 'snap_070', 'snap_090', 'final')
HEAVY = ('snap_050', 'final')
LOC, GLO = slice(0, 2), slice(2, 4)
V13_DIR = HERE.parent / 'v13_tiny'
NAN = float('nan')


# ------------------------------------------------------------------------------------------------ loading and rows
def load(path, dev, tk):
    z = torch.load(path, map_location='cpu', weights_only=False)
    m = M.V14(M.Cfg(V=tk.vocab)); m.load_state_dict(z['model']); m = m.to(dev).eval()
    tw = V.make_twin(m)
    miss, unexp = tw.load_state_dict({k: v for k, v in z['twin'].items() if not k.startswith('engram.table')},
                                     strict=False)
    assert not unexp and all(k.startswith('engram.table') for k in miss), (miss, unexp)
    tw.eval()
    cb = V.Codebook(m.c.d, m.c.pq_S, m.c.pq_N); cb.load_state_dict(z['codebook']); cb = cb.to(dev)
    maps = V.ResidueMaps(len(FAMS), 2 * m.c.d, m.c.d); maps.load_state_dict(z['maps']); maps = maps.to(dev)
    return m, tw, cb, maps, z


def dev_rows(fam, n, tk):
    if fam != 'text':
        return [D.make_row(fam, 10 ** 7 + k, 1.0, 99) for k in range(n)]
    out, rb = [], D.RowBuilder(tk, D.T_ROW)
    for ids in D.val_stories(tk, n=4000):
        if rb.add(ids, None, text=True): continue
        o = rb.out(); o['stats'] = {}; out.append(o)
        if len(out) >= n: break
        rb = D.RowBuilder(tk, D.T_ROW); rb.add(ids, None, text=True)
    return out


def train_rows(fam, n, tk):
    if fam != 'text':
        return [D.make_row(fam, k, 1.0, 1301) for k in range(n)]
    ts = D.TextStream(tk)
    return [ts.row() for _ in range(n)]


def to_dev(rows, dev):
    b = D.collate(rows)
    return torch.from_numpy(b['ids']).long().to(dev), torch.from_numpy(b['doc']).long().to(dev)


# ------------------------------------------------------------------------------------------------ one pass
@torch.no_grad()
def collect(m, tw, cb, maps, rows, fam, dev):
    """Per valid group: twin tau, its normalised form, the online view, both codes, dl and dG. Per horizon: the head's
    argmax tuple, the target code and the current code. Horizon-1 probe inputs: u_g, p_g, L_enc at row 4g+3."""
    k = m.c.k; fi = FAM_IDX[fam]
    Gr = {n: [] for n in ('key', 'tau', 'z', 'aon', 'bon', 'code', 'code_on', 'dl', 'dG')}
    Pr = {h: {n: [] for n in ('pred', 'tgt', 'cur', 'has_cur')} for h in range(1, m.c.horizons + 1)}
    Pb = {n: [] for n in ('u', 'p', 'lrow', 'tgt', 'logits1')}
    same_num = same_den = 0.0
    for i in range(0, len(rows), 8):
        ids, doc = to_dev(rows[i:i + 8], dev)
        B, T = ids.shape; G = T // k
        eidx = m.engram.keys(ids, doc)
        P_rows = maps.P[fi].expand(B, -1, -1)
        o = m(ids, doc, eidx=eidx, return_all=True)
        tau, valid, dl, dG = V.twin_targets(tw, ids, doc, eidx, P_rows)
        gdoc = doc[:, ::k]
        ell, gbar = V.group_feats(o['x'], o['L_enc'], o['G_enc'], k)
        a_on, b_on, _, _, _ = V.target_view(m, ell, gbar, gdoc, P_rows)
        z = cb.normalise(tau)
        code = cb.assign(z); code_on = cb.assign(cb.normalise(torch.cat([a_on, b_on], -1)))
        hl = o['ncp']; pred = hl.argmax(-1)                                             # (B, G, K, S)
        real = float((gdoc > 0).sum())
        same_num += float(V.same_argmax_fraction(hl, gdoc)) * real; same_den += real
        rowid = (i + torch.arange(B, device=dev))[:, None].expand(B, G)
        key = rowid * 100000 + gdoc
        for n, t in (('key', key), ('tau', tau), ('z', z), ('aon', a_on), ('bon', b_on), ('code', code),
                     ('code_on', code_on), ('dl', dl), ('dG', dG)):
            Gr[n].append(t[valid])
        for h in Pr:
            pair = (gdoc[:, :G - h] == gdoc[:, h:]) & (gdoc[:, :G - h] > 0) & valid[:, h:]
            Pr[h]['pred'].append(pred[:, :G - h, h - 1][pair]); Pr[h]['tgt'].append(code[:, h:][pair])
            Pr[h]['cur'].append(code[:, :G - h][pair]); Pr[h]['has_cur'].append(valid[:, :G - h][pair])
        pair = (gdoc[:, :-1] == gdoc[:, 1:]) & (gdoc[:, :-1] > 0) & valid[:, 1:]
        rel = k * torch.arange(G, device=dev) + k - 1
        Pb['u'].append(o['u'][:, :-1][pair]); Pb['p'].append(m.wp(o['m'])[:, :-1][pair])
        Pb['lrow'].append(o['L_enc'][:, rel][:, :-1][pair]); Pb['tgt'].append(code[:, 1:][pair])
        Pb['logits1'].append(hl[:, :-1, 0][pair])
    cat = lambda d: {a: torch.cat(v).cpu() for a, v in d.items()}
    return cat(Gr), {h: cat(d) for h, d in Pr.items()}, cat(Pb), same_num / max(same_den, 1.0)


@torch.no_grad()
def train_codes(m, tw, cb, maps, rows, fam, dev):
    """The twin's codes of every valid group of training rows (the family marginal, the cross-pair table)."""
    k = m.c.k; out = []
    for i in range(0, len(rows), 8):
        ids, doc = to_dev(rows[i:i + 8], dev)
        tau, valid, _, _ = V.twin_targets(tw, ids, doc, m.engram.keys(ids, doc), maps.P[FAM_IDX[fam]].expand(len(ids), -1, -1))
        out.append(cb.assign(cb.normalise(tau))[valid].cpu())
    return torch.cat(out)


# ------------------------------------------------------------------------------------------------ reads
def acc(a, b, sl):
    return float((a[:, sl] == b[:, sl]).float().mean()) if len(a) else NAN


def ppl(codes, Nw):
    out = []
    for s in range(codes.shape[1]):
        p = torch.bincount(codes[:, s], minlength=Nw).double(); p = p / p.sum().clamp(min=1)
        out.append(round(float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum())), 3))
    return out


def chi(key, z):
    """Within-document share of variance (H-JEPA's measure), documents with >= 2 valid groups (v14_design/checks.py)."""
    key = key.numpy(); z = z.double().numpy()
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= 2
    z, inv = z[keep], inv[keep]
    if len(z) < 2: return NAN
    means = np.zeros((len(uniq), z.shape[1])); np.add.at(means, inv, z)
    means /= np.maximum(np.bincount(inv, minlength=len(uniq)), 1)[:, None]
    within = ((z - means[inv]) ** 2).sum(); total = ((z - z.mean(0)) ** 2).sum()
    return float(within / max(total, 1e-12))


def eff_rank(X):
    """exp of the entropy of the normalised singular values of the (uncentred) matrix (RankMe's definition)."""
    if len(X) < 2: return NAN
    s = torch.linalg.svdvals(X.double()); p = s / s.sum().clamp(min=1e-12); p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def msq_xcorr(a, b):
    if len(a) < 3: return NAN
    za = (a.double() - a.double().mean(0)) / (a.double().std(0) + 1e-8)
    zb = (b.double() - b.double().mean(0)) / (b.double().std(0) + 1e-8)
    return float(((za.t() @ zb / (len(a) - 1)) ** 2).mean())


def r2(dl, dG, P):
    if len(dl) < 2: return NAN
    res = dG.double() - dl.double() @ P.cpu().double(); tot = dG.double() - dG.double().mean(0)
    return float(1 - (res ** 2).sum() / (tot ** 2).sum().clamp(min=1e-12))


def probe(X, Y, S, Nw, epochs=8, seed=0):
    """V13-tiny's diagnostics probe; returns the held-out accuracy per segment."""
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
        return (net(Xn[te]).view(-1, S, Nw).argmax(-1) == Y[te]).float().mean(0).tolist()


def head_reads(Pr, marg):
    out = {}
    for h, d in Pr.items():
        cc = d['has_cur']; n = int(cc.sum())
        mg = marg[None].expand(n, -1)
        out[f'h{h}'] = dict(pairs=len(d['tgt']), head_a=acc(d['pred'], d['tgt'], LOC), head_b=acc(d['pred'], d['tgt'], GLO),
                            pairs_cc=n, head_a_cc=acc(d['pred'][cc], d['tgt'][cc], LOC),
                            head_b_cc=acc(d['pred'][cc], d['tgt'][cc], GLO),
                            copy_a=acc(d['cur'][cc], d['tgt'][cc], LOC), copy_b=acc(d['cur'][cc], d['tgt'][cc], GLO),
                            marg_a=acc(mg, d['tgt'][cc], LOC), marg_b=acc(mg, d['tgt'][cc], GLO))
    return out


def signal_share(m, tw, cb, maps, rows, fam, dev):
    """||dL_a/du|| / (||dL_a/du|| + ||dL_b/du||) per horizon on one batch (the head alone, two backward passes)."""
    k = m.c.k; ids, doc = to_dev(rows[:8], dev)
    with torch.no_grad():
        eidx = m.engram.keys(ids, doc); P_rows = maps.P[FAM_IDX[fam]].expand(len(ids), -1, -1)
        o = m(ids, doc, eidx=eidx, return_all=True)
        tau, valid, _, _ = V.twin_targets(tw, ids, doc, eidx, P_rows)
        code = cb.assign(cb.normalise(tau))
    u = o['u'].detach().clone().requires_grad_(True)
    gdoc = doc[:, ::k]; G = gdoc.shape[1]; out = []
    with torch.enable_grad():
        hl = m.horizon_logits(u)
        for h in range(1, m.c.horizons + 1):
            pair = ((gdoc[:, :G - h] == gdoc[:, h:]) & (gdoc[:, :G - h] > 0) & valid[:, h:]).float()
            lg = hl[:, :G - h, h - 1]; tg = code[:, h:]
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            npair = pair.sum().clamp(min=1.0)
            La = (ce[..., LOC].sum(-1) * pair).sum() / npair; Lb = (ce[..., GLO].sum(-1) * pair).sum() / npair
            ga, = torch.autograd.grad(La, u, retain_graph=True); gb, = torch.autograd.grad(Lb, u, retain_graph=True)
            na, nb = float(ga.norm()), float(gb.norm())
            out.append(round(na / max(na + nb, 1e-12), 4))
    return out


@torch.no_grad()
def action_sensitivity(m, tw, maps, rows, fam, dev, tk, n_prefix=64, K=8, T_alt=256, per_batch=4, n_control=4,
                       seed=5150):
    """§6.3's target-side action sensitivity. Per prefix (a document's first j+1 groups, j >= 1, at most T_alt - 4
    tokens) K next groups are sampled from the model at temperature 1 and each is run through the twin; spread/motion
    = sum over prefixes of the variance of the pre-normaliser b_{j+1} (and a_{j+1}) across the K alternatives, over the
    sum of its mean squared size. The control repeats one alternative K times (must give 0)."""
    k = m.c.k; fi = FAM_IDX[fam]; gen = torch.Generator().manual_seed(seed); rng = np.random.default_rng(seed)
    cands = []
    for r, row in enumerate(rows):
        dc = row['doc']
        for d_ in range(1, int(dc.max()) + 1):
            pos = np.nonzero(dc == d_)[0]
            if len(pos) == 0: continue
            s0, Gd = int(pos[0]), len(pos) // k
            jmax = min(Gd - 2, T_alt // k - 2)
            if jmax >= 1: cands.append((r, s0, jmax))
    if not cands: return dict(prefixes=0)
    pick = rng.choice(len(cands), size=min(n_prefix, len(cands)), replace=False)
    pre = []
    for c in pick:
        r, s0, jmax = cands[int(c)]; j = int(rng.integers(1, jmax + 1))
        pre.append((rows[r]['ids'][s0:s0 + k * (j + 1)].astype(np.int64), j))
    pad = tk.pad
    sums = dict(sa=0.0, ma=0.0, sb=0.0, mb=0.0); ctrl = []
    for i0 in range(0, len(pre), per_batch):
        chunk = pre[i0:i0 + per_batch]; nb = len(chunk) * K
        ids = torch.full((nb, T_alt), pad, dtype=torch.long); doc = torch.zeros(nb, T_alt, dtype=torch.long)
        last = torch.zeros(nb, dtype=torch.long); grp = torch.zeros(nb, dtype=torch.long)
        for ci, (p_, j) in enumerate(chunk):
            sl = slice(ci * K, (ci + 1) * K); Lp = len(p_)
            ids[sl, :Lp] = torch.from_numpy(p_); doc[sl, :Lp + k] = 1; last[sl] = Lp - 1; grp[sl] = j + 1
        ids, doc = ids.to(dev), doc.to(dev)
        for t in range(k):                                                               # sample the next group
            lg = m(ids, doc)[0]
            pos = (last + t).to(dev)
            pr = torch.softmax(lg[torch.arange(nb, device=dev), pos].float(), -1).cpu()
            nxt = torch.multinomial(pr, 1, generator=gen)[:, 0]
            ids[torch.arange(nb, device=dev), pos + 1] = nxt.to(dev)
        P_rows = maps.P[fi].expand(nb, -1, -1)
        tau, _, _, _ = V.twin_targets(tw, ids, doc, m.engram.keys(ids, doc), P_rows)
        tg = tau[torch.arange(nb, device=dev), grp.to(dev)].cpu().double().view(len(chunk), K, -1)
        a_, b_ = tg[..., :m.c.d // 2], tg[..., m.c.d // 2:]
        sums['sa'] += float(((a_ - a_.mean(1, keepdim=True)) ** 2).sum(-1).mean(1).sum())
        sums['ma'] += float((a_ ** 2).sum(-1).mean(1).sum())
        sums['sb'] += float(((b_ - b_.mean(1, keepdim=True)) ** 2).sum(-1).mean(1).sum())
        sums['mb'] += float((b_ ** 2).sum(-1).mean(1).sum())
        if len(ctrl) < n_control:                                                        # K identical alternatives
            ids_c = ids[0:1].expand(K, -1).contiguous(); doc_c = doc[0:1].expand(K, -1).contiguous()
            tc, _, _, _ = V.twin_targets(tw, ids_c, doc_c, m.engram.keys(ids_c, doc_c), maps.P[fi].expand(K, -1, -1))
            bc = tc[:, int(grp[0])].cpu().double()
            ctrl.append(float(((bc - bc.mean(0)) ** 2).sum(-1).mean() / (bc ** 2).sum(-1).mean().clamp(min=1e-30)))
    ra = sums['sa'] / max(sums['ma'], 1e-30); rb = sums['sb'] / max(sums['mb'], 1e-30)
    return dict(prefixes=len(pre), K=K, ratio_a=round(ra, 5), ratio_b=round(rb, 5), control_max=max(ctrl) if ctrl else NAN,
                b_below_gate=bool(rb < 0.05), b_below_tenth_of_a=bool(rb < 0.1 * ra))


def cross_pair(Pb, table, gen, Nw):
    tid = lambda c: c[:, 0] + Nw * c[:, 1] + Nw ** 2 * c[:, 2] + Nw ** 3 * c[:, 3]
    if len(Pb['tgt']) == 0: return {}
    head = Pb['logits1'].argmax(-1)
    smp = torch.multinomial(torch.softmax(Pb['logits1'].float().reshape(-1, Nw), -1), 1, generator=gen).view(head.shape)
    rate = lambda c: round(float((~table[tid(c)]).float().mean()), 4)
    out = dict(head_argmax=rate(head), head_sampled=rate(smp), targets=rate(Pb['tgt']), table_tuples=int(table.sum()))
    out['head_above_twice_targets'] = bool(out['head_argmax'] > 2 * out['targets'])
    return out


def reassignment(c1, c2):
    if c1 is None or c2 is None or len(c1) != len(c2) or len(c1) == 0: return None
    ne = (c1 != c2).float()
    return dict(all=round(float(ne.mean()), 4), a=round(float(ne[:, LOC].mean()), 4), b=round(float(ne[:, GLO].mean()), 4),
                groups=len(c1))


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='runs/base'); ap.add_argument('--out', default='runs/eval/diag')
    ap.add_argument('--device', default='mps'); ap.add_argument('--rows', type=int, default=48)
    ap.add_argument('--train-rows', type=int, default=64); ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    dev = torch.device(a.device); run = HERE / a.run; out = HERE / a.out; out.mkdir(parents=True, exist_ok=True)
    nrow, ntr = (8, 8) if a.quick else (a.rows, a.train_rows)
    epochs = 1 if a.quick else 8
    asens = dict(n_prefix=4, per_batch=2, K=4, n_control=1) if a.quick else {}
    D._init_worker()
    tk = TK.Tok(); t_all = time.time()
    DEV = {f: dev_rows(f, nrow, tk) for f in FAMS}
    TRN = {f: train_rows(f, ntr, tk) for f in FAMS}
    (out / 'dev_rows.pkl').write_bytes(pickle.dumps({f: [dict(ids=r['ids'], doc=r['doc']) for r in DEV[f]] for f in FAMS}))
    torch.manual_seed(1301); w0 = float(M.V14(M.Cfg(V=tk.vocab)).t_glob.weight.norm())          # train.py's step 0
    res = dict(meta=dict(rows_per_family=nrow, train_rows=ntr, quick=a.quick, started=time.strftime('%Y-%m-%d %H:%M:%S'),
                         device=a.device), per_ckpt={}, V={}, guard={})
    CODES = {}
    for ck in CKPTS:
        p = run / f'{ck}.pt'
        if not p.exists(): print('missing', p, flush=True); continue
        t0 = time.time()
        m, tw, cb, maps, z = load(p, dev, tk)
        S, Nw = m.c.pq_S, m.c.pq_N
        R = dict(step=z.get('step'), tokens=z.get('tokens'), t_glob_gain=round(float(m.t_glob.weight.norm()) / w0, 4),
                 frozen=bool(cb.frozen), p_ok=[bool(x) for x in (maps.cnt.cpu() >= maps.min_pairs)], fam={})
        gen = torch.Generator().manual_seed(77)
        AG = dict(n=0, a=0.0, b=0.0)
        for f in FAMS:
            Gr, Pr, Pb, same = collect(m, tw, cb, maps, DEV[f], f, dev)
            trc = train_codes(m, tw, cb, maps, TRN[f], f, dev)
            marg = torch.tensor([int(torch.bincount(trc[:, s], minlength=Nw).argmax()) if len(trc) else 0 for s in range(S)])
            CODES[(ck, f)] = Gr['code']
            agree = (Gr['code'] == Gr['code_on']).float()
            AG['n'] += len(agree); AG['a'] += float(agree[:, LOC].sum(0).mean()); AG['b'] += float(agree[:, GLO].sum(0).mean())
            h = m.c.d // 2
            r = dict(groups=len(Gr['code']), pairs_h1=len(Pb['tgt']), same_argmax_all_horizons=round(same, 4),
                     head=head_reads(Pr, marg), marginal_words=marg.tolist(), chance=1 / Nw,
                     usage_perplexity=ppl(Gr['code'], Nw),
                     chi=dict(online_a=chi(Gr['key'], Gr['aon']), online_b=chi(Gr['key'], Gr['bon']),
                              twin_norm_a=chi(Gr['key'], Gr['z'][:, :h]), twin_norm_b=chi(Gr['key'], Gr['z'][:, h:])),
                     agreement_twin_online=dict(a=acc(Gr['code'], Gr['code_on'], LOC), b=acc(Gr['code'], Gr['code_on'], GLO)),
                     residue_r2=dict(own=r2(Gr['dl'], Gr['dG'], maps.P[FAM_IDX[f]]), pooled=r2(Gr['dl'], Gr['dG'], maps.P[len(FAMS)])),
                     xcorr_ab=msq_xcorr(Gr['tau'][:, :h], Gr['tau'][:, h:]),
                     eff_rank=dict(twin_a=eff_rank(Gr['tau'][:, :h]), twin_b=eff_rank(Gr['tau'][:, h:]),
                                   online_a=eff_rank(Gr['aon']), online_b=eff_rank(Gr['bon'])))
            if len(Pb['tgt']) >= 50:
                pr = {src: probe(Pb[src], Pb['tgt'], S, Nw, epochs=epochs) for src in ('u', 'p', 'lrow')}
                r['probes'] = {src: dict(a=round(float(np.mean(v[LOC])), 4), b=round(float(np.mean(v[GLO])), 4),
                                         per_segment=[round(x, 4) for x in v]) for src, v in pr.items()}
            if ck in HEAVY:
                tt = time.time()
                table = torch.zeros(Nw ** S, dtype=torch.bool)
                if len(trc): table[trc[:, 0] + Nw * trc[:, 1] + Nw ** 2 * trc[:, 2] + Nw ** 3 * trc[:, 3]] = True
                r['cross_pair'] = cross_pair(Pb, table, gen, Nw)
                r['signal_share_local'] = signal_share(m, tw, cb, maps, DEV[f], f, dev)
                r['action_sensitivity'] = action_sensitivity(m, tw, maps, DEV[f], f, dev, tk, **asens)
                r['s_heavy'] = round(time.time() - tt, 1)
            R['fam'][f] = r
            print(ck, f, json.dumps({k_: r[k_] for k_ in ('groups', 'chi', 'agreement_twin_online')}), flush=True)
        R['agreement_pooled'] = dict(a=round(AG['a'] / max(AG['n'], 1), 4), b=round(AG['b'] / max(AG['n'], 1), 4))
        R['seconds'] = round(time.time() - t0, 1)
        res['per_ckpt'][ck] = R
        (out / 'diagnostics_v14.json').write_text(json.dumps(res, indent=1))
        del m, tw, cb, maps, z
    # ---------------------------------------------------------------------------------------------- V5 and decisions
    v5 = {}
    for x, y in (('snap_010', 'snap_050'), ('snap_030', 'snap_070')):
        per = {f: reassignment(CODES.get((x, f)), CODES.get((y, f))) for f in FAMS}
        if all(v is not None for v in per.values()):
            c1 = torch.cat([CODES[(x, f)] for f in FAMS]); c2 = torch.cat([CODES[(y, f)] for f in FAMS])
            per['pooled'] = reassignment(c1, c2)
        v5[f'{x}-{y}'] = per
    v13 = None
    try:
        cp = subprocess.run([sys.executable, str(HERE / 'v5_v13.py'), '--v13', str(V13_DIR), '--rows', str(out / 'dev_rows.pkl'),
                             '--device', a.device], capture_output=True, text=True, timeout=3600)
        line = [l for l in cp.stdout.splitlines() if l.startswith('{')]
        v13 = json.loads(line[-1]) if line else dict(error=cp.stderr[-2000:])
    except Exception as e:                                                               # reported, not fatal
        v13 = dict(error=repr(e))
    res['V5_reassignment'] = dict(v14=v5, v13=v13)
    fin = res['per_ckpt'].get('final')
    if fin is not None:
        d1 = {}
        for f in REASONING:
            h1 = fin['fam'][f]['head']['h1']
            mgn = h1['head_b_cc'] - max(h1['copy_b'], h1['marg_b'])
            d1[f] = dict(head_b=h1['head_b_cc'], copy_b=h1['copy_b'], marg_b=h1['marg_b'], margin=round(mgn, 4),
                         passes=bool(mgn >= 0.10))
        res['V']['V1'] = dict(per_family=d1, passes=sum(v['passes'] for v in d1.values()) >= 3)
        pv = {f: fin['fam'][f].get('probes') for f in REASONING}
        if all(v is not None for v in pv.values()):
            d2 = {f: dict(u_b=pv[f]['u']['b'], lrow_b=pv[f]['lrow']['b'], margin=round(pv[f]['u']['b'] - pv[f]['lrow']['b'], 4))
                  for f in REASONING}
            for v in d2.values(): v['passes'] = bool(v['margin'] >= 0.02)
            res['V']['V2'] = dict(per_family=d2, passes=sum(v['passes'] for v in d2.values()) >= 2)
            try:
                v13d = json.loads((V13_DIR / 'runs/eval/diag/diagnostics.json').read_text())['probes']
            except Exception:
                v13d = {}
            d3 = {f: dict(u_b=pv[f]['u']['b'], p_b=pv[f]['p']['b'], margin=round(pv[f]['u']['b'] - pv[f]['p']['b'], 4),
                          u_a=pv[f]['u']['a'], p_a=pv[f]['p']['a'], margin_a=round(pv[f]['u']['a'] - pv[f]['p']['a'], 4),
                          v13_u_minus_p=v13d.get(f, {}).get('u_minus_p')) for f in REASONING}
            for v in d3.values(): v['passes'] = bool(v['margin'] >= 0.05)
            res['V']['V3'] = dict(per_family=d3, families_passing=[f for f in REASONING if d3[f]['passes']])
        d4 = {f: round(fin['fam'][f]['chi']['twin_norm_b'], 4) for f in FAMS}
        res['V']['V4'] = dict(chi_twin_norm_b=d4, passes=all(v >= 0.5 for v in d4.values()),
                              passes_reasoning_only=all(d4[f] >= 0.5 for f in REASONING))
        res['guard'] = dict(agreement=fin['agreement_pooled'],
                            passes=bool(fin['agreement_pooled']['a'] >= 0.8 and fin['agreement_pooled']['b'] >= 0.8))
    p14 = v5.get('snap_010-snap_050', {}).get('pooled'); p13 = (v13 or {}).get('snap_010-snap_050', {}).get('pooled')
    if p14 is not None and p13 is not None:
        res['V']['V5'] = dict(v14=p14['all'], v13=p13, passes=bool(p14['all'] < p13))
    res['meta']['seconds'] = round(time.time() - t_all, 1); res['meta']['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (out / 'diagnostics_v14.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(V=res['V'], guard=res['guard']), indent=1), flush=True)


if __name__ == '__main__':
    main()
