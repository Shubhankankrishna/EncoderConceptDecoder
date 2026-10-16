"""Two pre-run checks on V13-tiny's final checkpoint (CPU only), pre-registered in v14_Architecture.md.

a7 (§4.3, the projector check). A linear map T: 256 -> 64 fitted by document-centred SIGReg *alone* (§8.3's form: M = 256
   random directions per step, Epps-Pulley on 17 knots over [-5, 5], sigma = 1, first groups of documents masked,
   documents with >= 4 valid groups) on the local features l_g = [MeanPool(x)_g ; MeanPool(L_enc)_g], against the
   top-64 principal subspace of the same features (whitened), on
     - retained variance: tr(Pi Sigma) / tr(Sigma) for the projector Pi onto the map's row space (global and
       within-document covariance), and
     - a linear token-identity probe: least-squares classifier from the 64 outputs to each of the group's 4 token ids,
       fitted on even rows, scored on odd rows (unseen tokens count as errors).
   Rule (§4.3): if the SIGReg map's probe is more than 5 points below the principal subspace's, the statistic-projector
   arm becomes the default.
b4 (§6.2, soft code targets). On V13's own codebook over normalised m: the isotropic-Gaussian-mixture temperature
   T_s = 2 sigma_s^2 per segment (sigma_s^2 the per-dimension squared distance to the nearest word), the mean posterior
   entropy at T_s, and the temperature that gives a mean entropy of 1 bit (the arm's stated rule).

    taskpolicy -b python3 v14_design/check_projectors.py [--rows 24] [--steps 3000]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'v13_tiny'))
import checks as C                                              # noqa: E402


@torch.no_grad()
def collect(m, cb, rows, k):
    out = []
    for i in range(0, len(rows), 8):
        b = C.D.collate(rows[i:i + 8]); ids = torch.from_numpy(b['ids']).long(); doc = torch.from_numpy(b['doc']).long()
        o = m(ids, doc, return_all=True); x = m.emb(ids)
        B, T = ids.shape; G = T // k
        mp = lambda t: t.reshape(B, G, k, -1).mean(2)
        ell = torch.cat([mp(x), mp(o['L_enc'])], -1)                      # (B, G, 256)
        gdoc = doc[:, ::k]; toks = ids.reshape(B, G, k)
        first = torch.ones_like(gdoc, dtype=torch.bool); first[:, 1:] = gdoc[:, 1:] != gdoc[:, :-1]
        for r in range(B):
            real = gdoc[r] > 0
            out.append(dict(ell=ell[r][real].double(), doc=gdoc[r][real], first=first[r][real], tok=toks[r][real],
                            m=o['m'][r][real].double()))
    return out


def sigreg(z, gen, M=256, knots=17):
    w = torch.randn(z.shape[1], M, generator=gen, dtype=z.dtype); w = w / w.norm(dim=0, keepdim=True)
    s = z @ w
    t = torch.linspace(-5, 5, knots, dtype=z.dtype)
    ts = s[:, :, None] * t
    err = (torch.cos(ts).mean(0) - torch.exp(-t ** 2 / 2)) ** 2 + torch.sin(ts).mean(0) ** 2
    return torch.trapezoid(err * torch.exp(-t ** 2), t, dim=-1).mean() * z.shape[0]


def centre(z, doc, keep):
    """Document-centred samples: first groups masked, documents with >= 4 valid groups."""
    z, doc = z[keep], doc[keep]
    u, inv, cnt = torch.unique(doc, return_inverse=True, return_counts=True)
    ok = cnt[inv] >= 4
    mu = torch.zeros(len(u), z.shape[1], dtype=z.dtype).index_add_(0, inv, z) / cnt[:, None]
    return (z - mu[inv])[ok]


def retained(A, S):
    Pi = A.T @ torch.linalg.solve(A @ A.T, A)
    return float(torch.trace(Pi @ S) / torch.trace(S))


def probe(Ztr, Ttr, Zte, Tte):
    """Least-squares token-identity probe, one per token position; mean held-out accuracy over the 4 positions."""
    mu, sd = Ztr.mean(0), Ztr.std(0).clamp(min=1e-8)
    f = lambda Z: torch.cat([(Z - mu) / sd, torch.ones(len(Z), 1, dtype=Z.dtype)], 1)
    Xtr, Xte = f(Ztr), f(Zte)
    G = Xtr.T @ Xtr; G += 1e-3 * torch.trace(G) / G.shape[0] * torch.eye(G.shape[0], dtype=G.dtype)
    accs = []
    for j in range(Ttr.shape[1]):
        vocab, y = torch.unique(Ttr[:, j], return_inverse=True)
        XtY = torch.zeros(Xtr.shape[1], len(vocab), dtype=Xtr.dtype).index_add_(1, y, Xtr.T.contiguous())
        W = torch.linalg.solve(G, XtY)
        pred = vocab[(Xte @ W).argmax(1)]
        accs.append(float((pred == Tte[:, j]).double().mean()))
    return float(np.mean(accs)), accs


def a7(data, steps, seed):
    tr = [d for i, d in enumerate(data) if i % 2 == 0]; te = [d for i, d in enumerate(data) if i % 2 == 1]
    cat = lambda ds, key: torch.cat([d[key] for d in ds])
    Ltr, Lte = cat(tr, 'ell'), cat(te, 'ell')
    Ttr, Tte = cat(tr, 'tok'), cat(te, 'tok')
    S = torch.cov(Ltr.T)
    Cw = centre(Ltr, cat(tr, 'doc') + 0, ~cat(tr, 'first'))
    Sw = Cw.T @ Cw / len(Cw)
    ev, U = torch.linalg.eigh(S); U, ev = U[:, -64:], ev[-64:]
    A_pca = (U / ev.sqrt()).T                                            # whitened top-64 principal directions
    # SIGReg-only map, trained on whole training rows (8 per step), as T_loc would be
    gen = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    lin = torch.nn.Linear(256, 64, bias=False).double()
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    # rows in `tr` carry their own document ids; offset them per row so documents never merge across rows
    for i, d in enumerate(tr): d['docu'] = d['doc'] + 100000 * i
    log = []
    for step in range(steps):
        pick = torch.randint(0, len(tr), (8,), generator=gen)
        L = torch.cat([tr[i]['ell'] for i in pick]); D = torch.cat([tr[i]['docu'] for i in pick])
        F = torch.cat([tr[i]['first'] for i in pick])
        z = centre(lin(L), D, ~F)
        loss = sigreg(z, gen)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == steps - 1: log.append((step, round(float(loss.detach()), 3)))
    A_sig = lin.weight.detach()
    # SIGReg statistic of each map on held-out documents (lower = closer to N(0, I) per direction)
    for i, d in enumerate(te): d['docu'] = d['doc'] + 100000 * i
    Dte, Fte = cat(te, 'docu'), cat(te, 'first')
    g2 = torch.Generator().manual_seed(seed + 1)
    ep = {nm: float(sigreg(centre((Lte - Ltr.mean(0)) @ A.T, Dte, ~Fte), g2)) for nm, A in (('sigreg', A_sig), ('pca', A_pca))}
    rnd = torch.linalg.qr(torch.randn(256, 64, generator=torch.Generator().manual_seed(seed + 2), dtype=torch.float64))[0].T
    res = dict(n_train_groups=len(Ltr), n_test_groups=len(Lte), sigreg_loss_log=log, heldout_EP=ep)
    for nm, A in (('sigreg_map', A_sig), ('pca64', A_pca), ('random64', rnd)):
        acc, per = probe(Ltr @ A.T, Ttr, Lte @ A.T, Tte)
        res[nm] = dict(retained_var=retained(A, S), retained_var_within=retained(A, Sw), probe=acc, probe_per_pos=per)
    acc, per = probe(Ltr, Ttr, Lte, Tte)
    res['full256'] = dict(probe=acc, probe_per_pos=per)
    gap = res['pca64']['probe'] - res['sigreg_map']['probe']
    res['rule'] = dict(gap_points=100 * gap, statistic_projectors_default=bool(gap > 0.05))
    return res


def b4(data, cb):
    M = torch.cat([d['m'] for d in data]).float()
    z = cb.normalise(M).reshape(-1, cb.S, cb.ds).double()
    code = cb.code.double()
    out = {}
    for s in range(cb.S):
        d2 = (z[:, s, None, :] - code[s][None]).pow(2).sum(-1)              # (n, N)
        sig2 = float(d2.min(1).values.mean() / cb.ds)
        def H(T):
            p = torch.softmax(-d2 / T, 1); return float(-(p * p.clamp(min=1e-300).log2()).sum(1).mean())
        T_gmm = 2 * sig2
        lo, hi = 1e-4, 1e4
        for _ in range(100):                                                # bisection on log T for H = 1 bit
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if H(mid) < 1.0 else (lo, mid)
        T1 = math.sqrt(lo * hi)
        lg = -d2 / T_gmm
        out[f'seg{s}'] = dict(sigma2=sig2, T_gmm=T_gmm, H_at_T_gmm_bits=H(T_gmm), T_1bit=T1, ratio_T1_over_Tgmm=T1 / T_gmm,
                              logit_std_at_T_gmm=float(lg.std(1).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, default=24); ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=14); ap.add_argument('--out', default=str(HERE / 'checks' / 'projectors.json'))
    a = ap.parse_args()
    torch.set_num_threads(2)
    t0 = time.time()
    tk = C.TK.Tok(); m, cb, z = C.FE.load(C.V13 / 'runs' / 'base' / 'final.pt', torch.device('cpu'), tk)
    data = []
    for f in C.FAMS:
        rows = C.rows_for(f, a.rows, tk)
        for d in collect(m, cb, rows, m.c.k): d['fam'] = f; data.append(d)
    print(f'collected {len(data)} rows in {time.time() - t0:.0f}s', flush=True)
    res = dict(a7=a7(data, a.steps, a.seed), b4=b4(data, cb), rows_per_family=a.rows, seconds=round(time.time() - t0))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
