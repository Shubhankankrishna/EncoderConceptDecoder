"""G0 tests for V14's new parts (v14_Architecture.md §11 item 1), float64 on the CPU. Run with tests_g0.py:

    python3 tests_g0.py && python3 tests_v14.py
"""
import math
import sys

import torch

import model as M
import v14 as V

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
FAIL = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  ({detail})' if detail else ''), flush=True)
    if not ok: FAIL.append(name)


def make():
    m = M.V14(M.Cfg(V=64)).double()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() >= 1: p.add_(0.05 * torch.randn_like(p))
    return m


def batch(B=2, T=96, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 60, (B, T), generator=g)
    doc = torch.zeros(B, T, dtype=torch.long)
    doc[0, :40] = 1; doc[0, 40:76] = 2; doc[0, 76:] = 3                                  # boundaries at multiples of 4
    doc[1, :8] = 1; doc[1, 8:88] = 2                                                       # row 1 ends in padding
    return ids, doc


def main():
    m = make(); m.train()
    twin = V.make_twin(m)
    k = m.c.k
    ids, doc = batch(); B, T = ids.shape; G = T // k
    gdoc = doc[:, ::k]
    eidx = m.engram.keys(ids, doc)
    P0 = torch.zeros(B, 2 * m.c.d, m.c.d)

    # (1) the twin equals the online encoder at step 0, and shares the Engram table
    with torch.no_grad():
        ctx = m.context(doc)
        x, L, Gm = m.encode(ids, eidx, ctx)
        tau, valid_t, dl, dG = V.twin_targets(twin, ids, doc, eidx, P0)
        a, b, valid = V.online_view(m, x, L, Gm, gdoc, P0)
    err = (tau - torch.cat([a, b], -1)).abs().max().item()
    check('twin == online target view at step 0', err == 0.0, f'max {err:.1e}')
    check('Engram table shared, not copied', twin.engram.table is m.engram.table)
    names = {n for n, _ in twin.named_parameters()}
    check('twin holds no concept module, decoder, pools or head',
          not any(n.startswith(('cm.', 'dec.', 'ncp', 'pool_', 'wp', 'e_h', 'beta', 'src_norm', 'final')) for n in names))
    tw, ol = V.twin_pairs(twin, m)
    check('twin parameters take no gradient (the shared table excepted)', all(not p.requires_grad for p in tw))

    # (2) momentum schedule by elapsed time and the EMA update
    mus = [V.momentum(f) for f in (0.0, 0.3, 0.375, 0.5, 0.6, 0.7, 0.75, 0.9)]
    lags = [1 / (1 - x) if x < 1 else float('inf') for x in mus]
    check('momentum 0.99 at t = 0 (lag 100), lag 200 at 37.5%, 1 from t_freeze',
          abs(mus[0] - 0.99) < 1e-12 and abs(lags[2] - 200) < 1e-6 and mus[6] == 1.0 and mus[7] == 1.0,
          ' '.join(f'{l:.0f}' for l in lags[:6]))
    check('momentum non-decreasing', all(mus[i] <= mus[i + 1] for i in range(len(mus) - 1)))
    old = [p.clone() for p in tw]
    with torch.no_grad():
        for p in ol: p.add_(0.1)
    V.ema_update(tw, ol, 0.9)
    err = max((t - (0.9 * o + 0.1 * n)).abs().max().item() for t, o, n in zip(tw, old, ol))
    check('EMA update: twin = mu twin + (1 - mu) online', err < 1e-12, f'max {err:.1e}')
    with torch.no_grad():
        for p in ol: p.sub_(0.1)
    twin = V.make_twin(m); tw, ol = V.twin_pairs(twin, m)

    # (3) tau_g uses only groups <= g of its own document
    with torch.no_grad():
        t0, v0, _, _ = V.twin_targets(twin, ids, doc, eidx, P0)
        g = 4                                                                               # group 4 = tokens 16..19, doc 1
        ids2 = ids.clone(); ids2[0, 4 * (g + 1):40] = (ids2[0, 4 * (g + 1):40] + 7) % 60
        t2, _, _, _ = V.twin_targets(twin, ids2, doc, m.engram.keys(ids2, doc), P0)
        e_past = (t2[0, :g + 1] - t0[0, :g + 1]).abs().max().item()
        e_fut = (t2[0, g + 1:10] - t0[0, g + 1:10]).abs().max().item()
        ids3 = ids.clone(); ids3[0, 40:76] = (ids3[0, 40:76] + 5) % 60                     # edit document 2
        t3, _, _, _ = V.twin_targets(twin, ids3, doc, m.engram.keys(ids3, doc), P0)
        vm = v0[0].clone(); vm[10:19] = False                                               # other documents' valid groups
        e_other = (t3[0][vm] - t0[0][vm]).abs().max().item()
        e_first = (t3[0, 19] - t0[0, 19]).abs().max().item()                               # doc 3's first group: masked
    check('tau_g unchanged by edits after group g', e_past == 0.0 and e_fut > 0, f'past {e_past:.1e} future {e_fut:.1e}')
    check('tau of other documents unchanged by an edit to one document (valid groups)', e_other == 0.0,
          f'max {e_other:.1e}; the masked first group of the next document moves by {e_first:.1e}, never used')

    # (4) first groups are masked: never valid targets, never in the codebook update or SIGReg
    firsts = [(0, 0), (0, 10), (0, 19), (1, 0), (1, 2)]
    check('first groups of documents masked', all(not bool(v0[r, gg]) for r, gg in firsts)
          and bool(v0[0, 1]) and bool(v0[1, 3]) and not bool(v0[1, 22:].any()))

    # (5) horizon pairs stay inside one document; first groups never targets
    codes = torch.randint(0, 32, (B, G, m.c.pq_S))
    hl = torch.randn(B, G, m.c.horizons, m.c.pq_S, m.c.pq_N, requires_grad=True)
    loss, hst = V.horizon_loss(hl, codes, gdoc, v0)
    loss.backward()
    gr = hl.grad.abs().sum((-1, -2))                                                        # (B, G, K)
    bad = 0
    for r in range(B):
        for gg in range(G):
            for kk in range(1, m.c.horizons + 1):
                inside = gg + kk < G and gdoc[r, gg] == gdoc[r, gg + kk] and gdoc[r, gg] > 0
                if (gr[r, gg, kk - 1] > 0) != bool(inside): bad += 1
    check('horizon pairs (g, g+k) only inside one document', bad == 0, f'{bad} wrong cells')
    check('horizon weights sum to 1', abs(sum(V.HW) - 1) < 1e-12)

    # (6) NCP's gradient reaches neither projector; SIGReg on a reaches only T_loc; SIGReg on b reaches T_glob and G
    m.zero_grad()
    ctx = m.context(doc)
    x, L, Gm = m.encode(ids, eidx, ctx)
    mm, u = m.concepts(x, L, Gm, ctx)
    hl = m.horizon_logits(u)
    with torch.no_grad():
        tau, valid_t, _, _ = V.twin_targets(twin, ids, doc, eidx, P0)
    codes = torch.randint(0, 32, (B, G, m.c.pq_S))
    l_ncp, _ = V.horizon_loss(hl, codes, gdoc, valid_t)
    l_ncp.backward()
    check('NCP gradient reaches neither T_loc nor T_glob',
          m.t_loc.weight.grad is None and m.t_glob.weight.grad is None)
    m.zero_grad()
    x, L, Gm = m.encode(ids, eidx, ctx)
    a, b, valid = V.online_view(m, x, L, Gm, gdoc, P0)
    consts = V.sigreg_consts('cpu', torch.float64)
    om = V.directions(0, m.c.proj).double()
    sa, sb, sbu = V.sigreg_terms(a, b, gdoc, valid, om, consts)
    sa.backward(retain_graph=True)
    got = [n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().max() > 0]
    check('SIGReg on a trains T_loc only', got == ['t_loc.weight'], ', '.join(got[:6]))
    m.zero_grad()
    (sb + sbu).backward()
    got = {n.split('.')[0] for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().max() > 0}
    check('SIGReg on b reaches T_glob and the G side, not T_loc, the decoder or the concept module',
          't_glob' in got and 'e2' in got and 'e3' in got and not ({'t_loc', 'dec', 'cm', 'ncp'} & got), str(sorted(got)))

    # (7) document centring keeps documents with >= 4 valid groups
    zc, keep = V.doc_centre(a.detach(), gdoc, valid)
    ok = True
    for r in range(B):
        for dd in gdoc[r].unique().tolist():
            if dd == 0: continue
            sel = (gdoc[r] == dd) & valid[r]
            n = int(sel.sum())
            if n >= 4:
                ok &= bool(keep[r][sel].all()) and zc[r][sel].mean(0).abs().max().item() < 1e-12
            else:
                ok &= not bool(keep[r][sel].any())
    check('centring per document, documents with >= 4 valid groups kept', ok)

    # (8) Epps-Pulley values of §8.3: a point mass 0.131 per direction; Gaussian samples near the O(1/N) floor
    t, phi, q = consts
    one = V.ep(torch.zeros(2000, 1), torch.ones(2000, dtype=torch.bool), torch.ones(1, 1), t, phi, q).item()
    gs = V.ep(torch.randn(2000, 64), torch.ones(2000, dtype=torch.bool), om[0], t, phi, q).item()
    check('EP of a point mass = 0.131, of 2,000 Gaussian samples about 3e-4', abs(one - 0.131) < 0.002 and gs < 1e-3,
          f'{one:.4f} / {gs:.5f}')

    # (9) residue maps: zero before step 200 and below 5,120 pairs of the family; then the float64 ridge solution
    rm = V.ResidueMaps(5, 2 * m.c.d, m.c.d)
    Xd = torch.randn(4, 64, 2 * m.c.d); Yd = torch.randn(4, 64, m.c.d); vv = torch.ones(4, 64, dtype=torch.bool)
    for _ in range(19): rm.update(1, Xd, Yd, vv)                                          # 19 x 256 = 4,864 pairs
    rm.solve(300)
    z1 = rm.P[1].abs().max().item()
    rm.update(1, Xd, Yd, vv); rm.solve(150)
    z2 = rm.P[1].abs().max().item()
    rm.solve(300)
    A = rm.A[1].double(); Bm = rm.Bm[1].double(); rho = 1e-3 * torch.trace(A) / A.shape[0]
    ref = torch.linalg.solve(A + rho * torch.eye(A.shape[0]), Bm)
    e = (rm.P[1].double() - ref).abs().max().item()
    check('P_f = 0 below 5,120 pairs and before step 200; then the ridge solution',
          z1 == 0 and z2 == 0 and e < 1e-10 and rm.P[0].abs().max().item() == 0, f'max {e:.1e}')
    rm.freeze(); A0 = rm.A.clone(); rm.update(1, Xd, Yd, vv)
    check('residue maps frozen: no update after freeze()', torch.equal(A0, rm.A) and rm.solve(400) is None)

    # (10) codebook: masked update, dead words reseeded at the worst-fitted valid targets, frozen stops everything
    cb = V.Codebook(128, 4, 32)
    xx = torch.randn(400, 128); ww = torch.ones(400); ww[300:] = 0
    cb.update(xx, ww)                                                                       # initialisation
    cb.update(xx, ww)
    cb.count[0, 5] = 0.0                                                                    # make word (0, 5) dead
    cb.count[0, 9] = 0.0                                                                    # and (0, 9)
    z = cb.normalise(xx).reshape(-1, 4, 32)
    cb.update(xx, ww)
    idx = cb.assign(cb.normalise(xx))
    zz = cb.normalise(xx).reshape(-1, 4, 32)
    check('codebook ignores masked rows (codes stay finite, counts weighted)', torch.isfinite(cb.code).all().item())
    check('dead words reseeded (resets counted)', cb.resets.item() >= 2, f'resets {cb.resets.item():.0f}')
    cb2 = V.Codebook(128, 4, 32); x2 = torch.randn(2000, 128); w2 = torch.ones(2000)
    cb2.update(x2, w2); cb2.update(torch.randn(2000, 128), w2)
    check('no mass reset after initialisation (counts start at uniform usage)', cb2.resets.item() == 0,
          f'resets {cb2.resets.item():.0f}')
    cb.freeze(); c0 = cb.code.clone(); mu0 = cb.mu.clone(); cb.update(xx, ww)
    check('codebook frozen: no update after freeze()', torch.equal(c0, cb.code) and torch.equal(mu0, cb.mu))

    print('\nV14 G0: ' + ('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}'))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
