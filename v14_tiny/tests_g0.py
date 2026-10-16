"""G0 correctness tests (v13_NOTES.md §12.5), float64 on the CPU, exact. Run before any training:

    python3 tests_g0.py

(a) reach: L_enc rows move for an edit exactly R_L = 53 tokens back and not 54; logits, with the concept read held
    fixed, move for an edit R_D = 68 back and not 69. (b) causality of the whole model, concept path included.
(c) packed documents are isolated: editing one document leaves every other document's logits bit-identical, in
    either order. (d) release: concept g reaches row 4g+3 and not row 4g+2. (e) chunkwise GDN-2 equals the
    token-by-token recurrence with document resets. (f) the codebook buffers receive no gradient.
"""
import sys

import torch

import model as M

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
# Exact reach of this layout: Engram follows the first S layer, so its n-gram (+2) and that layer's window (+7) are
# parallel paths into the convolution (+9): R_L = 7 + 9 + 5 x 7 = 51 and R_D = R_L + 15 = 66, inside the design
# bound 6 x 7 + 2 + 9 = 53 / 68 (v13_NOTES.md §3.8 states the bound, not the exact value).
R_L, R_D = 51, 66
FAIL = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  ({detail})' if detail else ''), flush=True)
    if not ok: FAIL.append(name)


def make():
    m = M.V13(M.Cfg(V=64)).double().eval()
    with torch.no_grad():                                         # move every zero-initialised path off zero
        for n, p in m.named_parameters():
            if p.dim() >= 1: p.add_(0.05 * torch.randn_like(p))
    return m


def run(m, ids, doc, hook=None):
    with torch.no_grad():
        return m(ids, doc, u_hook=hook, return_all=True)


def main():
    m = make(); T = 192; B = 1
    ids = torch.randint(3, 64, (B, T)); doc = torch.ones(B, T, dtype=torch.long)
    base = run(m, ids, doc)
    u0 = base['u']

    # (a) reach of L_enc and of the decoder with the concept read held fixed
    p = 100
    ids2 = ids.clone(); ids2[0, p] = (ids[0, p] + 7) % 61 + 3
    o2 = run(m, ids2, doc, hook=lambda u, mm, c: u0)
    dL = (o2['L_enc'] - base['L_enc']).abs().amax(-1)[0]
    moved = torch.nonzero(dL > 0).flatten()
    check(f'reach L_enc = {R_L} (bound 53)', moved.min().item() == p and moved.max().item() == p + R_L,
          f'rows {moved.min().item()}..{moved.max().item()}, edit at {p}')
    dlog = (o2['logits'] - base['logits']).abs().amax(-1)[0]
    moved = torch.nonzero(dlog > 0).flatten()
    check(f'reach decoder = {R_D} (bound 68, concept read fixed)', moved.min().item() == p and moved.max().item() == p + R_D,
          f'rows {moved.min().item()}..{moved.max().item()}')

    # (b) causality of the full model (concept path included)
    o3 = run(m, ids2, doc)
    d = (o3['logits'] - base['logits']).abs().amax(-1)[0]
    check('causal: rows before the edit unchanged', d[:p].abs().max().item() == 0.0, f'max {d[:p].max().item():.2e}')
    check('global: the concept path carries the edit beyond R_D', (d[p + R_D + 1:] > 0).any().item())

    # (c) packed-document isolation, both orders, different boundary offsets
    for cut in (64, 100, 128):
        docp = torch.cat([torch.ones(1, cut, dtype=torch.long), 2 * torch.ones(1, T - cut, dtype=torch.long)], 1)
        a = run(m, ids, docp)
        idsA = ids.clone(); idsA[0, :cut] = torch.randint(3, 64, (cut,))
        b = run(m, idsA, docp)
        dB = (b['logits'][0, cut:] - a['logits'][0, cut:]).abs().max().item()
        check(f'isolation: later document unchanged (boundary {cut})', dB == 0.0, f'max {dB:.2e}')
        idsB = ids.clone(); idsB[0, cut:] = torch.randint(3, 64, (T - cut,))
        c = run(m, idsB, docp)
        dA = (c['logits'][0, :cut] - a['logits'][0, :cut]).abs().max().item()
        check(f'isolation: earlier document unchanged (boundary {cut})', dA == 0.0, f'max {dA:.2e}')
        # a document alone at the start of a row equals the same document after another one (positions shift by cut)
        if (T - cut) % 16: continue
        solo = run(m, ids[:, cut:].contiguous(), torch.ones(1, T - cut, dtype=torch.long))
        dd = (solo['logits'][0, :T - cut - 0] - a['logits'][0, cut:]).abs().max().item()
        check(f'shift invariance: document after a boundary at {cut} = document alone', dd < 1e-9, f'max {dd:.2e}')

    # (d) release rule: concept g reaches row 4g+3, not 4g+2
    g = 20
    def bump(u, mm, c):
        u = u.clone(); u[:, g] += 1.0; return u
    o4 = run(m, ids, doc, hook=bump)
    d = (o4['logits'] - base['logits']).abs().amax(-1)[0]
    check('release: row 4g+2 unchanged, row 4g+3 changed', d[4 * g + 2].item() == 0.0 and d[4 * g + 3].item() > 0,
          f'd[4g+2]={d[4 * g + 2].item():.1e} d[4g+3]={d[4 * g + 3].item():.1e}')
    check('release: no row before 4g+3 changed', d[:4 * g + 3].max().item() == 0.0)

    # (e) chunkwise GDN-2 vs the token recurrence, with document resets
    Bz, Tz, H, dk = 2, 96, 2, 8
    q = torch.nn.functional.normalize(torch.randn(Bz, Tz, H, dk), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(Bz, Tz, H, dk), dim=-1)
    u = torch.randn(Bz, Tz, H, dk); a = torch.rand(Bz, Tz, H, dk) * k
    g_ = -torch.rand(Bz, Tz, H, dk) * 2.0
    dz = torch.ones(Bz, Tz, dtype=torch.long); dz[0, 37:] = 2; dz[0, 70:] = 3; dz[1, 16:] = 2; dz[1, 17:] = 3
    oc = M.gdn2_chunk(q, k, u, a, g_, dz, 16); orf = M.gdn2_reference(q, k, u, a, g_, dz)
    err = (oc - orf).abs().max().item()
    check('GDN-2 chunkwise = recurrence (float64)', err < 1e-10, f'max {err:.1e}')
    oc32 = M.gdn2_chunk(*(z.float() for z in (q, k, u, a, g_)), dz, 16)
    err32 = (oc32.double() - orf).abs().max().item()
    check('GDN-2 chunkwise in float32 within 1e-4', err32 < 1e-4, f'max {err32:.1e}')

    # (f) gradient reaches every parameter, codebook-free
    m.train(); m.zero_grad()
    out = m(ids, doc, return_all=True)
    (out['logits'].pow(2).mean() + out['ncp'].pow(2).mean()).backward()
    dead = [n for n, p_ in m.named_parameters() if (p_.grad is None or p_.grad.abs().max().item() == 0)
            and not n.startswith(('t_loc', 't_glob'))]                  # V14: the projectors train only by SIGReg
    check('every parameter except T_loc, T_glob receives gradient from NTP + NCP', not dead, ', '.join(dead[:8]))
    tg = [n for n, p_ in m.named_parameters() if n.startswith(('t_loc', 't_glob')) and p_.grad is not None
          and p_.grad.abs().max().item() > 0]
    check('T_loc and T_glob receive no gradient from NTP + NCP', not tg, ', '.join(tg))

    print('\nG0:', 'ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
