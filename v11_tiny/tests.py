"""Correctness tests for v11_tiny (v11.md §7.6 subset). CPU fp32 unless stated.

  python -B -m v11_tiny.tests        (writes v11_tiny/tests_result.json)
"""
import copy, json, math, time
from pathlib import Path
import torch
from torch.nn import functional as F
from v10_10M.data import Corpus
from .model import build, counts, V11Tiny, ARMS
from .losses import compute, gaussian_nll, bridge_losses, probe_loss, sample_boundaries

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'v10_3_tiny' / 'data'
RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append(dict(test=name, passed=bool(ok), detail=str(detail)))
    print(('PASS ' if ok else 'FAIL ') + name + (' :: ' + str(detail) if detail else ''), flush=True)


def batches():
    c = Corpus('train', seed=7, data_dir=DATA); out = {}
    while len(out) < 4:
        b = c.batch(4, 256, progress=0)
        out.setdefault(b['family'], b)
    return out


def fresh(arm, seed=0):
    torch.manual_seed(seed); m = build(arm); m.train()
    if isinstance(m, V11Tiny):
        b = BATCHES['text']; o = m(b['ids'], b['lengths']); m.update_target_buffers(o['pooled'], o['valid_groups'])
    return m


def grads(model, loss, prefixes):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    g = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
    out = {}
    for (n, p), gi in zip(named, g):
        for pre in prefixes:
            if n.startswith(pre):
                out[pre] = out.get(pre, 0.) + (0. if gi is None else float(gi.abs().sum()))
    return out


def parts_for(arm, model, b, step=500):
    out = model(b['ids'], b['lengths'])
    return out, compute(arm, model, out, b, step=step, warmup=200, generator=torch.Generator().manual_seed(3), S=2)


def test_counts():
    expect = dict(C0=(311704, 311704), ORD=(475664, 475664), ORDE=(483224, 483224), X8=(503428, 474364), NOAUX=(501388, 472324),
                  K1NB=(501390, 472324), K1=(539414, 472324))
    for arm in ARMS:
        c = counts(build(arm)); ok = (c['dense'], c['dense_deployed']) == expect[arm]
        check(f'parameter count {arm}', ok, c)
    m = build('K1'); check('bridge = one block (38,024)', sum(p.numel() for p in m.bridge.parameters()) == 38024)


def test_gradient_contracts():
    m = fresh('K1'); b = BATCHES['text']
    out, p = parts_for('K1', m, b)
    trunk = ('embedding.', 'engram.', 'e1.', 'e2.', 'exit_reader.', 'e3.', 'pooler.')
    g = grads(m, p['v2'], trunk + ('core.layers', 'head.', 'encoder_concept_crc.', 'decoder.', 'log_sigma2'))
    check('L_v2 reaches no encoder/pooler/embedding parameter', all(g.get(k, 0.) == 0. for k in trunk), g)
    check('L_v2 reaches core blocks, head, log_sigma2', g['core.layers'] > 0 and g['head.'] > 0 and g['log_sigma2'] > 0)
    check('L_v2 reaches no decoder parameter', g.get('decoder.', 0.) == 0.)
    g = grads(m, p['v1'], ('decoder.', 'pooler.', 'e1.', 'core.layers', 'bridge.', 'probe.'))
    check('L_v1: decoder 0, pooler/encoder/core > 0', g['decoder.'] == 0. and g['pooler.'] > 0 and g['e1.'] > 0 and g['core.layers'] > 0, g)
    g = grads(m, p['vq'], ('head.codebook', 'head.logits', 'core.', 'pooler.', 'e1.'))
    check('L_vq reaches only the codebook', g['head.codebook'] > 0 and all(g[k] == 0. for k in ('head.logits', 'core.', 'pooler.', 'e1.')), g)
    bridge_total = p['bridge_h'] + p['bridge_kl']
    g = grads(m, bridge_total, ('decoder.', 'bridge.', 'core.layers', 'pooler.', 'e1.', 'engram.', 'probe.', 'head.'))
    check('bridge losses: decoder (incl. outn) 0; bridge, core, pooler, encoder > 0',
          g['decoder.'] == 0. and all(g[k] > 0 for k in ('bridge.', 'core.layers', 'pooler.', 'e1.')), g)
    check('bridge losses: PQ head 0 (anchor is u1, not the forecast)', g['head.'] == 0.)
    # Detached-view contract: embedding gradient from L_KL equals that of an identical model whose
    # vocabulary view is an explicit frozen clone, i.e. only the encoder-input use of E receives gradient.
    gk = torch.autograd.grad(p['bridge_kl'], [m.embedding.weight], retain_graph=True)[0]
    m2 = copy.deepcopy(m); out2 = m2(b['ids'], b['lengths'])
    import v11_tiny.losses as L
    lh2, lkl2, _ = L.bridge_losses(m2, out2, b['ids'], b['lengths'])
    gk2 = torch.autograd.grad(lkl2, [m2.embedding.weight])[0]
    check('L_KL embedding gradient reproducible', torch.allclose(gk, gk2, atol=1e-7))
    # Gradient through the encoder-input use only: zero it out by detaching the encoder's embedding input.
    m3 = copy.deepcopy(m)
    real = m3.embedding.forward
    m3.embedding.forward = lambda ids: real(ids).detach()
    out3 = m3(b['ids'], b['lengths']); _, lkl3, _ = L.bridge_losses(m3, out3, b['ids'], b['lengths'])
    g3 = torch.autograd.grad(lkl3, [m3.embedding.weight], allow_unused=True)[0]
    check('L_KL: embedding-as-head receives no gradient (zero once the encoder-input use is cut)',
          g3 is None or float(g3.abs().sum()) == 0., None if g3 is None else float(g3.abs().sum()))
    check('L_KL: embedding-as-encoder-input receives gradient', float(gk.abs().sum()) > 0)
    g = grads(m, p['probe'], ('probe.', 'embedding.', 'core.', 'head.', 'decoder.', 'pooler.'))
    check('L_probe reaches only probe.*', g['probe.'] > 0 and all(g[k] == 0. for k in ('embedding.', 'core.', 'head.', 'decoder.', 'pooler.')), g)
    g = grads(m, p['token'], ('bridge.', 'probe.', 'log_sigma1', 'log_sigma2', 'core.', 'pooler.', 'head.'))
    check('L_NTP: bridge/probe/sigmas 0; core/pooler/head > 0',
          all(g.get(k, 0.) == 0. for k in ('bridge.', 'probe.', 'log_sigma1', 'log_sigma2')) and g['core.'] > 0 and g['pooler.'] > 0 and g['head.'] > 0, g)


def test_noaux_k1nb():
    for arm in ('NOAUX', 'K1NB'):
        m = fresh(arm); out, p = parts_for(arm, m, BATCHES['text'])
        absent = [k for k in ('bridge_h', 'bridge_kl') if p[k] is not None]
        if arm == 'NOAUX': absent += [k for k in ('v1', 'v2', 'vq') if p[k] is not None]
        check(f'{arm}: absent terms are None (not 0)', not absent, absent)
        g = grads(m, p['total'] - p['probe'], ('core.', 'pooler.'))
        check(f'{arm}: trunk still trained', g['core.'] > 0)


def test_gaussian():
    pred = torch.randn(3, 5, 2, 4); tgt = torch.randn(3, 5, 2, 4); valid = torch.ones(3, 5)
    s = torch.tensor(.3, requires_grad=True)
    nll, mse = gaussian_nll(pred, tgt, s, valid)
    D = 8; sq = (pred - tgt).square().sum((-1, -2))
    hand = (sq / (2 * D * math.exp(.3)) + .5 * (math.log(2 * math.pi) + .3)).mean()
    check('Gaussian NLL closed form (K = 1)', abs(float(nll) - float(hand)) < 1e-6, (float(nll), float(hand)))
    s_opt = torch.tensor(math.log(float(mse)), requires_grad=True)
    g = torch.autograd.grad(gaussian_nll(pred, tgt, s_opt, valid)[0], s_opt)[0]
    check('dNLL/dlog_sigma = 0 at sigma^2 = mean sq/D', abs(float(g)) < 1e-5, float(g))
    s_low = torch.tensor(-20., requires_grad=True)
    check('log sigma floor applied', float(gaussian_nll(pred, tgt, s_low, valid)[0]) < 1e6)


def test_boundaries():
    elig = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.bool)
    rows, groups = sample_boundaries(elig, 2, torch.Generator().manual_seed(0))
    pairs = sorted(zip(rows.tolist(), groups.tolist()))
    ok = all(elig[r, g] for r, g in pairs) and sum(r == 0 for r, _ in pairs) == 2 and sum(r == 2 for r, _ in pairs) == 1 and not any(r == 1 for r, _ in pairs)
    check('boundary sampler: only eligible, at most S per document, distinct', ok and len(set(pairs)) == len(pairs), pairs)
    r2, g2 = sample_boundaries(elig, 2, torch.Generator().manual_seed(0))
    check('boundary sampler deterministic under seed', torch.equal(rows, r2) and torch.equal(groups, g2))


@torch.no_grad()
def test_forward_contracts():
    m = fresh('K1'); m.eval(); b = BATCHES['math']
    mu, var = m.target_mu.clone(), m.target_var.clone()
    o2 = m(b['ids'], b['lengths']); o1 = m(b['ids'], b['lengths'], visits=1)
    check('visits=1 equals states[0] of visits=2', torch.allclose(o1['u'], o2['states'][0], atol=1e-6))
    ov = m(b['ids'], b['lengths'], use_visit1=True)
    check('use_visit1 feeds u1 to the decoder', torch.allclose(ov['u'], o2['u1']) and not torch.allclose(ov['logits'], o2['logits']))
    m.train(); m.eval()
    check('eval forward leaves target buffers identical', torch.equal(mu, m.target_mu) and torch.equal(var, m.target_var))
    m.update_target_buffers(o2['pooled'], o2['valid_groups'])
    check('update_target_buffers is a no-op in eval()', torch.equal(mu, m.target_mu))
    logits = m(b['ids'], b['lengths'])['logits']; m.bridge = None; m.probe = None
    check('deleting bridge and probe leaves logits identical', torch.equal(logits, m(b['ids'], b['lengths'])['logits']))
    m = fresh('K1'); m.eval()
    flags = dict(ablate=True, ablate_aligned=True, ablate_reader=True, zero_state=True, zero_forecast_memory=True, use_visit1=True)
    for k in flags:
        check(f'intervention {k} changes logits', not torch.allclose(m(b['ids'], b['lengths'], **{k: True})['logits'], logits, atol=0))
    try:
        m(b['ids'], b['lengths'], ablate_aligend=True); check('unknown forward kwarg raises', False)
    except TypeError:
        check('unknown forward kwarg raises', True)
    m.train(); m.target_ready.fill_(False); o = m(b['ids'], b['lengths'])
    m.update_target_buffers(o['pooled'], o['valid_groups'])
    x = o['pooled'][o['valid_groups']]
    check('buffers initialised from the first batch', torch.allclose(m.target_mu, x.mean(0), atol=1e-6))
    tau = m.normalized_target(o['pooled'])[o['valid_groups']]
    check('normalized target ~ zero mean, unit variance on that batch', float(tau.mean(0).abs().max()) < 1e-4 and abs(float(tau.var(0, unbiased=False).mean()) - 1) < .05)


@torch.no_grad()
def test_causality():
    torch.manual_seed(5); seq = torch.randint(4, 1024, (1, 40))
    for arm in ('ORD', 'ORDE', 'NOAUX', 'K1'):
        m = fresh(arm); m.eval(); worst = 0.
        for cut in (1, 7, 8, 9, 15, 16, 17, 33):
            other = seq.clone(); other[0, cut:] = torch.randint(4, 1024, (40 - cut,))
            a = m(seq, torch.tensor([40]))['logits'][0, :cut]; c = m(other, torch.tensor([40]))['logits'][0, :cut]
            worst = max(worst, float((a - c).abs().max()))
        check(f'future-token perturbation {arm} (cuts 1,7,8,9,15,16,17,33)', worst == 0., worst)
        for n in (1, 7, 8, 9, 15, 16, 17, 33):
            l = m(seq[:, :n], torch.tensor([n]))['logits']
            full = m(seq, torch.tensor([40]))['logits'][:, :n]
            worst = max(worst, float((l - full).abs().max()))
        check(f'prefix length invariance {arm} (lengths 1..33)', worst < 1e-5, worst)
        # padding: a right-padded row must give the same logits on its valid rows
        padded = torch.cat((seq[:, :17], torch.zeros(1, 23, dtype=torch.long)), 1)
        d = float((m(padded, torch.tensor([17]))['logits'][0, :17] - m(seq[:, :17], torch.tensor([17]))['logits'][0]).abs().max())
        check(f'right padding isolation {arm}', d < 1e-5, d)


def test_bridge_edge():
    m = fresh('K1'); ids = torch.randint(4, 1024, (2, 8)); lengths = torch.tensor([8, 8])
    out = m(ids, lengths); r = bridge_losses(m, out, ids, lengths)
    check('bridge with no rows past the last boundary returns None', r[0] is None and r[2] == 0, r[2])
    ids = torch.randint(4, 1024, (1, 9)); out = m(ids, torch.tensor([9])); r = bridge_losses(m, out, ids, torch.tensor([9]))
    check('length 9: exactly one bridge row (t = 8)', int(r[2]) == 1, r[2])


def test_mps_parity():
    if not torch.backends.mps.is_available(): check('CPU/MPS parity', True, 'MPS unavailable: skipped'); return
    for fam in ('text', 'math'):
        b = BATCHES[fam]; m = fresh('K1'); m2 = copy.deepcopy(m).place('mps')
        p = parts_for('K1', m, b)[1]
        bm = {k: (v.to('mps') if torch.is_tensor(v) else v) for k, v in b.items()}
        out = m2(bm['ids'], bm['lengths'])
        p2 = compute('K1', m2, out, bm, step=500, warmup=200, generator=torch.Generator().manual_seed(3), S=2)
        dl = max(abs(float(p[k]) - float(p2[k])) for k in ('token', 'v1', 'v2', 'bridge_h', 'bridge_kl'))
        p['total'].backward(); p2['total'].backward()
        dg = max(float((a.grad - c.grad.cpu()).abs().max()) for (n, a), (_, c) in zip(m.named_parameters(), m2.named_parameters())
                 if a.grad is not None and n != 'engram.table.weight')
        check(f'CPU/MPS parity K1 {fam} (loss, grads)', dl < 1e-4 and dg < 1e-4, dict(loss=dl, grad=dg))


if __name__ == '__main__':
    t0 = time.time(); torch.set_num_threads(4)
    BATCHES = batches()
    for f in (test_counts, test_gradient_contracts, test_noaux_k1nb, test_gaussian, test_boundaries,
              test_forward_contracts, test_causality, test_bridge_edge, test_mps_parity):
        try: f()
        except Exception as e:
            import traceback; traceback.print_exc(); check(f.__name__ + ' raised', False, repr(e))
    passed = sum(r['passed'] for r in RESULTS)
    (ROOT / 'tests_result.json').write_text(json.dumps(dict(passed=passed, total=len(RESULTS), seconds=time.time() - t0,
                                                            results=RESULTS), indent=1) + '\n')
    print(f'{passed}/{len(RESULTS)} passed in {time.time() - t0:.0f}s')
