"""Bit-exact contract tests for the end-to-end pipeline (float64, randomised weights so no path is hidden by zero-init)."""
import torch, math, torch.nn.functional as F
from e2e import *
torch.set_default_dtype(torch.float64); torch.manual_seed(0)

def randomize(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if 'tab.' not in n and not n.endswith('core.e'): p.add_(torch.randn_like(p) * 0.2)

g = torch.Generator().manual_seed(1)
ids, tgt, meta = make_batch(1, g, hops=(3,))
EQPOS = [q[0] + 8 for q in meta[0][1]]
for arm, K in [('loop', 1), ('loopz', 4), ('stack4', 1), ('ident', 1)]:
    m = Model(arm, K); randomize(m); m.eval()
    k = 1 if K > 1 else None
    f = lambda i, doc=None, off=False, R=3: m(i, torch.zeros_like(i) if doc is None else doc, R, k, off)[0]
    with torch.no_grad():
        base = f(ids)
        # 1. causality: editing token p never changes logits at t < p
        for p in (37, 70, 110):
            i2 = ids.clone(); i2[0, p] = (ids[0, p] + 5) % VT; d = (f(i2) - base).abs().amax(-1)[0]
            assert not (d[:p] != 0).any() and d[p] != 0, (arm, p)
        # 2. far information reaches '=' ONLY through the concept plan
        i2 = ids.clone(); i2[0, 4] = (ids[0, 4] + 1) % NE
        far = (f(i2) - base).abs().amax(-1)[0, EQPOS]
        off = (f(i2, off=True) - f(ids, off=True)).abs().amax(-1)[0, EQPOS]
        assert (off == 0).all(), (arm, off)
        # 3. document isolation (two documents packed in one row)
        doc = torch.zeros_like(ids); doc[0, 64:] = 1
        i3 = ids.clone(); i3[0, 60] = (ids[0, 60] + 3) % VT
        assert torch.equal(f(ids, doc)[0, 64:], f(i3, doc)[0, 64:]), arm
    print(f"PASS {arm:6s}{'(K=4)' if K > 1 else '     '}: causal at every position | far fact -> '=' changes by {far.max():.2e} with plan, "
          f"exactly 0 with plan off | no leakage across packed documents")

# 4. gradient paths (float32 training model)
torch.set_default_dtype(torch.float32)
m = Model('loopz', 4); ids, tgt, _ = make_batch(8, g); doc = torch.zeros_like(ids)
tok, c_hat, _ = m.enc(ids, doc); m.core.vq_init(c_hat.detach())
u, lg, cp = m.core(c_hat, doc[:, ::4], 3, 2); out = m.dec(tok, u, cp, doc); lt = out @ m.enc.E.weight.T
mask = tgt >= 0; ntp = F.cross_entropy(lt[mask], tgt[mask]); ncp = ncp_ce(m.core, lg, c_hat, doc[:, ::4]).mean(); vq = m.core.vq_loss(c_hat, 1)
gr = lambda L, P: [torch.autograd.grad(L, p, retain_graph=True, allow_unused=True)[0] for p in P]
P = [m.enc.layers[7].attn.q.weight, m.enc.layers[3].ffn.w1.weight, m.core.layer.attn.q.weight, m.core.z, m.core.e, m.dec.Wp.weight]
names = ['enc FULL layer 8', 'enc FULL layer 4', 'core layer', 'core z', 'codebook e', 'decoder plan proj']
for L, nm in [(ntp, 'NTP'), (ncp, 'NCP'), (vq, 'VQ')]:
    gg = gr(L, P); print(f"grad from {nm}: " + ", ".join(f"{n}={'0' if x is None or x.abs().sum() == 0 else 'yes'}" for n, x in zip(names, gg)))
