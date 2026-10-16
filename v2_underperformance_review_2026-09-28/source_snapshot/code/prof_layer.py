import time, torch, torch.nn.functional as F
from model import build_model
from concept_core_d6 import moe_dispatch
dev = torch.device("mps"); torch.manual_seed(0)
m, _, _ = build_model(); m = m.to(dev).train(); core = m.core; L = core.layers[0]
B, M, d = 2, 256, 256
X = torch.randn(B, M, d, device=dev, requires_grad=True); S = torch.randn(B, M, d, device=dev, requires_grad=True)
cdoc = torch.zeros(B, M, dtype=torch.long, device=dev); mask = core.mask(cdoc)
def t(name, fn, n=20):
    for _ in range(3): y = fn()
    torch.mps.synchronize(); t0 = time.time()
    for _ in range(n): y = fn()
    torch.mps.synchronize(); print(f"{name:28s} {1000*(time.time()-t0)/n:6.2f} ms")
    return y
x = L.read_attn([X, S])
t("GR read (2 streams)", lambda: L.read_attn([X, S]))
t("attention", lambda: L.attn(x, mask))
t("write scale", lambda: L.write_attn(x))
xf = x.reshape(-1, d)
t("router", lambda: L.router(xf))
lg = L.router(xf)
t("moe dispatch (cf 1.25)", lambda: moe_dispatch(xf, lg, core.pool, 2, 1.25))
t("moe dense (all experts)", lambda: moe_dispatch(xf, lg, core.pool, 2, None, dense=True))
t("full layer fwd", lambda: L(X, S, mask))
def fb():
    y, aux = L(X, S, mask); (y.sum() + aux["lb"]).backward()
t("full layer fwd+bwd", fb)
u = core.readout(S)
t("ncp head fwd", lambda: core.head(u))
