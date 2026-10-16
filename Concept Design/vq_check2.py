import torch
torch.manual_seed(0); ds, N = 8, 128
cent = torch.randn(64, ds) * 0.6 + torch.tensor([1.5] + [0.0] * (ds - 1)); scale = torch.linspace(0.2, 1.0, ds)
def data(n, seed=None):
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    return cent[torch.randint(0, 64, (n,), generator=g)] + torch.randn(n, ds, generator=g) * 0.25 * scale
def kmeanspp(x, k):
    c = [x[torch.randint(len(x), (1,))]]
    for _ in range(k - 1):
        d = torch.cdist(x, torch.cat(c)).min(1).values.pow(2); c.append(x[torch.multinomial(d / d.sum(), 1)])
    return torch.cat(c)
def run(kind, steps=3000, restart=0):
    torch.manual_seed(1); B = torch.randn(N, ds)
    if kind == 'plain': P = [torch.nn.Parameter(B.clone())]; codes = lambda: P[0]
    elif kind == 'kmpp': P = [torch.nn.Parameter(kmeanspp(data(8192, 7), N))]; codes = lambda: P[0]
    elif kind == 'simvq': P = [torch.nn.Parameter(torch.eye(ds))]; codes = lambda: B @ P[0]
    elif kind == 'simvq+b': P = [torch.nn.Parameter(torch.eye(ds)), torch.nn.Parameter(torch.zeros(ds))]; codes = lambda: B @ P[0] + P[1]
    opt = torch.optim.Adam(P, lr=1e-3); hits = torch.zeros(N)
    for it in range(steps):
        c = data(1024); e = codes(); n = torch.cdist(c, e.detach()).argmin(1); hits += torch.bincount(n, minlength=N)
        loss = (c - e[n]).pow(2).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if restart and (it + 1) % restart == 0:
            dead = hits == 0
            if dead.any() and kind in ('plain', 'kmpp'):
                with torch.no_grad(): P[0][dead] = c[torch.randint(0, len(c), (int(dead.sum()),))]
            hits.zero_()
    with torch.no_grad():
        c = data(32768, 3); e = codes(); n = torch.cdist(c, e).argmin(1)
        cnt = torch.bincount(n, minlength=N).float(); p = cnt / cnt.sum()
        ve = 1 - (c - e[n]).pow(2).sum(1).mean() / (c - c.mean(0)).pow(2).sum(1).mean()
    return (cnt > 0).sum().item(), (-(p[p > 0] * p[p > 0].log()).sum()).exp().item(), ve.item()
for k, r in [('plain', 0), ('simvq', 0), ('simvq+b', 0), ('plain', 100), ('kmpp', 0), ('kmpp', 100)]:
    used, perp, ve = run(k, restart=r)
    print(f"{k:8s} restart={r:3d}: codes used {used:3d}/{N} | perplexity {perp:5.1f} | variance explained {ve:.3f}")
