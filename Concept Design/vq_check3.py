"""Drifting targets (the encoder keeps training): cluster centres rotate + translate over 3000 steps.
Adam-trained codes vs EMA k-means codes (decay 0.99), both with k-means++ init and dead-code restart every 100 steps."""
import torch, math
torch.manual_seed(0); ds, N, steps = 8, 128, 3000
cent0 = torch.randn(64, ds) * 0.6; scale = torch.linspace(0.2, 1.0, ds)
Q, _ = torch.linalg.qr(torch.randn(ds, ds))
def centres(it):
    a = 1.5 * it / steps; R = torch.eye(ds); R[0, 0] = R[1, 1] = math.cos(a); R[0, 1], R[1, 0] = -math.sin(a), math.sin(a)
    return (cent0 @ (Q @ R @ Q.T)) + torch.tensor([2.0 * it / steps] + [0.0] * (ds - 1))
def data(n, it): c = centres(it); return c[torch.randint(0, 64, (n,))] + torch.randn(n, ds) * 0.25 * scale
def kmeanspp(x, k):
    c = [x[torch.randint(len(x), (1,))]]
    for _ in range(k - 1):
        d = torch.cdist(x, torch.cat(c)).min(1).values.pow(2); c.append(x[torch.multinomial(d / d.sum(), 1)])
    return torch.cat(c)
def evaluate(e, it):
    c = data(32768, it); n = torch.cdist(c, e).argmin(1); cnt = torch.bincount(n, minlength=N).float(); p = cnt / cnt.sum()
    return (cnt > 0).sum().item(), 1 - ((c - e[n]).pow(2).sum(1).mean() / (c - c.mean(0)).pow(2).sum(1).mean()).item()
for kind in ('adam', 'ema'):
    torch.manual_seed(1); e = kmeanspp(data(8192, 0), N).clone()
    if kind == 'adam': P = torch.nn.Parameter(e); opt = torch.optim.Adam([P], lr=1e-3)
    else: sums, cnts = e.clone(), torch.ones(N)
    hits = torch.zeros(N); log = []
    for it in range(steps):
        c = data(1024, it); cur = P.detach() if kind == 'adam' else sums / cnts[:, None]
        n = torch.cdist(c, cur).argmin(1); b = torch.bincount(n, minlength=N).float(); hits += b
        if kind == 'adam':
            loss = (c - P[n]).pow(2).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        else:
            s = torch.zeros(N, ds).index_add_(0, n, c); sums = 0.99 * sums + 0.01 * s; cnts = 0.99 * cnts + 0.01 * b
        if (it + 1) % 100 == 0:
            dead = hits == 0; k = int(dead.sum())
            if k:
                new = c[torch.randint(0, len(c), (k,))]
                if kind == 'adam':
                    with torch.no_grad(): P[dead] = new
                else: sums[dead] = new * cnts[dead, None].clamp_min(1e-3); cnts[dead] = cnts[dead].clamp_min(1e-3)
            hits.zero_()
        if (it + 1) % 1000 == 0:
            cur = P.detach() if kind == 'adam' else sums / cnts[:, None]; log.append(evaluate(cur, it))
    print(f"{kind:4s}: " + " | ".join(f"step {1000*(i+1)}: used {u}/{N}, VE {v:.3f}" for i, (u, v) in enumerate(log)))
