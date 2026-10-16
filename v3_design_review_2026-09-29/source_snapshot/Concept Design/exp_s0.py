"""Review A/B: does a learned initial state S0 = s0 (position-independent, carries no input information) fix the
pass-1 read-scale mismatch (S0 = 0 => rms(S) = 0 => pass-1 read is half-scale), and does it change hop extrapolation?
Same task, sizes and schedule as exp_v2.py. Arms: L1S (baseline, S0 = 0) and L1S0 (S0 = learned s0, init 0.02*randn, no WD)."""
import sys, json, math, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '..'); sys.path.insert(0, '.')
import exp_v2
from exp_v2 import Core, chain_batch, evaluate
from exp_concept import kmeans_pq

class CoreS0(Core):
    def __init__(s, *a, **k):
        big = k.pop('big', False); super().__init__(*a, **k); s.s0 = nn.Parameter((1.0 if big else 0.02) * torch.randn(s.X.shape[1]))
    def forward(s, ids, R=None):
        doc = torch.zeros_like(ids); Xs = s.X[ids]; St = s.s0.expand_as(Xs)
        for _ in range(R):
            for L in s.layers: St = L(Xs, St, doc)
        return s.head(St)

def run(arm, seed, steps=6000, d=64, f=176, lr=1e-3, wd=0.1, B=64):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    X = F.normalize(torch.randn(exp_v2.V, d, generator=torch.Generator().manual_seed(99)), dim=-1) * math.sqrt(d)
    codes, assign = kmeans_pq(X, 8, 32)
    m = CoreS0('L1S', X, codes, assign, d, f, big=(arm == 'L1S0big')) if arm.startswith('L1S0') else Core('L1S', X, codes, assign, d, f)
    nowd = [p for n_, p in m.named_parameters() if 'log_tau' in n_ or n_.endswith('.w') or n_ == 's0']
    yes = [p for n_, p in m.named_parameters() if all(p is not q for q in nowd)]
    opt = torch.optim.AdamW([dict(params=yes, weight_decay=wd), dict(params=nowd, weight_decay=0)], lr=lr, betas=(0.9, 0.95))
    hist = []; t0 = time.time()
    for it in range(steps):
        for gp in opt.param_groups: gp['lr'] = lr * min(1, (it + 1) / 200)
        ids, tgt, _ = chain_batch(B, g); R = int(torch.randint(1, 5, (1,), generator=g))
        logits = m(ids, R); sel = tgt >= 0
        loss = F.cross_entropy(logits[sel].flatten(0, -2), assign[tgt[sel]].flatten())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (it + 1) % 250 == 0 or it == steps - 1:
            hist.append(dict(step=it + 1, loss=round(loss.item(), 4), **evaluate(m, [1, 2, 3, 4, 6, 8, 12])))
            print(arm, seed, hist[-1], f"{time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    torch.save(m.state_dict(), f"{arm}_{seed}.pt")
    json.dump(dict(arm=arm, seed=seed, hist=hist, secs=time.time() - t0), open(f"{arm}_{seed}.json", "w"))

if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
