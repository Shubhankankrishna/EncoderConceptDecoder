"""H1 test: GradCuit-style test-time latent steering in the concept stream of a trained v2 core (frozen weights).

Latents = offsets dz on the concept-stream inputs X at the 2 positions just before the query (SEP, Q), i.e. between the
'prompt' (facts) and the 'continuation' (the chain answers). Update exactly as GradCuit: loss = -R * sum_t log p(pred_t)
over the model's own greedy continuation, R = -1 if the verifier rejects the final answer, Adam, <= 10 updates, stop when correct.
Baselines at the same verifier budget (11 calls): random-direction steering, sampling (BoN), rank enumeration; and extra passes.
"""
import sys, json, math, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from exp_v2 import Core, chain_batch, V
from exp_concept import kmeans_pq
torch.set_num_threads(1); torch.manual_seed(0)

d = 64
X = F.normalize(torch.randn(V, d, generator=torch.Generator().manual_seed(99)), dim=-1) * math.sqrt(d)
codes, assign = kmeans_pq(X, 8, 32)
m = Core('L1S', X, codes, assign, d, 176); m.load_state_dict(torch.load(sys.argv[1] if len(sys.argv) > 1 else 'v2_L1S_0.pt')); m.eval()
for p in m.parameters(): p.requires_grad_(False)
SLOTS = [24, 25]


def fwd(ids, R, dz=None):
    Xs = m.X[ids].clone()
    if dz is not None: Xs[:, SLOTS] = Xs[:, SLOTS] + dz
    St = torch.zeros_like(Xs); doc = torch.zeros_like(ids)
    for _ in range(R):
        for L in m.layers: St = L(Xs, St, doc)
    return m.concept_logp(m.head(St))                                         # B, T, V


def regime(h, R, n=400, budget=10):
    g = torch.Generator().manual_seed(1000 + h * 10 + R)
    ids, tgt, kk = chain_batch(n, g, hops=(h,), nq=1)
    pos = list(range(27, 27 + h)); last = pos[-1]; ans = tgt[:, last]
    with torch.no_grad(): lp0 = fwd(ids, R)
    wrong = lp0[:, last].argmax(-1) != ans
    ids, tgt, ans, lp0 = ids[wrong], tgt[wrong], ans[wrong], lp0[wrong]; B = len(ids)
    out = dict(h=h, R=R, n=n, base_acc=1 - B / n, n_fail=B)
    # enumeration by rank, and sampling
    rank = (lp0[:, last] > lp0[:, last].gather(-1, ans[:, None])).sum(-1) + 1
    out['enumerate_solved'] = (rank <= budget + 1).float().mean().item(); out['enumerate_calls'] = rank.clamp(max=budget + 1).float().mean().item()
    gs = torch.Generator().manual_seed(5); pr = lp0[:, last].exp(); hit = torch.zeros(B, dtype=torch.bool)
    for _ in range(budget): hit |= torch.multinomial(pr, 1, generator=gs)[:, 0] == ans
    out['sample_solved'] = hit.float().mean().item()
    with torch.no_grad(): out['extra_passes_solved'] = (fwd(ids, R + 4)[:, last].argmax(-1) == ans).float().mean().item()
    inter0 = (lp0[:, pos[:-1]].argmax(-1) == tgt[:, pos[:-1]]).all(-1) if h > 1 else torch.ones(B, dtype=torch.bool)
    out['intermediate_ok_before'] = inter0.float().mean().item()

    for mode in ('gradcuit', 'random'):
        for lr in (0.05, 0.2, 0.5):
            torch.manual_seed(1); dz = torch.zeros(B, len(SLOTS), d, requires_grad=True)
            opt = torch.optim.Adam([dz], lr=lr); solved = torch.zeros(B, dtype=torch.bool); calls = torch.full((B,), budget + 1.0)
            for step in range(budget):
                active = ~solved
                if not active.any(): break
                keep = dz.detach().clone()
                if mode == 'gradcuit':
                    lp = fwd(ids, R, dz); pred = lp[:, pos].argmax(-1)
                    loss = (lp[:, pos].gather(-1, pred[..., None]).squeeze(-1).sum(-1) * active).sum()   # -R*sum log p, R=-1
                    opt.zero_grad(); loss.backward(); opt.step()
                else:
                    with torch.no_grad():
                        e = torch.randn_like(dz); dz += lr * e / e.flatten(1).norm(dim=1)[:, None, None]
                with torch.no_grad():
                    dz[solved] = keep[solved]                                                             # freeze finished instances
                    lp = fwd(ids, R, dz); ok = lp[:, last].argmax(-1) == ans
                    new = ok & ~solved; calls[new] = step + 2; solved |= ok
            with torch.no_grad():
                lp = fwd(ids, R, dz)
                inter = (lp[:, pos[:-1]].argmax(-1) == tgt[:, pos[:-1]]).all(-1) if h > 1 else torch.ones(B, dtype=torch.bool)
            out[f'{mode}_lr{lr}'] = dict(solved=round(solved.float().mean().item(), 3), calls=round(calls[solved].mean().item(), 2) if solved.any() else None,
                                         intermediates_ok_when_solved=round(inter[solved].float().mean().item(), 3) if solved.any() else None,
                                         dz_norm=round(dz.detach().flatten(1).norm(dim=1).mean().item(), 2))
    return out


if __name__ == "__main__":
    res = [regime(6, 6), regime(5, 4), regime(4, 3)]
    for r in res: print(json.dumps(r))
    json.dump(res, open('gradcuit_results.json', 'w'))
