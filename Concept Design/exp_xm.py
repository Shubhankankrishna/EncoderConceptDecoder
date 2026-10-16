"""H2 test: Explorative-Modeling latent z in the v2 concept core (GRAM-style hypotheses without a VAE).

Task: branching in-context graph. An 8-cycle f plus 2 nodes with a second successor g, so "x R" has 1 or 2 valid answers
and "x R R" up to 4 valid chains. Each training query carries ONE sampled valid path (data realise one mode).
The two R positions are predicted without seeing each other's answers -> joint prediction of a multi-part future,
the same situation as a concept plan describing 4 future tokens at once.

Arms:  K1  v2 core, no latent (deterministic)        XM4  v2 core + read-only latent stream Z = z_k (k = 1..4),
                                                            Forward XM: per query, train only the best candidate.
Both: one shared layer looped, input re-read every pass, S0 = 0, R ~ U{1..3}.
"""
import sys, json, math, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '.')
from encoder_v1 import GRRead, GRWrite, Attn, SwiGLU, ZCRMSNorm
from exp_concept import kmeans_pq
torch.set_num_threads(1)
ENT, SEP, RM, QM, PAD = 128, 128, 129, 130, 131; V = 132; QLEN = 4


def branch_batch(B, g, nq=2, force_branch=False):
    n = 8
    ents = torch.stack([torch.randperm(ENT, generator=g)[:n] for _ in range(B)])
    cyc = torch.gather(ents, 1, torch.stack([torch.randperm(n, generator=g) for _ in range(B)]))
    succ = [[None] * n for _ in range(B)]; seqs, tgts, info = [], [], []
    for b in range(B):
        c = cyc[b].tolist(); S = {c[i]: [c[(i + 1) % n]] for i in range(n)}
        for a in torch.randperm(n, generator=g)[:2].tolist():                 # 2 branching nodes
            opts = [e for e in c if e != c[a] and e not in S[c[a]]]
            S[c[a]].append(opts[torch.randint(len(opts), (1,), generator=g).item()])
        facts = [(a, s) for a in S for s in S[a]]; order = torch.randperm(len(facts), generator=g).tolist()
        seq = [x for i in order for x in (facts[i][0], RM, facts[i][1])] + [SEP]; tgt = [-1] * len(seq); qi = []
        for q in range(nq):
            if force_branch: x = [a for a in S if len(S[a]) == 2][q % 2]
            else: x = c[torch.randint(n, (1,), generator=g).item()]
            h1 = S[x][torch.randint(len(S[x]), (1,), generator=g).item()]; h2 = S[h1][torch.randint(len(S[h1]), (1,), generator=g).item()]
            base = len(seq); seq += [QM, x, RM, RM]; tgt += [-1, -1, h1, h2]; qi.append((base, x))
        seqs.append(seq); tgts.append(tgt); info.append((S, qi))
    return torch.tensor(seqs), torch.tensor(tgts), info


class Layer3(nn.Module):
    """reads {X, S, Z} (Z optional), writes S."""
    def __init__(s, d, f, n_read, H=4, kvh=2):
        super().__init__(); s.r1, s.w1, s.r2, s.w2 = GRRead(d, n_read), GRWrite(d, 1), GRRead(d, n_read), GRWrite(d, 1)
        s.attn, s.ffn = Attn(d, H, kvh, d // H, window=None), SwiGLU(d, f)
    def forward(s, streams, S, doc):
        x = s.r1(streams + [S]); S = S + s.w1(x) * s.attn(x, doc)
        x = s.r2(streams + [S]); return S + s.w2(x) * s.ffn(x)


class CoreZ(nn.Module):
    def __init__(s, K, X, codes, assign, d=64, f=176):
        super().__init__(); s.K = K
        s.layer = Layer3(d, f, 3 if K > 1 else 2)
        if K > 1: s.z = nn.Parameter(torch.randn(K, d))                           # K learned latent embeddings (XMDLM-style)
        s.register_buffer('X', X); s.register_buffer('codes', codes); s.register_buffer('assign', assign)
        s.S, s.N, s.ds = codes.shape; s.out_norm = ZCRMSNorm(d); s.Wq = nn.Linear(d, d, bias=False)
        s.log_tau = nn.Parameter(torch.full((s.S,), math.log(2 * s.ds)))
    def head(s, R):
        q = s.Wq(s.out_norm(R)).unflatten(-1, (s.S, s.ds))
        return -(q.unsqueeze(-2) - s.codes).pow(2).sum(-1) / s.log_tau.exp()[:, None]
    def forward(s, ids, R, zvec=None):
        doc = torch.zeros_like(ids); Xs = s.X[ids]; St = torch.zeros_like(Xs)
        streams = [Xs] if zvec is None else [Xs, zvec[:, None, :].expand_as(Xs)]
        for _ in range(R): St = s.layer(streams, St, doc)
        return s.head(St)
    def concept_logp(s, logits):
        lp = logits.log_softmax(-1); return lp[..., torch.arange(s.S)[None, :], s.assign].sum(-1).log_softmax(-1)


def seg_ce(m, logits, tgt):                      # per-position NCP CE (sum over segments / S); 0 where no target
    sel = tgt >= 0; t = torch.where(sel, tgt, torch.zeros_like(tgt))
    lp = logits.log_softmax(-1).gather(-1, m.assign[t][..., None]).squeeze(-1).mean(-1)
    return -(lp * sel)


@torch.no_grad()
def evaluate(m, R=3):
    m.eval(); g = torch.Generator().manual_seed(777)
    ids, tgt, info = branch_batch(256, g, nq=2, force_branch=True)                # every query starts at a branching node
    Z = [None] if m.K == 1 else [m.z[k].expand(len(ids), -1) for k in range(m.K)]
    preds = [m.concept_logp(m(ids, R, z)) for z in Z]                            # list of (B, T, V) log-probs
    coh = []; cov = []; samp_coh = []
    for b, (S, qi) in enumerate(info):
        for (base, x) in qi:
            valid = {(h1, h2) for h1 in S[x] for h2 in S[h1]}
            chains = [(p[b, base + 2].argmax().item(), p[b, base + 3].argmax().item()) for p in preds]
            coh += [c in valid for c in chains]; cov.append(len(set(chains) & valid) / len(valid))
            if m.K == 1:                                                           # independent sampling of the two positions
                p1, p2 = preds[0][b, base + 2].exp(), preds[0][b, base + 3].exp()
                s = [(torch.multinomial(p1, 1).item(), torch.multinomial(p2, 1).item()) for _ in range(4)]
                samp_coh += [c in valid for c in s]; cov[-1] = max(cov[-1], len(set(s) & valid) / len(valid))
    # unambiguous check: queries from non-branching nodes whose successor is also non-branching
    ids2, tgt2, info2 = branch_batch(256, torch.Generator().manual_seed(778), nq=2)
    p2 = [m.concept_logp(m(ids2, R, z)) for z in Z]; ok = []
    for b, (S, qi) in enumerate(info2):
        for (base, x) in qi:
            if len(S[x]) == 1 and len(S[S[x][0]]) == 1:
                ok += [all(p[b, base + 2].argmax().item() == S[x][0] and p[b, base + 3].argmax().item() == S[S[x][0]][0] for p in p2)]
    m.train()
    out = dict(chain_coherent=sum(coh) / len(coh), coverage=sum(cov) / len(cov), unambiguous_acc=sum(ok) / len(ok))
    if samp_coh: out['sampled_coherent'] = sum(samp_coh) / len(samp_coh)
    return out


def run(K, seed, steps, B=32, lr=1e-3, log=None):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed); d = 64
    X = F.normalize(torch.randn(V, d, generator=torch.Generator().manual_seed(99)), dim=-1) * math.sqrt(d)
    codes, assign = kmeans_pq(X, 8, 32); m = CoreZ(K, X, codes, assign)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    wins = torch.zeros(K); hist = []; t0 = time.time()
    for it in range(steps):
        for gp in opt.param_groups: gp['lr'] = lr * min(1, (it + 1) / 200)
        ids, tgt, _ = branch_batch(B, g); R = int(torch.randint(1, 4, (1,), generator=g))
        if K == 1:
            loss = seg_ce(m, m(ids, R), tgt).sum() / (tgt >= 0).sum()
        else:
            ce = torch.stack([seg_ce(m, m(ids, R, m.z[k].expand(B, -1)), tgt) for k in range(K)])     # K, B, T
            T0 = ids.shape[1] - 2 * QLEN
            q = torch.stack([ce[:, :, T0 + i * QLEN: T0 + (i + 1) * QLEN].sum(-1) for i in range(2)], -1)   # K, B, nq
            best = q.min(0); wins += torch.bincount(best.indices.flatten(), minlength=K)
            loss = best.values.sum() / (tgt >= 0).sum()                                                  # Forward XM: min over k
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (it + 1) % 500 == 0 or it == steps - 1:
            e = evaluate(m); e.update(step=it + 1, loss=round(loss.item(), 4))
            if K > 1: e['win_share'] = [round(w, 3) for w in (wins / wins.sum()).tolist()]; e['z_cos'] = F.cosine_similarity(m.z[:, None], m.z[None], dim=-1).mean().item()
            hist.append(e); wins.zero_()
            if log: print(K, seed, e, f"{time.time()-t0:.0f}s", file=log, flush=True)
    torch.save(m.state_dict(), f"xm_K{K}_{seed}.pt")
    return dict(K=K, seed=seed, hist=hist, secs=time.time() - t0)


if __name__ == "__main__":
    K, seed, steps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    json.dump(run(K, seed, steps, log=sys.stderr), open(f"xm_K{K}_{seed}.json", "w"))
