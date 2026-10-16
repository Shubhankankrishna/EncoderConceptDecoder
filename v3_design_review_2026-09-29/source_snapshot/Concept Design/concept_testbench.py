"""Concept-module test bench (isolated from encoder/decoder).

Inputs are concept vectors c_hat (fixed random points on the sphere of radius sqrt(d), standing in for a frozen encoder).
Output: NCP over a product-quantised vocabulary (k-means codebook of the concept vectors), CE on the next concept's codes.
Block = encoder_v1 FULL layer anatomy (gated attention, NoPE + per-head temperature, QKV conv) in a 1-stream gated residual.

Arms (layer applications in brackets):
  N4        4 untied layers                                   [4]   NCP-style stack
  U6        6 untied layers                                   [6]   equal FLOPs to the loops
  SUB       P + (B1 B2) x2 + C                                [6]   loop a subset (v1)
  SUB_D     SUB + DiscoLoop channel between passes            [6]   v1 as proposed
  FULL      (L1 L2 L3) x2, no prelude/coda                    [6]   loop the whole module, equal FLOPs
  FULL_D    FULL + DiscoLoop channel                          [6]
Tasks:
  ic  : in-context k-hop pointer chasing (facts [a R b] in random order, queries [Q_h x] -> f^h(x), h in 1..4)
  iw  : in-weight two-hop (DiscoLoop's two-graph task): atomic [a r b], two-hop [a r1 r2 c]; ID/OOD splits
"""
import sys, json, math, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '.')
from encoder_v1 import GRRead, GRWrite, Attn, SwiGLU, ZCRMSNorm, rms
torch.set_num_threads(1)


class GRLayer(nn.Module):
    def __init__(s, d, f, H, kvh, dh):
        super().__init__()
        s.r1, s.w1, s.r2, s.w2 = GRRead(d, 1), GRWrite(d, 1), GRRead(d, 1), GRWrite(d, 1)
        s.attn, s.ffn = Attn(d, H, kvh, dh, window=None), SwiGLU(d, f)
    def forward(s, R, doc):
        x = s.r1([R]); R = R + s.w1(x) * s.attn(x, doc)
        x = s.r2([R]); return R + s.w2(x) * s.ffn(x)


def kmeans_pq(X, S, N, iters=40, seed=0):
    g = torch.Generator().manual_seed(seed); V, d = X.shape; ds = d // S
    xs = X.view(V, S, ds); codes = torch.zeros(S, N, ds)
    for s_ in range(S):
        x = xs[:, s_]; pick = [x[torch.randint(V, (1,), generator=g)]]
        for _ in range(N - 1):
            dd = torch.cdist(x, torch.cat(pick)).min(1).values.pow(2); pick.append(x[torch.multinomial(dd / dd.sum(), 1, generator=g)])
        c = torch.cat(pick)
        for _ in range(iters):
            a = torch.cdist(x, c).argmin(1)
            for n in range(N):
                if (a == n).any(): c[n] = x[a == n].mean(0)
        codes[s_] = c
    assign = torch.stack([torch.cdist(xs[:, s_], codes[s_]).argmin(1) for s_ in range(S)], 1)   # V, S
    return codes, assign


class ConceptModule(nn.Module):
    def __init__(s, arm, X, codes, assign, d, f, H=4, kvh=2):
        super().__init__(); s.arm = arm; dh = d // H
        mk = lambda: GRLayer(d, f, H, kvh, dh)
        s.pre, s.loop, s.post, s.R = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), 1
        if arm == 'N4': s.loop = nn.ModuleList([mk() for _ in range(4)])
        elif arm == 'U6': s.loop = nn.ModuleList([mk() for _ in range(6)])
        elif arm.startswith('SUB'): s.pre, s.loop, s.post, s.R = nn.ModuleList([mk()]), nn.ModuleList([mk(), mk()]), nn.ModuleList([mk()]), 2
        elif arm.startswith('FULL'): s.loop, s.R = nn.ModuleList([mk() for _ in range(3)]), 2
        suf = arm.split('_')[1] if '_' in arm else ''
        s.disco = suf.startswith('D'); s.gated = s.disco and 'G' in suf
        if s.gated: s.gw = nn.Linear(d, 1); nn.init.zeros_(s.gw.weight); nn.init.zeros_(s.gw.bias)   # DiscoLoop eq.6, alpha0 = 0.5
        s.register_buffer('X', X); s.register_buffer('codes', codes); s.register_buffer('assign', assign)
        s.S, s.N, s.ds = codes.shape
        s.out_norm = ZCRMSNorm(d); s.Wq = nn.Linear(d, d, bias=False)
        s.log_tau = nn.Parameter(torch.full((s.S,), math.log(2 * s.ds)))
    def head(s, R):
        q = s.Wq(s.out_norm(R)).unflatten(-1, (s.S, s.ds))
        return -(q.unsqueeze(-2) - s.codes).pow(2).sum(-1) / s.log_tau.exp()[:, None]      # ..., S, N
    def phi(s, R):
        pi = s.head(R).softmax(-1); return torch.einsum('...sn,snd->...sd', pi, s.codes).flatten(-2)
    def forward(s, ids, R_run=None):
        R_run = R_run or s.R; doc = torch.zeros_like(ids); R = s.X[ids]; mids = []
        for L in s.pre: R = L(R, doc)
        for r in range(R_run):
            for L in s.loop: R = L(R, doc)
            mids.append(R)
            if s.disco and r < R_run - 1:
                ph = s.phi(R); a = torch.sigmoid(s.gw(ph)) if s.gated else 1.0         # fixed alpha = 1, or learned token gate
                R = R + a * rms(ph)
        for L in s.post: R = L(R, doc)
        return s.head(R), mids
    def concept_logp(s, logits):                                                        # joint score over the vocabulary
        lp = logits.log_softmax(-1)                                                     # ..., S, N
        sc = lp[..., torch.arange(s.S)[None, :], s.assign].sum(-1)                      # ..., V
        return sc.log_softmax(-1)


# ---------------------------------------------------------------- tasks
ENT, SEP, REL, Q0 = 128, 128, 129, 130                       # ic: entity ids 0..127, SEP, relation marker, Q1..Q4 = 130..133
def ic_batch(B, g, hops=(1, 2), nq=6):
    n = 6; ents = torch.stack([torch.randperm(ENT, generator=g)[:n] for _ in range(B)])      # B, n
    order = torch.stack([torch.randperm(n, generator=g) for _ in range(B)])
    cyc = torch.gather(ents, 1, order)                                                     # the 8-cycle
    nxt = torch.roll(cyc, -1, 1)
    fperm = torch.stack([torch.randperm(n, generator=g) for _ in range(B)])
    a, b = torch.gather(cyc, 1, fperm), torch.gather(nxt, 1, fperm)
    facts = torch.stack([a, torch.full_like(a, REL), b], 2).flatten(1)                     # B, 18
    seq = [facts, torch.full((B, 1), SEP)]; tgt_pos, tgt, hop, bridge = [], [], [], []
    pos = 19
    for qi in range(nq):
        h = torch.tensor(hops)[torch.randint(len(hops), (B,), generator=g)]
        k = torch.randint(n, (B,), generator=g)
        ans = torch.gather(cyc, 1, ((k + h) % n)[:, None])[:, 0]; br = torch.gather(cyc, 1, ((k + 1) % n)[:, None])[:, 0]
        x = torch.gather(cyc, 1, k[:, None])[:, 0]
        seq.append(torch.stack([Q0 + h - 1, x, ans], 1)); tgt_pos.append(pos + 1); tgt.append(ans); hop.append(h); bridge.append(br); pos += 3
    return torch.cat(seq, 1), torch.tensor(tgt_pos), torch.stack(tgt, 1), torch.stack(hop, 1), torch.stack(bridge, 1)


def iw_build(g, NE=60, NR=8, deg=5, id_frac=0.7):
    """Two entity-disjoint graphs sharing relations. ids: entities 0..2NE-1, relations 2NE..2NE+NR-1."""
    graphs = []
    for gi in range(2):
        base = gi * NE; edges = {}
        for a in range(NE):
            rels = torch.randperm(NR, generator=g)[:deg]
            for r in rels.tolist(): edges[(base + a, r)] = base + torch.randint(NE, (1,), generator=g).item()
        graphs.append(edges)
    def twohop(edges):
        out = []
        for (a, r1), b in edges.items():
            for r2 in range(NR):
                if (b, r2) in edges: out.append((a, r1, r2, edges[(b, r2)], b))
        return out
    atom = [(a, r, b) for e in graphs for (a, r), b in e.items()]
    thA, thB = twohop(graphs[0]), twohop(graphs[1])
    perm = torch.randperm(len(thA), generator=g).tolist(); cut = int(id_frac * len(thA))
    train2 = [thA[i] for i in perm[:cut]]; testid = [thA[i] for i in perm[cut:]]
    ood = [thB[i] for i in torch.randperm(len(thB), generator=g)[:1000].tolist()]
    R0 = 2 * NE
    def enc_atom(lst): return torch.tensor([[a, R0 + r, b, b] for a, r, b in lst])       # predict b at pos 1 (padded to len 4)
    def enc_two(lst): return torch.tensor([[a, R0 + r1, R0 + r2, c] for a, r1, r2, c, _ in lst]), torch.tensor([bb for *_, bb in lst])
    return dict(atom=enc_atom(atom), train2=enc_two(train2), testid=enc_two(testid), ood=enc_two(ood), V=2 * NE + NR)


def run(task, arm, seed, steps, d, f, lr=1e-3, wd=0.1, B=128, eval_every=500, log=None):
    if task == 'ic': eval_every = 250
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    if task == 'ic':
        V = 134; S, N = 8, 32
    else:
        data = iw_build(torch.Generator().manual_seed(1234)); V = data['V']; S, N = d // 8, 32
    X = F.normalize(torch.randn(V, d, generator=torch.Generator().manual_seed(99)), dim=-1) * math.sqrt(d)
    codes, assign = kmeans_pq(X, S, N)
    if 'X' in (arm.split('_')[1] if '_' in arm else ''):                                 # exact vocabulary head (tied, like DiscoLoop's LM head)
        codes, assign = X.clone().view(1, V, d), torch.arange(V)[:, None]
    uniq = len(set(map(tuple, assign.tolist())))
    m = ConceptModule(arm, X, codes, assign, d, f)
    nowd = [p for n_, p in m.named_parameters() if 'log_tau' in n_ or n_.endswith('.w')]
    yes = [p for n_, p in m.named_parameters() if not ('log_tau' in n_ or n_.endswith('.w'))]
    opt = torch.optim.AdamW([dict(params=yes, weight_decay=wd), dict(params=nowd, weight_decay=0)], lr=lr, betas=(0.9, 0.95))
    sched = lambda it: min(1, (it + 1) / 200)
    hist = []; t0 = time.time()

    def ce_codes(logits, tgt):                                      # per-segment CE on the target concept's codes
        return F.cross_entropy(logits.flatten(0, -2), assign[tgt].flatten())

    if task == 'iw':
        atom, (tr2, trb) = data['atom'], data['train2']
        pool = torch.cat([torch.cat([atom, torch.ones(len(atom), 1, dtype=torch.long)], 1),
                          torch.cat([tr2, torch.full((len(tr2), 1), 2)], 1)])          # last col: target position
    for it in range(steps):
        for gp in opt.param_groups: gp['lr'] = lr * sched(it)
        if task == 'ic':
            ids, tp, tgt, hop, br = ic_batch(64, g)
            logits, _ = m(ids); lg = logits[:, tp]; loss = ce_codes(lg, tgt)
        else:
            idx = torch.randint(len(pool), (B,), generator=g); rows = pool[idx]
            ids = rows[:, :4]; tp = rows[:, 4]
            tgt = torch.where(tp == 1, ids[:, 2], ids[:, 3])
            logits, _ = m(ids); lg = logits[torch.arange(B), tp]; loss = ce_codes(lg, tgt)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (it + 1) % eval_every == 0 or it == steps - 1:
            hist.append(dict(step=it + 1, loss=loss.item(), **evaluate(m, task, data if task == 'iw' else None)))
            if log: print(arm, seed, hist[-1], f"{time.time()-t0:.0f}s", file=log, flush=True)
            if task == 'ic' and hist[-1]['acc12'] >= 0.98 and len(hist) >= 2 and hist[-2]['acc12'] >= 0.98: break   # learned: stop early
    final = evaluate(m, task, data if task == 'iw' else None, diag=True)
    torch.save(m.state_dict(), f"model_{task}_{arm}_{seed}.pt")
    return dict(task=task, arm=arm, seed=seed, d=d, unique_codes=uniq, V=V, hist=hist, final=final,
                params=sum(p.numel() for p in m.parameters()), secs=time.time() - t0)


@torch.no_grad()
def evaluate(m, task, data=None, diag=False):
    m.eval(); out = {}
    def acc_of(logits, tgt): return (m.concept_logp(logits).argmax(-1) == tgt).float()
    def stage1(mids, pos_sel, tgt, br):                          # logit-lens on the pass-1 state (DiscoLoop's diagnostic)
        R1 = pos_sel(mids[0]); lp = m.concept_logp(m.head(R1))
        cosb = F.cosine_similarity(R1, m.X[br], dim=-1)
        return dict(s1_p_bridge=lp.gather(-1, br[..., None]).exp().mean().item(), s1_acc_answer=(lp.argmax(-1) == tgt).float().mean().item(),
                    s1_cos_bridge=cosb.mean().item())
    if task == 'ic':
        g = torch.Generator().manual_seed(777); ids, tp, tgt, hop, br = ic_batch(512, g, hops=(1, 2, 3))
        logits, mids = m(ids); a = acc_of(logits[:, tp], tgt)
        for h in (1, 2, 3): out[f'h{h}'] = a[hop == h].mean().item()
        out['acc12'] = a[hop <= 2].mean().item()
        if diag and m.R > 1:
            sel2 = hop == 2
            d2 = stage1(mids, lambda R: R[:, tp][sel2], tgt[sel2], br[sel2]); out.update({k + '_h2': v for k, v in d2.items()})
            for Rr in (1, 3):
                lg, _ = m(ids, R_run=Rr); ar = acc_of(lg[:, tp], tgt)
                for h in (1, 2, 3): out[f'h{h}_R{Rr}'] = ar[hop == h].mean().item()
    else:
        for split in ('testid', 'ood'):
            ids, br = data[split]; logits, mids = m(ids); tgt = ids[:, 3]
            out[split] = acc_of(logits[:, 2], tgt).mean().item()
            if diag and m.R > 1:
                d2 = stage1(mids, lambda R: R[:, 1], tgt, br); out.update({f'{k}_{split}': v for k, v in d2.items()})   # bridge lives at r1's position (DiscoLoop)
        ids = data['atom']; logits, _ = m(ids); out['atom'] = acc_of(logits[:, 1], ids[:, 2]).mean().item()
        ids, _ = data['train2']; logits, mids = m(ids); out['train2'] = acc_of(logits[:, 2], ids[:, 3]).mean().item()
        if m.R > 1:
            R1 = mids[0][:, 2]; out['s1_train2'] = (m.concept_logp(m.head(R1)).argmax(-1) == ids[:, 3]).float().mean().item()
    m.train(); return out


if __name__ == "__main__":
    task, arm, seed, steps, d, f = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    wd = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
    res = run(task, arm, seed, steps, d, f, wd=wd, log=sys.stderr)
    json.dump(res, open(f"res_{task}_{arm}_{seed}.json", "w"))
    print(json.dumps(res['final']))
