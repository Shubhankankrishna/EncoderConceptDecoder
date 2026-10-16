"""Concept module v2 test: loop-only core with input injection and sampled depth, vs NCP's plain stack.

Task (compositional chains, in-context): facts [a R b] for an 8-cycle f in random order, SEP, then 2 queries
[Q x R R ... R PAD..] with h relation markers. At the k-th R the target is f^k(x) (dense, like NCP at every position).
Train chains h in {1,2,3}; test h up to 6, so positions k = 4..6 are unseen depths (hop-count extrapolation).

Arms:
  N4      : 4 untied GR layers on one stream (NCP stack, option A)
  U6      : 6 untied layers
  L1S     : ONE shared layer looped; state S (starts at 0), input X re-read every pass (GR reads {X,S}, writes S);
            passes R ~ Uniform{1..4} each training step  (v2)
  L1F     : same, fixed R = 4 in training (isolates depth sampling)
  L2S     : TWO shared layers per pass, R ~ Uniform{1..3}
"""
import sys, json, math, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '.')
from encoder_v1 import GRRead, GRWrite, Attn, SwiGLU, ZCRMSNorm, rms
from exp_concept import GRLayer, kmeans_pq
torch.set_num_threads(1)

ENT, SEP, RM, QM, PAD = 128, 128, 129, 130, 131
V = 132; HMAX = 6; QLEN = 2 + HMAX


def chain_batch(B, g, hops=(1, 2, 3), nq=2):
    n = 8
    ents = torch.stack([torch.randperm(ENT, generator=g)[:n] for _ in range(B)])
    cyc = torch.gather(ents, 1, torch.stack([torch.randperm(n, generator=g) for _ in range(B)]))
    nxt = torch.roll(cyc, -1, 1); fp = torch.stack([torch.randperm(n, generator=g) for _ in range(B)])
    a, b = torch.gather(cyc, 1, fp), torch.gather(nxt, 1, fp)
    facts = torch.stack([a, torch.full_like(a, RM), b], 2).flatten(1)                    # B, 24
    seq = [facts, torch.full((B, 1), SEP)]; T0 = 25
    tgt = torch.full((B, T0 + nq * QLEN), -1, dtype=torch.long); kk = torch.zeros_like(tgt)
    for q in range(nq):
        h = torch.tensor(hops)[torch.randint(len(hops), (B,), generator=g)]
        s0 = torch.randint(n, (B,), generator=g); x = torch.gather(cyc, 1, s0[:, None])[:, 0]
        blk = torch.full((B, QLEN), PAD); blk[:, 0] = QM; blk[:, 1] = x
        base = T0 + q * QLEN
        for k in range(1, HMAX + 1):
            on = k <= h
            blk[:, 1 + k] = torch.where(on, torch.full_like(x, RM), blk[:, 1 + k])
            fk = torch.gather(cyc, 1, ((s0 + k) % n)[:, None])[:, 0]
            tgt[:, base + 1 + k] = torch.where(on, fk, tgt[:, base + 1 + k]); kk[:, base + 1 + k] = torch.where(on, torch.full_like(x, k), kk[:, base + 1 + k])
        seq.append(blk)
    return torch.cat(seq, 1), tgt, kk


class LoopLayer(nn.Module):
    """GR layer that reads {X, S} and writes S only (X = input stream, never written)."""
    def __init__(s, d, f, H, kvh, dh):
        super().__init__()
        s.r1, s.w1, s.r2, s.w2 = GRRead(d, 2), GRWrite(d, 1), GRRead(d, 2), GRWrite(d, 1)
        s.attn, s.ffn = Attn(d, H, kvh, dh, window=None), SwiGLU(d, f)
    def forward(s, X, S, doc):
        x = s.r1([X, S]); S = S + s.w1(x) * s.attn(x, doc)
        x = s.r2([X, S]); return S + s.w2(x) * s.ffn(x)


class Core(nn.Module):
    def __init__(s, arm, X, codes, assign, d, f, H=4, kvh=2):
        super().__init__(); s.arm = arm; dh = d // H
        s.loop = arm.startswith('L')
        if s.loop:
            s.layers = nn.ModuleList([LoopLayer(d, f, H, kvh, dh) for _ in range(int(arm[1]))])
            s.sampled = arm.endswith('S'); s.Rmax = 4 if arm[1] == '1' else 3
        else:
            s.layers = nn.ModuleList([GRLayer(d, f, H, kvh, dh) for _ in range(int(arm[1]))])
        s.register_buffer('X', X); s.register_buffer('codes', codes); s.register_buffer('assign', assign)
        s.S, s.N, s.ds = codes.shape
        s.out_norm = ZCRMSNorm(d); s.Wq = nn.Linear(d, d, bias=False)
        s.log_tau = nn.Parameter(torch.full((s.S,), math.log(2 * s.ds)))
    def head(s, R):
        q = s.Wq(s.out_norm(R)).unflatten(-1, (s.S, s.ds))
        return -(q.unsqueeze(-2) - s.codes).pow(2).sum(-1) / s.log_tau.exp()[:, None]
    def forward(s, ids, R=None):
        doc = torch.zeros_like(ids); Xs = s.X[ids]
        if not s.loop:
            H_ = Xs
            for L in s.layers: H_ = L(H_, doc)
            return s.head(H_)
        St = torch.zeros_like(Xs)
        for _ in range(R):
            for L in s.layers: St = L(Xs, St, doc)
        return s.head(St)
    def concept_logp(s, logits):
        lp = logits.log_softmax(-1); return lp[..., torch.arange(s.S)[None, :], s.assign].sum(-1).log_softmax(-1)


@torch.no_grad()
def evaluate(m, Rs):
    m.eval(); g = torch.Generator().manual_seed(777)
    ids, tgt, kk = chain_batch(512, g, hops=(6,))                                         # every query is a 6-chain: positions k=1..6
    out = {}
    for R in (Rs if m.loop else [None]):
        pred = m.concept_logp(m(ids, R)).argmax(-1); ok = (pred == tgt)
        out[f'R{R}' if R else 'stack'] = [round(ok[kk == k].float().mean().item(), 3) for k in range(1, HMAX + 1)]
    m.train(); return out


def run(arm, seed, steps, d=64, f=176, lr=1e-3, wd=0.1, B=64, log=None):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    X = F.normalize(torch.randn(V, d, generator=torch.Generator().manual_seed(99)), dim=-1) * math.sqrt(d)
    codes, assign = kmeans_pq(X, 8, 32)
    m = Core(arm, X, codes, assign, d, f)
    nowd = [p for n_, p in m.named_parameters() if 'log_tau' in n_ or n_.endswith('.w')]
    yes = [p for n_, p in m.named_parameters() if not ('log_tau' in n_ or n_.endswith('.w'))]
    opt = torch.optim.AdamW([dict(params=yes, weight_decay=wd), dict(params=nowd, weight_decay=0)], lr=lr, betas=(0.9, 0.95))
    Rs = [1, 2, 3, 4, 6, 8] if arm.startswith('L1') else [1, 2, 3, 4]
    hist = []; t0 = time.time()
    for it in range(steps):
        for gp in opt.param_groups: gp['lr'] = lr * min(1, (it + 1) / 200)
        ids, tgt, _ = chain_batch(B, g)
        if m.loop: R = int(torch.randint(1, m.Rmax + 1, (1,), generator=g)) if m.sampled else m.Rmax
        else: R = None
        logits = m(ids, R); sel = tgt >= 0
        loss = F.cross_entropy(logits[sel].flatten(0, -2), assign[tgt[sel]].flatten())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (it + 1) % 250 == 0 or it == steps - 1:
            hist.append(dict(step=it + 1, loss=round(loss.item(), 4), **evaluate(m, Rs)))
            if log: print(arm, seed, hist[-1], f"{time.time()-t0:.0f}s", file=log, flush=True)
    torch.save(m.state_dict(), f"v2_{arm}_{seed}.pt")
    return dict(arm=arm, seed=seed, hist=hist, params=sum(p.numel() for p in m.parameters()), secs=time.time() - t0)


if __name__ == "__main__":
    arm, seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    res = run(arm, seed, steps, log=sys.stderr)
    json.dump(res, open(f"v2_{arm}_{seed}.json", "w")); print(json.dumps(res['hist'][-1]))
