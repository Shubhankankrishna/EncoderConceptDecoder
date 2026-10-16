"""End-to-end test bench for the concept module: tokens -> encoder_v1 (frozen spec) -> concept core -> contract decoder.

Encoder: encoder_v1 exactly (S S S F S S S F, two streams, Engram, CRC pooling, sphere norm) at test width d=64
         (4 heads x 16, partial RoPE 4/16, Engram tables 1021 rows).
Core arms:  loop   = v2  : ONE shared GR layer looped; streams X (input, re-read every pass), S (state, S0 = 0); R ~ U{1..4}
            loopz  = v2.1: loop + read-only latent stream Z = z_k (K = 4 learned), one hypothesis per document,
                     Forward XM with relaxed winner-take-all on the NCP loss; decoder reads the selected hypothesis
                     (hard-EM); likelihood is always evaluated as the K-mixture.
            stack4 = NCP-style 4 untied layers; ident = no core computation (plan = pooled concept).
NCP head: PQ codebook (8 segments x 32 codes), k-means++ init from the first batch, VQ loss on sg(c), dead-code restarts;
          distance-softmax head, CE on the next concept's codes.
Decoder (contract): continues the encoder's L stream; reads (u_j*, c_pred_j*) at j*(t) = floor((t+1)/4) - 1 only, null plan
          when j* is outside the document; floored gate s in [0.5, 2]; 2 SWA (w=8) layers; LM head tied to the embedding.
"""
import sys, os, math, json, time, torch, torch.nn as nn, torch.nn.functional as F
ALIGN = os.environ.get('E2E_ALIGN') == '1'   # facts padded to 4 tokens = exactly one concept each (diagnostic)
sys.path.insert(0, '.')
import encoder_v1 as E
_Attn = E.Attn
E.Attn = lambda d, window=None: _Attn(d, 4, 2, d // 4, window, 4)
from encoder_v1 import GRRead, GRWrite, SwiGLU, ZCRMSNorm, rms
torch.set_num_threads(1)

NE, RT, QT, EQ, PADT = 64, 64, 65, 66, 67; FIL0, NFIL = 68, 16; VT = FIL0 + NFIL
T, T0, QU = 188, 104, 12


# ------------------------------------------------------------------ data
def make_batch(B, g, hops=(1, 2, 3), branch=False, path=False, nq=7):
    """Facts [a R b] (8-cycle; if branch, 2 nodes get a 2nd successor) at 0..23, filler to T0, then nq query units:
       [Q x R^h PAD..](8) [= ans PAD..](4).  path=True -> answer is the whole path b1..bh (h <= 3)."""
    ids = torch.full((B, T), PADT); tgt = torch.full((B, T), -1); meta = []
    for b in range(B):
        ents = torch.randperm(NE, generator=g)[:8].tolist(); S = {ents[i]: [ents[(i + 1) % 8]] for i in range(8)}
        if branch:
            for a in torch.randperm(8, generator=g)[:2].tolist():
                opts = [e for e in ents if e != ents[a] and e not in S[ents[a]]]
                S[ents[a]].append(opts[torch.randint(len(opts), (1,), generator=g).item()])
        facts = [(a, s) for a in S for s in S[a]][:8] if not branch else [(a, s) for a in S for s in S[a]]
        order = torch.randperm(len(facts), generator=g).tolist(); seq = [x for i in order for x in ((facts[i][0], RT, facts[i][1], PADT) if ALIGN else (facts[i][0], RT, facts[i][1]))]
        seq += (FIL0 + torch.randint(NFIL, (T0 - len(seq),), generator=g)).tolist(); qs = []
        for q in range(nq):
            h = hops[torch.randint(len(hops), (1,), generator=g).item()]
            x = ents[torch.randint(8, (1,), generator=g).item()]
            if branch: x = [a for a in S if len(S[a]) == 2][q % 2]
            p = [x]
            for _ in range(h): p.append(S[p[-1]][torch.randint(len(S[p[-1]]), (1,), generator=g).item()])
            unit = [QT, x] + [RT] * h; unit += [PADT] * (8 - len(unit))
            ans = p[1:] if path else [p[-1]]
            unit += [EQ] + ans + [PADT] * (3 - len(ans)); base = len(seq)
            for i, a in enumerate(ans): tgt[b, base + 8 + i] = a                      # predicted at '=' and following answer tokens
            seq += unit; qs.append((base, x, h))
        ids[b] = torch.tensor(seq); meta.append((S, qs))
    return ids, tgt, meta


# ------------------------------------------------------------------ concept core
class ZLayer(nn.Module):
    def __init__(s, d, f, n_read):
        super().__init__(); s.r1, s.w1, s.r2, s.w2 = GRRead(d, n_read), GRWrite(d, 1), GRRead(d, n_read), GRWrite(d, 1)
        s.attn, s.ffn = _Attn(d, 4, 2, d // 4, None, 4), SwiGLU(d, f)
    def forward(s, streams, S, doc):
        x = s.r1(streams + [S]); S = S + s.w1(x) * s.attn(x, doc)
        x = s.r2(streams + [S]); return S + s.w2(x) * s.ffn(x)


class GRL(nn.Module):
    def __init__(s, d, f, window=None):
        super().__init__(); s.r1, s.w1, s.r2, s.w2 = GRRead(d, 1), GRWrite(d, 1), GRRead(d, 1), GRWrite(d, 1)
        s.attn, s.ffn = _Attn(d, 4, 2, d // 4, window, 4), SwiGLU(d, f)
    def forward(s, R, doc):
        x = s.r1([R]); R = R + s.w1(x) * s.attn(x, doc)
        x = s.r2([R]); return R + s.w2(x) * s.ffn(x)


class Core(nn.Module):
    def __init__(s, arm, d, f, K=1, Sg=8, N=32):
        super().__init__(); s.arm, s.K = arm, K
        if arm.startswith('loop'): s.layer = ZLayer(d, f, 3 if K > 1 else 2)
        if arm == 'stack4': s.layers = nn.ModuleList([GRL(d, f) for _ in range(4)])
        if K > 1: s.z = nn.Parameter(torch.randn(K, d))
        s.Sg, s.N, s.ds = Sg, N, d // Sg
        s.e = nn.Parameter(torch.randn(Sg, N, s.ds)); s.register_buffer('hits', torch.zeros(Sg, N)); s.register_buffer('inited', torch.zeros(()))
        s.out_norm = ZCRMSNorm(d); s.Wq = nn.Linear(d, d, bias=False); s.log_tau = nn.Parameter(torch.full((Sg,), math.log(2 * s.ds)))
    def core(s, X, doc_c, R, k=None):
        if s.arm == 'ident': return X
        if s.arm == 'stack4':
            H = X
            for L in s.layers: H = L(H, doc_c)
            return H
        streams = [X] if k is None else [X, s.z[k].view(1, 1, -1).expand_as(X) if isinstance(k, int) else s.z[k][:, None, :].expand_as(X)]
        St = torch.zeros_like(X)
        for _ in range(R): St = s.layer(streams, St, doc_c)
        return St
    def head(s, u):
        q = s.Wq(u).unflatten(-1, (s.Sg, s.ds)); e = s.e.detach().clone()   # own copy: restarts rewrite e in place
        logits = -(q.unsqueeze(-2) - e).pow(2).sum(-1) / s.log_tau.exp()[:, None]
        return logits, torch.einsum('...sn,snd->...sd', logits.softmax(-1), e).flatten(-2)
    def forward(s, X, doc_c, R, k=None):
        u = s.out_norm(s.core(X, doc_c, R, k)); logits, cpred = s.head(u); return u, logits, cpred
    # ---- vocabulary (trained only by VQ; k-means++ init; restarts)
    def assign(s, c):
        return (c.unflatten(-1, (s.Sg, s.ds)).unsqueeze(-2) - s.e.detach()).pow(2).sum(-1).argmin(-1)
    @torch.no_grad()
    def vq_init(s, c):
        x = c.reshape(-1, s.Sg, s.ds)
        for j in range(s.Sg):
            xs = x[:, j]; pick = [xs[torch.randint(len(xs), (1,))]]
            for _ in range(s.N - 1):
                dd = torch.cdist(xs, torch.cat(pick)).min(1).values.pow(2); pick.append(xs[torch.multinomial(dd / dd.sum(), 1)])
            s.e[j] = torch.cat(pick)
        s.inited.fill_(1)
    def vq_loss(s, c, step, every=100):
        c = c.detach().reshape(-1, s.Sg, s.ds); n = s.assign(c.flatten(1))
        s.hits += F.one_hot(n, s.N).sum(0)
        if step % every == 0 and step > 0:
            with torch.no_grad():
                for j in range(s.Sg):
                    dead = s.hits[j] == 0
                    if dead.any(): s.e[j, dead] = c[torch.randint(len(c), (int(dead.sum()),)), j]
            s.hits.zero_()
        return (c - s.e[torch.arange(s.Sg), n]).pow(2).sum(-1).mean()


class Decoder(nn.Module):
    def __init__(s, d, f, n_layers=2):
        super().__init__(); s.Wp = nn.Linear(2 * d, d, bias=False); s.null = nn.Parameter(torch.zeros(d)); s.a = nn.Parameter(torch.zeros(1))
        s.layers = nn.ModuleList([GRL(d, f, window=8) for _ in range(n_layers)]); s.norm = ZCRMSNorm(d)
    def plan_index(s, doc):
        t = torch.arange(doc.shape[1]); j = (t + 1) // 4 - 1
        ok = (j >= 0)[None] & (doc[:, (j.clamp_min(0) * 4)] == doc); return j.clamp_min(0), ok
    def forward(s, tok, u, cpred, doc, plan_off=False):
        j, ok = s.plan_index(doc)
        p = s.Wp(torch.cat([u[:, j], cpred[:, j]], -1))
        if plan_off: ok = torch.zeros_like(ok)
        p = torch.where(ok[..., None], p, s.null.expand_as(p))
        D = tok + (0.5 + 1.5 * torch.sigmoid(s.a)) * rms(p)                       # floored gate: can never close
        for L in s.layers: D = L(D, doc)
        return s.norm(D)


class Model(nn.Module):
    def __init__(s, arm, K=1, d=64, f=176):
        super().__init__(); s.K = K
        s.enc = E.Encoder(V=VT, d=d, f=f, rows=1021); s.core = Core(arm, d, f, K); s.dec = Decoder(d, f)
    def forward(s, ids, doc, R, k=None, plan_off=False):
        tok, c_hat, _ = s.enc(ids, doc); doc_c = doc[:, ::4]
        u, logits, cpred = s.core(c_hat, doc_c, R, k)
        out = s.dec(tok, u, cpred, doc, plan_off)
        return out @ s.enc.E.weight.T / math.sqrt(out.shape[-1]), logits, c_hat, doc_c


def ncp_ce(core, logits, c_hat, doc_c):                     # per concept position, CE on the next concept's codes
    n = core.assign(c_hat.detach())
    valid = (doc_c[:, 1:] == doc_c[:, :-1]).float()
    lp = logits[:, :-1].log_softmax(-1).gather(-1, n[:, 1:, :, None]).squeeze(-1).mean(-1)
    return -(lp * valid)                                     # B, M-1


# ------------------------------------------------------------------ train
def train(arm, seed, steps, K=1, task="chain", B=24, lr=1e-3, alpha=0.1, eps=0.05, log=None, eval_every=500, cur_steps=1500):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    m = Model(arm, K); nowd = [p for n_, p in m.named_parameters() if ('tab.' in n_ or 'log_tau' in n_ or n_.startswith('enc.E.') or n_.endswith('.w'))]
    yes = [p for n_, p in m.named_parameters() if all(p is not q for q in nowd)]
    opt = torch.optim.AdamW([dict(params=yes, weight_decay=0.1), dict(params=nowd, weight_decay=0)], lr=lr, betas=(0.9, 0.95))
    hist = []; t0 = time.time(); wins = torch.zeros(max(K, 1))
    for it in range(steps):
        for gp in opt.param_groups: gp['lr'] = lr * min(1, (it + 1) / 200)
        if task == 'branch': hop_set = (2,)
        elif task == 'chain1': hop_set = (1,)
        elif task == 'chaincur': hop_set = (1,) if it < cur_steps else (1, 2, 3)
        else: hop_set = (1, 2, 3)
        ids, tgt, _ = make_batch(B, g, hops=hop_set, branch=task == 'branch', path=task == 'branch')
        doc = torch.zeros_like(ids); R = int(torch.randint(1, 5, (1,), generator=g))
        tok, c_hat, _ = m.enc(ids, doc); doc_c = doc[:, ::4]
        if not m.core.inited: m.core.vq_init(c_hat.detach())
        if K == 1:
            u, lg, cp = m.core(c_hat, doc_c, R); ncp = ncp_ce(m.core, lg, c_hat, doc_c).mean()
        else:
            outs = [m.core(c_hat, doc_c, R, k) for k in range(K)]
            per = torch.stack([ncp_ce(m.core, o[1], c_hat, doc_c).sum(-1) for o in outs])          # K, B  (per-document NCP)
            kstar = per.argmin(0); wins += torch.bincount(kstar, minlength=K)
            w = torch.full_like(per, eps / (K - 1)).scatter_(0, kstar[None], 1 - eps)
            ncp = (w * per).sum(0).mean() / (c_hat.shape[1] - 1)
            sel = lambda i: torch.stack([o[i] for o in outs])[kstar, torch.arange(B)]               # decoder reads k* (hard-EM)
            u, cp = sel(0), sel(2)
        out = m.dec(tok, u, cp, doc); logit_t = out @ m.enc.E.weight.T / math.sqrt(out.shape[-1])
        mask = tgt >= 0; ntp = F.cross_entropy(logit_t[mask], tgt[mask])
        loss = ntp + alpha * ncp + m.core.vq_loss(c_hat, it)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (it + 1) % eval_every == 0 or it == steps - 1:
            ev = evaluate(m, 'chain' if task.startswith('chain') else task); ev.update(step=it + 1, ntp=round(ntp.item(), 3), ncp=round(ncp.item(), 3))
            if K > 1: ev['win_share'] = [round(x, 3) for x in (wins / wins.sum()).tolist()]; wins.zero_()
            hist.append(ev)
            if log: print(arm, K, seed, ev, f"{time.time()-t0:.0f}s", file=log, flush=True)
    torch.save(m.state_dict(), f"e2e_{task}_{arm}_K{K}_{seed}_a{alpha}.pt")
    return dict(arm=arm, K=K, seed=seed, task=task, hist=hist, secs=time.time() - t0,
                params={n: sum(p.numel() for p in getattr(m, n).parameters()) for n in ('enc', 'core', 'dec')})


@torch.no_grad()
def answer_logits(m, ids, R, k=None, plan_off=False):
    doc = torch.zeros_like(ids); lt, _, _, _ = m(ids, doc, R, k, plan_off); return lt.log_softmax(-1)


@torch.no_grad()
def evaluate(m, task, n=96, Rs=(4,)):
    m.eval(); out = {}
    ks = [None] if m.K == 1 else list(range(m.K))
    if task == 'chain':
        for R in Rs:
            g = torch.Generator().manual_seed(4242); ids, tgt, meta = make_batch(n, g, hops=(1, 2, 3, 4, 5))
            lps = [answer_logits(m, ids, R, k) for k in ks]
            lp = torch.logsumexp(torch.stack(lps), 0) - math.log(len(ks))                            # K-mixture predictive
            pos = torch.stack([torch.tensor([q[0] + 8 for q in mt[1]]) for mt in meta]); hs = torch.tensor([[q[2] for q in mt[1]] for mt in meta])
            pred = lp.gather(1, pos[..., None].expand(-1, -1, VT)).argmax(-1); truth = tgt.gather(1, pos)
            for h in (1, 2, 3, 4, 5): out[f'h{h}'] = round((pred == truth)[hs == h].float().mean().item(), 3)
            sel = hs <= 3; out['ans_nll'] = round(-lp.gather(1, pos[..., None].expand(-1, -1, VT)).gather(-1, truth[..., None]).squeeze(-1)[sel].mean().item(), 3)
    m.train(); return out


if __name__ == "__main__":
    arm, K, seed, steps, task = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    alpha = float(sys.argv[6]) if len(sys.argv) > 6 else 0.1
    res = train(arm, seed, steps, K=K, task=task, alpha=alpha, log=sys.stderr, eval_every=250 if task == 'chain1' else 500)
    res['alpha'] = alpha; json.dump(res, open(f"e2e_{task}_{arm}_K{K}_{seed}_a{alpha}.json", "w"))
