"""HRM THINKER concept core v0 (design draft) - reference implementation + checks.

Reads encoder v1.0's c_hat (one sphere-normed vector per 4-token chunk), runs a looped two-state core
(TRM-style: Z = scratch state that reads the input, Y = plan state that reads only Z and itself),
predicts the next concept over a product-quantised vocabulary (NCP), and emits the plan for the decoder.
Block anatomy = encoder v1.0's FULL layer (gated residual, QKV conv, NoPE, per-head temperature) at concept rate.
Run `python concept_v0.py` for the checks.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from encoder_v1 import Encoder, GRRead, GRWrite, Attn, SwiGLU, ZCRMSNorm, rms


class CoreLayer(nn.Module):
    """One layer of the shared step function f. Attention + FFN weights are shared by both modes;
    each mode has its own read/write gates.  z-mode: read {X, Z, Y} -> write Z.  y-mode: read {Z, Y} -> write Y."""
    def __init__(s, d, f, s0=0.1):
        super().__init__()
        s.attn, s.ffn = Attn(d, window=None), SwiGLU(d, f)                      # NoPE + log_tau + QKV conv
        s.rz = nn.ModuleList([GRRead(d, 3) for _ in range(2)]); s.wz = nn.ModuleList([GRWrite(d, 1) for _ in range(2)])
        s.ry = nn.ModuleList([GRRead(d, 2) for _ in range(2)]); s.wy = nn.ModuleList([GRWrite(d, 1) for _ in range(2)])
        a = math.log((s0 / 2) / (1 - s0 / 2))                                    # 2*sigmoid(a) = s0
        for w in list(s.wz) + list(s.wy): nn.init.constant_(w.u.bias, a)
    def forward(s, X, Z, Y, doc, mode):
        for k in range(2):
            src = [X, Z, Y] if mode == 'z' else [Z, Y]
            x = (s.rz if mode == 'z' else s.ry)[k](src)
            out = s.attn(x, doc) if k == 0 else s.ffn(x)
            g = (s.wz if mode == 'z' else s.wy)[k](x)
            if mode == 'z': Z = Z + g * out
            else: Y = Y + g * out
        return Z, Y


class ConceptVocab(nn.Module):
    """Product-quantised concept vocabulary: S segments x N codes x ds dims (S*ds = dc).
    Trained ONLY by L_VQ = ||sg(c^s) - e^s_n||^2 (NCP-ArchPreview form); everything else reads sg(e).
    Health (F3): k-means++ init from the first batch of encoder outputs, and every `restart_every` steps each code
    that received no assignment is re-seeded to a random current c^s (vq_check2/3: 128/128 used, VE 0.94, also under drift)."""
    def __init__(s, dc, S=64, N=128, restart_every=100):
        super().__init__(); s.S, s.N, s.ds, s.every = S, N, dc // S, restart_every
        s.e = nn.Parameter(torch.randn(S, N, s.ds))                              # replaced by init_from() before step 1
        s.register_buffer('hits', torch.zeros(S, N)); s.register_buffer('step', torch.zeros((), dtype=torch.long))
    def codes(s): return s.e
    def assign(s, c):                                                             # c: ..., dc (already sg)
        cs = c.unflatten(-1, (s.S, s.ds))
        return (cs.unsqueeze(-2) - s.e.detach()).pow(2).sum(-1).argmin(-1)       # ..., S
    @torch.no_grad()
    def init_from(s, c):                                                          # k-means++ per segment
        x = c.reshape(-1, s.S, s.ds)
        for j in range(s.S):
            xs = x[:, j]; pick = [xs[torch.randint(len(xs), (1,))]]
            for _ in range(s.N - 1):
                d = torch.cdist(xs, torch.cat(pick)).min(1).values.pow(2)
                pick.append(xs[torch.multinomial(d / d.sum() if d.sum() > 0 else torch.ones_like(d), 1)])
            s.e[j] = torch.cat(pick)
    @torch.no_grad()
    def _restart(s, c):
        x = c.reshape(-1, s.S, s.ds)
        for j in range(s.S):
            dead = s.hits[j] == 0
            if dead.any(): s.e[j, dead] = x[torch.randint(len(x), (int(dead.sum()),)), j]
        s.hits.zero_()
    def vq_loss(s, c):
        c = c.detach(); cs = c.unflatten(-1, (s.S, s.ds)); n = s.assign(c)
        if s.training:
            s.hits += F.one_hot(n.reshape(-1, s.S), s.N).sum(0); s.step += 1
            if s.step % s.every == 0: s._restart(c)
        q = s.e[torch.arange(s.S), n]                                             # ..., S, ds (grad -> e only)
        return (cs - q).pow(2).sum(-1).mean(), n


class NCPHead(nn.Module):
    """pi^s_n = softmax_n( -||q^s - sg(e^s_n)||^2 / tau_s ),  q = W_q u.   c_pred^s = sum_n pi^s_n sg(e^s_n)."""
    def __init__(s, dc, vocab):
        super().__init__(); s.v = vocab; s.Wq = nn.Linear(dc, dc, bias=False)
        s.log_tau = nn.Parameter(torch.full((vocab.S,), math.log(2 * vocab.ds)))     # tau0 = E||a-b||^2 for unit-variance dims -> near-uniform pi at init; no WD
    def forward(s, u):
        e = s.v.codes().detach(); q = s.Wq(u).unflatten(-1, (s.v.S, s.v.ds))
        logits = -(q.unsqueeze(-2) - e).pow(2).sum(-1) / s.log_tau.exp()[:, None]
        pi = logits.softmax(-1)
        c_pred = torch.einsum('...sn,snd->...sd', pi, e).flatten(-2)
        return logits, c_pred


class ConceptCore(nn.Module):
    def __init__(s, dc=512, f=1408, layers=2, n_z=2, cycles=2, S=64, N=128, s0=0.1):
        super().__init__(); s.n_z, s.cycles = n_z, cycles
        s.f = nn.ModuleList([CoreLayer(dc, f, s0) for _ in range(layers)])
        s.vocab = ConceptVocab(dc, S, N); s.head = NCPHead(dc, s.vocab); s.out_norm = ZCRMSNorm(dc)
    def forward(s, c_hat, doc_c):
        X = c_hat; Z = X; Y = torch.zeros_like(X)                                 # Z0 = input (identity), Y0 = 0 (C1)
        us = []
        for _ in range(s.cycles):
            for _ in range(s.n_z):
                for L in s.f: Z, Y = L(X, Z, Y, doc_c, 'z')
            for L in s.f: Z, Y = L(X, Z, Y, doc_c, 'y')
            us.append(s.out_norm(Y))                                              # plan state after each cycle
        heads = [s.head(u) for u in us]
        return us, heads                                                          # plan = (u_m, c_pred_{m+1}) from last cycle

    def losses(s, c_hat, us, heads, doc_c):
        """Deep-supervised NCP (CE on codes, + MSE arm) and VQ. u_m predicts concept m+1 of the same document."""
        tgt = c_hat.detach()
        vq, n = s.vocab.vq_loss(tgt)
        valid = (doc_c[:, 1:] == doc_c[:, :-1]).float()                           # never predict across documents
        ce = mse = 0
        for logits, c_pred in heads:
            l = F.cross_entropy(logits[:, :-1].flatten(0, 2), n[:, 1:].flatten(), reduction='none').view(*n[:, 1:].shape)
            ce = ce + (l.mean(-1) * valid).sum() / valid.sum()
            mse = mse + (((c_pred[:, :-1] - tgt[:, 1:]).pow(2).mean(-1)) * valid).sum() / valid.sum()
        return ce / len(heads), mse / len(heads), vq


def token_plan(u, c_pred, doc, k=4):
    """Plan visible to decoder position t (0-indexed, predicting x_{t+1}): concept j*(t) = floor((t+1)/k) - 1.
    Null plan (zeros here; a learned vector in the decoder) when j* < 0 OR chunk j* belongs to another document:
    the first k-1 tokens of EVERY document, not only of the row."""
    T = doc.shape[1]; t = torch.arange(T); j = (t + 1) // k - 1
    ok = (j >= 0)[None] & (doc[:, (j.clamp_min(0) * k)] == doc)                 # B, T
    idx = torch.where(ok, j[None] + 1, torch.zeros_like(ok, dtype=torch.long))
    pad = lambda v: torch.cat([torch.zeros_like(v[:, :1]), v], 1)
    take = lambda v: torch.gather(pad(v), 1, idx[..., None].expand(-1, -1, v.shape[-1]))
    return take(u), take(c_pred)


# ------------------------------------------------------------------ checks
def _randomize(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if 'tab.' not in n and not n.endswith('vocab.e'): p.add_(torch.randn_like(p) * 0.2)

def run():
    torch.manual_seed(0); torch.set_default_dtype(torch.float64)
    enc = Encoder(V=64, d=64, f=128, rows=1021); core = ConceptCore(dc=64, f=128, S=8, N=16)
    _randomize(enc); _randomize(core); enc.eval(); core.eval()
    T, p = 256, 101
    ids = torch.randint(0, 64, (1, T)); doc = torch.zeros(1, T, dtype=torch.long)
    ids2 = ids.clone(); ids2[0, p] = (ids[0, p] + 7) % 64
    def plan(i, d):
        _, c, _ = enc(i, d); us, hs = core(c, d[:, ::4]); return token_plan(us[-1], hs[-1][1], d)
    with torch.no_grad(): (u1, c1), (u2, c2) = plan(ids, doc), plan(ids2, doc)
    ch = ((u1 - u2).abs().amax(-1) + (c1 - c2).abs().amax(-1))[0]
    first = torch.nonzero(ch != 0).min().item()
    assert not (ch[:p] != 0).any(), "plan at t < p must not see token p"
    assert first == (p // 4) * 4 + 3, first                                      # first used at the end of p's chunk
    print(f"PASS plan causality: editing token {p} first changes the plan at t={first} (end of its chunk), never before")
    print(f"     => plan at t never contains x_(t+1): the decoder cannot copy its target through the core")

    doc2 = torch.zeros(1, T, dtype=torch.long); doc2[0, 128:] = 1
    ids3 = ids.clone(); ids3[0, 120] = (ids[0, 120] + 5) % 64
    with torch.no_grad(): (a, b), (a2, b2) = plan(ids, doc2), plan(ids3, doc2)
    assert torch.equal(a[0, 128:], a2[0, 128:]) and torch.equal(b[0, 128:], b2[0, 128:])
    print("PASS no leakage across documents through the core (attention, QKV conv at concept rate, NCP targets)")

    # gradient paths at float32, M2 size
    torch.set_default_dtype(torch.float32); torch.manual_seed(0)
    enc, core = Encoder(), ConceptCore()
    ids = torch.randint(0, 8192, (2, 256)); doc = torch.zeros(2, 256, dtype=torch.long); dc_ = doc[:, ::4]
    _, c_hat, _ = enc(ids, doc); core.vocab.init_from(c_hat.detach()); us, heads = core(c_hat, dc_)
    ce, mse, vq = core.losses(c_hat, us, heads, dc_)
    gW = lambda loss, ps: [torch.autograd.grad(loss, p, retain_graph=True, allow_unused=True)[0] for p in ps]
    g = gW(vq, [core.vocab.e, core.head.Wq.weight, enc.pool.weight])
    assert g[0] is not None and g[1] is None and g[2] is None
    g = gW(ce, [core.vocab.e, core.head.Wq.weight, enc.pool.weight])
    assert g[0] is None and g[1].abs().sum() > 0 and g[2].abs().sum() > 0
    plan_u, plan_c = token_plan(us[-1], heads[-1][1], doc)
    g = gW(plan_u.sum() + plan_c.sum(), [core.vocab.e, enc.pool.weight, enc.layers[7].attn.q.weight])
    assert g[0] is None and g[1].abs().sum() > 0 and g[2].abs().sum() > 0
    print("PASS gradients: codebook <- L_VQ only | NCP -> core + encoder (via history, target is sg) | plan -> core + encoder incl. FULL layers")

    # init behaviour
    with torch.no_grad():
        u = us[-1][0]; un = F.normalize(u, dim=-1); cos = (un @ un.T)[torch.triu(torch.ones(64, 64, dtype=torch.bool), 1)]
    print(f"INFO init: plan u mean pairwise cos {cos.mean():.3f} (not a constant plan) | NCP CE {ce.item():.2f} nats"
          f" (uniform = ln 128 = {math.log(128):.2f})")

    ne = lambda m: sum(q.numel() for q in m.parameters())
    per_layer = ne(core.f[0]); M = 256
    mac_layer = (per_layer - sum(ne(x) for x in list(core.f[0].rz) + list(core.f[0].ry) + list(core.f[0].wz) + list(core.f[0].wy)))
    apps = core.cycles * (core.n_z + 1) * len(core.f)
    attn_mac = 2 * (M / 2) * 512                                                  # QK^T + AV per position, causal average, T=1024
    mflop = 6 * apps * (mac_layer + attn_mac) / 4 / 1e6                           # train FLOPs per TOKEN (core runs per 4 tokens)
    print(f"INFO core params {ne(core)/1e6:.2f}M ({per_layer/1e6:.2f}M per layer x {len(core.f)}, head+vocab {(ne(core.head)-ne(core.vocab)+ne(core.vocab))/1e6:.2f}M)"
          f" | {apps} layer applications per concept | ~{mflop:.0f} MFLOP per training token at T=1024")


if __name__ == "__main__":
    run()
