"""v5_tiny models (d=128, ~2-3M parameters, built for throughput).

Block: pre-RMSNorm, 4-head causal attention with RoPE (fused scaled_dot_product_attention), SwiGLU FFN (384 hidden).

NCPModel (Phase A, reduced NCP [R1]):
  encoder (3 blocks over T tokens) -> H
  pool: c_j = RMS(mean of H over chunk j)          (no-affine RMS normalisation of the pooled vector; declared)
  concept module (2 blocks over M=T/4 concepts, causal) -> u_j        (u_j has seen c_0..c_j and predicts c_{j+1})
  PQ heads: S=8 segments x V=32 codes x 16 dims; pi = softmax(W u); chat_j = sum_v pi_v Q_v (soft expectation)
  one CRC port: A_j = W_A RMS(u_j)                  (IRC omitted; CRC reduced to one concept->decoder port; declared)
  causal shift: b_t = chat_{(t-3)//4} for t >= 3 else 0; same shift for A.  The position predicting the first token
  of chunk j+1 (t = 4j+3) is the first to receive chat_j.
  decoder input: h_t + b_t + gamma * A_t ; decoder: 3 blocks over T tokens; tied LM head.
  losses: NTP + alpha * NCP-MSE (per-feature mean, target detached) + beta * VQ (codebook-only, nearest code).

TransformerLM (V0): plain causal LM, 8 blocks, same width/tokenizer; parameter-matched to NCPModel.

GramCore (Phases B-E): replaces the concept module.  It runs in parallel over all concept boundaries (causal
attention over the concept axis), with GRAM's hierarchical recursion [R2]: R high transitions, n_L low refinements,
stochastic guidance eps added to the high state (prior p(eps|u) / posterior q(eps|u,y)), plus the external control Z
(d_z=8) held over windows of chunks with prior p(Z|context at window start) and posterior q(Z|window targets).
Readouts: u_new (consumed by the frozen PQ heads) and A_new (a distinct readout for the CRC port).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

D, NH, FF = 128, 4, 384
S_PQ, V_PQ = 8, 32
K = 4


def rope_cache(n, hd, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(n, device=device).float()
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


_ROPE = {}


def rope_cache(n, hd, device, base=10000.0):  # noqa: F811  (cached, half-split layout)
    key = (n, hd, str(device))
    if key not in _ROPE:
        inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
        f = torch.outer(torch.arange(n, device=device).float(), inv)
        f = torch.cat([f, f], -1)
        _ROPE[key] = (torch.cos(f), torch.sin(f))
    return _ROPE[key]


def apply_rope(x, cos, sin):
    # x [B,H,T,hd], half-split rotation
    h = x.shape[-1] // 2
    T = x.shape[2]
    rot = torch.cat([-x[..., h:], x[..., :h]], -1)
    return x * cos[:T] + rot * sin[:T]


class Block(nn.Module):
    def __init__(self, d=D, nh=NH, ff=FF):
        super().__init__()
        self.nh = nh
        self.n1 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.n2 = nn.RMSNorm(d)
        self.w12 = nn.Linear(d, 2 * ff, bias=False)
        self.w3 = nn.Linear(ff, d, bias=False)

    def forward(self, x, rope):
        B, T, Dm = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.nh, Dm // self.nh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, Dm))
        g, u = self.w12(self.n2(x)).chunk(2, dim=-1)
        return x + self.w3(F.silu(g) * u)


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def shift_to_tokens(v, T):
    """v [B,M,d] (index j = prediction made at boundary j for chunk j+1) -> [B,T,d] with b_t = v_{(t-3)//4}, t>=3."""
    rep = v.repeat_interleave(K, dim=1)[:, :T - 3]
    return F.pad(rep, (0, 0, 3, 0))


class TransformerLM(nn.Module):
    def __init__(self, vocab, n_layers=8, T=192):
        super().__init__()
        self.emb = nn.Embedding(vocab, D)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.nf = nn.RMSNorm(D)
        self.T = T
        self.apply(_init)

    def forward(self, x):
        rope = rope_cache(x.shape[1], D // NH, x.device)
        h = self.emb(x)
        for b in self.blocks:
            h = b(h, rope)
        return self.nf(h) @ self.emb.weight.t()


def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)


class NCPModel(nn.Module):
    def __init__(self, vocab, n_enc=3, n_con=2, n_dec=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, D)
        self.enc = nn.ModuleList([Block() for _ in range(n_enc)])
        self.enc_norm = nn.RMSNorm(D)
        self.con = nn.ModuleList([Block() for _ in range(n_con)])
        self.con_norm = nn.RMSNorm(D)
        self.heads = nn.Linear(D, S_PQ * V_PQ)
        self.Q = nn.Parameter(torch.randn(S_PQ, V_PQ, D // S_PQ) * 0.25)
        self.W_A = nn.Linear(D, D, bias=False)
        self.gamma = nn.Parameter(torch.full((D,), 0.1))
        self.dec = nn.ModuleList([Block() for _ in range(n_dec)])
        self.nf = nn.RMSNorm(D)
        self.apply(_init)
        self.register_buffer("q_init", torch.zeros((), dtype=torch.bool))
        self.register_buffer("usage", torch.zeros(S_PQ, V_PQ))

    # ---- pieces (reused by later phases)
    def encode(self, x):
        rope = rope_cache(x.shape[1], D // NH, x.device)
        h = self.emb(x)
        for b in self.enc:
            h = b(h, rope)
        return self.enc_norm(h)

    @staticmethod
    def pool(H):
        B, T, Dm = H.shape
        return rms(H.view(B, T // K, K, Dm).mean(2))

    def concept(self, C):
        rope = rope_cache(C.shape[1], D // NH, C.device)
        u = C
        for b in self.con:
            u = b(u, rope)
        return self.con_norm(u)

    def pq(self, u):
        B, M, _ = u.shape
        logits = self.heads(u).view(B, M, S_PQ, V_PQ)
        pi = logits.float().softmax(-1)
        chat = torch.einsum("bmsv,svd->bmsd", pi, self.Q).reshape(B, M, D)
        return logits, pi, chat

    def port(self, u):
        return self.W_A(rms(u))

    def decode(self, H, chat, A):
        T = H.shape[1]
        h = H + shift_to_tokens(chat, T) + self.gamma * shift_to_tokens(A, T)
        rope = rope_cache(T, D // NH, H.device)
        for b in self.dec:
            h = b(h, rope)
        return self.nf(h) @ self.emb.weight.t()

    # ---- VQ
    def vq(self, C):
        """Nearest-code labels and codebook-only loss on detached pooled concepts."""
        B, M, _ = C.shape
        c = C.detach().view(B, M, S_PQ, 1, D // S_PQ)
        d2 = (c - self.Q.unsqueeze(0).unsqueeze(0)).pow(2).sum(-1)       # [B,M,S,V]
        q = d2.argmin(-1)                                                 # [B,M,S]
        sel = self.Q[torch.arange(S_PQ, device=q.device), q]           # [B,M,S,d/S]
        loss = (c.squeeze(3) - sel).pow(2).sum(-1).mean()                # squared distance per segment, mean
        return q, loss

    @torch.no_grad()
    def count_usage(self, q):
        """q [B,M,S] nearest-code labels -> accumulate per-segment usage (outside the compiled region)."""
        flat = (q.reshape(-1, S_PQ) + torch.arange(S_PQ, device=q.device) * V_PQ).flatten()
        self.usage += torch.bincount(flat, minlength=S_PQ * V_PQ).view(S_PQ, V_PQ).float()

    @torch.no_grad()
    def vq_init_or_restart(self, C, q=None):
        """Data init on first call; afterwards restart codes unused since the last restart."""
        cs = C.detach().reshape(-1, S_PQ, D // S_PQ)
        n = cs.shape[0]
        if not bool(self.q_init):
            for s in range(S_PQ):
                self.Q.data[s] = cs[torch.randperm(n, device=cs.device)[:V_PQ], s]
            self.q_init.fill_(True)
            return 0
        dead = (self.usage == 0)
        nd = int(dead.sum())
        for s in range(S_PQ):
            idx = dead[s].nonzero().flatten()
            if len(idx):
                self.Q.data[s, idx] = cs[torch.randperm(n, device=cs.device)[:len(idx)], s]
        self.usage.zero_()
        return nd

    # ---- full forward
    def forward(self, x, plan_mode="normal", plan_stats=None, perm=None):
        H = self.encode(x)
        C = self.pool(H)
        u = self.concept(C)
        logits, pi, chat = self.pq(u)
        A = self.port(u)
        chat, A = apply_plan_mode(chat, A, plan_mode, plan_stats, perm)
        out = self.decode(H, chat, A)
        return dict(logits=out, H=H, C=C, u=u, pi=pi, chat=chat, A=A, pq_logits=logits)


def apply_plan_mode(chat, A, mode, stats, perm):
    """Route-authority interventions (eval only).  'mean' uses training-mean vectors per domain (not a null)."""
    if mode == "normal":
        return chat, A
    if mode == "mean":
        return stats["chat"].expand_as(chat), stats["A"].expand_as(A)
    if mode == "pq_mean":       # PQ route replaced by its mean, CRC route intact
        return stats["chat"].expand_as(chat), A
    if mode == "crc_mean":      # CRC route replaced by its mean, PQ route intact
        return chat, stats["A"].expand_as(A)
    if mode == "swap":          # another row's plans (same domain)
        return chat[perm], A[perm]
    if mode == "zero":
        return torch.zeros_like(chat), torch.zeros_like(A)
    raise ValueError(mode)


# ----------------------------------------------------------------------------- losses shared by phases
def token_nll(logits, x, m):
    """Per-position NLL [B,T-1] for predicting x[:,1:] from positions 0..T-2, masked by m[:, :T-1]."""
    lp = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1),
                         reduction="none").view(x.shape[0], -1)
    return lp * m[:, :-1]


def concept_mask(m):
    """cm[j]=1 if the target chunk j+1 has a scored token: inputs 4j+3..4j+6 (j = 0..M-2)."""
    B, T = m.shape
    M = T // K
    mm = F.pad(m, (0, 1))[:, 3:3 + 4 * (M - 1)].view(B, M - 1, 4).amax(-1)
    return mm


def ncp_mse(chat, C, cm):
    """Per-feature MSE between chat_j and detached c_{j+1}, averaged over scored targets. Returns (loss, per-row)."""
    err = (chat[:, :-1] - C[:, 1:].detach()).pow(2).mean(-1)   # [B,M-1]
    return (err * cm).sum() / cm.sum().clamp(min=1), err


# ----------------------------------------------------------------------------- GRAM core (Phases B-E)
class MLP2(nn.Module):
    def __init__(self, din, dout, hid=D):
        super().__init__()
        self.n = nn.RMSNorm(din)
        self.a = nn.Linear(din, 2 * hid)
        self.b = nn.Linear(hid, dout)

    def forward(self, x):
        g, u = self.a(self.n(x)).chunk(2, -1)
        return self.b(F.silu(g) * u)


class GramCore(nn.Module):
    """variant: 'det' (V2: no Z, no eps), 'adapter' (V2a: one non-recursive block, no Z/eps),
    'z' (V3: Z + posterior/prior, no eps), 'gram' (V4/V6: Z + eps)."""

    def __init__(self, variant="gram", R=2, nL=2, dz=8):
        super().__init__()
        self.variant, self.R, self.nL, self.dz = variant, R, nL, dz
        self.in_proj = nn.Linear(D, D, bias=False)
        if variant == "adapter":
            self.block = Block()
        else:
            self.fL = Block()
            self.fH = Block()
            self.post_L = nn.RMSNorm(D)     # post-norm on the recursive states (HRM/TRM use post-norm layers);
            self.post_H = nn.RMSNorm(D)     # without it the re-injected residual stream grew 12 -> 53 in 2 transitions
            self.h0 = nn.Parameter(torch.randn(D) * 0.02)   # fixed initial states (learned constants)
            self.l0 = nn.Parameter(torch.randn(D) * 0.02)
        self.use_z = variant in ("z", "gram")
        self.use_eps = variant == "gram"
        if self.use_z:
            self.A_Z = nn.Linear(dz, D, bias=False)
            self.W_Z = nn.Linear(dz, D, bias=False)
            self.prior_z = MLP2(D, 2 * dz)
            self.post_z = MLP2(2 * D, 2 * dz)
        if self.use_eps:
            self.prior_e = MLP2(D, 2 * D)
            self.post_e = MLP2(2 * D, 2 * D)
        self.track_norm = False
        self._hnorm = [0.0] * R
        self.out_n = nn.RMSNorm(D)
        self.B_u = nn.Linear(D, D, bias=False)
        self.B_A = nn.Linear(D, D, bias=False)
        self.apply(_init)
        if self.use_z:     # small non-zero lift (spec B2): candidates must differ from step 0
            nn.init.normal_(self.A_Z.weight, std=0.1)
            nn.init.normal_(self.W_Z.weight, std=0.1)
        if self.use_eps:   # start guidance near zero with small scale
            for mlp in (self.prior_e, self.post_e):
                nn.init.zeros_(mlp.b.weight); nn.init.constant_(mlp.b.bias[:D], 0.0); nn.init.constant_(mlp.b.bias[D:], -3.0)

    # ---- Z distributions
    def z_prior(self, e, starts):
        """starts [B,M] bool marks window starts; prior read at each position (used at starts)."""
        mu, ls = self.prior_z(e).chunk(2, -1)
        return mu, ls.clamp(-5, 2)

    def z_post(self, e, y, win):
        """Posterior per window from e at the window start and the mean target over the window."""
        B, M, _ = e.shape
        nW = int(win.max()) + 1
        oh = F.one_hot(win, nW).float()                          # [B,M,nW]
        cnt = oh.sum(1).clamp(min=1)                             # [B,nW]
        ymean = torch.einsum("bmw,bmd->bwd", oh, y) / cnt.unsqueeze(-1)
        first = (oh.cumsum(1) == 1).float() * oh                 # first position of each window
        estart = torch.einsum("bmw,bmd->bwd", first, e)
        mu, ls = self.post_z(torch.cat([estart, ymean], -1)).chunk(2, -1)
        return mu, ls.clamp(-5, 2), oh, first                   # per window [B,nW,dz]

    def forward(self, C, Z_pos=None, eps_mode="prior", y=None, noise=None, return_e=False, e=None):
        """C [B,M,d] clean pooled history (detached).  Z_pos [B,M,dz] control per position (None for det/adapter).
        eps_mode: 'prior' | 'post' | 'mean' (prior mean, no sampling) | 'off'.  y [B,M,d]: target c_{j+1} (post only).
        noise [B,R,M,d]: base draws for replay.  Returns u_new, A_new, kl_eps [B,M] (per position, summed dims)."""
        if e is None:
            e = self.in_proj(C)
        rope = rope_cache(C.shape[1], D // NH, C.device)
        B, M, _ = e.shape
        kl = torch.zeros(B, M, device=C.device)
        if self.variant == "adapter":
            h = self.block(e, rope)
        else:
            zl = self.W_Z(Z_pos) if self.use_z else 0.0
            h = self.h0.expand(B, M, D) + (self.A_Z(Z_pos) if self.use_z else 0.0)
            l = self.l0.expand(B, M, D)
            for r in range(self.R):
                for _ in range(self.nL):
                    l = self.post_L(self.fL(l + h + e + zl, rope))
                u = self.post_H(self.fH(h + l, rope))
                if self.use_eps and eps_mode != "off":
                    mp, lsp = self.prior_e(u).chunk(2, -1); lsp = lsp.clamp(-6, 1)
                    eta = noise[:, r] if noise is not None else torch.randn_like(u)
                    if eps_mode == "post":
                        mq, lsq = self.post_e(torch.cat([u, y], -1)).chunk(2, -1); lsq = lsq.clamp(-6, 1)
                        eps = mq + lsq.exp() * eta
                        kl = kl + gauss_kl(mq, lsq, mp, lsp).sum(-1)
                    elif eps_mode == "mean":
                        eps = mp
                    else:
                        eps = mp + lsp.exp() * eta
                    h = u + eps
                else:
                    h = u
                if not torch.compiler.is_compiling():
                    self._hnorm[r] = h.detach().norm(dim=-1).mean().item() if self.track_norm else 0.0
        hn = self.out_n(h)
        out = (self.B_u(hn), self.B_A(hn), kl)
        return out + (e,) if return_e else out


def gauss_kl(mq, lsq, mp, lsp):
    """KL(N(mq, e^{2lsq}) || N(mp, e^{2lsp})) elementwise."""
    return lsp - lsq + (torch.exp(2 * lsq) + (mq - mp).pow(2)) / (2 * torch.exp(2 * lsp)) - 0.5


def random_windows(B, M, rng_t, lo=1, hi=4, device="cpu"):
    """Window id per position, lengths uniform in {lo..hi}.  Returns win [B,M] long, starts [B,M] bool."""
    lens = torch.randint(lo, hi + 1, (B, M), generator=rng_t)
    starts = torch.zeros(B, M, dtype=torch.bool)
    for b in range(B):
        p = 0
        i = 0
        while p < M:
            starts[b, p] = True
            p += int(lens[b, i]); i += 1
    win = starts.long().cumsum(1) - 1
    return win.to(device), starts.to(device)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
