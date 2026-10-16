"""Concept module v2 = D6 LOOPFORMER-MoE-DS (model_v2/MODEL_V2_19M.md §6, concept_v2/DESIGN_D6.md).

Four DISTINCT layers per pass, d 256, full causal attention over the chunk positions (doc-masked, NoPE, per-head
temperature, sigmoid output gate, no q/k/v convs), each sublayer reads [X~, S] through the gated-residual read and writes
the carry S. The FFN of every layer is top-2 of ONE shared pool of E SwiGLU experts (per-layer routers). Identity at init:
exactly one zero-init per path (attention W_o, experts' W_2).

The TRAINER owns the pass/window loop (per-window backward, detached carry, per-pass NCP, final-window consistency);
this module exposes the pieces:
    X, S = core.init_state(c_hat)            X~ = W_in c_hat (re-read every pass), S_0 = W_s c_hat
    S, aux = core.one_pass(X, S, cdoc, mask)  one cycle through the 4 layers (input carry is LN_h(S) for passes >= 2)
    u = core.readout(S)                       separately normed readout of the pass
    logits, c_pred = core.head(u)             PQ distance head (as v1's ConceptCoreV2.head, W_q 256 -> 384)
and the evaluation forward  u, logits, c_pred = core(c_hat, cdoc, R)  (R passes, no windows), used by the decoder path.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hrm_encoder import GRRead, GRWrite, ZCRMSNorm, rms, build_masks
from concept_core_v2 import ConceptVocab


class CoreAttention(nn.Module):
    """Gated full causal attention at concept rate: q/k zcRMSNorm, per-head temperature, NoPE, sigmoid output gate."""
    def __init__(self, d, H, Hk, dh):
        super().__init__()
        self.H, self.Hk, self.dh = H, Hk, dh
        self.q = nn.Linear(d, H * dh, bias=False)
        self.k = nn.Linear(d, Hk * dh, bias=False)
        self.v = nn.Linear(d, Hk * dh, bias=False)
        self.gate = nn.Linear(d, H * dh, bias=False)
        self.o = nn.Linear(H * dh, d, bias=False)
        nn.init.zeros_(self.o.weight)                                   # identity at init (one zero per path)
        self.q_norm, self.k_norm = ZCRMSNorm(dh), ZCRMSNorm(dh)
        self.log_tau = nn.Parameter(torch.zeros(H))

    def forward(self, x, mask):
        B, M, _ = x.shape
        H, Hk, dh = self.H, self.Hk, self.dh
        q = self.q_norm(self.q(x).view(B, M, H, dh).transpose(1, 2)) * self.log_tau.exp().view(1, H, 1, 1)
        k = self.k_norm(self.k(x).view(B, M, Hk, dh).transpose(1, 2))
        v = self.v(x).view(B, M, Hk, dh).transpose(1, 2)
        k, v = k.repeat_interleave(H // Hk, 1), v.repeat_interleave(H // Hk, 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, M, H * dh)
        return self.o(y * torch.sigmoid(self.gate(x)))


class ExpertPool(nn.Module):
    """E SwiGLU experts d -> de -> d stored as batched weights; W2 zero-init."""
    def __init__(self, d, de, E):
        super().__init__()
        self.E, self.d, self.de = E, d, de
        self.w1 = nn.Parameter(torch.randn(E, d, de) / math.sqrt(d))
        self.w3 = nn.Parameter(torch.randn(E, d, de) / math.sqrt(d))
        self.w2 = nn.Parameter(torch.zeros(E, de, d))

    def expert_fn(self, xe):                                           # xe (E, C, d) -> (E, C, d)
        return torch.bmm(F.silu(torch.bmm(xe, self.w1)) * torch.bmm(xe, self.w3), self.w2)


def moe_dispatch(x, logits, pool, top_k=2, capacity_factor=1.25, dense=False):
    """Top-k routing of the rows of x (N, d) over the shared pool with Switch-style capacity.
    Returns y (N, d), aux dict (lb, z, dropped fraction, load (E,), top-k indices (N, k)).
    dense=True evaluates every expert on every row and masks (reference path for the exact test; C = infinity)."""
    N, d = x.shape
    E = pool.E
    probs = logits.float().softmax(-1)
    if top_k == 2:                                                        # 2 x max: same result as topk, ~20x faster on MPS
        v1, i1 = probs.max(-1)
        v2, i2 = probs.scatter(-1, i1.unsqueeze(-1), -1.0).max(-1)
        gate_v, idx = torch.stack([v1, v2], -1), torch.stack([i1, i2], -1)
    else:
        gate_v, idx = probs.topk(top_k, dim=-1)                           # (N, k)
    gate_v = (gate_v / gate_v.sum(-1, keepdim=True)).to(x.dtype)
    # aux losses (Switch / ST-MoE)
    counts = F.one_hot(idx, E).sum(1).float()                             # (N, E)
    f = counts.mean(0) / top_k                                            # fraction of assignments per expert
    P = probs.mean(0)
    lb = E * (f * P).sum()
    z = torch.logsumexp(logits.float(), -1).pow(2).mean()
    if dense:
        ye = pool.expert_fn(x.unsqueeze(0).expand(E, N, d))                # (E, N, d)
        w = torch.zeros(N, E, dtype=x.dtype, device=x.device).scatter(1, idx, gate_v)
        y = torch.einsum("ne,end->nd", w, ye)
        return y, dict(lb=lb, z=z, dropped=torch.zeros((), device=x.device), load=f, idx=idx)
    if capacity_factor is None:                                           # evaluation: exact (no drops) at the observed max load
        C = max(1, int(F.one_hot(idx.reshape(-1), E).sum(0).max().item()))
    else:
        C = max(1, min(N, int(math.ceil(capacity_factor * N * top_k / E))))
    flat_e = idx.reshape(-1)                                              # (A,) expert of each assignment, A = N k
    flat_n = torch.arange(N, device=x.device).repeat_interleave(top_k)    # (A,) row of each assignment
    flat_g = gate_v.reshape(-1)
    oh = F.one_hot(flat_e, E)                                             # (A, E)
    pos = (oh.cumsum(0) * oh).sum(-1) - 1                                 # position inside its expert's buffer
    keep = pos < C
    slot = torch.where(keep, flat_e * C + pos, torch.full_like(pos, E * C))   # E*C = trash slot
    src = torch.full((E * C + 1,), N, dtype=torch.long, device=x.device)       # N = the zero row
    src = src.scatter(0, slot, flat_n)
    x_pad = torch.cat([x, x.new_zeros(1, d)], 0)
    xe = x_pad[src[:E * C]].view(E, C, d)
    ye = pool.expert_fn(xe).reshape(E * C, d)
    ye = torch.cat([ye, ye.new_zeros(1, d)], 0)
    contrib = ye[slot] * (flat_g * keep.to(flat_g.dtype)).unsqueeze(-1)  # dropped assignments contribute 0
    y = x.new_zeros(N, d).index_add(0, flat_n, contrib)
    return y, dict(lb=lb, z=z, dropped=1 - keep.float().mean(), load=f, idx=idx)


def gr_read(R, X, S, cache, key):
    """GRRead over [X~, S] with the X~ branch cached: X~ is constant within a window, so its gated term
    sigma(U V rms(X~) + b) * rms(X~) * (1 + gamma) is computed once per window per read module (same arithmetic)."""
    tX = cache.get(key) if cache is not None else None
    if tX is None:
        nX = rms(X)
        tX = torch.sigmoid(R.U[0](R.V[0](nX))) * nX * (1 + R.gamma[0])
        if cache is not None:
            cache[key] = tX
    nS = rms(S)
    return (tX + torch.sigmoid(R.U[1](R.V[1](nS))) * nS * (1 + R.gamma[1])) / 2


class CoreLayer(nn.Module):
    def __init__(self, cfg, pool):
        super().__init__()
        d = cfg["d"]
        self.read_attn = GRRead(d, 2, cfg["gr_rank"], cfg["gr_read_bias"])
        self.read_ffn = GRRead(d, 2, cfg["gr_rank"], cfg["gr_read_bias"])
        self.write_attn, self.write_ffn = GRWrite(d, 1), GRWrite(d, 1)
        self.attn = CoreAttention(d, cfg["H"], cfg["Hk"], cfg["dh"])
        self.router = nn.Linear(d, cfg["E"], bias=False)
        nn.init.normal_(self.router.weight, std=0.02)                     # small random: not tie-broken uniform top-2
        self.pool = [pool]                                                 # shared, not registered twice
        self.top_k, self.cf = cfg["top_k"], cfg["capacity_factor"]

    def forward(self, X, S, mask, dense=False, cache=None):
        x = gr_read(self.read_attn, X, S, cache, (id(self), 0))
        S = S + self.write_attn(x) * self.attn(x, mask)
        x = gr_read(self.read_ffn, X, S, cache, (id(self), 1))
        B, M, d = x.shape
        xf = x.reshape(B * M, d)
        cf = self.cf if self.training else None                            # evaluation never drops (exact, doc-isolated)
        y, aux = moe_dispatch(xf, self.router(xf), self.pool[0], self.top_k, cf, dense=dense)
        return S + self.write_ffn(x) * y.view(B, M, d), aux


class ConceptCoreD6(nn.Module):
    def __init__(self, dc=384, d=256, n_layers=4, H=4, Hk=2, dh=64, E=16, d_expert=512, top_k=2, gr_rank=32,
                 gr_read_bias=2.0, capacity_factor=1.25, S=48, N=128):
        super().__init__()
        self.cfg = dict(dc=dc, d=d, n_layers=n_layers, H=H, Hk=Hk, dh=dh, E=E, d_expert=d_expert, top_k=top_k,
                        gr_rank=gr_rank, gr_read_bias=gr_read_bias, capacity_factor=capacity_factor)
        self.W_in = nn.Linear(dc, d, bias=False)
        self.W_s = nn.Linear(dc, d, bias=False)
        self.pool = ExpertPool(d, d_expert, E)
        self.layers = nn.ModuleList([CoreLayer(self.cfg, self.pool) for _ in range(n_layers)])
        self.carry_norm = ZCRMSNorm(d)                                     # LN_h: bounded carry, own gain
        self.out_norm = ZCRMSNorm(d)                                       # separately normed readout
        self.vocab = ConceptVocab(dc, S, N)
        self.Wq = nn.Linear(d, dc, bias=False)
        self.log_tau = nn.Parameter(torch.full((S,), math.log(2 * dc // S)))
        self.dense_moe = False                                             # exact-test switch only

    # ---------------------------------------------------------------- pieces used by the trainer
    def init_state(self, c_hat):
        return self.W_in(c_hat), self.W_s(c_hat)

    def one_pass(self, X, S, mask, first=False, cache=None):
        """cache: a dict shared by the passes that read the SAME X~ tensor (one window / one evaluation forward)."""
        if not first:
            S = self.carry_norm(S)
        auxs = []
        for layer in self.layers:
            S, aux = layer(X, S, mask, dense=self.dense_moe, cache=cache)
            auxs.append(aux)
        return S, auxs

    def readout(self, S):
        return self.out_norm(S)

    def head(self, u, pred=True):
        e = self.vocab.e.detach(); V = self.vocab
        q = self.Wq(u).unflatten(-1, (V.S, V.ds))
        # ||q - e||^2 = ||q||^2 - 2 q.e + ||e||^2 (same value; avoids the (B, M, S, N, ds) difference tensor per pass)
        d2 = q.pow(2).sum(-1, keepdim=True) - 2 * torch.einsum("...sd,snd->...sn", q, e) + e.pow(2).sum(-1)
        logits = -d2 / self.log_tau.exp()[:, None]
        if not pred:                                                      # per-pass NCP needs only the logits
            return logits, None
        c_pred = torch.einsum("...sn,snd->...sd", logits.softmax(-1), e).flatten(-2)
        return logits, c_pred

    @staticmethod
    def mask(cdoc):
        return build_masks(cdoc, 1)[1]

    # ---------------------------------------------------------------- evaluation forward (R passes)
    def forward(self, c_hat, cdoc, R, return_passes=False):
        mask = self.mask(cdoc)
        X, S = self.init_state(c_hat)
        us, auxs, cache = [], [], {}
        for r in range(R):
            S, aux = self.one_pass(X, S, mask, first=(r == 0), cache=cache)
            if return_passes:
                us.append(self.readout(S)); auxs.append(aux)
        u = self.readout(S)
        logits, c_pred = self.head(u)
        if return_passes:
            return u, logits, c_pred, dict(us=us, auxs=auxs)
        return u, logits, c_pred

    def ncp_ce(self, logits, targets, cdoc):
        """Mean over valid chunk positions of the per-segment CE of the NEXT chunk's codes (targets = assign(c_hat))."""
        valid = (cdoc[:, 1:] == cdoc[:, :-1]).to(logits.dtype)
        ce = F.cross_entropy(logits[:, :-1].flatten(0, 2), targets[:, 1:].flatten(), reduction="none").view(targets[:, 1:].shape)
        return (ce.mean(-1) * valid).sum() / valid.sum().clamp_min(1)

    def losses(self, c_hat, logits, cdoc):
        """v1-compatible (ncp, vq) for evaluate.py: NCP from the given logits, VQ on sg(c_hat)."""
        vq, n = self.vocab.vq_loss(c_hat)
        return self.ncp_ce(logits, n, cdoc), vq
