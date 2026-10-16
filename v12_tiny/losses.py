"""V12 objectives.

NTP           weighted next-token CE (actions 1, positions lam_obs, END 1; header unscored).
NCP-CE        next-concept prediction as cross-entropy over *discrete segment codes* (FIRST_PRINCIPLES P2).
              The target tau_{g+1} is a stopped, EMA-normalised group summary, quantised by a product codebook
              (S segments x N words, EMA k-means with dead-code reset; no gradient). A head on u_g predicts,
              per segment, logits over the N words; the loss is sum_s CE(logits_s, idx_s(tau_{g+1})). The joint
              over segments is the product of per-segment categoricals (factorised), which is mode-preserving
              per segment, unlike MSE regression to the conditional mean.
              target='enc': next group's pooled encoder state (NCP-faithful; the encoder is upstream of the
                            concept path, so the target cannot be made trivially predictable through it).
              target='dec': HiLP-style upper-level target, the next group's mean *decoder* state (stopped).
                            Shortcut monitor (advisor #1): the decoder reads concepts, so `dec` targets are also
                            computed with the concept route zeroed and the FVU gap is logged (evaluate.py).
NextLat       p_psi([d_t ; E x_{t+1}]) -> d^_{t+1} (3-layer GELU MLP, residual), SmoothL1 to sg[d_{t+1}] after RMS
              normalisation + KL(softmax(sg logits_{t+1}) || softmax(head_sg(d^))) through a frozen view of the
              output head, d = 1 (NextLat §3.2, Algorithm 1). Applied to the token-rate final residual.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import RMSNorm


def ntp(out, b):
    lp = F.cross_entropy(out['logits'].flatten(0, 1).float(), b['targets'].flatten(), reduction='none').view_as(b['weights'])
    w = b['weights']
    act = (lp * b['is_act']).sum() / b['is_act'].sum().clamp(min=1)
    return (lp * w).sum() / w.sum(), act, lp


class PQCodebook(nn.Module):
    """Product quantiser over an EMA-normalised target; updated by EMA k-means, never by gradient."""
    def __init__(self, d, segments=4, words=32, decay=0.99, norm_decay=0.99):
        super().__init__()
        assert d % segments == 0
        self.S, self.N, self.ds, self.decay, self.nd = segments, words, d // segments, decay, norm_decay
        self.register_buffer('code', torch.randn(segments, words, d // segments))
        self.register_buffer('count', torch.ones(segments, words))
        self.register_buffer('mu', torch.zeros(d)); self.register_buffer('var', torch.ones(d))
        self.register_buffer('initialised', torch.zeros((), dtype=torch.bool))
        self.register_buffer('rows_ema', torch.ones(()))

    def normalise(self, x):
        return (x - self.mu) / (self.var.clamp(min=1e-4).sqrt())

    @torch.no_grad()
    def assign(self, tau):
        z = tau.reshape(-1, self.S, self.ds)                                       # (M, S, ds)
        d2 = (z[:, :, None] - self.code[None]).pow(2).sum(-1)                      # (M, S, N)
        return d2.argmin(-1)                                                       # (M, S)

    @torch.no_grad()
    def update(self, x):
        """x: raw (unnormalised) target rows (M, d). Sync-free after the first call (no .item()/bool on device)."""
        if x.shape[0] == 0: return
        if not getattr(self, '_init_py', False) and not bool(self.initialised):
            self.mu.copy_(x.mean(0)); self.var.copy_(x.var(0) + 1e-4)
            z = self.normalise(x).reshape(-1, self.S, self.ds)
            pick = torch.randint(0, z.shape[0], (self.S, self.N), device=z.device)
            self.code.copy_(z[pick, torch.arange(self.S, device=z.device)[:, None]])
            self.initialised.fill_(True); self.rows_ema.fill_(float(x.shape[0])); self._init_py = True; return
        self._init_py = True
        self.mu.mul_(self.nd).add_(x.mean(0), alpha=1 - self.nd)
        self.var.mul_(self.nd).add_(((x - self.mu) ** 2).mean(0), alpha=1 - self.nd)
        z = self.normalise(x).reshape(-1, self.S, self.ds)
        idx = self.assign(self.normalise(x))                                       # (M, S)
        onehot = F.one_hot(idx, self.N).float()                                    # (M, S, N)
        cnt = onehot.sum(0)                                                        # (S, N)
        sums = torch.einsum('msn,msd->snd', onehot, z)
        self.count.mul_(self.decay).add_(cnt, alpha=1 - self.decay)
        self.rows_ema.mul_(self.decay).add_(float(z.shape[0]) * (1 - self.decay))   # EMA of rows per step (shared codebook)
        upd = sums / cnt.clamp(min=1)[..., None]
        live = (cnt > 0)[..., None]
        self.code.copy_(torch.where(live, self.decay * self.code + (1 - self.decay) * upd, self.code))
        # dead-code reset (masked, no host sync): words whose EMA usage fell below 2% of uniform get a random row
        dead = self.count < (0.02 * self.rows_ema / self.N)   # both sides are EMAs across steps and families
        rows = torch.randint(0, z.shape[0], (self.S, self.N), device=z.device)
        cand = z[rows, torch.arange(self.S, device=z.device)[:, None]]             # (S, N, ds)
        self.code.copy_(torch.where(dead[..., None], cand, self.code))
        self.count.copy_(torch.where(dead, cnt.mean(), self.count))

    def usage_perplexity(self):
        p = self.count / self.count.sum(-1, keepdim=True)
        return torch.exp(-(p * p.clamp(min=1e-12).log()).sum(-1)).mean()


class NCPHead(nn.Module):
    def __init__(self, d, segments=4, words=32):
        super().__init__()
        self.S, self.N = segments, words
        self.n = RMSNorm(d); self.fc1 = nn.Linear(d, 2 * d); self.fc2 = nn.Linear(2 * d, segments * words)

    def forward(self, u):
        return self.fc2(F.gelu(self.fc1(self.n(u)))).view(*u.shape[:-1], self.S, self.N)


def group_targets(out, k, kind, concept_off_final=None):
    """Raw (unnormalised) per-group target summaries, shape (B, G, d)."""
    if kind == 'enc':
        return out['pooled_raw'].detach()
    src = out['final'] if concept_off_final is None else concept_off_final
    B, T, D = src.shape; G = out['G']
    return src[:, :G * k].reshape(B, G, k, D).mean(2).detach()


def group_valid(b, G, k):
    """Group g+1 is a valid target if it lies entirely inside the trajectory of the episode (after p_0)."""
    L = b['lengths'][:, None]; h = b['header'][:, None]
    g = torch.arange(G, device=L.device)[None]
    start, end = k * g, k * g + k - 1
    return (start >= h) & (end < L)                                                # (B, G) as targets


def ncp_ce(model, out, b, codebook, head, kind='enc'):
    k = model.k
    raw = group_targets(out, k, kind)                                              # (B, G, d)
    G = raw.shape[1]
    if G < 2: return raw.new_zeros(()), {}
    valid_t = group_valid(b, G, k)                                                 # target group valid
    ctx_ok = valid_t[:, 1:]                                                        # predict g+1 from u_g
    tgt = raw[:, 1:][ctx_ok]                                                       # (M, d)
    codebook.update(tgt)
    idx = codebook.assign(codebook.normalise(tgt))                                 # (M, S)
    logits = head(out['u'][:, :-1][ctx_ok])                                        # (M, S, N)
    ce = F.cross_entropy(logits.reshape(-1, codebook.N).float(), idx.reshape(-1), reduction='none').view(-1, codebook.S)
    loss = ce.sum(-1).mean()
    with torch.no_grad():
        acc = (logits.argmax(-1) == idx).float().mean()
    # tensors, not floats: converted only on logging steps (avoids a host sync every step)
    return loss, dict(ncp_ce=loss.detach(), ncp_acc=acc, code_ppl=codebook.usage_perplexity(), ncp_rows=ctx_ok.sum())


class NextLat(nn.Module):
    def __init__(self, d, emb, hidden_mult=2):
        super().__init__()
        self.emb = emb                                                             # shared table, read detached
        self.n = nn.LayerNorm(2 * d)
        self.f = nn.Sequential(nn.Linear(2 * d, hidden_mult * d), nn.GELU(), nn.Linear(hidden_mult * d, hidden_mult * d),
                               nn.GELU(), nn.Linear(hidden_mult * d, d))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)

    def forward(self, h, nxt):
        e = F.embedding(nxt, self.emb.weight.detach())
        return h + self.f(self.n(torch.cat([h, e], -1)))


def rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def nextlat_loss(model, out, b, nl):
    """d = 1. Rows t whose next token x_{t+1} and target state h_{t+1} lie in the scored trajectory."""
    h = out['final']                                                               # (B, T, d), pre-norm residual
    B, T, D = h.shape
    # NextLat §C: the latent loss runs on every real position (belief states form during context, incl. prompts and
    # the pinpad header); only the token-space KL is restricted to scored targets.
    col = torch.arange(T - 1, device=h.device)[None]
    valid_h = col + 1 < b['lengths'][:, None]
    valid = b['weights'][:, :-1] > 0
    pred = nl(h[:, :-1], b['ids'][:, 1:])                                          # predicts h_{t+1}
    tgt = h[:, 1:].detach()
    sl1 = F.smooth_l1_loss(rms(pred), rms(tgt), reduction='none').mean(-1)
    l_h = (sl1 * valid_h).sum() / valid_h.sum().clamp(min=1)
    # frozen view of the output head (final norm gain and tied embedding enter as constants)
    W = model.emb.weight.detach(); g = model.nf.g.detach()
    lt = out['logits'][:, 1:].detach().float()
    ls = ((rms(pred) * g) @ W.t()).float()
    kl = (F.softmax(lt, -1) * (F.log_softmax(lt, -1) - F.log_softmax(ls, -1))).sum(-1)
    l_kl = (kl * valid).sum() / valid.sum().clamp(min=1)
    return l_h, l_kl
