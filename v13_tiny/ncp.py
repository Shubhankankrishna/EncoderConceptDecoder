"""EMA product-quantiser codebook for next-concept prediction (v13_NOTES.md §6.1), copied from
v12_tiny/losses.py `PQCodebook` (measured there, including the dead-code fix that compares two EMAs).
Never trained by gradient; S = 4 segments x N = 32 words over the normalised concept input m."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PQCodebook(nn.Module):
    def __init__(self, d, segments=4, words=32, decay=0.99, norm_decay=0.99):
        super().__init__()
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
        z = tau.reshape(-1, self.S, self.ds)
        d2 = (z[:, :, None] - self.code[None]).pow(2).sum(-1)
        return d2.argmin(-1).view(*tau.shape[:-1], self.S)

    @torch.no_grad()
    def update(self, x):
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
        idx = self.assign(self.normalise(x)).reshape(-1, self.S)
        onehot = F.one_hot(idx, self.N).float()
        cnt = onehot.sum(0)
        sums = torch.einsum('msn,msd->snd', onehot, z)
        self.count.mul_(self.decay).add_(cnt, alpha=1 - self.decay)
        self.rows_ema.mul_(self.decay).add_(float(z.shape[0]) * (1 - self.decay))
        upd = sums / cnt.clamp(min=1)[..., None]
        live = (cnt > 0)[..., None]
        self.code.copy_(torch.where(live, self.decay * self.code + (1 - self.decay) * upd, self.code))
        dead = self.count < (0.02 * self.rows_ema / self.N)
        rows = torch.randint(0, z.shape[0], (self.S, self.N), device=z.device)
        cand = z[rows, torch.arange(self.S, device=z.device)[:, None]]
        self.code.copy_(torch.where(dead[..., None], cand, self.code))
        self.count.copy_(torch.where(dead, cnt.mean(), self.count))

    def usage_perplexity(self):
        p = self.count / self.count.sum(-1, keepdim=True)
        return torch.exp(-(p * p.clamp(min=1e-12).log()).sum(-1)).mean()


def ncp_targets(m, gdoc, codebook, update=True):
    """Valid (g -> g+1) pairs inside one document, the codebook update on their targets, and target codes."""
    with torch.no_grad():
        valid = (gdoc[:, :-1] == gdoc[:, 1:]) & (gdoc[:, :-1] > 0)
        tgt = m[:, 1:].detach()
        if update: codebook.update(tgt[valid])
        idx = codebook.assign(codebook.normalise(tgt))
    return valid, idx


def ncp_loss(ncp_logits, valid, idx):
    B, G1, S = idx.shape
    ce = F.cross_entropy(ncp_logits[:, :-1].reshape(-1, ncp_logits.shape[-1]), idx.reshape(-1), reduction='none')
    ce = ce.view(B, G1, S).sum(-1)
    vf = valid.float()
    loss = (ce * vf).sum() / vf.sum().clamp(min=1)
    with torch.no_grad():
        acc = ((ncp_logits[:, :-1].argmax(-1) == idx).float().mean(-1) * vf).sum() / vf.sum().clamp(min=1)
    return loss, acc
