"""Stand-in for the concept module (v2 as tested in the concept test bench, K = 1, i.e. without the XM latent).

ONE shared layer looped R times: reads {X, S}, writes S; X = c_hat (re-read every pass, gradient flows), S0 = 0;
R ~ U{1..4} in training.  Layer = the encoder's FULL layer at concept rate (gated attention, NoPE + per-head
temperature, q/k/v convs) wrapped in the gated-residual grammar.  Output u = zcRMSNorm(S).
NCP: product-quantised vocabulary (k-means++ init, dead-code restarts, trained by L_VQ only), distance head,
cross-entropy on the next concept's code indices.  c_pred = sum_n pi_n sg(e_n).
Only its interface matters for the decoder: (u, c_pred) per chunk, post-loop.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hrm_encoder import GRRead, GRWrite, GatedAttention, SwiGLU, ZCRMSNorm, build_masks


class LoopLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.read_attn, self.read_ffn = GRRead(d, 2, cfg.gr_rank, cfg.gr_read_bias), GRRead(d, 2, cfg.gr_rank, cfg.gr_read_bias)
        self.write_attn, self.write_ffn = GRWrite(d, 1), GRWrite(d, 1)
        self.attn, self.ffn = GatedAttention(cfg, "F"), SwiGLU(d, cfg.d_ff)

    def forward(self, X, S, cdoc, mask):
        x = self.read_attn([X, S])
        S = S + self.write_attn(x) * self.attn(x, cdoc, mask, None)
        x = self.read_ffn([X, S])
        return S + self.write_ffn(x) * self.ffn(x)


class ConceptVocab(nn.Module):
    def __init__(self, dc, S, N, restart_every=100):
        super().__init__()
        self.S, self.N, self.ds, self.every = S, N, dc // S, restart_every
        self.e = nn.Parameter(torch.randn(S, N, dc // S))
        self.register_buffer("hits", torch.zeros(S, N))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def assign(self, c):
        cs = c.unflatten(-1, (self.S, self.ds))
        return (cs.unsqueeze(-2) - self.e.detach()).pow(2).sum(-1).argmin(-1)

    @torch.no_grad()
    def init_from(self, c):
        x = c.reshape(-1, self.S, self.ds)
        for j in range(self.S):
            xs = x[:, j]; pick = [xs[torch.randint(len(xs), (1,))]]
            for _ in range(self.N - 1):
                dd = torch.cdist(xs, torch.cat(pick)).min(1).values.pow(2)
                pick.append(xs[torch.multinomial(dd / dd.sum() if dd.sum() > 0 else torch.ones_like(dd), 1)])
            self.e[j] = torch.cat(pick)

    def vq_loss(self, c):
        c = c.detach(); cs = c.unflatten(-1, (self.S, self.ds)); n = self.assign(c)
        if self.training:
            with torch.no_grad():
                self.hits += F.one_hot(n.reshape(-1, self.S), self.N).sum(0); self.step += 1
        q = self.e[torch.arange(self.S), n]
        return (cs - q).pow(2).sum(-1).mean(), n

    @torch.no_grad()
    def restart_dead(self, c):
        """Call AFTER the optimizer step (it edits e in place): re-seed codes unused since the last restart."""
        if self.step % self.every:
            return
        x = c.detach().reshape(-1, self.S, self.ds)
        for j in range(self.S):
            dead = self.hits[j] == 0
            if dead.any():
                self.e[j, dead] = x[torch.randint(len(x), (int(dead.sum()),)), j]
        self.hits.zero_()


class ConceptCoreV2(nn.Module):
    def __init__(self, cfg, S=16, N=64):
        super().__init__()
        dc = cfg.d_model
        self.layer = LoopLayer(cfg)
        self.out_norm = ZCRMSNorm(dc)
        self.vocab = ConceptVocab(dc, S, N)
        self.Wq = nn.Linear(dc, dc, bias=False)
        self.log_tau = nn.Parameter(torch.full((S,), math.log(2 * dc // S)))

    def head(self, u):
        e = self.vocab.e.detach(); V = self.vocab
        q = self.Wq(u).unflatten(-1, (V.S, V.ds))
        logits = -(q.unsqueeze(-2) - e).pow(2).sum(-1) / self.log_tau.exp()[:, None]
        c_pred = torch.einsum("...sn,snd->...sd", logits.softmax(-1), e).flatten(-2)
        return logits, c_pred

    def forward(self, c_hat, cdoc, R):
        _, mask = build_masks(cdoc, 1)
        X, S = c_hat, torch.zeros_like(c_hat)
        for _ in range(R):
            S = self.layer(X, S, cdoc, mask)
        u = self.out_norm(S)
        logits, c_pred = self.head(u)
        return u, logits, c_pred

    def losses(self, c_hat, logits, cdoc):
        vq, n = self.vocab.vq_loss(c_hat)
        valid = (cdoc[:, 1:] == cdoc[:, :-1]).to(logits.dtype)
        ce = F.cross_entropy(logits[:, :-1].flatten(0, 2), n[:, 1:].flatten(), reduction="none").view(n[:, 1:].shape)
        return (ce.mean(-1) * valid).sum() / valid.sum().clamp_min(1), vq
