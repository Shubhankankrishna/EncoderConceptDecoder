"""Frozen NCP skeleton + replacement core: forward modes shared by Phases B-E."""
import torch
import torch.nn.functional as F

import data_v5 as dv
from model_v5 import NCPModel, GramCore, token_nll, K, D, apply_plan_mode
from common_v5 import DEV


def load_ncp(path, dev=DEV):
    ck = torch.load(path, map_location="cpu")
    m = NCPModel(dv.VOCAB)
    m.load_state_dict(ck["model"])
    m.to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def window_tensors(win):
    nW = int(win.max()) + 1
    oh = F.one_hot(win, nW).float()                 # [B,M,nW]
    first = (oh.cumsum(1) == 1).float() * oh
    return oh, first


def chunk_sums(nll, M):
    """nll [B,T-1] over input positions -> [B,M]: chunk j collects positions 4j+3..4j+6 (those reading chat_j)."""
    B, Tm1 = nll.shape
    pad = F.pad(nll, (0, 4 * M + 3 - Tm1))[:, 3:3 + 4 * M]
    return pad.view(B, M, 4).sum(-1)


class System:
    """teacher: frozen NCPModel.  core: GramCore or None (None = original concept module)."""

    def __init__(self, teacher, core=None):
        self.t = teacher
        self.core = core

    @torch.no_grad()
    def context(self, x):
        H = self.t.encode(x)
        C = self.t.pool(H)
        return H, C

    def z_prior_schedule(self, e, starts, eta_z, mode="sample"):
        """Prior Z at window starts (from e at the start), forward-filled over each window.
        mode 'sample' uses eta_z [B,M,dz]; 'mean' uses the prior mean."""
        mu, ls = self.core.z_prior(e, starts)
        z_start = mu if mode == "mean" else mu + ls.exp() * eta_z
        win = starts.long().cumsum(1) - 1
        oh, first = window_tensors(win)
        z_win = torch.einsum("bmw,bmd->bwd", first, z_start)
        return torch.einsum("bmw,bwd->bmd", oh, z_win), (mu, ls)

    def plans(self, H, C, Z_pos=None, eps_mode="prior", y=None, noise=None, lam=1.0):
        """Returns chat, A (decoder-facing interface) and extras.  lam blends old (0) and new (1) outputs."""
        if self.core is None or lam == 0.0:
            u = self.t.concept(C)
            _, pi, chat = self.t.pq(u)
            return chat, self.t.port(u), dict(kl_eps=None)
        u_new, A_new, kl = self.core(C, Z_pos, eps_mode=eps_mode, y=y, noise=noise)
        _, pi, chat_new = self.t.pq(u_new)
        if lam < 1.0:
            with torch.no_grad():
                u_old = self.t.concept(C)
                _, _, chat_old = self.t.pq(u_old)
                A_old = self.t.port(u_old)
            chat_new = (1 - lam) * chat_old + lam * chat_new
            A_new = (1 - lam) * A_old + lam * A_new
        return chat_new, A_new, dict(kl_eps=kl, u=u_new, pi=pi)

    def logits(self, H, chat, A):
        return self.t.decode(H, chat, A)
