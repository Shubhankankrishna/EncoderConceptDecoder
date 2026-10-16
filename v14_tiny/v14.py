"""V14's new parts (v14_Architecture.md §3-§8), written with static shapes so the compiled step never syncs:
the EMA twin, the target view tau = [a; b], the per-family residue maps P_f, the PQ codebook over tau with the stricter
dead-code rule, document-centred SIGReg (plus the uncentred test on b, on by H2b), and the horizon loss.

Shapes: B rows x T tokens, G = T / k groups; gdoc (B, G) is each group's document id (0 = padding, 1.. per row).
"""
import copy
import math

import torch
import torch.nn.functional as F

from model import rope_tables

FAMILIES = ('grid', 'prog', 'state', 'plan', 'text')
HW = (0.4, 0.3, 0.2, 0.1)                       # horizon weights w_k (§6.2)
T_FREEZE = 0.75                                 # twin, normaliser, codebook and P_f freeze at 75% of training time
MU0 = 0.99                                      # twin momentum at t = 0 (a 100-step lag)


# ------------------------------------------------------------------------------------------------ the twin
TWIN_KEEP = ('emb', 'e1', 'engram', 'e2', 's0', 'e_r', 'e2_exit', 'e3', 'exit_L', 'exit_G', 't_loc', 't_glob')


def make_twin(model):
    """EMA twin of every module on the path from the ids to the target view (§3.3); the Engram table is shared."""
    twin = copy.deepcopy(model)
    for name in list(twin._modules):
        if name not in TWIN_KEEP: delattr(twin, name)
    for name in list(twin._parameters):
        if name not in TWIN_KEEP: delattr(twin, name)
    twin.engram.table = model.engram.table                                              # shared, not copied
    for n, p in twin.named_parameters():
        if not n.startswith('engram.table'): p.requires_grad_(False)
    twin.ckpt = False
    return twin


def twin_pairs(twin, model):
    on = dict(model.named_parameters())
    tw, ol = [], []
    for n, p in twin.named_parameters():
        if n.startswith('engram.table'): continue
        tw.append(p); ol.append(on[n])
    return tw, ol


def momentum(frac):
    """mu(t) = 1 - (1 - mu0) * (1 + cos(pi t / t_freeze)) / 2 before t_freeze, 1 after (§3.4)."""
    if frac >= T_FREEZE: return 1.0
    return 1.0 - (1.0 - MU0) * 0.5 * (1.0 + math.cos(math.pi * frac / T_FREEZE))


@torch.no_grad()
def ema_update(tw, ol, mu):
    if mu >= 1.0: return
    torch._foreach_lerp_(tw, ol, 1.0 - mu)


# ------------------------------------------------------------------------------------------------ the target view
def group_feats(x, L_enc, G_enc, k):
    B, T, d = x.shape; G = T // k
    mp = lambda t: t.reshape(B, G, k, t.shape[-1]).mean(2)
    return torch.cat([mp(x), mp(L_enc)], -1), mp(G_enc)                                  # ell (B, G, 2d), gbar (B, G, d)


def has_prev(gdoc):
    """True where group g has a previous group in the same document (b_g is defined; first groups are masked)."""
    prev = F.pad(gdoc, (1, 0), value=-1)[:, :-1]                                          # integer pad, no bool ops
    return (gdoc == prev) & (gdoc > 0)


def target_view(mod, ell, gbar, gdoc, P_rows):
    """a = T_loc ell; b = T_glob(dG - dl P_f) (§4.3). P_rows (B, 2d, d): each row's family map. Returns a, b, valid,
    dl, dG (dl and dG are the residue-map statistics' inputs)."""
    dl = ell - F.pad(ell, (0, 0, 1, 0))[:, :-1]
    dG = gbar - F.pad(gbar, (0, 0, 1, 0))[:, :-1]
    a = mod.t_loc(ell)
    b = mod.t_glob(dG - torch.einsum('bgl,bld->bgd', dl, P_rows))
    return a, b, has_prev(gdoc), dl, dG


def twin_targets(twin, ids, doc, eidx, P_rows):
    """The twin's tau = [a; b] (no gradient: call under torch.no_grad())."""
    ctx = twin_context(twin, doc)
    x, L, Gm = twin.encode(ids, eidx, ctx)
    ell, gbar = group_feats(x, L, Gm, twin.c.k)
    a, b, valid, dl, dG = target_view(twin, ell, gbar, doc[:, ::twin.c.k], P_rows)
    return torch.cat([a, b], -1), valid, dl, dG


def twin_context(twin, doc):
    """The encoder's masks only (the twin has no concept module or decoder)."""
    c = twin.c; B, T = doc.shape; dev = doc.device
    t = torch.arange(T, device=dev)
    ctx = dict(doc=doc)
    ctx['full_mask'] = ((t[None, :] <= t[:, None])[None] & (doc[:, :, None] == doc[:, None, :]))[:, None]
    dt = twin.emb.weight.dtype
    ctx['rope_tok'] = tuple(z.to(dt) for z in rope_tables(t, c.d // c.H))
    ctx['rope_tok_part'] = tuple(z.to(dt) for z in rope_tables(t, int(c.rope_frac_S * c.d // c.H)))
    return ctx


def online_view(model, x, L_enc, G_enc, gdoc, P_rows):
    """The online target view for the barrier terms: a from sg[x], sg[L_enc] (only T_loc trains), b with G's gradient
    and the local change stopped (§4.3)."""
    ell, gbar = group_feats(x.detach(), L_enc.detach(), G_enc, model.c.k)
    a, b, valid, _, _ = target_view(model, ell, gbar, gdoc, P_rows)
    return a, b, valid


# ------------------------------------------------------------------------------------------------ SIGReg (§8.3)
KNOTS = 17


def sigreg_consts(device, dtype=torch.float32):
    t = torch.linspace(-5.0, 5.0, KNOTS, device=device, dtype=dtype)
    h = 10.0 / (KNOTS - 1)
    q = torch.full((KNOTS,), h, device=device, dtype=dtype); q[0] = q[-1] = h / 2           # trapezoid weights
    return t, torch.exp(-t ** 2 / 2), q * torch.exp(-t ** 2)                              # knots, phi, weights (sigma 1)


def directions(step, dim, M=256, device='cpu'):
    g = torch.Generator().manual_seed(1_000_003 + int(step))
    w = torch.randn(2, dim, M, generator=g)
    return (w / w.norm(dim=1, keepdim=True)).to(device)                                   # (2, dim, M): for a and b


def ep(z, keep, om, t, phi, q):
    """Mean over the M directions of the Epps-Pulley statistic of the kept rows of z (N, D) (no N factor, §8.3)."""
    s = z @ om                                                                            # (N, M)
    wk = keep.to(z.dtype)
    n = wk.sum().clamp(min=1.0)
    ts = s[..., None] * t                                                                 # (N, M, K)
    c = (torch.cos(ts) * wk[:, None, None]).sum(0) / n
    si = (torch.sin(ts) * wk[:, None, None]).sum(0) / n
    return (((c - phi) ** 2 + si ** 2) @ q).mean()


def doc_centre(z, gdoc, valid, min_groups=4):
    """Each document's valid groups centred on their own mean; keep = valid groups of documents with >= 4 of them."""
    B, G, D = z.shape
    w = valid.to(z.dtype)
    idx = gdoc.long()
    sums = z.new_zeros(B, G + 1, D).scatter_add(1, idx[..., None].expand(B, G, D), z * w[..., None])
    cnt = z.new_zeros(B, G + 1).scatter_add(1, idx, w)
    mean = sums / cnt.clamp(min=1.0)[..., None]
    zc = z - torch.gather(mean, 1, idx[..., None].expand(B, G, D))
    return zc, valid & (torch.gather(cnt, 1, idx) >= min_groups)


def sigreg_terms(a, b, gdoc, valid, om, consts):
    """L_SIG,a and L_SIG,b (document-centred) and L_SIG,b,uncentred, each a mean over directions."""
    t, phi, q = consts
    D = a.shape[-1]
    ac, keep = doc_centre(a, gdoc, valid); bc, _ = doc_centre(b, gdoc, valid)
    keep = keep.reshape(-1); v = valid.reshape(-1)
    return (ep(ac.reshape(-1, D), keep, om[0], t, phi, q), ep(bc.reshape(-1, D), keep, om[1], t, phi, q),
            ep(b.reshape(-1, D), v, om[1], t, phi, q))


# ------------------------------------------------------------------------------------------------ the horizon loss
def horizon_loss(hl, codes, gdoc, valid_t, hw=HW):
    """L_NCP = sum_k w_k mean over valid (g, g+k) of sum_s CE (§6.2). Returns the loss and a (K, 7) stats tensor:
    pairs, head accuracy on halves a / b, copy-current accuracy on halves a / b, pairs with tau_g defined, CE."""
    B, G, K, S, N = hl.shape
    tot = hl.new_zeros(()); rows = []
    for k in range(1, K + 1):
        lg = hl[:, :G - k, k - 1]; tg = codes[:, k:]
        pair = ((gdoc[:, :G - k] == gdoc[:, k:]) & (gdoc[:, :G - k] > 0) & valid_t[:, k:]).to(hl.dtype)
        ce = F.cross_entropy(lg.reshape(-1, N), tg.reshape(-1), reduction='none').view(B, G - k, S)
        npair = pair.sum().clamp(min=1.0)
        lk = (ce.sum(-1) * pair).sum() / npair
        tot = tot + hw[k - 1] * lk
        with torch.no_grad():
            hit = (lg.argmax(-1) == tg).to(hl.dtype)
            pc = pair * valid_t[:, :G - k].to(hl.dtype)                                    # copy-current needs tau_g
            cc = (codes[:, :G - k] == tg).to(hl.dtype)
            npc = pc.sum().clamp(min=1.0)
            rows.append(torch.stack([pair.sum(), (hit[..., :S // 2].mean(-1) * pair).sum() / npair,
                                     (hit[..., S // 2:].mean(-1) * pair).sum() / npair,
                                     (cc[..., :S // 2].mean(-1) * pc).sum() / npc,
                                     (cc[..., S // 2:].mean(-1) * pc).sum() / npc, pc.sum(), lk.detach()]))
    return tot, torch.stack(rows)


def same_argmax_fraction(hl, gdoc):
    """Fraction of real groups whose argmax tuple is identical at every horizon (§6.2's read)."""
    am = hl.argmax(-1)                                                                    # (B, G, K, S)
    same = (am == am[:, :, :1]).all(-1).all(-1).to(hl.dtype)
    real = (gdoc > 0).to(hl.dtype)
    return (same * real).sum() / real.sum().clamp(min=1.0)


# ------------------------------------------------------------------------------------------------ the codebook (§6.1)
class Codebook(torch.nn.Module):
    """PQ over the normalised tau: S segments x N words, EMA k-means (0.99), running normaliser (0.99), dead words
    (usage below 10% of uniform) reseeded at the worst-fitted valid targets, distinct ones, in order of error.
    Masked updates (weights), so shapes are fixed. `frozen` stops every update (set at t_freeze)."""
    def __init__(self, d=128, S=4, N=32, decay=0.99, nd=0.99, dead_frac=0.10):
        super().__init__()
        self.S, self.N, self.ds, self.decay, self.nd, self.dead_frac = S, N, d // S, decay, nd, dead_frac
        self.register_buffer('code', torch.randn(S, N, d // S))
        self.register_buffer('count', torch.ones(S, N))
        self.register_buffer('mu', torch.zeros(d)); self.register_buffer('var', torch.ones(d))
        self.register_buffer('rows_ema', torch.ones(()))
        self.register_buffer('initialised', torch.zeros((), dtype=torch.bool))
        self.register_buffer('frozen', torch.zeros((), dtype=torch.bool))
        self.register_buffer('resets', torch.zeros(()))
        self._init_py = None; self._frozen_py = None

    def freeze(self):
        self.frozen.fill_(True); self._frozen_py = True

    def normalise(self, x):
        return (x - self.mu) / self.var.clamp(min=1e-4).sqrt()

    def assign(self, z):
        zz = z.reshape(-1, self.S, self.ds)
        d2 = (zz[:, :, None] - self.code[None]).pow(2).sum(-1)
        return d2.argmin(-1).view(*z.shape[:-1], self.S)

    @torch.no_grad()
    def update(self, x, w):
        """x (n, d) candidate targets, w (n,) 1 for valid targets, 0 otherwise."""
        if self._init_py is None: self._init_py = bool(self.initialised)
        if self._frozen_py is None: self._frozen_py = bool(self.frozen)
        if self._frozen_py: return
        n = w.sum().clamp(min=1.0)
        if not self._init_py:
            mu = (x * w[:, None]).sum(0) / n
            self.mu.copy_(mu); self.var.copy_(((x - mu) ** 2 * w[:, None]).sum(0) / n + 1e-4)
            z = self.normalise(x).reshape(-1, self.S, self.ds)
            score = torch.rand(x.shape[0], device=x.device) + w * 2                       # valid rows first
            pick = score.topk(self.S * self.N).indices.view(self.S, self.N)
            self.code.copy_(z[pick, torch.arange(self.S, device=x.device)[:, None]])
            self.rows_ema.copy_(n); self.count.copy_((n / self.N).expand_as(self.count))      # uniform usage at init
            self.initialised.fill_(True); self._init_py = True
            return
        self.mu.mul_(self.nd).add_((x * w[:, None]).sum(0) / n, alpha=1 - self.nd)
        self.var.mul_(self.nd).add_((((x - self.mu) ** 2) * w[:, None]).sum(0) / n, alpha=1 - self.nd)
        z = self.normalise(x).reshape(-1, self.S, self.ds)
        idx = self.assign(z.reshape(x.shape[0], -1))                                       # (n, S)
        oh = F.one_hot(idx, self.N).to(x.dtype) * w[:, None, None]
        cnt = oh.sum(0)
        sums = torch.einsum('msn,msd->snd', oh, z)
        self.count.mul_(self.decay).add_(cnt, alpha=1 - self.decay)
        self.rows_ema.mul_(self.decay).add_(n * (1 - self.decay))
        upd = sums / cnt.clamp(min=1.0)[..., None]
        self.code.copy_(torch.where((cnt > 0)[..., None], self.decay * self.code + (1 - self.decay) * upd, self.code))
        dead = self.count < (self.dead_frac * self.rows_ema / self.N)                        # (S, N)
        err = (z - self.code[torch.arange(self.S, device=x.device)[None], idx]).pow(2).sum(-1)   # (n, S)
        err = torch.where(w[:, None] > 0, err, torch.full_like(err, -1.0))
        worst = err.topk(self.N, dim=0).indices                                            # (N, S): worst first
        rank = (dead.long().cumsum(-1) - 1).clamp(min=0)                                   # i-th dead word -> i-th worst
        src = worst.gather(0, rank.t()).t()                                                # (S, N) row indices
        cand = z[src, torch.arange(self.S, device=x.device)[:, None]]                     # (S, N, ds)
        self.code.copy_(torch.where(dead[..., None], cand, self.code))
        self.count.copy_(torch.where(dead, cnt.mean(-1, keepdim=True).expand_as(cnt), self.count))
        self.resets.add_(dead.sum())

    def usage_perplexity(self):
        p = self.count / self.count.sum(-1, keepdim=True)
        return torch.exp(-(p * p.clamp(min=1e-12).log()).sum(-1))                          # (S,)


def batch_perplexity(codes, valid, N):
    """Code-usage perplexity per segment of one batch's valid targets (batches are single-family: §6.3's read)."""
    oh = F.one_hot(codes, N).to(torch.float32) * valid[..., None, None].to(torch.float32)
    p = oh.sum((0, 1)); p = p / p.sum(-1, keepdim=True).clamp(min=1.0)
    return torch.exp(-(p * p.clamp(min=1e-12).log()).sum(-1))                              # (S,)


# ------------------------------------------------------------------------------------------------ residue maps (§4.3)
class ResidueMaps(torch.nn.Module):
    """P_f = (A_f + rho_f I)^-1 B_f per family from EMAs (0.99 per update of that family) of dl^T dl and dl^T dG over
    the twin's valid groups, plus a pooled map kept for §6.3's read. P_f = 0 before step 200 and until the family has
    20 valid pairs per dimension of ell; re-solved every 100 steps in float64; frozen with the codebook."""
    def __init__(self, nf=5, dl=256, dg=128, decay=0.99):
        super().__init__()
        self.nf, self.dl, self.dg, self.decay = nf, dl, dg, decay
        self.register_buffer('A', torch.zeros(nf + 1, dl, dl)); self.register_buffer('Bm', torch.zeros(nf + 1, dl, dg))
        self.register_buffer('cnt', torch.zeros(nf + 1)); self.register_buffer('P', torch.zeros(nf + 1, dl, dg))
        self.register_buffer('frozen', torch.zeros((), dtype=torch.bool))
        self.min_pairs = 20 * dl; self._frozen_py = None

    def freeze(self):
        self.frozen.fill_(True); self._frozen_py = True

    def is_frozen(self):
        if self._frozen_py is None: self._frozen_py = bool(self.frozen)
        return self._frozen_py

    @torch.no_grad()
    def update(self, f, dl, dG, valid):
        if self.is_frozen(): return
        w = valid.reshape(-1).to(dl.dtype); n = w.sum().clamp(min=1.0)
        X = dl.reshape(-1, self.dl); Y = dG.reshape(-1, self.dg)
        Xw = X * w[:, None]
        Ab = Xw.t() @ X / n; Bb = Xw.t() @ Y / n
        for i in (f, self.nf):                                                            # family f, then pooled
            self.A[i].mul_(self.decay).add_(Ab, alpha=1 - self.decay)
            self.Bm[i].mul_(self.decay).add_(Bb, alpha=1 - self.decay)
            self.cnt[i].add_(w.sum())

    @torch.no_grad()
    def solve(self, step):
        if self.is_frozen(): return None
        cnt = self.cnt.cpu()
        out = torch.zeros(self.nf + 1, self.dl, self.dg, dtype=torch.float64)
        if step >= 200:
            A = self.A.cpu().double(); Bm = self.Bm.cpu().double()
            for i in range(self.nf + 1):
                if cnt[i] >= self.min_pairs:
                    rho = 1e-3 * torch.trace(A[i]) / self.dl
                    out[i] = torch.linalg.solve(A[i] + rho * torch.eye(self.dl, dtype=torch.float64), Bm[i])
        self.P.copy_(out.to(self.P.dtype))
        return [bool(cnt[i] >= self.min_pairs and step >= 200) for i in range(self.nf + 1)]

    def rows(self, fam_idx):
        """(B, dl, dg) maps for a (B,) tensor of family indices."""
        return self.P[fam_idx]
