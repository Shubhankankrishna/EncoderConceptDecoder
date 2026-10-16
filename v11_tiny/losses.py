"""V11 objectives (v11.md §4) for the K = 1 arms, plus NTP for ORD.

  L = L_NTP + ramp * ( lambda_c (L_v1/3 + 2 L_v2/3) + lambda_vq L_vq + lambda_h L_h + lambda_KL L_KL ) + L_probe

  L_v1   Gaussian NLL per dimension of the visit-1 forecast vs tau_{g+1}, all eligible groups
  L_v2   same for visit 2 (K = 1 special case of the mixture) on S sampled boundaries per document,
         from stopped committed inputs (v11_tiny.model.V11Tiny.visit2_from_stopped)
  L_vq   NCP Eq. 21 codebook fitting to the normalized target (codebook only)
  L_h    SmoothL1 between RMS-normalised bridge state and sg[pre-outn decoder state]
  L_KL   KL(sg teacher logits || student logits through a detached view of outn + tied embedding)
  L_probe detached forecast probe CE (V10.2 future_ce weighting); trains the probe only

NOAUX computes L_NTP and L_probe only. K1NB omits the bridge. Absent terms are returned as None.
"""
import math
import torch
from torch.nn import functional as F
from v10_1.model import token_loss, chunk_batch
from v10_2.model import future_ce

LAMBDA = dict(c=.1, vq=.01, h=1., kl=1.)
SIGMA_FLOOR = math.log(.01)


def gaussian_nll(pred, target, log_sigma, valid):
    """Per-dimension isotropic Gaussian NLL, mean over valid groups. pred/target [..., 2, d]; valid [...]."""
    D = pred.shape[-1] * pred.shape[-2]
    s = log_sigma.clamp_min(SIGMA_FLOOR)
    sq = (pred - target).square().sum((-1, -2))
    nll = sq / (2 * D * s.exp()) + .5 * (math.log(2 * math.pi) + s)
    return (nll * valid).sum() / valid.sum().clamp_min(1), (sq / D * valid).sum() / valid.sum().clamp_min(1)


def sample_boundaries(eligible, S, generator):
    """Up to S eligible groups per document, uniformly without replacement; returns (rows, groups)."""
    keys = torch.rand(eligible.shape, generator=generator).to(eligible.device)
    keys = keys.masked_fill(~eligible, -1.)
    k = min(S, eligible.shape[1])
    top = keys.topk(k, -1)
    ok = top.values >= 0
    rows = torch.arange(len(eligible), device=eligible.device)[:, None].expand(-1, k)
    return rows[ok], top.indices[ok]


class _Detached:
    def __init__(self, weight): self.weight = weight.detach()


def rmsn(x): return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype)


def bridge_losses(model, out, ids, lengths):
    """Vectorised over groups; loop over the 8 within-group steps (v11.md §5.3)."""
    u1, valid = out['u1'], out['valid_groups']
    b, g = valid.shape; t_len = ids.shape[1]
    if not g: return None, None, 0
    tokens = model.embedding(ids).detach()                                   # sg[E x]
    target = out['pre_outn'].detach(); teacher = out['logits'].detach()
    head_w, emb_w = model.decoder.outn.weight.detach(), model.embedding.weight.detach()   # frozen parameter view
    beta = model.bridge.init(u1.flatten(-2))                                 # anchor at r_g, NOT detached
    base = 8 * torch.arange(g, device=ids.device) + 7
    lh = lkl = 0.; n = 0
    for j in range(1, 9):
        t = base + j
        ok = valid & (t[None, :] < lengths[:, None])
        if not bool(ok.any()): break
        tc = t.clamp_max(t_len - 1)
        beta = model.bridge.step(beta, tokens[:, tc])
        okf = ok.to(beta.dtype)
        lh = lh + (F.smooth_l1_loss(rmsn(beta), rmsn(target[:, tc]), reduction='none').mean(-1) * okf).sum()
        student = F.log_softmax(F.linear(rmsn(beta) * head_w, emb_w).float(), -1)
        tl = F.log_softmax(teacher[:, tc].float(), -1)
        lkl = lkl + ((tl.exp() * (tl - student)).sum(-1) * okf).sum()
        n = n + ok.sum()
    if not torch.is_tensor(n) or int(n) == 0: return None, None, 0
    return lh / n, lkl / n, n


def probe_loss(model, out, ids, lengths, targets, target_mask, generator):
    chunk = chunk_batch(ids, lengths, targets, target_mask, out['valid_groups'], generator)
    if chunk is None: return None
    f = out['forecast'][chunk['rows'], chunk['boundaries']].detach()
    logits = model.probe(f, _Detached(model.embedding.weight))
    ce = F.cross_entropy(logits.flatten(0, 1), chunk['labels'].flatten(), reduction='none').reshape_as(chunk['mask'])
    w = torch.arange(1, 9, device=ce.device, dtype=ce.dtype).rsqrt() * chunk['mask']
    return ((ce * w).sum(-1) / w.sum(-1).clamp_min(1e-12)).mean()


def compute(arm, model, out, batch, step, warmup, generator, S):
    ids, lengths, targets, tmask = batch['ids'], batch['lengths'], batch['targets'], batch['target_mask']
    ntp = token_loss(out, targets, tmask)
    parts = dict(token=ntp, v1=None, v2=None, vq=None, bridge_h=None, bridge_kl=None, probe=None,
                 v1_mse=None, v2_mse=None, bridge_rows=0, v2_boundaries=0)
    if arm in ('ORD', 'ORDE'):
        parts['total'] = ntp; return parts
    probe = probe_loss(model, out, ids, lengths, targets, tmask, generator)
    parts['probe'] = probe
    total = ntp + (probe if probe is not None else 0.)
    if arm == 'NOAUX':
        parts['total'] = total; return parts
    ramp = min(1., step / warmup)
    valid = out['valid_groups']; aux = 0.
    if valid.shape[1] > 1:
        tau = model.normalized_target(out['pooled'])
        elig = valid[:, 1:]
        f1, _ = model.head(out['u1'])
        v1, v1_mse = gaussian_nll(f1[:, :-1], tau[:, 1:], model.log_sigma1, elig.to(f1.dtype))
        rows, groups = sample_boundaries(elig, S, generator)
        if len(rows):
            u2s = model.visit2_from_stopped(out)[rows, groups]
            f2, _ = model.head(u2s)
            ones = torch.ones(len(rows), device=f2.device, dtype=f2.dtype)
            v2, v2_mse = gaussian_nll(f2, tau[rows, groups + 1], model.log_sigma2, ones)
        else:
            v2 = v2_mse = None
        vq = model.head.fitting_loss(tau[:, 1:], elig)
        parts.update(v1=v1, v2=v2, vq=vq, v1_mse=v1_mse, v2_mse=v2_mse, v2_boundaries=len(rows))
        aux = aux + LAMBDA['c'] * (v1 / 3 + (2 * v2 / 3 if v2 is not None else 0.)) + LAMBDA['vq'] * vq
    if arm == 'K1':
        lh, lkl, n = bridge_losses(model, out, ids, lengths)
        if lh is not None:
            parts.update(bridge_h=lh, bridge_kl=lkl, bridge_rows=int(n))
            aux = aux + LAMBDA['h'] * lh + LAMBDA['kl'] * lkl
    parts['total'] = total + ramp * aux
    return parts
