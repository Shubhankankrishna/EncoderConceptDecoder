"""Objectives for the tiny arms.

C0  token CE only.
A   v10_1.model.losses unchanged (latent MSE, codeword fitting, chunk head,
    hard-argmin exploration with selector CE).
X   token CE + 0.1 * forecast-only future CE on the committed codes (V10.2)
    + 0.1 * mixture likelihood on candidate updates (V10.3):

      ll_k  = -sum_j mask_j * CE_{k,j}           visit-2 forecast of code k
      log_pi = log_softmax(selector[rows, at])    prefix-only prior
      L_mix = -mean_doc( logsumexp_k(log_pi_k + ll_k) / n_doc )

    Every one of the K branches carries gradient, weighted by its posterior
    responsibility; the selector is trained as the mixture prior rather than by
    cross-entropy on a hard winner. This is the smooth (maximum-likelihood)
    form of best-of-K training in Explorative Modeling, with a learned prior.
"""
import torch
from torch.nn import functional as F
from v10_1.model import token_loss, chunk_batch
from v10_1.model import losses as losses_a
from v10_2.model import future_ce


def mixture(model, output, chunk, lengths):
    rows, at = chunk['rows'], chunk['boundaries']
    subset = {k: output[k][rows] for k in ('pooled', 'sources', 'codes', 'valid_groups')}
    k = model.c.codes
    predictions = model.candidate_forecasts(subset, lengths[rows], at)                 # [n, K, 2, d]
    expanded = {name: chunk[name].repeat_interleave(k, 0) for name in ('labels', 'mask')}
    _, ce = future_ce(model, predictions.flatten(0, 1), expanded, details=True)        # [n*K, 8]
    mask = expanded['mask'].to(ce.dtype)
    ll = -(ce * mask).sum(-1).reshape(len(rows), k)
    log_prior = F.log_softmax(output['selector'][rows, at], -1)
    joint = log_prior + ll
    n_doc = chunk['mask'].sum(-1).to(ce.dtype).clamp_min(1)
    loss = -(torch.logsumexp(joint, -1) / n_doc).mean()
    with torch.no_grad():
        resp = joint.softmax(-1)
        per_token = -ll / n_doc[:, None]
        ranked = per_token.sort(-1).values
        stats = dict(responsibility_entropy=float(-(resp * resp.clamp_min(1e-12).log()).sum(-1).mean()),
                     prior_entropy=float(-(log_prior.exp() * log_prior).sum(-1).mean()),
                     winner_prior_ce=float(-log_prior.gather(1, ll.argmax(-1, keepdim=True)).mean()),
                     spread=float((ranked[:, -1] - ranked[:, 0]).mean()), gap=float((ranked[:, 1] - ranked[:, 0]).mean()),
                     winners=ll.argmax(-1).tolist())
    return loss, stats


def losses_x(model, output, ids, lengths, targets, target_mask, candidate_update=False, generator=None, boundaries=None):
    token = token_loss(output, targets, target_mask); zero = token * 0
    chunk = chunk_batch(ids, lengths, targets, target_mask, output['valid_groups'], generator, boundaries)
    future = mix = zero; visits = [zero, zero]; stats = {}
    if chunk is not None:
        rows, at = chunk['rows'], chunk['boundaries']
        visits = [future_ce(model, f[rows, at], chunk).mean() for f in output['forecasts']]
        future = visits[0] / 3 + 2 * visits[-1] / 3
        if candidate_update:
            mix, stats = mixture(model, output, chunk, lengths)
    total = token + .1 * future + .1 * mix
    return dict(total=total, token=token, future=future, mixture=mix, future_visits=visits,
                candidate_executed=bool(candidate_update and chunk is not None), **stats)


def losses_c0(model, output, ids, lengths, targets, target_mask, **_):
    token = token_loss(output, targets, target_mask)
    return dict(total=token, token=token)


def compute(arm, model, output, batch, candidate_update, generator):
    args = (model, output, batch['ids'], batch['lengths'], batch['targets'], batch['target_mask'])
    if arm == 'C0': return losses_c0(*args)
    if arm == 'A':
        out = losses_a(*args, candidate_update=candidate_update, generator=generator)
        out['latent_visits'] = out.pop('latent_visits'); return out
    return losses_x(*args, candidate_update=candidate_update, generator=generator)
